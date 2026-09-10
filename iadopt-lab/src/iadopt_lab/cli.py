"""Thin command entry point: parse arguments, delegate, and return an exit code.

This module owns no experiment logic. It does not loop, call providers, decide
retries, execute SQL, score predictions, or manage workers. Every command
delegates to the component that owns the behaviour and prints its result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .domain import LabError

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILED = 1
# Protocol overhead beyond the serialized messages: role wrappers, chat template markers
# and the request envelope. Deliberately generous; it is added to an upper bound.
_MESSAGE_OVERHEAD_TOKENS = 256
# Stop reasons that mean the campaign's work is done. `already_complete` is the runner
# recognizing finished work, which is a success for an idempotent resume.
# Stop reasons meaning the campaign is finished. `tasks_terminal` is finished WITH
# recorded operational failures: every remaining task is in a state no further work
# can advance, so refusing to finalize would discard the results that did succeed.
_SUCCESSFUL_STOPS = frozenset({"tasks_complete", "already_complete", "tasks_terminal"})


def _root(value: str | None) -> Path:
    """Resolve the lab root that owns parameters.yml.

    Args: value: Explicit --root argument or None for the packaged project root.
    Returns: Existing directory containing parameters.yml.
    Raises: LabError when the directory does not hold the experiment configuration.
    Side effects: Filesystem existence checks only.
    """
    root = Path(value).resolve() if value else Path(__file__).resolve().parents[2]
    if not (root / "parameters.yml").is_file():
        raise LabError(f"No parameters.yml under {root}; pass --root")
    return root


def _emit(payload: Any, as_json: bool) -> None:
    """Print a command result as canonical JSON or a short human summary.

    Args: payload: JSON-compatible result mapping; as_json: whether to print raw JSON.
    Returns: None.
    Raises: TypeError for non-serializable payloads.
    Side effects: Writes to stdout only.
    """
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return
    for key, value in payload.items():
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        elif isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        else:
            print(f"{key}: {value}")


def _configuration(root: Path, *, synthetic: bool = False):
    """Load and resolve the experiment configuration for one command.

    Args: root: Lab root; synthetic: whether to derive the labelled offline fixture snapshot.
    Returns: Resolved Configuration snapshot.
    Raises: ConfigurationError for invalid YAML or schema violations.
    Side effects: Reads parameters.yml and its schema.
    """
    from .configuration import load_parameters, resolve_configuration
    from .planning import synthetic_configuration

    resolved = resolve_configuration(load_parameters(root / "parameters.yml"))
    return synthetic_configuration(resolved) if synthetic else resolved


def cmd_preflight(args: argparse.Namespace) -> int:
    """Report draft-readiness issues and, with --live, the full live-dispatch gate.

    Args: args: Parsed namespace with root, live and json flags.
    Returns: EXIT_OK when no blocking issue remains, otherwise EXIT_FAILED.
    Raises: ConfigurationError for an unreadable or invalid configuration.
    Side effects: Reads configuration; never contacts a provider or database.
    """
    from .configuration import validate_live_readiness

    root = _root(args.root)
    resolved = _configuration(root)
    if args.live:
        report = validate_live_readiness(resolved)
        _emit({"scope": "live", "ready": report["ready"], "configuration_sha256": report["configuration_sha256"],
               "issue_count": len(report["issues"]), "issues": report["issues"]}, args.json)
        return EXIT_OK if report["ready"] else EXIT_FAILED
    issues = list(resolved.issues)
    _emit({"scope": "draft", "ready": not issues, "configuration_sha256": resolved.sha256,
           "issue_count": len(issues), "issues": issues}, args.json)
    return EXIT_OK if not issues else EXIT_FAILED


def cmd_ingest(args: argparse.Namespace) -> int:
    """Materialize and verify the pinned corpus bundle from a read-only source.

    Args: args: Namespace with root, optional --source-repository / --source-directory.
    Returns: EXIT_OK after a complete verified bundle is written.
    Raises: ValueError for any source, projection, or integrity conflict.
    Side effects: Writes immutable corpus, canonical, derived, and manifest artifacts.
    """
    from .corpus.ingestion import SOURCE_TAG, ingest_corpus

    root = _root(args.root)
    result = ingest_corpus(root, source_repository=args.source_repository,
                           source_directory=args.source_directory)
    manifest = result["manifest"]
    _emit({"tag": SOURCE_TAG, "commit": manifest["commit"], "tree": manifest["tree"],
           "files": manifest["file_count"], "records": len(result["records"]),
           "demonstrations": len(result["demonstrations"]["demonstrations"]),
           "population": result["evaluation_population"]["member_count"],
           "manifest_sha256": manifest["manifest_sha256"],
           "regression_counts": manifest["regression_counts"]}, args.json)
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-check every materialized corpus artifact against its stored identity.

    Args: args: Namespace with root and json flags.
    Returns: EXIT_OK when the full bundle verifies.
    Raises: ValueError for corruption, hash mismatch, or an incomplete population.
    Side effects: Filesystem reads only; no artifact is rewritten.
    """
    from .corpus.ingestion import (
        SOURCE_TAG,
        load_canonical_records,
        verify_derived_projections,
    )

    root = _root(args.root)
    records = load_canonical_records(root)
    derived = verify_derived_projections(root, records)
    _emit({"tag": SOURCE_TAG, "records": len(records),
           "demonstrations": sum(1 for row in records if row["demonstration_position"]),
           "population": sum(1 for row in records if not row["demonstration_position"]),
           "derived_files_verified": derived, "status": "verified"}, args.json)
    return EXIT_OK


def cmd_plan(args: argparse.Namespace) -> int:
    """Expand the frozen grid and report deterministic run, task, and call counts.

    Args: args: Namespace with root, synthetic, out, and json flags.
    Returns: EXIT_OK when a plan is expanded; EXIT_FAILED when the draft is unresolved.
    Raises: ConfigurationError for unresolved settings or an invalid population.
    Side effects: Reads corpus artifacts; writes the plan only when --out is given.
    """
    from .artifacts import collect_input_artifacts
    from .corpus.ingestion import load_canonical_records
    from .planning import expand_campaign

    root = _root(args.root)
    resolved = _configuration(root, synthetic=args.synthetic)
    if resolved.issues and not args.synthetic:
        _emit({"error": "configuration is not plan-ready", "issues": list(resolved.issues)}, args.json)
        return EXIT_FAILED
    records = list(load_canonical_records(root))
    targets = [row for row in records if not row["demonstration_position"]]
    _, similarity = _similarity_for(root, resolved.data, args.synthetic)
    if args.synthetic:
        targets = targets[:3]
    # Identities describe the planned population, not the full benchmark it came from.
    bundle = collect_input_artifacts(root, records, targets, similarity)
    plan = expand_campaign(resolved, targets, artifact_identities=bundle["identities"],
                           mode="synthetic" if args.synthetic else "live")
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _emit({"mode": plan["mode"], "plan_sha256": plan["sha256"],
           "population_size": plan["population_size"], **plan["counts"],
           "written": args.out or None}, args.json)
    return EXIT_OK


def cmd_estimate(args: argparse.Namespace) -> int:
    """Calculate the pre-run cost estimate from an explicit evidence file.

    Args: args: Namespace with root, plan, evidence, and json flags.
    Returns: EXIT_OK when the estimate is produced.
    Raises: LabError or ValueError for incomplete price, token, or assumption evidence.
    Side effects: Reads the plan and evidence files; performs no provider call.
    """
    from .costing import estimate_campaign_cost

    _root(args.root)
    plan = json.loads(Path(args.plan).read_bytes())
    evidence = json.loads(Path(args.evidence).read_bytes())
    estimate = estimate_campaign_cost(plan, evidence["prompt_artifacts"],
                                      evidence["billing_evidence"], evidence["assumptions"])
    _emit(estimate, args.json)
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    """Rank complete configurations from stored observations and export a derivative.

    Args: args: Namespace with root, plan, observations, format, out, and json flags.
    Returns: EXIT_OK when a ranking is produced.
    Raises: ValueError for observations outside the frozen plan or inconsistent receipts.
    Side effects: Reads plan/observation files; writes only beneath outputs/ when exporting.
    """
    from iadopt_eval.core import CLOSE_THRESHOLD, SCORER_VERSION

    from .artifacts import scorer_identity_hash
    from .corpus.ingestion import load_canonical_records
    from .reporting import build_configuration_ranking, export_report

    root = _root(args.root)
    plan = json.loads(Path(args.plan).read_bytes())
    observations = json.loads(Path(args.observations).read_bytes())
    # The same scientific checks the finalizer applies. Leaving them optional here meant
    # a report's integrity depended on which entry point produced it: this command would
    # accept evidence scored by any backend and infer category coverage from whatever
    # rows it was handed.
    resolved = _configuration(root)
    _, similarity_identity = _similarity_for(root, resolved.data, plan.get("mode") == "synthetic")
    records = load_canonical_records(root)
    # Recomputed from this checkout's evaluator source and the backend actually loaded,
    # never copied from the plan. Copying it made the binding check compare the plan to
    # itself, which passes for any checkout and so proves nothing about the report.
    scorer = scorer_identity_hash(root, similarity_identity)
    # Denominators come from the population the plan selected, not the whole corpus. The
    # five demonstrations are excluded from scoring by design, so counting them as
    # expected members reported a fully scored campaign as missing coverage.
    population = set(plan.get("population") or ())
    report = build_configuration_ranking(
        plan, observations,
        scorer_identity={"scorer_version": SCORER_VERSION,
                         "similarity_identity": similarity_identity,
                         "close_threshold": CLOSE_THRESHOLD,
                         "plan_scorer": scorer},
        population_categories={row["variable_id"]: {"category": row["category"],
                                                    "category_path": row["category_path"]}
                               for row in records if row["variable_id"] in population})
    written = None
    if args.out:
        written = export_report(report, args.format, args.out, outputs_root=root / "outputs")
    _emit({"policy_version": report["policy_version"], "mode": report["mode"],
           "synthetic": report["synthetic"], "population_size": report["population_size"],
           "configurations": report["configuration_count"], "rankable": report["rankable_count"],
           "complete": report["complete"], "written": written}, args.json)
    return EXIT_OK


def cmd_database(args: argparse.Namespace) -> int:
    """Prepare, start, initialize, or migrate the isolated local PostgreSQL instance.

    Args: args: Namespace with root, action, image, and dsn flags.
    Returns: EXIT_OK when the requested action completes.
    Raises: LabError or subprocess errors when Docker or the server is unavailable.
    Side effects: Container and database side effects for the named action only.
    """
    from .local_database import (
        initialize_local_roles,
        local_dsn,
        prepare_local_database,
        start_local_database,
    )
    from .persistence import migrate

    root = _root(args.root)
    if args.action == "prepare":
        _emit(prepare_local_database(root, args.image), args.json)
    elif args.action == "start":
        _emit(start_local_database(root), args.json)
    elif args.action == "init":
        _emit(initialize_local_roles(root), args.json)
    else:
        migrate(args.dsn or local_dsn(root, role="migrator"))
        _emit({"action": "migrate", "status": "applied"}, args.json)
    return EXIT_OK



def _database_url(root: Path, explicit: str | None) -> str:
    """Resolve the PostgreSQL connection string for an execution command.

    Args: root: Lab root; explicit: --dsn value, else DATABASE_URL, else the local app role.
    Returns: A connection string; its secret is used immediately and never stored.
    Raises: LabError when no local credential file exists and nothing was supplied.
    Side effects: Reads the ignored local credential file only when no DSN is given.
    """
    import os

    from .local_database import local_dsn

    return explicit or os.environ.get("DATABASE_URL") or local_dsn(root, role="app")


def _corpus_rows(root: Path, records: list[dict]) -> list[dict]:
    """Map canonical corpus records onto the persistence registration contract.

    Args: root: Lab root holding the immutable source snapshot; records: canonical records.
    Returns: One registration mapping per variable, carrying exact source bytes.
    Raises: OSError when a source Turtle file is missing.
    Side effects: Reads the pinned Turtle snapshot; writes nothing.
    """
    from .corpus.ingestion import _corpus_dir

    rows = []
    for record in records:
        raw = (_corpus_dir(root) / record["source_path"]).read_bytes()
        # `label` and `demonstration_position` are optional to the repository, which
        # falls back to the opaque URN when they are absent. Passing them keeps the
        # variable table readable and makes demonstrations identifiable in SQL.
        rows.append({key: record[key] for key in (
            "variable_id", "source_path", "label", "definition", "source_iri", "category",
            "subcategory", "category_path", "gold", "gold_sha256", "source_sha256",
            "demonstration_position")} | {"source_content": raw})
    return rows


def _similarity_backend(root: Path, evaluation: dict) -> Any:
    """Load the frozen local scoring backend by convention, never by download.

    The manifest lives at ``data/manifests/scorer-model-v1.yml`` and the immutable
    model bytes at ``data/embeddings/<model name>``. The revision recorded in
    ``parameters.yml`` must equal the manifest revision, so a swapped artifact
    cannot be scored under an unchanged configuration identity.

    Args: root: Lab root; evaluation: resolved evaluation policy block.
    Returns: Verified callable similarity backend exposing its frozen identity.
    Raises: LabError when the manifest is absent or disagrees with the configuration;
        ValueError when the artifact bytes fail verification.
    Side effects: Reads and hashes local model files only.
    """
    import yaml

    from iadopt_eval.embeddings import load_local_similarity

    path = root / "data/manifests/scorer-model-v1.yml"
    if not path.is_file():
        raise LabError("Scoring needs a frozen embedding manifest at " + str(path))
    manifest = yaml.safe_load(path.read_bytes())
    if manifest.get("revision") != evaluation.get("similarity_model_revision"):
        raise LabError("Embedding manifest revision differs from parameters.yml")
    if manifest.get("model_id") != evaluation.get("similarity_model"):
        raise LabError("Embedding manifest model differs from parameters.yml")
    directory = root / "data/embeddings" / str(manifest["model_id"]).split("/")[-1]
    return load_local_similarity(directory, manifest)


def _live_services(root: Path, data: dict, plan: dict) -> tuple[Any, Any, Any]:
    """Build the conservative token bound and price-backed cost policy for live dispatch.

    Args: root: lab root; data: resolved configuration; plan: frozen expanded plan.
    Returns: (token_bound, cost_policy, settlement_policy) callables. The first two run
        before dispatch; the third normalizes provider usage into an accounting record
        after the response returns.
    Raises: LabError when the frozen price card cannot back a metered reservation.
    Side effects: None; both callables are pure functions of the request body.
    """
    from decimal import Decimal

    import yaml

    from .costing import decimal_value

    ceiling = int(data["parameter_grid"]["max_output_tokens"])
    card_path = Path(data["cost_accounting"]["price_card_manifest"])
    card = yaml.safe_load((root / card_path).read_bytes())

    # Coverage is checked here, once, against the whole plan. `cost_policy` raises for an
    # uncovered model, and it raises from inside the worker pool, where the failure
    # cancels unrelated in-flight tasks. Discovering a missing card entry on the first
    # task of the second model is the worst possible moment: the estimate has already
    # reported the campaign ready, and part of the work is already dispatched. A
    # non-billed provider is not exempt - an evidenced zero is still evidence.
    missing = sorted({f"{count['provider']}/{count['model_id']}" for count in plan["counts"]["by_model"]}
                     - set(card.get("models") or {}))
    if missing:
        raise LabError(f"Frozen price card {card_path.as_posix()} has no entry for: "
                       + ", ".join(missing)
                       + ". Every planned model needs price evidence before dispatch.")

    def token_bound(task: dict, body: dict) -> int:
        """Bound one request's tokens from exact message bytes plus the output ceiling.

        The ratio is one token per UTF-8 byte plus a fixed protocol overhead.

        Conditions under which this is an upper bound, stated because a tokenizer-family
        property alone does not establish a provider's request accounting:

        1. The deployment uses a byte-level BPE tokenizer, where no token covers fewer
           than one input byte, so token count cannot exceed byte count. Both currently
           selected families (Qwen3, GLM) are byte-level BPE.
        2. Chat-template and special-token overhead stays within
           `_MESSAGE_OVERHEAD_TOKENS`, which is generous for a single user message.
        3. The provider bills the request it was sent, without server-side expansion.

        It is deliberately not the measured average. Qwen3 encodes these prompts at about
        0.23 tokens per byte, but a correction prompt embeds arbitrary model output whose
        density is unknown in advance, and PSNC publishes no tokenizer for its deployment.
        An average is an expectation; admission control needs a ceiling. The headroom is
        free here: the bound stays far inside both context windows and the token rate.

        Adding a deployment that violates condition 1 (a character- or word-level
        tokenizer) requires re-establishing this bound before it can be trusted.
        """
        raw = json.dumps(body.get("messages", []), ensure_ascii=False).encode("utf-8")
        return len(raw) + _MESSAGE_OVERHEAD_TOKENS + ceiling

    def cost_policy(task: dict, body: dict) -> dict[str, Any]:
        """Reserve a bounded upper-bound cost for one attempt from frozen prices."""
        key = task["provider"] + "/" + task["run"]["model_id"]
        entry = card["models"].get(key)
        if not entry:
            raise LabError("Frozen price card has no entry for " + key)
        if entry["mode"] == "non_billed":
            return {"reservation_amount": "0", "bounded": True, "price_evidence": entry}
        raw = json.dumps(body.get("messages", []), ensure_ascii=False).encode("utf-8")
        inputs = Decimal(len(raw) + _MESSAGE_OVERHEAD_TOKENS)
        # The same FX factor settlement applies. Without it a reservation is held in the
        # provider's currency while its settlement is recorded in the reporting currency,
        # so the two describe different units and cap accounting compares them wrongly.
        amount = ((inputs * decimal_value(entry["input_per_million"])
                   + Decimal(ceiling) * decimal_value(entry["output_per_million"]))
                  / Decimal(1000000) * decimal_value(entry["fx_to_reporting"]))
        return {"reservation_amount": str(amount.quantize(Decimal("0.00000001"))),
                "bounded": True, "price_evidence": entry,
                "currency": card.get("currency"), "fx_to_reporting": str(entry["fx_to_reporting"])}

    def settlement_policy(task: dict, response: dict) -> dict[str, Any]:
        """Normalize provider usage into an accounting record after the call returns.

        Without this the repository finds no `cost` key on the response payload and
        records every metered call as `unavailable`, leaving its reservation unsettled.
        Precedence is deliberate: a cost the provider itself reports is `actual`;
        otherwise real token counts priced with the frozen card give `estimated`;
        absent usage stays `unavailable` rather than being assumed free.

        Args: task: leased task carrying provider/model identity; response: ProviderResult dict.
        Returns: Settlement record for the persistence layer.
        Raises: LabError when the frozen price card has no entry for the model.
        Side effects: None; pricing comes from the frozen card, never a live lookup.
        """
        key = task["provider"] + "/" + task["run"]["model_id"]
        entry = card["models"].get(key)
        if not entry:
            raise LabError("Frozen price card has no entry for " + key)
        if entry["mode"] == "non_billed":
            return {"amount": "0", "state": "confirmed_zero", "price_evidence": entry}
        usage = response.get("usage") or {}
        reported = usage.get("cost", usage.get("total_cost"))
        if reported is not None:
            # A provider reports cost in its own billing currency; convert it like every
            # other amount so stored values share one unit.
            converted = decimal_value(reported) * decimal_value(entry["fx_to_reporting"])
            return {"amount": str(converted.quantize(Decimal("0.00000001"))), "state": "actual",
                    "price_evidence": entry, "basis": "provider-reported cost, FX applied",
                    "currency": card.get("currency"),
                    "fx_to_reporting": str(entry["fx_to_reporting"]), "usage": usage}
        prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
            return {"state": "unavailable", "price_evidence": entry,
                    "basis": "provider returned no usable token usage", "usage": usage}
        amount = (Decimal(prompt_tokens) * decimal_value(entry["input_per_million"])
                  + Decimal(completion_tokens) * decimal_value(entry["output_per_million"])
                  ) / Decimal(1000000) * decimal_value(entry["fx_to_reporting"])
        return {"amount": str(amount.quantize(Decimal("0.00000001"))), "state": "estimated",
                "price_evidence": entry, "usage": usage,
                "currency": card.get("currency"), "fx_to_reporting": str(entry["fx_to_reporting"]),
                "basis": "actual token counts priced with the frozen price card, FX applied"}

    return token_bound, cost_policy, settlement_policy


def _authorize(repository: Any, campaign_id: str, plan: dict, estimate: dict, actor: str,
               channel: str) -> str:
    """Record the disclosed estimate and the operator's explicit live approval.

    Args: repository: open repository; campaign_id: planned live campaign;
        plan: frozen plan; estimate: a usable pre-run-estimate-v1 result; actor: approver;
        channel: the exact invocation that carried the approval, recorded verbatim.
    Returns: The stored authorization identifier.
    Raises: LabError when the estimate is unusable; persistence errors on plan mismatch.
    Side effects: Writes one immutable live_authorization row. It records approval; it
        never infers or requests it.
    """
    from datetime import UTC, datetime

    from .canonical import content_hash
    from .persistence.repository import _clean, _hash

    if not estimate.get("ready") or estimate.get("issues"):
        raise LabError("Refusing to authorize an estimate that reports issues")
    if estimate.get("plan_sha256") != plan["sha256"]:
        raise LabError("Estimate is bound to a different plan")
    # The estimate carries its own hash, so an edited file can keep a matching plan hash
    # and a `ready` flag while its numbers say something else. Recomputing that hash is
    # what makes the receipt describe the document actually disclosed.
    claimed = estimate.get("sha256")
    recomputed = content_hash({key: value for key, value in estimate.items() if key != "sha256"})
    if claimed != recomputed:
        raise LabError("Estimate contents do not match the hash it carries; it was edited "
                       "after it was produced")
    # An estimate priced from one set of cards while the runner reserves against another
    # is not a disclosure of this campaign's cost. Both must name the same models.
    priced = set((estimate.get("billing_evidence") or {}).get("models") or {})
    planned = {f"{count['provider']}/{count['model_id']}" for count in plan["counts"]["by_model"]}
    if priced != planned:
        raise LabError("Estimate prices " + ", ".join(sorted(priced)) + " but the plan runs "
                       + ", ".join(sorted(planned)))
    fingerprint = repository.get_campaign(campaign_id)["plan"]["fingerprint"]
    receipt = {"plan_fingerprint": fingerprint, "policy_version": "pre-run-estimate-v1",
               "usable": True, "price_evidence": estimate["by_model"],
               "expected_cost": estimate["totals"]["expected_cost"],
               "conditional_maximum_cost": estimate["totals"]["conditional_maximum_cost"],
               "currency": estimate["currency"], "estimate_sha256": estimate["sha256"]}
    digest = _hash(_clean(receipt))
    now = datetime.now(UTC).isoformat()
    stored = repository.record_live_authorization(
        campaign_id, receipt,
        {"estimate_hash": digest, "disclosed_at": now, "channel": channel},
        {"estimate_hash": digest, "plan_fingerprint": fingerprint, "authorized_at": now,
         "actor": actor, "explicit": True})
    return stored["id"]


def _similarity_for(root: Path, data: dict, synthetic: bool) -> tuple[Any, dict]:
    """Resolve the scorer callable and its identity for every command that plans.

    Scorer identity is part of the plan fingerprint, so `plan`, `estimate` and `run`
    must derive it the same way. Fabricating a descriptive identity in one command
    and loading the real backend in another silently produces two different plans.

    Args: root: Lab root; data: resolved configuration; synthetic: fixture mode.
    Returns: (similarity callable, path-free identity mapping).
    Raises: LabError when the frozen embedding artifact is unusable.
    Side effects: Reads and hashes local model files in live mode only.
    """
    from .workflow import synthetic_similarity

    if synthetic:
        return synthetic_similarity, {"backend": "synthetic-equality-only-v1", "synthetic": True}
    backend = _similarity_backend(root, data["evaluation"])
    return backend, backend.identity


def _provider_endpoint(profile: dict, secrets: Any) -> str | None:
    """Resolve the one effective base URL for a provider, from its declared override.

    Probing and generation must describe the same deployment. When they resolve the URL
    differently, capability evidence can be measured against one endpoint and written into
    a configuration that dispatches to another, and nothing downstream would notice.

    Args: profile: Provider block from parameters.yml; secrets: loaded runtime secrets.
    Returns: The override URL when one is configured and set, otherwise None so the
        caller falls back to the frozen profile default.
    Raises: Nothing.
    Side effects: Reads an already-loaded secret value; never logs it.
    """
    name = profile.get("base_url_env")
    return (secrets.get(name) or None) if name else None


def _prepare_campaign(root: Path, *, synthetic: bool, dsn: str | None, env_file: str | None,
                      estimate: dict | None = None, actor: str | None = None,
                      channel: str | None = None,
                      expected_campaign: str | None = None) -> tuple[str, Any, Any]:
    """Freeze one campaign end to end and return it ready to advance.

    Args: root: Lab root; synthetic: labelled offline fixture mode; dsn: explicit database;
        env_file: explicit credential file, required for live provider adapters.
    Returns: (campaign_id, Services, plan). Nothing is dispatched by this function.
    Raises: ConfigurationError for an unfrozen configuration; LabError for missing
        credentials or an unusable scorer backend; persistence errors on conflict.
    Side effects: Registers corpus, campaign and task rows; allocates provider clients.
    """
    from .artifacts import collect_input_artifacts
    from .configuration import load_runtime_secrets
    from .corpus.ingestion import load_canonical_records
    from .persistence import Repository
    from .planning import expand_campaign
    from .providers import openrouter, psnc
    from .workflow import Services

    resolved = _configuration(root, synthetic=synthetic)
    if resolved.issues:
        raise LabError("Configuration is not frozen: " + "; ".join(resolved.issues))
    data = resolved.data
    records = list(load_canonical_records(root))
    targets = [row for row in records if not row["demonstration_position"]]
    demonstrations = tuple(sorted((row for row in records if row["demonstration_position"]),
                                  key=lambda row: row["demonstration_position"]))

    similarity, similarity_identity = _similarity_for(root, data, synthetic)

    # Collect artifacts for exactly the targets that will be planned. Hashing all 97
    # while planning three made the identity called `population` describe a different
    # set from the plan's population, so the dry run could not exercise the identity
    # contract it exists to test.
    planned = targets[:3] if synthetic else targets
    inputs = collect_input_artifacts(root, records, planned, similarity_identity)
    plan = expand_campaign(resolved, planned, artifact_identities=inputs["identities"],
                           mode="synthetic" if synthetic else "live")
    bundle = {"plan": plan, "configuration": data, "demonstrations": demonstrations,
              "similarity_identity": similarity_identity,
              "artifact_index": inputs["index"], "runtime": inputs["runtime"]}

    adapters = {}
    if not synthetic:
        import os

        # Credential and endpoint names are declared per provider in parameters.yml, never
        # guessed. The base-URL override is read here as well as in probing: leaving it out
        # meant `PSNC_API_BASE_URL` steered capability probes while generation silently used
        # the frozen default, so the two could describe different deployments.
        names = [value for name in data["campaign"]["providers"]
                 for value in (data["providers"][name]["api_key_env"],
                               data["providers"][name].get("base_url_env")) if value]
        secrets = load_runtime_secrets(env_file, names, os.environ)
        for name in data["campaign"]["providers"]:
            profile = data["providers"][name]
            key = secrets.get(profile["api_key_env"])
            if not key:
                raise LabError(f"{profile['api_key_env']} is not available for provider {name}")
            factory = psnc if name == "psnc" else openrouter
            adapters[name] = factory.create_adapter(
                profile, key, base_url_override=_provider_endpoint(profile, secrets))

    repository = Repository(_database_url(root, dsn))
    if expected_campaign is not None:
        # Check before writing anything. Registering first meant a rejected resume left
        # a second campaign's registration and plan behind before reporting the mismatch.
        stored = repository.get_campaign(expected_campaign)
        if stored["configuration"].get("lab_plan_sha256") != plan["sha256"]:
            repository.close()
            raise LabError(
                f"Campaign {expected_campaign} was planned from different inputs; resume "
                "requires the checkout that produced it, or start a new campaign")
    corpus = repository.register_corpus(
        {"repository": data["dataset"]["repository"], "release": data["dataset"]["release"],
         "commit": data["dataset"]["commit"], "tree": data["dataset"]["tree"],
         "expected_count": len(records)}, _corpus_rows(root, records))
    # Persist the freeze before anything becomes dispatchable. The plan hash can detect
    # drift but cannot recover the bytes that drifted, so a campaign whose checkout is
    # later lost or changed had no route back to what it actually ran. `register_campaign`
    # accepted artifact references all along and `get_campaign_artifact` reads exactly one
    # row of a kind, so the whole freeze is registered as a single `experiment-bundle`.
    import base64

    from .canonical import canonical_json_bytes

    freeze = {"version": "experiment-bundle-v1", "plan": plan,
              "identities": inputs["identities"], "artifact_index": inputs["index"],
              "runtime": inputs["runtime"], "similarity_identity": similarity_identity,
              "configuration_sha256": resolved.sha256,
              "files": {row["path"]: base64.b64encode(row["content"]).decode("ascii")
                        for row in inputs["artifacts"]}}
    stored = repository.register_artifact(
        "experiment-bundle", canonical_json_bytes(freeze),
        {"plan_sha256": plan["sha256"], "mode": plan["mode"],
         "configuration_sha256": resolved.sha256, "file_count": len(inputs["artifacts"])})
    campaign_id = repository.register_campaign({**data, "lab_plan_sha256": plan["sha256"]},
                                               mode=plan["mode"],
                                               artifact_refs=[stored["id"]])
    registered = {row["variable_id"]: row for row in corpus["variables"]}
    repository.plan_tasks(campaign_id, plan["runs"],
                          [registered[vid] for vid in plan["population"]])
    services = Services(repository=repository, root=root, bundle=bundle,
                        similarity=similarity, adapters=adapters)
    if not synthetic:
        (services.token_bound, services.cost_policy,
         services.settlement_policy) = _live_services(root, data, plan)
        if estimate is not None:
            _authorize(repository, campaign_id, plan, estimate, actor, channel)
    return campaign_id, services, plan


def _advance(campaign_id: str, services: Any, *, stop_after: int | None) -> dict[str, Any]:
    """Run the campaign loop to completion or a safe stop.

    Args: campaign_id: frozen campaign; services: assembled runtime; stop_after: offline limit.
    Returns: Final state counts and stop reason from the workflow runner.
    Raises: Whatever the runner raises; failures are never swallowed into a success.
    Side effects: Provider requests in live mode; database writes throughout.
    """
    import asyncio

    from .workflow import finalize_campaign, run_campaign

    async def advance() -> dict[str, Any]:
        """Run the campaign, finalize it when every task is done, then close clients.

        Finishing the tasks is not finishing the campaign: the ranking and report
        evidence must be stored and the campaign marked complete, or the database keeps
        a `running` campaign with no scientific result while the caller exits zero.
        """
        try:
            result = await run_campaign(campaign_id, services, stop_after=stop_after)
            # `already_complete` also finalizes: an interruption between the last task and
            # the ranking write must be recoverable by simply running resume again.
            if result["stop_reason"] in _SUCCESSFUL_STOPS:
                result = {**result, "finalized": await finalize_campaign(campaign_id, services)}
            return result
        finally:
            for adapter in services.adapters.values():
                await adapter.close()

    return asyncio.run(advance())


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Execute a labelled synthetic campaign with no provider and no cost.

    Args: args: Namespace with root, dsn, stop_after and json flags.
    Returns: EXIT_OK when the fixture campaign reaches a terminal state.
    Raises: LabError or persistence errors; a synthetic run never contacts a provider.
    Side effects: Database writes only. Results are labelled synthetic and are not scientific.
    """
    root = _root(args.root)
    campaign_id, services, plan = _prepare_campaign(root, synthetic=True, dsn=args.dsn, env_file=None)
    try:
        result = _advance(campaign_id, services, stop_after=args.stop_after)
    finally:
        services.repository.close()
    _emit({"mode": "synthetic", "campaign_id": campaign_id, "plan_sha256": plan["sha256"],
           "tasks": plan["counts"]["tasks"], **result,
           "warning": "SYNTHETIC fixture results; not scientific"}, args.json)
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """Freeze and execute a live campaign against the selected providers.

    Args: args: Namespace with root, dsn, env_file and json flags.
    Returns: EXIT_OK when the runner stops; the stop reason states whether it completed.
    Raises: LabError when the configuration, credentials, estimate or authorization is absent.
    Side effects: Real provider requests that consume quota and money, plus database writes.
    """
    root = _root(args.root)
    estimate = json.loads(Path(args.estimate).read_bytes()) if args.estimate else None
    if estimate is None:
        raise LabError("Live execution requires --estimate naming a disclosed pre-run estimate")
    # Authorization is recorded as an explicit act, so it has to be one. Inferring it from
    # the presence of an estimate file meant the stored receipt asserted an approval that
    # nothing in the invocation expressed, and named a `--authorize` flag that did not
    # exist. An identified approver is required for the same reason: `operator` records
    # that somebody ran a command, not who accepted the cost.
    if not args.authorize:
        raise LabError("Live dispatch requires --authorize, confirming the disclosed estimate "
                       "in " + str(args.estimate) + " has been reviewed and accepted")
    if not args.actor:
        raise LabError("Live dispatch requires --actor naming who authorizes this campaign")
    campaign_id, services, plan = _prepare_campaign(root, synthetic=False, dsn=args.dsn,
                                                    env_file=args.env_file,
                                                    estimate=estimate, actor=args.actor,
                                                    channel="iadopt-lab run --authorize")
    try:
        result = _advance(campaign_id, services, stop_after=None)
    finally:
        services.repository.close()
    _emit({"mode": "live", "campaign_id": campaign_id, "plan_sha256": plan["sha256"],
           "tasks": plan["counts"]["tasks"], **result}, args.json)
    return EXIT_OK if result["stop_reason"] in _SUCCESSFUL_STOPS else EXIT_FAILED


def cmd_resume(args: argparse.Namespace) -> int:
    """Continue an interrupted campaign from its durable checkpoints.

    Args: args: Namespace with root, campaign, dsn, env_file, synthetic and json flags.
    Returns: EXIT_OK when the runner stops having completed every task.
    Raises: LabError on scientific drift between the stored campaign and the checkout.
    Side effects: Same as the original run; completed work is never repeated.
    """
    root = _root(args.root)
    campaign_id, services, plan = _prepare_campaign(root, synthetic=args.synthetic, dsn=args.dsn,
                                                    env_file=args.env_file,
                                                    expected_campaign=args.campaign)
    if args.campaign and args.campaign != campaign_id:
        raise LabError(f"Frozen inputs resolve to campaign {campaign_id}, not {args.campaign}")
    try:
        result = _advance(campaign_id, services, stop_after=None)
    finally:
        services.repository.close()
    _emit({"resumed": campaign_id, "tasks": plan["counts"]["tasks"], **result}, args.json)
    # A campaign already finished is a success: resume is idempotent by design, and
    # exiting non-zero here made automation treat completed work as a failure.
    return EXIT_OK if result["stop_reason"] in _SUCCESSFUL_STOPS else EXIT_FAILED


def cmd_probe_models(args: argparse.Namespace) -> int:
    """Measure per-model capabilities against a live deployment and report the evidence.

    parameters.yml refuses to infer capabilities from a model name, so the fields that
    gate live execution have to come from observation. This command supplies them. It
    sends three short throwaway probes per model, never a corpus prompt, and freezes
    nothing; --write only edits the models block of the named provider.

    Args: args: Namespace with root, provider, env_file, model, concurrency, timeout,
        write and json flags.
    Returns: EXIT_OK when at least one model can run with reasoning off.
    Raises: LabError for a missing provider block, credentials, or an unreadable catalog.
    Side effects: Three provider requests per model, and with --write one edit to parameters.yml.
    """
    import asyncio

    from .configuration import load_parameters, load_runtime_secrets
    from .probing import (
        chat_models,
        fetch_catalog,
        models_block,
        probe_provider,
        write_models_block,
    )

    root = _root(args.root)
    data = load_parameters(root / "parameters.yml").data
    provider = (data.get("providers") or {}).get(args.provider)
    if provider is None:
        raise LabError(f"No providers.{args.provider} block in parameters.yml")

    names = [name for name in (provider.get("api_key_env"), provider.get("base_url_env")) if name]
    secrets = load_runtime_secrets(args.env_file or str(root.parent / ".env"), names, os.environ)
    api_key = secrets.get(provider.get("api_key_env") or "")
    if not api_key:
        raise LabError(f"{provider.get('api_key_env')} is not set; pass --env-file")
    base_url = (_provider_endpoint(provider, secrets)
                or provider.get("base_url") or provider.get("default_base_url"))
    if not base_url:
        raise LabError(f"No base URL for provider {args.provider}")

    catalog = chat_models(fetch_catalog(base_url=base_url, api_key=api_key))
    if args.model:
        wanted = set(args.model)
        missing = wanted - {entry["id"] for entry in catalog}
        if missing:
            raise LabError("Provider catalog has no chat model named: " + ", ".join(sorted(missing)))
        catalog = [entry for entry in catalog if entry["id"] in wanted]
    if not catalog:
        raise LabError("Provider catalog lists no chat-capable models")

    results = asyncio.run(probe_provider(
        base_url=base_url, api_key=api_key,
        path=provider.get("chat_completions_path") or "/chat/completions",
        models=catalog, concurrency=args.concurrency, timeout_seconds=args.timeout))

    block = models_block(results, provider_label=args.provider.upper(), base_url=base_url)
    if args.write:
        write_models_block(root / "parameters.yml", args.provider, block)

    usable = [row["id"] for row in results if row["usable"]]
    if args.json:
        _emit({"provider": args.provider, "probed": len(results), "usable": usable,
               "results": results, "models_block": block}, True)
        return EXIT_OK if usable else EXIT_FAILED

    print(f"Probed {len(results)} chat models on {args.provider}\n")
    print(f"{'model':32} {'verdict':26} {'default':>9} {'off':>7} {'lat':>7}  use")
    for row in results:
        base_reasoning = row["baseline"].get("reasoning_chars")
        off_reasoning = row["switched"].get("reasoning_chars")
        print(f"{row['id']:32} {row['verdict']:26} "
              f"{'-' if base_reasoning is None else str(base_reasoning) + 'ch':>9} "
              f"{'-' if off_reasoning is None else str(off_reasoning) + 'ch':>7} "
              f"{row['switched'].get('seconds', 0):6.1f}s  {'yes' if row['usable'] else 'NO'}")
    print(f"\n{len(usable)} of {len(results)} models can run with reasoning off.")
    if args.write:
        print(f"Wrote the providers.{args.provider}.models block to parameters.yml.")
    else:
        print("\nRe-run with --write to apply this block, or paste it yourself:\n")
        print(block)
    return EXIT_OK if usable else EXIT_FAILED


def cmd_evidence(args: argparse.Namespace) -> int:
    """Derive the cost-estimate evidence document for a frozen plan.

    Every prompt in the plan is rendered and tokenized, and billing comes from the frozen
    price card the runner reserves against, so the estimate cannot report a campaign ready
    on evidence execution lacks. The ceiling assertion is not derived: without
    --ceiling-basis the estimate reports it unverified, which blocks a metered campaign and
    leaves a fully evidenced non-billed one carrying a recorded warning instead.

    Args: args: Namespace with root, plan, out, tokenizer, assumption and ceiling flags.
    Returns: EXIT_OK when the document is written.
    Raises: LabError for an unavailable tokenizer, a null ceiling, or underivable billing.
    Side effects: Reads corpus/prompt/schema files and writes the evidence JSON. No network.
    """
    from .evidence import build_cost_evidence

    root = _root(args.root)
    plan = json.loads(Path(args.plan).read_bytes())
    document = build_cost_evidence(
        plan, _configuration(root).data, root=root, tokenizer_id=args.tokenizer,
        expected_output_tokens=args.expected_output_tokens,
        attempt2_fraction=args.attempt2_fraction, attempt3_fraction=args.attempt3_fraction,
        correction_error_tokens=args.correction_error_tokens,
        ceiling_basis=args.ceiling_basis)
    Path(args.out).write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    counts = [len(v) for v in document["prompt_artifacts"]["input_tokens_by_run"].values()]
    _emit({"out": args.out, "runs": len(counts), "targets_per_run": counts[0] if counts else 0,
           "models": sorted(document["billing_evidence"]["models"]),
           "ceiling_verified": document["assumptions"]["ceiling_verified"]}, args.json)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Define every command without executing any of them.

    Args: None.
    Returns: Configured argument parser whose defaults name a handler function.
    Raises: None.
    Side effects: None.
    """
    parser = argparse.ArgumentParser(prog="iadopt-lab",
                                     description="Reproducible I-ADOPT lexical decomposition experiments.")
    parser.add_argument("--root", help="Lab directory containing parameters.yml")
    parser.add_argument("--json", action="store_true", help="Print the raw JSON result")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    preflight = commands.add_parser("preflight", help="Report draft or live readiness issues")
    preflight.add_argument("--live", action="store_true", help="Apply the full live-dispatch gate")
    preflight.set_defaults(handler=cmd_preflight)

    ingest = commands.add_parser("ingest", help="Materialize and verify the pinned corpus")
    ingest.add_argument("--source-repository", help="Read-only Git clone containing the pinned tag")
    ingest.add_argument("--source-directory", help="Ordinary directory holding the exact release files")
    ingest.set_defaults(handler=cmd_ingest)

    verify = commands.add_parser("verify", help="Re-check materialized corpus artifacts")
    verify.set_defaults(handler=cmd_verify)

    plan = commands.add_parser("plan", help="Expand the frozen grid and report counts")
    plan.add_argument("--synthetic", action="store_true", help="Use the labelled offline fixture snapshot")
    plan.add_argument("--out", help="Write the expanded plan to this path")
    plan.set_defaults(handler=cmd_plan)

    estimate = commands.add_parser("estimate", help="Produce the pre-run cost estimate")
    estimate.add_argument("--plan", required=True, help="Expanded plan JSON")
    estimate.add_argument("--evidence", required=True, help="Prompt/billing/assumption evidence JSON")
    estimate.set_defaults(handler=cmd_estimate)

    evidence = commands.add_parser("evidence", help="Derive the cost-estimate evidence document")
    evidence.add_argument("--plan", required=True, help="Expanded plan JSON from `plan --out`")
    evidence.add_argument("--out", required=True, help="Path to write the evidence document")
    evidence.add_argument("--tokenizer", default="Qwen/Qwen3-32B",
                          help="Locally cached tokenizer used to count prompt tokens")
    evidence.add_argument("--expected-output-tokens", type=int, default=150)
    evidence.add_argument("--attempt2-fraction", default="0.15")
    evidence.add_argument("--attempt3-fraction", default="0.05")
    evidence.add_argument("--correction-error-tokens", type=int, default=800)
    evidence.add_argument("--ceiling-basis",
                          help="The specific observation establishing that the planned output "
                               "ceiling bounds all-in generation. Supplying it marks the ceiling "
                               "verified; without it a metered campaign is blocked and a "
                               "non-billed one carries a recorded warning.")
    evidence.set_defaults(handler=cmd_evidence)

    report = commands.add_parser("report", help="Rank configurations and export results")
    report.add_argument("--plan", required=True, help="Frozen plan JSON")
    report.add_argument("--observations", required=True, help="Stored observation rows JSON")
    report.add_argument("--format", default="json", choices=("json", "csv"), help="Export format")
    report.add_argument("--out", help="Destination beneath outputs/")
    report.set_defaults(handler=cmd_report)

    dry = commands.add_parser("dry-run", help="Synthetic offline campaign; no provider, no cost")
    dry.add_argument("--dsn", help="Explicit database connection string")
    dry.add_argument("--stop-after", type=int, help="Stop after N task advancements")
    dry.set_defaults(handler=cmd_dry_run)

    run = commands.add_parser("run", help="Freeze and execute a live campaign")
    run.add_argument("--dsn", help="Explicit database connection string")
    run.add_argument("--env-file", help="Explicit credential file supplying provider API keys")
    run.add_argument("--estimate", help="Disclosed pre-run estimate JSON; required for live dispatch")
    run.add_argument("--actor", help="Person recorded as granting explicit live authorization")
    run.add_argument("--authorize", action="store_true",
                     help="Confirm the disclosed estimate is accepted. Required for live dispatch; "
                          "the stored authorization receipt records this exact invocation.")
    run.set_defaults(handler=cmd_run)

    resume = commands.add_parser("resume", help="Continue an interrupted campaign")
    resume.add_argument("--campaign", help="Expected campaign id; verified against frozen inputs")
    resume.add_argument("--dsn", help="Explicit database connection string")
    resume.add_argument("--env-file", help="Explicit credential file supplying provider API keys")
    resume.add_argument("--synthetic", action="store_true", help="Resume a synthetic campaign")
    resume.set_defaults(handler=cmd_resume)

    probe = commands.add_parser("probe-models",
                                help="Measure provider model capabilities and emit evidence")
    probe.add_argument("--provider", default="psnc", help="Provider key under providers: in parameters.yml")
    probe.add_argument("--env-file", help="Explicit credential file supplying provider API keys")
    probe.add_argument("--model", action="append", help="Probe only this model id; repeatable")
    probe.add_argument("--concurrency", type=int, default=2, help="Models probed in parallel")
    probe.add_argument("--timeout", type=float, default=120.0, help="Per-probe request timeout")
    probe.add_argument("--write", action="store_true",
                       help="Apply the resulting models block to parameters.yml")
    probe.set_defaults(handler=cmd_probe_models)

    database = commands.add_parser("database", help="Manage the isolated local PostgreSQL instance")
    database.add_argument("action", choices=("prepare", "start", "init", "migrate"))
    database.add_argument("--image", default="postgres:16", help="Container image for prepare")
    database.add_argument("--dsn", help="Explicit DSN for migrate")
    database.set_defaults(handler=cmd_database)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch one command, and translate failures into exit codes.

    Args: argv: Optional explicit argument list; defaults to sys.argv[1:].
    Returns: Process exit code. Usage errors return 2; handled failures return 1.
    Raises: Nothing for an expected experiment error; unexpected exceptions propagate.
    Side effects: Only those of the dispatched command; writes diagnostics to stderr.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return args.handler(args)
    except LabError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILED
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
