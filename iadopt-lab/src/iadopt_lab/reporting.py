"""Pure complete-population ranking and safe, reproducible derivative exports."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from iadopt_eval import aggregate_items

from .canonical import canonical_json_bytes, content_hash, sha256_bytes

POLICY_VERSION = "mean-repetition-micro-close-f1-v1"
REPORT_VERSION = "configuration-report-v1"
EXPORT_VERSION = "safe-report-export-v1"


def _fraction(receipt: Mapping[str, Any]) -> Fraction:
    """Read a reduced metric receipt without consulting rounded presentation text.

    Args:
        receipt: Evaluator numerator/denominator mapping.
    Returns:
        Exact rational metric value.
    Raises:
        ValueError: Receipt integer types, sign, denominator, or reduction is invalid.
    Side Effects:
        None.
    """
    numerator, denominator = receipt.get("numerator"), receipt.get("denominator")
    if type(numerator) is not int or type(denominator) is not int or numerator < 0 or denominator <= 0:
        raise ValueError("Invalid report fraction receipt")
    result = Fraction(numerator, denominator)
    if (result.numerator, result.denominator) != (numerator, denominator):
        raise ValueError("Report fraction receipt must be reduced")
    return result


def _receipt(value: Fraction | int) -> dict[str, int | float]:
    """Render exact rational report values with non-authoritative display derivatives.

    Args:
        value: Exact rational or integer value.
    Returns:
        Reduced numerator/denominator and floating display value.
    Raises:
        OverflowError: An impractically large value cannot render as a float.
    Side Effects:
        None.
    """
    rational = Fraction(value)
    return {"numerator": rational.numerator, "denominator": rational.denominator, "value": float(rational)}


def _number(value: Any, name: str) -> Decimal:
    """Parse a finite nonnegative usage, latency, or monetary quantity exactly.

    Args:
        value: Integer, finite float, or explicit decimal string.
        name: Field name for a credential-free error message.
    Returns:
        Decimal parsed from the original textual representation.
    Raises:
        ValueError: Quantity is boolean, nonnumeric, negative, or non-finite.
    Side Effects:
        None.
    """
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {name} quantity") from exc
    if isinstance(value, bool) or not result.is_finite() or result < 0:
        raise ValueError(f"Invalid {name} quantity")
    return result


def _observations(observations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate, detach, and canonically order explicit run-variable evidence.

    Args:
        observations: Rows with run_id, variable_id, evaluation or null,
            terminal_invalid, exact category fields, and an attempts list.
    Returns:
        Detached rows sorted by run and variable identity.
    Raises:
        ValueError: Duplicate task, corrupt evaluation, missing identity/classification,
            inconsistent gold/category provenance, or incompatible scorer backends.
        TypeError: Evidence is not canonical-JSON-compatible.
    Side Effects:
        Consumes the input iterable once; no provider, database, or filesystem access.
    """
    rows = json.loads(canonical_json_bytes(list(observations)))
    seen, lineage, backend = set(), {}, None
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Observation must be a mapping")
        for name in ("run_id", "variable_id", "category", "subcategory", "category_path"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"Observation requires nonempty {name}")
        parts = row["category_path"].split("/")
        if len(parts) < 2 or parts[:2] != [row["category"], row["subcategory"]] or any(part in ("", ".", "..") for part in parts):
            raise ValueError("Observation classification disagrees with source category path")
        identity = (row["run_id"], row["variable_id"])
        if identity in seen:
            raise ValueError("Duplicate run-variable observation")
        seen.add(identity)
        if type(row.get("terminal_invalid")) is not bool or not isinstance(row.get("attempts"), list):
            raise ValueError("Observation requires terminal_invalid boolean and attempts list")
        if not all(isinstance(attempt, dict) for attempt in row["attempts"]):
            raise ValueError("Attempt evidence must contain mappings")
        if "evaluation" not in row:
            raise ValueError("Observation must explicitly provide evaluation or null")
        evaluation = row["evaluation"]
        gold_hash = None
        if evaluation is not None:
            if not isinstance(evaluation, dict) or evaluation.get("metadata", {}).get("variable_id") != row["variable_id"]:
                raise ValueError("Evaluation metadata.variable_id disagrees with its observation")
            aggregate_items([evaluation], expected_variable_ids=[row["variable_id"]])
            current_backend = (evaluation["scorer_version"], content_hash(evaluation["similarity_identity"]), evaluation["close_threshold"])
            if backend is not None and current_backend != backend:
                raise ValueError("Incompatible scorer/backend identities in observations")
            backend = current_backend
            gold_hash = evaluation["gold_hash"]
        provenance = (row["category"], row["subcategory"], row["category_path"])
        previous = lineage.get(row["variable_id"])
        if previous and (previous[:3] != provenance or (gold_hash and previous[3] and gold_hash != previous[3])):
            raise ValueError("Variable gold or classification changes across observations")
        lineage[row["variable_id"]] = (*provenance, gold_hash or (previous[3] if previous else None))
    return sorted(rows, key=lambda row: (row["run_id"], row["variable_id"]))


def _attempt_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only evidenced calls, errors, usage, latency, and scoped money.

    Args:
        rows: Validated observations with full attempt evidence.
    Returns:
        Summary counts and values with observed/missing denominators; absent money
        remains unavailable, and reasoning tokens are never added to completion tokens.
    Raises:
        ValueError: Available numeric evidence or a non-billed cost receipt is invalid.
    Side Effects:
        None; all original attempt records remain in the parent report.
    """
    attempts = [attempt for row in rows for attempt in row["attempts"]]
    token_values = {name: [] for name in ("input_tokens", "output_tokens", "reasoning_tokens")}
    latencies, costs, unavailable_cost = [], {}, 0
    outcomes: Counter[str] = Counter()
    validation_errors: Counter[str] = Counter()
    invalid_attempts = 0
    for attempt in attempts:
        response = attempt.get("response") if isinstance(attempt.get("response"), dict) else attempt
        outcome = response.get("outcome", attempt.get("outcome", "unavailable"))
        outcomes[str(outcome)] += 1
        validation = attempt.get("validation", {})
        if isinstance(validation, dict):
            errors = validation.get("errors", [])
            if errors:
                invalid_attempts += 1
                for error in errors:
                    key = error.get("code", error.get("kind", "unclassified")) if isinstance(error, dict) else "unclassified"
                    validation_errors[str(key)] += 1
        usage = response.get("usage")
        if isinstance(usage, dict):
            for name, alternatives in (("input_tokens", ("prompt_tokens", "input_tokens")),
                                       ("output_tokens", ("completion_tokens", "output_tokens"))):
                value = next((usage[key] for key in alternatives if usage.get(key) is not None), None)
                if value is not None:
                    numeric = _number(value, name)
                    if numeric != numeric.to_integral_value():
                        raise ValueError("Token counts must be integers")
                    token_values[name].append(int(numeric))
            details = usage.get("completion_tokens_details", usage.get("output_tokens_details", {}))
            value = details.get("reasoning_tokens") if isinstance(details, dict) else None
            if value is not None:
                numeric = _number(value, "reasoning_tokens")
                if numeric != numeric.to_integral_value():
                    raise ValueError("Reasoning token counts must be integers")
                token_values["reasoning_tokens"].append(int(numeric))
        if response.get("latency_seconds") is not None:
            latencies.append(_number(response["latency_seconds"], "latency_seconds"))
        cost = attempt.get("cost")
        if not isinstance(cost, dict) or any(cost.get(key) is None for key in ("mode", "amount", "currency", "basis", "kind")):
            unavailable_cost += 1
            continue
        if cost["mode"] not in {"non_billed", "metered"} or cost["kind"] not in {"actual", "estimated"} or not cost["basis"] or not cost["currency"]:
            raise ValueError("Unknown or unevidenced monetary receipt")
        amount = _number(cost["amount"], "cost")
        if cost["mode"] == "non_billed" and amount != 0:
            raise ValueError("Non-billed receipt must have zero amount")
        identity = (str(cost["currency"]), cost["mode"], cost["kind"], content_hash(cost["basis"]))
        bucket = costs.setdefault(identity, {"currency": cost["currency"], "mode": cost["mode"], "kind": cost["kind"],
                                            "basis": cost["basis"], "amount": Decimal(0), "attempt_count": 0})
        bucket["amount"] += amount
        bucket["attempt_count"] += 1
    return {"attempt_records": len(attempts), "retries_recorded": sum(max(0, len(row["attempts"]) - 1) for row in rows),
        "attempt_distribution": dict(sorted(Counter(str(len(row["attempts"])) for row in rows).items())),
        "outcomes": dict(sorted(outcomes.items())), "content_invalid_attempts": invalid_attempts,
        "validation_error_counts": dict(sorted(validation_errors.items())),
        "tokens": {name: {"reported_total": sum(values), "reported_attempts": len(values),
                           "missing_attempts": len(attempts) - len(values),
                           "total": sum(values) if len(values) == len(attempts) else None}
                   for name, values in token_values.items()},
        "latency_seconds": {"reported_total": str(sum(latencies, Decimal(0))), "reported_attempts": len(latencies),
                            "missing_attempts": len(attempts) - len(latencies),
                            "total": str(sum(latencies, Decimal(0))) if len(latencies) == len(attempts) else None},
        "cost": {"groups": [{**value, "amount": str(value["amount"])} for _, value in sorted(costs.items())],
                 "reported_attempts": len(attempts) - unavailable_cost, "missing_attempts": unavailable_cost,
                 "available": unavailable_cost == 0, "unknown_is_zero": False}}


def _scope_summary(rows: list[dict[str, Any]], expected: list[str] | None = None) -> dict[str, Any]:
    """Build an explicit complete or partial population summary from item receipts.

    Args:
        rows: Validated observations belonging to one run/slice.
        expected: Optional immutable full expected population for this scope.
    Returns:
        Coverage, named micro aggregates, terminal-validity rates, and attempt summary.
    Raises:
        ValueError: Evaluator receipts or population membership are inconsistent.
    Side Effects:
        None.
    """
    scored = [row for row in rows if row["evaluation"] is not None]
    scored_ids = sorted(row["variable_id"] for row in scored)
    expected_ids = sorted(expected) if expected is not None else sorted(row["variable_id"] for row in rows)
    missing = sorted(set(expected_ids) - set(scored_ids))
    invalid = sum(row["terminal_invalid"] for row in scored)
    aggregate = aggregate_items([row["evaluation"] for row in scored], expected_variable_ids=scored_ids) if scored else None
    return {"expected_variables": len(expected_ids), "observed_variables": len(rows), "scored_variables": len(scored),
        "expected_variable_ids": expected_ids, "scored_variable_ids": scored_ids, "missing_variable_ids": missing,
        "complete": not missing and bool(expected_ids), "metric_scope": "complete-population" if not missing else "partial-population-not-rankable",
        "aggregate": aggregate, "terminal_invalid": invalid, "schema_valid": len(scored) - invalid,
        "validity_denominator": len(scored), "schema_valid_rate": _receipt(Fraction(len(scored) - invalid, len(scored))) if scored else None,
        "terminal_invalid_rate": _receipt(Fraction(invalid, len(scored))) if scored else None,
        "attempt_summary": _attempt_summary(rows)}


def build_category_summary(observations: Iterable[Mapping[str, Any]],
                           population_categories: Mapping[str, Mapping[str, str]] | None = None) -> list[dict[str, Any]]:
    """Calculate exact source-category and subcategory micro slices within each run.

    Args:
        observations: Explicit run-variable rows accepted by the reporting boundary.
            Optional provider/model/configuration/repetition context is retained;
            build_configuration_ranking supplies it from the frozen plan.
    Returns:
        Canonically ordered category and subcategory rows with exact aggregates,
        support/coverage, run identity, classification, and source lineage hashes.
    Raises:
        ValueError: Duplicate/incompatible items or ambiguous category provenance.
    Side Effects:
        None; the caller retrieves database evidence and persists results separately.
    """
    rows = _observations(observations)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    contexts: dict[str, dict[str, Any]] = {}
    context_keys = ("provider", "model_id", "configuration_id", "repetition", "reasoning_mode")
    for row in rows:
        contexts.setdefault(row["run_id"], {key: row.get(key) for key in context_keys})
        groups[(row["run_id"], "category", row["category"], "", row["category"])].append(row)
        groups[(row["run_id"], "subcategory", row["category"], row["subcategory"], row["category_path"])].append(row)
    # Expected membership must come from frozen corpus metadata. Deriving it from the
    # rows supplied makes the denominator equal the numerator, so a category missing
    # two of its ten variables reports complete. When no membership is supplied the
    # coverage claim is withheld rather than assumed.
    expected_members: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for variable_id, meta in (population_categories or {}).items():
        expected_members[("category", meta["category"], meta["category"])].append(variable_id)
        expected_members[("subcategory", meta["category"], meta["category_path"])].append(variable_id)
    # A category that produced no rows at all must still appear. Building groups only from
    # observations made a run that scored nothing for a whole category indistinguishable
    # from one where that category does not exist, which silently hides the worst case.
    if population_categories is not None:
        present = {(run_id, scope, category, path)
                   for run_id, scope, category, _subcategory, path in groups}
        for run_id in contexts:
            for scope, category, path in expected_members:
                if (run_id, scope, category, path) not in present:
                    groups[(run_id, scope, category, "", path)] = []
    results = []
    for (run_id, scope, category, subcategory, path), members in sorted(groups.items()):
        context = contexts[run_id]
        if any(any(row.get(key) != value for key, value in context.items()) for row in members):
            raise ValueError("Run context changes inside a category summary")
        record = {"version": "category-summary-v1", "run_id": run_id, **context, "scope": scope,
                  "category": category, "subcategory": subcategory or None, "category_path": path,
                  **_scope_summary(members, expected_members.get((scope, category, path))
                                   if population_categories is not None else None),
                  "coverage_basis": "frozen-population" if population_categories is not None else "observed-rows-only",
                  "source_observation_hashes": sorted(content_hash(row) for row in members)}
        results.append({**record, "sha256": content_hash(record)})
    return results


def build_configuration_ranking(plan: Mapping[str, Any], observations: Iterable[Mapping[str, Any]],
                                *, scorer_identity: Mapping[str, Any] | None = None,
                                population_categories: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, Any]:
    """Rank every complete configuration by mean repetition micro Close F1.

    Args:
        plan: Hash-verified experiment-plan-v1 with explicit configurations, runs,
            population and synthetic/live mode. Only live mode requires 97 variables.
        observations: Every available planned task observation; unresolved tasks
            may be absent or have evaluation=null and always block rankability.
    Returns:
        Hashed report with persistence-compatible policy_version/configurations,
        exact primary receipts and shared competition ranks, every lower/incomplete
        configuration, full observations, repetition metrics, and category summaries.
    Raises:
        ValueError: Corrupt plan/evaluation, unplanned/duplicate task, incompatible
            provenance, invalid repetition membership, or incomplete live population.
        TypeError: Input evidence is not JSON-compatible.
    Side Effects:
        None. No provider, database, clock, model, or filesystem access occurs.
    """
    frozen = json.loads(canonical_json_bytes(dict(plan)))
    if frozen.get("version") != "experiment-plan-v1" or frozen.get("mode") not in {"live", "synthetic"}:
        raise ValueError("Unknown experiment plan version or mode")
    if frozen.get("sha256") != content_hash({key: value for key, value in frozen.items() if key != "sha256"}):
        raise ValueError("Experiment plan hash mismatch")
    population = frozen.get("population", [])
    if (not population or not all(isinstance(value, str) and value for value in population)
            or len(set(population)) != len(population) or frozen.get("population_size") != len(population)
            or (frozen["mode"] == "live" and len(population) != 97)):
        raise ValueError("Invalid declared evaluation population")
    configurations = frozen.get("configurations", [])
    runs = frozen.get("runs", [])
    if not configurations or not runs:
        raise ValueError("Plan must enumerate configurations and repetition runs")
    by_configuration = {row["configuration_id"]: row for row in configurations}
    by_run = {row["id"]: row for row in runs}
    if len(by_configuration) != len(configurations) or len(by_run) != len(runs):
        raise ValueError("Duplicate configuration or run identity")
    configured_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        config = by_configuration.get(run["configuration_id"])
        if config is None or any(run.get(key) != config.get(key) for key in ("provider", "model_id", "reasoning_mode")):
            raise ValueError("Run ownership disagrees with its configuration")
        if type(run.get("repetition")) is not int or run["repetition"] < 1:
            raise ValueError("Repetition identity must be a positive integer")
        configured_runs[run["configuration_id"]].append(run)
    for config_id in by_configuration:
        repetitions = sorted(run["repetition"] for run in configured_runs[config_id])
        if not repetitions or repetitions != list(range(1, len(repetitions) + 1)):
            raise ValueError("Configuration repetition membership is missing or duplicated")
    rows = _observations(observations)
    # `_observations` only proves the evaluations agree with each other, so a run scored
    # end to end with the wrong backend passes it. Bind them to the artifact the plan
    # froze, or the report cannot claim to be that plan's scientific result.
    frozen_scorer = frozen.get("artifact_identities", {}).get("scorer")
    if scorer_identity is not None:
        # Checked unconditionally: an identity from another plan is wrong whether or not
        # any observations happen to have been supplied with it. `plan_scorer` is
        # required, not optional: allowing it to be absent let a self-consistent identity
        # pass without ever being tied to the plan it claims to describe.
        if scorer_identity.get("plan_scorer") != frozen_scorer:
            raise ValueError("Supplied scorer identity does not belong to this plan")
        expected = content_hash({"scorer_version": scorer_identity["scorer_version"],
                                 "similarity_identity": scorer_identity["similarity_identity"],
                                 "close_threshold": scorer_identity["close_threshold"]})
        for row in rows:
            evaluation = row.get("evaluation")
            if not evaluation:
                continue
            observed = content_hash({"scorer_version": evaluation["scorer_version"],
                                     "similarity_identity": evaluation["similarity_identity"],
                                     "close_threshold": evaluation["close_threshold"]})
            if observed != expected:
                raise ValueError("Observations were scored with a different scorer or similarity backend than the plan froze")
    by_observed_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        run = by_run.get(row["run_id"])
        if run is None or row["variable_id"] not in population:
            raise ValueError("Observation lies outside the frozen plan")
        for key in ("provider", "model_id", "configuration_id", "repetition", "reasoning_mode"):
            if key in row and row[key] != run.get(key):
                raise ValueError("Observation run context conflicts with the plan")
            row[key] = run.get(key)
        by_observed_run[row["run_id"]].append(row)
    output_configurations = []
    for config_id, config in sorted(by_configuration.items()):
        repetition_summaries = []
        for run in sorted(configured_runs[config_id], key=lambda item: item["repetition"]):
            repetition_summaries.append({"run_id": run["id"], "repetition": run["repetition"],
                "provider": run["provider"], "model_id": run["model_id"], "reasoning_mode": run.get("reasoning_mode"),
                **_scope_summary(by_observed_run[run["id"]], population)})
        rankable = all(row["complete"] for row in repetition_summaries)
        means = None
        if rankable:
            means = {mode: {metric: _receipt(sum((_fraction(row["aggregate"][mode]["metrics"][metric]) for row in repetition_summaries), Fraction(0)) / len(repetition_summaries))
                            for metric in ("precision", "recall", "f1")} for mode in ("exact", "close")}
        reasons = [f"run:{row['run_id']}:unscored:{','.join(row['missing_variable_ids'])}"
                   for row in repetition_summaries if not row["complete"]]
        output_configurations.append({**config, "rankable": rankable, "rank": None,
            "reason": "; ".join(reasons) if reasons else None, "not_rankable_reasons": reasons,
            "expected_repetitions": len(repetition_summaries), "repetitions": repetition_summaries,
            "mean_repetition_micro": means, "primary": means["close"]["f1"] if means else None,
            "primary_metric": "mean-repetition-micro-close-f1", "policy_version": POLICY_VERSION})
    ranked = sorted((row for row in output_configurations if row["rankable"]),
                    key=lambda row: (-_fraction(row["primary"]), row["configuration_id"]))
    previous, shared_rank = None, None
    for position, row in enumerate(ranked, 1):
        value = _fraction(row["primary"])
        if value != previous:
            shared_rank = position
        row["rank"] = shared_rank
        previous = value
    ordered = ranked + sorted((row for row in output_configurations if not row["rankable"]), key=lambda row: row["configuration_id"])
    report = {"schema_version": "1.0", "version": REPORT_VERSION, "policy_version": POLICY_VERSION,
        "plan_sha256": frozen["sha256"], "mode": frozen["mode"],
        "synthetic": frozen["mode"] == "synthetic", "artifact_identities": frozen.get("artifact_identities", {}),
        "population": sorted(population), "population_sha256": content_hash(sorted(population)),
        "population_size": len(population), "complete": all(row["rankable"] for row in ordered),
        "configurations": ordered, "configuration_count": len(ordered), "rankable_count": len(ranked),
        "observations": rows, "source_observation_hashes": sorted(content_hash(row) for row in rows),
        "category_summaries": build_category_summary(rows, population_categories),
        "scorer_binding": "verified-against-plan" if scorer_identity is not None else "unverified",
        "deferred_statistics": "No within-configuration variance or confidence interval is inferred from a singleton repetition."}
    return {**report, "sha256": content_hash(report)}


def _csv_cell(value: Any) -> str:
    """Escape every potentially executable spreadsheet text cell conservatively.

    Args:
        value: Scalar or nested JSON cell value.
    Returns:
        Text with a leading apostrophe for formula/control-character prefixes;
        nested evidence uses canonical JSON and numbers are never rounded here.
    Raises:
        TypeError: Nested evidence is not JSON-compatible.
    Side Effects:
        None.
    """
    if value is None:
        text = ""
    elif isinstance(value, (dict, list)):
        text = canonical_json_bytes(value).decode("utf-8")
    else:
        text = str(value)
    if text and (text[0] in "\t\r\n" or text.lstrip().startswith(("=", "+", "-", "@"))):
        text = "'" + text
    return text


def _render_csv(report: Mapping[str, Any]) -> tuple[bytes, list[str], int]:
    """Render one complete source-preserving CSV row per configuration.

    Args:
        report: Validated configuration report including full observations.
    Returns:
        UTF-8 CSV bytes, stable column names, and data-row count.
    Raises:
        TypeError: A report cell cannot be canonically rendered.
    Side Effects:
        None; rendering uses an in-memory buffer only.
    """
    columns = ["report_sha256", "plan_sha256", "mode", "configuration_id", "provider", "model_id",
               "reasoning_mode", "prompt_variant", "shot_count", "temperature", "rankable", "rank", "reason"]
    columns += [f"mean_repetition_micro_{mode}_{metric}_fraction" for mode in ("exact", "close") for metric in ("precision", "recall", "f1")]
    columns += ["configuration_record_json", "source_observations_json"]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for configuration in report["configurations"]:
        row = {"report_sha256": report["sha256"], "plan_sha256": report["plan_sha256"], "mode": report["mode"], **configuration}
        for mode in ("exact", "close"):
            for metric in ("precision", "recall", "f1"):
                receipt = configuration["mean_repetition_micro"][mode][metric] if configuration["mean_repetition_micro"] else None
                row[f"mean_repetition_micro_{mode}_{metric}_fraction"] = f"{receipt['numerator']}/{receipt['denominator']}" if receipt else None
        row["configuration_record_json"] = configuration
        row["source_observations_json"] = [observation for observation in report["observations"] if observation["configuration_id"] == configuration["configuration_id"]]
        writer.writerow([_csv_cell(row.get(column)) for column in columns])
    return stream.getvalue().encode("utf-8"), columns, len(report["configurations"])


def _safe_destination(destination: Path | str, outputs_root: Path | str) -> tuple[Path, Path]:
    """Resolve an explicit derivative file strictly beneath a non-symlink root.

    Args:
        destination: Absolute output filename or relative name under outputs_root.
        outputs_root: Explicit permitted output directory, created if absent.
    Returns:
        Resolved safe destination and resolved root.
    Raises:
        ValueError: Path escapes the root, names a directory, or traverses a symlink.
        OSError: The permitted output directories cannot be created.
    Side Effects:
        Creates only required directories beneath the explicit output root.
    """
    root = Path(outputs_root).absolute()
    if root.is_symlink():
        raise ValueError("Output root may not be a symlink")
    root = root.resolve()
    candidate = Path(destination)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Export destination escapes outputs root") from exc
    if not relative.parts or any(part in ("..", ".") for part in relative.parts):
        raise ValueError("Export destination must name a file beneath outputs root")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Export destination traverses a symlink")
    if candidate.exists() and not candidate.is_file():
        raise ValueError("Export destination is not a regular file")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate, root


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish an idempotent immutable derivative through a flushed temporary file.

    Args:
        path: Previously validated destination under the explicit outputs root.
        data: Complete canonical bytes to publish.
    Returns:
        None after verifying stored bytes match the intended content.
    Raises:
        FileExistsError: A different derivative already occupies the destination.
        OSError: Writing, atomic publication, synchronization, or verification fails.
    Side Effects:
        Writes the named derivative and a temporary sibling, removing only its
        own temporary file. Existing different artifacts are never overwritten.
    """
    if path.exists():
        if path.read_bytes() == data:
            return
        raise FileExistsError("Different report artifact exists; choose a new destination")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".iadopt-report-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # link publishes without a check/replace race that could overwrite another writer.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FileExistsError("Concurrent different report artifact exists") from None
        if path.read_bytes() != data:
            raise OSError("Published report bytes failed verification")
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_report(report: Mapping[str, Any], format: str, destination: Path | str,
                  *, outputs_root: Path | str) -> dict[str, Any]:
    """Export canonical JSON or formula-safe CSV plus a deterministic byte manifest.

    Args:
        report: Complete hashed configuration-report-v1 from the pure builder;
            incomplete configurations are valid report content and remain visible.
        format: 'canonical-json' (or 'json') or 'csv'; XLSX is deliberately optional.
        destination: Named derivative file constrained beneath outputs_root.
        outputs_root: Explicit permitted output directory.
    Returns:
        Manifest with schema_version, kind, file hashes/lengths, source plan/report
        identities, exporter code hash, stable columns, row count, and manifest path.
    Raises:
        ValueError: Unsupported format, corrupt report, or unsafe output path.
        FileExistsError: Different artifact or manifest already exists at the target.
        OSError: Atomic write or byte verification fails.
    Side Effects:
        Writes only the named derivative and sibling .manifest.json under outputs_root;
        reads exporter source bytes for provenance. Never writes database evidence.
    """
    frozen = json.loads(canonical_json_bytes(dict(report)))
    if frozen.get("version") != REPORT_VERSION or frozen.get("policy_version") != POLICY_VERSION:
        raise ValueError("Unknown report or ranking policy version")
    if frozen.get("sha256") != content_hash({key: value for key, value in frozen.items() if key != "sha256"}):
        raise ValueError("Report hash mismatch")
    if format in {"canonical-json", "json"}:
        format = "canonical-json"
        data, columns, row_count = canonical_json_bytes(frozen), [], len(frozen["configurations"])
    elif format == "csv":
        data, columns, row_count = _render_csv(frozen)
    else:
        raise ValueError("Supported export formats are canonical-json and csv")
    path, root = _safe_destination(destination, outputs_root)
    manifest_path, _ = _safe_destination(path.with_name(path.name + ".manifest.json"), root)
    manifest = {"schema_version": "1.0", "kind": "experiment-report-export", "exporter_version": EXPORT_VERSION,
        "exporter_code_sha256": sha256_bytes(Path(__file__).read_bytes()), "format": format,
        "report_sha256": frozen["sha256"], "plan_sha256": frozen["plan_sha256"],
        "policy_version": POLICY_VERSION, "mode": frozen["mode"], "final": frozen["complete"],
        "files": [{"path": path.relative_to(root).as_posix(), "sha256": sha256_bytes(data), "byte_length": len(data)}],
        "row_count": row_count, "columns": columns, "manifest_path": manifest_path.relative_to(root).as_posix()}
    manifest["sha256"] = content_hash(manifest)
    manifest_bytes = canonical_json_bytes(manifest)
    for target, contents in ((path, data), (manifest_path, manifest_bytes)):
        if target.exists() and target.read_bytes() != contents:
            raise FileExistsError("Different report artifact/manifest exists; choose a new destination")
    _atomic_write(path, data)
    _atomic_write(manifest_path, manifest_bytes)
    return manifest
