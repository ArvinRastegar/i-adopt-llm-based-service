"""Build the cost-estimate evidence file that `estimate` consumes.

`estimate_campaign_cost` refuses to guess: it needs a real token count for every
rendered prompt, an evidenced billing card per model, and explicit retry and
ceiling assumptions. Assembling that by hand does not scale past one model, so
this module derives every part that IS derivable — prompt tokens by tokenizing
the exact prompts the runner will send, billing cards from the provider block
that already carries its own basis and provenance — and leaves the rest as
explicit operator input.

What it deliberately does not derive is the ceiling verification. Asserting that
an output ceiling covers a model's all-in behaviour is a claim about a
deployment, not a calculation, so `ceiling_verified` is false until an operator
says otherwise and supplies the basis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .domain import LabError

# Matches the estimate the first live campaigns used. Qwen3's byte-level BPE is a
# consistent estimator across these deployments; PSNC publishes no per-model tokenizer.
DEFAULT_TOKENIZER = "Qwen/Qwen3-32B"


def _tokenizer(name: str) -> tuple[Any, dict]:
    """Load a local tokenizer without contacting a model hub, and identify it.

    Naming a tokenizer does not identify it: two caches can hold different revisions
    under the same id, and the token counts would differ silently. The returned identity
    records what was actually loaded so a count can be reproduced or refuted.

    Args: name: Hub id of a tokenizer already present in the local cache.
    Returns: (loaded fast tokenizer, identity mapping).
    Raises: LabError when the tokenizer is not available offline.
    Side effects: Reads the local Hugging Face cache. Never downloads.
    """
    try:
        import transformers
        from transformers import AutoTokenizer

        loaded = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as error:  # noqa: BLE001 - surfaced as an operator-facing failure
        raise LabError(
            f"Tokenizer {name} is not available offline ({type(error).__name__}). "
            "Pass --tokenizer with a locally cached id.") from error
    return loaded, {"id": name, "class": type(loaded).__name__,
                    "vocab_size": int(getattr(loaded, "vocab_size", 0)),
                    "transformers_version": transformers.__version__}


def prompt_tokens_by_run(plan: dict, root: Any, tokenizer_id: str,
                         tokenizer: Any = None) -> tuple[dict[str, list[int]], dict]:
    """Tokenize every prompt the plan will send, once per distinct prompt shape.

    A prompt depends on the variant, the shot count and the target, never on the model,
    so runs that share a shape share one token vector rather than re-rendering it.

    Args: plan: Expanded plan; root: lab root; tokenizer_id: locally cached tokenizer;
        tokenizer: an already-loaded encoder, injected instead of read from the local cache.
    Returns: (mapping of run id to per-target token counts in population order, evidence mapping).
    Raises: LabError for an unavailable tokenizer or a target missing from the corpus.
    Side effects: Reads prompt, schema and corpus files; performs no provider call.
    """
    from .corpus.ingestion import load_canonical_records
    from .prompting.renderer import load_prompt_version, render_base_prompt
    from .validation import load_schema_bytes

    if tokenizer is None:
        tokenizer, tokenizer_identity = _tokenizer(tokenizer_id)
    else:
        tokenizer_identity = {"id": tokenizer_id, "class": type(tokenizer).__name__,
                              "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
                              "injected": True}
    records = list(load_canonical_records(root))
    by_id = {row["variable_id"]: row for row in records}
    demonstrations = tuple(sorted((row for row in records if row["demonstration_position"]),
                                  key=lambda row: row["demonstration_position"]))
    schema = load_schema_bytes(root)

    missing = [target for target in plan["population"] if target not in by_id]
    if missing:
        raise LabError(f"{len(missing)} planned targets are absent from the materialized corpus")

    shapes: dict[tuple[str, int], list[int]] = {}
    for run in plan["runs"]:
        shape = (run["prompt_variant"], run["shot_count"])
        if shape in shapes:
            continue
        template = load_prompt_version(shape[0], root)
        shapes[shape] = [
            len(tokenizer.encode(render_base_prompt(
                template, by_id[target]["definition"], schema,
                demonstrations[:shape[1]], target_id=target).content))
            for target in plan["population"]]

    evidence = {"method": "tokenization of every rendered prompt in population order",
                "tokenizer": tokenizer_id, "tokenizer_identity": tokenizer_identity,
                "distinct_prompt_shapes": [{"prompt_variant": variant, "shot_count": shots}
                                           for variant, shots in sorted(shapes)],
                "note": ("Prompts depend on variant, shot count and target only, so models "
                         "sharing a prompt shape share one measured token vector."),
                "limitations": [
                    "Counts cover rendered message content only. Chat-template markers and "
                    "special tokens are not included here; admission adds them separately as "
                    "the message-overhead allowance.",
                    "One tokenizer stands in for every deployment, because PSNC publishes no "
                    "per-model tokenizer. These are consistent estimates against a recorded "
                    "local tokenizer, not provider-reported input counts."]}
    return ({run["id"]: shapes[(run["prompt_variant"], run["shot_count"])] for run in plan["runs"]},
            evidence)


def billing_cards(plan: dict, parameters: dict, root: Any) -> dict:
    """Read the frozen price card execution will use, and check it covers the plan.

    The estimate must not be able to claim a campaign is affordable using different price
    evidence from the one the runner reserves against. So this reads the same card
    `cost_policy` loads rather than deriving cards from the provider declaration, and
    treats a model missing from it as a hard failure with the message the runner would
    have raised later. The provider declaration is used only to cross-check the mode.

    Args: plan: Expanded plan; parameters: resolved configuration data; root: lab root.
    Returns: A billing_evidence mapping keyed provider/model_id.
    Raises: LabError when the card is unreadable, incomplete, or disagrees with the
        provider's declared billing mode.
    Side effects: Reads the price-card manifest.
    """
    import yaml

    accounting = parameters.get("cost_accounting") or {}
    manifest = accounting.get("price_card_manifest")
    if not manifest:
        raise LabError("cost_accounting.price_card_manifest is null; frozen price evidence is required")
    card = yaml.safe_load((Path(root) / manifest).read_bytes()) or {}
    entries = card.get("models") or {}
    currency = card.get("currency")
    if not currency:
        raise LabError(f"Price card {manifest} declares no reporting currency")
    declared = accounting.get("reporting_currency")
    if declared and declared != currency:
        raise LabError(f"Price card currency {currency} disagrees with "
                       f"cost_accounting.reporting_currency {declared}")

    providers = parameters.get("providers") or {}
    keys = {f"{count['provider']}/{count['model_id']}" for count in plan["counts"]["by_model"]}
    absent = sorted(keys - set(entries))
    if absent:
        raise LabError(f"Frozen price card {manifest} has no entry for: " + ", ".join(absent)
                       + ". Every planned model needs price evidence before dispatch.")

    models: dict[str, Any] = {}
    for count in plan["counts"]["by_model"]:
        key = count["provider"] + "/" + count["model_id"]
        entry = dict(entries[key])
        if not entry.get("mode") or not entry.get("basis") or not entry.get("provenance"):
            raise LabError(f"{key}: price card entry needs mode, basis and provenance")
        billing = (providers.get(count["provider"]) or {}).get("billing") or {}
        if billing.get("mode") and billing["mode"] != entry["mode"]:
            raise LabError(f"{key}: price card says {entry['mode']} while "
                           f"providers.{count['provider']}.billing says {billing['mode']}")
        if entry["mode"] == "metered" and any(
                entry.get(field) is None
                for field in ("input_per_million", "output_per_million", "fx_to_reporting")):
            raise LabError(f"{key}: metered entry needs input/output rates and an FX factor")
        entry.setdefault("fx_to_reporting", "1")
        models[key] = entry
    return {"currency": currency, "models": models}


def plan_output_ceiling(plan: dict, parameters: dict) -> int:
    """Take the all-in output ceiling from the plan, and refuse a plan/parameter mismatch.

    Reading the ceiling from current parameters lets an old plan acquire a nominally
    plan-bound number it was never expanded with: the runs would send one limit while the
    estimate bounded a different one. The plan is authoritative because it is what the
    runner dispatches; disagreement with today's parameters means the plan is stale.

    Args: plan: Expanded plan; parameters: resolved configuration data.
    Returns: The single output ceiling every planned run shares.
    Raises: LabError when runs disagree, or when the plan and parameters disagree.
    Side effects: None.
    """
    ceilings = {run.get("max_output_tokens") for run in plan["runs"]}
    if len(ceilings) != 1 or None in ceilings:
        raise LabError("Planned runs do not share one explicit output ceiling")
    ceiling = int(ceilings.pop())
    configured = (parameters.get("parameter_grid") or {}).get("max_output_tokens")
    if configured is None:
        raise LabError("parameter_grid.max_output_tokens is null; an all-in ceiling is required")
    if int(configured) != ceiling:
        raise LabError(f"Plan was expanded with an output ceiling of {ceiling} but parameters now "
                       f"declare {configured}. Re-expand the plan instead of estimating a stale one.")
    return ceiling


def reasoning_accounting(plan: dict) -> str:
    """State how reasoning tokens are accounted for, from the plan's own reasoning modes.

    The three modes make different claims, and collapsing them would overstate the
    evidence. `disabled` sends a control that was measured to silence reasoning.
    `not_applicable` sends no control at all: it records that no switch exists, which is
    not the same as establishing the model never reasons. `enabled` expects reasoning.

    Args: plan: Expanded plan.
    Returns: A sentence describing the reasoning mode of every planned run.
    Raises: Nothing.
    Side effects: None.
    """
    modes: dict[str, list[str]] = {}
    for run in plan["runs"]:
        modes.setdefault(run["reasoning_mode"], []).append(f"{run['provider']}/{run['model_id']}")
    described = "; ".join(
        f"{mode}: {', '.join(sorted(set(keys)))}" for mode, keys in sorted(modes.items()))
    claims = {
        "disabled": ("a measured control suppresses reasoning, so reasoning tokens are not "
                     "expected in the billed output"),
        "not_applicable": ("no reasoning control exists to send; this records the absence of a "
                           "switch, NOT a measurement that the model never reasons, so reasoning "
                           "tokens remain possible and are covered by the output ceiling"),
        "enabled": "reasoning is requested and its tokens are expected within the output ceiling",
    }
    stated = " ".join(f"For {mode}, {claims[mode]}." for mode in sorted(modes) if mode in claims)
    return ("Reasoning mode per planned run - " + described + ". " + stated
            + " Any reasoning tokens a provider does report are recorded in the stored usage and "
              "counted once, within output tokens, never separately.")


def build_cost_evidence(plan: dict, parameters: dict, *, root: Any,
                        tokenizer_id: str = DEFAULT_TOKENIZER, expected_output_tokens: int = 150,
                        attempt2_fraction: str = "0.15", attempt3_fraction: str = "0.05",
                        correction_error_tokens: int = 800, ceiling_basis: str | None = None,
                        tokenizer: Any = None) -> dict:
    """Assemble the complete evidence document `estimate` consumes.

    Two things are deliberately not derived. The ceiling assertion needs a specific basis
    naming the observation behind it, because asserting that a ceiling bounds a
    deployment's all-in output is a claim about that deployment rather than a
    calculation; a generic default sentence would launder an unmade assertion into
    evidence. And billing comes from the frozen price card the runner reserves against,
    so an estimate cannot report a campaign ready on evidence execution does not have.

    Args: plan: Expanded plan; parameters: resolved configuration data; root: lab root;
        tokenizer_id: locally cached tokenizer; expected_output_tokens/attempt2_fraction/
        attempt3_fraction/correction_error_tokens: operator retry and output assumptions;
        ceiling_basis: the specific evidence that the planned ceiling bounds all-in output.
        Without it the estimate reports the ceiling unverified, blocking a metered campaign
        and leaving a non-billed one with a recorded warning (see `estimate_campaign_cost`);
        tokenizer: an already-loaded encoder, injected instead of read from the local cache.
    Returns: A mapping with prompt_artifacts, billing_evidence and assumptions.
    Raises: LabError for an unavailable tokenizer, a stale plan, or incomplete price evidence.
    Side effects: Reads corpus, prompt, schema and price-card files. No provider or network call.
    """
    ceiling = plan_output_ceiling(plan, parameters)
    billing = billing_cards(plan, parameters, root)
    tokens, token_evidence = prompt_tokens_by_run(plan, root, tokenizer_id, tokenizer)
    if expected_output_tokens > ceiling:
        raise LabError(f"Expected output {expected_output_tokens} exceeds the planned ceiling {ceiling}")
    assumptions = {
        "expected_output_tokens": expected_output_tokens,
        "attempt2_fraction": attempt2_fraction, "attempt3_fraction": attempt3_fraction,
        "correction_error_tokens": correction_error_tokens,
        "total_output_token_ceiling": ceiling,
        "ceiling_verified": bool(ceiling_basis),
        "reasoning_accounting_evidence": reasoning_accounting(plan)}
    if ceiling_basis:
        assumptions["ceiling_basis"] = ceiling_basis
    return {"prompt_artifacts": {"plan_sha256": plan["sha256"], "evidence": token_evidence,
                                 "input_tokens_by_run": tokens},
            "billing_evidence": billing, "assumptions": assumptions}
