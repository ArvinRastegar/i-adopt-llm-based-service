"""Build the cost-estimate evidence file that `estimate` consumes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .domain import LabError

# Matches the estimate the first live campaigns used. Qwen3's byte-level BPE is a
# consistent estimator across these deployments; PSNC publishes no per-model tokenizer.
DEFAULT_TOKENIZER = "Qwen/Qwen3-32B"


def _tokenizer(name: str) -> tuple[Any, dict]:
    """Load a local tokenizer without contacting a model hub, and identify it."""
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
    """Tokenize every prompt the plan will send, once per distinct prompt shape."""
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
    evidence from the one the runner reserves against."""
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
    """Take the all-in output ceiling from the plan, and refuse a plan/parameter mismatch."""
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
    """State how reasoning tokens are accounted for, from the plan's own reasoning modes."""
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
    """Assemble the complete evidence document `estimate` consumes."""
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
