"""Exact-decimal, plan-bound estimates; never an authorization or fabricated price."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from .canonical import content_hash
from .domain import LabError


def decimal_value(value: str | int | float | Decimal) -> Decimal:
    """Validate a nonnegative finite decimal quantity without binary-float arithmetic.

    Args: value: Explicit amount, rate or token estimate, preferably a decimal string.
    Returns: Exact Decimal parsed from its textual representation.
    Raises: LabError for boolean, negative, non-finite or malformed input.
    Side Effects: None; deterministic.
    """
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise LabError("Invalid decimal quantity") from None
    if isinstance(value, bool) or not result.is_finite() or result < 0:
        raise LabError("Decimal quantities must be finite and nonnegative")
    return result


def estimate_campaign_cost(plan: dict, prompt_artifacts: dict, billing_evidence: dict,
                           assumptions: dict) -> dict:
    """Calculate expected and conditional three-attempt scenarios from frozen evidence.

    Args:
        plan: Expanded plan with per-model counts and SHA-256.
        prompt_artifacts: plan_sha256, evidence and input_tokens_by_run mapping.
            Each run's token list covers all rendered target prompts in population order.
        billing_evidence: Reporting currency and model cards keyed provider/model_id.
            Each card has mode/basis/provenance, input_per_million/output_per_million,
            and fx_to_reporting. Unknown rates are not zero.
        assumptions: expected_output_tokens, attempt2_fraction, attempt3_fraction,
            correction_error_tokens, total_output_token_ceiling, ceiling_verified and
            reasoning_accounting_evidence. Output includes billed reasoning tokens.
    Returns:
        Hashed estimate with provider/model totals, assumptions and readiness issues.
        A conditional bound requires an evidenced all-in output ceiling, not a guess.
    Raises:
        LabError: Mismatched plan, invalid quantities or inconsistent retry fractions.
    Side Effects:
        None; no calibration, price fetch, persistence, approval or cap enforcement.
    """
    if prompt_artifacts.get("plan_sha256") != plan["sha256"]:
        raise LabError("Prompt estimate is not bound to this plan")
    issues, rows = [], []
    currency = billing_evidence.get("currency")
    if not currency:
        issues.append("Reporting currency is missing")
    if not prompt_artifacts.get("evidence"):
        issues.append("Prompt tokenizer/estimation evidence is missing")
    required = ("expected_output_tokens", "attempt2_fraction", "attempt3_fraction",
                "correction_error_tokens", "total_output_token_ceiling")
    missing = [name for name in required if assumptions.get(name) is None]
    issues.extend("Missing assumption: " + name for name in missing)
    values = {name: decimal_value(assumptions[name]) for name in required if name not in missing}
    if "attempt2_fraction" in values and "attempt3_fraction" in values:
        if not 0 <= values["attempt3_fraction"] <= values["attempt2_fraction"] <= 1:
            raise LabError("Retry fractions require 0 <= attempt3 <= attempt2 <= 1")
    if not assumptions.get("reasoning_accounting_evidence"):
        issues.append("Billed reasoning/output-token accounting is unverified")
    if assumptions.get("ceiling_verified") is not True:
        issues.append("All-in output/reasoning ceiling is not verified")
    by_run = prompt_artifacts.get("input_tokens_by_run", {})
    for count in plan["counts"]["by_model"]:
        key = count["provider"] + "/" + count["model_id"]
        card = billing_evidence.get("models", {}).get(key)
        row = {**count, "billing_key": key, "expected_cost": None, "conditional_maximum_cost": None}
        tokens = []
        for run in plan["runs"]:
            if (run["provider"], run["model_id"]) != (count["provider"], count["model_id"]):
                continue
            run_tokens = by_run.get(run["id"])
            if not isinstance(run_tokens, list) or len(run_tokens) != plan["population_size"]:
                issues.append(key + ": full rendered prompt token coverage missing")
                continue
            tokens.extend(decimal_value(token) for token in run_tokens)
        row["initial_input_tokens"] = str(sum(tokens)) if len(tokens) == count["initial_calls"] else None
        if not card or not card.get("basis") or not card.get("provenance"):
            issues.append(key + ": evidenced price/billing basis missing")
        elif card.get("mode") == "non_billed":
            row.update(expected_cost="0", conditional_maximum_cost="0", billing=card)
        elif card.get("mode") != "metered":
            issues.append(key + ": unknown billing mode")
        elif any(card.get(field) is None for field in ("input_per_million", "output_per_million", "fx_to_reporting")):
            issues.append(key + ": input/output price or FX missing")
        elif not missing and len(tokens) == count["initial_calls"]:
            rate_in = decimal_value(card["input_per_million"]) / Decimal(1000000)
            rate_out = decimal_value(card["output_per_million"]) / Decimal(1000000)
            fx = decimal_value(card["fx_to_reporting"])
            if fx == 0:
                raise LabError("FX rate must be positive")
            n, base = Decimal(count["initial_calls"]), sum(tokens)
            out, errors, ceiling = (values[name] for name in
                ("expected_output_tokens", "correction_error_tokens", "total_output_token_ceiling"))
            if out > ceiling:
                raise LabError("Expected output exceeds declared all-in ceiling")
            a2, a3 = values["attempt2_fraction"], values["attempt3_fraction"]
            # Each correction uses the base plus immediately previous raw answer/errors.
            expected_in = base + (a2 + a3) * (base + n * (out + errors))
            expected_out = n * out * (1 + a2 + a3)
            bounded_in = 3 * base + 2 * n * (ceiling + errors)
            bounded_out = 3 * n * ceiling
            row.update(expected_cost=str((expected_in * rate_in + expected_out * rate_out) * fx),
                conditional_maximum_cost=str((bounded_in * rate_in + bounded_out * rate_out) * fx),
                expected_input_tokens=str(expected_in), expected_output_tokens=str(expected_out),
                conditional_input_tokens=str(bounded_in), conditional_output_tokens=str(bounded_out), billing=card)
        rows.append(row)
    by_provider = []
    for name in sorted({row["provider"] for row in rows}):
        members = [row for row in rows if row["provider"] == name]
        by_provider.append({"provider": name, "initial_calls": sum(row["initial_calls"] for row in members),
            "maximum_calls": sum(row["maximum_calls"] for row in members),
            **{field: str(sum(Decimal(row[field]) for row in members))
               if all(row[field] is not None for row in members) else None
               for field in ("expected_cost", "conditional_maximum_cost")}})
    estimate = {"version": "pre-run-estimate-v1", "plan_sha256": plan["sha256"], "ready": not issues,
        "currency": currency, "issues": sorted(set(issues)), "assumptions": assumptions,
        "prompt_evidence_sha256": content_hash(prompt_artifacts), "billing_evidence": billing_evidence,
        "by_model": rows, "by_provider": by_provider,
        "totals": {"initial_calls": plan["counts"]["initial_calls"], "maximum_calls": plan["counts"]["maximum_calls"],
            **{field: str(sum(Decimal(row[field]) for row in rows))
               if not issues and all(row[field] is not None for row in rows) else None
               for field in ("expected_cost", "conditional_maximum_cost")}},
        "warning": "Conditional estimate, not a spending cap or live authorization. Actual cost may exceed estimates."}
    return {**estimate, "sha256": content_hash(estimate)}
