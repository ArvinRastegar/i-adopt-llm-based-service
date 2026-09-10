"""Exact-decimal estimates, retry correction sizes and explicit non-billed evidence."""

import copy
from decimal import Decimal
from pathlib import Path

import pytest

from iadopt_lab.configuration import load_parameters
from iadopt_lab.costing import decimal_value, estimate_campaign_cost
from iadopt_lab.domain import LabError
from iadopt_lab.planning import expand_campaign, synthetic_configuration

ROOT = Path(__file__).resolve().parents[2]


def _estimate_inputs():
    """Build a six-task/two-provider synthetic plan with fully artificial price evidence.

    Args: none. Returns: (plan,prompt_evidence,billing,assumptions) fixture tuple.
    Raises: fixture configuration errors. Side effects: public parameter read only.
    """
    config = synthetic_configuration(load_parameters(ROOT / "parameters.yml"))
    identities = {key: key + "-fixture" for key in ("corpus", "population", "prompts", "schema", "scorer", "runtime")}
    targets = [{"variable_id": "fixture-" + str(number)} for number in range(3)]
    plan = expand_campaign(config, targets, artifact_identities=identities, mode="synthetic")
    prompts = {"plan_sha256": plan["sha256"], "evidence": "fixed300 artificial tokens, not live tokenizer evidence",
               "input_tokens_by_run": {run["id"]: [300, 300, 300] for run in plan["runs"]}}
    billing = {"currency": "USD", "models": {
        "psnc/synthetic-fixture-v1": {"mode": "non_billed", "basis": "artificial free-account fixture", "provenance": "offline fixture"},
        "openrouter/synthetic-fixture-v1": {"mode": "metered", "basis": "artificial price card", "provenance": "offline fixture", "input_per_million": "0.1", "output_per_million": "0.2", "fx_to_reporting": "1"},
    }}
    assumptions = {"expected_output_tokens": 300, "attempt2_fraction": "0", "attempt3_fraction": "0",
                   "correction_error_tokens": 50, "total_output_token_ceiling": 600,
                   "ceiling_verified": True, "reasoning_accounting_evidence": "artificial all-in fixture"}
    return plan, prompts, billing, assumptions


def test_exact_cost_excludes_nonbilled_provider_but_retains_calls():
    """Calculate hand-checked costs while retaining PSNC request counts at explicit zero charge.

    Args: none. Returns: None.
    Raises: AssertionError on Decimal arithmetic or non-billed data loss.
    Side effects: public YAML read only; no price fetch or live call.
    """
    result = estimate_campaign_cost(*_estimate_inputs())
    assert result["ready"] and result["totals"]["initial_calls"] == 6
    assert result["totals"]["maximum_calls"] == 18
    assert Decimal(result["totals"]["expected_cost"]) == Decimal("0.00027")
    assert Decimal(result["totals"]["conditional_maximum_cost"]) == Decimal("0.00174")
    psnc = next(row for row in result["by_provider"] if row["provider"] == "psnc")
    assert psnc["initial_calls"] == 3 and psnc["expected_cost"] == "0"
    assert "not a spending cap or live authorization" in result["warning"]


def test_retry_input_contains_base_and_immediate_previous_output():
    """Count correction-request input growth instead of multiplying only original call cost.

    Args: none. Returns: None.
    Raises: AssertionError on incorrect retry cost model. Side effects: YAML read only.
    """
    plan, prompts, billing, assumptions = _estimate_inputs()
    assumptions.update(attempt2_fraction="0.5", attempt3_fraction="0.25")
    result = estimate_campaign_cost(plan, prompts, billing, assumptions)
    paid = next(row for row in result["by_model"] if row["provider"] == "openrouter")
    assert Decimal(paid["expected_input_tokens"]) == Decimal("2362.50")
    assert Decimal(paid["expected_output_tokens"]) == Decimal("1575.00")
    assert Decimal(paid["expected_cost"]) == Decimal("0.00055125")


@pytest.mark.parametrize("change", ["price", "fx", "billing_basis", "token_coverage", "ceiling", "reasoning", "output_assumption"])
def test_missing_evidence_never_becomes_zero_or_ready(change):
    """Keep missing rates/tokens/ceilings visible as unavailable or blocked estimates.

    Args: missing-evidence mutation name. Returns: None.
    Raises: AssertionError on silently invented zero/ready state. Side effects: YAML read only.
    """
    plan, prompts, billing, assumptions = _estimate_inputs()
    paid = billing["models"]["openrouter/synthetic-fixture-v1"]
    if change == "price":
        paid["input_per_million"] = None
    elif change == "fx":
        paid["fx_to_reporting"] = None
    elif change == "billing_basis":
        paid["basis"] = None
    elif change == "token_coverage":
        run = next(run for run in plan["runs"] if run["provider"] == "openrouter")
        prompts["input_tokens_by_run"][run["id"]] = [300]
    elif change == "ceiling":
        assumptions["ceiling_verified"] = False
    elif change == "reasoning":
        assumptions["reasoning_accounting_evidence"] = None
    else:
        assumptions["expected_output_tokens"] = None
    result = estimate_campaign_cost(plan, prompts, billing, assumptions)
    assert not result["ready"] and result["issues"]
    assert result["totals"]["expected_cost"] is None


def test_estimate_plan_binding_and_deterministic_identity():
    """Bind every estimate to exact plan and evidence; changes require a new identity.

    Args: none. Returns: None.
    Raises: AssertionError for stale-plan or hash errors. Side effects: YAML read only.
    """
    inputs = _estimate_inputs()
    first = estimate_campaign_cost(*inputs)
    assert first == estimate_campaign_cost(*copy.deepcopy(inputs))
    plan, prompts, billing, assumptions = inputs
    assumptions["expected_output_tokens"] = 301
    assert estimate_campaign_cost(plan, prompts, billing, assumptions)["sha256"] != first["sha256"]
    prompts["plan_sha256"] = "different-plan"
    with pytest.raises(LabError):
        estimate_campaign_cost(plan, prompts, billing, assumptions)


@pytest.mark.parametrize("a2,a3", [("0.25", "0.5"), ("1.1", "0"), ("0", "-0.1")])
def test_invalid_retry_fractions(a2, a3):
    """Reject impossible continuation fractions before calculating an estimate.

    Args: invalid second/third-attempt fractions. Returns: None.
    Raises: AssertionError on invalid estimate acceptance. Side effects: YAML read only.
    """
    plan, prompts, billing, assumptions = _estimate_inputs()
    assumptions.update(attempt2_fraction=a2, attempt3_fraction=a3)
    with pytest.raises(LabError):
        estimate_campaign_cost(plan, prompts, billing, assumptions)


@pytest.mark.parametrize("value", [True, False, "NaN", "Infinity", "-1", "not-number"])
def test_invalid_money_quantities(value):
    """Reject boolean, negative, malformed and non-finite monetary values.

    Args: invalid numeric input. Returns: None.
    Raises: AssertionError on unsafe Decimal acceptance. Side effects: none.
    """
    with pytest.raises(LabError):
        decimal_value(value)


def test_price_card_must_cover_every_planned_model(tmp_path):
    """The blocker: a ready zero-cost estimate preceded a mid-run crash on model two.

    `cost_policy` raises for an uncovered model, and it raises from inside the worker
    pool where the failure cancels unrelated in-flight tasks. The check therefore has to
    happen once, up front, against the whole plan - and a non-billed provider is not
    exempt, because an evidenced zero is still evidence.
    """
    import yaml

    from iadopt_lab.cli import _live_services
    from iadopt_lab.domain import LabError

    card = tmp_path / "card.yml"
    card.write_text(yaml.safe_dump({"currency": "USD", "models": {"psnc/Covered": {
        "mode": "non_billed", "basis": "free", "provenance": "declared",
        "fx_to_reporting": "1"}}}), encoding="utf-8")
    data = {"parameter_grid": {"max_output_tokens": 16000},
            "cost_accounting": {"price_card_manifest": card.name}}
    plan = {"counts": {"by_model": [{"provider": "psnc", "model_id": "Covered"},
                                    {"provider": "psnc", "model_id": "Absent"}]}}

    with pytest.raises(LabError, match="psnc/Absent"):
        _live_services(tmp_path, data, plan)

    # The covered-only plan builds, and reserves an evidenced zero rather than failing.
    plan["counts"]["by_model"] = [{"provider": "psnc", "model_id": "Covered"}]
    _, cost_policy, _ = _live_services(tmp_path, data, plan)
    reservation = cost_policy({"provider": "psnc", "run": {"model_id": "Covered"}},
                              {"messages": [{"role": "user", "content": "hi"}]})
    assert reservation["reservation_amount"] == "0" and reservation["bounded"] is True
