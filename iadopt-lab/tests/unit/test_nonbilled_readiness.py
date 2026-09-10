"""Non-billed readiness must disclose uncertainty without relaxing paid gates."""

import pytest

from iadopt_lab.costing import estimate_campaign_cost


@pytest.mark.parametrize("mode,provenance,ready", [
    ("non_billed", "owner declaration", True),
    ("non_billed", "", False),
    ("metered", "price card", False),
    ("unknown", "unknown", False),
])
def test_unverified_ceiling_requires_complete_nonbilled_coverage(mode, provenance, ready):
    """Input billing variants; assert readiness and unchanged uncertainty; no I/O."""
    plan = {"sha256": "fixture", "population_size": 1,
            "runs": [{"id": "run", "provider": "psnc", "model_id": "model"}],
            "counts": {"initial_calls": 1, "maximum_calls": 3, "by_model": [
                {"provider": "psnc", "model_id": "model", "initial_calls": 1, "maximum_calls": 3}]}}
    prompts = {"plan_sha256": "fixture", "evidence": "fixture tokenizer",
               "input_tokens_by_run": {"run": [100]}}
    billing = {"currency": "USD", "models": {"psnc/model": {
        "mode": mode, "basis": "fixture", "provenance": provenance,
        "input_per_million": "1", "output_per_million": "1", "fx_to_reporting": "1"}}}
    assumptions = {"expected_output_tokens": 100, "attempt2_fraction": "0.1",
                   "attempt3_fraction": "0", "correction_error_tokens": 50,
                   "total_output_token_ceiling": 16000, "ceiling_verified": False,
                   "reasoning_accounting_evidence": "fixture"}
    result = estimate_campaign_cost(plan, prompts, billing, assumptions)
    assert result["ready"] is ready
    assert result["assumptions"]["ceiling_verified"] is False
    if ready:
        assert result["warnings"] and not result["issues"]
        assert result["totals"]["expected_cost"] == "0"
        assert result["totals"]["conditional_maximum_cost"] == "0"
    else:
        assert result["issues"]
