"""Guards for the derived cost-estimate evidence document.

The estimate refuses to guess, so the generator must either derive a value from
something real or refuse. These tests pin the places that matters: billing comes
from the frozen price card execution reserves against rather than a parallel
derivation, a model missing from that card is refused rather than defaulted to
free, the output ceiling comes from the plan so a stale plan cannot borrow
today's number, and the ceiling assertion stays unmade until an operator
supplies a specific basis.

The tokenizer is injected throughout. Loading a real cached model would make
these tests depend on an undeclared local prerequisite, so only the one test
that is about the real tokenizer asks for it, and it skips when absent.
"""

import json
from pathlib import Path

import pytest
import yaml

from iadopt_lab.domain import LabError
from iadopt_lab.evidence import (
    billing_cards,
    build_cost_evidence,
    plan_output_ceiling,
    prompt_tokens_by_run,
    reasoning_accounting,
)

ROOT = Path(__file__).resolve().parents[2]


class FakeTokenizer:
    """A deterministic stand-in: one token per whitespace-separated word."""

    vocab_size = 7

    def encode(self, text):
        return text.split()


def _card(tmp_path, models, currency="USD"):
    path = tmp_path / "card.yml"
    path.write_text(yaml.safe_dump({"currency": currency, "models": models}), encoding="utf-8")
    # Absolute, so the card resolves the same whether root is the repo or tmp_path.
    return str(path)


def _parameters(tmp_path, models, *, ceiling=16000, billing_mode="non_billed", currency="USD"):
    return {"providers": {"psnc": {"billing": {"mode": billing_mode, "basis": "free access"}},
                          "openrouter": {"billing": {"mode": "metered", "basis": "credits"}}},
            "parameter_grid": {"max_output_tokens": ceiling},
            "cost_accounting": {"price_card_manifest": _card(tmp_path, models, currency),
                                "reporting_currency": currency}}


def _plan(models, *, ceiling=16000, shapes=(("strict-minimal", 5),), population=("a", "b")):
    runs, by_model = [], []
    for index, (provider, model) in enumerate(models):
        variant, shots = shapes[index % len(shapes)]
        runs.append({"id": f"run{index}", "provider": provider, "model_id": model,
                     "prompt_variant": variant, "shot_count": shots,
                     "reasoning_mode": "disabled", "max_output_tokens": ceiling})
        by_model.append({"provider": provider, "model_id": model,
                         "initial_calls": 2, "maximum_calls": 6})
    return {"sha256": "plan-hash", "population_size": len(population), "population": list(population),
            "runs": runs, "counts": {"by_model": by_model}}


NON_BILLED = {"mode": "non_billed", "basis": "free access", "provenance": "owner report",
              "fx_to_reporting": "1"}


# --- billing is the card execution uses, or nothing --------------------------------

def test_cards_come_from_the_frozen_price_card(tmp_path):
    plan = _plan([("psnc", "GLM-5.2"), ("psnc", "Qwen3.8-27B")])
    parameters = _parameters(tmp_path, {"psnc/GLM-5.2": NON_BILLED, "psnc/Qwen3.8-27B": NON_BILLED})
    cards = billing_cards(plan, parameters, tmp_path)
    assert sorted(cards["models"]) == ["psnc/GLM-5.2", "psnc/Qwen3.8-27B"]
    assert all(card["mode"] == "non_billed" for card in cards["models"].values())


def test_a_model_missing_from_the_card_is_refused_even_when_the_provider_is_free(tmp_path):
    """The exact failure that let a ready zero-cost estimate precede a mid-run crash."""
    plan = _plan([("psnc", "GLM-5.2"), ("psnc", "DeepSeek-V4-Flash")])
    parameters = _parameters(tmp_path, {"psnc/GLM-5.2": NON_BILLED})
    with pytest.raises(LabError, match="DeepSeek-V4-Flash"):
        billing_cards(plan, parameters, tmp_path)


def test_card_disagreeing_with_the_provider_declaration_is_refused(tmp_path):
    plan = _plan([("psnc", "GLM-5.2")])
    parameters = _parameters(tmp_path, {"psnc/GLM-5.2": NON_BILLED}, billing_mode="metered")
    with pytest.raises(LabError, match="price card says"):
        billing_cards(plan, parameters, tmp_path)


def test_metered_entry_without_rates_is_refused(tmp_path):
    plan = _plan([("openrouter", "qwen/qwen3-32b")])
    parameters = _parameters(tmp_path, {"openrouter/qwen/qwen3-32b": {
        "mode": "metered", "basis": "credits", "provenance": "price page"}})
    with pytest.raises(LabError, match="rates"):
        billing_cards(plan, parameters, tmp_path)


def test_currency_mismatch_between_card_and_configuration_is_refused(tmp_path):
    plan = _plan([("psnc", "GLM-5.2")])
    parameters = _parameters(tmp_path, {"psnc/GLM-5.2": NON_BILLED}, currency="USD")
    parameters["cost_accounting"]["reporting_currency"] = "EUR"
    with pytest.raises(LabError, match="disagrees"):
        billing_cards(plan, parameters, tmp_path)


# --- the ceiling belongs to the plan, not to today's parameters --------------------

def test_stale_plan_cannot_borrow_the_current_ceiling(tmp_path):
    """An old 100,000-token plan must not acquire today's 16,000-token bound."""
    plan = _plan([("psnc", "GLM-5.2")], ceiling=100000)
    parameters = _parameters(tmp_path, {"psnc/GLM-5.2": NON_BILLED}, ceiling=16000)
    with pytest.raises(LabError, match="Re-expand the plan"):
        plan_output_ceiling(plan, parameters)


def test_runs_disagreeing_on_a_ceiling_are_refused(tmp_path):
    plan = _plan([("psnc", "A"), ("psnc", "B")])
    plan["runs"][1]["max_output_tokens"] = 4000
    parameters = _parameters(tmp_path, {"psnc/A": NON_BILLED, "psnc/B": NON_BILLED})
    with pytest.raises(LabError, match="one explicit output ceiling"):
        plan_output_ceiling(plan, parameters)


def test_matching_plan_and_parameters_yield_the_planned_ceiling(tmp_path):
    plan = _plan([("psnc", "A")])
    assert plan_output_ceiling(plan, _parameters(tmp_path, {"psnc/A": NON_BILLED})) == 16000


# --- reasoning modes make different claims ----------------------------------------

def test_not_applicable_is_not_claimed_as_absence_of_reasoning():
    plan = _plan([("psnc", "A"), ("psnc", "B")])
    plan["runs"][1]["reasoning_mode"] = "not_applicable"
    statement = reasoning_accounting(plan)
    assert "disabled: psnc/A" in statement and "not_applicable: psnc/B" in statement
    # The distinction the accounting previously collapsed.
    assert "NOT a measurement that the model never reasons" in statement
    assert "counted once" in statement


# --- document assembly -------------------------------------------------------------

def _real_plan(models, **kwargs):
    from iadopt_lab.corpus.ingestion import load_canonical_records

    targets = [row["variable_id"] for row in load_canonical_records(ROOT)
               if not row["demonstration_position"]][:3]
    return json.loads(json.dumps(_plan(models, population=tuple(targets), **kwargs)))


def test_models_sharing_a_prompt_shape_share_one_measured_vector(tmp_path):
    plan = _real_plan([("psnc", "A"), ("psnc", "B")])
    parameters = _parameters(tmp_path, {"psnc/A": NON_BILLED, "psnc/B": NON_BILLED})
    document = build_cost_evidence(plan, parameters, root=ROOT, tokenizer=FakeTokenizer())
    tokens = document["prompt_artifacts"]["input_tokens_by_run"]
    assert tokens["run0"] == tokens["run1"] and len(tokens["run0"]) == 3
    assert all(count > 0 for count in tokens["run0"])
    assert document["prompt_artifacts"]["evidence"]["distinct_prompt_shapes"] == [
        {"prompt_variant": "strict-minimal", "shot_count": 5}]
    assert document["prompt_artifacts"]["plan_sha256"] == "plan-hash"


def test_tokenizer_identity_and_limits_are_recorded(tmp_path):
    """Naming a tokenizer does not identify it; two caches can differ silently."""
    plan = _real_plan([("psnc", "A")])
    parameters = _parameters(tmp_path, {"psnc/A": NON_BILLED})
    evidence = build_cost_evidence(plan, parameters, root=ROOT,
                                   tokenizer=FakeTokenizer())["prompt_artifacts"]["evidence"]
    assert evidence["tokenizer_identity"]["class"] == "FakeTokenizer"
    assert evidence["tokenizer_identity"]["vocab_size"] == 7
    assert any("content only" in line for line in evidence["limitations"])


def test_ceiling_stays_unverified_until_a_specific_basis_is_given(tmp_path):
    plan = _real_plan([("psnc", "A")])
    parameters = _parameters(tmp_path, {"psnc/A": NON_BILLED})
    default = build_cost_evidence(plan, parameters, root=ROOT, tokenizer=FakeTokenizer())
    assert default["assumptions"]["ceiling_verified"] is False
    # No generated sentence stands in for an assertion nobody made.
    assert "ceiling_basis" not in default["assumptions"]

    asserted = build_cost_evidence(plan, parameters, root=ROOT, tokenizer=FakeTokenizer(),
                                   ceiling_basis="measured 197 max completion tokens")
    assert asserted["assumptions"]["ceiling_verified"] is True
    assert asserted["assumptions"]["ceiling_basis"] == "measured 197 max completion tokens"
    assert asserted["assumptions"]["total_output_token_ceiling"] == 16000


def test_expected_output_above_the_ceiling_is_refused(tmp_path):
    plan = _real_plan([("psnc", "A")])
    parameters = _parameters(tmp_path, {"psnc/A": NON_BILLED})
    with pytest.raises(LabError, match="exceeds the planned ceiling"):
        build_cost_evidence(plan, parameters, root=ROOT, tokenizer=FakeTokenizer(),
                            expected_output_tokens=20000)


def test_unavailable_tokenizer_is_a_clear_failure(tmp_path):
    plan = _real_plan([("psnc", "A")])
    parameters = _parameters(tmp_path, {"psnc/A": NON_BILLED})
    with pytest.raises(LabError, match="not available offline"):
        build_cost_evidence(plan, parameters, root=ROOT, tokenizer_id="no/such-tokenizer-xyz")


def test_the_configured_tokenizer_reproduces_its_counts():
    """The one test that is about the real tokenizer, skipped when it is not cached."""
    from iadopt_lab.evidence import DEFAULT_TOKENIZER, _tokenizer

    try:
        tokenizer, identity = _tokenizer(DEFAULT_TOKENIZER)
    except LabError:
        pytest.skip(f"{DEFAULT_TOKENIZER} is not in the local cache")
    plan = _real_plan([("psnc", "A")])
    first, evidence = prompt_tokens_by_run(plan, ROOT, DEFAULT_TOKENIZER, tokenizer)
    second, _ = prompt_tokens_by_run(plan, ROOT, DEFAULT_TOKENIZER, tokenizer)
    assert first == second and all(count > 100 for count in first["run0"])
    assert evidence["tokenizer_identity"]["vocab_size"] == identity["vocab_size"]
