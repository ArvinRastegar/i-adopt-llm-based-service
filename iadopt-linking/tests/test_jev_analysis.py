"""Analysis of a Jev run on hand-built passes, where every figure can be worked out on paper."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from iadopt_linking.jev_analysis import (
    abstention,
    agreement_with,
    compare,
    confidence_block,
    confidence_formula,
    confidence_signals,
    consensus,
    correctness_changes,
    disagreements,
    exact_margin,
    exact_margins,
    headline,
    positions,
    repeatability,
    transitions,
    unanimity,
)
from iadopt_linking.population import Slot
from iadopt_linking_eval import choose, evaluate_slot


def cand(source: str, rank: int, local_id: str, score: float = 0.0) -> dict[str, Any]:
    return {"source": source, "rank": rank, "vocabulary": source, "local_id": local_id,
            "label": local_id.lower(), "score": score, "extras": {}}


def base(slot_id: str, gold: tuple[str, str], ids: list[str], term: str = "t") -> dict[str, Any]:
    pool = [cand("wikidata", i + 1, local_id) for i, local_id in enumerate(ids)]
    return {"slot_id": slot_id, "variable_id": "v", "role": "hasProperty", "slot": "hasProperty",
            "is_member": False, "term": term, "query_key": term,
            "gold": {"vocabulary": gold[0], "local_id": gold[1]}, "pool": pool}


def result(record: dict[str, Any], scores: list[float], *, p_none: float | None = 0.0,
           order: list[int] | None = None, jev_index: int | None = -1,
           confidence: float = 0.5) -> dict[str, Any]:
    """Judge scores the way pass_results does, for a hand-built Choice result."""
    pool = [{**c, "score": s} for c, s in zip(record["pool"], scores, strict=True)]
    scored = {**record, "pool": pool}
    chosen = choose(pool)
    own = chosen if jev_index == -1 else jev_index
    margin = exact_margin(scores)
    return {"slot_id": record["slot_id"], "asked": True, "scores": scores, "p_none": p_none,
            "margin": None if margin is None else float(margin),
            "jev_index": own, "jev_chose_none": own is None, "jev_choice_key": None,
            "jev_confidence": confidence, "order": order or list(range(len(scores))),
            "selected_index": chosen, "outcome": evaluate_slot(scored)}


A = base("a", ("wikidata", "Q1"), ["Q1", "Q2", "Q3"], term="alpha")
B = base("b", ("wikidata", "Q5"), ["Q4", "Q5"], term="beta")
C = base("c", ("wikidata", "Q8"), ["Q7"], term="gamma")           # gold not retrieved


def test_headline_counts_all_and_reachable_slots() -> None:
    rows = [result(A, [0.8, 0.1, 0.1]), result(B, [0.7, 0.3]), result(C, [1.0])]
    h = headline(rows)
    assert h["all"] == {"numerator": 1, "denominator": 3}
    assert h["reachable"] == {"numerator": 1, "denominator": 3} and h["no_selection"] == 0


def test_comparing_systems_counts_agreement_and_disagreement() -> None:
    jev = [result(A, [0.8, 0.1, 0.1])["outcome"], result(B, [0.7, 0.3])["outcome"]]
    qwen = [result(A, [0.2, 0.7, 0.1])["outcome"], result(B, [0.1, 0.9])["outcome"]]
    c = compare(jev, qwen, lambda o: o["reachable"], resamples=50, seed=1)
    assert (c["both"], c["only_first"], c["only_second"], c["neither"]) == (0, 1, 1, 0)
    assert c["mcnemar_p"] == {"numerator": 1, "denominator": 1}
    assert c["difference"]["difference"] == {"numerator": 0, "denominator": 1}
    assert (c["terms"]["terms"], c["terms"]["first_better"], c["terms"]["second_better"]) == (2, 1, 1)


def test_an_empty_population_has_no_p_value_and_no_interval() -> None:
    outcomes = [result(A, [0.8, 0.1, 0.1])["outcome"]]
    c = compare(outcomes, outcomes, lambda o: False)
    assert c["n"] == 0 and c["mcnemar_p"] is None and c["difference"] is None
    assert c["terms"] is None


def test_systems_that_judged_different_slots_cannot_be_compared() -> None:
    one = [result(A, [0.8, 0.1, 0.1])["outcome"]]
    other = [result(B, [0.7, 0.3])["outcome"]]
    with pytest.raises(ValueError, match="same slots"):
        compare(one, other, lambda o: True)


def test_repeatability_measures_selection_and_score_differences() -> None:
    first = [result(A, [0.8, 0.1, 0.1]), result(B, [0.6, 0.4])]
    second = [result(A, [0.8, 0.1, 0.1]), result(B, [0.4, 0.6])]
    r = repeatability(first, second)
    assert r["same_selection"] == {"numerator": 1, "denominator": 2}
    assert r["identical_scores"] == {"numerator": 1, "denominator": 2}
    assert r["max_abs_difference"] == {"numerator": 1, "denominator": 5}          # exactly 0.2
    assert r["mean_abs_difference"] == {"numerator": 2, "denominator": 25}        # 0.4 over 5


def test_agreement_is_the_share_of_all_passes_that_select_what_the_primary_selected() -> None:
    primary = [result(A, [0.8, 0.1, 0.1]), result(B, [0.6, 0.4])]
    others = [[result(A, [0.7, 0.2, 0.1]), result(B, [0.4, 0.6])],
              [result(A, [0.1, 0.8, 0.1]), result(B, [0.9, 0.1])]]
    assert agreement_with(primary, others) == [Fraction(2, 3), Fraction(2, 3)]


def test_consensus_averages_scores_before_selecting() -> None:
    slots = [Slot("a", "v", "p", "l", "d", "hasProperty", "hasProperty", False, "Property", "alpha",
                  "alpha", "iri", "wikidata", "Q1", "canonical")]
    passes = [[result(A, [0.9, 0.05, 0.05])], [result(A, [0.0, 0.6, 0.4])], [result(A, [0.3, 0.5, 0.2])]]
    mean = consensus(passes, slots, {"a": A})[0]
    assert mean["scores"] == pytest.approx([0.4, 0.3833333, 0.2166667])
    assert mean["outcome"]["correct"] is True


def test_abstention_leaves_the_slots_outside_the_population_apart() -> None:
    inside = result(A, [0.2, 0.1, 0.1], p_none=0.6, jev_index=None, confidence=0.95)
    outside_record = base("o", ("nerc", "S27"), ["Q1"], term="omega")
    outside = result(outside_record, [0.1], p_none=0.9, jev_index=None)
    a = abstention([inside, outside])
    assert a["asked"] == 1 and a["none_when_gold_in_pool"] == {"numerator": 1, "denominator": 1}
    assert a["none_when_gold_absent"] == {"numerator": 0, "denominator": 0}
    assert (a["outside_population_asked"], a["outside_population_none"]) == (1, 1)
    assert a["refusals_at_confidence_0.9"] == 1


def test_abstention_separates_wrong_and_right_refusals() -> None:
    present = result(A, [0.2, 0.1, 0.1], p_none=0.6, jev_index=None)       # refused, gold present
    absent = result(C, [0.1], p_none=0.9, jev_index=None)                  # refused, gold absent
    answered = result(B, [0.1, 0.8], p_none=0.1, jev_index=1)              # answered correctly
    a = abstention([present, absent, answered])
    assert a["none_when_gold_in_pool"] == {"numerator": 1, "denominator": 2}
    assert a["none_when_gold_absent"] == {"numerator": 1, "denominator": 1}
    assert a["answered"] == {"numerator": 1, "denominator": 3}
    assert a["precision_when_answered"] == {"numerator": 1, "denominator": 1}
    assert a["forced_choice_correct_when_none_chosen"] == {"numerator": 1, "denominator": 2}


def test_position_bias_counts_selections_shown_first_against_chance() -> None:
    first = result(A, [0.8, 0.1, 0.1], order=[0, 1, 2])             # selected candidate 0, shown first
    later = result(B, [0.1, 0.9], order=[1, 0])                      # selected candidate 1, shown first
    third = result(A, [0.1, 0.1, 0.8], order=[0, 1, 2])             # selected candidate 2, shown third
    p = positions([first, later, third])
    assert p["slots"] == 3 and p["selected_first"] == 2
    assert p["expected_if_no_bias"] == {"numerator": 7, "denominator": 6}      # 1/3 + 1/2 + 1/3
    assert p["jev_choice_first"] == 2


def test_the_documented_confidence_formula_is_checked_against_each_answer() -> None:
    fits = {"probabilities": {"c01": 0.6, "c02": 0.3, "none_of_these": 0.1}, "confidence": 0.4}
    off = {"probabilities": {"c01": 0.6, "c02": 0.3, "none_of_these": 0.1}, "confidence": 0.9}
    result_ = confidence_formula([fits, off])
    assert result_["answers"] == {"numerator": 1, "denominator": 2}
    assert result_["max_abs_difference"] == pytest.approx(0.5)


def test_transitions_and_disagreements_list_how_two_systems_differ() -> None:
    jev = [result(A, [0.8, 0.1, 0.1])["outcome"], result(B, [0.7, 0.3])["outcome"]]
    qwen = [result(A, [0.1, 0.8, 0.1])["outcome"], result(B, [0.1, 0.9])["outcome"]]
    assert transitions(qwen, jev) == {"correct>wrong_within_source": 1,
                                      "wrong_within_source>correct": 1}
    d = disagreements(jev, qwen)
    assert d["first_only_right"]["slots"] == 1 and d["second_only_right"]["slots"] == 1
    assert d["first_only_right"]["examples"][0]["term"] == "alpha"
    assert d["first_only_right"]["loser_is_equivalent"] == 0
    assert d["second_only_right"]["examples"][0]["second"]["local_id"] == "Q5"


def test_confidence_signals_are_judged_on_the_chosen_population() -> None:
    primary = [result(A, [0.8, 0.1, 0.1], p_none=0.1, confidence=0.7),
               result(B, [0.6, 0.3], p_none=0.1, confidence=0.4),
               result(C, [0.5], p_none=0.5, confidence=0.1)]
    qwen = [r["outcome"] for r in primary]
    signals = confidence_signals(primary, [], qwen)
    assert signals["jev_selected_probability"] == [0.8, 0.6, 0.5]
    assert signals["jev_candidate_mass"] == [pytest.approx(0.9), pytest.approx(0.9), 0.5]
    assert signals["jev_pass_agreement"] == [1.0, 1.0, 1.0]
    block = confidence_block(signals, {"jev": qwen, "qwen3": qwen}, lambda o: o["reachable"])
    entry = block["jev_selected_probability"]
    assert entry["slots"] == 3 and entry["answered"] == 3
    assert entry["auroc"] == {"numerator": 1, "denominator": 1}      # the right answer is most sure
    assert "reliability" in entry and "reliability" not in block["jev_margin"]


def test_a_margin_is_the_exact_gap_between_decimals_not_a_float_difference() -> None:
    assert 0.34 - 0.33 != 0.01                       # the trap
    assert exact_margin([0.34, 0.33, 0.1]) == Fraction(1, 100)
    assert exact_margin([0.5]) is None
    rows = [result(A, [0.34, 0.33, 0.33]), result(B, [0.9, 0.1])]
    counts = exact_margins(rows)
    assert counts["below_1e-2"] == {"numerator": 0, "denominator": 2}     # 0.01 is not below 0.01
    assert counts["below_1e-3"] == {"numerator": 0, "denominator": 2}
    signals = confidence_signals(rows, [], [r["outcome"] for r in rows])
    assert signals["jev_margin"] == [0.01, 0.8]
    assert confidence_signals([result(A, [0.5, 0.2, 0.1], p_none=0.9)], [], [rows[0]["outcome"]])[
        "jev_candidate_mass"] == [pytest.approx(0.1)]
    assert 1.0 - 0.9 != 0.1                          # the trap again: the mass must not be a float difference


def test_order_effects_are_counted_per_slot_not_only_in_total() -> None:
    primary = [result(A, [0.8, 0.1, 0.1]), result(B, [0.3, 0.7])]           # right, right
    other = [result(A, [0.1, 0.8, 0.1]), result(B, [0.2, 0.8])]             # wrong, right
    third = [result(A, [0.7, 0.2, 0.1]), result(B, [0.9, 0.1])]             # right, wrong
    changes = correctness_changes(primary, {"x": other, "y": third})
    assert changes["x"] == {"slots": 2, "changed": 1, "became_right": 0, "became_wrong": 1}
    assert changes["y"] == {"slots": 2, "changed": 1, "became_right": 0, "became_wrong": 1}
    assert unanimity([primary, other, third]) == {"slots": 2, "same_selection": 0,
                                                  "same_correctness": 0}
    assert unanimity([primary, primary]) == {"slots": 2, "same_selection": 2, "same_correctness": 2}
