"""Metric semantics on small synthetic runs, one case per outcome category."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from iadopt_linking_eval import choose, evaluate_slot, summarise


def cand(source: str, rank: int, local_id: str, score: float, vocabulary: str | None = None) -> dict:
    return {"source": source, "rank": rank, "vocabulary": vocabulary or source,
            "local_id": local_id, "label": local_id.lower(), "score": score}


def slot(slot_id: str, gold: tuple[str, str], pool: list[dict], variable: str = "v1",
         term: str = "t", role: str = "hasProperty", is_member: bool = False) -> dict[str, Any]:
    return {"slot_id": slot_id, "variable_id": variable, "role": role, "slot": role.split(".")[0],
            "is_member": is_member, "term": term, "query_key": term,
            "gold": {"vocabulary": gold[0], "local_id": gold[1]}, "pool": pool}


def test_choice_prefers_score_then_source_then_rank() -> None:
    pool = [cand("qudt", 1, "A", 0.5), cand("wikidata", 2, "B", 0.5), cand("wikidata", 1, "C", 0.5)]
    assert pool[choose(pool)]["local_id"] == "C"
    assert choose([]) is None
    other = [cand("om", 1, "X", 0.9), cand("wikidata", 1, "Y", 0.1)]
    assert other[choose(other)]["local_id"] == "X"


@pytest.mark.parametrize(("pool", "gold", "category"), [
    ([cand("wikidata", 1, "Q1", 0.9), cand("wikidata", 2, "Q2", 0.1)], ("wikidata", "Q1"), "correct"),
    ([cand("wikidata", 1, "Q1", 0.2), cand("wikidata", 2, "Q2", 0.9)], ("wikidata", "Q1"),
     "wrong_within_source"),
    ([cand("wikidata", 1, "Q9", 0.9), cand("qudt", 1, "Mass", 0.1)], ("qudt", "Mass"), "wrong_source"),
    ([cand("wikidata", 1, "Q9", 0.9)], ("wikidata", "Q1"), "gold_not_retrieved"),
    ([], ("qudt", "Mass"), "no_candidates"),
    ([cand("wikidata", 1, "Q9", 0.9)], ("nerc", "S27/current/CS003625"), "unreachable_vocabulary"),
])
def test_every_category(pool: list[dict], gold: tuple[str, str], category: str) -> None:
    outcome = evaluate_slot(slot("s", gold, pool))
    assert outcome["category"] == category
    assert outcome["correct"] is (category == "correct")


def test_diagnostics_separate_routing_from_ranking() -> None:
    pool = [cand("wikidata", 1, "Q9", 0.9), cand("qudt", 1, "Area", 0.3), cand("qudt", 2, "Mass", 0.2)]
    outcome = evaluate_slot(slot("s", ("qudt", "Mass"), pool))
    assert outcome["category"] == "wrong_source"
    assert outcome["routed_correctly"] is False
    assert outcome["gold_in_pool"] is True
    assert outcome["oracle_routed_correct"] is False      # inside QUDT the reranker prefers Area
    assert outcome["native_first_correct"] is False       # QUDT's own first hit is Area too
    assert outcome["wikidata_only_correct"] is None       # only defined for Wikidata gold


def test_a_source_that_returned_nothing_counts_as_a_miss_in_the_diagnostics() -> None:
    outcome = evaluate_slot(slot("s", ("qudt", "Mass"), [cand("wikidata", 1, "Q9", 0.9)]))
    assert outcome["category"] == "gold_not_retrieved"
    assert outcome["oracle_routed_correct"] is False and outcome["native_first_correct"] is False
    wikidata_gold = evaluate_slot(slot("s", ("wikidata", "Q1"), [cand("qudt", 1, "Mass", 0.9)]))
    assert wikidata_gold["wikidata_only_correct"] is False
    unreachable = evaluate_slot(slot("s", ("nerc", "x"), [cand("wikidata", 1, "Q9", 0.9)]))
    assert unreachable["oracle_routed_correct"] is None and unreachable["native_first_correct"] is None


def test_wikidata_only_and_native_order() -> None:
    pool = [cand("wikidata", 1, "Q9", 0.1), cand("wikidata", 2, "Q1", 0.8), cand("qudt", 1, "Mass", 0.95)]
    outcome = evaluate_slot(slot("s", ("wikidata", "Q1"), pool))
    assert outcome["category"] == "wrong_source"
    assert outcome["wikidata_only_correct"] is True       # a Wikidata-only pipeline would be right
    assert outcome["native_first_correct"] is False


def test_summary_receipts_and_populations() -> None:
    records = [
        slot("a", ("wikidata", "Q1"), [cand("wikidata", 1, "Q1", 0.9)], variable="v1"),
        slot("b", ("wikidata", "Q2"), [cand("wikidata", 1, "Q3", 0.9)], variable="v1"),
        slot("c", ("qudt", "Mass"), [cand("qudt", 1, "Mass", 0.8)], variable="v2", role="hasObjectOfInterest.hasPart",
             is_member=True),
        slot("d", ("iobc", "1"), [cand("wikidata", 1, "Q4", 0.7)], variable="v3"),
    ]
    summary = summarise([evaluate_slot(r) for r in records])
    everything = summary["populations"]["all"]
    assert everything["top1"] == {"numerator": 2, "denominator": 4}
    assert everything["reachable_slots"] == {"numerator": 3, "denominator": 4}
    assert everything["gold_in_pool"] == {"numerator": 2, "denominator": 3}
    # macro by variable: v1 = 1/2, v2 = 1, v3 = 0 -> (1/2 + 1 + 0) / 3 = 1/2
    assert everything["top1_macro_by_variable"] == {"numerator": 1, "denominator": 2}
    assert everything["categories"]["unreachable_vocabulary"] == 1
    assert summary["populations"]["gold:qudt"]["top1"] == {"numerator": 1, "denominator": 1}
    assert summary["populations"]["role:member"]["n"] == 1
    assert summary["populations"]["reachable"]["n"] == 3
    unreachable = summary["populations"]["unreachable"]
    assert (unreachable["n"], unreachable["top1"]) == (1, {"numerator": 0, "denominator": 1})


def test_routing_confusion_and_term_length_groups() -> None:
    records = [
        slot("a", ("wikidata", "Q1"), [cand("wikidata", 1, "Q1", 0.9)], term="mass"),
        slot("b", ("wikidata", "Q2"), [cand("wikidata", 1, "Q3", 0.2), cand("qudt", 1, "Mass", 0.9)],
             term="mass flux"),
        slot("c", ("qudt", "Mass"), [cand("wikidata", 1, "Q3", 0.9)], term="a b c d"),
        slot("d", ("iobc", "1"), [cand("wikidata", 1, "Q3", 0.9)], term="x"),
    ]
    summary = summarise([evaluate_slot(r) for r in records])
    assert summary["routing_confusion"] == {"qudt>wikidata": 1, "wikidata>qudt": 1, "wikidata>wikidata": 1}
    pops = summary["populations"]
    assert (pops["words:1"]["n"], pops["words:2"]["n"], pops["words:3+"]["n"]) == (2, 1, 1)


def test_thresholds_trade_coverage_for_precision() -> None:
    records = [slot("a", ("wikidata", "Q1"), [cand("wikidata", 1, "Q1", 0.95)]),
               slot("b", ("wikidata", "Q2"), [cand("wikidata", 1, "Q3", 0.35)])]
    rows = {Fraction(r["threshold"]["numerator"], r["threshold"]["denominator"]): r
            for r in summarise([evaluate_slot(r) for r in records])["thresholds_all"]}
    assert rows[Fraction(0)]["coverage"] == {"numerator": 2, "denominator": 2}
    assert rows[Fraction(9, 10)]["coverage"] == {"numerator": 1, "denominator": 2}
    assert rows[Fraction(9, 10)]["precision"] == {"numerator": 1, "denominator": 1}
    assert rows[Fraction(3, 10)]["precision"] == {"numerator": 1, "denominator": 2}


def test_consistency_ceiling_counts_the_majority_identity_per_term() -> None:
    pool = [cand("wikidata", 1, "Q5", 0.9)]
    records = [slot(f"s{i}", ("wikidata", gold), pool, term="person")
               for i, gold in enumerate(["Q5", "Q5", "Q5", "Q215627"])]
    records.append(slot("z", ("wikidata", "Q7"), pool, term="water"))
    block = summarise([evaluate_slot(r) for r in records])["populations"]["all"]
    assert block["consistency_ceiling"] == {"numerator": 4, "denominator": 5}
    assert block["distinct_terms"] == 2 and block["distinct_term_gold_pairs"] == 3


def test_a_ceiling_is_computed_for_each_population_it_labels() -> None:
    pool = [cand("wikidata", 1, "Q5", 0.9)]
    records = [slot("a", ("wikidata", "Q5"), pool, term="person"),
               slot("b", ("wikidata", "Q6"), pool, term="person"),
               slot("c", ("iobc", "1"), pool, term="person")]
    pops = summarise([evaluate_slot(r) for r in records])["populations"]
    assert pops["all"]["consistency_ceiling"] == {"numerator": 1, "denominator": 3}
    assert pops["reachable"]["consistency_ceiling"] == {"numerator": 1, "denominator": 2}


def with_wikidata_match(candidate: dict, *items: str) -> dict:
    return {**candidate, "extras": {"wikidata_match": [f"http://www.wikidata.org/entity/{q}" for q in items]}}


def test_equivalence_is_a_diagnostic_that_never_changes_the_headline() -> None:
    qudt = with_wikidata_match(cand("qudt", 1, "Mass", 0.9), "Q11423")
    gold_wikidata = evaluate_slot(slot("s", ("wikidata", "Q11423"), [qudt, cand("wikidata", 1, "Q11423", 0.1)]))
    assert gold_wikidata["correct"] is False and gold_wikidata["equivalent_correct"] is True
    other_way = evaluate_slot(slot("s", ("qudt", "Mass"), [
        cand("wikidata", 1, "Q11423", 0.9), with_wikidata_match(cand("qudt", 1, "Mass", 0.1), "Q11423")]))
    assert other_way["correct"] is False and other_way["equivalent_correct"] is True
    unrelated = evaluate_slot(slot("s", ("wikidata", "Q1"), [with_wikidata_match(cand("qudt", 1, "Mass", 0.9), "Q2")]))
    assert unrelated["equivalent_correct"] is False
    nothing = evaluate_slot(slot("s", ("wikidata", "Q1"), []))
    assert nothing["equivalent_correct"] is False
    block = summarise([gold_wikidata, other_way, unrelated, nothing])["populations"]["all"]
    assert block["top1"] == {"numerator": 0, "denominator": 4}
    assert block["top1_with_qudt_equivalence"] == {"numerator": 2, "denominator": 4}


def test_the_margin_is_the_gap_between_the_two_best_scores() -> None:
    wide = evaluate_slot(slot("a", ("wikidata", "Q1"), [cand("wikidata", 1, "Q1", 0.9), cand("wikidata", 2, "Q2", 0.4)]))
    tight = evaluate_slot(slot("b", ("wikidata", "Q1"), [cand("wikidata", 1, "Q1", 0.5), cand("wikidata", 2, "Q2", 0.4995)]))
    alone = evaluate_slot(slot("c", ("wikidata", "Q1"), [cand("wikidata", 1, "Q1", 0.5)]))
    assert wide["margin"] == pytest.approx(0.5) and tight["margin"] == pytest.approx(0.0005)
    assert alone["margin"] is None
    margins = summarise([wide, tight, alone])["margins"]
    assert margins["below_1e-3"] == {"numerator": 1, "denominator": 2}
    assert margins["below_1e-2"] == {"numerator": 1, "denominator": 2}
    assert margins["below_1e-4"] == {"numerator": 0, "denominator": 2}


def test_the_most_repeated_terms_are_listed_with_their_misses() -> None:
    pool_right, pool_wrong = [cand("wikidata", 1, "Q1", 0.9)], [cand("wikidata", 1, "Q9", 0.9)]
    records = ([slot(f"p{i}", ("wikidata", "Q1"), pool_right if i else pool_wrong, term="person") for i in range(3)]
               + [slot("w", ("wikidata", "Q1"), pool_wrong, term="water")])
    top = summarise([evaluate_slot(r) for r in records])["most_repeated_terms"]
    assert top[0] == {"term": "person", "slots": 3, "wrong": 1}
    assert top[1] == {"term": "water", "slots": 1, "wrong": 1}


def test_wikidata_only_accuracy_is_averaged_per_variable_too() -> None:
    right, wrong = [cand("wikidata", 1, "Q1", 0.9)], [cand("qudt", 1, "Mass", 0.9)]
    records = [slot("a", ("wikidata", "Q1"), right, variable="v1"),
               slot("b", ("wikidata", "Q1"), wrong, variable="v1"),
               slot("c", ("wikidata", "Q1"), right, variable="v2")]
    block = summarise([evaluate_slot(r) for r in records])["populations"]["gold:wikidata"]
    assert block["wikidata_only_top1"] == {"numerator": 2, "denominator": 3}
    assert block["wikidata_only_top1_macro_by_variable"] == {"numerator": 3, "denominator": 4}


def test_unknown_source_is_an_error_not_a_silent_skip() -> None:
    with pytest.raises(ValueError):
        choose([cand("elsewhere", 1, "X", 0.5)])
