"""The cross-device check: which slots it rescores, what it measures, what it refuses."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from iadopt_linking.devices import (
    DEVICE_TOLERANCE,
    NEAR_TIE_MARGIN,
    check_slots,
    describe_device_check,
    device_check,
    device_passed,
    reference_device,
    top_two_margin,
)
from iadopt_linking.rerank import ScoreError


def record(slot_id: str, scores: Sequence[float], gold: int = 0, term: str = "t") -> dict:
    pool = [{"source": "wikidata", "rank": rank, "iri": f"http://www.wikidata.org/entity/Q{rank}",
             "vocabulary": "wikidata", "local_id": f"Q{rank}", "label": f"q{rank}",
             "document": f"doc {slot_id} {rank}", "score": score}
            for rank, score in enumerate(scores, 1)]
    return {"slot_id": slot_id, "variable_id": "v", "role": "hasProperty", "slot": "hasProperty",
            "is_member": False, "term": term, "query_key": term, "query": f"query {slot_id}",
            "gold": {"vocabulary": "wikidata", "local_id": f"Q{gold + 1}"}, "pool": pool}


class Replay:
    """Returns the stored scores of each document, shifted by a fixed amount or per-slot."""

    def __init__(self, records: dict, shift: float = 0.0, flip: set[str] = frozenset()) -> None:
        self.by_document = {c["document"]: c["score"] for r in records.values() for c in r["pool"]}
        self.shift, self.flip = shift, flip
        self.calls = 0

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls += 1
        values = [self.by_document[d] + self.shift for d in documents]
        slot_id = query.split()[1]
        if slot_id in self.flip:
            values[0], values[1] = values[1], values[0]
        return values


def test_the_other_device_is_cpu_unless_the_scores_came_from_cpu() -> None:
    assert reference_device("mps") == "cpu" and reference_device(None) == "cpu"
    assert reference_device("cpu") == "mps"


def test_the_margin_needs_two_candidates() -> None:
    assert top_two_margin(record("a", [0.9, 0.4])) == pytest.approx(0.5)
    assert top_two_margin(record("a", [0.9])) is None and top_two_margin(record("a", [])) is None


def test_every_near_tie_is_checked_and_single_candidate_slots_are_not() -> None:
    records = {f"s{i}": record(f"s{i}", [0.9, 0.5]) for i in range(30)}
    records["tie1"] = record("tie1", [0.5, 0.5 - NEAR_TIE_MARGIN / 2])
    records["tie2"] = record("tie2", [0.7, 0.7])
    records["lone"] = record("lone", [0.9])
    chosen, near = check_slots(records, spread=5)
    assert near == 2 and chosen[:2] == ["tie1", "tie2"]
    assert len(chosen) == 7 and "lone" not in chosen and len(set(chosen)) == 7


def test_the_spread_is_evenly_spaced_and_capped_by_what_exists() -> None:
    records = {f"s{i}": record(f"s{i}", [0.9, 0.5]) for i in range(40)}
    chosen, near = check_slots(records, spread=10)
    assert near == 0 and chosen == [f"s{i}" for i in range(0, 40, 4)]
    few = {f"s{i}": record(f"s{i}", [0.9, 0.5]) for i in range(3)}
    assert check_slots(few, spread=10)[0] == ["s0", "s1", "s2"]
    assert check_slots(few, spread=0)[0] == []


def test_identical_scores_on_both_devices_pass_cleanly() -> None:
    records = {f"s{i}": record(f"s{i}", [0.9, 0.5, 0.1]) for i in range(5)}
    result = device_check(records, Replay(records), Replay(records), primary_device="mps",
                          reference_device_name="cpu", spread=3)
    assert device_passed(result) and result["slots_checked"] == 3 and result["near_ties"] == 0
    assert result["max_abs_difference"] == 0.0 and result["selection_changes"] == []
    assert result["correct_on_primary"] == result["correct_on_reference"] == 3


def test_a_near_tie_that_flips_is_reported_with_its_effect_but_does_not_fail() -> None:
    records = {"tie": record("tie", [0.50005, 0.50000], gold=0), "ok": record("ok", [0.9, 0.2])}
    reference = Replay(records, flip={"tie"})
    result = device_check(records, Replay(records), reference, primary_device="mps",
                          reference_device_name="cpu", spread=1)
    assert result["near_ties"] == 1 and result["slots_checked"] == 2
    (change,) = result["selection_changes"]
    assert change["slot_id"] == "tie" and change["primary"] == ["wikidata", "Q1"]
    assert change["reference"] == ["wikidata", "Q2"]
    assert (change["primary_correct"], change["reference_correct"]) == (True, False)
    assert (result["correct_on_primary"], result["correct_on_reference"]) == (2, 1)
    assert result["max_abs_difference"] == pytest.approx(0.00005)
    assert device_passed(result)
    assert "1 selections change on cpu" in describe_device_check(result)


def test_scores_that_differ_beyond_the_tolerance_fail() -> None:
    records = {"a": record("a", [0.9, 0.2])}
    result = device_check(records, Replay(records), Replay(records, shift=DEVICE_TOLERANCE * 3),
                          primary_device="mps", reference_device_name="cpu", spread=1)
    assert not device_passed(result)


def test_a_primary_device_that_does_not_reproduce_fails() -> None:
    records = {"a": record("a", [0.9, 0.2])}
    result = device_check(records, Replay(records, shift=1e-5), Replay(records),
                          primary_device="mps", reference_device_name="cpu", spread=1)
    assert result["max_reproducibility_difference"] == pytest.approx(1e-5)
    assert not device_passed(result)


def test_an_impossible_score_from_either_device_is_an_error() -> None:
    records = {"a": record("a", [0.9, 0.2])}
    with pytest.raises(ScoreError):
        device_check(records, Replay(records), Replay(records, shift=2.0),
                     primary_device="mps", reference_device_name="cpu", spread=1)


def test_a_check_that_rescored_nothing_does_not_pass() -> None:
    lone = {"a": record("a", [0.9]), "b": record("b", [])}
    result = device_check(lone, Replay(lone), Replay(lone), primary_device="mps",
                          reference_device_name="cpu")
    assert result["slots_checked"] == 0 and result["max_abs_difference"] == 0.0
    assert not device_passed(result)
    nothing = device_check({}, Replay({}), Replay({}), primary_device="mps",
                           reference_device_name="cpu")
    assert not device_passed(nothing)


def test_progress_is_reported_after_every_slot() -> None:
    records = {f"s{i}": record(f"s{i}", [0.9, 0.5]) for i in range(3)}
    seen: list[tuple[int, int]] = []
    device_check(records, Replay(records), Replay(records), primary_device="mps",
                 reference_device_name="cpu", spread=3, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]
