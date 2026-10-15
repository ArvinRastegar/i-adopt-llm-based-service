"""Run health: tri-state checks, retrieval summary and the notes that depend on the numbers."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from iadopt_linking.errors import RunError
from iadopt_linking.health import (
    CHECK_ACCOUNTED,
    CHECK_ANSWERED,
    CHECK_CONFIG,
    CHECK_DEVICE,
    CHECK_HASHES,
    CHECK_LIMIT,
    CHECK_RANGE,
    CHECK_RELEASES,
    CHECK_RETRIEVED,
    CHECK_SCORED,
    REQUIRED_CHECKS,
    build_checks,
    ensure_checks,
    evidence_period,
    notes_for,
    retrieval_summary,
)
from iadopt_linking.population import Slot
from iadopt_linking_eval import evaluate_slot, summarise

KEYS = [f"key{i}" for i in range(20)]
SOURCES = ("wikidata", "qudt", "om")
RELEASES = {name: {"latest": {"probed_at": "t"}, "probes": 1, "failed_probes": 0,
                   "distinct_releases": 1} for name in ("om", "qudt")}


def candidate(source: str, rank: int = 1, description: str = "d", **extras: Any) -> dict:
    return {"source": source, "rank": rank, "iri": f"http://{source}.test/{rank}", "label": "l",
            "description": description, "extras": extras}


def retrieval(source: str, key: str, n: int = 1, *, error: str | None = None, attempts: int = 1,
              text: str = "{}", causes: tuple[str, ...] = (), description: str = "d",
              at: str = "2026-10-14T10:00:00+00:00", **extras: Any) -> dict:
    return {"source": source, "query_key": key, "retrieved_at": at, "http_status": 200,
            "attempts": attempts, "attempt_errors": list(causes), "latency_s": 2.0,
            "request_s": 0.5, "error": error, "parse_error": None,
            "response_sha256": hashlib.sha256(text.encode()).hexdigest(), "response_text": text,
            "candidates": None if error else [candidate(source, r, description, **extras)
                                              for r in range(1, n + 1)],
            "skipped_reason": None, "request": {"url": "u", "params": {}}}


def full_history(**overrides: Any) -> list[dict]:
    return [retrieval(s, k, **overrides) for k in KEYS for s in SOURCES]


def slots(n: int) -> list[Slot]:
    return [Slot(f"s{i}", "v", "a/b.ttl", "V", "def", "hasProperty", "hasProperty", False,
                 "Property", "t", "t", "http://www.wikidata.org/entity/Q1", "wikidata", "Q1",
                 "scheme") for i in range(n)]


def score_record(slot_id: str, scores: list[float] | None = None, **scoring: Any) -> dict:
    pool = [{"source": "wikidata", "rank": r, "score": s}
            for r, s in enumerate(scores if scores is not None else [0.9, 0.1], 1)]
    return {"slot_id": slot_id, "pool": pool, "scoring": {
        "model": "m", "revision": "r" * 12, "device": "mps", "batch_size": 16,
        "max_description_chars": 300, **scoring}}


def records(n: int, **kwargs: Any) -> dict[str, dict]:
    return {f"s{i}": score_record(f"s{i}", **kwargs) for i in range(n)}


COUNTS = {"gold_links": 7, "links": 7, "slots": 5, "excluded_label_only": 1,
          "excluded_non_vocabulary": 1}


def named(checks: list[dict], name: str) -> dict:
    return next(c for c in checks if c["name"] == name)


def all_checks(**overrides: Any) -> list[dict]:
    base: dict[str, Any] = {"slots": slots(5), "counts": COUNTS, "keys": KEYS,
                            "max_description_chars": 300, "history": full_history(),
                            "releases": RELEASES, "records": records(5)}
    return build_checks(**{**base, **overrides})


def test_a_complete_healthy_run_passes_every_check_but_the_unrun_device_check() -> None:
    checks = all_checks()
    assert [c["name"] for c in checks] == list(REQUIRED_CHECKS)
    assert [c["passed"] for c in checks] == [True] * 9 + [None]
    ensure_checks(checks, final=True)


def test_phases_that_have_not_run_report_not_run_rather_than_passing() -> None:
    checks = build_checks(slots=slots(5), counts=COUNTS, keys=KEYS, max_description_chars=300)
    states = {c["name"]: c["passed"] for c in checks}
    assert states[CHECK_ACCOUNTED] is True
    assert all(states[name] is None for name in REQUIRED_CHECKS if name != CHECK_ACCOUNTED)
    ensure_checks(checks, final=False)
    with pytest.raises(RunError, match="did not run: every key retrieved"):
        ensure_checks(checks, final=True)


def test_linked_nodes_that_do_not_add_up_fail() -> None:
    assert named(all_checks(counts={**COUNTS, "links": 8}), CHECK_ACCOUNTED)["passed"] is False
    assert named(all_checks(slots=slots(4)), CHECK_ACCOUNTED)["passed"] is False
    assert named(all_checks(counts={**COUNTS, "excluded_label_only": 0}), CHECK_ACCOUNTED)["passed"] is False


def test_a_link_the_population_loop_never_saw_is_caught_by_the_independent_count() -> None:
    counts = {**COUNTS, "gold_links": 8}      # the gold lists one more link than the loop counted
    check = named(all_checks(counts=counts), CHECK_ACCOUNTED)
    assert check["passed"] is False and "8 linked nodes in the gold" in check["detail"]
    assert named(all_checks(counts={k: v for k, v in COUNTS.items() if k != "gold_links"}),
                 CHECK_ACCOUNTED)["passed"] is False


def test_a_pair_without_a_successful_retrieval_fails_the_retrieval_check() -> None:
    history = [r for r in full_history() if (r["source"], r["query_key"]) != ("qudt", "key3")]
    check = named(all_checks(history=history), CHECK_RETRIEVED)
    assert check["passed"] is False and "1 of 60" in check["detail"]
    failed = history + [retrieval("qudt", "key3", error="HTTP 503")]
    assert named(all_checks(history=failed), CHECK_RETRIEVED)["passed"] is False


def test_a_failure_that_was_later_repeated_passes_and_is_counted() -> None:
    history = [retrieval("qudt", "key3", error="HTTP 503"), *full_history(attempts=2)]
    check = named(all_checks(history=history), CHECK_RETRIEVED)
    assert check["passed"] is True
    assert "1 failed requests were repeated successfully and 60 needed an HTTP retry" in check["detail"]


def test_a_source_that_is_almost_always_empty_fails_even_though_nothing_errored() -> None:
    history = [retrieval(s, k, n=0 if s == "wikidata" else 1) for k in KEYS for s in SOURCES]
    check = named(all_checks(history=history), CHECK_ANSWERED)
    assert check["passed"] is False and "wikidata 0/20" in check["detail"]
    sparse = [retrieval(s, k, n=0 if s == "om" and i else 1)
              for i, k in enumerate(KEYS) for s in SOURCES]
    assert named(all_checks(history=sparse), CHECK_ANSWERED)["passed"] is True  # 1/20 is the floor


def test_a_stored_response_that_no_longer_matches_its_hash_fails() -> None:
    history = full_history()
    history[7] = {**history[7], "response_text": "tampered"}
    check = named(all_checks(history=history), CHECK_HASHES)
    assert check["passed"] is False and "1 stored responses do not match" in check["detail"]


def test_a_service_without_a_usable_release_probe_fails() -> None:
    missing = {**RELEASES, "om": {**RELEASES["om"], "latest": None}}
    check = named(all_checks(releases=missing), CHECK_RELEASES)
    assert check["passed"] is False and "no usable probe of om" in check["detail"]


def test_scoring_checks_catch_missing_slots_bad_scores_and_oversized_pools() -> None:
    assert named(all_checks(records=records(4)), CHECK_SCORED)["passed"] is False
    for bad in ([0.5, float("nan")], [0.5, 1.2], [-0.01, 0.5]):
        assert named(all_checks(records={**records(4), "s4": score_record("s4", bad)}),
                     CHECK_RANGE)["passed"] is False
    crowded = score_record("s4", [0.5] * 11)
    assert named(all_checks(records={**records(4), "s4": crowded}), CHECK_LIMIT)["passed"] is False


def test_mixed_or_unidentified_scoring_configurations_fail() -> None:
    mixed = {**records(4), "s4": score_record("s4", revision="other")}
    assert named(all_checks(records=mixed), CHECK_CONFIG)["passed"] is False
    assert named(all_checks(records=records(5, max_description_chars=None)), CHECK_CONFIG)["passed"] is False
    unknown = {name: {k: v for k, v in r.items() if k != "scoring"} for name, r in records(5).items()}
    assert named(all_checks(records=unknown), CHECK_CONFIG)["passed"] is False


def test_devices_may_differ_but_are_listed() -> None:
    mixed = {**records(4), "s4": score_record("s4", device="cpu")}
    check = named(all_checks(records=mixed), CHECK_CONFIG)
    assert check["passed"] is True and "cpu 1 slots" in check["detail"] and "mps 4 slots" in check["detail"]


def test_a_failed_device_check_stops_the_run_and_an_unrun_one_does_not() -> None:
    device = {"primary_device": "mps", "reference_device": "cpu", "slots_checked": 3, "near_ties": 1,
              "spread": 2, "near_tie_margin": 1e-3, "tolerance": 1e-4,
              "reproducibility_tolerance": 1e-6, "max_abs_difference": 5e-4,
              "max_reproducibility_difference": 0.0, "selection_changes": [],
              "correct_on_primary": 2, "correct_on_reference": 2}
    bad = named(all_checks(device=device), CHECK_DEVICE)
    assert bad["passed"] is False and "5.00e-04" in bad["detail"]
    with pytest.raises(RunError, match="health check failed: device consistency"):
        ensure_checks(all_checks(device=device), final=False)
    good = named(all_checks(device={**device, "max_abs_difference": 1e-6}), CHECK_DEVICE)
    assert good["passed"] is True


def test_the_summary_counts_repeated_failures_retries_and_empty_descriptions() -> None:
    history = [
        retrieval("wikidata", "key0", error="HTTP 429 slow down"),
        retrieval("wikidata", "key0", attempts=2, causes=("HTTP 429, Retry-After 60; waited 60 s",)),
        retrieval("wikidata", "key1", n=3, description=""),
        retrieval("qudt", "key0", n=2, defined_by=["http://qudt.org/vocab/quantitykind"]),
        retrieval("qudt", "key1", defined_by=["http://ext.test/vocab"]),
        retrieval("om", "key0", n=0),
        retrieval("om", "key9", n=1)]
    out = retrieval_summary(history, ["key0", "key1"])
    wikidata = out["wikidata"]
    assert wikidata["keys"] == 2 and wikidata["failed_requests"] == 1
    assert wikidata["failure_causes"] == {"HTTP 429 slow down": 1}
    assert wikidata["retried"] == 1 and wikidata["retry_causes"] == {"HTTP 429, Retry-After 60": 1}
    assert wikidata["candidate_counts"] == {1: 1, 3: 1} and wikidata["candidates"] == 4
    assert wikidata["empty_descriptions"] == 3 and wikidata["longest_description_chars"] == 1
    assert wikidata["service_seconds"] == {"median": 0.5, "max": 0.5}
    assert out["qudt"]["distinct_kinds"] == 2 and out["qudt"]["extension_kinds"] == 1
    assert out["om"]["keys"] == 1 and out["om"]["empty"] == 1


def test_the_evidence_period_spans_the_answers_in_use() -> None:
    history = [retrieval("wikidata", "key0", at="2026-10-14T09:00:00+00:00"),
               retrieval("wikidata", "key0", at="2026-10-14T11:00:00+00:00"),
               retrieval("qudt", "key0", at="2026-10-14T10:00:00+00:00"),
               retrieval("qudt", "elsewhere", at="2026-10-01T00:00:00+00:00")]
    assert evidence_period(history, ["key0"]) == {
        "first_retrieval": "2026-10-14T10:00:00+00:00", "last_retrieval": "2026-10-14T11:00:00+00:00"}
    assert evidence_period([], ["key0"]) == {"first_retrieval": None, "last_retrieval": None}


def outcome_summary() -> dict:
    def cand(source: str, local: str, score: float) -> dict:
        return {"source": source, "rank": 1, "vocabulary": source, "local_id": local,
                "label": local, "score": score}

    def slot(sid: str, gold: tuple[str, str], pool: list[dict]) -> dict:
        return {"slot_id": sid, "variable_id": "v", "role": "hasProperty", "slot": "hasProperty",
                "is_member": False, "term": "t", "query_key": "t",
                "gold": {"vocabulary": gold[0], "local_id": gold[1]}, "pool": pool}

    both = [cand("wikidata", "Q1", 0.5), cand("wikidata", "Q2", 0.4995)]
    return summarise([evaluate_slot(slot("a", ("wikidata", "Q1"), both)),
                      evaluate_slot(slot("b", ("iobc", "1"), both))])


def notes(**overrides: Any) -> list[str]:
    retrieval_info = retrieval_summary(full_history(description=""), KEYS)
    device = {"near_ties": 1, "spread": 4, "slots_checked": 5, "primary_device": "mps",
              "reference_device": "cpu", "selection_changes": [{}],
              "correct_on_primary": 3, "correct_on_reference": 2}
    base: dict[str, Any] = {
        "counts": {"gold_form_repaired": 2, "gold_form_variant": 5}, "retrieval": retrieval_info,
        "summary": outcome_summary(), "device": device,
        "releases": {"om": {"distinct_releases": 1}, "qudt": {"distinct_releases": 1}},
        "period": {"first_retrieval": "A", "last_retrieval": "B"}, "max_description_chars": 300}
    return notes_for(**{**base, **overrides})


def test_notes_state_this_runs_numbers_not_constants() -> None:
    text = "\n".join(notes())
    assert "between A and B" in text and "cut to 300 characters" in text
    assert "no description for any of the 20 OM candidates" in text
    assert "1 slots have gold in a vocabulary these sources cannot reach (iobc)" in text
    assert "2 malformed IRIs, 5 in a variant path form" in text
    assert "1 of the 1 distinct kinds retrieved are defined outside the core" in text
    assert ("rescored 1 near-ties and 4 other slots on cpu: 1 selections changed; over those 5 "
            "slots the correct count was 3 on mps and 2 on cpu") in text
    assert "2 of 2 selections were decided by a top-two margin below 1e-3" in text


def test_notes_without_a_device_check_say_so_and_uncapped_descriptions_say_not_cut() -> None:
    text = "\n".join(notes(device=None, max_description_chars=None))
    assert "the cross-device check was not run for this report" in text and "Descriptions are not cut" in text


def test_notes_flag_retrieval_that_spanned_two_releases() -> None:
    releases = {"om": {"distinct_releases": 2}, "qudt": {"distinct_releases": 1}}
    assert any("spanned more than one release of om" in n for n in notes(releases=releases))
