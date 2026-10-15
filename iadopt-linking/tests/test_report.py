"""The Markdown report and the outcomes CSV, rendered from small synthetic runs."""

from __future__ import annotations

import csv
import io
from typing import Any

from iadopt_linking.report import (
    CSV_COLUMNS,
    EXAMPLES_PER_CATEGORY,
    outcomes_csv,
    ratio,
    render_report,
    value,
)
from iadopt_linking_eval import evaluate_slot, summarise


def cand(source: str, rank: int, local_id: str, score: float, **extras: Any) -> dict:
    return {"source": source, "rank": rank, "vocabulary": source, "local_id": local_id,
            "label": local_id.lower(), "score": score, "extras": extras}


def slot(sid: str, gold: tuple[str, str], pool: list[dict], variable: str = "v1",
         term: str = "mass") -> dict:
    return {"slot_id": sid, "variable_id": variable, "source_path": f"A/{variable}.ttl",
            "role": "hasProperty", "slot": "hasProperty", "is_member": False, "term": term,
            "query_key": term, "gold": {"vocabulary": gold[0], "local_id": gold[1]}, "pool": pool}


def manifest(checks: list[dict] | None = None, notes: list[str] | None = None) -> dict:
    return {"title": "A title", "run_id": "run-1", "created_at": "2026-10-14T12:00:00+00:00",
            "evidence_period": {"first_retrieval": "2026-10-14T09:00:00+00:00",
                                "last_retrieval": "2026-10-14T09:20:00+00:00"},
            "corpus": {"tag": "v2.0.1", "commit": "0123456789abcdef"},
            "evidence_code_sha256": "e" * 64, "code": {"source_sha256": "s" * 64},
            "model": {"model": "the-model", "revision": "r" * 12, "devices": {"mps": 5, "cpu": 1}},
            "checks": checks or [], "notes": notes or []}


def run(records: list[dict]) -> tuple[dict, list[dict]]:
    outcomes = [evaluate_slot(r) for r in records]
    return summarise(outcomes), outcomes


RECORDS = [
    slot("a", ("wikidata", "Q1"), [cand("wikidata", 1, "Q1", 0.9)], "v1"),
    slot("b", ("wikidata", "Q2"), [cand("wikidata", 1, "Q3", 0.9), cand("wikidata", 2, "Q2", 0.1)], "v1"),
    slot("c", ("qudt", "Mass"), [cand("wikidata", 1, "Q9", 0.9), cand("qudt", 1, "Mass", 0.2)], "v2"),
    slot("d", ("wikidata", "Q4"), [cand("om", 1, "X", 0.5)], "v2"),
    slot("e", ("wikidata", "Q5"), [], "v3"),
    slot("f", ("iobc", "9"), [cand("wikidata", 1, "Q9", 0.9)], "v3"),
]


def test_receipts_are_shown_as_counts_and_three_decimals() -> None:
    assert ratio({"numerator": 194, "denominator": 294}) == "194/294 = 0.660"
    assert value({"numerator": 194, "denominator": 294}) == "0.660"
    assert ratio({"numerator": 0, "denominator": 0}) == "n/a" and value({"numerator": 0, "denominator": 0}) == "n/a"


def test_the_csv_has_one_row_per_slot_and_blank_selection_fields_when_nothing_was_selected() -> None:
    _, outcomes = run(RECORDS)
    rows = list(csv.DictReader(io.StringIO(outcomes_csv(outcomes))))
    assert tuple(rows[0]) == CSV_COLUMNS and len(rows) == len(RECORDS)
    first, empty = rows[0], rows[4]
    assert (first["slot_id"], first["category"], first["correct"], first["selected_local_id"]) == (
        "a", "correct", "1", "Q1")
    assert first["selected_score"] == "0.900000" and first["source_path"] == "A/v1.ttl"
    assert (empty["category"], empty["selected_source"], empty["selected_score"], empty["pool_size"]) == (
        "no_candidates", "", "", "0")


def test_the_report_has_every_section_and_this_runs_numbers() -> None:
    summary, outcomes = run(RECORDS)
    checks = [{"name": "first", "passed": True, "detail": "d1"},
              {"name": "second", "passed": False, "detail": "d2"},
              {"name": "third", "passed": None, "detail": "d3"}]
    text = render_report(manifest(checks, ["a note"]), summary, outcomes)
    for heading in ("# A title", "## Result", "## Comparison with the manuscript",
                    "## Where the misses come from", "## Diagnostics", "## Which source wins",
                    "## Confidence threshold", "## By slot, role and term length",
                    "## Most repeated terms", "## Examples of misses", "## Run health", "## Notes"):
        assert heading in text, heading
    assert "| All controlled-vocabulary slots | 6 |" in text and "1/6 = 0.167" in text
    assert "| Gold in another vocabulary (not searchable here) | 1 | 1 | 0/1 = 0.000 | 0.000 |" in text
    assert "| first | pass | d1 |" in text and "| second | FAIL | d2 |" in text
    assert "| third | not run | d3 |" in text and "- a note" in text
    assert "OM candidates have no description" not in text  # stated in the notes, from the data
    assert "Reranker `the-model` (revision `rrrrrrrrrr`), scored on mps (5 slots), cpu (1 slots)" in text
    assert "2026-10-14T09:00:00+00:00 to 2026-10-14T09:20:00+00:00" in text and "0123456789" in text


def test_the_manuscript_section_quotes_the_paper_and_labels_what_is_compared() -> None:
    summary, outcomes = run(RECORDS)
    text = render_report(manifest(), summary, outcomes)
    section = text.split("## Comparison with the manuscript")[1].split("## Where the misses")[0]
    assert "0.656" in section and "0.759" in section and "quoted, not reproduced" in section
    assert "First Wikidata hit, no reranker (naive)" in section
    assert "Qwen3 reranker on the Wikidata candidates alone" in section
    assert "paper/tracking" not in text and "Table 10" not in text


def test_without_wikidata_gold_there_is_no_manuscript_comparison() -> None:
    summary, outcomes = run([slot("c", ("qudt", "Mass"), [cand("qudt", 1, "Mass", 0.9)])])
    assert "Comparison with the manuscript" not in render_report(manifest(), summary, outcomes)


def test_examples_are_capped_per_outcome_and_ordered_by_slot() -> None:
    many = [slot(f"s{i:02d}", ("wikidata", "Q1"), [cand("wikidata", 1, "Q2", 0.9)], term=f"t{i}")
            for i in range(EXAMPLES_PER_CATEGORY + 4)]
    summary, outcomes = run(many)
    text = render_report(manifest(), summary, outcomes)
    block = text.split("**gold not retrieved**")[1].split("## Run health")[0]
    listed = [line for line in block.splitlines() if line.startswith("- `")]
    assert len(listed) == EXAMPLES_PER_CATEGORY
    assert listed[0].startswith("- `t0`") and listed[-1].startswith(f"- `t{EXAMPLES_PER_CATEGORY - 1}`")
    assert f"({EXAMPLES_PER_CATEGORY + 4} slots)" in text


def test_the_routing_table_and_threshold_grid_come_from_the_summary() -> None:
    summary, outcomes = run(RECORDS)
    text = render_report(manifest(), summary, outcomes)
    assert "| Gold in \\ chosen from | wikidata | qudt | om |" in text
    assert "| qudt | 1 | 0 | 0 |" in text
    assert "| 0.999 | 0/6 = 0.000 | n/a |" in text
    assert "| 0.000 | 5/6 = 0.833 | 1/5 = 0.200 |" in text
