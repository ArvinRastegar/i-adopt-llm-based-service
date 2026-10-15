"""Run-to-run comparison, on two tiny synthetic result folders."""

from __future__ import annotations

from pathlib import Path

import pytest

from iadopt_linking.cli import main
from iadopt_linking.compare import compare_runs, render_comparison
from iadopt_linking.errors import UsageError
from iadopt_linking.recorder import JsonlStore


def retrieval(source: str, key: str, iris: list[str]) -> dict:
    return {"source": source, "query_key": key, "error": None, "parse_error": None,
            "candidates": [{"iri": i} for i in iris]}


def scored(slot_id: str, chosen: str, score: float, gold: str = "Q1") -> dict:
    candidates = [("Q1", 0.4), ("Q2", 0.6)]
    pool = [{"source": "wikidata", "rank": rank, "iri": f"http://www.wikidata.org/entity/{q}",
             "vocabulary": "wikidata", "local_id": q, "label": q.lower(),
             "score": score if q == chosen else s}
            for rank, (q, s) in enumerate(candidates, 1)]
    return {"slot_id": slot_id, "variable_id": "v", "source_path": "a/b.ttl", "role": "hasProperty",
            "slot": "hasProperty", "is_member": False, "term": "t", "query_key": "t",
            "gold": {"vocabulary": "wikidata", "local_id": gold}, "pool": pool}


def build(root: Path, wikidata_iris: list[str], chosen: str) -> Path:
    root.mkdir(parents=True)
    JsonlStore(root / "candidates.jsonl").append(retrieval("wikidata", "t", wikidata_iris))
    JsonlStore(root / "candidates.jsonl").append(retrieval("qudt", "t", []))
    JsonlStore(root / "scores.jsonl").append(scored("s1", chosen, 0.9))
    return root


def test_identical_runs_agree_completely(tmp_path: Path) -> None:
    first = build(tmp_path / "a", ["x", "y"], "Q2")
    second = build(tmp_path / "b", ["x", "y"], "Q2")
    result = compare_runs(first, second)
    assert result["retrieval"]["wikidata"] == {
        "keys": 1, "identical": 1, "reordered": 0, "different": 0, "unavailable": 0}
    assert result["selection_changed"] == 0 and result["max_score_delta"] == 0.0
    assert result["correct_first"] == result["correct_second"] == 0


def test_reordered_changed_and_flipped_results_are_counted(tmp_path: Path) -> None:
    first = build(tmp_path / "a", ["x", "y"], "Q2")
    second = build(tmp_path / "b", ["y", "x"], "Q1")
    result = compare_runs(first, second)
    assert result["retrieval"]["wikidata"]["reordered"] == 1
    assert result["selection_changed"] == 1 and (result["correct_first"], result["correct_second"]) == (0, 1)
    assert result["changed_examples"][0]["first"] == ("wikidata", "Q2")
    assert result["max_score_delta"] > 0
    text = render_comparison(result, ("a", "b"))
    assert "Slots whose selection changed" in text and "wrong -> correct" in text


def test_a_failed_retrieval_is_unavailable_not_an_empty_list(tmp_path: Path) -> None:
    first = build(tmp_path / "a", ["x"], "Q2")
    second = build(tmp_path / "b", ["x"], "Q2")
    failed = retrieval("wikidata", "t", [])
    failed.update(error="HTTP 503", candidates=None)
    JsonlStore(second / "candidates.jsonl").append(failed)
    result = compare_runs(first, second)
    assert result["retrieval"]["wikidata"] == {
        "keys": 1, "identical": 0, "reordered": 0, "different": 0, "unavailable": 1}


def test_the_compare_command_prints_markdown(tmp_path: Path, capsys) -> None:
    first = build(tmp_path / "a", ["x"], "Q2")
    second = build(tmp_path / "b", ["x", "z"], "Q2")
    assert main(["compare", "--first", str(first), "--second", str(second)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Comparison of a and b") and "| wikidata | 1 | 0 | 0 | 1 | 0 |" in out


def test_a_folder_without_evidence_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    good = build(tmp_path / "a", ["x"], "Q2")
    with pytest.raises(UsageError, match="candidates.jsonl is missing"):
        compare_runs(empty, good)
    with pytest.raises(SystemExit) as stopped:
        main(["compare", "--first", str(empty), "--second", str(good)])
    assert stopped.value.code == 2
