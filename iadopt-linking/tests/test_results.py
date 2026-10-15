"""Committed results must still be what their manifest says, made by the code that is here."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest

from iadopt_linking import experiment
from iadopt_linking.artifacts import file_sha256
from iadopt_linking.experiment import evidence_code_identity, load_population
from iadopt_linking.health import REQUIRED_CHECKS
from iadopt_linking.recorder import JsonlStore, latest
from iadopt_linking.report import outcomes_csv, render_report
from iadopt_linking.retrieve import response_hash_matches
from iadopt_linking_eval import evaluate_slot, summarise

RESULTS = Path(__file__).resolve().parents[1] / "results"
MANIFESTS = sorted(RESULTS.glob("*/manifest.json"))
EACH = pytest.mark.parametrize("manifest_path", MANIFESTS, ids=lambda p: p.parent.name)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@functools.cache
def outcomes_of(folder: Path) -> tuple[Any, ...]:
    """Judge a folder's stored scores with the evaluation code in this tree."""
    manifest = read(folder / "manifest.json")
    scored = latest(JsonlStore(folder / "scores.jsonl").read(), lambda r: r["slot_id"])
    _, slots, _ = load_population(only=manifest["pilot_variables"])
    return tuple(evaluate_slot(scored[slot.slot_id]) for slot in slots)


def test_there_is_at_least_one_committed_result() -> None:
    assert MANIFESTS, "no results/*/manifest.json: a results folder is missing or incomplete"


@EACH
def test_files_are_the_ones_the_manifest_hashed(manifest_path: Path) -> None:
    manifest = read(manifest_path)
    assert set(manifest["files"]) == set(experiment.PUBLISHED)
    for name, expected in manifest["files"].items():
        assert file_sha256(manifest_path.parent / name) == expected, (
            f"{manifest_path.parent.name}/{name} changed after the run")


@EACH
def test_evidence_was_made_by_the_evidence_code_in_this_tree(manifest_path: Path) -> None:
    manifest = read(manifest_path)
    assert manifest["evidence_code_sha256"] == evidence_code_identity(), (
        "retrieval, scoring or gold-reading code changed since this run; its evidence is stale. "
        "Start a new results folder (a rerun in this folder is refused on purpose).")
    assert read(manifest_path.parent / "plan.json")["evidence_code_sha256"] == (
        manifest["evidence_code_sha256"])


@EACH
def test_a_committed_result_is_complete_and_passed_every_check(manifest_path: Path) -> None:
    manifest = read(manifest_path)
    assert [c["name"] for c in manifest["checks"]] == list(REQUIRED_CHECKS)
    assert all(c["passed"] is True for c in manifest["checks"]), (
        "a committed result passes every check, the cross-device check included")
    assert manifest["pilot_variables"] is None, "a pilot is not a result"
    assert manifest["device_check"]["scores_sha256"] == manifest["files"]["scores.jsonl"]


@EACH
def test_the_journal_ends_in_a_completed_invocation_by_the_same_evidence_code(
        manifest_path: Path) -> None:
    manifest = read(manifest_path)
    journal = read(manifest_path.parent / "provenance.json")
    entries = journal["invocations"]
    assert entries and entries[-1]["outcome"] == "completed"
    assert all(e["outcome"] != "running" and e["finished_at"] for e in entries)
    assert {e["evidence_code_sha256"] for e in entries} == {manifest["evidence_code_sha256"]}
    assert len(manifest["invocations"]) == len(entries)


@EACH
def test_every_stored_response_matches_its_hash(manifest_path: Path) -> None:
    records = JsonlStore(manifest_path.parent / "candidates.jsonl").read()
    assert records and all(response_hash_matches(r) for r in records)


@EACH
def test_the_summary_csv_and_report_are_what_the_stored_scores_give(manifest_path: Path) -> None:
    folder = manifest_path.parent
    manifest = read(manifest_path)
    outcomes = list(outcomes_of(folder))
    summary = summarise(outcomes)
    assert read(folder / "summary.json") == summary, "rebuild the report: iadopt-linking run --out"
    assert (folder / "outcomes.csv").read_text(encoding="utf-8") == outcomes_csv(outcomes)
    rendered = render_report({k: v for k, v in manifest.items() if k != "files"}, summary, outcomes)
    assert (folder / "report.md").read_text(encoding="utf-8") == rendered
