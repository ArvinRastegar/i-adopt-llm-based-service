"""Committed Jev results must still be what their manifest says, made by the code that is here."""

from __future__ import annotations

import functools
import json
import re
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from iadopt_linking.artifacts import file_sha256
from iadopt_linking.chat_experiment import EXPERIMENT_ID as PSNC_EXPERIMENT_ID
from iadopt_linking.jev import PASSES, candidate_key, pass_spec, prompt_fingerprint
from iadopt_linking.jev_experiment import (
    EXPERIMENT_ID,
    PUBLISHED_JEV,
    JevPaths,
    build_outputs,
    jev_evidence_identity,
    run_jev,
)
from iadopt_linking.jev_health import REQUIRED_JEV_CHECKS, hash_matches
from iadopt_linking.jev_inputs import build_tasks, load_baseline
from iadopt_linking.recorder import JsonlStore

RESULTS = Path(__file__).resolve().parents[1] / "results"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


MANIFESTS = sorted(p for p in RESULTS.glob("*/manifest.json")
                   if read(p).get("experiment") == EXPERIMENT_ID)
EACH = pytest.mark.parametrize("manifest_path", MANIFESTS, ids=lambda p: p.parent.name)


@functools.cache
def rebuilt(folder: Path) -> tuple[Any, ...]:
    """Re-derive everything a folder publishes from its stored responses and this tree's code."""
    manifest, plan = read(folder / "manifest.json"), read(folder / "plan.json")
    baseline = load_baseline(RESULTS / plan["baseline"]["folder"], only=plan["only"])
    specs = [pass_spec(name) for name, _arm, _order in plan["passes"]]
    tasks, _ = build_tasks(baseline, specs, session_id=f"iadopt-linking/{plan['run_id']}")
    cap = manifest["spend"]["cap_usd"]
    return build_outputs(JevPaths(folder), baseline, tasks, plan, read(folder / "provenance.json"),
                         specs, None if cap is None else Decimal(cap), manifest["created_at"])


@EACH
def test_files_are_the_ones_the_manifest_hashed(manifest_path: Path) -> None:
    manifest = read(manifest_path)
    assert set(manifest["files"]) == set(PUBLISHED_JEV)
    for name, expected in manifest["files"].items():
        assert file_sha256(manifest_path.parent / name) == expected, (
            f"{manifest_path.parent.name}/{name} changed after the run")


@EACH
def test_evidence_was_made_by_the_evidence_code_in_this_tree(manifest_path: Path) -> None:
    manifest = read(manifest_path)
    assert manifest["evidence_code_sha256"] == jev_evidence_identity(), (
        "the code that decides what Jev is asked or how it is read changed since this run; its "
        "evidence is stale. Start a new results folder.")
    assert read(manifest_path.parent / "plan.json")["evidence_code_sha256"] == (
        manifest["evidence_code_sha256"])


@EACH
def test_a_committed_result_is_complete_and_passed_every_check(manifest_path: Path) -> None:
    manifest = read(manifest_path)
    assert [c["name"] for c in manifest["checks"]] == list(REQUIRED_JEV_CHECKS)
    assert all(c["passed"] is True for c in manifest["checks"])
    assert manifest["pilot_variables"] is None, "a pilot is not a result"
    assert [p[0] for p in manifest["passes"]] == [p.name for p in PASSES], "every pass was run"


@EACH
def test_the_journal_ends_in_a_completed_invocation_by_the_same_evidence_code(
        manifest_path: Path) -> None:
    manifest = read(manifest_path)
    entries = read(manifest_path.parent / "provenance.json")["invocations"]
    assert entries and entries[-1]["outcome"] == "completed"
    assert all(e["outcome"] != "running" and e["finished_at"] for e in entries)
    assert {e["evidence_code_sha256"] for e in entries} == {manifest["evidence_code_sha256"]}
    assert len(manifest["invocations"]) == len(entries)


@EACH
def test_every_stored_response_matches_its_hash(manifest_path: Path) -> None:
    records = JsonlStore(manifest_path.parent / "responses.jsonl").read()
    assert records and all(hash_matches(r) for r in records)


@EACH
def test_it_was_compared_with_the_baseline_that_is_committed(manifest_path: Path) -> None:
    plan = read(manifest_path.parent / "plan.json")
    baseline = read(RESULTS / plan["baseline"]["folder"] / "manifest.json")
    assert plan["baseline"]["scores_sha256"] == baseline["files"]["scores.jsonl"]
    assert plan["baseline"]["candidates_sha256"] == baseline["files"]["candidates.jsonl"]


@EACH
def test_the_prompt_in_the_plan_is_the_frozen_one(manifest_path: Path) -> None:
    assert read(manifest_path.parent / "plan.json")["prompt"]["fingerprint"] == prompt_fingerprint()


@EACH
def test_no_candidate_was_shown_that_the_baseline_did_not_score(manifest_path: Path) -> None:
    plan = read(manifest_path.parent / "plan.json")
    baseline = load_baseline(RESULTS / plan["baseline"]["folder"], only=plan["only"])
    for record in JsonlStore(manifest_path.parent / "responses.jsonl").read():
        pool = {candidate_key(c) for c in baseline.records[record["slot_id"]]["pool"]}
        assert set(record["shown"]) == pool and len(record["shown"]) == len(pool)


@EACH
def test_the_summary_tables_and_reports_are_what_the_stored_responses_give(
        manifest_path: Path) -> None:
    folder = manifest_path.parent
    manifest = read(manifest_path)
    expected_manifest, summary, texts = rebuilt(folder)
    assert read(folder / "summary.json") == summary, "rebuild the report: iadopt-linking jev-run"
    for name, text in texts.items():
        assert (folder / name).read_text(encoding="utf-8") == text, f"{name} differs from a rebuild"
    assert {k: v for k, v in manifest.items() if k != "files"} == expected_manifest


@EACH
def test_the_repeat_pass_sent_exactly_the_requests_of_the_first_pass(manifest_path: Path) -> None:
    sent: dict[tuple[str, str], str] = {}
    for record in JsonlStore(manifest_path.parent / "responses.jsonl").read():
        sent[(record["slot_id"], record["pass"])] = record["request_sha256"]
    first = {slot: sha for (slot, name), sha in sent.items() if name == "choice-a1"}
    again = {slot: sha for (slot, name), sha in sent.items() if name == "choice-a1-repeat"}
    assert first and first == again, "the repeat measures Jev's own variability only if it is identical"


@EACH
def test_the_other_orders_differ_from_the_first(manifest_path: Path) -> None:
    sent: dict[tuple[str, str], str] = {}
    for record in JsonlStore(manifest_path.parent / "responses.jsonl").read():
        sent[(record["slot_id"], record["pass"])] = record["request_sha256"]
    first = {slot: sha for (slot, name), sha in sent.items() if name == "choice-a1"}
    for other in ("choice-a2-reversed", "choice-a3"):
        changed = sum(sent[(slot, other)] != sha for slot, sha in first.items())
        assert changed > 0.8 * len(first), f"{other} should show most pools in another order"


def test_there_is_at_least_one_committed_jev_result() -> None:
    assert MANIFESTS, ("no results/*/manifest.json names the Jev experiment: a Jev folder without a "
                       "manifest is not a result, and every test below would silently be skipped")


def test_every_committed_result_belongs_to_a_suite_that_checks_it() -> None:
    for path in RESULTS.glob("*/manifest.json"):
        experiment = read(path).get("experiment")
        assert experiment in (None, EXPERIMENT_ID, PSNC_EXPERIMENT_ID), (
            f"{path.parent.name} names experiment {experiment!r}, which no integrity test covers: "
            "add its suite before committing it")


def test_a_folder_with_evidence_but_no_manifest_is_not_a_result() -> None:
    for folder in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        evidence = any((folder / name).is_file() for name in ("responses.jsonl", "scores.jsonl"))
        assert not evidence or (folder / "manifest.json").is_file(), (
            f"{folder.name} holds evidence but no manifest.json (a run that stopped removes it); "
            "finish the run or move the folder out of results/")


def test_no_committed_result_holds_a_key_or_an_account_id() -> None:
    patterns = ("sk-or-", '"user_id"', "Authorization", "Bearer ")
    for folder in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        for path in sorted(folder.iterdir()):
            text = path.read_text(encoding="utf-8")
            for pattern in patterns:
                assert pattern not in text, f"{folder.name}/{path.name} contains {pattern!r}"


class NoRequestsAllowed:
    """A client factory for a rebuild, which must not ask for anything."""

    def __call__(self) -> Any:
        raise AssertionError("rebuilding a finished folder must not build a client")


def at_no_time(text: str) -> str:
    """Blank what a rebuild changes in a report's text: when it was built and by which code."""
    text = re.sub(r"report built \d{4}-\d{2}-\d{2}T[\d:+\-]+", "report built <time>", text)
    return re.sub(r"all code `[0-9a-f]+`", "all code `<hash>`", text)


@EACH
def test_the_documented_rebuild_binds_to_the_committed_plan_and_changes_none_of_its_data(
        manifest_path: Path, tmp_path: Path) -> None:
    """Rerunning jev-run in a copy is how report-side changes are published: it must not alter data.

    The engine was generalised after these folders were committed; this is what shows that a Jev
    folder still binds to the plan it holds and that every file but the report and the manifest
    comes out byte for byte as committed.
    """
    committed = manifest_path.parent
    copy = tmp_path / committed.name
    shutil.copytree(committed, copy)
    plan, manifest = read(committed / "plan.json"), read(manifest_path)
    run_jev(copy, baseline_dir=RESULTS / plan["baseline"]["folder"],
            max_cost_usd=manifest["spend"]["cap_usd"], only=plan["only"],
            client_factory=NoRequestsAllowed())
    files = manifest["files"]
    for name in ("plan.json", "responses.jsonl", "summary.json", "outcomes.csv", "confidence.csv",
                 "probabilities.csv", "comparison.md"):
        assert file_sha256(copy / name) == files[name], f"{name} is not what the committed run holds"
    assert at_no_time((copy / "report.md").read_text(encoding="utf-8")) == at_no_time(
        (committed / "report.md").read_text(encoding="utf-8"))
    assert read(copy / "manifest.json")["checks"] == manifest["checks"]
