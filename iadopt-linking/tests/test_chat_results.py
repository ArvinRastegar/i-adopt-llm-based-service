"""Committed PSNC results must still be what their manifest says, made by the code that is here."""

from __future__ import annotations

import csv
import functools
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from iadopt_linking.artifacts import file_sha256
from iadopt_linking.chat import scrub
from iadopt_linking.chat_choice import CHAT_PASSES, build_tasks, prompt_fingerprint
from iadopt_linking.chat_experiment import (
    EXPERIMENT_ID,
    PSNC_EVIDENCE_FILES,
    REQUIRED_CHAT_CHECKS,
    chat_experiment,
    run_chat,
)
from iadopt_linking.jev import candidate_key
from iadopt_linking.jev_experiment import PUBLISHED_JEV, JevPaths, build_outputs
from iadopt_linking.jev_health import hash_matches
from iadopt_linking.jev_inputs import load_baseline
from iadopt_linking.provenance import evidence_code_identity
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
    by_name = {spec.name: spec for spec in CHAT_PASSES}
    specs = [by_name[name] for name, _arm, _order in plan["passes"]]
    model = plan["model"]["requested"]
    tasks, _ = build_tasks(baseline, specs, model=model)
    return build_outputs(JevPaths(folder), baseline, tasks, plan, read(folder / "provenance.json"),
                         specs, None, manifest["created_at"], chat_experiment(model),
                         manifest["spend"]["max_requests"])


def same_up_to_rounding(expected: Any, actual: Any, where: str = "summary.json") -> None:
    """Assert two JSON values are equal, floats to a relative 1e-9.

    A probability here is exp() of a log-probability, and another platform's C library may round
    exp() differently in the last digit, which would move every float derived from it by a few units
    of the sixteenth place and nothing else: this must not look like a changed result.
    """
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(expected, (int, float)) and isinstance(actual, (int, float)), where
        assert math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-12), (
            f"{where}: {expected!r} against {actual!r}")
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), where
        for key in expected:
            same_up_to_rounding(expected[key], actual[key], f"{where}/{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), where
        for index, (item, other) in enumerate(zip(expected, actual, strict=True)):
            same_up_to_rounding(item, other, f"{where}[{index}]")
    else:
        assert expected == actual and type(expected) is type(actual), f"{where}: {expected!r}"


def test_the_comparison_that_tolerates_rounding_tolerates_nothing_else() -> None:
    same_up_to_rounding({"a": [1, 0.1 + 0.2, "x"], "b": None}, {"a": [1, 0.3, "x"], "b": None})
    for other in ({"a": [1, 0.30000001, "x"], "b": None}, {"a": [1, 0.3, "y"], "b": None},
                  {"a": [2, 0.3, "x"], "b": None}, {"a": [1, 0.3], "b": None},
                  {"a": [1, 0.3, "x"], "c": None}, {"a": [1, True, "x"], "b": None}):
        with pytest.raises(AssertionError):
            same_up_to_rounding({"a": [1, 0.3, "x"], "b": None}, other)


def test_there_is_at_least_one_committed_psnc_result() -> None:
    assert MANIFESTS, ("no results/*/manifest.json names the PSNC experiment: a folder without a "
                       "manifest is not a result, and every test below would silently be skipped")


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
    assert manifest["evidence_code_sha256"] == evidence_code_identity(files=PSNC_EVIDENCE_FILES), (
        "the code that decides what the model is asked or how it is read changed since this run; "
        "its evidence is stale. Start a new results folder.")
    assert read(manifest_path.parent / "plan.json")["evidence_code_sha256"] == (
        manifest["evidence_code_sha256"])


@EACH
def test_a_committed_result_is_complete_and_passed_every_check(manifest_path: Path) -> None:
    manifest = read(manifest_path)
    assert [c["name"] for c in manifest["checks"]] == list(REQUIRED_CHAT_CHECKS)
    assert all(c["passed"] is True for c in manifest["checks"])
    assert manifest["pilot_variables"] is None, "a pilot is not a result"
    assert [p[0] for p in manifest["passes"]] == [p.name for p in CHAT_PASSES], "every pass was run"


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
def test_it_was_compared_with_the_baseline_and_the_jev_run_that_are_committed(
        manifest_path: Path) -> None:
    plan = read(manifest_path.parent / "plan.json")
    baseline = read(RESULTS / plan["baseline"]["folder"] / "manifest.json")
    assert plan["baseline"]["scores_sha256"] == baseline["files"]["scores.jsonl"]
    assert plan["baseline"]["candidates_sha256"] == baseline["files"]["candidates.jsonl"]
    reference = plan["reference"]
    jev = read(RESULTS / reference["folder"] / "manifest.json")
    assert reference["experiment"] == jev["experiment"] == "jev-entity-linking-v1"
    for field, name in (("responses_sha256", "responses.jsonl"), ("plan_sha256", "plan.json"),
                        ("outcomes_sha256", "outcomes.csv")):
        assert reference[field] == jev["files"][name] == file_sha256(
            RESULTS / reference["folder"] / name), f"the Jev run's {name} is not the one compared"
    assert "manifest_sha256" not in reference, "a rebuilt reference report must not break this run"


@EACH
def test_the_prompt_in_the_plan_is_the_frozen_one(manifest_path: Path) -> None:
    plan = read(manifest_path.parent / "plan.json")
    assert plan["prompt"]["fingerprint"] == prompt_fingerprint(plan["model"]["requested"])


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
    same_up_to_rounding(read(folder / "summary.json"), summary)
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
    assert first and first == again, "the repeat measures the service's own variability only if identical"


@EACH
def test_the_other_orders_differ_from_the_first(manifest_path: Path) -> None:
    sent: dict[tuple[str, str], str] = {}
    for record in JsonlStore(manifest_path.parent / "responses.jsonl").read():
        sent[(record["slot_id"], record["pass"])] = record["request_sha256"]
    first = {slot: sha for (slot, name), sha in sent.items() if name == "choice-a1"}
    for other in ("choice-a2-reversed", "choice-a3"):
        changed = sum(sent[(slot, other)] != sha for slot, sha in first.items())
        assert changed > 0.8 * len(first), f"{other} should show most pools in another order"


@EACH
def test_reasoning_was_off_in_every_request_and_in_every_stored_response(
        manifest_path: Path) -> None:
    plan = read(manifest_path.parent / "plan.json")
    assert plan["prompt"]["request_parameters"]["chat_template_kwargs"] == {"enable_thinking": False}
    for record in JsonlStore(manifest_path.parent / "responses.jsonl").read():
        if record["response_text"] is None:
            continue
        message = json.loads(record["response_text"])["choices"][0]["message"]
        assert not message.get("reasoning_content") and not message.get("reasoning")


@EACH
def test_requests_were_sent_one_at_a_time_to_the_published_gateway(manifest_path: Path) -> None:
    entries = read(manifest_path.parent / "provenance.json")["invocations"]
    asking = [e for e in entries if e["asking"]]
    assert asking and all(e["workers"] == 1 for e in asking)
    assert all(e["client"].get("host") == "llm.hpc.psnc.pl" for e in asking)
    assert all(e["client"]["redirects"] == "not followed" for e in asking)


@EACH
def test_nothing_secret_or_internal_was_published(manifest_path: Path) -> None:
    """The stored text is a fixed point of the scrubber: it holds no key, id, token or private address."""
    folder = manifest_path.parent
    for path in sorted(folder.iterdir()):
        text = path.read_text(encoding="utf-8")
        assert scrub(text) == text, f"{folder.name}/{path.name} holds something the scrubber removes"
        assert not re.search(r"(?i)x-litellm|llm_provider-", text), f"{path.name} holds a gateway header"


# --- an independent reading of the raw files (no project code) ---

SOURCES = ("wikidata", "qudt", "om")
LABELS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def independent_top1(folder: Path) -> tuple[dict[str, int], int, dict[str, bool]]:
    """Recompute each pass's reachable top-1 from the raw responses and the baseline's scores.

    Uses nothing from the project: the probability of an option is the sum of exp(logprob) over the
    first token's top log-probabilities whose stripped text is its label, normalised over the labels
    found; the choice is the highest probability among the real candidates, ties by source order
    (wikidata, qudt, om) then rank. This is the rule written in DECISIONS L-016 and L-017.
    """
    plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    base = {}
    for line in (RESULTS / plan["baseline"]["folder"] / "scores.jsonl").read_text().splitlines():
        record = json.loads(line)
        base[record["slot_id"]] = record
    latest: dict[tuple[str, str], dict] = {}
    for line in (folder / "responses.jsonl").read_text().splitlines():
        record = json.loads(line)
        latest[(record["slot_id"], record["pass"])] = record
    reachable = {s for s, r in base.items() if r["gold"]["vocabulary"] in SOURCES}
    right: dict[str, int] = {}
    primary_right: dict[str, bool] = {}
    for name in sorted({p for _, p in latest}):
        count = 0
        for slot_id in reachable:
            pool = base[slot_id]["pool"]
            if not pool:                      # nothing to choose from: a miss, as in the baseline
                primary_right[slot_id] = False
                continue
            record = latest[(slot_id, name)]
            data = json.loads(record["response_text"])
            tops = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            mass: dict[str, float] = {}
            for entry in tops:
                label = entry["token"].strip()
                if label == "0" or label in LABELS:
                    mass[label] = mass.get(label, 0.0) + math.exp(entry["logprob"])
            total = math.fsum(mass.values())
            index = {f"{c['vocabulary']}:{c['local_id']}": i for i, c in enumerate(pool)}
            score = [0.0] * len(pool)
            for place, key in enumerate(record["shown"], start=1):
                score[index[key]] = mass.get(LABELS[place - 1], 0.0) / total
            best = min(range(len(pool)), key=lambda i: (
                -score[i], SOURCES.index(pool[i]["source"]), pool[i]["rank"]))
            gold = (base[slot_id]["gold"]["vocabulary"], base[slot_id]["gold"]["local_id"])
            hit = (pool[best]["vocabulary"], pool[best]["local_id"]) == gold
            count += hit
            if name == "choice-a1":
                primary_right[slot_id] = hit
        right[name] = count
    return right, len(reachable), primary_right


@EACH
def test_the_headline_is_what_an_independent_reading_of_the_raw_files_gives(
        manifest_path: Path) -> None:
    folder = manifest_path.parent
    summary = read(folder / "summary.json")
    right, reachable, primary = independent_top1(folder)
    assert reachable == summary["primary"]["populations"]["reachable"]["top1"]["denominator"]
    for name, headline in summary["analysis"]["passes"].items():
        assert headline["reachable"]["numerator"] == right[name], f"{name} differs from a plain reading"
    # and the paired counts against Jev's published per-slot outcomes
    plan = read(folder / "plan.json")
    with (RESULTS / plan["reference"]["folder"] / "outcomes.csv").open(newline="") as handle:
        jev = {row["slot_id"]: row["jev_correct"] == "1" for row in csv.DictReader(handle)}
    paired = summary["analysis"]["reference"]["primary_vs_reference"]["reachable"]
    assert paired["both"] == sum(primary[s] and jev[s] for s in primary)
    assert paired["only_first"] == sum(primary[s] and not jev[s] for s in primary)
    assert paired["only_second"] == sum(jev[s] and not primary[s] for s in primary)
    assert paired["neither"] == sum(not primary[s] and not jev[s] for s in primary)
    with (RESULTS / plan["baseline"]["folder"] / "outcomes.csv").open(newline="") as handle:
        qwen3 = {row["slot_id"]: row["correct"] == "1" for row in csv.DictReader(handle)}
    against = summary["analysis"]["primary_vs_qwen3"]["reachable"]
    assert against["both"] == sum(primary[s] and qwen3[s] for s in primary)
    assert against["only_first"] == sum(primary[s] and not qwen3[s] for s in primary)
    assert against["only_second"] == sum(qwen3[s] and not primary[s] for s in primary)
    assert against["neither"] == sum(not primary[s] and not qwen3[s] for s in primary)


def at_no_time(text: str) -> str:
    """Blank the moment a report was built, the one thing a rebuild changes in its text."""
    return re.sub(r"report built \d{4}-\d{2}-\d{2}T[\d:+\-]+", "report built <time>", text)


class NoRequestsAllowed:
    """A client factory for a rebuild, which must not ask for anything."""

    def __call__(self) -> Any:
        raise AssertionError("rebuilding a finished folder must not build a client")


@EACH
def test_the_documented_rebuild_binds_to_the_committed_plan_and_reproduces_its_data(
        manifest_path: Path, tmp_path: Path) -> None:
    """Rerunning psnc-run in a copy of the folder is how report-side changes are published."""
    committed = manifest_path.parent
    copy = tmp_path / committed.name
    shutil.copytree(committed, copy)
    plan = read(committed / "plan.json")
    run_chat(copy, baseline_dir=RESULTS / plan["baseline"]["folder"], max_requests=None,
             only=plan["only"], reference=RESULTS / plan["reference"]["folder"],
             client_factory=NoRequestsAllowed())
    for name in ("plan.json", "responses.jsonl", "outcomes.csv", "confidence.csv", "probabilities.csv"):
        assert file_sha256(copy / name) == file_sha256(committed / name), name
    same_up_to_rounding(read(copy / "summary.json"), read(committed / "summary.json"))
    for name in ("report.md", "comparison.md"):
        assert at_no_time((copy / name).read_text(encoding="utf-8")) == at_no_time(
            (committed / name).read_text(encoding="utf-8")), f"{name} is not what this code writes"
    assert read(copy / "manifest.json")["checks"] == read(committed / "manifest.json")["checks"]
