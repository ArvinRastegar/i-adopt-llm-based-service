"""The code that made the committed results: pinned hashes, same logic, rebuilt, never extended."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from iadopt_linking.artifacts import file_sha256
from iadopt_linking.chat_experiment import PSNC_EVIDENCE_FILES
from iadopt_linking.errors import RunError
from iadopt_linking.experiment import USER_AGENT, run
from iadopt_linking.http import PoliteClient
from iadopt_linking.jev_experiment import JEV_EVIDENCE_FILES, run_jev
from iadopt_linking.provenance import EVIDENCE_FILES, PROJECT_ROOT, RESULTS_CODE, RESULTS_EVIDENCE

RESULTS = PROJECT_ROOT / "results"
BASELINE = RESULTS / "baseline-20261014"
ALL_EVIDENCE = sorted({*EVIDENCE_FILES, *JEV_EVIDENCE_FILES, *PSNC_EVIDENCE_FILES})
EXEMPT = {"iadopt_linking/jev_inputs.py": {"_verify_manifest"}}  # the guard that accepts RESULTS_CODE


def at_results_code(name: str) -> bytes:
    """A file of iadopt-linking as it was at RESULTS_CODE; skips when this clone lacks that history."""
    git = ["git", "-C", str(PROJECT_ROOT)]
    try:
        held = subprocess.run([*git, "cat-file", "-e", f"{RESULTS_CODE}^{{commit}}"],
                              capture_output=True).returncode == 0
    except OSError:
        held = False
    if not held:
        pytest.skip(f"needs the git history that holds commit {RESULTS_CODE[:7]}")
    shown = subprocess.run([*git, "show", f"{RESULTS_CODE}:./{name}"], capture_output=True)
    assert shown.returncode == 0, f"{name} is missing at {RESULTS_CODE[:7]}"
    return shown.stdout


def logic(source: str | bytes, exempt: set[str]) -> str:
    """A module's syntax tree without docstrings, its exempt definitions or their provenance import."""
    tree = ast.parse(source)
    if exempt:
        tree.body = [node for node in tree.body if getattr(node, "name", None) not in exempt
                     and not (isinstance(node, ast.ImportFrom) and node.level == 1
                              and node.module == "provenance")]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class NoRequestsAllowed:
    def __call__(self) -> Any:
        raise AssertionError("no client may be built")


@pytest.mark.parametrize("name, files", [("baseline", EVIDENCE_FILES), ("jev", JEV_EVIDENCE_FILES),
                                         ("psnc", PSNC_EVIDENCE_FILES)])
def test_the_pinned_evidence_hashes_are_those_of_the_results_code(name: str, files) -> None:
    digest = hashlib.sha256()
    for module in files:
        digest.update(module.encode() + b"\0" + at_results_code(f"src/{module}"))
    assert RESULTS_EVIDENCE[name] == digest.hexdigest()


@pytest.mark.parametrize("module", ALL_EVIDENCE)
def test_the_evidence_logic_is_unchanged_since_the_results_code(module: str) -> None:
    """Docstrings may change; a change in what the code does needs a new results folder."""
    exempt = EXEMPT.get(module, set())
    now = (PROJECT_ROOT / "src" / module).read_text(encoding="utf-8")
    assert logic(now, exempt) == logic(at_results_code(f"src/{module}"), exempt), (
        f"{module} behaves differently from {RESULTS_CODE[:7]}; start a new results folder")


def test_a_results_code_folder_with_a_request_pending_is_refused(tmp_path: Path) -> None:
    """A folder made at RESULTS_CODE may be rebuilt, never extended by different code."""
    committed = RESULTS / "jev-20261015"
    copy = tmp_path / committed.name
    shutil.copytree(committed, copy)
    responses = copy / "responses.jsonl"
    lines = responses.read_text(encoding="utf-8").splitlines(keepends=True)
    responses.write_text("".join(lines[:-1]), encoding="utf-8")
    plan, manifest = read(committed / "plan.json"), read(committed / "manifest.json")
    with pytest.raises(RunError, match="different evidence-producing code"):
        run_jev(copy, baseline_dir=RESULTS / plan["baseline"]["folder"],
                max_cost_usd=manifest["spend"]["cap_usd"], only=plan["only"],
                client_factory=NoRequestsAllowed())


def no_transport(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("a rebuild must not send a request")


def no_scorer(*_args: Any) -> Any:
    raise AssertionError("a rebuild must not load a scorer")


def without_time_or_code(report: str) -> str:
    """Blank what a rebuild changes in a report: when it was built and the hash of all code."""
    report = re.sub(r"report built \d{4}-\d{2}-\d{2}T[\d:+\-]+", "report built <time>", report)
    return re.sub(r"all code `[0-9a-f]+`", "all code <hash>", report)


def rebuild_baseline(copy: Path) -> None:
    plan = read(BASELINE / "plan.json")
    run(copy, only=plan["only"], max_description_chars=plan["config"]["max_description_chars"],
        client_factory=lambda: PoliteClient(USER_AGENT, transport=no_transport),
        scorer_factory=no_scorer)


def test_the_committed_baseline_rebuilds_in_place_without_a_request(tmp_path: Path) -> None:
    """Its report may be rebuilt by this tree's code; its evidence and numbers stay as they are."""
    copy = tmp_path / BASELINE.name
    shutil.copytree(BASELINE, copy)
    rebuild_baseline(copy)
    for name in ("candidates.jsonl", "scores.jsonl", "plan.json", "outcomes.csv"):
        assert file_sha256(copy / name) == file_sha256(BASELINE / name), name
    assert read(copy / "summary.json") == read(BASELINE / "summary.json")
    assert without_time_or_code((copy / "report.md").read_text(encoding="utf-8")) == (
        without_time_or_code((BASELINE / "report.md").read_text(encoding="utf-8")))
    rebuilt, committed = read(copy / "manifest.json"), read(BASELINE / "manifest.json")
    assert rebuilt["evidence_code_sha256"] == RESULTS_EVIDENCE["baseline"]
    assert rebuilt["checks"] == committed["checks"]


@pytest.mark.parametrize("name, fields", [("scores.jsonl", ("slot_id",)),
                                          ("candidates.jsonl", ("source", "query_key"))])
def test_a_results_code_baseline_with_work_pending_is_refused(tmp_path: Path, name: str,
                                                              fields: tuple[str, ...]) -> None:
    """A missing score or retrieval would be made by this tree's code, so the folder is refused."""
    copy = tmp_path / BASELINE.name
    shutil.copytree(BASELINE, copy)
    lines = (copy / name).read_text(encoding="utf-8").splitlines(keepends=True)
    gone = [json.loads(lines[0])[field] for field in fields]
    kept = [line for line in lines if [json.loads(line)[field] for field in fields] != gone]
    (copy / name).write_text("".join(kept), encoding="utf-8")
    with pytest.raises(RunError, match="different evidence-producing code"):
        rebuild_baseline(copy)
