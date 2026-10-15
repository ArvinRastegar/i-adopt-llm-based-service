"""Provenance: what code made the evidence, what the services served, what each invocation did."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from iadopt_linking.errors import RunError
from iadopt_linking.http import Fetched
from iadopt_linking.provenance import (
    EVIDENCE_FILES,
    JOURNAL_SCHEMA,
    begin_invocation,
    code_identity,
    evidence_code_identity,
    fetch_om_release,
    fetch_qudt_release,
    invocation_summaries,
    load_journal,
    merge_releases,
    model_summary,
    summed_seconds,
)
from iadopt_linking.sources import OLS_ONTOLOGY_ENDPOINT, QUDT_ENDPOINT


class Client:
    def __init__(self, status: int = 200, body: str | None = "{}", error: str | None = None) -> None:
        self.status, self.body, self.error = status, body, error
        self.calls: list[str] = []

    def get(self, url: str, params: Any = None, headers: Any = None) -> Fetched:
        self.calls.append(url)
        return Fetched(url, self.status, self.body, 0.1, 1, self.error)


def project(tmp_path: Path) -> Path:
    for name in EVIDENCE_FILES:
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return tmp_path


def test_the_evidence_hash_follows_every_listed_module_and_nothing_else(tmp_path: Path) -> None:
    root = project(tmp_path)
    before = evidence_code_identity(root)
    (root / "src/iadopt_linking/report.py").write_text("# unrelated\n", encoding="utf-8")
    assert evidence_code_identity(root) == before
    for name in EVIDENCE_FILES:
        path = root / "src" / name
        path.write_text(path.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8")
        assert evidence_code_identity(root) != before, name
        path.write_text(f"# {name}\n", encoding="utf-8")
    assert evidence_code_identity(root) == before


def test_the_real_evidence_list_names_files_that_exist() -> None:
    assert len(evidence_code_identity()) == 64


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.test",
                    *args], check=True, capture_output=True)


def test_code_identity_outside_a_repository_has_no_git_facts(tmp_path: Path) -> None:
    identity = code_identity(project(tmp_path))
    assert identity["git_base_commit"] is None and identity["git_dirty"] is None
    assert len(identity["source_sha256"]) == 64 and identity["files"] == len(EVIDENCE_FILES) + 1


def test_code_identity_is_dirty_only_for_code_not_for_results_or_outputs(tmp_path: Path) -> None:
    root = project(tmp_path)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "base")
    clean = code_identity(root)
    assert clean["git_dirty"] is False and len(clean["git_base_commit"]) == 40
    (root / "results").mkdir()
    (root / "results/x.json").write_text("{}", encoding="utf-8")
    (root / "outputs").mkdir()
    (root / "outputs/log").write_text("x", encoding="utf-8")
    assert code_identity(root)["git_dirty"] is False
    (root / "src/iadopt_linking/report.py").write_text("# new\n", encoding="utf-8")
    dirty = code_identity(root)
    assert dirty["git_dirty"] is True and dirty["source_sha256"] != clean["source_sha256"]


def test_an_om_release_probe_keeps_what_the_service_says_and_when() -> None:
    body = json.dumps({"version": "2.0.1", "loaded": "2026-10-09", "fileHash": "h",
                       "numberOfTerms": 7, "status": "LOADED", "ignored": 1})
    client = Client(body=body)
    probe = fetch_om_release(client)
    assert client.calls == [OLS_ONTOLOGY_ENDPOINT] and "ignored" not in probe
    assert (probe["version"], probe["numberOfTerms"], probe["status"]) == ("2.0.1", 7, "LOADED")
    assert probe["probed_at"].endswith("+00:00")


@pytest.mark.parametrize("client", [
    Client(status=503, body=None, error="HTTP 503"), Client(body="not json"), Client(body="[]"),
    Client(body=json.dumps({"status": "LOADED"}))])
def test_an_om_probe_without_release_data_is_an_error_not_an_empty_success(client: Client) -> None:
    assert "error" in fetch_om_release(client)


def test_a_qudt_release_probe_reads_the_graph_modification_time() -> None:
    body = json.dumps({"results": {"bindings": [{"modified": {"value": "2026-09-22T13:10:01Z"}}]}})
    client = Client(body=body)
    probe = fetch_qudt_release(client)
    assert client.calls == [QUDT_ENDPOINT] and probe["graph_modified"] == "2026-09-22T13:10:01Z"


@pytest.mark.parametrize("client", [
    Client(status=500, body=None, error="HTTP 500"), Client(body="not json"), Client(body="{}"),
    Client(body=json.dumps({"results": {"bindings": []}})),
    Client(body=json.dumps({"results": {"bindings": [{"other": {}}]}}))])
def test_a_qudt_probe_without_a_modification_time_is_an_error(client: Client) -> None:
    assert "error" in fetch_qudt_release(client)


def invocation(om: dict | None = None, qudt: dict | None = None, **extra: Any) -> dict:
    return {"source_releases": None if om is None and qudt is None else {
        **({"om": om} if om is not None else {}), **({"qudt": qudt} if qudt is not None else {})},
        "seconds": {}, "model": None, **extra}


def test_releases_are_merged_across_invocations_and_failed_probes_are_counted() -> None:
    first = invocation({"probed_at": "t1", "version": "1"}, {"probed_at": "t1", "graph_modified": "m"})
    second = invocation({"probed_at": "t2", "version": "1"}, {"probed_at": "t2", "error": "HTTP 500"})
    third = invocation()
    merged = merge_releases([first, second, third])
    assert merged["om"] == {"latest": {"probed_at": "t2", "version": "1"}, "probes": 2,
                            "failed_probes": 0, "distinct_releases": 1}
    assert merged["qudt"]["latest"] == {"probed_at": "t1", "graph_modified": "m"}
    assert (merged["qudt"]["probes"], merged["qudt"]["failed_probes"]) == (2, 1)
    assert merged["wikidata"].startswith("live search API")


def test_a_changed_release_between_invocations_is_visible() -> None:
    merged = merge_releases([invocation({"probed_at": "a", "version": "1"}, None),
                             invocation({"probed_at": "b", "version": "2"}, None)])
    assert merged["om"]["distinct_releases"] == 2 and merged["qudt"]["latest"] is None


def test_a_new_folder_gets_an_empty_journal(tmp_path: Path) -> None:
    journal = load_journal(tmp_path / "provenance.json", evidence_exists=False)
    assert journal == {"schema": JOURNAL_SCHEMA, "invocations": [], "device_check": None}


def test_evidence_without_a_journal_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="holds evidence but no provenance.json"):
        load_journal(tmp_path / "provenance.json", evidence_exists=True)


@pytest.mark.parametrize("text", ["not json", "[]", json.dumps({"schema": 99, "invocations": []}),
                                  json.dumps({"schema": JOURNAL_SCHEMA, "invocations": {}})])
def test_a_journal_that_cannot_be_trusted_is_refused_not_replaced(tmp_path: Path, text: str) -> None:
    path = tmp_path / "provenance.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RunError):
        load_journal(path, evidence_exists=True)
    assert path.read_text(encoding="utf-8") == text


def test_beginning_an_invocation_opens_a_running_entry_with_the_environment(tmp_path: Path) -> None:
    journal = {"schema": JOURNAL_SCHEMA, "invocations": [], "device_check": None}
    entry = begin_invocation(journal, evidence_sha256="e" * 64, http_settings={"max_attempts": 3})
    assert journal["invocations"] == [entry]
    assert entry["outcome"] == "running" and entry["finished_at"] is None
    assert entry["evidence_code_sha256"] == "e" * 64 and entry["http"] == {"max_attempts": 3}
    assert entry["python"] and entry["platform"] and len(entry["code"]["source_sha256"]) == 64


def test_phase_times_add_up_over_invocations() -> None:
    entries = [{"seconds": {"retrieval": 10.04, "scoring": 1.0}}, {"seconds": {"scoring": 2.26}}, {"seconds": {}}]
    assert summed_seconds(entries) == {"retrieval": 10.0, "scoring": 3.3}


def test_the_manifest_summary_of_invocations_keeps_outcome_and_code() -> None:
    entry = {"started_at": "a", "finished_at": "b", "outcome": "completed", "seconds": {"scoring": 1.26},
             "retrieval": {"requested": 3}, "scoring": None, "python": "3.12.1", "platform": "p",
             "code": {"source_sha256": "s", "git_dirty": True}, "http": {"x": 1}, "model": None}
    (summary,) = invocation_summaries([entry])
    assert summary == {"started_at": "a", "finished_at": "b", "outcome": "completed",
                       "seconds": {"scoring": 1.3}, "retrieval": {"requested": 3}, "scoring": None,
                       "python": "3.12.1", "platform": "p", "source_sha256": "s", "git_dirty": True}


def test_the_model_summary_comes_from_the_stored_scores_and_the_latest_loaded_model() -> None:
    def rec(device: str) -> dict:
        return {"scoring": {"model": "m", "revision": "r", "batch_size": 16, "device": device}}

    summary = model_summary({"a": rec("mps"), "b": rec("mps"), "c": rec("cpu")}, [
        {"model": None}, {"model": {"torch": "2.9", "sentence_transformers": "5.1", "transformers": "4.5"}}])
    assert summary == {"model": "m", "revision": "r", "batch_size": 16,
                       "devices": {"cpu": 1, "mps": 2},
                       "libraries": {"torch": "2.9", "sentence_transformers": "5.1", "transformers": "4.5"}}
    assert model_summary({}, [])["devices"] == {} and model_summary({}, [])["libraries"] == {}
