"""The whole run, offline: canned responses, a deterministic scorer, every gate exercised."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from iadopt_linking import experiment
from iadopt_linking.artifacts import file_sha256
from iadopt_linking.cli import main
from iadopt_linking.errors import RunError, UsageError
from iadopt_linking.experiment import (
    PUBLISHED,
    LazyScorer,
    Paths,
    evidence_code_identity,
    load_population,
    query_keys,
    run,
)
from iadopt_linking.http import Fetched
from iadopt_linking.retrieve import RetrievalTimeout
from iadopt_linking.sources import (
    OLS_ENDPOINT,
    OLS_ONTOLOGY_ENDPOINT,
    QUDT_ENDPOINT,
    WIKIDATA_ENDPOINT,
)

FIXTURES = Path(__file__).parent / "fixtures"
PILOT = ["EWCT", "C8_CWDFraction"]
EMPTY_WIKIDATA = json.dumps({"search": [], "success": 1})
# Reading the corpus takes about a second; the pinned corpus cannot change under the tests.
CACHED_GOLD = functools.cache(experiment.load_gold_links)


@pytest.fixture(autouse=True)
def read_the_corpus_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment, "load_gold_links", CACHED_GOLD)


class CannedClient:
    """Serves the captured responses; can fail one host or answer one source with nothing."""

    def __init__(self, fail: str | None = None, silent: str | None = None,
                 release_fails: bool = False) -> None:
        self.fail, self.silent, self.release_fails = fail, silent, release_fails
        self.calls: list[str] = []
        self.closed = False

    def settings(self) -> dict[str, Any]:
        return {"user_agent": "canned", "max_attempts": 3}

    def get(self, url: str, params: Any = None, headers: Any = None) -> Fetched:
        self.calls.append(url)
        if self.fail and self.fail in url:
            return Fetched(url, 503, None, 0.1, 3, "HTTP 503")
        if url == OLS_ONTOLOGY_ENDPOINT:
            if self.release_fails:
                return Fetched(url, 500, None, 0.1, 3, "HTTP 500")
            body = json.dumps({"version": "2.0.test", "loaded": "2026-10-09", "fileHash": "h",
                               "numberOfTerms": 1, "status": "LOADED"})
        elif url == WIKIDATA_ENDPOINT:
            body = EMPTY_WIKIDATA if self.silent == "wikidata" else (
                FIXTURES / "wikidata_blood.json").read_text(encoding="utf-8")
        elif url == QUDT_ENDPOINT and "dcterms:modified" in params["query"]:
            if self.release_fails:
                return Fetched(url, 500, None, 0.1, 3, "HTTP 500")
            body = json.dumps({"results": {"bindings": [
                {"modified": {"value": "2026-09-22T13:10:01Z"}}]}})
        elif url == QUDT_ENDPOINT:
            body = (FIXTURES / "qudt_mass_concentration.json").read_text(encoding="utf-8")
        else:
            body = (FIXTURES / "om_area_fraction.json").read_text(encoding="utf-8")
        return Fetched(url, 200, body, 0.1, 1, None)

    def close(self) -> None:
        self.closed = True


class Overlap:
    """Deterministic scorer: the share of the document's words that the query also contains."""

    def __init__(self, device: str = "stub", invert: bool = False, fail_after: int | None = None):
        self.device, self.invert, self.fail_after, self.calls = device, invert, fail_after, 0

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt
        words = set(query.lower().split())
        values = [len(words & set(d.lower().split())) / max(1, len(d.split())) for d in documents]
        return [1.0 - v for v in values] if self.invert else values

    def describe(self) -> dict[str, Any]:
        return {"model": "overlap-stub", "revision": "stub-revision", "device": self.device,
                "batch_size": 16, "torch": "t", "sentence_transformers": "s", "transformers": "x"}


class Factory:
    """Builds stub scorers and remembers which devices were asked for."""

    def __init__(self, invert_cpu: bool = False, fail_after: int | None = None) -> None:
        self.invert_cpu, self.fail_after = invert_cpu, fail_after
        self.built: list[str | None] = []

    def __call__(self, device: str | None) -> Overlap:
        self.built.append(device)
        return Overlap(device or "stub", invert=device == "cpu" and self.invert_cpu,
                       fail_after=None if device == "cpu" else self.fail_after)


def execute(out: Path, client: CannedClient | None = None, factory: Factory | None = None,
            **kwargs: Any) -> tuple[dict[str, Any], CannedClient, Factory]:
    client, factory = client or CannedClient(), factory or Factory()
    kwargs.setdefault("only", PILOT)
    manifest = run(out, client_factory=lambda: client, scorer_factory=factory,
                   run_id="test-run", **kwargs)
    return manifest, client, factory


def journal(out: Path) -> dict[str, Any]:
    return json.loads((out / "provenance.json").read_text(encoding="utf-8"))


def test_a_pilot_run_writes_every_artifact_and_passes_every_gate(tmp_path: Path) -> None:
    manifest, client, _ = execute(tmp_path)
    for name in (*PUBLISHED, "manifest.json"):
        assert (tmp_path / name).is_file(), name
    assert [c["passed"] for c in manifest["checks"]] == [True] * 10
    assert manifest["population"]["slots"] == len(load_population(only=PILOT)[1])
    assert manifest["pilot_variables"] == sorted(PILOT)
    assert manifest["source_releases"]["om"]["latest"]["version"] == "2.0.test"
    assert manifest["source_releases"]["qudt"]["latest"]["graph_modified"] == "2026-09-22T13:10:01Z"
    assert manifest["retrieval"]["qudt"]["extension_kinds"] == 0
    assert manifest["model"]["model"] == "overlap-stub" and manifest["model"]["devices"] == {
        "stub": manifest["population"]["slots"]}
    assert manifest["device_check"]["reference_device"] == "cpu" and manifest["device_check"]["passed"]
    assert client.closed
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert set(on_disk["files"]) == set(PUBLISHED)
    for name, digest in on_disk["files"].items():
        assert file_sha256(tmp_path / name) == digest, name
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    for heading in ("## Result", "## Where the misses come from", "## Which source wins",
                    "## By slot, role and term length", "## Run health", "## Notes"):
        assert heading in report
    assert "| device consistency | pass |" in report


def test_retrieval_asks_each_source_once_per_key_and_the_journal_says_what_happened(
        tmp_path: Path) -> None:
    manifest, client, _ = execute(tmp_path)
    keys = query_keys(load_population(only=PILOT)[1])
    assert client.calls.count(WIKIDATA_ENDPOINT) == len(keys)
    assert client.calls.count(OLS_ENDPOINT) == len(keys)
    (entry,) = journal(tmp_path)["invocations"]
    assert entry["outcome"] == "completed" and entry["finished_at"] and entry["retrieval"]["requested"]
    assert set(entry["seconds"]) == {"retrieval", "scoring", "device_check"}
    assert entry["model"]["revision"] == "stub-revision" and entry["http"]["user_agent"] == "canned"
    assert manifest["invocations"][0]["outcome"] == "completed"
    assert set(manifest["timings_s"]) == {"retrieval", "scoring", "device_check"}
    assert manifest["evidence_period"]["first_retrieval"] <= manifest["evidence_period"]["last_retrieval"]


def test_a_second_run_calls_no_service_loads_no_model_and_changes_no_evidence(
        tmp_path: Path) -> None:
    execute(tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in ("candidates.jsonl", "scores.jsonl", "plan.json")}
    manifest, client, factory = execute(tmp_path)
    assert client.calls == [] and factory.built == []
    assert before == {name: (tmp_path / name).read_bytes() for name in before}
    assert [c["passed"] for c in manifest["checks"]] == [True] * 10
    assert [i["outcome"] for i in manifest["invocations"]] == ["completed", "completed"]
    assert manifest["invocations"][1]["retrieval"] == {"skipped": 3 * len(query_keys(
        load_population(only=PILOT)[1]))}


def test_recheck_devices_repeats_the_cross_check(tmp_path: Path) -> None:
    execute(tmp_path)
    _, _, factory = execute(tmp_path, recheck_devices=True)
    assert factory.built == [None, "cpu"]


def test_changed_scores_invalidate_the_stored_device_check(tmp_path: Path) -> None:
    execute(tmp_path)
    scores = (tmp_path / "scores.jsonl")
    last = scores.read_text(encoding="utf-8").splitlines()[-1]
    scores.write_text(scores.read_text(encoding="utf-8") + last + "\n", encoding="utf-8")
    _, _, factory = execute(tmp_path)
    assert factory.built == [None, "cpu"]


def test_the_manifest_binds_the_evidence_to_its_code_and_the_folder_to_its_plan(
        tmp_path: Path) -> None:
    manifest, _, _ = execute(tmp_path)
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert manifest["evidence_code_sha256"] == evidence_code_identity() == plan["evidence_code_sha256"]
    assert plan["only"] == sorted(PILOT) and plan["config"]["max_description_chars"] == 300
    assert plan["corpus"]["tag"] and plan["query_keys"] == query_keys(load_population(only=PILOT)[1])
    assert manifest["platform"] and manifest["code"]["source_sha256"]


def edit_plan(field: str, value: Any) -> Callable[[Path], None]:
    def edit(out: Path) -> None:
        plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        plan[field] = value
        (out / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return edit


@pytest.mark.parametrize(("damage", "message"), [
    (edit_plan("evidence_code_sha256", "0" * 64), "different evidence-producing code"),
    (edit_plan("only", ["EWCT"]), "a different set of variables"),
    (edit_plan("config", {"max_description_chars": 5}), "a different configuration"),
    (edit_plan("corpus", {"tag": "v0"}), "a different corpus release"),
    (edit_plan("query_keys", ["other"]), "a different set of queries"),
    (lambda out: (out / "plan.json").write_text("{not json", encoding="utf-8"), "cannot be read"),
    (lambda out: (out / "plan.json").unlink(), "holds evidence but no plan.json"),
    (lambda out: (out / "provenance.json").unlink(), "holds evidence but no provenance.json"),
    (lambda out: (out / "provenance.json").write_text("[]", encoding="utf-8"), "is not a version"),
])
def test_a_folder_that_does_not_match_this_run_is_not_resumed(
        tmp_path: Path, damage: Callable[[Path], None], message: str) -> None:
    execute(tmp_path)
    damage(tmp_path)
    before = (tmp_path / "scores.jsonl").read_bytes()
    with pytest.raises(RunError, match=message):
        execute(tmp_path)
    assert (tmp_path / "scores.jsonl").read_bytes() == before
    assert (tmp_path / "manifest.json").exists()  # a refused request touches nothing


def test_a_different_description_cap_is_not_resumed(tmp_path: Path) -> None:
    execute(tmp_path)
    with pytest.raises(RunError, match="a different configuration"):
        execute(tmp_path, max_description_chars=120)


def test_a_failing_source_stops_the_run_before_scoring_and_the_next_run_resumes(
        tmp_path: Path) -> None:
    with pytest.raises(RunError, match="retrievals failed"):
        execute(tmp_path, client=CannedClient(fail="qudt.org"))
    for name in ("report.md", "manifest.json", "scores.jsonl", "summary.json"):
        assert not (tmp_path / name).exists(), name
    (stopped,) = journal(tmp_path)["invocations"]
    assert stopped["outcome"].startswith("stopped: RunError") and stopped["finished_at"]
    manifest, client, _ = execute(tmp_path)
    assert WIKIDATA_ENDPOINT not in client.calls and OLS_ENDPOINT not in client.calls
    assert [c["passed"] for c in manifest["checks"]] == [True] * 10
    assert [i["outcome"][:7] for i in manifest["invocations"]] == ["stopped", "complet"]
    retrieved = next(c for c in manifest["checks"] if c["name"] == "every key retrieved from every source")
    assert "failed requests were repeated successfully" in retrieved["detail"]
    assert manifest["retrieval"]["qudt"]["failed_requests"] > 0


def test_a_source_that_answers_nothing_is_caught_before_any_scoring(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="health check failed: every source answered for some keys"):
        execute(tmp_path, client=CannedClient(silent="wikidata"))
    assert not (tmp_path / "scores.jsonl").exists() and not (tmp_path / "report.md").exists()


def test_a_deadline_stops_retrieval_and_the_next_run_finishes_it(tmp_path: Path) -> None:
    with pytest.raises(RetrievalTimeout):
        execute(tmp_path, max_retrieval_minutes=1e-9)
    assert journal(tmp_path)["invocations"][0]["outcome"].startswith("stopped: RetrievalTimeout")
    manifest, _, _ = execute(tmp_path)
    assert [c["passed"] for c in manifest["checks"]] == [True] * 10


def test_a_release_probe_that_fails_stops_the_run_and_only_the_probe_is_retried(
        tmp_path: Path) -> None:
    with pytest.raises(RunError, match="source releases recorded"):
        execute(tmp_path, client=CannedClient(release_fails=True))
    assert not (tmp_path / "scores.jsonl").exists()
    manifest, client, _ = execute(tmp_path)
    assert client.calls == [OLS_ONTOLOGY_ENDPOINT, QUDT_ENDPOINT]
    assert manifest["source_releases"]["om"]["probes"] == 2
    assert manifest["source_releases"]["om"]["failed_probes"] == 1


def test_a_device_that_disagrees_stops_the_run_and_leaves_no_report(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="health check failed: device consistency"):
        execute(tmp_path, factory=Factory(invert_cpu=True))
    for name in ("report.md", "manifest.json"):
        assert not (tmp_path / name).exists(), name
    stored = journal(tmp_path)["device_check"]
    assert stored["passed"] is False and stored["max_abs_difference"] > stored["tolerance"]
    assert journal(tmp_path)["invocations"][0]["outcome"].startswith("stopped: RunError")


def test_the_device_check_can_be_skipped_but_the_report_then_says_it_did_not_run(
        tmp_path: Path) -> None:
    manifest, _, factory = execute(tmp_path, check_devices=False)
    assert factory.built == [None] and manifest["device_check"] is None
    last = manifest["checks"][-1]
    assert (last["name"], last["passed"]) == ("device consistency", None)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| device consistency | not run |" in report
    assert "the cross-device check was not run for this report" in report


def test_a_failed_recheck_removes_the_manifest_of_the_earlier_run(tmp_path: Path) -> None:
    execute(tmp_path)
    assert (tmp_path / "manifest.json").exists()
    with pytest.raises(RunError, match="device consistency"):
        execute(tmp_path, factory=Factory(invert_cpu=True), recheck_devices=True)
    assert not (tmp_path / "manifest.json").exists()


def test_an_interruption_is_journalled_and_scoring_resumes_where_it_stopped(
        tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, factory=Factory(fail_after=3))
    (stopped,) = journal(tmp_path)["invocations"]
    assert stopped["outcome"].startswith("stopped: KeyboardInterrupt")
    partial = (tmp_path / "scores.jsonl").read_bytes()
    assert partial.count(b"\n") == 3 and not (tmp_path / "manifest.json").exists()
    manifest, _, _ = execute(tmp_path)
    assert (tmp_path / "scores.jsonl").read_bytes().startswith(partial)
    assert manifest["invocations"][1]["scoring"]["skipped"] == 3
    assert [c["passed"] for c in manifest["checks"]] == [True] * 10


def test_scores_made_in_two_invocations_are_attributed_to_the_devices_that_made_them(
        tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, factory=Factory(fail_after=3), device="stub-a")
    manifest, _, _ = execute(tmp_path, device="stub-b")
    assert manifest["model"]["devices"] == {"stub-a": 3, "stub-b": manifest["population"]["slots"] - 3}


def test_a_scorer_is_not_built_until_it_is_used() -> None:
    built: list[str | None] = []

    def factory(device: str | None) -> Overlap:
        built.append(device)
        return Overlap()

    lazy = LazyScorer(factory, "mps")
    assert not lazy.loaded and built == []
    assert lazy.score("q", ["a b"]) == [0.0] and lazy.loaded and built == ["mps"]
    assert lazy.describe()["model"] == "overlap-stub" and built == ["mps"]

    class Bare:
        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            return [0.5 for _ in documents]

    assert LazyScorer(lambda device: Bare(), None).describe() == {}


def test_a_folder_is_new_until_evidence_is_written(tmp_path: Path) -> None:
    paths = Paths(tmp_path)
    assert not paths.has_evidence()
    (tmp_path / "candidates.jsonl").write_text("", encoding="utf-8")
    assert not paths.has_evidence()
    (tmp_path / "candidates.jsonl").write_text("{}\n", encoding="utf-8")
    assert paths.has_evidence()


def test_the_plan_command_reports_without_calling_anything(capsys: pytest.CaptureFixture) -> None:
    assert main(["plan", "--only", ",".join(PILOT)]) == 0
    out = capsys.readouterr().out
    assert "population counts" in out and "distinct queries" in out


def test_an_unknown_variable_is_rejected_before_anything_is_written(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="No such variable"):
        load_population(only=["NoSuchVariable"])
    with pytest.raises(UsageError, match="No such variable"):
        execute(tmp_path / "out", only=["NoSuchVariable"])
    assert not (tmp_path / "out" / "plan.json").exists()
