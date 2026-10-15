"""Retrieval: every outcome is recorded, only successes are skipped on resume, loops are bounded."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iadopt_linking.http import Fetched
from iadopt_linking.recorder import JsonlStore
from iadopt_linking.retrieve import RetrievalTimeout, is_ok, response_hash_matches, retrieve
from iadopt_linking.sources import Candidate, Request, SourceError

NOW = "2026-10-13T12:00:00+00:00"


class FakeSource:
    def __init__(self, name: str, fail_parse_for: str | None = None, blank_for: str | None = None):
        self.name, self.fail_parse_for, self.blank_for = name, fail_parse_for, blank_for

    def request(self, key: str) -> Request | None:
        if key == self.blank_for:
            return None
        return Request(f"https://{self.name}.test/search", {"q": key}, {})

    def parse(self, text: str) -> list[Candidate]:
        data = json.loads(text)
        if data["q"] == self.fail_parse_for:
            raise SourceError("bad shape")
        return [Candidate(self.name, 1, f"http://{self.name}.test/{data['q']}", data["q"], "d", {})]


class FakeClient:
    """Answers with the query echoed back; can fail a key a set number of times."""

    def __init__(self, failures: dict[str, int] | None = None) -> None:
        self.failures = dict(failures or {})
        self.calls: list[str] = []

    def get(self, url: str, params: dict, headers: dict) -> Fetched:
        key = params["q"]
        self.calls.append(f"{url.split('//')[1].split('.')[0]}:{key}")
        if self.failures.get(key, 0) > 0:
            self.failures[key] -= 1
            return Fetched(url, 503, None, 0.1, 3, "HTTP 503", ("HTTP 429", "HTTP 503"))
        return Fetched(url, 200, json.dumps({"q": key}), 0.4, 1, None, (), 0.05)


def run(store: JsonlStore, client: FakeClient, keys: list[str], sources=None, **kwargs):
    sources = sources or {"a": FakeSource("a"), "b": FakeSource("b")}
    return retrieve(keys, sources, client, store, now=lambda: NOW, **kwargs)


def test_every_request_is_recorded_with_its_candidates(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "c.jsonl")
    stats = run(store, FakeClient(), ["mass", "time"])
    assert stats["requested"] == 4 and stats["ok:a"] == 2 and stats["ok:b"] == 2
    records = store.read()
    assert len(records) == 4 and all(is_ok(r) for r in records)
    first = records[0]
    assert first["candidates"][0]["label"] == "mass"
    assert json.loads(first["response_text"]) == {"q": "mass"}
    assert first["retrieved_at"] == NOW and len(first["response_sha256"]) == 64
    assert first["request"] == {"url": "https://a.test/search", "params": {"q": "mass"}}
    assert (first["attempts"], first["latency_s"], first["request_s"]) == (1, 0.4, 0.05)
    assert first["attempt_errors"] == [] and first["error"] is None and first["parse_error"] is None
    assert all(response_hash_matches(r) for r in records)


def test_the_stored_hash_exposes_a_changed_response(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "c.jsonl")
    run(store, FakeClient(), ["mass"])
    (record,) = store.read()[:1]
    assert response_hash_matches(record)
    assert not response_hash_matches({**record, "response_text": '{"q": "other"}'})
    assert not response_hash_matches({**record, "response_text": None})
    assert response_hash_matches({"response_text": None, "response_sha256": None})


def test_a_rerun_makes_no_request_for_what_succeeded(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "c.jsonl")
    run(store, FakeClient(), ["mass", "time"])
    client = FakeClient()
    stats = run(store, client, ["mass", "time"])
    assert client.calls == [] and stats["skipped"] == 4 and stats["requested"] == 0


def test_a_failure_is_recorded_counted_and_retried_alone_on_resume(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "c.jsonl")
    stats = run(store, FakeClient({"time": 1}), ["mass", "time"])
    assert stats["failed:a"] == 1 and stats["failed:b"] == 0 and stats["ok:a"] == 1
    bad = [r for r in store.read() if not is_ok(r)]
    assert len(bad) == 1 and bad[0]["error"] == "HTTP 503" and bad[0]["candidates"] is None
    assert bad[0]["attempt_errors"] == ["HTTP 429", "HTTP 503"]
    client = FakeClient()
    stats = run(store, client, ["mass", "time"])
    assert client.calls == ["a:time"] and stats["ok:a"] == 1 and stats["skipped"] == 3


def test_a_response_with_the_wrong_shape_is_a_failure_and_keeps_its_text(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "c.jsonl")
    sources = {"a": FakeSource("a", fail_parse_for="mass")}
    stats = run(store, FakeClient(), ["mass"], sources=sources)
    (record,) = store.read()
    assert stats["failed:a"] == 1 and record["parse_error"] == "bad shape"
    assert json.loads(record["response_text"]) == {"q": "mass"} and not is_ok(record)


def test_a_key_with_no_searchable_word_is_an_empty_success_without_a_request(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "c.jsonl")
    client = FakeClient()
    stats = run(store, client, ["of"], sources={"a": FakeSource("a", blank_for="of")})
    (record,) = store.read()
    assert client.calls == [] and is_ok(record) and record["candidates"] == []
    assert record["skipped_reason"] and stats["empty:a"] == 1


def test_the_deadline_stops_the_loop_before_the_next_request(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "c.jsonl")
    client = FakeClient()
    ticks = iter([0.0, 99.0, 99.0])
    with pytest.raises(RetrievalTimeout):
        run(store, client, ["mass", "time"], deadline=10.0, clock=lambda: next(ticks))
    assert len(client.calls) == 1 and len(store.read()) == 1


def test_a_bug_in_a_parser_is_not_recorded_as_a_failed_retrieval(tmp_path: Path) -> None:
    class Buggy(FakeSource):
        def parse(self, text: str) -> list[Candidate]:
            return [{}["missing"]]

    store = JsonlStore(tmp_path / "c.jsonl")
    with pytest.raises(KeyError):
        run(store, FakeClient(), ["mass"], sources={"a": Buggy("a")})
    assert store.read() == []


def test_a_missing_record_is_not_ok() -> None:
    assert not is_ok(None) and not is_ok({})


def test_a_corrupt_store_is_reported_with_its_line(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text('{"source": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="c.jsonl:2"):
        JsonlStore(path).read()
