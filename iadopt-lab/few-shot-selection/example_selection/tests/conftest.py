"""Fixtures for the example-selection tests. No provider access; no database."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def records() -> list[dict]:
    """The real 102-row canonical corpus; these tests assert on its actual structure."""
    from iadopt_lab.corpus.ingestion import load_canonical_records
    return list(load_canonical_records(ROOT))


@pytest.fixture(scope="session")
def split(records):
    from design import corpus_split
    return corpus_split(records)


@pytest.fixture(scope="session")
def pool(split):
    return split["P"]


@pytest.fixture(scope="session")
def corpus_ids(records) -> list[str]:
    return sorted(row["variable_id"] for row in records)


def _stub_context(tmp_path, records_by_id, mode: str):
    """A HarnessContext whose transport is a stub, so no provider is ever contacted.

    `mode` selects what the fake provider does: "valid" returns a well-formed decomposition,
    "invalid" returns unparseable text, "fail" simulates a transport error.
    """
    import sys
    from pathlib import Path as _Path

    from harness import HarnessContext
    root = _Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "src"))
    from iadopt_lab.cli import _configuration, _similarity_backend
    from iadopt_lab.prompting.renderer import load_prompt_version
    from iadopt_lab.validation import load_schema_bytes

    schema = load_schema_bytes(root)
    context = HarnessContext(
        records=records_by_id,
        template=load_prompt_version("matrix-decomposition", root),
        schema=schema,
        schema_text=schema.decode("utf-8"),
        similarity=_similarity_backend(root, _configuration(root).data["evaluation"]),
        url="https://stub.invalid/chat/completions",
        headers={"Authorization": "Bearer SECRET-TEST-KEY"},
        cache={},
        calls_path=tmp_path / "calls.jsonl",
    )
    context.stub_calls = 0
    context.stub_bodies = []

    async def transport(url, headers, body):
        context.stub_calls += 1
        context.stub_bodies.append(body)
        if mode == "fail":
            return {"ok": False, "error": "ConnectError", "latency": 0.0}
        answer = ('{"hasObjectOfInterest": "water", "hasProperty": "temperature", '
                  '"hasMatrix": "", "hasContextObject": "", "hasStatisticalModifier": "", '
                  '"hasConstraint": []}') if mode == "valid" else "not json at all"
        return {"ok": True, "latency": 0.01, "answer": answer, "finish_reason": "stop",
                "prompt_tokens": 100, "completion_tokens": 20}

    context.transport = transport
    return context


@pytest.fixture
def stub_context(tmp_path, records):
    return _stub_context(tmp_path, {r["variable_id"]: r for r in records}, "valid")


@pytest.fixture
def stub_context_invalid(tmp_path, records):
    return _stub_context(tmp_path, {r["variable_id"]: r for r in records}, "invalid")


@pytest.fixture
def stub_context_failing(tmp_path, records):
    return _stub_context(tmp_path, {r["variable_id"]: r for r in records}, "fail")
