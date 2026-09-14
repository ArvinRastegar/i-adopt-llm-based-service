"""Acceptance tests for contracts/harness.md. A stub provider replaces PSNC throughout."""

from __future__ import annotations

import json

import pytest
from harness import CANDIDATE_SIZE, candidate_hash, evaluate, load_cache


def test_hash_is_order_independent():
    """AT-1: a set identity must not depend on the order it was written in."""
    a = [f"urn:v:{i}" for i in range(5)]
    assert candidate_hash(a) == candidate_hash(list(reversed(a)))


def test_hash_distinguishes_different_sets():
    """AT-2: a one-member change must change the identity."""
    a = [f"urn:v:{i}" for i in range(5)]
    assert candidate_hash(a) != candidate_hash(a[:-1] + ["urn:v:99"])


@pytest.mark.asyncio
async def test_rejects_candidate_overlapping_targets(stub_context, corpus_ids):
    """AT-3: an example appearing in the eval set is the leak this design exists to stop."""
    candidate = corpus_ids[:CANDIDATE_SIZE]
    targets = corpus_ids[CANDIDATE_SIZE - 1:CANDIDATE_SIZE + 5]
    with pytest.raises(ValueError):
        await evaluate(candidate, targets, 1, stub_context)


@pytest.mark.asyncio
async def test_rejects_wrong_candidate_size(stub_context, corpus_ids):
    """AT-4: the experiment is defined at exactly 25 examples."""
    with pytest.raises(ValueError):
        await evaluate(corpus_ids[:24], corpus_ids[40:50], 1, stub_context)


@pytest.mark.asyncio
async def test_official_denominator_is_fixed(stub_context_invalid, corpus_ids):
    """AT-6: under the official rule an unparseable answer stays in the population (INV-3)."""
    candidate, targets = corpus_ids[:CANDIDATE_SIZE], corpus_ids[40:50]
    result = await evaluate(candidate, targets, 1, stub_context_invalid)
    assert result["invalid"] > 0
    assert result["denominator_official"] == len(targets)
    assert result["scored"] < len(targets)


@pytest.mark.asyncio
async def test_resume_issues_no_second_call(stub_context, corpus_ids):
    """AT-7: a repeated evaluation is served from cache and pays for nothing."""
    candidate, targets = corpus_ids[:CANDIDATE_SIZE], corpus_ids[40:50]
    await evaluate(candidate, targets, 1, stub_context)
    before = stub_context.stub_calls
    await evaluate(candidate, targets, 1, stub_context)
    assert stub_context.stub_calls == before


@pytest.mark.asyncio
async def test_transport_failure_is_recorded_not_raised(stub_context_failing, corpus_ids):
    """AT-8: a failed call is counted, never silently dropped and never fatal (INV-6)."""
    candidate, targets = corpus_ids[:CANDIDATE_SIZE], corpus_ids[40:50]
    result = await evaluate(candidate, targets, 1, stub_context_failing)
    assert result["failed_calls"] == len(targets)


@pytest.mark.asyncio
async def test_reasoning_is_disabled_on_every_request(stub_context, corpus_ids):
    """AT-9: every request must carry enable_thinking=false (INV-7)."""
    candidate, targets = corpus_ids[:CANDIDATE_SIZE], corpus_ids[40:50]
    await evaluate(candidate, targets, 1, stub_context)
    assert stub_context.stub_bodies
    for body in stub_context.stub_bodies:
        assert body["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_writes_one_json_line_per_call(stub_context, corpus_ids, tmp_path):
    """AT-10: the calls JSONL carries exactly one parseable line per provider call."""
    candidate, targets = corpus_ids[:CANDIDATE_SIZE], corpus_ids[40:50]
    await evaluate(candidate, targets, 1, stub_context)
    lines = [json.loads(line) for line in
             stub_context.calls_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == len(targets)


@pytest.mark.asyncio
async def test_never_writes_the_api_key(stub_context, corpus_ids):
    """AT-11: the credential must not reach any artifact."""
    candidate, targets = corpus_ids[:CANDIDATE_SIZE], corpus_ids[40:50]
    await evaluate(candidate, targets, 1, stub_context)
    assert "SECRET-TEST-KEY" not in stub_context.calls_path.read_text(encoding="utf-8")


def test_load_cache_tolerates_missing_file(tmp_path):
    """A resume before the first call must start empty, not crash."""
    assert load_cache(tmp_path / "absent.jsonl") == {}
