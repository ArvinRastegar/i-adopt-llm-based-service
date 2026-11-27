"""Offline tests for stronger_reader_test.py: stop rules, model routing, percentiles and the Step 2 prompts."""

from __future__ import annotations

import asyncio

import answer_form_check as afc
import constraint_types_test as cy
import knowledge as kn
import pytest
import rag_tuning as rt
import stronger_reader_test as sr


def test_guard_allows_reasoning_only_when_thinking_and_caps_spend():
    """Reasoning text stops a non-thinking reader, not a thinking one; the list-price cap stops both."""
    ok = {"ok": True, "status": 200, "prompt_tokens": 10, "completion_tokens": 10, "reasoning_chars": 500}
    off, on = sr.Guard("GLM-5.3", False), sr.Guard("GLM-5.3", True)
    for _ in range(5):
        off.record(ok)
        on.record(ok)
    assert off.reason and "reasoning" in off.reason and on.reason is None
    on.record(dict(ok, prompt_tokens=4_000_000, reasoning_chars=0))
    assert on.reason and "list price" in on.reason and on.cost > sr.CAP_USD


def test_model_transport_routes_every_request():
    """The wrapped transport replaces the model and keeps the rest of the body."""
    seen = []

    async def transport(body):
        seen.append(body)
        return {"ok": True}

    asyncio.run(sr.model_transport(transport, "GLM-5.3")({"model": "Qwen3.8-27B", "messages": [], "max_tokens": 5}))
    assert seen == [{"model": "GLM-5.3", "messages": [], "max_tokens": 5}]


def test_percentile_by_nearest_rank():
    """The value at rank ceil(share * n)."""
    assert sr.percentile(list(range(1, 11)), 0.95) == 10
    assert sr.percentile(list(range(1, 103)), 0.95) == 97
    assert sr.percentile([3.0], 0.5) == 3.0
    with pytest.raises(ValueError):
        sr.percentile([], 0.5)


def test_requests():
    """The two non-thinking readers share the finder's settings; the thinking one leaves sampling to the server."""
    assert sr.REQUESTS["qwen3.8"] == sr.REQUESTS["glm-5.3"]
    assert sr.REQUESTS["qwen3.8"]["temperature"] == 0.0 and sr.REQUESTS["qwen3.8"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "temperature" not in sr.REQUESTS["glm-5.3-think"] and sr.REQUESTS["glm-5.3-think"]["chat_template_kwargs"] == {"enable_thinking": True}
    for r in ("qwen3.5", "deepseek-v4.1-flash", "glm-5.3-flash"):
        assert sr.REQUESTS[r]["temperature"] == 0.0 and sr.REQUESTS[r]["max_tokens"] == sr.REQUESTS["qwen3.8"]["max_tokens"]
    assert set(sr.MODELS) == set(sr.READERS) == set(sr.REQUESTS) and sr.THINKING <= set(sr.READERS)
    guard = sr.Guard("Unknown-Model", False)
    guard.record({"ok": True, "status": 200, "prompt_tokens": 1_000_000, "completion_tokens": 0})
    assert guard.cost == pytest.approx(sr.PRICES["GLM-5.3"][0])


@pytest.fixture(scope="module")
def setup():
    """The context and stage 5's world with a word-overlap table.

    Args: none.
    Returns: (ctx, world, found).
    Raises: as the loaders.
    Side Effects: reads local files.
    """
    def overlap(first: str, second: str) -> float:
        """Shared words over all words.

        Args: first, second: texts.
        Returns: the share.
        Raises: nothing.
        Side Effects: none.
        """
        a, b = set(kn.words(first)), set(kn.words(second))
        return len(a & b) / len(a | b) if a | b else 0.0

    ctx = afc.load()
    base = kn.load()
    world = rt.corpus_world(rt.load(table=rt.similarity_table(base["ids"], lambda v: base["records"][v]["definition"], overlap)), None)
    found = {v: kn.walk.facts_from_direct(ctx, ctx["demos"][v]["patterns"]) for v in ctx["ids"]}
    return ctx, world, found


def test_step2_prompts(setup):
    """`best` is the constraint-types test's; the reader's matrix alone gives the answer-form prompt with its facts."""
    ctx, world, found = setup
    settings = {"flagged": [], "confident": {v: [] for v in ctx["ids"]}}
    v = next(u for u in ctx["ids"] if ctx["key"][u]["constraint"])
    control = cy.prompt2(ctx, world, ("best", 1, v), found, settings)
    assert sr.prompt2(ctx, world, ("best", 1, v), found, found, ["matrix"]) == control
    assert sr.prompt2(ctx, world, ("best+reader", 1, v), found, {v: None}, ["matrix"]) == control
    assert sr.prompt2(ctx, world, ("best+reader", 1, v), found, found, ["matrix"]) == \
        afc.prompt_b(ctx, world, ("rag-25+finder-matrix", 1, v), found)
    both = sr.prompt2(ctx, world, ("best+reader", 1, v), found, found, ["constraint", "matrix"])
    assert both == cy.prompt2(ctx, world, ("best+experts", 1, v), found, settings)
    with pytest.raises(KeyError):
        sr.prompt2(ctx, world, ("best+experts", 1, v), found, found, ["matrix"])
