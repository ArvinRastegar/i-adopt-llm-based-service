"""Offline tests for settings.py: what a walk may be shown, mixed walks and arm names. No provider."""

from __future__ import annotations

import json

import engine
import hints
import pytest
import settings
import walk
from engine import ft

GOOD = {"hasStatisticalModifier": "", "hasProperty": "count", "hasObjectOfInterest": "x",
        "hasMatrix": "", "hasContextObject": "", "hasConstraint": []}


@pytest.fixture(scope="module")
def ctx():
    """Load the inputs once per module.

    Args: none.
    Returns: the mapping from `walk.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files.
    """
    return walk.load()


def test_a_walk_is_never_shown_its_target_or_a_held_out_variable(ctx):
    """A practice target sees 77 tagged variables, a held-out target 78, and never a held-out one."""
    held_out = set(ctx["inputs"]["targets"])
    for target in ctx["practice"][:5]:
        shown = settings.shown_ids(ctx, target)
        assert len(shown) == 77 and target not in shown and not set(shown) & held_out
    for target in sorted(held_out)[:5]:
        shown = settings.shown_ids(ctx, target)
        assert len(shown) == 78 and not set(shown) & held_out


def test_direct_all_prompt_differs_only_in_its_examples(ctx):
    """The prompt keeps the pattern lists, shows each permitted variable once and ends on the target."""
    target = ctx["practice"][0]
    definition = ctx["records"][target]["definition"]
    prompt = settings.direct_prompt_all(ctx, target)
    base = walk.direct_prompt(ctx, definition)
    assert prompt.split("\nEXAMPLES\n")[0] == base.split("\nEXAMPLES\n")[0]
    assert prompt.count("\nPatterns: {") == 77 and prompt.endswith(f"Definition: {definition}\nPatterns:")
    assert prompt.count(f"Definition: {definition}\n") == 1
    for held in ctx["inputs"]["targets"]:
        assert f"Definition: {ctx['records'][held]['definition']}\n" not in prompt
    own = json.dumps(walk.direct_answer(ctx, ctx["key"][target]), ensure_ascii=False)
    assert f"Definition: {definition}\nPatterns: {own}" not in prompt


def test_mixed_walk_takes_matrix_and_context_from_the_steps():
    """Matrix and context object come from the step-by-step walk; a missing walk leaves the other alone."""
    first = {"a": {"facts": {"property": [104], "matrix": [], "context": [], "constraint": [92]}},
             "b": {"facts": None}, "c": {"facts": {"property": [109], "matrix": [41], "context": [50]}}}
    steps = {"a": {"facts": {"property": [132], "matrix": [44], "context": [50], "constraint": []}},
             "b": {"facts": {"property": [101], "matrix": [], "context": []}}, "c": {"facts": None}}
    out = settings.mixed(first, steps)
    assert out["a"]["facts"] == {"property": [104], "matrix": [44], "context": [50], "constraint": [92]}
    assert out["b"]["facts"]["property"] == [101] and out["c"]["facts"]["matrix"] == [41]


def test_arm_names():
    """An arm name gives the mode, the walk and the guidance features."""
    assert settings.parse_arm("base") is None and settings.parse_arm("base+retry") is None
    assert settings.parse_arm("guided+check@direct-all/full") == (
        "guided+check", "direct-all", frozenset({"form", "names", "types", "texts"}))
    assert settings.parse_arm("guided@mixed/form+types")[2] == frozenset({"form", "types"})
    with pytest.raises(ValueError):
        settings.parse_arm("magic@direct/full")
    with pytest.raises(KeyError):
        settings.parse_arm("guided@direct/everything")


@pytest.mark.asyncio
async def test_run_uses_each_arm_walk_and_keeps_arm_names(ctx, tmp_path, monkeypatch):
    """Base arms get the plain prompt, guided arms the guidance of their own walk, under their own name."""
    ids = ctx["practice"][:2]
    facts = {"property": [104], "object": [], "matrix": [], "context": [], "constraint": [],
             "statistical_modifier": False, "names": {}}
    monkeypatch.setattr(settings, "walks_named", lambda c, name, where: {v: {"facts": facts} for v in ids})
    monkeypatch.setattr(settings, "REPETITIONS", 1)
    seen = []

    async def transport(body):
        """Reply with a conforming answer and note whether the prompt was guided.

        Args: body: the request body.
        Returns: a successful reply.
        Raises: nothing.
        Side Effects: appends to `seen`.
        """
        seen.append(hints.HEADING in body["messages"][0]["content"])
        return {"ok": True, "status": 200, "latency": 0.1, "answer": json.dumps(GOOD), "reasoning": "",
                "finish_reason": "stop", "prompt_tokens": 5000, "completion_tokens": 50, "served_model": "x"}

    arms = ["base", "guided+check@direct-all/full"]
    path = tmp_path / "calls.jsonl"
    result = await settings.run(ctx, arms, ids, "practice", transport, path, concurrency=2)
    assert result["issued"] == 4 and sorted(seen) == [False, False, True, True]
    stored = engine.latest(path)
    assert {key[0] for key in stored} == set(arms)
    assert all(r["guided"] == (key[0] != "base") for key, r in stored.items())
    assert len(ft.read_jsonl(path)) == 4


def test_settings_runs_this_folders_pipeline():
    """The runner uses guided/pipeline.py, not the example-selection module of the same name."""
    assert settings.pipeline.__file__ == str(settings.HERE / "pipeline.py")
    assert callable(settings.pipeline.attempt) and callable(settings.pipeline.paths)
