"""Offline tests for groups.py: leave-one-out for the example group, cleaned texts, arms, pooling. No provider."""

from __future__ import annotations

import json
import re

import ceiling
import engine
import groups
import hints
import pytest
import recompute
import settings
import walk
from engine import ft

FULL = frozenset({"form", "names", "types", "texts"})
GOOD = {"hasStatisticalModifier": "", "hasProperty": "count", "hasObjectOfInterest": "x",
        "hasMatrix": "", "hasContextObject": "", "hasConstraint": []}


@pytest.fixture(scope="module")
def ctx():
    """Load this step's context once per module.

    Args: none.
    Returns: the mapping from `groups.load`, with cleaned pattern texts.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files.
    """
    return groups.load()


@pytest.fixture(scope="module")
def raw():
    """Load the context of steps 1 to 4, with the pattern texts as built.

    Args: none.
    Returns: the mapping from `walk.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files.
    """
    return walk.load()


def test_the_three_groups_are_the_whole_corpus(ctx):
    """25 examples, 53 practice and 24 held-out variables, disjoint, 102 in all."""
    sets = {group: set(groups.targets(ctx, group)) for group in ("examples", "practice", "held-out")}
    assert [len(sets[g]) for g in ("examples", "practice", "held-out")] == [25, 53, 24]
    assert len(set.union(*sets.values())) == 102 == len(ctx["records"])


def test_only_the_two_example_terms_are_removed(ctx, raw):
    """The cleaned table differs from the built one in exactly the two design texts."""
    changed = [(n, field) for n, pattern in raw["patterns"].items() for field, text in pattern.items()
               if ctx["patterns"][n][field] != text]
    assert sorted(changed) == [("129", "design"), ("132", "design")]
    assert ctx["patterns"]["132"]["design"] == "Index must be specified. OoI is a single entity."
    assert ctx["patterns"]["129"]["design"].endswith("Often the property is domain-specific labelled.")
    text = json.dumps(ctx["patterns"]).lower()
    assert "body mass" not in text and "actinic" not in text
    with pytest.raises(ValueError):
        groups.clean_patterns(ctx["patterns"])      # the terms are gone, so a second pass must refuse


def test_no_guidance_of_a_new_target_names_a_removed_term(ctx):
    """With the experts' tags and every feature on, no guidance block holds either term."""
    for group in groups.GROUPS:
        for target in groups.targets(ctx, group):
            block = hints.render(hints.facts_from_key(ctx["key"][target]), FULL, ctx["patterns"], ctx["types"]).lower()
            assert "body mass index" not in block and "actinic flux" not in block


def test_an_example_is_never_shown_its_own_answer(ctx):
    """Each example target's fill prompt holds the other 24 examples and not its own block."""
    answers = [ft.answer_text(row["gold"]) for row in ctx["inputs"]["examples"]]
    assert len(set(answers)) == 25                  # so "its own answer" is unambiguous
    for target in groups.targets(ctx, "examples"):
        record = ctx["records"][target]
        own = groups.context_for(ctx, target)
        assert [row["variable_id"] for row in own["inputs"]["examples"]] == groups.shown_examples(target)
        assert len(own["inputs"]["examples"]) == 24 and len(ctx["inputs"]["examples"]) == 25
        for arm, walks in (("base", {}), (groups.KEY_ARM, groups.walks_named(ctx, "key", "examples"))):
            prompt = groups.prompt_of(ctx, arm, target, walks)
            block = ft.examples_block(prompt)
            assert len(re.findall(r"(?m)^Example \d+$", block)) == 24
            assert ft.answer_text(record["gold"]) not in block
            assert f"Definition: {record['definition']}\nOutput:" not in block


def test_other_targets_get_the_usual_prompt(ctx):
    """A held-out or practice target keeps all 25 examples and today's prompt byte for byte."""
    for group in ("held-out", "practice"):
        for target in groups.targets(ctx, group):
            assert groups.context_for(ctx, target) is ctx
            definition = ctx["records"][target]["definition"]
            assert groups.prompt_of(ctx, "base", target, {}) == ft.prompt_for(ctx["inputs"], "B", definition)


def test_the_key_arm_is_the_ceiling_full_arm(raw, ctx):
    """With the texts as built, `guided@key/full` sends step 1's `full` prompt; cleaning changes four."""
    keyed = groups.walks_named(raw, "key", "practice")
    differing = []
    for target in raw["practice"]:
        definition = raw["records"][target]["definition"]
        facts = hints.facts_from_key(raw["key"][target])
        assert groups.prompt_of(raw, groups.KEY_ARM, target, keyed) == ceiling.guided_prompt(raw, definition, facts, "full")
        if groups.prompt_of(ctx, groups.KEY_ARM, target, keyed) != groups.prompt_of(raw, groups.KEY_ARM, target, keyed):
            differing.append(tuple(raw["key"][target]["property"]))
    assert sorted(differing) == [(129,), (129,), (132,), (132,)]


def test_step_demonstrations_leave_the_target_out(ctx):
    """A held-out target gets the walk's own demonstrations; an example target never sees itself."""
    full = walk.demo_index(ctx)
    assert groups.demos_for(ctx, groups.targets(ctx, "held-out")[0]) == full
    definitions = [ctx["records"][v]["definition"] for v in ft.EXAMPLE_IDS]
    assert len(set(definitions)) == 25
    for target in groups.targets(ctx, "examples"):
        definition = ctx["records"][target]["definition"]
        own = groups.demos_for(ctx, target)
        for qid, pairs in full.items():
            assert own.get(qid, []) == [pair for pair in pairs if pair[0] != definition]
        assert not set(own) - set(full)


def test_direct_prompt_leaves_the_target_out(ctx):
    """The one-call walk prompt is the walk's own for other targets and drops an example target's tags."""
    held = groups.targets(ctx, "held-out")[0]
    assert groups.direct_prompt(ctx, held) == walk.direct_prompt(ctx, ctx["records"][held]["definition"])
    for target in groups.targets(ctx, "examples"):
        definition = ctx["records"][target]["definition"]
        prompt = groups.direct_prompt(ctx, target)
        assert prompt.count("\nPatterns: {") == 24 and prompt.endswith(f"Definition: {definition}\nPatterns:")
        assert prompt.count(f"Definition: {definition}\n") == 1
        assert prompt.split("\nEXAMPLES\n")[0] == walk.direct_prompt(ctx, definition).split("\nEXAMPLES\n")[0]


def test_the_all_variables_walk_leaves_an_example_target_out(ctx):
    """Shown every permitted tagged variable, an example target sees 77 others and never a held-out one."""
    held_out = set(ctx["inputs"]["targets"])
    for target in groups.targets(ctx, "examples"):
        shown = settings.shown_ids(ctx, target)
        assert len(shown) == 77 and target not in shown and not set(shown) & held_out
        definition = ctx["records"][target]["definition"]
        assert settings.direct_prompt_all(ctx, target).count(f"Definition: {definition}\n") == 1


def test_the_two_scripts_name_the_same_sources():
    """The recomputation states the table's sources on its own; the two statements agree."""
    assert recompute.TABLE_ARMS == groups.ARMS
    assert {arm: tuple(sources) for arm, sources in recompute.TABLE_SOURCES["practice"].items()} == groups.PRACTICE_SOURCES
    assert recompute.TABLE_ROWS == dict(groups.ROWS)


def test_pooling_takes_every_answer_of_every_group(monkeypatch):
    """A pooled repetition scores all pairs of all groups together; an incomplete group is refused."""
    monkeypatch.setattr(groups, "_aggregate", lambda evaluations, ids: {"f1": float(sum(evaluations))})
    first = {1: [(1, "a"), (2, "b")], 2: [(3, "a"), (4, "b")]}
    second = {1: [(10, "c")], 2: [(20, "c")]}
    assert groups.pooled_f1([first, second]) == [13.0, 27.0]
    with pytest.raises(ValueError):
        groups.pooled_f1([first, {1: [(10, "c")]}])
    with pytest.raises(ValueError):
        groups.pooled_f1([{1: [(1, "a")], 2: [(3, "a"), (4, "b")]}])


@pytest.mark.asyncio
async def test_run_answers_each_example_in_its_own_context(ctx, tmp_path, monkeypatch):
    """Every arm of an example target is sent without that example; arm names and modes are kept."""
    ids = groups.targets(ctx, "examples")[:2]
    facts = {"property": [104], "object": [], "matrix": [], "context": [], "constraint": [],
             "statistical_modifier": False, "names": {}}
    monkeypatch.setattr(groups, "targets", lambda c, group: ids)
    found = {v: {"facts": facts} for v in ids}
    usual = groups.walks_named
    monkeypatch.setattr(groups, "walks_named", lambda c, name, group: usual(c, name, group) if name == "key" else found)
    seen: list[str] = []

    async def transport(body):
        """Reply with a conforming answer and keep the first prompt of each request.

        Args: body: the request body.
        Returns: a successful reply.
        Raises: nothing.
        Side Effects: appends to `seen`.
        """
        seen.append(body["messages"][0]["content"])
        return {"ok": True, "status": 200, "latency": 0.1, "answer": json.dumps(GOOD), "reasoning": "",
                "finish_reason": "stop", "prompt_tokens": 5000, "completion_tokens": 50, "served_model": "x"}

    arms = [groups.BASELINE, groups.KEY_ARM, "guided+check@direct-all/full"]
    path = tmp_path / "calls.jsonl"
    result = await groups.run(ctx, "examples", arms, transport, path, concurrency=2, repetitions=1)
    assert result["issued"] == 6 and result["stopped"] is None
    for prompt in seen:
        assert len(re.findall(r"(?m)^Example \d+$", ft.examples_block(prompt))) == 24
    for target in ids:
        record = ctx["records"][target]
        mine = [p for p in seen if record["definition"] in p.split("TARGET DEFINITION", 1)[1]]
        assert len(mine) >= 3 and all(ft.answer_text(record["gold"]) not in ft.examples_block(p) for p in mine)
    stored = engine.latest(path)
    assert {key[0] for key in stored} == set(arms)
    assert all(record["guided"] == (key[0] != groups.BASELINE) for key, record in stored.items())
    assert all(record["calls"] == 1 for key, record in stored.items() if key[0] == groups.KEY_ARM)


@pytest.mark.asyncio
async def test_walks_of_an_example_never_show_it(ctx, tmp_path):
    """Step by step and in one call, no walk prompt of an example target holds it as a demonstration."""
    target = groups.targets(ctx, "examples")[0]
    definition = ctx["records"][target]["definition"]
    seen: list[str] = []

    async def transport(body):
        """Answer a step with the first option and a direct request with a pattern set.

        Args: body: the request body.
        Returns: a successful reply in the shape of `walk.chat_transport`.
        Raises: nothing.
        Side Effects: appends to `seen`.
        """
        seen.append(body["messages"][0]["content"])
        one_token = body.get("max_tokens") == 1
        answer = "1" if one_token else json.dumps({"property": 104, "matrix": None, "context": False,
                                                    "modifier": False, "constraints": []})
        return {"ok": True, "status": 200, "latency": 0.1, "answer": answer, "reasoning": "",
                "top": [{"token": "1", "logprob": 0.0}], "prompt_tokens": 900, "completion_tokens": 1,
                "served_model": "x"}

    for style in ("steps", "direct"):
        state = await groups.walk_group(ctx, style, [target], transport, tmp_path / f"{style}.jsonl", concurrency=1)
        assert state == {"walked": 1, "skipped": 0, "stopped": None}
    assert len(seen) > 40
    for prompt in seen:
        assert prompt.count(f"Definition: {definition}\n") == 1
        assert prompt.endswith(f"Definition: {definition}\nAnswer:") or prompt.endswith(f"Definition: {definition}\nPatterns:")
    rows = walk.stored(tmp_path / "direct.jsonl")
    assert rows[target]["facts"]["property"] == [104] and rows[target]["shown"] == 24 and rows[target]["complete"]
    assert walk.stored(tmp_path / "steps.jsonl")[target]["facts"] is not None
