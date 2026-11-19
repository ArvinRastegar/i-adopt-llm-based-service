"""Offline tests for hints.py, engine.py and ceiling.py. A stub replaces PSNC."""

from __future__ import annotations

import json

import ceiling
import engine
import hints
import pytest
from engine import ft

FACTS = {"property": [110], "object": [35], "matrix": [], "context": [], "constraint": [76, 92],
         "statistical_modifier": False, "names": {}}


@pytest.fixture(scope="module")
def ctx():
    """Load the ceiling test's inputs once per module.

    Args: none.
    Returns: the mapping from `ceiling.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files.
    """
    return ceiling.load()


def test_practice_set_is_53_and_touches_no_example_or_held_out_variable(ctx):
    """The practice set is the 38 search variables plus the 15 unused pool variables."""
    practice = set(ctx["practice"])
    assert len(practice) == 53 and not practice & set(ft.EXAMPLE_IDS)
    assert not practice & set(ctx["inputs"]["targets"])
    assert practice <= set(ctx["key"])


def test_no_guidance_for_the_baseline(ctx):
    """The baseline arm is today's prompt, byte for byte."""
    variable = ctx["practice"][0]
    definition = ctx["records"][variable]["definition"]
    assert hints.render(FACTS, frozenset(), ctx["patterns"], ctx["types"]) == ""
    assert ceiling.prompt_of(ctx, ("base", 1, variable)) == ft.prompt_for(ctx["inputs"], "B", definition)


def test_guidance_only_adds_a_block_after_the_definition(ctx):
    """Every guided arm equals the baseline prompt plus one block right after the definition."""
    variable = ctx["practice"][0]
    definition = ctx["records"][variable]["definition"]
    base = ceiling.prompt_of(ctx, ("base", 1, variable))
    for arm, features in ceiling.ARMS.items():
        if arm == "base":
            continue
        block = hints.render(hints.facts_from_key(ctx["key"][variable]), features, ctx["patterns"], ctx["types"])
        prompt = ceiling.prompt_of(ctx, (arm, 1, variable))
        assert block.startswith("\n\n" + hints.HEADING) and definition + block in prompt
        assert prompt.replace(block, "", 1) == base


def test_form_lines_state_shapes_and_absences(ctx):
    """A ratio object with no matrix, context or modifier is spelled out part by part."""
    block = hints.render(FACTS, {"form"}, ctx["patterns"], ctx["types"])
    assert "- hasObjectOfInterest: an asymmetric system with hasNumerator and hasDenominator." in block
    assert '- hasMatrix: none; use "".' in block and '- hasContextObject: none; use "".' in block
    assert '- hasStatisticalModifier: none; use "".' in block
    assert "pattern" not in block.split(hints.HEADING)[1] and "hasConstraint" not in block


def test_types_names_and_texts_are_separate_features(ctx):
    """Constraint types, pattern names and rulebook texts appear only when asked for."""
    types = hints.render(FACTS, {"form", "types"}, ctx["patterns"], ctx["types"])
    assert "of no other type: normalization; state." in types and "Property pattern" not in types
    none = hints.render(dict(FACTS, constraint=[]), {"types"}, ctx["patterns"], ctx["types"])
    assert "- hasConstraint: none; use []." in none
    names = hints.render(FACTS, {"names"}, ctx["patterns"], ctx["types"])
    assert "- Property pattern 110: fraction." in names and "- Object of interest pattern 35: ratio." in names
    full = hints.render(FACTS, set(hints.FEATURES), ctx["patterns"], ctx["types"])
    assert ctx["patterns"]["110"]["definition"] in full
    assert "- state: " + next(t["definition"] for t in ctx["types"] if t["name"] == "state") in full
    with pytest.raises(ValueError):
        hints.render(FACTS, {"colour"}, ctx["patterns"], ctx["types"])


def test_conformance_checks_each_part(ctx):
    """An answer conforms only when every part has the form the facts imply."""
    form = hints.expected_form(FACTS, ctx["patterns"], ctx["types"])
    assert form == {"object": "ratio", "matrix": "empty", "context": False,
                    "statistical_modifier": False, "constraint_types": ["normalization", "state"]}
    good = {"hasStatisticalModifier": "", "hasProperty": "mass fraction",
            "hasObjectOfInterest": {"AsymmetricSystem": "", "hasNumerator": "a", "hasDenominator": "b"},
            "hasMatrix": "", "hasContextObject": "",
            "hasConstraint": [{"label": "state: dry", "on": "a"}, {"label": "normalization: per m2", "on": "b"}]}
    assert hints.conformance(good, form)["all"] is True
    bad = dict(good, hasMatrix="soil", hasConstraint=[])
    result = hints.conformance(bad, form)
    assert (result["matrix"], result["constraint_types"], result["object"], result["all"]) == (False, False, True, False)
    assert hints.conformance(None, form)["all"] is False


def test_invented_items_are_not_corpus_variables(ctx):
    """The wording is developed on made-up definitions only."""
    definitions = {r["definition"].strip().lower() for r in ctx["records"].values()}
    labels = {r["label"].strip().lower() for r in ctx["records"].values()}
    for item in ceiling.INVENTED:
        assert item["definition"].lower() not in definitions | labels
        assert all(str(n) in ctx["patterns"] for n in item["property"] + item["object"] + item["matrix"])


def test_schedule_covers_the_grid_once_with_arms_side_by_side(ctx):
    """Each arm, repetition and variable appears once; one variable's arms are adjacent."""
    arms = list(ceiling.ARMS)
    jobs = engine.schedule(arms, ctx["practice"], ceiling.REPETITIONS, ceiling.SCHEDULE_SEED)
    assert len(jobs) == len(set(jobs)) == 6 * 53 * 15
    assert jobs == engine.schedule(arms, ctx["practice"], ceiling.REPETITIONS, ceiling.SCHEDULE_SEED)
    for start in range(0, len(jobs), len(arms)):
        block = jobs[start:start + len(arms)]
        assert {j[0] for j in block} == set(arms) and len({j[1:] for j in block}) == 1


def _stub(answer="{}", *, status=200, reasoning=""):
    """Build a stub transport that records every request and returns one fixed reply.

    Args: answer: reply text; status: HTTP status (non-200 means a failed call); reasoning:
        reasoning text to return.
    Returns: (transport, calls), where `calls` collects every request body sent.
    Raises: nothing.
    Side Effects: none until the transport is awaited.
    """
    calls = []

    async def transport(body):
        """Record the request and return the fixed reply.

        Args: body: the request body.
        Returns: a transport result shaped like `format_test.psnc_transport`'s.
        Raises: nothing.
        Side Effects: appends `body` to `calls`.
        """
        calls.append(body)
        if status != 200:
            return {"ok": False, "status": status, "error": f"HTTP {status}", "latency": 0.1}
        text = answer(body) if callable(answer) else answer
        return {"ok": True, "status": 200, "latency": 0.1, "answer": text, "reasoning": reasoning,
                "finish_reason": "stop", "prompt_tokens": 5000, "completion_tokens": 100,
                "served_model": body["model"]}

    return transport, calls


@pytest.mark.asyncio
async def test_run_uses_the_study_settings_and_resumes_without_repeats(ctx, tmp_path):
    """Calls go to Qwen3.8-27B with reasoning off, and a second run sends nothing new."""
    jobs = engine.schedule(["base", "form"], ctx["practice"][:2], 2, 1)
    transport, calls = _stub()
    path = tmp_path / "calls.jsonl"
    result = await engine.run_jobs(jobs, lambda job: ceiling.prompt_of(ctx, job), transport, path, concurrency=3)
    assert result["issued"] == 8 and result["stopped"] is None
    body = calls[0]
    assert body["model"] == "Qwen3.8-27B" and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert (body["temperature"], body["top_p"], body["max_tokens"]) == (0.5, 1.0, 16000)
    assert {engine.key_of(r) for r in ft.read_jsonl(path)} == set(jobs)
    again = await engine.run_jobs(jobs, lambda job: ceiling.prompt_of(ctx, job), transport, path, concurrency=3)
    assert again["issued"] == 0 and len(calls) == 8


@pytest.mark.asyncio
async def test_stop_rules(ctx, tmp_path):
    """A refused credential, early failures and reasoning text each stop the run."""
    jobs = engine.schedule(["base"], ctx["practice"], 1, 1)
    for name, kwargs, expected in (("auth", {"status": 401}, 1), ("fail", {"status": 503}, engine.EARLY_FAILURES + 1),
                                   ("think", {"reasoning": "hmm"}, engine.THINKING_LIMIT + 1)):
        transport, calls = _stub(**kwargs)
        result = await engine.run_jobs(jobs, lambda job: "p", transport, tmp_path / f"{name}.jsonl", concurrency=1)
        assert len(calls) == expected and result["stopped"], name


@pytest.mark.asyncio
async def test_failed_calls_are_retried(ctx, tmp_path):
    """A failed call is not treated as answered: the next run retries it."""
    jobs = engine.schedule(["base"], ctx["practice"][:1], 1, 1)
    path = tmp_path / "calls.jsonl"
    failing, _ = _stub(status=503)
    await engine.run_jobs(jobs, lambda job: "p", failing, path, concurrency=1)
    working, calls = _stub()
    await engine.run_jobs(jobs, lambda job: "p", working, path, concurrency=1)
    assert len(calls) == 1 and all(r["ok"] for r in engine.latest(path).values())


@pytest.mark.asyncio
async def test_scoring_and_comparison_end_to_end(ctx, tmp_path):
    """Gold answers score 1, unusable answers score 0 and stay in the population."""
    variables = ctx["practice"][:3]
    gold = {v: json.dumps(ctx["records"][v]["gold"]) for v in variables}
    by_prompt = {ceiling.prompt_of(ctx, ("form", 1, v)): gold[v] for v in variables}
    transport, _ = _stub(lambda body: by_prompt.get(body["messages"][0]["content"], "not json"))
    jobs = engine.schedule(["base", "form"], variables, 2, 1)
    calls_path, scores_path = tmp_path / "calls.jsonl", tmp_path / "scores.jsonl"
    await engine.run_jobs(jobs, lambda job: ceiling.prompt_of(ctx, job), transport, calls_path, concurrency=2)
    scores = engine.score(calls_path, scores_path, ctx["records"], ctx["inputs"]["schema"])
    assert all(row["valid_prediction"] == (key[0] == "form") for key, row in scores.items())
    summary = engine.summarise(["base", "form"], "base", variables, 2, engine.latest(calls_path), scores)
    assert summary["cells"]["form"]["close_f1_official_mean"] == 1.0
    assert summary["cells"]["base"]["close_f1_official_mean"] == 0.0
    assert summary["cells"]["base"]["invalid"] == 6 and summary["variables"] == 3
    assert summary["comparisons"]["form"]["difference"] == 1.0
    rates = ceiling.conformance_rates(ctx, scores)
    assert rates["base"]["all"] == 0.0 and rates["form"]["object"] == 1.0
    assert engine.score(calls_path, scores_path, ctx["records"], ctx["inputs"]["schema"]).keys() == scores.keys()
    assert len(ft.read_jsonl(scores_path)) == 12
