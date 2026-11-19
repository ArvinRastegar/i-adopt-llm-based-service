"""Offline tests for pipeline.py: check, repair, fallback and the four arms. A stub replaces PSNC."""

from __future__ import annotations

import json

import engine
import hints
import pytest
import walk
from settings import pipeline  # this folder's pipeline.py, loaded by path; see settings.py for why

FEATURES = frozenset({"form", "names", "types"})
FACTS = {"property": [110], "object": [], "matrix": [], "context": [], "constraint": [76],
         "statistical_modifier": False, "names": {}}
GOOD = {"hasStatisticalModifier": "", "hasProperty": "mass fraction",
        "hasObjectOfInterest": {"AsymmetricSystem": "", "hasNumerator": "a", "hasDenominator": "b"},
        "hasMatrix": "", "hasContextObject": "", "hasConstraint": [{"label": "state: dry", "on": "a"}]}
WRONG_FORM = dict(GOOD, hasMatrix="soil", hasConstraint=[])
INVALID = dict(GOOD, hasConstraint=[{"label": "state: dry", "on": "nothing here"}])


@pytest.fixture(scope="module")
def ctx():
    """Load the end-to-end test's inputs once per module.

    Args: none.
    Returns: the mapping from `walk.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files.
    """
    return walk.load()


def stub(first, repair=None, plain=None):
    """Build a transport that replies by the kind of request it gets.

    Args: first: reply to a guided first prompt, or to any first prompt when `plain` is None;
        repair: reply to a three-message repair request; plain: reply to an unguided first prompt.
    Returns: (transport, calls), where `calls` collects every request body sent.
    Raises: nothing.
    Side Effects: none until the transport is awaited.
    """
    calls = []

    async def transport(body):
        """Record the request and reply by its kind.

        Args: body: the request body.
        Returns: a successful transport result.
        Raises: nothing.
        Side Effects: appends `body` to `calls`.
        """
        calls.append(body)
        messages = body["messages"]
        if len(messages) == 3:
            reply = repair
        elif hints.HEADING in messages[0]["content"] or plain is None:
            reply = first
        else:
            reply = plain
        return {"ok": True, "status": 200, "latency": 0.5, "answer": json.dumps(reply), "reasoning": "",
                "finish_reason": "stop", "prompt_tokens": 5000, "completion_tokens": 100,
                "served_model": body["model"]}

    return transport, calls


def test_stated_form_follows_the_features(ctx):
    """Only what the guidance states can be checked."""
    full = pipeline.stated_form(ctx, FACTS, FEATURES)
    assert full == {"matrix": "empty", "context": False, "statistical_modifier": False, "object": "ratio",
                    "constraint_types": ["state"]}
    assert "constraint_types" not in pipeline.stated_form(ctx, FACTS, frozenset({"form"}))
    assert pipeline.stated_form(ctx, FACTS, frozenset({"names"})) == {}
    assert pipeline.stated_form(ctx, None, FEATURES) == {}


def test_check_reports_schema_and_form_problems(ctx):
    """A conforming answer passes; schema and form failures are named part by part."""
    form = pipeline.stated_form(ctx, FACTS, FEATURES)
    assert pipeline.check(ctx, json.dumps(GOOD), form) == (GOOD, [])
    prediction, complaints = pipeline.check(ctx, json.dumps(WRONG_FORM), form)
    assert prediction == WRONG_FORM and complaints == ['hasMatrix must be "".',
                                                       "hasConstraint must have one or more constraints of each "
                                                       "of these types and of no other type: state."]
    prediction, complaints = pipeline.check(ctx, json.dumps(INVALID), form)
    assert prediction is None and complaints and "/hasConstraint/0/on" in complaints[0]
    assert pipeline.check(ctx, "not json", {}) == (None, ["The answer is not exactly one JSON object."])
    assert pipeline.check(ctx, json.dumps(WRONG_FORM), {}) == (WRONG_FORM, [])


def test_repair_conversation_has_the_prompt_the_answer_and_the_problems():
    """The repair request is the first prompt, the first reply and the listed problems."""
    messages = pipeline.repair_messages("PROMPT", "ANSWER", ["one", "two"])
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "PROMPT" and messages[1]["content"] == "ANSWER"
    assert messages[2]["content"].startswith(pipeline.REPAIR_INTRO + "\n- one\n- two\n\n")


@pytest.mark.asyncio
async def test_the_four_arms(ctx):
    """Each arm makes the calls its rules allow and keeps the right final answer."""
    variable = ctx["practice"][0]
    walks = {variable: {"facts": FACTS}}

    async def go(arm, transport):
        """Run one job of one arm.

        Args: arm: arm name; transport: the stub.
        Returns: the job's record.
        Raises: nothing.
        Side Effects: stub calls.
        """
        return await pipeline.attempt(ctx, (arm, 1, variable), transport, FEATURES, walks, engine.Guard())

    transport, calls = stub(WRONG_FORM)
    record = await go("base", transport)
    assert record["calls"] == 1 and not record["guided"] and json.loads(record["answer"]) == WRONG_FORM
    assert hints.HEADING not in calls[0]["messages"][0]["content"]

    transport, calls = stub(INVALID, repair=GOOD)
    record = await go("base+retry", transport)
    assert record["calls"] == 2 and record["repaired"] and not record["fell_back"]
    assert json.loads(record["answer"]) == GOOD and record["prompt_tokens"] == 10000
    transport, calls = stub(WRONG_FORM)
    record = await go("base+retry", transport)
    assert record["calls"] == 1 and not record["repaired"]

    transport, calls = stub(WRONG_FORM)
    record = await go("guided", transport)
    assert record["calls"] == 1 and record["guided"] and hints.HEADING in calls[0]["messages"][0]["content"]

    transport, calls = stub(WRONG_FORM, repair=GOOD)
    record = await go("guided+check", transport)
    assert record["calls"] == 2 and record["repaired"] and not record["fell_back"]
    assert json.loads(record["answer"]) == GOOD and record["first_complaints"]

    transport, calls = stub(WRONG_FORM, repair=WRONG_FORM, plain=INVALID)
    record = await go("guided+check", transport)
    assert record["calls"] == 3 and record["fell_back"] and json.loads(record["answer"]) == INVALID
    assert hints.HEADING not in calls[2]["messages"][0]["content"]

    transport, calls = stub(GOOD)
    record = await go("guided+check", transport)
    assert record["calls"] == 1 and not record["repaired"] and json.loads(record["answer"]) == GOOD


@pytest.mark.asyncio
async def test_a_variable_without_facts_gets_the_plain_prompt(ctx):
    """When the walk found nothing, a guided arm falls back to today's prompt and checks the schema only."""
    variable = ctx["practice"][0]
    transport, calls = stub(WRONG_FORM)
    record = await pipeline.attempt(ctx, ("guided+check", 1, variable), transport, FEATURES,
                                    {variable: {"facts": None}}, engine.Guard())
    assert not record["guided"] and record["calls"] == 1
    assert hints.HEADING not in calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_run_scores_final_answers_and_resumes(ctx, tmp_path):
    """Records carry the final answer for scoring, and a second run sends nothing."""
    variables = ctx["practice"][:2]
    walks = {v: {"facts": FACTS} for v in variables}
    jobs = engine.schedule(["base", "guided+check"], variables, 2, 3)
    transport, calls = stub(WRONG_FORM, repair=GOOD)
    path = tmp_path / "calls.jsonl"
    result = await pipeline.run(ctx, jobs, transport, path, FEATURES, walks, concurrency=2)
    assert result["issued"] == 8 and result["stopped"] is None and len(calls) == 12
    stored = engine.latest(path)
    assert all(json.loads(r["answer"]) == (GOOD if key[0] == "guided+check" else WRONG_FORM)
               for key, r in stored.items())
    how = pipeline.behaviour(stored)
    assert how["guided+check"] == {"guided": 1.0, "repaired": 1.0, "fell_back": 0.0, "calls_per_answer": 2.0}
    assert how["base"]["calls_per_answer"] == 1.0
    await pipeline.run(ctx, jobs, transport, path, FEATURES, walks, concurrency=2)
    assert len(calls) == 12
    scores = engine.score(path, tmp_path / "scores.jsonl", ctx["records"], ctx["inputs"]["schema"])
    assert len(scores) == 8 and all(row["valid_prediction"] for row in scores.values())
