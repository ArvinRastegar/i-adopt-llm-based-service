"""Offline tests for walk.py: questions, demonstrations, walking and marking. No provider."""

from __future__ import annotations

import json
import math

import hints
import pytest
import rulebook
import walk
from engine import ft


@pytest.fixture(scope="module")
def ctx():
    """Load the route test's inputs once per module.

    Args: none.
    Returns: the mapping from `walk.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files.
    """
    return walk.load()


def scripted(ctx, entry):
    """Build an `ask` that answers every question as the answer key would.

    Args: ctx: from `walk.load`; entry: an answer-key entry.
    Returns: (ask, asked), where `asked` collects the ids of the questions put.
    Raises: nothing.
    Side Effects: none until the function is awaited.
    """
    answers = walk.right_answers(ctx, entry)
    for step in rulebook.routes_to(ctx["trees"], entry["property"][0])[0]:
        answers.setdefault(f"{step['tree']}/{step['node']}", step["answer"])
    asked = []

    async def ask(question, definition):
        """Answer one question from the script.

        Args: question: the question; definition: ignored.
        Returns: the scripted option with full confidence.
        Raises: KeyError when the walk asks a question off the scripted route.
        Side Effects: appends the question id to `asked`.
        """
        asked.append(question["id"])
        return {"answer": answers[question["id"]], "confidence": 1.0, "probabilities": [], "ok": True, "status": 200}

    return ask, asked


def test_our_matrix_question_gives_a_no_matrix_exit(ctx):
    """DT8 gains one question of ours; six noes end on "no matrix", and 42 is still reachable."""
    tree = ctx["matrix_tree"]
    assert tree["nodes"]["OUR1"] == {"kind": "question", "text": walk.OUR_MATRIX_QUESTION, "ours": True}
    none = rulebook.routes_to({"DT8": tree}, walk.NO_MATRIX, start_tree="DT8")
    assert [s["answer"] for s in none[0]] == ["false"] * 6 and len(none) == 1
    to_42 = rulebook.routes_to({"DT8": tree}, 42, start_tree="DT8")
    assert [s["node"] for s in to_42[0]][-1] == "OUR1" and to_42[0][-1]["answer"] == "true"
    assert "OUR1" not in ctx["trees"]["DT8"]["nodes"]


def test_one_question_of_ours_per_constraint_type_name(ctx):
    """The two "process" rows merge, so 42 table rows give 41 questions, all marked ours."""
    questions = ctx["constraint_questions"]
    assert len(questions) == 41 and all(q["ours"] and q["options"] == ["yes", "no"] for q in questions)
    process = next(q for q in questions if q["name"] == "process")
    assert " Or: " in process["text"] and process["number"] == 90


def test_question_prompt_numbers_options_and_demonstrations():
    """Options and demonstration answers are numbered, and the prompt ends ready for the answer."""
    question = {"text": "Is it length?", "options": ["true", "false"]}
    prompt = walk.question_prompt(question, [("width of a road", "true"), ("mass of a rock", "false")], "depth of snow")
    assert "Question: Is it length?\nOptions:\n1. true\n2. false" in prompt
    assert "Definition: width of a road\nAnswer: 1" in prompt and "Definition: mass of a rock\nAnswer: 2" in prompt
    assert prompt.endswith("Definition: depth of snow\nAnswer:")
    assert "Examples:" not in walk.question_prompt(question, [], "depth of snow")


def test_right_answers_follow_the_key_and_skip_ambiguous_questions(ctx):
    """A one-route pattern answers every question on its route; a two-route pattern skips the fork."""
    one = walk.right_answers(ctx, {"property": [127], "object": [35], "matrix": [41], "context": [], "constraint": [76],
                                   "statistical_modifier": False, "names": {}})
    assert one["DT0/B"] == "true" and one["DT5/C"] == "Entity" and one["DT8/H"] == "true"
    assert one["OUR/context"] == "no" and one["OUR/constraint/state"] == "yes" and one["OUR/constraint/age"] == "no"
    two = walk.right_answers(ctx, {"property": [110], "object": [35], "matrix": [], "context": [50], "constraint": [],
                                   "statistical_modifier": True, "names": {}})
    assert "DT0/B" not in two and two["DT8/OUR1"] == "false" and two["OUR/modifier"] == "yes"


def test_demonstrations_come_from_the_examples_only(ctx):
    """Every demonstration is an example variable's definition; none is a practice or held-out one."""
    examples = {ctx["records"][v]["definition"] for v in ft.EXAMPLE_IDS}
    others = {ctx["records"][v]["definition"] for v in ctx["records"] if v not in ft.EXAMPLE_IDS}
    index = walk.demo_index(ctx)
    for demos in index.values():
        assert 0 < len(demos) <= walk.DEMOS_PER_QUESTION
        assert all(text in examples and text not in others for text, _ in demos)
    assert {answer for _, answer in index["DT3/N"]} == {"true", "false"}
    assert len(index["OUR/context"]) == 25 and sum(a == "yes" for _, a in index["OUR/context"]) == 2


def test_tree_questions_say_what_they_are_about_and_where_answers_lead(ctx):
    """With the pattern table, a question names its part and each option lists its patterns."""
    question = walk.tree_question("DT1", ctx["trees"]["DT1"], "B", ctx["trees"], ctx["patterns"])
    assert question["about"] == walk.ABOUT_PROPERTY and question["options"] == ["false", "true"]
    assert question["leads"][0] == "index" and "fraction" in question["leads"][1]
    last = walk.tree_question("DT8", ctx["matrix_tree"], "OUR1", {"DT8": ctx["matrix_tree"]}, ctx["patterns"])
    assert last["about"] == walk.ABOUT_MATRIX and last["leads"] == ["artificial container", "no matrix"]
    prompt = walk.question_prompt(question, [], "x")
    assert "1. false (leads to: index)" in prompt and walk.ABOUT_PROPERTY in prompt
    assert walk.reachable(ctx["trees"], "DT0", "B") == sorted(n for n in ctx["reachable"])


def test_option_probabilities_sum_spellings_and_ignore_other_tokens():
    """" 1" and "1" count for option one; unrelated tokens are ignored; the rest is normalised."""
    top = [{"token": "1", "logprob": math.log(0.6)}, {"token": " 1", "logprob": math.log(0.1)},
           {"token": "2", "logprob": math.log(0.2)}, {"token": "The", "logprob": math.log(0.1)}]
    assert walk.option_probabilities(top, 2) == pytest.approx([0.7 / 0.9, 0.2 / 0.9])
    assert walk.option_probabilities([{"token": "x", "logprob": -1.0}], 2) == [0.0, 0.0]


@pytest.mark.asyncio
async def test_tree_walk_reaches_the_guide_pattern_and_skips_one_option_questions(ctx):
    """Scripted answers lead to pattern 127 in ten questions; a one-option question costs no call."""
    ask, asked = scripted(ctx, {"property": [127], "object": [35], "matrix": [], "context": [], "constraint": [],
                                "statistical_modifier": False, "names": {}})
    result = await walk.walk_tree(ctx["trees"], "DT0", ask, "x")
    assert result["pattern"] == 127 and len(result["steps"]) == 10 and result["confidence"] == 1.0
    ask, asked = scripted(ctx, {"property": [134], "object": [33], "matrix": [], "context": [], "constraint": [],
                                "statistical_modifier": False, "names": {}})
    result = await walk.walk_tree(ctx["trees"], "DT0", ask, "x")
    assert result["pattern"] == 134 and "DT5/R" not in asked


@pytest.mark.asyncio
async def test_walking_with_the_key_answers_reproduces_the_key_for_every_variable(ctx):
    """For all 102 variables a walk answered from the key lands on the key's own form.

    The object shape is read off the property pattern, which is wrong for exactly one variable:
    its property pattern (117 direction) demands a symmetric object and its tag says single.
    """
    object_misses = []
    for variable_id, entry in ctx["key"].items():
        ask, _ = scripted(ctx, entry)
        result = await walk.walk_steps(ctx, "x", ask)
        marks = walk.mark(ctx, result["facts"], entry)
        assert all(marks[k] for k in ("property", "matrix", "matrix_pattern", "context", "modifier")), variable_id
        if {n for n in entry["constraint"]} <= {t["number"] for t in ctx["types"]}:
            assert marks["constraint_set"], variable_id
        if not marks["object"]:
            object_misses.append(entry["property"])
    assert object_misses == [[117]]


def test_direct_reply_round_trip_and_rejections(ctx):
    """The key in direct form reads back as itself; junk and unknown patterns give no facts."""
    for entry in ctx["key"].values():
        facts = walk.facts_from_direct(ctx, json.dumps(walk.direct_answer(ctx, entry)))
        marks = walk.mark(ctx, facts, entry)
        assert marks["property"] and marks["matrix"] and marks["context"] and marks["modifier"]
    assert walk.facts_from_direct(ctx, "no json here") is None
    assert walk.facts_from_direct(ctx, '{"property": 999}') is None
    assert walk.facts_from_direct(ctx, '{"property": 41}') is None
    facts = walk.facts_from_direct(ctx, '{"property": 104, "matrix": 7, "constraints": ["state", "bogus"], "context": true}')
    assert facts["matrix"] == [] and facts["constraint"] == [76] and facts["context"] == [50]


def test_marking_counts_each_decision(ctx):
    """A walk with the wrong matrix and one extra constraint type is marked part by part."""
    entry = {"property": [104], "object": [33], "matrix": [], "context": [], "constraint": [92],
             "statistical_modifier": False, "names": {}}
    facts = {"property": [104], "object": [], "matrix": [41], "context": [], "constraint": [76, 92],
             "statistical_modifier": False, "names": {}}
    marks = walk.mark(ctx, facts, entry)
    assert marks["property"] and marks["object"] and not marks["matrix"] and not marks["constraint_set"]
    assert (marks["constraint_tp"], marks["constraint_fp"], marks["constraint_fn"]) == (1, 1, 0)
    assert walk.mark(ctx, None, entry)["constraint_fn"] == 1


@pytest.mark.asyncio
async def test_run_stores_one_row_per_item_resumes_and_stops_on_reasoning(ctx, tmp_path):
    """Walks are stored per item, a second run walks nothing, and reasoning text stops the run."""
    calls = []

    async def transport(body):
        """Answer every question with option 2 and report its probability.

        Args: body: the request body.
        Returns: a successful reply.
        Raises: nothing.
        Side Effects: appends `body` to `calls`.
        """
        calls.append(body)
        return {"ok": True, "status": 200, "latency": 0.01, "answer": "2", "reasoning": "",
                "top": [{"token": "2", "logprob": math.log(0.9)}, {"token": "1", "logprob": math.log(0.1)}],
                "prompt_tokens": 300, "completion_tokens": 1}

    path = tmp_path / "route.jsonl"
    state = await walk.run(ctx, "steps", {"a": "one", "b": "two"}, transport, path, concurrency=2)
    assert state == {"walked": 2, "skipped": 0, "stopped": None}
    body = calls[0]
    assert body["model"] == "Qwen3.8-27B" and body["temperature"] == 0.0 and body["max_tokens"] == 1
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["logprobs"] is True
    rows = walk.stored(path)
    assert set(rows) == {"a", "b"} and all(r["complete"] for r in rows.values())
    assert rows["a"]["confidence"]["property"] == pytest.approx(0.9)
    before = len(calls)
    await walk.run(ctx, "steps", {"a": "one", "b": "two"}, transport, path, concurrency=2)
    assert len(calls) == before

    async def thinking(body):
        """Reply with reasoning text.

        Args: body: the request body.
        Returns: a reply that carries reasoning.
        Raises: nothing.
        Side Effects: none.
        """
        return {"ok": True, "status": 200, "latency": 0.01, "answer": "{}", "reasoning": "hmm", "top": []}

    state = await walk.run(ctx, "direct", {"a": "one", "b": "two", "c": "three"}, thinking, tmp_path / "d.jsonl", concurrency=1)
    assert state["stopped"] and state["walked"] == 1 and state["skipped"] == 2
    assert hints.expected_form(rows["a"]["facts"], ctx["patterns"], ctx["types"])["matrix"] in ("empty", "single", "source_target")
