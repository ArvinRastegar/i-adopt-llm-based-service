"""Offline tests for draft_rag.py: what each arm shows, the gates, the rule check, units and the summary.

No provider and no embedding model: the similarity order here is word overlap, so the tests run
without the gitignored local model.
"""

from __future__ import annotations

import json
import random
import re

import decision_value as dv
import draft_rag as dr
import groups
import hints
import knowledge as kn
import orclient
import pytest
from engine import ft

GOOD = {"hasStatisticalModifier": "", "hasProperty": "count", "hasObjectOfInterest": "bicycle",
        "hasMatrix": "", "hasContextObject": "", "hasConstraint": []}


def overlap(first: str, second: str) -> float:
    """Word overlap of two texts, standing in for the embedding model.

    Args: first, second: texts.
    Returns: shared words over all words, 0.0 for two empty texts.
    Raises: nothing.
    Side Effects: none.
    """
    a, b = set(kn.words(first)), set(kn.words(second))
    return len(a & b) / len(a | b) if a | b else 0.0


@pytest.fixture(scope="module")
def ctx():
    """The stage's context with a word-overlap similarity order.

    Args: none.
    Returns: the mapping from `draft_rag.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files; none of them is stored output.
    """
    base = kn.load()
    order = dr.similarity_order(base["ids"], lambda v: base["records"][v]["definition"], overlap)
    return dr.load(order=order)


@pytest.fixture(scope="module")
def world(ctx):
    """What corpus targets may be shown.

    Args: ctx: as above.
    Returns: from `corpus_world`.
    Raises: nothing.
    Side Effects: none.
    """
    return dr.corpus_world(ctx)


def test_the_similarity_order_is_complete_symmetric_in_its_scores_and_fixed():
    """Every other item once, by falling similarity, equals by id."""
    texts = {"a": "red apple", "b": "red apple pie", "c": "blue sky", "d": "red apple"}
    order = dr.similarity_order(list(texts), texts.get, overlap)
    assert order["a"] == ["d", "b", "c"] and order["c"][-1] in "abd"
    assert all(sorted(others) == sorted(set(texts) - {v}) for v, others in order.items())
    assert dr.similarity_order(list(texts)[::-1], texts.get, overlap) == order


def test_near_relatives_are_the_21_of_the_stage_3_review(ctx):
    """21 variables share at least half their words with a variable of another fold."""
    assert len(ctx["relatives"]) == 21 and set(ctx["relatives"]) <= set(ctx["ids"])


def test_the_baseline_is_todays_prompt(ctx, world):
    """`base` sends today's prompt byte for byte; an example variable sees the other 24 examples."""
    for variable_id in ctx["ids"]:
        prompt = dr.fill_prompt(world["base_inputs"](variable_id), world["definition"](variable_id))
        assert prompt == groups.prompt_of(ctx, "base", variable_id, {})


def test_retrieval_never_shows_the_target_or_its_fold(ctx, world):
    """25 retrieved examples, 85 finder examples and 5 voters, all from other folds, in the stated order."""
    for variable_id in ctx["ids"]:
        fold = ctx["fold_of"][variable_id]
        near = dr.nearest(ctx, variable_id)
        assert len(near) == 85 and all(ctx["fold_of"][u] != fold for u in near)
        retrieved = [record["variable_id"] for record in world["rag_records"](variable_id)]
        assert retrieved == near[:25][::-1]                                  # most similar last
        assert [d["id"] for d in world["finder_rag"](variable_id)] == retrieved
        assert [d["id"] for d in world["finder_draft"](variable_id)] == sorted(near)
        assert [vt["id"] for vt in world["voters"](variable_id)] == near[:5]
        prompt = dr.fill_prompt(dict(ctx["inputs"], examples=world["rag_records"](variable_id)),
                                world["definition"](variable_id))
        examples = ft.examples_block(prompt)
        assert len(re.findall(r"(?m)^Example \d+$", examples)) == 25
        assert ft.answer_text(ctx["records"][variable_id]["gold"]) not in examples
        assert f"Definition: {world['definition'](variable_id)}\nOutput:" not in examples


def test_the_finder_prompt_shows_every_variable_with_its_property_and_object(ctx, world):
    """Examples carry the experts' property and object; the target carries its draft's, last."""
    variable_id = ctx["ids"][17]
    definition = world["definition"](variable_id)
    draft = dict(GOOD, hasProperty="mass", hasObjectOfInterest={"AsymmetricSystem": "a / b", "hasNumerator": "a",
                                                                  "hasDenominator": "b"})
    prompt = dr.finder_prompt(ctx, world["finder_draft"](variable_id), definition, dr.anchor_of(draft))
    assert prompt.startswith(kn.head(ctx)) and f"\n{dr.ANCHOR_LINE}\n\nEXAMPLES\n" in prompt
    assert prompt.endswith(f"Definition: {definition}\n{dr.ANCHOR_LABEL}{dr.anchor_of(draft)}\nPatterns:")
    assert prompt.count(f"Definition: {definition}\n") == 1 and prompt.count(dr.ANCHOR_LABEL) == 86
    for demo in world["finder_draft"](variable_id)[:10]:
        gold = ctx["records"][demo["id"]]["gold"]
        assert json.loads(demo["anchor"]) == {"hasProperty": gold["hasProperty"],
                                              "hasObjectOfInterest": gold["hasObjectOfInterest"]}
        assert f"Definition: {demo['definition']}\n{dr.ANCHOR_LABEL}{demo['anchor']}\nPatterns: {demo['patterns']}\n" in prompt
    assert json.loads(dr.anchor_of(None)) == {"hasProperty": "", "hasObjectOfInterest": ""}
    gold_anchor = dr.anchor_of(ctx["records"][variable_id]["gold"])
    assert f"{dr.ANCHOR_LABEL}{gold_anchor}\nPatterns: " not in prompt or gold_anchor in [
        d["anchor"] for d in world["finder_draft"](variable_id)]


def test_all_decisions_stated_is_the_block_of_steps_1_to_5(ctx):
    """With exact constraint types the block is `decision_value.render`, byte for byte."""
    for variable_id in ctx["ids"]:
        facts = hints.facts_from_key(ctx["key"][variable_id])
        stmt = dr.statement(facts, dv.ALL)
        assert dr.render(ctx, stmt) == dv.render(facts, dv.ALL, ctx["patterns"], ctx["types"])
        assert dr.render(ctx, dr.statement(facts, {"matrix", "context"})) == dv.render(
            facts, {"matrix", "context"}, ctx["patterns"], ctx["types"])
    with pytest.raises(ValueError):
        dr.statement(hints.facts_from_key(ctx["key"][ctx["ids"][0]]), {"shape"})


def test_at_least_wording_and_its_check(ctx):
    """"At least" names the types without excluding others, and the check holds the answer to that."""
    variable_id = next(v for v in ctx["ids"] if len(ctx["key"][v]["constraint"]) >= 2)
    facts = hints.facts_from_key(ctx["key"][variable_id])
    names = hints.expected_form(facts, ctx["patterns"], ctx["types"])["constraint_types"]
    stmt = dr.statement(facts, {"property", "constraint"}, at_least=True)
    block = dr.render(ctx, stmt)
    assert dr.AT_LEAST.format(names="; ".join(names)) in block and "of no other type" not in block
    assert block.index(dv.FORM_HARD) < block.index("at least one constraint") < block.index("Constraint types:")
    only = dr.render(ctx, dr.statement(facts, {"constraint"}, at_least=True))
    assert only.split("\n")[2:5] == [hints.HEADING, dv.FORM_HARD, dr.AT_LEAST.format(names="; ".join(names))]
    form = dr.form_of(ctx, stmt)
    assert form["constraint_types_at_least"] == names and "constraint_types" not in form
    answer = dict(GOOD, hasConstraint=[{"label": f"{name}: something", "on": "bicycle"} for name in names]
                  + [{"label": "time: an extra one", "on": "bicycle"}])
    assert dr.check(ctx, json.dumps(answer), {"constraint_types_at_least": names})[1] == []
    missing = dict(answer, hasConstraint=answer["hasConstraint"][1:])
    assert dr.check(ctx, json.dumps(missing), {"constraint_types_at_least": names})[1] == [
        dr.AT_LEAST_COMPLAINT.format(names="; ".join(names))]
    exact = dr.form_of(ctx, dr.statement(facts, {"constraint"}))
    assert dr.check(ctx, json.dumps(answer), exact)[1]                     # an extra type breaks "exactly"
    assert dr.check(ctx, "not json", {})[0] is None


def test_the_object_shape_is_checked_where_its_line_is_stated(ctx):
    """With the property stated, the object's shape it implies is checked; with only the matrix, not."""
    variable_id = next(v for v in ctx["ids"] if ctx["key"][v]["property"] == [128])     # concentration: a ratio
    facts = dict(hints.facts_from_key(ctx["key"][variable_id]), object=[])
    assert dr.form_of(ctx, dr.statement(facts, {"property"}))["object"] == "ratio"
    assert "object" not in dr.form_of(ctx, dr.statement(facts, {"matrix"}))
    _, complaints = dr.check(ctx, json.dumps(GOOD), dr.form_of(ctx, dr.statement(facts, {"property"})))
    assert complaints == [f"hasObjectOfInterest must be {hints.SHAPE_TEXT['ratio']}"]


def test_what_a_draft_decides():
    """The matrix shape, the context object and the constraint types a decomposition names."""
    draft = dict(GOOD, hasMatrix={"AsymmetricSystem": "a → b", "hasSource": "a", "hasTarget": "b"},
                 hasContextObject="sea", hasConstraint=[{"label": "State: dissolved", "on": "x"},
                                                        {"label": "no type here", "on": "x"},
                                                        {"label": "state: solid", "on": "x"}])
    assert dr.view(draft) == {"matrix": "source_target", "context": True, "types": ["state"]}
    assert dr.view(GOOD) == {"matrix": "empty", "context": False, "types": []} and dr.view(None) is None


def test_the_vote_of_the_nearest():
    """Four of five agreeing shapes state a matrix; the most common pattern among them, the nearer first."""
    def voters(*rows):
        """Voters from (shape, patterns) pairs.

        Args: rows: (shape, tagged matrix patterns).
        Returns: voter dicts.
        Raises: nothing.
        Side Effects: none.
        """
        return [{"id": f"v{n}", "shape": shape, "matrix": list(matrix)} for n, (shape, matrix) in enumerate(rows)]

    empty, single = ("empty", []), ("single", [41])
    assert dr.vote(voters(empty, empty, single, empty, empty)) == []
    assert dr.vote(voters(("single", [42]), ("single", [41]), ("single", [41]), ("single", [42]), empty)) == [42]
    assert dr.vote(voters(("single", [41]), ("single", [42]), ("single", [41]), ("single", [43]), empty)) == [41]
    assert dr.vote(voters(single, single, empty, empty, single)) is None
    assert dr.vote(voters(single, single, single, single, single, empty)) == [41]       # only five vote
    assert dr.vote([]) is None


def test_the_gates(ctx):
    """Agreement states the matrix and the context; shared types are stated as "at least"; the vote wins."""
    finder = {"property": [128], "object": [], "matrix": [41], "context": [], "constraint": [76, 70],
              "statistical_modifier": True, "names": {}}
    agree = {"matrix": "single", "context": False, "types": ["state", "time"]}
    stmt = dr.gated(ctx, agree, finder, None)
    assert stmt["decisions"] == ["constraint", "context", "matrix", "property"] and stmt["at_least"]
    assert hints.expected_form(stmt["facts"], ctx["patterns"], ctx["types"])["constraint_types"] == ["state"]
    assert stmt["facts"]["matrix"] == [41] and stmt["facts"]["statistical_modifier"] is True
    disagree = {"matrix": "empty", "context": True, "types": ["part"]}
    assert dr.gated(ctx, disagree, finder, None)["decisions"] == ["property"]
    voted = dr.gated(ctx, disagree, finder, [])
    assert voted["decisions"] == ["matrix", "property"] and voted["facts"]["matrix"] == []
    assert dr.gated(ctx, None, finder, None)["decisions"] == ["property"]
    only_vote = dr.gated(ctx, agree, None, [42])
    assert only_vote["decisions"] == ["matrix"] and only_vote["facts"]["matrix"] == [42]
    assert dr.gated(ctx, agree, None, None) is None
    assert finder["matrix"] == [41] and finder["constraint"] == [76, 70]        # the finder's facts are untouched
    text = dr.render(ctx, stmt)
    assert "modifier" not in text.lower() or "hasStatisticalModifier" not in text


def test_a_finder_answer_is_read_strictly(ctx):
    """Only the requested JSON object counts; a bare property does not."""
    good = {"property": 104, "matrix": None, "context": False, "modifier": False, "constraints": ["condition"]}
    assert dr.finder_facts(ctx, json.dumps(good))["constraint"] == [70]
    assert dr.finder_facts(ctx, json.dumps({"property": 104})) is None
    assert dr.finder_facts(ctx, json.dumps(dict(good, matrix="41"))) is None and dr.finder_facts(ctx, None) is None


def reply(answer: str, host: str = "Near AI", ok: bool = True) -> dict:
    """A transport result.

    Args: answer: the text; host: who answered; ok: whether it succeeded.
    Returns: the result in the shape of `orclient.read`.
    Raises: nothing.
    Side Effects: none.
    """
    return {"ok": ok, "status": 200 if ok else 500, "error": None if ok else "HTTP 500", "answer": answer,
            "reasoning": "", "reasoning_tokens": 0, "finish_reason": "stop", "prompt_tokens": 1000,
            "completion_tokens": 50, "cost_usd": 0.0002, "host": host, "served_model": "qwen/qwen3.8-27b",
            "latency": 0.1, "attempts": 1}


class Script:
    """A fake transport that answers by the kind of request and keeps what it was sent."""

    def __init__(self, guided_answers: list[str], finder: dict, draft: dict = GOOD, fail_on: str | None = None,
                 rag_draft: dict | None = None):
        """Hold the canned answers.

        Args: guided_answers: answers to guided prompts and repairs, used in order, the last one
            repeating; finder: the pattern-finding answer; draft: the answer to plain prompts;
            fail_on: a definition whose own pattern-finding request fails (the definition that
                ends the prompt, not one shown as an example); rag_draft: the answer to the second
                plain prompt of a unit (`rag`), when it should differ from the first (`base`).
        Returns: nothing.
        Raises: nothing.
        Side Effects: none.
        """
        self.guided, self.finder, self.draft, self.fail_on = list(guided_answers), finder, draft, fail_on
        self.rag_draft = rag_draft if rag_draft is not None else draft
        self.sent: list[tuple[str, dict]] = []
        self.fills = 0

    async def __call__(self, body: dict) -> dict:
        """Answer one request.

        Args: body: the request body.
        Returns: a result.
        Raises: nothing.
        Side Effects: appends to `sent`.
        """
        messages = body["messages"]
        text = messages[0]["content"]
        kind = ("repair" if len(messages) == 3 else "finder" if text.startswith("Classify the scientific variable")
                else "guided" if hints.HEADING in text else "fill")
        self.sent.append((kind, body))
        if self.fail_on and kind == "finder" and text.rsplit("\nDefinition: ", 1)[-1].startswith(self.fail_on):
            return reply("", ok=False)
        if kind == "finder":
            return reply(json.dumps(self.finder))
        if kind == "fill":
            self.fills += 1
            return reply(json.dumps(self.draft if self.fills % 2 else self.rag_draft))
        answer = self.guided.pop(0) if len(self.guided) > 1 else self.guided[0]
        return reply(answer)


def guard(tmp_path):
    """A fresh guard over a ledger in a temporary folder.

    Args: tmp_path: the folder.
    Returns: the guard.
    Raises: nothing.
    Side Effects: none.
    """
    return orclient.Guard(orclient.Spend(ledger=tmp_path / "ledger.jsonl"))


async def test_a_unit_answers_every_arm_from_the_same_drafts(ctx, world, tmp_path):
    """Base and rag are the drafts; a failing guided answer is repaired; a second failure falls back."""
    variable_id = ctx["ids"][3]
    finder = {"property": 104, "matrix": 41, "context": False, "modifier": False, "constraints": ["condition"]}
    fixed = dict(GOOD, hasMatrix="road", hasConstraint=[{"label": "condition: rain", "on": "bicycle"}])
    script = Script([json.dumps(GOOD), json.dumps(fixed)], finder)
    records = await dr.unit(ctx, world, 2, variable_id, script, guard(tmp_path))
    assert [r["arm"] for r in records] == list(dr.ARMS) and all(r["repetition"] == 2 for r in records)
    by_arm = {r["arm"]: r for r in records}
    assert by_arm["base"]["answer"] == by_arm["rag"]["answer"] == json.dumps(GOOD)
    first = by_arm["draft-first"]
    assert first["first_complaints"] and first["repaired"] and not first["repair_complaints"]
    assert first["answer"] == json.dumps(fixed) and not first["fell_back"]
    assert first["statement"]["decisions"] == sorted(dv.ALL) and first["finder_read"]
    gated = by_arm["draft-first/gated"]          # the draft says no matrix, the finder says one: not stated
    assert gated["statement"]["decisions"] == ["context", "property"] and gated["form"] == {
        "object": "single", "context": False}
    kinds = [kind for kind, _ in script.sent]
    assert kinds[:2] == ["fill", "finder"] and kinds.count("finder") == 2 and kinds.count("fill") == 2
    fills = [body["messages"][0]["content"] for kind, body in script.sent if kind == "fill"]
    retrieved_block = ft.examples_block(fills[1])                     # the second plain prompt is `rag`
    assert retrieved_block != ft.examples_block(fills[0])
    for kind, body in script.sent:
        if kind == "guided":                                          # every guided prompt keeps its draft's examples
            arm_block = ft.examples_block(body["messages"][0]["content"])
            assert arm_block in (ft.examples_block(fills[0]), retrieved_block)
    combined_prompt = [body for kind, body in script.sent if kind == "guided"][-1]["messages"][0]["content"]
    assert ft.examples_block(combined_prompt) == retrieved_block
    voted = dr.vote(world["voters"](variable_id))
    assert by_arm["rag/gated"]["no_guidance"] == (voted is None)
    assert by_arm["base"]["unit_calls"] == len(script.sent)
    assert by_arm["base"]["unit_cost_usd"] == pytest.approx(0.0002 * len(script.sent))
    assert all(r["host"] == "Near AI" and r["ok"] for r in records)
    finder_prompt = next(body for kind, body in script.sent if kind == "finder")["messages"][0]["content"]
    assert finder_prompt.endswith(f"{dr.ANCHOR_LABEL}{dr.anchor_of(GOOD)}\nPatterns:")
    for kind, body in script.sent:
        assert body["temperature"] == (0.0 if kind == "finder" else 0.5)

    still_wrong = dict(GOOD, hasProperty="another count")           # never conforms, and is not the draft
    broken = Script([json.dumps(still_wrong)], finder)
    again = {r["arm"]: r for r in await dr.unit(ctx, world, 1, variable_id, broken, guard(tmp_path))}
    assert again["draft-first"]["fell_back"] and again["draft-first"]["answer"] == json.dumps(GOOD)
    assert again["draft-first"]["first_answer"] == json.dumps(still_wrong) and again["draft-first"]["repair_complaints"]


async def test_every_guided_arm_keeps_to_its_own_draft_and_examples(ctx, world, tmp_path):
    """Draft-first arms start from `base` and keep today's examples; retrieval arms from `rag` and its examples."""
    variable_id = next(v for v in ctx["ids"] if dr.vote(world["voters"](v)) is not None)
    finder = {"property": 104, "matrix": 41, "context": False, "modifier": False, "constraints": ["condition"]}
    rag_draft = dict(GOOD, hasProperty="number")
    never = dict(GOOD, hasProperty="yet another count", hasMatrix="", hasConstraint=[])
    script = Script([json.dumps(never)], finder, rag_draft=rag_draft)
    records = {r["arm"]: r for r in await dr.unit(ctx, world, 1, variable_id, script, guard(tmp_path))}
    assert records["base"]["answer"] == json.dumps(GOOD) and records["rag"]["answer"] == json.dumps(rag_draft)
    for arm in dr.GUIDED:
        own = json.dumps(GOOD) if arm.startswith("draft") else json.dumps(rag_draft)
        assert records[arm]["draft_sha256"] == ft.sha256(own)
        if records[arm]["fell_back"] or records[arm]["no_guidance"]:
            assert records[arm]["answer"] == own
    assert records["draft-first"]["fell_back"] and records["combined"]["fell_back"] or records["combined"]["no_guidance"]
    fills = [body["messages"][0]["content"] for kind, body in script.sent if kind == "fill"]
    today, retrieved = ft.examples_block(fills[0]), ft.examples_block(fills[1])
    guided = [body["messages"][0]["content"] for kind, body in script.sent if kind == "guided"]
    assert [ft.examples_block(p) for p in guided] == [today, today, retrieved, retrieved]
    calls = {(r["arm"], 1, variable_id): r for r in records.values()}
    assert dr.unit_integrity(calls, [1], [variable_id])["mismatched"] == 0
    swapped = dict(calls)
    swapped[("combined", 1, variable_id)] = dict(records["combined"], draft_sha256=ft.sha256(json.dumps(GOOD)))
    assert dr.unit_integrity(swapped, [1], [variable_id])["mismatched"] == 1
    torn = dict(calls)
    torn[("rag/gated", 1, variable_id)] = dict(records["rag/gated"], started_at="another time")
    assert dr.unit_integrity(torn, [1], [variable_id])["examples"][0]["problems"] == ["start times differ"]


async def test_a_stop_rule_stops_a_unit_at_its_next_call(ctx, world, tmp_path):
    """Once a stop rule has fired, a unit already under way sends nothing more."""
    stopped = guard(tmp_path)
    stopped.stop("answers carry reasoning despite the switch")
    script = Script([json.dumps(GOOD)], {"property": 104})
    with pytest.raises(dr.Failed, match="stopped"):
        await dr.unit(ctx, world, 1, ctx["ids"][0], script, stopped)
    assert script.sent == []


async def test_without_patterns_the_draft_is_the_answer(ctx, world, tmp_path):
    """An unreadable finder answer and no vote: no guided call, the drafts stand."""
    variable_id = next(v for v in ctx["ids"] if dr.vote(world["voters"](v)) is None)
    script = Script([json.dumps(GOOD)], {"property": "none"})
    records = {r["arm"]: r for r in await dr.unit(ctx, world, 1, variable_id, script, guard(tmp_path))}
    for arm in dr.GUIDED:
        assert records[arm]["no_guidance"] and records[arm]["answer"] == json.dumps(GOOD) and records[arm]["calls"] == 0
    assert [kind for kind, _ in script.sent] == ["fill", "finder", "fill", "finder"]


async def test_a_failed_call_or_another_host_stores_nothing(ctx, world, tmp_path):
    """A unit with a failed call, or an answer from another host, is not stored and is logged."""
    finder = {"property": 104, "matrix": None, "context": False, "modifier": False, "constraints": []}
    ids = ctx["ids"][:3]
    units = [(1, v) for v in ids]
    calls, failed = tmp_path / "calls.jsonl", tmp_path / "failed.jsonl"
    bad = Script([json.dumps(GOOD)], finder, fail_on=ctx["records"][ids[1]]["definition"])
    result = await dr.run(ctx, world, units, bad, guard(tmp_path), calls, failed, concurrency=1)
    assert result["done"] == 2 and result["failed"] == 1
    assert dr.complete_units(calls) == {(1, ids[0]), (1, ids[2])}
    assert json.loads(failed.read_text().splitlines()[0])["variable_id"] == ids[1]
    good = Script([json.dumps(GOOD)], finder)
    again = await dr.run(ctx, world, units, good, guard(tmp_path), calls, failed, concurrency=1)
    assert again["done"] == 1 and dr.complete_units(calls) == set(units)

    class Elsewhere(Script):
        """Answers from another host."""

        async def __call__(self, body: dict) -> dict:
            """Answer from Wafer.

            Args: body: the request body.
            Returns: a result from another host.
            Raises: nothing.
            Side Effects: none.
            """
            return dict(await super().__call__(body), host="Wafer")

    other = guard(tmp_path)
    moved = await dr.run(ctx, world, [(2, ids[0])], Elsewhere([json.dumps(GOOD)], finder), other,
                         tmp_path / "other.jsonl", tmp_path / "other-failed.jsonl", concurrency=1)
    assert moved["failed"] == 1 and "not the pinned host" in moved["stopped"]
    assert not (tmp_path / "other.jsonl").exists() or dr.complete_units(tmp_path / "other.jsonl") == set()


def test_a_unit_counts_as_done_only_with_all_six_arms(tmp_path):
    """Stored records of only some arms, or a failed record, leave the unit to be done again."""
    path = tmp_path / "calls.jsonl"
    rows = [{"arm": arm, "repetition": 1, "variable_id": "a", "ok": True} for arm in dr.ARMS]
    rows += [{"arm": arm, "repetition": 1, "variable_id": "b", "ok": True} for arm in dr.ARMS[:3]]
    rows += [{"arm": arm, "repetition": 2, "variable_id": "a", "ok": arm != "combined"} for arm in dr.ARMS]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert dr.complete_units(path) == {(1, "a")}


def test_units_are_every_variable_in_every_repetition():
    """Ten repetitions of all variables, shuffled within each, each unit once."""
    units = dr.schedule([f"v{n}" for n in range(102)])
    assert len(units) == len(set(units)) == 1020
    assert [r for r, _ in units] == sorted(r for r, _ in units)
    assert units == dr.schedule([f"v{n}" for n in range(102)][::-1])


def invented_run(ctx):
    """Stored calls and scores for every arm, variable and repetition, with steady differences.

    Args: ctx: from `load`.
    Returns: (calls, scores) keyed like `engine.latest` and `engine.score`.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    rng = random.Random(4)
    calls, scores = {}, {}
    for position, arm in enumerate(dr.ARMS):
        for repetition in range(1, dr.REPETITIONS + 1):
            for variable_id in ctx["ids"]:
                right = sum(rng.random() < 0.4 + 0.03 * position for _ in range(6))
                key = (arm, repetition, variable_id)
                stmt = None
                if arm in dr.GUIDED and rng.random() < 0.8:
                    stmt = {"decisions": ["matrix", "property"], "at_least": False, "property": [104],
                            "matrix_shape": "empty", "matrix": []}
                calls[key] = {"arm": arm, "repetition": repetition, "variable_id": variable_id, "ok": True,
                              "started_at": f"{repetition}/{variable_id}", "draft_sha256": ft.sha256("{}"),
                              "answer": "{}", "calls": 1, "cost_usd": 0.001, "unit_cost_usd": 0.006,
                              "statement": stmt, "no_guidance": stmt is None, "first_complaints": [],
                              "repaired": False, "repair_complaints": [], "fell_back": False,
                              "draft_complaints": ["x"]}
                scores[key] = {"key": list(key), "valid_prediction": True, "prediction": None,
                               "evaluation": {"close": {"totals": {
                                   "tp": {"numerator": right, "denominator": 1},
                                   "fp": {"numerator": rng.randint(0, 2), "denominator": 1},
                                   "fn": {"numerator": 6 - right, "denominator": 1}}}}}
    return calls, scores


def totals_f1(evaluations, ids=None):
    """Micro F1 from Close totals, standing in for the study's aggregator.

    Args: evaluations: score rows' evaluations; ids: unused.
    Returns: {"f1": value}.
    Raises: nothing.
    Side Effects: none.
    """
    tp = sum(e["close"]["totals"]["tp"]["numerator"] for e in evaluations)
    fp = sum(e["close"]["totals"]["fp"]["numerator"] for e in evaluations)
    fn = sum(e["close"]["totals"]["fn"]["numerator"] for e in evaluations)
    return {"f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def test_the_summary_compares_each_arm_with_todays_prompt(ctx, monkeypatch):
    """Five comparisons in the family, three secondary ones, the relatives split and the pattern marks."""
    monkeypatch.setattr(dr, "_aggregate", totals_f1)
    calls, scores = invented_run(ctx)
    summary = dr.summarise(ctx, calls, scores, draws=300)
    assert summary["complete"] and summary["repetitions_used"] == list(range(1, 11))
    assert list(summary["comparisons"]) == list(dr.COMPARED) and dr.FAMILY == 5 and dr.STRICT == pytest.approx(0.01)
    every = list(range(1, 11))
    for arm in dr.COMPARED:
        row = summary["comparisons"][arm]
        mine = (totals_f1([scores[(arm, r, v)]["evaluation"] for r in every for v in ctx["ids"]])["f1"]
                - totals_f1([scores[("base", r, v)]["evaluation"] for r in every for v in ctx["ids"]])["f1"])
        assert row["gain"] == pytest.approx(mine) and (row["first"], row["second"]) == (arm, "base")
        assert row["strict_low"] <= row["low"] <= row["high"] <= row["strict_high"]
        assert summary["without_relatives"][arm]["variables"] == 81
        assert summary["with_relatives"][arm]["variables"] == 21
    assert set(summary["secondary"]) == {"draft-first/gated vs draft-first", "rag/gated vs rag", "combined vs rag"}
    marks = summary["patterns"]["rag/gated"]
    assert marks["answers"] == 1020 and marks["guided"] + marks["no_guidance"] == 1020
    tagged_empty = sum(not ctx["key"][v]["matrix"] for v in ctx["ids"])
    assert 0 < marks["matrix_right"] <= marks["matrix_stated"] and tagged_empty == 51
    assert summary["spent_usd"] == pytest.approx(1020 * 0.006)
    broken = dict(calls)
    broken[("combined", 4, ctx["ids"][9])] = dict(calls[("combined", 4, ctx["ids"][9])], ok=False)
    partial = dr.summarise(ctx, broken, scores, draws=50)
    assert not partial["complete"] and 4 not in partial["repetitions_used"] and len(partial["repetitions_used"]) == 9
    assert summary["unit_integrity"] == {"units": 1020, "mismatched": 0, "examples": []}
    torn = dict(calls)
    torn[("rag/gated", 7, ctx["ids"][2])] = dict(calls[("rag/gated", 7, ctx["ids"][2])], started_at="another write")
    mixed = dr.summarise(ctx, torn, scores, draws=50)
    assert mixed["unit_integrity"]["mismatched"] == 1 and not mixed["complete"]
    assert mixed["repetitions_used"] == list(range(1, 11))       # the figures are still given, flagged


def test_marking_what_was_stated(ctx):
    """Stated decisions are marked against the experts' tags; the check's outcome is counted."""
    with_matrix = next(v for v in ctx["ids"] if ctx["key"][v]["matrix"] and ctx["key"][v]["constraint"])
    want = hints.expected_form(hints.facts_from_key(ctx["key"][with_matrix]), ctx["patterns"], ctx["types"])
    right = {"decisions": ["constraint", "matrix", "property"], "at_least": True,
             "property": list(ctx["key"][with_matrix]["property"]), "matrix_shape": want["matrix"],
             "matrix": list(ctx["key"][with_matrix]["matrix"]), "constraint_types": want["constraint_types"][:1]}
    wrong = dict(right, property=[0], matrix_shape="empty", matrix=[], constraint_types=["no such type"])
    records = [{"variable_id": with_matrix, "statement": right, "first_complaints": [], "repaired": False,
                "repair_complaints": [], "fell_back": False, "draft_complaints": []},
               {"variable_id": with_matrix, "statement": wrong, "first_complaints": ["x"], "repaired": True,
                "repair_complaints": ["x"], "fell_back": True, "draft_complaints": ["x"]},
               {"variable_id": with_matrix, "statement": None}]
    marks = dr.patterns_marked(ctx, records)
    assert (marks["answers"], marks["guided"], marks["no_guidance"]) == (3, 2, 1)
    assert (marks["property_right"], marks["matrix_right"], marks["matrix_pattern_right"]) == (1, 1, 1)
    assert marks["types_stated"] == 2 and marks["types_tagged"] == 1 and marks["types_contained"] == 1
    assert (marks["passed_first"], marks["repaired"], marks["fell_back"], marks["draft_would_pass"]) == (1, 1, 1, 1)
    assert marks["matrix_precision"] == 0.5 and marks["fell_back_share"] == 0.5


def test_the_dev_world_keeps_to_the_invented_variables(ctx):
    """In the dev check everything retrieved is an invented variable of the other half."""
    world, order = dr.dev_world(ctx, overlap)
    assert len(order) == 16
    for position, name in enumerate(order):
        other = {u for i, u in enumerate(order) if i % 2 != position % 2}
        assert {r["id"] for r in world["rag_records"](name)} == other
        assert {d["id"] for d in world["finder_draft"](name)} == other
        assert {vt["id"] for vt in world["voters"](name)} <= other and len(world["voters"](name)) == 5
        assert world["base_inputs"](name) is ctx["inputs"]
    corpus = {record["definition"] for record in ctx["records"].values()}
    assert not {world["definition"](name) for name in order} & corpus
