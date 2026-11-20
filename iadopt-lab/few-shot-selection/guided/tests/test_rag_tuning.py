"""Offline tests for rag_tuning.py: what each arm shows, the retry, units, the choice by fold, the summary.

No provider and no embedding model: similarity here is word overlap.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction

import draft_rag as dr
import groups
import knowledge as kn
import orclient
import pytest
import rag_tuning as rt
from engine import ft

GOOD = {"hasStatisticalModifier": "", "hasProperty": "count", "hasObjectOfInterest": "bicycle",
        "hasMatrix": "", "hasContextObject": "", "hasConstraint": []}


def overlap(first: str, second: str) -> float:
    """Word overlap of two texts, standing in for the embedding model.

    Args: first, second: texts.
    Returns: shared words over all words.
    Raises: nothing.
    Side Effects: none.
    """
    a, b = set(kn.words(first)), set(kn.words(second))
    return len(a & b) / len(a | b) if a | b else 0.0


@pytest.fixture(scope="module")
def ctx():
    """The stage's context with a word-overlap similarity table.

    Args: none.
    Returns: the mapping from `rag_tuning.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files; none of them is stored output.
    """
    base = kn.load()
    return rt.load(table=rt.similarity_table(base["ids"], lambda v: base["records"][v]["definition"], overlap))


@pytest.fixture(scope="module")
def world(ctx):
    """What corpus targets may be shown, labels compared by word overlap.

    Args: ctx: as above.
    Returns: from `corpus_world`.
    Raises: nothing.
    Side Effects: none.
    """
    return rt.corpus_world(ctx, overlap)


def test_the_similarity_table_and_its_order():
    """Symmetric, rounded, every pair once; the order is `draft_rag`'s for the same similarity."""
    texts = {"a": "red apple", "b": "red apple pie", "c": "blue sky", "d": "red apple"}
    table = rt.similarity_table(list(texts), texts.get, overlap)
    assert table["a"]["b"] == table["b"]["a"] and set(table["a"]) == {"b", "c", "d"} and "a" not in table["a"]
    assert rt.order_of(table) == dr.similarity_order(list(texts), texts.get, overlap)


def test_the_context_takes_its_order_from_the_table(ctx):
    """The pool of every target is the 85 other-fold variables by falling similarity in the table."""
    for variable_id in ctx["ids"]:
        pool = dr.nearest(ctx, variable_id)
        row = ctx["table"][variable_id]
        assert len(pool) == 85 and all(row[a] >= row[b] for a, b in zip(pool, pool[1:]))
        assert all(ctx["fold_of"][u] != ctx["fold_of"][variable_id] for u in pool)
    assert len(ctx["relatives"]) == 21


def test_what_each_arm_shows(ctx, world):
    """Every retrieved example is another fold's variable; sizes and orders are as planned."""
    for variable_id in ctx["ids"]:
        pool = world["pool"](variable_id)
        fold = ctx["fold_of"][variable_id]
        similar = lambda u, v=variable_id: ctx["table"][v][u]          # noqa: E731
        shown = {arm: rt.examples(world, arm, variable_id) for arm in rt.CONFIGS if arm != "rag-25-labels"}
        assert shown["rag-25"] == pool[:25][::-1] == shown["rag-25+retry"]
        assert shown["rag-10"] == pool[:10][::-1] and shown["rag-50"] == pool[:50][::-1]
        assert shown["rag-85"] == pool[::-1] and shown["rag-25-first"] == pool[:25]
        fixed = world["fixed"](variable_id)
        assert shown["fixed+rag-10"][:len(fixed)] == fixed and len(shown["fixed+rag-10"]) == len(fixed) + 10
        extra = shown["fixed+rag-10"][len(fixed):]
        assert extra == [u for u in pool if u not in fixed][:10][::-1]
        assert len(fixed) == (24 if variable_id in ft.EXAMPLE_IDS else 25) and variable_id not in fixed
        demos = [{"id": u, "definition": ctx["records"][u]["definition"]} for u in pool]
        assert shown["rag-25-words"] == [d["id"] for d in kn.nearest_last(world["definition"](variable_id), demos)][-25:]
        mmr = shown["rag-25-mmr"]
        assert len(set(mmr)) == 25 and set(mmr) <= set(pool) and pool[0] in mmr
        assert [similar(u) for u in mmr] == sorted(similar(u) for u in mmr)
        for arm, ids in shown.items():
            retrieved = ids[len(fixed):] if arm == "fixed+rag-10" else ids
            assert variable_id not in ids and all(ctx["fold_of"][u] != fold for u in retrieved)
            assert len(set(ids)) == len(ids)


def test_diversity_moves_away_from_near_duplicates():
    """With equal relevance, maximal marginal relevance prefers an item unlike the ones picked."""
    sims = {("t", "a"): 0.9, ("t", "b"): 0.85, ("t", "c"): 0.84, ("a", "b"): 0.99, ("a", "c"): 0.1, ("b", "c"): 0.1}
    similar = lambda x, y: sims.get((x, y), sims.get((y, x), 0.0))          # noqa: E731
    assert rt.mmr("t", ["a", "b", "c"], similar, 2) == ["c", "a"]           # b is a near duplicate of a
    assert rt.mmr("t", ["a", "b", "c"], similar, 2, weight=1.0) == ["b", "a"]
    assert rt.mmr("t", ["a"], similar, 5) == ["a"]


def test_retrieval_by_the_drafts_labels(ctx, world):
    """A draft that names a candidate's labels pulls that candidate in; without a draft, plain order."""
    variable_id = ctx["ids"][12]
    pool = world["pool"](variable_id)
    assert rt.examples(world, "rag-25-labels", variable_id, None) == pool[:25][::-1]
    far = pool[-1]                                                     # the least similar definition
    draft = dict(ctx["records"][far]["gold"])
    chosen = rt.examples(world, "rag-25-labels", variable_id, draft)
    assert far in chosen and far not in pool[:25] and len(chosen) == 25 and len(set(chosen)) == 25
    assert all(ctx["fold_of"][u] != ctx["fold_of"][variable_id] for u in chosen)


def test_labels_of_a_decomposition():
    """Plain labels, the parts of systems and the constraint labels, in order, empty ones left out."""
    decomposition = {"hasStatisticalModifier": "mean", "hasProperty": "mass concentration",
                     "hasObjectOfInterest": {"AsymmetricSystem": "nitrate / water", "hasNumerator": "nitrate",
                                             "hasDenominator": "water"},
                     "hasMatrix": "", "hasContextObject": "catchment",
                     "hasConstraint": [{"label": "state: dissolved", "on": "nitrate"}]}
    assert rt.labels_of(decomposition) == "mass concentration; nitrate; water; catchment; mean; state: dissolved"
    assert rt.labels_of(GOOD) == "count; bicycle"


def test_prompts_are_todays_with_other_examples(ctx, world):
    """`base` is today's prompt; other arms change the examples only."""
    for variable_id in ctx["ids"][:30]:
        assert dr.fill_prompt(world["base_inputs"](variable_id), world["definition"](variable_id)) == groups.prompt_of(
            ctx, "base", variable_id, {})
        ids = rt.examples(world, "rag-25", variable_id)
        prompt = rt.prompt_of(world, variable_id, ids)
        assert prompt == dr.fill_prompt(dict(ctx["inputs"], examples=[ctx["records"][u] for u in ids]),
                                        world["definition"](variable_id))
        block = ft.examples_block(prompt)
        assert ft.answer_text(ctx["records"][variable_id]["gold"]) not in block


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
    """A fake transport: valid answers, except where told; keeps what it was sent."""

    def __init__(self, invalid_calls: set[int] = frozenset(), host: str = "Near AI", fail_at: int | None = None,
                 first: str | None = None):
        """Hold the plan of answers.

        Args: invalid_calls: positions (from 1) of calls answered with text that is no JSON; host:
            who answers; fail_at: a position whose call fails; first: the text of the first answer
            (a unit's draft), when it should be a given one.
        Returns: nothing.
        Raises: nothing.
        Side Effects: none.
        """
        self.invalid, self.host, self.fail_at, self.first = set(invalid_calls), host, fail_at, first
        self.sent: list[dict] = []

    async def __call__(self, body: dict) -> dict:
        """Answer one request.

        Args: body: the request body.
        Returns: a result.
        Raises: nothing.
        Side Effects: appends to `sent`.
        """
        self.sent.append(body)
        position = len(self.sent)
        if position == self.fail_at:
            return reply("", ok=False)
        if position in self.invalid:
            return reply(f"not a decomposition ({position})", host=self.host)
        if position == 1 and self.first is not None:
            return reply(self.first, host=self.host)
        return reply(json.dumps(dict(GOOD, hasProperty=f"count {position}")), host=self.host)


def guard(tmp_path):
    """A fresh guard over a ledger in a temporary folder.

    Args: tmp_path: the folder.
    Returns: the guard.
    Raises: nothing.
    Side Effects: none.
    """
    return orclient.Guard(orclient.Spend(ledger=tmp_path / "ledger.jsonl"))


async def test_a_unit_answers_eleven_arms_with_ten_calls(ctx, world, tmp_path):
    """One call per arm except the retry, which reuses `rag-25`'s answer; one start time; the right draft."""
    variable_id = ctx["ids"][5]
    far = world["pool"](variable_id)[-1]          # a draft with this variable's labels must pull it in
    script = Script(first=json.dumps(ctx["records"][far]["gold"]))
    records = await rt.unit(ctx, world, 3, variable_id, script, guard(tmp_path))
    assert [r["arm"] for r in records] == list(rt.ARMS) and len(script.sent) == 10
    by_arm = {r["arm"]: r for r in records}
    assert len({r["started_at"] for r in records}) == 1 and all(r["repetition"] == 3 for r in records)
    assert by_arm["rag-25-labels"]["draft_sha256"] == ft.sha256(by_arm["base"]["answer"])
    assert by_arm["rag-25+retry"]["answer"] == by_arm["rag-25"]["answer"] and not by_arm["rag-25+retry"]["repaired"]
    assert by_arm["rag-25+retry"]["first_sha256"] == ft.sha256(by_arm["rag-25"]["answer"])
    assert by_arm["base"]["unit_calls"] == 10 and by_arm["base"]["unit_cost_usd"] == pytest.approx(0.002)
    for arm in rt.CONFIGS:
        if arm not in ("rag-25-labels", "rag-25+retry"):
            assert by_arm[arm]["examples"] == rt.examples(world, arm, variable_id)
    draft, _ = dr.check(ctx, by_arm["base"]["answer"], {})
    labels = by_arm["rag-25-labels"]
    assert draft is not None and labels["examples"] == rt.examples(world, "rag-25-labels", variable_id, draft)
    assert far in labels["examples"] and far not in by_arm["rag-25"]["examples"]
    assert labels["prompt_sha256"] == ft.sha256(rt.prompt_of(world, variable_id, labels["examples"]))
    assert labels["prompt_sha256"] in {ft.sha256(body["messages"][0]["content"]) for body in script.sent}
    assert all(body["temperature"] == 0.5 and body["max_tokens"] == 16000 for body in script.sent)
    calls = {(r["arm"], 3, variable_id): r for r in records}
    assert rt.unit_integrity(calls, [3], [variable_id])["mismatched"] == 0


async def test_the_retry_keeps_a_valid_repair_and_otherwise_the_first_answer(ctx, world, tmp_path):
    """An unusable `rag-25` answer goes back once; a valid repair replaces it, an unusable one does not."""
    variable_id = ctx["ids"][6]
    fixed = await rt.unit(ctx, world, 1, variable_id, Script(invalid_calls={2}), guard(tmp_path))
    retry = {r["arm"]: r for r in fixed}["rag-25+retry"]
    assert retry["repaired"] and retry["repair_valid"] and retry["calls"] == 1
    assert json.loads(retry["answer"])["hasProperty"] == "count 11"            # the eleventh call
    assert {r["arm"]: r for r in fixed}["rag-25"]["answer"] == "not a decomposition (2)"
    script = Script(invalid_calls={2, 11})
    kept = {r["arm"]: r for r in await rt.unit(ctx, world, 1, variable_id, script, guard(tmp_path))}["rag-25+retry"]
    assert kept["repaired"] and kept["repair_valid"] is False and kept["answer"] == "not a decomposition (2)"
    repair = script.sent[-1]["messages"]
    assert len(repair) == 3 and repair[1]["content"] == "not a decomposition (2)"


async def test_a_unit_stops_on_failure_another_host_or_a_stop_rule(ctx, world, tmp_path):
    """Nothing is returned for a failed call, an answer from another host, or after a stop rule fired."""
    variable_id = ctx["ids"][7]
    with pytest.raises(dr.Failed):
        await rt.unit(ctx, world, 1, variable_id, Script(fail_at=4), guard(tmp_path))
    other = guard(tmp_path)
    with pytest.raises(dr.Failed):
        await rt.unit(ctx, world, 1, variable_id, Script(host="Wafer"), other)
    assert "not the pinned host" in other.reason
    stopped = guard(tmp_path)
    stopped.stop("answers carry reasoning despite the switch")
    script = Script()
    with pytest.raises(dr.Failed, match="stopped"):
        await rt.unit(ctx, world, 1, variable_id, script, stopped)
    assert script.sent == []


async def test_only_complete_units_are_stored(ctx, world, tmp_path):
    """A failed unit leaves nothing but a line in the failure log, and is done on the next run."""
    calls, failed = tmp_path / "calls.jsonl", tmp_path / "failed.jsonl"
    units = [(1, v) for v in ctx["ids"][:2]]
    result = await rt.run(ctx, world, units, Script(fail_at=3), guard(tmp_path), calls, failed, concurrency=1)
    assert result["failed"] == 1 and result["done"] == 1 and len(rt.complete_units(calls)) == 1
    again = await rt.run(ctx, world, units, Script(), guard(tmp_path), calls, failed, concurrency=1)
    assert again["done"] == 1 and rt.complete_units(calls) == set(units)
    partial = tmp_path / "partial.jsonl"
    partial.write_text("".join(json.dumps({"arm": arm, "repetition": 1, "variable_id": "x", "ok": True}) + "\n"
                               for arm in rt.ARMS[:-1]))
    assert rt.complete_units(partial) == set()


def counts_for(ctx, rule):
    """Per-variable counts of every arm, from a rule.

    Args: ctx: from `load`; rule: (arm, variable) -> right items of ten.
    Returns: {arm: {variable: [TP, FP, FN]}}.
    Raises: nothing.
    Side Effects: none.
    """
    return {arm: {v: [float(rule(arm, v)), 0.0, float(10 - rule(arm, v))] for v in ctx["ids"]} for arm in rt.ARMS}


def test_a_fold_has_no_say_in_its_own_choice(ctx):
    """A setting best only on a fold's own variables is not chosen for it; equal ones go to the first."""
    fold0 = {v for v in ctx["ids"] if ctx["fold_of"][v] == 0}

    def rule(arm, v):
        """`rag-50` wins outside fold 0, `rag-85` wins hugely inside it; others are level.

        Args: arm, v: the arm and the variable.
        Returns: right items.
        Raises: nothing.
        Side Effects: none.
        """
        if arm == "rag-85":
            return 10 if v in fold0 else 6
        return 8 if arm == "rag-50" and v not in fold0 else 6

    choice = rt.choose(ctx, counts_for(ctx, rule))
    assert choice[0] == "rag-50" and all(choice[f] == "rag-50" for f in range(1, 6))
    level = rt.choose(ctx, counts_for(ctx, lambda arm, v: 7))
    assert set(level.values()) == {"rag-25"}                           # the first setting among equals
    assert rt.pooled_f1({"a": [1.0, 0.0, 0.0]}, ["a"]) == Fraction(1)
    assert rt.pooled_f1({"a": [Fraction(1, 3), Fraction(0), Fraction(1, 3)]}, ["a"]) == Fraction(2, 3)
    split = {"rag-25": {v: [Fraction(1, 85) if v == ctx["ids"][0] else Fraction(0), 0, 1] for v in ctx["ids"]}}
    assert rt.pooled_f1(split["rag-25"], ctx["ids"]) == Fraction(2, 85) / (Fraction(2, 85) + 102)


def invented_run(ctx):
    """Stored records and scores for every arm, variable and repetition, with steady differences.

    Args: ctx: from `load`.
    Returns: (calls, scores).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    rng = random.Random(6)
    calls, scores = {}, {}
    for repetition in range(1, rt.REPETITIONS + 1):
        for variable_id in ctx["ids"]:
            answers = {arm: json.dumps({"n": rng.random()}) for arm in rt.ARMS}
            answers["rag-25+retry"] = answers["rag-25"]
            for position, arm in enumerate(rt.ARMS):
                key = (arm, repetition, variable_id)
                right = sum(rng.random() < 0.4 + 0.02 * position for _ in range(6))
                calls[key] = {"arm": arm, "repetition": repetition, "variable_id": variable_id, "ok": True,
                              "started_at": f"{repetition}/{variable_id}", "answer": answers[arm], "calls": 1,
                              "prompt_tokens": 5000, "cost_usd": 0.0003,
                              "unit_cost_usd": 0.003 if arm == "base" else None,
                              "draft_sha256": ft.sha256(answers["base"]) if arm == "rag-25-labels" else None,
                              "first_sha256": ft.sha256(answers["rag-25"]) if arm == "rag-25+retry" else None,
                              "repaired": arm == "rag-25+retry" and rng.random() < 0.02, "repair_valid": None}
                scores[key] = {"key": list(key), "valid_prediction": rng.random() > 0.02, "prediction": None,
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


def test_the_summary(ctx, monkeypatch):
    """Arms, the setting chosen by fold, the family of two, the descriptive comparisons, integrity."""
    monkeypatch.setattr(rt, "_aggregate", totals_f1)
    calls, scores = invented_run(ctx)
    summary = rt.summarise(ctx, calls, scores, draws=200)
    assert summary["complete"] and summary["unit_integrity"]["mismatched"] == 0
    assert list(summary["arms"]) == list(rt.ARMS)
    assert set(summary["comparisons"]) == {"chosen vs base", "chosen vs rag-25"} and rt.STRICT == 0.025
    for row in summary["comparisons"].values():
        assert row["strict_low"] <= row["low"] <= row["high"] <= row["strict_high"]
    assert set(summary["against_base"]) == set(rt.CONFIGS)
    assert set(summary["against_reference"]) == set(rt.CONFIGS) - {"rag-25"}
    assert set(summary["without_relatives"]) == {"chosen", *rt.CONFIGS}
    assert all(row["variables"] == 81 for row in summary["without_relatives"].values())
    choice = summary["chosen"]["by_fold"]
    pick = {v: choice[str(ctx["fold_of"][v])] for v in ctx["ids"]}
    every = list(range(1, 11))
    mine = [totals_f1([scores[(pick[v], r, v)]["evaluation"] for v in ctx["ids"]])["f1"] for r in every]
    assert summary["chosen"]["f1_per_repetition"] == pytest.approx(mine)
    assert summary["spent_usd"] == pytest.approx(1020 * 0.003)
    torn = dict(calls)
    torn[("rag-85", 2, ctx["ids"][4])] = dict(calls[("rag-85", 2, ctx["ids"][4])], started_at="another write")
    mixed = rt.summarise(ctx, torn, scores, draws=50)
    assert mixed["unit_integrity"]["mismatched"] == 1 and not mixed["complete"]
    other = dict(calls)
    other[("rag-25-labels", 5, ctx["ids"][1])] = dict(calls[("rag-25-labels", 5, ctx["ids"][1])], draft_sha256="x")
    assert rt.summarise(ctx, other, scores, draws=50)["unit_integrity"]["examples"][0]["problems"] == [
        "rag-25-labels: not its own draft"]
    gap = dict(calls)
    gap[("base", 9, ctx["ids"][0])] = dict(calls[("base", 9, ctx["ids"][0])], ok=False)
    assert 9 not in rt.summarise(ctx, gap, scores, draws=50)["repetitions_used"]


def test_the_dev_world_keeps_to_the_invented_variables(ctx):
    """In the dev check the pool is the other half of the invented variables; today's examples stay."""
    world, order = rt.dev_world(ctx, overlap)
    assert len(order) == 16
    for position, name in enumerate(order):
        other = {u for i, u in enumerate(order) if i % 2 != position % 2}
        assert set(world["pool"](name)) == other
        assert world["fixed"](name) == [r["variable_id"] for r in ctx["inputs"]["examples"]]
        for arm in rt.CONFIGS:
            if arm != "rag-25-labels":
                ids = rt.examples(world, arm, name)
                assert all(world["record"](u)["definition"] for u in ids)


def test_the_fixed_prompts_of_base_are_those_of_earlier_stages(ctx):
    """The digest of today's 102 prompts is the one stages 1 and 4 recorded."""
    assert rt.fingerprints(ctx)["prompts"]["base"]["sha256"] == "78fcde54455ecd22"
