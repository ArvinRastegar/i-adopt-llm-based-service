"""Offline tests for decision_value.py: what each arm states, and the comparisons. No provider."""

from __future__ import annotations

import re

import decision_value as dv
import engine
import groups
import hints
import pytest
import scoreboard
from engine import ft

FULL = frozenset({"form", "names", "types", "texts"})
HEADERS = {dv.FORM_HARD, dv.PATTERNS_HARD, dv.FORM_SOFT, dv.PATTERNS_SOFT, dv.SOFT_NOTE, hints.HEADING,
           "Constraint types:", ""}


@pytest.fixture(scope="module")
def ctx():
    """Load the stage's context once per module.

    Args: none.
    Returns: the mapping from `decision_value.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files, among them the stored walks of steps 2 to 5.
    """
    if not (engine.OUTPUT / "route-examples-direct-all.jsonl").exists():
        pytest.skip("the stored walks of steps 2 to 5 are not on this machine (gitignored output)")
    return dv.load()


def lines_of(ctx, arm, variable_id):
    """The lines of one arm's block for one variable, without the headings.

    Args: ctx: from `load`; arm: an arm name; variable_id: the variable.
    Returns: the block's lines that state something, in order.
    Raises: KeyError for an unknown arm.
    Side Effects: none.
    """
    facts = hints.facts_from_key(ctx["key"][variable_id])
    text = dv.block(ctx, arm, facts, ctx["found"].get(variable_id), variable_id, ctx["table"])
    return [line for line in text.split("\n") if line not in HEADERS]


def test_the_25_arms(ctx):
    """One baseline, the key, three arms per decision, and the five wording arms; 25,500 calls."""
    assert len(dv.ARMS) == len(set(dv.ARMS)) == 25 and dv.DECISIONS == tuple(scoreboard.DECISIONS)
    jobs = dv.schedule(ctx)
    assert len(jobs) == len(set(jobs)) == 25 * 102 * dv.REPETITIONS == 25500
    assert {job[0] for job in jobs} == set(dv.ARMS) and {job[2] for job in jobs} == set(ctx["ids"])
    arms = set(dv.ARMS)
    assert all(first in arms and second in arms for _, first, second in dv.contrasts())
    assert len({name for name, _, _ in dv.contrasts()}) == len(dv.contrasts())


def test_all_decisions_stated_is_the_guidance_of_steps_1_to_5(ctx):
    """With everything stated the block is `hints.render` with all features, byte for byte."""
    for variable_id in ctx["ids"]:
        for facts in (hints.facts_from_key(ctx["key"][variable_id]), ctx["found"][variable_id]):
            assert dv.render(facts, dv.ALL, ctx["patterns"], ctx["types"]) == hints.render(
                facts, FULL, ctx["patterns"], ctx["types"])
    assert dv.render(hints.facts_from_key(ctx["key"][ctx["ids"][0]]), set(), ctx["patterns"], ctx["types"]) == ""
    with pytest.raises(ValueError):
        dv.render(ctx["found"][ctx["ids"][0]], {"shape"}, ctx["patterns"], ctx["types"])


def test_base_key_and_walk_send_the_prompts_of_step_5(ctx):
    """`base`, `key` and `walk` are step 5's baseline, key arm and unchecked `direct-all` guidance."""
    keyed = {group: groups.walks_named(ctx, "key", group) for group in scoreboard.GROUPS}
    for variable_id in ctx["ids"]:
        group = ctx["group_of"][variable_id]
        assert dv.prompt_of(ctx, ("base", 1, variable_id)) == groups.prompt_of(ctx, "base", variable_id, {})
        assert dv.prompt_of(ctx, ("key", 1, variable_id)) == groups.prompt_of(
            ctx, groups.KEY_ARM, variable_id, keyed[group])
        own = groups.context_for(ctx, variable_id)
        assert dv.prompt_of(ctx, ("walk", 1, variable_id)) == groups.pipeline.guided_prompt(
            own, ctx["records"][variable_id]["definition"], ctx["found"][variable_id], FULL)


def test_the_six_decisions_share_out_the_lines_of_the_key_block(ctx):
    """Each stated line belongs to exactly one decision; dropping a decision removes its lines only."""
    for variable_id in ctx["ids"]:
        whole = lines_of(ctx, "key", variable_id)
        alone = {d: lines_of(ctx, f"only-{d}", variable_id) for d in dv.DECISIONS}
        assert all(alone.values())                                  # every decision states something
        assert sorted(line for lines in alone.values() for line in lines) == sorted(whole)
        for decision in dv.DECISIONS:
            rest = lines_of(ctx, f"drop-{decision}", variable_id)
            assert rest == [line for line in whole if line not in alone[decision]]
    variable_id = ctx["ids"][0]
    assert lines_of(ctx, "only-property", variable_id)[0].startswith("- Property pattern ")
    assert lines_of(ctx, "only-matrix", variable_id)[0].startswith("- hasMatrix: ")
    assert lines_of(ctx, "only-modifier", variable_id) == [next(
        line for line in lines_of(ctx, "key", variable_id) if line.startswith("- hasStatisticalModifier: "))]


def test_a_wrong_arm_is_wrong_in_its_decision_and_nowhere_else(ctx):
    """Marked against the experts' tags, `wrong-D` fails decision D and passes the other five."""
    for variable_id in ctx["ids"]:
        facts = hints.facts_from_key(ctx["key"][variable_id])
        for decision in dv.DECISIONS:
            wrong = dv.wrong_facts(facts, decision, ctx["table"], ctx["patterns"])
            row = scoreboard.marks(ctx, {variable_id: wrong})[variable_id]
            assert row == {**{d: d != decision for d in dv.DECISIONS}, "complete": False}
            assert lines_of(ctx, f"wrong-{decision}", variable_id) != lines_of(ctx, "key", variable_id)
        assert facts == hints.facts_from_key(ctx["key"][variable_id])       # the right set is untouched
    with pytest.raises(KeyError):
        dv.wrong_facts(facts, "shape", ctx["table"], ctx["patterns"])


def test_the_usual_alternatives(ctx):
    """The wrong values come from the tags and the stored walks, and are never the right pattern."""
    table = ctx["table"]
    assert (table["matrix"], table["context"], table["constraint"], table["object_other"]) == (41, 50, 70, 35)
    kinds = {int(n): pattern["kind"] for n, pattern in ctx["patterns"].items()}
    for right in sorted({entry["property"][0] for entry in ctx["key"].values()}):
        for count in (1, 2, 5):
            given = dv.wrong_properties(right, table, count)
            assert len(given) == len(set(given)) == count and right not in given
            assert all(kinds[n] == "property" for n in given)
        confused = table["confused"].get(right)
        if confused:
            assert dv.wrong_properties(right, table, 1) == [n for n in confused if n != right][:1]
    with pytest.raises(ValueError):
        dv.wrong_properties(101, table, 10 ** 3)


def test_soft_wording_changes_the_headings_only(ctx):
    """A soft block states the same lines under hedged headings and one note."""
    for variable_id in ctx["ids"]:
        for hard, soft in (("key", "soft-key"), ("walk", "soft-walk")):
            assert lines_of(ctx, hard, variable_id) == lines_of(ctx, soft, variable_id)
        facts = hints.facts_from_key(ctx["key"][variable_id])
        text = dv.block(ctx, "soft-key", facts, None, variable_id, ctx["table"])
        assert text.count(dv.SOFT_NOTE) == 1 and dv.FORM_SOFT in text and dv.PATTERNS_SOFT in text
        assert dv.FORM_HARD not in text and dv.PATTERNS_HARD not in text


def test_candidates_hold_the_right_pattern_once_in_a_mixed_order(ctx):
    """The right property pattern is one of two or three listed, not always first, the rest unchanged."""
    first = {2: 0, 3: 0}
    for variable_id in ctx["ids"]:
        right = ctx["key"][variable_id]["property"][0]
        whole = lines_of(ctx, "key", variable_id)
        for count in (2, 3):
            lines = lines_of(ctx, f"cand-property-{count}", variable_id)
            listed = [int(re.match(r"  - Property pattern (\d+):", line).group(1))
                      for line in lines if line.startswith("  - Property pattern ")]
            assert len(listed) == len(set(listed)) == count and listed.count(right) == 1
            assert lines.count(dv.CANDIDATES_LINE) == 1
            assert lines == lines_of(ctx, f"cand-property-{count}", variable_id)      # fixed order
            others = [line for line in lines if not line.startswith("  - ") and line != dv.CANDIDATES_LINE]
            assert others == [line for line in whole if not line.startswith("- Property pattern ")]
            first[count] += listed[0] == right
    assert 30 <= first[2] <= 72 and 18 <= first[3] <= 52          # near a half and a third of 102


def test_an_example_is_never_shown_its_own_answer_in_any_arm(ctx):
    """Every arm's prompt for an example variable holds the other 24 examples only."""
    for variable_id in groups.targets(ctx, "examples"):
        record = ctx["records"][variable_id]
        for arm in dv.ARMS:
            examples = ft.examples_block(dv.prompt_of(ctx, (arm, 1, variable_id)))
            assert len(re.findall(r"(?m)^Example \d+$", examples)) == 24
            assert ft.answer_text(record["gold"]) not in examples
    other = groups.targets(ctx, "held-out")[0]
    assert len(re.findall(r"(?m)^Example \d+$", ft.examples_block(dv.prompt_of(ctx, ("key", 1, other))))) == 25


def test_no_block_names_a_removed_term(ctx):
    """No arm's block holds the two example terms that name a corpus variable."""
    for variable_id in ctx["ids"]:
        facts = hints.facts_from_key(ctx["key"][variable_id])
        for arm in dv.ARMS:
            text = dv.block(ctx, arm, facts, ctx["found"][variable_id], variable_id, ctx["table"]).lower()
            assert "body mass index" not in text and "actinic flux" not in text
    with pytest.raises(KeyError):
        dv.block(ctx, "drop-shape", facts, None, variable_id, ctx["table"])


def test_a_comparison_resamples_the_variables():
    """The pooled gain and its interval over resampled variables, on counts worked out by hand."""
    counts = {"a": {"v1": [8.0, 2.0, 2.0], "v2": [2.0, 8.0, 8.0]},
              "b": {"v1": [5.0, 5.0, 5.0], "v2": [5.0, 5.0, 5.0]}}
    out = dv.gain(counts, "a", "b", ["v1", "v2"], draws=400, seed=3)
    assert out["gain"] == pytest.approx(0.0)                      # both arms: 10 right of 20 and 20
    assert out["low"] == pytest.approx(-0.3) and out["high"] == pytest.approx(0.3)
    assert 0.15 < out["share_positive"] < 0.35                    # a quarter of the draws take v1 twice
    assert dv.gain(counts, "a", "b", ["v1"], draws=50, seed=3)["low"] == pytest.approx(0.3)
    assert dv.gain(counts, "a", "b", ["v1", "v2"], draws=400, seed=3) == out


def test_the_break_even_accuracy():
    """Stating a decision pays once it is right more often than harm / (value + harm)."""
    def arm(f1_tp):
        """Counts of one variable with a chosen number of right items out of ten.

        Args: f1_tp: right items.
        Returns: {variable id: [TP, FP, FN]} for two equal variables.
        Raises: nothing.
        Side Effects: none.
        """
        return {v: [float(f1_tp), 10.0 - f1_tp, 10.0 - f1_tp] for v in ("v1", "v2")}

    counts = {"key": arm(8), "drop-matrix": arm(6), "wrong-matrix": arm(3)}
    out = dv.break_even(counts, "matrix", ["v1", "v2"], draws=50, seed=1)
    assert out["value"] == pytest.approx(0.2) and out["harm"] == pytest.approx(0.3)
    assert out["accuracy"] == pytest.approx(0.6) and out["low"] == pytest.approx(0.6)
    assert out["high"] == pytest.approx(0.6) and out["defined_share"] == 1.0
    flat = dv.break_even({"key": arm(6), "drop-matrix": arm(6), "wrong-matrix": arm(6)}, "matrix", ["v1"], draws=10)
    assert flat["accuracy"] is None and flat["defined_share"] == 0.0 and flat["low"] is None
    helps = dv.break_even({"key": arm(8), "drop-matrix": arm(6), "wrong-matrix": arm(7)}, "matrix", ["v1"], draws=10)
    assert helps["accuracy"] == 0.0                              # even a wrong statement beats silence
