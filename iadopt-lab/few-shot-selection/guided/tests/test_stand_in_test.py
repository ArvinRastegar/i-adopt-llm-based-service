"""Offline tests for stand_in_test.py: the categories, the two new prompts, their reading and the arms."""

from __future__ import annotations

import answer_form_check as afc
import constraint_types_test as cy
import engine
import knowledge as kn
import pytest
import rag_tuning as rt
import stand_in_test as si


@pytest.fixture(scope="module")
def setup():
    """The context (without stored calls), stage 5's world with a word-overlap table, and facts.

    Args: none.
    Returns: (ctx, world, found): found holds the experts' patterns as the finder's facts.
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
    ctx["categories"] = si.categories()
    ctx["definitions"] = {q["name"]: q["definition"] for q in ctx["constraint_questions"]}
    base = kn.load()
    world = rt.corpus_world(rt.load(table=rt.similarity_table(base["ids"], lambda v: base["records"][v]["definition"], overlap)), None)
    found = {v: kn.walk.facts_from_direct(ctx, ctx["demos"][v]["patterns"]) for v in ctx["ids"]}
    return ctx, world, found


def test_categories_cover_the_completed_table(setup):
    """Fifteen categories; every one of the 44 type names is in exactly one of them."""
    ctx, _, _ = setup
    cats = ctx["categories"]
    names = [q["name"] for q in ctx["constraint_questions"]]
    listed = [t for c in cats for t in c["types"]]
    assert len(cats) == 15 and len(names) == 44
    assert sorted(listed) == sorted(names)
    assert len({c["name"] for c in cats}) == 15 and all(c["name"] == c["name"].strip().lower() for c in cats)
    assert si.NOT_A_RESTRICTION not in {c["name"] for c in cats}


def test_stand_in_prompt_shows_the_two_screens_in_plain_words(setup):
    """The phrases, every kind with its help text, no proposed kinds and no I-ADOPT wording."""
    ctx, _, _ = setup
    text = si.stand_in_prompt("Dissolved nitrate in river water, per litre.", ["dissolved", "per litre"], ctx["categories"])
    assert '"Dissolved nitrate in river water, per litre."' in text
    assert '- "dissolved"' in text and '- "per litre"' in text and si.SCREEN_1 in text and si.SCREEN_2 in text
    for c in ctx["categories"]:
        assert f"- {c['name']}: {c['help']}" in text
    assert f"- {si.NOT_A_RESTRICTION}:" in text
    assert "i-adopt" not in text.lower() and "constraint" not in text.lower()
    empty = si.stand_in_prompt("Air temperature.", [], ctx["categories"])
    assert si.SCREEN_1_NONE in empty and si.SCREEN_1 not in empty and si.SCREEN_2 in empty


def test_read_answer():
    """Kinds are trimmed and lower-cased; null lists are empty; anything else is unusable."""
    answer = '{"phrases": [{"phrase": " dissolved ", "kind": " State or Form "}, {"kind": "cause"}, "x"], "added": null}'
    assert si.read_answer(answer) == [{"phrase": "dissolved", "kind": "state or form", "source": "listed"}]
    fenced = '```json\n{"added": [{"phrase": "at night", "kind": "time or period"}]}\n```'
    assert si.read_answer(fenced) == [{"phrase": "at night", "kind": "time or period", "source": "added"}]
    assert si.read_answer('{"phrases": [], "added": []}') == []
    for bad in (None, "", "no JSON here", '{"answer": 1}', '{"phrases": "dissolved"}', "[1, 2]"):
        assert si.read_answer(bad) is None


def test_read_type():
    """The whole answer first, then the longest type name it contains; else nothing."""
    place = {"types": ["location", "region", "vicinity", "spatial extent", "spatial distribution"]}
    towards = {"types": ["reference", "direction"]}
    assert si.read_type("reference", towards) == "reference"
    assert si.read_type('"Direction".', towards) == "direction"
    assert si.read_type("**reference**", towards) == "reference"
    assert si.read_type("It is a direction.", towards) == "direction"
    assert si.read_type("spatial extent", place) == "spatial extent"
    assert si.read_type("region or location", place) == "location"
    assert si.read_type("spatial", place) is None
    assert si.read_type("directional", towards) is None
    assert si.read_type(None, towards) is None


def test_types_from_answers():
    """Single-type kinds need no call; not a restriction and unknown kinds add nothing; screen 1 alone skips screen 2."""
    cats = si.categories()
    answers = {"a": [{"phrase": "dissolved", "kind": "state or form", "source": "listed"},
                     {"phrase": "river", "kind": si.NOT_A_RESTRICTION, "source": "listed"},
                     {"phrase": "odd", "kind": "colour", "source": "added"},
                     {"phrase": "per litre", "kind": "rate (per something)", "source": "listed"},
                     {"phrase": "at night", "kind": "time or period", "source": "added"},
                     {"phrase": "by a laboratory", "kind": "who did it, or what for", "source": "added"}],
               "b": None, "c": []}
    jobs = si.typing_jobs(cats, answers)
    assert list(jobs) == ["a#0", "a#4", "a#5"] and jobs["a#0"]["category"]["name"] == "state or form"
    typed = {"a#0": "state", "a#4": "time", "a#5": None}
    assert si.stand_in_types(cats, answers, typed) == {"a": ["normalization", "state", "time"], "b": None, "c": []}
    assert si.stand_in_types(cats, answers, typed, ("listed",)) == {"a": ["normalization", "state"], "b": None, "c": []}
    assert si.stand_in_types(cats, answers, {"a#0": None})["a"] == ["normalization"]
    assert si.STAND_IN_ARMS == {"best+stand-in": ("listed", "added"), "best+screen-1": ("listed",)}


def test_type_prompt_lists_only_the_category(setup):
    """The typing prompt offers the category's types with their definitions, no others."""
    ctx, _, _ = setup
    towards = next(c for c in ctx["categories"] if c["name"] == "measured from or towards")
    text = si.type_prompt("Wind speed 10 m above ground.", "10 m above ground", towards, ctx["definitions"])
    assert f"- reference: {ctx['definitions']['reference']}" in text and f"- direction: {ctx['definitions']['direction']}" in text
    assert text.count("\n- ") == 2


def test_arms(setup):
    """`best` and `best+experts` are the constraint-types test's; the stand-in's exact gold types give `best+experts`."""
    ctx, world, found = setup
    settings = {"flagged": [], "confident": {v: [] for v in ctx["ids"]}}
    none = {arm: {} for arm in si.STAND_IN_ARMS}
    for v in ctx["ids"]:
        control = cy.prompt2(ctx, world, ("best", 1, v), found, settings)
        experts = cy.prompt2(ctx, world, ("best+experts", 1, v), found, settings)
        gold = sorted(cy.gold_types(ctx, v))
        assert si.prompt2(ctx, world, ("best", 1, v), found, none) == control
        assert si.prompt2(ctx, world, ("best+experts", 1, v), found, none) == experts
        for arm in si.STAND_IN_ARMS:
            other = next(a for a in si.STAND_IN_ARMS if a != arm)
            assert si.prompt2(ctx, world, (arm, 1, v), found, {arm: {v: None}, other: {v: gold}}) == control
            assert si.prompt2(ctx, world, (arm, 1, v), found, {arm: {v: gold}, other: {v: None}}) == experts
    v = next(u for u in ctx["ids"] if ctx["key"][u]["constraint"])
    best = cy.prompt2(ctx, world, ("best", 1, v), found, settings)
    assert si.prompt2(ctx, world, ("best+screen-1", 1, v), found, {"best+screen-1": {v: []}, "best+stand-in": {}}) != best
    assert si.prompt2(ctx, world, ("best+stand-in", 1, v), {}, {"best+stand-in": {v: ["state"]}}) == cy.prompt2(
        ctx, world, ("best", 1, v), {}, settings)
    with pytest.raises(KeyError):
        si.prompt2(ctx, world, ("best+person", 1, v), found, none)


def test_guard_stops_after_failures_in_a_row():
    """Ten failed calls in a row stop the run; a success in between resets the count."""
    guard = si.Guard()
    for _ in range(engine.EARLY_WINDOW + 1):
        guard.record({"ok": True, "status": 200, "prompt_tokens": 1, "completion_tokens": 1})
    for _ in range(si.FAILURES_IN_A_ROW - 1):
        guard.record({"ok": False, "status": 429})
    guard.record({"ok": True, "status": 200, "prompt_tokens": 1, "completion_tokens": 1})
    for _ in range(si.FAILURES_IN_A_ROW - 1):
        guard.record({"ok": False, "status": 429})
    assert guard.reason is None
    guard.record({"ok": False, "status": 429})
    assert guard.reason and "HTTP 429" in guard.reason
