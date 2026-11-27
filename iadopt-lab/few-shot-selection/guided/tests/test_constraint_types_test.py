"""Offline tests for constraint_types_test.py: Step 1's rules and Step 2's prompts."""

from __future__ import annotations

import answer_form_check as afc
import constraint_types_test as cy
import knowledge as kn
import pytest
import rag_tuning as rt


@pytest.fixture(scope="module")
def ctx():
    """The answer-form check's context.

    Args: none.
    Returns: the mapping from `answer_form_check.load`.
    Raises: as the loaders.
    Side Effects: reads local files.
    """
    return afc.load()


def test_pool_types_by_prefix_or_the_one_left(ctx):
    """A label's own prefix when it is a type of its variable, else the variable's one unused type."""
    entries = cy.pool(ctx)
    assert entries and all(e["type"] in cy.gold_types(ctx, e["variable"]) for e in entries)
    by_label = {(e["variable"], e["position"]): e for e in entries}
    for v in ctx["ids"]:
        types = cy.gold_types(ctx, v)
        for position, c in enumerate(ctx["records"][v]["gold"]["hasConstraint"]):
            prefix = c["label"].split(":", 1)[0].strip().lower()
            if prefix in types:
                assert by_label[(v, position)]["type"] == prefix


def test_nearest_takes_the_most_similar_and_the_first_on_ties():
    """Pool order breaks ties; the similarity is the scorer's policy (equal texts are 1.0)."""
    pool = [{"variable": "a", "position": 0, "value": "dissolved", "type": "state"},
            {"variable": "b", "position": 0, "value": "Dissolved ", "type": "condition"},
            {"variable": "c", "position": 1, "value": "per hour", "type": "normalization"}]
    score = cy.cosine(lambda a, b: 0.1)
    assert cy.nearest("dissolved", pool, score)["type"] == "state"
    assert cy.nearest("per minute", pool, cy.cosine(lambda a, b: 0.9 if "per" in a else 0.2))["type"] == "normalization"
    with pytest.raises(ValueError):
        cy.nearest("x", [], score)


def test_precision_recall_and_majority():
    """S1a's counts and the most frequent set with ties to the earliest."""
    k = {("v1", "state"): 10, ("v1", "part"): 3, ("v2", "cause"): 6}
    gold = {"v1": frozenset({"state"}), "v2": frozenset({"cause", "event"})}
    row = cy.precision_recall(k, gold, ["v1", "v2"], 5)
    assert (row["named"], row["right"], row["precision"], row["recall"]) == (2, 2, 1.0, 2 / 3)
    assert cy.precision_recall(k, gold, ["v1", "v2"], 11)["precision"] is None
    a, b = frozenset({"x"}), frozenset({"y"})
    assert cy.majority([a, b, b, a]) == a and cy.majority([b, a, a]) == a


def test_signal_rule():
    """Best accuracy among signals flagging at most half; else the one flagging fewest."""
    rows = {"R": {"flagged": 0.6, "accuracy": 0.9}, "N": {"flagged": 0.4, "accuracy": 0.7},
            "R or N": {"flagged": 0.5, "accuracy": 0.8}}
    assert cy.choose_signal(rows) == "R or N"
    rows["R or N"]["flagged"] = 0.7
    assert cy.choose_signal(rows) == "N"
    rows["N"]["flagged"] = 0.8
    assert cy.choose_signal(rows) == "R"


@pytest.fixture(scope="module")
def world():
    """Stage 5's world with a word-overlap similarity table.

    Args: none.
    Returns: the world.
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

    base = kn.load()
    return rt.corpus_world(rt.load(table=rt.similarity_table(base["ids"], lambda v: base["records"][v]["definition"], overlap)), None)


def test_step2_prompts(ctx, world):
    """Control as the answer-form check; confident types as "at least"; a flagged variable gets the experts' set."""
    v = next(u for u in ctx["ids"] if ctx["key"][u]["constraint"] and ctx["key"][u]["matrix"])
    w = next(u for u in ctx["ids"] if u != v and ctx["key"][u]["constraint"])
    found = {u: kn.walk.facts_from_direct(ctx, ctx["demos"][u]["patterns"]) for u in (v, w)}
    names = sorted(cy.gold_types(ctx, v))
    settings = {"confident": {v: names[:1], w: []}, "flagged": [v]}
    control = cy.prompt2(ctx, world, ("best", 1, v), found, settings)
    assert control == afc.prompt_b(ctx, world, ("rag-25+finder-matrix", 1, v), found)
    confident = cy.prompt2(ctx, world, ("best+confident", 1, v), found, settings)
    assert confident != control and "at least" in confident
    experts = cy.prompt2(ctx, world, ("best+experts", 1, v), found, settings)
    assert cy.prompt2(ctx, world, ("best+person", 1, v), found, settings) == experts != confident
    assert cy.prompt2(ctx, world, ("best+confident", 1, w), found, settings) == cy.prompt2(ctx, world, ("best", 1, w), found, settings)
    assert cy.prompt2(ctx, world, ("best+person", 1, w), found, settings) == cy.prompt2(ctx, world, ("best", 1, w), found, settings)
    assert cy.arms_to_run({"confident_arm_runs": False}) == ("best", "best+person", "best+experts")
    with pytest.raises(KeyError):
        cy.prompt2(ctx, world, ("other", 1, v), found, settings)
