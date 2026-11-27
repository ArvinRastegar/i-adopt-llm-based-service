"""Offline tests for answer_form_check.py: the two answer forms and the Part B prompts."""

from __future__ import annotations

import json

import answer_form_check as af
import conventions_test as ct
import knowledge as kn
import pytest
import rag_tuning as rt


@pytest.fixture(scope="module")
def ctx():
    """Part A's context.

    Args: none.
    Returns: the mapping from `answer_form_check.load`.
    Raises: as the loaders.
    Side Effects: reads local files.
    """
    return af.load()


def test_the_two_forms_differ_only_in_the_answer(ctx):
    """Five keys against the named-first answer; same examples, decompositions, order and completed table."""
    v = ctx["ids"][3]
    five, named = (af.prompt_a(ctx, (arm, 1, v)) for arm in af.ARMS)
    assert ct.INSTRUCTION in named and ct.INSTRUCTION not in five
    assert ct.INSTRUCTION_START in five.split("\n")[1]
    assert "volume:" in five and "thickness:" in five
    assert named == ct.prompt1(ctx, ("plain", 1, v))
    assert five == kn.build(ctx, "D+S", ctx["records"][v]["definition"],
                            [ctx["five_demos"][u] for u in ctx["ids"] if ctx["fold_of"][u] != ctx["fold_of"][v]])
    five_answer = json.loads(five.split("Patterns: ")[1].split("\n")[0])
    assert set(five_answer) == {"property", "matrix", "context", "modifier", "constraints"}
    assert five.count("Decomposition: ") == named.count("Decomposition: ") == 85
    with pytest.raises(KeyError):
        af.build(ctx, "other", "x", 0)


def test_gate_of_part_b():
    """The finder's matrix arm runs only above the break-even accuracy."""
    assert af.arms_b({"finder_arm_runs": True}) == af.PART_B
    assert af.arms_b({"finder_arm_runs": False}) == ("rag-25", "rag-25+experts-matrix")


@pytest.fixture(scope="module")
def world():
    """Stage 5's world with a word-overlap similarity table (no stored output needed).

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


def test_part_b_prompts_state_only_the_matrix(ctx, world):
    """The control is stage 5's prompt; the guided arms add a matrix statement and nothing about constraints."""
    v = next(u for u in ctx["ids"] if ctx["key"][u]["matrix"] and ctx["key"][u]["constraint"])
    control = af.prompt_b(ctx, world, ("rag-25", 1, v), {})
    assert control == rt.prompt_of(world, v, rt.examples(world, "rag-25", v))
    experts = af.prompt_b(ctx, world, ("rag-25+experts-matrix", 1, v), {})
    both = ct.prompt2(ctx, world, ("rag-25+experts", 1, v), {}, [])
    assert experts != control and experts != both and len(control) < len(experts) < len(both)
    assert af.prompt_b(ctx, world, ("rag-25+finder-matrix", 1, v), {v: None}) == control
    facts = ct.hints.facts_from_key(ctx["key"][v])
    assert af.prompt_b(ctx, world, ("rag-25+finder-matrix", 1, v), {v: facts}) == experts
    with pytest.raises(KeyError):
        af.prompt_b(ctx, world, ("other", 1, v), {})
