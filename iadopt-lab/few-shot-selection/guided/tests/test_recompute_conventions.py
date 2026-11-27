"""Offline tests: recompute_conventions.py marks answers as conventions_test.py does."""

from __future__ import annotations

import json
import random

import conventions_test as ct
import pytest
import recompute_conventions as rc

COLUMNS = ("matrix", "constraint", "matrix_pattern", "both", "parsed", "form", "tp", "fp", "fn")


@pytest.fixture(scope="module")
def ctx():
    """Part 1's context without fresh entries.

    Args: none.
    Returns: the mapping from `conventions_test.load`.
    Raises: as the loaders.
    Side Effects: reads local files.
    """
    return ct.load(fresh=False)


def answers(ctx, variable_id: str, rng: random.Random) -> list[str]:
    """The experts' own answer for a variable and variations of it, some broken on purpose.

    Args: ctx: from `load`; variable_id: the variable; rng: the random source.
    Returns: answer texts.
    Raises: nothing.
    Side Effects: draws from `rng`.
    """
    right = json.loads(ctx["demos"][variable_id]["patterns"])
    names = sorted(q["name"] for q in ctx["constraint_questions"])
    out = [json.dumps(right), "```json\n" + json.dumps(right) + "\n```", "no JSON", "{\"property\": 1}"]
    for _ in range(6):
        wrong = dict(right)
        wrong["matrix"] = rng.choice([None, 40, 41, 42, 43, 44, 45])
        wrong["constraints"] = rng.sample(names, rng.randint(0, 3)) + (["not a type"] if rng.random() < 0.2 else [])
        if rng.random() < 0.2:
            wrong.pop("restrictions")
        if rng.random() < 0.1:
            wrong["property"] = 9999
        out.append(json.dumps(wrong))
    return out


def test_both_mark_alike(ctx):
    """Every column of every answer, for all 102 variables."""
    rng = random.Random(3)
    by_number, by_name = rc.table()
    patterns = json.loads((rc.DATA / "patterns.json").read_text(encoding="utf-8"))
    kinds = {number: row.get("kind") for number, row in patterns.get("patterns", patterns).items()}
    for v in ctx["ids"]:
        for answer in answers(ctx, v, rng):
            mine = ct.outcome(ctx, ctx["key"][v], answer)
            theirs = rc.mark(ctx["key"][v], answer, kinds, by_number, by_name)
            assert {c: mine[c] for c in COLUMNS} == {c: theirs[c] for c in COLUMNS}, (v, answer)


def test_both_read_the_same_table(ctx):
    """The completed table: the same names for the same numbers."""
    by_number, _ = rc.table()
    assert by_number == {row["number"]: row["name"].lower() for row in ctx["types"]}
