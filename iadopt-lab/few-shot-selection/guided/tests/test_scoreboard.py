"""Offline tests for scoreboard.py: the marks, the folds and the fresh list. No provider."""

from __future__ import annotations

from collections import Counter

import engine
import groups
import hints
import pytest
import rulebook
import scoreboard


@pytest.fixture(scope="module")
def ctx():
    """Load the context once per module.

    Args: none.
    Returns: the mapping from `groups.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files.
    """
    return groups.load()


def test_the_experts_tags_are_complete_against_themselves(ctx):
    """Marked against itself, the answer key is right in every decision for all 102 variables."""
    found = {v: hints.facts_from_key(entry) for v, entry in ctx["key"].items()}
    cells = scoreboard.board(ctx, found)
    assert set(cells) == {"all", *scoreboard.GROUPS}
    assert [cells[name]["complete"]["of"] for name in ("all", *scoreboard.GROUPS)] == [102, 25, 53, 24]
    for row in cells.values():
        assert all(cell["right"] == cell["of"] for cell in row.values())
    assert set(cells["all"]) == {*scoreboard.DECISIONS, "complete"}


def test_one_wrong_decision_is_marked_once_and_spoils_complete(ctx):
    """A found set with a wrong matrix is wrong there and nowhere else, and is not complete."""
    variable_id = next(v for v, entry in sorted(ctx["key"].items()) if entry["matrix"])
    facts = dict(hints.facts_from_key(ctx["key"][variable_id]), matrix=[])
    row = scoreboard.marks(ctx, {variable_id: facts})[variable_id]
    assert row == {"property": True, "object": True, "matrix": False, "context": True, "modifier": True,
                   "constraint": True, "complete": False}
    assert not any(scoreboard.marks(ctx, {variable_id: None})[variable_id].values())


def test_the_interval_of_a_share():
    """The Wilson interval holds the share, narrows with more trials and handles the edges."""
    low, high = scoreboard.wilson(22, 102)
    assert low < 22 / 102 < high and round(low, 3) == 0.147 and round(high, 3) == 0.305
    assert scoreboard.wilson(0, 0) == (0.0, 1.0)
    assert scoreboard.wilson(10, 10)[1] == pytest.approx(1.0) and scoreboard.wilson(0, 10)[0] == pytest.approx(0.0)
    assert scoreboard.wilson(220, 1020)[1] - scoreboard.wilson(220, 1020)[0] < high - low


def test_the_folds_are_six_fixed_equal_parts_of_every_group(ctx):
    """Six parts of 17, each with every group and with matrix and no-matrix variables; fixed."""
    fold_of = scoreboard.folds(ctx)
    assert fold_of == scoreboard.folds(ctx) and len(fold_of) == 102
    assert Counter(fold_of.values()) == {fold: 17 for fold in range(6)}
    by_group = scoreboard.ids_by_group(ctx)
    for fold in range(6):
        members = {v for v, f in fold_of.items() if f == fold}
        assert [len(members & set(by_group[g])) for g in scoreboard.GROUPS] in ([4, 9, 4], [5, 8, 4])
        with_matrix = sum(bool(ctx["key"][v]["matrix"]) for v in members)
        assert 7 <= with_matrix <= 10
    digest = engine.ft.sha256("\n".join(f"{v} {fold_of[v]}" for v in sorted(fold_of)))
    assert digest == "80f37c48d2cf004cc00a1237247c7a89269463b13698bdc3aec7d2723cc4f4e3"
    assert scoreboard.folds(ctx, seed=1) != fold_of


def test_a_target_may_only_use_the_other_parts(ctx):
    """What a learning method may use for a target: the 85 variables of the other five parts."""
    fold_of = scoreboard.folds(ctx)
    for target in sorted(fold_of)[:12]:
        usable = scoreboard.usable_for(target, fold_of)
        assert len(usable) == 85 and target not in usable
        assert all(fold_of[v] != fold_of[target] for v in usable)


def test_the_fresh_entries_are_none_of_the_102(ctx):
    """Every fresh entry has a property tag and an issue that the answer key does not use."""
    rows = scoreboard.fresh(rulebook.load("issues_snapshot")["issues"], ctx["key"])
    used = {entry["issue"] for entry in ctx["key"].values()}
    assert len(rows) == 31 and sum(bool(row["flags"]) for row in rows) == 10
    assert not {row["issue"] for row in rows} & used and all(row["property"] for row in rows)
    assert [row["issue"] for row in rows] == sorted(row["issue"] for row in rows)


def test_the_stored_walks_on_the_scoreboard(ctx):
    """The stored `direct-all` walk has the complete set right for 22 of the 102 variables."""
    if not (engine.OUTPUT / "route-examples-direct-all.jsonl").exists():
        pytest.skip("the stored walks of steps 2 to 5 are not on this machine (gitignored output)")
    cells = scoreboard.board(ctx, scoreboard.stored_walk(ctx, "direct-all"))["all"]
    assert {name: cell["right"] for name, cell in cells.items()} == {
        "property": 75, "object": 98, "matrix": 63, "context": 92, "modifier": 98, "constraint": 38,
        "complete": 22}
