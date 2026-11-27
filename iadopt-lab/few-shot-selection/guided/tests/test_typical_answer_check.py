"""Offline tests for typical_answer_check.py: the picking rule, the F1 helper and the comparison."""

from __future__ import annotations

import typical_answer_check as ta


def totals(tp, fp, fn):
    """Scorer-style totals.

    Args: tp, fp, fn: integer counts.
    Returns: the totals as numerator/denominator.
    Raises: nothing.
    Side Effects: none.
    """
    return {n: {"numerator": x, "denominator": 1} for n, x in (("tp", tp), ("fp", fp), ("fn", fn))}


def test_pick_takes_the_most_agreeing_run_and_breaks_ties_low():
    """Highest mean agreement wins; ties go to the lowest run number."""
    assert ta.pick((1, 2, 3), {(1, 2): 0.9, (1, 3): 0.5, (2, 3): 0.8}) == 2
    assert ta.pick((1, 2, 3), {(1, 2): 0.5, (1, 3): 0.5, (2, 3): 0.5}) == 1
    assert ta.pick((4, 5, 6), {(4, 5): 0.0, (4, 6): 0.0, (5, 6): 0.7}) == 5


def test_f1():
    """Close F1 from totals; zero without a true positive."""
    assert ta.f1(totals(2, 1, 1)) == 2 * 2 / (2 * 2 + 1 + 1)
    assert ta.f1(totals(0, 3, 3)) == 0.0


def test_compare_keeps_the_typical_answer(monkeypatch):
    """With groups of three, the kept answer's counts replace the group's mean counts."""
    monkeypatch.setattr(ta, "GROUPS", {3: ((1, 2, 3),)})
    monkeypatch.setattr(ta, "DRAWS", 50)
    rows = {1: (1, 1, 1), 2: (3, 0, 0), 3: (2, 1, 0)}
    scores = {("arm", r, "v"): {"evaluation": {"close": {"totals": totals(*c)}}} for r, c in rows.items()}
    agree = {"v": {"1-2": 0.8, "1-3": 0.2, "2-3": 0.9}}         # means: run 1 0.5, run 2 0.85, run 3 0.55
    out = ta.compare(scores, "arm", ["v"], agree, 3)
    assert out["kept_runs"] == {"v": [2]}
    assert out["kept_f1"] == 1.0
    mean = [2.0, 2 / 3, 1 / 3]
    assert abs(out["one_answer_f1"] - 2 * mean[0] / (2 * mean[0] + mean[1] + mean[2])) < 1e-12
