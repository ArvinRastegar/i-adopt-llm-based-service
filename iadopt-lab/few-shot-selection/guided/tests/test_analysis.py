"""Offline tests for analysis.py: the nearest-neighbour vote and the sign test. No model, no provider."""

from __future__ import annotations

import math

import analysis
import pytest


def test_a_target_is_predicted_from_its_most_similar_other_items():
    """The target never votes for itself; the k most similar others decide by majority."""
    scores = [[1.0, 0.9, 0.8, 0.1], [0.9, 1.0, 0.7, 0.2], [0.8, 0.7, 1.0, 0.3], [0.1, 0.2, 0.3, 1.0]]
    labels = [True, True, False, False]
    assert analysis.neighbour_votes(scores, labels, [0, 1, 2, 3], 1) == 3   # item 2's nearest is item 0
    assert analysis.neighbour_votes(scores, labels, [0, 1, 2, 3], 3) == 0   # each is outvoted by the rest
    assert analysis.neighbour_votes(scores, labels, [3], 1) == 1


def test_equal_similarities_are_ordered_by_index():
    """A tie in similarity goes to the lower index, so the result does not depend on sort internals."""
    scores = [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]]
    assert analysis.neighbour_votes(scores, [False, True, False], [0], 1) == 0
    assert analysis.neighbour_votes(scores, [False, False, True], [0], 1) == 1


def test_an_even_or_oversized_vote_is_refused():
    """An even k could tie and a k as large as the pool has no neighbours left to leave out."""
    scores = [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]
    for k in (2, 3):
        with pytest.raises(ValueError):
            analysis.neighbour_votes(scores, [True, False, True], [0], k)


def test_sign_test_is_two_sided_and_ignores_ties():
    """Eleven against three gives the exact two-sided binomial p; nothing moved gives 1."""
    assert analysis.sign_test(0, 0) == 1.0
    assert analysis.sign_test(11, 3) == pytest.approx(2 * sum(math.comb(14, i) for i in range(11, 15)) / 2 ** 14)
    assert analysis.sign_test(3, 11) == analysis.sign_test(11, 3)


def _row(tp: int, fp: int, fn: int) -> dict:
    """A stored score row holding only the Close totals the views read.

    Args: tp, fp, fn: whole counts.
    Returns: the row.
    Raises: nothing.
    Side Effects: none.
    """
    return {"evaluation": {"close": {"totals": {k: {"numerator": n, "denominator": 1}
                                                for k, n in (("tp", tp), ("fp", fp), ("fn", fn))}}}}


def test_resampling_variables_separates_a_general_gain_from_one_carried_by_a_few():
    """An arm better on every variable keeps a positive interval; one better on a single variable does not."""
    scores = {}
    for i in range(20):
        scores[("base", 1, f"v{i}")] = _row(5, 5, 5)
        scores[("everywhere", 1, f"v{i}")] = _row(7, 3, 3)
        scores[("one", 1, f"v{i}")] = _row(40, 0, 0) if i == 0 else _row(4, 6, 6)
        scores[("same", 1, f"v{i}")] = _row(5, 5, 5)
    everywhere = analysis.bootstrap_over_variables(scores, "everywhere", "base", draws=500)
    assert everywhere["gain"] == pytest.approx(0.2) and everywhere["low"] > 0 and everywhere["share_positive"] == 1.0
    one = analysis.bootstrap_over_variables(scores, "one", "base", draws=500)
    assert one["gain"] > 0 and one["low"] < 0 < one["high"]
    same = analysis.bootstrap_over_variables(scores, "same", "base", draws=500)
    assert same["gain"] == 0 and same["low"] == 0 and same["high"] == 0
    assert analysis.bootstrap_over_variables(scores, "one", "base", draws=500) == one     # seeded


def test_counts_are_summed_over_repetitions_before_the_f1():
    """Two repetitions of one variable add up; the pooled gain is a difference of micro F1, not a mean."""
    scores = {("base", 1, "a"): _row(1, 1, 1), ("base", 2, "a"): _row(3, 1, 1), ("arm", 1, "a"): _row(4, 0, 0),
              ("arm", 2, "a"): _row(4, 0, 0), ("base", 1, "b"): _row(0, 2, 2), ("arm", 1, "b"): _row(0, 2, 2)}
    counts = analysis.variable_counts(scores)
    assert counts["base"]["a"] == [4.0, 2.0, 2.0] and counts["arm"]["a"] == [8.0, 0.0, 0.0]
    assert analysis.pooled_gain(counts["arm"], counts["base"], ["a", "b"]) == pytest.approx(16 / 20 - 8 / 16)


def test_pooling_groups_keeps_each_setup_with_its_own_baseline():
    """Groups are joined variable by variable; an overlap or a one-sided variable is refused."""
    one = ({"a": [8.0, 0.0, 0.0], "b": [4.0, 2.0, 2.0]}, {"a": [4.0, 2.0, 2.0], "b": [4.0, 2.0, 2.0]})
    two = ({"c": [0.0, 4.0, 4.0]}, {"c": [2.0, 2.0, 2.0]})
    view = analysis.pooled_bootstrap([one, two], draws=200)
    assert view["variables"] == 3
    assert view["gain"] == pytest.approx(24 / 36 - 20 / 32) and view["low"] <= view["gain"] <= view["high"]
    with pytest.raises(ValueError):
        analysis.pooled_bootstrap([one, one], draws=10)
    with pytest.raises(ValueError):
        analysis.pooled_bootstrap([({"a": [1.0, 0.0, 0.0]}, {"z": [1.0, 0.0, 0.0]})], draws=10)

