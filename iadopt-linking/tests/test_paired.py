"""Paired comparison: exact tests and the cluster bootstrap, on cases small enough to check by hand."""

from __future__ import annotations

from fractions import Fraction

import pytest

from iadopt_linking_eval.paired import (
    cluster_bootstrap_difference,
    mcnemar_exact,
    paired_counts,
    term_level,
)


def test_counts_cross_tabulate_two_systems() -> None:
    counts = paired_counts([True, True, False, False, True], [True, False, True, False, False])
    assert counts == {"n": 5, "both": 1, "only_first": 2, "only_second": 1, "neither": 1}


def test_columns_of_different_length_are_refused() -> None:
    with pytest.raises(ValueError, match="different numbers"):
        paired_counts([True], [True, False])


@pytest.mark.parametrize(("only_first", "only_second", "expected"), [
    (5, 0, Fraction(1, 16)),                 # 2 * (1 / 32)
    (3, 3, Fraction(1)),                     # the tails cover the whole distribution
    (10, 2, Fraction(79, 2048)),             # 2 * (1 + 12 + 66) / 4096, the usual exact value
    (2, 10, Fraction(79, 2048)),             # symmetric
    (0, 0, Fraction(1)),                     # never disagreeing is no evidence of a difference
])
def test_mcnemar_exact_matches_the_binomial_tail(only_first: int, only_second: int,
                                                 expected: Fraction) -> None:
    assert mcnemar_exact(only_first, only_second) == expected


def test_mcnemar_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="negative"):
        mcnemar_exact(-1, 2)


def test_identical_systems_have_a_zero_difference_and_a_degenerate_interval() -> None:
    outcome = [True, False, True, True, False, False]
    result = cluster_bootstrap_difference(outcome, outcome, list("abcdef"), resamples=200, seed=1)
    assert result["difference"] == {"numerator": 0, "denominator": 1}
    assert result["interval"]["lower"] == result["interval"]["upper"] == 0.0


def test_a_clear_winner_has_an_interval_above_zero() -> None:
    first = [True] * 40
    second = [False] * 40
    result = cluster_bootstrap_difference(first, second, [str(i) for i in range(40)],
                                          resamples=500, seed=3)
    assert result["difference"] == {"numerator": 1, "denominator": 1}
    assert result["interval"]["lower"] == 1.0 and result["clusters"] == 40


def test_the_difference_is_exact_and_signed() -> None:
    result = cluster_bootstrap_difference([True, False, False], [True, True, True],
                                          ["a", "b", "c"], resamples=50, seed=0)
    assert result["difference"] == {"numerator": -2, "denominator": 3}
    assert (result["first_correct"], result["second_correct"]) == (1, 3)


def test_slots_that_share_a_cluster_widen_the_interval() -> None:
    # Ten slots favour the first system; ten are ties. If the ten sit in one cluster, a resample
    # can miss them entirely or draw them twice, so the interval must be wider than when each slot
    # is its own cluster.
    first = [True] * 10 + [True] * 10
    second = [False] * 10 + [True] * 10
    independent = cluster_bootstrap_difference(first, second, [str(i) for i in range(20)],
                                               resamples=2000, seed=7)["interval"]
    grouped = cluster_bootstrap_difference(first, second, ["win"] * 10 + ["tie"] * 10,
                                           resamples=2000, seed=7)["interval"]
    assert grouped["upper"] - grouped["lower"] > independent["upper"] - independent["lower"]


def test_the_bootstrap_is_reproducible_from_its_seed() -> None:
    first = [True, False, True, False, True, True, False, True]
    second = [False, False, True, True, True, False, False, True]
    clusters = list("aabbccdd")
    a = cluster_bootstrap_difference(first, second, clusters, resamples=300, seed=11)
    b = cluster_bootstrap_difference(first, second, clusters, resamples=300, seed=11)
    assert a == b
    assert a["interval"]["method"] == "percentile bootstrap over clusters"


@pytest.mark.parametrize("arguments", [
    ([], [], [], {}),
    ([True], [True], ["a"], {"level": Fraction(0)}),
    ([True], [True], ["a"], {"level": Fraction(1)}),
    ([True], [True], ["a"], {"resamples": 0}),
    ([True, False], [True], ["a", "b"], {}),
])
def test_unusable_requests_are_refused(arguments: tuple) -> None:
    first, second, clusters, options = arguments
    with pytest.raises(ValueError):
        cluster_bootstrap_difference(first, second, clusters, **options)


def test_a_term_asked_many_times_counts_once_at_the_term_level() -> None:
    # "water" is asked six times and the first system wins all six; four other terms are asked once
    # each and the second system wins three of them. Per slot the first system leads 7 to 3; per
    # term it wins 2 and loses 3.
    clusters = ["water"] * 6 + ["a", "b", "c", "d"]
    first = [True] * 6 + [True, False, False, False]
    second = [False] * 6 + [False, True, True, True]
    result = term_level(first, second, clusters)
    assert (result["terms"], result["first_better"], result["second_better"], result["equal"]) == (
        5, 2, 3, 0)
    assert result["sign_test_p"] == {"numerator": 1, "denominator": 1}
    # per-term differences are +1 (water), +1, -1, -1, -1, so the unweighted mean is -1/5
    assert result["mean_difference"] == {"numerator": -1, "denominator": 5}


def test_the_term_level_difference_is_an_unweighted_mean_over_terms() -> None:
    result = term_level([True, True, False], [False, False, False], ["x", "x", "y"])
    # term x: 1.0 - 0.0; term y: 0.0 - 0.0; the mean over two terms is 1/2
    assert result["mean_difference"] == {"numerator": 1, "denominator": 2}
    assert (result["first_better"], result["second_better"], result["equal"]) == (1, 0, 1)


def test_the_term_level_comparison_needs_slots() -> None:
    with pytest.raises(ValueError):
        term_level([], [], [])
