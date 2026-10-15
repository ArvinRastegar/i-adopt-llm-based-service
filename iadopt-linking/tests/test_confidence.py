"""Confidence statistics on tiny cases whose answers can be worked out on paper."""

from __future__ import annotations

from fractions import Fraction

import pytest

from iadopt_linking_eval.confidence import (
    auroc,
    brier_score,
    calibration_gap,
    coverage_precision,
    escalation_table,
    reliability_bins,
)

GRID = (Fraction(0), Fraction(1, 2), Fraction(1))


def test_auroc_is_one_for_perfect_separation_and_zero_for_the_opposite() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0


def test_auroc_counts_ties_as_half_and_ignores_unanswered_slots() -> None:
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == Fraction(1, 2)
    assert auroc([0.9, None, 0.1], [True, True, False]) == 1


def test_auroc_is_undefined_when_only_one_kind_of_answer_exists() -> None:
    assert auroc([0.9, 0.8], [True, True]) is None
    assert auroc([None, None], [True, False]) is None


def test_coverage_and_precision_use_a_greater_or_equal_cut() -> None:
    rows = coverage_precision([1.0, 0.5, 0.4, None], [True, True, False, False], GRID)
    assert [(r["coverage"], r["precision"]) for r in rows] == [
        ({"numerator": 3, "denominator": 4}, {"numerator": 2, "denominator": 3}),
        ({"numerator": 2, "denominator": 4}, {"numerator": 2, "denominator": 2}),
        ({"numerator": 1, "denominator": 4}, {"numerator": 1, "denominator": 1}),
    ]
    assert rows[1]["threshold"] == {"numerator": 1, "denominator": 2}


def test_escalation_reports_the_errors_it_could_fix_and_the_work_it_repeats() -> None:
    rows = escalation_table([1.0, 0.9, 0.4, 0.3, None], [True, False, True, False, False], GRID)
    half = rows[1]
    assert half["escalated"] == {"numerator": 3, "denominator": 5}
    assert (half["escalated_wrong"], half["escalated_right"]) == (2, 1)
    assert half["kept_precision"] == {"numerator": 1, "denominator": 2}
    # keep 1 right, escalate 3 and suppose all succeed: (1 + 3) / 5
    assert half["ceiling_if_escalations_succeed"] == {"numerator": 4, "denominator": 5}
    # a threshold of 0 escalates only the slot with no answer
    assert rows[0]["escalated"] == {"numerator": 1, "denominator": 5}


def test_the_last_reliability_bin_includes_one() -> None:
    rows = reliability_bins([1.0, 0.95, 0.05], [True, True, False], bins=10)
    assert [(r["lower"]["numerator"], r["slots"]) for r in rows] == [(0, 1), (9, 2)]
    assert rows[1]["mean_confidence"] == pytest.approx(0.975)


def test_a_calibrated_forecaster_has_no_calibration_gap() -> None:
    rows = reliability_bins([0.5] * 4, [True, True, False, False], bins=10)
    assert calibration_gap(rows) == pytest.approx(0.0)
    assert calibration_gap([]) is None


def test_a_confident_wrong_forecaster_has_a_large_gap_and_brier_score() -> None:
    rows = reliability_bins([0.95, 0.95], [False, False], bins=10)
    assert calibration_gap(rows) == pytest.approx(0.95)
    assert brier_score([1.0, 0.0, 0.5], [True, False, False]) == pytest.approx(0.25 / 3)
    assert brier_score([None], [True]) is None


@pytest.mark.parametrize("call", [
    lambda: auroc([0.5], [True, False]),
    lambda: auroc([1.5], [True]),
    lambda: coverage_precision([-0.1], [True], GRID),
    lambda: reliability_bins([0.5], [True], bins=0),
])
def test_bad_columns_are_refused(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_escalation_separates_errors_a_better_scorer_could_fix_from_those_it_could_not() -> None:
    rows = escalation_table([0.9, 0.2, 0.1, 0.1], [True, False, False, True], (Fraction(1, 2),),
                            fixable=[True, True, False, True])
    row = rows[0]
    assert row["escalated"] == {"numerator": 3, "denominator": 4}
    assert (row["escalated_right"], row["escalated_wrong_fixable"],
            row["escalated_wrong_unfixable"]) == (1, 1, 1)
    # keep 1 right; escalate 1 right + 1 fixable error; the unfixable one stays wrong: 1 + 1 + 1
    assert row["ceiling_if_escalations_succeed"] == {"numerator": 3, "denominator": 4}


def test_escalation_flags_must_match_the_slots() -> None:
    with pytest.raises(ValueError, match="different numbers"):
        escalation_table([0.5], [True], GRID, fixable=[True, False])
