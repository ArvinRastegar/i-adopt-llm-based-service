"""How well a confidence value separates right answers from wrong ones, and what a cut-off costs.

Pure standard-library helpers in the same contract as the rest of this package. Counts are
receipts. The statistics that are real-valued by nature (the calibration gap, the Brier score) are
floats and meant for display. A confidence of None means the system gave no answer for the slot;
no threshold ever accepts it, and the rank statistics leave it out.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from .metrics import _count, _exact


def _check(confidences: Sequence[float | None], correct: Sequence[bool]) -> None:
    """Refuse columns that cannot describe the same slots or a confidence outside [0, 1].

    Args: confidences: one value or None per slot; correct: whether each answer was right.
    Returns: None.
    Raises: ValueError for different lengths or a value outside [0, 1].
    Side effects: none.
    """
    if len(confidences) != len(correct):
        raise ValueError("Confidences and outcomes describe different numbers of slots")
    for value in confidences:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"A confidence must lie in [0, 1], got {value!r}")


def auroc(confidences: Sequence[float | None], correct: Sequence[bool]) -> Fraction | None:
    """Give the chance that a right answer is more confident than a wrong one.

    Args: confidences: one value or None per slot; correct: whether each answer was right.
    Returns: P(confidence of a random right answer > that of a random wrong one), counting ties
    as one half, as an exact fraction over the answered slots; None when only one kind exists.
    Raises: ValueError for mismatched columns or a value outside [0, 1].
    Side effects: none.
    """
    _check(confidences, correct)
    answered = [(c, ok) for c, ok in zip(confidences, correct, strict=True) if c is not None]
    right = [c for c, ok in answered if ok]
    wrong = [c for c, ok in answered if not ok]
    if not right or not wrong:
        return None
    twice = sum(2 if r > w else 1 if r == w else 0 for r in right for w in wrong)
    return Fraction(twice, 2 * len(right) * len(wrong))


def coverage_precision(confidences: Sequence[float | None], correct: Sequence[bool],
                       thresholds: Sequence[Fraction]) -> list[dict[str, Any]]:
    """Trade coverage against precision by answering only at or above a confidence.

    Args: confidences: one value or None per slot; correct: whether each answer was right;
    thresholds: cut-offs to try.
    Returns: one row per threshold: the threshold, how many slots were answered, and how many of
    those were right.
    Raises: ValueError for mismatched columns or a value outside [0, 1].
    Side effects: none.
    """
    _check(confidences, correct)
    rows = []
    for threshold in thresholds:
        cut = float(threshold)
        accepted = [ok for c, ok in zip(confidences, correct, strict=True)
                    if c is not None and c >= cut]
        rows.append({"threshold": _exact(threshold),
                     "coverage": _count(len(accepted), len(correct)),
                     "precision": _count(sum(accepted), len(accepted))})
    return rows


def escalation_table(confidences: Sequence[float | None], correct: Sequence[bool],
                     thresholds: Sequence[Fraction], fixable: Sequence[bool] | None = None
                     ) -> list[dict[str, Any]]:
    """Show what handing the less confident slots to a stronger system would involve.

    Args: confidences: one value or None per slot (None is always escalated); correct: whether
    each answer was right; thresholds: slots below a threshold are escalated; fixable: whether a
    better scorer could be right on each slot (the gold is among the candidates), default all.
    Returns: one row per threshold: the escalated slots; among them the right ones (work a
    stronger system would repeat), the wrong ones it could fix and the wrong ones it could not;
    the kept slots' precision; and the accuracy over all slots if every escalated slot that can
    be fixed were then answered correctly. That last figure is a ceiling, not a prediction.
    Raises: ValueError for mismatched columns or a value outside [0, 1].
    Side effects: none.
    """
    _check(confidences, correct)
    flags = [True] * len(correct) if fixable is None else list(fixable)
    if len(flags) != len(correct):
        raise ValueError("Fixable flags and outcomes describe different numbers of slots")
    total = len(correct)
    rows = []
    for threshold in thresholds:
        cut = float(threshold)
        kept = [ok for c, ok in zip(confidences, correct, strict=True) if c is not None and c >= cut]
        sent = [(ok, can) for c, ok, can in zip(confidences, correct, flags, strict=True)
                if c is None or c < cut]
        right = sum(ok for ok, _ in sent)
        fixable_wrong = sum(can and not ok for ok, can in sent)
        rows.append({"threshold": _exact(threshold),
                     "escalated": _count(len(sent), total),
                     "escalated_wrong": len(sent) - right, "escalated_right": right,
                     "escalated_wrong_fixable": fixable_wrong,
                     "escalated_wrong_unfixable": len(sent) - right - fixable_wrong,
                     "kept_precision": _count(sum(kept), len(kept)),
                     "ceiling_if_escalations_succeed": _count(sum(kept) + right + fixable_wrong,
                                                              total)})
    return rows


def reliability_bins(confidences: Sequence[float | None], correct: Sequence[bool],
                     bins: int = 10) -> list[dict[str, Any]]:
    """Compare stated confidence with observed accuracy in equal-width bins.

    Args: confidences: one value or None per slot; correct: whether each answer was right; bins:
    number of equal-width bins on [0, 1]; the last bin includes 1.
    Returns: one row per non-empty bin: its bounds, the slots in it, how many were right and the
    mean confidence (a float).
    Raises: ValueError for mismatched columns, a value outside [0, 1] or fewer than one bin.
    Side effects: none.
    """
    _check(confidences, correct)
    if bins < 1:
        raise ValueError("At least one bin is needed")
    members: dict[int, list[tuple[float, bool]]] = {}
    for c, ok in zip(confidences, correct, strict=True):
        if c is not None:
            members.setdefault(min(int(c * bins), bins - 1), []).append((c, ok))
    rows = []
    for index in sorted(members):
        pairs = members[index]
        rows.append({"lower": _exact(Fraction(index, bins)), "upper": _exact(Fraction(index + 1, bins)),
                     "slots": len(pairs), "right": sum(ok for _, ok in pairs),
                     "mean_confidence": sum(c for c, _ in pairs) / len(pairs)})
    return rows


def calibration_gap(rows: Sequence[dict[str, Any]]) -> float | None:
    """Average the distance between confidence and accuracy over bins, weighted by their size.

    Args: rows: reliability_bins output.
    Returns: the expected calibration error as a float, or None when there are no answers.
    Raises: KeyError for a malformed row.
    Side effects: none.
    """
    total = sum(row["slots"] for row in rows)
    if total == 0:
        return None
    return sum(row["slots"] / total * abs(row["right"] / row["slots"] - row["mean_confidence"])
               for row in rows)


def brier_score(confidences: Sequence[float | None], correct: Sequence[bool]) -> float | None:
    """Average the squared gap between confidence and the 0/1 outcome.

    Args: confidences: one value or None per slot; correct: whether each answer was right.
    Returns: the Brier score over answered slots as a float (0 is perfect, 0.25 is a coin flip
    stated at 0.5), or None when no slot was answered.
    Raises: ValueError for mismatched columns or a value outside [0, 1].
    Side effects: none.
    """
    _check(confidences, correct)
    answered = [(c, ok) for c, ok in zip(confidences, correct, strict=True) if c is not None]
    if not answered:
        return None
    return sum((c - float(ok)) ** 2 for c, ok in answered) / len(answered)
