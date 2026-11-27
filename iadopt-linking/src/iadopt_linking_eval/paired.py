"""Paired comparison of two linkers judged on the same slots."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from .metrics import _exact

DEFAULT_RESAMPLES = 10_000
DEFAULT_LEVEL = Fraction(95, 100)


def _length(*columns: Sequence[Any]) -> int:
    """Check that parallel columns describe the same slots."""
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError(f"Columns describe different numbers of slots: {sorted(lengths)}")
    return lengths.pop()


def paired_counts(first: Sequence[bool], second: Sequence[bool]) -> dict[str, int]:
    """Cross-tabulate two systems' correctness on the same slots."""
    n = _length(first, second)
    both = sum(a and b for a, b in zip(first, second, strict=True))
    only_first = sum(a and not b for a, b in zip(first, second, strict=True))
    only_second = sum(b and not a for a, b in zip(first, second, strict=True))
    return {"n": n, "both": both, "only_first": only_first, "only_second": only_second,
            "neither": n - both - only_first - only_second}


def mcnemar_exact(only_first: int, only_second: int) -> Fraction:
    """Two-sided exact McNemar p-value."""
    if only_first < 0 or only_second < 0:
        raise ValueError("Discordant counts cannot be negative")
    n = only_first + only_second
    if n == 0:
        return Fraction(1)
    tail = sum(math.comb(n, k) for k in range(min(only_first, only_second) + 1))
    return min(Fraction(1), Fraction(2 * tail, 2**n))


def cluster_bootstrap_difference(
        first: Sequence[bool], second: Sequence[bool], clusters: Sequence[str], *,
        resamples: int = DEFAULT_RESAMPLES, seed: int = 0,
        level: Fraction = DEFAULT_LEVEL) -> dict[str, Any]:
    """Estimate the accuracy difference (first minus second) and a percentile interval for it."""
    n = _length(first, second, clusters)
    if n == 0:
        raise ValueError("Nothing to compare")
    if not 0 < level < 1 or resamples < 1:
        raise ValueError("The level must lie in (0, 1) and at least one resample is needed")
    members: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        members[cluster].append(index)
    totals = [(len(indexes), sum(first[i] for i in indexes), sum(second[i] for i in indexes))
              for indexes in members.values()]
    count = len(totals)
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(resamples):
        size = got_first = got_second = 0
        for _ in range(count):
            drawn = totals[int(rng.random() * count)]
            size += drawn[0]
            got_first += drawn[1]
            got_second += drawn[2]
        differences.append((got_first - got_second) / size)
    differences.sort()
    tail = (1 - level) / 2
    lower = differences[math.floor(tail * (resamples - 1))]
    upper = differences[math.ceil((1 - tail) * (resamples - 1))]
    correct_first, correct_second = sum(first), sum(second)
    difference = Fraction(correct_first - correct_second, n)
    return {"n": n, "clusters": count, "first_correct": correct_first,
            "second_correct": correct_second,
            "difference": _exact(difference),
            "interval": {"level": str(float(level)), "lower": lower, "upper": upper,
                         "resamples": resamples, "seed": seed,
                         "method": "percentile bootstrap over clusters"}}


def term_level(first: Sequence[bool], second: Sequence[bool], clusters: Sequence[str]) -> dict[str, Any]:
    """Compare two systems once per cluster, so that a term asked many times counts once."""
    n = _length(first, second, clusters)
    if n == 0:
        raise ValueError("Nothing to compare")
    members: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        members[cluster].append(index)
    better = worse = equal = 0
    total = Fraction(0)
    for indexes in members.values():
        a = Fraction(sum(first[i] for i in indexes), len(indexes))
        b = Fraction(sum(second[i] for i in indexes), len(indexes))
        better += a > b
        worse += a < b
        equal += a == b
        total += a - b
    p_value = mcnemar_exact(better, worse)
    mean = total / len(members)
    return {"terms": len(members), "first_better": better, "second_better": worse, "equal": equal,
            "sign_test_p": _exact(p_value), "mean_difference": _exact(mean)}
