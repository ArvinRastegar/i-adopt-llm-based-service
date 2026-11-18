#!/usr/bin/env python
"""Independent recomputation of the example-layout test's headline numbers.

Standard library only: imports nothing from the project, numpy or scipy, so a mistake in
`format_test.py`'s aggregation or statistics cannot repeat itself here. It reads the stored call
and score records, rebuilds micro Close F1 per repetition from the per-item TP/FP/FN counts,
redoes Welch's test with its own t distribution, and compares everything with
`format-test-summary.json`.

    ./format_test_recompute.py                 # recompute and compare
    ./format_test_recompute.py --validate-study # first check the F1 formula on the study's data
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "output"
STUDY_CANDIDATE = "2d56af0e65d42748"


def jsonl(path: Path):
    """Stream the rows of a JSON-lines file.

    Args: path: the file.
    Returns: an iterator over the parsed rows; blank lines are skipped.
    Raises: OSError when the file cannot be read; json.JSONDecodeError on a malformed line.
    Side Effects: reads the file lazily.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def counts(evaluation: dict) -> tuple[Fraction, Fraction, Fraction]:
    """Read one item's exact Close counts from its stored evaluation.

    Args: evaluation: a stored per-item evaluation record.
    Returns: (TP, FP, FN) as exact fractions.
    Raises: KeyError when the record lacks Close totals.
    Side Effects: none. Deterministic.
    """
    totals = evaluation["close"]["totals"]
    return tuple(Fraction(totals[k]["numerator"], totals[k]["denominator"])
                 for k in ("tp", "fp", "fn"))


def micro_f1(items: list[tuple[Fraction, Fraction, Fraction]]) -> float:
    """Micro F1 from summed exact counts, zero when precision and recall are both zero.

    Args: items: (TP, FP, FN) per item.
    Returns: F1 as a float, computed exactly and converted once at the end.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    tp = sum(i[0] for i in items)
    fp = sum(i[1] for i in items)
    fn = sum(i[2] for i in items)
    precision = tp / (tp + fp) if tp + fp else Fraction(0)
    recall = tp / (tp + fn) if tp + fn else Fraction(0)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _betacf(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete beta continued fraction (modified Lentz).

    Args: a, b: positive shape parameters; x: point in (0, 1).
    Returns: the continued-fraction factor used by `betainc`.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Args: a, b: positive shape parameters; x: point, clipped to [0, 1].
    Returns: I_x(a, b).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of Student's t.

    Args: t: the statistic; df: degrees of freedom, positive.
    Returns: P(|T| >= |t|).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_quantile_975(df: float) -> float:
    """The 97.5% quantile of Student's t, by bisection on the two-sided p-value.

    Args: df: degrees of freedom, positive.
    Returns: the quantile, to about 1e-13.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    low, high = 0.0, 100.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if t_two_sided_p(mid, df) > 0.05:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def welch(b: list[float], a: list[float]) -> dict[str, float]:
    """Welch's two-sided t-test of mean(b) - mean(a), with a 95% confidence interval.

    Args: b, a: per-repetition scores, at least two in each with non-zero spread.
    Returns: difference, t, df, p, ci_low, ci_high.
    Raises: statistics.StatisticsError with fewer than two values; ZeroDivisionError when
        both groups are constant.
    Side Effects: none. Deterministic.
    """
    diff = statistics.fmean(b) - statistics.fmean(a)
    vb, va = statistics.variance(b) / len(b), statistics.variance(a) / len(a)
    se = math.sqrt(vb + va)
    df = (vb + va) ** 2 / (vb ** 2 / (len(b) - 1) + va ** 2 / (len(a) - 1))
    t = diff / se
    half = t_quantile_975(df) * se
    return {"difference": diff, "t": t, "df": df, "p": t_two_sided_p(abs(t), df),
            "ci_low": diff - half, "ci_high": diff + half}


def latest_calls(path: Path) -> dict[tuple, dict]:
    """Keep one record per call: the first successful one, else the last failure.

    Args: path: the calls file.
    Returns: {(model, layout, repetition, variable_id): record}.
    Raises: as `jsonl`.
    Side Effects: reads the file.
    """
    chosen: dict[tuple, dict] = {}
    for row in jsonl(path):
        key = (row["model"], row["format"], row["repetition"], row["variable_id"])
        held = chosen.get(key)
        if held is None or not held["ok"]:
            chosen[key] = row
    return chosen


def recompute() -> dict:
    """Rebuild per-repetition Close F1 for every cell from the stored records.

    Args: none.
    Returns: {"targets", "per_rep", "cells"}; a repetition missing a call is left out.
    Raises: KeyError when a call has no stored score.
    Side Effects: reads the calls and scores files.
    """
    calls = latest_calls(OUTPUT / "format-test-calls.jsonl")
    scores = {tuple(r["key"]): r for r in jsonl(OUTPUT / "format-test-scores.jsonl")}
    targets = sorted({key[3] for key in calls})
    per_rep: dict[tuple, dict] = {}
    groups: dict[tuple, list] = defaultdict(list)
    for key in calls:
        groups[key[:3]].append(key)
    for (model, fmt, rep), keys in sorted(groups.items()):
        if len(keys) != len(targets):
            continue
        items = [counts(scores[k]["evaluation"]) for k in keys]
        valid = [counts(scores[k]["evaluation"]) for k in keys if scores[k]["valid_prediction"]]
        per_rep[(model, fmt, rep)] = {"official": micro_f1(items), "as_run": micro_f1(valid),
                                      "invalid": len(keys) - len(valid)}
    cells: dict[str, list[float]] = defaultdict(list)
    for (model, fmt, _), row in sorted(per_rep.items()):
        cells[f"{model}|{fmt}"].append(row["official"])
    return {"targets": len(targets), "per_rep": per_rep, "cells": dict(cells)}


def validate_study() -> None:
    """Check the F1 formula against values the study's own pipeline stored.

    Args: none.
    Returns: nothing; prints the largest difference over the study's 15 top25 repetitions.
    Raises: ValueError when the study's files hold no top25 confirmation rows.
    Side Effects: reads the study's calls and observations files (about 400 MB).
    """
    stored = {row["repetition"]: row["result"]["close_f1_as_run"]
              for row in jsonl(OUTPUT / "example-selection-observations.jsonl")
              if row["stage"] == "confirm" and row["name"] == "top25"}
    c_targets = {row["variable_id"] for row in jsonl(OUTPUT / "format-test-calls.jsonl")}
    items: dict[int, list] = defaultdict(list)
    for row in jsonl(OUTPUT / "example-selection-calls.jsonl"):
        if (row.get("candidate_hash") == STUDY_CANDIDATE and row["variable_id"] in c_targets
                and row.get("evaluation")):
            items[row["repetition"]].append(counts(row["evaluation"]))
    worst = max(abs(micro_f1(items[rep]) - value) for rep, value in stored.items())
    print(f"study check: {len(stored)} repetitions, largest difference from the stored "
          f"as-run Close F1 = {worst:.2e}")


def main() -> int:
    """Recompute, compare with the summary and print the comparison.

    Args: none; reads --validate-study from the command line.
    Returns: 0.
    Raises: KeyError when the summary lacks a recomputed cell or comparison.
    Side Effects: reads the output files; prints to standard output.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-study", action="store_true")
    args = parser.parse_args()
    if args.validate_study:
        validate_study()
    result = recompute()
    summary = json.loads((OUTPUT / "format-test-summary.json").read_text(encoding="utf-8"))
    print(f"targets per repetition: {result['targets']}")
    worst = 0.0
    for name, values in result["cells"].items():
        stored = summary["cells"][name]
        mine = [r["close_f1_official"] for r in stored["per_repetition"]]
        worst = max([worst] + [abs(x - y) for x, y in zip(values, mine, strict=True)])
        print(f"{name:16} reps {len(values)}  mean {statistics.fmean(values):.4f}  "
              f"SD {statistics.stdev(values):.4f}  (summary {stored['close_f1_official_mean']:.4f})")
    print(f"largest per-repetition difference from the summary: {worst:.2e}")
    for model in sorted({name.split('|')[0] for name in result["cells"]}):
        a, b = result["cells"].get(f"{model}|A"), result["cells"].get(f"{model}|B")
        if a and b:
            mine = welch(b, a)
            theirs = summary["primary"][model]
            print(f"{model}: B-A {mine['difference']:+.4f} [{mine['ci_low']:+.4f}, "
                  f"{mine['ci_high']:+.4f}] t {mine['t']:.3f} df {mine['df']:.2f} "
                  f"p {mine['p']:.3g}  | summary p {theirs['p']:.3g}, CI "
                  f"[{theirs['ci_low']:+.4f}, {theirs['ci_high']:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
