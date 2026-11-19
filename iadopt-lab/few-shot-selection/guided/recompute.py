#!/usr/bin/env python
"""Independent recomputation of a guided-decomposition comparison.

Standard library only: it imports nothing from the project, numpy or scipy, so a mistake in
`engine.py`'s aggregation or statistics cannot repeat itself here. It reads the stored call and
score records, rebuilds micro Close F1 per arm and repetition from the per-item TP/FP/FN counts,
redoes Welch's test with its own t distribution and Holm's correction, and compares everything
with the summary file.

    ./recompute.py --name ceiling --baseline base
    ./recompute.py --check-table      # every cell of groups-table.json, pooled rows included
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "output" / "guided"


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
    return tuple(Fraction(totals[k]["numerator"], totals[k]["denominator"]) for k in ("tp", "fp", "fn"))


def micro_f1(items: list[tuple[Fraction, Fraction, Fraction]]) -> float:
    """Micro F1 from summed exact counts, zero when precision and recall are both zero.

    Args: items: (TP, FP, FN) per item.
    Returns: F1 as a float, computed exactly and converted once at the end.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    tp, fp, fn = (sum(i[k] for i in items) for k in range(3))
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
    Raises: statistics.StatisticsError with fewer than two values; ZeroDivisionError when both
        groups are constant.
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


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm's step-down correction.

    Args: pvalues: {name: p}.
    Returns: {name: adjusted p}, capped at 1 and monotone in rank.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out, running = {}, 0.0
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        out[name] = running
    return out


def recompute(name: str) -> dict[str, list[float]]:
    """Rebuild per-repetition Close F1 for every arm from the stored records.

    Args: name: the comparison's file prefix, such as "ceiling".
    Returns: {arm: per-repetition F1, in repetition order}; a repetition missing a call or a
        score is left out.
    Raises: OSError when the files are missing.
    Side Effects: reads the calls and scores files.
    """
    calls: dict[tuple, dict] = {}
    for row in jsonl(OUTPUT / f"{name}-calls.jsonl"):
        key = (row["arm"], row["repetition"], row["variable_id"])
        if key not in calls or not calls[key]["ok"]:
            calls[key] = row
    scores = {tuple(row["key"]): row for row in jsonl(OUTPUT / f"{name}-scores.jsonl")}
    variables = sorted({key[2] for key in calls})
    groups: dict[tuple, list] = defaultdict(list)
    for key in calls:
        groups[key[:2]].append(key)
    cells: dict[str, list[float]] = defaultdict(list)
    for (arm, _), keys in sorted(groups.items()):
        if len(keys) == len(variables) and all(key in scores for key in keys):
            cells[arm].append(micro_f1([counts(scores[key]["evaluation"]) for key in keys]))
    return dict(cells)


# Stated here on its own, not imported: which stored comparison and arm holds each setup's answers
# for each group. `groups.py` has its own copy; a wrong mapping in either shows as a mismatch.
FOUND = ("guided+check@direct/full", "guided+check@direct-all/full", "guided+check@mixed-all/full",
         "guided+check@direct-all/form+types", "guided+check@mixed/full")
TABLE_ARMS = ("base", "guided@key/full", *FOUND)
TABLE_SOURCES = {
    "examples": {arm: [("groups-examples", arm)] for arm in TABLE_ARMS},
    "held-out": {arm: [("groups-held-out", arm)] for arm in TABLE_ARMS},
    "practice": {"base": [("ceiling", "base"), ("e2e", "base"), ("tune", "base")],
                 "guided@key/full": [("ceiling", "full")],
                 "guided+check@direct/full": [("e2e", "guided+check")],
                 **{arm: [("tune", arm)] for arm in FOUND[1:]}}}
TABLE_ROWS = {"examples": ("examples",), "practice": ("practice",), "held-out": ("held-out",),
              "practice + held-out": ("practice", "held-out"), "all": ("examples", "practice", "held-out")}


def item_counts(name: str) -> dict[str, dict[int, list]]:
    """Every arm's exact per-answer Close counts of one stored comparison, by repetition.

    Args: name: the comparison's file prefix.
    Returns: {arm: {repetition: [(TP, FP, FN), ...]}}, one entry per answer (the last stored score).
    Raises: OSError when the scores file is missing.
    Side Effects: reads the scores file.
    """
    last: dict[tuple, tuple] = {}
    for row in jsonl(OUTPUT / f"{name}-scores.jsonl"):
        last[tuple(row["key"])] = counts(row["evaluation"])
    out: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for (arm, repetition, _), item in last.items():
        out[arm][repetition].append(item)
    return out


def check_table() -> int:
    """Recompute every cell of `groups-table.json` from the stored scores and compare.

    Args: none.
    Returns: 0 when every cell matches to 1e-9, else 1.
    Raises: OSError when a stored file is missing.
    Side Effects: reads the stored score files and the table; prints to standard output.
    """
    table = json.loads((OUTPUT / "groups-table.json").read_text(encoding="utf-8"))["rows"]
    names = sorted({name for group in TABLE_SOURCES.values() for sources in group.values() for name, _ in sources})
    stored = {name: item_counts(name) for name in names}
    worst = 0.0
    for row, groups in TABLE_ROWS.items():
        line = []
        for arm in TABLE_ARMS:
            choices = [TABLE_SOURCES[group][arm] for group in groups]
            means = []
            for i in range(max(len(choice) for choice in choices)):
                picked = [stored[name][source] for name, source in
                          (choice[min(i, len(choice) - 1)] for choice in choices)]
                repetitions = sorted(set.intersection(*(set(series) for series in picked)))
                means.append(statistics.fmean(micro_f1([item for series in picked for item in series[r]])
                                              for r in repetitions))
            mine = statistics.fmean(means)
            worst = max(worst, abs(mine - table[row]["arms"][arm]["mean"]))
            line.append(f"{mine:.4f}")
        print(f"{row:20} " + "  ".join(line))
    print(f"largest difference from groups-table.json: {worst:.2e}")
    print("MATCH" if worst < 1e-9 else "MISMATCH")
    return 0 if worst < 1e-9 else 1


def main() -> int:
    """Recompute, compare with the summary and print the comparison.

    Args: none; reads --name and --baseline, or --check-table, from the command line.
    Returns: 0 when everything matches the stored figures to 1e-9, else 1.
    Raises: KeyError when the summary lacks a recomputed arm.
    Side Effects: reads the output files; prints to standard output.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--baseline")
    parser.add_argument("--check-table", action="store_true")
    args = parser.parse_args()
    if args.check_table:
        return check_table()
    if not (args.name and args.baseline):
        parser.error("--name and --baseline are required unless --check-table is given")
    cells = recompute(args.name)
    summary = json.loads((OUTPUT / f"{args.name}-summary.json").read_text(encoding="utf-8"))
    worst = 0.0
    for arm, values in cells.items():
        stored = [r["close_f1_official"] for r in summary["cells"][arm]["per_repetition"]]
        worst = max([worst] + [abs(x - y) for x, y in zip(values, stored, strict=True)])
        print(f"{arm:20} reps {len(values):2}  mean {statistics.fmean(values):.4f}  "
              f"SD {statistics.stdev(values):.4f}  (summary {summary['cells'][arm]['close_f1_official_mean']:.4f})")
    print(f"largest per-repetition difference from the summary: {worst:.2e}")
    tests = {arm: welch(values, cells[args.baseline]) for arm, values in cells.items() if arm != args.baseline}
    adjusted = holm({arm: result["p"] for arm, result in tests.items()})
    for arm, mine in tests.items():
        theirs = summary["comparisons"][arm]
        worst = max(worst, abs(mine["difference"] - theirs["difference"]), abs(mine["ci_low"] - theirs["ci_low"]),
                    abs(mine["ci_high"] - theirs["ci_high"]), abs(adjusted[arm] - theirs["p_holm"]))
        print(f"{arm:20} vs {args.baseline}: {mine['difference']:+.4f} [{mine['ci_low']:+.4f}, "
              f"{mine['ci_high']:+.4f}] t {mine['t']:.2f} df {mine['df']:.1f} holm p {adjusted[arm]:.3g}"
              f"  | summary holm p {theirs['p_holm']:.3g}")
    print("MATCH" if worst < 1e-9 else f"MISMATCH (largest difference {worst:.2e})")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
