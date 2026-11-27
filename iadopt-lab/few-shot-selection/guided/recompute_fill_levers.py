#!/usr/bin/env python
"""Independent recomputation of the fill-levers test (`fill_levers_test.py`).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/FILL_LEVERS_TEST_20261127.md` (private repository).

Standard library only. From the stored calls, scores and agreements: the most typical of three by the
plan's rule, every arm's pooled Close F1 per run, the comparisons with their intervals, the latencies
and the re-ask counts.

    ./recompute_fill_levers.py     # prints the comparison; exit status 1 on any mismatch
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recompute_conventions as rc  # noqa: E402

ARMS = ("best", "best+typical3", "best+retry", "best+form")
SAMPLES = ("best", "best#2", "best#3")
RUNS = 10
TOLERANCE = 1e-9


def rank(values: list[float], share: float) -> float:
    """The nearest-rank percentile.

    Args: values: the values; share: between 0 and 1.
    Returns: the value at rank ceil(share * n).
    Raises: nothing for non-empty values.
    Side Effects: none.
    """
    ordered = sorted(values)
    return ordered[math.ceil(round(share * len(ordered), 9)) - 1]


def main() -> int:
    """Recompute and compare.

    Args: none.
    Returns: 0 when every figure matches, 1 otherwise.
    Raises: OSError when a stored file is missing.
    Side Effects: reads stored files; prints the comparison.
    """
    problems: list[str] = []
    key = json.loads((rc.DATA / "route_key.json").read_text(encoding="utf-8"))["variables"]
    ids = sorted(key)
    summary = json.loads((rc.OUT / "fill-levers-summary.json").read_text(encoding="utf-8"))
    picks = json.loads((rc.OUT / "fill-levers-typical3.json").read_text(encoding="utf-8"))
    calls = rc.kept(rc.OUT / "fill-levers-calls.jsonl")
    scored = {(tuple(r["key"]), r["answer_sha256"]): r for r in rc.rows(rc.OUT / "fill-levers-scores.jsonl")}

    def score(arm: str, run: int, v: str) -> dict:
        """The stored score row of one answer.

        Args: arm, run, v: the call.
        Returns: the row.
        Raises: KeyError when it is missing.
        Side Effects: none.
        """
        r = calls[(arm, run, v)]
        digest = hashlib.sha256(r["answer"].encode("utf-8")).hexdigest() if r.get("ok") and r.get("answer") is not None else "no-answer"
        return scored[((arm, run, v), digest)]

    rows: dict[tuple, dict] = {}
    for run in range(1, RUNS + 1):
        for v in ids:
            for arm in (*SAMPLES, "best+retry", "best+form"):
                rows[(arm, run, v)] = score(arm, run, v)
            agree = picks[f"{run}|{v}"]["agreements"]
            means = {i: sum(agree[f"{min(i, j)}-{max(i, j)}"] for j in (1, 2, 3) if j != i) / 2 for i in (1, 2, 3)}
            top = max(means.values())
            chosen = min(i for i in (1, 2, 3) if means[i] == top)
            if chosen != picks[f"{run}|{v}"]["chosen"]:
                problems.append(f"typical3 {run} {v}: {chosen} here, {picks[f'{run}|{v}']['chosen']} stored")
            for pair, value in agree.items():
                a, b = (int(x) for x in pair.split("-"))
                usable = all(rows[(SAMPLES[i - 1], run, v)]["prediction"] is not None for i in (a, b))
                if not usable and value != 0.0:
                    problems.append(f"{run} {v}: pair {pair} agrees although an answer is unusable")
            rows[("best+typical3", run, v)] = rows[(SAMPLES[chosen - 1], run, v)]

    def counts(row: dict) -> list[Fraction]:
        """Close TP, FP, FN of a score row.

        Args: row: a score row.
        Returns: [TP, FP, FN].
        Raises: KeyError when a count is missing.
        Side Effects: none.
        """
        t = row["evaluation"]["close"]["totals"]
        return [Fraction(t[n]["numerator"], t[n]["denominator"]) for n in ("tp", "fp", "fn")]

    per_var = {}
    for arm in ARMS:
        per_run = []
        for run in range(1, RUNS + 1):
            total = [sum((counts(rows[(arm, run, v)])[i] for v in ids), Fraction(0)) for i in range(3)]
            per_run.append(float(2 * total[0] / (2 * total[0] + total[1] + total[2])) if total[0] else 0.0)
        mine = {"f1_mean": statistics.fmean(per_run), "f1_sd": statistics.stdev(per_run),
                "invalid": sum(1 for run in range(1, RUNS + 1) for v in ids if not rows[(arm, run, v)]["valid_prediction"])}
        for field, value in mine.items():
            if abs(value - summary["arms"][arm][field]) > TOLERANCE:
                problems.append(f"{arm} {field}: {value} here, {summary['arms'][arm][field]} stored")
        per_var[arm] = {v: [float(sum((counts(rows[(arm, run, v)])[i] for run in range(1, RUNS + 1)), Fraction(0)))
                            for i in range(3)] for v in ids}

    def pooled(first: str, sample: list[int]) -> float:
        """F1 of an arm minus `best` over positions into the ids.

        Args: first: the arm; sample: positions.
        Returns: the difference.
        Raises: nothing.
        Side Effects: none.
        """
        out = []
        for arm in (first, "best"):
            tp = fp = fn = 0.0
            for position in sample:
                a = per_var[arm][ids[position]]
                tp, fp, fn = tp + a[0], fp + a[1], fn + a[2]
            out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        return out[0] - out[1]

    for arm in ARMS[1:]:
        rng = random.Random(rc.SEED2)
        drawn = sorted(pooled(arm, rng.choices(range(len(ids)), k=len(ids))) for _ in range(rc.DRAWS))
        mine = {"gain": pooled(arm, list(range(len(ids)))), "low": drawn[int(0.025 * rc.DRAWS)], "high": drawn[int(0.975 * rc.DRAWS) - 1]}
        for field, value in mine.items():
            theirs = summary["against_best"][arm][field]
            if not math.isclose(value, theirs, rel_tol=0.0, abs_tol=TOLERANCE):
                problems.append(f"{arm} {field}: {value} here, {theirs} stored")
        print(f"{arm} - best: {mine['gain']:+.4f} [{mine['low']:+.4f}, {mine['high']:+.4f}]")
    units = [(run, v) for run in range(1, RUNS + 1) for v in ids]
    times = {"best": [calls[("best", r, v)]["latency_seconds"] for r, v in units],
             "best+typical3": [max(calls[(s, r, v)]["latency_seconds"] for s in SAMPLES) for r, v in units],
             "best+retry": [calls[("best+retry", r, v)]["latency_total"] for r, v in units],
             "best+form": [calls[("best+form", r, v)]["latency_seconds"] for r, v in units]}
    for arm, values in times.items():
        for field, value in (("p50", statistics.median(values)), ("p95", rank(values, 0.95)), ("max", max(values))):
            if abs(value - summary["latency"][arm][field]) > TOLERANCE:
                problems.append(f"{arm} latency {field}: {value} here, {summary['latency'][arm][field]} stored")
    retried = [calls[("best+retry", r, v)] for r, v in units]
    mine = {"repaired": sum(bool(x.get("repaired")) for x in retried), "repair_valid": sum(bool(x.get("repair_valid")) for x in retried)}
    for field, value in mine.items():
        if value != summary["retry"][field]:
            problems.append(f"retry {field}: {value} here, {summary['retry'][field]} stored")
    for problem in problems[:40]:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
