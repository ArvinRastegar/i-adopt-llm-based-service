#!/usr/bin/env python
"""Independent recomputation of the answer-form check (`answer_form_check.py`).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/ANSWER_FORM_CHECK_20261127.md` (private repository).

Standard library only. Nothing is imported from the lab or from the modules that ran the check;
the marking, extraction and type table come from `recompute_conventions.py`, which is itself
standard library only and was written apart from them. Part A is marked again from the stored
answers; Part B is worked out again from the stored scores.

    ./recompute_answer_form.py     # prints the comparison; exit status 1 on any mismatch
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recompute_conventions as rc  # noqa: E402

ARMS = ("five-key", "named-first")
RUNS_A, RUNS_B = 5, 10
BREAK_EVEN_MATRIX = 0.57
TOLERANCE = 1e-9


def main() -> int:
    """Recompute and compare.

    Args: none.
    Returns: 0 when every figure matches, 1 otherwise.
    Raises: OSError when a stored file is missing.
    Side Effects: reads stored files; prints the comparison.
    """
    problems: list[str] = []
    key = json.loads((rc.DATA / "route_key.json").read_text(encoding="utf-8"))["variables"]
    patterns = json.loads((rc.DATA / "patterns.json").read_text(encoding="utf-8"))
    kinds = {number: row.get("kind") for number, row in patterns.get("patterns", patterns).items()}
    by_number, by_name = rc.table()
    ids = sorted(key)
    calls = rc.kept(rc.OUT / "answer-form-a-calls.jsonl")
    summary_a = json.loads((rc.OUT / "answer-form-a-summary.json").read_text(encoding="utf-8"))
    marked = {k: rc.mark(key[k[2]], r.get("answer") if r.get("ok") else None, kinds, by_number, by_name)
              for k, r in calls.items()}
    for arm in ARMS:
        rows = [row for (a, _, _), row in marked.items() if a == arm]
        if len(rows) != len(ids) * RUNS_A:
            problems.append(f"{arm}: {len(rows)} answers, {len(ids) * RUNS_A} expected")
        for column in ("matrix", "constraint", "matrix_pattern", "both", "parsed"):
            mine = sum(r[column] for r in rows) / len(rows)
            if abs(mine - summary_a["arms"][arm][column]) > TOLERANCE:
                problems.append(f"{arm} {column}: {mine} here, {summary_a['arms'][arm][column]} stored")
        tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if abs(f1 - summary_a["arms"][arm]["constraint_f1"]) > TOLERANCE:
            problems.append(f"{arm} constraint F1: {f1} here, {summary_a['arms'][arm]['constraint_f1']} stored")

    def share(arm: str, column: str) -> dict[str, float]:
        """Each variable's share right over its runs.

        Args: arm: the arm; column: the column.
        Returns: {variable: share}.
        Raises: ZeroDivisionError for a variable without answers.
        Side Effects: none.
        """
        tally = defaultdict(list)
        for (a, _, v), row in marked.items():
            if a == arm:
                tally[v].append(row[column])
        return {v: sum(tally[v]) / len(tally[v]) for v in ids}

    for column in ("matrix", "constraint"):
        first, second = share("named-first", column), share("five-key", column)
        mine = rc.interval([first[v] - second[v] for v in ids])
        for field, value in mine.items():
            if abs(value - summary_a["comparisons"][column][field]) > TOLERANCE:
                problems.append(f"comparison {column} {field}: {value} here, {summary_a['comparisons'][column][field]} stored")
    runs = summary_a["arms"]["named-first"]["matrix"] > BREAK_EVEN_MATRIX
    if runs != summary_a["finder_arm_runs"]:
        problems.append(f"gate: {runs} here, {summary_a['finder_arm_runs']} stored")
    print(f"Part A: {len(calls)} answers checked; the finder arm runs: {runs}")

    path_b = rc.OUT / "answer-form-b-summary.json"
    if path_b.exists():
        summary_b = json.loads(path_b.read_text(encoding="utf-8"))
        arms = list(summary_b.get("arms", {}))
        calls_b = rc.kept(rc.OUT / "answer-form-b-calls.jsonl")
        stored = {(tuple(r["key"]), r["answer_sha256"]): r for r in rc.rows(rc.OUT / "answer-form-b-scores.jsonl")}
        scores = {}
        for k, r in calls_b.items():
            digest = hashlib.sha256(r["answer"].encode("utf-8")).hexdigest() if r.get("ok") and r.get("answer") is not None else "no-answer"
            if (k, digest) in stored:
                scores[k] = stored[(k, digest)]

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
        for arm in arms:
            per_run = []
            for r in range(1, RUNS_B + 1):
                total = [sum((counts(scores[(arm, r, v)])[k] for v in ids), Fraction(0)) for k in range(3)]
                per_run.append(float(2 * total[0] / (2 * total[0] + total[1] + total[2])) if total[0] else 0.0)
            for field, value in (("f1_mean", statistics.fmean(per_run)), ("f1_sd", statistics.stdev(per_run))):
                if abs(value - summary_b["arms"][arm][field]) > TOLERANCE:
                    problems.append(f"Part B {arm} {field}: {value} here, {summary_b['arms'][arm][field]} stored")
            per_var[arm] = {v: [float(sum((counts(scores[(arm, r, v)])[k] for r in range(1, RUNS_B + 1)), Fraction(0)))
                                for k in range(3)] for v in ids}

        def pooled(first: str, sample: list[int]) -> float:
            """F1 of an arm minus `rag-25` over positions into the ids.

            Args: first: the arm; sample: positions, repeats allowed.
            Returns: the difference.
            Raises: nothing.
            Side Effects: none.
            """
            out = []
            for arm in (first, "rag-25"):
                tp = fp = fn = 0.0
                for position in sample:
                    a = per_var[arm][ids[position]]
                    tp, fp, fn = tp + a[0], fp + a[1], fn + a[2]
                out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
            return out[0] - out[1]

        for arm in arms:
            if arm == "rag-25":
                continue
            rng = random.Random(rc.SEED2)
            drawn = sorted(pooled(arm, rng.choices(range(len(ids)), k=len(ids))) for _ in range(rc.DRAWS))
            mine = {"gain": pooled(arm, list(range(len(ids)))), "low": drawn[int(0.025 * rc.DRAWS)],
                    "high": drawn[int(0.975 * rc.DRAWS) - 1]}
            for field, value in mine.items():
                theirs = summary_b["against_rag25"][arm][field]
                if not math.isclose(value, theirs, rel_tol=0.0, abs_tol=TOLERANCE):
                    problems.append(f"Part B {arm} {field}: {value} here, {theirs} stored")
        print("Part B: checked arms", arms)
    for problem in problems[:40]:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
