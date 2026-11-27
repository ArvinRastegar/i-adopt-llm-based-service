#!/usr/bin/env python
"""Independent recomputation of the stronger-reader test (`stronger_reader_test.py`).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/STRONGER_READER_TEST_20261127.md` (private repository).

Standard library only. Nothing is imported from the lab or from the modules that ran the test; the
JSON extraction, the marking, the completed type table and the resampling come from
`recompute_conventions.py`. Step 1 is worked out again from the stored answers and latencies, Step 2
(when it ran) from the stored scores.

    ./recompute_stronger_reader.py     # prints the comparison; exit status 1 on any mismatch
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

BASELINE = "qwen3.8"
COLUMNS = ("parsed", "matrix", "matrix_pattern", "constraint", "both")
BREAK_EVEN, LIMIT = 0.54, 30.0
STEP2, RUNS = ("best", "best+reader"), 10
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
    patterns = json.loads((rc.DATA / "patterns.json").read_text(encoding="utf-8"))
    kinds = {number: row.get("kind") for number, row in patterns.get("patterns", patterns).items()}
    by_number, by_name = rc.table()
    ids = sorted(key)
    stored = json.loads((rc.OUT / "stronger-reader-summary.json").read_text(encoding="utf-8"))
    calls = rc.kept(rc.OUT / "stronger-reader-calls.jsonl")
    finder = rc.kept(rc.OUT / "answer-form-a-calls.jsonl")
    readers = sorted({k[0] for k in calls})
    answers = {r: {v: calls[(r, 1, v)].get("answer") if calls.get((r, 1, v), {}).get("ok") else None for v in ids} for r in readers}
    answers["finder-openrouter"] = {v: finder[("named-first", 1, v)].get("answer") for v in ids}
    marks = {r: {v: rc.mark(key[v], a[v], kinds, by_number, by_name) for v in ids} for r, a in answers.items()}
    if set(answers) != set(stored["readers"]):
        problems.append(f"readers: {sorted(answers)} here, {sorted(stored['readers'])} stored")
    for r, m in marks.items():
        tp, fp, fn = (sum(x[k] for x in m.values()) for k in ("tp", "fp", "fn"))
        row = {c: sum(bool(x[c]) for x in m.values()) / len(ids) for c in COLUMNS}
        row["type_f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        for field, value in row.items():
            if abs(value - stored["readers"][r][field]) > TOLERANCE:
                problems.append(f"{r} {field}: {value} here, {stored['readers'][r][field]} stored")
    fill = [r["latency_seconds"] for k, r in rc.kept(rc.OUT / "stand-in-decomposition-calls.jsonl").items() if k[0] == "best" and r.get("ok")]
    fill95 = rank(fill, 0.95)
    if abs(fill95 - stored["latency"]["fill"]["p95"]) > TOLERANCE:
        problems.append(f"fill p95: {fill95} here, {stored['latency']['fill']['p95']} stored")
    chosen = None
    for r in readers:
        lat = [calls[(r, 1, v)]["latency_seconds"] for v in ids if (r, 1, v) in calls]
        mine = {"p50": statistics.median(lat), "p95": rank(lat, 0.95), "max": max(lat)}
        for field, value in mine.items():
            if abs(value - stored["latency"][r][field]) > TOLERANCE:
                problems.append(f"{r} latency {field}: {value} here, {stored['latency'][r][field]} stored")
        if r == BASELINE:
            continue
        gaps = {}
        for c in ("matrix", "constraint"):
            gaps[c] = rc.interval([float(marks[r][v][c]) - float(marks[BASELINE][v][c]) for v in ids])
            for field in ("mean", "low", "high"):
                theirs = stored["differences"][f"{r} - {BASELINE}"][c][field]
                if not math.isclose(gaps[c][field], theirs, rel_tol=0.0, abs_tol=TOLERANCE):
                    problems.append(f"{r} {c} {field}: {gaps[c][field]} here, {theirs} stored")
        share = sum(bool(marks[r][v]["constraint"]) for v in ids) / len(ids)
        gate = {"types": share > BREAK_EVEN and gaps["constraint"]["low"] > 0, "matrix": gaps["matrix"]["low"] > 0,
                "time": mine["p95"] + fill95 <= LIMIT}
        if gate != stored["gates"][r]:
            problems.append(f"{r} gates: {gate} here, {stored['gates'][r]} stored")
        if gate["time"] and (gate["types"] or gate["matrix"]):
            candidate = (not gate["types"], -share, mine["p95"], r)
            chosen = min(chosen, candidate) if chosen else candidate
    expected = None if chosen is None else chosen[3]
    if expected != (stored["step2"] or {}).get("reader"):
        problems.append(f"step 2 reader: {expected} here, {(stored['step2'] or {}).get('reader')} stored")
    print(f"Step 1 checked: readers {readers}; Step 2 reader: {expected}")

    path = rc.OUT / "stronger-reader-decomposition-summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        dcalls = rc.kept(rc.OUT / "stronger-reader-decomposition-calls.jsonl")
        scored = {(tuple(r["key"]), r["answer_sha256"]): r for r in rc.rows(rc.OUT / "stronger-reader-decomposition-scores.jsonl")}
        scores = {}
        for kk, r in dcalls.items():
            digest = hashlib.sha256(r["answer"].encode("utf-8")).hexdigest() if r.get("ok") and r.get("answer") is not None else "no-answer"
            if (kk, digest) in scored:
                scores[kk] = scored[(kk, digest)]

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
        for arm in STEP2:
            per_run = []
            for run in range(1, RUNS + 1):
                total = [sum((counts(scores[(arm, run, v)])[i] for v in ids), Fraction(0)) for i in range(3)]
                per_run.append(float(2 * total[0] / (2 * total[0] + total[1] + total[2])) if total[0] else 0.0)
            for field, value in (("f1_mean", statistics.fmean(per_run)), ("f1_sd", statistics.stdev(per_run))):
                if abs(value - summary["arms"][arm][field]) > TOLERANCE:
                    problems.append(f"{arm} {field}: {value} here, {summary['arms'][arm][field]} stored")
            per_var[arm] = {v: [float(sum((counts(scores[(arm, run, v)])[i] for run in range(1, RUNS + 1)), Fraction(0)))
                                for i in range(3)] for v in ids}

        def pooled(sample: list[int]) -> float:
            """F1 of `best+reader` minus `best` over positions into the ids.

            Args: sample: positions.
            Returns: the difference.
            Raises: nothing.
            Side Effects: none.
            """
            out = []
            for arm in ("best+reader", "best"):
                tp = fp = fn = 0.0
                for position in sample:
                    a = per_var[arm][ids[position]]
                    tp, fp, fn = tp + a[0], fp + a[1], fn + a[2]
                out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
            return out[0] - out[1]

        rng = random.Random(rc.SEED2)
        drawn = sorted(pooled(rng.choices(range(len(ids)), k=len(ids))) for _ in range(rc.DRAWS))
        mine = {"gain": pooled(list(range(len(ids)))), "low": drawn[int(0.025 * rc.DRAWS)], "high": drawn[int(0.975 * rc.DRAWS) - 1]}
        for field, value in mine.items():
            theirs = summary["against_best"]["best+reader"][field]
            if not math.isclose(value, theirs, rel_tol=0.0, abs_tol=TOLERANCE):
                problems.append(f"best+reader - best {field}: {value} here, {theirs} stored")
        print("Step 2 checked")
    for problem in problems[:40]:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
