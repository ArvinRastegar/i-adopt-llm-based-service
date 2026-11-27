#!/usr/bin/env python
"""Independent recomputation of the constraint-types test (`constraint_types_test.py`).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/CONSTRAINT_TYPES_TEST_20261127.md` (private repository).

Standard library only. Nothing is imported from the lab or from the modules that ran the test; the
JSON extraction and the completed type table come from `recompute_conventions.py` (standard library
only, written apart from them). Step 1 is worked out again from the stored answers (S1a, the agreement,
the folds' thresholds, S1d's signals and the chosen one) and from the stored nearest-value choices
(S1c's sets; the embedding similarities are not recomputed, so S1b is not checked). Step 2 is worked
out again from the stored scores.

    ./recompute_constraint_types.py     # prints the comparison; exit status 1 on any mismatch
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recompute_conventions as rc  # noqa: E402

SOURCES = (("conventions-part1-calls.jsonl", "plain"), ("answer-form-a-calls.jsonl", "named-first"))
RUN_ONE, RUNS2 = 5, 10
FOLDS, FOLD_SEED = 6, 20261123
GROUP_FILES = {"examples": "groups-examples-calls.jsonl", "practice": "ceiling-calls.jsonl",
               "held-out": "groups-held-out-calls.jsonl"}
TOLERANCE = 1e-9


def folds(key: dict, ids: list[str]) -> dict[str, int]:
    """Stage 0's folds, dealt again.

    Args: key: the experts' tags; ids: the variables.
    Returns: {variable: fold}.
    Raises: OSError when a group file is missing.
    Side Effects: reads three stored files.
    """
    member_of = {}
    for name, file in GROUP_FILES.items():
        for row in rc.rows(rc.OUT.parent / "guided" / file):
            member_of[row["variable_id"]] = name
    rng = random.Random(FOLD_SEED)
    out: dict[str, int] = {}
    for group in ("examples", "practice", "held-out"):
        for has_matrix in (True, False):
            stratum = sorted(v for v in ids if member_of[v] == group and bool(key[v]["matrix"]) == has_matrix)
            rng.shuffle(stratum)
            for v in stratum:
                out[v] = len(out) % FOLDS
    return out


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
    fold_of = folds(key, ids)
    step1 = json.loads((rc.OUT / "constraint-types-step1.json").read_text(encoding="utf-8"))

    def types_of(answer: str | None):
        """The type set of one answer, read strictly.

        Args: answer: the reply or None.
        Returns: a frozenset of names, or None when the answer is not the requested JSON.
        Raises: nothing.
        Side Effects: none.
        """
        data = rc.extract(answer)
        proper = bool(data is not None and {"property", "matrix", "context", "modifier", "constraints"} <= set(data)
                      and (data["matrix"] is None or (type(data["matrix"]) is int and 40 <= data["matrix"] <= 45))
                      and type(data["context"]) is bool and type(data["modifier"]) is bool
                      and isinstance(data["constraints"], list) and all(isinstance(n, str) for n in data["constraints"]))
        if not proper or kinds.get(str(data["property"])) != "property":
            return None
        return frozenset(by_number[by_name[n.lower()]] for n in data["constraints"] if n.lower() in by_name)

    gold = {v: frozenset(by_number.get(n) or key[v]["names"].get(str(n), str(n)) for n in key[v]["constraint"]) for v in ids}
    sets: dict[str, list] = {v: [] for v in ids}
    for file, arm in SOURCES:
        calls = rc.kept(rc.OUT / file)
        for v in ids:
            for run in range(1, 6):
                record = calls[(arm, run, v)]
                sets[v].append(types_of(record.get("answer") if record.get("ok") else None))
    k: Counter = Counter()
    for v in ids:
        for s in sets[v]:
            for t in s or ():
                k[(v, t)] += 1

    def pr(members: list[str], threshold: int):
        """Precision and recall at a threshold.

        Args: members: variables; threshold: K.
        Returns: (named, right, precision or None, recall).
        Raises: nothing.
        Side Effects: none.
        """
        keep = set(members)
        named = [(v, t) for (v, t), c in k.items() if v in keep and c >= threshold]
        right = sum(1 for v, t in named if t in gold[v])
        total = sum(len(gold[v]) for v in keep)
        return len(named), right, (right / len(named) if named else None), right / total

    for K in range(1, 11):
        named, right, precision, recall = pr(ids, K)
        stored = step1["s1a"][str(K)]
        if (named, right) != (stored["named"], stored["right"]) or abs(recall - stored["recall"]) > TOLERANCE:
            problems.append(f"S1a K={K}: {named}, {right}, {recall} here; {stored} stored")
    agree = [v for v in ids if len(set(sets[v])) == 1 and sets[v][0] is not None]
    others = [v for v in ids if v not in agree]
    mine = {"agreeing": len(agree), "agreeing_right": sum(sets[v][0] == gold[v] for v in agree),
            "other_answers": 10 * len(others), "other_answers_right": sum(s == gold[v] for v in others for s in sets[v])}
    if mine != step1["agreement"]:
        problems.append(f"agreement: {mine} here, {step1['agreement']} stored")
    thresholds = {}
    for f in range(FOLDS):
        rest = [v for v in ids if fold_of[v] != f]
        thresholds[str(f)] = next((K for K in range(2, 11) if (pr(rest, K)[2] or 0) >= 0.90), None)
    if thresholds != step1["thresholds"]:
        problems.append(f"thresholds: {thresholds} here, {step1['thresholds']} stored")

    def majority(items: list):
        """The most frequent item, ties to the earliest.

        Args: items: the items.
        Returns: the item.
        Raises: nothing.
        Side Effects: none.
        """
        tally = Counter(items)
        top = max(tally.values())
        return next(i for i in items if tally[i] == top)

    model_sets = {v: [frozenset(row["type"] for row in answer) for answer in step1["nearest"][v]] for v in ids}
    for name, chosen in (("finder", sets), ("no model", model_sets)):
        right = sum(s == gold[v] for v in ids for s in chosen[v])
        tp = sum(len((s or frozenset()) & gold[v]) for v in ids for s in chosen[v])
        fp = sum(len((s or frozenset()) - gold[v]) for v in ids for s in chosen[v])
        fn = sum(len(gold[v] - (s or frozenset())) for v in ids for s in chosen[v])
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        stored = step1["s1c"][name]
        if right != stored["sets_right"] or abs(f1 - stored["type_f1"]) > TOLERANCE:
            problems.append(f"S1c {name}: {right}, {f1} here; {stored} stored")
    flags = {"R": {v for v in ids if len(set(sets[v])) > 1},
             "N": {v for v in ids if majority(sets[v]) != majority(model_sets[v])}}
    flags["R or N"] = flags["R"] | flags["N"]
    rows = {}
    for signal, flagged in flags.items():
        accuracy = sum(1 for v in ids if v in flagged or sets[v][RUN_ONE] == gold[v]) / len(ids)
        rows[signal] = {"flagged": len(flagged) / len(ids), "accuracy": accuracy}
        stored = step1["s1d"][signal]
        if len(flagged) != stored["flagged_count"] or abs(accuracy - stored["accuracy"]) > TOLERANCE:
            problems.append(f"S1d {signal}: {len(flagged)}, {accuracy} here; {stored} stored")
    allowed = [s for s in ("R", "N", "R or N") if rows[s]["flagged"] <= 0.5]
    chosen_signal = (max(allowed, key=lambda s: (rows[s]["accuracy"], -("R", "N", "R or N").index(s))) if allowed
                     else min(("R", "N", "R or N"), key=lambda s: (rows[s]["flagged"], ("R", "N", "R or N").index(s))))
    if chosen_signal != step1["signal"] or sorted(flags[chosen_signal]) != step1["flagged"]:
        problems.append(f"signal: {chosen_signal} here, {step1['signal']} stored")
    print(f"Step 1 checked: signal {chosen_signal}, {len(flags[chosen_signal])} questions, thresholds {thresholds}")

    path = rc.OUT / "constraint-types-summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        arms = list(summary.get("arms", {}))
        calls = rc.kept(rc.OUT / "constraint-types-calls.jsonl")
        stored_scores = {(tuple(r["key"]), r["answer_sha256"]): r for r in rc.rows(rc.OUT / "constraint-types-scores.jsonl")}
        scores = {}
        for kk, r in calls.items():
            digest = hashlib.sha256(r["answer"].encode("utf-8")).hexdigest() if r.get("ok") and r.get("answer") is not None else "no-answer"
            if (kk, digest) in stored_scores:
                scores[kk] = stored_scores[(kk, digest)]

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
            for r in range(1, RUNS2 + 1):
                total = [sum((counts(scores[(arm, r, v)])[i] for v in ids), Fraction(0)) for i in range(3)]
                per_run.append(float(2 * total[0] / (2 * total[0] + total[1] + total[2])) if total[0] else 0.0)
            for field, value in (("f1_mean", statistics.fmean(per_run)), ("f1_sd", statistics.stdev(per_run))):
                if abs(value - summary["arms"][arm][field]) > TOLERANCE:
                    problems.append(f"Step 2 {arm} {field}: {value} here, {summary['arms'][arm][field]} stored")
            per_var[arm] = {v: [float(sum((counts(scores[(arm, r, v)])[i] for r in range(1, RUNS2 + 1)), Fraction(0)))
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

        for arm in arms:
            if arm == "best":
                continue
            rng = random.Random(rc.SEED2)
            drawn = sorted(pooled(arm, rng.choices(range(len(ids)), k=len(ids))) for _ in range(rc.DRAWS))
            mine2 = {"gain": pooled(arm, list(range(len(ids)))), "low": drawn[int(0.025 * rc.DRAWS)],
                     "high": drawn[int(0.975 * rc.DRAWS) - 1]}
            for field, value in mine2.items():
                theirs = summary["against_best"][arm][field]
                if not math.isclose(value, theirs, rel_tol=0.0, abs_tol=TOLERANCE):
                    problems.append(f"Step 2 {arm} {field}: {value} here, {theirs} stored")
        print("Step 2 checked arms", arms)
    for problem in problems[:40]:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
