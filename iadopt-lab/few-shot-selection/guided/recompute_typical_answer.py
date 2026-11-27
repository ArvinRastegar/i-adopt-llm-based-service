#!/usr/bin/env python
"""Independent recomputation of the most-typical-answer check (`typical_answer_check.py`).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/TYPICAL_ANSWER_CHECK_20261127.md` (private repository).

Standard library only. From the stored agreements and the stored scores: the kept answer of every group,
the pooled Close F1 of the kept answers and of one answer, and the intervals.

    ./recompute_typical_answer.py     # prints the comparison; exit status 1 on any mismatch
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recompute_conventions as rc  # noqa: E402

SOURCES = {"psnc": ("stand-in-decomposition-calls.jsonl", "stand-in-decomposition-scores.jsonl", "best"),
           "openrouter": ("question-budget-calls.jsonl", "question-budget-scores.jsonl", "ask-0")}
GROUPS = {3: ((1, 2, 3), (4, 5, 6), (7, 8, 9)), 5: ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))}
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
    ids = sorted(key)
    agree = json.loads((rc.OUT / "typical-answer-agreements.json").read_text(encoding="utf-8"))
    summary = json.loads((rc.OUT / "typical-answer-summary.json").read_text(encoding="utf-8"))
    for name, (calls_file, scores_file, arm) in SOURCES.items():
        calls = rc.kept(rc.OUT / calls_file)
        scored = {(tuple(r["key"]), r["answer_sha256"]): r for r in rc.rows(rc.OUT / scores_file)}

        def counts(run: int, v: str) -> list[Fraction]:
            """Close TP, FP, FN of one stored answer.

            Args: run: the run; v: the variable.
            Returns: [TP, FP, FN].
            Raises: KeyError when the score is missing.
            Side Effects: none.
            """
            r = calls[(arm, run, v)]
            digest = hashlib.sha256(r["answer"].encode("utf-8")).hexdigest() if r.get("ok") and r.get("answer") is not None else "no-answer"
            t = scored[((arm, run, v), digest)]["evaluation"]["close"]["totals"]
            return [Fraction(t[n]["numerator"], t[n]["denominator"]) for n in ("tp", "fp", "fn")]

        for k, groups in GROUPS.items():
            kept, one, runs = {}, {}, {}
            for v in ids:
                kept[v], one[v], runs[v] = [0.0] * 3, [0.0] * 3, []
                for g in groups:
                    means = {}
                    for run in g:
                        others = [o for o in g if o != run]
                        means[run] = sum(agree[name][v][f"{min(run, o)}-{max(run, o)}"] for o in others) / len(others)
                    best = max(means.values())
                    chosen = min(run for run in g if means[run] == best)
                    runs[v].append(chosen)
                    kept[v] = [a + float(b) for a, b in zip(kept[v], counts(chosen, v), strict=True)]
                    avg = [sum(counts(run, v)[i] for run in g) / len(g) for i in range(3)]
                    one[v] = [a + float(b) for a, b in zip(one[v], avg, strict=True)]

            def pooled(table: dict, sample: list[str]) -> float:
                """Pooled F1 over a sample of variables.

                Args: table: {variable: [TP, FP, FN]}; sample: variables.
                Returns: the F1.
                Raises: nothing.
                Side Effects: none.
                """
                tp, fp, fn = (sum(table[v][i] for v in sample) for i in range(3))
                return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

            rng = random.Random(rc.SEED2)
            drawn = []
            for _ in range(rc.DRAWS):
                sample = rng.choices(ids, k=len(ids))
                drawn.append(pooled(kept, sample) - pooled(one, sample))
            drawn.sort()
            mine = {"kept_f1": pooled(kept, ids), "one_answer_f1": pooled(one, ids),
                    "gain": pooled(kept, ids) - pooled(one, ids), "low": drawn[int(0.025 * rc.DRAWS)],
                    "high": drawn[int(0.975 * rc.DRAWS) - 1]}
            theirs = summary["results"][f"{name}/k{k}"]
            for field, value in mine.items():
                if not math.isclose(value, theirs[field], rel_tol=0.0, abs_tol=TOLERANCE):
                    problems.append(f"{name} k={k} {field}: {value} here, {theirs[field]} stored")
            if runs != theirs["kept_runs"]:
                problems.append(f"{name} k={k}: the kept runs differ")
            print(f"{name} k={k}: gain {mine['gain']:+.4f} [{mine['low']:+.4f}, {mine['high']:+.4f}]")
    for problem in problems[:40]:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
