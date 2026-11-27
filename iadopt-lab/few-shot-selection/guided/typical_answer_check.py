#!/usr/bin/env python
"""Pick the most typical of several stored answers: does it beat using one answer?

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/TYPICAL_ANSWER_CHECK_20261127.md` (private repository).

Arvin, 2026-11-27: "run until work that needs people" (the free check first among the remaining items).

No model calls. Today's best pipeline has ten stored answers per variable in two runs: on PSNC (the
stand-in test's `best` arm, primary) and on OpenRouter (the question-budget test's `ask-0`, the same
prompts, replication). The runs are split into groups of three (1-3, 4-6, 7-9) and of five (1-5, 6-10).
In each group the answer with the highest mean agreement with the others is kept; agreement between two
answers is the frozen scorer's Close F1 of one against the other, averaged over both directions, and an
unusable answer agrees with nothing. The kept answers are scored against the experts as stored, and
compared with the mean of the group's answers (what one answer gives).

    ./typical_answer_check.py --compute     # writes the agreements and the summary
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import engine  # noqa: E402
import rag_tuning as rt  # noqa: E402
from engine import ft  # noqa: E402

OUT = HERE.parent / "output" / "retrieval"
SOURCES = {"psnc": (OUT / "stand-in-decomposition-calls.jsonl", OUT / "stand-in-decomposition-scores.jsonl", "best"),
           "openrouter": (OUT / "question-budget-calls.jsonl", OUT / "question-budget-scores.jsonl", "ask-0")}
GROUPS = {3: ((1, 2, 3), (4, 5, 6), (7, 8, 9)), 5: ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))}
DRAWS, SEED = 10000, 20261130
AGREEMENTS = OUT / "typical-answer-agreements.json"
SUMMARY = OUT / "typical-answer-summary.json"


def f1(totals: Mapping[str, Mapping[str, int]]) -> float:
    """Close F1 from a scorer's totals.

    Args: totals: {"tp", "fp", "fn"} as numerator/denominator.
    Returns: the F1; 0 when there is no true positive.
    Raises: KeyError when a count is missing.
    Side Effects: none. Deterministic.
    """
    tp, fp, fn = (Fraction(totals[n]["numerator"], totals[n]["denominator"]) for n in ("tp", "fp", "fn"))
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def counts(row: Mapping[str, Any]) -> list[Fraction]:
    """Close TP, FP, FN of a stored score row.

    Args: row: a score row.
    Returns: [TP, FP, FN].
    Raises: KeyError when a count is missing.
    Side Effects: none. Deterministic.
    """
    t = row["evaluation"]["close"]["totals"]
    return [Fraction(t[n]["numerator"], t[n]["denominator"]) for n in ("tp", "fp", "fn")]


def pick(group: Sequence[int], agreement: Mapping[tuple[int, int], float]) -> int:
    """The most typical run of a group.

    Args: group: run numbers; agreement: {(run, other run): mean agreement}, symmetric.
    Returns: the run with the highest mean agreement with the others; ties go to the lowest run number.
    Raises: KeyError when a pair is missing.
    Side Effects: none. Deterministic.
    """
    def mean(run: int) -> float:
        """Mean agreement of one run with the others of the group.

        Args: run: the run.
        Returns: the mean.
        Raises: KeyError when a pair is missing.
        Side Effects: none.
        """
        others = [o for o in group if o != run]
        return sum(agreement[(min(run, o), max(run, o))] for o in others) / len(others)

    return max(group, key=lambda run: (mean(run), -run))


def agreements(scores: Mapping[engine.Job, Mapping[str, Any]], arm: str, ids: Sequence[str],
               similarity: Any) -> dict[str, dict[str, float]]:
    """Every needed pairwise agreement of one stored run.

    Args: scores: from `engine.score`; arm: the arm; ids: the variables; similarity: the scorer's backend.
    Returns: {variable: {"a-b": agreement}} for every pair of runs inside a group of `GROUPS`.
    Raises: KeyError when a score is missing.
    Side Effects: embeds texts through the backend.
    """
    from iadopt_eval import evaluate_item

    pairs = sorted({(a, b) for groups in GROUPS.values() for g in groups for a in g for b in g if a < b})
    out: dict[str, dict[str, float]] = {}
    for v in ids:
        out[v] = {}
        for a, b in pairs:
            first, second = scores[(arm, a, v)]["prediction"], scores[(arm, b, v)]["prediction"]
            if first is None or second is None:
                out[v][f"{a}-{b}"] = 0.0
                continue
            both = [f1(evaluate_item(gold, pred, similarity, metadata={"variable_id": v},
                                     similarity_identity=similarity.identity)["close"]["totals"])
                    for gold, pred in ((first, second), (second, first))]
            out[v][f"{a}-{b}"] = sum(both) / 2
    return out


def compare(scores: Mapping[engine.Job, Mapping[str, Any]], arm: str, ids: Sequence[str],
            agree: Mapping[str, Mapping[str, float]], k: int) -> dict[str, Any]:
    """The kept answers against the mean of their groups, pooled over variables.

    Args: scores: from `engine.score`; arm: the arm; ids: the variables; agree: from `agreements`; k: group size.
    Returns: {"kept_f1", "one_answer_f1", "gain", "low", "high", "kept_runs"}: pooled Close F1 summed over
        the groups, the difference with a 95% interval from resampling the variables (`DRAWS`, `SEED`).
    Raises: KeyError when a score or an agreement is missing.
    Side Effects: none. Deterministic.
    """
    kept: dict[str, list[float]] = {}
    one: dict[str, list[float]] = {}
    runs: dict[str, list[int]] = {}
    for v in ids:
        table = {tuple(int(x) for x in p.split("-")): a for p, a in agree[v].items()}
        kept[v], one[v], runs[v] = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], []
        for group in GROUPS[k]:
            chosen = pick(group, table)
            runs[v].append(chosen)
            kept[v] = [x + float(c) for x, c in zip(kept[v], counts(scores[(arm, chosen, v)]), strict=True)]
            mean = [sum(counts(scores[(arm, r, v)])[i] for r in group) / len(group) for i in range(3)]
            one[v] = [x + float(c) for x, c in zip(one[v], mean, strict=True)]

    def pooled(table: Mapping[str, list[float]], sample: Sequence[str]) -> float:
        """Pooled F1 over a sample of variables.

        Args: table: {variable: [TP, FP, FN]}; sample: variables, with repeats.
        Returns: the F1.
        Raises: nothing.
        Side Effects: none.
        """
        tp = sum(table[v][0] for v in sample)
        fp = sum(table[v][1] for v in sample)
        fn = sum(table[v][2] for v in sample)
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    rng = random.Random(SEED)
    drawn = []
    for _ in range(DRAWS):
        sample = rng.choices(list(ids), k=len(ids))
        drawn.append(pooled(kept, sample) - pooled(one, sample))
    drawn.sort()
    return {"kept_f1": pooled(kept, ids), "one_answer_f1": pooled(one, ids), "gain": pooled(kept, ids) - pooled(one, ids),
            "low": drawn[int(0.025 * DRAWS)], "high": drawn[int(0.975 * DRAWS) - 1], "kept_runs": runs}


def main(argv: Sequence[str] | None = None) -> int:
    """Compute the agreements and the comparisons.

    Args: argv: arguments; `sys.argv` when None.
    Returns: 0.
    Raises: SystemExit on bad arguments.
    Side Effects: reads the stored scores, loads the scorer's similarity model, writes `AGREEMENTS` and `SUMMARY`.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compute", action="store_true", required=True)
    parser.parse_args(argv)
    from iadopt_lab.cli import _configuration, _similarity_backend

    ctx = rt.load()
    ids = ctx["ids"]
    similarity = _similarity_backend(engine.ROOT, _configuration(engine.ROOT).data["evaluation"])
    stored: dict[str, Any] = {}
    summary: dict[str, Any] = {"generated_at": ft.now(), "sources": {}, "results": {}}
    for name, (calls, scores_path, arm) in SOURCES.items():
        scores = engine.score(calls, scores_path, ctx["records"], ctx["inputs"]["schema"])
        stored[name] = agreements(scores, arm, ids, similarity)
        summary["sources"][name] = {"calls": str(calls.name), "arm": arm,
                                    "unusable": sum(1 for k, s in scores.items() if k[0] == arm and not s["valid_prediction"])}
        for k in GROUPS:
            row = compare(scores, arm, ids, stored[name], k)
            summary["results"][f"{name}/k{k}"] = {x: row[x] for x in ("kept_f1", "one_answer_f1", "gain", "low", "high")}
            summary["results"][f"{name}/k{k}"]["kept_runs"] = row["kept_runs"]
            print(f"{name} k={k}: kept {row['kept_f1']:.4f} one {row['one_answer_f1']:.4f} gain {row['gain']:+.4f} "
                  f"[{row['low']:+.4f}, {row['high']:+.4f}]")
    AGREEMENTS.write_text(json.dumps(stored, indent=0) + "\n", encoding="utf-8")
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
