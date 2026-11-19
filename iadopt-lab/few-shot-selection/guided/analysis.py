#!/usr/bin/env python
"""Secondary views of a guided-decomposition comparison: by part, by variable, arm against arm.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.

These read the stored score records of a finished comparison and change nothing. The primary
analysis lives in `engine.summarise`; what is printed here is descriptive.

    ./analysis.py --name ceiling --baseline base --contrast form+types:form --contrast full:form+names+types
    ./analysis.py --name groups-held-out --baseline base --bootstrap   # does a gain hold across variables?
    ./analysis.py --pool groups-examples:ARM:base tune:ARM:base groups-held-out:ARM:base   # several groups together
    ./analysis.py --neighbours     # post hoc: do similar definitions share the matrix decision?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine  # noqa: E402
from engine import ft  # noqa: E402

PARTS = ("hasProperty", "hasObjectOfInterest", "hasMatrix", "hasContextObject",
         "hasStatisticalModifier", "hasConstraint")
NEIGHBOUR_COUNTS = (1, 3, 5, 7)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10000, 20261119
NEIGHBOURS_FILE = engine.OUTPUT / "neighbours.json"


def _f1(tp: Fraction, fp: Fraction, fn: Fraction) -> tuple[float, float, float]:
    """Precision, recall and F1 from exact counts.

    Args: tp, fp, fn: summed counts.
    Returns: (precision, recall, F1), each 0 when undefined.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    precision = tp / (tp + fp) if tp + fp else Fraction(0)
    recall = tp / (tp + fn) if tp + fn else Fraction(0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else Fraction(0)
    return float(precision), float(recall), float(f1)


def _counts(node: dict[str, Any]) -> list[Fraction]:
    """Exact TP, FP and FN of one stored contribution block.

    Args: node: a mapping with "tp", "fp" and "fn" receipts.
    Returns: [TP, FP, FN].
    Raises: KeyError when a count is missing.
    Side Effects: none. Deterministic.
    """
    return [Fraction(node[k]["numerator"], node[k]["denominator"]) for k in ("tp", "fp", "fn")]


def part_scores(scores: dict[engine.Job, dict]) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Micro Close precision, recall and F1 per arm and part, pooled over all answers.

    Args: scores: from `engine.score`.
    Returns: {arm: {part: (precision, recall, F1)}}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    totals: dict[str, dict[str, list[Fraction]]] = defaultdict(lambda: {p: [Fraction(0)] * 3 for p in PARTS})
    for (arm, _, _), row in scores.items():
        components = row["evaluation"]["close"]["components"]
        for part in PARTS:
            add = _counts(components[part]["contributions"])
            totals[arm][part] = [a + b for a, b in zip(totals[arm][part], add, strict=True)]
    return {arm: {part: _f1(*values) for part, values in parts.items()} for arm, parts in totals.items()}


def variable_means(scores: dict[engine.Job, dict]) -> dict[str, dict[str, float]]:
    """Mean per-answer Close F1 of every variable, per arm.

    Args: scores: from `engine.score`.
    Returns: {arm: {variable_id: mean F1 over repetitions}}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (arm, _, variable_id), row in scores.items():
        values[arm][variable_id].append(_f1(*_counts(row["evaluation"]["close"]["totals"]))[2])
    return {arm: {v: statistics.fmean(xs) for v, xs in per.items()} for arm, per in values.items()}


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact sign test.

    Args: better, worse: counts of variables that moved each way; ties are left out.
    Returns: the p-value; 1.0 when nothing moved.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    n, k = better + worse, max(better, worse)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def by_variable(scores: dict[engine.Job, dict], arm: str, baseline: str) -> dict[str, Any]:
    """How many variables an arm improves and worsens against another arm.

    Args: scores: from `engine.score`; arm, baseline: the two arms.
    Returns: {"better", "worse", "same", "p", "mean_gain", "median_gain", "gains": {id: gain}}.
    Raises: KeyError when an arm has no scores.
    Side Effects: none. Deterministic.
    """
    means = variable_means(scores)
    gains = {v: means[arm][v] - means[baseline][v] for v in means[baseline] if v in means[arm]}
    better = sum(g > 0 for g in gains.values())
    worse = sum(g < 0 for g in gains.values())
    ordered = sorted(gains.values())
    return {"better": better, "worse": worse, "same": len(gains) - better - worse,
            "p": sign_test(better, worse), "mean_gain": statistics.fmean(ordered),
            "median_gain": statistics.median(ordered), "gains": gains}


def variable_counts(scores: dict[engine.Job, dict]) -> dict[str, dict[str, list[float]]]:
    """Each variable's Close counts summed over its repetitions, per arm.

    Args: scores: from `engine.score`.
    Returns: {arm: {variable_id: [TP, FP, FN]}} as floats.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    totals: dict[str, dict[str, list[Fraction]]] = defaultdict(lambda: defaultdict(lambda: [Fraction(0)] * 3))
    for (arm, _, variable_id), row in scores.items():
        add = _counts(row["evaluation"]["close"]["totals"])
        totals[arm][variable_id] = [a + b for a, b in zip(totals[arm][variable_id], add, strict=True)]
    return {arm: {v: [float(x) for x in counts] for v, counts in per.items()} for arm, per in totals.items()}


def pooled_gain(first: dict[str, list[float]], second: dict[str, list[float]], variables: Sequence[str]) -> float:
    """Micro Close F1 of one arm minus another over a list of variables, each counted as often as listed.

    Args: first, second: {variable_id: [TP, FP, FN]} of the two arms; variables: the ids to pool.
    Returns: F1(first) - F1(second), each from the summed counts.
    Raises: KeyError when an arm lacks a listed variable.
    Side Effects: none. Deterministic.
    """
    f1 = []
    for arm in (first, second):
        tp, fp, fn = (sum(arm[v][k] for v in variables) for k in range(3))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return f1[0] - f1[1]


def bootstrap_over_variables(scores: dict[engine.Job, dict], arm: str, baseline: str,
                             draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    """How far an arm's gain depends on which variables were scored.

    The comparison over repetitions keeps the variables fixed, so it answers "is this run-to-run
    noise?". This resamples the variables with replacement instead and answers "would the gain
    hold on other variables like these?".

    Args: scores: from `engine.score`; arm, baseline: the two arms; draws: resamples; seed: of the
        resampling.
    Returns: {"gain": pooled F1 difference over all variables and repetitions, "low", "high": the
        2.5% and 97.5% points of the resampled differences, "share_positive"}.
    Raises: KeyError when an arm has no scores.
    Side Effects: none. Deterministic for a given seed.
    """
    counts = variable_counts(scores)
    variables = sorted(counts[baseline])
    rng = random.Random(seed)
    gains = sorted(pooled_gain(counts[arm], counts[baseline], rng.choices(variables, k=len(variables)))
                   for _ in range(draws))
    return {"gain": pooled_gain(counts[arm], counts[baseline], variables),
            "low": gains[int(0.025 * draws)], "high": gains[int(0.975 * draws) - 1],
            "share_positive": sum(g > 0 for g in gains) / draws}


def load_scores(name: str) -> dict[engine.Job, dict]:
    """The stored score of every answer of one comparison.

    Args: name: the comparison's file prefix.
    Returns: {(arm, repetition, variable_id): row}; a re-scored answer keeps its last row.
    Raises: OSError when the scores file cannot be read.
    Side Effects: reads the scores file.
    """
    return {tuple(row["key"]): row for row in ft.read_jsonl(engine.OUTPUT / f"{name}-scores.jsonl")}


def pooled_bootstrap(parts: Sequence[tuple[dict[str, list[float]], dict[str, list[float]]]],
                     draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    """One setup against today's prompt over several groups of variables put together.

    Each group brings the setup's counts and the counts of the baseline of the same run, so a
    setup is never compared with a baseline from another run. The variables of all groups are
    then resampled together.

    Args: parts: per group, ({variable_id: [TP, FP, FN]} of the setup, the same of its baseline);
        the groups must not share a variable; draws, seed: of the resampling.
    Returns: {"variables", "gain", "low", "high", "share_positive"} as in `bootstrap_over_variables`.
    Raises: ValueError when two groups share a variable or a group's two sides differ in variables.
    Side Effects: none. Deterministic for a given seed.
    """
    first: dict[str, list[float]] = {}
    second: dict[str, list[float]] = {}
    for arm, baseline in parts:
        if set(arm) != set(baseline) or set(arm) & set(first):
            raise ValueError("the groups overlap or a group's two sides hold different variables")
        first.update(arm)
        second.update(baseline)
    variables = sorted(first)
    rng = random.Random(seed)
    gains = sorted(pooled_gain(first, second, rng.choices(variables, k=len(variables))) for _ in range(draws))
    return {"variables": len(variables), "gain": pooled_gain(first, second, variables),
            "low": gains[int(0.025 * draws)], "high": gains[int(0.975 * draws) - 1],
            "share_positive": sum(g > 0 for g in gains) / draws}


def contrast(summary: dict[str, Any], first: str, second: str) -> dict[str, float]:
    """Welch's test of one arm against another on per-repetition Close F1.

    Args: summary: from `engine.summarise`; first, second: arm names.
    Returns: the test of first minus second.
    Raises: KeyError when an arm is missing.
    Side Effects: none. Deterministic.
    """
    series = {arm: [r["close_f1_official"] for r in summary["cells"][arm]["per_repetition"]]
              for arm in (first, second)}
    return ft.welch(series[first], series[second])


def neighbour_votes(scores: Sequence[Sequence[float]], labels: Sequence[bool], targets: Sequence[int],
                    k: int) -> int:
    """Count the targets whose label a vote of their k most similar other items gets right.

    Args: scores: square similarity matrix; labels: one boolean per item; targets: indices of the
        items to predict; k: how many neighbours vote.
    Returns: the number of targets predicted correctly.
    Raises: ValueError when k is even (a vote could tie) or not smaller than the number of items.
    Side Effects: none. Deterministic: equal similarities are ordered by index.
    """
    if k % 2 == 0 or k >= len(labels):
        raise ValueError("k must be odd and smaller than the number of items")
    right = 0
    for target in targets:
        nearest = sorted((i for i in range(len(labels)) if i != target),
                         key=lambda i: (-scores[target][i], i))[:k]
        right += (2 * sum(labels[i] for i in nearest) > k) == labels[target]
    return right


def neighbours() -> dict[str, Any]:
    """Post hoc check: can the matrix and context decisions be read off similar definitions?

    Each practice variable is predicted from the k most similar definitions among the other 77
    variables that are not held out (the 25 examples and the other 52 practice variables), with the
    scorer's local similarity model and the expert answer key. No model call.

    Args: none.
    Returns: {"pool", "targets", "always_no_matrix", "right": {k: {"matrix", "context"}}} as counts.
    Raises: as `walk.load` and the scorer's model loader.
    Side Effects: reads local files and loads the local similarity model. Deterministic.
    """
    import walk

    from iadopt_lab.cli import _configuration, _similarity_backend

    ctx = walk.load()
    pool = sorted(set(ft.EXAMPLE_IDS) | set(ctx["practice"]))
    similarity = _similarity_backend(engine.ROOT, _configuration(engine.ROOT).data["evaluation"])
    texts = [ctx["records"][v]["definition"] for v in pool]
    scores = [[similarity(a, b) for b in texts] for a in texts]
    targets = [i for i, v in enumerate(pool) if v in set(ctx["practice"])]
    labels = {part: [bool(ctx["key"][v][part]) for v in pool] for part in ("matrix", "context")}
    return {"pool": len(pool), "targets": len(targets),
            "always_no_matrix": sum(not labels["matrix"][i] for i in targets),
            "right": {str(k): {part: neighbour_votes(scores, labels[part], targets, k) for part in labels}
                      for k in NEIGHBOUR_COUNTS}}


def main(argv: list[str] | None = None) -> int:
    """Print the secondary views of one stored comparison, or run the neighbour check.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: 0.
    Raises: SystemExit on bad arguments; OSError when the comparison's files are missing.
    Side Effects: reads output files; prints to standard output; --neighbours writes its result.
    """
    parser = argparse.ArgumentParser(description="Secondary views of a comparison (exploratory).")
    parser.add_argument("--name")
    parser.add_argument("--baseline")
    parser.add_argument("--contrast", action="append", default=[], help="first:second")
    parser.add_argument("--neighbours", action="store_true", help="post hoc nearest-neighbour check")
    parser.add_argument("--bootstrap", action="store_true", help="resample the variables for each arm's gain")
    parser.add_argument("--pool", nargs="+", metavar="NAME:ARM:BASELINE",
                        help="one setup over several stored comparisons, each against its own baseline")
    args = parser.parse_args(argv)
    if args.neighbours:
        result = neighbours()
        NEIGHBOURS_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        for k, right in result["right"].items():
            print(f"k={k}: " + "  ".join(f"{part} {count}/{result['targets']} = {count / result['targets']:.2f}"
                                         for part, count in right.items()))
        print(f"always no matrix: {result['always_no_matrix']}/{result['targets']}")
        return 0
    if args.pool:
        parts = []
        for source in args.pool:
            name, arm, baseline = source.split(":")
            counts = variable_counts(load_scores(name))
            parts.append((counts[arm], counts[baseline]))
        view = pooled_bootstrap(parts)
        print(f"{view['variables']} variables: {view['gain']:+.4f} [{view['low']:+.4f}, {view['high']:+.4f}]  "
              f"positive in {view['share_positive']:.0%} of resamples")
        return 0
    if not (args.name and args.baseline):
        parser.error("--name and --baseline are required unless --neighbours or --pool is given")
    scores = load_scores(args.name)
    summary = json.loads((engine.OUTPUT / f"{args.name}-summary.json").read_text(encoding="utf-8"))
    parts = part_scores(scores)
    print("Close F1 by part (precision/recall in brackets):")
    for arm in summary["cells"]:
        print(f"  {arm:20} " + "  ".join(f"{part[3:9]} {parts[arm][part][2]:.3f}" for part in PARTS))
    print("precision / recall by part:")
    for arm in summary["cells"]:
        print(f"  {arm:20} " + "  ".join(f"{part[3:9]} {parts[arm][part][0]:.2f}/{parts[arm][part][1]:.2f}" for part in PARTS))
    print("variables against the baseline:")
    for arm in summary["cells"]:
        if arm != args.baseline:
            view = by_variable(scores, arm, args.baseline)
            print(f"  {arm:20} better {view['better']:2}  worse {view['worse']:2}  same {view['same']:2}  "
                  f"sign p {view['p']:.4f}  mean gain {view['mean_gain']:+.3f}  median {view['median_gain']:+.3f}")
    if args.bootstrap:
        print(f"gain over {args.baseline}, variables resampled {BOOTSTRAP_DRAWS} times (95% interval):")
        for arm in summary["cells"]:
            if arm != args.baseline:
                view = bootstrap_over_variables(scores, arm, args.baseline)
                print(f"  {arm:38} {view['gain']:+.4f} [{view['low']:+.4f}, {view['high']:+.4f}]  "
                      f"positive in {view['share_positive']:.0%} of resamples")
    for pair in args.contrast:
        first, second = pair.split(":")
        result = contrast(summary, first, second)
        print(f"contrast {first} minus {second}: {result['difference']:+.4f} "
              f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] p {result['p']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
