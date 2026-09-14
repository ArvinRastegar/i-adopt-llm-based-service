#!/usr/bin/env python
"""Entry point for the example-selection optimization — see README.md and contracts/.

EXPLORATORY side experiment. Writes no campaign, run or task row; reads nothing from
`parameters.yml`; writes only under `few-shot-selection/example_selection/output/`. Its results
cannot enter an official ranking.

    ./run.py --plan-only            # budget and split, no provider calls
    ./run.py --subsets 12 --reps 2  # end-to-end smoke test
    ./run.py                        # full run, resumable
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import design  # noqa: E402
import pipeline  # noqa: E402
from attribution import ranking  # noqa: E402
from harness import build_context  # noqa: E402

from iadopt_lab.corpus.ingestion import load_canonical_records  # noqa: E402


def _falsification(confirm: dict) -> dict:
    """Decide whether the attribution model learned anything real.

    Args: confirm: per-candidate held-out statistics from stage D.
    Returns: {"passed": bool, "margin_bottom": float, "margin_random": float}.
    Raises: nothing.
    Side Effects: none. The top-25 must beat BOTH controls; the bottom-25 arm is attenuated
        by its necessary 10-member overlap, so the random arm carries equal weight.
    """
    top = confirm.get("top25", {}).get("mean", 0.0)
    bottom = confirm.get("bottom25", {}).get("mean", 0.0)
    randoms = [row["mean"] for name, row in confirm.items() if name.startswith("random-")]
    random_mean = statistics.mean(randoms) if randoms else 0.0
    return {"passed": top > bottom and top > random_mean,
            "margin_bottom": top - bottom, "margin_random": top - random_mean}


async def main() -> int:
    parser = argparse.ArgumentParser(description="example-selection optimization")
    parser.add_argument("--subsets", type=int, default=pipeline.DEFAULT_SUBSETS)
    parser.add_argument("--reps", type=int, default=pipeline.DEFAULT_SEARCH_REPS)
    parser.add_argument("--confirm-reps", type=int, default=pipeline.DEFAULT_CONFIRM_REPS)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    records = list(load_canonical_records(ROOT))
    split = design.corpus_split(records)
    planned = pipeline.planned_calls(args.subsets, args.reps, args.confirm_reps,
                                     n_candidates=6, n_eval=len(split["E"]),
                                     n_confirm=len(split["C"]))
    print(f"split: P={len(split['P'])}  E={len(split['E'])}  C={len(split['C'])}")
    print(f"planned provider calls: {planned:,}  (~{planned / 6 / 60:.0f} min at concurrency 24)")
    print(f"ceiling: {pipeline.CALL_CEILING:,}")
    if args.plan_only:
        return 0

    context = build_context(ROOT)
    # The budget gate must cost the run the caller actually asked for, not the defaults (OQ-1).
    context.search_reps, context.confirm_reps = args.reps, args.confirm_reps
    print("\n[A] attribution", flush=True)
    attribution = await pipeline.stage_attribution(context, split, args.subsets)
    coefficients = attribution["fit"]["coefficients"]
    print(f"    r2 {attribution['fit']['r2']:.3f}   split-half {attribution['fit']['split_half']:.3f}"
          f"   n {attribution['fit']['n']}", flush=True)

    print("[B] construct", flush=True)
    construct = await pipeline.stage_construct(context, split, coefficients, reps=args.reps)
    for name, values in sorted(construct["scores"].items()):
        print(f"    {name:<14} E {statistics.mean(values):.4f} (selection-biased)", flush=True)

    print("[C] refine", flush=True)
    seeded = {name: {"subset": construct["candidates"][name], "scores": values}
              for name, values in construct["scores"].items()}
    refine = await pipeline.stage_refine(context, split, seeded, reps=args.reps)

    print("[D] confirm on held-out C", flush=True)
    finalists = dict(construct["candidates"])
    finalists["refined"] = refine["best"]
    confirm = await pipeline.stage_confirm(context, split, finalists, reps=args.confirm_reps)

    results = {"split": {k: len(v) for k, v in split.items()},
               "search": {n: {"mean": statistics.mean(v)} for n, v in construct["scores"].items()},
               "confirm": confirm, "falsification": _falsification(confirm)}
    report = HERE / "output" / "example-selection-report.md"
    pipeline.write_report(report, results)

    print("\nheld-out C (the result):")
    for name, row in sorted(confirm.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"    {name:<14} {row['mean']:.4f} +/- {row['sd']:.4f}")
    print(f"\nfalsification passed: {results['falsification']['passed']}")
    print("top 5 examples by contribution:")
    for variable_id, coefficient in ranking(coefficients)[:5]:
        label = context.records[variable_id].get("label") or ""
        print(f"    {coefficient:+.4f}  {label[:60]}")
    print(f"\nreport: {report.relative_to(ROOT)}")
    if context.client is not None:
        await context.client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
