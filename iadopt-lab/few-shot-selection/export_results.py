"""Export the small public tables in `results/` from the stored answers (which are not in Git).

Run from `iadopt-lab/`: `uv run python -I few-shot-selection/export_results.py`.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from iadopt_eval.core import aggregate_items

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
RESULTS = HERE.parents[1] / "results"
SHOTS = (0, 1, 3, 5, 7, 10, 15, 20, 25, 30)
SETTINGS = {  # retrieval arm -> plain name
    "base": "One fixed set of 25 examples",
    "rag-10": "10 most similar examples",
    "rag-25": "25 most similar examples",
    "rag-25+retry": "25 most similar, unusable answers asked again once",
    "rag-50": "50 most similar examples",
    "rag-85": "All 85 examples from the other folds",
    "rag-25-first": "25 most similar, most similar first",
    "fixed+rag-10": "The fixed 25 plus the 10 most similar",
    "rag-25-mmr": "25 similar and varied examples",
    "rag-25-labels": "25 examples picked with a draft's labels",
    "rag-25-words": "25 most similar by shared words",
}


def shot_count() -> list[dict]:
    """Micro Close F1 by number of worked examples, over the scored answers (Qwen3.8-27B, 72 variables)."""
    path = OUTPUT / "shot-pool-30-ablation-results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    table = []
    for shots in SHOTS:
        batch = [r for r in rows if r["shot_count"] == shots]
        scored = [r for r in batch if r.get("evaluation")]
        total = aggregate_items([r["evaluation"] for r in scored],
                                expected_variable_ids=[r["variable_id"] for r in scored])
        table.append({"examples": shots, "answers": len(batch), "scored": len(scored),
                      "close_f1": round(total["close"]["metrics"]["f1"]["value"], 4)})
    return table


def retrieval() -> list[dict]:
    """Close F1 of each way of choosing the examples, on the 102 variables, ten runs each.

    Gains are against the fixed 25 and against the 25 most similar, with 95% intervals from
    resampling the variables. The last row is the setting chosen fold by fold on the other folds.
    """
    summary = json.loads((OUTPUT / "retrieval" / "tuning-summary.json").read_text(encoding="utf-8"))
    chosen = summary["chosen"]
    scores = {**summary["arms"], "chosen": {"f1_mean": chosen["f1_mean"],
                                            "f1_sd": statistics.stdev(chosen["f1_per_repetition"])}}
    vs_fixed = {**summary["against_base"], "chosen": summary["comparisons"]["chosen vs base"]}
    vs_25 = {**summary["against_reference"], "chosen": summary["comparisons"]["chosen vs rag-25"]}
    table = []
    for arm, name in {**SETTINGS, "chosen": "Setting chosen on the other five folds"}.items():
        row = {"setting": arm, "name": name, "close_f1": round(scores[arm]["f1_mean"], 4),
               "sd": round(scores[arm]["f1_sd"], 4)}
        for prefix, gains in (("vs_fixed", vs_fixed), ("vs_25_nearest", vs_25)):
            gain = gains.get(arm)
            row.update({f"{prefix}_{key}": round(gain[key], 4) if gain else ""
                        for key in ("gain", "low", "high")})
        table.append(row)
    return table


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "shot-count.csv", shot_count())
    write_csv(RESULTS / "pattern-retrieval.csv", retrieval())
    print(f"wrote {RESULTS / 'shot-count.csv'} and {RESULTS / 'pattern-retrieval.csv'}")
