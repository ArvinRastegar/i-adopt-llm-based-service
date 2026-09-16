#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
# I-ADOPT - Rebuild the Phase-1 error-analysis workbook from the lab database.
#
# Produces a workbook structurally identical to the template
#   error_analysis/excel_schema/onlyPhaseOne20251208_142819.xlsx
# (same sheets, columns, column order, JSON conventions, Summary schema), but
# populated from PostgreSQL experiment records instead of live LLM calls.
#
# Every metric is computed by importing the ORIGINAL scorer from
#   misc/benchmarking_example/onlyPhaseOne.py
# so no metric definition is re-implemented or allowed to drift.
#
# Two stages, because no single project virtualenv carries every dependency:
#   1. fetch  - needs psycopg          (iadopt-lab/.venv)
#   2. build  - needs pandas + scorer  (.venv)
#
#   iadopt-lab/.venv/bin/python export_from_database.py fetch --records recs.json
#   .venv/bin/python           export_from_database.py build --records recs.json
# --------------------------------------------------------------------------- #
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

REPO = pathlib.Path(__file__).resolve().parents[2]
LAB = REPO / "iadopt-lab"
BENCH = REPO / "misc" / "benchmarking_example"
HERE = pathlib.Path(__file__).resolve().parent

sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(BENCH))

# --------------------------------------------------------------------------- #
# Target experiment. Campaign 79b5ec5f is the only campaign with full coverage
# of this grid point; the other two sharing it hold 4-9 complete tasks each.
# --------------------------------------------------------------------------- #
CAMPAIGN_PREFIX = "79b5ec5f"
MODEL_ID = "z-ai/glm-5.2"
TEMPERATURE = 0.5
SHOT_COUNT = 5

# Block order mirrors the template's ordering convention
# (strict_minimal -> object_matrix_tree -> constraint_first).
PROMPT_VARIANTS = ["strict-minimal", "matrix-decomposition", "constraint-decomposition"]

# The prompt_version strings written into the workbook. Default is the exact
# database identifier, which is what the data actually is.
#
# The template workbook predates two renamings of these prompt families, so the
# consuming scripts still filter on the oldest generation of names:
#   metrics.py, conformance_*.py  -> "strict_minimal"
#   metrics_jaccard.py            -> "constraint_first"
#   matrix_confusion.py           -> "strict_min"   (matches nothing even in the template)
# With database identifiers those filters select zero rows: the scripts run
# clean but produce empty output. Pass --legacy-labels to write the template's
# names instead, which restores those filters at the cost of labelling rows with
# identifiers that no longer exist anywhere in the project.
LEGACY_LABELS = {
    "strict-minimal": "strict_minimal",
    "matrix-decomposition": "object_matrix_tree",
    "constraint-decomposition": "constraint_first",
}

# Key order taken from the template's own predicted_json cells.
JSON_KEY_ORDER = [
    "label", "definition", "comment",
    "hasProperty", "hasObjectOfInterest", "hasMatrix",
    "hasStatisticalModifier", "hasContextObject", "hasConstraint",
]

# Best-ranked complete configuration per prompt variant for this model, taken from
# iadopt_lab.configuration_rank across every campaign (primary metric
# mean_repetition_micro_close_f1). Unlike the fixed grid point above, the
# temperature differs per variant. All three are 97/97 complete.
BEST_CONFIGURATIONS = {
    # prompt_variant: (configuration_id, temperature, shot_count, close_f1, rank)
    "strict-minimal": (
        "1a6412bd7319d7d7e489ad13ff0e3cc09670d120af362cee746436122f4fb2c3", 1.0, 5, 0.3978, 4),
    "matrix-decomposition": (
        "6bcf8cdc1adafdad4a41d6a6a65441a6bf9541ee4bd97fcac88c184740211837", 0.5, 5, 0.4206, 1),
    "constraint-decomposition": (
        "b1f5b244d9526fd15121ecb32fbb8a3d6d3d278ee303a47ce65039faa3c16892", 1.0, 5, 0.3779, 6),
}

BEST_QUERY = """
SELECT v.label, v.definition, v.gold, p.canonical,
       rr.temperature, rr.shot_count, rr.model_id, rr.prompt_variant,
       t.id AS task_id, p.id AS prediction_id, rr.configuration_id
FROM iadopt_lab.resolved_run rr
JOIN iadopt_lab.task       t ON t.run_id = rr.id AND t.state = 'complete'
JOIN iadopt_lab.variable   v ON v.id = t.variable_id
JOIN iadopt_lab.prediction p ON p.task_id = t.id AND p.terminal_invalid = false
WHERE rr.configuration_id = %(config)s
  AND rr.campaign_id::text LIKE %(campaign)s
ORDER BY v.label
"""

QUERY = """
SELECT v.label, v.definition, v.gold, p.canonical,
       rr.temperature, rr.shot_count, rr.model_id, rr.prompt_variant,
       t.id AS task_id, p.id AS prediction_id, rr.configuration_id
FROM iadopt_lab.resolved_run rr
JOIN iadopt_lab.task       t ON t.run_id = rr.id AND t.state = 'complete'
JOIN iadopt_lab.variable   v ON v.id = t.variable_id
JOIN iadopt_lab.prediction p ON p.task_id = t.id AND p.terminal_invalid = false
WHERE rr.model_id = %(model)s
  AND rr.temperature = %(temperature)s
  AND rr.shot_count = %(shot)s
  AND rr.prompt_variant = %(variant)s
  AND rr.campaign_id::text LIKE %(campaign)s
ORDER BY v.label
"""


def shape_json(label: str, definition: str, components: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble one workbook JSON object in the template's key order.

    'comment' is carried as an empty string: the current prompts no longer ask
    the model for it and the v2.0.1 corpus gold does not store one, but
    validation_gt.py / validation_pred.py index obj["comment"] directly and
    would raise KeyError without the key. Those two scripts derive their actual
    metric from 'definition', which is real data, so an empty comment changes
    no computed value.
    """
    out: Dict[str, Any] = {"label": label, "definition": definition, "comment": ""}
    for key in JSON_KEY_ORDER[3:]:
        out[key] = components.get(key, [] if key == "hasConstraint" else "")
    return out


# Component order of the Summary sheet's per-key columns, matching the template
# header exactly (it is onlyPhaseOne.ONTO_KEYS order).
ONTO_KEYS = ["hasStatisticalModifier", "hasProperty", "hasObjectOfInterest",
             "hasMatrix", "hasContextObject", "hasConstraint"]

# Exact per-variable/component confusion receipts written by the lab scorer.
# numerator/denominator are an exact rational: summing numerators alone is wrong.
FACTS_QUERY = """
SELECT component, mode, metric, numerator, denominator
FROM iadopt_lab.evaluation_facts
WHERE configuration_id = %(config)s AND metric IN ('tp','fp','fn')
"""

CONFIG_QUERY = """
SELECT model_id, temperature, prompt_variant, shot_count
FROM iadopt_lab.resolved_run WHERE configuration_id = %(config)s LIMIT 1
"""


def _prf(tp, fp, fn):
    """Micro precision/recall/F1 over exact rational confusion counts.

    Same definition the template Summary uses (pool TP/FP/FN, then derive),
    evaluated in exact arithmetic so it reproduces the database's own ranking
    values digit for digit.
    """
    from fractions import Fraction  # noqa: PLC0415
    precision = tp / (tp + fp) if tp + fp else Fraction(0)
    recall = tp / (tp + fn) if tp + fn else Fraction(0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else Fraction(0)
    return round(float(precision), 3), round(float(recall), 3), round(float(f1), 3)


def summary_from_database(conn, configuration_ids: List[str],
                          legacy_labels: bool = False) -> List[Dict[str, Any]]:
    """Build the Summary sheet from the metrics the database already holds.

    The Summary is NOT recomputed with the legacy onlyPhaseOne scorer: these are
    the authoritative scores the lab's own scorer stored, so the workbook agrees
    with iadopt_lab.configuration_rank instead of contradicting it. The metric
    definition is unchanged (micro-pooled TP/FP/FN -> P/R/F1, rounded to 3dp);
    only the source of the confusion counts differs.
    """
    from collections import defaultdict  # noqa: PLC0415
    from fractions import Fraction  # noqa: PLC0415

    summary: List[Dict[str, Any]] = []
    for config in configuration_ids:
        model_id, temperature, prompt_variant, shot = conn.execute(
            CONFIG_QUERY, {"config": config}).fetchone()

        counts: Dict[tuple, Fraction] = defaultdict(Fraction)
        for component, mode, metric, num, den in conn.execute(
                FACTS_QUERY, {"config": config}):
            if component == "__variable__":      # per-variable rollup; would double count
                continue
            value = Fraction(int(num), int(den))
            counts[(component, mode, metric)] += value
            counts[("__all__", mode, metric)] += value

        row: Dict[str, Any] = {
            "Model": model_id,
            "Temperature": float(temperature),
            "PromptVersion": (LEGACY_LABELS[prompt_variant] if legacy_labels else prompt_variant),
            "Shot": int(shot),
        }
        for scope, prefix in [("__all__", "")] + [(k, k + "_") for k in ONTO_KEYS]:
            for mode in ("exact", "close"):
                p, r, f = _prf(*(counts[(scope, mode, m)] for m in ("tp", "fp", "fn")))
                row[f"{prefix}P_{mode}"] = p
                row[f"{prefix}R_{mode}"] = r
                row[f"{prefix}F_{mode}"] = f
        summary.append(row)

    summary.sort(key=lambda r: r["F_exact"], reverse=True)   # template ordering
    return summary


# --------------------------------------------------------------------------- #
# Stage 1: fetch
# --------------------------------------------------------------------------- #
def fetch(records_path: pathlib.Path, legacy_labels: bool = False,
          best: bool = False) -> None:
    from iadopt_lab.local_database import local_dsn  # noqa: PLC0415
    import psycopg  # noqa: PLC0415

    print(f"Model {MODEL_ID} | campaign {CAMPAIGN_PREFIX} | "
          + ("best-ranked configuration per prompt variant"
             if best else f"temperature {TEMPERATURE} | shot {SHOT_COUNT}"))
    records: List[Dict[str, Any]] = []
    configuration_ids: set[str] = set()
    with psycopg.connect(local_dsn(LAB, role="reader")) as conn:
        for variant in PROMPT_VARIANTS:
            if best:
                config, temp, shot, f1, rank = BEST_CONFIGURATIONS[variant]
                rows = conn.execute(BEST_QUERY, {
                    "config": config, "campaign": CAMPAIGN_PREFIX + "%",
                }).fetchall()
                print(f"  {variant:28s} {len(rows):3d} rows  "
                      f"(temp {temp}, {shot}-shot, close F1 {f1}, rank {rank})")
            else:
                rows = conn.execute(QUERY, {
                    "model": MODEL_ID, "temperature": TEMPERATURE, "shot": SHOT_COUNT,
                    "variant": variant, "campaign": CAMPAIGN_PREFIX + "%",
                }).fetchall()
                print(f"  {variant:28s} {len(rows):3d} rows")
            configuration_ids.add(rows[0][10]) if rows else None
            for (label, definition, gold, canonical, temperature, shot,
                 model_id, prompt_variant, task_id, prediction_id, _cfg) in rows:
                records.append({
                    "variable": label,
                    "model": model_id,
                    "temperature": float(temperature),
                    "prompt_version": (LEGACY_LABELS[prompt_variant]
                                       if legacy_labels else prompt_variant),
                    "shot": int(shot),
                    "ground_truth_json": shape_json(label, definition, gold),
                    "predicted_json": shape_json(label, definition, canonical),
                    "task_id": str(task_id),
                    "prediction_id": str(prediction_id),
                })
        summary = summary_from_database(conn, sorted(configuration_ids), legacy_labels)
    payload = {"rows": records, "summary": summary}
    records_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  total {len(records)} rows -> {records_path}")
    for row in summary:
        print(f"    Summary {row['PromptVersion']:26s} "
              f"F_exact={row['F_exact']:.3f}  F_close={row['F_close']:.3f}")


# --------------------------------------------------------------------------- #
# Stage 2: build
# --------------------------------------------------------------------------- #
def build(records_path: pathlib.Path, out_xlsx: pathlib.Path) -> None:
    import pandas as pd  # noqa: PLC0415
    import onlyPhaseOne as O  # noqa: PLC0415  (original scorer, imported not copied)

    payload = json.loads(records_path.read_text(encoding="utf-8"))
    records = payload["rows"] if isinstance(payload, dict) else payload
    summary = payload.get("summary") if isinstance(payload, dict) else None
    results = [dict(r, confusion=O.compute_confusion_for_pair(
        r["ground_truth_json"], r["predicted_json"])) for r in records]

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as wr:
        # 1) One sheet per ONTO_KEY
        for key in O.ONTO_KEYS:
            rows = []
            for r in results:
                gt, pred = r["ground_truth_json"], r["predicted_json"]
                rows.append({
                    "variable": r["variable"],
                    "model": r["model"],
                    "temperature": r["temperature"],
                    "prompt_version": r["prompt_version"],
                    "shot": r["shot"],
                    "ground_truth": json.dumps(gt.get(key, ""), ensure_ascii=False, indent=2),
                    "predicted": json.dumps(pred.get(key, ""), ensure_ascii=False, indent=2),
                })
            pd.DataFrame(rows).to_excel(wr, sheet_name=f"{key} concepts"[:31], index=False)

        # 2) Full LLM outputs
        json_rows = []
        for r in results:
            json_rows.append({
                "variable": r["variable"],
                "model": r["model"],
                "temperature": r["temperature"],
                "prompt_version": r["prompt_version"],
                "shot": r["shot"],
                "ground_truth_json": json.dumps(r["ground_truth_json"], ensure_ascii=False, indent=2),
                "predicted_json": json.dumps(r["predicted_json"], ensure_ascii=False, indent=2),
            })
        pd.DataFrame(json_rows).to_excel(wr, sheet_name="LLM outputs", index=False)

        # 3) Summary metrics. Prefer the database's own scores so the workbook
        #    agrees with iadopt_lab.configuration_rank; fall back to the legacy
        #    recompute only for a records file that predates them.
        if summary:
            df_summary = pd.DataFrame(summary, columns=list(
                O.compute_summary_metrics(results[:1]).columns))
        else:
            df_summary = O.compute_summary_metrics(results)
        df_summary.to_excel(wr, sheet_name="Summary", index=False)

    print(f"Wrote {out_xlsx}")

    manifest = out_xlsx.with_suffix(".provenance.json")
    manifest.write_text(json.dumps({
        "model_id": MODEL_ID, "temperature": TEMPERATURE, "shot_count": SHOT_COUNT,
        "campaign_prefix": CAMPAIGN_PREFIX, "prompt_variants": PROMPT_VARIANTS,
        "row_count": len(results),
        "rows": [{"variable": r["variable"], "prompt_version": r["prompt_version"],
                  "task_id": r["task_id"], "prediction_id": r["prediction_id"]}
                 for r in results],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild the Phase-1 workbook from the lab database.")
    ap.add_argument("stage", choices=["fetch", "build"])
    ap.add_argument("--records", default=str(HERE / "records.json"))
    ap.add_argument("--out", default=str(HERE / "onlyPhaseOne_glm52_from_database.xlsx"))
    ap.add_argument("--best", action="store_true",
                    help="fetch: use each prompt variant's best-ranked configuration "
                         "instead of the fixed temp 0.5 / 5-shot grid point")
    ap.add_argument("--legacy-labels", action="store_true",
                    help="fetch: write the template's old prompt_version names "
                         "instead of the database identifiers")
    args = ap.parse_args()

    if args.stage == "fetch":
        fetch(pathlib.Path(args.records), legacy_labels=args.legacy_labels, best=args.best)
    else:
        build(pathlib.Path(args.records), pathlib.Path(args.out))


if __name__ == "__main__":
    main()
