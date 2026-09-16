"""Best configuration per model across every ranked full-grid campaign.

Reasoning is treated as one parameter among prompt variant, shot count and temperature:
each model appears once, showing whichever configuration scored highest, with the losing
reasoning setting reported as a delta where both were actually measured.

Only the three full 36-configuration-per-model grids are pooled. They are comparable by
construction and the generator verifies it: identical population hash, evaluation size,
scorer version and Close threshold, checked before anything is reported. Every metric is
recomputed from per-variable receipts with exact rational arithmetic and cross-checked
against the value the ranking stored.

Usage:  .venv/bin/python ops/report-model-comparison.py [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402

from iadopt_lab.local_database import local_dsn  # noqa: E402

# The three full grids: 36 configurations per model, 97 variables, same scorer.
GRIDS = {
    "5cdc9417-28c0-50b9-b24b-8048a6f58ffc": "5cdc60cd",
    "844df00e-846c-50c2-9d0b-1a6deca6f3e7": "7bdfd213",
    "79b5ec5f-601b-54fd-9313-c0a3b99c3ed0": "c5e9333e",
}
COMPONENTS = ("hasProperty", "hasObjectOfInterest", "hasMatrix",
              "hasContextObject", "hasStatisticalModifier", "hasConstraint")
SHORT = {"hasProperty": "Prop", "hasObjectOfInterest": "Obj", "hasMatrix": "Matrix",
         "hasContextObject": "Ctx", "hasStatisticalModifier": "StatMod",
         "hasConstraint": "Constr"}


def _f(cell: dict) -> Fraction:
    return Fraction(cell["numerator"], cell["denominator"])


def _f1(tp: Fraction, fp: Fraction, fn: Fraction) -> Fraction | None:
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else 2 * tp / denominator


def _n(value: Fraction | None, places: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{places}f}"


def verify_comparable(conn) -> dict:
    """Refuse to pool grids that were not scored the same way."""
    seen = {}
    for campaign in GRIDS:
        population, size = conn.execute(
            "SELECT population_hash, population_size FROM campaign_plan WHERE campaign_id=%s",
            (campaign,)).fetchone()
        record = conn.execute(
            "SELECT e.evidence FROM evaluation_item e JOIN task t ON t.id=e.task_id "
            "WHERE t.campaign_id=%s LIMIT 1", (campaign,)).fetchone()[0]
        seen[campaign] = (population, size, record["scorer_version"],
                          record["close_threshold"],
                          record["similarity_identity"].get("manifest", {}).get("model"))
    distinct = set(seen.values())
    if len(distinct) != 1:
        raise SystemExit(f"grids are not comparable, refusing to pool: {seen}")
    population, size, scorer, threshold, _ = distinct.pop()
    return {"population": population, "size": size, "scorer": scorer, "threshold": threshold}


def metrics_for(conn, campaign: str, configuration: str) -> dict:
    """Recompute every metric for one configuration from its stored receipts."""
    evidence = conn.execute(
        "SELECT e.evidence FROM evaluation_item e JOIN task t ON t.id=e.task_id "
        "JOIN resolved_run r ON r.id=t.run_id "
        "WHERE t.campaign_id=%s AND r.configuration_id=%s", (campaign, configuration)).fetchall()
    totals = {branch: dict.fromkeys(("tp", "fp", "fn", "tn"), Fraction(0))
              for branch in ("close", "exact")}
    parts = {name: dict.fromkeys(("tp", "fp", "fn"), Fraction(0)) for name in COMPONENTS}
    for (record,) in evidence:
        for branch in ("close", "exact"):
            for key in ("tp", "fp", "fn", "tn"):
                totals[branch][key] += _f(record[branch]["totals"][key])
        for name in COMPONENTS:
            contribution = record["close"]["components"][name]["contributions"]
            for key in ("tp", "fp", "fn"):
                parts[name][key] += _f(contribution[key])
    close, exact = totals["close"], totals["exact"]
    mass = sum(close.values())
    cost = conn.execute(
        "SELECT coalesce(sum(cs.amount),0) FROM cost_settlement cs "
        "JOIN attempt a ON a.id=cs.attempt_id JOIN task t ON t.id=a.task_id "
        "JOIN resolved_run r ON r.id=t.run_id "
        "WHERE t.campaign_id=%s AND r.configuration_id=%s", (campaign, configuration)).fetchone()[0]
    return {
        "variables": len(evidence),
        "close_f1": _f1(close["tp"], close["fp"], close["fn"]),
        "exact_f1": _f1(exact["tp"], exact["fp"], exact["fn"]),
        "precision": None if close["tp"] + close["fp"] == 0 else close["tp"] / (close["tp"] + close["fp"]),
        "recall": None if close["tp"] + close["fn"] == 0 else close["tp"] / (close["tp"] + close["fn"]),
        "accuracy": None if mass == 0 else (close["tp"] + close["tn"]) / mass,
        "components": {name: _f1(p["tp"], p["fp"], p["fn"]) for name, p in parts.items()},
        "cost": float(cost),
    }


def collect(conn) -> list[dict]:
    """One entry per model deployment: its best configuration, and its reasoning runner-up."""
    arms: dict[tuple[str, str], list[dict]] = {}
    for campaign, ranking_prefix in GRIDS.items():
        ranking = conn.execute(
            "SELECT id FROM ranking_run WHERE campaign_id=%s AND id::text LIKE %s",
            (campaign, ranking_prefix + "%")).fetchone()[0]
        for provider, model, configuration, numerator, denominator in conn.execute(
            "SELECT provider, model_id, configuration_id, primary_numerator, primary_denominator "
            "FROM configuration_rank WHERE ranking_id=%s AND rank IS NOT NULL", (ranking,)):
            run = conn.execute(
                "SELECT prompt_variant, shot_count, temperature, reasoning_mode "
                "FROM resolved_run WHERE campaign_id=%s AND configuration_id=%s LIMIT 1",
                (campaign, configuration)).fetchone()
            arms.setdefault((provider, model), []).append({
                "campaign": campaign, "configuration": configuration,
                "score": Fraction(int(numerator), int(denominator)),
                "prompt": run[0], "shots": run[1], "temperature": float(run[2]),
                "reasoning": run[3],
            })

    rows = []
    for (provider, model), candidates in arms.items():
        best = max(candidates, key=lambda item: item["score"])
        computed = metrics_for(conn, best["campaign"], best["configuration"])
        if computed["close_f1"] != best["score"]:
            raise SystemExit(f"{model}: recomputed {computed['close_f1']} != ranked {best['score']}")

        modes = {item["reasoning"] for item in candidates}
        per_mode = {mode: max((i for i in candidates if i["reasoning"] == mode),
                              key=lambda item: item["score"]) for mode in modes}
        rankable = {mode: sum(1 for i in candidates if i["reasoning"] == mode) for mode in modes}
        # The mean over a mode's rankable configurations matters as much as its maximum:
        # a maximum taken over 3 survivors of 36 is a survivorship artefact, and only the
        # pair of numbers together shows whether a reasoning verdict can be trusted.
        means = {mode: sum((i["score"] for i in candidates if i["reasoning"] == mode),
                           Fraction(0)) / rankable[mode] for mode in modes}
        alternative = None
        if len(modes) > 1:
            loser = min((per_mode[m] for m in modes if m != best["reasoning"]),
                        key=lambda item: item["score"])
            alternative = {"reasoning": loser["reasoning"], "score": loser["score"],
                           "delta": best["score"] - loser["score"],
                           "rankable": rankable[loser["reasoning"]],
                           "mean": means[loser["reasoning"]]}
        best_mean = means[best["reasoning"]]
        rows.append({**best, **computed, "provider": provider, "model": model,
                     "mean": best_mean,
                     "rankable": rankable[best["reasoning"]],
                     "rankable_total": len(candidates),
                     "modes_tested": sorted(modes), "alternative": alternative})
    rows.sort(key=lambda item: item["close_f1"], reverse=True)
    return rows


def render(rows: list[dict], identity: dict) -> str:
    out: list[str] = []
    add = out.append
    campaigns = {"5cdc9417-28c0-50b9-b24b-8048a6f58ffc": "5cdc9417",
                 "844df00e-846c-50c2-9d0b-1a6deca6f3e7": "844df00e",
                 "79b5ec5f-601b-54fd-9313-c0a3b99c3ed0": "79b5ec5f"}

    add("# Every model at its best configuration")
    add("")
    add("One row per model deployment, showing the single highest-scoring configuration "
        "it produced anywhere. **Reasoning is treated as a parameter like any other**: "
        "where a model was measured both ways, the row shows whichever setting won and "
        "the other is reported as a delta.")
    add("")
    add(f"Pooled from the three full grids — `5cdc9417` (PSNC), `844df00e` and "
        f"`79b5ec5f` (OpenRouter) — each 36 configurations per model over the same "
        f"**{identity['size']} variables**. The generator verifies they are comparable "
        f"before reporting anything: identical population hash `{identity['population'][:12]}…`, "
        f"scorer `{identity['scorer']}`, Close threshold {identity['threshold']}. Every "
        f"figure is recomputed from per-variable receipts and cross-checked against the "
        f"stored ranking.")
    add("")
    add("## Best configuration per model")
    add("")
    add("| Model | Provider | Prompt | Shots | T | Reasoning | Close F1 | Exact F1 | Accuracy | Cost |")
    add("|---|---|---|--:|--:|---|--:|--:|--:|--:|")
    for row in rows:
        cost = "not billed" if row["provider"] == "psnc" else f"${row['cost']:.4f}"
        choice = "**" if row["alternative"] else ""
        add(f"| `{row['model']}` | {row['provider']} | {row['prompt']} | {row['shots']} | "
            f"{row['temperature']:g} | {choice}{row['reasoning']}{choice} | "
            f"**{_n(row['close_f1'])}** | {_n(row['exact_f1'])} | {_n(row['accuracy'])} | {cost} |")
    add("")
    add("A **bold** reasoning setting marks a model where both settings were actually "
        "measured and this one won. Every other row had only one setting available, so "
        "no choice was made — see the coverage table below.")
    add("")

    add("## Where reasoning was a real choice")
    add("")
    tested = [row for row in rows if row["alternative"]]
    if tested:
        add("| Model | Winner | Best | Mean | n | Runner-up | Best | Mean | n | Delta (best) |")
        add("|---|---|--:|--:|--:|---|--:|--:|--:|--:|")
        for row in rows:
            alt = row["alternative"]
            if not alt:
                continue
            add(f"| `{row['model']}` | {row['reasoning']} | {_n(row['close_f1'])} | "
                f"{_n(row['mean'])} | {row['rankable']} | {alt['reasoning']} | "
                f"{_n(alt['score'])} | {_n(alt['mean'])} | {alt['rankable']} | "
                f"**+{_n(alt['delta'])}** |")
        add("")
    add("Only these models were run both ways, on the same gateway and the same grid, so "
        "these are the experiment's only controlled reasoning comparisons. Everything "
        "else in the table above is a model's single measured setting.")
    add("")
    add("**The two verdicts are not equally trustworthy.** `qwen/qwen3-32b` had 36 "
        "rankable configurations with reasoning off against 22 with it on, and off wins "
        "on both the maximum and the mean — that is a credible result. "
        "`qwen/qwen3-8b` had only **3 of 36** configurations survive with reasoning off, "
        "because upstream rate limiting destroyed the rest (D-048); its off arm is a "
        "maximum over three survivors and its higher mean reflects which configurations "
        "happened to complete, not which setting is better. Treat the qwen3-8b delta as "
        "unusable and the qwen3-32b one as the experiment's actual reasoning finding.")
    add("")

    add("## Coverage — how much evidence each row rests on")
    add("")
    add("A configuration is rankable only if all 97 of its variables completed. Losses "
        "are permanent, so a low count means the row's winner was chosen from a smaller "
        "field and its margin is correspondingly less certain.")
    add("")
    add("| Model | Reasoning settings measured | Rankable configurations | Winner drawn from | Campaign |")
    add("|---|---|--:|--:|---|")
    for row in rows:
        add(f"| `{row['model']}` | {', '.join(row['modes_tested'])} | "
            f"{row['rankable_total']} | {row['rankable']} | "
            f"`{campaigns[row['campaign']]}` |")
    add("")

    add("## Fine-grained: Close F1 per I-ADOPT field")
    add("")
    add("Each model's winning configuration, scored one field at a time over the same "
        "97 variables. Computed from each field's own contributions, so these neither "
        "average nor sum to the overall column.")
    add("")
    add("| Model | " + " | ".join(SHORT[name] for name in COMPONENTS) + " |")
    add("|---|" + "--:|" * len(COMPONENTS))
    for row in rows:
        cells = " | ".join(_n(row["components"][name], 3) for name in COMPONENTS)
        add(f"| `{row['model']}` | {cells} |")
    add("")
    return "\n".join(out)


NOTES = """
## Reading this table safely

**Provider is not a nuisance variable here, it is a confound.** The three PSNC rows ran
on different hardware under a different serving stack from the six OpenRouter rows. A
PSNC row scoring above an OpenRouter row does not establish that the model is better —
only that this deployment of it scored better on this corpus. `GLM-5.2` (PSNC) and
`z-ai/glm-5.2` (OpenRouter) are the same model family served two ways, and the gap
between those two rows mixes deployment with reasoning; it is not a reasoning result.

**Cost is not comparable across providers.** PSNC was non-billed on the owner's own
hardware, so its rows read `not billed` rather than `$0.0000`. That is an absence of
metering, not a price. Only the OpenRouter rows can be compared with each other on cost.

**Reasoning-off was never measured for `z-ai/glm-5.2`.** Its row is its only setting.
PSNC reasoning-enabled runs were excluded from the experiment because they could not fit
the 120-second timeout (D-044, D-045), so no controlled on/off comparison exists for the
GLM family at all.

**A model's own best row is a maximum over 36 configurations**, so it is optimistically
biased relative to that model's typical behaviour — the more configurations a model had
rankable, the more the maximum is favoured. The coverage table is there to make that
visible: `qwen/qwen3-8b` picked its reasoning-off winner from a field of 3.

## How the metrics are calculated

Identical to the per-configuration report, which documents the scoring in full:
[results-top-configurations.md](results-top-configurations.md). In brief — each of the
six I-ADOPT fields contributes exactly one unit of confusion mass per variable; TP, FP,
FN and TN are summed across all 97 variables as exact fractions, then:

```
precision = TP / (TP + FP)          F1       = 2·TP / (2·TP + FP + FN)
recall    = TP / (TP + FN)          accuracy = (TP + TN) / (TP + FP + FN + TN)
```

**Close** accepts a cosine similarity of 0.80 or better; **Exact** requires identical
normalised strings. Accuracy credits true negatives, and 39% of this corpus's field mass
is fields that should be left empty, so it rewards caution rather than skill — F1 is the
ranked metric and the one to read.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/results-model-comparison.md")
    args = parser.parse_args()
    with psycopg.connect(local_dsn(ROOT, role="app"), connect_timeout=10) as conn:
        conn.read_only = True
        conn.execute("SET search_path TO iadopt_lab, public")
        identity = verify_comparable(conn)
        rows = collect(conn)
    document = render(rows, identity) + NOTES
    (ROOT / args.out).write_text(document, encoding="utf-8")
    print(f"wrote {ROOT / args.out} ({len(document):,} chars, {len(rows)} models)\n")
    for row in rows:
        alt = row["alternative"]
        extra = (f"  (vs {alt['reasoning']} {float(alt['score']):.4f}, "
                 f"+{float(alt['delta']):.4f})") if alt else ""
        print(f"  {row['model']:<34} {row['provider']:<11} {row['reasoning']:<15} "
              f"F1={float(row['close_f1']):.4f}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
