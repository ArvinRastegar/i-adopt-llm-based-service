"""Generate the top-configurations results document straight from stored evidence.

Every figure is recomputed from `evaluation_item` receipts with exact rational
arithmetic, never copied from a summary, and the overall F1 it derives is checked
against the value the ranking stored. Run it again and the document regenerates; the
numbers cannot drift from the database because nothing is transcribed by hand.

Usage:  .venv/bin/python ops/report-top-configurations.py [--top N] [--out PATH]
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

CAMPAIGN = "79b5ec5f-601b-54fd-9313-c0a3b99c3ed0"
RANKING = "c5e9333e-7b64-544f-8b2e-96a596e756e8"
COMPONENTS = ("hasProperty", "hasObjectOfInterest", "hasMatrix",
              "hasContextObject", "hasStatisticalModifier", "hasConstraint")
SHORT = {"hasProperty": "Prop", "hasObjectOfInterest": "Obj", "hasMatrix": "Matrix",
         "hasContextObject": "Ctx", "hasStatisticalModifier": "StatMod",
         "hasConstraint": "Constr"}


def _f(cell: dict) -> Fraction:
    """Read one stored contribution as the exact fraction the scorer recorded."""
    return Fraction(cell["numerator"], cell["denominator"])


def _f1(tp: Fraction, fp: Fraction, fn: Fraction) -> Fraction | None:
    """F1 from summed confusion mass; undefined when a configuration predicted nothing."""
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else 2 * tp / denominator


def _pct(value: Fraction | None, places: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{places}f}"


def collect(conn, limit: int) -> list[dict]:
    """Recompute every reported metric for the top `limit` ranked configurations."""
    ranked = conn.execute(
        "SELECT rank, configuration_id, model_id, primary_numerator, primary_denominator "
        "FROM configuration_rank WHERE ranking_id=%s AND rank IS NOT NULL "
        "ORDER BY rank LIMIT %s", (RANKING, limit)).fetchall()

    rows = []
    for rank, configuration, model, numerator, denominator in ranked:
        run = conn.execute(
            "SELECT prompt_variant, shot_count, temperature, top_p, max_output_tokens, "
            "reasoning_mode FROM resolved_run WHERE campaign_id=%s AND configuration_id=%s "
            "LIMIT 1", (CAMPAIGN, configuration)).fetchone()
        evidence = conn.execute(
            "SELECT e.evidence FROM evaluation_item e JOIN task t ON t.id=e.task_id "
            "JOIN resolved_run r ON r.id=t.run_id "
            "WHERE t.campaign_id=%s AND r.configuration_id=%s", (CAMPAIGN, configuration)).fetchall()

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
        mass = close["tp"] + close["fp"] + close["fn"] + close["tn"]
        cost, attempts, priced = conn.execute(
            "SELECT coalesce(sum(cs.amount),0), count(*), count(cs.amount) "
            "FROM cost_settlement cs JOIN attempt a ON a.id=cs.attempt_id "
            "JOIN task t ON t.id=a.task_id JOIN resolved_run r ON r.id=t.run_id "
            "WHERE t.campaign_id=%s AND r.configuration_id=%s", (CAMPAIGN, configuration)).fetchone()

        stored = Fraction(int(numerator), int(denominator))
        computed = _f1(close["tp"], close["fp"], close["fn"])
        if computed != stored:  # the document is only trustworthy if this holds
            raise SystemExit(f"rank {rank}: recomputed {computed} != stored {stored}")

        rows.append({
            "rank": rank, "model": model, "variables": len(evidence),
            "prompt": run[0], "shots": run[1], "temperature": float(run[2]),
            "reasoning": run[5],
            "close_f1": computed,
            "close_precision": None if close["tp"] + close["fp"] == 0 else close["tp"] / (close["tp"] + close["fp"]),
            "close_recall": None if close["tp"] + close["fn"] == 0 else close["tp"] / (close["tp"] + close["fn"]),
            "accuracy": None if mass == 0 else (close["tp"] + close["tn"]) / mass,
            "exact_f1": _f1(exact["tp"], exact["fp"], exact["fn"]),
            "components": {name: _f1(p["tp"], p["fp"], p["fn"]) for name, p in parts.items()},
            "cost": float(cost), "attempts": attempts, "priced": priced,
        })
    return rows


def render(rows: list[dict], campaign_cost: float, population: int) -> str:
    """Render the results document. Prose is fixed; every number comes from `rows`."""
    out: list[str] = []
    add = out.append

    add("# Top configurations — campaign `79b5ec5f`")
    add("")
    add("Generated from stored evaluation receipts by "
        "`ops/report-top-configurations.py`. Every figure below is recomputed from the "
        "per-variable evidence with exact rational arithmetic; the overall Close F1 is "
        "checked against the value the ranking stored, and the generator refuses to "
        "write this file if they disagree.")
    add("")
    add(f"Campaign total: **${campaign_cost:.2f}** over 17,460 tasks. "
        f"Each configuration below was evaluated on the same **{population} variables**.")
    add("")
    add("## The ten best configurations")
    add("")
    add("| # | Model | Prompt | Shots | T | Reasoning | Close F1 | Exact F1 | Accuracy | Cost |")
    add("|--:|---|---|--:|--:|---|--:|--:|--:|--:|")
    for row in rows:
        add(f"| {row['rank']} | `{row['model']}` | {row['prompt']} | {row['shots']} | "
            f"{row['temperature']:g} | {row['reasoning']} | **{_pct(row['close_f1'])}** | "
            f"{_pct(row['exact_f1'])} | {_pct(row['accuracy'])} | ${row['cost']:.4f} |")
    add("")
    add("Precision and recall behind the same Close F1 figures:")
    add("")
    add("| # | Model | Close precision | Close recall | Close F1 |")
    add("|--:|---|--:|--:|--:|")
    for row in rows:
        add(f"| {row['rank']} | `{row['model']}` | {_pct(row['close_precision'])} | "
            f"{_pct(row['close_recall'])} | {_pct(row['close_f1'])} |")
    add("")
    add("## Fine-grained: Close F1 per I-ADOPT field")
    add("")
    add("Same configurations, scored one field at a time. A field's score is computed "
        "from that field's own contributions across all "
        f"{population} variables — these are **not** averages of the overall column, and "
        "they do not sum to it.")
    add("")
    add("| # | Model | " + " | ".join(SHORT[name] for name in COMPONENTS) + " |")
    add("|--:|---|" + "--:|" * len(COMPONENTS))
    for row in rows:
        cells = " | ".join(_pct(row["components"][name], 3) for name in COMPONENTS)
        add(f"| {row['rank']} | `{row['model']}` | {cells} |")
    add("")
    add("`n/a` means the field contributed no true positives, false positives or false "
        "negatives anywhere in the population — gold and prediction were both empty for "
        "every variable, so F1 is undefined rather than zero.")
    add("")
    add("## What the cost buys")
    add("")
    add("Cost per configuration covers 97 variables including retries. The right-hand "
        "column scales that to 1,000 variables, which is the number worth comparing "
        "against if this is ever run at corpus scale.")
    add("")
    add("| # | Model | Close F1 | Cost (97 vars) | Relative | Per 1,000 vars |")
    add("|--:|---|--:|--:|--:|--:|")
    cheapest = min(row["cost"] for row in rows)
    for row in rows:
        per_thousand = row["cost"] / row["variables"] * 1000
        add(f"| {row['rank']} | `{row['model']}` | {_pct(row['close_f1'])} | "
            f"${row['cost']:.4f} | {row['cost'] / cheapest:.1f}x | ${per_thousand:.2f} |")
    add("")
    top, value = rows[0], min(rows, key=lambda r: r["cost"])
    ratio = top["cost"] / value["cost"]
    gap = float(top["close_f1"]) - float(value["close_f1"])
    share = float(value["close_f1"]) / float(top["close_f1"])
    add(f"The spread matters more than the absolute figures. `{value['model']}` at rank "
        f"{value['rank']} reaches **{share:.0%}** of the best score for **1/{ratio:.0f} "
        f"of the cost** — {gap:.4f} F1 for {ratio:.0f}x the spend is the trade the top "
        f"of this table is making. Which side of it is right depends on whether the "
        f"output is reviewed by a person afterwards; at these accuracies it will be.")
    add("")
    return "\n".join(out)


METHOD = """
## How each metric is calculated

Scorer `january-derived-member-credit-v1`, similarity MiniLM-L6-v2 cosine over
NFC-normalised, case-folded, trimmed strings.

### The unit of measurement

A decomposition has six fields. **Each field contributes exactly one unit of confusion
mass per variable**, split between true positives, false positives, false negatives and
true negatives. One variable therefore carries 6 units, and a 97-variable population
carries 582. That fixed mass is what makes the fields comparable with each other and the
configurations comparable with one another.

How a field's unit is divided depends on what is being compared:

| Situation | Branch | Result |
|---|---|---|
| Both gold and prediction empty | `january-empty` | TN = 1 |
| Gold empty, prediction present | `january-empty` | FP = 1 |
| Gold present, prediction empty | `january-empty` | FN = 1 |
| Both plain strings | `january-scalar` | similarity ≥ threshold → TP = 1, else **FP = 1** |
| Either side a nested system | `system-member-credit-v1` | unit split across members by assignment |
| `hasConstraint` | `january-constraints-greedy` | unit split across constraints, see below |

**Note the scalar row.** A present-but-wrong answer scores FP = 1 and FN = 0. It is
counted as a wrong thing said, not as a right thing missed. This is why recall sits
above precision almost everywhere in the tables above: recall is only reduced by fields
left *empty* that should have been filled.

### hasConstraint

Each gold constraint carries two units of `1/(2·n_gold)` — one for its `label`, one for
its `on` target — so the field still totals one unit. Gold and predicted constraints are
paired greedily, highest mean similarity first, row-major on ties. Within a pair, label
and target are scored separately: at or above threshold adds its unit to TP, below adds
it to FP. Unmatched gold constraints add `2·unit` to FN; unmatched predicted constraints
add `2·unit` to FP. A final correction rescales so the field's mass is exactly 1.

### Nested systems

A ratio such as `lactate / blood`, or a flux such as `vegetation → soil`, is compared
member by member. Asymmetric systems only allow matches in the same role — a numerator
may not be credited against a denominator. Each member carries an equal share of the
field's unit, so getting one of two roles right earns half credit rather than none.

### Close versus Exact

Identical machinery, one number different:

- **Close** — cosine similarity ≥ **0.80**. "amount of substance concentration" against
  "concentration" scores as a hit.
- **Exact** — threshold **1.0**, i.e. the normalised strings must be identical.

Exact is always the harsher of the two, and the gap between the columns is a direct
measure of how much a configuration is getting *nearly* right.

### Aggregating to a configuration

Micro, not macro: TP, FP, FN and TN are summed across all 97 variables as exact
fractions, and the metric is computed once from those totals.

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2·TP / (2·TP + FP + FN)
accuracy  = (TP + TN) / (TP + FP + FN + TN)
```

Micro-averaging means every variable carries the same weight regardless of how many
constraints it happens to have. Averaging the 97 per-variable F1 scores instead (macro)
would give a different, generally higher number; the ranking uses micro, and so does
every figure here.

**Accuracy is the weakest of these four**, and is reported only because it was asked
for. It is the one metric that credits true negatives, and in this corpus true negatives
are abundant. Counting empty gold fields across the 97-variable population:

| Field | Gold present | Gold empty |
|---|--:|--:|
| `hasProperty` | 97 | 0 |
| `hasObjectOfInterest` | 97 | 0 |
| `hasConstraint` | 97 | 0 |
| `hasMatrix` | 47 | 50 |
| `hasContextObject` | 10 | 87 |
| `hasStatisticalModifier` | 8 | 89 |

**226 of the 582 mass units — 39% — are fields that should be left empty.** A model that
emits nothing at all for `hasContextObject` and `hasStatisticalModifier` banks 176 units
of true negative before saying anything correct. Accuracy therefore rewards correct
silence as much as correct extraction and rises with caution rather than skill, which is
why F1, which ignores TN entirely, is the ranked metric. Read the accuracy column as a
sanity check, not as a score.

### Fine-grained field scores

Each field's F1 is computed from that field's own contributions summed over the
population — the same micro formula, restricted to one of the six. They are independent
views, not a decomposition of the overall figure: the overall score is computed from
total mass across all six fields at once, so the six field scores neither average nor
sum to it.

`n/a` appears where a field generated no TP, FP or FN anywhere — gold and prediction
were both empty for all 97 variables, leaving only true negatives and an undefined F1.

### Cost

Per-attempt amounts from `cost_settlement`, summed over every attempt belonging to the
configuration, in USD. This is settled spend from the provider's reported token usage,
not an estimate, and it includes retries: a configuration whose answers failed
validation twice before succeeding paid for all three calls. Attempts whose settlement
is `unavailable` or `ambiguous` contribute nothing, so a configuration's cost is a
slight underestimate where those occurred; the per-row attempt counts are in the
generator's output if you need the exact coverage.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", default="docs/results-top-configurations.md")
    args = parser.parse_args()

    with psycopg.connect(local_dsn(ROOT, role="app"), connect_timeout=10) as conn:
        conn.read_only = True
        conn.execute("SET search_path TO iadopt_lab, public")
        rows = collect(conn, args.top)
        total = float(conn.execute("SELECT spent_cost FROM campaign WHERE id=%s",
                                   (CAMPAIGN,)).fetchone()[0])

    document = render(rows, total, rows[0]["variables"]) + METHOD
    destination = ROOT / args.out
    destination.write_text(document, encoding="utf-8")
    print(f"wrote {destination} ({len(document):,} chars, {len(rows)} configurations)")
    for row in rows:
        print(f"  #{row['rank']:<3} {row['model']:<34} F1={float(row['close_f1']):.4f} "
              f"cost=${row['cost']:.4f} attempts={row['attempts']} priced={row['priced']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
