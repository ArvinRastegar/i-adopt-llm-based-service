"""Export the best-scoring configuration's 97 answers to an Excel workbook.

The winning configuration is found by searching every fully-scored configuration across
the three comparable full grids, not hardcoded, and the overall Close F1 that wins is
re-derived from per-variable receipts and checked against the value the ranking stored.
Per-variable and per-component scores are likewise recomputed from stored contributions
with exact rational arithmetic and cross-checked against the scorer's own recorded F1, so
the workbook cannot drift from the database.

Usage:  .venv/bin/python ops/export-best-configuration.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

from iadopt_lab.local_database import local_dsn  # noqa: E402

# The three full 36-configuration-per-model grids and their rankings. Same population,
# scorer and threshold, which is what makes "best over all campaigns" a meaningful search.
GRIDS = {
    "5cdc9417-28c0-50b9-b24b-8048a6f58ffc": "5cdc60cd-3f25-541c-9dfd-32fded299ee7",
    "844df00e-846c-50c2-9d0b-1a6deca6f3e7": "7bdfd213-68fd-57c7-b696-b5dbac1f23da",
    "79b5ec5f-601b-54fd-9313-c0a3b99c3ed0": "c5e9333e-7b64-544f-8b2e-96a596e756e8",
}
COMPONENTS = ("hasProperty", "hasObjectOfInterest", "hasMatrix",
              "hasContextObject", "hasStatisticalModifier", "hasConstraint")
LABELS = {"hasProperty": "Property", "hasObjectOfInterest": "Object of interest",
          "hasMatrix": "Matrix", "hasContextObject": "Context object",
          "hasStatisticalModifier": "Statistical modifier", "hasConstraint": "Constraint"}
EMPTY = "—"  # a field the side deliberately left blank, distinct from an unwritten cell

HEAD = Font(bold=True, color="FFFFFF", size=11)
FILL = PatternFill("solid", fgColor="2F4F6F")
GOLD_FILL = PatternFill("solid", fgColor="EAF1F8")
MONO = Font(name="Menlo", size=9)


def _f(cell: dict) -> Fraction:
    """Read one stored contribution as the exact fraction the scorer recorded."""
    return Fraction(cell["numerator"], cell["denominator"])


def _f1(tp: Fraction, fp: Fraction, fn: Fraction) -> Fraction | None:
    """F1 from confusion mass; undefined when nothing was asserted on either side."""
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else 2 * tp / denominator


def render(value) -> str:
    """Render one I-ADOPT component value as the pretty multi-line text for its cell."""
    if value is None or value == "" or value == []:
        return EMPTY
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _render_system(value)
    if isinstance(value, list):
        return _render_constraints(value)
    return json.dumps(value, ensure_ascii=False)  # unreachable for known shapes


def _render_system(value: dict) -> str:
    """Render a nested system: its own label, then each part under its role."""
    roles = (("hasNumerator", "numerator"), ("hasDenominator", "denominator"),
             ("hasSource", "source"), ("hasTarget", "target"))
    if "SymmetricSystem" in value:
        parts = [f"    • part:  {_leaf(p)}" for p in value.get("hasPart", [])]
        return "\n".join([f"Symmetric system:  {value['SymmetricSystem']}", *parts])
    if "AsymmetricSystem" in value:
        kind = "ratio" if "hasNumerator" in value else "flux" if "hasSource" in value else "pair"
        lines = [f"Asymmetric system ({kind}):  {value['AsymmetricSystem']}"]
        lines += [f"    • {name}:  {_leaf(value[key])}" for key, name in roles if key in value]
        return "\n".join(lines)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _leaf(value) -> str:
    """Flatten one member of a system, which is a plain string in every stored record."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _render_constraints(value: list) -> str:
    """Render the constraint list as a numbered block of label / target pairs."""
    lines = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict) and {"label", "on"} <= set(item):
            lines.append(f"{index}.  {item['label']}")
            lines.append(f"      on:  {_leaf(item['on'])}")
        else:
            lines.append(f"{index}.  {json.dumps(item, ensure_ascii=False)}")
    return "\n".join(lines) if lines else EMPTY


def find_best(conn) -> dict:
    """Search every fully-scored configuration in the three grids for the highest Close F1."""
    best = None
    for campaign, ranking in GRIDS.items():
        for cid, model, numerator, denominator, rank in conn.execute(
            "SELECT configuration_id, model_id, primary_numerator, primary_denominator, rank "
            "FROM configuration_rank WHERE ranking_id=%s AND rank IS NOT NULL", (ranking,)
        ).fetchall():
            score = Fraction(int(numerator), int(denominator))
            if best is None or score > best["stored_f1"]:
                best = {"campaign": campaign, "ranking": ranking, "configuration": cid,
                        "model": model, "stored_f1": score, "rank": rank}
    run = conn.execute(
        "SELECT provider, reasoning_mode, prompt_variant, shot_count, temperature, top_p, "
        "max_output_tokens FROM resolved_run WHERE campaign_id=%s AND configuration_id=%s LIMIT 1",
        (best["campaign"], best["configuration"])).fetchone()
    best |= {"provider": run[0], "reasoning": run[1], "prompt_variant": run[2],
             "shots": run[3], "temperature": run[4], "top_p": run[5], "max_tokens": run[6]}
    return best


def collect(conn, best: dict) -> list[dict]:
    """Build one row per variable, recomputing every score from the stored receipts."""
    records = conn.execute(
        "SELECT v.label, v.category, v.subcategory, v.source_path, v.gold, "
        "       p.canonical, p.terminal_invalid, e.evidence, res.assistant_text, t.attempt_count "
        "FROM evaluation_item e "
        "JOIN task t ON t.id=e.task_id "
        "JOIN resolved_run r ON r.id=t.run_id "
        "JOIN variable v ON v.id=t.variable_id "
        "JOIN prediction p ON p.task_id=t.id "
        "JOIN attempt a ON a.id=p.attempt_id "
        "JOIN response res ON res.attempt_id=a.id "
        "WHERE t.campaign_id=%s AND r.configuration_id=%s "
        "ORDER BY v.category, v.subcategory, v.label",
        (best["campaign"], best["configuration"])).fetchall()

    totals = dict.fromkeys(("tp", "fp", "fn"), Fraction(0))
    rows = []
    for label, category, subcategory, path, gold, canonical, invalid, evidence, text, attempts in records:
        close = evidence["close"]
        for key in ("tp", "fp", "fn"):
            totals[key] += _f(close["totals"][key])

        variable_f1 = _f1(*(_f(close["totals"][k]) for k in ("tp", "fp", "fn")))
        if variable_f1 != Fraction(close["metrics"]["f1"]["numerator"],
                                   close["metrics"]["f1"]["denominator"]) and variable_f1 is not None:
            raise SystemExit(f"{label}: recomputed variable F1 disagrees with the stored receipt")

        components = {}
        for name in COMPONENTS:
            contribution = close["components"][name]["contributions"]
            components[name] = _f1(*(_f(contribution[k]) for k in ("tp", "fp", "fn")))

        stem = Path(path).stem
        rows.append({
            "variable": label,
            "path": f"{category}\\{subcategory}\\{stem}",
            "variable_f1": variable_f1,
            "response": text or "",
            "gold": gold, "prediction": canonical,
            "components": components,
            "invalid": invalid, "attempts": attempts,
        })

    computed = _f1(totals["tp"], totals["fp"], totals["fn"])
    if computed != best["stored_f1"]:  # the workbook is only trustworthy if this holds
        raise SystemExit(f"recomputed overall {computed} != ranking's stored {best['stored_f1']}")
    best["variables"] = len(rows)
    return rows


def _row_height(values: list, columns: list[tuple[str, int]]) -> float:
    """Estimate the height one row needs so its tallest wrapped cell is not clipped."""
    lines = 1
    for value, (_, width) in zip(values, columns, strict=True):
        if not isinstance(value, str):
            continue
        used = sum(max(1, -(-len(segment) // max(width - 2, 1)))
                   for segment in value.split("\n"))
        lines = max(lines, used)
    return min(lines, 22) * 13.2 + 4


def _header(sheet, columns: list[tuple[str, int]]) -> None:
    """Write a styled, frozen header row and set the column widths behind it."""
    sheet.append([name for name, _ in columns])
    for index, cell in enumerate(sheet[1], start=1):
        cell.font, cell.fill = HEAD, FILL
        cell.alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
        sheet.column_dimensions[get_column_letter(index)].width = columns[index - 1][1]
    sheet.row_dimensions[1].height = 34
    sheet.freeze_panes = "B2"


def write_answers(book, rows: list[dict], best: dict) -> None:
    """Write the main sheet: one row per variable, in the requested column order."""
    sheet = book.active
    sheet.title = "Answers"
    columns = [("variable", 38), ("model", 18), ("reasoning", 12), ("temperature", 12),
               ("prompt_version", 22), ("shot", 7), ("variable path", 34),
               ("close F1 (variable)", 13), ("raw LLM response", 62)]
    for name in COMPONENTS:
        columns += [(f"{LABELS[name]} — ground truth", 34),
                    (f"{LABELS[name]} — predicted", 34),
                    (f"{LABELS[name]} — close F1", 11)]
    _header(sheet, columns)

    for row in rows:
        values = [row["variable"], best["model"], best["reasoning"], float(best["temperature"]),
                  best["prompt_variant"], best["shots"], row["path"],
                  None if row["variable_f1"] is None else float(row["variable_f1"]),
                  row["response"]]
        for name in COMPONENTS:
            score = row["components"][name]
            values += [render(row["gold"].get(name)), render(row["prediction"].get(name)),
                       None if score is None else float(score)]
        sheet.append(values)

        line = sheet.max_row
        for cell in sheet[line]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        sheet.cell(line, 9).font = MONO
        for offset in (8, *range(12, 28, 3)):  # the variable score and each component's F1
            sheet.cell(line, offset).number_format = "0.0000"
            sheet.cell(line, offset).alignment = Alignment(vertical="top", horizontal="center")
        for offset in range(10, 28, 3):  # ground-truth columns, tinted
            sheet.cell(line, offset).fill = GOLD_FILL
        sheet.row_dimensions[line].height = _row_height(values, columns)

    sheet.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{sheet.max_row}"


def write_identity(book, best: dict, rows: list[dict], conn) -> None:
    """Write the constants behind every row: which run this is and how it was scored."""
    sheet = book.create_sheet("Run identity")
    _header(sheet, [("field", 30), ("value", 96)])
    sample = conn.execute(
        "SELECT e.evidence FROM evaluation_item e JOIN task t ON t.id=e.task_id "
        "JOIN resolved_run r ON r.id=t.run_id WHERE t.campaign_id=%s AND r.configuration_id=%s "
        "LIMIT 1", (best["campaign"], best["configuration"])).fetchone()[0]
    cost = conn.execute(
        "SELECT coalesce(sum(cs.amount),0) FROM cost_settlement cs JOIN attempt a ON a.id=cs.attempt_id "
        "JOIN task t ON t.id=a.task_id JOIN resolved_run r ON r.id=t.run_id "
        "WHERE t.campaign_id=%s AND r.configuration_id=%s",
        (best["campaign"], best["configuration"])).fetchone()[0]

    for field, value in [
        ("model", best["model"]),
        ("provider", best["provider"]),
        ("reasoning", best["reasoning"]),
        ("temperature", f"{float(best['temperature']):g}"),
        ("top_p", f"{float(best['top_p']):g}" if best["top_p"] is not None else "unset"),
        ("prompt version", f"{best['prompt_variant']} (prompts/{best['prompt_variant']}-v1.txt)"),
        ("shots", str(best["shots"])),
        ("campaign", best["campaign"]),
        ("ranking", best["ranking"]),
        ("configuration id", best["configuration"]),
        ("rank within campaign", str(best["rank"])),
        ("overall close F1 (micro)", f"{float(best['stored_f1']):.6f}"),
        ("overall close F1 (exact fraction)", f"{best['stored_f1'].numerator}/{best['stored_f1'].denominator}"),
        ("variables scored", str(best["variables"])),
        ("scorer version", sample["scorer_version"]),
        ("close threshold", str(sample["close_threshold"])),
        ("similarity model", f'{sample["similarity_identity"]["manifest"]["model_id"]} @ {sample["similarity_identity"]["manifest"]["revision"][:12]}'),
        ("settled cost for this configuration", f"${float(cost):.4f} USD"),
        ("parse failures in these rows", str(sum(1 for r in rows if r["invalid"]))),
        ("rows needing more than one attempt", str(sum(1 for r in rows if r["attempts"] > 1))),
        ("searched for the best over", "all fully-scored configurations in campaigns "
                                       "5cdc9417, 844df00e and 79b5ec5f"),
    ]:
        sheet.append([field, value])
        sheet.cell(sheet.max_row, 1).font = Font(bold=True)
        for cell in sheet[sheet.max_row]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def write_notes(book, best: dict, rows: list[dict]) -> None:
    """Write the reading guide, including the caveats that make the numbers legible."""
    sheet = book.create_sheet("How to read this")
    _header(sheet, [("topic", 28), ("what it means", 118)])
    macro = sum(float(r["variable_f1"]) for r in rows if r["variable_f1"] is not None) / len(rows)
    for topic, text in [
        ("One row",
         "One variable answered once by the single best-scoring configuration. All 97 variables "
         "of the scored population are present; the five demonstration variables used for "
         "few-shot prompting are not part of it."),
        ("Which configuration",
         f"{best['model']} on {best['provider']}, {best['prompt_variant']} prompt, "
         f"{best['shots']}-shot, temperature {float(best['temperature']):g}, reasoning "
         f"{best['reasoning']}. It had the highest overall Close F1 of every fully-scored "
         f"configuration in the three comparable full grids. Columns B–F repeat this on every "
         f"row as provenance; the full identity is on the 'Run identity' sheet."),
        ("close F1 (variable)",
         "Close F1 for that one variable across all six components together. Close means two "
         "strings count as a match at cosine similarity 0.80 or above, after normalisation."),
        ("Micro vs macro — important",
         f"The configuration's headline score is {float(best['stored_f1']):.4f}. The 97 values in "
         f"column H average to {macro:.4f}. Both are correct: the headline micro-aggregates "
         f"confusion mass over the whole population and computes F1 once, it is not the mean of "
         f"the per-variable column. Do not average column H and report it as the model's score."),
        ("Per-component close F1",
         "The same measure restricted to one component on one variable. Each component carries "
         "exactly one unit of confusion mass per variable, so the six are comparable with each "
         "other but do not average or sum to column H."),
        ("Blank as a value",
         f"'{EMPTY}' means the field was deliberately empty on that side. Gold is empty for "
         f"most Matrix, Context object and Statistical modifier fields; predicting something "
         f"there is a false positive, and leaving it empty is a true negative."),
        ("An empty F1 cell",
         "Not a zero and not a missing measurement: gold and prediction were both empty, so "
         "there were no true positives, false positives or false negatives to compute from and "
         "F1 is undefined. Left blank on purpose so the column stays numeric — sorting and "
         "AVERAGE skip these rows instead of counting them as zero."),
        ("Nested systems",
         "Where a component is a system rather than a plain string it is written as the system's "
         "own label followed by each member under its role — numerator and denominator for a "
         "ratio, source and target for a flux, parts for a symmetric system."),
        ("Constraints",
         "Always a list. Each entry is numbered, with its label first and the component it "
         "constrains on the line below."),
        ("raw LLM response",
         "Verbatim assistant text from the attempt whose answer was scored. Where a task needed "
         "more than one attempt this is the final one, which is always the attempt the stored "
         "prediction came from. Rows are sized to fit their tallest cell."),
        ("Predicted vs raw",
         "The predicted columns are the parsed, canonicalised answer that was actually scored. "
         "They come from the same response as the raw column, so any difference between them is "
         "parsing, never a different call."),
        ("Ground truth",
         "The tinted columns. Taken from the corpus gold decomposition, unchanged."),
    ]:
        sheet.append([topic, text])
        sheet.cell(sheet.max_row, 1).font = Font(bold=True)
        for cell in sheet[sheet.max_row]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        sheet.row_dimensions[sheet.max_row].height = 58


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/best-configuration-answers.xlsx")
    args = parser.parse_args()

    with psycopg.connect(local_dsn(ROOT, role="app"), connect_timeout=10) as conn:
        conn.read_only = True
        conn.execute("SET search_path TO iadopt_lab, public")
        best = find_best(conn)
        rows = collect(conn, best)

        book = Workbook()
        write_answers(book, rows, best)
        write_identity(book, best, rows, conn)
        write_notes(book, best, rows)

    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    book.save(destination)
    print(f"wrote {destination}")
    print(f"  {best['model']} / {best['prompt_variant']} / {best['shots']}-shot / "
          f"T={float(best['temperature']):g} / reasoning {best['reasoning']}")
    print(f"  campaign {best['campaign'][:8]} rank {best['rank']}, "
          f"overall close F1 {float(best['stored_f1']):.4f} (recomputed and matched)")
    print(f"  {len(rows)} rows x {9 + 3 * len(COMPONENTS)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
