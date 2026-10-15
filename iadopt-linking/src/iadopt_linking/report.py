"""Render a run's summary as a CSV of per-slot outcomes and a Markdown report."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from typing import Any

CSV_COLUMNS = (
    "slot_id", "source_path", "role", "term", "gold_vocabulary", "gold_local_id", "category",
    "correct", "selected_source", "selected_local_id", "selected_label", "selected_score",
    "gold_in_pool", "pool_size")
MISS_CATEGORIES = ("wrong_within_source", "wrong_source", "gold_not_retrieved", "no_candidates")
EXAMPLES_PER_CATEGORY = 6
# The manuscript's entity-linking table (URI accuracy: average accuracy per variable, only cases
# with a Wikidata ground-truth link). Its inputs are not in this repository, so these are quoted.
MANUSCRIPT_NAIVE = "0.656"
MANUSCRIPT_RERANKER = "0.759"


def ratio(receipt: Mapping[str, int]) -> str:
    """Show a receipt as counts and a three-decimal value.

    Args: receipt: {"numerator", "denominator"}.
    Returns: "n/d = 0.123", or "n/a" when nothing was judged.
    Raises: None.
    Side effects: none; the decimal is for display only.
    """
    numerator, denominator = receipt["numerator"], receipt["denominator"]
    return "n/a" if denominator == 0 else f"{numerator}/{denominator} = {numerator / denominator:.3f}"


def value(receipt: Mapping[str, int]) -> str:
    """Show a receipt as a three-decimal value only.

    Args: receipt: {"numerator", "denominator"}.
    Returns: "0.123", or "n/a" when the denominator is zero.
    Raises: None.
    Side effects: none.
    """
    denominator = receipt["denominator"]
    return "n/a" if denominator == 0 else f"{receipt['numerator'] / denominator:.3f}"


def outcomes_csv(outcomes: Sequence[Mapping[str, Any]]) -> str:
    """Write the per-slot outcomes as CSV.

    Args: outcomes: evaluate_slot results.
    Returns: CSV text with one row per slot, in the order given.
    Raises: None.
    Side effects: none.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for o in outcomes:
        selected = o["selected"] or {}
        writer.writerow([
            o["slot_id"], o["source_path"], o["role"], o["term"], o["gold"][0], o["gold"][1],
            o["category"], int(o["correct"]), selected.get("source", ""),
            selected.get("local_id", ""), selected.get("label", ""),
            "" if "score" not in selected else f"{selected['score']:.6f}",
            int(o["gold_in_pool"]), o["pool_size"]])
    return buffer.getvalue()


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Lay out a Markdown table.

    Args: header: column titles; rows: cell text per row.
    Returns: the table's lines.
    Raises: None.
    Side effects: none.
    """
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _verdict(passed: bool | None) -> str:
    """Name a check's tri-state result.

    Args: passed: True, False, or None for a check that was not run.
    Returns: "pass", "FAIL" or "not run".
    Raises: None.
    Side effects: none.
    """
    return "not run" if passed is None else "pass" if passed else "FAIL"


def _header(manifest: Mapping[str, Any]) -> list[str]:
    """Write the title and the identity paragraph.

    Args: manifest: run manifest.
    Returns: Markdown lines.
    Raises: KeyError if the manifest lacks a field.
    Side effects: none.
    """
    model = manifest["model"]
    devices = ", ".join(f"{name} ({count} slots)" for name, count in model["devices"].items())
    period = manifest["evidence_period"]
    return [
        f"# {manifest['title']}", "",
        f"Run `{manifest['run_id']}`, report built {manifest['created_at']}. Evidence retrieved "
        f"{period['first_retrieval']} to {period['last_retrieval']}. Corpus I-ADOPT "
        f"{manifest['corpus']['tag']} ({manifest['corpus']['commit'][:10]}). Evidence code "
        f"`{manifest['evidence_code_sha256'][:12]}`, all code `{manifest['code']['source_sha256'][:12]}`. "
        f"Reranker `{model['model']}` (revision `{(model['revision'] or '-')[:10]}`), "
        f"scored on {devices}.",
        "",
        "For every gold slot whose IRI names a controlled-vocabulary term, the slot's term was "
        "searched in Wikidata, QUDT quantity kinds and OM quantities (at most 10 candidates "
        "each), the pooled candidates were scored by the existing Qwen3 reranker with the "
        "variable's definition as context, and the best-scoring candidate was compared with "
        "the gold identity (vocabulary and local identifier).", ""]


def _result(pops: Mapping[str, Any]) -> list[str]:
    """Write the headline table.

    Args: pops: summary populations.
    Returns: Markdown lines.
    Raises: KeyError if a population is missing.
    Side effects: none.
    """
    rows = []
    for name, label in (("all", "All controlled-vocabulary slots"),
                        ("reachable", "Gold in Wikidata, QUDT or OM"),
                        ("gold:wikidata", "Gold in Wikidata"), ("gold:qudt", "Gold in QUDT"),
                        ("gold:om", "Gold in OM")):
        if name in pops:
            p = pops[name]
            rows.append([label, str(p["n"]), str(p["distinct_terms"]), ratio(p["top1"]),
                         value(p["top1_macro_by_variable"])])
    if "unreachable" in pops:
        p = pops["unreachable"]
        rows.append(["Gold in another vocabulary (not searchable here)", str(p["n"]),
                     str(p["distinct_terms"]), ratio(p["top1"]), value(p["top1_macro_by_variable"])])
    return ["## Result", ""] + _table(
        ["Population", "Slots", "Distinct terms", "Top-1 accuracy", "Mean per variable"], rows) + [
        "", "Mean per variable is the manuscript's style of measure (accuracy per variable, then "
        "averaged); slots repeat terms, so the distinct-term column is the number of independent "
        "questions asked.", ""]


def _manuscript(pops: Mapping[str, Any]) -> list[str]:
    """Write the comparison with the manuscript's entity-linking table.

    Args: pops: summary populations.
    Returns: Markdown lines; empty when there is no Wikidata gold.
    Raises: KeyError if the Wikidata population lacks a metric.
    Side effects: none.
    """
    if "gold:wikidata" not in pops:
        return []
    w = pops["gold:wikidata"]
    rows = [
        ["First Wikidata hit, no reranker (naive)", ratio(w["native_first_hit_top1"]),
         value(w["native_first_hit_top1_macro_by_variable"]), MANUSCRIPT_NAIVE],
        ["Qwen3 reranker on the Wikidata candidates alone", ratio(w["wikidata_only_top1"]),
         value(w["wikidata_only_top1_macro_by_variable"]), MANUSCRIPT_RERANKER],
        ["Qwen3 reranker on all three sources pooled (this baseline)", ratio(w["top1"]),
         value(w["top1_macro_by_variable"]), "-"]]
    return ["## Comparison with the manuscript", "",
            "The manuscript's entity-linking table reports URI accuracy as the average accuracy "
            "per variable, over cases that have a Wikidata ground-truth link. The first two rows "
            "below use that measure on every Wikidata-gold slot of this report, members of "
            "ratios and source/target parts included; whether the manuscript counted exactly "
            "these cases is not established here.", ""] + _table(
        ["Measure (Wikidata gold)", "Slots correct", "Mean per variable", "Manuscript"], rows) + [
        "", "The manuscript's figures are quoted, not reproduced: their inputs are not in this "
        "repository. A difference is not explained by this run. Candidates here come from the "
        "live services on the run date, Wikidata is asked for ten candidates where the API "
        "default is seven, and the slot population and prompts may differ from the manuscript's.",
        ""]


def _misses(pops: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    """Write where the misses come from, the diagnostics and the source-confusion table.

    Args: pops: summary populations; summary: metrics.summarise output.
    Returns: Markdown lines.
    Raises: KeyError if the reachable population or a metric is missing.
    Side effects: none.
    """
    out = ["## Where the misses come from (reachable slots)", ""]
    cats = pops["reachable"]["categories"]
    reach_n = pops["reachable"]["n"]
    out += _table(["Outcome", "Slots", "Share"], [
        [name.replace("_", " "), str(cats[name]), f"{cats[name] / reach_n:.3f}"]
        for name in ("correct", "wrong_within_source", "wrong_source", "gold_not_retrieved",
                     "no_candidates")])
    r = pops["reachable"]
    margins = summary["margins"]["below_1e-3"]
    out += ["", "## Diagnostics (reachable slots)", ""]
    out += _table(["Question", "Answer"], [
        ["Is the gold term among the pooled candidates?", ratio(r["gold_in_pool"])],
        ["Does the reranker pick it when it is?", ratio(r["top1_given_gold_in_pool"])],
        ["Is the chosen candidate from the gold's own source? (slots with any candidate)",
         ratio(r["routed_to_gold_source"])],
        ["Top-1 if the source were known (best score within the gold's source)",
         ratio(r["oracle_routed_top1"])],
        ["Top-1 of the gold source's own first hit (no reranker)", ratio(r["native_first_hit_top1"])],
        ["Top-1 if a QUDT kind and its `wikidataMatch` item counted as the same entity",
         ratio(r["top1_with_qudt_equivalence"])],
        ["Ceiling for a system that sees only the term (majority gold per term), reachable slots",
         ratio(r["consistency_ceiling"])],
        ["Ceiling for such a system over all slots",
         ratio(pops["all"]["consistency_ceiling"])],
        ["Selections decided by a top-two score margin below 1e-3 (slots with 2+ candidates)",
         ratio(margins)]])
    confusion = summary["routing_confusion"]
    sources = ("wikidata", "qudt", "om")
    out += ["", "## Which source wins (reachable slots with any candidate)", "",
            "Rows are the gold's source, columns the source of the selected candidate.", ""]
    out += _table(["Gold in \\ chosen from", *sources], [
        [name, *[str(confusion.get(f"{name}>{chosen}", 0)) for chosen in sources]]
        for name in sources])
    return out


def _tables(pops: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    """Write the threshold, grouping and repetition tables.

    Args: pops: summary populations; summary: metrics.summarise output.
    Returns: Markdown lines.
    Raises: KeyError if a group lacks a metric.
    Side effects: none.
    """
    out = ["", "## Confidence threshold (all slots)", "",
           "A selection counts as an answer only if its score reaches the threshold. Scores "
           "saturate near 1, so the grid continues above 0.9.", ""]
    out += _table(["Threshold", "Answered", "Precision when answered"], [
        [value(t["threshold"]), ratio(t["coverage"]), ratio(t["precision"])]
        for t in summary["thresholds_all"]])
    out += ["", "## By slot, role and term length", ""]
    rows = []
    for name in sorted(n for n in pops if n.startswith(("slot:", "role:", "words:"))):
        p = pops[name]
        rows.append([name, str(p["n"]), ratio(p["top1"]), ratio(p["gold_in_pool"])])
    out += _table(["Group", "Slots", "Top-1", "Gold in pool"], rows)
    out += ["", "## Most repeated terms", "",
            "A few terms are asked many times, so their outcome weighs on every slot-weighted "
            "figure above.", ""]
    out += _table(["Term", "Slots", "Wrong"], [
        [f"`{t['term']}`", str(t["slots"]), str(t["wrong"])] for t in summary["most_repeated_terms"]])
    return out


def _examples(outcomes: Sequence[Mapping[str, Any]]) -> list[str]:
    """Write a few examples of each kind of miss.

    Args: outcomes: evaluate_slot results.
    Returns: Markdown lines.
    Raises: KeyError for a malformed outcome.
    Side effects: none.
    """
    out = ["", "## Examples of misses", "",
           f"Up to {EXAMPLES_PER_CATEGORY} per outcome, first by slot id. Gold is "
           "vocabulary:identifier.", ""]
    for category in MISS_CATEGORIES:
        shown = [o for o in sorted(outcomes, key=lambda o: o["slot_id"]) if o["category"] == category]
        if not shown:
            continue
        out += [f"**{category.replace('_', ' ')}** ({len(shown)} slots)", ""]
        for o in shown[:EXAMPLES_PER_CATEGORY]:
            chosen = o["selected"]
            picked = "nothing" if chosen is None else (
                f"{chosen['label']!r} ({chosen['source']}:{chosen['local_id']}, {chosen['score']:.2f})")
            out.append(f"- `{o['term']}` ({o['source_path'].rsplit('/', 1)[-1]}) -> {picked}; "
                       f"gold {o['gold'][0]}:{o['gold'][1]}")
        out.append("")
    return out


def render_report(manifest: Mapping[str, Any], summary: Mapping[str, Any],
                  outcomes: Sequence[Mapping[str, Any]]) -> str:
    """Write the Markdown report of a finished run.

    Args: manifest: run identity, configuration, health checks; summary: metrics.summarise
    output; outcomes: evaluate_slot results.
    Returns: Markdown text.
    Raises: KeyError if the summary lacks a population the report needs.
    Side effects: none.
    """
    pops = summary["populations"]
    out = _header(manifest) + _result(pops) + _manuscript(pops) + _misses(pops, summary)
    out += _tables(pops, summary) + _examples(outcomes)
    out += ["## Run health", ""]
    out += _table(["Check", "Result", "Detail"], [
        [c["name"], _verdict(c["passed"]), c["detail"]] for c in manifest["checks"]])
    out += ["", "## Notes", ""]
    out += [f"- {note}" for note in manifest["notes"]]
    out += ["", "Contains data retrieved from Wikidata (CC0), QUDT (CC BY 4.0, attribution to "
            "QUDT.org) and the Ontology of units of Measure (CC BY 4.0) via the EBI Ontology "
            "Lookup Service.", ""]
    return "\n".join(out)
