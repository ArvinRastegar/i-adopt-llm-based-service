"""Render a scorer's run: the report, the head-to-head with Qwen3, and the tables for later analysis.

Pure functions of the stored summary and outcomes, so a committed report can always be rebuilt from
the evidence beside it. The report keeps the baseline's shape (result table, where misses come
from, diagnostics, source confusion, thresholds, groups, examples, health) so the experiments
read side by side, and adds what only a scorer that answers with probabilities has: passes and
order, abstention, and what its confidence is worth.

A ``Voice`` names the scorer and holds the few paragraphs that depend on what it is (how it was
asked, what its own confidence means, where its cost comes from). ``JEV_VOICE`` is Jev's; the
text it holds is what the committed Jev report says, word for word.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .jev import ARM_NOUL, PRIMARY_PASS
from .report import _examples, _table, _verdict, ratio, value


@dataclass(frozen=True)
class Voice:
    """How one scorer is named and described in the files a run publishes.

    ``key`` prefixes the published names of its fields, columns and signals; ``name`` is what the
    prose calls it. ``own_confidence_word`` names the scorer's own confidence in a sentence,
    ``own_confidence_short`` in a table heading and ``own_confidence_label`` in a table row. ``cost_sentence``, ``intro`` and ``confidence_notes``
    write the paragraphs that depend on what the scorer is, ``reliability_note`` ends the
    reliability table, ``attribution`` closes the report and ``extra`` adds sections (such as a
    comparison with another run) after the paired table. ``pointwise`` says whether a pointwise
    arm can exist, which decides whether its columns are written. ``post_hoc``,
    ``term_level_note`` and ``term_heading`` say how the analyses Jev's report gained after its
    first results are labelled: post hoc for Jev, exploratory for a run that planned them.
    """

    key: str
    name: str
    own_confidence_word: str
    own_confidence_short: str
    own_confidence_label: str
    pointwise: bool
    cost_sentence: Callable[[Mapping[str, Any]], str]
    intro: Callable[[Mapping[str, Any]], str]
    confidence_notes: Callable[[Mapping[str, Any]], list[str]]
    reliability_note: str
    attribution: str
    post_hoc: str
    term_level_note: str
    term_heading: str
    extra: Callable[[Mapping[str, Any]], list[str]] = lambda summary: []


def outcome_columns(key: str, pointwise: bool) -> tuple[str, ...]:
    """Name the columns of outcomes.csv.

    Args: key: the scorer's key; pointwise: whether the pointwise columns are written.
    Returns: the column names: the slot, the scorer's answer and signals, optionally the pointwise
    arm, and the Qwen3 baseline's answer.
    Raises: None.
    Side effects: none.
    """
    return ("slot_id", "source_path", "role", "term", "query_key", "gold_vocabulary",
            "gold_local_id", "reachable", "pool_size", "gold_in_pool",
            f"{key}_category", f"{key}_correct", f"{key}_equivalent_correct",
            f"{key}_selected_source", f"{key}_selected_local_id", f"{key}_selected_label",
            f"{key}_selected_probability", f"{key}_margin", f"{key}_confidence", f"{key}_p_none",
            f"{key}_own_choice", f"{key}_selected_position", f"{key}_pass_agreement",
            *(("pointwise_correct", "pointwise_selected_local_id",
               "pointwise_selected_probability") if pointwise else ()),
            "qwen3_category", "qwen3_correct", "qwen3_selected_source", "qwen3_selected_local_id",
            "qwen3_selected_label", "qwen3_selected_score", "qwen3_margin")


def confidence_columns(key: str) -> tuple[str, ...]:
    """Name the columns of confidence.csv.

    Args: key: the scorer's key.
    Returns: the column names: one row is a slot in a pass.
    Raises: None.
    Side effects: none.
    """
    return ("slot_id", "pass", "arm", "term", "query_key", "gold_vocabulary", "gold_local_id",
            "reachable", "asked", "pool_size", "gold_in_pool", "category", "correct",
            "equivalent_correct", "selected_source", "selected_local_id", "selected_score",
            "margin", f"{key}_confidence", "p_none", f"{key}_own_choice", "selected_position")


def signal_labels(voice: Voice) -> dict[str, str]:
    """Label the confidence signals for a table.

    Args: voice: the scorer.
    Returns: signal name -> label, the scorer's signals first and the baseline's after them.
    Raises: None.
    Side effects: none.
    """
    return {
        f"{voice.key}_selected_probability": f"{voice.name}: probability of the selected candidate",
        f"{voice.key}_confidence": voice.own_confidence_label,
        f"{voice.key}_margin": f"{voice.name}: top-two probability margin",
        f"{voice.key}_candidate_mass": f"{voice.name}: probability not given to `none_of_these`",
        f"{voice.key}_pass_agreement": f"{voice.name}: share of the Choice passes that agree",
        "qwen3_score": "Qwen3: score of the selected candidate",
        "qwen3_margin": "Qwen3: top-two score margin"}


PROBABILITY_COLUMNS = (
    "slot_id", "pass", "arm", "position", "source", "local_id", "label", "probability",
    "is_gold", "is_selected")


def num(x: float | None) -> str:
    """Write a number for a CSV cell: short, exact enough, empty for None.

    Args: x: a float or None.
    Returns: the number to six decimals without trailing zeros, or "".
    Raises: None.
    Side effects: none.
    """
    if x is None:
        return ""
    text = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _signed(receipt: Mapping[str, int]) -> str:
    """Show a signed difference as a decimal.

    Args: receipt: {"numerator", "denominator"}.
    Returns: "+0.035" style text.
    Raises: None.
    Side effects: none.
    """
    return f"{receipt['numerator'] / receipt['denominator']:+.3f}"


def _terms(block: Mapping[str, Any] | None, name: str = "Jev", other: str = "Qwen3") -> str:
    """Describe a comparison made once per term.

    Args: block: paired.term_level output, or None; name: the first system; other: the second.
    Returns: a sentence fragment with the terms won by each system and the sign-test p-value.
    Raises: KeyError for a malformed block.
    Side effects: none.
    """
    if block is None:
        return "no terms"
    mean = block["mean_difference"]
    return (f"{name} better on {block['first_better']} of {block['terms']} terms, {other} better on "
            f"{block['second_better']}, equal on {block['equal']} (exact sign test p "
            f"{_p(block['sign_test_p'])}; term-averaged difference {_signed(mean)})")


def _p(receipt: Mapping[str, int] | None) -> str:
    """Show an exact p-value as a decimal.

    Args: receipt: the p-value as a fraction, or None.
    Returns: "0.0123", "<0.0001" or "-".
    Raises: None.
    Side effects: none.
    """
    if receipt is None:
        return "-"
    p = receipt["numerator"] / receipt["denominator"]
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def _difference(block: Mapping[str, Any] | None) -> str:
    """Show an accuracy difference with its interval.

    Args: block: compare output.
    Returns: "+0.035 (95% interval -0.012 to +0.080)", or "-" without a difference.
    Raises: KeyError for a malformed block.
    Side effects: none.
    """
    if block is None or block["difference"] is None:
        return "-"
    d = block["difference"]
    interval = d["interval"]
    return (f"{_signed(d['difference'])} ({float(interval['level']):.0%} interval "
            f"{interval['lower']:+.3f} to {interval['upper']:+.3f})")


def outcomes_csv(own: Sequence[Mapping[str, Any]], qwen: Sequence[Mapping[str, Any]],
                 results: Mapping[str, Sequence[Mapping[str, Any]]],
                 agreement: Sequence[Fraction | None], slot_paths: Mapping[str, str],
                 specs: Mapping[str, Any], voice: Voice | None = None) -> str:
    """Write one row per slot: the scorer's primary answer and signals beside Qwen3's.

    Args: own, qwen: evaluate_slot outcomes for the same slots (the scorer's, the baseline's);
    results: pass_results by pass name; agreement: per slot, the exact share of Choice passes that
    agree with the primary selection; slot_paths: slot id -> source path; specs: pass name ->
    PassSpec; voice: the scorer (Jev's by default).
    Returns: CSV text, one row per slot in population order.
    Raises: KeyError for a malformed outcome.
    Side effects: none.
    """
    voice = voice or JEV_VOICE
    primary = results[PRIMARY_PASS]
    pointwise = next((results[name] for name, spec in specs.items() if spec.arm == ARM_NOUL), None)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(outcome_columns(voice.key, voice.pointwise))
    for index, (j, q) in enumerate(zip(own, qwen, strict=True)):
        r = primary[index]
        chosen, other = j["selected"] or {}, q["selected"] or {}
        own_choice = r["own_choice_key"] or ""
        position = ("" if r["selected_index"] is None
                    else str(r["order"].index(r["selected_index"]) + 1))
        point = pointwise[index]["outcome"] if pointwise else None
        point_chosen = (point["selected"] or {}) if point else {}
        row = [
            j["slot_id"], slot_paths[j["slot_id"]], j["role"], j["term"], j["query_key"],
            j["gold"][0], j["gold"][1], int(j["reachable"]), j["pool_size"], int(j["gold_in_pool"]),
            j["category"], int(j["correct"]), int(j["equivalent_correct"]),
            chosen.get("source", ""), chosen.get("local_id", ""), chosen.get("label", ""),
            num(chosen.get("score")), num(r["margin"]), num(r["own_confidence"]), num(r["p_none"]),
            own_choice, position,
            num(None if agreement[index] is None else float(agreement[index]))]
        if voice.pointwise:
            row += ["" if point is None else int(point["correct"]), point_chosen.get("local_id", ""),
                    num(point_chosen.get("score"))]
        row += [q["category"], int(q["correct"]), other.get("source", ""), other.get("local_id", ""),
                other.get("label", ""), num(other.get("score")), num(q["margin"])]
        writer.writerow(row)
    return buffer.getvalue()


def confidence_csv(results: Mapping[str, Sequence[Mapping[str, Any]]],
                   specs: Mapping[str, Any], voice: Voice | None = None) -> str:
    """Write one row per slot and pass with every number that says how sure the scorer was.

    Args: results: pass_results by pass name; specs: pass name -> PassSpec; voice: the scorer
    (Jev's by default).
    Returns: CSV text, pass by pass in population order.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
    voice = voice or JEV_VOICE
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(confidence_columns(voice.key))
    for name, rows in results.items():
        for r in rows:
            o = r["outcome"]
            chosen = o["selected"] or {}
            own = r.get("own_choice_key") or ""
            position = ("" if r["selected_index"] is None
                        else str(r["order"].index(r["selected_index"]) + 1))
            writer.writerow([
                o["slot_id"], name, specs[name].arm, o["term"], o["query_key"], o["gold"][0],
                o["gold"][1], int(o["reachable"]), int(r["asked"]), o["pool_size"],
                int(o["gold_in_pool"]),
                o["category"], int(o["correct"]), int(o["equivalent_correct"]),
                chosen.get("source", ""), chosen.get("local_id", ""), num(chosen.get("score")),
                num(r["margin"]), num(r["own_confidence"]), num(r["p_none"]), own, position])
    return buffer.getvalue()


def probabilities_csv(results: Mapping[str, Sequence[Mapping[str, Any]]],
                      specs: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> str:
    """Write one row per slot, pass and candidate with the probability the scorer gave it.

    Args: results: pass_results by pass name; specs: pass name -> PassSpec; records: the
    baseline's scores records by slot id (they hold the candidates and the gold).
    Returns: CSV text; ``position`` is where the candidate was shown, 1-based. For a Choice pass
    ``probability`` is the option's probability, for a pointwise pass it is the probability of
    yes.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(PROBABILITY_COLUMNS)
    for name, rows in results.items():
        for r in rows:
            if not r["asked"]:
                continue
            base = records[r["slot_id"]]
            gold = (base["gold"]["vocabulary"], base["gold"]["local_id"])
            for place, index in enumerate(r["order"], start=1):
                c = base["pool"][index]
                writer.writerow([r["slot_id"], name, specs[name].arm, place, c["source"],
                                 c["local_id"], c["label"], num(r["scores"][index]),
                                 int((c["vocabulary"], c["local_id"]) == gold),
                                 int(index == r["selected_index"])])
    return buffer.getvalue()


def _pass_phrase(spec: list[Any]) -> str:
    """Describe one further pass in words.

    Args: spec: [name, arm, order] as the plan lists it.
    Returns: a short phrase.
    Raises: None.
    Side effects: none.
    """
    _, arm, order = spec
    if arm == ARM_NOUL:
        return "one yes/no question per candidate (TypeSafe's re-ranking pattern)"
    if order == "shuffle-a":
        return "an identical repeat of the primary request"
    return "the reverse order" if order == "reverse-a" else f"another candidate order ({order})"


def _jev_cost_sentence(manifest: Mapping[str, Any]) -> str:
    """Say how many requests Jev answered and what OpenRouter reports they cost.

    Args: manifest: run manifest.
    Returns: the sentence that follows the period of the requests in the header.
    Raises: KeyError if the manifest lacks a field.
    Side effects: none.
    """
    spend = manifest["spend"]
    return (f"{spend['requests']} requests answered ({spend['input_tokens']:,} input tokens), "
            f"${spend['cost_usd']} reported by OpenRouter in `usage.cost` for all "
            f"{spend['responses_stored']} stored responses (not reconciled with the account's "
            "billing).")


def _jev_intro(manifest: Mapping[str, Any]) -> str:
    """Describe how Jev was asked and what the primary result is.

    Args: manifest: run manifest.
    Returns: the paragraph that follows the identity paragraph.
    Raises: KeyError if the manifest lacks a field.
    Side effects: none.
    """
    cap = manifest["prompt"]["max_description_chars"]
    others = [p for p in manifest["passes"] if p[0] != PRIMARY_PASS]
    again = ("The same pool was also asked, to see how much the result depends on order and on "
             "chance: " + "; ".join(_pass_phrase(p) for p in others) + ". ") if others else ""
    return ("Jev is a typed-decision model: it answers a question with a probability for each option "
            "instead of generating text. Each slot's pooled candidates, exactly as the Qwen3 baseline "
            f"saw them (label and description {'uncapped' if cap is None else f'capped at {cap} characters'}"
            ", no source tag), were put to Jev as one Choice question with an extra `none_of_these` "
            "option. The primary result, fixed in the run's plan and in DECISIONS L-012 before the "
            "first request (no commit predates the run), is the first Choice pass (a seeded candidate "
            "order) with the forced choice: the candidate with the highest probability, "
            "`none_of_these` ignored, ties decided as in the baseline. " + again + "Everything except "
            "the primary result is exploratory.")


def _header(manifest: Mapping[str, Any], voice: Voice) -> list[str]:
    """Write the title and the identity paragraphs.

    Args: manifest: run manifest; voice: the scorer.
    Returns: Markdown lines.
    Raises: KeyError if the manifest lacks a field.
    Side effects: none.
    """
    model, baseline = manifest["model"], manifest["baseline"]
    period = manifest["evidence_period"]
    return [
        f"# {manifest['title']}", "",
        f"Run `{manifest['run_id']}`, report built {manifest['created_at']}. Requests sent "
        f"{period['first_request']} to {period['last_request']}: {voice.cost_sentence(manifest)} "
        f"Corpus "
        f"I-ADOPT {manifest['corpus']['tag']} ({manifest['corpus']['commit'][:10]}). Candidates "
        f"and Qwen3 scores come from baseline `{baseline['folder']}` (scores "
        f"`{baseline['scores_sha256'][:12]}`), not from new retrieval. Evidence code "
        f"`{manifest['evidence_code_sha256'][:12]}`, all code `{manifest['code']['source_sha256'][:12]}`, "
        f"prompt `{manifest['prompt']['fingerprint'][:12]}`. Model requested `{model['requested']}`, "
        f"served as `{', '.join(model['served'])}` by {', '.join(model['providers'])}.", "",
        voice.intro(manifest), ""]


def _result(pops: Mapping[str, Any], base: Mapping[str, Any], analysis: Mapping[str, Any],
            voice: Voice) -> list[str]:
    """Write the headline table with the baseline beside it.

    Args: pops: the primary pass's summary populations; base: the baseline's summary populations;
    analysis: the analysis block; voice: the scorer.
    Returns: Markdown lines.
    Raises: KeyError if a population is missing.
    Side effects: none.
    """
    labels = (("reachable", "Gold in Wikidata, QUDT or OM (primary)"),
              ("all", "All controlled-vocabulary slots"), ("gold:wikidata", "Gold in Wikidata"),
              ("gold:qudt", "Gold in QUDT"), ("gold:om", "Gold in OM"))
    rows = []
    for name, label in labels:
        if name not in pops:
            continue
        p, b = pops[name], base[name]
        compared = analysis["primary_vs_qwen3"].get(name)
        rows.append([label, str(p["n"]), str(p["distinct_terms"]), ratio(p["top1"]),
                     value(p["top1_macro_by_variable"]), ratio(b["top1"]),
                     _difference(compared), _p(compared["mcnemar_p"]) if compared else "-"])
    if "unreachable" in pops:
        p = pops["unreachable"]
        rows.append(["Gold in another vocabulary (not searchable here)", str(p["n"]),
                     str(p["distinct_terms"]), ratio(p["top1"]), value(p["top1_macro_by_variable"]),
                     ratio(base["unreachable"]["top1"]), "-", "-"])
    return ["## Result", ""] + _table(
        ["Population", "Slots", "Distinct terms", f"{voice.name} top-1",
         f"{voice.name} mean per variable", "Qwen3 top-1", f"{voice.name} minus Qwen3",
         "McNemar p"], rows) + [
        "", "Both systems were judged by the same code on the same slots and the same candidates. "
        f"The interval comes from resampling whole terms ({pops['all']['distinct_terms']} terms ask "
        f"{pops['all']['n']} slots, so slots are not independent evidence); McNemar's p treats "
        "slots as independent and is therefore optimistic. Mean per variable is the manuscript's "
        "style of measure.", ""]


def _paired(analysis: Mapping[str, Any], voice: Voice) -> list[str]:
    """Write the paired table for the primary population.

    Args: analysis: the analysis block; voice: the scorer.
    Returns: Markdown lines.
    Raises: KeyError if the comparison is missing.
    Side effects: none.
    """
    c = analysis["primary_vs_qwen3"]["reachable"]
    eq = analysis["primary_vs_qwen3_equivalence"]
    t, te = c["terms"], eq["terms"]
    name = voice.name
    return [f"## {name} against Qwen3 on the reachable slots", ""] + _table(
        ["", "Qwen3 right", "Qwen3 wrong"],
        [[f"{name} right", str(c["both"]), str(c["only_first"])],
         [f"{name} wrong", str(c["only_second"]), str(c["neither"])]]) + [
        "",
        f"Strict identity: {name} minus Qwen3 {_difference(c)}; exact McNemar p {_p(c['mcnemar_p'])}.",
        f"Counted once per term instead of once per slot ({voice.term_level_note}): "
        f"{_terms(t, name)}.",
        f"If a QUDT kind and its `wikidataMatch` item counted as one entity: {name} minus Qwen3 "
        f"{_difference(eq)}; {_terms(te, name)}. (An undecided question of the project; strict "
        "identity is the headline.)", ""]


def _misses(pops: Mapping[str, Any], base: Mapping[str, Any], voice: Voice) -> list[str]:
    """Write where misses come from and the diagnostics, the scorer beside Qwen3.

    Args: pops: the primary pass's summary populations; base: the baseline's; voice: the scorer.
    Returns: Markdown lines.
    Raises: KeyError if the reachable population or a metric is missing.
    Side effects: none.
    """
    r, b = pops["reachable"], base["reachable"]
    n = r["n"]
    out = ["## Where the misses come from (reachable slots)", ""]
    out += _table(["Outcome", f"{voice.name} slots", f"{voice.name} share", "Qwen3 slots",
                   "Qwen3 share"], [
        [name.replace("_", " "), str(r["categories"][name]), f"{r['categories'][name] / n:.3f}",
         str(b["categories"][name]), f"{b['categories'][name] / n:.3f}"]
        for name in ("correct", "wrong_within_source", "wrong_source", "gold_not_retrieved",
                     "no_candidates")])
    out += ["", "## Diagnostics (reachable slots)", ""]
    out += _table(["Question", voice.name, "Qwen3"], [
        ["Is the gold term among the pooled candidates?", ratio(r["gold_in_pool"]),
         ratio(b["gold_in_pool"])],
        ["Does the scorer pick it when it is?", ratio(r["top1_given_gold_in_pool"]),
         ratio(b["top1_given_gold_in_pool"])],
        ["Is the chosen candidate from the gold's own source? (slots with any candidate)",
         ratio(r["routed_to_gold_source"]), ratio(b["routed_to_gold_source"])],
        ["Top-1 if the source were known (best score within the gold's source)",
         ratio(r["oracle_routed_top1"]), ratio(b["oracle_routed_top1"])],
        ["Top-1 of the gold source's own first hit (no scorer)", ratio(r["native_first_hit_top1"]),
         ratio(b["native_first_hit_top1"])],
        ["Top-1 if a QUDT kind and its `wikidataMatch` item counted as the same entity",
         ratio(r["top1_with_qudt_equivalence"]), ratio(b["top1_with_qudt_equivalence"])],
        ["Ceiling for a system that sees only the term (majority gold per term)",
         ratio(r["consistency_ceiling"]), ratio(b["consistency_ceiling"])]])
    if "gold:wikidata" in pops:
        w, bw = pops["gold:wikidata"], base["gold:wikidata"]
        out += ["", f"On the Wikidata-gold slots ({voice.post_hoc}), picking the best-scoring "
                "Wikidata candidate only "
                f"(for {voice.name} the probabilities of the pooled question, restricted to "
                "Wikidata; for Qwen3 the manuscript's \"reranker alone\" setting): "
                f"{voice.name} {ratio(w['wikidata_only_top1'])}, "
                f"Qwen3 {ratio(bw['wikidata_only_top1'])}."]
    return out


def _routing(summary: Mapping[str, Any], voice: Voice) -> list[str]:
    """Write the source-confusion tables for both systems.

    Args: summary: the run summary; voice: the scorer.
    Returns: Markdown lines.
    Raises: KeyError for a malformed summary.
    Side effects: none.
    """
    sources = ("wikidata", "qudt", "om")
    out = ["", "## Which source wins (reachable slots with any candidate)", "",
           "Rows are the gold's source, columns the source of the selected candidate."]
    for label, block in ((voice.name, summary["primary"]), ("Qwen3", summary["baseline"])):
        confusion = block["routing_confusion"]
        out += ["", f"**{label}**", ""]
        out += _table(["Gold in \\ chosen from", *sources], [
            [name, *[str(confusion.get(f"{name}>{chosen}", 0)) for chosen in sources]]
            for name in sources])
    return out


def _passes(analysis: Mapping[str, Any], specs: Mapping[str, Any], voice: Voice) -> list[str]:
    """Write the passes, repeatability, order effects and the pointwise arm.

    Args: analysis: the analysis block; specs: pass name -> PassSpec; voice: the scorer.
    Returns: Markdown lines.
    Raises: KeyError for a malformed analysis.
    Side effects: none.
    """
    versus = {PRIMARY_PASS: analysis["primary_vs_qwen3"]["reachable"],
              **analysis["passes_vs_qwen3"]}
    rows = []
    for name, spec in specs.items():
        h = analysis["passes"][name]
        same = analysis["agreement_with_primary"].get(name)
        rows.append([f"`{name}`", spec.arm, spec.order, ratio(h["reachable"]), ratio(h["all"]),
                     "-" if same is None else ratio(same["same_selection"]),
                     _difference(versus[name])])
    out = ["", "## Passes, order and chance", ""]
    out += _table(["Pass", "Arm", "Order", "Top-1 reachable", "Top-1 all",
                   "Same selection as primary", "Minus Qwen3 (reachable)"], rows)
    if "repeatability" in analysis:
        rep = analysis["repeatability"]
        out += ["", f"Identical request sent twice (`choice-a1` and `choice-a1-repeat`), {rep['slots']} "
                f"slots: the same candidate was selected for {ratio(rep['same_selection'])}; every "
                f"candidate's probability was identical for {ratio(rep['identical_scores'])}; the "
                f"largest difference in a candidate's probability was "
                f"{_largest_difference(rep['max_abs_difference'])} (mean "
                f"{_decimal(rep['mean_abs_difference'], 4)})."]
    change = analysis.get("order_and_chance")
    if change:
        u = change["unanimity"]
        per_pass = "; ".join(
            f"`{name}` changes {c['changed']} of {c['slots']} ({c['became_right']} become right, "
            f"{c['became_wrong']} wrong)" for name, c in change["correctness_changes"].items())
        out += ["", f"Accuracy totals hide offsetting changes ({voice.post_hoc}). Reachable slots "
                f"whose correctness differs from the primary pass: {per_pass}. All the Choice "
                f"passes select the same candidate on {u['same_selection']} of {u['slots']} slots "
                f"and are equally right or wrong on {u['same_correctness']}."]
    out += ["", f"Does {voice.name} favour the first option it is shown? Share of slots with at "
            "least two candidates in which the selected candidate was shown first, against the "
            "share if position played no part (the mean of 1/n):", ""]
    pos_rows = []
    for name, p in analysis["positions"].items():
        expected = p["expected_if_no_bias"]
        pos_rows.append([f"`{name}`", str(p["slots"]), str(p["selected_first"]),
                         f"{expected['numerator'] / expected['denominator']:.1f}"
                         if expected else "-"])
    out += _table(["Pass", "Slots", "Selected candidate shown first",
                   "Expected number if position played no part"], pos_rows)
    if "consensus" in analysis:
        c = analysis["consensus"]
        out += ["", f"Exploratory: averaging the probabilities of the Choice passes "
                f"({', '.join(f'`{p}`' for p in c['passes'])}) and selecting from the average gives "
                f"{ratio(c['headline']['reachable'])} on the reachable slots; minus Qwen3 "
                f"{_difference(c['vs_qwen3'])}."
                + (" An identical repeat counts as a pass here, so its order weighs twice."
                   if "repeatability" in analysis else "")]
    if "pointwise_vs_primary" in analysis:
        pv = analysis["pointwise_vs_primary"]
        out += ["", f"Exploratory: the pointwise arm (`{pv['pass']}`, one yes/no question per "
                f"candidate) against the primary Choice pass: {_difference(pv['vs_primary'])} "
                f"(pointwise minus primary); it disagrees with the primary pass on "
                f"{_slots(pv['vs_primary']['only_first'] + pv['vs_primary']['only_second'])}."]
    return out


def _abstention(analysis: Mapping[str, Any], voice: Voice) -> list[str]:
    """Write how the scorer used its own `none_of_these` option.

    Args: analysis: the analysis block; voice: the scorer.
    Returns: Markdown lines.
    Raises: KeyError for a malformed analysis.
    Side effects: none.
    """
    a = analysis["abstention"]
    refusals = a["forced_choice_correct_when_none_chosen"]["denominator"]
    name = voice.name
    return ["", f"## {name}'s own `none_of_these` (primary pass, reachable slots)", ""] + _table(
        ["Question", "Answer"], [
            ["Slots asked", str(a["asked"])],
            [f"`none_of_these` was {name}'s choice although the gold was among the candidates",
             ratio(a["none_when_gold_in_pool"])],
            [f"`none_of_these` was {name}'s choice when the gold was not among the candidates "
             "(a right abstention)", ratio(a["none_when_gold_absent"])],
            [f"Slots {name} answered with a candidate", ratio(a["answered"])],
            ["Precision of the forced choice on those", ratio(a["precision_when_answered"])],
            [f"Forced choice correct on the slots where {name} chose `none_of_these`",
             ratio(a["forced_choice_correct_when_none_chosen"])],
            [f"Refusals made at {voice.own_confidence_word} of 0.9 or more",
             f"{a['refusals_at_confidence_0.9']} of {refusals}"]]) + [
        "", "The headline always uses the forced choice, as the baseline always picks its top "
        f"candidate; this table shows what honouring {name}'s refusals would change. On the "
        f"{a['outside_population_asked']} asked slots whose gold lies in a vocabulary the sources "
        f"cannot reach, {name} chose `none_of_these` for {a['outside_population_none']}; they are "
        "left out above because a pool may hold the same concept under another identifier.", ""]


def _jev_confidence_notes(analysis: Mapping[str, Any]) -> list[str]:
    """Say what Jev's own confidence value is, and what it measures when Jev refuses.

    Args: analysis: the analysis block.
    Returns: the paragraphs that follow the signals table, as Markdown lines.
    Raises: KeyError for a malformed analysis.
    Side effects: none.
    """
    formula = analysis["confidence_formula"]
    a = analysis["abstention"]
    return ["", f"TypeSafe documents Jev's `confidence` as (p_max - 1/n) / (1 - 1/n) over its n options. "
            f"That holds within 0.01 for {ratio(formula['answers'])} of the Choice responses (largest "
            f"difference {formula['max_abs_difference']:.3f}; this check is post hoc, L-014), so "
            "where it holds `confidence` is a function of the top probability and the number of "
            "options and carries no other information.",
            "", "Jev's `confidence` describes its own choice. For "
            f"{a['forced_choice_correct_when_none_chosen']['denominator']} reachable slots that "
            f"choice was `none_of_these` ({a['refusals_at_confidence_0.9']} of them at 0.9 or "
            "more), so for those slots it measures confidence in the refusal, not in the forced "
            "candidate whose correctness is scored here."]


def _confidence(analysis: Mapping[str, Any], voice: Voice) -> list[str]:
    """Write what the confidence signals are worth.

    Args: analysis: the analysis block; voice: the scorer.
    Returns: Markdown lines.
    Raises: KeyError for a malformed analysis.
    Side effects: none.
    """
    block = analysis["confidence"]
    out = ["## What the confidence is worth (reachable slots)", "",
           "Any number that says how much to trust a selection can decide when to hand a slot to a "
           "stronger model. AUROC is the chance that a right answer is more confident than a wrong "
           "one (0.5 is no information). The calibration gap and Brier score treat the value as a "
           "probability of being right. All of this is exploratory and uses the data it describes; "
           "choose a threshold on held-out terms before relying on one.", ""]
    rows = []
    for name, label in signal_labels(voice).items():
        e = block[name]
        auc = e["auroc"]
        rows.append([label, str(e["answered"]),
                     "-" if auc is None else f"{auc['numerator'] / auc['denominator']:.3f}",
                     "-" if e.get("calibration_gap") is None else f"{e['calibration_gap']:.3f}",
                     "-" if e.get("brier") is None else f"{e['brier']:.3f}"])
    out += _table(["Signal", "Slots answered", "AUROC", "Calibration gap", "Brier"], rows)
    out += voice.confidence_notes(analysis)
    out += ["", "Coverage and precision when only selections at or above a threshold are answered "
            "(reachable slots; the rest would go to a stronger model):", ""]
    probability, qwen = block[f"{voice.key}_selected_probability"], block["qwen3_score"]
    own = block[f"{voice.key}_confidence"]
    rows = []
    for a, b, c in zip(probability["thresholds"], qwen["thresholds"], own["thresholds"],
                       strict=True):
        rows.append([value(a["threshold"]), ratio(a["coverage"]), ratio(a["precision"]),
                     ratio(c["coverage"]), ratio(c["precision"]), ratio(b["coverage"]),
                     ratio(b["precision"])])
    out += _table(["Threshold", f"{voice.name} probability: answered", "precision",
                   f"{voice.name} {voice.own_confidence_short}: answered", "precision",
                   "Qwen3 score: answered", "precision"], rows)
    out += ["", f"Handing the slots below a threshold on {voice.name}'s selected probability to a "
            f"stronger model (the split into fixable and unfixable errors is {voice.post_hoc}). Of "
            "the "
            "escalated slots, the right ones are work a stronger model repeats; the "
            "wrong ones with the gold among the candidates are errors it could fix; the wrong ones "
            "without are beyond any scorer. The ceiling assumes the stronger model fixes every "
            "fixable error, so it is an upper bound, not a prediction:", ""]
    rows = []
    for row in probability["escalation"]:
        rows.append([value(row["threshold"]), ratio(row["escalated"]), str(row["escalated_right"]),
                     str(row["escalated_wrong_fixable"]), str(row["escalated_wrong_unfixable"]),
                     ratio(row["kept_precision"]), ratio(row["ceiling_if_escalations_succeed"])])
    out += _table(["Threshold", "Escalated", "Right among them", "Wrong, fixable",
                   "Wrong, unfixable", "Precision of the rest", "Ceiling"], rows)
    out += ["", f"Reliability of {voice.name}'s selected probability (bins of width 0.1):", ""]
    out += _table(["Bin", "Slots", "Right", "Mean probability"], [
        [f"{value(b['lower'])}-{value(b['upper'])}", str(b["slots"]), str(b["right"]),
         f"{b['mean_confidence']:.3f}"] for b in probability["reliability"]])
    out += ["", voice.reliability_note]
    return out


def _tables(pops: Mapping[str, Any], base: Mapping[str, Any], summary: Mapping[str, Any],
            voice: Voice) -> list[str]:
    """Write the grouping and repetition tables.

    Args: pops: the primary pass's populations; base: the baseline's; summary: the run summary;
    voice: the scorer.
    Returns: Markdown lines.
    Raises: KeyError if a group lacks a metric.
    Side effects: none.
    """
    out = ["", "## By slot, role and term length", ""]
    rows = []
    for name in sorted(n for n in pops if n.startswith(("slot:", "role:", "words:"))):
        p = pops[name]
        rows.append([name, str(p["n"]), ratio(p["top1"]), ratio(base[name]["top1"]),
                     ratio(p["gold_in_pool"])])
    out += _table(["Group", "Slots", f"{voice.name} top-1", "Qwen3 top-1", "Gold in pool"], rows)
    out += ["", "## Most repeated terms", "",
            "A few terms are asked many times, so their outcome weighs on every slot-weighted "
            "figure above.", ""]
    qwen_terms = {t["term"]: t for t in summary["baseline"]["most_repeated_terms"]}
    out += _table(["Term", "Slots", f"{voice.name} wrong", "Qwen3 wrong"], [
        [f"`{t['term']}`", str(t["slots"]), str(t["wrong"]),
         str(qwen_terms[t["term"]]["wrong"]) if t["term"] in qwen_terms else "-"]
        for t in summary["primary"]["most_repeated_terms"]])
    return out


def render_report(manifest: Mapping[str, Any], summary: Mapping[str, Any],
                  outcomes: Sequence[Mapping[str, Any]], specs: Mapping[str, Any],
                  voice: Voice | None = None) -> str:
    """Write the Markdown report of a finished run.

    Args: manifest: run identity, configuration, health checks; summary: the run summary;
    outcomes: the primary pass's evaluate_slot outcomes; specs: pass name -> PassSpec; voice: the
    scorer (Jev's by default).
    Returns: Markdown text.
    Raises: KeyError if the summary lacks a population or block the report needs.
    Side effects: none.
    """
    voice = voice or JEV_VOICE
    pops, base, analysis = (summary["primary"]["populations"], summary["baseline"]["populations"],
                            summary["analysis"])
    out = _header(manifest, voice) + _result(pops, base, analysis, voice) + _paired(analysis, voice)
    out += voice.extra(summary)
    out += _misses(pops, base, voice) + _routing(summary, voice) + _passes(analysis, specs, voice)
    out += _abstention(analysis, voice) + _confidence(analysis, voice)
    out += _tables(pops, base, summary, voice)
    out += _examples(outcomes)
    out += ["## Run health", ""]
    out += _table(["Check", "Result", "Detail"], [
        [c["name"], _verdict(c["passed"]), c["detail"]] for c in manifest["checks"]])
    out += ["", "## Notes", ""]
    out += [f"- {note}" for note in manifest["notes"]]
    out += ["", "Contains data retrieved from Wikidata (CC0), QUDT (CC BY 4.0, attribution to "
            "QUDT.org) and the Ontology of units of Measure (CC BY 4.0) via the EBI Ontology "
            f"Lookup Service, and {voice.attribution}.", ""]
    return "\n".join(out)


def render_comparison(manifest: Mapping[str, Any], summary: Mapping[str, Any],
                      voice: Voice | None = None) -> str:
    """Write the head-to-head of the primary pass and the Qwen3 baseline.

    Args: manifest: run identity; summary: the run summary; voice: the scorer (Jev's by default).
    Returns: Markdown text.
    Raises: KeyError for a malformed summary.
    Side effects: none.
    """
    voice = voice or JEV_VOICE
    scorer = voice.name
    analysis = summary["analysis"]
    out = [f"# {scorer} against the Qwen3 baseline: {manifest['run_id']}", "",
           f"Primary pass `{PRIMARY_PASS}` of run `{manifest['run_id']}` against baseline "
           f"`{manifest['baseline']['folder']}`, on the same slots and the same candidates. "
           f"Differences are in top-1 correct slots ({scorer} minus Qwen3); intervals resample whole "
           "terms.", "", "## Paired counts by population", ""]
    rows = []
    for name, c in analysis["primary_vs_qwen3"].items():
        rows.append([name, str(c["n"]), str(c["both"]), str(c["only_first"]), str(c["only_second"]),
                     str(c["neither"]), _difference(c), _p(c["mcnemar_p"])])
    out += _table(["Population", "Slots", "Both right", f"Only {scorer}", "Only Qwen3",
                   "Both wrong", f"{scorer} minus Qwen3", "McNemar p"], rows)
    out += ["", f"## Counted once per term ({voice.term_heading}; reachable and by gold vocabulary)",
            ""]
    out += _table(["Population", "Terms", f"{scorer} better", "Qwen3 better", "Equal",
                   "Sign-test p", "Term-averaged difference"], [
        [name, str(c["terms"]["terms"]), str(c["terms"]["first_better"]),
         str(c["terms"]["second_better"]), str(c["terms"]["equal"]),
         _p(c["terms"]["sign_test_p"]), _signed(c["terms"]["mean_difference"])]
        for name, c in analysis["primary_vs_qwen3"].items() if c["terms"] is not None])
    out += ["", f"## How slots moved between categories (Qwen3 to {scorer})", ""]
    moves = analysis[f"transitions_qwen3_to_{voice.key}"]
    out += _table([f"Qwen3 > {scorer}", "Slots"], [
        [key.replace("_", " ").replace(">", " > "), str(count)] for key, count in moves.items()])
    out += ["", "## Every pass against Qwen3 (reachable slots)", ""]
    rows = [[f"`{name}`", ratio(h["reachable"]), ratio(h["reachable_with_qudt_equivalence"])]
            for name, h in analysis["passes"].items()]
    rows.append(["Qwen3 baseline", ratio(analysis["qwen3"]["reachable"]),
                 ratio(analysis["qwen3"]["reachable_with_qudt_equivalence"])])
    out += _table(["Pass", "Strict top-1", "With QUDT/Wikidata equivalence"], rows)
    for key, title in (("first_only_right", f"Slots only {scorer} got right"),
                       ("second_only_right", "Slots only Qwen3 got right")):
        block = analysis[f"disagreements_{voice.key}_vs_qwen3"][key]
        loser = "Qwen3" if key == "first_only_right" else scorer
        out += ["", f"## {title} ({block['slots']})", "",
                f"{block['loser_is_equivalent']} of these {block['slots']} slots would not be "
                f"misses for {loser} if a QUDT kind and its `wikidataMatch` Wikidata item counted "
                f"as one entity ({voice.post_hoc}).", ""]
        for e in block["examples"]:
            own_view, qwen_view = e["first"], e["second"]
            out.append(f"- `{e['term']}` ({e['slot_id'].split('#')[0].rsplit('/', 1)[-1]}): gold "
                       f"{e['gold'][0]}:{e['gold'][1]}; {scorer} {_picked(own_view)}; Qwen3 "
                       f"{_picked(qwen_view)}")
        if not block["examples"]:
            out.append("None.")
    out += voice.extra(summary)
    return "\n".join(out) + "\n"


def _decimal(receipt: Mapping[str, int], digits: int) -> str:
    """Show an exact fraction as a decimal.

    Args: receipt: {"numerator", "denominator"}; digits: decimals to show.
    Returns: the decimal text.
    Raises: None.
    Side effects: none.
    """
    return f"{receipt['numerator'] / receipt['denominator']:.{digits}f}"


def _largest_difference(receipt: Mapping[str, int]) -> str:
    """Show the largest probability difference between two identical requests so that it is seen.

    Args: receipt: {"numerator", "denominator"}.
    Returns: two decimals; four when the difference exists but is below 0.005, where two would
    print 0.00 beside a count of slots that were not identical; "under 0.0001" below that.
    Raises: None.
    Side effects: none.
    """
    value = receipt["numerator"] / receipt["denominator"]
    digits = 4 if 0 < value < 0.005 else 2
    text = f"{value:.{digits}f}"
    return text if value == 0 or float(text) > 0 else f"under {10 ** -digits:g}"


def _slots(n: int) -> str:
    """Count slots in words.

    Args: n: a number of slots.
    Returns: "1 slot" or "n slots".
    Raises: None.
    Side effects: none.
    """
    return f"{n} slot" if n == 1 else f"{n} slots"


def _picked(view: Mapping[str, Any] | None) -> str:
    """Describe a selection in one phrase.

    Args: view: a selected candidate view or None.
    Returns: "'label' (source:id, 0.93)" or "nothing".
    Raises: KeyError for a malformed view.
    Side effects: none.
    """
    if view is None:
        return "nothing"
    return f"{view['label']!r} ({view['source']}:{view['local_id']}, {view['score']:.2f})"


JEV_VOICE = Voice(
    key="jev", name="Jev", own_confidence_word="a `confidence`", own_confidence_short="`confidence`",
    own_confidence_label="Jev: its own `confidence` value", pointwise=True,
    cost_sentence=_jev_cost_sentence, intro=_jev_intro, confidence_notes=_jev_confidence_notes,
    reliability_note="Jev reports probabilities rounded to two decimals, so a threshold moves in "
                     "steps of 0.01 and many slots share a value near 1.",
    attribution="answers from TypeSafe's Jev model obtained through OpenRouter",
    post_hoc="post hoc, L-014",
    term_level_note="post hoc: added after the first results, because slots cluster by term",
    term_heading="post hoc")
