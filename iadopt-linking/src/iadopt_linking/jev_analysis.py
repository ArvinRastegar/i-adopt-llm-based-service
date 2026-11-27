"""What a Jev run shows: each pass judged by the baseline's own evaluator, then compared."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from iadopt_linking_eval import choose, evaluate_slot
from iadopt_linking_eval.confidence import (
    auroc,
    brier_score,
    calibration_gap,
    coverage_precision,
    escalation_table,
    reliability_bins,
)
from iadopt_linking_eval.metrics import MARGIN_BOUNDS, THRESHOLDS, _count, _exact
from iadopt_linking_eval.paired import (
    DEFAULT_RESAMPLES,
    cluster_bootstrap_difference,
    mcnemar_exact,
    paired_counts,
    term_level,
)

from .jev import (
    ARM_CHOICE,
    ARM_NOUL,
    NONE_KEY,
    PRIMARY_PASS,
    PassSpec,
    candidate_key,
    choice_scores,
    noul_scores,
    scored_record,
)
from .population import Slot

BOOTSTRAP_SEED = 20261015
EXAMPLES = 12
POPULATIONS: tuple[tuple[str, Callable[[Mapping[str, Any]], bool]], ...] = (
    ("reachable", lambda o: o["reachable"]), ("all", lambda o: True),
    ("gold:wikidata", lambda o: o["gold"][0] == "wikidata"),
    ("gold:qudt", lambda o: o["gold"][0] == "qudt"), ("gold:om", lambda o: o["gold"][0] == "om"))


def _exact_or_none(value: Fraction | None) -> dict[str, int] | None:
    """Record an exact rational as a receipt, or None for None."""
    return None if value is None else _exact(value)


def _dec(x: float) -> Fraction:
    """Read a probability that the service wrote as a decimal, exactly."""
    return Fraction(repr(x))


def exact_margin(scores: Sequence[float]) -> Fraction | None:
    """Give the gap between the two highest scores, exactly."""
    if len(scores) < 2:
        return None
    first, second = sorted((_dec(s) for s in scores), reverse=True)[:2]
    return first - second


def exact_margins(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Count the selections decided by a small margin, exactly."""
    margins = [m for m in (exact_margin(r["scores"]) for r in results if r["asked"])
               if m is not None]
    return {f"below_{label}": _count(sum(m < bound for m in margins), len(margins))
            for label, bound in MARGIN_BOUNDS}


def pass_results(spec: PassSpec, slots: Sequence[Slot], records: Mapping[str, Mapping[str, Any]],
                 tasks: Mapping[tuple[str, str], Any], parsed: Mapping[tuple[str, str], Any]
                 ) -> list[dict[str, Any]]:
    """Judge one pass: turn every answer into scores and evaluate them like the baseline's."""
    results = []
    for slot in slots:
        base = records[slot.slot_id]
        key = (slot.slot_id, spec.name)
        if key not in tasks and base["pool"]:
            raise ValueError(f"{slot.slot_id} has candidates but pass {spec.name} sent no request")
        result: dict[str, Any] = {
            "slot_id": slot.slot_id, "asked": key in tasks, "scores": [], "margin": None,
            "p_none": None, "own_index": None, "own_chose_none": None, "own_choice_key": None,
            "own_confidence": None, "order": [], "selected_index": None}
        if key in tasks:
            order, answer = list(tasks[key].order), parsed[key]
            result["order"] = order
            if spec.arm == ARM_CHOICE:
                scores, p_none, own_index = choice_scores(answer, order, len(base["pool"]))
                result.update(scores=scores, p_none=p_none, own_index=own_index,
                              own_chose_none=own_index is None,
                              own_choice_key=(NONE_KEY if own_index is None
                                              else candidate_key(base["pool"][own_index])),
                              own_confidence=answer["confidence"])
            else:
                result["scores"] = noul_scores(answer, order, len(base["pool"]))
            margin = exact_margin(result["scores"])
            result["margin"] = None if margin is None else float(margin)
        scored = scored_record(base, result["scores"])
        result["outcome"] = evaluate_slot(scored)
        result["selected_index"] = choose(scored["pool"])
        results.append(result)
    return results


def headline(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count how often a pass was right."""
    outcomes = [r["outcome"] for r in results]
    reachable = [o for o in outcomes if o["reachable"]]
    return {"all": _count(sum(o["correct"] for o in outcomes), len(outcomes)),
            "reachable": _count(sum(o["correct"] for o in reachable), len(reachable)),
            "reachable_with_qudt_equivalence": _count(
                sum(o["equivalent_correct"] for o in reachable), len(reachable)),
            "no_selection": sum(o["selected"] is None for o in outcomes)}


def compare(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]],
            population: Callable[[Mapping[str, Any]], bool], field: str = "correct", *,
            resamples: int = DEFAULT_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Compare two systems on the slots of one population."""
    if [o["slot_id"] for o in first] != [o["slot_id"] for o in second]:
        raise ValueError("The two systems did not judge the same slots in the same order")
    chosen = [i for i, o in enumerate(first) if population(o)]
    a = [bool(first[i][field]) for i in chosen]
    b = [bool(second[i][field]) for i in chosen]
    counts = paired_counts(a, b)
    if not chosen:
        return {**counts, "mcnemar_p": None, "difference": None, "terms": None}
    clusters = [first[i]["query_key"] for i in chosen]
    return {**counts, "mcnemar_p": _exact(mcnemar_exact(counts["only_first"], counts["only_second"])),
            "difference": cluster_bootstrap_difference(a, b, clusters, resamples=resamples,
                                                       seed=seed),
            "terms": term_level(a, b, clusters)}


def repeatability(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
                  ) -> dict[str, Any]:
    """Measure how much two passes that sent the same request differ."""
    if [r["slot_id"] for r in first] != [r["slot_id"] for r in second]:
        raise ValueError("The passes cover different slots")
    both = [(a, b) for a, b in zip(first, second, strict=True) if a["asked"] and b["asked"]]
    deltas = [abs(_dec(x) - _dec(y)) for a, b in both
              for x, y in zip(a["scores"], b["scores"], strict=True)]
    mean = sum(deltas, Fraction(0)) / len(deltas) if deltas else Fraction(0)
    return {"slots": len(both),
            "same_selection": _count(sum(a["selected_index"] == b["selected_index"]
                                         for a, b in both), len(both)),
            "identical_scores": _count(sum(a["scores"] == b["scores"] for a, b in both), len(both)),
            "max_abs_difference": _exact(max(deltas, default=Fraction(0))),
            "mean_abs_difference": _exact(mean)}


def consensus(passes: Sequence[Sequence[Mapping[str, Any]]], slots: Sequence[Slot],
              records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Average several passes' scores per candidate and judge the result."""
    if any([r["slot_id"] for r in one] != [s.slot_id for s in slots] for one in passes):
        raise ValueError("The passes cover different slots")
    results = []
    for index, slot in enumerate(slots):
        rows = [one[index] for one in passes]
        base = records[slot.slot_id]
        asked = all(r["asked"] for r in rows)
        scores = ([math.fsum(r["scores"][i] for r in rows) / len(rows)
                   for i in range(len(base["pool"]))] if asked else [])
        scored = scored_record(base, scores)
        results.append({"slot_id": slot.slot_id, "asked": asked, "scores": scores,
                        "outcome": evaluate_slot(scored), "selected_index": choose(scored["pool"])})
    return results


def agreement_with(primary: Sequence[Mapping[str, Any]],
                   others: Sequence[Sequence[Mapping[str, Any]]]) -> list[Fraction | None]:
    """Say, per slot, which share of passes selected what the primary pass selected."""
    shares: list[Fraction | None] = []
    for index, result in enumerate(primary):
        if result["selected_index"] is None:
            shares.append(None)
            continue
        same = 1 + sum(one[index]["selected_index"] == result["selected_index"] for one in others)
        shares.append(Fraction(same, 1 + len(others)))
    return shares


def confidence_formula(answers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check Jev's documented ``confidence`` against the probabilities it came with."""
    gaps = []
    for answer in answers:
        n = len(answer["probabilities"])
        expected = (max(answer["probabilities"].values()) - 1 / n) / (1 - 1 / n)
        gaps.append(abs(answer["confidence"] - expected))
    return {"answers": _count(sum(g <= 0.01 for g in gaps), len(gaps)),
            "max_abs_difference": max(gaps, default=0.0)}


def abstention(primary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe how the scorer used its own "none of these" option."""
    asked = [r for r in primary if r["asked"] and r["outcome"]["reachable"]]
    present = [r for r in asked if r["outcome"]["gold_in_pool"]]
    absent = [r for r in asked if not r["outcome"]["gold_in_pool"]]
    answered = [r for r in asked if not r["own_chose_none"]]
    refused = [r for r in asked if r["own_chose_none"]]
    outside = [r for r in primary if r["asked"] and not r["outcome"]["reachable"]]
    return {
        "asked": len(asked),
        "none_when_gold_in_pool": _count(sum(r["own_chose_none"] for r in present), len(present)),
        "none_when_gold_absent": _count(sum(r["own_chose_none"] for r in absent), len(absent)),
        "answered": _count(len(answered), len(asked)),
        "precision_when_answered": _count(sum(r["outcome"]["correct"] for r in answered),
                                          len(answered)),
        "forced_choice_correct_when_none_chosen": _count(
            sum(r["outcome"]["correct"] for r in refused), len(refused)),
        "refusals_at_confidence_0.9": sum(r["own_confidence"] >= 0.9 for r in refused),
        "outside_population_asked": len(outside),
        "outside_population_none": sum(bool(r["own_chose_none"]) for r in outside)}


def positions(results: Sequence[Mapping[str, Any]], key: str = "jev") -> dict[str, Any]:
    """Measure whether a pass favours the first option it is shown."""
    several = [r for r in results if r["asked"] and len(r["scores"]) >= 2]
    expected = sum(Fraction(1, len(r["scores"])) for r in several)
    first = sum(r["order"].index(r["selected_index"]) == 0 for r in several)
    own = [r for r in several if r["own_index"] is not None]
    own_first = sum(r["order"].index(r["own_index"]) == 0 for r in own)
    own_expected = sum(Fraction(1, len(r["scores"])) for r in own)
    return {"slots": len(several), "selected_first": first, "expected_if_no_bias": _exact(expected),
            f"{key}_choice_slots": len(own), f"{key}_choice_first": own_first,
            f"{key}_choice_expected_if_no_bias": _exact(own_expected)}


def correctness_changes(primary: Sequence[Mapping[str, Any]],
                        others: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Count the slots whose correctness differs between the primary pass and another."""
    chosen = [i for i, r in enumerate(primary) if r["asked"] and r["outcome"]["reachable"]]
    changes = {}
    for name, rows in others.items():
        right = sum(rows[i]["outcome"]["correct"] and not primary[i]["outcome"]["correct"]
                    for i in chosen)
        wrong = sum(primary[i]["outcome"]["correct"] and not rows[i]["outcome"]["correct"]
                    for i in chosen)
        changes[name] = {"slots": len(chosen), "changed": right + wrong, "became_right": right,
                         "became_wrong": wrong}
    return changes


def unanimity(passes: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    """Count the slots on which every Choice pass agrees."""
    chosen = [i for i, r in enumerate(passes[0]) if r["asked"] and r["outcome"]["reachable"]]
    return {"slots": len(chosen),
            "same_selection": sum(len({one[i]["selected_index"] for one in passes}) == 1
                                  for i in chosen),
            "same_correctness": sum(len({bool(one[i]["outcome"]["correct"]) for one in passes}) == 1
                                    for i in chosen)}


def transitions(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
                ) -> dict[str, int]:
    """Count how slots moved between failure categories from one system to the other."""
    counted = Counter(f"{a['category']}>{b['category']}" for a, b in zip(first, second, strict=True))
    return dict(sorted(counted.items()))


def _view(outcome: Mapping[str, Any]) -> dict[str, Any] | None:
    """Show what a system selected for one slot."""
    chosen = outcome["selected"]
    return None if chosen is None else {key: chosen[key] for key in
                                        ("source", "local_id", "label", "score")}


def disagreements(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
                  ) -> dict[str, Any]:
    """List slots where exactly one of two systems was right."""
    def pick(want_first: bool) -> dict[str, Any]:
        """List the slots one system got right and the other wrong."""
        rows = sorted(((a, b) for a, b in zip(first, second, strict=True)
                       if a["reachable"] and bool(a["correct"]) == want_first
                       and bool(b["correct"]) != want_first), key=lambda pair: pair[0]["slot_id"])
        loser_equivalent = sum(bool((b if want_first else a)["equivalent_correct"])
                               for a, b in rows)
        return {"slots": len(rows), "loser_is_equivalent": loser_equivalent, "examples": [
            {"slot_id": a["slot_id"], "term": a["term"], "gold": a["gold"],
             "first": _view(a), "second": _view(b)} for a, b in rows[:EXAMPLES]]}
    return {"first_only_right": pick(True), "second_only_right": pick(False)}


def confidence_signals(primary: Sequence[Mapping[str, Any]],
                       choice_passes: Sequence[Sequence[Mapping[str, Any]]],
                       qwen: Sequence[Mapping[str, Any]], key: str = "jev"
                       ) -> dict[str, list[float | None]]:
    """Collect the numbers that say how much to trust a selection, per slot."""
    def selected_score(outcome: Mapping[str, Any]) -> float | None:
        """Give the score of the selected candidate."""
        return None if outcome["selected"] is None else outcome["selected"]["score"]

    return {
        f"{key}_selected_probability": [selected_score(r["outcome"]) for r in primary],
        f"{key}_confidence": [r["own_confidence"] if r["asked"] else None for r in primary],
        f"{key}_margin": [r["margin"] if r["asked"] else None for r in primary],
        f"{key}_candidate_mass": [None if r["p_none"] is None else float(1 - _dec(r["p_none"]))
                                  for r in primary],
        f"{key}_pass_agreement": [None if share is None else float(share)
                                  for share in agreement_with(primary, choice_passes)],
        "qwen3_score": [selected_score(o) for o in qwen],
        "qwen3_margin": [o["margin"] for o in qwen]}


def confidence_block(signals: Mapping[str, Sequence[float | None]],
                     outcomes: Mapping[str, Sequence[Mapping[str, Any]]],
                     population: Callable[[Mapping[str, Any]], bool],
                     key: str = "jev") -> dict[str, Any]:
    """Judge each confidence signal on one population."""
    block: dict[str, Any] = {}
    for name, values in signals.items():
        system = "qwen3" if name.startswith("qwen3") else key
        judged = outcomes[system]
        chosen = [i for i, o in enumerate(judged) if population(o)]
        confidence = [values[i] for i in chosen]
        correct = [bool(judged[i]["correct"]) for i in chosen]
        entry: dict[str, Any] = {
            "slots": len(chosen), "answered": sum(c is not None for c in confidence),
            "auroc": _exact_or_none(auroc(confidence, correct)),
            "thresholds": coverage_precision(confidence, correct, THRESHOLDS),
            "escalation": escalation_table(confidence, correct, THRESHOLDS,
                                           [bool(judged[i]["gold_in_pool"]) for i in chosen])}
        if name in (f"{key}_selected_probability", f"{key}_confidence", "qwen3_score"):
            bins = reliability_bins(confidence, correct)
            entry.update(reliability=bins, calibration_gap=calibration_gap(bins),
                         brier=brier_score(confidence, correct))
        block[name] = entry
    return block


def analyse(slots: Sequence[Slot], records: Mapping[str, Mapping[str, Any]],
            specs: Sequence[PassSpec], tasks: Mapping[tuple[str, str], Any],
            parsed: Mapping[tuple[str, str], Any], *, resamples: int = DEFAULT_RESAMPLES,
            seed: int = BOOTSTRAP_SEED, key: str = "jev", formula_check: bool = True
            ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Judge every pass and compare the primary pass with the Qwen3 baseline."""
    names = [spec.name for spec in specs]
    if PRIMARY_PASS not in names:
        raise ValueError(f"The primary pass {PRIMARY_PASS!r} was not run")
    results = {spec.name: pass_results(spec, slots, records, tasks, parsed) for spec in specs}
    qwen = [evaluate_slot(records[slot.slot_id]) for slot in slots]
    primary = results[PRIMARY_PASS]
    own = [r["outcome"] for r in primary]
    choice_names = [spec.name for spec in specs if spec.arm == ARM_CHOICE]
    others = [results[n] for n in choice_names if n != PRIMARY_PASS]
    analysis: dict[str, Any] = {
        "passes": {name: headline(rows) for name, rows in results.items()},
        "qwen3": headline([{"outcome": o} for o in qwen]),
        "primary_vs_qwen3": {
            name: compare(own, qwen, population, resamples=resamples, seed=seed)
            for name, population in POPULATIONS},
        "primary_vs_qwen3_equivalence": compare(
            own, qwen, POPULATIONS[0][1], "equivalent_correct", resamples=resamples, seed=seed),
        "passes_vs_qwen3": {
            name: compare([r["outcome"] for r in rows], qwen, POPULATIONS[0][1],
                          resamples=resamples, seed=seed)
            for name, rows in results.items() if name != PRIMARY_PASS},
        f"transitions_qwen3_to_{key}": transitions(qwen, own),
        f"disagreements_{key}_vs_qwen3": disagreements(own, qwen),
        "abstention": abstention(primary),
        "positions": {name: positions(results[name], key) for name in choice_names},
        "agreement_with_primary": {
            name: {"same_selection": _count(
                sum(a["selected_index"] == b["selected_index"] for a, b in zip(
                    primary, results[name], strict=True) if a["asked"] and b["asked"]),
                sum(a["asked"] and b["asked"] for a, b in zip(primary, results[name], strict=True)))}
            for name in results if name != PRIMARY_PASS}}
    if formula_check:
        analysis["confidence_formula"] = confidence_formula([
            parsed[slot_pass] for slot_pass, task in tasks.items() if task.spec.arm == ARM_CHOICE])
    if "choice-a1-repeat" in results:
        analysis["repeatability"] = repeatability(primary, results["choice-a1-repeat"])
    if len(choice_names) > 1:
        mean = consensus([results[n] for n in choice_names], slots, records)
        analysis["consensus"] = {
            "passes": choice_names, "headline": headline(mean),
            "vs_qwen3": compare([r["outcome"] for r in mean], qwen, POPULATIONS[0][1],
                                resamples=resamples, seed=seed)}
    pointwise = next((spec.name for spec in specs if spec.arm == ARM_NOUL), None)
    if pointwise is not None:
        analysis["pointwise_vs_primary"] = {
            "pass": pointwise, "vs_primary": compare(
                [r["outcome"] for r in results[pointwise]], own, POPULATIONS[0][1],
                resamples=resamples, seed=seed)}
    if others:
        analysis["order_and_chance"] = {
            "correctness_changes": correctness_changes(
                primary, {n: results[n] for n in choice_names if n != PRIMARY_PASS}),
            "unanimity": unanimity([results[n] for n in choice_names])}
    signals = confidence_signals(primary, others, qwen, key)
    analysis["confidence"] = confidence_block(signals, {key: own, "qwen3": qwen}, POPULATIONS[0][1],
                                              key)
    return results, analysis
