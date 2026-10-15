"""What a Jev run shows: each pass judged by the baseline's own evaluator, then compared.

Pure functions over stored evidence. Every pass is turned into the same kind of scores record the
Qwen3 baseline produced and judged by ``evaluate_slot``, so "correct", the failure categories and
the diagnostics mean exactly what they meant there. Only counts, exact fractions and clearly marked
simulations (the bootstrap) come out; nothing is tuned on the data.

Terminology: a *result* is one pass's answer for one slot, a *selection* is the candidate with the
highest score (ties by source, then rank, as the baseline decided), and a *confidence signal* is
any number that says how much to trust a selection.
"""

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
    """Record an exact rational as a receipt, or None for None.

    Args: value: a Fraction or None.
    Returns: {"numerator", "denominator"} in lowest terms, or None.
    Raises: None.
    Side effects: none.
    """
    return None if value is None else _exact(value)


def _dec(x: float) -> Fraction:
    """Read a probability that the service wrote as a decimal, exactly.

    Args: x: a float parsed from a decimal such as 0.34.
    Returns: the decimal's exact value; the shortest repr round-trips, so this is 34/100 and not
    the binary double nearest to it. Differences of two such floats are not exact (0.34 - 0.33
    is 0.010000000000000009), which is why margins are computed from these.
    Raises: None.
    Side effects: none.
    """
    return Fraction(repr(x))


def exact_margin(scores: Sequence[float]) -> Fraction | None:
    """Give the gap between the two highest scores, exactly.

    Args: scores: one score per candidate, written by the service as decimals.
    Returns: the highest minus the second highest, or None with fewer than two candidates.
    Raises: None.
    Side effects: none.
    """
    if len(scores) < 2:
        return None
    first, second = sorted((_dec(s) for s in scores), reverse=True)[:2]
    return first - second


def exact_margins(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Count the selections decided by a small margin, exactly.

    Args: results: pass_results of one pass.
    Returns: the same structure as the evaluator's ``margins`` block, from exact margins.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
    margins = [m for m in (exact_margin(r["scores"]) for r in results if r["asked"])
               if m is not None]
    return {f"below_{label}": _count(sum(m < bound for m in margins), len(margins))
            for label, bound in MARGIN_BOUNDS}


def pass_results(spec: PassSpec, slots: Sequence[Slot], records: Mapping[str, Mapping[str, Any]],
                 tasks: Mapping[tuple[str, str], Any], parsed: Mapping[tuple[str, str], Any]
                 ) -> list[dict[str, Any]]:
    """Judge one pass: turn every answer into scores and evaluate them like the baseline's.

    Args: spec: the pass; slots: the population in order; records: the baseline's scores records;
    tasks: the requests by (slot id, pass); parsed: parse_choice or parse_noul output by the same
    key, for every slot that was asked.
    Returns: one result per slot: slot id, whether it was asked, the score of each pool
    candidate, the exact gap between the two highest scores (``margin``, a float of an exact
    value), the probability of "none" and Jev's own choice (Choice only), Jev's confidence, the
    presentation order, the pool index selected, and the evaluate_slot outcome. A slot with no
    candidate was not asked and has no scores.
    Raises: KeyError if an asked slot has no parsed answer; ValueError if a slot has candidates
    but no request, which would otherwise score as a row of ties.
    Side effects: none.
    """
    results = []
    for slot in slots:
        base = records[slot.slot_id]
        key = (slot.slot_id, spec.name)
        if key not in tasks and base["pool"]:
            raise ValueError(f"{slot.slot_id} has candidates but pass {spec.name} sent no request")
        result: dict[str, Any] = {
            "slot_id": slot.slot_id, "asked": key in tasks, "scores": [], "margin": None,
            "p_none": None, "jev_index": None, "jev_chose_none": None, "jev_choice_key": None,
            "jev_confidence": None, "order": [], "selected_index": None}
        if key in tasks:
            order, answer = list(tasks[key].order), parsed[key]
            result["order"] = order
            if spec.arm == ARM_CHOICE:
                scores, p_none, jev_index = choice_scores(answer, order, len(base["pool"]))
                result.update(scores=scores, p_none=p_none, jev_index=jev_index,
                              jev_chose_none=jev_index is None,
                              jev_choice_key=(NONE_KEY if jev_index is None
                                              else candidate_key(base["pool"][jev_index])),
                              jev_confidence=answer["confidence"])
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
    """Count how often a pass was right.

    Args: results: pass_results output.
    Returns: receipts over all slots and over the reachable ones for strict identity and for the
    QUDT/Wikidata equivalence, and the number of slots with no selection.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
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
    """Compare two systems on the slots of one population.

    Args: first, second: evaluate_slot outcomes for the same slots in the same order; population:
    which outcomes to include; field: the boolean outcome compared; resamples and seed: the
    bootstrap's.
    Returns: the paired counts, the exact McNemar p-value, the accuracy difference (first minus
    second) with a percentile interval from resampling whole terms, and the same comparison made
    once per term (a sign test over terms and the unweighted mean difference).
    Raises: ValueError if the systems judged different slots. A population with no slot gives
    zero counts and no p-value or interval.
    Side effects: none.
    """
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
    """Measure how much two passes that sent the same request differ.

    Args: first, second: pass_results of two passes over the same slots.
    Returns: over the slots both asked: how many selected the same candidate, how many gave
    identical scores for every candidate, and the largest and mean absolute difference in a
    candidate's score, as receipts (the scores are decimals, so the differences are exact).
    Raises: ValueError if the passes cover different slots.
    Side effects: none.
    """
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
    """Average several passes' scores per candidate and judge the result.

    Args: passes: pass_results of passes over the same slots (the Choice passes); slots: the
    population; records: the baseline's scores records.
    Returns: one result per slot like pass_results, with the mean scores. Every pass counts once,
    so a pass that repeats another request counts twice for that order.
    Raises: ValueError if the passes cover different slots, or a slot with candidates was not
    asked in every pass.
    Side effects: none.
    """
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
    """Say, per slot, which share of passes selected what the primary pass selected.

    Args: primary: the primary pass's results; others: the other Choice passes' results.
    Returns: per slot the exact share of ALL passes (the primary included) whose selection equals
    the primary's, or None where the primary selected nothing. A pass that repeats another
    request votes like any other.
    Raises: None.
    Side effects: none.
    """
    shares: list[Fraction | None] = []
    for index, result in enumerate(primary):
        if result["selected_index"] is None:
            shares.append(None)
            continue
        same = 1 + sum(one[index]["selected_index"] == result["selected_index"] for one in others)
        shares.append(Fraction(same, 1 + len(others)))
    return shares


def confidence_formula(answers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check Jev's documented ``confidence`` against the probabilities it came with.

    TypeSafe documents the Choice confidence as (p_max - 1/n) / (1 - 1/n) over the n options, "none"
    included. If that holds, the value adds nothing to the top probability and the option count.

    Args: answers: parse_choice output for every Choice response.
    Returns: how many answers were checked, how many agree with the formula within 0.01 (both
    numbers are rounded to two decimals) and the largest absolute difference.
    Raises: KeyError for a malformed answer.
    Side effects: none.
    """
    gaps = []
    for answer in answers:
        n = len(answer["probabilities"])
        expected = (max(answer["probabilities"].values()) - 1 / n) / (1 - 1 / n)
        gaps.append(abs(answer["confidence"] - expected))
    return {"answers": _count(sum(g <= 0.01 for g in gaps), len(gaps)),
            "max_abs_difference": max(gaps, default=0.0)}


def abstention(primary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe how Jev used its own "none of these" option.

    Args: primary: the primary Choice pass's results.
    Returns: over the asked slots of the primary (reachable) population: how often Jev chose
    "none" where the gold was among the candidates (a wrong abstention) and where it was not
    (a right one), how many refusals were made at a confidence of 0.9 or more, and what accuracy
    would be if refusals were left unanswered: coverage and precision among answered slots. The
    asked slots whose gold lies in a vocabulary the sources cannot reach are counted apart: their
    pool may hold the same concept under another identifier, so a refusal there is not
    necessarily right.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
    asked = [r for r in primary if r["asked"] and r["outcome"]["reachable"]]
    present = [r for r in asked if r["outcome"]["gold_in_pool"]]
    absent = [r for r in asked if not r["outcome"]["gold_in_pool"]]
    answered = [r for r in asked if not r["jev_chose_none"]]
    refused = [r for r in asked if r["jev_chose_none"]]
    outside = [r for r in primary if r["asked"] and not r["outcome"]["reachable"]]
    return {
        "asked": len(asked),
        "none_when_gold_in_pool": _count(sum(r["jev_chose_none"] for r in present), len(present)),
        "none_when_gold_absent": _count(sum(r["jev_chose_none"] for r in absent), len(absent)),
        "answered": _count(len(answered), len(asked)),
        "precision_when_answered": _count(sum(r["outcome"]["correct"] for r in answered),
                                          len(answered)),
        "forced_choice_correct_when_none_chosen": _count(
            sum(r["outcome"]["correct"] for r in refused), len(refused)),
        "refusals_at_confidence_0.9": sum(r["jev_confidence"] >= 0.9 for r in refused),
        "outside_population_asked": len(outside),
        "outside_population_none": sum(bool(r["jev_chose_none"]) for r in outside)}


def positions(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure whether a pass favours the first option it is shown.

    Args: results: pass_results of a Choice pass.
    Returns: over the slots with at least two candidates: how many selections sit in the first
    position, how many Jev choices (its own, "none" excluded) do, and how many each would
    be if position played no part (the mean of 1/n), as exact fractions of the slot count.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
    several = [r for r in results if r["asked"] and len(r["scores"]) >= 2]
    expected = sum(Fraction(1, len(r["scores"])) for r in several)
    first = sum(r["order"].index(r["selected_index"]) == 0 for r in several)
    own = [r for r in several if r["jev_index"] is not None]
    own_first = sum(r["order"].index(r["jev_index"]) == 0 for r in own)
    own_expected = sum(Fraction(1, len(r["scores"])) for r in own)
    return {"slots": len(several), "selected_first": first, "expected_if_no_bias": _exact(expected),
            "jev_choice_slots": len(own), "jev_choice_first": own_first,
            "jev_choice_expected_if_no_bias": _exact(own_expected)}


def correctness_changes(primary: Sequence[Mapping[str, Any]],
                        others: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Count the slots whose correctness differs between the primary pass and another.

    Accuracy totals can hide offsetting changes: a pass that gets as many slots right may get
    different ones right.

    Args: primary: the primary pass's results; others: pass name -> results of another Choice
    pass over the same slots.
    Returns: per pass, over the asked reachable slots: how many changed correctness, and in which
    direction.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
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
    """Count the slots on which every Choice pass agrees.

    Args: passes: pass_results of the Choice passes over the same slots.
    Returns: over the asked reachable slots: how many have every pass selecting the same
    candidate, and how many have every pass equally right or wrong.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
    chosen = [i for i, r in enumerate(passes[0]) if r["asked"] and r["outcome"]["reachable"]]
    return {"slots": len(chosen),
            "same_selection": sum(len({one[i]["selected_index"] for one in passes}) == 1
                                  for i in chosen),
            "same_correctness": sum(len({bool(one[i]["outcome"]["correct"]) for one in passes}) == 1
                                    for i in chosen)}


def transitions(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
                ) -> dict[str, int]:
    """Count how slots moved between failure categories from one system to the other.

    Args: first, second: evaluate_slot outcomes for the same slots.
    Returns: "<first category>><second category>" -> slots.
    Raises: KeyError for a malformed outcome.
    Side effects: none.
    """
    counted = Counter(f"{a['category']}>{b['category']}" for a, b in zip(first, second, strict=True))
    return dict(sorted(counted.items()))


def _view(outcome: Mapping[str, Any]) -> dict[str, Any] | None:
    """Show what a system selected for one slot.

    Args: outcome: evaluate_slot result.
    Returns: source, identifier, label and score of the selection, or None.
    Raises: None.
    Side effects: none.
    """
    chosen = outcome["selected"]
    return None if chosen is None else {key: chosen[key] for key in
                                        ("source", "local_id", "label", "score")}


def disagreements(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
                  ) -> dict[str, Any]:
    """List slots where exactly one of two systems was right.

    Args: first, second: evaluate_slot outcomes for the same slots (reachable slots only are
    listed).
    Returns: for each direction the number of slots, how many of them the losing system would
    have won had a QUDT kind and its Wikidata twin counted as one entity, and the first EXAMPLES
    by slot id, with the term, gold and what each system selected.
    Raises: KeyError for a malformed outcome.
    Side effects: none.
    """
    def pick(want_first: bool) -> dict[str, Any]:
        """List the slots one system got right and the other wrong.

        Args: want_first: True for the first system right and the second wrong, False for the
        opposite.
        Returns: the count, how many of the losing system's slots are QUDT/Wikidata twins, and the
        first examples.
        Raises: KeyError for a malformed outcome.
        Side effects: none.
        """
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
                       qwen: Sequence[Mapping[str, Any]]) -> dict[str, list[float | None]]:
    """Collect the numbers that say how much to trust a selection, per slot.

    Args: primary: the primary Choice pass's results; choice_passes: the other Choice passes'
    results; qwen: the baseline's evaluate_slot outcomes.
    Returns: signal name -> one value or None per slot. Signals: the probability of the selected
    candidate, Jev's own confidence, the top-two margin and the probability mass on real
    candidates (1 minus "none"), both computed exactly from the decimals the service wrote, the
    share of the Choice passes that agree with the primary selection (an identical repeat counts
    as a pass), and the Qwen3 score and margin for reference.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
    def selected_score(outcome: Mapping[str, Any]) -> float | None:
        """Give the score of the selected candidate.

        Args: outcome: evaluate_slot result.
        Returns: the score, or None when nothing was selected.
        Raises: None.
        Side effects: none.
        """
        return None if outcome["selected"] is None else outcome["selected"]["score"]

    return {
        "jev_selected_probability": [selected_score(r["outcome"]) for r in primary],
        "jev_confidence": [r["jev_confidence"] if r["asked"] else None for r in primary],
        "jev_margin": [r["margin"] if r["asked"] else None for r in primary],
        "jev_candidate_mass": [None if r["p_none"] is None else float(1 - _dec(r["p_none"]))
                               for r in primary],
        "jev_pass_agreement": [None if share is None else float(share)
                               for share in agreement_with(primary, choice_passes)],
        "qwen3_score": [selected_score(o) for o in qwen],
        "qwen3_margin": [o["margin"] for o in qwen]}


def confidence_block(signals: Mapping[str, Sequence[float | None]],
                     outcomes: Mapping[str, Sequence[Mapping[str, Any]]],
                     population: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any]:
    """Judge each confidence signal on one population.

    Args: signals: confidence_signals output; outcomes: name of the system that made each signal
    ("jev" or "qwen3") -> its evaluate_slot outcomes; population: which slots to include.
    Returns: per signal the AUROC for telling right from wrong, coverage and precision at each
    threshold, the escalation table, and, for the probability-like signals, reliability bins,
    the calibration gap and the Brier score.
    Raises: KeyError for a signal with no matching outcomes.
    Side effects: none.
    """
    block: dict[str, Any] = {}
    for name, values in signals.items():
        system = "qwen3" if name.startswith("qwen3") else "jev"
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
        if name in ("jev_selected_probability", "jev_confidence", "qwen3_score"):
            bins = reliability_bins(confidence, correct)
            entry.update(reliability=bins, calibration_gap=calibration_gap(bins),
                         brier=brier_score(confidence, correct))
        block[name] = entry
    return block


def analyse(slots: Sequence[Slot], records: Mapping[str, Mapping[str, Any]],
            specs: Sequence[PassSpec], tasks: Mapping[tuple[str, str], Any],
            parsed: Mapping[tuple[str, str], Any], *, resamples: int = DEFAULT_RESAMPLES,
            seed: int = BOOTSTRAP_SEED) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Judge every pass and compare the primary pass with the Qwen3 baseline.

    Args: slots, records: the population and the baseline's scores; specs: the passes run (the
    primary pass is required); tasks and parsed: requests and parsed answers by (slot id, pass);
    resamples and seed: the bootstrap's.
    Returns: results by pass name, and the analysis: per-pass headlines, the paired comparison with
    Qwen3 on each population, other passes against Qwen3, the pointwise arm against the primary,
    repeatability, agreement, consensus, abstention, position bias, category transitions,
    disagreement examples and the confidence signals on the reachable slots.
    Raises: ValueError if the primary pass was not run; KeyError for malformed input.
    Side effects: none.
    """
    names = [spec.name for spec in specs]
    if PRIMARY_PASS not in names:
        raise ValueError(f"The primary pass {PRIMARY_PASS!r} was not run")
    results = {spec.name: pass_results(spec, slots, records, tasks, parsed) for spec in specs}
    qwen = [evaluate_slot(records[slot.slot_id]) for slot in slots]
    primary = results[PRIMARY_PASS]
    jev = [r["outcome"] for r in primary]
    choice_names = [spec.name for spec in specs if spec.arm == ARM_CHOICE]
    others = [results[n] for n in choice_names if n != PRIMARY_PASS]
    analysis: dict[str, Any] = {
        "passes": {name: headline(rows) for name, rows in results.items()},
        "qwen3": headline([{"outcome": o} for o in qwen]),
        "primary_vs_qwen3": {
            name: compare(jev, qwen, population, resamples=resamples, seed=seed)
            for name, population in POPULATIONS},
        "primary_vs_qwen3_equivalence": compare(
            jev, qwen, POPULATIONS[0][1], "equivalent_correct", resamples=resamples, seed=seed),
        "passes_vs_qwen3": {
            name: compare([r["outcome"] for r in rows], qwen, POPULATIONS[0][1],
                          resamples=resamples, seed=seed)
            for name, rows in results.items() if name != PRIMARY_PASS},
        "transitions_qwen3_to_jev": transitions(qwen, jev),
        "disagreements_jev_vs_qwen3": disagreements(jev, qwen),
        "abstention": abstention(primary),
        "confidence_formula": confidence_formula([
            parsed[key] for key, task in tasks.items() if task.spec.arm == ARM_CHOICE]),
        "positions": {name: positions(results[name]) for name in choice_names},
        "agreement_with_primary": {
            name: {"same_selection": _count(
                sum(a["selected_index"] == b["selected_index"] for a, b in zip(
                    primary, results[name], strict=True) if a["asked"] and b["asked"]),
                sum(a["asked"] and b["asked"] for a, b in zip(primary, results[name], strict=True)))}
            for name in results if name != PRIMARY_PASS}}
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
                [r["outcome"] for r in results[pointwise]], jev, POPULATIONS[0][1],
                resamples=resamples, seed=seed)}
    if others:
        analysis["order_and_chance"] = {
            "correctness_changes": correctness_changes(
                primary, {n: results[n] for n in choice_names if n != PRIMARY_PASS}),
            "unanimity": unanimity([results[n] for n in choice_names])}
    signals = confidence_signals(primary, others, qwen)
    analysis["confidence"] = confidence_block(signals, {"jev": jev, "qwen3": qwen}, POPULATIONS[0][1])
    return results, analysis
