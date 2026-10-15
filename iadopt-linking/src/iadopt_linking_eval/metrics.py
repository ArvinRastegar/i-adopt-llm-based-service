"""Exact-match link metrics as numerator/denominator receipts.

Input is one mapping per scored slot (see ``evaluate_slot``). Nothing here reads a file, calls a
service or consults a clock, and no ratio is ever held as a float.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from .identity import REACHABLE

SOURCE_ORDER = REACHABLE
# Reranker scores saturate near 1, so the grid keeps resolving above 0.9.
THRESHOLDS = (*(Fraction(step, 10) for step in range(10)),
              Fraction(95, 100), Fraction(99, 100), Fraction(999, 1000))
MARGIN_BOUNDS = (("1e-4", Fraction(1, 10_000)), ("1e-3", Fraction(1, 1_000)),
                 ("1e-2", Fraction(1, 100)))
CATEGORIES = ("correct", "wrong_within_source", "wrong_source", "gold_not_retrieved",
              "no_candidates", "unreachable_vocabulary")
SCORER_VERSION = "exact-identity-pooled-top1-v3"


def _count(numerator: int, denominator: int) -> dict[str, int]:
    """Record a count ratio without reducing it.

    Args: numerator: items satisfying a condition; denominator: items considered.
    Returns: {"numerator", "denominator"}.
    Raises: None.
    Side effects: none.
    """
    return {"numerator": numerator, "denominator": denominator}


def _exact(value: Fraction) -> dict[str, int]:
    """Record an exact rational, reduced.

    Args: value: any Fraction.
    Returns: {"numerator", "denominator"} in lowest terms.
    Raises: None.
    Side effects: none.
    """
    return {"numerator": value.numerator, "denominator": value.denominator}


def choose(pool: Sequence[Mapping[str, Any]]) -> int | None:
    """Pick the candidate the reranker would select.

    Args: pool: candidates with ``score``, ``source`` and ``rank``.
    Returns: index of the highest score; ties go to the earlier source, then the earlier rank;
    None for an empty pool.
    Raises: ValueError for a source outside SOURCE_ORDER.
    Side effects: none; deterministic.
    """
    best: tuple[Any, ...] | None = None
    chosen: int | None = None
    for index, candidate in enumerate(pool):
        key = (-candidate["score"], SOURCE_ORDER.index(candidate["source"]), candidate["rank"])
        if best is None or key < best:
            best, chosen = key, index
    return chosen


def _best_in(pool: Sequence[Mapping[str, Any]], source: str) -> Mapping[str, Any] | None:
    """Pick the best-scoring candidate of one source.

    Args: pool: candidates; source: one of SOURCE_ORDER.
    Returns: the candidate, or None when that source returned nothing.
    Raises: ValueError for a source outside SOURCE_ORDER.
    Side effects: none.
    """
    members = [candidate for candidate in pool if candidate["source"] == source]
    index = choose(members)
    return None if index is None else members[index]


def _is(candidate: Mapping[str, Any] | None, gold: tuple[str, str]) -> bool:
    """Say whether a candidate is the gold entity.

    Args: candidate: a pool entry or None; gold: (vocabulary, local identifier).
    Returns: False when there is no candidate.
    Raises: KeyError if the candidate lacks its identity fields.
    Side effects: none.
    """
    return candidate is not None and (candidate["vocabulary"], candidate["local_id"]) == gold


def _wikidata_matches(candidate: Mapping[str, Any]) -> set[str]:
    """List the Wikidata items a QUDT candidate says it matches.

    Args: candidate: a pool entry.
    Returns: Wikidata local identifiers from its ``wikidata_match`` extra; empty for others.
    Raises: None.
    Side effects: none.
    """
    return {iri.rsplit("/", 1)[-1] for iri in candidate.get("extras", {}).get("wikidata_match", [])}


def _equivalent(chosen: Mapping[str, Any] | None, gold: tuple[str, str],
                pool: Sequence[Mapping[str, Any]]) -> bool:
    """Say whether a selection is the gold entity or its QUDT/Wikidata counterpart.

    Args: chosen: the selected pool entry or None; gold: (vocabulary, local identifier); pool:
    the slot's candidates, which carry QUDT's wikidataMatch.
    Returns: True for the gold itself, for a QUDT kind whose wikidataMatch is the gold Wikidata
    item, and for the Wikidata item that the gold QUDT kind (when retrieved) matches.
    Raises: None.
    Side effects: none.
    """
    if chosen is None:
        return False
    if (chosen["vocabulary"], chosen["local_id"]) == gold:
        return True
    if chosen["source"] == "qudt" and gold[0] == "wikidata":
        return gold[1] in _wikidata_matches(chosen)
    if chosen["source"] == "wikidata" and gold[0] == "qudt":
        counterpart = next((c for c in pool if (c["vocabulary"], c["local_id"]) == gold), None)
        return counterpart is not None and chosen["local_id"] in _wikidata_matches(counterpart)
    return False


def evaluate_slot(record: Mapping[str, Any]) -> dict[str, Any]:
    """Judge one scored slot against its gold identity.

    Args: record: slot fields plus ``gold`` {vocabulary, local_id} and ``pool`` candidates, each
    with source, rank, vocabulary, local_id, label and score.
    Returns: the slot's outcome: category, correctness, the selected candidate and the
    diagnostics (gold in pool, routing, per-source and native-order comparisons).
    Raises: KeyError for a missing field; ValueError for an unknown source.
    Side effects: none.
    """
    gold = (record["gold"]["vocabulary"], record["gold"]["local_id"])
    pool = record["pool"]
    keys = [(candidate["vocabulary"], candidate["local_id"]) for candidate in pool]
    chosen = choose(pool)
    reachable = gold[0] in REACHABLE
    in_pool = gold in keys
    if not reachable:
        category = "unreachable_vocabulary"
    elif chosen is None:
        category = "no_candidates"
    elif keys[chosen] == gold:
        category = "correct"
    elif not in_pool:
        category = "gold_not_retrieved"
    elif pool[chosen]["source"] != gold[0]:
        category = "wrong_source"
    else:
        category = "wrong_within_source"
    own_best = _best_in(pool, gold[0]) if reachable else None
    own_first = next((c for c in pool if c["source"] == gold[0] and c["rank"] == 1), None)
    wikidata_best = _best_in(pool, "wikidata")
    ranked = sorted((candidate["score"] for candidate in pool), reverse=True)
    return {
        "slot_id": record["slot_id"], "variable_id": record["variable_id"],
        "source_path": record.get("source_path", ""), "role": record["role"], "slot": record["slot"], "is_member": record["is_member"],
        "term": record["term"], "query_key": record["query_key"], "gold": list(gold),
        "category": category, "correct": category == "correct", "reachable": reachable,
        "gold_in_pool": in_pool, "pool_size": len(pool),
        "selected": None if chosen is None else {
            "source": pool[chosen]["source"], "local_id": pool[chosen]["local_id"],
            "label": pool[chosen]["label"], "score": pool[chosen]["score"]},
        "routed_correctly": (None if chosen is None or not reachable
                             else pool[chosen]["source"] == gold[0]),
        # A source that returned nothing is a miss, so these compare with the headline top-1.
        "oracle_routed_correct": (None if not reachable else _is(own_best, gold)),
        "native_first_correct": (None if not reachable else _is(own_first, gold)),
        "wikidata_only_correct": (None if gold[0] != "wikidata" else _is(wikidata_best, gold)),
        # Top score minus runner-up: how decisively the selection was made.
        "margin": ranked[0] - ranked[1] if len(ranked) >= 2 else None,
        # Diagnostic only: counts a QUDT kind and its wikidataMatch item as the same entity.
        "equivalent_correct": _equivalent(None if chosen is None else pool[chosen], gold, pool),
    }


def _macro(outcomes: Sequence[Mapping[str, Any]], field: str = "correct") -> dict[str, int]:
    """Average a boolean outcome per variable, then across variables.

    Args: outcomes: evaluate_slot results; field: boolean outcome to average.
    Returns: the exact mean of per-variable accuracies; denominator 1 when nothing was judged.
    Raises: None.
    Side effects: none.
    """
    groups: dict[str, list[bool]] = defaultdict(list)
    for outcome in outcomes:
        if outcome[field] is not None:
            groups[outcome["variable_id"]].append(bool(outcome[field]))
    if not groups:
        return _exact(Fraction(0))
    return _exact(sum((Fraction(sum(v), len(v)) for v in groups.values()), Fraction(0))
                  / len(groups))


def _block(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise one population of outcomes.

    Args: outcomes: evaluate_slot results for the population.
    Returns: counts and receipts for accuracy, retrieval recall, routing and diagnostics.
    Raises: None.
    Side effects: none.
    """
    total = len(outcomes)
    reachable = [o for o in outcomes if o["reachable"]]
    routed = [o for o in reachable if o["routed_correctly"] is not None]
    in_pool = [o for o in reachable if o["gold_in_pool"]]

    def tally(field: str, population: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        """Count a boolean diagnostic over the slots where it is defined.

        Args: field: boolean diagnostic; population: outcomes to consider.
        Returns: count receipt over slots whose diagnostic is not None.
        Raises: None.
        Side effects: none.
        """
        judged = [o for o in population if o[field] is not None]
        return _count(sum(bool(o[field]) for o in judged), len(judged))

    return {
        "n": total,
        "distinct_terms": len({o["query_key"] for o in outcomes}),
        "distinct_term_gold_pairs": len({(o["query_key"], tuple(o["gold"])) for o in outcomes}),
        "top1": _count(sum(o["correct"] for o in outcomes), total),
        "top1_macro_by_variable": _macro(outcomes),
        "top1_with_qudt_equivalence": _count(sum(o["equivalent_correct"] for o in outcomes), total),
        "reachable_slots": _count(len(reachable), total),
        "gold_in_pool": _count(len(in_pool), len(reachable)),
        "top1_given_gold_in_pool": _count(sum(o["correct"] for o in in_pool), len(in_pool)),
        "routed_to_gold_source": _count(sum(bool(o["routed_correctly"]) for o in routed), len(routed)),
        "oracle_routed_top1": tally("oracle_routed_correct", reachable),
        "oracle_routed_top1_macro_by_variable": _macro(reachable, "oracle_routed_correct"),
        "native_first_hit_top1": tally("native_first_correct", reachable),
        "native_first_hit_top1_macro_by_variable": _macro(reachable, "native_first_correct"),
        "wikidata_only_top1": tally("wikidata_only_correct", outcomes),
        "wikidata_only_top1_macro_by_variable": _macro(outcomes, "wikidata_only_correct"),
        "consistency_ceiling": term_consistency_ceiling(outcomes),
        "categories": {name: sum(o["category"] == name for o in outcomes) for name in CATEGORIES},
    }


def _threshold_rows(outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Trade coverage against precision by accepting only confident selections.

    Args: outcomes: evaluate_slot results.
    Returns: one row per threshold: slots accepted (selected score >= threshold) and correct.
    Raises: None.
    Side effects: none.
    """
    rows = []
    total = len(outcomes)
    for threshold in THRESHOLDS:
        accepted = [o for o in outcomes
                    if o["selected"] is not None and o["selected"]["score"] >= float(threshold)]
        rows.append({"threshold": _exact(threshold),
                     "coverage": _count(len(accepted), total),
                     "precision": _count(sum(o["correct"] for o in accepted), len(accepted))})
    return rows


def term_consistency_ceiling(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Bound what any system that sees only the term can reach.

    Args: outcomes: evaluate_slot results.
    Returns: slots explained by the majority gold identity of their term, over all slots.
    Raises: None.
    Side effects: none.
    """
    by_term: dict[str, Counter] = defaultdict(Counter)
    for outcome in outcomes:
        by_term[outcome["query_key"]][tuple(outcome["gold"])] += 1
    return _count(sum(max(counter.values()) for counter in by_term.values()), len(outcomes))


def summarise(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate a run into the reported metrics.

    Args: outcomes: evaluate_slot results for every slot in the population.
    Returns: populations (all controlled-vocabulary slots, reachable and unreachable
    vocabularies, by gold vocabulary, by slot, top-level versus members, by term length), the source-confusion
    counts, the threshold table, how many selections were decided by a small score margin and
    the most repeated terms.
    Raises: None.
    Side effects: none.
    """
    populations: dict[str, list[Mapping[str, Any]]] = {
        "all": list(outcomes), "reachable": [o for o in outcomes if o["reachable"]],
        "unreachable": [o for o in outcomes if not o["reachable"]]}
    for vocabulary in sorted({o["gold"][0] for o in outcomes}):
        populations[f"gold:{vocabulary}"] = [o for o in outcomes if o["gold"][0] == vocabulary]
    for slot in sorted({o["slot"] for o in outcomes}):
        populations[f"slot:{slot}"] = [o for o in outcomes if o["slot"] == slot]
    populations["role:top-level"] = [o for o in outcomes if not o["is_member"]]
    populations["role:member"] = [o for o in outcomes if o["is_member"]]
    for label, wanted in (("1", lambda n: n == 1), ("2", lambda n: n == 2), ("3+", lambda n: n >= 3)):
        populations[f"words:{label}"] = [
            o for o in outcomes if wanted(len(o["query_key"].split()))]
    confusion: Counter[str] = Counter(
        f"{o['gold'][0]}>{o['selected']['source']}"
        for o in outcomes if o["reachable"] and o["selected"] is not None)
    margins = [o["margin"] for o in outcomes if o["margin"] is not None]
    terms = Counter(o["query_key"] for o in outcomes)
    wrong = Counter(o["query_key"] for o in outcomes if not o["correct"])
    return {
        "scorer_version": SCORER_VERSION,
        "populations": {name: _block(members) for name, members in populations.items() if members},
        "routing_confusion": dict(sorted(confusion.items())),
        "thresholds_all": _threshold_rows(outcomes),
        "margins": {f"below_{label}": _count(sum(m < float(bound) for m in margins), len(margins))
                    for label, bound in MARGIN_BOUNDS},
        "most_repeated_terms": [
            {"term": term, "slots": count, "wrong": wrong[term]}
            for term, count in sorted(terms.items(), key=lambda item: (-item[1], item[0]))[:5]],
    }
