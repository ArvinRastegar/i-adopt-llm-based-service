"""January-derived scoring with explicit D-022 member-credit corrections.

The module has no experiment, network, database, clock, or filesystem dependency.
All externally produced similarities must be supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

COMPONENTS = (
    "hasStatisticalModifier", "hasProperty", "hasObjectOfInterest",
    "hasMatrix", "hasContextObject", "hasConstraint",
)
ENTITY_COMPONENTS = frozenset(COMPONENTS[2:5])
SCORER_VERSION = "january-derived-member-credit-v1"
CLOSE_THRESHOLD = 0.80
_CONFUSION_KEYS = ("tp", "fp", "fn", "tn")
_ON_PREFIX = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.+)$")
Similarity = Callable[[str, str], float]


def _canonical(value: Any) -> str:
    """Serialize JSON evidence canonically without non-finite numbers.

    Args:
        value: A JSON-compatible value to serialize.
    Returns:
        Stable compact Unicode JSON with sorted mapping keys.
    Raises:
        TypeError: A value is not JSON-compatible.
        ValueError: A numeric value is non-finite.
    Side Effects:
        None.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    """Hash a canonical JSON value for timestamp-free evidence identity.

    Args:
        value: JSON-compatible evidence.
    Returns:
        Lowercase SHA-256 hex digest of canonical UTF-8 bytes.
    Raises:
        TypeError: The evidence cannot be serialized.
        ValueError: The evidence contains a non-finite number.
    Side Effects:
        None.
    """
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _receipt(value: Fraction | int | float) -> dict[str, int | float]:
    """Encode an exact rational and its non-authoritative display derivative.

    Args:
        value: Rational, integer, or finalized historical finite float.
    Returns:
        Numerator, denominator, and floating-point display value.
    Raises:
        ValueError: A floating input is non-finite.
        OverflowError: The rational cannot be rendered as a finite float.
    Side Effects:
        None.
    """
    rational = Fraction(value)
    return {"numerator": rational.numerator, "denominator": rational.denominator,
            "value": float(rational)}


def _read_receipt(receipt: Mapping[str, Any]) -> Fraction:
    """Validate and decode an authoritative contribution receipt.

    Args:
        receipt: Mapping with integer numerator/denominator and display value.
    Returns:
        Exact nonnegative rational contribution or metric.
    Raises:
        ValueError: Receipt fields, sign, reduction, or display are inconsistent.
    Side Effects:
        None.
    """
    numerator, denominator = receipt.get("numerator"), receipt.get("denominator")
    if type(numerator) is not int or type(denominator) is not int:
        raise ValueError("Fraction receipt requires integer numerator/denominator")
    if numerator < 0 or denominator <= 0:
        raise ValueError("Fraction receipt requires nonnegative value and positive denominator")
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise ValueError("Fraction receipt must be reduced")
    if type(receipt.get("value")) not in (int, float) or receipt.get("value") != float(result):
        raise ValueError("Fraction display disagrees with its exact receipt")
    return result


def _metrics(contributions: Mapping[str, Fraction]) -> dict[str, Any]:
    """Calculate exact micro metrics with January zero-denominator behavior.

    Args:
        contributions: Nonnegative TP, FP, FN, and TN rational counts.
    Returns:
        Precision, recall, and F1 fraction receipts.
    Raises:
        KeyError: A TP, FP, or FN count is missing.
    Side Effects:
        None.
    """
    tp, fp, fn = (contributions[key] for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else Fraction(0)
    recall = tp / (tp + fn) if tp + fn else Fraction(0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else Fraction(0)
    return {"precision": _receipt(precision), "recall": _receipt(recall), "f1": _receipt(f1)}


def _nonempty_text(value: Any) -> bool:
    """Check the structural nonempty lexical string requirement.

    Args:
        value: An arbitrary candidate label.
    Returns:
        True only for a string containing a non-whitespace character.
    Raises:
        None.
    Side Effects:
        None.
    """
    return isinstance(value, str) and bool(value.strip())


def _validate_entity(value: Any, path: str) -> None:
    """Defensively validate a scalar or one supported system representation.

    Args:
        value: Entity string or system mapping; an empty string is allowed.
        path: Evidence pointer used in actionable validation errors.
    Returns:
        None when the representation is structurally valid.
    Raises:
        ValueError: Unsupported/malformed structure, duplicate parts, or empty roles.
    Side Effects:
        None.
    """
    if isinstance(value, str):
        if value and not value.strip():
            raise ValueError(f"{path}: whitespace-only entity is invalid")
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a string or system mapping")
    if "SymmetricSystem" in value:
        if set(value) != {"SymmetricSystem", "hasPart"} or not isinstance(value["SymmetricSystem"], str):
            raise ValueError(f"{path}: malformed symmetric system")
        parts = value.get("hasPart")
        if not isinstance(parts, list) or len(parts) < 2 or not all(_nonempty_text(part) for part in parts):
            raise ValueError(f"{path}: symmetric system requires at least two nonempty parts")
        if len(set(parts)) != len(parts):
            raise ValueError(f"{path}: exact duplicate symmetric parts are invalid")
        return
    if "AsymmetricSystem" in value:
        for first, second in (("hasNumerator", "hasDenominator"), ("hasSource", "hasTarget")):
            if set(value) == {"AsymmetricSystem", first, second}:
                if isinstance(value["AsymmetricSystem"], str) and all(_nonempty_text(value[key]) for key in (first, second)):
                    return
        raise ValueError(f"{path}: asymmetric system requires exactly one complete role pair")
    raise ValueError(f"{path}: unrecognized system representation")


def _validate_decomposition(value: Mapping[str, Any], owner: str) -> None:
    """Check the six-field evaluator boundary without importing schema code.

    Args:
        value: Canonical lexical decomposition; metadata belongs outside it.
        owner: Gold or prediction identifier used in validation errors.
    Returns:
        None for a structurally valid decomposition.
    Raises:
        ValueError: Missing/extra keys, invalid text, or unsupported nested shapes.
    Side Effects:
        None. Semantic target resolution remains the generation validator's job.
    """
    if not isinstance(value, Mapping) or set(value) != set(COMPONENTS):
        raise ValueError(f"{owner}: expected exactly the six lexical decomposition keys")
    for key in COMPONENTS[:2]:
        item = value[key]
        if not isinstance(item, str) or (item and not item.strip()):
            raise ValueError(f"{owner}/{key}: expected a string, empty or containing non-whitespace")
    for key in ENTITY_COMPONENTS:
        _validate_entity(value[key], f"{owner}/{key}")
    constraints = value["hasConstraint"]
    if not isinstance(constraints, list):
        raise ValueError(f"{owner}/hasConstraint: expected a list")
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, Mapping) or set(constraint) != {"label", "on"}:
            raise ValueError(f"{owner}/hasConstraint/{index}: expected label and on only")
        if not all(_nonempty_text(constraint[key]) for key in ("label", "on")):
            raise ValueError(f"{owner}/hasConstraint/{index}: fields must be nonempty strings")


def _scalar_similarity(gold: str, prediction: str, close: bool, similarity: Similarity) -> float:
    """Apply the frozen January scalar normalization and similarity policy.

    Args:
        gold: Original lexical gold value.
        prediction: Original lexical prediction value.
        close: Whether cosine fallback is enabled.
        similarity: Deterministic callable receiving normalized unequal strings.
    Returns:
        Zero for empties, one for normalized equality, otherwise zero or cosine.
    Raises:
        ValueError: The supplied backend returns a non-finite value.
        Exception: Backend exceptions propagate without substituting a score.
    Side Effects:
        Calls the supplied backend only for nonempty unequal Close values.
    """
    if not gold or not prediction:
        return 0.0
    normalized_gold, normalized_prediction = gold.lower().strip(), prediction.lower().strip()
    if normalized_gold == normalized_prediction:
        return 1.0
    score = float(similarity(normalized_gold, normalized_prediction)) if close else 0.0
    if not math.isfinite(score):
        raise ValueError("Similarity backend returned a non-finite value")
    return score


def _members(value: Any, path: str) -> tuple[str, list[dict[str, Any]]]:
    """Expand a nonempty entity into canonical members without its container label.

    Args:
        value: Already validated nonempty scalar or structured entity.
        path: Original component pointer for provenance.
    Returns:
        Representation name and canonical members with original source pointers.
    Raises:
        KeyError: A caller bypassed validation with malformed system input.
    Side Effects:
        None; original inputs remain unchanged.
    """
    if isinstance(value, str):
        entries, representation = [(value, path, None)], "simple"
    elif "SymmetricSystem" in value:
        entries = [(part, f"{path}/hasPart/{index}", None) for index, part in enumerate(value["hasPart"])]
        entries.sort(key=lambda entry: entry[0].encode("utf-8"))
        representation = "symmetric"
    else:
        roles = ("hasNumerator", "hasDenominator") if "hasNumerator" in value else ("hasSource", "hasTarget")
        entries = [(value[role], f"{path}/{role}", role) for role in roles]
        representation = "asymmetric"
    return representation, [
        {"index": index, "original": label, "normalized": label.lower().strip(),
         "source_pointer": pointer, "role": role}
        for index, (label, pointer, role) in enumerate(entries)
    ]


@dataclass
class _Edge:
    """Mutable residual-graph edge used only inside a pure assignment call."""

    target: int
    reverse: int
    capacity: int
    similarity_cost: Fraction
    lexical_cost: int


def _add_edge(graph: list[list[_Edge]], source: int, target: int,
              similarity_cost: Fraction, lexical_cost: int) -> _Edge:
    """Add a unit-capacity edge and its exact reverse to a residual graph.

    Args:
        graph: Function-local adjacency lists to mutate.
        source: Origin node index.
        target: Destination node index.
        similarity_cost: Exact primary cost for a matching edge.
        lexical_cost: Exact secondary canonical tie cost.
    Returns:
        The forward edge, whose residual capacity reveals final selection.
    Raises:
        IndexError: Node indices lie outside the graph.
    Side Effects:
        Mutates only the caller-owned local residual graph.
    """
    forward = _Edge(target, len(graph[target]), 1, similarity_cost, lexical_cost)
    reverse = _Edge(source, len(graph[source]), 0, -similarity_cost, -lexical_cost)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _assignment(gold_count: int, prediction_count: int,
                candidates: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """Find a cardinality-first, similarity-second, canonical-tie matching.

    Args:
        gold_count: Number of canonical gold member occurrences.
        prediction_count: Number of canonical predicted member occurrences.
        candidates: Pair records with gold_index, prediction_index, eligible, similarity.
    Returns:
        Sorted selected pair indices; each occurrence is used at most once.
    Raises:
        ValueError: A candidate similarity is non-finite.
        RuntimeError: An internally inconsistent residual path is encountered.
    Side Effects:
        None outside local graph mutation. No tolerance changes stored similarities.
    """
    sink = gold_count + prediction_count + 1
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    for index in range(gold_count):
        _add_edge(graph, 0, index + 1, Fraction(0), 0)
    for index in range(prediction_count):
        _add_edge(graph, gold_count + index + 1, sink, Fraction(0), 0)
    edge_records: list[tuple[int, int, _Edge]] = []
    pair_count = gold_count * prediction_count
    for candidate in candidates:
        if not candidate["eligible"]:
            continue
        i, j = candidate["gold_index"], candidate["prediction_index"]
        # For fixed cardinality, the highest differing bit prefers the earliest pair.
        priority = 1 << (pair_count - 1 - (i * prediction_count + j))
        edge = _add_edge(graph, i + 1, gold_count + j + 1,
                         -Fraction(candidate["similarity"]), -priority)
        edge_records.append((i, j, edge))
    while True:
        distance: list[tuple[Fraction, int] | None] = [None] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distance[0] = (Fraction(0), 0)
        for _ in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distance[node] is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    if not edge.capacity:
                        continue
                    cost = (distance[node][0] + edge.similarity_cost,
                            distance[node][1] + edge.lexical_cost)
                    if distance[edge.target] is None or cost < distance[edge.target]:
                        distance[edge.target] = cost
                        previous[edge.target] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        visited: set[int] = set()
        while node:
            if node in visited or previous[node] is None:
                raise RuntimeError("Invalid augmenting path in member assignment")
            visited.add(node)
            parent, edge_index = previous[node]
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
    return sorted((i, j) for i, j, edge in edge_records if edge.capacity == 0)


def _component_record(branch: str, counts: Sequence[Fraction | float | int],
                      evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Build one component result from complete counts and branch evidence.

    Args:
        branch: Versioned dispatch rule name.
        counts: TP, FP, FN, TN in fixed order; finalized float values are retained.
        evidence: Normalization, pairing, and correction evidence.
    Returns:
        JSON-compatible component with authoritative counts and exact metrics.
    Raises:
        ValueError: Counts are negative or not exactly four entries.
    Side Effects:
        None.
    """
    if len(counts) != 4:
        raise ValueError("Expected TP/FP/FN/TN")
    fractions = dict(zip(_CONFUSION_KEYS, map(Fraction, counts), strict=True))
    if any(value < 0 for value in fractions.values()):
        raise ValueError("Negative component contribution")
    return {"branch": branch, "contributions": {key: _receipt(value) for key, value in fractions.items()},
            "metrics": _metrics(fractions), "evidence": dict(evidence)}


def _system_component(gold: Any, prediction: Any, key: str,
                      close: bool, similarity: Similarity) -> dict[str, Any]:
    """Score D-022 members with corresponding-role or unordered assignment.

    Args:
        gold: Nonempty validated gold entity.
        prediction: Nonempty validated predicted entity.
        key: The entity-bearing component name.
        close: Whether to use the Close scalar matching policy where permitted.
        similarity: Normalized-string cosine callable.
    Returns:
        Component record with exact unit-mass partial credit and all candidates.
    Raises:
        ValueError: Similarity is invalid or the resulting mass invariant fails.
    Side Effects:
        Calls only the supplied deterministic similarity backend when required.
    """
    gold_type, gold_members = _members(gold, f"/gold/{key}")
    prediction_type, prediction_members = _members(prediction, f"/prediction/{key}")
    ordered = gold_type == prediction_type == "asymmetric"
    literal = gold_type == prediction_type == "symmetric"
    threshold = CLOSE_THRESHOLD if close else 1.0
    candidates = []
    for i, gold_member in enumerate(gold_members):
        for j, prediction_member in enumerate(prediction_members):
            allowed = not ordered or i == j
            if allowed:
                score = float(gold_member["original"] == prediction_member["original"]) if literal else _scalar_similarity(
                    gold_member["original"], prediction_member["original"], close, similarity)
            else:
                score = None
            candidates.append({"gold_index": i, "prediction_index": j,
                               "role_allowed": allowed, "similarity": score,
                               "eligible": allowed and score >= threshold})
    selected = _assignment(len(gold_members), len(prediction_members), candidates)
    selected_set = set(selected)
    for candidate in candidates:
        candidate["selected"] = (candidate["gold_index"], candidate["prediction_index"]) in selected_set
    g, p, m = len(gold_members), len(prediction_members), len(selected)
    denominator = g + p - m
    counts = (Fraction(m, denominator), Fraction(p - m, denominator), Fraction(g - m, denominator), Fraction(0))
    if sum(counts) != 1:
        raise ValueError("System contribution mass must equal one")
    selected_sum = sum((Fraction(candidate["similarity"]) for candidate in candidates if candidate["selected"]), Fraction(0))
    role_evidence = "available" if ordered else "unavailable" if "asymmetric" in (gold_type, prediction_type) else "not_applicable"
    return _component_record("system-member-credit-v1", counts, {
        "correction": "D-022", "gold_representation": gold_type,
        "prediction_representation": prediction_type, "gold_members": gold_members,
        "prediction_members": prediction_members, "equivalence": "literal" if literal else "scalar-normalized",
        "matching_mode": "corresponding-role" if ordered else "unordered-membership",
        "asymmetric_role_evidence": role_evidence, "threshold": threshold,
        "candidates": candidates, "selected_pairs": [list(pair) for pair in selected],
        "unmatched_gold_indices": [i for i in range(g) if all(i != pair[0] for pair in selected)],
        "unmatched_prediction_indices": [j for j in range(p) if all(j != pair[1] for pair in selected)],
        "g": g, "p": p, "m": m, "U": denominator,
        "assignment_objective": {"cardinality": m, "similarity_sum": _receipt(selected_sum),
                                 "tie_policy": "lexicographically-smallest-canonical-pair-list"},
    })


def _normalize_constraint(constraint: Mapping[str, str]) -> dict[str, str]:
    """Preserve January prefix removal and Constraint whitespace normalization.

    Args:
        constraint: Validated label/on strings.
    Returns:
        New normalized label/on mapping; original values are never modified.
    Raises:
        KeyError: A caller bypassed validation and omitted a field.
    Side Effects:
        None.
    """
    target = constraint["on"]
    match = _ON_PREFIX.match(target)
    if match and match.group(1) in COMPONENTS:
        target = match.group(2).strip()
    return {"label": re.sub(r"\s+", " ", constraint["label"].strip().lower()),
            "on": re.sub(r"\s+", " ", target.strip().lower())}


def _constraints(gold: list[Mapping[str, str]], prediction: list[Mapping[str, str]],
                 close: bool, similarity: Similarity) -> dict[str, Any]:
    """Execute the January greedy Constraint branch with original float arithmetic.

    Args:
        gold: Original ordered gold Constraints.
        prediction: Original ordered predicted Constraints.
        close: Whether scalar cosine fallback and threshold 0.80 are enabled.
        similarity: Deterministic normalized-string cosine callable.
    Returns:
        Constraint record with all normalized candidates, selected order, and exact
        numerical ratios of finalized historical floating-point contributions.
    Raises:
        ValueError: The supplied similarity backend returns non-finite values.
    Side Effects:
        Calls only the supplied similarity backend; no external state is accessed.
    """
    gold_normalized = [_normalize_constraint(item) for item in gold]
    prediction_normalized = [_normalize_constraint(item) for item in prediction]
    evidence: dict[str, Any] = {"correction": None, "gold_original": gold,
        "prediction_original": prediction, "gold_normalized": gold_normalized,
        "prediction_normalized": prediction_normalized,
        "threshold": CLOSE_THRESHOLD if close else 1.0,
        "matching_mode": "january-greedy-row-major", "candidates": [], "selected_pairs": []}
    if not gold:
        return _component_record("january-constraints-empty", (0, int(bool(prediction)), 0, int(not prediction)), evidence)
    n_gold, n_prediction = len(gold), len(prediction)
    unit = 1.0 / (2 * n_gold)
    threshold = evidence["threshold"]
    for i, gold_item in enumerate(gold_normalized):
        for j, prediction_item in enumerate(prediction_normalized):
            label_score = _scalar_similarity(gold_item["label"], prediction_item["label"], close, similarity)
            target_score = _scalar_similarity(gold_item["on"], prediction_item["on"], close, similarity)
            evidence["candidates"].append({"gold_index": i, "prediction_index": j,
                "label_similarity": label_score, "on_similarity": target_score,
                "similarity": (label_score + target_score) / 2.0, "selected": False})
    tp = fp = fn = 0.0
    used_gold: set[int] = set()
    used_prediction: set[int] = set()
    while True:
        available = [candidate for candidate in evidence["candidates"]
                     if candidate["gold_index"] not in used_gold and candidate["prediction_index"] not in used_prediction]
        if not available:
            break
        # max returns the first equal maximum; candidates were appended row-major.
        chosen = max(available, key=lambda candidate: candidate["similarity"])
        if chosen["similarity"] < 0:
            break
        chosen["selected"] = True
        i, j = chosen["gold_index"], chosen["prediction_index"]
        used_gold.add(i)
        used_prediction.add(j)
        evidence["selected_pairs"].append([i, j])
        for score in (chosen["label_similarity"], chosen["on_similarity"]):
            if score >= threshold:
                tp += unit
            else:
                fp += unit
    fn += (n_gold - len(used_gold)) * 2 * unit
    fp += (n_prediction - len(used_prediction)) * 2 * unit
    total = tp + fp + fn
    before = {"tp": tp, "fp": fp, "fn": fn, "total": total}
    numerical_correction = "none"
    if 1.0 - total > 1e-6:
        fp += 1.0 - total
        numerical_correction = "deficit-added-to-fp"
    elif total - 1.0 > 1e-6:
        tp /= total
        fp /= total
        fn /= total
        numerical_correction = "excess-divided-by-total"
    evidence.update({"unit_source_float": unit, "before_numerical_correction": before,
                     "numerical_correction": numerical_correction,
                     "unmatched_gold_indices": sorted(set(range(n_gold)) - used_gold),
                     "unmatched_prediction_indices": sorted(set(range(n_prediction)) - used_prediction)})
    return _component_record("january-constraints-greedy", (tp, fp, fn, 0.0), evidence)


def _compare(gold: Any, prediction: Any, key: str, close: bool,
             similarity: Similarity) -> dict[str, Any]:
    """Select the frozen empty, scalar, Constraint, or D-022 member branch.

    Args:
        gold: Validated gold component.
        prediction: Validated prediction component.
        key: One of the six component names.
        close: Exact versus Close mode.
        similarity: Normalized-string cosine callable.
    Returns:
        Complete component record with evidence and exact fraction receipts.
    Raises:
        ValueError: A supplied similarity or derived contribution is invalid.
    Side Effects:
        May call the supplied similarity callable for Close comparisons.
    """
    if key == "hasConstraint":
        return _constraints(gold, prediction, close, similarity)
    if not gold or not prediction:
        counts = (0, 1, 0, 0) if prediction else (0, 0, 1, 0) if gold else (0, 0, 0, 1)
        return _component_record("january-empty", counts, {"correction": None,
            "gold_present": bool(gold), "prediction_present": bool(prediction)})
    if key in ENTITY_COMPONENTS and (isinstance(gold, Mapping) or isinstance(prediction, Mapping)):
        return _system_component(gold, prediction, key, close, similarity)
    score = _scalar_similarity(gold, prediction, close, similarity)
    threshold = CLOSE_THRESHOLD if close else 1.0
    return _component_record("january-scalar", (int(score >= threshold), int(score < threshold), 0, 0), {
        "correction": None, "gold_original": gold, "prediction_original": prediction,
        "gold_normalized": gold.lower().strip(), "prediction_normalized": prediction.lower().strip(),
        "similarity": score, "threshold": threshold})


def _sum_components(components: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Sum exact component receipts before calculating item-level micro metrics.

    Args:
        components: The six component records for one scoring mode.
    Returns:
        Component mapping, summed confusion receipts, and micro metric receipts.
    Raises:
        ValueError: A component contains an invalid receipt.
    Side Effects:
        None.
    """
    for component in components.values():
        counts = {key: _read_receipt(component["contributions"][key]) for key in _CONFUSION_KEYS}
        if component["metrics"] != _metrics(counts):
            raise ValueError("Component metrics disagree with contribution receipts")
    totals = {key: sum((_read_receipt(component["contributions"][key]) for component in components.values()), Fraction(0))
              for key in _CONFUSION_KEYS}
    return {"components": dict(components), "totals": {key: _receipt(value) for key, value in totals.items()},
            "metrics": _metrics(totals)}


def evaluate_item(gold: Mapping[str, Any], prediction: Mapping[str, Any],
                  similarity: Similarity, *, metadata: Mapping[str, Any] | None = None,
                  similarity_identity: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
    """Evaluate a single immutable six-field lexical gold/prediction pair.

    Args:
        gold: Exactly six structurally valid canonical lexical fields.
        prediction: The same six fields, including explicit empties for terminal
            content-invalid outcomes; operational failures must not be passed here.
        similarity: Callable accepting normalized unequal strings and returning a
            finite cosine value, without downloading or making provider calls.
        metadata: Optional JSON metadata such as variable_id, category, hashes,
            and terminal_invalid. It is retained but never used to match labels.
        similarity_identity: Frozen backend identity or an explicitly named test
            fixture. Omission records an unverified identity, never scientific parity.
    Returns:
        JSON-compatible Exact/Close components, totals, precision/recall/F1,
        full comparison evidence, immutable input hashes, and a result hash.
    Raises:
        ValueError: Invalid lexical structures, non-finite similarities, or an
            optional gold_hash/prediction_hash in metadata does not match the input.
        TypeError: Evidence is not JSON-compatible or similarity is not callable.
        Exception: Similarity backend failures propagate without scoring fallback.
    Side Effects:
        Only invokes the supplied similarity callable. Never writes inputs, files,
        database state, timestamps, random IDs, or network requests itself.
    """
    _validate_decomposition(gold, "gold")
    _validate_decomposition(prediction, "prediction")
    if not callable(similarity):
        raise TypeError("similarity must be a callable")
    evidence_metadata = json.loads(_canonical(dict(metadata or {})))
    gold_hash, prediction_hash = _hash(gold), _hash(prediction)
    for key, expected in (("gold_hash", gold_hash), ("prediction_hash", prediction_hash)):
        if key in evidence_metadata and evidence_metadata[key] != expected:
            raise ValueError(f"Metadata {key} disagrees with the supplied lexical input")
    identity = similarity_identity if similarity_identity is not None else {"kind": "unverified-supplied-callable"}
    result: dict[str, Any] = {"schema_version": "1.0", "scorer_version": SCORER_VERSION,
        "similarity_identity": json.loads(_canonical(identity)), "close_threshold": CLOSE_THRESHOLD,
        "gold_hash": gold_hash, "prediction_hash": prediction_hash, "metadata": evidence_metadata}
    for mode, close in (("exact", False), ("close", True)):
        result[mode] = _sum_components({key: _compare(gold[key], prediction[key], key, close, similarity) for key in COMPONENTS})
    result["result_hash"] = _hash(result)
    # Detach nested original Constraint evidence from caller-owned mutable inputs.
    return json.loads(_canonical(result))


def aggregate_items(records: Iterable[Mapping[str, Any]], *,
                    expected_variable_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Regenerate exact micro metrics for one explicitly scoped item collection.

    Args:
        records: evaluate_item records sharing scorer/backend identity. Input order
            is irrelevant; callers must scope one configuration and repetition.
        expected_variable_ids: Optional complete population for strict coverage.
            Each record must then carry a unique matching metadata.variable_id.
    Returns:
        Exact/Close summed counts and micro metrics, item/component support,
        canonical source hashes, optional variable population, and aggregate hash.
    Raises:
        ValueError: Empty scope, corrupt result/receipt, duplicate variable identity,
            incompatible scorer/backend, or incomplete expected coverage.
        TypeError: A record or expected variable identifier is not serializable.
    Side Effects:
        Consumes the supplied iterable once; no persistence or network access.
    """
    items = list(records)
    if not items:
        raise ValueError("Cannot aggregate an empty evaluation scope")
    identity = (items[0].get("scorer_version"), _canonical(items[0].get("similarity_identity")), items[0].get("close_threshold"))
    ids: list[str] = []
    for item in items:
        if (item.get("scorer_version"), _canonical(item.get("similarity_identity")), item.get("close_threshold")) != identity:
            raise ValueError("Incompatible scorer or similarity identities in aggregate")
        if (item.get("scorer_version") != SCORER_VERSION or item.get("schema_version") != "1.0"
                or item.get("close_threshold") != CLOSE_THRESHOLD):
            raise ValueError("Unsupported evaluation record version")
        unhashed = {key: value for key, value in item.items() if key != "result_hash"}
        if item.get("result_hash") != _hash(unhashed):
            raise ValueError("Evaluation record hash mismatch")
        variable_id = item.get("metadata", {}).get("variable_id")
        if variable_id is not None:
            if not isinstance(variable_id, str) or not variable_id:
                raise ValueError("metadata.variable_id must be a nonempty string")
            ids.append(variable_id)
        for mode in ("exact", "close"):
            if set(item[mode]["components"]) != set(COMPONENTS):
                raise ValueError("Evaluation record does not cover all six components")
            if _sum_components(item[mode]["components"]) != item[mode]:
                raise ValueError("Evaluation totals/metrics disagree with component receipts")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate variable identity in aggregate scope")
    source_hashes = [item["result_hash"] for item in items]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("Duplicate evaluation record in aggregate scope")
    if expected_variable_ids is not None:
        expected = list(expected_variable_ids)
        if not all(isinstance(item, str) and item for item in expected) or len(expected) != len(set(expected)):
            raise ValueError("Expected population must contain unique nonempty string IDs")
        if len(ids) != len(items) or set(ids) != set(expected):
            raise ValueError("Evaluation records do not cover the expected population")
    result: dict[str, Any] = {"schema_version": "1.0", "scorer_version": SCORER_VERSION,
        "similarity_identity": items[0]["similarity_identity"], "close_threshold": CLOSE_THRESHOLD,
        "support": {"items": len(items), "components": len(items) * len(COMPONENTS)},
        "variable_ids": sorted(ids), "source_record_hashes": sorted(item["result_hash"] for item in items)}
    for mode in ("exact", "close"):
        totals = {key: sum((_read_receipt(item[mode]["totals"][key]) for item in items), Fraction(0)) for key in _CONFUSION_KEYS}
        result[mode] = {"totals": {key: _receipt(value) for key, value in totals.items()}, "metrics": _metrics(totals)}
    result["aggregate_hash"] = _hash(result)
    return json.loads(_canonical(result))
