"""Hand-calculated scoring boundaries, audit evidence, and exact aggregates."""

from copy import deepcopy
from fractions import Fraction
from itertools import combinations, permutations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from iadopt_eval import COMPONENTS, aggregate_items, evaluate_item
from iadopt_eval.core import _assignment, _canonical, _hash


def blank(**values):
    """Build a valid six-field fixture with explicit empty defaults."""
    return {**{key: [] if key == "hasConstraint" else "" for key in COMPONENTS}, **values}


def symmetric(*parts, label="unscored"):
    """Build a symmetric-system test value without canonicalizing its evidence."""
    return {"SymmetricSystem": label, "hasPart": list(parts)}


def asymmetric(first, second, *, ratio=False, label="unscored"):
    """Build a complete ordered asymmetric test system."""
    return {"AsymmetricSystem": label,
            "hasNumerator" if ratio else "hasSource": first,
            "hasDenominator" if ratio else "hasTarget": second}


def zero_similarity(left, right):
    """Return an explicit offline non-match for unequal scalar strings."""
    return 0.0


def fraction(receipt):
    """Decode a receipt for exact test assertions."""
    return Fraction(receipt["numerator"], receipt["denominator"])


def component(gold, prediction, *, similarity=zero_similarity, key="hasObjectOfInterest", mode="exact"):
    """Evaluate one changed component while retaining five explicitly empty slots."""
    result = evaluate_item(blank(**{key: gold}), blank(**{key: prediction}), similarity,
                           similarity_identity="offline-zero-fixture")
    return result[mode]["components"][key]


@pytest.mark.parametrize("gold,prediction,expected", [
    ("Water", " water ", (1, 0, 0, 0)), ("water", "soil", (0, 1, 0, 0)),
    ("water", "", (0, 0, 1, 0)), ("", "soil", (0, 1, 0, 0)), ("", "", (0, 0, 0, 1)),
    (symmetric("water", "air"), "", (0, 0, 1, 0)),
    ("", asymmetric("water", "air"), (0, 1, 0, 0)),
])
def test_january_scalar_and_empty_boundary(gold, prediction, expected):
    """Preserve January unit confusion, especially wrong-scalar FP without FN."""
    for mode in ("exact", "close"):
        record = component(gold, prediction, mode=mode)
        assert tuple(fraction(record["contributions"][key]) for key in ("tp", "fp", "fn", "tn")) == expected


@pytest.mark.parametrize("score,expected_tp", [(0.799999999, 0), (0.8, 1), (0.800000001, 1)])
def test_close_threshold_inclusive_and_normalized_inputs(score, expected_tp):
    """Check stored cosine boundaries and the exact normalized backend operands."""
    calls = []

    def cosine(left, right):
        """Record normalized fixture operands and return the fixed test score."""
        calls.append((left, right))
        return score

    record = component(" WATER ", "Liquid", similarity=cosine, mode="close")
    assert fraction(record["contributions"]["tp"]) == expected_tp
    assert calls == [("water", "liquid")]
    assert record["evidence"]["similarity"] == score


@pytest.mark.parametrize("gold,prediction,counts,f1", [
    (symmetric("water", "air"), "water", (Fraction(1, 2), 0, Fraction(1, 2)), Fraction(2, 3)),
    ("water", symmetric("water", "air"), (Fraction(1, 2), Fraction(1, 2), 0), Fraction(2, 3)),
    (symmetric("water", "air"), symmetric("water", "soil"), (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)), Fraction(1, 2)),
    (symmetric("water", "air"), symmetric("air", "water"), (1, 0, 0), 1),
    (symmetric("water", "air", "soil"), "water", (Fraction(1, 3), 0, Fraction(2, 3)), Fraction(1, 2)),
    (symmetric("water", "air"), "rock", (0, Fraction(1, 3), Fraction(2, 3)), 0),
    (symmetric("water", "air"), symmetric("soil", "rock"), (0, Fraction(1, 2), Fraction(1, 2)), 0),
    (asymmetric("water", "air"), asymmetric("water", "soil"), (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)), Fraction(1, 2)),
])
def test_all_documented_partial_credit_examples(gold, prediction, counts, f1):
    """Assert exact D-022 member fractions separately from harmonic component F1."""
    for mode in ("exact", "close"):
        record = component(gold, prediction, mode=mode)
        assert tuple(fraction(record["contributions"][key]) for key in ("tp", "fp", "fn")) == counts
        assert sum(fraction(value) for value in record["contributions"].values()) == 1
        assert fraction(record["metrics"]["f1"]) == f1
        assert record["evidence"]["correction"] == "D-022"


def test_symmetric_literal_equivalence_container_exclusion_and_permutation():
    """Preserve literal symmetric identity without embedding container labels."""
    def forbidden(left, right):
        """Fail if the literal symmetric branch invokes a similarity backend."""
        raise AssertionError("Symmetric/symmetric must never embed")

    gold = symmetric("water", "air", label="DO NOT EMBED GOLD")
    prediction = symmetric("air", "Water", label="DO NOT EMBED PREDICTION")
    first = component(gold, prediction, similarity=forbidden, mode="close")
    second = component(symmetric("air", "water", label="changed"),
                       symmetric("Water", "air", label="also changed"), similarity=forbidden, mode="close")
    assert first["contributions"] == second["contributions"]
    assert first["evidence"]["selected_pairs"] == second["evidence"]["selected_pairs"]
    assert fraction(first["contributions"]["tp"]) == Fraction(1, 3)
    assert "DO NOT EMBED" not in _canonical(first)
    assert component(gold, symmetric("water ", "air"), mode="exact")["contributions"] == first["contributions"]


def test_asymmetric_roles_fallback_mixed_membership_and_no_reuse():
    """Keep role slots ordered while explicitly marking missing shared role evidence."""
    ordered = component(asymmetric("water", "air"), asymmetric("air", "water", ratio=True))
    assert fraction(ordered["contributions"]["tp"]) == 0
    assert ordered["evidence"]["asymmetric_role_evidence"] == "available"
    assert all(candidate["similarity"] is None for candidate in ordered["evidence"]["candidates"] if not candidate["role_allowed"])
    fallback = component(asymmetric("water", "air"), asymmetric(" WATER ", "AIR", ratio=True))
    assert fraction(fallback["contributions"]["tp"]) == 1
    assert [member["role"] for member in fallback["evidence"]["gold_members"]] == ["hasSource", "hasTarget"]
    assert [member["role"] for member in fallback["evidence"]["prediction_members"]] == ["hasNumerator", "hasDenominator"]
    mixed = component(asymmetric("water", "air"), symmetric("air", "water"))
    assert fraction(mixed["contributions"]["tp"]) == 1
    assert mixed["evidence"]["asymmetric_role_evidence"] == "unavailable"
    duplicate_roles = component(asymmetric("water", "Water"), "water")
    assert duplicate_roles["evidence"]["m"] == 1
    assert fraction(duplicate_roles["contributions"]["tp"]) == Fraction(1, 2)
    duplicate_normalized = component(symmetric("water", "Water"), "water")
    assert duplicate_normalized["evidence"]["g"] == 2
    assert duplicate_normalized["evidence"]["m"] == 1


def test_max_cardinality_counterexample_and_exact_lexical_ties():
    """Choose two eligible pairs rather than the highest individual cosine edge."""
    values = {("a", "x"): 0.95, ("a", "y"): 0.85, ("b", "x"): 0.84, ("b", "y"): 0.10}
    record = component(asymmetric("a", "b"), symmetric("x", "y"),
                       similarity=lambda a, b: values[a, b], mode="close")
    assert record["evidence"]["selected_pairs"] == [[0, 1], [1, 0]]
    assert fraction(record["contributions"]["tp"]) == 1
    tied = component(asymmetric("a", "b"), symmetric("x", "y"), similarity=lambda a, b: 0.9, mode="close")
    assert tied["evidence"]["selected_pairs"] == [[0, 0], [1, 1]]
    values[("b", "y")] = 0.99
    largest_sum = component(asymmetric("a", "b"), symmetric("x", "y"), similarity=lambda a, b: values[a, b], mode="close")
    assert largest_sum["evidence"]["selected_pairs"] == [[0, 0], [1, 1]]


@settings(max_examples=100, deadline=None)
@given(st.integers(1, 4), st.integers(1, 4), st.lists(st.sampled_from([0.1, 0.8, 0.81, 0.9, 1.0]), min_size=16, max_size=16))
def test_assignment_matches_exhaustive_oracle(gold_count, prediction_count, scores):
    """Compare the polynomial assignment with exhaustive small-graph optimization."""
    candidates = [{"gold_index": i, "prediction_index": j, "similarity": scores[i * prediction_count + j],
                   "eligible": scores[i * prediction_count + j] >= 0.8}
                  for i in range(gold_count) for j in range(prediction_count)]
    possible = []
    for cardinality in range(min(gold_count, prediction_count) + 1):
        for gold_indices in combinations(range(gold_count), cardinality):
            for prediction_indices in permutations(range(prediction_count), cardinality):
                pairs = tuple(zip(gold_indices, prediction_indices, strict=True))
                if all(scores[i * prediction_count + j] >= 0.8 for i, j in pairs):
                    score = sum((Fraction(scores[i * prediction_count + j]) for i, j in pairs), Fraction(0))
                    possible.append((-cardinality, -score, pairs))
    assert _assignment(gold_count, prediction_count, candidates) == list(min(possible)[2])


def test_constraints_prefix_normalization_fractional_fields_and_negative_pairs():
    """Preserve prefix case sensitivity, two-field credit, and negative-pair stopping."""
    gold = [{"label": " Daily   maximum ", "on": "hasProperty: Water"}]
    predicted = [{"label": "daily maximum", "on": "water"}]
    normalized = component(gold, predicted, key="hasConstraint")
    assert fraction(normalized["contributions"]["tp"]) == 1
    wrong_case = component(gold, [{"label": "daily maximum", "on": "HasProperty: water"}], key="hasConstraint")
    assert fraction(wrong_case["contributions"]["tp"]) == Fraction(1, 2)
    assert fraction(wrong_case["contributions"]["fp"]) == Fraction(1, 2)
    negative = component(gold, [{"label": "x", "on": "y"}], similarity=lambda a, b: -0.5, key="hasConstraint", mode="close")
    assert negative["evidence"]["selected_pairs"] == []
    assert negative["evidence"]["numerical_correction"] == "excess-divided-by-total"
    assert fraction(negative["contributions"]["fn"]) == Fraction(1, 2)
    assert fraction(negative["contributions"]["fp"]) == Fraction(1, 2)


def test_aggregate_micro_receipts_coverage_input_order_and_detached_evidence():
    """Sum receipts before F1, retain explicit failures, and freeze immutable results."""
    gold = blank(hasProperty="temperature", hasObjectOfInterest=symmetric("air", "water"))
    prediction = blank(hasProperty="temperature", hasObjectOfInterest="water")
    first = evaluate_item(gold, prediction, zero_similarity, metadata={"variable_id": "one"}, similarity_identity="fixture")
    second = evaluate_item(gold, blank(), zero_similarity,
                           metadata={"variable_id": "two", "terminal_invalid": True}, similarity_identity="fixture")
    original_bytes = _canonical(first)
    gold["hasObjectOfInterest"]["hasPart"].append("soil")
    assert _canonical(first) == original_bytes
    aggregate = aggregate_items([first, second], expected_variable_ids=["one", "two"])
    assert aggregate == aggregate_items([second, first], expected_variable_ids=["two", "one"])
    assert aggregate["support"] == {"items": 2, "components": 12}
    assert fraction(aggregate["exact"]["totals"]["tp"]) == Fraction(3, 2)
    assert fraction(aggregate["exact"]["totals"]["fn"]) == Fraction(5, 2)
    assert fraction(aggregate["exact"]["metrics"]["f1"]) == Fraction(6, 11)
    assert fraction(aggregate["exact"]["metrics"]["f1"]) != (fraction(first["exact"]["metrics"]["f1"]) + fraction(second["exact"]["metrics"]["f1"])) / 2
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_items([first, first])
    with pytest.raises(ValueError, match="cover"):
        aggregate_items([first], expected_variable_ids=["one", "two"])
    corrupt = deepcopy(first)
    corrupt["exact"]["totals"]["tp"]["numerator"] = 99
    with pytest.raises(ValueError, match="hash"):
        aggregate_items([corrupt])
    corrupt["result_hash"] = _hash({key: value for key, value in corrupt.items() if key != "result_hash"})
    with pytest.raises(ValueError, match="disagree"):
        aggregate_items([corrupt])
    incompatible = evaluate_item(blank(), blank(), zero_similarity, similarity_identity="different fixture")
    with pytest.raises(ValueError, match="Incompatible"):
        aggregate_items([first, incompatible])


@pytest.mark.parametrize("invalid", [
    {}, blank(extra="x"), blank(hasProperty=None), blank(hasProperty="  "),
    blank(hasObjectOfInterest={"SymmetricSystem": "x", "hasPart": ["air", "air"]}),
    blank(hasObjectOfInterest={"AsymmetricSystem": "x", "hasSource": "air"}),
    blank(hasObjectOfInterest={"AsymmetricSystem": "x", "hasSource": "air", "hasDenominator": "water"}),
    blank(hasConstraint={}), blank(hasConstraint=[{"label": "", "on": "air"}]),
    blank(hasConstraint=[{"label": "x", "on": "air", "extra": "y"}]),
])
def test_malformed_lexical_inputs_fail_before_scoring(invalid):
    """Reject schema-invalid representations at the independent evaluator boundary."""
    with pytest.raises(ValueError):
        evaluate_item(invalid, blank(), zero_similarity)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_backend_and_metadata_hash_errors(score):
    """Reject non-finite similarities and incorrect ownership hashes without fallback."""
    with pytest.raises(ValueError, match="non-finite"):
        component("air", "water", similarity=lambda a, b: score)
    with pytest.raises(ValueError, match="gold_hash"):
        evaluate_item(blank(), blank(), zero_similarity, metadata={"gold_hash": "wrong"})
