"""Gold-independent lexical shape, target and canonicalization regressions."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from iadopt_lab.validation import (
    canonicalize_prediction,
    empty_prediction,
    load_schema_bytes,
    validate_prediction,
)


def test_schema_meta_and_empty_prediction():
    """Validate shared schema and valid empty outputs.

    Args: none. Returns: None after assertions.
    Raises: AssertionError on protocol regression. Side effects: reads schema only.
    """
    Draft202012Validator.check_schema(json.loads(load_schema_bytes()))
    result = validate_prediction(empty_prediction())
    assert result.valid and result.canonical_prediction == empty_prediction()
    assert result.errors == ()


@pytest.mark.parametrize("field,value", [
    ("hasProperty", None), ("hasMatrix", []), ("hasConstraint", ""),
    ("hasObjectOfInterest", 1), ("hasStatisticalModifier", False),
])
def test_invalid_field_types(field, value):
    """Reject incompatible scalar/entity/array types.

    Args: invalid field/value case. Returns: None.
    Raises: AssertionError on acceptance or missing path evidence. Side effects: schema read.
    """
    candidate = empty_prediction()
    candidate[field] = value
    result = validate_prediction(candidate)
    assert not result.valid
    assert result.errors[0]["pointer"] == "/" + field


@pytest.mark.parametrize("system", [
    {"SymmetricSystem": "", "hasPart": ["water", "air"]},
    {"AsymmetricSystem": "", "hasSource": "water", "hasTarget": "air"},
    {"AsymmetricSystem": "", "hasNumerator": "water", "hasDenominator": "air"},
])
@pytest.mark.parametrize("field", ["hasObjectOfInterest", "hasMatrix", "hasContextObject"])
def test_all_system_alternatives(field, system):
    """Accept every supported system alternative in every entity-bearing position.

    Args: field and complete system fixture. Returns: None.
    Raises: AssertionError on rejection or scored-role mutation. Side effects: schema read.
    """
    candidate = empty_prediction()
    candidate[field] = system
    result = validate_prediction(candidate)
    assert result.valid
    assert result.canonical_prediction[field] != ""
    assert candidate[field] == system


@pytest.mark.parametrize("system", [
    {"SymmetricSystem": "", "hasPart": ["water"]},
    {"SymmetricSystem": "", "hasPart": ["water", "water"]},
    {"SymmetricSystem": "", "hasPart": ["Water", " water "]},
    {"SymmetricSystem": "", "hasPart": ["water", " "]},
    {"AsymmetricSystem": "", "hasSource": "water"},
    {"AsymmetricSystem": "", "hasSource": "water", "hasTarget": "air", "hasNumerator": "water", "hasDenominator": "air"},
    {"AsymmetricSystem": "", "hasNumerator": "water", "hasDenominator": "air", "unknown": "x"},
])
def test_invalid_systems(system):
    """Reject ambiguous, duplicate, incomplete and mixed systems.

    Args: invalid system fixture. Returns: None.
    Raises: AssertionError if a malformed system is accepted. Side effects: schema read.
    """
    candidate = empty_prediction()
    candidate["hasObjectOfInterest"] = system
    assert not validate_prediction(candidate).valid


def test_required_unknown_and_nonfinite():
    """Reject absent/unknown fields and non-finite JSON values without parser exceptions.

    Args: none. Returns: None.
    Raises: AssertionError on validation regression. Side effects: schema reads only.
    """
    missing = empty_prediction()
    del missing["hasMatrix"]
    assert not validate_prediction(missing).valid
    unknown = {**empty_prediction(), "extra": ""}
    assert not validate_prediction(unknown).valid
    assert validate_prediction(float("nan")).errors[0]["code"] == "not_json_value"


def test_targets_aliases_and_preserved_order():
    """Resolve whole-system aliases while retaining raw text, scored members and Constraint order.

    Args: none. Returns: None.
    Raises: AssertionError for mutation or incorrect canonicalization. Side effects: schema read.
    """
    candidate = empty_prediction()
    candidate["hasProperty"] = "temperature"
    candidate["hasObjectOfInterest"] = {"SymmetricSystem": "custom", "hasPart": ["water", "Air"]}
    candidate["hasConstraint"] = [{"label": "z:last", "on": "custom"}, {"label": "a:first", "on": " WATER "}]
    untouched = copy.deepcopy(candidate)
    result = validate_prediction(candidate)
    assert result.valid and candidate == untouched
    canonical = result.canonical_prediction
    assert canonical["hasObjectOfInterest"] == {"SymmetricSystem": "Air + water", "hasPart": ["Air", "water"]}
    assert canonical["hasConstraint"] == [{"label": "z:last", "on": "Air + water"}, {"label": "a:first", "on": "water"}]


@pytest.mark.parametrize("target,code", [("unknown", "unknown_constraint_target"), ("shared", "ambiguous_constraint_target")])
def test_bad_constraint_targets(target, code):
    """Reject missing targets and aliases competing with another emitted component.

    Args: target and expected stable error code. Returns: None.
    Raises: AssertionError on incorrect diagnostics. Side effects: schema read.
    """
    candidate = empty_prediction()
    candidate["hasProperty"] = "shared"
    candidate["hasObjectOfInterest"] = {"SymmetricSystem": "shared", "hasPart": ["water", "air"]}
    candidate["hasConstraint"] = [{"label": "wet", "on": target}]
    result = validate_prediction(candidate)
    assert not result.valid
    assert result.errors[0]["code"] == code
    assert result.errors[0]["pointer"] == "/hasConstraint/0/on"


def test_same_lexical_label_in_multiple_roles():
    """Allow indistinguishable repeated lexical labels without inventing role identity.

    Args: none. Returns: None.
    Raises: AssertionError on overstrict target resolution. Side effects: schema read.
    """
    candidate = empty_prediction()
    candidate.update(hasProperty="water", hasObjectOfInterest="water", hasConstraint=[{"label": "wet", "on": "water"}])
    assert validate_prediction(candidate).valid
    with pytest.raises(ValueError):
        canonicalize_prediction({})


def test_readable_prediction_attaches_corpus_values_without_touching_the_score():
    """Prove D-031 attaches label/definition for readers only, never for scoring.

    Args: none. Returns: None.
    Raises: AssertionError if the scored six-field record or its hash changes.
    Side effects: none.
    """
    import copy as _copy

    from iadopt_eval import evaluate_item
    from iadopt_lab.canonical import content_hash
    from iadopt_lab.validation import empty_prediction, readable_prediction

    prediction = empty_prediction()
    prediction["hasProperty"] = "temperature"
    prediction["hasObjectOfInterest"] = "air"
    original = _copy.deepcopy(prediction)
    before = content_hash(prediction)

    readable = readable_prediction(prediction, label="Air temperature",
                                   definition="Thermodynamic temperature of the air.")

    assert readable["label"] == "Air temperature"
    assert readable["definition"] == "Thermodynamic temperature of the air."
    assert list(readable)[:2] == ["label", "definition"]
    # The scored record is untouched, so its identity cannot drift.
    assert prediction == original and content_hash(prediction) == before
    # Mutating the readable view cannot reach the scored prediction.
    readable["hasProperty"] = "mutated"
    assert prediction["hasProperty"] == "temperature"

    # The evaluator still refuses anything that is not exactly the six fields.
    with pytest.raises(ValueError):
        evaluate_item(prediction, readable, similarity=lambda left, right: 0.0)
    with pytest.raises(ValueError):
        readable_prediction(readable, label="x", definition="y")
