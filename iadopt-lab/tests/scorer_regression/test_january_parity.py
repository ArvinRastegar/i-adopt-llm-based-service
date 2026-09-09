"""Execute only hash-verified January scoring AST nodes, never the legacy runner.

Provider initialization, prompts, filesystem setup, logging, and model downloads
are excluded before compilation. Cosines are deterministic offline fixtures.
"""

import __future__

import ast
import hashlib
import random
import re
from pathlib import Path

import numpy as np
import pytest

from iadopt_eval import COMPONENTS, evaluate_item

REFERENCE_COMMIT = "b9683d2242aa5ca5b987440ea4b6f70bc1253c7e"
REFERENCE_PATH = "benchmarking_example/randomShotsPhaseOne.py"
REFERENCE_SHA256 = "2d4a6271fd86a076127dfb3f85589391cff9744efdfdee8ec570b52c74921df0"


def lexical(**values):
    """Construct six-field lexical fixtures independent of the production validator.

    Args:
        values: Original component overrides.
    Returns:
        Six-field mapping with explicit empties.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {**{key: [] if key == "hasConstraint" else "" for key in COMPONENTS}, **values}


def offline_cosine(left, right, model_name=None):
    """Supply stable threshold and negative examples without an embedding model.

    Args:
        left: Normalized first string.
        right: Normalized second string.
        model_name: Ignored January compatibility argument.
    Returns:
        Fixed finite fixture cosine, not an estimate from scientific embeddings.
    Raises:
        None.
    Side Effects:
        None.
    """
    if {left, right} == {"water", "liquid"}:
        return 0.8
    if {left, right} == {"air", "atmosphere"}:
        return 0.799999999999
    if {left, right} == {"a", "b"}:
        return -0.2
    return 0.0


@pytest.fixture
def january():
    """Load isolated scorer definitions from the retained hash-verified source copy.

    Args:
        None; the reference path is inside this standalone experiment directory.
    Returns:
        Namespace of original scoring functions with a deterministic cosine fixture.
    Raises:
        AssertionError: Historical bytes do not match the accepted source hash.
        OSError: The retained historical reference file cannot be read.
        SyntaxError: Historical definitions cannot compile under the test interpreter.
    Side Effects:
        Reads one retained source file and compiles selected pure AST definitions;
        no legacy module imports, logging setup, writes, provider calls, or downloads.
    """
    project = Path(__file__).resolve().parents[2]
    source = (project / "reference/january/randomShotsPhaseOne.py.txt").read_bytes()
    assert hashlib.sha256(source).hexdigest() == REFERENCE_SHA256
    module = ast.parse(source.decode("utf-8"))
    names = {"sim_string", "_asym_parts", "sim_asym", "_sym_parts", "sim_sym",
             "canonical_on", "normalize_constraint", "sim_constraint", "confusion",
             "confusion_constraints", "prf", "compute_confusion_for_pair"}
    constants = {"_ON_PREFIX_RE", "ONTO_KEYS", "CLOSE_THR", "EMBED_MODEL_NAME"}
    selected = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets):
            selected.append(node)
    assert {node.name for node in selected if isinstance(node, ast.FunctionDef)} == names
    namespace = {"np": np, "re": re, "_cosine": offline_cosine}
    isolated = ast.Module(body=selected, type_ignores=[])
    exec(compile(isolated, "january-hash-verified-scoring-only", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


@pytest.mark.parametrize("gold,prediction", [
    ("water", " WATER "), ("water", "soil"), ("water", "liquid"),
    ("air", "atmosphere"), ("", ""), ("water", ""), ("", "water"),
    ("a", "b"), ("Daily maximum", "daily   maximum"),
])
def test_scalar_and_empty_contributions_equal_january(january, gold, prediction):
    """Compare all unchanged scalar/empty contribution floats with original code.

    Args:
        january: Isolated hash-verified historical function namespace.
        gold: Parameterized original gold text.
        prediction: Parameterized predicted text.
    Returns:
        None.
    Raises:
        AssertionError: Any original scalar contribution differs exactly.
    Side Effects:
        Calls only deterministic offline cosine fixtures.
    """
    current = evaluate_item(lexical(hasProperty=gold), lexical(hasProperty=prediction), offline_cosine,
                            similarity_identity="january-regression-offline-cosines")
    for mode, close in (("exact", False), ("close", True)):
        expected = january["confusion"](gold, prediction, close)
        actual = current[mode]["components"]["hasProperty"]["contributions"]
        assert tuple(actual[key]["value"] for key in ("tp", "fp", "fn", "tn")) == expected


@pytest.mark.parametrize("n_gold", range(8))
@pytest.mark.parametrize("n_prediction", range(8))
def test_constraints_match_original_numpy_greedy_and_float_correction(january, n_gold, n_prediction):
    """Compare real January NumPy matching and float correction over 128 mode cases.

    Args:
        january: Isolated historical function namespace with actual NumPy.
        n_gold: Number of original ordered gold Constraints.
        n_prediction: Number of predicted Constraints.
    Returns:
        None.
    Raises:
        AssertionError: Any finalized historical TP/FP/FN/TN float differs.
    Side Effects:
        Uses a locally seeded test generator only; no global random state changes.
    """
    generator = random.Random(n_gold * 100 + n_prediction)
    labels = ["a", "b", "water", "liquid", "air", "atmosphere", " daily   Maximum "]
    targets = ["a", "b", "water", "liquid", " hasProperty: Water ", "HasProperty: water"]
    gold = [{"label": generator.choice(labels), "on": generator.choice(targets)} for _ in range(n_gold)]
    prediction = [{"label": generator.choice(labels), "on": generator.choice(targets)} for _ in range(n_prediction)]
    current = evaluate_item(lexical(hasConstraint=gold), lexical(hasConstraint=prediction), offline_cosine,
                            similarity_identity="january-regression-offline-cosines")
    for mode, close in (("exact", False), ("close", True)):
        expected = january["confusion_constraints"](gold, prediction, close)
        actual = current[mode]["components"]["hasConstraint"]["contributions"]
        assert tuple(actual[key]["value"] for key in ("tp", "fp", "fn", "tn")) == expected


def test_constraint_row_major_tie_fixture_preserves_original_order_sensitivity(january):
    """Retain the historical greedy tie choice rather than silently optimizing it.

    Args:
        january: Isolated hash-verified original scoring namespace.
    Returns:
        None.
    Raises:
        AssertionError: Selected original row-major pair or contribution differs.
    Side Effects:
        None outside local fixtures.
    """
    gold = [{"label": "a", "on": "a"}, {"label": "a", "on": "b"}]
    first_prediction = [{"label": "a", "on": "c"}, {"label": "b", "on": "a"}]
    records = []
    for prediction in (first_prediction, list(reversed(first_prediction))):
        current = evaluate_item(lexical(hasConstraint=gold), lexical(hasConstraint=prediction), offline_cosine)
        record = current["exact"]["components"]["hasConstraint"]
        records.append(record)
        expected = january["confusion_constraints"](gold, prediction, False)
        assert tuple(record["contributions"][key]["value"] for key in ("tp", "fp", "fn", "tn")) == expected
        assert record["evidence"]["selected_pairs"][0] == [0, 0]
    assert records[0]["contributions"] != records[1]["contributions"]


@pytest.mark.parametrize("gold,prediction,old,new", [
    ({"SymmetricSystem": "same", "hasPart": ["water", "air"]}, "water", (0, 1, 0, 0), (0.5, 0, 0.5, 0)),
    ("water", {"SymmetricSystem": "same", "hasPart": ["water", "air"]}, (0, 1, 0, 0), (0.5, 0.5, 0, 0)),
    ({"SymmetricSystem": "same", "hasPart": ["water", "air"]},
     {"SymmetricSystem": "same", "hasPart": ["water", "soil"]}, (0, 1, 0, 0), (1 / 3, 1 / 3, 1 / 3, 0)),
    ({"SymmetricSystem": "one", "hasPart": ["water", "air"]},
     {"SymmetricSystem": "two", "hasPart": ["water", "air"]}, (0, 1, 0, 0), (1, 0, 0, 0)),
    ({"AsymmetricSystem": "one", "hasSource": "water", "hasTarget": "air"},
     {"AsymmetricSystem": "two", "hasSource": "water", "hasTarget": "soil"}, (0, 1, 0, 0), (1 / 3, 1 / 3, 1 / 3, 0)),
])
def test_approved_correction_has_explicit_before_after_receipts(january, gold, prediction, old, new):
    """Demonstrate each approved difference instead of claiming unchanged parity.

    Args:
        january: Isolated original function namespace.
        gold: Original system/scalar fixture.
        prediction: Original paired prediction.
        old: Hand-calculated historical whole-component contributions.
        new: Hand-calculated D-022 fraction display derivatives.
    Returns:
        None.
    Raises:
        AssertionError: The before/after correction boundary differs from approval.
    Side Effects:
        Calls only offline cosine fixtures for historical mixed representations.
    """
    current = evaluate_item(lexical(hasObjectOfInterest=gold), lexical(hasObjectOfInterest=prediction), offline_cosine)
    for mode, close in (("exact", False), ("close", True)):
        assert january["confusion"](gold, prediction, close) == old
        component = current[mode]["components"]["hasObjectOfInterest"]
        assert tuple(component["contributions"][key]["value"] for key in ("tp", "fp", "fn", "tn")) == new
        assert component["evidence"]["correction"] == "D-022"
