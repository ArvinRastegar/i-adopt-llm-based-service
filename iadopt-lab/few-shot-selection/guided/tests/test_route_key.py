"""Offline tests for route_key.py: the expert answer key and its audit."""

from __future__ import annotations

import route_key
import rulebook

GOLD = {"hasStatisticalModifier": "maximum", "hasProperty": "mass flux density",
        "hasObjectOfInterest": {"AsymmetricSystem": "a / b", "hasNumerator": "a", "hasDenominator": "b"},
        "hasMatrix": {"AsymmetricSystem": "", "hasSource": "air", "hasTarget": "ground"},
        "hasContextObject": "",
        "hasConstraint": [{"label": "state: dissolved", "on": "a"},
                          {"label": "reporting frequency: daily", "on": "maximum"}]}


def test_labels_are_sorted_into_parts():
    """Numbered tags go to their part; plain tags become flags or the modifier switch."""
    parsed = route_key.parse_labels(["132 property: index", "33 object of interest: single",
                                     "44 matrix: reference area", "72 constraint: cause", "corpus",
                                     "statistical modifier", "to be agreed on", "50 context: environment"])
    assert parsed["property"] == [132] and parsed["object"] == [33] and parsed["matrix"] == [44]
    assert parsed["constraint"] == [72] and parsed["context"] == [50]
    assert parsed["statistical_modifier"] is True and parsed["flags"] == ["to be agreed on"]
    assert parsed["names"]["132"] == "index" and parsed["other"] == []


def test_corpus_link_is_read_in_both_forms():
    """Entries link the corpus through the published site or through GitHub."""
    site = "see https://i-adopt.github.io/Corpus/Social%20Sciences/Demography/X.ttl here"
    hub = "[x](https://github.com/i-adopt/Corpus/blob/main/Social%20Sciences/Demography/X.ttl)"
    assert route_key.corpus_path(site) == route_key.corpus_path(hub) == "Social Sciences/Demography/X.ttl"
    assert route_key.corpus_path("no link") is None and route_key.corpus_path(None) is None


def test_key_covers_every_corpus_variable_once(records, key, trees):
    """All 102 variables have one entry with exactly one reachable property pattern."""
    assert set(key) == {r["variable_id"] for r in records} and len(key) == 102
    assert len({entry["issue"] for entry in key.values()}) == 102
    for entry in key.values():
        assert len(entry["property"]) == 1
        assert rulebook.routes_to(trees, entry["property"][0])


def test_shapes_names_and_targets():
    """Shapes are named, and a constraint target is traced to the part that carries it."""
    assert route_key.shape("") == "empty" and route_key.shape("soil") == "single"
    assert route_key.shape(GOLD["hasObjectOfInterest"]) == "ratio"
    assert route_key.shape(GOLD["hasMatrix"]) == "source_target"
    assert route_key.shape({"SymmetricSystem": "", "hasPart": ["x", "y"]}) == "symmetric"
    assert route_key.names_of(GOLD["hasMatrix"]) == {"air", "ground", "air → ground"}
    assert route_key.target_part(GOLD, "a") == "object of interest"
    assert route_key.target_part(GOLD, "maximum") == "statistical modifier"
    assert route_key.target_part(GOLD, "mass flux density") == "property"
    assert route_key.target_part(GOLD, "nothing") == "unknown"


def test_audit_reports_each_kind_of_disagreement(patterns, trees):
    """A fraction tagged on a non-ratio gold, a missing matrix and an unknown type are all found."""
    types = rulebook.load("constraint_types")["types"]
    gold = dict(GOLD, hasObjectOfInterest="a", hasMatrix="",
                hasConstraint=[{"label": "colour: red", "on": "a"}])
    entry = route_key.parse_labels(["110 property: fraction", "35 object of interest: ratio",
                                    "41 matrix: natural container", "76 constraint: state"])
    categories = {c for c, _ in route_key.audit_variable(gold, entry, patterns, types, trees)}
    assert {"object shape against the property pattern", "object shape against the object tag",
            "matrix present or absent", "constraint type tagged but not in gold",
            "constraint type in gold but not tagged", "gold constraint type not in the table",
            "statistical modifier present or absent"} <= categories


def test_tags_fix_the_form_of_every_gold_answer(records, key):
    """Across the corpus the tags never disagree with the gold on object, matrix or context form."""
    result = route_key.audit(records, key)
    assert result["variables"] == 102
    for category in ("object shape against the object tag", "matrix present or absent",
                     "matrix shape against the matrix tag", "context object present or absent",
                     "property tag count", "property pattern not in any tree"):
        assert category not in result["categories"], category
    assert result["categories"]["statistical modifier present or absent"] == 1
