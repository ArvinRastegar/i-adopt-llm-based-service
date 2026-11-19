"""Offline tests for rulebook.py: the trees, patterns and constraint types as data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rulebook

SAMPLE = """---
title: SAMPLE
---
flowchart TD
    A[Start <br> here] --> B{Are units <br> present?}:::white
    B --> |true|N[QUANTITY <br> <a href='https://x/blob/main/Decision_trees/DT2.mmd'>DT2</a><br>]
    B --> |false|C{Which <br> scale?}:::white
    C -- binary --> E[BOOLEAN <br> <a href='https://x/blob/main/pattern/VDP103.yaml'>VDP103</a><br>]
    C --ordinal-->G[ORDINAL <br> <a href='https://x/blob/main/pattern/VDP104.yaml'>VDP105</a]
    E --> G
    classDef white fill:#f3f6f4
"""
UPSTREAM = rulebook.HERE.parents[2] / "external" / "I-ADOPT-patterns-playground"


def test_parser_reads_all_three_edge_forms():
    """Pipe labels, dash labels and unlabelled arrows are all read, with markup stripped."""
    tree = rulebook.parse_tree(SAMPLE)
    assert tree["title"] == "SAMPLE" and tree["start"] == "A"
    assert tree["nodes"]["A"] == {"kind": "start", "text": "Start here"}
    assert tree["nodes"]["B"]["text"] == "Are units present?"
    assert [(e["from"], e["to"], e["label"]) for e in tree["edges"]] == [
        ("A", "B", ""), ("B", "N", "true"), ("B", "C", "false"), ("C", "E", "binary"),
        ("C", "G", "ordinal"), ("E", "G", "")]
    assert tree["nodes"]["N"]["kind"] == "jump" and tree["nodes"]["N"]["tree"] == "DT2"
    assert tree["nodes"]["E"]["pattern"] == 103 and tree["nodes"]["E"]["text"] == "BOOLEAN"


def test_parser_prefers_the_label_and_reports_the_slip():
    """A leaf labelled VDP105 that links VDP104.yaml is pattern 105, and the slip is listed."""
    tree = rulebook.parse_tree(SAMPLE)
    assert tree["nodes"]["G"]["pattern"] == 105
    assert rulebook.tree_slips("S", tree) == ["S node G: labelled VDP105, links VDP104.yaml"]


def test_nine_trees_with_the_expected_questions(trees):
    """Every tree is present with the number of questions the flowcharts show."""
    counts = {name: sum(n["kind"] == "question" for n in tree["nodes"].values())
              for name, tree in trees.items()}
    assert counts == {"DT0": 5, "DT1": 5, "DT2": 6, "DT3": 10, "DT4": 5, "DT5": 6, "DT6": 6,
                      "DT7": 0, "DT8": 5}


def test_trees_are_well_formed(trees, patterns):
    """Edges join known nodes, questions have answers, and every leaf pattern is in the table."""
    for name, tree in trees.items():
        for edge in tree["edges"]:
            assert edge["from"] in tree["nodes"] and edge["to"] in tree["nodes"], name
        for ident, node in tree["nodes"].items():
            if node["kind"] == "question":
                labels = [e["label"] for e in rulebook.options(tree, ident)]
                assert labels and all(labels) and len(set(labels)) == len(labels), (name, ident)
            if node["kind"] == "pattern":
                assert str(node["pattern"]) in patterns, (name, ident)
            if node["kind"] == "jump":
                assert node["tree"] in trees, (name, ident)
            assert "link" not in node


def test_route_to_areal_density_matches_the_authors_guide(trees):
    """The guide's demo route (ozone column) is the one route to pattern 127."""
    routes = rulebook.routes_to(trees, 127)
    assert len(routes) == 1
    assert [(s["tree"], s["answer"]) for s in routes[0]] == [
        ("DT0", "true"), ("DT2", "false"), ("DT2", "false"), ("DT3", "false"), ("DT3", "false"),
        ("DT3", "true"), ("DT3", "false"), ("DT3", "true"), ("DT5", "Area"), ("DT5", "Entity")]


def test_every_property_pattern_but_two_is_reachable(trees, patterns):
    """Speed (119) and acceleration (141) are in no tree; every other property pattern is."""
    unreachable = sorted(int(n) for n, p in patterns.items()
                         if p["kind"] == "property" and not rulebook.routes_to(trees, int(n)))
    assert unreachable == [119, 141]


def test_our_correction_is_applied_and_recorded(trees):
    """DT3's mislabelled leaf is pattern 114, and the correction is stored with its reason."""
    assert trees["DT3"]["nodes"]["F"] == {"kind": "pattern", "pattern": 114, "text": "RATIOLESS DERIVED"}
    stored = rulebook.load("trees")
    assert [(c["tree"], c["node"]) for c in stored["corrections"]] == [("DT3", "F")]
    assert len(stored["slips"]) == 4
    assert all(e["label"] != "false:VolumicDensity" for e in trees["DT5"]["edges"])


def test_patterns_carry_shapes_and_no_examples(patterns):
    """Shapes are normalised, and no example or issue link survives the build."""
    assert patterns["110"]["object_shape"] == "ratio" and patterns["104"]["object_shape"] == "single"
    assert patterns["130"]["matrix_shape"] == "source_target"
    assert patterns["35"]["shape"] == "ratio" and patterns["45"]["shape"] == "source_target"
    assert patterns["44"]["kind"] == "matrix" and patterns["103"]["name"] == "boolean"
    assert patterns["124"]["required_constraint"].lower() == "normalization"
    text = (rulebook.DATA / "patterns.json").read_text(encoding="utf-8")
    assert '"example"' not in text and "gh_issue" not in text and "/issues/" not in text
    assert "BMI" not in text


def test_constraint_types_table():
    """Forty-two types in six groups, each with a definition."""
    types = rulebook.load("constraint_types")["types"]
    assert len(types) == 42 and len({t["category"] for t in types}) == 6
    assert all(t["name"] and t["definition"] for t in types)
    assert next(t for t in types if t["number"] == 74)["targets"] == [
        "property", "object of interest", "matrix", "statistical modifier"]


def test_data_files_match_the_pinned_upstream():
    """A rebuild from the local clone gives the stored bytes; skipped when the clone is absent."""
    stored = json.loads((rulebook.DATA / "trees.json").read_text(encoding="utf-8"))["source"]["commit"]
    if not (UPSTREAM / ".git").exists():
        pytest.skip("no local clone of the upstream patterns repository")
    import subprocess
    head = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if head != stored:
        pytest.skip("the local clone is not at the pinned commit")
    for name, text in rulebook.build(Path(UPSTREAM)).items():
        assert (rulebook.DATA / name).read_text(encoding="utf-8") == text, name
