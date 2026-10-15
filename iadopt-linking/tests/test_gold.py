"""Contract tests: this project's reading of the corpus must agree with iadopt-lab's."""

from __future__ import annotations

import collections
from pathlib import Path

import pytest
from iadopt_lab.corpus.ingestion import load_canonical_records

from iadopt_linking.gold import (
    extract_links,
    lexical_mismatches,
    load_gold_links,
    resolve_lab_root,
)

PREFIXES = (
    "@prefix iop: <https://w3id.org/iadopt/ont/> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
)


@pytest.fixture(scope="module")
def lab_root() -> Path:
    """Resolve the sibling iadopt-lab project once."""
    return resolve_lab_root()


@pytest.fixture(scope="module")
def records(lab_root: Path) -> dict:
    """Index the lab's verified canonical records by source path."""
    return {record["source_path"]: record for record in load_canonical_records(lab_root)}


@pytest.fixture(scope="module")
def variables(lab_root: Path) -> tuple:
    """Load the gold links of all 102 variables once."""
    return load_gold_links(lab_root)


def test_default_lab_root_is_the_sibling_project(lab_root: Path) -> None:
    assert lab_root.name == "iadopt-lab"
    assert (lab_root / "data/manifests/corpus-v2.0.1.json").is_file()


def test_environment_variable_selects_the_lab(monkeypatch: pytest.MonkeyPatch, lab_root: Path) -> None:
    monkeypatch.setenv("IADOPT_LAB_ROOT", str(lab_root))
    assert resolve_lab_root() == lab_root


def test_a_directory_that_is_not_the_lab_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_lab_root(tmp_path)


def test_release_pinned_link_counts(variables: tuple) -> None:
    links = [link for variable in variables for link in variable.links]
    assert len(variables) == 102
    assert len(links) == 313
    assert sum(link.iri is not None for link in links) == 296
    assert sum(link.is_member for link in links) == 76
    assert collections.Counter(link.node_class for link in links) == {
        "Entity": 202, "Property": 102, "StatisticalModifier": 9}


def test_top_level_iris_equal_the_labs(variables: tuple, records: dict) -> None:
    for variable in variables:
        mine = {link.slot: link.iri for link in variable.links if not link.is_member and link.iri}
        assert mine == records[variable.source_path]["iris"], variable.source_path


def test_the_lab_view_misses_exactly_the_member_iris(variables: tuple, records: dict) -> None:
    links = [link for variable in variables for link in variable.links]
    member_iris = sum(1 for link in links if link.is_member and link.iri)
    lab_iris = sum(len(records[variable.source_path]["iris"]) for variable in variables)
    assert (member_iris, lab_iris, sum(1 for link in links if link.iri)) == (68, 228, 296)


def test_lexical_check_has_teeth(variables: tuple, records: dict) -> None:
    variable = next(v for v in variables if any(link.is_member for link in v.links))
    gold = records[variable.source_path]["gold"]
    assert lexical_mismatches(variable.links, gold) == []
    renamed = {**gold, "hasProperty": gold["hasProperty"] + " (changed)"}
    assert lexical_mismatches(variable.links, renamed)
    without_members = tuple(link for link in variable.links if not link.is_member)
    assert lexical_mismatches(without_members, gold)


def test_extract_links_rejects_a_file_without_a_variable() -> None:
    with pytest.raises(ValueError, match="Variable root"):
        extract_links(PREFIXES)


def test_extract_links_rejects_an_unlabeled_node() -> None:
    ttl = PREFIXES + (
        '<urn:v> a iop:Variable ; rdfs:label "v" ; iop:hasProperty <urn:p> .\n'
        "<urn:p> a iop:Property .\n")
    with pytest.raises(ValueError, match="rdfs:label"):
        extract_links(ttl)


def test_a_blank_node_is_a_label_without_an_iri() -> None:
    ttl = PREFIXES + (
        '<urn:v> a iop:Variable ; rdfs:label "v" ;\n'
        '  iop:hasProperty [ a iop:Property ; rdfs:label "mass" ] .\n')
    (link,) = extract_links(ttl)
    assert (link.role, link.node_class, link.term, link.iri) == ("hasProperty", "Property", "mass", None)


def test_the_lab_never_imports_this_project(lab_root: Path) -> None:
    offenders = [path for path in (lab_root / "src").rglob("*.py")
                 if "iadopt_linking" in path.read_text(encoding="utf-8")]
    assert offenders == []
