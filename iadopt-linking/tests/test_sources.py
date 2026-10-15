"""Source request builders and strict response parsers, against real captured responses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iadopt_linking.sources import (
    OM_QUANTITY,
    RESULT_LIMIT,
    SOURCES,
    QudtSource,
    SourceError,
    WikidataSource,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_wikidata_request_asks_for_ten_english_items() -> None:
    request = WikidataSource().request("blood")
    assert request.params["search"] == "blood"
    assert request.params["limit"] == str(RESULT_LIMIT) == "10"
    assert (request.params["language"], request.params["type"]) == ("en", "item")


def test_wikidata_parse_keeps_the_api_order_and_the_concept_uri() -> None:
    candidates = WikidataSource().parse(fixture("wikidata_blood.json"))
    assert [c.rank for c in candidates] == [1, 2, 3, 4]
    assert candidates[0].iri == "http://www.wikidata.org/entity/Q16860120"
    assert (candidates[0].label, candidates[0].description) == ("Blood", "family name")
    assert all(c.source == "wikidata" for c in candidates)


@pytest.mark.parametrize("body", [
    "not json",
    json.dumps({"error": {"code": "internal_api_error", "info": "x"}}),
    json.dumps({"searchinfo": {}}),
    json.dumps({"search": [{"id": "P31", "label": "instance of"}]}),
    json.dumps({"search": [{"id": f"Q{i}"} for i in range(RESULT_LIMIT + 1)]}),
])
def test_wikidata_parse_refuses_what_is_not_a_result(body: str) -> None:
    with pytest.raises(SourceError):
        WikidataSource().parse(body)


def test_qudt_query_requires_every_word_and_drops_function_words() -> None:
    request = QudtSource().request("amount of substance concentration")
    query = request.params["query"]
    for word in ("amount", "substance", "concentration"):
        assert f'CONTAINS(LCASE(STR(?l)), "{word}")' in query
    assert 'CONTAINS(LCASE(STR(?l)), "of")' not in query
    assert f"LIMIT {RESULT_LIMIT}" in query
    assert "qudt:deprecated true" in query and 'STRSTARTS(LANG(?l), "en")' in query
    assert QudtSource().request("of the") is None


def test_qudt_words_cannot_break_out_of_the_query() -> None:
    request = QudtSource().request('mass" ) } DROP ALL { "')
    assert request is not None
    filters = [line.strip() for line in request.params["query"].splitlines() if "CONTAINS" in line]
    assert filters == [f'FILTER(CONTAINS(LCASE(STR(?l)), "{word}"))'
                       for word in ("mass", "drop", "all")]


def test_qudt_parse_groups_rows_and_orders_by_label_length() -> None:
    candidates = QudtSource().parse(fixture("qudt_mass_concentration.json"))
    assert [c.rank for c in candidates] == list(range(1, len(candidates) + 1))
    assert candidates[0].iri == "http://qudt.org/vocab/quantitykind/MassConcentration"
    assert candidates[0].label == "Mass Concentration"
    assert "Q589446" in candidates[0].extras["wikidata_match"][0]
    assert len({c.iri for c in candidates}) == len(candidates)
    lengths = [len(c.label) for c in candidates]
    assert lengths == sorted(lengths)


def test_qudt_prefers_english_but_accepts_en_us_only_labels() -> None:
    (candidate,) = QudtSource().parse(fixture("qudt_mass_flux_density.json"))
    assert candidate.label == "Mass Flux Density"
    assert candidate.iri.endswith("/MassFluxDensity")


def test_qudt_parse_refuses_malformed_and_oversized_results() -> None:
    with pytest.raises(SourceError):
        QudtSource().parse(json.dumps({"boolean": True}))
    rows = [{"qk": {"value": f"http://qudt.org/vocab/quantitykind/K{i}"}, "len": {"value": "1"},
             "label": {"value": f"k{i}"}} for i in range(RESULT_LIMIT + 1)]
    with pytest.raises(SourceError):
        QudtSource().parse(json.dumps({"results": {"bindings": rows}}))


def test_om_request_restricts_to_quantities_and_ten_rows() -> None:
    request = SOURCES["om"].request("area fraction")
    assert request.params["allChildrenOf"] == OM_QUANTITY
    assert (request.params["ontology"], request.params["rows"]) == ("om", "10")


def test_om_parse_reads_labels_and_iris() -> None:
    candidates = SOURCES["om"].parse(fixture("om_area_fraction.json"))
    assert candidates[0].iri == "http://www.ontology-of-units-of-measure.org/resource/om-2/AreaFraction"
    assert candidates[0].label == "area fraction"
    assert candidates[0].description == ""


def test_om_parse_refuses_a_foreign_ontology() -> None:
    body = json.dumps({"response": {"docs": [{"iri": "http://x/y", "ontology_name": "qudt"}]}})
    with pytest.raises(SourceError):
        SOURCES["om"].parse(body)


def om_body(docs: list, found: int | None = None, status: int = 0) -> str:
    return json.dumps({"responseHeader": {"status": status},
                       "response": {"numFound": len(docs) if found is None else found, "docs": docs}})


def om_doc(**overrides: object) -> dict:
    return {"iri": "http://x/y", "ontology_name": "om", "label": "area", "description": [], **overrides}


@pytest.mark.parametrize("body", [
    json.dumps({"search": [], "success": 0}),
    json.dumps({"search": []}),
    json.dumps({"search": "none", "success": 1}),
    json.dumps({"search": ["Q1"], "success": 1}),
    json.dumps({"search": [{"id": "Q1", "label": 5}], "success": 1}),
    json.dumps({"search": [{"id": "Q1", "label": "a", "description": ["x"]}], "success": 1}),
    json.dumps([1, 2]),
])
def test_wikidata_parse_turns_every_wrong_shape_into_a_source_error(body: str) -> None:
    with pytest.raises(SourceError):
        WikidataSource().parse(body)


@pytest.mark.parametrize("body", [
    json.dumps({"results": {"bindings": "x"}}),
    json.dumps({"results": {"bindings": [{"len": {"value": "1"}}]}}),
    json.dumps({"results": {"bindings": [{"qk": {"value": "http://q/K"}, "len": {"value": "long"},
                                          "label": {"value": "k"}}]}}),
    json.dumps({"results": {"bindings": [5]}}),
])
def test_qudt_parse_turns_every_wrong_shape_into_a_source_error(body: str) -> None:
    with pytest.raises(SourceError):
        QudtSource().parse(body)


@pytest.mark.parametrize("body", [
    om_body([om_doc()], status=1),
    om_body([om_doc()], found=5),
    om_body([om_doc(), om_doc()], found=1),
    om_body([om_doc(label=None)]),
    om_body([om_doc(label="")]),
    om_body([om_doc(description="text")]),
    om_body([om_doc(description=[3])]),
    om_body([om_doc(iri="")]),
    om_body(["x"]),
    om_body([om_doc() for _ in range(RESULT_LIMIT + 1)]),
    json.dumps({"response": {"docs": [], "numFound": 0}}),
    "not json",
])
def test_om_parse_turns_every_wrong_shape_into_a_source_error(body: str) -> None:
    with pytest.raises(SourceError):
        SOURCES["om"].parse(body)


def test_om_numfound_may_exceed_the_ten_documents_returned() -> None:
    docs = [om_doc(iri=f"http://x/{i}") for i in range(RESULT_LIMIT)]
    assert len(SOURCES["om"].parse(om_body(docs, found=250))) == RESULT_LIMIT


def test_om_keeps_the_first_non_empty_description() -> None:
    (candidate,) = SOURCES["om"].parse(om_body([om_doc(description=["", "  a  \n b  "])]))
    assert candidate.description == "a b"


def test_a_source_error_keeps_its_own_message() -> None:
    with pytest.raises(SourceError, match="not an item id"):
        WikidataSource().parse(json.dumps({"search": [{"id": "P31"}], "success": 1}))
