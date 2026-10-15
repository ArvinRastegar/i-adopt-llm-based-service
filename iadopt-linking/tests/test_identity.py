"""Identity classification: every IRI form the corpus and the sources actually write."""

from __future__ import annotations

import pytest

from iadopt_linking_eval import Identity, classify, is_controlled_vocabulary


@pytest.mark.parametrize(("iri", "expected"), [
    ("http://www.wikidata.org/entity/Q184395", Identity("wikidata", "Q184395", "canonical")),
    ("https://www.wikidata.org/entity/Q184395", Identity("wikidata", "Q184395", "scheme")),
    ("https://www.wikidata.org/wiki/Q19033", Identity("wikidata", "Q19033", "variant")),
    ("hhttps://www.wikidata.org/entity/Q5", Identity("wikidata", "Q5", "repaired")),
    ("https://www.wikidata.org/enity/Q104053", Identity("wikidata", "Q104053", "repaired")),
    ("http://qudt.org/vocab/quantitykind/Mass", Identity("qudt", "Mass", "canonical")),
    ("https://qudt.org/vocab/quantitykind/Mass", Identity("qudt", "Mass", "scheme")),
    ("http://www.ontology-of-units-of-measure.org/resource/om-2/AreaFraction",
     Identity("om", "AreaFraction", "canonical")),
    ("http://purl.jp/bio/4/id/200906065003062504", Identity("iobc", "200906065003062504", "opaque")),
    ("http://vocab.nerc.ac.uk/collection/S27/current/CS003625/",
     Identity("nerc", "S27/current/CS003625", "opaque")),
    ("https://vocab.nerc.ac.uk/collection/S20/current/S2000002/",
     Identity("nerc", "S20/current/S2000002", "opaque")),
    ("https://lod.nal.usda.gov/nalt/137027", Identity("nalt", "137027", "opaque")),
    ("http://vocabs.lter-europe.net/EnvThes/20813", Identity("envthes", "20813", "opaque")),
    ("https://goldbook.iupac.org/terms/view/A00086", Identity("iupac-goldbook", "A00086", "opaque")),
    ("http://purl.bioontology.org/ontology/RCD/X76JX", Identity("bioportal", "RCD/X76JX", "opaque")),
])
def test_known_forms(iri: str, expected: Identity) -> None:
    assert classify(iri) == expected


def test_scheme_variants_name_the_same_entity() -> None:
    assert (classify("http://www.wikidata.org/entity/Q5").key
            == classify("https://www.wikidata.org/entity/Q5").key
            == classify("hhttps://www.wikidata.org/entity/Q5").key)
    assert (classify("http://qudt.org/vocab/quantitykind/Count").key
            == classify("https://qudt.org/vocab/quantitykind/Count").key)


@pytest.mark.parametrize("iri", [
    "https://example.org/data/PipeConnection",
    "https://github.com/i-adopt/Corpus/blob/main/Natural%20Sciences/Oceanography/RDVA.ttl",
])
def test_placeholder_and_repository_iris_are_not_vocabularies(iri: str) -> None:
    assert not is_controlled_vocabulary(classify(iri))


def test_unrecognised_hosts_and_paths_are_labelled_not_guessed() -> None:
    assert classify("https://unknown.example.net/x").vocabulary == "unknown"
    assert classify("https://www.wikidata.org/entity/P31").vocabulary == "unknown"
    assert classify("https://qudt.org/vocab/unit/KiloGM").vocabulary == "unknown"
    assert not is_controlled_vocabulary(classify("https://unknown.example.net/x"))


def test_text_that_is_not_an_iri_is_rejected() -> None:
    with pytest.raises(ValueError, match="Not an IRI"):
        classify("Q5")
