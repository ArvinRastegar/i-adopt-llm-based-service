"""Pooling and scoring: fixed source order, strict checks, resume, replay of identical queries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from iadopt_linking.population import Slot
from iadopt_linking.recorder import JsonlStore
from iadopt_linking.score import PoolError, pool_candidates, score_all, score_slot
from iadopt_linking.sources import Candidate

SOURCES = ("wikidata", "qudt", "om")


def slot(slot_id: str = "f.ttl#hasProperty#0", term: str = "Blood", variable: str = "urn:v1") -> Slot:
    return Slot(slot_id, variable, "A/f.ttl", "Var", "A fluid definition.", "hasProperty",
                "hasProperty", False, "Property", term, term.lower(), "https://www.wikidata.org/entity/Q7873",
                "wikidata", "Q7873", "scheme")


def retrieval(source: str, key: str, candidates: list[Candidate]) -> dict[str, Any]:
    return {"source": source, "query_key": key, "error": None, "parse_error": None,
            "candidates": [asdict(c) for c in candidates]}


WD = [Candidate("wikidata", 1, "http://www.wikidata.org/entity/Q1", "Blood", "family name", {}),
      Candidate("wikidata", 2, "http://www.wikidata.org/entity/Q7873", "blood", "organic fluid " + "x" * 400, {})]
QK = [Candidate("qudt", 1, "http://qudt.org/vocab/quantitykind/Mass", "Mass", "", {})]


def retrieved_for(key: str = "blood", wd=WD, qk=QK, om=()) -> dict[tuple[str, str], dict[str, Any]]:
    return {("wikidata", key): retrieval("wikidata", key, list(wd)),
            ("qudt", key): retrieval("qudt", key, list(qk)),
            ("om", key): retrieval("om", key, list(om))}


class Fake:
    """Scores 0.9 for the document that mentions an organic fluid, 0.1 otherwise."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls += 1
        return [0.9 if "organic fluid" in d else 0.1 for d in documents]


def test_pool_follows_the_fixed_source_order_and_classifies_identities() -> None:
    pool = pool_candidates(slot(), retrieved_for(), SOURCES)
    assert [(c["source"], c["rank"], c["local_id"]) for c in pool] == [
        ("wikidata", 1, "Q1"), ("wikidata", 2, "Q7873"), ("qudt", 1, "Mass")]
    assert pool[0]["vocabulary"] == "wikidata"


def test_a_missing_or_failed_retrieval_stops_the_pool() -> None:
    retrieved = retrieved_for()
    del retrieved[("om", "blood")]
    with pytest.raises(PoolError, match="om"):
        pool_candidates(slot(), retrieved, SOURCES)
    retrieved = retrieved_for()
    retrieved[("qudt", "blood")]["error"] = "HTTP 503"
    with pytest.raises(PoolError, match="qudt"):
        pool_candidates(slot(), retrieved, SOURCES)


def test_a_candidate_from_the_wrong_vocabulary_is_an_error() -> None:
    odd = [Candidate("qudt", 1, "http://www.wikidata.org/entity/Q1", "x", "", {})]
    with pytest.raises(PoolError, match="wikidata IRI"):
        pool_candidates(slot(), retrieved_for(qk=odd), SOURCES)


def test_more_than_ten_candidates_from_one_source_is_an_error() -> None:
    many = [Candidate("wikidata", i, f"http://www.wikidata.org/entity/Q{i}", "x", "", {})
            for i in range(1, 12)]
    with pytest.raises(PoolError, match="more than 10"):
        pool_candidates(slot(), retrieved_for(wd=many), SOURCES)


def test_documents_use_the_description_cap_and_scores_attach_in_order() -> None:
    pool = pool_candidates(slot(), retrieved_for(), SOURCES)
    record = score_slot(slot(), pool, Fake(), 50)
    assert record["query"] == 'Definition of "Blood" in context: "A fluid definition."'
    assert record["pool"][1]["document"].startswith('label: "blood", description: "organic fluid')
    assert len(record["pool"][1]["document"]) < 100 and record["pool"][1]["document"].endswith('…"')
    assert [c["score"] for c in record["pool"]] == [0.1, 0.9, 0.1]
    assert record["gold"] == {"iri": "https://www.wikidata.org/entity/Q7873", "vocabulary": "wikidata",
                              "local_id": "Q7873", "form": "scheme"}


def test_scoring_resumes_and_reuses_an_identical_query(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "scores.jsonl")
    twins = [slot("f.ttl#hasProperty#0"), slot("f.ttl#hasObjectOfInterest#0"),
             slot("g.ttl#hasProperty#0", variable="urn:v2")]
    scorer = Fake()
    stats = score_all(twins, retrieved_for(), scorer, store, sources=SOURCES, max_description_chars=300)
    assert dict(stats) == {"scored": 2, "reused": 1} and scorer.calls == 2
    assert len(store.read()) == 3
    again = Fake()
    stats = score_all(twins, retrieved_for(), again, store, sources=SOURCES, max_description_chars=300)
    assert dict(stats) == {"skipped": 3} and again.calls == 0 and len(store.read()) == 3


def test_a_twin_whose_query_differs_only_in_letter_case_is_scored_not_replayed(tmp_path: Path) -> None:
    class CaseSensitive:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            self.queries.append(query)
            return [0.9 if "Water" in query else 0.1 for _ in documents]

    store = JsonlStore(tmp_path / "scores.jsonl")
    scorer = CaseSensitive()
    pair = [slot("f.ttl#hasProperty#0", term="Water"), slot("f.ttl#hasObjectOfInterest#0", term="water")]
    stats = score_all(pair, {**retrieved_for("water")}, scorer, store, sources=SOURCES,
                      max_description_chars=300)
    assert dict(stats) == {"scored": 2} and len(scorer.queries) == 2
    upper, lower = store.read()
    assert 'Definition of "Water"' in upper["query"] and 'Definition of "water"' in lower["query"]
    assert {c["score"] for c in upper["pool"]} == {0.9} and {c["score"] for c in lower["pool"]} == {0.1}


def test_an_empty_pool_is_recorded_not_skipped(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "scores.jsonl")
    stats = score_all([slot()], retrieved_for(wd=[], qk=[]), Fake(), store, sources=SOURCES,
                      max_description_chars=300)
    (record,) = store.read()
    assert record["pool"] == [] and stats["empty_pool"] == 1


def test_a_scorer_that_returns_the_wrong_count_is_refused(tmp_path: Path) -> None:
    class Short:
        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            return [0.5]

    with pytest.raises(ValueError, match="scores for"):
        score_all([slot()], retrieved_for(), Short(), JsonlStore(tmp_path / "s.jsonl"),
                  sources=SOURCES, max_description_chars=300)
    assert not (tmp_path / "s.jsonl").exists()
