"""The evaluation population, built from the real verified gold."""

from __future__ import annotations

import collections

import pytest

from iadopt_linking.gold import load_gold_links
from iadopt_linking.population import build_population, normalise_term, query_key


@pytest.fixture(scope="module")
def population() -> tuple:
    slots, counts = build_population(load_gold_links())
    return slots, counts


def test_term_normalisation() -> None:
    assert normalise_term("  radial velocity of water current ") == "radial velocity of water current"
    assert normalise_term("uranium‑236") == "uranium-236"
    assert normalise_term("a  b\n c") == "a b c"
    assert query_key("Sea  Water") == "sea water"


def test_counts_account_for_every_linked_node(population: tuple) -> None:
    slots, counts = population
    assert counts["links"] == 313
    assert counts["slots"] == len(slots) == 294
    assert counts["excluded_label_only"] == 17
    assert counts["excluded_non_vocabulary"] == 2
    assert counts["links"] == counts["slots"] + counts["excluded_label_only"] + counts[
        "excluded_non_vocabulary"]


def test_gold_vocabularies(population: tuple) -> None:
    slots, _ = population
    assert collections.Counter(s.gold_vocabulary for s in slots) == {
        "wikidata": 218, "qudt": 61, "om": 5, "iobc": 4, "nerc": 2, "nalt": 1, "envthes": 1,
        "iupac-goldbook": 1, "bioportal": 1}


def test_gold_iris_that_needed_reading_are_counted(population: tuple) -> None:
    slots, counts = population
    assert counts["gold_form_repaired"] == 2 and counts["gold_form_variant"] == 1
    assert {s.gold_local_id for s in slots if s.gold_form == "repaired"} == {"Q5", "Q104053"}


def test_slot_ids_are_unique_and_members_are_flagged(population: tuple) -> None:
    slots, _ = population
    assert len({s.slot_id for s in slots}) == len(slots)
    assert sum(s.is_member for s in slots) == 68
    assert all(s.definition and s.term and s.query_key == query_key(s.term) for s in slots)
