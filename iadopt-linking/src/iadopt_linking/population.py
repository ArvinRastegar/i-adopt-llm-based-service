"""The evaluation population: gold slots whose IRI names a controlled-vocabulary term.

A slot is one linked node of one variable. Slots without an IRI (label only) and slots whose
IRI is not a vocabulary (placeholder, repository) have no gold identifier to score and are
counted, not silently dropped.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from iadopt_linking_eval import classify, is_controlled_vocabulary

from .gold import GoldVariable

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


@dataclass(frozen=True)
class Slot:
    """One gold link to score, with the context a linker is given."""

    slot_id: str
    variable_id: str
    source_path: str
    variable_label: str
    definition: str
    role: str
    slot: str
    is_member: bool
    node_class: str
    term: str
    query_key: str
    gold_iri: str
    gold_vocabulary: str
    gold_local_id: str
    gold_form: str


def normalise_term(term: str) -> str:
    """Make a corpus term safe to send to a search service.

    Args: term: slot text as the corpus writes it.
    Returns: NFKC text with dash variants unified, whitespace collapsed and ends stripped.
    Raises: None.
    Side effects: none; case is preserved.
    """
    text = unicodedata.normalize("NFKC", term).translate(_DASHES)
    return re.sub(r"\s+", " ", text).strip()


def query_key(term: str) -> str:
    """Name the retrieval query a term shares with every identical term.

    Args: term: slot text as the corpus writes it.
    Returns: the normalised term in lower case.
    Raises: None.
    Side effects: none.
    """
    return normalise_term(term).lower()


def build_population(variables: Iterable[GoldVariable]) -> tuple[tuple[Slot, ...], dict[str, int]]:
    """Select the slots whose gold is a controlled-vocabulary IRI.

    Args: variables: gold variables from iadopt_linking.gold.load_gold_links.
    Returns: the slots in corpus order and a count of every link seen, kept, and left out by
    reason (label only, non-vocabulary IRI), plus how many gold IRIs needed repair or a variant.
    Raises: ValueError for an IRI that is not an IRI at all.
    Side effects: none; deterministic.
    """
    slots: list[Slot] = []
    counts: Counter[str] = Counter()
    for variable in variables:
        ordinals: Counter[str] = Counter()
        for link in variable.links:
            counts["links"] += 1
            ordinal = ordinals[link.role]
            ordinals[link.role] += 1
            if link.iri is None:
                counts["excluded_label_only"] += 1
                continue
            identity = classify(link.iri)
            if not is_controlled_vocabulary(identity):
                counts["excluded_non_vocabulary"] += 1
                continue
            counts["slots"] += 1
            counts[f"gold_form_{identity.form}"] += 1
            slots.append(Slot(
                slot_id=f"{variable.source_path}#{link.role}#{ordinal}",
                variable_id=variable.variable_id, source_path=variable.source_path,
                variable_label=variable.label, definition=variable.definition.strip(),
                role=link.role, slot=link.slot, is_member=link.is_member,
                node_class=link.node_class, term=link.term, query_key=query_key(link.term),
                gold_iri=link.iri, gold_vocabulary=identity.vocabulary,
                gold_local_id=identity.local_id, gold_form=identity.form))
    return tuple(slots), dict(sorted(counts.items()))
