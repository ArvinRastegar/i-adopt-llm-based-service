"""Pool the candidates of all sources for each slot and score them with one reranker.

A scorer sees only recorded candidates, so it can never reach a retrieval service, and the same
pool is available to every scorer that is compared on it later. Every scored record carries the
configuration that produced it, so a folder can never silently mix description caps or models.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from iadopt_linking_eval import classify

from .population import Slot, normalise_term
from .recorder import JsonlStore, latest
from .rerank import Scorer, build_document, build_query, validate_scores
from .retrieve import is_ok
from .sources import RESULT_LIMIT

PROVENANCE_FIELDS = ("model", "revision", "device", "batch_size")


class PoolError(ValueError):
    """A slot's retrieved candidates are missing or inconsistent with their source."""


def scoring_provenance(scorer: Scorer, max_description_chars: int | None) -> dict[str, Any]:
    """Describe what produced a score, to be stored in every record it scores.

    Args: scorer: the reranker; max_description_chars: description cap in force.
    Returns: model, revision, device and batch size when the scorer reports them, plus the cap.
    Raises: None.
    Side effects: may load the model, because describing it needs its revision.
    """
    describe = getattr(scorer, "describe", None)
    described = describe() if describe is not None else {}
    return {**{k: described.get(k) for k in PROVENANCE_FIELDS},
            "max_description_chars": max_description_chars}


def pool_candidates(slot: Slot, retrieved: Mapping[tuple[str, str], Mapping[str, Any]],
                    sources: Sequence[str]) -> list[dict[str, Any]]:
    """Gather one slot's candidates from every source in a fixed source order.

    Args: slot: the slot; retrieved: (source, query key) -> successful retrieval record;
    sources: source names in pooling order.
    Returns: candidates with source, rank, IRI, identity, label, description and extras.
    Raises: PoolError for a missing retrieval, a candidate whose IRI belongs to another
    vocabulary than its source, or more than RESULT_LIMIT candidates from a source.
    Side effects: none.
    """
    pool: list[dict[str, Any]] = []
    for name in sources:
        record = retrieved.get((name, slot.query_key))
        if not is_ok(record):
            raise PoolError(f"No successful {name} retrieval for {slot.query_key!r}")
        if len(record["candidates"]) > RESULT_LIMIT:
            raise PoolError(f"{name} returned more than {RESULT_LIMIT} candidates for "
                            f"{slot.query_key!r}")
        for candidate in record["candidates"]:
            identity = classify(candidate["iri"])
            if identity.vocabulary != name:
                raise PoolError(f"{name} returned {candidate['iri']!r}, a {identity.vocabulary} IRI")
            pool.append({
                "source": name, "rank": candidate["rank"], "iri": candidate["iri"],
                "vocabulary": identity.vocabulary, "local_id": identity.local_id,
                "label": candidate["label"], "description": candidate["description"],
                "extras": candidate["extras"]})
    return pool


def slot_query(slot: Slot) -> str:
    """Form the plain query a slot is scored with.

    Args: slot: the slot.
    Returns: the legacy query text; the term keeps its case, the definition its words.
    Raises: None.
    Side effects: none.
    """
    return build_query(normalise_term(slot.term), slot.definition)


def score_slot(slot: Slot, pool: Sequence[dict[str, Any]], scorer: Scorer,
               max_description_chars: int | None,
               provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score one slot's pool.

    Args: slot: the slot; pool: pool_candidates output; scorer: the reranker;
    max_description_chars: description cap used to build documents; provenance: scoring
    provenance to store with the record.
    Returns: the scores record: slot fields, gold identity, query text, the scored pool and the
    provenance.
    Raises: ScoreError if the scorer returns a wrong count or an impossible score.
    Side effects: model inference.
    """
    query = slot_query(slot)
    documents = [build_document(c["label"], c["description"], max_description_chars) for c in pool]
    scores = validate_scores(scorer.score(query, documents), len(documents)) if documents else []
    return {
        "slot_id": slot.slot_id, "variable_id": slot.variable_id, "source_path": slot.source_path,
        "role": slot.role, "slot": slot.slot, "is_member": slot.is_member,
        "node_class": slot.node_class, "term": slot.term, "query_key": slot.query_key,
        "gold": {"iri": slot.gold_iri, "vocabulary": slot.gold_vocabulary,
                 "local_id": slot.gold_local_id, "form": slot.gold_form},
        "query": query, "scoring": dict(provenance or {}),
        "pool": [{**candidate, "document": document, "score": score}
                 for candidate, document, score in zip(pool, documents, scores, strict=True)]}


def score_all(slots: Sequence[Slot], retrieved: Mapping[tuple[str, str], Mapping[str, Any]],
              scorer: Scorer, store: JsonlStore, *, sources: Sequence[str],
              max_description_chars: int | None,
              progress: Callable[[int, int], None] | None = None) -> Counter[str]:
    """Score every slot that has no record yet.

    Args: slots: the population; retrieved: successful retrieval records; scorer: the reranker;
    store: scores JSONL; sources: pooling order; max_description_chars: description cap;
    progress: callback(done, total) after each slot.
    Returns: counters: scored, skipped, reused (same variable, same query text), empty_pool.
    Raises: PoolError for missing or inconsistent candidates, or when existing records were
    scored with another description cap or another model revision; ScoreError from scoring.
    Side effects: model inference; one appended, fsynced line per scored slot.
    """
    recorded = latest(store.read(), lambda record: record["slot_id"])
    caps = {(record.get("scoring") or {}).get("max_description_chars") for record in recorded.values()}
    if caps and caps != {max_description_chars}:
        raise PoolError(f"existing scores used description cap {sorted(caps, key=str)}, "
                        f"this run asks for {max_description_chars}")
    stats: Counter[str] = Counter()
    provenance: dict[str, Any] | None = None
    scored_here: dict[tuple[str, str], dict[str, Any]] = {}
    for done, slot in enumerate(slots, 1):
        if slot.slot_id in recorded:
            stats["skipped"] += 1
        else:
            if provenance is None:
                provenance = scoring_provenance(scorer, max_description_chars)
                revisions = {(r.get("scoring") or {}).get("revision") for r in recorded.values()}
                if revisions and revisions != {provenance["revision"]}:
                    raise PoolError(f"existing scores come from model revision "
                                    f"{sorted(revisions, key=str)}, this run has "
                                    f"{provenance['revision']}")
            pool = pool_candidates(slot, retrieved, sources)
            twin = scored_here.get((slot.variable_id, slot_query(slot)))
            if twin is not None:
                record = score_slot(slot, pool, _Replay(twin), max_description_chars, provenance)
                stats["reused"] += 1
            else:
                record = score_slot(slot, pool, scorer, max_description_chars, provenance)
                scored_here[(slot.variable_id, record["query"])] = record
                stats["scored"] += 1
            if not pool:
                stats["empty_pool"] += 1
            store.append(record)
        if progress is not None:
            progress(done, len(slots))
    return stats


class _Replay:
    """Reuse the scores of an identical earlier slot (same variable, same query text)."""

    def __init__(self, record: Mapping[str, Any]) -> None:
        """Remember the earlier record's scores.

        Args: record: a scores record for the identical query.
        Returns: None.
        Raises: None.
        Side effects: none.
        """
        self._scores = [candidate["score"] for candidate in record["pool"]]

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return the remembered scores.

        Args: query: ignored; documents: must match the remembered count.
        Returns: the remembered scores.
        Raises: ValueError if the document count differs.
        Side effects: none.
        """
        if len(documents) != len(self._scores):
            raise ValueError("Replayed pool differs from the original pool")
        return list(self._scores)
