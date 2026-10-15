"""The three live retrieval sources: Wikidata, QUDT quantity kinds and OM quantities.

Each source turns a query key (a lower-cased, normalised term) into a request and turns the
raw response text into at most ten ranked candidates. Parsing is strict: a response that does
not have the expected shape raises SourceError so it is recorded as a failure, never as an
empty result.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

RESULT_LIMIT = 10
WIKIDATA_ENDPOINT = "https://www.wikidata.org/w/api.php"
QUDT_ENDPOINT = "https://www.qudt.org/fuseki/qudt/sparql"
OLS_ENDPOINT = "https://www.ebi.ac.uk/ols4/api/search"
OM_QUANTITY = "http://www.ontology-of-units-of-measure.org/resource/om-2/Quantity"
OLS_ONTOLOGY_ENDPOINT = "https://www.ebi.ac.uk/ols4/api/ontologies/om"
# The endpoint publishes no version number, only when its graph was last modified.
QUDT_RELEASE_QUERY = (
    "PREFIX dcterms: <http://purl.org/dc/terms/> SELECT ?modified WHERE { "
    "<http://www.linkedmodel.org/schema/vaem#GMD_SHACLQUDT-All> dcterms:modified ?modified } LIMIT 1")
# Words that carry no meaning in a quantity-kind label; dropping them lets "amount of substance
# concentration" find QUDT's "Amount of Substance of Concentration".
QUDT_STOPWORDS = frozenset({"of", "the", "a", "an", "and"})


class SourceError(ValueError):
    """A response that does not have the shape the source promises."""


@dataclass(frozen=True)
class Candidate:
    """One ranked answer from one source, as the source wrote it."""

    source: str
    rank: int
    iri: str
    label: str
    description: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Request:
    """Everything needed to ask a source about one key."""

    url: str
    params: dict[str, str]
    headers: dict[str, str]


def _json(text: str, source: str) -> Any:
    """Decode a JSON response body.

    Args: text: response body; source: source name for the message.
    Returns: the decoded value.
    Raises: SourceError when the body is not JSON.
    Side effects: none.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SourceError(f"{source}: response is not JSON ({exc})") from exc


def _words(text: str) -> str:
    """Collapse whitespace so multi-line descriptions fit one document line.

    Args: text: any description.
    Returns: single-spaced text.
    Raises: None.
    Side effects: none.
    """
    return re.sub(r"\s+", " ", text).strip()


def _strict(parse: Callable[[Any, str], list[Candidate]]) -> Callable[[Any, str], list[Candidate]]:
    """Turn any structural failure of a parser into a SourceError.

    Args: parse: a source's parse method.
    Returns: the same method, raising SourceError for a response of the wrong inner shape, so it
    is recorded as a failure with its text instead of escaping as a KeyError or AttributeError.
    Raises: None here; the wrapped method raises SourceError.
    Side effects: none.
    """
    @functools.wraps(parse)
    def wrapper(self: Any, text: str) -> list[Candidate]:
        try:
            return parse(self, text)
        except SourceError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
            raise SourceError(
                f"{self.name}: malformed response ({type(exc).__name__}: {exc})") from exc
    return wrapper


class WikidataSource:
    """Wikidata items through the wbsearchentities API."""

    name = "wikidata"

    def request(self, key: str) -> Request:
        """Build the search request.

        Args: key: query key.
        Returns: GET request for at most RESULT_LIMIT English items.
        Raises: None.
        Side effects: none.
        """
        return Request(WIKIDATA_ENDPOINT, {
            "action": "wbsearchentities", "search": key, "language": "en", "uselang": "en",
            "type": "item", "limit": str(RESULT_LIMIT), "format": "json"},
            {"Accept": "application/json"})

    @_strict
    def parse(self, text: str) -> list[Candidate]:
        """Read the ranked items from a response.

        Args: text: response body.
        Returns: candidates in the API's own order.
        Raises: SourceError for an API error, a missing success flag, a malformed item, a
        non-item id or too many items.
        Side effects: none.
        """
        data = _json(text, self.name)
        if not isinstance(data, dict) or "error" in data or "search" not in data:
            raise SourceError(f"wikidata: unexpected response {str(data)[:160]}")
        if data.get("success") != 1:
            raise SourceError("wikidata: the API did not report success")
        items = data["search"]
        if not isinstance(items, list) or len(items) > RESULT_LIMIT:
            raise SourceError("wikidata: search is not a list of at most RESULT_LIMIT items")
        candidates = []
        for rank, item in enumerate(items, 1):
            identifier = item.get("id", "")
            if not isinstance(identifier, str) or not re.fullmatch(r"Q[0-9]+", identifier):
                raise SourceError(f"wikidata: not an item id: {identifier!r}")
            label, description = item.get("label", ""), item.get("description", "")
            if not isinstance(label, str) or not isinstance(description, str):
                raise SourceError(f"wikidata: label or description of {identifier} is not text")
            candidates.append(Candidate(
                self.name, rank,
                item.get("concepturi") or f"http://www.wikidata.org/entity/{identifier}",
                label, description, {}))
        return candidates


class QudtSource:
    """QUDT quantity kinds through the public SPARQL endpoint (no text index, so label filters)."""

    name = "qudt"

    def request(self, key: str) -> Request | None:
        """Build the query: kinds whose English label contains every word of the key.

        Args: key: query key.
        Returns: GET request ordered by label length (an exact label sorts first), at most
        RESULT_LIMIT kinds; None when the key has no searchable word.
        Raises: None.
        Side effects: none.
        """
        words = [w for w in re.findall(r"[^\W_]+", key.lower()) if w not in QUDT_STOPWORDS]
        if not words:
            return None
        filters = "\n      ".join(f'FILTER(CONTAINS(LCASE(STR(?l)), "{w}"))' for w in words)
        query = f"""PREFIX qudt: <http://qudt.org/schema/qudt/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
SELECT ?qk ?len ?label ?description ?wikidata ?definedBy WHERE {{
  {{ SELECT ?qk (MIN(STRLEN(STR(?l))) AS ?len) WHERE {{
      ?qk a qudt:QuantityKind ; rdfs:label ?l .
      FILTER(STRSTARTS(LANG(?l), "en") || LANG(?l) = "")
      FILTER NOT EXISTS {{ ?qk qudt:deprecated true }}
      {filters}
  }} GROUP BY ?qk ORDER BY ?len ?qk LIMIT {RESULT_LIMIT} }}
  ?qk rdfs:label ?label . FILTER(STRSTARTS(LANG(?label), "en") || LANG(?label) = "")
  OPTIONAL {{ ?qk dcterms:description ?d1 }}
  OPTIONAL {{ ?qk qudt:plainTextDescription ?d2 }}
  OPTIONAL {{ ?qk qudt:wikidataMatch ?wikidata }}
  OPTIONAL {{ ?qk rdfs:isDefinedBy ?definedBy }}
  BIND(COALESCE(?d1, ?d2) AS ?description)
}}"""
        return Request(QUDT_ENDPOINT, {"query": query},
                       {"Accept": "application/sparql-results+json"})

    @_strict
    def parse(self, text: str) -> list[Candidate]:
        """Group the result rows into one candidate per quantity kind.

        Args: text: SPARQL JSON results.
        Returns: candidates ordered by label length, then IRI; the English label is preferred
        to en-US, and the Wikidata match and release IRI travel in ``extras``.
        Raises: SourceError for a missing section, a row of the wrong shape or more than
        RESULT_LIMIT kinds.
        Side effects: none.
        """
        data = _json(text, self.name)
        try:
            rows = data["results"]["bindings"]
        except (KeyError, TypeError) as exc:
            raise SourceError(f"qudt: unexpected response {str(data)[:160]}") from exc
        if not isinstance(rows, list):
            raise SourceError("qudt: bindings is not a list")
        kinds: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = kinds.setdefault(row["qk"]["value"], {
                "length": int(row["len"]["value"]), "labels": {}, "descriptions": set(),
                "wikidata": set(), "defined_by": set()})
            entry["labels"].setdefault(row["label"].get("xml:lang", ""), set()).add(
                row["label"]["value"])
            if "description" in row:
                entry["descriptions"].add(_words(row["description"]["value"]))
            if "wikidata" in row:
                entry["wikidata"].add(row["wikidata"]["value"])
            if "definedBy" in row:
                entry["defined_by"].add(row["definedBy"]["value"])
        if len(kinds) > RESULT_LIMIT:
            raise SourceError(f"qudt: {len(kinds)} kinds returned, limit is {RESULT_LIMIT}")
        candidates = []
        for rank, (iri, entry) in enumerate(
                sorted(kinds.items(), key=lambda item: (item[1]["length"], item[0])), 1):
            labels = entry["labels"].get("en") or entry["labels"].get("en-US") or next(
                iter(entry["labels"].values()))
            descriptions = sorted(d for d in entry["descriptions"] if d)
            candidates.append(Candidate(
                self.name, rank, iri, sorted(labels)[0], descriptions[0] if descriptions else "",
                {"wikidata_match": sorted(entry["wikidata"]),
                 "defined_by": sorted(entry["defined_by"])}))
        return candidates


class OmSource:
    """OM 2 quantity classes through the EBI Ontology Lookup Service search."""

    name = "om"

    def request(self, key: str) -> Request:
        """Build the search request restricted to descendants of om:Quantity.

        Args: key: query key.
        Returns: GET request for at most RESULT_LIMIT OM quantity classes.
        Raises: None.
        Side effects: none.
        """
        return Request(OLS_ENDPOINT, {
            "q": key, "ontology": "om", "allChildrenOf": OM_QUANTITY, "rows": str(RESULT_LIMIT),
            "fieldList": "iri,label,description,type,ontology_name"},
            {"Accept": "application/json"})

    @_strict
    def parse(self, text: str) -> list[Candidate]:
        """Read the ranked classes from a response.

        Args: text: OLS search JSON.
        Returns: candidates in OLS relevance order.
        Raises: SourceError for a missing section, a non-zero service status, a document count
        that contradicts numFound, a foreign ontology, a label or description that is not text,
        or too many documents.
        Side effects: none.
        """
        data = _json(text, self.name)
        try:
            documents = data["response"]["docs"]
            found = data["response"]["numFound"]
        except (KeyError, TypeError) as exc:
            raise SourceError(f"om: unexpected response {str(data)[:160]}") from exc
        status = (data.get("responseHeader") or {}).get("status")
        if status != 0:
            raise SourceError(f"om: the service reported status {status!r}")
        if not isinstance(documents, list) or len(documents) > RESULT_LIMIT:
            raise SourceError("om: docs is not a list of at most RESULT_LIMIT items")
        if isinstance(found, bool) or not isinstance(found, int) or len(documents) != min(
                found, RESULT_LIMIT):
            raise SourceError(f"om: numFound {found!r} contradicts {len(documents)} documents")
        candidates = []
        for rank, document in enumerate(documents, 1):
            if document.get("ontology_name") != "om" or not document.get("iri"):
                raise SourceError(f"om: unexpected document {str(document)[:160]}")
            label = document.get("label")
            raw = document.get("description") or []
            if not isinstance(label, str) or not label:
                raise SourceError(f"om: {document['iri']} has no text label")
            if not isinstance(raw, list) or not all(isinstance(d, str) for d in raw):
                raise SourceError(f"om: description of {document['iri']} is not a list of text")
            descriptions = [d for d in raw if d]
            candidates.append(Candidate(
                self.name, rank, document["iri"], label,
                _words(descriptions[0]) if descriptions else "", {}))
        return candidates


SOURCES = {source.name: source for source in (WikidataSource(), QudtSource(), OmSource())}
