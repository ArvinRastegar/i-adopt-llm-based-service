"""Controlled-vocabulary identity of an IRI: which vocabulary, which local identifier."""

from __future__ import annotations

import re
from dataclasses import dataclass

REACHABLE = ("wikidata", "qudt", "om")
NON_VOCABULARY = "non-vocabulary"
UNKNOWN = "unknown"

_IRI = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*)://(?P<host>[^/?#]+)(?P<path>/[^?#]*)?$")
_WIKIDATA = re.compile(r"^/(?P<kind>entity|wiki|enity)/(?P<id>Q[0-9]+)/?$")
_QUDT = re.compile(r"^/vocab/quantitykind/(?P<id>[A-Za-z0-9_.-]+)/?$")
_OM = re.compile(r"^/resource/om-2/(?P<id>[A-Za-z0-9_.-]+)$")
# Vocabularies the three retrieval sources cannot reach; the identity only has to be stable.
_OTHER = {
    "purl.jp": ("iobc", re.compile(r"^/bio/4/id/(?P<id>[0-9]+)/?$")),
    "vocab.nerc.ac.uk": (
        "nerc", re.compile(r"^/collection/(?P<id>[A-Za-z0-9]+/current/[A-Za-z0-9]+)/?$")),
    "lod.nal.usda.gov": ("nalt", re.compile(r"^/nalt/(?P<id>[0-9]+)/?$")),
    "vocabs.lter-europe.net": ("envthes", re.compile(r"^/EnvThes/(?P<id>[0-9]+)/?$")),
    "goldbook.iupac.org": ("iupac-goldbook", re.compile(r"^/terms/view/(?P<id>[A-Za-z0-9]+)/?$")),
    "purl.bioontology.org": (
        "bioportal", re.compile(r"^/ontology/(?P<id>[A-Za-z0-9]+/[A-Za-z0-9]+)/?$")),
}
_NON_VOCABULARY_HOSTS = ("example.org", "www.example.org", "github.com")


@dataclass(frozen=True)
class Identity:
    """One IRI read as a vocabulary and a local identifier."""

    vocabulary: str
    local_id: str
    form: str

    @property
    def key(self) -> tuple[str, str]:
        """Give the pair that two IRIs must share to name the same entity."""
        return (self.vocabulary, self.local_id)


def classify(iri: str) -> Identity:
    """Read an IRI as a controlled-vocabulary identity."""
    match = _IRI.match(iri.strip())
    if match is None:
        raise ValueError(f"Not an IRI: {iri!r}")
    scheme, host, path = match["scheme"].lower(), match["host"].lower(), match["path"] or ""
    well_formed = scheme in ("http", "https")
    for vocabulary, hostname, pattern in (
            ("wikidata", "www.wikidata.org", _WIKIDATA), ("qudt", "qudt.org", _QUDT),
            ("om", "www.ontology-of-units-of-measure.org", _OM)):
        if host != hostname:
            continue
        found = pattern.match(path)
        if found is None:
            return Identity(UNKNOWN, iri.strip(), "opaque")
        kind = found.groupdict().get("kind")
        if not well_formed or kind == "enity":
            form = "repaired"
        elif kind == "wiki":
            form = "variant"
        else:
            form = "canonical" if scheme == "http" else "scheme"
        return Identity(vocabulary, found["id"], form)
    if host in _OTHER:
        vocabulary, pattern = _OTHER[host]
        found = pattern.match(path)
        if found is not None:
            return Identity(vocabulary, found["id"], "opaque")
    if host in _NON_VOCABULARY_HOSTS:
        return Identity(NON_VOCABULARY, host + path, "opaque")
    return Identity(UNKNOWN, iri.strip(), "opaque")


def is_controlled_vocabulary(identity: Identity) -> bool:
    """Say whether an identity names a term of a controlled vocabulary."""
    return identity.vocabulary not in (NON_VOCABULARY, UNKNOWN)
