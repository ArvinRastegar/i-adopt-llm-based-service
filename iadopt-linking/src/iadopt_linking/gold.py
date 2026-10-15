"""Gold entity links of the pinned I-ADOPT corpus, read from lab-verified Turtle.

iadopt-lab owns the corpus and its parser. That parser keeps IRIs only for top-level slots and
drops those of ratio, source/target and part members, so linking needs its own reading of the
same Turtle. ``load_gold_links`` asks the lab to verify every source byte first, binds the bytes
it parses to the hashes the lab recorded, and proves that the lexical text it extracted equals
the lab's gold before returning anything.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iadopt_lab.corpus.ingestion import load_canonical_records
from rdflib import RDF, RDFS, Graph, Literal, Namespace, URIRef

IOP = Namespace("https://w3id.org/iadopt/ont/")
LAB_ROOT_ENV = "IADOPT_LAB_ROOT"
SLOTS = (
    "hasProperty", "hasStatisticalModifier", "hasObjectOfInterest", "hasMatrix", "hasContextObject",
)
MEMBERS = ("hasNumerator", "hasDenominator", "hasSource", "hasTarget", "hasPart")
NODE_CLASSES = ("StatisticalModifier", "Property", "Entity")


@dataclass(frozen=True)
class GoldLink:
    """One linked slot value exactly as the corpus Turtle writes it.

    ``role`` is the slot name, or ``<slot>.<member>`` for a ratio, source/target or part member.
    ``iri`` is None for a blank node, which is a label with no identifier at all.
    """

    role: str
    slot: str
    node_class: str
    term: str
    iri: str | None

    @property
    def is_member(self) -> bool:
        """Report whether the link sits inside a ratio, source/target or part system.

        Args: none.
        Returns: True for system members, False for a slot's own node.
        Raises: None.
        Side effects: none.
        """
        return "." in self.role


@dataclass(frozen=True)
class GoldVariable:
    """One corpus variable with the context a linker sees and the gold links to score."""

    variable_id: str
    label: str
    definition: str
    source_path: str
    source_sha256: str
    links: tuple[GoldLink, ...]


def resolve_lab_root(lab_root: Path | str | None = None) -> Path:
    """Locate the iadopt-lab project root that owns the corpus.

    Args: lab_root: explicit root; else $IADOPT_LAB_ROOT; else the sibling ``iadopt-lab`` folder.
    Returns: resolved root directory holding ``data/manifests``.
    Raises: FileNotFoundError when the chosen directory is not an iadopt-lab root.
    Side effects: filesystem existence checks only.
    """
    candidate = lab_root or os.environ.get(LAB_ROOT_ENV)
    root = Path(candidate) if candidate else Path(__file__).resolve().parents[3] / "iadopt-lab"
    root = root.resolve()
    if not (root / "data/manifests").is_dir():
        raise FileNotFoundError(f"Not an iadopt-lab root (no data/manifests): {root}")
    return root


def _label(graph: Graph, node: Any) -> str:
    """Read the sole non-blank rdfs:label of a linked node with whitespace preserved.

    Args: graph: parsed variable; node: linked RDF node.
    Returns: exact label text.
    Raises: ValueError when the node has no label, several, or a blank or non-literal one.
    Side effects: none.
    """
    values = set(graph.objects(node, RDFS.label))
    if len(values) != 1:
        raise ValueError(f"Linked node needs exactly one rdfs:label, found {len(values)}: {node}")
    value = next(iter(values))
    if not isinstance(value, Literal) or not str(value).strip():
        raise ValueError(f"Linked node rdfs:label must be a non-blank literal: {node}")
    return str(value)


def _node_class(graph: Graph, node: Any) -> str:
    """Name the I-ADOPT component class of a linked node.

    Args: graph: parsed variable; node: linked RDF node.
    Returns: ``StatisticalModifier``, ``Property`` or ``Entity``.
    Raises: ValueError when the node carries none of them.
    Side effects: none.
    """
    names = {str(kind)[len(IOP):] for kind in graph.objects(node, RDF.type)
             if str(kind).startswith(IOP)}
    for name in NODE_CLASSES:
        if name in names:
            return name
    raise ValueError(f"Linked node has no I-ADOPT component class: {node}")


def _link(graph: Graph, node: Any, role: str) -> GoldLink:
    """Describe one linked node.

    Args: graph: parsed variable; node: linked RDF node; role: slot or ``<slot>.<member>``.
    Returns: the node's class, label and IRI (None for a blank node).
    Raises: ValueError for an unlabeled or unclassified node.
    Side effects: none.
    """
    return GoldLink(role=role, slot=role.split(".")[0], node_class=_node_class(graph, node),
                    term=_label(graph, node), iri=str(node) if isinstance(node, URIRef) else None)


def extract_links(ttl: bytes | str) -> tuple[GoldLink, ...]:
    """Extract every linked node of one corpus variable, system members included.

    Args: ttl: the variable's Turtle as UTF-8 bytes or text.
    Returns: links ordered by slot, role, term and IRI, independent of RDF iteration order.
    Raises: ValueError for a missing or repeated Variable root or an unlabeled node;
    RDF parser errors for invalid Turtle.
    Side effects: none; no ontology or service is consulted. Deterministic.
    """
    text = ttl.decode("utf-8") if isinstance(ttl, bytes) else ttl
    graph = Graph().parse(data=text, format="turtle")
    roots = set(graph.subjects(RDF.type, IOP.Variable))
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one Variable root, received {len(roots)}")
    root = next(iter(roots))
    links: list[GoldLink] = []
    for slot in SLOTS:
        for node in graph.objects(root, IOP[slot]):
            is_system = ((node, RDF.type, IOP.SymmetricSystem) in graph
                         or (node, RDF.type, IOP.AsymmetricSystem) in graph)
            if not is_system:
                links.append(_link(graph, node, slot))
                continue
            for member in MEMBERS:
                links.extend(_link(graph, target, f"{slot}.{member}")
                             for target in graph.objects(node, IOP[member]))
    return tuple(sorted(
        links, key=lambda link: (SLOTS.index(link.slot), link.role, link.term, link.iri or "")))


def lexical_mismatches(links: Sequence[GoldLink], gold: Mapping[str, Any]) -> list[str]:
    """Compare extracted link text with the lab's lexical gold for the same variable.

    Args: links: extract_links output; gold: the lab record's ``gold`` mapping.
    Returns: one description per slot where the two readings differ; empty when they agree.
    Raises: None.
    Side effects: none.
    """
    problems: list[str] = []
    for slot in SLOTS:
        value = gold.get(slot) or ""
        expected: dict[str, list[str]] = {}
        if isinstance(value, str):
            if value:
                expected[slot] = [value]
        else:
            for member in MEMBERS:
                if member in value:
                    terms = value[member]
                    expected[f"{slot}.{member}"] = [terms] if isinstance(terms, str) else list(terms)
        found: dict[str, list[str]] = {}
        for link in links:
            if link.slot == slot:
                found.setdefault(link.role, []).append(link.term)
        if ({role: sorted(terms) for role, terms in expected.items()}
                != {role: sorted(terms) for role, terms in found.items()}):
            problems.append(f"{slot}: lab gold {expected!r} differs from linked nodes {found!r}")
    return problems


def load_gold_links(lab_root: Path | str | None = None) -> tuple[GoldVariable, ...]:
    """Load every variable's gold links after the lab has verified the corpus.

    Args: lab_root: iadopt-lab root; see resolve_lab_root.
    Returns: the lab's 102 variables in the lab's order, each with its linked nodes.
    Raises: ValueError when the lab's verification fails, a source file differs from the hash the
    lab recorded for it, or the extracted text disagrees with the lab's gold.
    Side effects: filesystem reads only.
    """
    root = resolve_lab_root(lab_root)
    variables = []
    for record in load_canonical_records(root):
        path = record["source_path"]
        raw = (root / "data/corpus" / record["tag"] / path).read_bytes()
        if (hashlib.sha256(raw).hexdigest(), len(raw)) != (
                record["source_sha256"], record["source_byte_length"]):
            raise ValueError(f"Source bytes differ from the lab's record: {path}")
        links = extract_links(raw)
        problems = lexical_mismatches(links, record["gold"])
        if problems:
            raise ValueError(f"Link reading disagrees with the lab's gold for {path}: {problems}")
        variables.append(GoldVariable(
            variable_id=record["variable_id"], label=record["label"],
            definition=record["definition"], source_path=path,
            source_sha256=record["source_sha256"], links=links))
    return tuple(variables)
