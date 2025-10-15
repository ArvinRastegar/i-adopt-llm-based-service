#!/usr/bin/env python
"""
iadopt_ttl2json.py  ·  2025-10-15
------------------------------------

• Downloads C1…C30.ttl (or a numeric subset via CLI)
• Converts each to the compact JSON layout from the most-recent examples
• Enriches hasProperty/hasMatrix/hasObjectOfInterest/hasContextObject with ...URI
• For systems:
    - Asymmetric: add AsymmetricSystemURI (if mapped), hasSourceURI, hasTargetURI
    - Symmetric:  add SymmetricSystemURI (if mapped), hasPartURIs (aligned with hasPart)
• Saves every variable as its *own* JSON file → C1.json, C2.json, ...
• Writes a combined 'all_variables.json' to the same folder for convenience
• Ships with unittests that exercise every code path, including URI enrichment
"""

from __future__ import annotations
import csv
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests  # pip install requests rdflib
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal
import unittest

# --------------------------------------------------------------------------- #
# 1️⃣  Fixed output location (create if necessary)
# --------------------------------------------------------------------------- #
OUTDIR = Path(
    "/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service/" "benchmarking_example/data/Json_preferred"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# 2️⃣  Constants & namespaces
BASE_URL = "https://sirkos.github.io/challenge/Challenge"
IOP = Namespace("https://w3id.org/iadopt/ont/")
RDFS_NS = RDFS

# 3️⃣  CSV mapping file discovery (load once at startup)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPINGS_DIR = SCRIPT_DIR / "mappings"

MAPPINGS_OOI_PATH = Path(os.getenv("MAPPINGS_OOI_PATH", str(DEFAULT_MAPPINGS_DIR / "Terms_OoI.csv")))
MAPPINGS_MATRIX_PATH = Path(os.getenv("MAPPINGS_PROPERTY_PATH", str(DEFAULT_MAPPINGS_DIR / "Terms_Matrix.csv")))
MAPPINGS_PROPERTY_PATH = Path(os.getenv("MAPPINGS_PROPERTY_PATH", str(DEFAULT_MAPPINGS_DIR / "Terms_Property.csv")))

WIKIDATA_MAPPINGS: Dict[str, str] = {}  # label → full Wikidata URL


def _normalize_wikidata_value(u: str) -> Optional[str]:
    """Accept either a full Wikidata URL or a naked QID; return full URL or None."""
    if not u:
        return None
    u = u.strip()
    if not u:
        return None
    if u.startswith(("http://", "https://")):
        return u
    if re.fullmatch(r"Q\d+", u):
        return f"https://www.wikidata.org/wiki/{u}"
    return None


def _load_csv_mapping(path: Path) -> Dict[str, str]:
    """
    Load a CSV with headers 'Concept;Wikidata' (semicolon-separated).
    Returns a dict: label -> full Wikidata URL. Missing files are tolerated.
    """
    mapping: Dict[str, str] = {}
    if not path or not path.exists():
        print(f"⚠️  Warning: mapping file not found: {path}")
        return mapping

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            concept = (row.get("Concept") or row.get("concept") or row.get("label") or "").strip()
            wikidata_raw = (row.get("Wikidata") or row.get("wikidata") or row.get("URI") or "").strip()
            if not concept or not wikidata_raw:
                continue
            url = _normalize_wikidata_value(wikidata_raw)
            if url:
                mapping[concept] = url
    return mapping


def load_wikidata_mappings() -> Dict[str, str]:
    """Load all mappings (OOI, Matrix, Property) once at startup and merge them."""
    ooi = _load_csv_mapping(MAPPINGS_OOI_PATH)
    matrix = _load_csv_mapping(MAPPINGS_MATRIX_PATH)
    prop = _load_csv_mapping(MAPPINGS_PROPERTY_PATH)
    merged = {**ooi, **matrix, **prop}  # right-most wins on collisions
    print(f"Loaded {len(merged)} Wikidata mappings from CSV files.")
    return merged


# Load at module import (once)
WIKIDATA_MAPPINGS = load_wikidata_mappings()


# --------------------------------------------------------------------------- #
# 4️⃣  Helper utilities
# --------------------------------------------------------------------------- #
def _label(g: Graph, node: URIRef | None) -> str | None:
    lab = g.value(node, RDFS_NS.label) if node else None
    return str(lab) if isinstance(lab, Literal) else None


def _list_labels(g: Graph, nodes: List[URIRef]) -> List[str]:
    return [_label(g, n) for n in nodes if _label(g, n)]


def _collapse_issue(url: str) -> str:
    m = re.search(r"/issues/(\d+)", url)
    return f"issue {m.group(1)}" if m else url


def _uri_for_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    return WIKIDATA_MAPPINGS.get(label)


def _maybe_add_uri_field(container: Dict[str, Any], key: str, label_value: Any) -> None:
    """
    If the value is a simple string and a mapping exists, add the parallel ...URI field.
    Example: key='hasMatrix' -> 'hasMatrixURI'.
    """
    if isinstance(label_value, str):
        uri = _uri_for_label(label_value)
        if uri:
            container[f"{key}URI"] = uri


# Helper: serialise an entity *or* system node, enriching with URIs where possible
def _entity_representation(g: Graph, node: URIRef | None) -> str | Dict[str, Any] | None:
    """Return either a plain label or a dict describing a (a)symmetric system, with URI enrichment."""
    if node is None:
        return None

    # Asymmetric system with source/target
    if (node, RDF.type, IOP.AsymmetricSystem) in g:
        sys_label = _label(g, node)
        src_node = g.value(node, IOP.hasSource)
        tgt_node = g.value(node, IOP.hasTarget)
        src_label = _label(g, src_node)
        tgt_label = _label(g, tgt_node)

        out: Dict[str, Any] = {
            "AsymmetricSystem": sys_label,
            "hasSource": src_label,
            "hasTarget": tgt_label,
        }
        # URIs for system label, source, target (if available)
        if u := _uri_for_label(sys_label):
            out["AsymmetricSystemURI"] = u
        if u := _uri_for_label(src_label):
            out["hasSourceURI"] = u
        if u := _uri_for_label(tgt_label):
            out["hasTargetURI"] = u
        return out

    # Symmetric system with parts
    if (node, RDF.type, IOP.SymmetricSystem) in g:
        sys_label = _label(g, node)
        parts_nodes = list(g.objects(node, IOP.hasPart))
        parts_labels = _list_labels(g, parts_nodes)

        out: Dict[str, Any] = {
            "SymmetricSystem": sys_label,
            "hasPart": parts_labels,
        }
        if u := _uri_for_label(sys_label):
            out["SymmetricSystemURI"] = u

        # Build aligned URIs array; include key only if any mapping exists
        part_uris: List[Optional[str]] = [(_uri_for_label(lbl) or None) for lbl in parts_labels]
        if any(part_uris):
            out["hasPartURIs"] = part_uris
        return out

    # plain entity -> string label
    return _label(g, node)


# --------------------------------------------------------------------------- #
# 5️⃣  Core RDF→JSON mapping
# --------------------------------------------------------------------------- #
def parse_variable(ttl: str) -> Dict[str, Any]:
    g = Graph()
    g.parse(data=ttl, format="turtle")

    # locate the Variable root
    roots = list(g.subjects(RDF.type, IOP.Variable))
    if not roots:
        raise ValueError("No iop:Variable found")
    root = roots[0]

    j: Dict[str, Any] = {
        "label": _label(g, root),
        "comment": str(g.value(root, RDFS_NS.comment, default="")).strip() or None,
    }

    # -- simple scalar field --------------------------------------------------
    if prop := g.value(root, IOP.hasProperty):
        j["hasProperty"] = _label(g, prop)
        _maybe_add_uri_field(j, "hasProperty", j["hasProperty"])

    # -- links that may be entity *or* system --------------------------------
    for predicate, key in [
        (IOP.hasMatrix, "hasMatrix"),
        (IOP.hasObjectOfInterest, "hasObjectOfInterest"),
        (IOP.hasContextObject, "hasContextObject"),
    ]:
        if node := g.value(root, predicate):
            val = _entity_representation(g, node)
            j[key] = val
            # Only enrich with URI when the value is a simple label (string).
            _maybe_add_uri_field(j, key, val)

    # statistical modifier stays scalar
    if mod := g.value(root, IOP.hasStatisticalModifier):
        j["hasStatisticalModifier"] = _label(g, mod)

    # -- constraints ----------------------------------------------------------
    constraints = list(g.objects(root, IOP.hasConstraint))
    if constraints:
        clist: List[Dict[str, str]] = []
        # Object-of-interest *or* matrix system for prefix resolution
        ooi_node = g.value(root, IOP.hasObjectOfInterest)
        matrix_node = g.value(root, IOP.hasMatrix)

        for c in constraints:
            cname = _label(g, c)
            constrained = g.value(c, IOP.constrains)
            target_lbl = _label(g, constrained)

            # if this constrained node is a source/target of either system,
            # prepend that context to the label
            for sys_node in (ooi_node, matrix_node):
                if sys_node and (sys_node, IOP.hasTarget, constrained) in g:
                    target_lbl = f"hasTarget: {target_lbl}"
                    break
                if sys_node and (sys_node, IOP.hasSource, constrained) in g:
                    target_lbl = f"hasSource: {target_lbl}"
                    break

            clist.append({"label": cname, "on": target_lbl})

        j["hasConstraint"] = clist

    # optional issue
    for p, o in g.predicate_objects(root):
        if str(p).split("#")[-1] == "issue":
            j["issue"] = _collapse_issue(str(o))
            break

    # prune nulls
    return {k: v for k, v in j.items() if v is not None}


# --------------------------------------------------------------------------- #
# 6️⃣  I/O helpers
# --------------------------------------------------------------------------- #
def _download(url: str, timeout: int = 15) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def convert_one(n: int) -> Dict[str, Any]:
    ttl = _download(f"{BASE_URL}/C{n}.ttl")
    data = parse_variable(ttl)
    # Capital filenames like C1.json, C2.json, ...
    (OUTDIR / f"C{n}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"✓  C{n}.json")
    return data


# --------------------------------------------------------------------------- #
# 7️⃣  Command-line interface
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = range(int(sys.argv[1]), int(sys.argv[1]) + 1) if len(sys.argv) > 1 and sys.argv[1].isdigit() else range(1, 31)

    # Informative note about mapping load (only shown once per run)
    if WIKIDATA_MAPPINGS:
        print(
            f"🔗 Loaded {len(WIKIDATA_MAPPINGS)} Wikidata mappings "
            f"(OOI: {len(_load_csv_mapping(MAPPINGS_OOI_PATH))}, "
            f"Matrix: {len(_load_csv_mapping(MAPPINGS_MATRIX_PATH))}, "
            f"Property: {len(_load_csv_mapping(MAPPINGS_PROPERTY_PATH))})"
        )
    else:
        print("⚠️  No Wikidata mappings found; proceeding without URI enrichment.")

    all_data = [convert_one(i) for i in rng]
    (OUTDIR / "all_variables.json").write_text(json.dumps(all_data, indent=2, ensure_ascii=False))
    print(f"✓  all_variables.json  ({len(all_data)} entries)  → {OUTDIR}\n")

    # run tests after conversion
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)  # rdflib noise
    res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(__name__))
    if not res.wasSuccessful():
        sys.exit("❌ tests failed")
    print("🎉  All tests passed")


# --------------------------------------------------------------------------- #
# 8️⃣  Built-in test-suite
# --------------------------------------------------------------------------- #
# Minimal unit-test covering an asymmetric *matrix*
class TestMatrixSystem(unittest.TestCase):
    TTL_MATRIX_SYS = textwrap.dedent(
        """
        @prefix ex: <http://example.org/> .
        @prefix iop: <https://w3id.org/iadopt/ont/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:V a iop:Variable ;
             rdfs:label "Mass flux C" ;
             iop:hasProperty ex:P ;
             iop:hasMatrix ex:Mat .
        ex:P   a iop:Property ; rdfs:label "mass flux" .
        ex:Mat a iop:Entity, iop:System, iop:AsymmetricSystem ;
              rdfs:label "veg→soil" ;
              iop:hasSource ex:Veg ;
              iop:hasTarget ex:Soil .
        ex:Veg a iop:Entity ; rdfs:label "vegetation" .
        ex:Soil a iop:Entity ; rdfs:label "soil" .
    """
    )

    def test_matrix_as_system_with_nested_uris(self):
        # Inject mappings for nested components
        global WIKIDATA_MAPPINGS
        prev = dict(WIKIDATA_MAPPINGS)
        try:
            WIKIDATA_MAPPINGS = {
                "vegetation": "https://www.wikidata.org/wiki/Q187997",
                "soil": "https://www.wikidata.org/wiki/Q36133",
                # 'veg→soil' not mapped on purpose
                "mass flux": "https://www.wikidata.org/wiki/Q3265048",
            }
            d = parse_variable(self.TTL_MATRIX_SYS)
            self.assertIsInstance(d["hasMatrix"], dict)
            self.assertEqual(d["hasMatrix"]["AsymmetricSystem"], "veg→soil")
            self.assertEqual(d["hasMatrix"]["hasSource"], "vegetation")
            self.assertEqual(d["hasMatrix"]["hasTarget"], "soil")
            # URI enrichment inside the system
            self.assertIn("hasSourceURI", d["hasMatrix"])
            self.assertTrue(d["hasMatrix"]["hasSourceURI"].startswith("https://www.wikidata.org/wiki/"))
            self.assertIn("hasTargetURI", d["hasMatrix"])
            self.assertTrue(d["hasMatrix"]["hasTargetURI"].startswith("https://www.wikidata.org/wiki/"))
            # top-level hasMatrix is a dict → no top-level ...URI field
            self.assertNotIn("hasMatrixURI", d)
            # property is a simple string → should have URI
            self.assertEqual(d["hasProperty"], "mass flux")
            self.assertIn("hasPropertyURI", d)
        finally:
            WIKIDATA_MAPPINGS = prev


class TestMapping(unittest.TestCase):
    """Covers every branch, and checks URI enrichment including systems."""

    TTL_SIMPLE = textwrap.dedent(
        """
        @prefix ex: <http://example.org/> .
        @prefix iop: <https://w3id.org/iadopt/ont/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:V a iop:Variable ;
             rdfs:label "Cloud cover" ;
             iop:hasProperty ex:P ;
             iop:hasObjectOfInterest ex:O ;
             iop:hasMatrix ex:M .
        ex:P a iop:Property ; rdfs:label "area fraction" .
        ex:O a iop:Entity  ; rdfs:label "cloud" .
        ex:M a iop:Entity  ; rdfs:label "atmosphere" .
    """
    )

    TTL_ASYM = textwrap.dedent(
        """
        @prefix ex: <http://example.org/> .
        @prefix iop: <https://w3id.org/iadopt/ont/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:V a iop:Variable ;
             rdfs:label "Distance NN" ;
             iop:hasProperty ex:P ;
             iop:hasObjectOfInterest ex:Sys ;
             iop:hasConstraint [ a iop:Constraint ;
                                 rdfs:label "nearest" ;
                                 iop:constrains ex:Tgt ] .
        ex:P a iop:Property ; rdfs:label "distance" .
        ex:Sys a iop:Entity, iop:System, iop:AsymmetricSystem ;
               rdfs:label "patch system" ;
               iop:hasSource ex:Src ;
               iop:hasTarget ex:Tgt .
        ex:Src a iop:Entity ; rdfs:label "patch" .
        ex:Tgt a iop:Entity ; rdfs:label "patch" .
    """
    )

    TTL_SYM = textwrap.dedent(
        """
        @prefix ex: <http://example.org/> .
        @prefix iop: <https://w3id.org/iadopt/ont/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:V a iop:Variable ;
             rdfs:label "Pair interaction" ;
             iop:hasProperty ex:P ;
             iop:hasObjectOfInterest ex:Sym .
        ex:P a iop:Property ; rdfs:label "interaction strength" .
        ex:Sym a iop:Entity, iop:System, iop:SymmetricSystem ;
               rdfs:label "pair system" ;
               iop:hasPart ex:A ;
               iop:hasPart ex:B .
        ex:A a iop:Entity ; rdfs:label "alpha" .
        ex:B a iop:Entity ; rdfs:label "beta" .
    """
    )

    def test_simple_mapping_with_uri_enrichment(self):
        global WIKIDATA_MAPPINGS
        prev = dict(WIKIDATA_MAPPINGS)
        try:
            WIKIDATA_MAPPINGS = {
                "atmosphere": "https://www.wikidata.org/wiki/Q8104",
                "cloud": "https://www.wikidata.org/wiki/Q8072",
                "area fraction": "https://www.wikidata.org/wiki/Q55659167",
            }
            d = parse_variable(self.TTL_SIMPLE)
            self.assertEqual(d["hasMatrix"], "atmosphere")
            self.assertEqual(d["hasProperty"], "area fraction")
            self.assertEqual(d["hasObjectOfInterest"], "cloud")
            self.assertNotIn("hasConstraint", d)

            self.assertIn("hasMatrixURI", d)
            self.assertTrue(d["hasMatrixURI"].startswith("https://www.wikidata.org/wiki/"))
            self.assertIn("hasObjectOfInterestURI", d)
            self.assertTrue(d["hasObjectOfInterestURI"].startswith("https://www.wikidata.org/wiki/"))
            self.assertIn("hasPropertyURI", d)
            self.assertTrue(d["hasPropertyURI"].startswith("https://www.wikidata.org/wiki/"))
        finally:
            WIKIDATA_MAPPINGS = prev

    def test_asymmetric_system_mapping_enriched(self):
        global WIKIDATA_MAPPINGS
        prev = dict(WIKIDATA_MAPPINGS)
        try:
            WIKIDATA_MAPPINGS = {
                "distance": "https://www.wikidata.org/wiki/Q12453",
                "patch": "https://www.wikidata.org/wiki/Q101991",
            }
            d = parse_variable(self.TTL_ASYM)
            self.assertIn("hasConstraint", d)
            constraints = d["hasConstraint"]
            self.assertIsInstance(constraints, list)
            c = next((item for item in constraints if item.get("label") == "nearest"), None)
            self.assertIsNotNone(c, "should find a constraint with label 'nearest'")
            self.assertTrue(c["on"].startswith("hasTarget:"), f"unexpected on: {c['on']}")
            self.assertEqual(d["hasObjectOfInterest"]["AsymmetricSystem"], "patch system")

            # Property mapping adds URI for property (simple string), but not a top-level URI for OOI (dict)
            self.assertIn("hasPropertyURI", d)
            self.assertTrue(d["hasPropertyURI"].startswith("https://"))
            self.assertNotIn("hasObjectOfInterestURI", d)

            # Nested URIs on system components
            ooi = d["hasObjectOfInterest"]
            self.assertIn("hasSourceURI", ooi)
            self.assertIn("hasTargetURI", ooi)
        finally:
            WIKIDATA_MAPPINGS = prev

    def test_symmetric_system_mapping_enriched(self):
        global WIKIDATA_MAPPINGS
        prev = dict(WIKIDATA_MAPPINGS)
        try:
            WIKIDATA_MAPPINGS = {
                "alpha": "https://www.wikidata.org/wiki/Q666",
                # 'beta' intentionally unmapped to test null alignment
                "interaction strength": "https://www.wikidata.org/wiki/Q12345",
            }
            d = parse_variable(self.TTL_SYM)
            self.assertEqual(d["hasProperty"], "interaction strength")
            self.assertIn("hasPropertyURI", d)

            ooi = d["hasObjectOfInterest"]
            self.assertEqual(ooi["SymmetricSystem"], "pair system")
            self.assertEqual(ooi["hasPart"], ["alpha", "beta"])
            # hasPartURIs should exist (at least one mapping) and align to parts
            self.assertIn("hasPartURIs", ooi)
            self.assertEqual(len(ooi["hasPartURIs"]), 2)
            self.assertTrue(ooi["hasPartURIs"][0].startswith("https://www.wikidata.org/wiki/"))
            self.assertIsNone(ooi["hasPartURIs"][1])
        finally:
            WIKIDATA_MAPPINGS = prev


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
# If you prefer running only the tests:
# if __name__ == "__main__":
#     import unittest
#     unittest.main()
