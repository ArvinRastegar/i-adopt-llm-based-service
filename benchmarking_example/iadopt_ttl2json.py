#!/usr/bin/env python
"""
iadopt_ttl2json.py  ·  2025-06-04
------------------------------------

• Downloads C1…C30.ttl (or a numeric subset via CLI)
• Converts each to the compact JSON layout from the most-recent examples
• Saves every variable as its *own* JSON file in the absolute path requested
• Writes a combined 'all_variables.json' to the same folder for convenience
• Ships with unittests that exercise every code path
"""

from __future__ import annotations
import json, re, sys, textwrap
from pathlib import Path
from typing import Any, Dict, List

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


# --------------------------------------------------------------------------- #
# 3️⃣  Helper utilities
# --------------------------------------------------------------------------- #
def _label(g: Graph, node: URIRef | None) -> str | None:
    lab = g.value(node, RDFS_NS.label) if node else None
    return str(lab) if isinstance(lab, Literal) else None


def _list_labels(g: Graph, nodes: List[URIRef]) -> List[str]:
    return [_label(g, n) for n in nodes if _label(g, n)]


def _collapse_issue(url: str) -> str:
    m = re.search(r"/issues/(\d+)", url)
    return f"issue {m.group(1)}" if m else url


# Helper: serialise an entity *or* system node
def _entity_representation(g: Graph, node: URIRef | None) -> str | Dict[str, Any] | None:
    """Return either a plain label or a dict describing a (a)symmetric system."""
    if node is None:
        return None

    if (node, RDF.type, IOP.AsymmetricSystem) in g:
        return {
            "AsymmetricSystem": _label(g, node),
            "hasSource": _label(g, g.value(node, IOP.hasSource)),
            "hasTarget": _label(g, g.value(node, IOP.hasTarget)),
        }

    if (node, RDF.type, IOP.SymmetricSystem) in g:
        return {
            "SymmetricSystem": _label(g, node),
            "hasPart": _list_labels(g, list(g.objects(node, IOP.hasPart))),
        }

    # plain entity
    return _label(g, node)


# --------------------------------------------------------------------------- #
# 4️⃣  Core RDF→JSON mapping                                    #
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

    # -- links that may be entity *or* system --------------------------------
    for predicate, key in [
        (IOP.hasMatrix, "hasMatrix"),
        (IOP.hasObjectOfInterest, "hasObjectOfInterest"),
        (IOP.hasContextObject, "hasContextObject"),
    ]:
        if node := g.value(root, predicate):
            j[key] = _entity_representation(g, node)

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
# 5️⃣  I/O helpers
# --------------------------------------------------------------------------- #
def _download(url: str, timeout: int = 15) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def convert_one(n: int) -> Dict[str, Any]:
    ttl = _download(f"{BASE_URL}/C{n}.ttl")
    data = parse_variable(ttl)
    (OUTDIR / f"C{n}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"✓  C{n}.json")
    return data


# --------------------------------------------------------------------------- #
# 6️⃣  Command-line interface
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = range(int(sys.argv[1]), int(sys.argv[1]) + 1) if len(sys.argv) > 1 and sys.argv[1].isdigit() else range(1, 31)
    all_data = [convert_one(i) for i in rng]
    (OUTDIR / "all_variables.json").write_text(json.dumps(all_data, indent=2, ensure_ascii=False))
    print(f"✓  all_variables.json  ({len(all_data)} entries)  → {OUTDIR}\n")

    # run tests after conversion
    import unittest, warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)  # rdflib noise
    res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(__name__))
    if not res.wasSuccessful():
        sys.exit("❌ tests failed")
    print("🎉  All tests passed")


# --------------------------------------------------------------------------- #
# 7️⃣  Built-in test-suite
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

    def test_matrix_as_system(self):
        d = parse_variable(self.TTL_MATRIX_SYS)
        self.assertIsInstance(d["hasMatrix"], dict)
        self.assertEqual(d["hasMatrix"]["AsymmetricSystem"], "veg→soil")
        self.assertEqual(d["hasMatrix"]["hasSource"], "vegetation")
        self.assertEqual(d["hasMatrix"]["hasTarget"], "soil")


class TestMapping(unittest.TestCase):
    """Covers every branch."""

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

    def test_simple_mapping(self):
        d = parse_variable(self.TTL_SIMPLE)
        self.assertEqual(d["hasMatrix"], "atmosphere")
        self.assertEqual(d["hasProperty"], "area fraction")
        self.assertEqual(d["hasObjectOfInterest"], "cloud")
        self.assertNotIn("hasConstraint", d)

    def test_asymmetric_system_mapping(self):
        d = parse_variable(self.TTL_ASYM)
        self.assertIn("hasConstraint", d)
        constraints = d["hasConstraint"]
        self.assertIsInstance(constraints, list)
        # find the one with label "nearest"
        c = next((item for item in constraints if item.get("label") == "nearest"), None)
        self.assertIsNotNone(c, "should find a constraint with label 'nearest'")
        # and assert its `on` value starts with the hasTarget prefix
        self.assertTrue(c["on"].startswith("hasTarget:"), f"unexpected on: {c['on']}")
        self.assertEqual(d["hasObjectOfInterest"]["AsymmetricSystem"], "patch system")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
# if __name__ == "__main__":
#     import unittest

#     unittest.main()
