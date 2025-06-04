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


# --------------------------------------------------------------------------- #
# 4️⃣  Core RDF→JSON mapping (unchanged API from v2)
# --------------------------------------------------------------------------- #
def parse_variable(ttl: str) -> Dict[str, Any]:
    g = Graph()
    g.parse(data=ttl, format="turtle")

    root_candidates = list(g.subjects(RDF.type, IOP.Variable))
    if not root_candidates:
        raise ValueError("No iop:Variable in supplied Turtle")
    root: URIRef = root_candidates[0]

    j: Dict[str, Any] = {
        "label": _label(g, root),
        "comment": str(g.value(root, RDFS_NS.comment, default="")).strip() or None,
    }

    # simple links
    for p, k in [
        (IOP.hasProperty, "hasProperty"),
        (IOP.hasMatrix, "hasMatrix"),
        (IOP.hasStatisticalModifier, "hasStatisticalModifier"),
        (IOP.hasContextObject, "hasContextObject"),
    ]:
        if o := g.value(root, p):
            j[k] = _label(g, o)

    # object of interest
    ooi = g.value(root, IOP.hasObjectOfInterest)
    if ooi:
        if (ooi, RDF.type, IOP.AsymmetricSystem) in g:
            j["hasObjectOfInterest"] = {
                "AsymmetricSystem": _label(g, ooi),
                "hasSource": _label(g, g.value(ooi, IOP.hasSource)),
                "hasTarget": _label(g, g.value(ooi, IOP.hasTarget)),
            }
        elif (ooi, RDF.type, IOP.SymmetricSystem) in g:
            j["hasObjectOfInterest"] = {
                "SymmetricSystem": _label(g, ooi),
                "hasPart": _list_labels(g, list(g.objects(ooi, IOP.hasPart))),
            }
        else:
            j["hasObjectOfInterest"] = _label(g, ooi)

    # constraints
    c_nodes = list(g.objects(root, IOP.hasConstraint))
    if c_nodes:
        cdict: Dict[str, Dict[str, str]] = {}
        for c in c_nodes:
            cname = _label(g, c)
            targ = g.value(c, IOP.constrains)
            targ_lbl = _label(g, targ)
            if ooi and (ooi, IOP.hasTarget, targ) in g:
                targ_lbl = f"hasTarget: {targ_lbl}"
            elif ooi and (ooi, IOP.hasSource, targ) in g:
                targ_lbl = f"hasSource: {targ_lbl}"
            cdict[cname] = {"constrains": targ_lbl}
        j["hasConstraint"] = cdict

    # optional issue link
    for pred, obj in g.predicate_objects(root):
        if str(pred).split("#")[-1] == "issue":
            j["issue"] = _collapse_issue(str(obj))
            break

    # prune None values
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
# 7️⃣  Built-in test-suite (unchanged from v2 except for path independence)
# --------------------------------------------------------------------------- #
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
        self.assertTrue(d["hasConstraint"]["nearest"]["constrains"].startswith("hasTarget:"))
        self.assertEqual(d["hasObjectOfInterest"]["AsymmetricSystem"], "patch system")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
# if __name__ == "__main__":
#     import unittest

#     unittest.main()
