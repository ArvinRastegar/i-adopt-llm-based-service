#!/usr/bin/env python
"""
gt_json_maker.py · 2025-11-27
------------------------------------

• Reads ALL TTL files from a local folder (arbitrary names)
• Converts each TTL variable to compact JSON
• Extracts:
    - label
    - definition (skos:definition)
    - comment
    - hasProperty (+URI if TTL provides one)
    - hasMatrix (+URI if TTL provides one)
    - hasObjectOfInterest (+URI if TTL provides one)
    - hasContextObject (+URI if TTL provides one)
    - hasStatisticalModifier (+URI)
    - hasConstraint (cleaned label, correct ON target)
• URI enrichment ONLY for Wikidata URIs present in TTL
• All Wikidata URIs normalized to:
      https://www.wikidata.org/wiki/Qxxxx
• Writes one JSON per TTL + all_variables.json
"""

from __future__ import annotations
from rdflib import Graph, Namespace, RDF, RDFS, URIRef
from pathlib import Path
import json, sys, re
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Output Folder
# --------------------------------------------------------------------------- #
OUTDIR = Path("/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service/benchmarking_example/data/Json_preferred")
OUTDIR.mkdir(parents=True, exist_ok=True)

TTL_INPUT_DIR = Path(
    "/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service/benchmarking_example/data/variables_ttl_files"
)


# --------------------------------------------------------------------------- #
# Namespaces
# --------------------------------------------------------------------------- #
IOP = Namespace("https://w3id.org/iadopt/ont/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
RDFS_NS = RDFS


# --------------------------------------------------------------------------- #
# URI normalization helper
# --------------------------------------------------------------------------- #
def _maybe_uri(node: Any) -> Optional[str]:
    """
    Normalize ANY Wikidata URI to canonical format:
        https://www.wikidata.org/wiki/Qxxxx

    Accepts:
        - entity/Qxxxx
        - wiki/Qxxxx
        - malformed HTTP forms
        - raw "wikidata.org/entity/Qxxx"
    """
    if not isinstance(node, URIRef):
        return None

    raw = str(node).strip()

    # fix common protocol errors
    raw = re.sub(r"^hhttps://", "https://", raw)
    raw = re.sub(r"^httpss://", "https://", raw)
    raw = raw.replace("http://", "https://")

    # inject full domain if missing
    if raw.startswith("wikidata.org"):
        raw = "https://" + raw
    if raw.startswith("www.wikidata.org"):
        raw = "https://" + raw

    # extract Q-id
    m = re.search(r"(Q[0-9]+)", raw)
    if not m:
        return None

    qid = m.group(1)
    return f"https://www.wikidata.org/wiki/{qid}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _label(g: Graph, node: URIRef | None):
    if not node:
        return None
    lab = g.value(node, RDFS_NS.label)
    return str(lab) if lab else None


def _clean_constraint_label(label: str) -> str:
    """Remove leading 'XXX:' if present."""
    if not label:
        return label
    if ":" in label:
        return label.split(":", 1)[1].strip()
    return label.strip()


# --------------------------------------------------------------------------- #
# Entity / System Representation
# --------------------------------------------------------------------------- #
def _entity_representation(g: Graph, node: URIRef | None):
    if node is None:
        return None

    def name(n):
        return _label(g, n) or str(n).split("/")[-1]

    # Asymmetric
    if (node, RDF.type, IOP.AsymmetricSystem) in g:
        source = g.value(node, IOP.hasSource) or g.value(node, IOP.hasNumerator)
        target = g.value(node, IOP.hasTarget) or g.value(node, IOP.hasDenominator)

        if source == target and source is not None:
            return {"SymmetricSystem": name(node), "hasPart": [name(source)]}

        return {
            "AsymmetricSystem": name(node),
            "hasSource": name(source) if source else None,
            "hasTarget": name(target) if target else None,
        }

    # Symmetric
    if (node, RDF.type, IOP.SymmetricSystem) in g:
        parts = list(g.objects(node, IOP.hasPart))
        return {"SymmetricSystem": name(node), "hasPart": [name(p) for p in parts]}

    return name(node)


# --------------------------------------------------------------------------- #
# Root finder
# --------------------------------------------------------------------------- #
def _find_variable_root(g: Graph):
    roots = set()

    for s in g.subjects(RDF.type, IOP.Variable):
        roots.add(s)
    if roots:
        return list(roots)[0]

    for s, _, o in g.triples((None, RDF.type, None)):
        if isinstance(o, URIRef) and str(o).rstrip("/#").endswith("Variable"):
            roots.add(s)
    if roots:
        return list(roots)[0]

    for s in g.subjects(RDFS.label, None):
        if g.value(s, SKOS.definition) or g.value(s, RDFS.comment):
            roots.add(s)
    if roots:
        return list(roots)[0]

    raise ValueError("No Variable root found.")


# --------------------------------------------------------------------------- #
# Definition extractor
# --------------------------------------------------------------------------- #
def _get_definition(g: Graph, root):
    DCT = Namespace("http://purl.org/dc/terms/")

    for pred in (SKOS.definition, DCT.description, RDFS.comment):
        val = g.value(root, pred)
        if val:
            return str(val).strip()

    return None


# --------------------------------------------------------------------------- #
# MAIN TTL → JSON CONVERTER
# --------------------------------------------------------------------------- #
def parse_variable(ttl: str) -> Dict[str, Any]:
    g = Graph()
    g.parse(data=ttl, format="turtle")

    root = _find_variable_root(g)

    result = {
        "label": _label(g, root),
        "definition": _get_definition(g, root),
        "comment": (str(g.value(root, RDFS.comment) or "").strip() or None),
    }

    # ------------------------- Property -------------------------------------
    if prop := g.value(root, IOP.hasProperty):
        result["hasProperty"] = _label(g, prop)
        if u := _maybe_uri(prop):
            result["hasPropertyURI"] = u

    # ------------------- Statistical Modifier (NEW) --------------------------
    if stat := g.value(root, IOP.hasStatisticalModifier):
        result["hasStatisticalModifier"] = _label(g, stat)
        if u := _maybe_uri(stat):
            result["hasStatisticalModifierURI"] = u

    # --------------- Matrix / OOI / Context Objects -------------------------
    for pred, key in [
        (IOP.hasMatrix, "hasMatrix"),
        (IOP.hasObjectOfInterest, "hasObjectOfInterest"),
        (IOP.hasContextObject, "hasContextObject"),
    ]:
        nodes = list(g.objects(root, pred))
        if not nodes:
            continue

        if len(nodes) == 1:
            node = nodes[0]
            rep = _entity_representation(g, node)
            result[key] = rep
            if isinstance(node, URIRef):
                if u := _maybe_uri(node):
                    result[key + "URI"] = u
        else:
            result[key] = []
            for node in nodes:
                result[key].append(_entity_representation(g, node))

    # ---------------------------- Constraints --------------------------------
    constraints = list(g.objects(root, IOP.hasConstraint))
    if constraints:
        out = []
        for c in constraints:
            raw_label = _label(g, c)
            clean = _clean_constraint_label(raw_label)

            target_node = g.value(c, IOP.constrains)

            # Guarantee a target:
            if target_node is None:
                target_node = (
                    g.value(root, IOP.hasStatisticalModifier)
                    or g.value(root, IOP.hasObjectOfInterest)
                    or g.value(root, IOP.hasProperty)
                    or g.value(root, IOP.hasMatrix)
                    or root
                )

            target_label = _label(g, target_node)
            if not target_label:
                target_label = str(target_node).split("/")[-1]

            out.append({"label": clean, "on": target_label})

        result["hasConstraint"] = out

    # ---------------------- Clean output ------------------------------------
    return {k: v for k, v in result.items() if v is not None}


# --------------------------------------------------------------------------- #
# File conversion
# --------------------------------------------------------------------------- #
def convert_file(ttl_path: Path) -> Dict[str, Any]:
    ttl_text = ttl_path.read_text(encoding="utf-8")
    data = parse_variable(ttl_text)

    out_path = OUTDIR / f"{ttl_path.stem}.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✓ {ttl_path.stem}.json generated")
    return data


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ttl_files = sorted(TTL_INPUT_DIR.rglob("*.ttl"))
    if not ttl_files:
        print("❌ No TTL files found.")
        sys.exit(1)

    log_path = OUTDIR / "processing_log.txt"
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    all_data = []
    failed = []

    log("=== I-ADOPT TTL → JSON PROCESSING LOG ===\n")

    for f in ttl_files:
        try:
            d = convert_file(f)
            all_data.append(d)
            log(f"✓ SUCCESS: {f}")
        except Exception as e:
            log(f"❌ FAILED: {f}\n      ERROR: {e}")
            failed.append((f, e))

    if all_data:
        combined = OUTDIR / "all_variables.json"
        combined.write_text(json.dumps(all_data, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"\n✓ all_variables.json written with {len(all_data)} variables.\n")

    if failed:
        log("⚠️ FAILED TTL FILES:")
        for f, e in failed:
            log(f"  - {f}: {e}")

    log_path.write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
