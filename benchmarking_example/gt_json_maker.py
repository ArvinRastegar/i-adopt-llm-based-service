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
    - hasConstraint (cleaned label)
• URI enrichment ONLY for Wikidata URIs present in TTL
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
# Helpers
# --------------------------------------------------------------------------- #
def _label(g: Graph, node: URIRef | None):
    """Return the rdfs:label value as a Python string."""
    if not node:
        return None
    lab = g.value(node, RDFS_NS.label)
    return str(lab) if lab else None


def _clean_constraint_label(label: str) -> str:
    """Remove prefix before ': '."""
    if label and ": " in label:
        return label.split(": ", 1)[0].strip()
    return label


def _maybe_uri(node: Any) -> Optional[str]:
    """Return Wikidata URI only if node is a Wikidata URIRef."""
    if isinstance(node, URIRef):
        uri = str(node)
        if uri.startswith("https://www.wikidata.org/wiki/"):
            return uri
    return None


# --------------------------------------------------------------------------- #
# Entity & System Representation
# --------------------------------------------------------------------------- #
def _entity_representation(g: Graph, node: URIRef | None):
    if node is None:
        return None

    # Asymmetric system
    if (node, RDF.type, IOP.AsymmetricSystem) in g:
        return {
            "AsymmetricSystem": _label(g, node),
            "hasSource": _label(g, g.value(node, IOP.hasSource)),
            "hasTarget": _label(g, g.value(node, IOP.hasTarget)),
        }

    # Symmetric system
    if (node, RDF.type, IOP.SymmetricSystem) in g:
        parts_nodes = list(g.objects(node, IOP.hasPart))
        return {
            "SymmetricSystem": _label(g, node),
            "hasPart": [_label(g, p) for p in parts_nodes],
        }

    # Plain entity → return label
    return _label(g, node)


# --------------------------------------------------------------------------- #
# Core TTL → JSON Parser
# --------------------------------------------------------------------------- #
def parse_variable(ttl: str) -> Dict[str, Any]:
    g = Graph()
    g.parse(data=ttl, format="turtle")

    # Find root variable
    roots = list(g.subjects(RDF.type, IOP.Variable))
    if not roots:
        raise ValueError("No iop:Variable found in TTL file.")
    root = roots[0]

    result: Dict[str, Any] = {
        "label": _label(g, root),
        "definition": str(g.value(root, SKOS.definition) or "").strip() or None,
        "comment": str(g.value(root, RDFS_NS.comment) or "").strip() or None,
    }

    # --- Property (label + URI) --------------------------------------------
    if prop := g.value(root, IOP.hasProperty):
        result["hasProperty"] = _label(g, prop)
        if u := _maybe_uri(prop):
            result["hasPropertyURI"] = u

    # --- Matrix, OOI, Context ----------------------------------------------
    for pred, key in [
        (IOP.hasMatrix, "hasMatrix"),
        (IOP.hasObjectOfInterest, "hasObjectOfInterest"),
        (IOP.hasContextObject, "hasContextObject"),
    ]:
        node = g.value(root, pred)
        if node:
            val = _entity_representation(g, node)
            result[key] = val

            # Only add ...URI if node is a Wikidata URI
            if isinstance(val, str):
                if u := _maybe_uri(node):
                    result[f"{key}URI"] = u

    # --- Constraints --------------------------------------------------------
    constraints = list(g.objects(root, IOP.hasConstraint))
    if constraints:
        out = []
        for c in constraints:
            raw = _label(g, c)
            clean = _clean_constraint_label(raw)
            target = _label(g, g.value(c, IOP.constrains))
            out.append({"label": clean, "on": target})
        result["hasConstraint"] = out

    # Remove None values
    return {k: v for k, v in result.items() if v is not None}


# --------------------------------------------------------------------------- #
# File Conversion
# --------------------------------------------------------------------------- #
def convert_file(ttl_path: Path) -> Dict[str, Any]:
    ttl_text = ttl_path.read_text(encoding="utf-8")
    data = parse_variable(ttl_text)

    out_path = OUTDIR / f"{ttl_path.stem}.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✓ {ttl_path.stem}.json generated")
    return data


# --------------------------------------------------------------------------- #
# Main CLI
# --------------------------------------------------------------------------- #
def main():
    # Recursively find all TTL files
    ttl_files = sorted(TTL_INPUT_DIR.rglob("*.ttl"))
    if not ttl_files:
        print(f"❌ No TTL files found under {TTL_INPUT_DIR}")
        sys.exit(1)

    log_path = OUTDIR / "processing_log.txt"
    log_lines = []

    def log(msg: str):
        """Append to log list and print to terminal."""
        print(msg)
        log_lines.append(msg)

    all_data = []
    failed = []

    log("=== I-ADOPT TTL → JSON PROCESSING LOG ===\n")

    for f in ttl_files:
        try:
            data = convert_file(f)
            all_data.append(data)
            log(f"✓ SUCCESS: {f}")
        except Exception as e:
            error_msg = f"❌ FAILED: {f}\n      ERROR: {e}"
            log(error_msg)
            failed.append((f, e))

    # Write combined JSON only for successfully parsed variables
    if all_data:
        combined_path = OUTDIR / "all_variables.json"
        combined_path.write_text(json.dumps(all_data, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"\n✓ all_variables.json written with {len(all_data)} variables.\n")
    else:
        log("\n❌ No variables were successfully parsed, all failed.\n")

    # Summary of failures
    if failed:
        log("⚠️ FAILED TTL FILES:")
        for f, e in failed:
            log(f"  - {f}: {e}")

    # Save log file
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n📄 Log file written to: {log_path}\n")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
