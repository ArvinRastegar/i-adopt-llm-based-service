#!/usr/bin/env python
"""The I-ADOPT decision trees, patterns and constraint types as data.

EXPLORATORY side experiment (guided decomposition, started 2026-11-19). Not part of the official
experiment: nothing it produces may enter an official ranking, campaign, results table or the
paper. Plan and results: `paper/tracking/GUIDED_DECOMPOSITION_20261119.md` (private repository).

The upstream rulebook (github.com/mabablue/I-ADOPT-patterns-playground) was written for people:
nine Mermaid flowcharts, about a hundred small pattern files and a CSV of constraint types. This
module reads one pinned version of it into three JSON files that the experiments use, so that no
run depends on the ignored local clone:

    ./rulebook.py --build --patterns-repo ../../../external/I-ADOPT-patterns-playground
    ./rulebook.py --check --patterns-repo ../../../external/I-ADOPT-patterns-playground

Pattern `example` and `gh_issue` fields are dropped at build time: several name corpus variables
(pattern 132 names BMI, a held-out variable), and nothing built here may carry an answer.

Isolation as for the sibling runners: lives in `few-shot-selection/`, which no artifact collector
walks; writes only `few-shot-selection/guided/data/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
UPSTREAM = "https://github.com/mabablue/I-ADOPT-patterns-playground"
TREE_NAMES = tuple(f"DT{n}" for n in range(9))

# Corrections to the upstream trees, applied at build time and stored with the data. Each one is
# OUR reading, not upstream's; the reason is given so a reader can disagree.
TREE_CORRECTIONS = (
    {"tree": "DT3", "node": "F", "set": {"kind": "pattern", "pattern": 114, "text": "RATIOLESS DERIVED"},
     "reason": "Upstream labels this leaf 'DT1' and links DT2.mmd. Following the link loops "
               "(DT2 -> DT3 -> DT2) and DT1 is the tree for unitless ratios. Pattern 114 "
               "(ratioless derived) is defined for exactly this branch, appears in DT7, and its "
               "own stored route ends here (Ratio(F) -> PropOfProp(F))."},
)

# Kind by pattern number. Upstream marks 44 (reference area) as an object-of-interest pattern,
# but DT8 and the example tags use it as a matrix pattern.
KIND_RANGES = ((31, 39, "object"), (40, 45, "matrix"), (50, 50, "context"), (60, 99, "constraint"),
               (100, 100, "system"), (101, 142, "property"), (200, 209, "constraint"))

PATTERN_KEYS = ("type", "subtype", "suptype", "name", "definition", "comment", "design",
                "ObjectOfInterest", "Matrix", "required_constraint", "constrained_component",
                "constrained_components", "path", "alt_path", "altpath", "PDP", "ODP", "MDP", "CT")
NULLS = {"", "null", "~", "[]", "none"}


# --- trees -------------------------------------------------------------------------------

def _clean(text: str) -> tuple[str, str, str]:
    """Strip Mermaid markup from a node's text.

    Args: text: raw node text with <br> and <a href> markup.
    Returns: (plain text, link target file name or "", link label or "").
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    link = re.search(r"<a\s+href='([^']*)'\s*>\s*([^<]*)", text)
    target = link.group(1).rstrip("/").split("/")[-1] if link else ""
    label = link.group(2).strip() if link else ""
    plain = re.sub(r"<br\s*/?>", " ", text)
    plain = re.sub(r"<a\s+href='[^']*'\s*>", " ", plain)
    plain = re.sub(r"</a[>\]]?", " ", plain)
    return re.sub(r"\s+", " ", plain).strip(), target, label


def _node(token: str) -> tuple[str, str | None, str | None]:
    """Split a Mermaid node token into id, shape and raw text.

    Args: token: such as `B{Is it length?}:::white`, `N[QUANTITY]` or a bare `H`.
    Returns: (id, shape "question" | "box" | "round" | None, raw text or None).
    Raises: ValueError when the token is not a node.
    Side Effects: none. Deterministic.
    """
    token = re.sub(r":::\w+\s*$", "", token.strip()).strip()
    match = re.match(r"^([A-Za-z0-9_]+)\s*(.*)$", token, re.S)
    if not match:
        raise ValueError(f"not a Mermaid node: {token!r}")
    ident, rest = match.group(1), match.group(2).strip()
    if not rest:
        return ident, None, None
    shapes = {"{": ("}", "question"), "[": ("]", "box"), "(": (")", "round")}
    if rest[0] not in shapes or not rest.endswith(shapes[rest[0]][0]):
        raise ValueError(f"unreadable Mermaid node: {token!r}")
    return ident, shapes[rest[0]][1], rest[1:-1]


def parse_tree(text: str) -> dict[str, Any]:
    """Read one Mermaid flowchart into nodes and labelled edges.

    Args: text: the content of a `DTn.mmd` file.
    Returns: {"title", "start", "nodes": {id: {"kind", "text", ...}}, "edges": [{"from", "to",
        "label"}]}. Kinds: question, pattern (with "pattern": number), jump (with "tree"),
        start, other.
    Raises: ValueError when an edge line cannot be read.
    Side Effects: none. Deterministic.
    """
    title = (re.search(r"^title:\s*(.+)$", text, re.M) or [None, ""])[1].strip()
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    start = None
    for line in text.splitlines():
        if "-->" not in line or line.lstrip().startswith("%%"):
            continue
        pipe = re.match(r"^(?P<src>.+?)\s*-->\s*\|(?P<label>[^|]*)\|\s*(?P<dst>.+)$", line)
        dash = re.match(r"^(?P<src>[^-]+?)\s*--\s*(?P<label>[^>]+?)\s*-->\s*(?P<dst>.+)$", line)
        plain = re.match(r"^(?P<src>.+?)\s*-->\s*(?P<dst>.+)$", line)
        edge = pipe or dash or plain
        if edge is None:
            raise ValueError(f"unreadable Mermaid edge: {line!r}")
        label = edge.groupdict().get("label") or ""
        ends = []
        for token in (edge.group("src"), edge.group("dst")):
            ident, shape, raw = _node(token)
            if raw is not None:
                plain_text, target, link_label = _clean(raw)
                info: dict[str, Any] = {"kind": "question" if shape == "question" else "other",
                                        "text": plain_text}
                if shape != "question":
                    number = re.search(r"VDP\s*(\d+)", link_label) or re.search(r"VDP\s*(\d+)", plain_text)
                    jump = re.fullmatch(r"DT(\d+)", link_label) or re.fullmatch(r"DT(\d+)\.mmd", target)
                    if number:
                        info.update(kind="pattern", pattern=int(number.group(1)),
                                    text=re.sub(r"\s*VDP\s*\d+\s*", " ", plain_text).strip())
                    elif jump:
                        info.update(kind="jump", tree=f"DT{jump.group(1)}",
                                    text=re.sub(r"\s*DT\d+\s*$", "", plain_text).strip())
                    if target:
                        info["link"] = target
                        info["link_label"] = link_label
                nodes.setdefault(ident, info)
            ends.append(ident)
        if start is None:
            start = ends[0]
        edges.append({"from": ends[0], "to": ends[1], "label": re.sub(r"\s+", " ", label).strip()})
    for ident in {e["from"] for e in edges} | {e["to"] for e in edges}:
        nodes.setdefault(ident, {"kind": "other", "text": ""})
    if start and nodes[start]["kind"] == "other":
        nodes[start]["kind"] = "start"
    return {"title": title, "start": start, "nodes": nodes, "edges": edges}


def tree_slips(name: str, tree: dict[str, Any]) -> list[str]:
    """List places where a tree's link and its label disagree.

    Args: name: tree name; tree: from `parse_tree`.
    Returns: one sentence per disagreement.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    slips = []
    for ident, node in sorted(tree["nodes"].items()):
        link, label = node.get("link", ""), node.get("link_label", "")
        stem = link.rsplit(".", 1)[0]
        named = re.search(r"\b(?:VDP|DT)\s*\d+\b", label)
        valid = re.fullmatch(r"(?:VDP|DT)\d+", stem)
        if link and ((named and named.group(0).replace(" ", "") != stem) or not valid):
            slips.append(f"{name} node {ident}: labelled {label}, links {link}")
    for edge in tree["edges"]:
        if re.match(r"^(true|false):", edge["label"]):
            slips.append(f"{name} edge {edge['from']} -> {edge['to']}: answer written as {edge['label']!r}")
    return slips


def build_trees(repo: Path) -> dict[str, Any]:
    """Parse the nine upstream trees and apply our recorded corrections.

    Args: repo: a checkout of the upstream patterns repository.
    Returns: {"source", "trees", "slips", "corrections"}.
    Raises: OSError when a tree file is missing; ValueError when one cannot be read.
    Side Effects: reads the tree files and asks git for the checkout's commit.
    """
    trees, slips = {}, []
    for name in TREE_NAMES:
        tree = parse_tree((repo / "Decision_trees" / f"{name}.mmd").read_text(encoding="utf-8"))
        slips += tree_slips(name, tree)
        trees[name] = tree
    for tree in trees.values():
        for edge in tree["edges"]:
            edge["label"] = re.sub(r"^(true|false):.*$", r"\1", edge["label"])
    for fix in TREE_CORRECTIONS:
        node = trees[fix["tree"]]["nodes"][fix["node"]]
        node.pop("tree", None)
        node.update(fix["set"])
    for tree in trees.values():
        for node in tree["nodes"].values():
            node.pop("link", None)
            node.pop("link_label", None)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    return {"source": {"repository": UPSTREAM, "commit": commit}, "trees": trees, "slips": slips,
            "corrections": [dict(c) for c in TREE_CORRECTIONS]}


def options(tree: dict[str, Any], node: str) -> list[dict[str, str]]:
    """The labelled answers leaving a question, in file order.

    Args: tree: one parsed tree; node: a node id.
    Returns: the outgoing edges.
    Raises: nothing.
    Side Effects: none.
    """
    return [e for e in tree["edges"] if e["from"] == node]


def routes_to(trees: dict[str, Any], pattern: int, start_tree: str = "DT0") -> list[list[dict[str, str]]]:
    """Every way of answering the questions that ends at one property pattern.

    Args: trees: the "trees" mapping; pattern: pattern number; start_tree: where a walk begins.
    Returns: routes; each is a list of {"tree", "node", "question", "answer"} steps.
    Raises: nothing; an unreachable pattern gives [].
    Side Effects: none. Deterministic.
    """
    found: list[list[dict[str, str]]] = []

    def visit(tree_name: str, node: str, steps: list[dict[str, str]], seen: frozenset) -> None:
        """Follow one branch depth-first.

        Args: tree_name, node: where we are; steps: answers so far; seen: visited (tree, node).
        Returns: nothing.
        Raises: nothing.
        Side Effects: appends complete routes to `found`.
        """
        if (tree_name, node) in seen:
            return
        seen = seen | {(tree_name, node)}
        tree = trees[tree_name]
        info = tree["nodes"][node]
        if info["kind"] == "pattern":
            if info["pattern"] == pattern:
                found.append(steps)
            return
        if info["kind"] == "jump":
            target = info["tree"]
            visit(target, trees[target]["start"], steps, seen)
            return
        for edge in options(tree, node):
            step = steps
            if info["kind"] == "question":
                step = steps + [{"tree": tree_name, "node": node, "question": info["text"],
                                 "answer": edge["label"]}]
            visit(tree_name, edge["to"], step, seen)

    visit(start_tree, trees[start_tree]["start"], [], frozenset())
    return found


def implied_patterns(trees: dict[str, Any], pattern: int) -> list[int]:
    """Object and matrix patterns that DT5 and DT7 attach to a property pattern.

    Args: trees: the "trees" mapping; pattern: a property pattern number.
    Returns: sorted pattern numbers reached by unlabelled arrows from that leaf.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out: set[int] = set()
    for name in ("DT5", "DT7"):
        tree = trees[name]
        for ident, node in tree["nodes"].items():
            if node["kind"] == "pattern" and node.get("pattern") == pattern:
                for edge in options(tree, ident):
                    target = tree["nodes"][edge["to"]]
                    if target["kind"] == "pattern" and not edge["label"]:
                        out.add(target["pattern"])
    return sorted(out)


# --- patterns ----------------------------------------------------------------------------

def _fields(text: str) -> dict[str, str]:
    """Read the known top-level fields of a pattern file, YAML or Markdown, leniently.

    Args: text: file content. Two upstream files are not valid YAML, so no YAML parser is used.
    Returns: {field: value}; folded `>-` blocks are joined; trailing `# comments` are removed.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out: dict[str, str] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"^(?:- )?([A-Za-z_]+):\s*(.*)$", lines[index])
        index += 1
        if not match or match.group(1) not in PATTERN_KEYS:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value in (">-", ">", "|", "|-"):
            block = []
            while index < len(lines) and (lines[index].startswith("  ") or not lines[index].strip()):
                block.append(lines[index].strip())
                index += 1
            value = " ".join(part for part in block if part)
        value = re.sub(r"\s+#\s.*$", "", value).strip().strip('"').strip()
        value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
        out.setdefault(key, value)
    return out


def _shape(value: str | None) -> str | None:
    """Normalise an upstream shape note to one of our four shapes.

    Args: value: such as "Asymmetric System (Num/Denom)", "Entity" or None.
    Returns: "single", "ratio", "symmetric", "source_target", "variable" or None when unstated.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    text = (value or "").strip().lower()
    if text in NULLS:
        return None
    if "num" in text and "denom" in text:
        return "ratio"
    if "source" in text and "target" in text:
        return "source_target"
    if "symmetric" in text and "asymmetric" not in text:
        return "symmetric"
    if text == "variable":
        return "variable"
    return "single"


def kind_of(number: int) -> str:
    """The component a pattern number belongs to.

    Args: number: pattern number.
    Returns: property, object, matrix, constraint, context, system or other.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    for low, high, kind in KIND_RANGES:
        if low <= number <= high:
            return kind
    return "other"


def build_patterns(repo: Path, trees: dict[str, Any]) -> dict[str, Any]:
    """Collect every numbered pattern with what it demands of a decomposition.

    Args: repo: upstream checkout; trees: the "trees" mapping, for names of file-less leaves.
    Returns: {number as text: {"number", "kind", "name", "definition", "design", "object_shape",
        "matrix_shape", "required_constraint", "constrained_component", "sources"}}.
    Raises: OSError when the pattern folder is missing.
    Side Effects: reads the pattern files. `example` and `gh_issue` are never read into the result.
    """
    merged: dict[int, dict[str, Any]] = {}
    files = sorted((repo / "pattern").glob("*.yaml")) + sorted((repo / "pattern" / "yaml").glob("*.yaml"))
    files += sorted((repo / "pattern").glob("*.md"))
    rank = {"VDP": 0, "PDP": 1, "ODP": 1, "MDP": 1, "CT": 1}
    for path in sorted(files, key=lambda p: (rank.get(re.sub(r"\d.*", "", p.stem), 2), p.suffix != ".yaml", str(p))):
        match = re.fullmatch(r"(VDP|PDP|ODP|MDP|CT)(\d+)", path.stem)
        if not match:
            continue
        number = int(match.group(2))
        fields = _fields(path.read_text(encoding="utf-8"))
        entry = merged.setdefault(number, {"number": number, "kind": kind_of(number), "sources": []})
        entry["sources"].append(str(path.relative_to(repo)))
        for key, target in (("name", "name"), ("definition", "definition"), ("comment", "definition"),
                            ("design", "design"), ("required_constraint", "required_constraint"),
                            ("constrained_component", "constrained_component"),
                            ("constrained_components", "constrained_component")):
            value = fields.get(key, "")
            if value.lower() not in NULLS and not entry.get(target):
                entry[target] = value
        if match.group(1) == "VDP":
            for key, target in (("ObjectOfInterest", "object_shape"), ("Matrix", "matrix_shape")):
                if key in fields and target not in entry:
                    entry[target] = _shape(fields[key])
            if fields.get("ODP", "").lower() not in NULLS and "object_pattern" not in entry:
                entry["object_pattern"] = int(re.sub(r"\D", "", fields["ODP"]))
    for tree in trees.values():
        for node in tree["nodes"].values():
            if node["kind"] == "pattern":
                entry = merged.setdefault(node["pattern"], {"number": node["pattern"],
                                                           "kind": kind_of(node["pattern"]), "sources": []})
                entry.setdefault("tree_name", node["text"].lower())
    fixed = {33: "single", 34: "symmetric", 35: "ratio", 36: "source_target", 45: "source_target"}
    for number, entry in merged.items():
        entry["name"] = (entry.get("name") or entry.get("tree_name") or "").strip().lower()
        entry.pop("tree_name", None)
        for key in ("definition", "design", "required_constraint", "constrained_component"):
            entry[key] = entry.get(key) or None
        if entry["kind"] == "property" and entry.get("object_shape") is None and "object_pattern" in entry:
            entry["object_shape"] = fixed.get(entry["object_pattern"])
        entry.pop("object_pattern", None)
        entry["object_shape"] = entry.get("object_shape") if entry["kind"] == "property" else None
        entry["matrix_shape"] = entry.get("matrix_shape") if entry["kind"] == "property" else None
        if entry["kind"] in ("object", "matrix"):
            entry["shape"] = fixed.get(number, "single")
    return {str(number): merged[number] for number in sorted(merged)}


def build_constraint_types(repo: Path) -> list[dict[str, Any]]:
    """Read the constraint-type table.

    Args: repo: upstream checkout.
    Returns: rows with number, category, name, definition, observation and the parts each type
        may attach to.
    Raises: OSError when the table is missing.
    Side Effects: reads the CSV.
    """
    rows = []
    with (repo / "Decision_trees" / "Constraint_patterns.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "number": int(row["Number"]), "category": row["Category"].strip(),
                "name": row["Name"].strip(), "definition": re.sub(r"\s+", " ", row["Definition"]).strip(),
                "observation": row["Type of observation"].strip(),
                "targets": [part for part, column in (("property", "Property"),
                                                      ("object of interest", "Object of Interest"),
                                                      ("matrix", "Matrix"),
                                                      ("statistical modifier", "Statistical Modifier"))
                            if row[column].strip().upper() == "TRUE"]})
    return rows


# --- files -------------------------------------------------------------------------------

def dump(value: Any) -> str:
    """Serialise data files the same way every time.

    Args: value: JSON data.
    Returns: indented, key-sorted JSON text ending in a newline, non-ASCII kept.
    Raises: TypeError for values JSON cannot hold.
    Side Effects: none. Deterministic.
    """
    return json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def build(repo: Path) -> dict[str, str]:
    """Build the three data files' text from an upstream checkout.

    Args: repo: upstream checkout.
    Returns: {file name: text}.
    Raises: as the three builders.
    Side Effects: reads the checkout.
    """
    trees = build_trees(repo)
    return {"trees.json": dump(trees),
            "patterns.json": dump({"source": trees["source"], "patterns": build_patterns(repo, trees["trees"])}),
            "constraint_types.json": dump({"source": trees["source"], "types": build_constraint_types(repo)})}


def load(name: str) -> Any:
    """Load one built data file.

    Args: name: "trees", "patterns" or "constraint_types".
    Returns: the stored JSON.
    Raises: OSError when the file has not been built.
    Side Effects: reads the file.
    """
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Build the data files, or check them against a checkout.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: 0, or 1 when --check finds a difference.
    Raises: SystemExit on bad arguments.
    Side Effects: --build writes `data/*.json`.
    """
    parser = argparse.ArgumentParser(description="I-ADOPT rulebook as data (exploratory).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--patterns-repo", type=Path, required=True)
    args = parser.parse_args(argv)
    files = build(args.patterns_repo.resolve())
    status = 0
    for name, text in files.items():
        path = DATA / name
        if args.build:
            DATA.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(HERE)} ({len(text):,} characters)")
        elif not path.exists() or path.read_text(encoding="utf-8") != text:
            print(f"DIFFERS: {name}")
            status = 1
    if args.check and status == 0:
        print("data files match the checkout")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
