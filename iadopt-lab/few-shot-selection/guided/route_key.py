#!/usr/bin/env python
"""The expert answer key for the tree routes, and its audit against the corpus gold.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.

The pattern authors tag every worked example in their public examples repository
(github.com/mabablue/I-ADOPT-examples-playground) with the patterns it uses, for example
"132 property: index" and "44 matrix: reference area". Each corpus variable's entry links its
corpus file, which matches it to one corpus record. The tags name the leaf every tree should
reach, so they serve as the answer key for a tree walk.

    gh api 'repos/mabablue/I-ADOPT-examples-playground/issues?state=all&per_page=100' --paginate \\
        --jq '.[] | select(.pull_request|not) | {number, title, state, updated_at,
              labels: [.labels[].name], body}' > issues.jsonl
    ./route_key.py --build --issues issues.jsonl --date 2026-11-19
    ./route_key.py --audit

The audit compares the tags and the pattern rules with the corpus gold. It prints gold-derived
detail, so its output belongs in the private record and never in this repository.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import rulebook  # noqa: E402

EXAMPLES = "https://github.com/mabablue/I-ADOPT-examples-playground"
LABEL_KINDS = {"property": "property", "object of interest": "object", "matrix": "matrix",
               "constraint": "constraint", "context": "context", "part": "part"}
FLAGS = ("to be agreed on", "to be changed")


def parse_labels(labels: list[str]) -> dict[str, Any]:
    """Sort an entry's tags into pattern numbers per part.

    Args: labels: tag texts such as "132 property: index" or "statistical modifier".
    Returns: {"property", "object", "matrix", "constraint", "context", "part": sorted numbers,
        "statistical_modifier": bool, "flags": [...], "names": {number: tag name},
        "other": unrecognised tags}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out: dict[str, Any] = {kind: [] for kind in LABEL_KINDS.values()}
    out.update(statistical_modifier=False, flags=[], names={}, other=[])
    for label in labels:
        text = label.strip()
        match = re.match(r"^(\d+)\s+([a-z ]+?):\s*(.+)$", text)
        if match and match.group(2) in LABEL_KINDS:
            number = int(match.group(1))
            out[LABEL_KINDS[match.group(2)]].append(number)
            out["names"][str(number)] = match.group(3).strip()
        elif text == "statistical modifier":
            out["statistical_modifier"] = True
        elif text in FLAGS:
            out["flags"].append(text)
        elif text != "corpus":
            out["other"].append(text)
    for kind in LABEL_KINDS.values():
        out[kind] = sorted(out[kind])
    return out


def corpus_path(body: str | None) -> str | None:
    """Find the corpus file an entry links.

    Args: body: the entry's text.
    Returns: the path below the corpus root, URL-decoded, or None.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    match = (re.search(r"https?://i-adopt\.github\.io/Corpus/([^\s)\]>\"]+)", body or "")
             or re.search(r"https?://github\.com/i-adopt/Corpus/(?:blob|tree)/[^/]+/([^\s)\]>\"]+)", body or ""))
    return urllib.parse.unquote(match.group(1)) if match else None


def build(issues: list[dict[str, Any]], records: list[dict[str, Any]], date: str) -> dict[str, Any]:
    """Match tagged entries to corpus records and build the answer key.

    Args: issues: entries from the GitHub API; records: canonical corpus records; date: the day
        the entries were read.
    Returns: {"snapshot": [...], "key": {"source", "date", "variables": {variable_id: {...}}}}.
    Raises: ValueError when a record has no entry or more than one.
    Side Effects: none. Deterministic for given inputs.
    """
    by_path = {record["source_path"]: record for record in records}
    by_file: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_file.setdefault(record["source_path"].split("/")[-1].lower(), []).append(record)
    snapshot, matched = [], {}
    for issue in sorted(issues, key=lambda item: item["number"]):
        path = corpus_path(issue.get("body"))
        snapshot.append({"number": issue["number"], "title": issue["title"], "state": issue["state"],
                         "updated_at": issue["updated_at"], "labels": sorted(issue["labels"]),
                         "corpus_path": path})
        if path is None:
            continue
        record = by_path.get(path)
        if record is None:
            same_name = by_file.get(path.split("/")[-1].lower(), [])
            record = same_name[0] if len(same_name) == 1 else None
        if record is not None:
            matched.setdefault(record["variable_id"], []).append(issue)
    problems = [r["source_path"] for r in records if len(matched.get(r["variable_id"], [])) != 1]
    if problems:
        raise ValueError(f"records without exactly one tagged entry: {problems}")
    variables = {}
    for record in records:
        issue = matched[record["variable_id"]][0]
        parsed = parse_labels(issue["labels"])
        variables[record["variable_id"]] = {"issue": issue["number"], "source_path": record["source_path"],
                                            "labels": sorted(issue["labels"]), **parsed}
    return {"snapshot": snapshot,
            "key": {"source": EXAMPLES, "date": date, "variables": variables}}


def load() -> dict[str, Any]:
    """Load the built answer key.

    Args: none.
    Returns: {variable_id: entry}.
    Raises: OSError when it has not been built.
    Side Effects: reads `data/route_key.json`.
    """
    return json.loads((rulebook.DATA / "route_key.json").read_text(encoding="utf-8"))["variables"]


# --- audit -------------------------------------------------------------------------------

def shape(value: Any) -> str:
    """The shape of a gold or predicted entity.

    Args: value: a lexical entity: a string or a system object.
    Returns: "empty", "single", "symmetric", "source_target" or "ratio".
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if isinstance(value, str):
        return "single" if value else "empty"
    if "SymmetricSystem" in value:
        return "symmetric"
    return "source_target" if "hasSource" in value else "ratio"


def names_of(value: Any) -> set[str]:
    """Every label by which a constraint may point at an entity.

    Args: value: a lexical entity.
    Returns: the entity's own label, its members and its joined system names.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if isinstance(value, str):
        return {value} if value else set()
    if "SymmetricSystem" in value:
        parts = list(value["hasPart"])
        return {value["SymmetricSystem"], " + ".join(sorted(parts)), *parts} - {""}
    if "hasSource" in value:
        pair, joiner = (value["hasSource"], value["hasTarget"]), " → "
    else:
        pair, joiner = (value["hasNumerator"], value["hasDenominator"]), " / "
    return {value["AsymmetricSystem"], joiner.join(pair), *pair} - {""}


def target_part(gold: dict[str, Any], on: str) -> str:
    """Which part of a decomposition a constraint points at.

    Args: gold: a six-field decomposition; on: the constraint's target label.
    Returns: "property", "statistical modifier", "object of interest", "matrix",
        "context object" or "unknown".
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if on == gold["hasProperty"]:
        return "property"
    if gold["hasStatisticalModifier"] and on == gold["hasStatisticalModifier"]:
        return "statistical modifier"
    for field, name in (("hasObjectOfInterest", "object of interest"), ("hasMatrix", "matrix"),
                        ("hasContextObject", "context object")):
        if on in names_of(gold[field]):
            return name
    return "unknown"


def constraint_type(label: str) -> str:
    """The type prefix of a constraint label.

    Args: label: such as "state: dissolved".
    Returns: the lower-cased text before the first colon, or "" when there is none.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return label.split(":", 1)[0].strip().lower() if ":" in label else ""


def audit_variable(gold: dict[str, Any], entry: dict[str, Any], patterns: dict[str, Any],
                   types: list[dict[str, Any]], trees: dict[str, Any]) -> list[tuple[str, str]]:
    """Compare one variable's tags and pattern rules with its gold decomposition.

    Args: gold: the gold decomposition; entry: its answer-key entry; patterns, types, trees: the
        rulebook data.
    Returns: (category, detail) findings; empty when everything agrees.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out: list[tuple[str, str]] = []
    by_number = {row["number"]: row for row in types}
    by_name = {row["name"].lower(): row for row in types}
    if len(entry["property"]) != 1:
        out.append(("property tag count", f"{len(entry['property'])} property tags"))
    gold_object, gold_matrix = shape(gold["hasObjectOfInterest"]), shape(gold["hasMatrix"])
    for number in entry["property"]:
        pattern = patterns.get(str(number), {})
        if not rulebook.routes_to(trees, number):
            out.append(("property pattern not in any tree", f"{number} {pattern.get('name')}"))
        want = {"variable": "single"}.get(pattern.get("object_shape"), pattern.get("object_shape"))
        if want and want != gold_object:
            out.append(("object shape against the property pattern",
                        f"{number} {pattern.get('name')} wants {want}, gold has {gold_object}"))
        want_matrix = pattern.get("matrix_shape")
        if want_matrix and want_matrix != gold_matrix:
            out.append(("matrix against the property pattern",
                        f"{number} {pattern.get('name')} wants a {want_matrix} matrix, gold has {gold_matrix}"))
        need = (pattern.get("required_constraint") or "").lower()
        gold_types = {constraint_type(c["label"]) for c in gold["hasConstraint"]}
        if need and not any(word.strip() in gold_types for word in re.split(r"\bor\b", need)):
            out.append(("required constraint missing in gold",
                        f"{number} {pattern.get('name')} requires {need!r}"))
    for number in entry["object"]:
        want = patterns.get(str(number), {}).get("shape", "single")
        if want != gold_object:
            out.append(("object shape against the object tag", f"tag {number} wants {want}, gold has {gold_object}"))
    if bool(entry["matrix"]) != (gold_matrix != "empty"):
        out.append(("matrix present or absent", f"tags {entry['matrix']}, gold has {gold_matrix}"))
    for number in entry["matrix"]:
        want = patterns.get(str(number), {}).get("shape", "single")
        if gold_matrix != "empty" and want != gold_matrix:
            out.append(("matrix shape against the matrix tag", f"tag {number} wants {want}, gold has {gold_matrix}"))
    if bool(entry["context"]) != bool(gold["hasContextObject"]):
        out.append(("context object present or absent", f"tags {entry['context']}, gold {bool(gold['hasContextObject'])}"))
    if entry["statistical_modifier"] != bool(gold["hasStatisticalModifier"]):
        out.append(("statistical modifier present or absent",
                    f"tag {entry['statistical_modifier']}, gold {bool(gold['hasStatisticalModifier'])}"))
    tagged = {by_number[n]["name"].lower() if n in by_number else entry["names"].get(str(n), str(n))
              for n in entry["constraint"]}
    in_gold = {constraint_type(c["label"]) for c in gold["hasConstraint"]}
    for name in sorted(tagged - in_gold):
        out.append(("constraint type tagged but not in gold", name))
    for name in sorted(in_gold - tagged):
        out.append(("constraint type in gold but not tagged", name))
    for constraint in gold["hasConstraint"]:
        name = constraint_type(constraint["label"])
        if name not in by_name:
            out.append(("gold constraint type not in the table", name))
            continue
        part = target_part(gold, constraint["on"])
        if part not in by_name[name]["targets"]:
            out.append(("gold constraint sits on a part the table does not allow",
                        f"{name} on {part}; table allows {by_name[name]['targets'] or 'nothing'}"))
    for flag in entry["flags"]:
        out.append(("entry flagged by its authors", flag))
    return out


def audit(records: list[dict[str, Any]], key: dict[str, Any]) -> dict[str, Any]:
    """Audit every corpus variable.

    Args: records: canonical corpus records; key: from `load`.
    Returns: {"variables": count, "clean": count, "categories": {category: count},
        "findings": [{"label", "source_path", "issue", "category", "detail"}]}.
    Raises: KeyError when a record has no answer-key entry.
    Side Effects: reads the built rulebook data.
    """
    patterns = rulebook.load("patterns")["patterns"]
    types = rulebook.load("constraint_types")["types"]
    trees = rulebook.load("trees")["trees"]
    findings, categories, clean = [], Counter(), 0
    for record in records:
        entry = key[record["variable_id"]]
        found = audit_variable(record["gold"], entry, patterns, types, trees)
        clean += not found
        for category, detail in found:
            categories[category] += 1
            findings.append({"label": record["label"], "source_path": record["source_path"],
                             "issue": entry["issue"], "category": category, "detail": detail})
    return {"variables": len(records), "clean": clean, "categories": dict(categories.most_common()),
            "findings": findings}


def main(argv: list[str] | None = None) -> int:
    """Build the answer key, or print its audit against the gold.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: 0.
    Raises: SystemExit on bad arguments; ValueError when a record cannot be matched.
    Side Effects: --build writes `data/issues_snapshot.json` and `data/route_key.json`; --audit
        prints gold-derived detail.
    """
    from iadopt_lab.corpus.ingestion import load_canonical_records

    parser = argparse.ArgumentParser(description="Expert route answer key (exploratory).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--issues", type=Path)
    parser.add_argument("--date")
    args = parser.parse_args(argv)
    records = list(load_canonical_records(ROOT))
    if args.build:
        if not args.issues or not args.date:
            parser.error("--build needs --issues and --date")
        issues = [json.loads(line) for line in args.issues.read_text(encoding="utf-8").splitlines() if line.strip()]
        built = build(issues, records, args.date)
        for name, value in (("issues_snapshot.json", {"source": EXAMPLES, "date": args.date,
                                                      "issues": built["snapshot"]}),
                            ("route_key.json", built["key"])):
            (rulebook.DATA / name).write_text(rulebook.dump(value), encoding="utf-8")
            print(f"wrote data/{name}")
        print(f"matched {len(built['key']['variables'])} of {len(records)} corpus variables")
        return 0
    result = audit(records, load())
    print(json.dumps({k: result[k] for k in ("variables", "clean", "categories")}, indent=1))
    for finding in result["findings"]:
        print(f"#{finding['issue']:<4} {finding['label'][:46]:46} | {finding['category']} | {finding['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
