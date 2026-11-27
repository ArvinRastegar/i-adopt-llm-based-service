#!/usr/bin/env python
"""Independent recomputation of the conventions test (`conventions_test.py`).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/CONVENTIONS_TEST_20261127.md` (private repository).

Standard library only, and nothing imported from the lab or from the other modules of this
folder. The stored answers are read again and marked a second way: its own JSON extraction, its own
reading of the pattern numbers and type names (the rulebook table completed by the overlay), the
matrix by its shape (patterns 40 to 44 a single entity, 45 a source and target) and the constraint
types as a set. Then every figure of Part 1's summary except the "complete" column (which needs the
object and property shapes) is worked out again, and Part 2's from the stored scores.

    ./recompute_conventions.py     # prints the comparison; exit status 1 on any mismatch
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parent / "output" / "retrieval"
CANONICAL = HERE.parents[1] / "data" / "canonical"
ARMS = ("plain", "rules")
PART2 = ("rag-25", "rag-25+experts", "rag-25+rules")
RUNS1, RUNS2 = 5, 10
DRAWS, SEED, SEED2 = 10000, 20261204, 20261130
STRICT = 0.05 / 2
BREAK_EVEN = {"matrix": 0.57, "constraint": 0.54}
SOURCE_TARGET = 45            # the only matrix pattern whose matrix is a source and a target (flow)
TOLERANCE = 1e-9


def rows(path: Path) -> list[dict]:
    """A JSON-lines file.

    Args: path: the file.
    Returns: its rows; none when it is missing.
    Raises: ValueError for a line that is not JSON.
    Side Effects: reads the file.
    """
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def kept(path: Path) -> dict[tuple, dict]:
    """One record per (arm, run, target): the first answered one, else the last.

    Args: path: a calls file.
    Returns: {key: record}.
    Raises: as `rows`.
    Side Effects: reads the file.
    """
    out: dict[tuple, dict] = {}
    for record in rows(path):
        key = (record["arm"], record["repetition"], record["variable_id"])
        if key not in out or not out[key].get("ok"):
            out[key] = record
    return out


def strict(text: str):
    """JSON with duplicate keys and NaN-like constants refused.

    Args: text: a span.
    Returns: the value.
    Raises: ValueError when the span is not strict JSON.
    Side Effects: none.
    """
    def pairs(items):
        """Refuse repeated keys.

        Args: items: key-value pairs.
        Returns: a dict.
        Raises: ValueError for a repeated key.
        Side Effects: none.
        """
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("repeated key")
        return dict(items)

    def constant(name):
        """Refuse NaN and infinity.

        Args: name: the constant.
        Returns: nothing.
        Raises: ValueError always.
        Side Effects: none.
        """
        raise ValueError(name)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def extract(answer: str | None) -> dict | None:
    """The one JSON object of an answer: the whole text, else one fenced block, else one braced span.

    Args: answer: the reply.
    Returns: the object, or None.
    Raises: nothing.
    Side Effects: none.
    """
    text = (answer or "").strip()
    try:
        value = strict(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    found = []
    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S):
        try:
            value = strict(block.strip())
            if isinstance(value, dict):
                found.append(value)
        except ValueError:
            continue
    if len(found) == 1:
        return found[0]
    depth, start, spans = 0, None, []
    for position, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = position
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start:position + 1])
    parsed = []
    for span in spans:
        try:
            value = strict(span)
            if isinstance(value, dict):
                parsed.append(value)
        except ValueError:
            continue
    return parsed[0] if len(parsed) == 1 else None


def table() -> tuple[dict[int, str], dict[str, int]]:
    """The constraint-type table completed by the overlay.

    Args: none.
    Returns: ({number: name}, {name: the number of its first row}).
    Raises: KeyError when the files lack their fields.
    Side Effects: reads two data files.
    """
    types = json.loads((DATA / "constraint_types.json").read_text(encoding="utf-8"))["types"]
    overlay = json.loads((DATA / "types_overlay.json").read_text(encoding="utf-8"))
    names = [(row["number"], row["name"].lower()) for row in types]
    renamed = {change["number"]: change["name"].lower() for change in overlay["rename"]}
    names = [(number, renamed.get(number, name)) for number, name in names]
    names += [(added["number"], added["name"].lower()) for added in overlay["add"]]
    by_number = dict(names)
    by_name: dict[str, int] = {}
    for number, name in names:
        by_name.setdefault(name, number)
    return by_number, by_name


def mark(entry: dict, answer: str | None, kinds: dict, by_number: dict, by_name: dict) -> dict:
    """One answer against one variable's tags.

    Args: entry: the tags; answer: the reply or None; kinds: {pattern number: kind}; by_number,
        by_name: from `table`.
    Returns: {"matrix", "constraint", "matrix_pattern", "both", "parsed", "form", "tp", "fp", "fn"}.
    Raises: nothing.
    Side Effects: none.
    """
    data = extract(answer)
    proper = bool(data is not None and {"property", "matrix", "context", "modifier", "constraints"} <= set(data)
                  and (data["matrix"] is None or (type(data["matrix"]) is int and 40 <= data["matrix"] <= 45))
                  and type(data["context"]) is bool and type(data["modifier"]) is bool
                  and isinstance(data["constraints"], list) and all(isinstance(n, str) for n in data["constraints"]))
    form = bool(data is not None and isinstance(data.get("object_of_interest"), str) and "matrix_entity" in data
                and (data["matrix_entity"] is None or isinstance(data["matrix_entity"], str))
                and isinstance(data.get("restrictions"), list) and all(isinstance(x, str) for x in data["restrictions"]))
    parsed = proper and kinds.get(str(data["property"])) == "property"
    want_types = {by_number.get(n) or entry["names"].get(str(n), str(n)) for n in entry["constraint"]}
    want_shape = "empty" if not entry["matrix"] else ("source_target" if SOURCE_TARGET in entry["matrix"] else "single")
    if not parsed:
        return {"matrix": False, "constraint": False, "matrix_pattern": False, "both": False, "parsed": False,
                "form": form, "tp": 0, "fp": 0, "fn": len(want_types)}
    matrix = [data["matrix"]] if data["matrix"] is not None else []
    shape = "empty" if not matrix else ("source_target" if SOURCE_TARGET in matrix else "single")
    names = {n.lower() for n in data["constraints"]}
    invented = len(names - set(by_name))
    got = {by_number[by_name[n]] for n in names if n in by_name}
    row = {"matrix": shape == want_shape, "constraint": got == want_types and not invented,
           "matrix_pattern": matrix == list(entry["matrix"]), "parsed": True, "form": form,
           "tp": len(got & want_types), "fp": len(got - want_types) + invented, "fn": len(want_types - got)}
    row["both"] = row["matrix"] and row["constraint"]
    return row


def interval(gaps: list[float], draws: int = DRAWS, seed: int = SEED) -> dict:
    """The mean of per-variable gaps with resampled intervals.

    Args: gaps: one gap per variable, in id order; draws, seed: of the resampling.
    Returns: {"mean", "low", "high", "strict_low", "strict_high"}.
    Raises: nothing.
    Side Effects: none.
    """
    rng = random.Random(seed)
    drawn = sorted(statistics.fmean(rng.choices(gaps, k=len(gaps))) for _ in range(draws))
    return {"mean": statistics.fmean(gaps), "low": drawn[int(0.025 * draws)], "high": drawn[int(0.975 * draws) - 1],
            "strict_low": drawn[int(STRICT / 2 * draws)], "strict_high": drawn[int((1 - STRICT / 2) * draws) - 1]}


def main() -> int:
    """Recompute and compare.

    Args: none.
    Returns: 0 when every figure matches, 1 otherwise.
    Raises: OSError when a stored file is missing.
    Side Effects: reads stored files; prints the comparison.
    """
    problems: list[str] = []
    key = json.loads((DATA / "route_key.json").read_text(encoding="utf-8"))["variables"]
    patterns = json.loads((DATA / "patterns.json").read_text(encoding="utf-8"))
    patterns = patterns.get("patterns", patterns)
    kinds = {number: row.get("kind") for number, row in patterns.items()}
    by_number, by_name = table()
    records = {}
    for path in sorted(CANONICAL.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "variable_id" in record and "definition" in record:
            records[record["variable_id"]] = record
    ids = sorted(key)
    fresh = json.loads((OUT / "fresh" / "fresh-variables.json").read_text(encoding="utf-8"))["variables"]
    entries = {**{v: key[v] for v in ids}, **{v: row["entry"] for v, row in fresh.items()}}
    calls = kept(OUT / "conventions-part1-calls.jsonl")
    summary1 = json.loads((OUT / "conventions-part1-summary.json").read_text(encoding="utf-8"))
    marked = {k: mark(entries[k[2]], r.get("answer") if r.get("ok") else None, kinds, by_number, by_name)
              for k, r in calls.items()}
    deionized = {v for v in ids if "deionized water" in records[v]["definition"].lower()}
    populations = {"corpus": ids, "corpus without deionized water": [v for v in ids if v not in deionized],
                   "fresh, all": sorted(fresh), "fresh, unflagged": sorted(v for v in fresh if not fresh[v]["flags"]),
                   "fresh, no near relative": sorted(v for v in fresh if not fresh[v]["relative"])}
    for name, members in populations.items():
        stored = summary1["arms"].get(name)
        if stored is None:
            problems.append(f"population {name} missing from the summary")
            continue
        for arm in ARMS:
            got = [row for (a, _, v), row in marked.items() if a == arm and v in set(members)]
            if len(got) != len(members) * RUNS1:
                problems.append(f"{name} {arm}: {len(got)} answers, {len(members) * RUNS1} expected")
            for column in ("matrix", "constraint", "matrix_pattern", "both", "parsed", "form"):
                mine = sum(r[column] for r in got) / len(got)
                if abs(mine - stored[arm][column]) > TOLERANCE:
                    problems.append(f"{name} {arm} {column}: {mine} here, {stored[arm][column]} stored")
            tp, fp, fn = (sum(r[k] for r in got) for k in ("tp", "fp", "fn"))
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if abs(f1 - stored[arm]["constraint_f1"]) > TOLERANCE:
                problems.append(f"{name} {arm} constraint F1: {f1} here, {stored[arm]['constraint_f1']} stored")

    def share(arm: str, column: str, members: list[str]) -> dict[str, float]:
        """Each variable's share right over its runs.

        Args: arm: the arm; column: the column; members: the variables.
        Returns: {variable: share}.
        Raises: ZeroDivisionError for a variable without answers.
        Side Effects: none.
        """
        tally = defaultdict(list)
        for (a, _, v), row in marked.items():
            if a == arm and v in set(members):
                tally[v].append(row[column])
        return {v: sum(tally[v]) / len(tally[v]) for v in members}

    passed = []
    for column in ("matrix", "constraint"):
        first, second = share("rules", column, ids), share("plain", column, ids)
        mine = interval([first[v] - second[v] for v in ids])
        for field, value in mine.items():
            if abs(value - summary1["comparisons"][column][field]) > TOLERANCE:
                problems.append(f"comparison {column} {field}: {value} here, {summary1['comparisons'][column][field]} stored")
        if mine["low"] > 0 and summary1["arms"]["corpus"]["rules"][column] > BREAK_EVEN[column]:
            passed.append(column)
    if passed != summary1["gate"]:
        problems.append(f"gate: {passed} here, {summary1['gate']} stored")
    groups: dict[str, list[str]] = defaultdict(list)
    for v in ids:
        gold, entry = records[v]["gold"], key[v]
        pattern = entry["matrix"][0] if entry["matrix"] else 0
        if isinstance(gold["hasObjectOfInterest"], dict) and "hasDenominator" in gold["hasObjectOfInterest"]:
            groups["ratio object"].append(v)
        if gold["hasMatrix"] == "person":
            groups["matrix person"].append(v)
        elif pattern:
            groups[f"matrix {pattern} {patterns[str(pattern)]['name']}"].append(v)
        else:
            groups["no matrix"].append(v)
        if v in deionized:
            groups["deionized water"].append(v)
    if set(groups) != set(summary1["groups"]):
        problems.append(f"groups: {sorted(groups)} here, {sorted(summary1['groups'])} stored")
    for name, members in groups.items():
        for arm in ARMS:
            for column in ("matrix", "constraint"):
                mine = statistics.fmean(share(arm, column, sorted(members)).values())
                theirs = summary1["groups"].get(name, {}).get(arm, {}).get(column)
                if theirs is None or abs(mine - theirs) > TOLERANCE:
                    problems.append(f"group {name} {arm} {column}: {mine} here, {theirs} stored")
    print("Part 1:", "checked" if calls else "no calls", f"({len(calls)} answers); gate here {passed}")

    summary2_path = OUT / "conventions-part2-summary.json"
    if summary2_path.exists():
        summary2 = json.loads(summary2_path.read_text(encoding="utf-8"))
        arms2 = list(summary2.get("arms", {}))
        calls2 = kept(OUT / "conventions-part2-calls.jsonl")
        stored = {(tuple(r["key"]), r["answer_sha256"]): r for r in rows(OUT / "conventions-part2-scores.jsonl")}
        scores = {}
        for k, r in calls2.items():
            digest = hashlib.sha256(r["answer"].encode("utf-8")).hexdigest() if r.get("ok") and r.get("answer") is not None else "no-answer"
            if (k, digest) in stored:
                scores[k] = stored[(k, digest)]

        def counts(row: dict) -> list[Fraction]:
            """Close TP, FP, FN of a score row.

            Args: row: a score row.
            Returns: [TP, FP, FN].
            Raises: KeyError when a count is missing.
            Side Effects: none.
            """
            t = row["evaluation"]["close"]["totals"]
            return [Fraction(t[n]["numerator"], t[n]["denominator"]) for n in ("tp", "fp", "fn")]

        per_var: dict[str, dict[str, list[float]]] = {}
        for arm in arms2:
            per_run = []
            for r in range(1, RUNS2 + 1):
                total = [sum((counts(scores[(arm, r, v)])[k] for v in ids), Fraction(0)) for k in range(3)]
                per_run.append(float(2 * total[0] / (2 * total[0] + total[1] + total[2])) if total[0] else 0.0)
            for field, value in (("f1_mean", statistics.fmean(per_run)), ("f1_sd", statistics.stdev(per_run))):
                if abs(value - summary2["arms"][arm][field]) > TOLERANCE:
                    problems.append(f"Part 2 {arm} {field}: {value} here, {summary2['arms'][arm][field]} stored")
            invalid = sum(1 for r in range(1, RUNS2 + 1) for v in ids if not scores[(arm, r, v)]["valid_prediction"])
            if invalid != summary2["arms"][arm]["invalid"]:
                problems.append(f"Part 2 {arm} invalid: {invalid} here, {summary2['arms'][arm]['invalid']} stored")
            per_var[arm] = {v: [float(x) for x in [sum((counts(scores[(arm, r, v)])[k] for r in range(1, RUNS2 + 1)), Fraction(0))
                                                 for k in range(3)]] for v in ids}

        def pooled(first: str, sample: list[int]) -> float:
            """F1 of an arm minus `rag-25` over positions into the ids.

            Args: first: the arm; sample: positions, repeats allowed.
            Returns: the difference.
            Raises: nothing.
            Side Effects: none.
            """
            out = []
            for arm in (first, "rag-25"):
                tp = fp = fn = 0.0
                for position in sample:
                    a = per_var[arm][ids[position]]
                    tp, fp, fn = tp + a[0], fp + a[1], fn + a[2]
                out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
            return out[0] - out[1]

        for arm in arms2:
            if arm == "rag-25":
                continue
            rng = random.Random(SEED2)
            drawn = sorted(pooled(arm, rng.choices(range(len(ids)), k=len(ids))) for _ in range(DRAWS))
            mine = {"gain": pooled(arm, list(range(len(ids)))), "low": drawn[int(0.025 * DRAWS)],
                    "high": drawn[int(0.975 * DRAWS) - 1]}
            for field, value in mine.items():
                theirs = summary2["against_rag25"][arm][field]
                if not math.isclose(value, theirs, rel_tol=0.0, abs_tol=TOLERANCE):
                    problems.append(f"Part 2 {arm} {field}: {value} here, {theirs} stored")
        print("Part 2: checked arms", arms2)
    for problem in problems[:40]:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
