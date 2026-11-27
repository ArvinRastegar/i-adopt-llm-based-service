#!/usr/bin/env python
"""Independent recomputation of the stand-in user test (`stand_in_test.py`).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/STAND_IN_USER_TEST_20261127.md` (private repository).

Standard library only. Nothing is imported from the lab or from the modules that ran the test; the
JSON extraction and the completed type table come from `recompute_conventions.py`. The stand-in's and
the typing's stored answers are read again, the types and the descriptive measures rebuilt, the
decomposition prompts checked for consistency by their hashes, and Step 3 worked out from the stored
scores.

    ./recompute_stand_in.py     # prints the comparison; exit status 1 on any mismatch
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recompute_conventions as rc  # noqa: E402

SCREENS = {"best+stand-in": ("phrases", "added"), "best+screen-1": ("phrases",)}
ARMS = ("best", "best+stand-in", "best+screen-1", "best+experts")
PAIRS = (("best+stand-in", "best"), ("best+screen-1", "best"), ("best+experts", "best"),
         ("best+screen-1", "best+stand-in"), ("best+stand-in", "best+experts"), ("best+screen-1", "best+experts"))
RUNS = 10
TOLERANCE = 1e-9


def main() -> int:
    """Recompute and compare.

    Args: none.
    Returns: 0 when every figure matches, 1 otherwise.
    Raises: OSError when a stored file is missing.
    Side Effects: reads stored files; prints the comparison.
    """
    problems: list[str] = []
    key = json.loads((rc.DATA / "route_key.json").read_text(encoding="utf-8"))["variables"]
    patterns = json.loads((rc.DATA / "patterns.json").read_text(encoding="utf-8"))
    kinds = {number: row.get("kind") for number, row in patterns.get("patterns", patterns).items()}
    cats = json.loads((rc.DATA / "question_categories_v1.json").read_text(encoding="utf-8"))["categories"]
    category_of = {t: c["name"] for c in cats for t in c["types"]}
    by_number, by_name = rc.table()
    ids = sorted(key)
    stored = json.loads((rc.OUT / "stand-in-answers-summary.json").read_text(encoding="utf-8"))
    gold = {v: frozenset(by_number.get(n) or key[v]["names"].get(str(n), str(n)) for n in key[v]["constraint"]) for v in ids}

    def finder_types(answer):
        """The finder's type set, read strictly.

        Args: answer: the reply or None.
        Returns: a frozenset of names, or None.
        Raises: nothing.
        Side Effects: none.
        """
        data = rc.extract(answer)
        proper = bool(data is not None and {"property", "matrix", "context", "modifier", "constraints"} <= set(data)
                      and (data["matrix"] is None or (type(data["matrix"]) is int and 40 <= data["matrix"] <= 45))
                      and type(data["context"]) is bool and type(data["modifier"]) is bool
                      and isinstance(data["constraints"], list) and all(isinstance(n, str) for n in data["constraints"]))
        if not proper or kinds.get(str(data["property"])) != "property":
            return None
        return frozenset(by_number[by_name[n.lower()]] for n in data["constraints"] if n.lower() in by_name)

    finder_calls = rc.kept(rc.OUT / "answer-form-a-calls.jsonl")
    finder_answers = {v: finder_calls[("named-first", 1, v)].get("answer") if finder_calls[("named-first", 1, v)].get("ok")
                      else None for v in ids}
    shown = 0
    for v in ids:
        data = rc.extract(finder_answers[v]) or {}
        listed = data.get("restrictions") if isinstance(data.get("restrictions"), list) else []
        shown += sum(1 for p in listed if isinstance(p, str) and p.strip())
    if shown != stored["phrases_shown"]:
        problems.append(f"phrases shown: {shown} here, {stored['phrases_shown']} stored")

    calls = rc.kept(rc.OUT / "stand-in-answers-calls.jsonl")
    items: dict[str, list | None] = {}
    for v in ids:
        r = calls.get(("stand-in", 1, v), {})
        data = rc.extract(r.get("answer")) if r.get("ok") else None
        if data is None or not ({"phrases", "added"} & set(data)) or any(
                not isinstance(data.get(s) or [], list) for s in ("phrases", "added")):
            items[v] = None
            continue
        items[v] = [(s, item["phrase"].strip(), str(item.get("kind") or "").strip().lower())
                    for s in ("phrases", "added") for item in data.get(s) or []
                    if isinstance(item, dict) and isinstance(item.get("phrase"), str) and item["phrase"].strip()]

    names = {c["name"]: c["types"] for c in cats}
    typing = rc.kept(rc.OUT / "stand-in-typing-calls.jsonl")

    def typed(target: str, options: list[str]) -> str | None:
        """The type a typing answer names.

        Args: target: "<variable>#<position>"; options: the category's types.
        Returns: the type, or None.
        Raises: nothing.
        Side Effects: none.
        """
        r = typing.get(("typing", 1, target), {})
        words = " ".join(re.findall(r"[a-z0-9]+", (r.get("answer") or "").lower().replace("'", ""))) if r.get("ok") else ""
        if words in options:
            return words
        hits = [t for t in options if f" {t} " in f" {words} "]
        return sorted(hits, key=len)[-1] if hits else None

    needed = [f"{v}#{i}" for v in ids for i, (_, _, k) in enumerate(items[v] or []) if len(names.get(k, ())) > 1]
    types = {arm: {} for arm in SCREENS}
    for arm, screens in SCREENS.items():
        for v in ids:
            if items[v] is None:
                types[arm][v] = None
                continue
            chosen = set()
            for i, (s, _, k) in enumerate(items[v]):
                if k in names and s in screens:
                    t = names[k][0] if len(names[k]) == 1 else typed(f"{v}#{i}", names[k])
                    if t:
                        chosen.add(t)
            types[arm][v] = sorted(chosen)
        if types[arm] != stored["types"][arm]:
            problems.append(f"{arm}: the types differ for {sum(types[arm][v] != stored['types'][arm][v] for v in ids)} variables")
    flat = [x for v in ids for x in items[v] or []]
    mine = {"usable": sum(items[v] is not None for v in ids), "listed": sum(s == "phrases" for s, _, _ in flat),
            "listed_not_a_restriction": sum(s == "phrases" and k == "not a restriction" for s, _, k in flat),
            "added": sum(s == "added" for s, _, _ in flat),
            "added_not_a_restriction": sum(s == "added" and k == "not a restriction" for s, _, k in flat),
            "unknown_kind": sum(k not in names and k != "not a restriction" for _, _, k in flat),
            "typing_calls": len(needed),
            "not_typed": sum(typed(t, names[items[t.split("#")[0]][int(t.split("#")[1])][2]]) is None for t in needed)}
    for field, value in mine.items():
        if value != stored[field]:
            problems.append(f"{field}: {value} here, {stored[field]} stored")
    for who in (*SCREENS, "finder"):
        right = cats_right = tp = fp = fn = 0
        for v in ids:
            if who in SCREENS:
                got = frozenset(types[who][v] or ())
                picked = {k for s, _, k in items[v] or [] if k in names and s in SCREENS[who]}
            else:
                got = finder_types(finder_answers[v]) or frozenset()
                picked = {category_of[t] for t in got}
            right += got == gold[v]
            cats_right += picked == {category_of[t] for t in gold[v]}
            tp, fp, fn = tp + len(got & gold[v]), fp + len(got - gold[v]), fn + len(gold[v] - got)
        row = {"types_right": right / len(ids), "categories_right": cats_right / len(ids), "type_tp": tp, "type_fp": fp,
               "type_fn": fn, "type_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}
        for field, value in row.items():
            if abs(value - stored["by"][who][field]) > TOLERANCE:
                problems.append(f"{who} {field}: {value} here, {stored['by'][who][field]} stored")
    print(f"Steps 1 and 2 checked: {mine['usable']} usable answers, {len(needed)} typing calls")

    path = rc.OUT / "stand-in-decomposition-summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        dcalls = rc.kept(rc.OUT / "stand-in-decomposition-calls.jsonl")
        hashes = {}
        for (arm, run, v), r in dcalls.items():
            hashes.setdefault((arm, v), set()).add(r["prompt_sha256"])
        if any(len(h) != 1 for h in hashes.values()):
            problems.append("an arm's prompt differs between runs")
        h = {k: next(iter(s)) for k, s in hashes.items()}
        earlier = rc.kept(rc.OUT / "constraint-types-calls.jsonl")
        for arm in ("best", "best+experts"):
            if any(earlier[(arm, 1, v)]["prompt_sha256"] != h[(arm, v)] for v in ids):
                problems.append(f"{arm}: prompts differ from the constraint-types test's")
        for arm in SCREENS:
            for v in ids:
                no_facts = h[("best", v)] == h[("best+experts", v)]
                if no_facts or types[arm][v] is None:
                    ok = h[(arm, v)] == h[("best", v)]
                else:
                    ok = (h[(arm, v)] == h[("best+experts", v)]) == (frozenset(types[arm][v]) == gold[v]) \
                        and h[(arm, v)] != h[("best", v)]
                if not ok:
                    problems.append(f"{arm} {v}: the prompt does not match its types")
        print("Prompts checked:", sum(h[("best", v)] == h[("best+experts", v)] for v in ids), "variables without finder facts")

        stored_scores = {(tuple(r["key"]), r["answer_sha256"]): r for r in rc.rows(rc.OUT / "stand-in-decomposition-scores.jsonl")}
        scores = {}
        for kk, r in dcalls.items():
            digest = hashlib.sha256(r["answer"].encode("utf-8")).hexdigest() if r.get("ok") and r.get("answer") is not None else "no-answer"
            if (kk, digest) in stored_scores:
                scores[kk] = stored_scores[(kk, digest)]

        def counts(row: dict) -> list[Fraction]:
            """Close TP, FP, FN of a score row.

            Args: row: a score row.
            Returns: [TP, FP, FN].
            Raises: KeyError when a count is missing.
            Side Effects: none.
            """
            t = row["evaluation"]["close"]["totals"]
            return [Fraction(t[n]["numerator"], t[n]["denominator"]) for n in ("tp", "fp", "fn")]

        per_var = {}
        for arm in ARMS:
            per_run = []
            for r in range(1, RUNS + 1):
                total = [sum((counts(scores[(arm, r, v)])[i] for v in ids), Fraction(0)) for i in range(3)]
                per_run.append(float(2 * total[0] / (2 * total[0] + total[1] + total[2])) if total[0] else 0.0)
            for field, value in (("f1_mean", statistics.fmean(per_run)), ("f1_sd", statistics.stdev(per_run))):
                if abs(value - summary["arms"][arm][field]) > TOLERANCE:
                    problems.append(f"{arm} {field}: {value} here, {summary['arms'][arm][field]} stored")
            per_var[arm] = {v: [float(sum((counts(scores[(arm, r, v)])[i] for r in range(1, RUNS + 1)), Fraction(0)))
                                for i in range(3)] for v in ids}

        def pooled(first: str, second: str, sample: list[int]) -> float:
            """F1 of one arm minus another over positions into the ids.

            Args: first, second: arms; sample: positions.
            Returns: the difference.
            Raises: nothing.
            Side Effects: none.
            """
            out = []
            for arm in (first, second):
                tp = fp = fn = 0.0
                for position in sample:
                    a = per_var[arm][ids[position]]
                    tp, fp, fn = tp + a[0], fp + a[1], fn + a[2]
                out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
            return out[0] - out[1]

        for first, second in PAIRS:
            theirs = summary["against_best"][first] if second == "best" else summary["comparisons"][f"{first} - {second}"]
            rng = random.Random(rc.SEED2)
            drawn = sorted(pooled(first, second, rng.choices(range(len(ids)), k=len(ids))) for _ in range(rc.DRAWS))
            mine2 = {"gain": pooled(first, second, list(range(len(ids)))), "low": drawn[int(0.025 * rc.DRAWS)],
                     "high": drawn[int(0.975 * rc.DRAWS) - 1]}
            for field, value in mine2.items():
                if not math.isclose(value, theirs[field], rel_tol=0.0, abs_tol=TOLERANCE):
                    problems.append(f"{first} - {second} {field}: {value} here, {theirs[field]} stored")
        print("Step 3 checked:", len(PAIRS), "comparisons")
    for problem in problems[:40]:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
