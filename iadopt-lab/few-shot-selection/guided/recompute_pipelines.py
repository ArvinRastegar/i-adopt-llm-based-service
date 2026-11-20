#!/usr/bin/env python
"""Independent recomputation of stage 4 (a draft first, retrieval, and a rule check).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/PATTERN_RETRIEVAL_20261119.md` (private repository).

Standard library only, and nothing imported from the lab or from the other modules of this
folder. The stored answers and their scores are read again and every figure of
`pipelines-summary.json` is worked out a second way: each arm's Close F1 per repetition, its mean,
spread, groups, unusable answers, calls and cost; the five comparisons with today's prompt (the
list is held here) with both intervals, the share of positive resamples, Welch's test and Holm's
adjustment; the three secondary contrasts; the split by near relatives (the folds dealt again and
the relatives found again here); what the guided arms stated against the experts' tags; and the
rule check's outcomes. It also fails when a record came from another host or carries reasoning, when
the prompts fixed before the run are not the fingerprinted ones, and when the six records of a unit
do not belong together. (The records do not store the model that served each call; the request
names the model and the transport refuses any other host.)

    ./recompute_pipelines.py            # prints the comparison; exit status 1 on any mismatch
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "output"
CALLS = OUTPUT / "retrieval" / "pipelines-calls.jsonl"
SCORES = OUTPUT / "retrieval" / "pipelines-scores.jsonl"
SUMMARY = OUTPUT / "retrieval" / "pipelines-summary.json"
GROUP_FOLDER = OUTPUT / "guided"
GROUP_FILES = {"examples": "groups-examples-calls.jsonl", "practice": "ceiling-calls.jsonl",
               "held-out": "groups-held-out-calls.jsonl"}
DATA = HERE / "data"
CANONICAL = HERE.parents[1] / "data" / "canonical"
HOST = "Near AI"
ARMS = ("base", "draft-first", "draft-first/gated", "rag", "rag/gated", "combined")
GUIDED = ("draft-first", "draft-first/gated", "rag/gated", "combined")
COMPARED = ARMS[1:]
SECONDARY = (("draft-first/gated", "draft-first"), ("rag/gated", "rag"), ("combined", "rag"))
REPETITIONS, DRAWS, SEED = 10, 10000, 20261128
STRICT = 0.05 / len(COMPARED)
FOLDS, FOLD_SEED, FOLDS_DIGEST = 6, 20261123, "80f37c48d2cf004c"
OVERLAP = 0.5
TOLERANCE = 1e-9
# The digests of the prompts fixed before the run, as the plan records them: today's prompt and
# the retrieval prompt for all 102 variables, and the `rag/gated` prompt where the vote fires.
PLAN_DIGESTS = {"base": ("78fcde54455ecd22", 102), "rag": ("b4d93cce21277ac7", 102),
                "rag/gated": ("d8eb4c07243a275d", 50)}


def rows(path: Path) -> list[dict]:
    """Read a JSON-lines file.

    Args: path: the file.
    Returns: its rows; none when the file is missing.
    Raises: ValueError for a line that is not JSON.
    Side Effects: reads the file.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text_hash(text: str) -> str:
    """SHA-256 of a text.

    Args: text: any text.
    Returns: the hex digest.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def kept_calls() -> dict[tuple, dict]:
    """One record per (arm, repetition, variable): the first answered one, else the last.

    Args: none.
    Returns: {key: record}.
    Raises: OSError when the file cannot be read.
    Side Effects: reads the calls file.
    """
    out: dict[tuple, dict] = {}
    for record in rows(CALLS):
        key = (record["arm"], record["repetition"], record["variable_id"])
        if key not in out or not out[key]["ok"]:
            out[key] = record
    return out


def kept_scores(calls: dict[tuple, dict]) -> dict[tuple, dict]:
    """The stored score of each kept record's answer.

    Args: calls: from `kept_calls`.
    Returns: {key: score row}; a record without a stored score for its answer is left out.
    Raises: OSError when the file cannot be read.
    Side Effects: reads the scores file.
    """
    stored = {(tuple(row["key"]), row["answer_sha256"]): row for row in rows(SCORES)}
    out = {}
    for key, record in calls.items():
        answer = record["answer"] if record["ok"] else None
        digest = text_hash(answer) if answer is not None else "no-answer"
        if (key, digest) in stored:
            out[key] = stored[(key, digest)]
    return out


def counts_of(row: dict) -> list[Fraction]:
    """Exact Close counts of one scored answer.

    Args: row: a score row.
    Returns: [TP, FP, FN].
    Raises: KeyError when a count is missing.
    Side Effects: none. Deterministic.
    """
    totals = row["evaluation"]["close"]["totals"]
    return [Fraction(totals[name]["numerator"], totals[name]["denominator"]) for name in ("tp", "fp", "fn")]


def f1(total: list) -> float:
    """Micro F1 from summed counts.

    Args: total: [TP, FP, FN].
    Returns: 2TP / (2TP + FP + FN), 0.0 without a true positive.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    tp, fp, fn = total
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def groups() -> dict[str, list[str]]:
    """The three groups of the corpus, from the variables stored by steps 1 and 5.

    Args: none.
    Returns: {group: sorted variable ids}.
    Raises: ValueError when the groups overlap or are not 25, 53 and 24 variables.
    Side Effects: reads three stored calls files.
    """
    out = {name: sorted({row["variable_id"] for row in rows(GROUP_FOLDER / file)})
           for name, file in GROUP_FILES.items()}
    sizes = [len(out[name]) for name in ("examples", "practice", "held-out")]
    if sizes != [25, 53, 24] or len({v for ids in out.values() for v in ids}) != 102:
        raise ValueError(f"the stored groups are not 25, 53 and 24 distinct variables: {sizes}")
    return out


def deal_folds(by_group: dict[str, list[str]], tags: dict[str, dict]) -> dict[str, int]:
    """Deal the 102 variables into the six folds again, as stage 0 defines them.

    Args: by_group: from `groups`; tags: the experts' tags by variable.
    Returns: {variable: fold}.
    Raises: KeyError when a variable has no tags.
    Side Effects: none. Deterministic for the fixed seed.
    """
    rng = random.Random(FOLD_SEED)
    out: dict[str, int] = {}
    for group in ("examples", "practice", "held-out"):
        for has_matrix in (True, False):
            stratum = sorted(v for v in by_group[group] if bool(tags[v]["matrix"]) == has_matrix)
            rng.shuffle(stratum)
            for variable_id in stratum:
                out[variable_id] = len(out) % FOLDS
    return out


def definitions() -> dict[str, str]:
    """The definitions of the corpus records, read from the lab's canonical record files.

    Args: none.
    Returns: {variable id: definition}.
    Raises: ValueError when the files do not hold 102 distinct variables.
    Side Effects: reads the canonical record files (plain JSON; the lab's code is not used).
    """
    out = {}
    for path in sorted(CANONICAL.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "variable_id" in record and "definition" in record:
            out[record["variable_id"]] = record["definition"]
    if len(out) != 102:
        raise ValueError(f"the canonical records hold {len(out)} variables, not 102")
    return out


def relatives(definitions: dict[str, str], fold_of: dict[str, int]) -> set[str]:
    """The variables whose definition shares at least half its words with one of another fold.

    Args: definitions: by variable; fold_of: from `deal_folds`.
    Returns: the variables with such a relative.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    words = {v: set(re.findall(r"[a-z0-9]+", text.lower())) for v, text in definitions.items()}
    return {v for v in words
            if any(len(words[v] & words[u]) / len(words[v] | words[u]) >= OVERLAP
                   for u in words if fold_of[u] != fold_of[v])}


def tagged_form(entry: dict, patterns: dict, type_names: dict[int, str]) -> dict:
    """The matrix shape, context object and constraint types the experts' tags give.

    Args: entry: one variable's tags; patterns: the pattern table; type_names: name by number.
    Returns: {"matrix", "context", "types"}.
    Raises: KeyError when the tags lack a field.
    Side Effects: none. Deterministic.
    """
    shapes = [patterns.get(str(n), {}).get("shape", "single") for n in entry["matrix"]]
    matrix = "empty" if not shapes else ("source_target" if "source_target" in shapes else "single")
    names = entry.get("names", {})
    types = sorted({type_names.get(n) or names.get(str(n), str(n)) for n in entry["constraint"]})
    return {"matrix": matrix, "context": bool(entry["context"]), "types": types}


def incomplete_beta(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function, by its continued fraction.

    Args: a, b: positive shape values; x: between 0 and 1.
    Returns: I_x(a, b).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - incomplete_beta(b, a, 1.0 - x)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)) / a
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 400):
        for numerator in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                          -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            result *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return front * result


def welch(b: list[float], a: list[float]) -> dict[str, float]:
    """Welch's two-sided test of mean(b) - mean(a).

    Args: b, a: per-repetition scores, at least two each.
    Returns: {"difference", "t", "df", "p"}.
    Raises: statistics.StatisticsError with fewer than two values in either group.
    Side Effects: none. Deterministic.
    """
    difference = statistics.mean(b) - statistics.mean(a)
    var_b, var_a = statistics.variance(b) / len(b), statistics.variance(a) / len(a)
    if var_b + var_a == 0:
        return {"difference": difference, "t": math.nan, "df": math.nan, "p": 1.0 if difference == 0 else 0.0}
    df = (var_b + var_a) ** 2 / (var_b ** 2 / (len(b) - 1) + var_a ** 2 / (len(a) - 1))
    t = difference / math.sqrt(var_b + var_a)
    return {"difference": difference, "t": t, "df": df, "p": incomplete_beta(df / 2.0, 0.5, df / (df + t * t))}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm's step-down adjustment.

    Args: pvalues: {name: p}.
    Returns: {name: adjusted p}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(sorted(pvalues.items(), key=lambda item: item[1])):
        running = max(running, min(1.0, (len(pvalues) - rank) * p))
        out[name] = running
    return out


def comparison(per_variable: dict[str, dict[str, list[float]]], first: str, second: str, ids: list[str]) -> dict:
    """One arm against another, the variables resampled with the fixed seed.

    Args: per_variable: {arm: {variable: [TP, FP, FN] over the repetitions}}; first, second: arms;
        ids: the variables.
    Returns: {"gain", "low", "high", "strict_low", "strict_high", "share_positive", "ties"}.
    Raises: KeyError when an arm lacks a variable.
    Side Effects: none. Deterministic.
    """
    ids = sorted(ids)
    tables = [[per_variable[arm][v] for v in ids] for arm in (first, second)]

    def pooled(sample: list[int]) -> float:
        """F1 of the first arm minus the second over a list of positions.

        Args: sample: positions into `ids`, repeats allowed.
        Returns: the difference.
        Raises: nothing.
        Side Effects: none.
        """
        values = []
        for table in tables:
            tp = fp = fn = 0.0
            for position in sample:
                tp += table[position][0]
                fp += table[position][1]
                fn += table[position][2]
            values.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        return values[0] - values[1]

    rng = random.Random(SEED)
    drawn = sorted(pooled(rng.choices(range(len(ids)), k=len(ids))) for _ in range(DRAWS))
    return {"gain": pooled(list(range(len(ids)))), "low": drawn[int(0.025 * DRAWS)],
            "high": drawn[int(0.975 * DRAWS) - 1], "strict_low": drawn[int(STRICT / 2 * DRAWS)],
            "strict_high": drawn[int((1 - STRICT / 2) * DRAWS) - 1],
            "share_positive": sum(g > 0 for g in drawn) / DRAWS, "ties": sum(abs(g) <= 1e-12 for g in drawn) / DRAWS}


def main() -> int:
    """Recompute and compare.

    Args: none.
    Returns: 0 when every figure matches the stored summary, 1 otherwise.
    Raises: OSError when a stored file is missing.
    Side Effects: reads stored files; prints the comparison.
    """
    calls = kept_calls()
    scores = kept_scores(calls)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    tags = json.loads((DATA / "route_key.json").read_text(encoding="utf-8"))["variables"]
    patterns = json.loads((DATA / "patterns.json").read_text(encoding="utf-8"))["patterns"]
    type_names = {row["number"]: row["name"].lower()
                  for row in json.loads((DATA / "constraint_types.json").read_text(encoding="utf-8"))["types"]}
    by_group = groups()
    ids = sorted(v for members in by_group.values() for v in members)
    problems: list[str] = []

    def check(name: str, mine, theirs, relative: float = 0.0, room: float = 0.0, absolute: float = TOLERANCE) -> None:
        """Note a mismatch between a recomputed figure and the stored one.

        Args: name: what is compared; mine, theirs: the two figures; relative: room as a share of
            the stored value (for p-values worked out another way); room: absolute room.
        Returns: nothing.
        Raises: nothing.
        Side Effects: appends to `problems`.
        """
        if mine is None or theirs is None:
            if mine is not theirs:
                problems.append(f"{name}: recomputed {mine}, stored {theirs}")
        elif not abs(mine - theirs) <= absolute + relative * abs(theirs) + room:
            problems.append(f"{name}: recomputed {mine}, stored {theirs}")

    # What was sent and what came back.
    wanted = {(arm, r, v) for arm in ARMS for r in range(1, REPETITIONS + 1) for v in ids}
    answered = [record for record in calls.values() if record["ok"]]
    print(f"records kept {len(calls)}; answered {len(answered)}; scored {len(scores)}")
    if set(calls) != wanted:
        problems.append("the stored records are not 6 arms x 102 variables x 10 repetitions")
    if len(answered) != len(calls) or len(scores) != len(calls):
        problems.append(f"{len(calls) - len(answered)} records not answered and {len(calls) - len(scores)} not scored")
    hosts = Counter(record.get("host") for record in answered)
    thinking = sum(1 for record in answered if record.get("reasoning_chars") or record.get("reasoning_tokens"))
    print("hosts", dict(hosts), "| records with reasoning", thinking)
    if set(hosts) - {HOST}:
        problems.append(f"records from another host: {dict(hosts)}")
    if thinking:
        problems.append(f"{thinking} records carry reasoning")
    for arm, (digest, count) in PLAN_DIGESTS.items():
        by_variable: dict[str, set[str]] = defaultdict(set)
        for (name, _, variable_id), record in calls.items():
            if name == arm and record.get("prompt_sha256"):
                by_variable[variable_id].add(record["prompt_sha256"])
        if any(len(found) != 1 for found in by_variable.values()) or len(by_variable) != count:
            problems.append(f"{arm}: the fixed prompts were not sent once per variable ({len(by_variable)} variables)")
            continue
        mine = text_hash("\n".join(next(iter(by_variable[v])) for v in sorted(by_variable)))[:16]
        if mine != digest:
            problems.append(f"{arm}: the prompts sent ({mine}) are not the fingerprinted ones ({digest})")
    torn = 0
    for r in range(1, REPETITIONS + 1):
        for v in ids:
            if not all((arm, r, v) in calls for arm in ARMS):
                continue
            unit = {arm: calls[(arm, r, v)] for arm in ARMS}
            wrong = len({record.get("started_at") for record in unit.values()}) != 1
            for arm in GUIDED:
                own = text_hash(unit["base" if arm.startswith("draft") else "rag"]["answer"])
                wrong = wrong or unit[arm].get("draft_sha256") != own
                if unit[arm].get("no_guidance") or unit[arm].get("fell_back"):
                    wrong = wrong or text_hash(unit[arm]["answer"]) != own
            torn += wrong
    print("units whose records do not belong together:", torn)
    if torn:
        problems.append(f"{torn} units hold records that do not belong together")
    if summary.get("unit_integrity", {}).get("mismatched") != torn:
        problems.append(f"unit integrity: recomputed {torn}, stored {summary.get('unit_integrity', {}).get('mismatched')}")
    if summary.get("complete") is not True or summary.get("repetitions_used") != list(range(1, REPETITIONS + 1)):
        problems.append(f"the summary is not of a complete run: complete {summary.get('complete')}, "
                        f"repetitions used {summary.get('repetitions_used')}")
    if set(summary.get("arms", {})) != set(ARMS) or summary.get("variables") != len(ids):
        problems.append("the summary's arms or number of variables are not those of the plan")
    if problems and (set(calls) != wanted or set(scores) != wanted):
        for problem in problems:
            print("MISMATCH", problem)
        print(f"{len(problems)} mismatches")
        return 1

    # Scores per arm.
    reps = list(range(1, REPETITIONS + 1))
    exact: dict[str, dict[str, list[Fraction]]] = defaultdict(lambda: defaultdict(lambda: [Fraction(0)] * 3))
    for (arm, _, variable_id), row in scores.items():
        exact[arm][variable_id] = [a + b for a, b in zip(exact[arm][variable_id], counts_of(row), strict=True)]
    series: dict[str, list[float]] = {}
    for arm in ARMS:
        stored = summary["arms"].get(arm)
        if stored is None:
            continue
        values = [f1([sum(counts_of(scores[(arm, r, v)])[k] for v in ids) for k in range(3)]) for r in reps]
        series[arm] = values
        check(f"{arm} mean", statistics.mean(values), stored["f1_mean"])
        check(f"{arm} sd", statistics.stdev(values), stored["f1_sd"])
        for position, value in enumerate(values):
            check(f"{arm} repetition {position + 1}", value, stored["f1_per_repetition"][position])
        for group, members in by_group.items():
            grouped = [f1([sum(counts_of(scores[(arm, r, v)])[k] for v in members) for k in range(3)]) for r in reps]
            check(f"{arm} {group}", statistics.mean(grouped), stored["by_group"][group])
        check(f"{arm} invalid", float(sum(1 for r in reps for v in ids if not scores[(arm, r, v)]["valid_prediction"])),
              float(stored["invalid"]))
        check(f"{arm} calls", float(sum(calls[(arm, r, v)].get("calls") or 0 for r in reps for v in ids)),
              float(stored["calls"]))
        check(f"{arm} cost", sum(calls[(arm, r, v)].get("cost_usd") or 0.0 for r in reps for v in ids),
              stored["cost_usd"])
        print(f"{arm:18} F1 {statistics.mean(values):.4f}")
    check("spent", sum(calls[("base", r, v)].get("unit_cost_usd") or 0.0 for r in reps for v in ids),
          summary["spent_usd"])

    # Comparisons: the five of the plan, the three secondary contrasts, the split by relatives.
    per_variable = {arm: {v: [float(x) for x in counts] for v, counts in by.items()} for arm, by in exact.items()}
    if list(summary["comparisons"]) != list(COMPARED):
        problems.append(f"the summary's comparisons are not the five of the plan: {list(summary['comparisons'])}")
    pvalues = {}
    for arm in COMPARED:
        stored = summary["comparisons"].get(arm)
        if stored is None:
            continue
        if (stored.get("first"), stored.get("second")) != (arm, "base"):
            problems.append(f"comparison {arm}: compares {stored.get('first')} with {stored.get('second')}")
        mine = comparison(per_variable, arm, "base", ids)
        for field in ("gain", "low", "high", "strict_low", "strict_high"):
            check(f"{arm} {field}", mine[field], stored[field])
        check(f"{arm} share_positive", mine["share_positive"], stored["share_positive"], room=mine["ties"])
        test = welch(series[arm], series["base"])
        pvalues[arm] = test["p"]
        for field in ("difference", "t", "df"):
            if not math.isnan(test[field]):
                check(f"{arm} runs {field}", test[field], stored["welch"][field], relative=1e-9)
        check(f"{arm} runs p", test["p"], stored["welch"]["p"], relative=1e-6, absolute=0.0)
    for arm, value in holm(pvalues).items():
        check(f"{arm} Holm p", value, summary["comparisons"][arm]["p_holm"], relative=1e-6, absolute=0.0)
    names = {f"{first} vs {second}": (first, second) for first, second in SECONDARY}
    if set(summary["secondary"]) != set(names):
        problems.append(f"the summary's secondary contrasts are not the three of the plan: {sorted(summary['secondary'])}")
    for name, (first, second) in names.items():
        stored = summary["secondary"].get(name)
        if stored is None:
            continue
        mine = comparison(per_variable, first, second, ids)
        for field in ("gain", "low", "high"):
            check(f"{name} {field}", mine[field], stored[field])
    fold_of = deal_folds(by_group, tags)
    if text_hash("\n".join(f"{v} {fold_of[v]}" for v in sorted(fold_of)))[:16] != FOLDS_DIGEST:
        problems.append("the folds dealt again are not those of stage 0")
    found = relatives(definitions(), fold_of)
    with_relative = sorted(found)
    without_relative = [v for v in ids if v not in found]
    print("near relatives found again:", len(with_relative))
    for side, key in (("without", "without_relatives"), ("with", "with_relatives")):
        for arm in COMPARED:
            stored = summary[key].get(arm)
            if stored is None:
                problems.append(f"{side} relatives {arm}: missing")
                continue
            members = without_relative if side == "without" else with_relative
            check(f"{side} relatives {arm} variables", float(len(members)), float(stored["variables"]))
            mine = comparison(per_variable, arm, "base", members)
            for field in ("gain", "low", "high"):
                check(f"{side} relatives {arm} {field}", mine[field], stored[field])

    # What the guided arms stated, against the tags, and how the rule check went.
    for arm in GUIDED:
        stored = summary["patterns"].get(arm)
        if stored is None:
            problems.append(f"patterns {arm}: missing")
            continue
        tally: Counter = Counter()
        for r in reps:
            for v in ids:
                item = calls[(arm, r, v)]
                tally["answers"] += 1
                stmt = item.get("statement")
                if stmt is None:
                    tally["no_guidance"] += 1
                    continue
                tally["guided"] += 1
                want = tagged_form(tags[v], patterns, type_names)
                if "property" in stmt:
                    tally["property_stated"] += 1
                    tally["property_right"] += stmt["property"] == list(tags[v]["property"])
                if "matrix_shape" in stmt:
                    tally["matrix_stated"] += 1
                    tally["matrix_right"] += stmt["matrix_shape"] == want["matrix"]
                    tally["matrix_pattern_right"] += stmt["matrix"] == list(tags[v]["matrix"])
                if "context" in stmt:
                    tally["context_stated"] += 1
                    tally["context_right"] += stmt["context"] == want["context"]
                if "constraint_types" in stmt:
                    tally["constraint_stated"] += 1
                    said, tagged = set(stmt["constraint_types"]), set(want["types"])
                    tally["types_stated"] += len(said)
                    tally["types_tagged"] += len(said & tagged)
                    tally["types_exact"] += said == tagged
                    tally["types_contained"] += said <= tagged
                tally["passed_first"] += not item["first_complaints"]
                tally["repaired"] += bool(item["repaired"])
                tally["passed_after_repair"] += bool(item["repaired"]) and not item["repair_complaints"]
                tally["fell_back"] += bool(item["fell_back"])
                tally["draft_would_pass"] += not item["draft_complaints"]
        for name, value in tally.items():
            check(f"patterns {arm} {name}", float(value), float(stored.get(name, 0)))
        for name in [key for key in stored if not key.endswith(("_precision", "_share"))]:
            if name not in tally and stored[name]:
                problems.append(f"patterns {arm} {name}: stored {stored[name]}, recomputed nothing")
        print(f"{arm:18} stated in {tally['guided']} of {tally['answers']}; matrix right {tally['matrix_right']} of "
              f"{tally['matrix_stated']}; types tagged {tally['types_tagged']} of {tally['types_stated']}; fell back {tally['fell_back']}")
    for problem in problems:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
