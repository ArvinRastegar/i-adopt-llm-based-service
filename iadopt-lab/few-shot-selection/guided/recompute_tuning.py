#!/usr/bin/env python
"""Independent recomputation of stage 5 (tuning retrieval, chosen fold by fold).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/PATTERN_RETRIEVAL_20261119.md` (private repository).

Standard library only, and nothing imported from the lab or from the other modules of this
folder. The stored answers and their scores are read again and every figure of
`tuning-summary.json` is worked out a second way: each arm's Close F1 per repetition, mean,
spread, groups, unusable answers, calls, prompt length and cost; the setting chosen for each
fold on the other five (exactly, with fractions); the chosen setting's scores; the two
comparisons of the plan with both intervals and Welch's test; every setting against today's
prompt and against `rag-25`; the split without near relatives (the folds dealt again and the
relatives found again here); the retry's tallies. It also fails when a record came from another
host or carries reasoning, when the prompts fixed before the run are not the fingerprinted ones,
and when the eleven records of a unit do not belong together.

    ./recompute_tuning.py            # prints the comparison; exit status 1 on any mismatch
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
CALLS = OUTPUT / "retrieval" / "tuning-calls.jsonl"
SCORES = OUTPUT / "retrieval" / "tuning-scores.jsonl"
SUMMARY = OUTPUT / "retrieval" / "tuning-summary.json"
GROUP_FOLDER = OUTPUT / "guided"
GROUP_FILES = {"examples": "groups-examples-calls.jsonl", "practice": "ceiling-calls.jsonl",
               "held-out": "groups-held-out-calls.jsonl"}
DATA = HERE / "data"
CANONICAL = HERE.parents[1] / "data" / "canonical"
HOST = "Near AI"
ARMS = ("base", "rag-25", "rag-10", "rag-50", "rag-85", "rag-25-first", "fixed+rag-10", "rag-25-words",
        "rag-25-mmr", "rag-25-labels", "rag-25+retry")
CONFIGS = ARMS[1:]
BASELINE, REFERENCE = "base", "rag-25"
REPETITIONS, DRAWS, SEED = 10, 10000, 20261130
STRICT = 0.05 / 2
FOLDS, FOLD_SEED, FOLDS_DIGEST = 6, 20261123, "80f37c48d2cf004c"
OVERLAP = 0.5
TOLERANCE = 1e-9
# The digests of the prompts fixed before the run, as the plan records them (one per variable).
PLAN_DIGESTS: dict[str, str] = {
    "base": "78fcde54455ecd22",
    "rag-25": "b4d93cce21277ac7",
    "rag-10": "180670c70fd13efd",
    "rag-50": "883a1e9c8452b3b0",
    "rag-85": "b48b40a9b4ec1670",
    "rag-25-first": "2add7d8b445cd039",
    "fixed+rag-10": "a65aaa0af2848a21",
    "rag-25-words": "bf0ec35b180e6562",
    "rag-25-mmr": "6f4f9e0bc2d22cad"}


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


def relatives(texts: dict[str, str], fold_of: dict[str, int]) -> set[str]:
    """The variables whose definition shares at least half its words with one of another fold.

    Args: texts: definitions by variable; fold_of: from `deal_folds`.
    Returns: the variables with such a relative.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    words = {v: set(re.findall(r"[a-z0-9]+", text.lower())) for v, text in texts.items()}
    return {v for v in words
            if any(len(words[v] & words[u]) / len(words[v] | words[u]) >= OVERLAP
                   for u in words if fold_of[u] != fold_of[v])}


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
    by_group = groups()
    ids = sorted(v for members in by_group.values() for v in members)
    reps = list(range(1, REPETITIONS + 1))
    problems: list[str] = []

    def check(name: str, mine, theirs, relative: float = 0.0, room: float = 0.0, absolute: float = TOLERANCE) -> None:
        """Note a mismatch between a recomputed figure and the stored one.

        Args: name: what is compared; mine, theirs: the figures; relative: room as a share of the
            stored value; room: absolute room.
        Returns: nothing.
        Raises: nothing.
        Side Effects: appends to `problems`.
        """
        if mine is None or theirs is None:
            if mine is not theirs:
                problems.append(f"{name}: recomputed {mine}, stored {theirs}")
        elif not abs(mine - theirs) <= absolute + relative * abs(theirs) + room:
            problems.append(f"{name}: recomputed {mine}, stored {theirs}")

    wanted = {(arm, r, v) for arm in ARMS for r in reps for v in ids}
    answered = [record for record in calls.values() if record["ok"]]
    print(f"records kept {len(calls)}; answered {len(answered)}; scored {len(scores)}")
    if set(calls) != wanted:
        problems.append("the stored records are not 11 arms x 102 variables x 10 repetitions")
    if len(answered) != len(calls) or len(scores) != len(calls):
        problems.append(f"{len(calls) - len(answered)} records not answered and {len(calls) - len(scores)} not scored")
    hosts = Counter(record.get("host") for record in answered)
    thinking = sum(1 for record in answered if record.get("reasoning_chars") or record.get("reasoning_tokens"))
    print("hosts", dict(hosts), "| records with reasoning", thinking)
    if set(hosts) - {HOST}:
        problems.append(f"records from another host: {dict(hosts)}")
    if thinking:
        problems.append(f"{thinking} records carry reasoning")
    for arm, digest in PLAN_DIGESTS.items():
        by_variable: dict[str, set[str]] = defaultdict(set)
        for (name, _, variable_id), record in calls.items():
            if name == arm:
                by_variable[variable_id].add(record.get("prompt_sha256"))
        if any(len(found) != 1 for found in by_variable.values()) or len(by_variable) != len(ids):
            problems.append(f"{arm}: the fixed prompts were not sent once per variable")
            continue
        mine = text_hash("\n".join(next(iter(by_variable[v])) for v in sorted(by_variable)))[:16]
        if mine != digest:
            problems.append(f"{arm}: the prompts sent ({mine}) are not the fingerprinted ones ({digest})")
    if problems and (set(calls) != wanted or set(scores) != wanted):
        for problem in problems:
            print("MISMATCH", problem)
        print(f"{len(problems)} mismatches")
        return 1
    torn = 0
    for r in reps:
        for v in ids:
            unit = {arm: calls[(arm, r, v)] for arm in ARMS}
            wrong = len({record.get("started_at") for record in unit.values()}) != 1
            wrong = wrong or unit["rag-25-labels"].get("draft_sha256") != text_hash(unit["base"]["answer"])
            wrong = wrong or unit["rag-25+retry"].get("first_sha256") != text_hash(unit["rag-25"]["answer"])
            torn += wrong
    print("units whose records do not belong together:", torn)
    if torn:
        problems.append(f"{torn} units hold records that do not belong together")
    if summary.get("unit_integrity", {}).get("mismatched") != torn:
        problems.append(f"unit integrity: recomputed {torn}, stored {summary.get('unit_integrity', {}).get('mismatched')}")
    if summary.get("complete") is not True or summary.get("repetitions_used") != reps:
        problems.append(f"the summary is not of a complete run: complete {summary.get('complete')}, "
                        f"repetitions used {summary.get('repetitions_used')}")
    if list(summary.get("arms", {})) != list(ARMS) or summary.get("variables") != len(ids):
        problems.append("the summary's arms or number of variables are not those of the plan")

    exact: dict[str, dict[str, list[Fraction]]] = defaultdict(lambda: defaultdict(lambda: [Fraction(0)] * 3))
    for (arm, _, variable_id), row in scores.items():
        exact[arm][variable_id] = [a + b for a, b in zip(exact[arm][variable_id], counts_of(row), strict=True)]

    def series(pick: dict[str, str], members: list[str]) -> list[float]:
        """Close F1 per repetition when each variable is answered by the arm `pick` names.

        Args: pick: {variable: arm}; members: the variables.
        Returns: one F1 per repetition.
        Raises: KeyError when a score is missing.
        Side Effects: none.
        """
        return [f1([sum(counts_of(scores[(pick[v], r, v)])[k] for v in members) for k in range(3)]) for r in reps]

    per_arm: dict[str, list[float]] = {}
    for arm in ARMS:
        stored = summary["arms"].get(arm)
        if stored is None:
            continue
        everyone = {v: arm for v in ids}
        values = series(everyone, ids)
        per_arm[arm] = values
        check(f"{arm} mean", statistics.mean(values), stored["f1_mean"])
        check(f"{arm} sd", statistics.stdev(values), stored["f1_sd"])
        for position, value in enumerate(values):
            check(f"{arm} repetition {position + 1}", value, stored["f1_per_repetition"][position])
        for group, members in by_group.items():
            check(f"{arm} {group}", statistics.mean(series(everyone, members)), stored["by_group"][group])
        records = [calls[(arm, r, v)] for r in reps for v in ids]
        check(f"{arm} invalid", float(sum(1 for r in reps for v in ids if not scores[(arm, r, v)]["valid_prediction"])),
              float(stored["invalid"]))
        check(f"{arm} calls", float(sum(record.get("calls") or 0 for record in records)), float(stored["calls"]))
        check(f"{arm} cost", sum(record.get("cost_usd") or 0.0 for record in records), stored["cost_usd"])
        tokens = [record.get("prompt_tokens") for record in records if record.get("calls")]
        check(f"{arm} prompt tokens", statistics.mean(tokens) if tokens else None, stored["prompt_tokens_mean"])
        print(f"{arm:14} F1 {statistics.mean(values):.4f}")
    check("spent", sum(calls[("base", r, v)].get("unit_cost_usd") or 0.0 for r in reps for v in ids),
          summary["spent_usd"])

    fold_of = deal_folds(by_group, tags)
    if text_hash("\n".join(f"{v} {fold_of[v]}" for v in sorted(fold_of)))[:16] != FOLDS_DIGEST:
        problems.append("the folds dealt again are not those of stage 0")
    choice = {}
    for fold in range(FOLDS):
        others = [v for v in ids if fold_of[v] != fold]
        best, best_value = None, None
        for arm in CONFIGS:
            tp, fp, fn = (sum((exact[arm][v][k] for v in others), Fraction(0)) for k in range(3))
            value = 2 * tp / (2 * tp + fp + fn) if tp else Fraction(0)
            if best is None or value > best_value:
                best, best_value = arm, value
        choice[str(fold)] = best
    if choice != summary["chosen"].get("by_fold"):
        problems.append(f"chosen by fold: recomputed {choice}, stored {summary['chosen'].get('by_fold')}")
    pick = {v: choice[str(fold_of[v])] for v in ids}
    chosen_series = series(pick, ids)
    check("chosen mean", statistics.mean(chosen_series), summary["chosen"]["f1_mean"])
    for position, value in enumerate(chosen_series):
        check(f"chosen repetition {position + 1}", value, summary["chosen"]["f1_per_repetition"][position])
    for group, members in by_group.items():
        check(f"chosen {group}", statistics.mean(series(pick, members)), summary["chosen"]["by_group"][group])

    per_variable = {arm: {v: [float(x) for x in counts] for v, counts in by.items()} for arm, by in exact.items()}
    per_variable["chosen"] = {v: per_variable[pick[v]][v] for v in ids}
    wanted_names = {f"chosen vs {BASELINE}": BASELINE, f"chosen vs {REFERENCE}": REFERENCE}
    if set(summary["comparisons"]) != set(wanted_names):
        problems.append(f"the summary's comparisons are not the two of the plan: {sorted(summary['comparisons'])}")
    for name, second in wanted_names.items():
        stored = summary["comparisons"].get(name)
        if stored is None:
            continue
        mine = comparison(per_variable, "chosen", second, ids)
        for field in ("gain", "low", "high", "strict_low", "strict_high"):
            check(f"{name} {field}", mine[field], stored[field])
        check(f"{name} share_positive", mine["share_positive"], stored["share_positive"], room=mine["ties"])
        test = welch(chosen_series, per_arm[second])
        for field in ("difference", "t", "df"):
            if not math.isnan(test[field]):
                check(f"{name} runs {field}", test[field], stored["welch"][field], relative=1e-9)
        check(f"{name} runs p", test["p"], stored["welch"]["p"], relative=1e-6, absolute=0.0)
    if set(summary["against_base"]) != set(CONFIGS) or set(summary["against_reference"]) != set(CONFIGS) - {REFERENCE}:
        problems.append("the summary's descriptive comparisons are not every setting")
    for key, second, arms in (("against_base", BASELINE, CONFIGS), ("against_reference", REFERENCE,
                                                                   [a for a in CONFIGS if a != REFERENCE])):
        for arm in arms:
            stored = summary[key].get(arm)
            if stored is None:
                continue
            mine = comparison(per_variable, arm, second, ids)
            for field in ("gain", "low", "high"):
                check(f"{key} {arm} {field}", mine[field], stored[field])
    found = relatives(definitions(), fold_of)
    without = [v for v in ids if v not in found]
    print("near relatives found again:", len(found))
    if set(summary["without_relatives"]) != {"chosen", *CONFIGS}:
        problems.append("the summary's split without relatives is not for the chosen setting and every setting")
    for arm in ("chosen", *CONFIGS):
        stored = summary["without_relatives"].get(arm)
        if stored is None:
            continue
        check(f"without relatives {arm} variables", float(len(without)), float(stored["variables"]))
        mine = comparison(per_variable, arm, BASELINE, without)
        for field in ("gain", "low", "high"):
            check(f"without relatives {arm} {field}", mine[field], stored[field])
    retry = [calls[("rag-25+retry", r, v)] for r in reps for v in ids]
    for name, value in (("answers", len(retry)), ("repaired", sum(bool(item.get("repaired")) for item in retry)),
                        ("repair_valid", sum(bool(item.get("repair_valid")) for item in retry))):
        check(f"retry {name}", float(value), float(summary["retry"].get(name, -1)))
    print("chosen by fold", choice, f"| chosen F1 {statistics.mean(chosen_series):.4f}")
    for problem in problems:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
