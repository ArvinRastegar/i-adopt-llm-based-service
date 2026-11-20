#!/usr/bin/env python
"""Independent recomputation of stage 1 (the value of each pattern decision).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/PATTERN_RETRIEVAL_20261119.md` (private repository).

Standard library only, and nothing imported from the lab or from the other modules of this
folder. The stored calls and scores are read again and the stored summary is checked figure by
figure: per-arm scores (each repetition, the mean, the three groups, unusable and failed
answers, cost), the list of comparisons itself (held here, not taken from the summary), every
comparison's difference, interval over resampled variables and share of positive resamples, the
Welch test over repetitions and its Holm adjustment, the break-even accuracies, and the split by
whether the experts' tags have a part. It also fails when an answer came from another host or
model than the pinned one or carries reasoning, and when the prompts stored with the answers are
not the ones the plan fingerprinted before the run.

Inputs it does not take from the lab: the groups of the corpus come from the stored files of
steps 1 and 5, and the experts' tags from `data/route_key.json`.

    ./recompute_decisions.py            # prints the comparison; exit status 1 on any mismatch
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "output"
CALLS = OUTPUT / "retrieval" / "decisions-calls.jsonl"
SCORES = OUTPUT / "retrieval" / "decisions-scores.jsonl"
SUMMARY = OUTPUT / "retrieval" / "decisions-summary.json"
LEDGER = OUTPUT / "retrieval" / "openrouter-ledger.jsonl"
GROUP_FOLDER = OUTPUT / "guided"
GROUP_FILES = {"examples": "groups-examples-calls.jsonl", "practice": "ceiling-calls.jsonl",
               "held-out": "groups-held-out-calls.jsonl"}
TAGS = HERE / "data" / "route_key.json"
HOST, MODEL = "Near AI", "qwen/qwen3.8-27b"
DECISIONS = ("property", "object", "matrix", "context", "modifier", "constraint")
PRESENCE = {"matrix": "matrix", "context": "context", "modifier": "statistical_modifier",
            "constraint": "constraint"}
REPETITIONS, DRAWS, SEED = 10, 10000, 20261119
TOLERANCE = 1e-9
# The digest of each arm's 102 prompts as the plan recorded it before the run.
PLAN_DIGESTS = {
    "base": "78fcde54455ecd22", "key": "0ac593e6501ce470", "drop-property": "f526bd5fb61fe8a8",
    "drop-object": "039e700573f79744", "drop-matrix": "b17724729940647d", "drop-context": "77072a049bdc16db",
    "drop-modifier": "e6f7242f33b5e59c", "drop-constraint": "f31bee23c15c171b",
    "only-property": "12660fc5f6449781", "only-object": "e2d2058597e1b689", "only-matrix": "32a6bb36f886a919",
    "only-context": "ec5df6c9823a8cad", "only-modifier": "ecee170e6eafe9c0", "only-constraint": "589b73ea3e7040be",
    "wrong-property": "08ea1d66bfc06bc4", "wrong-object": "ec4db3efa4024e05", "wrong-matrix": "02b75fa7dc0ed7c0",
    "wrong-context": "1b60c7c1a70f28c5", "wrong-modifier": "7b4a138743babfd2",
    "wrong-constraint": "d994d9c80b1b488f", "soft-key": "851e19dc1e686da0", "walk": "287c58ed4a5098a8",
    "soft-walk": "8b96197895e235a7", "cand-property-2": "7e87609937c1862c", "cand-property-3": "d2f375b33e840317"}


def comparisons() -> dict[str, tuple[str, str]]:
    """The 34 comparisons of the plan, written out here a second time: {name: (first, second)}.

    Args: none.
    Returns: the comparisons; a difference is the first arm minus the second.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out = {"key vs base": ("key", "base"), "walk vs base": ("walk", "base"),
           "soft-key vs key": ("soft-key", "key"), "soft-walk vs walk": ("soft-walk", "walk"),
           "soft-walk vs base": ("soft-walk", "base"), "soft-key vs base": ("soft-key", "base")}
    for d in DECISIONS:
        out[f"value of {d} (key vs drop-{d})"] = ("key", f"drop-{d}")
        out[f"{d} alone (only-{d} vs base)"] = (f"only-{d}", "base")
        out[f"harm of a wrong {d} (drop-{d} vs wrong-{d})"] = (f"drop-{d}", f"wrong-{d}")
        out[f"wrong-{d} vs base"] = (f"wrong-{d}", "base")
    for count in (2, 3):
        out[f"cand-property-{count} vs drop-property"] = (f"cand-property-{count}", "drop-property")
        out[f"cand-property-{count} vs key"] = (f"cand-property-{count}", "key")
    return out


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
    """One call per (arm, repetition, variable): the first that was answered, else the last failure.

    Args: none.
    Returns: {key: record}.
    Raises: OSError when the calls file cannot be read.
    Side Effects: reads the calls file.
    """
    out: dict[tuple, dict] = {}
    for record in rows(CALLS):
        key = (record["arm"], record["repetition"], record["variable_id"])
        if key not in out or not out[key]["ok"]:
            out[key] = record
    return out


def kept_scores(calls: dict[tuple, dict]) -> dict[tuple, dict]:
    """The stored score that belongs to each kept call's answer.

    Args: calls: from `kept_calls`.
    Returns: {key: score row}; a call whose answer has no stored score is left out.
    Raises: OSError when the scores file cannot be read.
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


def resampled(per_variable: dict[str, dict[str, list[float]]], arms: list[str], ids: list[str]) -> dict[str, list[float]]:
    """Each arm's pooled F1 over the same resamples of some variables.

    Args: per_variable: {arm: {variable: [TP, FP, FN] summed over repetitions}}; arms: the arms
        wanted; ids: the variables to resample, with replacement.
    Returns: {arm: one F1 per resample}; position 0 holds the F1 over the variables as they are.
    Raises: KeyError when an arm lacks a variable.
    Side Effects: none. Deterministic for the fixed seed: the generator draws the same indices as
        drawing from the sorted ids does.
    """
    ids = sorted(ids)
    table = {arm: [per_variable[arm][v] for v in ids] for arm in arms}
    rng = random.Random(SEED)
    samples = [list(range(len(ids)))] + [rng.choices(range(len(ids)), k=len(ids)) for _ in range(DRAWS)]
    out: dict[str, list[float]] = {}
    for arm in arms:
        column = table[arm]
        values = []
        for sample in samples:
            tp = fp = fn = 0.0
            for index in sample:
                row = column[index]
                tp += row[0]
                fp += row[1]
                fn += row[2]
            values.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        out[arm] = values
    return out


def spread(first: list[float], second: list[float]) -> dict[str, float]:
    """One arm against another from their resampled F1 values.

    Args: first, second: from `resampled`, position 0 the plain value.
    Returns: {"gain", "low", "high", "share_positive", "ties"}; "ties" is the share of resamples in
        which the two arms score the same to twelve decimals. Which side of zero such a resample
        falls on depends on the order of the additions, so a share of positive resamples is only
        known to within the ties.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    drawn = sorted(a - b for a, b in zip(first[1:], second[1:], strict=True))
    return {"gain": first[0] - second[0], "low": drawn[int(0.025 * DRAWS)],
            "high": drawn[int(0.975 * DRAWS) - 1], "share_positive": sum(g > 0 for g in drawn) / DRAWS,
            "ties": sum(abs(g) <= 1e-12 for g in drawn) / DRAWS}


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
    Returns: {"difference", "t", "df", "p"}; p is 1 or 0 when both variances are zero.
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
    Returns: {name: adjusted p}, capped at 1 and never falling with rank.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(sorted(pvalues.items(), key=lambda item: item[1])):
        running = max(running, min(1.0, (len(pvalues) - rank) * p))
        out[name] = running
    return out


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
    by_group = groups()
    ids = sorted(v for members in by_group.values() for v in members)
    arms = sorted({key[0] for key in calls})
    problems: list[str] = []

    def check(name: str, mine: float | None, theirs: float | None, relative: float = 0.0,
              room: float = 0.0, absolute: float = TOLERANCE) -> None:
        """Note a mismatch between a recomputed figure and the stored one.

        Args: name: what is compared; mine: recomputed; theirs: stored; relative: extra room as a
            share of the stored value, for the p-values worked out by another method; room: extra
            room in the figure's own units, for the ties of a share of positive resamples.
        Returns: nothing.
        Raises: nothing.
        Side Effects: appends to `problems`.
        """
        if mine is None or theirs is None:
            if mine is not theirs:
                problems.append(f"{name}: recomputed {mine}, stored {theirs}")
        elif not abs(mine - theirs) <= absolute + relative * abs(theirs) + room:
            problems.append(f"{name}: recomputed {mine}, stored {theirs}")

    def check_spread(name: str, mine: dict[str, float], theirs: dict) -> None:
        """Compare one comparison's difference, interval and share of positive resamples.

        Args: name: what is compared; mine: from `spread`; theirs: the stored figures.
        Returns: nothing.
        Raises: nothing; a missing stored figure is a mismatch.
        Side Effects: appends to `problems`.
        """
        for field in ("gain", "low", "high"):
            check(f"{name} {field}", mine[field], theirs.get(field))
        check(f"{name} share_positive", mine["share_positive"], theirs.get("share_positive"), room=mine["ties"])

    # What was sent and what came back.
    answered = [record for record in calls.values() if record["ok"]]
    print(f"calls kept {len(calls)}; answered {len(answered)}; scored {len(scores)}")
    print("arms", len(arms), "variables", len({k[2] for k in calls}), "repetitions", sorted({k[1] for k in calls}))
    if len(calls) != 25 * 102 * REPETITIONS or set(arms) != set(PLAN_DIGESTS) or {k[2] for k in calls} != set(ids):
        problems.append("the stored calls are not 25 arms x 102 variables x 10 repetitions")
    if len(answered) != len(calls) or len(scores) != len(calls):
        problems.append(f"{len(calls) - len(answered)} calls failed and {len(calls) - len(scores)} have no score")
    hosts = Counter(record["host"] for record in answered)
    models = Counter(record["served_model"] for record in answered)
    thinking = sum(1 for record in answered if record.get("reasoning_chars") or record.get("reasoning_tokens"))
    print("hosts", dict(hosts), "models", dict(models), "answers with reasoning", thinking,
          "finish reasons", dict(Counter(record["finish_reason"] for record in answered)))
    if set(hosts) - {HOST} or set(models) - {MODEL}:
        problems.append(f"answers came from another host or model: {dict(hosts)} {dict(models)}")
    if thinking:
        problems.append(f"{thinking} answers carry reasoning")
    matched = 0
    for arm in arms:
        hashes = defaultdict(set)
        for (name, _, variable_id), record in calls.items():
            if name == arm:
                hashes[variable_id].add(record["prompt_sha256"])
        if any(len(found) != 1 for found in hashes.values()) or set(hashes) != set(ids):
            problems.append(f"{arm}: a variable was sent different prompts in different repetitions")
            continue
        digest = text_hash("\n".join(next(iter(hashes[v])) for v in ids))[:16]
        if digest != PLAN_DIGESTS.get(arm):
            problems.append(f"{arm}: the prompts sent ({digest}) are not the planned ones ({PLAN_DIGESTS.get(arm)})")
        else:
            matched += 1
    print("prompt digests match the plan for", matched, "arms")

    # Scores per arm.
    exact: dict[str, dict[str, list[Fraction]]] = defaultdict(lambda: defaultdict(lambda: [Fraction(0)] * 3))
    per_repetition: dict[str, dict[int, dict[str, list[Fraction]]]] = defaultdict(lambda: defaultdict(dict))
    for (arm, repetition, variable_id), row in scores.items():
        counts = counts_of(row)
        exact[arm][variable_id] = [a + b for a, b in zip(exact[arm][variable_id], counts, strict=True)]
        per_repetition[arm][repetition][variable_id] = counts
    if summary.get("complete") is not True or summary.get("repetitions_used") != list(range(1, REPETITIONS + 1)):
        problems.append(f"the summary is not of a complete run: complete {summary.get('complete')}, "
                        f"repetitions used {summary.get('repetitions_used')}")
    if set(summary["arms"]) != set(arms):
        problems.append("the summary's arms are not the stored arms")
    series: dict[str, list[float]] = {}
    spent = 0.0
    for arm in arms:
        cell = summary["arms"].get(arm)
        if cell is None:
            continue
        values = []
        for repetition in range(1, REPETITIONS + 1):
            got = per_repetition[arm].get(repetition, {})
            if set(got) == set(ids):
                values.append(f1([sum(got[v][k] for v in ids) for k in range(3)]))
        if len(values) != REPETITIONS or cell["repetitions"] != REPETITIONS:
            problems.append(f"{arm}: {len(values)} complete repetitions, stored {cell['repetitions']}")
            continue
        series[arm] = values
        check(f"{arm} mean", statistics.mean(values), cell["f1_mean"])
        check(f"{arm} sd", statistics.stdev(values), cell["f1_sd"])
        for position, value in enumerate(values):
            check(f"{arm} repetition {position + 1}", value, cell["f1_per_repetition"][position])
        for group, members in by_group.items():
            grouped = [f1([sum(per_repetition[arm][r][v][k] for v in members) for k in range(3)])
                       for r in range(1, REPETITIONS + 1)]
            check(f"{arm} {group}", statistics.mean(grouped), cell["by_group"][group])
        invalid = sum(1 for key, row in scores.items() if key[0] == arm and not row["valid_prediction"])
        failed = sum(1 for key, record in calls.items() if key[0] == arm and not record["ok"])
        cost = sum(record.get("cost_usd") or 0.0 for key, record in calls.items() if key[0] == arm)
        spent += cost
        if invalid != cell["invalid"] or failed != cell["failed_calls"]:
            problems.append(f"{arm}: invalid {invalid} / failed {failed}, stored {cell['invalid']} / {cell['failed_calls']}")
        check(f"{arm} cost", cost, cell["cost_usd"])
        print(f"{arm:18} F1 {statistics.mean(values):.4f}  invalid {invalid:3}  failed {failed}  ${cost:.2f}")
    check("spent", spent, summary["spent_usd"])

    # Comparisons: the list itself, the resampled intervals, the test over repetitions.
    wanted = comparisons()
    stored_pairs = {name: (row.get("first"), row.get("second")) for name, row in summary["contrasts"].items()}
    if stored_pairs != wanted:
        problems.append("the summary's comparisons are not the 34 of the plan: "
                        f"missing {sorted(set(wanted) - set(stored_pairs))}, extra {sorted(set(stored_pairs) - set(wanted))}, "
                        f"other arms {sorted(n for n in set(wanted) & set(stored_pairs) if wanted[n] != stored_pairs[n])}")
    per_variable = {arm: {v: [float(x) for x in counts] for v, counts in by.items()} for arm, by in exact.items()}
    usable = [arm for arm in arms if arm in series]
    drawn = resampled(per_variable, usable, ids)
    pvalues = {}
    for name, (first, second) in wanted.items():
        row = summary["contrasts"].get(name)
        if row is None or first not in drawn or second not in drawn:
            continue
        check_spread(name, spread(drawn[first], drawn[second]), row)
        test = welch(series[first], series[second])
        pvalues[name] = test["p"]
        for field in ("difference", "t", "df"):
            if not math.isnan(test[field]):
                check(f"{name} runs {field}", test[field], row["welch"][field], relative=1e-9)
        check(f"{name} runs p", test["p"], row["welch"]["p"], relative=1e-6, absolute=0.0)
    for name, value in holm(pvalues).items():
        check(f"{name} Holm p", value, summary["contrasts"][name]["p_holm"], relative=1e-6, absolute=0.0)

    # Break-even accuracies and the split by presence.
    for decision in DECISIONS:
        stored = summary["break_even"].get(decision)
        names = ("key", f"drop-{decision}", f"wrong-{decision}")
        if stored is None or any(name not in drawn for name in names):
            problems.append(f"break-even {decision}: missing")
            continue
        top, dropped, wrong = (drawn[name] for name in names)
        value, harm = top[0] - dropped[0], dropped[0] - wrong[0]
        check(f"break-even {decision} value", value, stored["value"])
        check(f"break-even {decision} harm", harm, stored["harm"])
        check(f"break-even {decision} accuracy",
              min(1.0, max(0.0, harm / (value + harm))) if value + harm > 0 else None, stored["accuracy"])
        defined = sorted(min(1.0, max(0.0, (d - w) / (t - w)))
                         for t, d, w in zip(top[1:], dropped[1:], wrong[1:], strict=True) if t - w > 0)
        check(f"break-even {decision} low", defined[int(0.025 * len(defined))] if defined else None, stored["low"])
        check(f"break-even {decision} high", defined[int(0.975 * len(defined)) - 1] if defined else None,
              stored["high"])
        check(f"break-even {decision} defined", len(defined) / DRAWS, stored["defined_share"])
    tags = json.loads(TAGS.read_text(encoding="utf-8"))["variables"]
    if set(summary["by_presence"]) != set(PRESENCE):
        problems.append("the summary's split by presence is not for matrix, context, modifier and constraint")
    for decision, field in PRESENCE.items():
        for label, wanted_presence in (("present", True), ("absent", False)):
            members = [v for v in ids if bool(tags[v][field]) == wanted_presence]
            stored = summary["by_presence"].get(decision, {}).get(label)
            if stored is None:
                problems.append(f"by presence {decision} {label}: missing")
                continue
            check(f"by presence {decision} {label} variables", float(len(members)), float(stored["variables"]))
            part = resampled(per_variable, ["key", f"drop-{decision}", f"wrong-{decision}"], members)
            for side, (first, second) in (("value", ("key", f"drop-{decision}")),
                                          ("harm", (f"drop-{decision}", f"wrong-{decision}"))):
                check_spread(f"by presence {decision} {label} {side}", spread(part[first], part[second]),
                             stored[side])

    by_purpose: dict[str, float] = defaultdict(float)
    for row in rows(LEDGER):
        booked = row.get("booked_usd")
        by_purpose[row["purpose"]] += float(booked if booked is not None else row.get("cost_usd") or 0.0)
    print("ledger by purpose:", {name: round(value, 4) for name, value in sorted(by_purpose.items())},
          "total", round(sum(by_purpose.values()), 4))
    print(f"comparisons checked {len(wanted)}; cost of the stored answers ${spent:.4f}")
    for problem in problems:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
