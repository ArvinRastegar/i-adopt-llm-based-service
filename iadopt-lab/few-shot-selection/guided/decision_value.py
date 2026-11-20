#!/usr/bin/env python
"""Stage 1 of the pattern-retrieval work: what is each pattern decision worth, right and wrong?

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/PATTERN_RETRIEVAL_20261119.md` (private repository).

The experts' patterns lift the decomposition; the patterns the model finds do not. Before trying
to find them better, this stage measures which of the six decisions of a pattern set carry the
lift, what a wrong one costs, and whether softer wording keeps the gain or limits the damage. A
pattern set is six decisions: the property pattern, the shape of the object of interest, the
matrix, the context object, the statistical modifier and the constraint types.

Every arm is today's prompt (the study's 25 examples in the output-shaped layout, an example
variable answered with the other 24) plus a guidance block after the definition:

- `base`: no block. `key`: the experts' six decisions, as in steps 1 and 5.
- `drop-D`: the experts' decisions without decision D. `only-D`: decision D alone.
- `wrong-D`: the experts' decisions with D stated wrongly.
- `soft-key`: the experts' decisions worded as a suggestion.
- `walk`, `soft-walk`: the patterns the model found in steps 4 and 5 (`direct-all`), worded as a
  must and as a suggestion.
- `cand-property-2`, `cand-property-3`: the right property pattern among two or three candidates.

One call per arm, variable and repetition; no check and no repair. The model is
`qwen/qwen3.8-27b` on OpenRouter's pinned host (see `orclient.py`).

    ./decision_value.py --plan-only    # fingerprints, counts and the cost estimate; no calls
    ./decision_value.py --dev          # the new wordings on invented variables only
    ./decision_value.py --run          # the grid, resumable, under the spending limit
    ./decision_value.py --score        # score the stored answers and write the summary
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import analysis  # noqa: E402
import ceiling  # noqa: E402
import engine  # noqa: E402
import groups  # noqa: E402
import hints  # noqa: E402
import orclient  # noqa: E402
import rulebook  # noqa: E402
import scoreboard  # noqa: E402
from engine import ft  # noqa: E402
from harness import _aggregate  # noqa: E402

DECISIONS = tuple(scoreboard.DECISIONS)
ALL = frozenset(DECISIONS)
BASELINE, KEY = "base", "key"
ARMS = (BASELINE, KEY, *(f"drop-{d}" for d in DECISIONS), *(f"only-{d}" for d in DECISIONS),
        *(f"wrong-{d}" for d in DECISIONS), "soft-key", "walk", "soft-walk",
        "cand-property-2", "cand-property-3")
FOUND = "direct-all"                # the stored walk the `walk` arms state
CONFUSION_WALKS = ("direct", "direct-all")
REPETITIONS = 10
SCHEDULE_SEED = 20261124
RUN_LIMIT_USD = 12.0                # the most this stage may book, over every launch of it
DEV_LIMIT_USD = 0.25                # the same for the dev check on invented variables
CONCURRENCY = 24
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = analysis.BOOTSTRAP_DRAWS, analysis.BOOTSTRAP_SEED
CALLS = orclient.OUTPUT / "decisions-calls.jsonl"
SCORES = orclient.OUTPUT / "decisions-scores.jsonl"
SUMMARY = orclient.OUTPUT / "decisions-summary.json"
DEV_CALLS = orclient.OUTPUT / "decisions-dev-calls.jsonl"
DEV_ARMS = ("key", "soft-key", "cand-property-2", "drop-matrix", "only-property", "wrong-matrix")
DEV_REPETITIONS = 2

FORM_HARD = "The decomposition must have this form:"
PATTERNS_HARD = "It follows these I-ADOPT patterns:"
SOFT_NOTE = ("This guidance is a suggestion and may be wrong. Where the definition clearly disagrees, "
             "follow the definition.")
FORM_SOFT = "The decomposition probably has this form:"
PATTERNS_SOFT = "It probably follows these I-ADOPT patterns:"
CANDIDATES_LINE = "- Property: one of the following patterns; choose the one that fits the definition."
PRESENCE = ("matrix", "context", "modifier", "constraint")   # decisions that are present or absent


def render(facts: Mapping[str, Any], stated: frozenset[str] | set[str], patterns: Mapping[str, Any],
           types: Sequence[Mapping[str, Any]], soft: bool = False, candidates: Sequence[int] = (),
           order_key: str = "") -> str:
    """Write the guidance block with only some decisions stated.

    With every decision stated, no candidates and hard wording this is `hints.render` with all
    four features, byte for byte. A decision that is left out loses its form line and its pattern
    lines; the line on the object's shape belongs to the object decision when an object pattern
    gives the shape and to the property decision when only the property pattern does.

    Args: facts: a pattern set; stated: the decisions to write; patterns, types: the rulebook
        data; soft: word the block as a suggestion; candidates: property patterns to list in
        place of the one property line; order_key: decides the candidates' order.
    Returns: the block, starting with two newlines; "" when no decision is stated.
    Raises: ValueError for an unknown decision.
    Side Effects: none. Deterministic.
    """
    stated = frozenset(stated)
    if stated - ALL:
        raise ValueError(f"unknown decisions: {sorted(stated - ALL)}")
    if not stated:
        return ""
    facts = dict(facts)
    form = hints.expected_form(facts, patterns, types)
    shape_owner = "object" if facts["object"] else "property"
    form_lines = []
    if form["object"] and shape_owner in stated:
        form_lines.append(f"- hasObjectOfInterest: {hints.SHAPE_TEXT[form['object']]}")
    if "matrix" in stated:
        text = hints.NONE_TEXT if form["matrix"] == "empty" else hints.SHAPE_TEXT[form["matrix"]]
        form_lines.append(f"- hasMatrix: {text}")
    if "context" in stated:
        form_lines.append(f"- hasContextObject: {'one term.' if form['context'] else hints.NONE_TEXT}")
    if "modifier" in stated:
        form_lines.append("- hasStatisticalModifier: "
                          f"{'present.' if form['statistical_modifier'] else hints.NONE_TEXT}")
    names = form["constraint_types"]
    if "constraint" in stated:
        if names:
            form_lines.append("- hasConstraint: one or more constraints of each of these types and of no "
                              f"other type: {'; '.join(names)}. Each label starts with its type and a colon.")
        else:
            form_lines.append("- hasConstraint: none; use [].")
    pattern_lines = []
    if "property" in stated and candidates:
        pattern_lines.append(CANDIDATES_LINE)
        ordered = sorted(candidates, key=lambda n: hashlib.sha256(f"{order_key}:{n}".encode()).hexdigest())
        pattern_lines += ["  " + hints._pattern_line("Property", n, patterns, facts["names"], True) for n in ordered]
    for part, key in (("Property", "property"), ("Object of interest", "object"),
                      ("Matrix", "matrix"), ("Context object", "context")):
        if key in stated and not (key == "property" and candidates):
            pattern_lines += [hints._pattern_line(part, n, patterns, facts["names"], True) for n in facts[key]]
    lines = ["", "", hints.HEADING]
    if soft:
        lines.append(SOFT_NOTE)
    if form_lines:
        lines += [FORM_SOFT if soft else FORM_HARD, *form_lines]
    if pattern_lines:
        lines += [PATTERNS_SOFT if soft else PATTERNS_HARD, *pattern_lines]
    if "constraint" in stated and names:
        by_name = {row["name"].lower(): row for row in types}
        lines.append("Constraint types:")
        for name in names:
            definition = (by_name.get(name, {}).get("definition") or "").strip()
            lines.append(f"- {name}: {definition}" if definition else f"- {name}")
    return "\n".join(lines)


def _ranked(counter: Counter) -> list[int]:
    """Numbers by falling count, the lower number first among equals.

    Args: counter: counts by number.
    Returns: the numbers.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return [number for number, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def tables(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """What a wrong statement says: the usual alternatives, read from the tags and the stored walks.

    Args: ctx: from `load`.
    Returns: {"matrix", "context", "constraint": the most frequent pattern of that kind in the
        experts' tags, "object_other": the most frequent object pattern that is not one term,
        "confused": {right property pattern: the patterns the stored walks gave instead, most
        frequent first}, "walk_errors": every wrong property the walks gave, most frequent first,
        "frequent": the experts' property patterns, most frequent first, "reachable": every
        property pattern the trees can reach}.
    Raises: ValueError when a kind has no tagged pattern.
    Side Effects: reads the stored walk files. Deterministic for given files.
    """
    key, patterns = ctx["key"], ctx["patterns"]
    counts = {field: Counter(n for entry in key.values() for n in entry[field])
              for field in ("matrix", "context", "constraint", "object", "property")}
    other = Counter({n: c for n, c in counts["object"].items()
                     if patterns.get(str(n), {}).get("shape", "single") != "single"})
    if not all((counts["matrix"], counts["context"], counts["constraint"], other)):
        raise ValueError("the tags lack a matrix, context, constraint or non-single object pattern")
    by_right: dict[int, Counter] = {}
    errors: Counter = Counter()
    for name in CONFUSION_WALKS:
        for variable_id, facts in scoreboard.stored_walk(ctx, name).items():
            right = key[variable_id]["property"][0]
            given = (facts or {}).get("property") or []
            if given and given[0] != right:
                by_right.setdefault(right, Counter())[given[0]] += 1
                errors[given[0]] += 1
    return {"matrix": _ranked(counts["matrix"])[0], "context": _ranked(counts["context"])[0],
            "constraint": _ranked(counts["constraint"])[0], "object_other": _ranked(other)[0],
            "confused": {right: _ranked(counter) for right, counter in by_right.items()},
            "walk_errors": _ranked(errors), "frequent": _ranked(counts["property"]),
            "reachable": list(ctx["reachable"])}


def wrong_properties(right: int, table: Mapping[str, Any], count: int) -> list[int]:
    """The property patterns stated in place of, or next to, the right one.

    Args: right: the right property pattern; table: from `tables`; count: how many to return.
    Returns: `count` distinct property patterns, none of them `right`: first what the stored walks
        confused it with, then the walks' most frequent wrong answers, then the experts' most
        frequent patterns, then every reachable pattern in number order.
    Raises: ValueError when fewer than `count` other patterns exist.
    Side Effects: none. Deterministic.
    """
    out: list[int] = []
    for number in (*table["confused"].get(right, ()), *table["walk_errors"], *table["frequent"],
                   *sorted(table["reachable"])):
        if number != right and number not in out:
            out.append(number)
    if len(out) < count:
        raise ValueError(f"fewer than {count} property patterns besides {right}")
    return out[:count]


def wrong_facts(facts: Mapping[str, Any], decision: str, table: Mapping[str, Any],
                patterns: Mapping[str, Any]) -> dict[str, Any]:
    """A pattern set with one decision changed to a wrong value and the others kept.

    Args: facts: the right pattern set; decision: the decision to get wrong; table: from
        `tables`; patterns: the pattern table.
    Returns: a copy: another property pattern; the other object shape (one term against a ratio);
        a matrix, a context object, a modifier or constraints stated where there are none and
        denied where there are some.
    Raises: KeyError for an unknown decision.
    Side Effects: none. Deterministic.
    """
    out = dict(facts, names=dict(facts.get("names", {})))
    if decision == "property":
        out["property"] = wrong_properties(facts["property"][0], table, 1)
    elif decision == "object":
        single = hints.object_shape(dict(facts), patterns) == "single"
        out["object"] = [table["object_other"]] if single else [33]
    elif decision == "modifier":
        out["statistical_modifier"] = not facts["statistical_modifier"]
    elif decision in ("matrix", "context", "constraint"):
        out[decision] = [] if facts[decision] else [table[decision]]
    else:
        raise KeyError(decision)
    return out


def block(ctx: Mapping[str, Any], arm: str, facts: Mapping[str, Any], found: Mapping[str, Any] | None,
          order_key: str, table: Mapping[str, Any]) -> str:
    """The guidance block one arm appends for one variable.

    Args: ctx: from `load`; arm: an arm name; facts: the experts' pattern set; found: the pattern
        set the model found, or None; order_key: the variable's id; table: from `tables`.
    Returns: the block; "" for the baseline and for a `walk` arm without a found set.
    Raises: KeyError for an unknown arm.
    Side Effects: none. Deterministic.
    """
    patterns, types = ctx["patterns"], ctx["types"]
    if arm == BASELINE:
        return ""
    if arm in (KEY, "soft-key"):
        return render(facts, ALL, patterns, types, soft=arm == "soft-key")
    if arm in ("walk", "soft-walk"):
        return render(found, ALL, patterns, types, soft=arm == "soft-walk") if found else ""
    kind, _, rest = arm.partition("-")
    if kind == "drop" and rest in ALL:
        return render(facts, ALL - {rest}, patterns, types)
    if kind == "only" and rest in ALL:
        return render(facts, {rest}, patterns, types)
    if kind == "wrong" and rest in ALL:
        return render(wrong_facts(facts, rest, table, patterns), ALL, patterns, types)
    if arm in ("cand-property-2", "cand-property-3"):
        count = int(arm.rsplit("-", 1)[1])
        numbers = [facts["property"][0], *wrong_properties(facts["property"][0], table, count - 1)]
        return render(facts, ALL, patterns, types, candidates=numbers, order_key=order_key)
    raise KeyError(arm)


def load() -> dict[str, Any]:
    """Everything this stage needs. No network, no secrets.

    Args: none.
    Returns: the context of `groups.load` (cleaned pattern texts) plus "ids" (the 102 variables,
        sorted), "group_of", "found" (the stored `direct-all` pattern sets) and "table".
    Raises: as `groups.load`; KeyError when a stored walk is missing.
    Side Effects: reads local files.
    """
    ctx = groups.load()
    by_group = scoreboard.ids_by_group(ctx)
    ctx["ids"] = sorted(v for ids in by_group.values() for v in ids)
    ctx["group_of"] = {v: group for group, ids in by_group.items() for v in ids}
    ctx["found"] = scoreboard.stored_walk(ctx, FOUND)
    ctx["table"] = tables(ctx)
    return ctx


def prompt_of(ctx: Mapping[str, Any], job: engine.Job) -> str:
    """The prompt of one grid call.

    Args: ctx: from `load`; job: (arm, repetition, variable_id).
    Returns: today's prompt for that variable (an example variable is shown the other 24
        examples) with the arm's block after the definition.
    Raises: KeyError for an unknown arm or variable.
    Side Effects: none. Deterministic.
    """
    arm, _, variable_id = job
    own = groups.context_for(ctx, variable_id)
    facts = hints.facts_from_key(ctx["key"][variable_id])
    text = block(ctx, arm, facts, ctx["found"].get(variable_id), variable_id, ctx["table"])
    return ft.prompt_for(own["inputs"], "B", ctx["records"][variable_id]["definition"] + text)


def schedule(ctx: Mapping[str, Any]) -> list[engine.Job]:
    """Every call of the grid, all arms of one variable side by side.

    Args: ctx: from `load`.
    Returns: (arm, repetition, variable_id) jobs, each exactly once.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return engine.schedule(ARMS, ctx["ids"], REPETITIONS, SCHEDULE_SEED)


def fingerprints(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Hash what the stage sends: code, data, stored walks, the wrong values and every prompt.

    Args: ctx: from `load`.
    Returns: hex digests and counts.
    Raises: OSError when a file cannot be read.
    Side Effects: reads files. Deterministic.
    """
    out: dict[str, Any] = {name: ceiling.file_sha256(HERE / name)[:16] for name in (
        "decision_value.py", "orclient.py", "scoreboard.py", "hints.py", "engine.py", "groups.py",
        "settings.py", "walk.py", "route_key.py", "rulebook.py", "ceiling.py", "analysis.py")}
    out.update({f"data/{name}": ceiling.file_sha256(rulebook.DATA / name)[:16] for name in (
        "trees.json", "patterns.json", "constraint_types.json", "route_key.json", "issues_snapshot.json")})
    for name in CONFUSION_WALKS:
        for prefix in ("route", "route-held-out", "route-examples"):
            path = engine.OUTPUT / f"{prefix}-{name}.jsonl"
            out[f"stored/{path.name}"] = ceiling.file_sha256(path)[:16]
    out["population_sha256"] = ft.sha256("\n".join(ctx["ids"]))[:16]
    out["table_sha256"] = ft.sha256(json.dumps(ctx["table"], sort_keys=True))[:16]
    out["prompts"] = {}
    for arm in ARMS:
        prompts = [prompt_of(ctx, (arm, 1, v)) for v in ctx["ids"]]
        out["prompts"][arm] = {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                               "chars_mean": round(sum(len(p) for p in prompts) / len(prompts), 1)}
    return out


# --- scoring and comparison --------------------------------------------------------------

def complete_repetitions(ctx: Mapping[str, Any], calls: Mapping[engine.Job, dict],
                         scores: Mapping[engine.Job, dict], arms: Sequence[str]) -> list[int]:
    """The repetitions in which every one of the given arms was answered for every variable.

    Args: ctx: from `load`; calls: from `engine.latest`; scores: from `engine.score`; arms: the
        arms that must be complete.
    Returns: the repetition numbers, in order. A repetition with a missing or failed call in any
        of the arms is left out, so no comparison mixes finished and unfinished runs.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return [repetition for repetition in range(1, REPETITIONS + 1)
            if all((arm, repetition, v) in scores and calls.get((arm, repetition, v), {}).get("ok")
                   for arm in arms for v in ctx["ids"])]


def series(scores: Mapping[engine.Job, dict], arm: str, ids: Sequence[str],
           repetitions: Sequence[int]) -> list[float]:
    """One arm's Close F1 per repetition over some variables; an unusable answer counts as empty.

    Args: scores: from `engine.score`; arm: an arm name; ids: the variables; repetitions: the
        repetitions to score, from `complete_repetitions`.
    Returns: one F1 per repetition, in order.
    Raises: KeyError when a repetition lacks a score.
    Side Effects: none. Deterministic.
    """
    out = []
    for repetition in repetitions:
        keys = [(arm, repetition, v) for v in ids]
        out.append(_aggregate([scores[k]["evaluation"] for k in keys], [k[2] for k in keys])["f1"])
    return out


def gain(counts: Mapping[str, Mapping[str, Sequence[float]]], first: str, second: str,
         ids: Sequence[str], draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    """One arm against another over some variables, with the variables resampled.

    Args: counts: from `analysis.variable_counts`; first, second: arm names; ids: the variables;
        draws, seed: of the resampling. Every comparison draws the same resamples.
    Returns: {"gain": pooled Close F1 of `first` minus `second`, "low", "high": the 2.5% and 97.5%
        points over resampled variables, "share_positive"}.
    Raises: KeyError when an arm lacks a variable.
    Side Effects: none. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    ids = sorted(ids)
    drawn = sorted(analysis.pooled_gain(counts[first], counts[second], rng.choices(ids, k=len(ids)))
                   for _ in range(draws))
    return {"gain": analysis.pooled_gain(counts[first], counts[second], ids),
            "low": drawn[int(0.025 * draws)], "high": drawn[int(0.975 * draws) - 1],
            "share_positive": sum(g > 0 for g in drawn) / draws}


def break_even(counts: Mapping[str, Mapping[str, Sequence[float]]], decision: str, ids: Sequence[str],
               draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """How often a decision must be right before stating it beats leaving it out.

    With the other five decisions right, stating decision D with accuracy p scores
    p * F1(key) + (1 - p) * F1(wrong-D), and leaving it out scores F1(drop-D). The two are equal
    at p = harm / (value + harm), where value = F1(key) - F1(drop-D) and
    harm = F1(drop-D) - F1(wrong-D).

    Args: counts: from `analysis.variable_counts`; decision: a decision; ids: the variables;
        draws, seed: of the resampling.
    Returns: {"value", "harm", "accuracy": the point values (accuracy None when value + harm is
        not positive), "low", "high": the 2.5% and 97.5% points of the accuracy over the resamples
        in which it is defined, "defined_share": the share of such resamples}.
    Raises: KeyError when an arm lacks a variable.
    Side Effects: none. Deterministic for a given seed.
    """
    def point(sample: Sequence[str]) -> tuple[float, float]:
        """Value and harm over one list of variables.

        Args: sample: variable ids, repeats allowed.
        Returns: (value, harm).
        Raises: KeyError when an arm lacks a variable.
        Side Effects: none.
        """
        return (analysis.pooled_gain(counts[KEY], counts[f"drop-{decision}"], sample),
                analysis.pooled_gain(counts[f"drop-{decision}"], counts[f"wrong-{decision}"], sample))

    ids = sorted(ids)
    value, harm = point(ids)
    rng = random.Random(seed)
    drawn = []
    for _ in range(draws):
        v, h = point(rng.choices(ids, k=len(ids)))
        if v + h > 0:
            drawn.append(min(1.0, max(0.0, h / (v + h))))
    drawn.sort()
    accuracy = min(1.0, max(0.0, harm / (value + harm))) if value + harm > 0 else None
    return {"value": value, "harm": harm, "accuracy": accuracy,
            "low": drawn[int(0.025 * len(drawn))] if drawn else None,
            "high": drawn[int(0.975 * len(drawn)) - 1] if drawn else None,
            "defined_share": len(drawn) / draws}


def contrasts() -> list[tuple[str, str, str]]:
    """The comparisons written down before the run: (name, first arm, second arm).

    Args: none.
    Returns: the list, in the order of the report.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out = [("key vs base", KEY, BASELINE), ("walk vs base", "walk", BASELINE),
           ("soft-key vs key", "soft-key", KEY), ("soft-walk vs walk", "soft-walk", "walk"),
           ("soft-walk vs base", "soft-walk", BASELINE), ("soft-key vs base", "soft-key", BASELINE)]
    out += [(f"value of {d} (key vs drop-{d})", KEY, f"drop-{d}") for d in DECISIONS]
    out += [(f"{d} alone (only-{d} vs base)", f"only-{d}", BASELINE) for d in DECISIONS]
    out += [(f"harm of a wrong {d} (drop-{d} vs wrong-{d})", f"drop-{d}", f"wrong-{d}") for d in DECISIONS]
    out += [(f"wrong-{d} vs base", f"wrong-{d}", BASELINE) for d in DECISIONS]
    for count in (2, 3):
        arm = f"cand-property-{count}"
        out += [(f"{arm} vs drop-property", arm, "drop-property"), (f"{arm} vs key", arm, KEY)]
    return out


def summarise(ctx: Mapping[str, Any], calls: Mapping[engine.Job, dict],
              scores: Mapping[engine.Job, dict], draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    """Per-arm scores, every pre-specified comparison and the break-even accuracies.

    Only repetitions that are complete for every arm that was run enter any figure, so a stopped
    or half-resumed run cannot pass for a finished one: "complete" is true only when all 25 arms
    have all planned repetitions.

    Args: ctx: from `load`; calls: from `engine.latest`; scores: from `engine.score`; draws:
        resamples of the variables.
    Returns: {"generated_at", "variables", "repetitions_planned", "repetitions_used", "complete",
        "arms": {arm: {...}}, "contrasts": {name: {...}}, "break_even": {decision: {...}},
        "by_presence": {decision: {"present" / "absent": {...}}}, "spent_usd"}. A comparison holds
        "gain", "low", "high" and "share_positive" over resampled variables and the Welch test
        over repetitions with its Holm-adjusted p-value.
    Raises: nothing; an arm that was not run at all is left out, with its comparisons.
    Side Effects: none.
    """
    by_group = {group: [v for v in ctx["ids"] if ctx["group_of"][v] == group] for group in scoreboard.GROUPS}
    run = [arm for arm in ARMS if any(key[0] == arm for key in calls)]
    used = complete_repetitions(ctx, calls, scores, run)
    out: dict[str, Any] = {"generated_at": ft.now(), "variables": len(ctx["ids"]),
                           "repetitions_planned": REPETITIONS, "repetitions_used": used,
                           "complete": list(run) == list(ARMS) and len(used) == REPETITIONS,
                           "arms": {}, "contrasts": {}, "break_even": {}, "by_presence": {}, "spent_usd": 0.0}
    if not used:
        return out
    arms: dict[str, Any] = out["arms"]
    for arm in run:
        rows = [calls[(arm, r, v)] for r in range(1, REPETITIONS + 1) for v in ctx["ids"] if (arm, r, v) in calls]
        every = series(scores, arm, ctx["ids"], used)
        answered = [row for row in rows if row.get("ok")]
        arms[arm] = {"repetitions": len(every), "f1_mean": statistics.mean(every),
                     "f1_sd": statistics.stdev(every) if len(every) > 1 else None, "f1_per_repetition": every,
                     "by_group": {g: statistics.mean(series(scores, arm, ids, used)) for g, ids in by_group.items()},
                     "invalid": sum(1 for key, row in scores.items()
                                    if key[0] == arm and key[1] in used and not row["valid_prediction"]),
                     "failed_calls": len(rows) - len(answered),
                     "reasoning_answers": sum(1 for row in answered if row.get("reasoning_chars")
                                              or row.get("reasoning_tokens")),
                     "prompt_tokens_mean": ft._mean(row.get("prompt_tokens") for row in answered),
                     "completion_tokens_mean": ft._mean(row.get("completion_tokens") for row in answered),
                     "cost_usd": sum(row.get("cost_usd") or 0.0 for row in rows)}
    counts = analysis.variable_counts({key: row for key, row in scores.items() if key[0] in arms and key[1] in used})
    table: dict[str, Any] = out["contrasts"]
    for name, first, second in contrasts():
        if first not in arms or second not in arms:
            continue
        table[name] = dict(gain(counts, first, second, ctx["ids"], draws=draws), first=first, second=second)
        if len(used) > 1:
            table[name]["welch"] = ft.welch(arms[first]["f1_per_repetition"], arms[second]["f1_per_repetition"])
    adjusted = ft.holm({name: row["welch"]["p"] for name, row in table.items() if "welch" in row})
    for name, value in adjusted.items():
        table[name]["p_holm"] = value
    out["spent_usd"] = sum(cell["cost_usd"] for cell in arms.values())
    for decision in DECISIONS:
        if all(name in arms for name in (KEY, f"drop-{decision}", f"wrong-{decision}")):
            out["break_even"][decision] = break_even(counts, decision, ctx["ids"], draws=draws)
    for decision in PRESENCE:
        if not all(name in arms for name in (KEY, f"drop-{decision}", f"wrong-{decision}")):
            continue
        has = {v: bool(ctx["key"][v]["statistical_modifier" if decision == "modifier" else decision])
               for v in ctx["ids"]}
        out["by_presence"][decision] = {}
        for label, wanted in (("present", True), ("absent", False)):
            ids = [v for v in ctx["ids"] if has[v] == wanted]
            out["by_presence"][decision][label] = {
                "variables": len(ids),
                "value": gain(counts, KEY, f"drop-{decision}", ids, draws=draws),
                "harm": gain(counts, f"drop-{decision}", f"wrong-{decision}", ids, draws=draws)}
    return out


async def dev(ctx: Mapping[str, Any], transport: orclient.Transport, guard: orclient.Guard) -> list[dict]:
    """Try the new wordings on the invented variables of `ceiling.py`.

    Args: ctx: from `load`; transport: sends one request; guard: the stop rules.
    Returns: one row per call: the answer, whether it is a valid decomposition, and whether it has
        the form the block states.
    Raises: nothing from the provider.
    Side Effects: provider calls for invented definitions only; appends to the dev calls file.
    """
    from iadopt_lab.generation.extractor import extract_json
    from iadopt_lab.validation import validate_prediction

    items = {item["id"]: dict(item, statistical_modifier=item.get("statistical_modifier", False), names={})
             for item in ceiling.INVENTED}

    def stated(arm: str, facts: Mapping[str, Any]) -> dict[str, Any]:
        """The pattern set an arm's block states.

        Args: arm: an arm name; facts: the invented variable's right pattern set.
        Returns: the wrong set for a `wrong-` arm, else the right one.
        Raises: nothing.
        Side Effects: none.
        """
        return wrong_facts(facts, arm.partition("-")[2], ctx["table"], ctx["patterns"]) if arm.startswith("wrong-") else dict(facts)

    jobs = engine.schedule(DEV_ARMS, list(items), DEV_REPETITIONS, SCHEDULE_SEED)
    await orclient.run_jobs(
        jobs, lambda job: ft.prompt_for(ctx["inputs"], "B", items[job[2]]["definition"]
                                        + block(ctx, job[0], items[job[2]], None, job[2], ctx["table"])),
        transport, DEV_CALLS, guard, concurrency=8)
    rows = []
    for key, record in sorted(engine.latest(DEV_CALLS).items()):
        prediction = None
        extracted = extract_json(record.get("answer") or "")
        if extracted.success and validate_prediction(extracted.candidate, ctx["inputs"]["schema"]).valid:
            prediction = extracted.candidate
        form = hints.expected_form(stated(key[0], items[key[2]]), ctx["patterns"], ctx["types"])
        rows.append({"key": key, "ok": record.get("ok"), "prediction": prediction,
                     "reasoning": bool(record.get("reasoning_chars") or record.get("reasoning_tokens")),
                     "conformance": hints.conformance(prediction, form)})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode: plan, dev, run or score.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status, 0 on success.
    Raises: SystemExit on bad arguments or a missing key.
    Side Effects: --dev and --run call the provider; --dev, --run and --score write output files.
    """
    parser = argparse.ArgumentParser(description="The value of each pattern decision (stage 1).")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--plan-only", "--dev", "--run", "--score"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--max-calls", type=int, default=None, help="issue at most this many new calls")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args(argv)
    ctx = load()
    jobs = schedule(ctx)

    if args.plan_only:
        print(json.dumps({"variables": len(ctx["ids"]), "arms": list(ARMS), "repetitions": REPETITIONS,
                          "calls": len(jobs), "run_limit_usd": RUN_LIMIT_USD, "host": orclient.HOST,
                          "model": orclient.MODEL, "table": ctx["table"], **fingerprints(ctx)}, indent=2))
        return 0
    if args.score:
        scores = engine.score(CALLS, SCORES, ctx["records"], ctx["inputs"]["schema"])
        summary = summarise(ctx, engine.latest(CALLS), scores)
        SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not summary["complete"]:
            print(f"NOT COMPLETE: {len(summary['arms'])} of {len(ARMS)} arms, repetitions used "
                  f"{summary['repetitions_used']} of {REPETITIONS}; the figures below cover those only")
        for arm, cell in summary["arms"].items():
            print(f"{arm:18} reps {cell['repetitions']:2}  F1 {cell['f1_mean']:.4f} (SD {cell['f1_sd'] or 0:.4f})  "
                  + "  ".join(f"{g} {value:.3f}" for g, value in cell["by_group"].items())
                  + f"  invalid {cell['invalid']:3}  failed {cell['failed_calls']}  ${cell['cost_usd']:.2f}")
        for name, row in summary["contrasts"].items():
            runs = f"runs p {row['p_holm']:.2g}" if "p_holm" in row else "runs p -"
            print(f"{name:46} {row['gain']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}]  {runs}")
        for decision, row in summary["break_even"].items():
            accuracy = "undefined" if row["accuracy"] is None else f"{row['accuracy']:.2f}"
            print(f"break-even {decision:11} value {row['value']:+.4f} harm {row['harm']:+.4f} "
                  f"accuracy {accuracy} [{row['low']}, {row['high']}] defined {row['defined_share']:.2f}")
        print(f"spent ${summary['spent_usd']:.2f}")
        return 0 if summary["complete"] else 1

    async def go() -> int:
        """Open the transport, run the dev check or the grid, and always close the client.

        Args: none.
        Returns: the exit status: 1 when a stop rule ended the run.
        Raises: SystemExit when the key is missing or another process holds the ledger.
        Side Effects: provider calls, the ledger and output files.
        """
        purpose = "decisions-dev" if args.dev else "decisions"
        spend = orclient.Spend(purpose=purpose, allowance=DEV_LIMIT_USD if args.dev else RUN_LIMIT_USD)
        guard = orclient.Guard(spend)
        transport, client = orclient.open_transport(spend, purpose, concurrency=args.concurrency)
        try:
            if args.dev:
                for row in await dev(ctx, transport, guard):
                    flags = "".join("Y" if ok else "n" for ok in row["conformance"].values())
                    print(row["key"], "ok" if row["ok"] else "FAILED", "reasoning" if row["reasoning"] else "",
                          flags, json.dumps(row["prediction"], ensure_ascii=False))
                print("stopped:", guard.reason, "spent in all: $%.4f" % spend.spent)
                return 0
            result = await orclient.run_jobs(jobs, lambda job: prompt_of(ctx, job), transport, CALLS, guard,
                                             concurrency=args.concurrency, max_calls=args.max_calls)
            print(json.dumps(result, indent=2))
            return 1 if result["stopped"] else 0
        finally:
            await orclient.close_transport(transport, client)

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
