#!/usr/bin/env python
"""Written conventions for the pattern finder (a pilot), and what right decisions add downstream.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/CONVENTIONS_TEST_20261127.md` (private repository).

Arvin, 2026-11-27: "plan the test of the rules in the walk", then in the question dialog "Pilot
now", "Yes, download", "Run it in any case" (the ceiling arm) and "Up to $4".

Part 1 is stage 3's one-call pattern finder (`knowledge.py`) in its setting D+S: the examples are
the variables of the other five folds with the experts' decompositions, the most similar last.
Both arms use the constraint-type table completed by `data/types_overlay.json` and an answer that
first names the object of interest, the matrix entity and the restricting phrases. Arm `rules`
adds the written conventions of `data/conventions_v1.txt` (worded with invented examples). Five
runs of the 102 variables, and the same for the fresh tagged entries of the examples repository,
which see all 102 as examples.

Part 2 is today's prompt with the 25 most similar decomposed variables as examples (stage 5's
`rag-25`): `rag-25`; `rag-25+experts`, the experts' matrix and constraint types stated (the
ceiling, run in any case); `rag-25+rules`, the decisions of arm `rules` (run 1) that passed the
gate. Ten runs. All calls book under one purpose, so the allowance covers every part.

    ./conventions_test.py --plan-only
    ./conventions_test.py --fresh         # read the downloaded files of the fresh entries
    ./conventions_test.py --dev           # both arms on invented variables only
    ./conventions_test.py --run-part1
    ./conventions_test.py --score-part1   # marks, comparisons and the gate
    ./conventions_test.py --run-part2
    ./conventions_test.py --score-part2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import analysis  # noqa: E402
import ceiling  # noqa: E402
import draft_rag as dr  # noqa: E402
import engine  # noqa: E402
import groups  # noqa: E402
import hints  # noqa: E402
import knowledge as kn  # noqa: E402
import orclient  # noqa: E402
import rag_tuning as rt  # noqa: E402
import route_key  # noqa: E402
import rulebook  # noqa: E402
import scoreboard  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402

ARMS = ("plain", "rules")
PART2_ARMS = ("rag-25", "rag-25+experts", "rag-25+rules")
REPETITIONS, PART2_REPETITIONS = 5, 10
SCHEDULE_SEED, FRESH_SEED, PART2_SEED = 20261201, 20261202, 20261203
REQUEST = dict(kn.REQUEST, max_tokens=700)           # the answer now names entities and phrases first
PURPOSE, ALLOWANCE_USD = "conventions", 4.0          # Arvin, 2026-11-27: "Up to $4", every part together
CONCURRENCY = 16
DRAWS, SEED = 10000, 20261204
FAMILY = 2                                           # the matrix and the constraint types
STRICT = 0.05 / FAMILY
BREAK_EVEN = {"matrix": 0.57, "constraint": 0.54}    # stage 1: right more often than this, or stating hurts
DECIDED = ("matrix", "constraint")
COLUMNS = ("matrix", "constraint", "matrix_pattern", "both", "complete", "parsed", "form")
OVERLAY = rulebook.DATA / "types_overlay.json"
CONVENTIONS = rulebook.DATA / "conventions_v1.txt"
CONVENTIONS_HEADING = "CONVENTIONS OF THE EXPERTS (written rules; follow them)"
INSTRUCTION_START = 'Return one JSON object with these keys: "property"'
INSTRUCTION = (
    'Return one JSON object with these keys, in this order: "object_of_interest" (the entity whose property is '
    'measured, in a few words), "matrix_entity" (the entity that is the matrix, in a few words, or null when '
    'there is no matrix), "restrictions" (a list of the phrases of the definition that narrow the variable; [] '
    'when there are none), "property" (one property pattern number), "matrix" (one matrix pattern number, or '
    'null when there is no matrix), "context" (true or false: is a context object needed), "modifier" (true or '
    'false: is a statistical modifier involved) and "constraints" (a list of the constraint type names of the '
    'restrictions; [] when there are none). Return the JSON only.')
DEIONIZED = "deionized water"
CALLS1 = orclient.OUTPUT / "conventions-part1-calls.jsonl"
SUMMARY1 = orclient.OUTPUT / "conventions-part1-summary.json"
CALLS2 = orclient.OUTPUT / "conventions-part2-calls.jsonl"
SCORES2 = orclient.OUTPUT / "conventions-part2-scores.jsonl"
SUMMARY2 = orclient.OUTPUT / "conventions-part2-summary.json"
DEV_CALLS = orclient.OUTPUT / "conventions-dev-calls.jsonl"
FRESH_DIR = orclient.OUTPUT / "fresh"
FRESH = FRESH_DIR / "fresh-variables.json"


# --- what the finder is shown -------------------------------------------------------------------

def overlay_types(types: Sequence[Mapping[str, Any]], overlay: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The constraint-type table with the overlay's renamings and additions.

    Args: types: the rulebook's table; overlay: the parsed overlay file.
    Returns: a new table: renamed rows keep their number, category and targets.
    Raises: ValueError when a renamed row is missing or does not carry its expected old name, or an
        added number is already in the table.
    Side Effects: none. Deterministic.
    """
    out = [dict(row) for row in types]
    by_number = {row["number"]: row for row in out}
    for change in overlay["rename"]:
        row = by_number.get(change["number"])
        if row is None or row["name"] != change["from"]:
            raise ValueError(f"type {change['number']} is not '{change['from']}' in the table")
        row.update(name=change["name"], definition=change["definition"])
    for added in overlay["add"]:
        if added["number"] in by_number:
            raise ValueError(f"type {added['number']} is already in the table")
        out.append({key: added[key] for key in ("category", "definition", "name", "number", "observation", "targets")})
    return out


def entity_text(value: Any) -> str | None:
    """An entity as words: a plain label, or a system's own name.

    Args: value: a lexical value of the decomposition.
    Returns: the text, or None for an empty value.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if isinstance(value, str):
        return value.strip() or None
    return (value.get("AsymmetricSystem") or value.get("SymmetricSystem") or "").strip() or None


def answer_of(ctx: Mapping[str, Any], entry: Mapping[str, Any], gold: Mapping[str, Any]) -> dict[str, Any]:
    """An example's answer in the form the prompt asks for, from the experts' tags and decomposition.

    Args: ctx: from `load`; entry: the experts' tags; gold: their decomposition.
    Returns: {"object_of_interest", "matrix_entity", "restrictions", then the five keys of
        `walk.direct_answer`}.
    Raises: IndexError when the entry has no property tag.
    Side Effects: none. Deterministic.
    """
    restrictions = [(c["label"].split(":", 1)[1] if ":" in c["label"] else c["label"]).strip()
                    for c in gold.get("hasConstraint", [])]
    # A missing field counts as empty, as `knowledge.one_line` reads the invented decompositions.
    return {"object_of_interest": entity_text(gold.get("hasObjectOfInterest", "")),
            "matrix_entity": entity_text(gold.get("hasMatrix", "")), "restrictions": restrictions,
            **walk.direct_answer(ctx, entry)}


def demo(ctx: Mapping[str, Any], item_id: str, definition: str, entry: Mapping[str, Any],
         gold: Mapping[str, Any]) -> dict[str, str]:
    """One example variable as the prompts show it.

    Args: ctx: from `load`; item_id: its id; definition: its definition; entry: its tags; gold: its
        decomposition.
    Returns: {"id", "definition", "gold" (the decomposition on one line), "patterns" (the answer)}.
    Raises: IndexError when the entry has no property tag.
    Side Effects: none. Deterministic.
    """
    return {"id": item_id, "definition": definition, "gold": kn.one_line(gold),
            "patterns": json.dumps(answer_of(ctx, entry, gold), ensure_ascii=False)}


def load(fresh: bool = True) -> dict[str, Any]:
    """Everything Part 1 needs. No network, no secrets.

    Args: fresh: whether to read the fresh entries (written by `--fresh`).
    Returns: the context of `groups.load` with the completed type table and its questions, plus
        "ids", "group_of", "fold_of", "demos", "conventions" and "fresh" ({id: entry row}).
    Raises: as `groups.load` and `overlay_types`.
    Side Effects: reads local files.
    """
    ctx = groups.load()
    ctx["types"] = overlay_types(ctx["types"], json.loads(OVERLAY.read_text(encoding="utf-8")))
    ctx["constraint_questions"] = walk.constraint_questions(ctx["types"])
    by_group = scoreboard.ids_by_group(ctx)
    ctx["ids"] = sorted(v for ids in by_group.values() for v in ids)
    ctx["group_of"] = {v: group for group, ids in by_group.items() for v in ids}
    ctx["fold_of"] = scoreboard.folds(ctx)
    ctx["demos"] = {v: demo(ctx, v, ctx["records"][v]["definition"], ctx["key"][v], ctx["records"][v]["gold"])
                    for v in ctx["ids"]}
    ctx["conventions"] = CONVENTIONS.read_text(encoding="utf-8").strip()
    ctx["fresh"] = json.loads(FRESH.read_text(encoding="utf-8"))["variables"] if fresh and FRESH.exists() else {}
    return ctx


def head(ctx: Mapping[str, Any]) -> str:
    """Stage 3's instruction and pattern lists, asking for the longer answer.

    Args: ctx: from `load`.
    Returns: the head of the prompt.
    Raises: ValueError when the instruction line is not where stage 3 has it.
    Side Effects: none. Deterministic.
    """
    lines = kn.head(ctx).split("\n")
    if len(lines) < 2 or not lines[1].startswith(INSTRUCTION_START):
        raise ValueError("the pattern finder's instruction is not its second line")
    lines[1] = INSTRUCTION
    return "\n".join(lines)


def build(ctx: Mapping[str, Any], arm: str, definition: str, demos: Sequence[Mapping[str, str]]) -> str:
    """The prompt of one arm for one target.

    Args: ctx: from `load`; arm: one of `ARMS`; definition: the target's definition; demos: the
        examples it may be shown.
    Returns: the prompt: head, the conventions for `rules`, the examples with their decompositions
        (most similar last), then the target.
    Raises: KeyError for an unknown arm.
    Side Effects: none. Deterministic.
    """
    if arm not in ARMS:
        raise KeyError(arm)
    lines = [head(ctx)]
    if arm == "rules":
        lines += [CONVENTIONS_HEADING, ctx["conventions"], ""]
    lines.append("EXAMPLES")
    for example in kn.nearest_last(definition, demos):
        lines += [f"Definition: {example['definition']}", f"Decomposition: {example['gold']}",
                  f"Patterns: {example['patterns']}", ""]
    lines += [f"Definition: {definition}", "Patterns:"]
    return "\n".join(lines)


def shown(ctx: Mapping[str, Any], target: str) -> list[Mapping[str, str]]:
    """The examples a target may be shown.

    Args: ctx: from `load`; target: a corpus variable, or a fresh entry ("fresh:<issue>").
    Returns: the 85 variables of the other five folds; for a fresh entry all 102.
    Raises: KeyError for an unknown target.
    Side Effects: none. Deterministic.
    """
    if target in ctx["fresh"]:
        return [ctx["demos"][v] for v in ctx["ids"]]
    return kn.usable_demos(ctx, ctx["fold_of"][target])


def definition_of(ctx: Mapping[str, Any], target: str) -> str:
    """A target's definition.

    Args: ctx: from `load`; target: as in `shown`.
    Returns: the definition.
    Raises: KeyError for an unknown target.
    Side Effects: none. Deterministic.
    """
    return ctx["fresh"][target]["definition"] if target in ctx["fresh"] else ctx["records"][target]["definition"]


def entry_of(ctx: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    """A target's tags.

    Args: ctx: from `load`; target: as in `shown`.
    Returns: the experts' tags.
    Raises: KeyError for an unknown target.
    Side Effects: none. Deterministic.
    """
    return ctx["fresh"][target]["entry"] if target in ctx["fresh"] else ctx["key"][target]


def prompt1(ctx: Mapping[str, Any], job: engine.Job) -> str:
    """The prompt of one Part 1 call.

    Args: ctx: from `load`; job: (arm, run, target).
    Returns: the prompt.
    Raises: KeyError for an unknown arm or target.
    Side Effects: none. Deterministic.
    """
    arm, _, target = job
    return build(ctx, arm, definition_of(ctx, target), shown(ctx, target))


def schedule1(ctx: Mapping[str, Any]) -> list[engine.Job]:
    """Every Part 1 call: the 102 variables, then the fresh entries.

    Args: ctx: from `load`.
    Returns: (arm, run, target) jobs, each once.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    jobs = engine.schedule(ARMS, ctx["ids"], REPETITIONS, SCHEDULE_SEED)
    if ctx["fresh"]:
        jobs += engine.schedule(ARMS, sorted(ctx["fresh"]), REPETITIONS, FRESH_SEED)
    return jobs


# --- the fresh entries ---------------------------------------------------------------------------

def read_definition(path: Path) -> tuple[str | None, str | None]:
    """The definition and label of the one variable in a downloaded Turtle file (untrusted data).

    Args: path: the file.
    Returns: (rdfs:comment, rdfs:label) of the resource typed iop:Variable; None where missing or
        when the file does not parse or does not hold exactly one variable.
    Raises: nothing.
    Side Effects: reads the file.
    """
    from rdflib import RDF, RDFS, Graph, URIRef

    graph = Graph()
    try:
        graph.parse(path, format="turtle")
    except Exception:  # an unparsable download is left out, not fixed
        return None, None
    variables = list(graph.subjects(RDF.type, URIRef("https://w3id.org/iadopt/ont/Variable")))
    if len(variables) != 1:
        return None, None
    comment, label = graph.value(variables[0], RDFS.comment), graph.value(variables[0], RDFS.label)
    return (" ".join(str(comment).split()) or None) if comment else None, (str(label).strip() or None) if label else None


def fresh_rows(ctx: Mapping[str, Any], snapshot: Mapping[str, Any], folder: Path) -> dict[str, dict[str, Any]]:
    """The fresh entries that have a downloaded definition, with their tags.

    Args: ctx: from `load(fresh=False)`; snapshot: the stored issue snapshot; folder: where the
        downloaded files are (`ttl/<issue>.ttl`).
    Returns: {"fresh:<issue>": {"issue", "title", "definition", "label", "entry", "flags",
        "relative" (a corpus definition shares at least half of all its words)}}.
    Raises: nothing.
    Side Effects: reads the downloaded files.
    """
    words = {v: set(kn.words(ctx["records"][v]["definition"])) for v in ctx["ids"]}
    issues = {issue["number"]: issue for issue in snapshot["issues"]}
    out = {}
    for row in scoreboard.fresh(snapshot["issues"], ctx["key"]):
        path = folder / "ttl" / f"{row['issue']}.ttl"
        definition, label = read_definition(path) if path.exists() else (None, None)
        if not definition:
            continue
        mine = set(kn.words(definition))
        relative = any(len(mine & other) / len(mine | other) >= dr.RELATIVE_OVERLAP for other in words.values() if mine | other)
        entry = route_key.parse_labels(list(issues[row["issue"]]["labels"]))
        entry["issue"] = row["issue"]
        out[f"fresh:{row['issue']}"] = {"issue": row["issue"], "title": row["title"], "definition": definition,
                                        "label": label, "entry": entry, "flags": row["flags"], "relative": relative}
    return out


# --- marking ---------------------------------------------------------------------------------------

def form_ok(answer: str | None) -> bool:
    """Whether an answer also has the three new keys, properly typed.

    Args: answer: the model's reply, or None.
    Returns: True when "object_of_interest" is text, "matrix_entity" text or null, and
        "restrictions" a list of texts.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    from iadopt_lab.generation.extractor import extract_json

    extracted = extract_json(answer or "")
    data = extracted.candidate if extracted.success and isinstance(extracted.candidate, dict) else None
    return bool(data is not None and isinstance(data.get("object_of_interest"), str)
                and (data.get("matrix_entity") is None or isinstance(data.get("matrix_entity"), str))
                and "matrix_entity" in data and isinstance(data.get("restrictions"), list)
                and all(isinstance(item, str) for item in data["restrictions"]))


def outcome(ctx: Mapping[str, Any], entry: Mapping[str, Any], answer: str | None) -> dict[str, Any]:
    """Mark one answer against the experts' tags, as stage 3 marks it, plus the new keys.

    Args: ctx: from `load`; entry: the tags; answer: the model's reply, or None.
    Returns: stage 3's row (`knowledge.outcome`) and "form".
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    proper, invented_types = kn.requested(ctx, answer)
    facts = walk.facts_from_direct(ctx, answer or "") if proper else None
    mark = walk.mark(ctx, facts, entry)
    row: dict[str, Any] = {decision: bool(mark[name]) for decision, name in scoreboard.DECISIONS.items()}
    if facts is not None and invented_types:
        row["constraint"] = False
    row["complete"] = all(row.values())
    row.update(matrix_pattern=bool(mark["matrix_pattern"]), both=row["matrix"] and row["constraint"],
               parsed=facts is not None, form=form_ok(answer), tp=mark["constraint_tp"],
               fp=mark["constraint_fp"] + (invented_types if facts is not None else 0), fn=mark["constraint_fn"])
    return row


def marks1(ctx: Mapping[str, Any], calls: Mapping[engine.Job, Mapping[str, Any]]) -> dict[engine.Job, dict]:
    """Mark every stored Part 1 call; a failed call counts as no answer.

    Args: ctx: from `load`; calls: from `engine.latest`.
    Returns: {job: outcome}.
    Raises: KeyError for a call of an unknown target.
    Side Effects: none. Deterministic.
    """
    return {key: outcome(ctx, entry_of(ctx, key[2]), record.get("answer") if record.get("ok") else None)
            for key, record in calls.items()}


def per_variable(marked: Mapping[engine.Job, Mapping[str, Any]], arm: str, column: str,
                 ids: Sequence[str]) -> dict[str, float]:
    """Each variable's share of right answers over its runs.

    Args: marked: from `marks1`; arm: an arm; column: a marked column; ids: the variables.
    Returns: {variable: share}.
    Raises: KeyError when a variable has no marked answer in the arm.
    Side Effects: none. Deterministic.
    """
    rows: dict[str, list[bool]] = defaultdict(list)
    for (name, _, v), row in marked.items():
        if name == arm and v in ids:
            rows[v].append(bool(row[column]))
    return {v: sum(rows[v]) / len(rows[v]) for v in ids}


def difference(first: Mapping[str, float], second: Mapping[str, float], ids: Sequence[str],
               draws: int = DRAWS, seed: int = SEED) -> dict[str, float]:
    """The mean of first minus second over variables, with intervals from resampling the variables.

    Args: first, second: {variable: share}; ids: the variables; draws, seed: of the resampling.
    Returns: {"mean", "low", "high" (95%), "strict_low", "strict_high" (the level divided by
        `FAMILY`)}.
    Raises: KeyError when a variable is missing.
    Side Effects: none. Deterministic for a given seed.
    """
    ids = sorted(ids)
    gaps = [first[v] - second[v] for v in ids]
    rng = random.Random(seed)
    drawn = sorted(statistics.fmean(rng.choices(gaps, k=len(gaps))) for _ in range(draws))
    return {"mean": statistics.fmean(gaps), "low": drawn[int(0.025 * draws)], "high": drawn[int(0.975 * draws) - 1],
            "strict_low": drawn[int(STRICT / 2 * draws)], "strict_high": drawn[int((1 - STRICT / 2) * draws) - 1]}


def gate(summary: Mapping[str, Any]) -> list[str]:
    """The decisions of arm `rules` that may be stated in Part 2.

    Args: summary: from `summarise1`.
    Returns: the decisions whose 95% interval of `rules` minus `plain` on the 102 lies above zero and
        whose share right in `rules` is above stage 1's break-even accuracy.
    Raises: KeyError when the summary lacks a comparison.
    Side Effects: none. Deterministic.
    """
    return [d for d in DECIDED if summary["comparisons"][d]["low"] > 0
            and summary["arms"]["corpus"]["rules"][d] > BREAK_EVEN[d]]


def groups_of(ctx: Mapping[str, Any]) -> dict[str, list[str]]:
    """The corpus variables by the matrix rule that covers them, and those the deionized-water rule names.

    Args: ctx: from `load`.
    Returns: {group: sorted variable ids}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out: dict[str, list[str]] = defaultdict(list)
    for v in ctx["ids"]:
        gold, entry = ctx["records"][v]["gold"], ctx["key"][v]
        pattern = entry["matrix"][0] if entry["matrix"] else 0
        if isinstance(gold["hasObjectOfInterest"], Mapping) and "hasDenominator" in gold["hasObjectOfInterest"]:
            out["ratio object"].append(v)
        if gold["hasMatrix"] == "person":
            out["matrix person"].append(v)
        elif pattern:
            out[f"matrix {pattern} {ctx['patterns'][str(pattern)]['name']}"].append(v)
        else:
            out["no matrix"].append(v)
        if DEIONIZED in ctx["records"][v]["definition"].lower():
            out["deionized water"].append(v)
    return {name: sorted(ids) for name, ids in sorted(out.items())}


def summarise1(ctx: Mapping[str, Any], calls: Mapping[engine.Job, Mapping[str, Any]]) -> dict[str, Any]:
    """Part 1: shares by arm and population, the two comparisons, the gate and the rule groups.

    Args: ctx: from `load`; calls: from `engine.latest`.
    Returns: the summary.
    Raises: KeyError when an answer is missing.
    Side Effects: none. Deterministic.
    """
    marked = marks1(ctx, calls)
    fresh = sorted(ctx["fresh"])
    populations = {"corpus": ctx["ids"],
                   "corpus without deionized water": [v for v in ctx["ids"] if v not in groups_of(ctx).get("deionized water", [])],
                   "fresh, all": fresh,
                   "fresh, unflagged": [v for v in fresh if not ctx["fresh"][v]["flags"]],
                   "fresh, no near relative": [v for v in fresh if not ctx["fresh"][v]["relative"]]}
    wanted = set(schedule1(ctx))
    out: dict[str, Any] = {"generated_at": ft.now(), "complete": wanted <= set(calls) and all(
        calls[job].get("ok") for job in wanted), "answers": len(calls),
        "failed": sum(1 for job in wanted if not calls.get(job, {}).get("ok")), "arms": {}, "comparisons": {}}
    for name, ids in populations.items():
        if not ids:
            continue
        out["arms"][name] = {"variables": len(ids)}
        for arm in ARMS:
            rows = [row for (a, _, v), row in marked.items() if a == arm and v in ids]
            cells = {column: sum(bool(r[column]) for r in rows) / len(rows) for column in COLUMNS}
            tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
            cells["constraint_f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            out["arms"][name][arm] = cells
    for column in DECIDED:
        out["comparisons"][column] = difference(per_variable(marked, "rules", column, ctx["ids"]),
                                                per_variable(marked, "plain", column, ctx["ids"]), ctx["ids"])
    out["gate"] = gate(out)
    out["groups"] = {}
    for name, ids in groups_of(ctx).items():
        out["groups"][name] = {"variables": len(ids), **{arm: {column: statistics.fmean(
            per_variable(marked, arm, column, ids).values()) for column in DECIDED} for arm in ARMS}}
    by_type: dict[str, Counter] = defaultdict(Counter)
    for (arm, run, v), row in marked.items():
        if v not in ctx["key"]:
            continue
        record = calls[(arm, run, v)]
        facts = walk.facts_from_direct(ctx, record.get("answer") or "") if record.get("ok") and row["parsed"] else None
        want = set(hints.expected_form(hints.facts_from_key(ctx["key"][v]), ctx["patterns"], ctx["types"])["constraint_types"])
        got = set(hints.expected_form(facts, ctx["patterns"], ctx["types"])["constraint_types"]) if facts else set()
        for name in want:
            by_type[name][f"{arm} have"] += 1
            by_type[name][f"{arm} found"] += name in got
        for name in got - want:
            by_type[name][f"{arm} extra"] += 1
    out["types"] = {name: dict(sorted(tally.items())) for name, tally in sorted(by_type.items())}
    return out


# --- Part 2 ----------------------------------------------------------------------------------------

def rules_facts(ctx: Mapping[str, Any], calls: Mapping[engine.Job, Mapping[str, Any]]) -> dict[str, Any]:
    """The patterns arm `rules` found in run 1 for each corpus variable.

    Args: ctx: from `load`; calls: Part 1's stored calls.
    Returns: {variable: facts, or None when the answer is missing or not the requested JSON}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out = {}
    for v in ctx["ids"]:
        record = calls.get(("rules", 1, v), {})
        answer = record.get("answer") if record.get("ok") else None
        proper, _ = kn.requested(ctx, answer)
        out[v] = walk.facts_from_direct(ctx, answer or "") if proper else None
    return out


def prompt2(ctx: Mapping[str, Any], world: Mapping[str, Any], job: engine.Job, decided: Mapping[str, Any],
            stated: Sequence[str]) -> str:
    """The prompt of one Part 2 call.

    Args: ctx: from `load` (the completed type table renders the statement); world: from
        `rag_tuning.corpus_world`; job: (arm, run, variable); decided: from `rules_facts`; stated:
        the decisions that passed the gate.
    Returns: today's prompt with the 25 most similar variables of the other folds as examples, and
        for the guided arms the stated matrix and constraint types (the gate's decisions for
        `rag-25+rules`; none stated when its run-1 answer is unusable).
    Raises: KeyError for an unknown arm.
    Side Effects: none. Deterministic.
    """
    arm, _, v = job
    ids = rt.examples(world, "rag-25", v)
    if arm == "rag-25":
        return rt.prompt_of(world, v, ids)
    if arm == "rag-25+experts":
        facts, decisions = hints.facts_from_key(ctx["key"][v]), set(DECIDED)
    elif arm == "rag-25+rules":
        facts, decisions = decided.get(v), set(stated)
    else:
        raise KeyError(arm)
    if facts is None or not decisions:
        return rt.prompt_of(world, v, ids)
    inputs = dict(world["base_inputs"](v), examples=[world["record"](u) for u in ids])
    return dr.fill_prompt(inputs, world["definition"](v), dr.render(ctx, dr.statement(facts, decisions)))


def summarise2(ctx2: Mapping[str, Any], scores: Mapping[engine.Job, Mapping[str, Any]],
               arms: Sequence[str]) -> dict[str, Any]:
    """Part 2: each arm's Close F1 per run and by part, and each guided arm against `rag-25`.

    Args: ctx2: from `rag_tuning.load`; scores: from `engine.score`; arms: the arms run.
    Returns: the summary.
    Raises: KeyError when a score is missing.
    Side Effects: none. Deterministic.
    """
    ids, runs = ctx2["ids"], range(1, PART2_REPETITIONS + 1)
    out: dict[str, Any] = {"generated_at": ft.now(), "arms": {}, "against_rag25": {}}
    complete = all((arm, r, v) in scores for arm in arms for r in runs for v in ids)
    out["complete"] = complete
    if not complete:
        return out
    for arm in arms:
        per_run = []
        for r in runs:
            total = [sum((analysis._counts(scores[(arm, r, v)]["evaluation"]["close"]["totals"])[k] for v in ids), Fraction(0))
                     for k in range(3)]
            per_run.append(float(2 * total[0] / (2 * total[0] + total[1] + total[2])) if total[0] else 0.0)
        out["arms"][arm] = {"f1_mean": statistics.fmean(per_run), "f1_sd": statistics.stdev(per_run),
                            "f1_per_run": per_run,
                            "invalid": sum(1 for r in runs for v in ids if not scores[(arm, r, v)]["valid_prediction"])}
    out["by_part"] = {arm: {part: list(values) for part, values in parts.items()}
                      for arm, parts in analysis.part_scores({k: s for k, s in scores.items() if k[0] in arms}).items()}
    counts = analysis.variable_counts({k: s for k, s in scores.items() if k[0] in arms})
    for arm in arms:
        if arm != "rag-25":
            out["against_rag25"][arm] = rt.gain(counts, arm, "rag-25", ids)
    return out


# --- the dev check -------------------------------------------------------------------------------

def invented(ctx: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The invented variables of stage 3, shown in this test's answer form.

    Args: ctx: from `load`.
    Returns: {id: {"entry", "definition", "demo"}}.
    Raises: KeyError when an invented variable has no decomposition.
    Side Effects: none. Deterministic.
    """
    out = {}
    for item in (*ceiling.INVENTED, *walk.INVENTED_MORE):
        entry = dict(item, statistical_modifier=item.get("statistical_modifier", False), names={})
        out[item["id"]] = {"entry": entry, "definition": item["definition"],
                           "demo": demo(ctx, item["id"], item["definition"], entry, kn.INVENTED_GOLD[item["id"]])}
    return out


async def dev(ctx: Mapping[str, Any], transport: orclient.Transport, guard: orclient.Guard) -> list[dict]:
    """Both arms on the invented variables, two halves teaching each other.

    Args: ctx: from `load`; transport, guard: from `orclient`.
    Returns: one row per call: parsed, form, matrix and constraint types right, the answer's start.
    Raises: nothing from the provider.
    Side Effects: provider calls for invented definitions only; appends to `DEV_CALLS`.
    """
    items = invented(ctx)
    order = sorted(items, key=lambda name: int(name.split("-")[1]))
    half = {name: position % 2 for position, name in enumerate(order)}
    taught = {side: [items[name]["demo"] for name in order if half[name] != side] for side in (0, 1)}
    jobs = engine.schedule(ARMS, order, 1, SCHEDULE_SEED)
    await orclient.run_jobs(jobs, lambda job: build(ctx, job[0], items[job[2]]["definition"], taught[half[job[2]]]),
                            transport, DEV_CALLS, guard, concurrency=8, request=REQUEST)
    rows = []
    for key, record in sorted(engine.latest(DEV_CALLS).items()):
        row = outcome(ctx, items[key[2]]["entry"], record.get("answer") if record.get("ok") else None)
        rows.append({"key": key, "ok": record.get("ok"), "parsed": row["parsed"], "form": row["form"],
                     "matrix": row["matrix"], "constraint": row["constraint"],
                     "answer": (record.get("answer") or "")[:200]})
    return rows


# --- fingerprints and the command line --------------------------------------------------------------

def fingerprints(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Hash what the test sends: code, data, the fresh entries and every fixed prompt.

    Args: ctx: from `load`.
    Returns: hex digests (16 characters) and counts. `rag-25+rules` depends on Part 1 and is
        hashed when it is run.
    Raises: OSError when a file cannot be read.
    Side Effects: reads files and, for Part 2, the stage 5 similarity table.
    """
    out: dict[str, Any] = {name: ceiling.file_sha256(HERE / name)[:16] for name in (
        "conventions_test.py", "knowledge.py", "walk.py", "hints.py", "engine.py", "orclient.py", "groups.py",
        "scoreboard.py", "rag_tuning.py", "draft_rag.py", "decision_value.py")}
    out.update({f"data/{name}": ceiling.file_sha256(rulebook.DATA / name)[:16] for name in (
        "types_overlay.json", "conventions_v1.txt", "patterns.json", "constraint_types.json", "route_key.json",
        "trees.json")})
    out["fresh"] = ceiling.file_sha256(FRESH)[:16] if FRESH.exists() else None
    out["prompts_part1"] = {}
    for arm in ARMS:
        for population, ids in (("corpus", ctx["ids"]), ("fresh", sorted(ctx["fresh"]))):
            prompts = [prompt1(ctx, (arm, 1, v)) for v in ids]
            if prompts:
                out["prompts_part1"][f"{arm} {population}"] = {
                    "sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                    "chars_mean": round(sum(len(p) for p in prompts) / len(prompts), 1)}
    ctx2 = rt.load()
    world = rt.corpus_world(ctx2, None)
    out["prompts_part2"] = {}
    for arm in PART2_ARMS[:2]:
        prompts = [prompt2(ctx, world, (arm, 1, v), {}, []) for v in ctx["ids"]]
        out["prompts_part2"][arm] = {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                                     "chars_mean": round(sum(len(p) for p in prompts) / len(prompts), 1)}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode.

    Args: argv: arguments; `sys.argv` when None.
    Returns: 0 on success; 1 when a stop rule ended a run or a summary is not complete.
    Raises: SystemExit on bad arguments or a step run out of order.
    Side Effects: --dev and --run-* call the provider; all modes but --plan-only write files.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--plan-only", "--fresh", "--dev", "--run-part1", "--score-part1", "--run-part2", "--score-part2"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args(argv)

    if args.fresh:
        ctx = load(fresh=False)
        rows = fresh_rows(ctx, rulebook.load("issues_snapshot"), FRESH_DIR)
        FRESH.write_text(json.dumps({"source": "https://github.com/mabablue/I-ADOPT-examples-playground",
                                     "variables": rows}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{len(rows)} fresh entries with a definition: {sum(not r['flags'] for r in rows.values())} unflagged, "
              f"{sum(r['relative'] for r in rows.values())} with a near relative among the 102")
        return 0
    ctx = load()
    if args.plan_only:
        jobs = schedule1(ctx)
        print(json.dumps({"variables": len(ctx["ids"]), "fresh": len(ctx["fresh"]), "arms": ARMS,
                          "repetitions": REPETITIONS, "part1_calls": len(jobs),
                          "part2_calls_per_arm": len(ctx["ids"]) * PART2_REPETITIONS, "allowance_usd": ALLOWANCE_USD,
                          "host": orclient.HOST, "model": orclient.MODEL, "request": REQUEST,
                          "fill_request": engine.REQUEST, **fingerprints(ctx)}, indent=2))
        return 0
    if args.score_part1:
        summary = summarise1(ctx, engine.latest(CALLS1))
        SUMMARY1.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for population, cells in summary["arms"].items():
            for arm in ARMS:
                c = cells[arm]
                print(f"{population:32} {arm:6} matrix {c['matrix']:.3f} constraints {c['constraint']:.3f} "
                      f"types F1 {c['constraint_f1']:.3f} both {c['both']:.3f} parsed {c['parsed']:.3f} form {c['form']:.3f}")
        for column, row in summary["comparisons"].items():
            print(f"rules - plain, {column}: {row['mean']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}] "
                  f"strict [{row['strict_low']:+.3f}, {row['strict_high']:+.3f}]")
        print("gate:", summary["gate"], "| complete:", summary["complete"], "| failed:", summary["failed"])
        return 0 if summary["complete"] else 1
    if args.score_part2:
        ctx2 = rt.load()
        summary1 = json.loads(SUMMARY1.read_text(encoding="utf-8"))
        arms = PART2_ARMS if summary1["gate"] else PART2_ARMS[:2]
        scores = engine.score(CALLS2, SCORES2, ctx2["records"], ctx2["inputs"]["schema"])
        summary = summarise2(ctx2, scores, arms)
        SUMMARY2.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for arm, row in summary.get("arms", {}).items():
            print(f"{arm:16} F1 {row['f1_mean']:.4f} sd {row['f1_sd']:.4f} invalid {row['invalid']}")
        for arm, row in summary.get("against_rag25", {}).items():
            print(f"{arm} - rag-25: {row['gain']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
        return 0 if summary["complete"] else 1

    async def go() -> int:
        """Open the transport, run the asked part, always close the client.

        Args: none.
        Returns: 1 when a stop rule ended the work, else 0.
        Raises: SystemExit when the key is missing, another process holds the ledger, or Part 2
            is asked for before Part 1 is scored.
        Side Effects: provider calls, the ledger and output files.
        """
        spend = orclient.Spend(purpose=PURPOSE, allowance=ALLOWANCE_USD)
        guard = orclient.Guard(spend)
        transport, client = orclient.open_transport(spend, PURPOSE, concurrency=args.concurrency)
        try:
            if args.dev:
                for row in await dev(ctx, transport, guard):
                    print(row["key"], "ok" if row["ok"] else "FAILED", "parsed" if row["parsed"] else "NOT PARSED",
                          "form" if row["form"] else "NO FORM", f"matrix {'Y' if row['matrix'] else 'n'}",
                          f"constraints {'Y' if row['constraint'] else 'n'}", row["answer"].replace("\n", " "))
            elif args.run_part1:
                result = await orclient.run_jobs(schedule1(ctx), lambda job: prompt1(ctx, job), transport, CALLS1,
                                                 guard, concurrency=args.concurrency, request=REQUEST,
                                                 max_calls=args.max_calls)
                print(json.dumps(result, indent=2))
            else:
                if not SUMMARY1.exists():
                    raise SystemExit("score Part 1 first")
                summary1 = json.loads(SUMMARY1.read_text(encoding="utf-8"))
                if not summary1["complete"]:
                    raise SystemExit("Part 1 is not complete")
                stated = summary1["gate"]
                arms = PART2_ARMS if stated else PART2_ARMS[:2]
                ctx2 = rt.load()
                world = rt.corpus_world(ctx2, None)
                decided = rules_facts(ctx, engine.latest(CALLS1))
                jobs = engine.schedule(arms, ctx["ids"], PART2_REPETITIONS, PART2_SEED)
                result = await orclient.run_jobs(jobs, lambda job: prompt2(ctx, world, job, decided, stated), transport,
                                                 CALLS2, guard, concurrency=args.concurrency, request=dict(engine.REQUEST),
                                                 max_calls=args.max_calls)
                print(json.dumps(result, indent=2))
            if guard.reason:
                print("stopped:", guard.reason)
            return 1 if guard.reason else 0
        finally:
            await orclient.close_transport(transport, client)

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
