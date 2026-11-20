#!/usr/bin/env python
"""Stage 3 of the pattern-retrieval work: does better knowledge find the matrix and the constraints?

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/PATTERN_RETRIEVAL_20261119.md` (private repository).

Stage 1 showed that two decisions of a pattern set carry most of the experts' lift, the matrix
and the constraint types, and that today's pattern-finding has them right for 62% and 37% of the
variables. This stage gives the pattern-finding step more to go on and measures how often those
two decisions come out right. It finds patterns only; no decomposition is asked for.

The step is the one-call prompt of steps 2 to 5: the pattern lists, tagged example variables and
the target's definition, answered with a JSON object of patterns. Four kinds of knowledge are
switched on and off, in every combination (16 settings, "cells"):

- `G`, the experts' procedure: the questions of their matrix decision tree in order (with our own
  exit to "no matrix", as in step 2) and the general sentences of their guide on the context
  object, the statistical modifier and constraints.
- `D`, the experts' decompositions: every example also shows how the experts decomposed it, so
  the model sees what they treat as a matrix and as a constraint.
- `S`, the nearest examples last: the examples in order of similarity to the target.
- `I`, rules written from the examples: the model first writes down the conventions it sees in
  the tagged examples (once per fold), and the classifier is given them.

For a target, only variables of the other five folds (`scoreboard.folds`) are examples or enter
the rules, in every cell. The matrix tree and the guide sentences are quoted from the I-ADOPT
Patterns repository (commit ecca347, CC BY 4.0, by Barbara Magagna, Sirko Schindler, Stuart Chalk
and Marvin Friedl); the tree is read from `data/trees.json`.

    ./knowledge.py --plan-only    # fingerprints, counts and the cost estimate; no calls
    ./knowledge.py --dev          # the new wordings on invented variables only
    ./knowledge.py --induce       # the six rule texts of the `I` cells
    ./knowledge.py --run          # the grid, resumable, under the spending limit
    ./knowledge.py --score        # mark the stored answers and write the summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import ceiling  # noqa: E402
import engine  # noqa: E402
import groups  # noqa: E402
import orclient  # noqa: E402
import rulebook  # noqa: E402
import scoreboard  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402

FACTORS = ("G", "D", "S", "I")
BASELINE = "base"
REPETITIONS = 2
SCHEDULE_SEED = 20261125
REQUEST = {"stream": False, "temperature": 0.0, "top_p": 1.0, "max_tokens": 400}
INDUCE_REQUEST = {"stream": False, "temperature": 0.0, "top_p": 1.0, "max_tokens": 900}
ALLOWANCE_USD = 5.0                 # the most this stage may book, over every launch of it
DEV_ALLOWANCE_USD = 0.25
CONCURRENCY = 16
DRAWS, SEED = 10000, 20261126
PRIMARY = ("matrix", "constraint")
COLUMNS = (*scoreboard.DECISIONS, "complete", "matrix_pattern", "both")
FAMILY = 2 * len(FACTORS) + 2       # the comparisons a claim may rest on: four factors and the
STRICT = 0.05 / FAMILY              # chosen setting, each for the matrix and for the constraints
CALLS = orclient.OUTPUT / "knowledge-calls.jsonl"
RULES = orclient.OUTPUT / "knowledge-rules.jsonl"
SUMMARY = orclient.OUTPUT / "knowledge-summary.json"
DEV_CALLS = orclient.OUTPUT / "knowledge-dev-calls.jsonl"
DEV_RULES = orclient.OUTPUT / "knowledge-dev-rules.jsonl"
DEV_CELLS = ("base", "G", "D", "S", "I", "G+D+S+I")

GUIDE_HEADING = "HOW THE EXPERTS DECIDE (from the I-ADOPT pattern guide and its matrix decision tree)"
GUIDE_MATRIX = ("Matrix. The matrix is the medium, material, body, area or process in which the object of "
                "interest occurs. Ask these questions about the object of interest in this order and stop at "
                "the first that is true:")
GUIDE_NO_MATRIX = "If none is true, there is no matrix."
# Quoted from Guide/guideline.md of the patterns repository, sections 8, 6 and 7, without the
# sentences that point to one example variable.
GUIDE_SENTENCES = (
    "Context object. Add a context object, if necessary, to add clarity. Typical context objects are spheres "
    "like atmosphere, biosphere, or any additional information required despite the matrix.",
    "Statistical modifier. Determine if statistical modifiers (like mean, standard deviation, maximum) are "
    "involved. If a statistical modifier modifies a value's unit, it turns into the property instead.",
    "Constraints. Determine constraints and place them under the correct component. Ensure each constraint is "
    "concise yet informative, and present only one distinct constraint per argument.")
RULES_HEADING = "CONVENTIONS SEEN IN THE EXAMPLES (written from the examples below)"
INDUCE_INTRO = (
    "Below are scientific variables that experts decomposed with the I-ADOPT patterns. For each you see its "
    "definition, the experts' decomposition and its patterns.\n"
    "Write down the conventions the experts follow for two decisions:\n"
    "MATRIX: when a variable has a matrix, when it has none, and which matrix pattern applies.\n"
    "CONSTRAINTS: which wording of a definition leads to which constraint type, and what is not a constraint.\n"
    "Write at most 8 short rules under each of the two headings MATRIX and CONSTRAINTS. The rules will be used "
    "to classify other variables, so keep them general and name no variable of the list. Return the two "
    "headings and their rules only.")

# The experts' decomposition of each invented variable of `ceiling.py` and `walk.py`, written here
# before the first call so that the `D` and `I` wordings can be tried without a corpus variable.
INVENTED_GOLD = {
    "dev-1": {"hasProperty": "count", "hasObjectOfInterest": "bicycle",
              "hasConstraint": [{"label": "normalization: per hour", "on": "count"}]},
    "dev-2": {"hasProperty": "mass fraction",
              "hasObjectOfInterest": {"AsymmetricSystem": "microplastic fibres / beach sand",
                                      "hasNumerator": "microplastic fibres", "hasDenominator": "beach sand"}},
    "dev-3": {"hasStatisticalModifier": "maximum", "hasProperty": "index", "hasObjectOfInterest": "noise annoyance",
              "hasMatrix": "residential district",
              "hasConstraint": [{"label": "reporting frequency: daily", "on": "maximum"}]},
    "dev-4": {"hasProperty": "mass flux", "hasObjectOfInterest": "fine sediment",
              "hasMatrix": {"AsymmetricSystem": "river → delta", "hasSource": "river", "hasTarget": "delta"}},
    "dev-5": {"hasProperty": "thermodynamic temperature", "hasObjectOfInterest": "coolant water",
              "hasMatrix": "reactor cooling loop"},
    "dev-6": {"hasProperty": "plane angle",
              "hasObjectOfInterest": {"SymmetricSystem": "roof ridge line + true north",
                                      "hasPart": ["roof ridge line", "true north"]}},
    "dev-7": {"hasProperty": "mass concentration",
              "hasObjectOfInterest": {"AsymmetricSystem": "nitrate / groundwater", "hasNumerator": "nitrate",
                                      "hasDenominator": "groundwater"},
              "hasContextObject": "agricultural catchment",
              "hasConstraint": [{"label": "state: dissolved", "on": "nitrate"}]},
    "dev-8": {"hasProperty": "colour", "hasObjectOfInterest": "orchid flower",
              "hasConstraint": [{"label": "part: petals", "on": "orchid flower"}]},
    "dev-9": {"hasProperty": "count", "hasObjectOfInterest": "student", "hasMatrix": "primary school"},
    "dev-10": {"hasStatisticalModifier": "mean", "hasProperty": "thermodynamic temperature",
               "hasObjectOfInterest": "air",
               "hasConstraint": [{"label": "reference: 2 m height", "on": "air"}]},
    "dev-11": {"hasProperty": "probability", "hasObjectOfInterest": "frost",
               "hasConstraint": [{"label": "time: night in April", "on": "frost"}]},
    "dev-12": {"hasProperty": "mass", "hasObjectOfInterest": "soil sample",
               "hasConstraint": [{"label": "state: dried", "on": "soil sample"}]},
    "dev-13": {"hasProperty": "volume fraction",
               "hasObjectOfInterest": {"AsymmetricSystem": "sand / river sediment", "hasNumerator": "sand",
                                       "hasDenominator": "river sediment"}},
    "dev-14": {"hasProperty": "number density",
               "hasObjectOfInterest": {"AsymmetricSystem": "bacteria / drinking water", "hasNumerator": "bacteria",
                                       "hasDenominator": "drinking water"}},
    "dev-15": {"hasProperty": "type", "hasObjectOfInterest": "roofing material"},
    "dev-16": {"hasProperty": "cover",
               "hasObjectOfInterest": {"AsymmetricSystem": "moss / forest floor", "hasNumerator": "moss",
                                       "hasDenominator": "forest floor"}},
}


def cells() -> tuple[str, ...]:
    """The 16 settings: every combination of the four kinds of knowledge.

    Args: none.
    Returns: the cell names, "base" first; a name joins its factors with "+", in the order of
        `FACTORS`.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return tuple("+".join(f for bit, f in enumerate(FACTORS) if mask >> bit & 1) or BASELINE
                 for mask in range(2 ** len(FACTORS)))


def factors_of(cell: str) -> frozenset[str]:
    """The kinds of knowledge one cell switches on.

    Args: cell: a cell name.
    Returns: its factors; none for "base".
    Raises: KeyError for a name that is no cell.
    Side Effects: none. Deterministic.
    """
    if cell not in cells():
        raise KeyError(cell)
    return frozenset() if cell == BASELINE else frozenset(cell.split("+"))


def one_line(decomposition: Mapping[str, Any]) -> str:
    """A decomposition as one line of JSON, fields in the order of the schema.

    Args: decomposition: a six-field decomposition; a missing field counts as empty.
    Returns: the JSON text.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    full = {"hasStatisticalModifier": "", "hasProperty": "", "hasObjectOfInterest": "", "hasMatrix": "",
            "hasContextObject": "", "hasConstraint": [], **decomposition}
    return json.dumps(json.loads(ft.answer_text(full)), ensure_ascii=False)


def demo(ctx: Mapping[str, Any], item_id: str, definition: str, entry: Mapping[str, Any],
         gold: Mapping[str, Any]) -> dict[str, str]:
    """One example variable as the prompts show it.

    Args: ctx: from `load`; item_id: its id; definition: its definition; entry: its pattern tags;
        gold: the experts' decomposition.
    Returns: {"id", "definition", "gold": the decomposition on one line, "patterns": the tags as
        the JSON object the classifier must return}.
    Raises: IndexError when the entry has no property tag.
    Side Effects: none. Deterministic.
    """
    return {"id": item_id, "definition": definition, "gold": one_line(gold),
            "patterns": json.dumps(walk.direct_answer(ctx, entry), ensure_ascii=False)}


def load() -> dict[str, Any]:
    """Everything this stage needs. No network, no secrets, no stored output of an earlier step.

    Args: none.
    Returns: the context of `groups.load` (cleaned pattern texts) plus "ids" (the 102 variables,
        sorted), "group_of", "fold_of" and "demos" (every variable as an example).
    Raises: as `groups.load`.
    Side Effects: reads local files.
    """
    ctx = groups.load()
    by_group = scoreboard.ids_by_group(ctx)
    ctx["ids"] = sorted(v for ids in by_group.values() for v in ids)
    ctx["group_of"] = {v: group for group, ids in by_group.items() for v in ids}
    ctx["fold_of"] = scoreboard.folds(ctx)
    ctx["demos"] = {v: demo(ctx, v, ctx["records"][v]["definition"], ctx["key"][v], ctx["records"][v]["gold"])
                    for v in ctx["ids"]}
    return ctx


def head(ctx: Mapping[str, Any]) -> str:
    """The instruction and the pattern lists of the one-call prompt of steps 2 to 5.

    Args: ctx: from `load`.
    Returns: everything of `walk.direct_prompt` before its examples.
    Raises: ValueError when that prompt no longer has a single EXAMPLES section.
    Side Effects: none. Deterministic.
    """
    text = walk.direct_prompt(ctx, "")
    if text.count("\nEXAMPLES\n") != 1:
        raise ValueError("the direct prompt has no single EXAMPLES section")
    return text.split("\nEXAMPLES\n")[0]


def matrix_steps(ctx: Mapping[str, Any]) -> list[tuple[str, int]]:
    """The questions of the matrix tree in the order a walk meets them, each with its pattern.

    Args: ctx: from `load`; uses the upstream tree with our exit to "no matrix".
    Returns: (question, pattern number its "true" answer ends on) pairs.
    Raises: ValueError when a question does not have one "true" answer ending on a pattern and
        one "false" answer leading on, so an upstream change of the tree is noticed.
    Side Effects: none. Deterministic.
    """
    tree = ctx["matrix_tree"]
    nodes, edges = tree["nodes"], tree["edges"]
    node = next(e["to"] for e in edges if e["from"] == tree["start"])
    out = []
    while nodes[node]["kind"] == "question":
        leaving = {e["label"]: e["to"] for e in edges if e["from"] == node}
        if set(leaving) != {"true", "false"} or nodes[leaving["true"]]["kind"] != "pattern":
            raise ValueError("the matrix tree is not a chain of yes-or-no questions")
        out.append((nodes[node]["text"], int(nodes[leaving["true"]]["pattern"])))
        node = leaving["false"]
    if nodes[node].get("pattern") != walk.NO_MATRIX:
        raise ValueError("the matrix tree does not end on \"no matrix\"")
    return out


def guide(ctx: Mapping[str, Any]) -> list[str]:
    """The experts' procedure as the `G` cells state it.

    Args: ctx: from `load`.
    Returns: the block's lines.
    Raises: as `matrix_steps`.
    Side Effects: none. Deterministic.
    """
    lines = [GUIDE_HEADING, GUIDE_MATRIX]
    for position, (question, number) in enumerate(matrix_steps(ctx), start=1):
        lines.append(f"{position}. {question} Then the matrix is pattern {number} "
                     f"({ctx['patterns'][str(number)]['name']}).")
    return [*lines, GUIDE_NO_MATRIX, *GUIDE_SENTENCES]


def words(text: str) -> list[str]:
    """The words of a text, for comparing definitions.

    Args: text: any text.
    Returns: its lower-case runs of letters and digits.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def nearest_last(definition: str, demos: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """Put the examples in order of similarity to the target, the most similar last.

    Similarity is the cosine of word counts weighted by how rare a word is among the target and
    these examples; it uses the definitions only.

    Args: definition: the target's definition; demos: the examples.
    Returns: the examples, least similar first; equally similar ones in id order.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    documents = [Counter(words(definition)), *(Counter(words(d["definition"])) for d in demos)]
    seen = Counter(word for document in documents for word in document)
    weight = {word: math.log((1 + len(documents)) / (1 + count)) + 1.0 for word, count in seen.items()}
    vectors = [{word: count * weight[word] for word, count in document.items()} for document in documents]
    norms = [math.sqrt(sum(value * value for value in vector.values())) for vector in vectors]

    def similarity(position: int) -> float:
        """Cosine between the target and one example.

        Args: position: the example's position among the documents.
        Returns: the cosine, 0.0 for an empty definition.
        Raises: nothing.
        Side Effects: none.
        """
        if not norms[0] or not norms[position]:
            return 0.0
        shared = sum(value * vectors[position].get(word, 0.0) for word, value in vectors[0].items())
        return shared / (norms[0] * norms[position])

    scored = [(round(similarity(position + 1), 12), d["id"], d) for position, d in enumerate(demos)]
    return [d for _, _, d in sorted(scored, key=lambda item: (item[0], item[1]))]


def build(ctx: Mapping[str, Any], cell: str, definition: str, demos: Sequence[Mapping[str, str]],
          rules: str | None = None) -> str:
    """The prompt of one cell for one target.

    Args: ctx: from `load`; cell: a cell name; definition: the target's definition; demos: the
        examples the target may be shown; rules: the conventions written from those examples,
        needed by a cell with `I`.
    Returns: the prompt. For "base" it is the one-call prompt of steps 4 and 5 with these
        examples in id order, byte for byte.
    Raises: KeyError for an unknown cell; ValueError when a cell with `I` gets no rules.
    Side Effects: none. Deterministic.
    """
    on = factors_of(cell)
    lines = [head(ctx)]
    if "G" in on:
        lines += [*guide(ctx), ""]
    if "I" in on:
        if not (rules or "").strip():
            raise ValueError(f"cell {cell} needs the rules written from the examples")
        lines += [RULES_HEADING, rules.strip(), ""]
    ordered = nearest_last(definition, demos) if "S" in on else sorted(demos, key=lambda d: d["id"])
    lines.append("EXAMPLES")
    for example in ordered:
        lines.append(f"Definition: {example['definition']}")
        if "D" in on:
            lines.append(f"Decomposition: {example['gold']}")
        lines += [f"Patterns: {example['patterns']}", ""]
    lines += [f"Definition: {definition}", "Patterns:"]
    return "\n".join(lines)


def induction_prompt(ctx: Mapping[str, Any], demos: Sequence[Mapping[str, str]]) -> str:
    """The prompt that asks for the conventions seen in some tagged variables.

    Args: ctx: from `load`; demos: the variables to learn from.
    Returns: the prompt: the task, the matrix patterns and constraint types of the classifier's
        prompt, and every variable with its decomposition and patterns, in id order.
    Raises: ValueError when the classifier's prompt has no MATRIX PATTERNS section.
    Side Effects: none. Deterministic.
    """
    lists = head(ctx)
    if lists.count("\nMATRIX PATTERNS\n") != 1:
        raise ValueError("the direct prompt has no single MATRIX PATTERNS section")
    lines = [INDUCE_INTRO, "", "MATRIX PATTERNS", lists.split("\nMATRIX PATTERNS\n")[1].rstrip("\n"), "", "VARIABLES"]
    for example in sorted(demos, key=lambda d: d["id"]):
        lines += [f"Definition: {example['definition']}", f"Decomposition: {example['gold']}",
                  f"Patterns: {example['patterns']}", ""]
    lines.append("CONVENTIONS")
    return "\n".join(lines)


def usable_demos(ctx: Mapping[str, Any], fold: int) -> list[Mapping[str, str]]:
    """The examples a target of one fold may be shown: every variable of the other five folds.

    Args: ctx: from `load`; fold: the target's fold.
    Returns: 85 examples in id order.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return [ctx["demos"][v] for v in ctx["ids"] if ctx["fold_of"][v] != fold]


def stored_rules(path: Path = RULES) -> dict[int, str]:
    """The rule text written for each fold, as stored.

    Args: path: the rules file.
    Returns: {fold: text} of the first usable row of each fold: answered, ended normally (a
        text cut at the token limit is not used), without reasoning, and not empty.
    Raises: OSError when the file exists but cannot be read.
    Side Effects: reads the file.
    """
    out: dict[int, str] = {}
    for row in ft.read_jsonl(path):
        usable = (row.get("ok") and row.get("finish_reason") == "stop" and (row.get("rules") or "").strip()
                  and not row.get("reasoning_chars") and not row.get("reasoning_tokens"))
        if usable and row["fold"] not in out:
            out[row["fold"]] = row["rules"]
    return out


def prompt_of(ctx: Mapping[str, Any], job: engine.Job, rules: Mapping[int, str]) -> str:
    """The prompt of one grid call.

    Args: ctx: from `load`; job: (cell, repetition, variable_id); rules: from `stored_rules`.
    Returns: the prompt, with examples and rules of the other five folds only.
    Raises: KeyError for an unknown cell or variable; ValueError when a cell with `I` has no
        rules for the target's fold.
    Side Effects: none. Deterministic.
    """
    cell, _, variable_id = job
    fold = ctx["fold_of"][variable_id]
    return build(ctx, cell, ctx["records"][variable_id]["definition"], usable_demos(ctx, fold), rules.get(fold))


def schedule(ctx: Mapping[str, Any]) -> list[engine.Job]:
    """Every call of the grid, all cells of one variable side by side.

    Args: ctx: from `load`.
    Returns: (cell, repetition, variable_id) jobs, each exactly once.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return engine.schedule(cells(), ctx["ids"], REPETITIONS, SCHEDULE_SEED)


def fingerprints(ctx: Mapping[str, Any], rules: Mapping[int, str]) -> dict[str, Any]:
    """Hash what the stage sends: code, data, the folds, the rule prompts and every cell's prompts.

    Args: ctx: from `load`; rules: from `stored_rules`; cells with `I` are hashed only when all
        six rule texts exist.
    Returns: hex digests and counts.
    Raises: OSError when a file cannot be read.
    Side Effects: reads files. Deterministic.
    """
    out: dict[str, Any] = {name: ceiling.file_sha256(HERE / name)[:16] for name in (
        "knowledge.py", "orclient.py", "scoreboard.py", "hints.py", "engine.py", "groups.py", "walk.py",
        "route_key.py", "rulebook.py", "ceiling.py")}
    out.update({f"data/{name}": ceiling.file_sha256(rulebook.DATA / name)[:16] for name in (
        "trees.json", "patterns.json", "constraint_types.json", "route_key.json")})
    out["population_sha256"] = ft.sha256("\n".join(ctx["ids"]))[:16]
    out["folds_sha256"] = ft.sha256("\n".join(f"{v} {ctx['fold_of'][v]}" for v in sorted(ctx["fold_of"])))[:16]
    out["guide_sha256"] = ft.sha256("\n".join(guide(ctx)))[:16]
    out["induction_prompts"] = {str(fold): ft.sha256(induction_prompt(ctx, usable_demos(ctx, fold)))[:16]
                                for fold in range(scoreboard.FOLDS)}
    out["rules"] = {str(fold): ft.sha256(text)[:16] for fold, text in sorted(rules.items())}
    out["prompts"] = {}
    for cell in cells():
        if "I" in factors_of(cell) and len(rules) < scoreboard.FOLDS:
            continue
        prompts = [prompt_of(ctx, (cell, 1, v), rules) for v in ctx["ids"]]
        out["prompts"][cell] = {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                                "chars_mean": round(sum(len(p) for p in prompts) / len(prompts), 1)}
    return out


# --- marking and comparison --------------------------------------------------------------

def requested(ctx: Mapping[str, Any], answer: str | None) -> tuple[bool, int]:
    """Whether an answer is the JSON object the prompt asks for, and how many types it invents.

    `walk.facts_from_direct` reads leniently: a missing key counts as "none", a matrix given as
    text as "no matrix", an unlisted constraint name is dropped. The plan marks an answer that is
    not the requested JSON as wrong in everything, so the answer is read strictly here first.

    Args: ctx: from `load`; answer: the model's reply, or None.
    Returns: (True when the answer is one JSON object with the five keys, a matrix that is null
        or a matrix pattern number, true or false for context and modifier, and a list of names
        for the constraints; the number of those names that are no constraint type).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    from iadopt_lab.generation.extractor import extract_json

    extracted = extract_json(answer or "")
    data = extracted.candidate if extracted.success and isinstance(extracted.candidate, dict) else None
    if data is None or not {"property", "matrix", "context", "modifier", "constraints"} <= set(data):
        return False, 0
    matrix = data["matrix"]
    if not (matrix is None or (type(matrix) is int and 40 <= matrix <= 45)):
        return False, 0
    if type(data["context"]) is not bool or type(data["modifier"]) is not bool:
        return False, 0
    names = data["constraints"]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return False, 0
    listed = {question["name"] for question in ctx["constraint_questions"]}
    return True, len({name.lower() for name in names} - listed)


def outcome(ctx: Mapping[str, Any], variable_id: str, answer: str | None) -> dict[str, Any]:
    """Mark one answer against the experts' tags of its variable.

    Args: ctx: from `load`; variable_id: the variable; answer: the model's reply, or None.
    Returns: {each of the six decisions, "complete", "matrix_pattern" (the same matrix pattern),
        "both" (matrix and constraint types right): booleans; "parsed"; "matrix_stated";
        "tp", "fp", "fn": constraint types right, extra and missing}. An answer that is not the
        requested JSON (see `requested`) is wrong in everything. A constraint name that is no
        constraint type counts as an extra type, so the constraint types are then wrong.
    Raises: KeyError when the variable has no tags.
    Side Effects: none. Deterministic.
    """
    proper, invented_types = requested(ctx, answer)
    facts = walk.facts_from_direct(ctx, answer or "") if proper else None
    mark = walk.mark(ctx, facts, ctx["key"][variable_id])
    row: dict[str, Any] = {decision: bool(mark[name]) for decision, name in scoreboard.DECISIONS.items()}
    if facts is not None and invented_types:
        row["constraint"] = False
    row["complete"] = all(row.values())
    row.update(matrix_pattern=bool(mark["matrix_pattern"]), both=row["matrix"] and row["constraint"],
               parsed=facts is not None, matrix_stated=bool(facts and facts["matrix"]),
               tp=mark["constraint_tp"], fp=mark["constraint_fp"] + (invented_types if facts is not None else 0),
               fn=mark["constraint_fn"])
    return row


def results(ctx: Mapping[str, Any], calls: Mapping[engine.Job, Mapping[str, Any]]) -> dict[engine.Job, dict]:
    """Mark every stored call.

    Args: ctx: from `load`; calls: from `engine.latest`.
    Returns: {(cell, repetition, variable_id): outcome}; a failed call is marked as no answer.
    Raises: KeyError for a call of an unknown variable.
    Side Effects: none. Deterministic.
    """
    return {key: outcome(ctx, key[2], record.get("answer") if record.get("ok") else None)
            for key, record in calls.items()}


def share(marked: Mapping[engine.Job, Mapping[str, Any]], cell: str, column: str, ids: Sequence[str],
          repetitions: Sequence[int]) -> float:
    """How often one cell is right in one column, over some variables and repetitions.

    Args: marked: from `results`; cell: a cell name; column: an outcome column; ids: variables;
        repetitions: repetition numbers.
    Returns: the share of right answers; 0.0 for no variables.
    Raises: KeyError when a call is missing.
    Side Effects: none. Deterministic.
    """
    values = [bool(marked[(cell, r, v)][column]) for r in repetitions for v in ids]
    return sum(values) / len(values) if values else 0.0


def resampled(per_variable: Mapping[str, float], ids: Sequence[str], draws: int = DRAWS,
              seed: int = SEED) -> dict[str, float]:
    """The mean of a per-variable figure, with the variables resampled.

    Args: per_variable: {variable id: figure}; ids: the variables; draws, seed: of the
        resampling. Every comparison draws the same resamples.
    Returns: {"mean", "low", "high": the 95% interval, "strict_low", "strict_high": the interval
        at the level divided by the comparisons of the family}.
    Raises: KeyError when a variable has no figure.
    Side Effects: none. Deterministic for a given seed.
    """
    ids = sorted(ids)
    values = [per_variable[v] for v in ids]
    rng = random.Random(seed)
    drawn = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(draws))
    return {"mean": sum(values) / len(values),
            "low": drawn[int(0.025 * draws)], "high": drawn[int(0.975 * draws) - 1],
            "strict_low": drawn[int(STRICT / 2 * draws)], "strict_high": drawn[int((1 - STRICT / 2) * draws) - 1]}


def by_variable(marked: Mapping[engine.Job, Mapping[str, Any]], names: Sequence[str], column: str,
                ids: Sequence[str], repetitions: Sequence[int]) -> dict[str, float]:
    """Each variable's share of right answers over some cells and repetitions.

    Args: marked: from `results`; names: cell names; column: an outcome column; ids: variables;
        repetitions: repetition numbers.
    Returns: {variable id: share}.
    Raises: KeyError when a call is missing.
    Side Effects: none. Deterministic.
    """
    return {v: sum(bool(marked[(cell, r, v)][column]) for cell in names for r in repetitions)
            / (len(names) * len(repetitions)) for v in ids}


def effect(marked: Mapping[engine.Job, Mapping[str, Any]], factor: str, column: str, ids: Sequence[str],
           repetitions: Sequence[int], draws: int = DRAWS) -> dict[str, float]:
    """What one kind of knowledge changes, averaged over the settings of the other three.

    Args: marked: from `results`; factor: one of `FACTORS`; column: an outcome column; ids:
        variables; repetitions: repetition numbers; draws: resamples.
    Returns: the share right in the eight cells with the factor minus the share in the eight
        without, with its intervals over resampled variables, as `resampled` gives them.
    Raises: KeyError for an unknown factor or a missing call.
    Side Effects: none. Deterministic.
    """
    if factor not in FACTORS:
        raise KeyError(factor)
    on = [cell for cell in cells() if factor in factors_of(cell)]
    off = [cell for cell in cells() if factor not in factors_of(cell)]
    with_it = by_variable(marked, on, column, ids, repetitions)
    without = by_variable(marked, off, column, ids, repetitions)
    return resampled({v: with_it[v] - without[v] for v in ids}, ids, draws)


def chosen(ctx: Mapping[str, Any], marked: Mapping[engine.Job, Mapping[str, Any]],
           repetitions: Sequence[int]) -> dict[int, str]:
    """For each fold, the cell that does best on the other five.

    "Best" is the share of right matrix decisions plus the share of right constraint types. A
    fold's own variables have no say in its choice, so the choice can be scored on them.

    Args: ctx: from `load`; marked: from `results`; repetitions: repetition numbers.
    Returns: {fold: cell name}; among equally good cells the first in the order of `cells`. The
        two shares have the same denominator, so the cells are compared by their whole number of
        right answers; a sum of two rounded shares could tell two equal cells apart.
    Raises: KeyError when a call is missing.
    Side Effects: none. Deterministic.
    """
    out = {}
    for fold in range(scoreboard.FOLDS):
        others = [v for v in ctx["ids"] if ctx["fold_of"][v] != fold]
        scores = [(sum(bool(marked[(cell, r, v)][column]) for column in PRIMARY for r in repetitions for v in others),
                   -position, cell) for position, cell in enumerate(cells())]
        out[fold] = max(scores)[2]
    return out


def summarise(ctx: Mapping[str, Any], calls: Mapping[engine.Job, Mapping[str, Any]],
              draws: int = DRAWS) -> dict[str, Any]:
    """Every cell's scoreboard, what each kind of knowledge changes, and the chosen setting.

    Only repetitions in which every cell was answered for every variable enter a figure;
    "complete" is true only when both planned repetitions did.

    Args: ctx: from `load`; calls: from `engine.latest`; draws: resamples of the variables.
    Returns: {"generated_at", "variables", "repetitions_planned", "repetitions_used", "complete",
        "cells": {cell: {...}}, "effects": {column: {factor: {...}}}, "chosen": {...},
        "agreement", "spent_usd"}.
    Raises: nothing; without a complete repetition only the header is returned.
    Side Effects: none.
    """
    names = cells()
    used = [r for r in range(1, REPETITIONS + 1)
            if all(calls.get((cell, r, v), {}).get("ok") for cell in names for v in ctx["ids"])]
    out: dict[str, Any] = {"generated_at": ft.now(), "variables": len(ctx["ids"]),
                           "repetitions_planned": REPETITIONS, "repetitions_used": used,
                           "complete": len(used) == REPETITIONS, "cells": {}, "effects": {}, "chosen": {},
                           "agreement": None, "spent_usd": sum(r.get("cost_usd") or 0.0 for r in calls.values())}
    if not used:
        return out
    marked = results(ctx, calls)
    ids = ctx["ids"]
    by_group = {group: [v for v in ids if ctx["group_of"][v] == group] for group in scoreboard.GROUPS}
    with_matrix = [v for v in ids if ctx["key"][v]["matrix"]]
    without_matrix = [v for v in ids if not ctx["key"][v]["matrix"]]
    for cell in names:
        rows = [marked[(cell, r, v)] for r in used for v in ids]
        tp, fp, fn = (sum(row[name] for row in rows) for name in ("tp", "fp", "fn"))
        precision, recall = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        out["cells"][cell] = {
            **{column: share(marked, cell, column, ids, used) for column in COLUMNS},
            "parsed": share(marked, cell, "parsed", ids, used),
            "constraint_precision": precision, "constraint_recall": recall,
            "constraint_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "matrix_found_where_there_is_one": share(marked, cell, "matrix", with_matrix, used),
            "no_matrix_where_there_is_none": share(marked, cell, "matrix", without_matrix, used),
            "by_group": {group: {column: share(marked, cell, column, members, used) for column in PRIMARY}
                         for group, members in by_group.items()},
            "intervals": {column: resampled(by_variable(marked, [cell], column, ids, used), ids, draws)
                          for column in PRIMARY},
            "prompt_tokens_mean": ft._mean(calls[(cell, r, v)].get("prompt_tokens") for r in used for v in ids),
            "cost_usd": sum(calls[(cell, r, v)].get("cost_usd") or 0.0
                            for r in range(1, REPETITIONS + 1) for v in ids if (cell, r, v) in calls)}
    for column in (*PRIMARY, "property", "complete", "both"):
        out["effects"][column] = {factor: effect(marked, factor, column, ids, used, draws) for factor in FACTORS}
    choice = chosen(ctx, marked, used)
    picked = {v: choice[ctx["fold_of"][v]] for v in ids}
    out["chosen"] = {"by_fold": {str(fold): cell for fold, cell in choice.items()}, "columns": {}}
    for column in (*PRIMARY, "both", "complete"):
        own = {v: by_variable(marked, [picked[v]], column, [v], used)[v] for v in ids}
        base = by_variable(marked, [BASELINE], column, ids, used)
        out["chosen"]["columns"][column] = {
            "share": sum(own.values()) / len(own), "base": sum(base.values()) / len(base),
            "against_base": resampled({v: own[v] - base[v] for v in ids}, ids, draws)}
    if len(used) > 1:
        same = [marked[(cell, used[0], v)] == marked[(cell, used[1], v)] for cell in names for v in ids]
        out["agreement"] = sum(same) / len(same)
    return out


# --- provider ----------------------------------------------------------------------------

async def induce(ctx: Mapping[str, Any], folds: Mapping[int, Sequence[Mapping[str, str]]],
                 transport: orclient.Transport, guard: orclient.Guard, path: Path) -> dict[int, str]:
    """Have the model write the conventions it sees, once for each fold that has none stored.

    Args: ctx: from `load`; folds: {fold: the examples that fold's targets may see}; transport:
        sends one request; guard: the stop rules; path: the rules file.
    Returns: {fold: rule text} of everything stored afterwards.
    Raises: nothing from the provider; a failed call is stored as failed.
    Side Effects: provider calls; appends to `path`.
    """
    have = stored_rules(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fold, demos in sorted(folds.items()):
        if fold in have or guard.reason:
            continue
        prompt = induction_prompt(ctx, demos)
        started_at = ft.now()
        result = await transport(dict(INDUCE_REQUEST, messages=[{"role": "user", "content": prompt}]))
        if result.get("refused"):
            guard.stop(str(result.get("error") or "spend cap reached"))
            continue
        record = {"fold": fold, "started_at": started_at, "ok": bool(result.get("ok")), "status": result.get("status"),
                  "error": result.get("error"), "rules": result.get("answer"),
                  "reasoning_chars": len(result.get("reasoning") or ""),
                  "reasoning_tokens": result.get("reasoning_tokens"), "finish_reason": result.get("finish_reason"),
                  "prompt_tokens": result.get("prompt_tokens"), "completion_tokens": result.get("completion_tokens"),
                  "cost_usd": result.get("cost_usd"), "host": result.get("host"),
                  "served_model": result.get("served_model"), "prompt_sha256": ft.sha256(prompt),
                  "examples": len(demos)}
        guard.record(record)
        if record["ok"] and record["host"] != guard.host["name"]:
            record["ok"], record["error"] = False, f"served by another host: {record['host']}"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return stored_rules(path)


def invented(ctx: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The invented variables with their right patterns and decompositions, for the dev check.

    Args: ctx: from `load`.
    Returns: {id: {"entry": pattern tags, "demo": the variable as an example, "definition"}}.
    Raises: KeyError when an invented variable has no decomposition here.
    Side Effects: none. Deterministic.
    """
    out = {}
    for item in (*ceiling.INVENTED, *walk.INVENTED_MORE):
        entry = dict(item, statistical_modifier=item.get("statistical_modifier", False), names={})
        out[item["id"]] = {"entry": entry, "definition": item["definition"],
                           "demo": demo(ctx, item["id"], item["definition"], entry, INVENTED_GOLD[item["id"]])}
    return out


async def dev(ctx: Mapping[str, Any], transport: orclient.Transport, guard: orclient.Guard) -> list[dict]:
    """Try every new wording on the invented variables, split into two halves that teach each other.

    Args: ctx: from `load`; transport: sends one request; guard: the stop rules.
    Returns: one row per call: the answer, whether it is the requested JSON, and whether its
        matrix decision and constraint types are the invented right answers.
    Raises: nothing from the provider.
    Side Effects: provider calls for invented definitions only; appends to the dev files.
    """
    items = invented(ctx)
    order = sorted(items, key=lambda name: int(name.split("-")[1]))
    half = {name: position % 2 for position, name in enumerate(order)}
    shown = {side: [items[name]["demo"] for name in order if half[name] != side] for side in (0, 1)}
    rules = await induce(ctx, shown, transport, guard, DEV_RULES)
    jobs = engine.schedule(DEV_CELLS, order, 1, SCHEDULE_SEED)
    await orclient.run_jobs(
        jobs, lambda job: build(ctx, job[0], items[job[2]]["definition"], shown[half[job[2]]],
                                rules.get(half[job[2]])),
        transport, DEV_CALLS, guard, concurrency=8, request=REQUEST)
    rows = []
    for key, record in sorted(engine.latest(DEV_CALLS).items()):
        facts = walk.facts_from_direct(ctx, record.get("answer") or "") if record.get("ok") else None
        mark = walk.mark(ctx, facts, items[key[2]]["entry"])
        rows.append({"key": key, "ok": record.get("ok"), "parsed": facts is not None,
                     "reasoning": bool(record.get("reasoning_chars") or record.get("reasoning_tokens")),
                     "matrix": bool(mark["matrix"]), "constraint": bool(mark["constraint_set"]),
                     "property": bool(mark["property"]), "answer": (record.get("answer") or "")[:160]})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode: plan, dev, induce, run or score.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status: 0 on success, 1 when a stop rule ended a run or the stored run is
        not complete.
    Raises: SystemExit on bad arguments, a missing key, or `--run` before `--induce`.
    Side Effects: --dev, --induce and --run call the provider; all but --plan-only write files.
    """
    parser = argparse.ArgumentParser(description="Knowledge for the pattern-finding step (stage 3).")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--plan-only", "--dev", "--induce", "--run", "--score"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--max-calls", type=int, default=None, help="issue at most this many new calls")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args(argv)
    ctx = load()
    rules = stored_rules()
    jobs = schedule(ctx)

    if args.plan_only:
        print(json.dumps({"variables": len(ctx["ids"]), "cells": list(cells()), "repetitions": REPETITIONS,
                          "calls": len(jobs), "allowance_usd": ALLOWANCE_USD, "host": orclient.HOST,
                          "model": orclient.MODEL, "request": REQUEST, "family": FAMILY,
                          **fingerprints(ctx, rules)}, indent=2))
        return 0
    if args.score:
        summary = summarise(ctx, engine.latest(CALLS))
        SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not summary["complete"]:
            print(f"NOT COMPLETE: repetitions used {summary['repetitions_used']} of {REPETITIONS}")
        for cell, row in summary["cells"].items():
            print(f"{cell:10} matrix {row['matrix']:.3f}  constraints {row['constraint']:.3f}  "
                  f"types F1 {row['constraint_f1']:.3f}  property {row['property']:.3f}  "
                  f"complete {row['complete']:.3f}  parsed {row['parsed']:.3f}  ${row['cost_usd']:.2f}")
        for column, rows in summary["effects"].items():
            for factor, row in rows.items():
                print(f"{column:10} {factor}: {row['mean']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}] "
                      f"strict [{row['strict_low']:+.3f}, {row['strict_high']:+.3f}]")
        for column, row in summary["chosen"].get("columns", {}).items():
            versus = row["against_base"]
            print(f"chosen {column:10} {row['share']:.3f} against base {row['base']:.3f}: {versus['mean']:+.3f} "
                  f"[{versus['low']:+.3f}, {versus['high']:+.3f}] strict [{versus['strict_low']:+.3f}, "
                  f"{versus['strict_high']:+.3f}]")
        print("chosen by fold:", summary["chosen"].get("by_fold"), "| agreement of the two repetitions:",
              summary["agreement"], f"| spent ${summary['spent_usd']:.2f}")
        return 0 if summary["complete"] else 1
    if args.run and len(rules) < scoreboard.FOLDS:
        raise SystemExit("run --induce first: the cells with I need the six rule texts")

    async def go() -> int:
        """Open the transport, do the asked work, and always close the client.

        Args: none.
        Returns: the exit status: 1 when a stop rule ended the work.
        Raises: SystemExit when the key is missing or another process holds the ledger.
        Side Effects: provider calls, the ledger and output files.
        """
        purpose = "knowledge-dev" if args.dev else "knowledge"
        spend = orclient.Spend(purpose=purpose, allowance=DEV_ALLOWANCE_USD if args.dev else ALLOWANCE_USD)
        guard = orclient.Guard(spend)
        transport, client = orclient.open_transport(spend, purpose, concurrency=args.concurrency)
        try:
            if args.dev:
                for row in await dev(ctx, transport, guard):
                    print(row["key"], "ok" if row["ok"] else "FAILED", "parsed" if row["parsed"] else "NOT PARSED",
                          "reasoning" if row["reasoning"] else "", f"matrix {'Y' if row['matrix'] else 'n'}",
                          f"constraints {'Y' if row['constraint'] else 'n'}",
                          f"property {'Y' if row['property'] else 'n'}", row["answer"].replace("\n", " "))
            elif args.induce:
                folds = {fold: usable_demos(ctx, fold) for fold in range(scoreboard.FOLDS)}
                written = await induce(ctx, folds, transport, guard, RULES)
                print(json.dumps({"folds_with_rules": sorted(written), "stopped": guard.reason,
                                  "spent_usd": round(spend.spent, 4)}, indent=2))
            else:
                result = await orclient.run_jobs(jobs, lambda job: prompt_of(ctx, job, rules), transport, CALLS,
                                                 guard, concurrency=args.concurrency, request=REQUEST,
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
