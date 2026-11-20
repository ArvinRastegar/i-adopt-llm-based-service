#!/usr/bin/env python
"""Stage 4 of the pattern-retrieval work: a draft first, retrieval, and a rule check, end to end.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/PATTERN_RETRIEVAL_20261119.md` (private repository).

Arvin asked for two pipelines that end with a rule check, "to see that the decomposition is
following the patterns that are chosen", and then the usual scoring, without the linking step:

- a draft first: the model decomposes the variable with today's prompt; the draft's property and
  object of interest are given to the pattern-finding step, which chooses the patterns of the
  other parts; the variable is decomposed again with those patterns stated;
- retrieval: the variables most like the target are retrieved with the experts' decompositions
  and patterns, and serve as the prompt's examples and as a source of patterns.

Six setups ("arms"), on `qwen/qwen3.8-27b` at the pinned host (see `orclient.py`):

- `base`: today's prompt. Its answer is the draft of the two `draft-first` arms.
- `draft-first`: every decision of the pattern-finding step is stated.
- `draft-first/gated`: only what passes a gate is stated: the property pattern; the matrix and the
  context object where draft and pattern-finding agree; the constraint types both name, as "at
  least these types".
- `rag`: today's prompt with the 25 most similar tagged variables of the other folds as its
  examples, most similar last. Its answer is the draft of `rag/gated` and `combined`.
- `rag/gated`: `rag` with the matrix stated where at least 4 of the 5 most similar variables agree
  on its shape.
- `combined`: `rag` with the gates of `draft-first/gated`, pattern-finding being shown the 25
  retrieved variables, and the matrix of `rag/gated` where that gate fires.

Rule check of every guided arm: the answer must pass the lab's schema validation and have the form
the stated patterns demand; a failing answer goes back once with its problems; if it still fails,
the draft is kept. Where nothing is stated, the draft is the answer and no call is made.

All arms of one variable and repetition (a "unit") are answered together, so every guided arm
starts from the same draft as the arm it is compared with. Only complete units are stored.

    ./draft_rag.py --neighbours   # the similarity order of the 102 definitions (local model)
    ./draft_rag.py --plan-only    # fingerprints, counts and the cost estimate; no calls
    ./draft_rag.py --dev          # every step on invented variables only
    ./draft_rag.py --run          # the grid, resumable, under the spending limit
    ./draft_rag.py --score        # score the stored answers and write the summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import analysis  # noqa: E402
import ceiling  # noqa: E402
import decision_value as dv  # noqa: E402
import engine  # noqa: E402
import groups  # noqa: E402
import hints  # noqa: E402
import knowledge as kn  # noqa: E402
import orclient  # noqa: E402
import route_key  # noqa: E402
import rulebook  # noqa: E402
import scoreboard  # noqa: E402
import settings  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402
from harness import _aggregate  # noqa: E402

pipeline = settings.pipeline        # this folder's pipeline.py, loaded by path; see settings.py

ARMS = ("base", "draft-first", "draft-first/gated", "rag", "rag/gated", "combined")
GUIDED = ("draft-first", "draft-first/gated", "rag/gated", "combined")
BASELINE = "base"
COMPARED = ARMS[1:]
SECONDARY = (("draft-first/gated", "draft-first"), ("rag/gated", "rag"), ("combined", "rag"))
REPETITIONS = 10
SCHEDULE_SEED = 20261127
RETRIEVED = 25                      # as many examples as today's prompt shows
VOTERS, VOTES_NEEDED = 5, 4
RELATIVE_OVERLAP = 0.5
FILL_REQUEST = dict(engine.REQUEST)  # the study's settings
FINDER_REQUEST = {"stream": False, "temperature": 0.0, "top_p": 1.0, "max_tokens": 400}
ALLOWANCE_USD = 8.0
DEV_ALLOWANCE_USD = 0.3
CONCURRENCY = 16
DRAWS, SEED = 10000, 20261128
FAMILY = len(COMPARED)
STRICT = 0.05 / FAMILY
CALLS = orclient.OUTPUT / "pipelines-calls.jsonl"
FAILED = orclient.OUTPUT / "pipelines-failed.jsonl"
SCORES = orclient.OUTPUT / "pipelines-scores.jsonl"
SUMMARY = orclient.OUTPUT / "pipelines-summary.json"
NEIGHBOURS = orclient.OUTPUT / "pipelines-neighbours.json"
DEV_CALLS = orclient.OUTPUT / "pipelines-dev-calls.jsonl"
DEV_FAILED = orclient.OUTPUT / "pipelines-dev-failed.jsonl"

ANCHOR_LINE = ("Every variable below is shown with its property and object of interest. Use them, with the "
               "definition, to choose the patterns.")
ANCHOR_LABEL = "Property and object of interest: "
AT_LEAST = ("- hasConstraint: at least one constraint of each of these types: {names}. Each label starts "
            "with its type and a colon.")
AT_LEAST_COMPLAINT = "hasConstraint must include at least one constraint of each of these types: {names}."
EMPTY_FACTS = {"property": [], "object": [], "matrix": [], "context": [], "constraint": [],
               "statistical_modifier": False, "names": {}}


class Failed(Exception):
    """A call of a unit failed, so the unit is not stored."""


# --- retrieval -----------------------------------------------------------------------------

def similarity_order(ids: Sequence[str], definition: Callable[[str], str],
                     similar: Callable[[str, str], float]) -> dict[str, list[str]]:
    """Every variable's other variables, most similar first.

    Args: ids: the variables; definition: gives a variable's definition; similar: a similarity
        of two texts.
    Returns: {variable: the other variables, by falling similarity, equals by id}.
    Raises: nothing from this function.
    Side Effects: calls `similar` once per pair. Deterministic for a deterministic `similar`.
    """
    ids = sorted(ids)
    score: dict[tuple[str, str], float] = {}
    for position, first in enumerate(ids):
        for second in ids[position + 1:]:
            score[(first, second)] = score[(second, first)] = round(float(similar(definition(first),
                                                                              definition(second))), 9)
    return {v: sorted((u for u in ids if u != v), key=lambda u: (-score[(v, u)], u)) for v in ids}


def local_similarity() -> tuple[Callable[[str, str], float], dict[str, Any]]:
    """The scorer's frozen local embedding model, used here to compare definitions.

    Args: none.
    Returns: (cosine similarity of two texts, the model's identity).
    Raises: as the lab's loader when the model files are missing or altered.
    Side Effects: reads and hashes the local model files; no download.
    """
    from iadopt_lab.cli import _configuration, _similarity_backend

    backend = _similarity_backend(engine.ROOT, _configuration(engine.ROOT).data["evaluation"])
    return backend, backend.identity


def write_neighbours(ctx: Mapping[str, Any], path: Path = NEIGHBOURS) -> dict[str, Any]:
    """Order the 102 definitions by similarity with the local model and store the order.

    Args: ctx: from `knowledge.load`; path: where to store it.
    Returns: {"model", "definitions_sha256", "order": {variable: other variables}}.
    Raises: as `local_similarity`.
    Side Effects: loads the local model; writes `path`.
    """
    similar, identity = local_similarity()
    ids = ctx["ids"]
    out = {"model": identity, "definitions_sha256": definitions_digest(ctx),
           "order": similarity_order(ids, lambda v: ctx["records"][v]["definition"], similar)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def definitions_digest(ctx: Mapping[str, Any]) -> str:
    """A fingerprint of the 102 definitions, in id order.

    Args: ctx: from `knowledge.load`.
    Returns: the SHA-256 hex digest.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return ft.sha256("\n".join(f"{v}\t{ctx['records'][v]['definition']}" for v in ctx["ids"]))


def load(order: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Everything this stage needs. No network, no secrets.

    Args: order: the similarity order of the definitions; read from `NEIGHBOURS` when None.
    Returns: the context of `knowledge.load` plus "order" and "relatives" (the variables with a
        variable of another fold whose definition shares at least half of all their words).
    Raises: SystemExit when the similarity order has not been written or belongs to other
        definitions.
    Side Effects: reads local files.
    """
    ctx = kn.load()
    if order is None:
        if not NEIGHBOURS.exists():
            raise SystemExit("run --neighbours first")
        stored = json.loads(NEIGHBOURS.read_text(encoding="utf-8"))
        if stored["definitions_sha256"] != definitions_digest(ctx):
            raise SystemExit("the stored similarity order belongs to other definitions")
        order = stored["order"]
    ctx["order"] = {v: list(order[v]) for v in ctx["ids"]}
    words = {v: set(kn.words(ctx["records"][v]["definition"])) for v in ctx["ids"]}
    ctx["relatives"] = sorted(
        v for v in ctx["ids"]
        if any(len(words[v] & words[u]) / len(words[v] | words[u]) >= RELATIVE_OVERLAP
               for u in ctx["ids"] if ctx["fold_of"][u] != ctx["fold_of"][v]))
    return ctx


def nearest(ctx: Mapping[str, Any], target: str) -> list[str]:
    """The tagged variables a target may be shown, most similar first: never its own fold.

    Args: ctx: from `load`; target: the variable.
    Returns: the 85 variables of the other folds, by falling similarity.
    Raises: KeyError for an unknown target.
    Side Effects: none. Deterministic.
    """
    return [u for u in ctx["order"][target] if ctx["fold_of"][u] != ctx["fold_of"][target]]


def anchor_of(decomposition: Mapping[str, Any] | None) -> str:
    """The property and object of interest of a decomposition, as one line of JSON.

    Args: decomposition: a six-field decomposition, or None for an unusable draft.
    Returns: {"hasProperty": ..., "hasObjectOfInterest": ...}; empty strings for None.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if decomposition is None:
        return json.dumps({"hasProperty": "", "hasObjectOfInterest": ""}, ensure_ascii=False)
    return json.dumps({"hasProperty": decomposition["hasProperty"],
                       "hasObjectOfInterest": decomposition["hasObjectOfInterest"]}, ensure_ascii=False)


def corpus_world(ctx: Mapping[str, Any]) -> dict[str, Callable[[str], Any]]:
    """What a corpus target may be shown, by kind.

    Args: ctx: from `load`.
    Returns: {"definition", "base_inputs" (today's prompt inputs; an example variable gets the
        other 24 examples), "rag_records" (the 25 nearest, most similar last), "finder_draft" (the
        85 variables of the other folds, in id order), "finder_rag" (the 25 nearest, most similar
        last), "voters" (the 5 nearest, nearest first)}, each a function of the target.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    def demo(u: str) -> dict[str, str]:
        """One tagged variable as the pattern-finding step sees it.

        Args: u: the variable.
        Returns: {"id", "definition", "anchor", "patterns"}.
        Raises: KeyError for an unknown variable.
        Side Effects: none.
        """
        return {"id": u, "definition": ctx["records"][u]["definition"], "anchor": anchor_of(ctx["records"][u]["gold"]),
                "patterns": json.dumps(walk.direct_answer(ctx, ctx["key"][u]), ensure_ascii=False)}

    def voter(u: str) -> dict[str, Any]:
        """One tagged variable's matrix, for the vote.

        Args: u: the variable.
        Returns: {"id", "shape", "matrix": tagged matrix patterns}.
        Raises: KeyError for an unknown variable.
        Side Effects: none.
        """
        form = hints.expected_form(hints.facts_from_key(ctx["key"][u]), ctx["patterns"], ctx["types"])
        return {"id": u, "shape": form["matrix"], "matrix": list(ctx["key"][u]["matrix"])}

    return {"definition": lambda v: ctx["records"][v]["definition"],
            "base_inputs": lambda v: groups.context_for(ctx, v)["inputs"],
            "rag_records": lambda v: [ctx["records"][u] for u in reversed(nearest(ctx, v)[:RETRIEVED])],
            "finder_draft": lambda v: [demo(u) for u in sorted(nearest(ctx, v))],
            "finder_rag": lambda v: [demo(u) for u in reversed(nearest(ctx, v)[:RETRIEVED])],
            "voters": lambda v: [voter(u) for u in nearest(ctx, v)[:VOTERS]]}


# --- prompts, statements and the rule check ----------------------------------------------------

def fill_prompt(inputs: Mapping[str, Any], definition: str, block: str = "") -> str:
    """Today's decomposition prompt with given examples and an optional guidance block.

    Args: inputs: prompt inputs with "template", "schema_text" and "examples"; definition: the
        target's definition; block: text appended to the definition.
    Returns: the prompt; with today's inputs and no block it is today's prompt, byte for byte.
    Raises: KeyError when an example lacks a definition or gold.
    Side Effects: none. Deterministic.
    """
    return ft.prompt_for(inputs, "B", definition + block)


def finder_prompt(ctx: Mapping[str, Any], demos: Sequence[Mapping[str, str]], definition: str, anchor: str) -> str:
    """The pattern-finding prompt, every variable shown with its property and object of interest.

    Args: ctx: from `load`; demos: the tagged variables to show, in order; definition: the
        target's definition; anchor: the target's property and object of interest, from its draft.
    Returns: the prompt: the instruction and pattern lists of steps 2 to 5, one line on the
        anchors, the examples, and the target last.
    Raises: as `knowledge.head`.
    Side Effects: none. Deterministic.
    """
    lines = [kn.head(ctx), ANCHOR_LINE, "", "EXAMPLES"]
    for demo in demos:
        lines += [f"Definition: {demo['definition']}", ANCHOR_LABEL + demo["anchor"], f"Patterns: {demo['patterns']}", ""]
    lines += [f"Definition: {definition}", ANCHOR_LABEL + anchor, "Patterns:"]
    return "\n".join(lines)


def statement(facts: Mapping[str, Any], decisions: set[str] | frozenset[str], at_least: bool = False) -> dict[str, Any]:
    """What a guided arm states: a pattern set, the decisions taken from it, and how.

    Args: facts: a pattern set; decisions: which of the six decisions to state; at_least: state
        the constraint types as "at least these" rather than "exactly these".
    Returns: {"facts", "decisions" (sorted), "at_least"}.
    Raises: ValueError for an unknown decision.
    Side Effects: none. Deterministic.
    """
    if set(decisions) - dv.ALL:
        raise ValueError(f"unknown decisions: {sorted(set(decisions) - dv.ALL)}")
    return {"facts": {k: (list(v) if isinstance(v, list) else v) for k, v in dict(facts).items()},
            "decisions": sorted(decisions), "at_least": bool(at_least)}


def render(ctx: Mapping[str, Any], stmt: Mapping[str, Any]) -> str:
    """The guidance block of a statement.

    Args: ctx: from `load`; stmt: from `statement`.
    Returns: with exact constraint types, the block of `decision_value.render` (with all six
        decisions that is the block of steps 1 to 5); with "at least", the same block in which the
        constraint line says "at least" and does not exclude other types.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    decisions = set(stmt["decisions"])
    facts = stmt["facts"]
    if not stmt["at_least"] or "constraint" not in decisions:
        return dv.render(facts, decisions, ctx["patterns"], ctx["types"])
    names = hints.expected_form(facts, ctx["patterns"], ctx["types"])["constraint_types"]
    text = dv.render(facts, decisions - {"constraint"}, ctx["patterns"], ctx["types"])
    line = AT_LEAST.format(names="; ".join(names))
    by_name = {row["name"].lower(): row for row in ctx["types"]}
    definitions = ["Constraint types:"] + [
        f"- {name}: {(by_name.get(name, {}).get('definition') or '').strip()}".rstrip(": ") for name in names]
    if not text:
        return "\n".join(["", "", hints.HEADING, dv.FORM_HARD, line, *definitions])
    lines = text.split("\n")
    if dv.FORM_HARD in lines:
        position = lines.index(dv.PATTERNS_HARD) if dv.PATTERNS_HARD in lines else len(lines)
        lines.insert(position, line)
    else:
        lines.insert(lines.index(hints.HEADING) + 1, line)
        lines.insert(lines.index(hints.HEADING) + 1, dv.FORM_HARD)
    return "\n".join([*lines, *definitions])


def form_of(ctx: Mapping[str, Any], stmt: Mapping[str, Any]) -> dict[str, Any]:
    """What the rule check demands of an answer: the form of the stated decisions only.

    Args: ctx: from `load`; stmt: from `statement`.
    Returns: a subset of {"object", "matrix", "context", "statistical_modifier",
        "constraint_types", "constraint_types_at_least"}. The object's shape is demanded when the
        decision that owns its line in the block is stated (see `decision_value.render`).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    facts, decisions = stmt["facts"], set(stmt["decisions"])
    full = hints.expected_form(facts, ctx["patterns"], ctx["types"])
    owner = "object" if facts["object"] else "property"
    out: dict[str, Any] = {}
    if full["object"] and owner in decisions:
        out["object"] = full["object"]
    if "matrix" in decisions:
        out["matrix"] = full["matrix"]
    if "context" in decisions:
        out["context"] = full["context"]
    if "modifier" in decisions:
        out["statistical_modifier"] = full["statistical_modifier"]
    if "constraint" in decisions:
        out["constraint_types_at_least" if stmt["at_least"] else "constraint_types"] = full["constraint_types"]
    return out


def check(ctx: Mapping[str, Any], answer: str | None, form: Mapping[str, Any]) -> tuple[dict | None, list[str]]:
    """The rule check: schema validation, then the stated form.

    Args: ctx: from `load`; answer: the model's reply, or None; form: from `form_of`, or {} for
        the schema only.
    Returns: (valid decomposition or None, complaints); no complaints means the answer passes.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    plain = {key: value for key, value in form.items() if key != "constraint_types_at_least"}
    prediction, complaints = pipeline.check(ctx, answer, plain)
    wanted = form.get("constraint_types_at_least")
    if prediction is not None and wanted:
        got = {route_key.constraint_type(c["label"]) for c in prediction["hasConstraint"]}
        if set(wanted) - got:
            complaints = [*complaints, AT_LEAST_COMPLAINT.format(names="; ".join(wanted))]
    return prediction, complaints


# --- the gates ------------------------------------------------------------------------------

def finder_facts(ctx: Mapping[str, Any], answer: str | None) -> dict[str, Any] | None:
    """The pattern set a pattern-finding answer states, read strictly.

    Args: ctx: from `load`; answer: the reply.
    Returns: the facts; None when the reply is not the requested JSON object (see
        `knowledge.requested`). Constraint names that are no type are dropped here; the strict
        reading has already refused answers whose other fields are malformed.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    proper, _ = kn.requested(ctx, answer)
    return walk.facts_from_direct(ctx, answer or "") if proper else None


def view(decomposition: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The decisions a draft decomposition makes about the matrix, context and constraints.

    Args: decomposition: a valid six-field decomposition, or None.
    Returns: {"matrix": shape, "context": bool, "types": sorted constraint types named}, or None.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if decomposition is None:
        return None
    return {"matrix": route_key.shape(decomposition["hasMatrix"]), "context": bool(decomposition["hasContextObject"]),
            "types": sorted({route_key.constraint_type(c["label"]) for c in decomposition["hasConstraint"]} - {""})}


def vote(voters: Sequence[Mapping[str, Any]]) -> list[int] | None:
    """The matrix the nearest tagged variables agree on.

    Args: voters: from the world's "voters", nearest first.
    Returns: None when fewer than `VOTES_NEEDED` of the first `VOTERS` share a matrix shape; else
        [] for "no matrix", or [pattern]: the matrix pattern most common among the agreeing
        voters, the nearer one among equals.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    voters = list(voters)[:VOTERS]
    shapes = Counter(voter["shape"] for voter in voters)
    if not shapes:
        return None
    shape, count = shapes.most_common(1)[0]
    if count < VOTES_NEEDED:
        return None
    if shape == "empty":
        return []
    agreeing = [voter["matrix"][0] for voter in voters if voter["shape"] == shape and voter["matrix"]]
    counts = Counter(agreeing)
    best = max(counts.values())
    return [next(number for number in agreeing if counts[number] == best)]


def gated(ctx: Mapping[str, Any], draft: Mapping[str, Any] | None, facts: Mapping[str, Any] | None,
          voted: list[int] | None) -> dict[str, Any] | None:
    """The statement of `draft-first/gated` (no vote) and of `combined` (with the vote).

    Args: ctx: from `load`; draft: `view` of the draft; facts: the pattern-finding answer's
        facts; voted: from `vote`, or None.
    Returns: the statement, or None when nothing passes:
        - the property pattern of pattern-finding, whenever there is one;
        - the matrix: the vote where it fires, else pattern-finding's where its shape is the
          draft's;
        - the context object where draft and pattern-finding agree;
        - the constraint types both name, as "at least these", where there are any.
        The object's shape and the statistical modifier are never stated on their own.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out = dict(facts) if facts is not None else dict(EMPTY_FACTS)
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in out.items()}
    out["names"] = dict(out.get("names") or {})
    decisions: set[str] = set()
    found = hints.expected_form(facts, ctx["patterns"], ctx["types"]) if facts is not None else None
    if facts is not None:
        decisions.add("property")
    if voted is not None:
        out["matrix"] = list(voted)
        decisions.add("matrix")
    elif found is not None and draft is not None and draft["matrix"] == found["matrix"]:
        decisions.add("matrix")
    if found is not None and draft is not None and draft["context"] == found["context"]:
        decisions.add("context")
    at_least = False
    if found is not None and draft is not None:
        common = set(draft["types"]) & set(found["constraint_types"])
        if common:
            by_name = {question["name"]: question["number"] for question in ctx["constraint_questions"]}
            out["constraint"] = sorted(by_name[name] for name in common)
            decisions.add("constraint")
            at_least = True
    if not decisions:
        return None
    if "constraint" not in decisions:
        out["constraint"] = []
    return statement(out, decisions, at_least)


def summary_of(ctx: Mapping[str, Any], stmt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What a statement says, in the words the summary marks against the experts' tags.

    Args: ctx: from `load`; stmt: a statement or None.
    Returns: {"decisions", "at_least", "property", "matrix_shape", "matrix", "context",
        "constraint_types"} for the stated decisions only; None for no statement.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if stmt is None:
        return None
    facts, decisions = stmt["facts"], set(stmt["decisions"])
    form = hints.expected_form(facts, ctx["patterns"], ctx["types"])
    out: dict[str, Any] = {"decisions": sorted(decisions), "at_least": stmt["at_least"]}
    if "property" in decisions:
        out["property"] = list(facts["property"])
    if "matrix" in decisions:
        out["matrix_shape"], out["matrix"] = form["matrix"], list(facts["matrix"])
    if "context" in decisions:
        out["context"] = form["context"]
    if "constraint" in decisions:
        out["constraint_types"] = form["constraint_types"]
    return out


# --- one unit: every arm of one variable and repetition ---------------------------------------

async def unit(ctx: Mapping[str, Any], world: Mapping[str, Callable[[str], Any]], repetition: int, target: str,
               transport: orclient.Transport, guard: orclient.Guard) -> list[dict[str, Any]]:
    """Answer every arm for one variable in one repetition.

    Args: ctx: from `load`; world: from `corpus_world` (or the dev world); repetition: its number;
        target: the variable; transport: from `orclient.open_transport`; guard: the stop rules.
    Returns: one record per arm, in the order of `ARMS`, with the final answer and everything
        that led to it.
    Raises: Failed when any call fails, is refused by the cap, or is answered by another host.
    Side Effects: provider calls (2 to 12).
    """
    definition = world["definition"](target)
    started_at = ft.now()
    cost: dict[str, Counter] = {arm: Counter() for arm in (*ARMS, "finder-draft", "finder-rag")}

    async def call(owner: str, messages: list[dict[str, str]], request: Mapping[str, Any]) -> str:
        """Send one request and keep its cost under its owner.

        Args: owner: the arm or step the call belongs to; messages: the chat; request: settings.
        Returns: the answer text.
        Raises: Failed on any failure, and before sending when a stop rule has fired, so a unit
            already under way stops at its next call.
        Side Effects: one provider call; updates the guard and the costs.
        """
        if guard.reason:
            raise Failed(f"stopped: {guard.reason}")
        result = await transport(dict(request, messages=messages))
        if result.get("refused"):
            guard.stop(str(result.get("error") or "spend cap reached"))
            raise Failed(str(result.get("error")))
        reasoning = result.get("reasoning") or ""
        guard.record({"ok": bool(result.get("ok")), "status": result.get("status"), "error": result.get("error"),
                      "host": result.get("host"), "reasoning_chars": len(reasoning),
                      "reasoning_tokens": result.get("reasoning_tokens")})
        if not result.get("ok"):
            raise Failed(str(result.get("error")))
        if result.get("host") != guard.host["name"]:
            raise Failed(f"answered by {result.get('host')}")
        spent = cost[owner]
        spent["calls"] += 1
        spent["prompt_tokens"] += result.get("prompt_tokens") or 0
        spent["completion_tokens"] += result.get("completion_tokens") or 0
        spent["reasoning_chars"] += len(reasoning)
        spent["reasoning_tokens"] += result.get("reasoning_tokens") or 0
        spent["cost_micro_usd"] += round((result.get("cost_usd") or 0.0) * 1e6)
        return result.get("answer") or ""

    base_inputs = world["base_inputs"](target)
    base_prompt = fill_prompt(base_inputs, definition)
    draft = await call("base", [{"role": "user", "content": base_prompt}], FILL_REQUEST)
    draft_prediction, _ = check(ctx, draft, {})
    finder_draft_prompt = finder_prompt(ctx, world["finder_draft"](target), definition, anchor_of(draft_prediction))
    finder_draft = await call("finder-draft", [{"role": "user", "content": finder_draft_prompt}], FINDER_REQUEST)
    facts_draft = finder_facts(ctx, finder_draft)
    rag_inputs = dict(ctx["inputs"], examples=list(world["rag_records"](target)))
    rag_prompt = fill_prompt(rag_inputs, definition)
    rag = await call("rag", [{"role": "user", "content": rag_prompt}], FILL_REQUEST)
    rag_prediction, _ = check(ctx, rag, {})
    voted = vote(world["voters"](target))
    finder_rag_prompt = finder_prompt(ctx, world["finder_rag"](target), definition, anchor_of(rag_prediction))
    finder_rag = await call("finder-rag", [{"role": "user", "content": finder_rag_prompt}], FINDER_REQUEST)
    facts_rag = finder_facts(ctx, finder_rag)
    statements = {
        "draft-first": statement(facts_draft, dv.ALL) if facts_draft is not None else None,
        "draft-first/gated": gated(ctx, view(draft_prediction), facts_draft, None),
        "rag/gated": statement(dict(EMPTY_FACTS, matrix=list(voted)), {"matrix"}) if voted is not None else None,
        "combined": gated(ctx, view(rag_prediction), facts_rag, voted)}
    finder_of = {"draft-first": (finder_draft, finder_draft_prompt), "draft-first/gated": (finder_draft, finder_draft_prompt),
                 "rag/gated": (None, None), "combined": (finder_rag, finder_rag_prompt)}

    def record(arm: str, **fields: Any) -> dict[str, Any]:
        """One arm's record.

        Args: arm: the arm; fields: what to add.
        Returns: the record.
        Raises: nothing.
        Side Effects: none.
        """
        spent = cost[arm]
        return {"arm": arm, "repetition": repetition, "variable_id": target, "started_at": started_at, "ok": True,
                "status": 200, "error": None, "host": guard.host["name"], "calls": spent["calls"],
                "prompt_tokens": spent["prompt_tokens"], "completion_tokens": spent["completion_tokens"],
                "reasoning_chars": spent["reasoning_chars"], "reasoning_tokens": spent["reasoning_tokens"],
                "cost_usd": spent["cost_micro_usd"] / 1e6, **fields}

    out = [record("base", answer=draft, prompt_sha256=ft.sha256(base_prompt), valid=draft_prediction is not None)]
    guided: dict[str, dict[str, Any]] = {}
    for arm in GUIDED:
        first_pass, inputs = (draft, base_inputs) if arm.startswith("draft") else (rag, rag_inputs)
        stmt = statements[arm]
        finder_answer, finder_text = finder_of[arm]
        details: dict[str, Any] = {"statement": summary_of(ctx, stmt), "draft_sha256": ft.sha256(first_pass),
                                   "finder_answer": finder_answer,
                                   "finder_prompt_sha256": ft.sha256(finder_text) if finder_text else None,
                                   "finder_read": (facts_draft if arm.startswith("draft") else facts_rag) is not None
                                   if finder_text else None,
                                   "voted": voted if arm in ("rag/gated", "combined") else None,
                                   "no_guidance": stmt is None, "first_answer": None, "first_complaints": [],
                                   "repaired": False, "repair_complaints": [], "fell_back": False,
                                   "draft_complaints": None, "form": None, "prompt_sha256": None}
        if stmt is None:
            guided[arm] = dict(details, answer=first_pass)
            continue
        form = form_of(ctx, stmt)
        prompt = fill_prompt(inputs, definition, render(ctx, stmt))
        answer = await call(arm, [{"role": "user", "content": prompt}], FILL_REQUEST)
        _, complaints = check(ctx, answer, form)
        details.update(form=form, prompt_sha256=ft.sha256(prompt), first_answer=answer, first_complaints=complaints,
                       draft_complaints=check(ctx, first_pass, form)[1])
        final = answer
        if complaints:
            second = await call(arm, pipeline.repair_messages(prompt, answer, complaints), FILL_REQUEST)
            _, still = check(ctx, second, form)
            details.update(repaired=True, repair_complaints=still)
            final = second
            if still:
                final = first_pass
                details["fell_back"] = True
        guided[arm] = dict(details, answer=final)
    out.append(record("draft-first", **guided["draft-first"]))
    out.append(record("draft-first/gated", **guided["draft-first/gated"]))
    out.append(record("rag", answer=rag, prompt_sha256=ft.sha256(rag_prompt), valid=rag_prediction is not None,
                      examples=[record_id(item) for item in world["rag_records"](target)]))
    out.append(record("rag/gated", **guided["rag/gated"]))
    out.append(record("combined", **guided["combined"]))
    shared = {"finder_draft_cost_usd": cost["finder-draft"]["cost_micro_usd"] / 1e6,
              "finder_rag_cost_usd": cost["finder-rag"]["cost_micro_usd"] / 1e6,
              "unit_cost_usd": sum(spent["cost_micro_usd"] for spent in cost.values()) / 1e6,
              "unit_calls": sum(spent["calls"] for spent in cost.values())}
    out[0].update(shared)
    return out


def record_id(record: Mapping[str, Any]) -> str:
    """The id of a corpus record, or of an invented variable.

    Args: record: a corpus record or a dev record.
    Returns: its "variable_id" or "id".
    Raises: KeyError when it has neither.
    Side Effects: none.
    """
    return record["variable_id"] if "variable_id" in record else record["id"]


def schedule(ids: Sequence[str], repetitions: int = REPETITIONS, seed: int = SCHEDULE_SEED) -> list[tuple[int, str]]:
    """Every unit, repetition by repetition, the variables shuffled within each.

    Args: ids: the variables; repetitions: how many; seed: of the shuffles.
    Returns: (repetition, variable) pairs, each exactly once.
    Raises: nothing.
    Side Effects: none. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    out = []
    for repetition in range(1, repetitions + 1):
        order = sorted(ids)
        rng.shuffle(order)
        out += [(repetition, v) for v in order]
    return out


def complete_units(path: Path) -> set[tuple[int, str]]:
    """The units whose six arms are all stored.

    Args: path: the calls file.
    Returns: {(repetition, variable)}.
    Raises: OSError when the file exists but cannot be read.
    Side Effects: reads the file.
    """
    arms: dict[tuple[int, str], set[str]] = {}
    for key, record in engine.latest(path).items():
        if record.get("ok"):
            arms.setdefault((key[1], key[2]), set()).add(key[0])
    return {unit_key for unit_key, have in arms.items() if have == set(ARMS)}


async def run(ctx: Mapping[str, Any], world: Mapping[str, Callable[[str], Any]], units: Sequence[tuple[int, str]],
              transport: orclient.Transport, guard: orclient.Guard, calls_path: Path, failed_path: Path,
              concurrency: int = CONCURRENCY, max_units: int | None = None) -> dict[str, Any]:
    """Answer every unit not yet stored; store a unit's six records only when all succeeded.

    Args: ctx: from `load`; world: what targets may be shown; units: from `schedule`; transport,
        guard: from `orclient`; calls_path: the records; failed_path: one line per failed unit;
        concurrency: workers; max_units: cap on new units.
    Returns: {"done", "failed", "skipped", "stopped", "spent_usd", "room_usd"}.
    Raises: nothing from the provider.
    Side Effects: provider calls; appends to the two files. Resumable.
    """
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    have = complete_units(calls_path)
    queue: asyncio.Queue = asyncio.Queue()
    for item in units:
        if item not in have:
            queue.put_nowait(item)
    lock = asyncio.Lock()
    counts: Counter = Counter()
    budget = [max_units if max_units is not None else len(units)]

    async def worker() -> None:
        """Take units until none are left or a stop rule fires.

        Args: none.
        Returns: nothing.
        Raises: nothing.
        Side Effects: provider calls; appends records under the lock.
        """
        while True:
            try:
                repetition, target = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if guard.reason or budget[0] <= 0:
                counts["skipped"] += 1
                continue
            budget[0] -= 1
            try:
                records = await unit(ctx, world, repetition, target, transport, guard)
            except Failed as error:
                counts["failed"] += 1
                async with lock:
                    with failed_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"repetition": repetition, "variable_id": target,
                                                 "error": str(error)[:300], "at": ft.now()}) + "\n")
                continue
            counts["done"] += 1
            async with lock:
                with calls_path.open("a", encoding="utf-8") as handle:
                    handle.write("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return {"done": counts["done"], "failed": counts["failed"], "skipped": counts["skipped"],
            "stopped": guard.reason, "spent_usd": round(guard.spend.spent, 4), "room_usd": round(guard.spend.room(), 4)}


# --- scoring and comparison --------------------------------------------------------------------

def gain(counts: Mapping[str, Mapping[str, Sequence[float]]], first: str, second: str, ids: Sequence[str],
         draws: int = DRAWS, seed: int = SEED) -> dict[str, float]:
    """One arm against another over some variables, with the variables resampled.

    Args: counts: from `analysis.variable_counts`; first, second: arms; ids: variables; draws,
        seed: of the resampling; every comparison draws the same resamples.
    Returns: {"gain", "low", "high" (95%), "strict_low", "strict_high" (the level divided by
        `FAMILY`), "share_positive"}.
    Raises: KeyError when an arm lacks a variable.
    Side Effects: none. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    ids = sorted(ids)
    drawn = sorted(analysis.pooled_gain(counts[first], counts[second], rng.choices(ids, k=len(ids)))
                   for _ in range(draws))
    return {"gain": analysis.pooled_gain(counts[first], counts[second], ids),
            "low": drawn[int(0.025 * draws)], "high": drawn[int(0.975 * draws) - 1],
            "strict_low": drawn[int(STRICT / 2 * draws)], "strict_high": drawn[int((1 - STRICT / 2) * draws) - 1],
            "share_positive": sum(g > 0 for g in drawn) / draws}


def patterns_marked(ctx: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How often what a guided arm stated is what the experts tagged, and how the rule check went.

    Args: ctx: from `load`; records: one guided arm's records.
    Returns: counts and shares: statements made, each stated decision right, constraint types
        stated and tagged, answers that passed the check first time, after the repair, and fell
        back, and drafts that would have passed.
    Raises: KeyError for a record of an unknown variable.
    Side Effects: none. Deterministic.
    """
    tally: Counter = Counter()
    for item in records:
        tally["answers"] += 1
        stmt = item.get("statement")
        if stmt is None:
            tally["no_guidance"] += 1
            continue
        tally["guided"] += 1
        key = ctx["key"][item["variable_id"]]
        want = hints.expected_form(hints.facts_from_key(key), ctx["patterns"], ctx["types"])
        if "property" in stmt:
            tally["property_stated"] += 1
            tally["property_right"] += stmt["property"] == list(key["property"])
        if "matrix_shape" in stmt:
            tally["matrix_stated"] += 1
            tally["matrix_right"] += stmt["matrix_shape"] == want["matrix"]
            tally["matrix_pattern_right"] += stmt["matrix"] == list(key["matrix"])
        if "context" in stmt:
            tally["context_stated"] += 1
            tally["context_right"] += stmt["context"] == want["context"]
        if "constraint_types" in stmt:
            tally["constraint_stated"] += 1
            stated, tagged = set(stmt["constraint_types"]), set(want["constraint_types"])
            tally["types_stated"] += len(stated)
            tally["types_tagged"] += len(stated & tagged)
            tally["types_exact"] += stated == tagged
            tally["types_contained"] += stated <= tagged
        tally["passed_first"] += not item["first_complaints"]
        tally["repaired"] += bool(item["repaired"])
        tally["passed_after_repair"] += bool(item["repaired"]) and not item["repair_complaints"]
        tally["fell_back"] += bool(item["fell_back"])
        tally["draft_would_pass"] += not item["draft_complaints"]
    out: dict[str, Any] = dict(tally)
    for name in ("property", "matrix", "context"):
        if tally[f"{name}_stated"]:
            out[f"{name}_precision"] = tally[f"{name}_right"] / tally[f"{name}_stated"]
    if tally["types_stated"]:
        out["types_precision"] = tally["types_tagged"] / tally["types_stated"]
    if tally["guided"]:
        out["passed_first_share"] = tally["passed_first"] / tally["guided"]
        out["fell_back_share"] = tally["fell_back"] / tally["guided"]
    return out


def unit_integrity(calls: Mapping[engine.Job, Mapping[str, Any]], repetitions: Sequence[int],
                   ids: Sequence[str]) -> dict[str, Any]:
    """Whether the six records of every unit belong together.

    Args: calls: from `engine.latest`; repetitions, ids: the units to check.
    Returns: {"units", "mismatched", "examples"}: a unit is mismatched when its records do not
        share one start time, when a guided arm's draft is not its own draft (`base` for the
        draft-first arms, `rag` for the others), or when an answer without guidance or after a
        fall-back is not that draft.
    Raises: KeyError when a record is missing.
    Side Effects: none. Deterministic.
    """
    mismatched = []
    for repetition in repetitions:
        for variable_id in ids:
            records = {arm: calls[(arm, repetition, variable_id)] for arm in ARMS}
            problems = []
            if len({record.get("started_at") for record in records.values()}) != 1:
                problems.append("start times differ")
            for arm in GUIDED:
                own = ft.sha256(records[BASELINE if arm.startswith("draft") else "rag"]["answer"])
                record = records[arm]
                if record.get("draft_sha256") != own:
                    problems.append(f"{arm}: not its own draft")
                if (record.get("no_guidance") or record.get("fell_back")) and ft.sha256(record["answer"]) != own:
                    problems.append(f"{arm}: the kept answer is not its own draft")
            if problems:
                mismatched.append({"repetition": repetition, "variable_id": variable_id, "problems": problems})
    return {"units": len(repetitions) * len(ids), "mismatched": len(mismatched), "examples": mismatched[:5]}


def summarise(ctx: Mapping[str, Any], calls: Mapping[engine.Job, Mapping[str, Any]],
              scores: Mapping[engine.Job, Mapping[str, Any]], draws: int = DRAWS) -> dict[str, Any]:
    """Every arm's score, the comparisons of the plan, and what the patterns and the check did.

    Only repetitions in which every arm was stored and scored for every variable enter a figure;
    "complete" is true only when all ten did and every unit's records belong together.

    Args: ctx: from `load`; calls: from `engine.latest`; scores: from `engine.score`; draws:
        resamples of the variables.
    Returns: {"generated_at", "variables", "repetitions_planned", "repetitions_used", "complete",
        "arms": {...}, "comparisons": {arm: {...}}, "secondary": {...}, "without_relatives",
        "with_relatives", "patterns": {arm: {...}}, "unit_integrity", "spent_usd"}.
    Raises: nothing; without a complete repetition only the header is returned.
    Side Effects: none.
    """
    ids = ctx["ids"]
    used = [r for r in range(1, REPETITIONS + 1)
            if all((arm, r, v) in scores and calls.get((arm, r, v), {}).get("ok") for arm in ARMS for v in ids)]
    out: dict[str, Any] = {"generated_at": ft.now(), "variables": len(ids), "repetitions_planned": REPETITIONS,
                           "repetitions_used": used, "complete": len(used) == REPETITIONS, "arms": {},
                           "comparisons": {}, "secondary": {}, "without_relatives": {}, "with_relatives": {},
                           "patterns": {},
                           "spent_usd": sum(r.get("unit_cost_usd") or 0.0 for k, r in calls.items() if k[0] == BASELINE)}
    if not used:
        return out
    out["unit_integrity"] = unit_integrity(calls, used, ids)
    out["complete"] = out["complete"] and out["unit_integrity"]["mismatched"] == 0
    by_group = {group: [v for v in ids if ctx["group_of"][v] == group] for group in scoreboard.GROUPS}
    for arm in ARMS:
        series = [_aggregate([scores[(arm, r, v)]["evaluation"] for v in ids], ids)["f1"] for r in used]
        out["arms"][arm] = {
            "f1_mean": statistics.mean(series), "f1_sd": statistics.stdev(series) if len(series) > 1 else None,
            "f1_per_repetition": series,
            "by_group": {group: statistics.mean(_aggregate([scores[(arm, r, v)]["evaluation"] for v in members],
                                                           members)["f1"] for r in used)
                         for group, members in by_group.items()},
            "invalid": sum(1 for r in used for v in ids if not scores[(arm, r, v)]["valid_prediction"]),
            "calls": sum(calls[(arm, r, v)].get("calls") or 0 for r in used for v in ids),
            "cost_usd": sum(calls[(arm, r, v)].get("cost_usd") or 0.0 for r in used for v in ids)}
    counts = analysis.variable_counts({key: row for key, row in scores.items() if key[0] in ARMS and key[1] in used})
    for arm in COMPARED:
        row = dict(gain(counts, arm, BASELINE, ids, draws), first=arm, second=BASELINE)
        if len(used) > 1:
            row["welch"] = ft.welch(out["arms"][arm]["f1_per_repetition"], out["arms"][BASELINE]["f1_per_repetition"])
        out["comparisons"][arm] = row
    adjusted = ft.holm({arm: row["welch"]["p"] for arm, row in out["comparisons"].items() if "welch" in row})
    for arm, value in adjusted.items():
        out["comparisons"][arm]["p_holm"] = value
    for first, second in SECONDARY:
        out["secondary"][f"{first} vs {second}"] = dict(gain(counts, first, second, ids, draws), first=first,
                                                        second=second)
    without = [v for v in ids if v not in set(ctx["relatives"])]
    for arm in COMPARED:
        out["without_relatives"][arm] = dict(gain(counts, arm, BASELINE, without, draws), variables=len(without))
        out["with_relatives"][arm] = dict(gain(counts, arm, BASELINE, ctx["relatives"], draws),
                                          variables=len(ctx["relatives"]))
    for arm in GUIDED:
        out["patterns"][arm] = patterns_marked(ctx, [calls[(arm, r, v)] for r in used for v in ids])
    return out


# --- the dev check on invented variables ----------------------------------------------------

def dev_world(ctx: Mapping[str, Any], similar: Callable[[str, str], float]) -> tuple[dict[str, Callable], list[str]]:
    """The invented variables of `knowledge.py`, in two halves that serve as each other's examples.

    Args: ctx: from `load`; similar: a similarity of two texts.
    Returns: (a world in the shape of `corpus_world`, the invented ids in order). Today's prompt
        keeps the study's 25 examples, as every dev run before; everything retrieved comes from
        the other half.
    Raises: KeyError when an invented variable has no decomposition.
    Side Effects: calls `similar` for every pair of invented definitions.
    """
    items = kn.invented(ctx)
    order = sorted(items, key=lambda name: int(name.split("-")[1]))
    half = {name: position % 2 for position, name in enumerate(order)}
    gold = {name: dict(json.loads(items[name]["demo"]["gold"])) for name in order}
    ranked = similarity_order(order, lambda name: items[name]["definition"], similar)

    def others(name: str) -> list[str]:
        """The other half, most similar first.

        Args: name: an invented id.
        Returns: ids.
        Raises: nothing.
        Side Effects: none.
        """
        return [u for u in ranked[name] if half[u] != half[name]]

    def demo(u: str) -> dict[str, str]:
        """An invented variable as pattern-finding sees it.

        Args: u: an invented id.
        Returns: {"id", "definition", "anchor", "patterns"}.
        Raises: nothing.
        Side Effects: none.
        """
        return {"id": u, "definition": items[u]["definition"], "anchor": anchor_of(gold[u]),
                "patterns": items[u]["demo"]["patterns"]}

    def voter(u: str) -> dict[str, Any]:
        """An invented variable's matrix, for the vote.

        Args: u: an invented id.
        Returns: {"id", "shape", "matrix"}.
        Raises: nothing.
        Side Effects: none.
        """
        entry = items[u]["entry"]
        return {"id": u, "shape": hints.expected_form(entry, ctx["patterns"], ctx["types"])["matrix"],
                "matrix": list(entry["matrix"])}

    world = {"definition": lambda v: items[v]["definition"],
             "base_inputs": lambda v: ctx["inputs"],
             "rag_records": lambda v: [{"id": u, "definition": items[u]["definition"], "gold": gold[u]}
                                       for u in reversed(others(v)[:RETRIEVED])],
             "finder_draft": lambda v: [demo(u) for u in sorted(others(v), key=lambda u: int(u.split("-")[1]))],
             "finder_rag": lambda v: [demo(u) for u in reversed(others(v)[:RETRIEVED])],
             "voters": lambda v: [voter(u) for u in others(v)[:VOTERS]]}
    return world, order


# --- fingerprints and the command line ---------------------------------------------------------

def fingerprints(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Hash what is fixed before the run: code, data, the similarity order and the fixed prompts.

    The prompts of pattern-finding and of the guided arms other than `rag/gated` depend on the
    run's own drafts; their hashes are stored with each answer.

    Args: ctx: from `load`.
    Returns: hex digests (first 16 characters) and counts.
    Raises: OSError when a file cannot be read.
    Side Effects: reads files. Deterministic.
    """
    world = corpus_world(ctx)
    out: dict[str, Any] = {name: ceiling.file_sha256(HERE / name)[:16] for name in (
        "draft_rag.py", "knowledge.py", "decision_value.py", "orclient.py", "scoreboard.py", "hints.py",
        "engine.py", "groups.py", "settings.py", "pipeline.py", "walk.py", "route_key.py", "rulebook.py")}
    out.update({f"data/{name}": ceiling.file_sha256(rulebook.DATA / name)[:16] for name in (
        "trees.json", "patterns.json", "constraint_types.json", "route_key.json")})
    out["neighbours_file"] = ceiling.file_sha256(NEIGHBOURS)[:16] if NEIGHBOURS.exists() else None
    out["population_sha256"] = ft.sha256("\n".join(ctx["ids"]))[:16]
    out["relatives"] = len(ctx["relatives"])
    fixed = {"base": [fill_prompt(world["base_inputs"](v), world["definition"](v)) for v in ctx["ids"]],
             "rag": [fill_prompt(dict(ctx["inputs"], examples=world["rag_records"](v)), world["definition"](v))
                     for v in ctx["ids"]]}
    votes = {v: vote(world["voters"](v)) for v in ctx["ids"]}
    fixed["rag/gated"] = [fill_prompt(dict(ctx["inputs"], examples=world["rag_records"](v)), world["definition"](v),
                                      render(ctx, statement(dict(EMPTY_FACTS, matrix=list(votes[v])), {"matrix"})))
                          for v in ctx["ids"] if votes[v] is not None]
    out["prompts"] = {name: {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                             "count": len(prompts), "chars_mean": round(sum(map(len, prompts)) / len(prompts), 1)}
                      for name, prompts in fixed.items()}
    out["vote_fires"] = sum(value is not None for value in votes.values())
    out["vote_sha256"] = ft.sha256(json.dumps(votes, sort_keys=True))[:16]
    prefixes = {}
    for fold in range(scoreboard.FOLDS):
        member = next(v for v in ctx["ids"] if ctx["fold_of"][v] == fold)
        prefixes[str(fold)] = ft.sha256(finder_prompt(ctx, world["finder_draft"](member), "", "").rsplit("\nDefinition: ", 1)[0])[:16]
    out["finder_draft_prefix_by_fold"] = prefixes
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status: 0 on success, 1 when a stop rule ended a run or the stored run is
        not complete.
    Raises: SystemExit on bad arguments, a missing key or a missing similarity order.
    Side Effects: --neighbours writes the order; --dev and --run call the provider; --score
        writes the scores and the summary.
    """
    parser = argparse.ArgumentParser(description="Draft first and retrieval, with a rule check (stage 4).")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--neighbours", "--plan-only", "--dev", "--run", "--score"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--max-units", type=int, default=None, help="answer at most this many new units")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args(argv)
    if args.neighbours:
        stored = write_neighbours(kn.load())
        print(json.dumps({"model": stored["model"], "definitions_sha256": stored["definitions_sha256"][:16],
                          "variables": len(stored["order"])}, indent=2))
        return 0
    ctx = load()
    if args.plan_only:
        print(json.dumps({"variables": len(ctx["ids"]), "arms": list(ARMS), "repetitions": REPETITIONS,
                          "units": len(ctx["ids"]) * REPETITIONS, "allowance_usd": ALLOWANCE_USD,
                          "host": orclient.HOST, "model": orclient.MODEL, "family": FAMILY,
                          **fingerprints(ctx)}, indent=2))
        return 0
    if args.score:
        scores = engine.score(CALLS, SCORES, ctx["records"], ctx["inputs"]["schema"])
        summary = summarise(ctx, engine.latest(CALLS), scores)
        SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not summary["complete"]:
            print(f"NOT COMPLETE: repetitions used {summary['repetitions_used']} of {REPETITIONS}")
        for arm, cell in summary["arms"].items():
            print(f"{arm:18} F1 {cell['f1_mean']:.4f} (SD {cell['f1_sd'] or 0:.4f})  "
                  + "  ".join(f"{g} {value:.3f}" for g, value in cell["by_group"].items())
                  + f"  invalid {cell['invalid']}  ${cell['cost_usd']:.2f}")
        for arm, row in summary["comparisons"].items():
            print(f"{arm:18} vs base {row['gain']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}] strict "
                  f"[{row['strict_low']:+.4f}, {row['strict_high']:+.4f}]  runs p {row.get('p_holm', float('nan')):.2g}")
        for name, row in summary["secondary"].items():
            print(f"{name:30} {row['gain']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}]")
        for arm, row in summary["without_relatives"].items():
            print(f"without relatives {arm:18} {row['gain']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}]")
        for arm, row in summary["patterns"].items():
            print(arm, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
        print(f"spent ${summary['spent_usd']:.2f}")
        return 0 if summary["complete"] else 1

    async def go() -> int:
        """Open the transport, do the asked work, and always close the client.

        Args: none.
        Returns: the exit status: 1 when a stop rule ended the work.
        Raises: SystemExit when the key is missing or another process holds the ledger.
        Side Effects: provider calls, the ledger and output files.
        """
        purpose = "pipelines-dev" if args.dev else "pipelines"
        spend = orclient.Spend(purpose=purpose, allowance=DEV_ALLOWANCE_USD if args.dev else ALLOWANCE_USD)
        guard = orclient.Guard(spend)
        transport, client = orclient.open_transport(spend, purpose, concurrency=args.concurrency)
        try:
            if args.dev:
                similar, _ = local_similarity()
                world, order = dev_world(ctx, similar)
                result = await run(ctx, world, [(1, name) for name in order], transport, guard, DEV_CALLS, DEV_FAILED,
                                   concurrency=min(8, args.concurrency))
            else:
                result = await run(ctx, corpus_world(ctx), schedule(ctx["ids"]), transport, guard, CALLS, FAILED,
                                   concurrency=args.concurrency, max_units=args.max_units)
            print(json.dumps(result, indent=2))
            return 1 if result["stopped"] else 0
        finally:
            await orclient.close_transport(transport, client)

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
