#!/usr/bin/env python
"""Step 2, the route test: can the model find the patterns by itself?

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/GUIDED_DECOMPOSITION_20261119.md` (private repository).

Two ways of asking are compared on the practice set, each at temperature 0:

- `steps`: one question per call, as Arvin described it. The property is found by walking the
  upstream trees from DT0. The matrix is found by walking DT8. The questions the trees lack are
  ours and are marked so: one more DT8 question that gives a "no matrix" exit, one question each
  for the context object and the statistical modifier, and one yes-or-no question per constraint
  type. Every answer is a single character, so the model's confidence can be read.
- `direct`: one call that names all patterns at once, given the lists of patterns and types.

Both see the same demonstrations: the 25 example variables with their expert tags. A walk's
result ("facts") has the same shape as an answer-key entry, so `hints.render` can turn it into
guidance and the answer key can mark it.

    ./walk.py --plan-only
    ./walk.py --dev                 # invented variables only
    ./walk.py --run --style steps   # or: direct
    ./walk.py --report
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import ceiling  # noqa: E402
import engine  # noqa: E402
import hints  # noqa: E402
import httpx  # noqa: E402
import rulebook  # noqa: E402
from engine import ft  # noqa: E402

STYLES = ("steps", "direct")
LABELS = "123456789"
DEMOS_PER_QUESTION = 25
CONCURRENCY = 6
NO_MATRIX = 0
STEP_REQUEST = {"stream": False, "temperature": 0.0, "top_p": 1.0, "max_tokens": 1, "logprobs": True,
                "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}
DIRECT_REQUEST = {"stream": False, "temperature": 0.0, "top_p": 1.0, "max_tokens": 300,
                  "chat_template_kwargs": {"enable_thinking": False}}

# Questions the upstream trees lack. They are OURS, not the pattern authors'.
OUR_MATRIX_QUESTION = "Is it within an artificial container?"
OUR_CONTEXT_QUESTION = ("Does the variable need a context object: a sphere or environment, such as the "
                        "atmosphere, that is not the matrix itself?")
OUR_MODIFIER_QUESTION = "Is a statistical modifier involved, such as mean, maximum or standard deviation?"
OUR_CONSTRAINT_QUESTION = "Does the definition contain a constraint of the type “{name}”? {definition}"
OUR_CONSTRAINT_NOTE = ("A constraint is an explicit restriction in the definition that narrows the property, the "
                       "object of interest, the matrix or the statistical modifier. The property, the object of "
                       "interest and the matrix themselves are not constraints.")
ABOUT_PROPERTY = ("The question is about the PROPERTY of the variable: the kind of quantity or quality that is "
                  "measured or observed.")
ABOUT_MATRIX = ("The question is about the MATRIX of the variable: the medium, material, body, area or process "
                "in which the object of interest occurs. “It” is the object of interest. Many variables have no "
                "matrix.")

# More invented variables for the wording of the walk; none is a corpus variable. Their patterns
# are the right answers, written down before the first call.
INVENTED_MORE = (
    {"id": "dev-9", "definition": "Number of students enrolled in a primary school",
     "property": [104], "object": [33], "matrix": [42], "context": [], "constraint": []},
    {"id": "dev-10", "definition": "Mean air temperature at 2 m height",
     "property": [109], "object": [33], "matrix": [], "context": [], "constraint": [68],
     "statistical_modifier": True},
    {"id": "dev-11", "definition": "Probability of frost occurring on a night in April",
     "property": [139], "object": [33], "matrix": [], "context": [], "constraint": [78]},
    {"id": "dev-12", "definition": "Mass of a dried soil sample",
     "property": [109], "object": [33], "matrix": [], "context": [], "constraint": [76]},
    {"id": "dev-13", "definition": "Volume fraction of sand in river sediment",
     "property": [110], "object": [35], "matrix": [], "context": [], "constraint": []},
    {"id": "dev-14", "definition": "Number of bacteria per millilitre of drinking water",
     "property": [125], "object": [35], "matrix": [], "context": [], "constraint": []},
    {"id": "dev-15", "definition": "Type of roofing material of a house",
     "property": [101], "object": [33], "matrix": [], "context": [], "constraint": []},
    {"id": "dev-16", "definition": "Cover of moss on the forest floor",
     "property": [111], "object": [35], "matrix": [], "context": [], "constraint": []},
)

Ask = Callable[[dict[str, Any], str], Awaitable[dict[str, Any]]]


def results_path(style: str, name: str = "route") -> Path:
    """Where one style's walk results are stored.

    Args: style: "steps" or "direct"; name: file prefix.
    Returns: the JSON-lines path under the ignored output folder.
    Raises: nothing.
    Side Effects: none.
    """
    return engine.OUTPUT / f"{name}-{style}.jsonl"


# --- questions ---------------------------------------------------------------------------

def matrix_tree(trees: Mapping[str, Any]) -> dict[str, Any]:
    """DT8 with our extra question, so that a walk can end on "no matrix".

    Args: trees: the upstream trees.
    Returns: a copy of DT8 in which "not within a natural container" leads to our question; yes
        ends on pattern 42 as upstream does, no ends on a leaf with pattern number 0.
    Raises: KeyError when DT8 no longer has the expected last question.
    Side Effects: none. Deterministic.
    """
    tree = copy.deepcopy(trees["DT8"])
    last = next(e for e in tree["edges"] if tree["nodes"][e["to"]].get("pattern") == 42)
    tree["nodes"]["OUR1"] = {"kind": "question", "text": OUR_MATRIX_QUESTION, "ours": True}
    tree["nodes"]["NONE"] = {"kind": "pattern", "pattern": NO_MATRIX, "text": "NO MATRIX", "ours": True}
    tree["edges"].append({"from": "OUR1", "to": last["to"], "label": "true"})
    tree["edges"].append({"from": "OUR1", "to": "NONE", "label": "false"})
    last["to"] = "OUR1"
    return tree


def constraint_questions(types: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One yes-or-no question of ours per constraint type name.

    Args: types: the constraint-type table.
    Returns: questions with "id", "text", "options", "name" and "number"; two table rows that
        share a name become one question with both definitions.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    merged: dict[str, dict[str, Any]] = {}
    for row in types:
        name = row["name"].lower()
        if name in merged:
            merged[name]["definition"] += " Or: " + row["definition"]
        else:
            merged[name] = {"number": row["number"], "definition": row["definition"]}
    return [{"id": f"OUR/constraint/{name}", "name": name, "number": info["number"], "ours": True,
             "about": OUR_CONSTRAINT_NOTE, "definition": info["definition"],
             "text": OUR_CONSTRAINT_QUESTION.format(name=name, definition=info["definition"]),
             "options": ["yes", "no"]} for name, info in sorted(merged.items())]


def reachable(trees: Mapping[str, Any], name: str, node: str) -> list[int]:
    """The pattern leaves that can still be reached from a node.

    Args: trees: {name: tree}; name: the tree; node: a node id.
    Returns: sorted pattern numbers; 0 stands for our "no matrix" leaf.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    found: set[int] = set()
    stack, seen = [(name, node)], set()
    while stack:
        here = stack.pop()
        if here in seen:
            continue
        seen.add(here)
        tree = trees[here[0]]
        info = tree["nodes"][here[1]]
        if info["kind"] == "pattern":
            found.add(info["pattern"])
        elif info["kind"] == "jump":
            stack.append((info["tree"], trees[info["tree"]]["start"]))
        else:
            stack += [(here[0], edge["to"]) for edge in rulebook.options(tree, here[1])]
    return sorted(found)


def tree_question(tree_name: str, tree: Mapping[str, Any], node: str, trees: Mapping[str, Any] | None = None,
                  patterns: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One tree question in the shape the prompt builder takes.

    Args: tree_name: such as "DT3"; tree: the parsed tree; node: a question node id; trees and
        patterns: when given, each option also says which patterns it can still lead to.
    Returns: {"id", "text", "options", "ours"}, plus "about" and "leads" when trees are given.
    Raises: KeyError for an unknown node.
    Side Effects: none. Deterministic.
    """
    info = tree["nodes"][node]
    edges = rulebook.options(tree, node)
    question = {"id": f"{tree_name}/{node}", "text": info["text"], "ours": bool(info.get("ours")),
                "options": [e["label"] for e in edges]}
    if trees is not None and patterns is not None:
        scope = dict(trees, **{tree_name: tree})
        question["about"] = ABOUT_MATRIX if tree_name == "DT8" else ABOUT_PROPERTY
        question["leads"] = [", ".join("no matrix" if n == NO_MATRIX else patterns.get(str(n), {}).get("name", str(n))
                                       for n in reachable(scope, tree_name, e["to"])) for e in edges]
    return question


def question_prompt(question: Mapping[str, Any], demos: Sequence[tuple[str, str]], definition: str) -> str:
    """The prompt for one question about one definition.

    Args: question: with "text" and "options"; demos: (definition, right option) pairs from the
        example variables; definition: the target definition.
    Returns: the prompt; the reply is the number of one option.
    Raises: ValueError when a demonstration names an unknown option.
    Side Effects: none. Deterministic.
    """
    options = list(question["options"])
    leads = question.get("leads") or [""] * len(options)
    lines = ["You are following the I-ADOPT decision trees to classify a scientific variable.",
             "Answer the question about the definition. Reply with the number of one option and nothing else."]
    if question.get("about"):
        lines.append(question["about"])
    lines += ["", f"Question: {question['text']}", "Options:"]
    lines += [f"{LABELS[i]}. {label}" + (f" (leads to: {leads[i]})" if leads[i] else "")
              for i, label in enumerate(options)]
    if demos:
        lines += ["", "Examples:"]
        for text, answer in demos:
            lines += [f"Definition: {text}", f"Answer: {LABELS[options.index(answer)]}", ""]
    else:
        lines.append("")
    lines += [f"Definition: {definition}", "Answer:"]
    return "\n".join(lines)


# --- demonstrations from the 25 example variables ------------------------------------------

def right_answers(ctx: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, str]:
    """The right answer to every question that one tagged variable's routes pass through.

    Args: ctx: from `load`; entry: an answer-key entry.
    Returns: {question id: option}; a tree question on which the variable's routes disagree is
        left out; every question of ours has an answer.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    seen: dict[str, set[str]] = {}
    for number in entry["property"]:
        for route in rulebook.routes_to(ctx["trees"], number):
            for step in route:
                seen.setdefault(f"{step['tree']}/{step['node']}", set()).add(step["answer"])
    target = entry["matrix"][0] if entry["matrix"] else NO_MATRIX
    for route in rulebook.routes_to({"DT8": ctx["matrix_tree"]}, target, start_tree="DT8"):
        for step in route:
            seen.setdefault(f"DT8/{step['node']}", set()).add(step["answer"])
    out = {qid: next(iter(answers)) for qid, answers in seen.items() if len(answers) == 1}
    out["OUR/context"] = "yes" if entry["context"] else "no"
    out["OUR/modifier"] = "yes" if entry["statistical_modifier"] else "no"
    names = set(hints.constraint_names(hints.facts_from_key(entry), ctx["types"]))
    for question in ctx["constraint_questions"]:
        out[question["id"]] = "yes" if question["name"] in names else "no"
    return out


def demo_index(ctx: Mapping[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """Demonstrations for every question, drawn from the 25 example variables only.

    Args: ctx: from `load`.
    Returns: {question id: up to DEMOS_PER_QUESTION (definition, right option) pairs}: every
        example the question applies to, in a fixed order, so the options keep the share they have
        among the examples.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    groups: dict[str, dict[str, list[str]]] = {}
    for variable_id in sorted(ft.EXAMPLE_IDS):
        definition = ctx["records"][variable_id]["definition"]
        for qid, answer in right_answers(ctx, ctx["key"][variable_id]).items():
            groups.setdefault(qid, {}).setdefault(answer, []).append(definition)
    order = {ctx["records"][v]["definition"]: position for position, v in enumerate(sorted(ft.EXAMPLE_IDS))}
    index: dict[str, list[tuple[str, str]]] = {}
    for qid, by_answer in groups.items():
        pairs = [(text, answer) for answer, texts in by_answer.items() for text in texts]
        index[qid] = sorted(pairs, key=lambda pair: order[pair[0]])[:DEMOS_PER_QUESTION]
    return index


# --- the model ---------------------------------------------------------------------------

def option_probabilities(top: Sequence[Mapping[str, Any]], count: int) -> list[float]:
    """Turn first-token log-probabilities into a probability per option.

    Args: top: the reported top tokens, each with "token" and "logprob"; count: number of options.
    Returns: one probability per option, summing to 1 over the options found; all zero when no
        option label was reported.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    mass = [0.0] * count
    for item in top:
        token = str(item.get("token", "")).strip()
        if len(token) == 1 and token in LABELS[:count]:
            mass[LABELS.index(token)] += math.exp(float(item.get("logprob", -math.inf)))
    total = sum(mass)
    return [m / total for m in mass] if total > 0 else mass


def chat_transport(concurrency: int = CONCURRENCY) -> tuple[Callable[[Mapping[str, Any]], Awaitable[dict]], Any]:
    """A PSNC chat transport that also returns the first token's log-probabilities.

    Args: concurrency: size of the connection pool.
    Returns: (transport, client); the caller closes the client.
    Raises: SystemExit when PSNC_API_KEY is not set.
    Side Effects: reads the repository's .env through the lab's allowlisted loader; stores no
        headers; opens no connection until the first call.
    """
    from iadopt_lab.configuration import load_runtime_secrets

    secrets = load_runtime_secrets(str(engine.ROOT.parent / ".env"), ["PSNC_API_KEY", "PSNC_API_BASE_URL"],
                                   os.environ)
    key = secrets.get("PSNC_API_KEY")
    if not key:
        raise SystemExit("PSNC_API_KEY is not set")
    url = (secrets.get("PSNC_API_BASE_URL") or "https://llm.hpc.psnc.pl").rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    client = httpx.AsyncClient(timeout=ft.TIMEOUT, follow_redirects=False,
                               limits=httpx.Limits(max_connections=concurrency * 2,
                                                   max_keepalive_connections=concurrency))

    async def transport(body: Mapping[str, Any]) -> dict:
        """Send one request; a failure comes back as data, never as an exception.

        Args: body: the chat-completions request body.
        Returns: {"ok", "status", "latency"}, plus answer, top tokens and token counts on success.
        Raises: nothing.
        Side Effects: one HTTPS request on the shared client.
        """
        started = time.monotonic()
        try:
            response = await client.post(url, headers=headers, json=body)
        except Exception as error:  # noqa: BLE001 - a failed call is recorded, never raised
            return {"ok": False, "status": None, "error": type(error).__name__,
                    "latency": time.monotonic() - started}
        latency = time.monotonic() - started
        if response.status_code != 200:
            return {"ok": False, "status": response.status_code, "error": f"HTTP {response.status_code}",
                    "latency": latency}
        try:
            payload = response.json()
        except ValueError:
            return {"ok": False, "status": 200, "error": "body is not JSON", "latency": latency}
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = payload.get("usage") or {}
        content = ((choice.get("logprobs") or {}).get("content") or [{}])
        top = [{"token": t.get("token"), "logprob": t.get("logprob")}
               for t in (content[0].get("top_logprobs") or [])] if content else []
        return {"ok": True, "status": 200, "latency": latency, "answer": message.get("content") or "",
                "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
                "top": top, "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"), "served_model": payload.get("model")}

    return transport, client


def make_ask(ctx: Mapping[str, Any], transport: Callable[[Mapping[str, Any]], Awaitable[dict]],
             demos: Mapping[str, Sequence[tuple[str, str]]], log: list[dict]) -> Ask:
    """Build the function that asks the model one question.

    Args: ctx: from `load`; transport: sends one request; demos: from `demo_index`; log: receives
        one record per call.
    Returns: ask(question, definition) -> {"answer", "confidence", "probabilities", "ok"}.
    Raises: nothing.
    Side Effects: the returned function calls the provider and appends to `log`.
    """
    async def ask(question: dict[str, Any], definition: str) -> dict[str, Any]:
        """Ask one question and read the chosen option and its probability.

        Args: question: with "id", "text" and "options"; definition: the target definition.
        Returns: the chosen option; when the reply is no option label, the most probable option
            is taken, and when no label was reported at all the answer is None.
        Raises: nothing.
        Side Effects: one provider call; one log record.
        """
        options = list(question["options"])
        prompt = question_prompt(question, demos.get(question["id"], ()), definition)
        body = dict(STEP_REQUEST, model=engine.MODEL, messages=[{"role": "user", "content": prompt}])
        result = await transport(body)
        probabilities = option_probabilities(result.get("top") or [], len(options)) if result.get("ok") else []
        text = (result.get("answer") or "").strip()
        index = LABELS.index(text) if len(text) == 1 and text in LABELS[:len(options)] else None
        if index is None and probabilities and max(probabilities) > 0:
            index = probabilities.index(max(probabilities))
        answer = options[index] if index is not None else None
        confidence = probabilities[index] if index is not None and probabilities else None
        log.append({"question": question["id"], "ok": bool(result.get("ok")), "status": result.get("status"),
                    "error": result.get("error"), "raw": result.get("answer"), "answer": answer,
                    "confidence": confidence, "probabilities": probabilities,
                    "reasoning_chars": len(result.get("reasoning") or ""),
                    "prompt_tokens": result.get("prompt_tokens"),
                    "completion_tokens": result.get("completion_tokens"),
                    "latency_seconds": round(result.get("latency") or 0.0, 3), "prompt_sha256": ft.sha256(prompt)})
        return {"answer": answer, "confidence": confidence, "probabilities": probabilities,
                "ok": bool(result.get("ok")), "status": result.get("status")}

    return ask


# --- walking -----------------------------------------------------------------------------

async def walk_tree(trees: Mapping[str, Any], start: str, ask: Ask, definition: str,
                    patterns: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Walk from a tree's start to a pattern leaf, asking one question at a time.

    Args: trees: {name: tree}; start: the tree to begin in; ask: asks one question; definition:
        the target definition; patterns: the pattern table, so that each option can say where it
        leads.
    Returns: {"pattern": number or None, "steps": [{"question", "answer", "confidence"}],
        "confidence": the lowest step confidence, or None}. A question with one option is
        answered without a call. The pattern is None when an answer could not be read.
    Raises: nothing.
    Side Effects: provider calls through `ask`.
    """
    name, node, steps, visited = start, trees[start]["start"], [], set()
    while (name, node) not in visited:
        visited.add((name, node))
        tree = trees[name]
        info = tree["nodes"][node]
        if info["kind"] == "pattern":
            scores = [s["confidence"] for s in steps if s["confidence"] is not None]
            return {"pattern": info["pattern"], "steps": steps, "confidence": min(scores) if scores else None}
        if info["kind"] == "jump":
            name, node = info["tree"], trees[info["tree"]]["start"]
            continue
        edges = rulebook.options(tree, node)
        if info["kind"] != "question" or len(edges) == 1:
            if not edges:
                break
            node = edges[0]["to"]
            continue
        question = tree_question(name, tree, node, trees, patterns)
        reply = await ask(question, definition)
        steps.append({"question": question["id"], "answer": reply["answer"], "confidence": reply["confidence"]})
        if reply["answer"] is None:
            break
        node = next(e["to"] for e in edges if e["label"] == reply["answer"])
    return {"pattern": None, "steps": steps, "confidence": None}


async def walk_steps(ctx: Mapping[str, Any], definition: str, ask: Ask) -> dict[str, Any]:
    """Find all facts of one variable step by step.

    Args: ctx: from `load`; definition: the target definition; ask: asks one question.
    Returns: {"facts": facts or None, "trace": {...}, "confidence": {part: lowest probability}}.
        Facts are None when the property walk did not reach a pattern.
    Raises: nothing.
    Side Effects: provider calls through `ask`.
    """
    prop = await walk_tree(ctx["trees"], "DT0", ask, definition, ctx["patterns"])
    matrix = await walk_tree({"DT8": ctx["matrix_tree"]}, "DT8", ask, definition, ctx["patterns"])
    context = await ask({"id": "OUR/context", "text": OUR_CONTEXT_QUESTION, "options": ["yes", "no"]}, definition)
    modifier = await ask({"id": "OUR/modifier", "text": OUR_MODIFIER_QUESTION, "options": ["yes", "no"]}, definition)
    constraints, weakest = [], None
    for question in ctx["constraint_questions"]:
        reply = await ask(question, definition)
        if reply["answer"] == "yes":
            constraints.append(question["number"])
        if reply["confidence"] is not None:
            weakest = reply["confidence"] if weakest is None else min(weakest, reply["confidence"])
    trace = {"property": prop["steps"], "matrix": matrix["steps"]}
    confidence = {"property": prop["confidence"], "matrix": matrix["confidence"],
                  "context": context["confidence"], "modifier": modifier["confidence"], "constraints": weakest}
    if prop["pattern"] is None:
        return {"facts": None, "trace": trace, "confidence": confidence}
    facts = {"property": [prop["pattern"]], "object": [],
             "matrix": [matrix["pattern"]] if matrix["pattern"] else [],
             "context": [50] if context["answer"] == "yes" else [],
             "constraint": sorted(constraints), "statistical_modifier": modifier["answer"] == "yes", "names": {}}
    return {"facts": facts, "trace": trace, "confidence": confidence}


def direct_prompt(ctx: Mapping[str, Any], definition: str) -> str:
    """The single prompt that asks for all patterns at once.

    Args: ctx: from `load`; definition: the target definition.
    Returns: the prompt, with the pattern lists and the 25 example variables' tags.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    patterns, lines = ctx["patterns"], [
        "Classify the scientific variable defined below with the I-ADOPT patterns.",
        'Return one JSON object with these keys: "property" (one property pattern number), "matrix" '
        "(one matrix pattern number, or null when there is no matrix), \"context\" (true or false: is a "
        'context object needed), "modifier" (true or false: is a statistical modifier involved) and '
        '"constraints" (a list of constraint type names; [] when there are none). Return the JSON only.',
        "", "PROPERTY PATTERNS"]
    for number in ctx["reachable"]:
        pattern = patterns[str(number)]
        lines.append(f"{number} {pattern['name']}: {pattern['definition'] or ''}".rstrip(": ").rstrip())
    lines += ["", "MATRIX PATTERNS"]
    lines += [f"{n} {patterns[str(n)]['name']}: {patterns[str(n)]['definition'] or ''}".rstrip(": ").rstrip()
              for n in range(40, 46)]
    lines += ["", "CONSTRAINT TYPES", OUR_CONSTRAINT_NOTE]
    lines += [f"{q['name']}: {q['definition']}" for q in ctx["constraint_questions"]]
    lines += ["", "EXAMPLES"]
    for variable_id in sorted(ft.EXAMPLE_IDS):
        lines += [f"Definition: {ctx['records'][variable_id]['definition']}",
                  "Patterns: " + json.dumps(direct_answer(ctx, ctx["key"][variable_id]), ensure_ascii=False), ""]
    lines += [f"Definition: {definition}", "Patterns:"]
    return "\n".join(lines)


def direct_answer(ctx: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    """An answer-key entry in the form the direct prompt asks for.

    Args: ctx: from `load`; entry: an answer-key entry.
    Returns: {"property", "matrix", "context", "modifier", "constraints"}.
    Raises: IndexError when the entry has no property tag.
    Side Effects: none. Deterministic.
    """
    return {"property": entry["property"][0], "matrix": entry["matrix"][0] if entry["matrix"] else None,
            "context": bool(entry["context"]), "modifier": bool(entry["statistical_modifier"]),
            "constraints": hints.constraint_names(hints.facts_from_key(entry), ctx["types"])}


def facts_from_direct(ctx: Mapping[str, Any], text: str) -> dict[str, Any] | None:
    """Read facts from a direct reply.

    Args: ctx: from `load`; text: the model's reply.
    Returns: facts, or None when the reply is not the requested JSON or names no known property
        pattern. Unknown matrix numbers and constraint names are dropped.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    from iadopt_lab.generation.extractor import extract_json

    extracted = extract_json(text or "")
    data = extracted.candidate if extracted.success and isinstance(extracted.candidate, dict) else None
    if data is None or str(data.get("property")) not in ctx["patterns"]:
        return None
    if ctx["patterns"][str(data["property"])]["kind"] != "property":
        return None
    by_name = {q["name"]: q["number"] for q in ctx["constraint_questions"]}
    matrix = data.get("matrix")
    names = data.get("constraints") if isinstance(data.get("constraints"), list) else []
    return {"property": [int(data["property"])], "object": [],
            "matrix": [int(matrix)] if isinstance(matrix, int) and 40 <= matrix <= 45 else [],
            "context": [50] if data.get("context") is True else [],
            "constraint": sorted({by_name[str(n).lower()] for n in names if str(n).lower() in by_name}),
            "statistical_modifier": data.get("modifier") is True, "names": {}}


async def walk_direct(ctx: Mapping[str, Any], definition: str,
                      transport: Callable[[Mapping[str, Any]], Awaitable[dict]], log: list[dict]) -> dict[str, Any]:
    """Find all facts of one variable in one call.

    Args: ctx: from `load`; definition: the target definition; transport: sends one request;
        log: receives one record for the call.
    Returns: {"facts": facts or None, "trace": {}, "confidence": {}}.
    Raises: nothing.
    Side Effects: one provider call; one log record.
    """
    prompt = direct_prompt(ctx, definition)
    result = await transport(dict(DIRECT_REQUEST, model=engine.MODEL, messages=[{"role": "user", "content": prompt}]))
    log.append({"question": "direct", "ok": bool(result.get("ok")), "status": result.get("status"),
                "error": result.get("error"), "raw": result.get("answer"),
                "reasoning_chars": len(result.get("reasoning") or ""),
                "prompt_tokens": result.get("prompt_tokens"), "completion_tokens": result.get("completion_tokens"),
                "latency_seconds": round(result.get("latency") or 0.0, 3), "prompt_sha256": ft.sha256(prompt)})
    facts = facts_from_direct(ctx, result.get("answer") or "") if result.get("ok") else None
    return {"facts": facts, "trace": {}, "confidence": {}}


async def run(ctx: Mapping[str, Any], style: str, items: Mapping[str, str],
              transport: Callable[[Mapping[str, Any]], Awaitable[dict]], path: Path,
              concurrency: int = CONCURRENCY) -> dict[str, Any]:
    """Walk every item not already stored, one stored row per item.

    Args: ctx: from `load`; style: "steps" or "direct"; items: {id: definition}; transport: sends
        one request; path: the results file; concurrency: items walked at once.
    Returns: {"walked", "skipped", "stopped"}.
    Raises: ValueError for an unknown style.
    Side Effects: provider calls; appends to `path`. Resumable per item. A refused credential or
        any reasoning text stops the run.
    """
    if style not in STYLES:
        raise ValueError(f"unknown style: {style}")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["id"] for row in ft.read_jsonl(path) if row.get("complete")}
    demos = demo_index(ctx)
    semaphore, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    state = {"walked": 0, "skipped": 0, "stopped": None}

    async def one(ident: str, definition: str) -> None:
        """Walk one item and store its row.

        Args: ident: the item's id; definition: its definition.
        Returns: nothing.
        Raises: nothing.
        Side Effects: provider calls; one appended row.
        """
        async with semaphore:
            if state["stopped"]:
                state["skipped"] += 1
                return
            log: list[dict] = []
            started = ft.now()
            if style == "steps":
                result = await walk_steps(ctx, definition, make_ask(ctx, transport, demos, log))
            else:
                result = await walk_direct(ctx, definition, transport, log)
            if any(c.get("status") in (401, 403) for c in log):
                state["stopped"] = "credential refused"
            elif any(c.get("reasoning_chars") for c in log):
                state["stopped"] = "answers carry reasoning text despite the switch"
            row = {"id": ident, "style": style, "started_at": started, "facts": result["facts"],
                   "trace": result["trace"], "confidence": result["confidence"], "calls": log,
                   "complete": all(c["ok"] for c in log)}
            state["walked"] += 1
            async with lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(i, d) for i, d in items.items() if i not in done))
    return state


def stored(path: Path) -> dict[str, dict]:
    """The stored walk of every item: the last complete row, else the last row.

    Args: path: a results file.
    Returns: {item id: row}.
    Raises: OSError when the file exists but cannot be read.
    Side Effects: reads the file.
    """
    out: dict[str, dict] = {}
    for row in ft.read_jsonl(path):
        if row["id"] not in out or row.get("complete") or not out[row["id"]].get("complete"):
            out[row["id"]] = row
    return out


# --- marking against the answer key ------------------------------------------------------

def mark(ctx: Mapping[str, Any], facts: Mapping[str, Any] | None, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Mark one walk against the expert answer key.

    Args: ctx: from `load`; facts: the walk's result, or None; entry: the answer-key entry.
    Returns: {"property", "object", "matrix", "matrix_pattern", "context", "modifier",
        "constraint_set": booleans, "constraint_tp", "constraint_fp", "constraint_fn": counts}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    want = hints.expected_form(hints.facts_from_key(entry), ctx["patterns"], ctx["types"])
    wanted_types = set(want["constraint_types"])
    if facts is None:
        return {"property": False, "object": False, "matrix": False, "matrix_pattern": False,
                "context": False, "modifier": False, "constraint_set": False,
                "constraint_tp": 0, "constraint_fp": 0, "constraint_fn": len(wanted_types)}
    got = hints.expected_form(facts, ctx["patterns"], ctx["types"])
    got_types = set(got["constraint_types"])
    return {"property": facts["property"] == list(entry["property"]), "object": got["object"] == want["object"],
            "matrix": got["matrix"] == want["matrix"], "matrix_pattern": facts["matrix"] == list(entry["matrix"]),
            "context": got["context"] == want["context"],
            "modifier": got["statistical_modifier"] == want["statistical_modifier"],
            "constraint_set": got_types == wanted_types, "constraint_tp": len(got_types & wanted_types),
            "constraint_fp": len(got_types - wanted_types), "constraint_fn": len(wanted_types - got_types)}


def report(ctx: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> dict[str, Any]:
    """Accuracy of a style's walks over a set of variables.

    Args: ctx: from `load`; rows: from `stored`; ids: the variables to mark.
    Returns: {"variables", "walked", "no_facts", shares per decision, "constraint_f1", "calls",
        "prompt_tokens"}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    marks = [mark(ctx, rows[v]["facts"], ctx["key"][v]) for v in ids if v in rows]
    count = len(marks)
    out: dict[str, Any] = {"variables": len(ids), "walked": count,
                           "no_facts": sum(rows[v]["facts"] is None for v in ids if v in rows)}
    if not count:
        return out
    for name in ("property", "object", "matrix", "matrix_pattern", "context", "modifier", "constraint_set"):
        out[name] = sum(m[name] for m in marks) / count
    tp, fp, fn = (sum(m[f"constraint_{k}"] for m in marks) for k in ("tp", "fp", "fn"))
    precision, recall = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    out["constraint_precision"], out["constraint_recall"] = precision, recall
    out["constraint_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    calls = [c for v in ids if v in rows for c in rows[v]["calls"]]
    out["calls"] = len(calls)
    out["prompt_tokens"] = sum(c.get("prompt_tokens") or 0 for c in calls)
    out["list_price_usd"] = sum(ft.list_price_usd(engine.MODEL, c) for c in calls)
    return out


def load() -> dict[str, Any]:
    """Everything the route test needs. No network, no secrets.

    Args: none.
    Returns: the ceiling test's context plus the trees, our matrix tree, our constraint questions
        and the reachable property patterns.
    Raises: as `ceiling.load`.
    Side Effects: reads local files.
    """
    ctx = ceiling.load()
    trees = rulebook.load("trees")["trees"]
    ctx.update(trees=trees, matrix_tree=matrix_tree(trees), constraint_questions=constraint_questions(ctx["types"]),
               reachable=[int(n) for n, p in sorted(ctx["patterns"].items(), key=lambda kv: int(kv[0]))
                          if p["kind"] == "property" and rulebook.routes_to(trees, int(n))])
    return ctx


def fingerprints(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Hash what the route test sends.

    Args: ctx: from `load`.
    Returns: digests of the code, of every question prompt for a fixed invented definition, and
        of every direct prompt for the practice set.
    Raises: nothing.
    Side Effects: reads this folder's files. Deterministic.
    """
    demos = demo_index(ctx)
    probe = ceiling.INVENTED[0]["definition"]
    scopes = [(name, tree, ctx["trees"]) for name, tree in ctx["trees"].items() if name not in ("DT6", "DT7", "DT8")]
    scopes.append(("DT8", ctx["matrix_tree"], {"DT8": ctx["matrix_tree"]}))
    questions = [tree_question(name, tree, node, scope, ctx["patterns"]) for name, tree, scope in scopes
                 for node, info in sorted(tree["nodes"].items()) if info["kind"] == "question"]
    questions += [{"id": "OUR/context", "text": OUR_CONTEXT_QUESTION, "options": ["yes", "no"]},
                  {"id": "OUR/modifier", "text": OUR_MODIFIER_QUESTION, "options": ["yes", "no"]}]
    questions += ctx["constraint_questions"]
    prompts = [question_prompt(q, demos.get(q["id"], ()), probe) for q in questions]
    direct = [direct_prompt(ctx, ctx["records"][v]["definition"]) for v in ctx["practice"]]
    return {"walk.py": ceiling.file_sha256(HERE / "walk.py"), "questions": len(questions),
            "questions_with_demonstrations": sum(bool(demos.get(q["id"])) for q in questions),
            "step_prompts_sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts)),
            "direct_prompts_sha256": ft.sha256("\n".join(ft.sha256(p) for p in direct)),
            "direct_prompt_chars_mean": round(sum(len(p) for p in direct) / len(direct), 1)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode: plan, dev, run or report.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status, 0 on success.
    Raises: SystemExit on bad arguments or a missing PSNC key.
    Side Effects: --dev and --run call the provider and write output files.
    """
    parser = argparse.ArgumentParser(description="Route test (exploratory).")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--plan-only", "--dev", "--run", "--report"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--style", choices=STYLES)
    parser.add_argument("--dev-name", default="route-dev", help="file prefix for a wording check")
    parser.add_argument("--population", choices=("practice", "held-out"), default="practice",
                        help="held-out is for the final exam only")
    args = parser.parse_args(argv)
    ctx = load()
    population = ctx["practice"] if args.population == "practice" else sorted(ctx["inputs"]["targets"])
    file_name = "route" if args.population == "practice" else "route-held-out"
    if args.plan_only:
        print(json.dumps(fingerprints(ctx), indent=2))
        return 0
    if args.report:
        for style in STYLES:
            print(style, json.dumps(report(ctx, stored(results_path(style, file_name)), population), indent=1))
        return 0
    styles = [args.style] if args.style else list(STYLES)

    async def go() -> int:
        """Open the transport, walk the invented or the practice variables, close the client.

        Args: none.
        Returns: the exit status.
        Raises: SystemExit when the PSNC key is missing.
        Side Effects: provider calls and output files.
        """
        transport, client = chat_transport()
        try:
            for style in styles:
                if args.dev:
                    invented = list(ceiling.INVENTED) + list(INVENTED_MORE)
                    items = {item["id"]: item["definition"] for item in invented}
                    state = await run(ctx, style, items, transport, results_path(style, args.dev_name))
                    rows = stored(results_path(style, args.dev_name))
                    for item in invented:
                        entry = dict(item, statistical_modifier=item.get("statistical_modifier", False), names={})
                        marks = mark(ctx, rows[item["id"]]["facts"], entry)
                        flags = "".join("Y" if marks[k] else "n" for k in
                                        ("property", "object", "matrix", "context", "modifier", "constraint_set"))
                        print(style, item["id"], flags, json.dumps(rows[item["id"]]["facts"]),
                              Counter(len(rows[item["id"]]["calls"]) for _ in [0]))
                else:
                    items = {v: ctx["records"][v]["definition"] for v in population}
                    state = await run(ctx, style, items, transport, results_path(style, file_name))
                print(style, json.dumps(state))
            return 0
        finally:
            await client.aclose()

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
