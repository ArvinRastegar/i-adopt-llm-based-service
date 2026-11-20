#!/usr/bin/env python
"""Stage 5 of the pattern-retrieval work: tuning retrieval on the 102 variables, chosen fold by fold.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/PATTERN_RETRIEVAL_20261119.md` (private repository).

Stage 4 found that today's prompt does better with the 25 most similar decomposed variables as its
examples (`draft_rag.py`, arm `rag`). Arvin: "just tune on the already 102 variables that we have
because those are the only variables aviable." No variable is left that no choice has seen, so the
setting is chosen fold by fold: for the variables of one fold, the setting that scores best on the
other five folds; the score of those choices is the estimate of what tuning buys.

Eleven arms, each one call of today's prompt with other examples (`base` excepted), on
`qwen/qwen3.8-27b` at the pinned host (see `orclient.py`). Every retrieved example comes from the
other five folds:

- `base`: today's prompt with its fixed 25 examples. Its answer is the draft of `rag-25-labels`.
- `rag-25`: the 25 most similar variables, most similar last (stage 4's `rag`).
- `rag-10`, `rag-50`, `rag-85`: the 10, 50 or all 85 most similar, most similar last.
- `rag-25-first`: the 25 most similar, most similar first.
- `fixed+rag-10`: today's 25 examples followed by the 10 most similar variables not among them.
- `rag-25-words`: the 25 most similar by shared words (`knowledge.nearest_last`) instead of the
  embedding model.
- `rag-25-mmr`: 25 chosen for similarity to the target and difference from each other (maximal
  marginal relevance), most similar last.
- `rag-25-labels`: 25 chosen by the similarity of definitions and of the draft's labels to the
  examples' gold labels, half each; the plain `rag-25` order when the draft is unusable.
- `rag-25+retry`: the `rag-25` answer; an answer that fails the schema goes back once with its
  problems, and the corrected answer is kept when it passes.

All arms of one variable and repetition (a "unit") are answered together and stored only when
every call succeeded.

    ./rag_tuning.py --similarities  # the cosine of every pair of definitions (local model)
    ./rag_tuning.py --plan-only     # fingerprints, counts and the cost estimate; no calls
    ./rag_tuning.py --dev           # every arm on invented variables only
    ./rag_tuning.py --run           # the grid, resumable, under the spending limit
    ./rag_tuning.py --score         # score the stored answers and write the summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
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
import knowledge as kn  # noqa: E402
import orclient  # noqa: E402
import rulebook  # noqa: E402
import scoreboard  # noqa: E402
import settings  # noqa: E402
from engine import ft  # noqa: E402
from harness import _aggregate  # noqa: E402

pipeline = settings.pipeline        # this folder's pipeline.py, loaded by path; see settings.py

ARMS = ("base", "rag-25", "rag-10", "rag-50", "rag-85", "rag-25-first", "fixed+rag-10", "rag-25-words",
        "rag-25-mmr", "rag-25-labels", "rag-25+retry")
CONFIGS = ARMS[1:]
BASELINE, REFERENCE = "base", "rag-25"
REPETITIONS = 10
SCHEDULE_SEED = 20261129
MMR_LAMBDA = 0.7
LABEL_WEIGHT = 0.5
FIXED_EXTRA = 10
ALLOWANCE_USD = 8.0
DEV_ALLOWANCE_USD = 0.3
CONCURRENCY = 16
DRAWS, SEED = 10000, 20261130
FAMILY = 2                          # the tuned setting against today's prompt and against `rag-25`
STRICT = 0.05 / FAMILY
CALLS = orclient.OUTPUT / "tuning-calls.jsonl"
FAILED = orclient.OUTPUT / "tuning-failed.jsonl"
SCORES = orclient.OUTPUT / "tuning-scores.jsonl"
SUMMARY = orclient.OUTPUT / "tuning-summary.json"
SIMILARITIES = orclient.OUTPUT / "tuning-similarities.json"
DEV_CALLS = orclient.OUTPUT / "tuning-dev-calls.jsonl"
DEV_FAILED = orclient.OUTPUT / "tuning-dev-failed.jsonl"


# --- similarity -------------------------------------------------------------------------------

def similarity_table(ids: Sequence[str], text: Callable[[str], str],
                     similar: Callable[[str, str], float]) -> dict[str, dict[str, float]]:
    """The similarity of every pair of items, rounded as `draft_rag.similarity_order` rounds it.

    Args: ids: the items; text: an item's text; similar: a similarity of two texts.
    Returns: {a: {b: similarity}} for every pair of different items, symmetric.
    Raises: nothing from this function.
    Side Effects: calls `similar` once per pair. Deterministic for a deterministic `similar`.
    """
    ids = sorted(ids)
    table: dict[str, dict[str, float]] = {v: {} for v in ids}
    for position, first in enumerate(ids):
        for second in ids[position + 1:]:
            value = round(float(similar(text(first), text(second))), 9)
            table[first][second] = table[second][first] = value
    return table


def order_of(table: Mapping[str, Mapping[str, float]]) -> dict[str, list[str]]:
    """Every item's other items, most similar first, equals by id.

    Args: table: from `similarity_table`.
    Returns: {item: other items}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return {v: sorted(row, key=lambda u: (-row[u], u)) for v, row in table.items()}


def write_similarities(path: Path = SIMILARITIES) -> dict[str, Any]:
    """Compute and store the cosine of every pair of the 102 definitions with the local model.

    Args: path: where to store it.
    Returns: {"model", "definitions_sha256", "table"}.
    Raises: SystemExit when the order it gives is not stage 4's stored order.
    Side Effects: loads the local model; reads stage 4's order; writes `path`.
    """
    ctx = dr.load()
    similar, identity = dr.local_similarity()
    table = similarity_table(ctx["ids"], lambda v: ctx["records"][v]["definition"], similar)
    if order_of(table) != ctx["order"]:
        raise SystemExit("the similarities do not give stage 4's order")
    out = {"model": identity, "definitions_sha256": dr.definitions_digest(ctx), "table": table}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load(table: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, Any]:
    """Everything this stage needs. No network, no secrets.

    Args: table: the similarities of the definitions; read from `SIMILARITIES` when None.
    Returns: the context of `draft_rag.load`, with "order" derived from the table, plus "table".
    Raises: SystemExit when the similarities have not been computed or belong to other
        definitions.
    Side Effects: reads local files.
    """
    if table is None:
        if not SIMILARITIES.exists():
            raise SystemExit("run --similarities first")
        stored = json.loads(SIMILARITIES.read_text(encoding="utf-8"))
        table = stored["table"]
        ctx = dr.load(order=order_of(table))
        if stored["definitions_sha256"] != dr.definitions_digest(ctx):
            raise SystemExit("the stored similarities belong to other definitions")
    else:
        ctx = dr.load(order=order_of(table))
    ctx["table"] = {v: dict(row) for v, row in table.items()}
    return ctx


def labels_of(decomposition: Mapping[str, Any]) -> str:
    """The labels of a decomposition as one text, for comparing decompositions.

    Args: decomposition: a six-field decomposition.
    Returns: property, the parts of the object of interest and of the matrix, context object,
        modifier and the constraint labels, joined with "; ", empty ones left out.
    Raises: KeyError when a field is missing.
    Side Effects: none. Deterministic.
    """
    out: list[str] = []

    def add(value: Any) -> None:
        """Add one field's labels.

        Args: value: a lexical value.
        Returns: nothing.
        Raises: nothing.
        Side Effects: appends to `out`.
        """
        if isinstance(value, str):
            if value:
                out.append(value)
        elif isinstance(value, Mapping):
            for name, part in value.items():
                if name not in ("AsymmetricSystem", "SymmetricSystem"):
                    for item in part if isinstance(part, list) else [part]:
                        add(item)

    for field in ("hasProperty", "hasObjectOfInterest", "hasMatrix", "hasContextObject", "hasStatisticalModifier"):
        add(decomposition[field])
    out += [constraint["label"] for constraint in decomposition["hasConstraint"]]
    return "; ".join(out)


# --- what each arm shows -------------------------------------------------------------------------

def corpus_world(ctx: Mapping[str, Any], label_similar: Callable[[str, str], float] | None) -> dict[str, Any]:
    """What a corpus target may be shown, and how items compare.

    Args: ctx: from `load`; label_similar: a similarity of two label texts (the local model), or
        None when no arm that needs it will run.
    Returns: {"definition", "base_inputs", "pool" (the 85 variables of the other folds, most
        similar first), "fixed" (today's examples for the target), "record", "similar" (of two
        items' definitions), "label_similar"}.
    Raises: nothing.
    Side Effects: none.
    """
    table = ctx["table"]
    return {"definition": lambda v: ctx["records"][v]["definition"],
            "base_inputs": lambda v: groups.context_for(ctx, v)["inputs"],
            "pool": lambda v: dr.nearest(ctx, v),
            "fixed": lambda v: [r["variable_id"] for r in groups.context_for(ctx, v)["inputs"]["examples"]],
            "record": lambda u: ctx["records"][u],
            "similar": lambda a, b: table[a][b],
            "label_similar": label_similar}


def mmr(target: str, pool: Sequence[str], similar: Callable[[str, str], float], count: int,
        weight: float = MMR_LAMBDA) -> list[str]:
    """Pick items similar to the target and different from each other (maximal marginal relevance).

    Args: target: the target; pool: candidates, most similar first; similar: a similarity of two
        items; count: how many to pick; weight: the weight of similarity to the target.
    Returns: the picked items, most similar to the target last, equals by id.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    picked: list[str] = []
    rest = list(pool)
    while rest and len(picked) < count:
        def value(item: str) -> float:
            """How much an item adds now.

            Args: item: a candidate.
            Returns: its marginal relevance.
            Raises: nothing.
            Side Effects: none.
            """
            redundancy = max((similar(item, other) for other in picked), default=0.0)
            return weight * similar(target, item) - (1 - weight) * redundancy

        best = max(rest, key=lambda item: (value(item), -rest.index(item)))
        picked.append(best)
        rest.remove(best)
    return sorted(picked, key=lambda item: (similar(target, item), item))


def examples(world: Mapping[str, Any], arm: str, target: str, draft: Mapping[str, Any] | None = None) -> list[str]:
    """The examples one arm shows a target, in prompt order.

    Args: world: from `corpus_world` (or the dev world); arm: an arm other than `base`; target:
        the variable; draft: the target's draft decomposition, for `rag-25-labels`.
    Returns: example ids.
    Raises: KeyError for `base` or an unknown arm.
    Side Effects: `rag-25-labels` compares label texts with the world's label similarity.
    """
    pool = list(world["pool"](target))
    if arm in ("rag-25", "rag-25+retry"):
        return pool[:25][::-1]
    if arm in ("rag-10", "rag-50"):
        return pool[:int(arm.split("-")[1])][::-1]
    if arm == "rag-85":
        return pool[::-1]
    if arm == "rag-25-first":
        return pool[:25]
    if arm == "fixed+rag-10":
        fixed = list(world["fixed"](target))
        return fixed + [u for u in pool if u not in fixed][:FIXED_EXTRA][::-1]
    if arm == "rag-25-words":
        demos = [{"id": u, "definition": world["record"](u)["definition"]} for u in pool]
        return [d["id"] for d in kn.nearest_last(world["definition"](target), demos)][-25:]
    if arm == "rag-25-mmr":
        return mmr(target, pool, world["similar"], 25)
    if arm == "rag-25-labels":
        if draft is None:
            return pool[:25][::-1]
        text = labels_of(draft)
        score = {u: (1 - LABEL_WEIGHT) * world["similar"](target, u)
                 + LABEL_WEIGHT * world["label_similar"](text, labels_of(world["record"](u)["gold"])) for u in pool}
        chosen = sorted(pool, key=lambda u: (-round(score[u], 9), pool.index(u)))[:25]
        return sorted(chosen, key=lambda u: (round(score[u], 9), u))
    raise KeyError(arm)


def prompt_of(world: Mapping[str, Any], target: str, ids: Sequence[str]) -> str:
    """Today's prompt for a target with given examples.

    Args: world: as above; target: the variable; ids: the examples, in order.
    Returns: the prompt.
    Raises: KeyError for an unknown example.
    Side Effects: none. Deterministic.
    """
    inputs = dict(world["base_inputs"](target), examples=[world["record"](u) for u in ids])
    return dr.fill_prompt(inputs, world["definition"](target))


# --- one unit --------------------------------------------------------------------------------

async def unit(ctx: Mapping[str, Any], world: Mapping[str, Any], repetition: int, target: str,
               transport: orclient.Transport, guard: orclient.Guard) -> list[dict[str, Any]]:
    """Answer every arm for one variable in one repetition.

    Args: ctx: from `load` (for the schema); world: what targets may be shown; repetition: its
        number; target: the variable; transport, guard: from `orclient`.
    Returns: one record per arm, in the order of `ARMS`.
    Raises: draft_rag.Failed when any call fails, is refused, is answered by another host, or a
        stop rule has fired.
    Side Effects: provider calls (10, and one more when the retry is needed).
    """
    started_at = ft.now()
    definition = world["definition"](target)
    cost: dict[str, Counter] = {arm: Counter() for arm in ARMS}

    async def call(arm: str, messages: list[dict[str, str]]) -> str:
        """Send one request with the study's settings and keep its cost under the arm.

        Args: arm: the arm; messages: the chat.
        Returns: the answer text.
        Raises: draft_rag.Failed on any failure, and before sending when a stop rule has fired.
        Side Effects: one provider call; updates the guard and the costs.
        """
        if guard.reason:
            raise dr.Failed(f"stopped: {guard.reason}")
        result = await transport(dict(dr.FILL_REQUEST, messages=messages))
        if result.get("refused"):
            guard.stop(str(result.get("error") or "spend cap reached"))
            raise dr.Failed(str(result.get("error")))
        reasoning = result.get("reasoning") or ""
        guard.record({"ok": bool(result.get("ok")), "status": result.get("status"), "error": result.get("error"),
                      "host": result.get("host"), "reasoning_chars": len(reasoning),
                      "reasoning_tokens": result.get("reasoning_tokens")})
        if not result.get("ok"):
            raise dr.Failed(str(result.get("error")))
        if result.get("host") != guard.host["name"]:
            raise dr.Failed(f"answered by {result.get('host')}")
        spent = cost[arm]
        spent["calls"] += 1
        spent["prompt_tokens"] += result.get("prompt_tokens") or 0
        spent["completion_tokens"] += result.get("completion_tokens") or 0
        spent["reasoning_chars"] += len(reasoning)
        spent["reasoning_tokens"] += result.get("reasoning_tokens") or 0
        spent["cost_micro_usd"] += round((result.get("cost_usd") or 0.0) * 1e6)
        return result.get("answer") or ""

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

    base_prompt = dr.fill_prompt(world["base_inputs"](target), definition)
    draft = await call("base", [{"role": "user", "content": base_prompt}])
    draft_prediction, _ = dr.check(ctx, draft, {})
    out = {"base": record("base", answer=draft, prompt_sha256=ft.sha256(base_prompt),
                          valid=draft_prediction is not None)}
    answers: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for arm in CONFIGS:
        if arm == "rag-25+retry":
            continue
        ids = examples(world, arm, target, draft_prediction)
        prompt = prompt_of(world, target, ids)
        answer = await call(arm, [{"role": "user", "content": prompt}])
        answers[arm], prompts[arm] = answer, prompt
        extra = {"draft_sha256": ft.sha256(draft), "draft_valid": draft_prediction is not None} \
            if arm == "rag-25-labels" else {}
        out[arm] = record(arm, answer=answer, prompt_sha256=ft.sha256(prompt), examples=ids, **extra)
    first = answers["rag-25"]
    _, complaints = dr.check(ctx, first, {})
    final, repaired, repair_valid = first, False, None
    if complaints:
        second = await call("rag-25+retry", pipeline.repair_messages(prompts["rag-25"], first, complaints))
        repaired = True
        repair_valid = dr.check(ctx, second, {})[0] is not None
        if repair_valid:
            final = second
    out["rag-25+retry"] = record("rag-25+retry", answer=final, prompt_sha256=ft.sha256(prompts["rag-25"]),
                                 examples=out["rag-25"]["examples"], first_sha256=ft.sha256(first),
                                 first_complaints=complaints, repaired=repaired, repair_valid=repair_valid)
    unit_cost = sum(spent["cost_micro_usd"] for spent in cost.values()) / 1e6
    out["base"].update(unit_cost_usd=unit_cost, unit_calls=sum(spent["calls"] for spent in cost.values()))
    return [out[arm] for arm in ARMS]


def complete_units(path: Path) -> set[tuple[int, str]]:
    """The units whose eleven arms are all stored.

    Args: path: the calls file.
    Returns: {(repetition, variable)}.
    Raises: OSError when the file exists but cannot be read.
    Side Effects: reads the file.
    """
    arms: dict[tuple[int, str], set[str]] = {}
    for key, item in engine.latest(path).items():
        if item.get("ok"):
            arms.setdefault((key[1], key[2]), set()).add(key[0])
    return {unit_key for unit_key, have in arms.items() if have == set(ARMS)}


async def run(ctx: Mapping[str, Any], world: Mapping[str, Any], units: Sequence[tuple[int, str]],
              transport: orclient.Transport, guard: orclient.Guard, calls_path: Path, failed_path: Path,
              concurrency: int = CONCURRENCY, max_units: int | None = None) -> dict[str, Any]:
    """Answer every unit not yet stored; store a unit's records only when all its calls succeeded.

    Args: ctx: from `load`; world: what targets may be shown; units: (repetition, variable)
        pairs; transport, guard: from `orclient`; calls_path, failed_path: the two files;
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
            except dr.Failed as error:
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


# --- scoring, the choice by fold, and comparison --------------------------------------------------

def gain(counts: Mapping[str, Mapping[str, Sequence[float]]], first: str, second: str, ids: Sequence[str],
         draws: int = DRAWS, seed: int = SEED) -> dict[str, float]:
    """One arm against another over some variables, the variables resampled.

    Args: counts: {arm: {variable: [TP, FP, FN]}}; first, second: arms; ids: variables; draws,
        seed: of the resampling; every comparison draws the same resamples.
    Returns: {"gain", "low", "high" (95%), "strict_low", "strict_high" (the level divided by
        `FAMILY`), "share_positive"}.
    Raises: KeyError when an arm lacks a variable.
    Side Effects: none. Deterministic for a given seed.
    """
    import random

    rng = random.Random(seed)
    ids = sorted(ids)
    drawn = sorted(analysis.pooled_gain(counts[first], counts[second], rng.choices(ids, k=len(ids)))
                   for _ in range(draws))
    return {"gain": analysis.pooled_gain(counts[first], counts[second], ids),
            "low": drawn[int(0.025 * draws)], "high": drawn[int(0.975 * draws) - 1],
            "strict_low": drawn[int(STRICT / 2 * draws)], "strict_high": drawn[int((1 - STRICT / 2) * draws) - 1],
            "share_positive": sum(g > 0 for g in drawn) / draws}


def exact_counts(scores: Mapping[engine.Job, Mapping[str, Any]]) -> dict[str, dict[str, list[Fraction]]]:
    """Each variable's exact Close counts summed over its repetitions, per arm.

    Args: scores: score rows by (arm, repetition, variable).
    Returns: {arm: {variable: [TP, FP, FN]}} as fractions.
    Raises: KeyError when a count is missing.
    Side Effects: none. Deterministic.
    """
    out: dict[str, dict[str, list[Fraction]]] = defaultdict(lambda: defaultdict(lambda: [Fraction(0)] * 3))
    for (arm, _, variable_id), row in scores.items():
        add = analysis._counts(row["evaluation"]["close"]["totals"])
        out[arm][variable_id] = [a + b for a, b in zip(out[arm][variable_id], add, strict=True)]
    return out


def pooled_f1(counts: Mapping[str, Sequence[Any]], ids: Sequence[str]) -> Fraction:
    """Micro F1 over some variables, exactly, so two settings with equal counts are equal.

    Args: counts: {variable: [TP, FP, FN]} summed over the repetitions; ids: the variables.
    Returns: 2TP / (2TP + FP + FN) as a fraction; 0 without a true positive.
    Raises: KeyError when a variable is missing.
    Side Effects: none. Deterministic.
    """
    tp, fp, fn = (sum((Fraction(counts[v][k]) for v in ids), Fraction(0)) for k in range(3))
    return 2 * tp / (2 * tp + fp + fn) if tp else Fraction(0)


def choose(ctx: Mapping[str, Any], counts: Mapping[str, Mapping[str, Sequence[Any]]]) -> dict[int, str]:
    """For each fold, the setting with the best Close F1 on the variables of the other five.

    Args: ctx: from `load`; counts: {arm: {variable: [TP, FP, FN]}} over the repetitions used,
        exact (from `exact_counts`).
    Returns: {fold: arm}; among equally good settings the first in the order of `CONFIGS`.
    Raises: KeyError when a setting lacks a variable.
    Side Effects: none. Deterministic.
    """
    out = {}
    for fold in range(scoreboard.FOLDS):
        others = [v for v in ctx["ids"] if ctx["fold_of"][v] != fold]
        best, best_value = None, None
        for arm in CONFIGS:
            value = pooled_f1(counts[arm], others)
            if best is None or value > best_value:
                best, best_value = arm, value
        out[fold] = best
    return out


def unit_integrity(calls: Mapping[engine.Job, Mapping[str, Any]], repetitions: Sequence[int],
                   ids: Sequence[str]) -> dict[str, Any]:
    """Whether the eleven records of every unit belong together.

    Args: calls: from `engine.latest`; repetitions, ids: the units to check.
    Returns: {"units", "mismatched", "examples"}: a unit is mismatched when its records do not
        share one start time, when `rag-25-labels` was not built on the unit's own draft, or when
        `rag-25+retry` did not start from the unit's own `rag-25` answer.
    Raises: KeyError when a record is missing.
    Side Effects: none. Deterministic.
    """
    mismatched = []
    for repetition in repetitions:
        for variable_id in ids:
            records = {arm: calls[(arm, repetition, variable_id)] for arm in ARMS}
            problems = []
            if len({item.get("started_at") for item in records.values()}) != 1:
                problems.append("start times differ")
            if records["rag-25-labels"].get("draft_sha256") != ft.sha256(records["base"]["answer"]):
                problems.append("rag-25-labels: not its own draft")
            if records["rag-25+retry"].get("first_sha256") != ft.sha256(records["rag-25"]["answer"]):
                problems.append("rag-25+retry: not its own first answer")
            if problems:
                mismatched.append({"repetition": repetition, "variable_id": variable_id, "problems": problems})
    return {"units": len(repetitions) * len(ids), "mismatched": len(mismatched), "examples": mismatched[:5]}


def summarise(ctx: Mapping[str, Any], calls: Mapping[engine.Job, Mapping[str, Any]],
              scores: Mapping[engine.Job, Mapping[str, Any]], draws: int = DRAWS) -> dict[str, Any]:
    """Every arm's score, the setting chosen fold by fold, and the comparisons of the plan.

    Only repetitions in which every arm was stored and scored for every variable enter a figure;
    "complete" is true only when all ten did and every unit's records belong together.

    Args: ctx: from `load`; calls: from `engine.latest`; scores: from `engine.score`; draws:
        resamples of the variables.
    Returns: {"generated_at", "variables", "repetitions_planned", "repetitions_used", "complete",
        "unit_integrity", "arms", "chosen": {"by_fold", "f1_mean", "f1_per_repetition",
        "by_group"}, "comparisons" (the family of two), "against_base", "against_reference",
        "without_relatives", "retry", "spent_usd"}.
    Raises: nothing; without a complete repetition only the header is returned.
    Side Effects: none.
    """
    ids = ctx["ids"]
    used = [r for r in range(1, REPETITIONS + 1)
            if all((arm, r, v) in scores and calls.get((arm, r, v), {}).get("ok") for arm in ARMS for v in ids)]
    out: dict[str, Any] = {"generated_at": ft.now(), "variables": len(ids), "repetitions_planned": REPETITIONS,
                           "repetitions_used": used, "complete": len(used) == REPETITIONS, "arms": {},
                           "chosen": {}, "comparisons": {}, "against_base": {}, "against_reference": {},
                           "without_relatives": {}, "retry": {},
                           "spent_usd": sum(r.get("unit_cost_usd") or 0.0 for k, r in calls.items() if k[0] == BASELINE)}
    if not used:
        return out
    out["unit_integrity"] = unit_integrity(calls, used, ids)
    out["complete"] = out["complete"] and out["unit_integrity"]["mismatched"] == 0
    by_group = {group: [v for v in ids if ctx["group_of"][v] == group] for group in scoreboard.GROUPS}

    def series(pick: Callable[[str], str], members: Sequence[str]) -> list[float]:
        """Close F1 per repetition when each variable is answered by the arm `pick` names.

        Args: pick: arm for a variable; members: the variables.
        Returns: one F1 per used repetition.
        Raises: KeyError when a score is missing.
        Side Effects: none.
        """
        return [_aggregate([scores[(pick(v), r, v)]["evaluation"] for v in members], list(members))["f1"] for r in used]

    for arm in ARMS:
        every = series(lambda v, arm=arm: arm, ids)
        rows = [calls[(arm, r, v)] for r in used for v in ids]
        out["arms"][arm] = {
            "f1_mean": statistics.mean(every), "f1_sd": statistics.stdev(every) if len(every) > 1 else None,
            "f1_per_repetition": every,
            "by_group": {g: statistics.mean(series(lambda v, arm=arm: arm, members)) for g, members in by_group.items()},
            "invalid": sum(1 for r in used for v in ids if not scores[(arm, r, v)]["valid_prediction"]),
            "calls": sum(item.get("calls") or 0 for item in rows),
            "prompt_tokens_mean": ft._mean(item.get("prompt_tokens") for item in rows if item.get("calls")),
            "cost_usd": sum(item.get("cost_usd") or 0.0 for item in rows)}
    kept = {key: row for key, row in scores.items() if key[0] in ARMS and key[1] in used}
    counts = analysis.variable_counts(kept)
    choice = choose(ctx, exact_counts(kept))
    pick = {v: choice[ctx["fold_of"][v]] for v in ids}
    counts["chosen"] = {v: counts[pick[v]][v] for v in ids}
    chosen_series = series(lambda v: pick[v], ids)
    out["chosen"] = {"by_fold": {str(fold): arm for fold, arm in choice.items()},
                     "f1_mean": statistics.mean(chosen_series), "f1_per_repetition": chosen_series,
                     "by_group": {g: statistics.mean(series(lambda v: pick[v], members)) for g, members in by_group.items()}}
    for second in (BASELINE, REFERENCE):
        row = dict(gain(counts, "chosen", second, ids, draws), first="chosen", second=second)
        if len(used) > 1:
            row["welch"] = ft.welch(chosen_series, out["arms"][second]["f1_per_repetition"])
        out["comparisons"][f"chosen vs {second}"] = row
    for arm in CONFIGS:
        out["against_base"][arm] = gain(counts, arm, BASELINE, ids, draws)
        if arm != REFERENCE:
            out["against_reference"][arm] = gain(counts, arm, REFERENCE, ids, draws)
    without = [v for v in ids if v not in set(ctx["relatives"])]
    for arm in ("chosen", *CONFIGS):
        out["without_relatives"][arm] = dict(gain(counts, arm, BASELINE, without, draws), variables=len(without))
    retry = [calls[("rag-25+retry", r, v)] for r in used for v in ids]
    out["retry"] = {"answers": len(retry), "repaired": sum(bool(item.get("repaired")) for item in retry),
                    "repair_valid": sum(bool(item.get("repair_valid")) for item in retry)}
    return out


# --- the dev check on invented variables ---------------------------------------------------------

def dev_world(ctx: Mapping[str, Any], similar: Callable[[str, str], float]) -> tuple[dict[str, Any], list[str]]:
    """The invented variables of `knowledge.py`, in two halves that serve as each other's pool.

    Args: ctx: from `load`; similar: a similarity of two texts (also used for label texts).
    Returns: (a world in the shape of `corpus_world`, the invented ids in order). Today's prompt
        keeps the study's 25 examples; everything retrieved comes from the other half.
    Raises: KeyError when an invented variable has no decomposition.
    Side Effects: calls `similar` for every pair of invented definitions.
    """
    items = kn.invented(ctx)
    order = sorted(items, key=lambda name: int(name.split("-")[1]))
    half = {name: position % 2 for position, name in enumerate(order)}
    gold = {name: dict(json.loads(items[name]["demo"]["gold"])) for name in order}
    table = similarity_table(order, lambda name: items[name]["definition"], similar)
    ranked = order_of(table)

    def record(u: str) -> Mapping[str, Any]:
        """An invented variable as an example, or a corpus record for today's examples.

        Args: u: an invented id or a corpus id.
        Returns: {"definition", "gold"} (and the id).
        Raises: KeyError for an unknown id.
        Side Effects: none.
        """
        if u in items:
            return {"id": u, "definition": items[u]["definition"], "gold": gold[u]}
        return ctx["records"][u]

    world = {"definition": lambda v: items[v]["definition"], "base_inputs": lambda v: ctx["inputs"],
             "pool": lambda v: [u for u in ranked[v] if half[u] != half[v]],
             "fixed": lambda v: [r["variable_id"] for r in ctx["inputs"]["examples"]],
             "record": record, "similar": lambda a, b: table[a][b], "label_similar": similar}
    return world, order


# --- fingerprints and the command line ----------------------------------------------------------

def fingerprints(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Hash what is fixed before the run: code, data, similarities and the prompts of nine arms.

    The prompts of `rag-25-labels` depend on the run's drafts; their hashes are stored with each
    answer. `rag-25+retry` sends the `rag-25` prompt.

    Args: ctx: from `load`.
    Returns: hex digests (first 16 characters) and counts.
    Raises: OSError when a file cannot be read.
    Side Effects: reads files. Deterministic.
    """
    world = corpus_world(ctx, None)
    out: dict[str, Any] = {name: ceiling.file_sha256(HERE / name)[:16] for name in (
        "rag_tuning.py", "draft_rag.py", "knowledge.py", "orclient.py", "scoreboard.py", "engine.py", "groups.py",
        "settings.py", "pipeline.py")}
    out.update({f"data/{name}": ceiling.file_sha256(rulebook.DATA / name)[:16] for name in ("route_key.json",)})
    out["similarities_file"] = ceiling.file_sha256(SIMILARITIES)[:16] if SIMILARITIES.exists() else None
    out["population_sha256"] = ft.sha256("\n".join(ctx["ids"]))[:16]
    out["relatives"] = len(ctx["relatives"])
    out["prompts"] = {}
    for arm in ARMS:
        if arm in ("rag-25-labels", "rag-25+retry"):
            continue
        prompts = [dr.fill_prompt(world["base_inputs"](v), world["definition"](v)) if arm == BASELINE
                   else prompt_of(world, v, examples(world, arm, v)) for v in ctx["ids"]]
        out["prompts"][arm] = {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                               "chars_mean": round(sum(map(len, prompts)) / len(prompts), 1)}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status: 0 on success, 1 when a stop rule ended a run or the stored run is
        not complete.
    Raises: SystemExit on bad arguments, a missing key or missing similarities.
    Side Effects: --similarities writes the table; --dev and --run call the provider; --score
        writes the scores and the summary.
    """
    parser = argparse.ArgumentParser(description="Tuning retrieval, chosen fold by fold (stage 5).")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--similarities", "--plan-only", "--dev", "--run", "--score"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--max-units", type=int, default=None, help="answer at most this many new units")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args(argv)
    if args.similarities:
        stored = write_similarities()
        print(json.dumps({"definitions_sha256": stored["definitions_sha256"][:16], "variables": len(stored["table"])}))
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
            print(f"NOT COMPLETE: repetitions used {summary['repetitions_used']}, "
                  f"integrity {summary.get('unit_integrity')}")
        for arm, cell in summary["arms"].items():
            print(f"{arm:14} F1 {cell['f1_mean']:.4f} (SD {cell['f1_sd'] or 0:.4f})  "
                  + "  ".join(f"{g} {value:.3f}" for g, value in cell["by_group"].items())
                  + f"  invalid {cell['invalid']}  tokens {cell['prompt_tokens_mean'] or 0:.0f}  ${cell['cost_usd']:.2f}")
        chosen = summary.get("chosen", {})
        if chosen:
            print(f"chosen by fold {chosen['by_fold']}  F1 {chosen['f1_mean']:.4f}  "
                  + "  ".join(f"{g} {value:.3f}" for g, value in chosen["by_group"].items()))
        for name, row in summary["comparisons"].items():
            print(f"{name:22} {row['gain']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}] strict "
                  f"[{row['strict_low']:+.4f}, {row['strict_high']:+.4f}]")
        for arm, row in summary["against_reference"].items():
            print(f"{arm:14} vs rag-25 {row['gain']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}]")
        for arm, row in summary["without_relatives"].items():
            print(f"without relatives {arm:14} vs base {row['gain']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}]")
        print("retry:", summary["retry"], f"| spent ${summary['spent_usd']:.2f}")
        return 0 if summary["complete"] else 1

    async def go() -> int:
        """Open the transport, do the asked work, and always close the client.

        Args: none.
        Returns: the exit status: 1 when a stop rule ended the work.
        Raises: SystemExit when the key is missing or another process holds the ledger.
        Side Effects: provider calls, the ledger and output files.
        """
        purpose = "tuning-dev" if args.dev else "tuning"
        spend = orclient.Spend(purpose=purpose, allowance=DEV_ALLOWANCE_USD if args.dev else ALLOWANCE_USD)
        guard = orclient.Guard(spend)
        similar, _ = dr.local_similarity()
        transport, client = orclient.open_transport(spend, purpose, concurrency=args.concurrency)
        try:
            if args.dev:
                world, order = dev_world(ctx, similar)
                result = await run(ctx, world, [(1, name) for name in order], transport, guard, DEV_CALLS, DEV_FAILED,
                                   concurrency=min(8, args.concurrency))
            else:
                result = await run(ctx, corpus_world(ctx, similar), dr.schedule(ctx["ids"], REPETITIONS, SCHEDULE_SEED),
                                   transport, guard, CALLS, FAILED, concurrency=args.concurrency, max_units=args.max_units)
            print(json.dumps(result, indent=2))
            return 1 if result["stopped"] else 0
        finally:
            await orclient.close_transport(transport, client)

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
