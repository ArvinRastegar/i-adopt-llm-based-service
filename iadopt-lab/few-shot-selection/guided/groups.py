#!/usr/bin/env python
"""Step 5: the groups that had no result, the held-out variables and the example variables.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/GUIDED_DECOMPOSITION_20261119.md` (private repository).

Steps 1 to 4 scored the 53 practice variables only. Arvin then asked for every empty cell of the
results table, so this module runs the same setups on the other two groups of the corpus:

- `held-out`: the 24 variables that had been kept for a final exam. Their prompts are built the
  way the practice prompts were.
- `examples`: the 25 variables that are the few-shot examples of every prompt. A variable cannot
  be shown its own answer, so each one is scored leave-one-out: its fill prompt holds the other 24
  examples, and its walks are shown demonstrations without it.

Setups ("arms"): today's prompt; guidance from the experts' tags (`guided@key/full`, the `full`
arm of step 1); and the five workflows of steps 3 and 4 in which the model finds the patterns.
`--table` pools the groups into "practice + held-out" and "all 102" from the stored records.

    ./groups.py --group held-out --plan-only
    ./groups.py --group held-out --walk steps        # also: direct, direct-all
    ./groups.py --group held-out --marks
    ./groups.py --group held-out --run
    ./groups.py --group held-out --score
    ./groups.py --table
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import engine  # noqa: E402
import hints  # noqa: E402
import settings  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402
from harness import _aggregate  # noqa: E402

pipeline = settings.pipeline  # this folder's pipeline.py, loaded by path; see settings.py for why

GROUPS = ("held-out", "examples")
BASELINE = "base"
KEY_ARM = "guided@key/full"
FOUND_ARMS = ("guided+check@direct/full", "guided+check@direct-all/full", "guided+check@mixed-all/full",
              "guided+check@direct-all/form+types", "guided+check@mixed/full")
ARMS = (BASELINE, KEY_ARM, *FOUND_ARMS)
REPETITIONS = 15
SCHEDULE_SEED = 20261122
WALK_STYLES = ("direct", "steps", "direct-all")
# Example terms in the rulebook's pattern texts that name one corpus variable. They are taken out
# of every prompt of this step: "Body Mass Index" is a held-out variable, and the independent
# review of 2026-11-19 found both terms in practice prompts of steps 1 to 4.
REMOVED_TERMS = ((132, "design", " (e.g. body mass index)"), (129, "design", " such as actinic flux"))
# Where each arm's practice-set answers are stored: (run name, arm name in that run).
PRACTICE_SOURCES = {BASELINE: (("ceiling", "base"), ("e2e", "base"), ("tune", "base")),
                    KEY_ARM: (("ceiling", "full"),),
                    "guided+check@direct/full": (("e2e", "guided+check"),),
                    "guided+check@direct-all/full": (("tune", "guided+check@direct-all/full"),),
                    "guided+check@mixed-all/full": (("tune", "guided+check@mixed-all/full"),),
                    "guided+check@direct-all/form+types": (("tune", "guided+check@direct-all/form+types"),),
                    "guided+check@mixed/full": (("tune", "guided+check@mixed/full"),)}
ROWS = (("examples", ("examples",)), ("practice", ("practice",)), ("held-out", ("held-out",)),
        ("practice + held-out", ("practice", "held-out")), ("all", ("examples", "practice", "held-out")))
Transport = Callable[[Mapping[str, Any]], Awaitable[dict]]


def clean_patterns(patterns: Mapping[str, Any]) -> dict[str, Any]:
    """The pattern table without the example terms that name a corpus variable.

    Args: patterns: the built pattern table.
    Returns: a deep copy in which each term of `REMOVED_TERMS` is cut out of its text.
    Raises: ValueError when a term is not found exactly once, so an upstream change is noticed.
    Side Effects: none. Deterministic.
    """
    out = copy.deepcopy(dict(patterns))
    for number, field, term in REMOVED_TERMS:
        text = out[str(number)].get(field) or ""
        if text.count(term) != 1:
            raise ValueError(f"pattern {number} {field} does not hold {term!r} exactly once")
        out[str(number)][field] = text.replace(term, "")
    return out


def load() -> dict[str, Any]:
    """Everything this step needs, with the cleaned pattern texts. No network, no secrets.

    Args: none.
    Returns: the context of `walk.load` with "patterns" replaced by `clean_patterns`.
    Raises: as `walk.load` and `clean_patterns`.
    Side Effects: reads local files.
    """
    ctx = walk.load()
    ctx["patterns"] = clean_patterns(ctx["patterns"])
    return ctx


def targets(ctx: Mapping[str, Any], group: str) -> list[str]:
    """The variables of one group.

    Args: ctx: from `load`; group: "held-out", "examples" or "practice".
    Returns: sorted variable ids.
    Raises: KeyError for an unknown group.
    Side Effects: none.
    """
    return {"held-out": sorted(ctx["inputs"]["targets"]), "examples": sorted(ft.EXAMPLE_IDS),
            "practice": list(ctx["practice"])}[group]


def shown_examples(target: str) -> list[str]:
    """The example variables a prompt for one target may show, in the study's order.

    Args: target: the variable being decomposed.
    Returns: the 25 example ids, without the target when it is one of them.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return [v for v in ft.EXAMPLE_IDS if v != target]


def context_for(ctx: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    """The context whose fill prompt never shows the target its own answer.

    Args: ctx: from `load`; target: the variable being decomposed.
    Returns: `ctx` itself for a target that is no example; else a copy whose examples leave the
        target out (24 examples).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if target not in ft.EXAMPLE_IDS:
        return ctx
    kept = [row for row in ctx["inputs"]["examples"] if row["variable_id"] != target]
    return dict(ctx, inputs=dict(ctx["inputs"], examples=kept))


def demos_for(ctx: Mapping[str, Any], target: str) -> dict[str, list[tuple[str, str]]]:
    """The step-by-step walk's demonstrations for one target: `walk.demo_index` without the target.

    Args: ctx: from `load`; target: the variable being walked.
    Returns: {question id: (definition, right option) pairs}, drawn from `shown_examples(target)`
        in the order `walk.demo_index` uses; equal to it for a target that is no example.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    shown = sorted(shown_examples(target))
    order = {ctx["records"][v]["definition"]: position for position, v in enumerate(shown)}
    groups: dict[str, dict[str, list[str]]] = {}
    for variable_id in shown:
        definition = ctx["records"][variable_id]["definition"]
        for qid, answer in walk.right_answers(ctx, ctx["key"][variable_id]).items():
            groups.setdefault(qid, {}).setdefault(answer, []).append(definition)
    index: dict[str, list[tuple[str, str]]] = {}
    for qid, by_answer in groups.items():
        pairs = [(text, answer) for answer, texts in by_answer.items() for text in texts]
        index[qid] = sorted(pairs, key=lambda pair: order[pair[0]])[:walk.DEMOS_PER_QUESTION]
    return index


def direct_prompt(ctx: Mapping[str, Any], target: str) -> str:
    """The one-call walk prompt for one target: `walk.direct_prompt` without the target as an example.

    Args: ctx: from `load`; target: the variable being walked.
    Returns: the prompt; byte for byte `walk.direct_prompt` for a target that is no example.
    Raises: ValueError when the base prompt no longer has a single EXAMPLES section.
    Side Effects: none. Deterministic.
    """
    definition = ctx["records"][target]["definition"]
    base = walk.direct_prompt(ctx, definition)
    if base.count("\nEXAMPLES\n") != 1:
        raise ValueError("the direct prompt has no single EXAMPLES section")
    lines = [base.split("\nEXAMPLES\n")[0], "EXAMPLES"]
    for shown in sorted(shown_examples(target)):
        lines += [f"Definition: {ctx['records'][shown]['definition']}",
                  "Patterns: " + json.dumps(walk.direct_answer(ctx, ctx["key"][shown]), ensure_ascii=False), ""]
    lines += [f"Definition: {definition}", "Patterns:"]
    return "\n".join(lines)


def walk_path(group: str, style: str) -> Path:
    """Where one group's walks of one style are stored.

    Args: group: "held-out" or "examples"; style: "direct", "steps" or "direct-all".
    Returns: the JSON-lines path under the ignored output folder.
    Raises: nothing.
    Side Effects: none.
    """
    return walk.results_path(style, f"route-{group}")


async def walk_group(ctx: Mapping[str, Any], style: str, ids: Sequence[str], transport: Transport, path: Path,
                     concurrency: int = walk.CONCURRENCY) -> dict[str, Any]:
    """Walk variables with `direct` or `steps`, each with demonstrations that leave it out.

    Args: ctx: from `load`; style: "direct" or "steps"; ids: the targets; transport: sends one
        request; path: the results file; concurrency: variables walked at once.
    Returns: {"walked", "skipped", "stopped"}.
    Raises: ValueError for another style.
    Side Effects: provider calls; appends one row per variable to `path`, in the row format of
        `walk.run`. Resumable per variable. A refused credential or any reasoning text stops it.
    """
    if style not in walk.STYLES:
        raise ValueError(f"unknown style: {style}")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["id"] for row in ft.read_jsonl(path) if row.get("complete")}
    semaphore, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    state: dict[str, Any] = {"walked": 0, "skipped": 0, "stopped": None}

    async def one(target: str) -> None:
        """Walk one variable and store its row.

        Args: target: the variable.
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
            definition = ctx["records"][target]["definition"]
            if style == "steps":
                result = await walk.walk_steps(ctx, definition, walk.make_ask(ctx, transport, demos_for(ctx, target), log))
            else:
                prompt = direct_prompt(ctx, target)
                reply = await transport(dict(walk.DIRECT_REQUEST, model=engine.MODEL,
                                             messages=[{"role": "user", "content": prompt}]))
                log.append({"question": "direct", "ok": bool(reply.get("ok")), "status": reply.get("status"),
                            "error": reply.get("error"), "raw": reply.get("answer"),
                            "reasoning_chars": len(reply.get("reasoning") or ""),
                            "prompt_tokens": reply.get("prompt_tokens"),
                            "completion_tokens": reply.get("completion_tokens"),
                            "latency_seconds": round(reply.get("latency") or 0.0, 3),
                            "prompt_sha256": ft.sha256(prompt)})
                facts = walk.facts_from_direct(ctx, reply.get("answer") or "") if reply.get("ok") else None
                result = {"facts": facts, "trace": {}, "confidence": {}}
            if any(c.get("status") in (401, 403) for c in log):
                state["stopped"] = "credential refused"
            elif any(c.get("reasoning_chars") for c in log):
                state["stopped"] = "answers carry reasoning text despite the switch"
            row = {"id": target, "style": style, "started_at": started, "facts": result["facts"],
                   "trace": result["trace"], "confidence": result["confidence"], "calls": log,
                   "complete": all(c["ok"] for c in log), "shown": len(shown_examples(target))}
            state["walked"] += 1
            async with lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(v) for v in ids if v not in done))
    return state


def walks_named(ctx: Mapping[str, Any], name: str, group: str) -> dict[str, dict[str, Any]]:
    """One source of facts for one group.

    Args: ctx: from `load`; name: "key" (the experts' tags), "direct", "steps", "direct-all",
        "mixed" or "mixed-all" (as in `settings.walks_named`); group: "held-out" or "examples".
    Returns: {variable id: row with "facts"}.
    Raises: KeyError for an unknown name.
    Side Effects: reads stored walk files, except for "key".
    """
    if name == "key":
        return {v: {"id": v, "style": "key", "facts": hints.facts_from_key(ctx["key"][v])}
                for v in targets(ctx, group)}
    stored = {style: walk.stored(walk_path(group, style)) for style in WALK_STYLES}
    return {"direct": stored["direct"], "steps": stored["steps"], "direct-all": stored["direct-all"],
            "mixed": settings.mixed(stored["direct"], stored["steps"]),
            "mixed-all": settings.mixed(stored["direct-all"], stored["steps"])}[name]


def prompt_of(ctx: Mapping[str, Any], arm: str, target: str, walks: Mapping[str, Mapping[str, Any]]) -> str:
    """The first prompt one arm sends for one target.

    Args: ctx: from `load`; arm: an arm name; target: the variable; walks: the arm's facts by
        variable id (unused for the baseline).
    Returns: the prompt; today's prompt when the arm is the baseline or the walk has no facts.
    Raises: KeyError or ValueError for a malformed arm name.
    Side Effects: none. Deterministic.
    """
    own = context_for(ctx, target)
    definition = ctx["records"][target]["definition"]
    spec = settings.parse_arm(arm)
    facts = walks.get(target, {}).get("facts") if spec else None
    if facts is None:
        return ft.prompt_for(own["inputs"], "B", definition)
    return pipeline.guided_prompt(own, definition, facts, spec[2])


async def run(ctx: Mapping[str, Any], group: str, arms: Sequence[str], transport: Transport, calls_path: Path,
              concurrency: int = engine.CONCURRENCY, repetitions: int = REPETITIONS) -> dict[str, Any]:
    """Answer every arm for every variable of a group and repetition, one record per answer.

    Args: ctx: from `load`; group: "held-out" or "examples"; arms: arm names; transport: sends
        one request; calls_path: the records file; concurrency: workers; repetitions: runs per arm.
    Returns: {"issued", "skipped", "stopped", "list_price_usd"}.
    Raises: SystemExit when an arm lacks a stored walk for a variable.
    Side Effects: provider calls; appends to `calls_path`. Resumable. Each target is answered in
        its own leave-one-out context.
    """
    ids = targets(ctx, group)
    specs = {arm: settings.parse_arm(arm) for arm in arms}
    loaded = {spec[1]: walks_named(ctx, spec[1], group) for spec in specs.values() if spec}
    for name, rows in loaded.items():
        missing = [v for v in ids if v not in rows]
        if missing:
            raise SystemExit(f"{len(missing)} variables have no stored {name} walk for {group}")
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    guard = engine.Guard()
    answered = {key for key, record in engine.latest(calls_path).items() if record.get("ok")}
    queue: asyncio.Queue = asyncio.Queue()
    for job in engine.schedule(list(arms), ids, repetitions, SCHEDULE_SEED):
        if job not in answered:
            queue.put_nowait(job)
    lock, counts = asyncio.Lock(), Counter()

    async def worker() -> None:
        """Take jobs until the queue is empty, honouring the stop rules.

        Args: none.
        Returns: nothing.
        Raises: nothing.
        Side Effects: provider calls; appends records under the lock.
        """
        while True:
            try:
                arm, repetition, variable_id = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if guard.reason:
                counts["skipped"] += 1
                continue
            own, spec = context_for(ctx, variable_id), specs[arm]
            if spec is None:
                record = await pipeline.attempt(own, (arm, repetition, variable_id), transport, frozenset(), {}, guard)
            else:
                record = await pipeline.attempt(own, (spec[0], repetition, variable_id), transport, spec[2],
                                                loaded[spec[1]], guard)
                record["arm"] = arm
            counts["issued"] += 1
            async with lock:
                with calls_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return {"issued": counts["issued"], "skipped": counts["skipped"], "stopped": guard.reason,
            "list_price_usd": round(guard.spend, 4)}


def paths(group: str) -> dict[str, Path]:
    """The files of one group's comparison.

    Args: group: "held-out" or "examples".
    Returns: {"calls", "scores", "summary"} paths under the ignored output folder.
    Raises: nothing.
    Side Effects: none.
    """
    return pipeline.paths(f"groups-{group}")


def _digest(prompts: Any) -> str:
    """One digest over a sequence of prompts.

    Args: prompts: an iterable of prompt texts, in a fixed order.
    Returns: the SHA-256 of the prompts' own digests, one per line.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return ft.sha256("\n".join(ft.sha256(p) for p in prompts))


def fingerprints(ctx: Mapping[str, Any], group: str) -> dict[str, Any]:
    """Hash what this step sends for one group, so the record can freeze it.

    Args: ctx: from `load`; group: "held-out" or "examples".
    Returns: file digests, the walk prompts' digests and, once the walks are stored, one digest
        per arm over its first prompts.
    Raises: OSError when a source file cannot be read.
    Side Effects: reads this folder's source files and the stored walks.
    """
    ids = targets(ctx, group)
    out: dict[str, Any] = {"group": group, "variables": len(ids), "arms": list(ARMS),
                           "answers": len(ARMS) * len(ids) * REPETITIONS,
                           "examples_per_prompt": sorted({len(context_for(ctx, v)["inputs"]["examples"]) for v in ids})}
    for name in ("groups", "pipeline", "settings", "walk", "hints", "engine"):
        out[f"{name}.py"] = ft.sha256((HERE / f"{name}.py").read_text(encoding="utf-8"))
    out["direct_prompts_sha256"] = _digest(direct_prompt(ctx, v) for v in ids)
    out["direct_all_prompts_sha256"] = _digest(settings.direct_prompt_all(ctx, v) for v in ids)
    out["direct_all_shown"] = sorted({len(settings.shown_ids(ctx, v)) for v in ids})
    demos = {v: demos_for(ctx, v) for v in ids}
    out["step_demos_sha256"] = ft.sha256(json.dumps([demos[v] for v in ids], sort_keys=True, ensure_ascii=False))
    for arm in ARMS:
        spec = settings.parse_arm(arm)
        rows = walks_named(ctx, spec[1], group) if spec else {}
        if spec and not all(v in rows for v in ids):
            out.setdefault("arms_without_walks", []).append(arm)
            continue
        out.setdefault("prompts_sha256", {})[arm] = _digest(prompt_of(ctx, arm, v, rows) for v in ids)
    return out


def score(ctx: Mapping[str, Any], group: str) -> dict[str, Any]:
    """Score one group's comparison and write its summary.

    Args: ctx: from `load`; group: "held-out" or "examples".
    Returns: the summary of `engine.summarise` with "behaviour" and "settings" added.
    Raises: OSError when the calls file is missing.
    Side Effects: loads the local similarity model for new answers; writes the scores and summary.
    """
    files = paths(group)
    scores = engine.score(files["calls"], files["scores"], ctx["records"], ctx["inputs"]["schema"])
    calls = engine.latest(files["calls"])
    summary = engine.summarise(list(ARMS), BASELINE, targets(ctx, group), REPETITIONS, calls, scores)
    summary["behaviour"] = pipeline.behaviour(calls)
    summary["settings"] = {"group": group, "arms": list(ARMS)}
    files["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def pooled_f1(series: Sequence[Mapping[int, Sequence[tuple[dict, str]]]]) -> list[float]:
    """Micro Close F1 per repetition over several groups put together.

    Args: series: one mapping per group, {repetition: (evaluation, variable id) pairs}.
    Returns: one F1 per repetition, from the summed counts of all pairs.
    Raises: ValueError when a group lacks a repetition or holds a different number of answers in
        different repetitions, so an incomplete comparison is never pooled silently.
    Side Effects: none. Deterministic.
    """
    repetitions = set(series[0])
    if any(set(s) != repetitions or len({len(pairs) for pairs in s.values()}) != 1 for s in series):
        raise ValueError("the groups do not hold the same complete repetitions")
    out = []
    for repetition in sorted(repetitions):
        pairs = [pair for s in series for pair in s[repetition]]
        out.append(_aggregate([p[0] for p in pairs], [p[1] for p in pairs])["f1"])
    return out


def stored_series(name: str) -> dict[str, dict[int, list[tuple[dict, str]]]]:
    """Every arm's scored answers of one stored comparison, by repetition.

    Args: name: the comparison's file prefix, such as "tune" or "groups-held-out".
    Returns: {arm: {repetition: (evaluation, variable id) pairs}}, one pair per answer.
    Raises: OSError when the scores file cannot be read.
    Side Effects: reads the scores file.
    """
    latest: dict[tuple, dict] = {}
    for row in ft.read_jsonl(engine.OUTPUT / f"{name}-scores.jsonl"):
        latest[tuple(row["key"])] = row     # a re-scored answer replaces its earlier row
    out: dict[str, dict[int, list[tuple[dict, str]]]] = {}
    for (arm, repetition, variable_id), row in latest.items():
        out.setdefault(arm, {}).setdefault(repetition, []).append((row["evaluation"], variable_id))
    return out


def table() -> dict[str, Any]:
    """Close F1 of every arm on every group and on the pooled rows, from the stored records.

    Args: none.
    Returns: {"rows": {row: {"variables", "arms": {arm: {"mean", "sd", "runs"}}}}}; the practice
        part of the baseline is averaged over its three stored runs.
    Raises: OSError when a stored comparison is missing.
    Side Effects: reads the stored score files (several hundred megabytes).
    """
    names = {name for sources in PRACTICE_SOURCES.values() for name, _ in sources} | {f"groups-{g}" for g in GROUPS}
    stored = {name: stored_series(name) for name in sorted(names)}
    sizes = {group: len({v for pairs in next(iter(stored[f"groups-{group}"].values())).values() for _, v in pairs})
             for group in GROUPS}
    sizes["practice"] = len({v for pairs in stored["tune"]["base"].values() for _, v in pairs})
    rows: dict[str, Any] = {}
    for row, groups in ROWS:
        cells = {}
        for arm in ARMS:
            choices = [[stored[name][source] for name, source in PRACTICE_SOURCES[arm]] if group == "practice"
                       else [stored[f"groups-{group}"][arm]] for group in groups]
            runs = [pooled_f1([choice[min(i, len(choice) - 1)] for choice in choices])
                    for i in range(max(len(choice) for choice in choices))]
            means = [statistics.fmean(run) for run in runs]
            cells[arm] = {"mean": statistics.fmean(means), "runs": means,
                          "sd": statistics.stdev(runs[0]) if len(runs) == 1 else None}
        rows[row] = {"variables": sum(sizes[g] for g in groups), "arms": cells}
    return {"generated_at": ft.now(), "repetitions": REPETITIONS, "rows": rows}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode: plan, walk, marks, run, score or table.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status, 0 on success.
    Raises: SystemExit on bad arguments, a missing PSNC key or missing stored walks.
    Side Effects: --walk and --run call the provider; --walk, --run, --score and --table write
        output files.
    """
    parser = argparse.ArgumentParser(description="Held-out and example groups (exploratory).")
    parser.add_argument("--group", choices=GROUPS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--walk", choices=WALK_STYLES)
    mode.add_argument("--marks", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--table", action="store_true")
    args = parser.parse_args(argv)
    if args.table:
        result = table()
        (engine.OUTPUT / "groups-table.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        for row, info in result["rows"].items():
            print(f"{row:20} {info['variables']:3}  " + "  ".join(f"{c['mean']:.4f}" for c in info["arms"].values()))
        return 0
    if not args.group:
        parser.error("--group is required for this mode")
    ctx = load()
    ids = targets(ctx, args.group)
    if args.plan_only:
        print(json.dumps(fingerprints(ctx, args.group), indent=2))
        return 0
    if args.marks:
        for name in ("direct", "steps", "mixed", "direct-all", "mixed-all"):
            rows = walks_named(ctx, name, args.group)
            if all(v in rows for v in ids):
                print(name, json.dumps(walk.report(ctx, rows, ids)))
        return 0
    if args.score:
        summary = score(ctx, args.group)
        for arm, cell in summary["cells"].items():
            how = summary["behaviour"].get(arm, {})
            print(f"{arm:38} F1 {cell['close_f1_official_mean']:.4f} (SD {cell['close_f1_official_sd'] or 0:.4f}) "
                  f"invalid {cell['invalid']:3}  repaired {how.get('repaired', 0):.2f}  fell back "
                  f"{how.get('fell_back', 0):.2f}  calls/answer {how.get('calls_per_answer', 1):.2f}")
        for arm, result in summary["comparisons"].items():
            print(f"{arm:38} vs {BASELINE}: {result['difference']:+.4f} [{result['ci_low']:+.4f}, "
                  f"{result['ci_high']:+.4f}] holm p {result['p_holm']:.2g} -> {result['verdict']}")
        return 0

    async def go() -> int:
        """Open a transport, walk or run, and always close the client.

        Args: none.
        Returns: the exit status.
        Raises: SystemExit when the PSNC key is missing.
        Side Effects: provider calls and output files.
        """
        if args.walk:
            transport, client = walk.chat_transport()
            try:
                path = walk_path(args.group, args.walk)
                if args.walk == "direct-all":
                    state = await settings.walk_direct_all(ctx, ids, transport, path)
                else:
                    state = await walk_group(ctx, args.walk, ids, transport, path)
                print(args.walk, json.dumps(state))
            finally:
                await client.aclose()
            return 0
        transport, client = ft.psnc_transport()
        try:
            print(json.dumps(await run(ctx, args.group, ARMS, transport, paths(args.group)["calls"]), indent=2))
        finally:
            await client.aclose()
        return 0

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
