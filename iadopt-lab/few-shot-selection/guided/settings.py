#!/usr/bin/env python
"""Step 4, settings: better walks, and the final exam on the held-out variables.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/GUIDED_DECOMPOSITION_20261119.md` (private repository).

The route test showed that the walk, not the guidance, limits the workflow. This module tries
walks that should decide better, marks them against the expert answer key, and compares the
resulting workflows end to end on Qwen3.8-27B:

- `direct-all`: the one-call walk of `walk.py`, but shown every tagged variable it may see instead
  of the 25 examples only. For a practice variable that is the 25 examples plus the other 52
  practice variables (leave-one-out); for a held-out variable it is all 78 variables that are not
  held out. A variable's own tags are never shown, and no held-out variable is ever shown.
- `mixed`: a walk put together from stored walks, without new calls: the matrix and the context
  object from the step-by-step walk, everything else from another walk.

    ./settings.py --walk direct-all                       # walk the practice set
    ./settings.py --marks                                 # mark every stored walk
    ./settings.py --name tune --arms ... --run / --score  # compare workflows end to end
    ./settings.py --population held-out ...               # the final exam only
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import engine  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402

# `example_selection/pipeline.py` has the same module name and comes earlier on the path these
# side-experiment modules build, so a plain `import pipeline` took that one: the first start of
# the step 4 run (2026-11-19) stopped on an AttributeError before any call. Load this folder's
# file by its path instead.
_spec = importlib.util.spec_from_file_location("guided_pipeline", HERE / "pipeline.py")
pipeline = importlib.util.module_from_spec(_spec)
sys.modules["guided_pipeline"] = pipeline
_spec.loader.exec_module(pipeline)

FEATURE_SETS = {"full": frozenset({"form", "names", "types", "texts"}),
                "form+names+types": frozenset({"form", "names", "types"}),
                "form+types": frozenset({"form", "types"}), "form": frozenset({"form"})}
REPETITIONS = 15
SCHEDULE_SEED = 20261121
Transport = Callable[[Mapping[str, Any]], Awaitable[dict]]


def population(ctx: Mapping[str, Any], name: str) -> list[str]:
    """The variables of one population.

    Args: ctx: from `walk.load`; name: "practice" or "held-out".
    Returns: sorted variable ids.
    Raises: KeyError for an unknown name.
    Side Effects: none.
    """
    return {"practice": list(ctx["practice"]), "held-out": sorted(ctx["inputs"]["targets"])}[name]


def shown_ids(ctx: Mapping[str, Any], variable_id: str) -> list[str]:
    """The tagged variables a walk for one target may be shown.

    Args: ctx: from `walk.load`; variable_id: the target.
    Returns: the 25 examples and the practice variables, without the target itself. Never a
        held-out variable.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    held_out = set(ctx["inputs"]["targets"])
    pool = (set(ft.EXAMPLE_IDS) | set(ctx["practice"])) - held_out - {variable_id}
    return sorted(pool)


def direct_prompt_all(ctx: Mapping[str, Any], variable_id: str) -> str:
    """The direct prompt of `walk.py` with every permitted tagged variable as a demonstration.

    Args: ctx: from `walk.load`; variable_id: the target.
    Returns: the prompt; only its EXAMPLES section differs from `walk.direct_prompt`.
    Raises: ValueError when the base prompt no longer has an EXAMPLES section.
    Side Effects: none. Deterministic.
    """
    definition = ctx["records"][variable_id]["definition"]
    base = walk.direct_prompt(ctx, definition)
    if base.count("\nEXAMPLES\n") != 1:
        raise ValueError("the direct prompt has no single EXAMPLES section")
    lines = [base.split("\nEXAMPLES\n")[0], "EXAMPLES"]
    for shown in shown_ids(ctx, variable_id):
        lines += [f"Definition: {ctx['records'][shown]['definition']}",
                  "Patterns: " + json.dumps(walk.direct_answer(ctx, ctx["key"][shown]), ensure_ascii=False), ""]
    lines += [f"Definition: {definition}", "Patterns:"]
    return "\n".join(lines)


async def walk_direct_all(ctx: Mapping[str, Any], ids: Sequence[str], transport: Transport, path: Path,
                          concurrency: int = walk.CONCURRENCY) -> dict[str, Any]:
    """Walk variables with the `direct-all` prompt, one stored row per variable.

    Args: ctx: from `walk.load`; ids: the targets; transport: sends one request; path: the
        results file; concurrency: variables walked at once.
    Returns: {"walked", "stopped"}.
    Raises: nothing from the provider.
    Side Effects: provider calls; appends to `path`. Resumable per variable. A refused credential
        or any reasoning text stops the run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["id"] for row in ft.read_jsonl(path) if row.get("complete")}
    semaphore, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    state: dict[str, Any] = {"walked": 0, "stopped": None}

    async def one(variable_id: str) -> None:
        """Walk one variable and store its row.

        Args: variable_id: the target.
        Returns: nothing.
        Raises: nothing.
        Side Effects: one provider call; one appended row.
        """
        async with semaphore:
            if state["stopped"]:
                return
            prompt = direct_prompt_all(ctx, variable_id)
            started = ft.now()
            result = await transport(dict(walk.DIRECT_REQUEST, model=engine.MODEL,
                                          messages=[{"role": "user", "content": prompt}]))
            call = {"question": "direct-all", "ok": bool(result.get("ok")), "status": result.get("status"),
                    "error": result.get("error"), "raw": result.get("answer"),
                    "reasoning_chars": len(result.get("reasoning") or ""),
                    "prompt_tokens": result.get("prompt_tokens"), "completion_tokens": result.get("completion_tokens"),
                    "latency_seconds": round(result.get("latency") or 0.0, 3), "prompt_sha256": ft.sha256(prompt)}
            if call["status"] in (401, 403):
                state["stopped"] = "credential refused"
            elif call["reasoning_chars"]:
                state["stopped"] = "answers carry reasoning text despite the switch"
            facts = walk.facts_from_direct(ctx, result.get("answer") or "") if result.get("ok") else None
            row = {"id": variable_id, "style": "direct-all", "started_at": started, "facts": facts, "trace": {},
                   "confidence": {}, "calls": [call], "complete": call["ok"], "shown": len(shown_ids(ctx, variable_id))}
            state["walked"] += 1
            async with lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(v) for v in ids if v not in done))
    return state


def mixed(first: Mapping[str, Mapping[str, Any]], steps: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Put a walk together: matrix and context object from the step-by-step walk, the rest from another.

    Args: first: stored walks that supply property, modifier and constraint types; steps: stored
        step-by-step walks.
    Returns: {variable id: row}; where either walk has no facts, the other's facts are used alone.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out: dict[str, dict[str, Any]] = {}
    for variable_id, row in first.items():
        facts = row.get("facts")
        other = steps.get(variable_id, {}).get("facts")
        if facts is not None and other is not None:
            facts = dict(facts, matrix=list(other["matrix"]), context=list(other["context"]))
        out[variable_id] = {"id": variable_id, "style": "mixed", "facts": facts if facts is not None else other,
                            "calls": []}
    return out


def walks_named(ctx: Mapping[str, Any], name: str, where: str) -> dict[str, dict[str, Any]]:
    """Load one walk variant for one population.

    Args: ctx: from `walk.load`; name: "direct", "steps", "direct-all", "mixed" (direct with the
        step-by-step matrix and context) or "mixed-all" (the same on `direct-all`); where:
        "practice" or "held-out".
    Returns: {variable id: row}.
    Raises: KeyError for an unknown name.
    Side Effects: reads stored walk files.
    """
    prefix = "route" if where == "practice" else "route-held-out"
    stored = {style: walk.stored(walk.results_path(style, prefix)) for style in ("direct", "steps", "direct-all")}
    return {"direct": stored["direct"], "steps": stored["steps"], "direct-all": stored["direct-all"],
            "mixed": mixed(stored["direct"], stored["steps"]),
            "mixed-all": mixed(stored["direct-all"], stored["steps"])}[name]


def parse_arm(arm: str) -> tuple[str, str, frozenset[str]] | None:
    """Read an arm name of the form `mode@walk/features`.

    Args: arm: "base", or for example "guided+check@direct-all/full".
    Returns: None for "base" and "base+retry"; else (mode, walk name, features).
    Raises: KeyError or ValueError for a malformed name.
    Side Effects: none.
    """
    if arm in ("base", "base+retry"):
        return None
    mode, rest = arm.split("@")
    name, features = rest.split("/")
    if mode not in ("guided", "guided+check"):
        raise ValueError(f"unknown mode in arm {arm!r}")
    return mode, name, FEATURE_SETS[features]


async def run(ctx: Mapping[str, Any], arms: Sequence[str], ids: Sequence[str], where: str, transport: Transport,
              calls_path: Path, concurrency: int = engine.CONCURRENCY) -> dict[str, Any]:
    """Answer every arm for every variable and repetition, appending one record per answer.

    Args: ctx: from `walk.load`; arms: arm names; ids: the variables; where: their population;
        transport: sends one request; calls_path: the records file; concurrency: workers.
    Returns: {"issued", "skipped", "stopped", "list_price_usd"}.
    Raises: SystemExit when a guided arm lacks a stored walk for a variable.
    Side Effects: provider calls; appends to `calls_path`. Resumable.
    """
    specs = {arm: parse_arm(arm) for arm in arms}
    loaded = {spec[1]: walks_named(ctx, spec[1], where) for spec in specs.values() if spec}
    for name, rows in loaded.items():
        missing = [v for v in ids if v not in rows]
        if missing:
            raise SystemExit(f"{len(missing)} variables have no stored {name} walk for {where}")
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    guard = engine.Guard()
    answered = {key for key, record in engine.latest(calls_path).items() if record.get("ok")}
    queue: asyncio.Queue = asyncio.Queue()
    for job in engine.schedule(list(arms), ids, REPETITIONS, SCHEDULE_SEED):
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
            spec = specs[arm]
            if spec is None:
                record = await pipeline.attempt(ctx, (arm, repetition, variable_id), transport, frozenset(), {}, guard)
            else:
                record = await pipeline.attempt(ctx, (spec[0], repetition, variable_id), transport, spec[2],
                                                loaded[spec[1]], guard)
                record["arm"] = arm
            counts["issued"] += 1
            async with lock:
                with calls_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return {"issued": counts["issued"], "skipped": counts["skipped"], "stopped": guard.reason,
            "list_price_usd": round(guard.spend, 4)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode: walk, marks, plan, run or score.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status, 0 on success.
    Raises: SystemExit on bad arguments, a missing PSNC key or missing stored walks.
    Side Effects: --walk and --run call the provider; --walk, --run and --score write output files.
    """
    parser = argparse.ArgumentParser(description="Settings and final exam (exploratory).")
    parser.add_argument("--walk", choices=("direct-all",))
    parser.add_argument("--marks", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--arms")
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--population", choices=("practice", "held-out"), default="practice",
                        help="held-out is for the final exam only")
    args = parser.parse_args(argv)
    ctx = walk.load()
    ids = population(ctx, args.population)
    prefix = "route" if args.population == "practice" else "route-held-out"

    if args.marks:
        for name in ("direct", "steps", "mixed", "direct-all", "mixed-all"):
            try:
                rows = walks_named(ctx, name, args.population)
            except OSError:
                continue
            if all(v in rows for v in ids):
                print(name, json.dumps(walk.report(ctx, rows, ids)))
        return 0
    arms = args.arms.split(",") if args.arms else []
    files = pipeline.paths(args.name) if args.name else {}
    if args.plan_only:
        prompts = [direct_prompt_all(ctx, v) for v in ids]
        print(json.dumps({"population": args.population, "variables": len(ids), "arms": arms,
                          "answers": len(arms) * len(ids) * REPETITIONS,
                          "settings.py": ft.sha256((HERE / "settings.py").read_text(encoding="utf-8")),
                          "direct_all_prompts_sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts)),
                          "direct_all_prompt_chars_mean": round(sum(len(p) for p in prompts) / len(prompts), 1),
                          "shown_per_target": sorted({len(shown_ids(ctx, v)) for v in ids})}, indent=2))
        return 0
    if args.score:
        scores = engine.score(files["calls"], files["scores"], ctx["records"], ctx["inputs"]["schema"])
        calls = engine.latest(files["calls"])
        summary = engine.summarise(arms, args.baseline, ids, REPETITIONS, calls, scores)
        summary["behaviour"] = pipeline.behaviour(calls)
        summary["settings"] = {"population": args.population, "arms": arms}
        files["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for arm, cell in summary["cells"].items():
            how = summary["behaviour"].get(arm, {})
            print(f"{arm:38} F1 {cell['close_f1_official_mean']:.4f} (SD {cell['close_f1_official_sd'] or 0:.4f}) "
                  f"invalid {cell['invalid']:3}  repaired {how.get('repaired', 0):.2f}  fell back "
                  f"{how.get('fell_back', 0):.2f}  calls/answer {how.get('calls_per_answer', 1):.2f}")
        for arm, result in summary["comparisons"].items():
            print(f"{arm:38} vs {args.baseline}: {result['difference']:+.4f} [{result['ci_low']:+.4f}, "
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
                print(json.dumps(await walk_direct_all(ctx, ids, transport, walk.results_path("direct-all", prefix))))
            finally:
                await client.aclose()
            return 0
        transport, client = ft.psnc_transport()
        try:
            print(json.dumps(await run(ctx, arms, ids, args.population, transport, files["calls"]), indent=2))
        finally:
            await client.aclose()
        return 0

    if not (args.walk or args.run):
        parser.error("choose --walk, --marks, --plan-only, --run or --score")
    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
