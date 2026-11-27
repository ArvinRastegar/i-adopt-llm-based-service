#!/usr/bin/env python
"""Does naming the object and the matrix first lift the pattern finder's matrix, and does stating it help?

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/ANSWER_FORM_CHECK_20261127.md` (private repository).

Arvin, 2026-11-27: "Yes, plan it" (a same-day check of the answer form, and stating the finder's own
matrix), then in the question dialog "Yes, include it" (the matrix-only ceiling) and "Up to $2".

Part A is the pattern finder of the conventions test (`conventions_test.py`), setting D+S with the
completed type table, in two answer forms, the same day and in one run:

- `five-key`: stage 3's prompt (`knowledge.build`, cell D+S), its five-key answer;
- `named-first`: the conventions test's `plain` prompt, which first names the object of interest, the
  matrix entity and the restricting phrases.

Part B is today's prompt with the 25 most similar decomposed variables (stage 5's `rag-25`), ten runs:
`rag-25`; `rag-25+finder-matrix`, the matrix of `named-first` (run 1) stated, run when its matrix is
right more often than stage 1's break-even 0.57; `rag-25+experts-matrix`, the experts' matrix alone.

    ./answer_form_check.py --plan-only
    ./answer_form_check.py --dev        # Part A's arms on invented variables only
    ./answer_form_check.py --run-a
    ./answer_form_check.py --score-a
    ./answer_form_check.py --run-b
    ./answer_form_check.py --score-b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import ceiling  # noqa: E402
import conventions_test as ct  # noqa: E402
import draft_rag as dr  # noqa: E402
import engine  # noqa: E402
import hints  # noqa: E402
import knowledge as kn  # noqa: E402
import orclient  # noqa: E402
import rag_tuning as rt  # noqa: E402
import rulebook  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402

ARMS = ("five-key", "named-first")
PART_B = ("rag-25", "rag-25+finder-matrix", "rag-25+experts-matrix")
REPETITIONS, PART_B_REPETITIONS = 5, 10
SEED_A, SEED_B = 20261211, 20261212
PURPOSE, ALLOWANCE_USD = "answer-form", 2.0          # Arvin, 2026-11-27: "Up to $2", every part together
CONCURRENCY = 16
BREAK_EVEN_MATRIX = 0.57                             # stage 1
COLUMNS = ("matrix", "constraint", "matrix_pattern", "both", "parsed")
CALLS_A = orclient.OUTPUT / "answer-form-a-calls.jsonl"
SUMMARY_A = orclient.OUTPUT / "answer-form-a-summary.json"
CALLS_B = orclient.OUTPUT / "answer-form-b-calls.jsonl"
SCORES_B = orclient.OUTPUT / "answer-form-b-scores.jsonl"
SUMMARY_B = orclient.OUTPUT / "answer-form-b-summary.json"
DEV_CALLS = orclient.OUTPUT / "answer-form-dev-calls.jsonl"


def load() -> dict[str, Any]:
    """Part A's context: the conventions test's, plus every variable as a five-key example.

    Args: none.
    Returns: the context of `conventions_test.load` (no fresh entries) with "five_demos".
    Raises: as `conventions_test.load`.
    Side Effects: reads local files.
    """
    ctx = ct.load(fresh=False)
    ctx["five_demos"] = {v: kn.demo(ctx, v, ctx["records"][v]["definition"], ctx["key"][v], ctx["records"][v]["gold"])
                         for v in ctx["ids"]}
    return ctx


def build(ctx: Mapping[str, Any], arm: str, definition: str, fold: int | None, invented: Mapping[str, Any] | None = None) -> str:
    """The prompt of one Part A arm.

    Args: ctx: from `load`; arm: one of `ARMS`; definition: the target's definition; fold: the
        target's fold (its examples are the other five folds); invented: for the dev check, the
        examples to show instead, as {"five": [...], "named": [...]}.
    Returns: the prompt.
    Raises: KeyError for an unknown arm.
    Side Effects: none. Deterministic.
    """
    if arm == "five-key":
        demos = invented["five"] if invented else [ctx["five_demos"][v] for v in ctx["ids"] if ctx["fold_of"][v] != fold]
        return kn.build(ctx, "D+S", definition, demos)
    if arm == "named-first":
        demos = invented["named"] if invented else kn.usable_demos(ctx, fold)
        return ct.build(ctx, "plain", definition, demos)
    raise KeyError(arm)


def prompt_a(ctx: Mapping[str, Any], job: engine.Job) -> str:
    """The prompt of one Part A call.

    Args: ctx: from `load`; job: (arm, run, variable).
    Returns: the prompt.
    Raises: KeyError for an unknown arm or variable.
    Side Effects: none. Deterministic.
    """
    arm, _, v = job
    return build(ctx, arm, ctx["records"][v]["definition"], ctx["fold_of"][v])


def summarise_a(ctx: Mapping[str, Any], calls: Mapping[engine.Job, Mapping[str, Any]]) -> dict[str, Any]:
    """Part A: shares by arm, `named-first` minus `five-key`, and whether Part B's finder arm may run.

    Args: ctx: from `load`; calls: from `engine.latest`.
    Returns: the summary.
    Raises: KeyError when an answer is missing.
    Side Effects: none. Deterministic.
    """
    marked = {k: ct.outcome(ctx, ctx["key"][k[2]], r.get("answer") if r.get("ok") else None) for k, r in calls.items()}
    wanted = set(engine.schedule(ARMS, ctx["ids"], REPETITIONS, SEED_A))
    out: dict[str, Any] = {"generated_at": ft.now(), "answers": len(calls),
                           "complete": wanted <= set(calls) and all(calls[j].get("ok") for j in wanted),
                           "failed": sum(1 for j in wanted if not calls.get(j, {}).get("ok")), "arms": {}, "comparisons": {}}
    for arm in ARMS:
        rows = [row for (a, _, _), row in marked.items() if a == arm]
        out["arms"][arm] = {column: sum(bool(r[column]) for r in rows) / len(rows) for column in COLUMNS}
        tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
        out["arms"][arm]["constraint_f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    for column in ("matrix", "constraint"):
        out["comparisons"][column] = ct.difference(ct.per_variable(marked, "named-first", column, ctx["ids"]),
                                                   ct.per_variable(marked, "five-key", column, ctx["ids"]), ctx["ids"])
    out["finder_arm_runs"] = out["arms"]["named-first"]["matrix"] > BREAK_EVEN_MATRIX
    return out


def finder_facts(ctx: Mapping[str, Any], calls: Mapping[engine.Job, Mapping[str, Any]]) -> dict[str, Any]:
    """The patterns `named-first` found in run 1 for each variable.

    Args: ctx: from `load`; calls: Part A's stored calls.
    Returns: {variable: facts, or None when the answer is missing or not the requested JSON}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out = {}
    for v in ctx["ids"]:
        record = calls.get(("named-first", 1, v), {})
        answer = record.get("answer") if record.get("ok") else None
        proper, _ = kn.requested(ctx, answer)
        out[v] = walk.facts_from_direct(ctx, answer or "") if proper else None
    return out


def prompt_b(ctx: Mapping[str, Any], world: Mapping[str, Any], job: engine.Job, found: Mapping[str, Any]) -> str:
    """The prompt of one Part B call.

    Args: ctx: from `load`; world: from `rag_tuning.corpus_world`; job: (arm, run, variable); found:
        from `finder_facts`.
    Returns: stage 5's `rag-25` prompt; for the guided arms with the matrix stated (stage 4's
        statement); `rag-25+finder-matrix` without a usable run-1 answer states nothing.
    Raises: KeyError for an unknown arm.
    Side Effects: none. Deterministic.
    """
    arm, _, v = job
    ids = rt.examples(world, "rag-25", v)
    if arm == "rag-25":
        return rt.prompt_of(world, v, ids)
    if arm == "rag-25+experts-matrix":
        facts = hints.facts_from_key(ctx["key"][v])
    elif arm == "rag-25+finder-matrix":
        facts = found.get(v)
    else:
        raise KeyError(arm)
    if facts is None:
        return rt.prompt_of(world, v, ids)
    inputs = dict(world["base_inputs"](v), examples=[world["record"](u) for u in ids])
    return dr.fill_prompt(inputs, world["definition"](v), dr.render(ctx, dr.statement(facts, {"matrix"})))


def arms_b(summary_a: Mapping[str, Any]) -> tuple[str, ...]:
    """The Part B arms to run, by Part A's gate.

    Args: summary_a: from `summarise_a`.
    Returns: all three, or the two without the finder's matrix.
    Raises: KeyError when the summary lacks the gate.
    Side Effects: none. Deterministic.
    """
    return PART_B if summary_a["finder_arm_runs"] else (PART_B[0], PART_B[2])


def fingerprints(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the code, the data and every fixed prompt.

    Args: ctx: from `load`.
    Returns: hex digests (16 characters) and mean prompt lengths; `rag-25+finder-matrix` depends on
        Part A and is hashed when it is run.
    Raises: OSError when a file cannot be read.
    Side Effects: reads files and stage 5's similarity table.
    """
    out: dict[str, Any] = {name: ceiling.file_sha256(HERE / name)[:16] for name in (
        "answer_form_check.py", "conventions_test.py", "knowledge.py", "walk.py", "rag_tuning.py", "draft_rag.py")}
    out.update({f"data/{name}": ceiling.file_sha256(rulebook.DATA / name)[:16] for name in (
        "types_overlay.json", "constraint_types.json", "patterns.json", "route_key.json")})
    out["prompts_a"] = {}
    for arm in ARMS:
        prompts = [prompt_a(ctx, (arm, 1, v)) for v in ctx["ids"]]
        out["prompts_a"][arm] = {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                                 "chars_mean": round(sum(len(p) for p in prompts) / len(prompts), 1)}
    world = rt.corpus_world(rt.load(), None)
    out["prompts_b"] = {}
    for arm in (PART_B[0], PART_B[2]):
        prompts = [prompt_b(ctx, world, (arm, 1, v), {}) for v in ctx["ids"]]
        out["prompts_b"][arm] = {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                                 "chars_mean": round(sum(len(p) for p in prompts) / len(prompts), 1)}
    return out


async def dev(ctx: Mapping[str, Any], transport: orclient.Transport, guard: orclient.Guard) -> list[dict]:
    """Part A's arms on stage 3's invented variables, two halves teaching each other.

    Args: ctx: from `load`; transport, guard: from `orclient`.
    Returns: one row per call: parsed, matrix and constraint types right, the answer's start.
    Raises: nothing from the provider.
    Side Effects: provider calls for invented definitions only; appends to `DEV_CALLS`.
    """
    items = ct.invented(ctx)
    order = sorted(items, key=lambda name: int(name.split("-")[1]))
    half = {name: position % 2 for position, name in enumerate(order)}
    five = {name: kn.demo(ctx, name, items[name]["definition"], items[name]["entry"], kn.INVENTED_GOLD[name]) for name in order}
    shown = {side: {"five": [five[n] for n in order if half[n] != side],
                    "named": [items[n]["demo"] for n in order if half[n] != side]} for side in (0, 1)}
    jobs = engine.schedule(ARMS, order, 1, SEED_A)
    await orclient.run_jobs(jobs, lambda job: build(ctx, job[0], items[job[2]]["definition"], None, shown[half[job[2]]]),
                            transport, DEV_CALLS, guard, concurrency=8, request=ct.REQUEST)
    rows = []
    for key, record in sorted(engine.latest(DEV_CALLS).items()):
        row = ct.outcome(ctx, items[key[2]]["entry"], record.get("answer") if record.get("ok") else None)
        rows.append({"key": key, "ok": record.get("ok"), "parsed": row["parsed"], "matrix": row["matrix"],
                     "constraint": row["constraint"], "answer": (record.get("answer") or "")[:160]})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode.

    Args: argv: arguments; `sys.argv` when None.
    Returns: 0 on success; 1 when a stop rule ended a run or a summary is not complete.
    Raises: SystemExit on bad arguments or a step run out of order.
    Side Effects: --dev and --run-* call the provider; all modes but --plan-only write files.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--plan-only", "--dev", "--run-a", "--score-a", "--run-b", "--score-b"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args(argv)
    ctx = load()
    if args.plan_only:
        print(json.dumps({"variables": len(ctx["ids"]), "arms_a": ARMS, "arms_b": PART_B, "repetitions": REPETITIONS,
                          "calls_a": len(ctx["ids"]) * REPETITIONS * len(ARMS),
                          "calls_b_per_arm": len(ctx["ids"]) * PART_B_REPETITIONS, "allowance_usd": ALLOWANCE_USD,
                          "model": orclient.MODEL, "host": orclient.HOST["name"], "request_a": ct.REQUEST,
                          "request_b": engine.REQUEST, **fingerprints(ctx)}, indent=2))
        return 0
    if args.score_a:
        summary = summarise_a(ctx, engine.latest(CALLS_A))
        SUMMARY_A.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for arm, cells in summary["arms"].items():
            print(f"{arm:12} " + " ".join(f"{c} {v:.3f}" for c, v in cells.items()))
        for column, row in summary["comparisons"].items():
            print(f"named-first - five-key, {column}: {row['mean']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
        print("finder arm runs:", summary["finder_arm_runs"], "| complete:", summary["complete"])
        return 0 if summary["complete"] else 1
    if args.score_b:
        ctx2 = rt.load()
        arms = arms_b(json.loads(SUMMARY_A.read_text(encoding="utf-8")))
        scores = engine.score(CALLS_B, SCORES_B, ctx2["records"], ctx2["inputs"]["schema"])
        summary = ct.summarise2(ctx2, scores, arms)
        SUMMARY_B.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for arm, row in summary.get("arms", {}).items():
            print(f"{arm:22} F1 {row['f1_mean']:.4f} sd {row['f1_sd']:.4f} invalid {row['invalid']}")
        for arm, row in summary.get("against_rag25", {}).items():
            print(f"{arm} - rag-25: {row['gain']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
        return 0 if summary["complete"] else 1

    async def go() -> int:
        """Open the transport, run the asked part, always close the client.

        Args: none.
        Returns: 1 when a stop rule ended the work, else 0.
        Raises: SystemExit when the key is missing, the ledger is held, or Part B is asked for
            before Part A is scored and complete.
        Side Effects: provider calls, the ledger and output files.
        """
        spend = orclient.Spend(purpose=PURPOSE, allowance=ALLOWANCE_USD)
        guard = orclient.Guard(spend)
        transport, client = orclient.open_transport(spend, PURPOSE, concurrency=args.concurrency)
        try:
            if args.dev:
                for row in await dev(ctx, transport, guard):
                    print(row["key"], "ok" if row["ok"] else "FAILED", "parsed" if row["parsed"] else "NOT PARSED",
                          f"matrix {'Y' if row['matrix'] else 'n'}", f"constraints {'Y' if row['constraint'] else 'n'}",
                          row["answer"].replace("\n", " "))
            elif args.run_a:
                jobs = engine.schedule(ARMS, ctx["ids"], REPETITIONS, SEED_A)
                print(json.dumps(await orclient.run_jobs(jobs, lambda job: prompt_a(ctx, job), transport, CALLS_A, guard,
                                                         concurrency=args.concurrency, request=ct.REQUEST,
                                                         max_calls=args.max_calls), indent=2))
            else:
                if not SUMMARY_A.exists() or not json.loads(SUMMARY_A.read_text(encoding="utf-8"))["complete"]:
                    raise SystemExit("score a complete Part A first")
                arms = arms_b(json.loads(SUMMARY_A.read_text(encoding="utf-8")))
                world = rt.corpus_world(rt.load(), None)
                found = finder_facts(ctx, engine.latest(CALLS_A))
                jobs = engine.schedule(arms, ctx["ids"], PART_B_REPETITIONS, SEED_B)
                print(json.dumps(await orclient.run_jobs(jobs, lambda job: prompt_b(ctx, world, job, found), transport,
                                                         CALLS_B, guard, concurrency=args.concurrency,
                                                         request=dict(engine.REQUEST), max_calls=args.max_calls), indent=2))
            if guard.reason:
                print("stopped:", guard.reason)
            return 1 if guard.reason else 0
        finally:
            await orclient.close_transport(transport, client)

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
