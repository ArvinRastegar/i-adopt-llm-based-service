#!/usr/bin/env python
"""A stronger reader for the matrix and constraint-type call: PSNC's GLM-5.3 in the finder's place.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/STRONGER_READER_TEST_20261127.md` (private repository).

Arvin, 2026-11-27: "no need to plan it just test it with the best smartest model that the psnc has
available", then "keep in mind that the full response should never take more than 30 seconds. otherwise
we should try other possiblities". GLM-5.3 was chosen as PSNC's strongest model from public comparisons
(see the record).

Step 1: the finder's own prompt (the answer-form check's `named-first`, every variable of the other five
folds as an example), sent byte for byte to three readers on PSNC, one answer per variable: `qwen3.8`
(Qwen3.8-27B, the finder's settings), `glm-5.3` (thinking off, the same settings) and `glm-5.3-think`
(thinking on, the server's own sampling). Each is marked against the experts as the finder was; latency
counts against the 30-second limit. Step 2, only for a reader that passes an accuracy gate and the time
gate: today's best pipeline with that reader's decisions stated, against `best`, ten runs.

    ./stronger_reader_test.py --plan-only
    ./stronger_reader_test.py --dev
    ./stronger_reader_test.py --run
    ./stronger_reader_test.py --score
    ./stronger_reader_test.py --run-decomposition
    ./stronger_reader_test.py --score-decomposition
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import answer_form_check as afc  # noqa: E402
import ceiling  # noqa: E402
import constraint_types_test as cy  # noqa: E402
import conventions_test as ct  # noqa: E402
import draft_rag as dr  # noqa: E402
import engine  # noqa: E402
import knowledge as kn  # noqa: E402
import rag_tuning as rt  # noqa: E402
import stand_in_test as si  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402

READERS = ("qwen3.8", "glm-5.3", "glm-5.3-think", "qwen3.5", "deepseek-v4.1-flash", "glm-5.3-flash", "gpt-oss")
MODELS = {"qwen3.8": "Qwen3.8-27B", "glm-5.3": "GLM-5.3", "glm-5.3-think": "GLM-5.3", "qwen3.5": "Qwen3.5-397B-A17B",
          "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash", "glm-5.3-flash": "GLM-5.3-Flash", "gpt-oss": "gpt-oss_120b"}
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
OFF_BOTH = {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}   # Qwen/GLM and DeepSeek switches
REQUESTS = {"qwen3.8": dict(ct.REQUEST, **OFF), "glm-5.3": dict(ct.REQUEST, **OFF),
            "glm-5.3-think": {"stream": False, "max_tokens": 16000, "chat_template_kwargs": {"enable_thinking": True}},
            "qwen3.5": dict(ct.REQUEST, **OFF_BOTH), "deepseek-v4.1-flash": dict(ct.REQUEST, **OFF_BOTH),
            "glm-5.3-flash": dict(ct.REQUEST, **OFF_BOTH),
            "gpt-oss": {"stream": False, "max_tokens": 4000, "reasoning_effort": "low"}}
THINKING = frozenset({"glm-5.3-think", "gpt-oss"})          # readers whose answers carry reasoning text by design
# The hub's list prices in US dollars per million tokens, input/output (EXAMPLE_FORMAT_TEST_20261118.md); a
# model without a known price is capped at GLM-5.3's, the highest known.
PRICES = {"Qwen3.8-27B": (0.089, 0.178), "GLM-5.3": (6.13, 12.27)}
CAP_USD = 20.0                     # list-price stop per reader
LIMIT_SECONDS = 30.0               # Arvin: the full response must stay under 30 seconds
BREAK_EVEN = 0.54                  # stage 1: constraint types must be right for more variables than this
TIMEOUT = 600.0                    # measure slow answers instead of cutting them at the study's 120 s
CONCURRENCY = 6
DEV_ITEMS = ("dev-1", "dev-2", "dev-3", "dev-4")
STEP2 = ("best", "best+reader")
REPETITIONS, SEED = 10, 20261251
OUT = HERE.parent / "output" / "retrieval"
CALLS = OUT / "stronger-reader-calls.jsonl"
DEV = OUT / "stronger-reader-dev-calls.jsonl"
SUMMARY = OUT / "stronger-reader-summary.json"
CALLS2 = OUT / "stronger-reader-decomposition-calls.jsonl"
SCORES2 = OUT / "stronger-reader-decomposition-scores.jsonl"
SUMMARY2 = OUT / "stronger-reader-decomposition-summary.json"


class Guard(si.Guard):
    """The study's stop rules, ten failures in a row, a list-price cap; reasoning text allowed when thinking is on."""

    def __init__(self, model: str, thinking: bool) -> None:
        """Start with no stop reason.

        Args: model: the model served; thinking: whether reasoning text is expected.
        Returns: nothing.
        Raises: nothing.
        Side Effects: none.
        """
        super().__init__()
        self.model, self.thinking_on, self.cost = model, thinking, 0.0

    def record(self, result: Mapping[str, Any]) -> None:
        """Apply the stop rules to one finished call.

        Args: result: the stored call record.
        Returns: nothing.
        Raises: nothing.
        Side Effects: updates counts and the list-price total; may set the stop reason.
        """
        super().record(dict(result, reasoning_chars=0) if self.thinking_on else result)
        if result.get("ok"):
            price_in, price_out = PRICES.get(self.model, PRICES["GLM-5.3"])
            self.cost += ((result.get("prompt_tokens") or 0) * price_in + (result.get("completion_tokens") or 0) * price_out) / 1e6
        if self.cost > CAP_USD:
            self.reason = self.reason or f"list price above ${CAP_USD:.0f}"


def model_transport(transport: Any, model: str) -> Any:
    """A transport that sends every request to one model.

    Args: transport: PSNC's; model: the model id.
    Returns: the wrapped transport.
    Raises: nothing.
    Side Effects: none until called.
    """
    async def send(body: Mapping[str, Any]) -> dict:
        """Send one request to the model.

        Args: body: the request body.
        Returns: the transport's result.
        Raises: nothing.
        Side Effects: one request.
        """
        return await transport(dict(body, model=model))

    return send


def percentile(values: Sequence[float], share: float) -> float:
    """A percentile by the nearest rank.

    Args: values: the values; share: between 0 and 1.
    Returns: the value at rank ceil(share * n).
    Raises: ValueError for no values.
    Side Effects: none. Deterministic.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    return ordered[max(0, -(-int(round(share * 1000)) * len(ordered) // 1000) - 1)]


def fill_latencies() -> list[float]:
    """Today's fill call's latencies on PSNC: the stand-in test's `best` arm, all ten runs.

    Args: none.
    Returns: the latencies in seconds.
    Raises: nothing.
    Side Effects: reads the calls file.
    """
    return [r["latency_seconds"] for k, r in engine.latest(si.CALLS).items() if k[0] == "best" and r.get("ok")]


def summarise(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Step 1: each reader against the experts, the GLM readers against `qwen3.8`, latencies and the gates.

    Args: ctx: from `answer_form_check.load`.
    Returns: shares by reader (and the stored OpenRouter finder run 1 for reference), differences,
        latencies, "gates" by reader and "step2" (the reader Step 2 uses and the decisions it states, or None).
    Raises: KeyError when a stored finder answer is missing.
    Side Effects: reads the calls files.
    """
    calls = engine.latest(CALLS)
    stored = engine.latest(afc.CALLS_A)
    present = [r for r in READERS if any(k[0] == r for k in calls)]
    answers = {r: {v: calls[(r, 1, v)].get("answer") if calls.get((r, 1, v), {}).get("ok") else None for v in ctx["ids"]}
               for r in present}
    answers["finder-openrouter"] = {v: stored[("named-first", 1, v)].get("answer") for v in ctx["ids"]}
    marks = {r: {v: ct.outcome(ctx, ctx["key"][v], a[v]) for v in ctx["ids"]} for r, a in answers.items()}
    out: dict[str, Any] = {"generated_at": ft.now(), "variables": len(ctx["ids"]), "readers": {}, "differences": {},
                           "latency": {}, "gates": {}}
    for r, m in marks.items():
        tp, fp, fn = (sum(x[k] for x in m.values()) for k in ("tp", "fp", "fn"))
        out["readers"][r] = {c: sum(bool(x[c]) for x in m.values()) / len(m) for c in ("parsed", "matrix", "matrix_pattern", "constraint", "both")}
        out["readers"][r]["type_f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        out["readers"][r]["answered"] = sum(a is not None for a in answers[r].values())
    fill = fill_latencies()
    out["latency"]["fill"] = {"p50": statistics.median(fill), "p95": percentile(fill, 0.95), "max": max(fill), "calls": len(fill)}
    step2 = None
    for r in present:
        lat = [calls[(r, 1, v)]["latency_seconds"] for v in ctx["ids"] if (r, 1, v) in calls]
        out["latency"][r] = {"p50": statistics.median(lat), "p95": percentile(lat, 0.95), "max": max(lat), "calls": len(lat)} if lat else None
        if r == "qwen3.8":
            continue
        diff = {c: ct.difference({v: float(marks[r][v][c]) for v in ctx["ids"]}, {v: float(marks["qwen3.8"][v][c]) for v in ctx["ids"]},
                                 ctx["ids"]) for c in ("matrix", "constraint")}
        out["differences"][f"{r} - qwen3.8"] = diff
        types = out["readers"][r]["constraint"] > BREAK_EVEN and diff["constraint"]["low"] > 0
        matrix = diff["matrix"]["low"] > 0
        fits = bool(lat) and out["latency"][r]["p95"] + out["latency"]["fill"]["p95"] <= LIMIT_SECONDS
        out["gates"][r] = {"types": types, "matrix": matrix, "time": fits}
        if fits and (types or matrix):
            candidate = {"reader": r, "decisions": sorted({"matrix", "constraint"} if types else {"matrix"}),
                         "constraint": out["readers"][r]["constraint"]}
            if step2 is None or (types and "constraint" not in step2["decisions"]) or \
                    (("constraint" in candidate["decisions"]) == ("constraint" in step2["decisions"])
                     and candidate["constraint"] > step2["constraint"]):
                step2 = candidate
    out["step2"] = step2
    return out


def reader_facts(ctx: Mapping[str, Any], reader: str) -> dict[str, Any]:
    """The patterns one reader found for each variable.

    Args: ctx: from `answer_form_check.load`; reader: one of `READERS`.
    Returns: {variable: facts, or None when its answer is missing or not the requested JSON}.
    Raises: nothing.
    Side Effects: reads the calls file.
    """
    calls = engine.latest(CALLS)
    out = {}
    for v in ctx["ids"]:
        record = calls.get((reader, 1, v), {})
        answer = record.get("answer") if record.get("ok") else None
        proper, _ = kn.requested(ctx, answer)
        out[v] = walk.facts_from_direct(ctx, answer or "") if proper else None
    return out


def prompt2(ctx: Mapping[str, Any], world: Mapping[str, Any], job: engine.Job, found: Mapping[str, Any],
            facts: Mapping[str, Any], decisions: Sequence[str]) -> str:
    """The prompt of one Step 2 call.

    Args: ctx: from `answer_form_check.load`; world: from `rag_tuning.corpus_world`; job: (arm, run,
        variable); found: the stored finder's facts (run 1); facts: the reader's facts; decisions: what
        the reader's facts state.
    Returns: `best` is the constraint-types test's `best`; `best+reader` states the reader's decisions
        (stage 4's statement) and falls back to `best` without usable reader facts.
    Raises: KeyError for an unknown arm.
    Side Effects: none. Deterministic.
    """
    arm, run, v = job
    if arm not in STEP2:
        raise KeyError(arm)
    control = afc.prompt_b(ctx, world, ("rag-25+finder-matrix", run, v), found)
    if arm == "best" or facts.get(v) is None:
        return control
    ids = rt.examples(world, "rag-25", v)
    inputs = dict(world["base_inputs"](v), examples=[world["record"](u) for u in ids])
    return dr.fill_prompt(inputs, world["definition"](v), dr.render(ctx, dr.statement(facts[v], set(decisions))))


def fingerprints(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the code and the 102 Step 1 prompts, and check them against the stored finder's run 1.

    Args: ctx: from `answer_form_check.load`.
    Returns: hex digests (16 characters), the mean prompt length and how many prompts equal the stored ones.
    Raises: OSError when a file cannot be read.
    Side Effects: reads files.
    """
    out: dict[str, Any] = {name: ceiling.file_sha256(HERE / name)[:16] for name in (
        "stronger_reader_test.py", "answer_form_check.py", "conventions_test.py", "constraint_types_test.py",
        "stand_in_test.py", "walk.py", "engine.py")}
    prompts = [afc.prompt_a(ctx, ("named-first", 1, v)) for v in ctx["ids"]]
    stored = engine.latest(afc.CALLS_A)
    out["prompts"] = {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                      "chars_mean": round(sum(len(p) for p in prompts) / len(prompts), 1),
                      "equal_to_stored_finder": sum(ft.sha256(p) == stored[("named-first", 1, v)]["prompt_sha256"]
                                                    for p, v in zip(prompts, ctx["ids"], strict=True))}
    return out


async def run_reader(ctx: Mapping[str, Any], transport: Any, reader: str, jobs: Sequence[engine.Job],
                     prompt_of: Any, path: Path) -> dict[str, Any]:
    """Run one reader's jobs with its model, request and stop rules.

    Args: ctx: from `answer_form_check.load`; transport: PSNC's; reader: one of `READERS`; jobs: its
        jobs; prompt_of: builds a job's prompt; path: the calls file.
    Returns: the runner's counts and the stop reason.
    Raises: nothing from the provider.
    Side Effects: PSNC calls; appends to `path`.
    """
    guard = Guard(MODELS[reader], reader in THINKING)
    result = await engine.run_jobs(jobs, prompt_of, model_transport(transport, MODELS[reader]), path,
                                   concurrency=CONCURRENCY, guard=guard, request=REQUESTS[reader])
    return dict(result, reader=reader, list_price_usd=round(guard.cost, 4))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode.

    Args: argv: arguments; `sys.argv` when None.
    Returns: 0 on success; 1 when a stop rule ended a run or a summary is not complete.
    Raises: SystemExit on bad arguments or a step run out of order.
    Side Effects: --dev and --run* call PSNC; the others read and write local files.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--plan-only", "--dev", "--run", "--score", "--run-decomposition", "--score-decomposition"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--readers", nargs="*", default=list(READERS))
    args = parser.parse_args(argv)
    ctx = afc.load()
    if args.plan_only:
        print(json.dumps({"provider": "PSNC", "readers": {r: MODELS[r] for r in READERS}, "requests": REQUESTS,
                          "limit_seconds": LIMIT_SECONDS, "cap_usd": CAP_USD, **fingerprints(ctx)}, indent=2))
        return 0
    if args.score:
        out = summarise(ctx)
        SUMMARY.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=1))
        return 0
    if args.score_decomposition:
        ctx2 = rt.load()
        summary = cy.summarise2(ctx2, engine.score(CALLS2, SCORES2, ctx2["records"], ctx2["inputs"]["schema"]), STEP2)
        SUMMARY2.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for arm, row in summary.get("arms", {}).items():
            print(f"{arm:12} F1 {row['f1_mean']:.4f} sd {row['f1_sd']:.4f} invalid {row['invalid']}")
        for arm, row in summary.get("against_best", {}).items():
            print(f"{arm} - best: {row['gain']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
        return 0 if summary["complete"] else 1

    async def go() -> int:
        """Open PSNC's transport, run the asked part, always close the client.

        Args: none.
        Returns: 1 when a stop rule ended the work, else 0.
        Raises: SystemExit when a step is run out of order.
        Side Effects: PSNC calls and the calls files.
        """
        ft.TIMEOUT = TIMEOUT       # read by `walk.chat_transport` when it opens the client
        transport, client = walk.chat_transport(concurrency=CONCURRENCY)
        stopped = False
        try:
            if args.dev:
                items = ct.invented(ctx)
                for r in args.readers:
                    res = await run_reader(ctx, transport, r, [(r, 1, n) for n in DEV_ITEMS],
                                           lambda job: afc.build(ctx, "named-first", items[job[2]]["definition"], None), DEV)
                    stopped |= bool(res["stopped"])
                    print(json.dumps(res))
                calls = engine.latest(DEV)
                for (r, _, n), rec in sorted(calls.items()):
                    mark = ct.outcome(ctx, items[n]["entry"], rec.get("answer") if rec.get("ok") else None)
                    print(r, n, rec.get("status"), f"{rec.get('latency_seconds')}s", "parsed" if mark["parsed"] else "NOT PARSED",
                          "matrix" if mark["matrix"] else "-", "types" if mark["constraint"] else "-",
                          f"out {rec.get('completion_tokens')} reasoning {rec.get('reasoning_chars')}")
            elif args.run:
                for r in args.readers:
                    res = await run_reader(ctx, transport, r, [(r, 1, v) for v in ctx["ids"]],
                                           lambda job: afc.prompt_a(ctx, ("named-first", 1, job[2])), CALLS)
                    stopped |= bool(res["stopped"])
                    print(json.dumps(res))
            else:
                if not SUMMARY.exists() or not json.loads(SUMMARY.read_text(encoding="utf-8")).get("step2"):
                    raise SystemExit("no reader passed the gates; Step 2 does not run")
                step2 = json.loads(SUMMARY.read_text(encoding="utf-8"))["step2"]
                world = rt.corpus_world(rt.load(), None)
                found = afc.finder_facts(ctx, engine.latest(afc.CALLS_A))
                facts = reader_facts(ctx, step2["reader"])
                guard = si.Guard()
                print(json.dumps(await engine.run_jobs(
                    engine.schedule(STEP2, ctx["ids"], REPETITIONS, SEED),
                    lambda job: prompt2(ctx, world, job, found, facts, step2["decisions"]), transport, CALLS2,
                    concurrency=16, guard=guard, request=dict(engine.REQUEST))))
                stopped = bool(guard.reason)
            return 1 if stopped else 0
        finally:
            await client.aclose()

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
