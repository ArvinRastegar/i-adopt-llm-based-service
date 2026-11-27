#!/usr/bin/env python
"""Three changes to today's fill call, in one run: the most typical of three, a re-ask, a short note first.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/FILL_LEVERS_TEST_20261127.md` (private repository).

Arvin, 2026-11-27: "run until work that needs people". Every call goes to PSNC (`Qwen3.8-27B`, the study's
fill settings). For each variable and run, one unit sends, side by side, today's best prompt three times
and the note-first variant once; when the first answer fails the schema it is sent back once.

- `best`: the first of the three answers (today's pipeline);
- `best+typical3`: of the three, the one that agrees most with the other two (`typical_answer_check.pick`,
  agreement by the frozen scorer's Close F1 both ways), chosen when scoring;
- `best+retry`: the first answer, or its re-ask when it failed the schema and the re-ask passes;
- `best+form`: the note-first variant: three short plain lines, then the JSON object.

    ./fill_levers_test.py --plan-only
    ./fill_levers_test.py --dev           # both forms on invented variables only
    ./fill_levers_test.py --run
    ./fill_levers_test.py --score
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
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
import settings  # noqa: E402
import stand_in_test as si  # noqa: E402
import stronger_reader_test as sr  # noqa: E402
import typical_answer_check as ta  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402

pipeline = settings.pipeline        # this folder's pipeline.py, loaded by path, as stage 5 does (`rag_tuning.py`)
ARMS = ("best", "best+typical3", "best+retry", "best+form")
SAMPLES = ("best", "best#2", "best#3")
REPETITIONS, SEED = 10, 20261261
UNITS_AT_ONCE = 6                                   # four calls each, so about 24 in flight
LIMIT_SECONDS = 30.0
PLAIN_HEAD = "Output only the JSON object."
PLAIN_TAIL = ("Return exactly one JSON object matching the schema, without prose, Markdown fences, quotation wrappers, "
              "or explanations.")
NOTE_HEAD = ("First write three short lines in plain words: what is measured and of what; what it is in or part of, "
             "if anything; the words that narrow it down. Then output the JSON object.")
NOTE_TAIL = ("Write the three short lines, then exactly one JSON object matching the schema, without Markdown fences "
             "or quotation wrappers.")
OUT = HERE.parent / "output" / "retrieval"
CALLS = OUT / "fill-levers-calls.jsonl"
SCORES = OUT / "fill-levers-scores.jsonl"
DEV = OUT / "fill-levers-dev-calls.jsonl"
PICKS = OUT / "fill-levers-typical3.json"
SUMMARY = OUT / "fill-levers-summary.json"


def note_first(prompt: str) -> str:
    """Today's prompt with the note-first output instructions.

    Args: prompt: a fill prompt with today's two output instructions.
    Returns: the prompt with both replaced.
    Raises: ValueError when either instruction is not there exactly once.
    Side Effects: none. Deterministic.
    """
    if prompt.count(PLAIN_HEAD) != 1 or prompt.count(PLAIN_TAIL) != 1:
        raise ValueError("the prompt does not carry today's output instructions once each")
    return prompt.replace(PLAIN_HEAD, NOTE_HEAD).replace(PLAIN_TAIL, NOTE_TAIL)


def body(messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """A fill request.

    Args: messages: the chat messages.
    Returns: the study's fill settings with the model and the messages.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return dict(engine.REQUEST, model=engine.MODEL, messages=list(messages))


def record(arm: str, run: int, v: str, result: Mapping[str, Any], prompt: str, started: str, **extra: Any) -> dict[str, Any]:
    """A call record in the study's form.

    Args: arm, run, v: the job; result: the transport's result; prompt: what was sent (hashed);
        started: the start time; extra: fields to add.
    Returns: the record.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    reasoning = result.get("reasoning") or ""
    return {"arm": arm, "repetition": run, "variable_id": v, "started_at": started, "ok": bool(result.get("ok")),
            "status": result.get("status"), "error": result.get("error"), "answer": result.get("answer"),
            "reasoning_chars": len(reasoning), "reasoning": reasoning or None, "finish_reason": result.get("finish_reason"),
            "prompt_tokens": result.get("prompt_tokens"), "completion_tokens": result.get("completion_tokens"),
            "latency_seconds": round(result.get("latency") or 0.0, 3), "served_model": result.get("served_model"),
            "prompt_sha256": ft.sha256(prompt), **extra}


async def unit(ctx2: Mapping[str, Any], prompt: str, run: int, v: str, transport: Any) -> list[dict[str, Any]]:
    """One variable and run: three plain calls and one note-first call side by side, then the re-ask if needed.

    Args: ctx2: from `rag_tuning.load` (for the schema check); prompt: today's best prompt; run, v: the
        job; transport: PSNC's.
    Returns: records for "best", "best#2", "best#3", "best+form" and "best+retry".
    Raises: nothing from the provider.
    Side Effects: four or five PSNC calls.
    """
    started = ft.now()
    form = note_first(prompt)
    results = await asyncio.gather(*(transport(body([{"role": "user", "content": p}])) for p in (prompt, prompt, prompt, form)))
    out = [record(arm, run, v, res, p, started) for arm, res, p in zip((*SAMPLES, "best+form"), results, (prompt,) * 3 + (form,), strict=True)]
    first = results[0].get("answer") if results[0].get("ok") else None
    _, complaints = dr.check(ctx2, first, {})
    final, extra = results[0], {"repaired": False, "repair_valid": None, "first_sha256": ft.sha256(first or ""),
                                "latency_total": round(results[0].get("latency") or 0.0, 3)}
    if complaints:
        second = await transport(body(pipeline.repair_messages(prompt, first, complaints)))
        valid = second.get("ok") and dr.check(ctx2, second.get("answer"), {})[0] is not None
        extra.update(repaired=True, repair_valid=bool(valid), complaints=complaints,
                     latency_total=round((results[0].get("latency") or 0.0) + (second.get("latency") or 0.0), 3))
        if valid:
            final = dict(second, latency=results[0].get("latency"))
    out.append(record("best+retry", run, v, final, prompt, started, **extra))
    return out


def summarise(ctx: Mapping[str, Any], ctx2: Mapping[str, Any]) -> dict[str, Any]:
    """Score every answer, choose the most typical of three, and compare the four arms with `best`.

    Args: ctx: from `answer_form_check.load`; ctx2: from `rag_tuning.load`.
    Returns: the constraint-types test's Step 2 summary for `ARMS`, with latencies (each arm's unit time
        and, with the finder's 95th percentile on PSNC, the full response) and the re-ask counts.
    Raises: KeyError when a score is missing.
    Side Effects: loads the scorer's similarity model; writes `PICKS`; appends new scores.
    """
    from iadopt_eval import evaluate_item
    from iadopt_lab.cli import _configuration, _similarity_backend

    scores = engine.score(CALLS, SCORES, ctx2["records"], ctx2["inputs"]["schema"])
    similarity = _similarity_backend(engine.ROOT, _configuration(engine.ROOT).data["evaluation"])
    picks: dict[str, Any] = {}
    seconds = []
    for run in range(1, REPETITIONS + 1):
        for v in ctx2["ids"]:
            if not all((arm, run, v) in scores for arm in SAMPLES):
                continue
            clock = time.perf_counter()
            table = {}
            for a in range(3):
                for b in range(a + 1, 3):
                    first, second = scores[(SAMPLES[a], run, v)]["prediction"], scores[(SAMPLES[b], run, v)]["prediction"]
                    both = [ta.f1(evaluate_item(g, p, similarity, metadata={"variable_id": v},
                                                similarity_identity=similarity.identity)["close"]["totals"])
                            for g, p in ((first, second), (second, first))] if first is not None and second is not None else [0.0, 0.0]
                    table[(a + 1, b + 1)] = sum(both) / 2
            chosen = ta.pick((1, 2, 3), table)
            seconds.append(time.perf_counter() - clock)
            picks[f"{run}|{v}"] = {"chosen": chosen, "agreements": {f"{a}-{b}": x for (a, b), x in table.items()}}
            scores[("best+typical3", run, v)] = scores[(SAMPLES[chosen - 1], run, v)]
    PICKS.write_text(json.dumps(picks, indent=0) + "\n", encoding="utf-8")
    summary = cy.summarise2(ctx2, {k: s for k, s in scores.items() if k[0] in ARMS}, ARMS)
    calls = engine.latest(CALLS)

    def spread(values: Sequence[float]) -> dict[str, float] | None:
        """Median, 95th percentile and maximum.

        Args: values: seconds.
        Returns: the three figures, or None for no values.
        Raises: nothing.
        Side Effects: none.
        """
        return {"p50": statistics.median(values), "p95": sr.percentile(values, 0.95), "max": max(values)} if values else None

    units = {(run, v) for (arm, run, v) in calls if arm == "best"}
    times = {"best": [calls[("best", r, v)]["latency_seconds"] for r, v in units],
             "best+typical3": [max(calls[(s, r, v)]["latency_seconds"] for s in SAMPLES) for r, v in units
                               if all((s, r, v) in calls for s in SAMPLES)],
             "best+retry": [calls[("best+retry", r, v)]["latency_total"] for r, v in units if ("best+retry", r, v) in calls],
             "best+form": [calls[("best+form", r, v)]["latency_seconds"] for r, v in units if ("best+form", r, v) in calls]}
    finder = json.loads(sr.SUMMARY.read_text(encoding="utf-8"))["latency"]["qwen3.8"]["p95"]
    summary["latency"] = {arm: spread(t) for arm, t in times.items()}
    summary["latency"]["finder_p95"] = finder
    summary["latency"]["typical3_choice_seconds"] = spread(seconds)
    summary["full_response_p95"] = {arm: (row["p95"] + finder if row else None) for arm, row in summary["latency"].items()
                                    if arm in ARMS}
    retried = [calls[("best+retry", r, v)] for r, v in units if ("best+retry", r, v) in calls]
    summary["retry"] = {"units": len(retried), "repaired": sum(bool(x.get("repaired")) for x in retried),
                        "repair_valid": sum(bool(x.get("repair_valid")) for x in retried)}
    return summary


async def dev(transport: Any, guard: si.Guard) -> list[dict[str, Any]]:
    """Both forms on stage 3's invented variables, with the 25 corpus variables sharing most words as examples.

    Args: transport: PSNC's; guard: the stop rules.
    Returns: one row per invented variable and form: usable, Close F1 against the invented gold, the answer's start.
    Raises: nothing from the provider.
    Side Effects: PSNC calls for invented definitions only; appends to `DEV`; loads the scorer.
    """
    from iadopt_eval import evaluate_item
    from iadopt_lab.cli import _configuration, _similarity_backend

    ctx = afc.load()
    ctx2 = rt.load()
    world = rt.corpus_world(ctx2, None)
    items = ct.invented(ctx)

    def similar(first: str, second: str) -> float:
        """Shared words over all words.

        Args: first, second: texts.
        Returns: the share.
        Raises: nothing.
        Side Effects: none.
        """
        a, b = set(kn.words(first)), set(kn.words(second))
        return len(a & b) / len(a | b) if a | b else 0.0

    def plain(name: str) -> str:
        """Today's prompt for an invented variable, without the matrix statement.

        Args: name: its id.
        Returns: the prompt.
        Raises: nothing.
        Side Effects: none.
        """
        text = items[name]["definition"]
        ids = sorted(ctx2["ids"], key=lambda u: (-similar(text, ctx2["records"][u]["definition"]), u))[:25][::-1]
        inputs = dict(world["base_inputs"](ctx2["ids"][0]), examples=[world["record"](u) for u in ids])
        return dr.fill_prompt(inputs, text)

    jobs = [(arm, 1, name) for name in sorted(items) for arm in ("plain", "note")]
    await engine.run_jobs(jobs, lambda job: plain(job[2]) if job[0] == "plain" else note_first(plain(job[2])), transport, DEV,
                          concurrency=8, guard=guard, request=dict(engine.REQUEST))
    similarity = _similarity_backend(engine.ROOT, _configuration(engine.ROOT).data["evaluation"])
    rows = []
    for (arm, _, name), rec in sorted(engine.latest(DEV).items()):
        prediction, _ = dr.check(ctx2, rec.get("answer") if rec.get("ok") else None, {})
        gold = kn.INVENTED_GOLD[name]
        full = {k: gold.get(k, [] if k == "hasConstraint" else "") for k in
                ("hasStatisticalModifier", "hasProperty", "hasObjectOfInterest", "hasMatrix", "hasContextObject", "hasConstraint")}
        f = ta.f1(evaluate_item(full, prediction, similarity, metadata={"variable_id": name},
                                similarity_identity=similarity.identity)["close"]["totals"]) if prediction else 0.0
        rows.append({"arm": arm, "name": name, "usable": prediction is not None, "f1": f,
                     "tokens": rec.get("completion_tokens"), "latency": rec.get("latency_seconds"), "answer": rec.get("answer") or ""})
    return rows


def fingerprints(ctx: Mapping[str, Any], world: Mapping[str, Any], found: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the code and the two prompt sets.

    Args: ctx: from `answer_form_check.load`; world: stage 5's; found: the stored finder's facts.
    Returns: hex digests (16 characters) and mean lengths.
    Raises: OSError when a file cannot be read.
    Side Effects: reads files.
    """
    out: dict[str, Any] = {name: ceiling.file_sha256(HERE / name)[:16] for name in (
        "fill_levers_test.py", "typical_answer_check.py", "answer_form_check.py", "draft_rag.py", "pipeline.py",
        "rag_tuning.py", "engine.py", "walk.py")}
    plain = [afc.prompt_b(ctx, world, ("rag-25+finder-matrix", 1, v), found) for v in ctx["ids"]]
    out["prompts"] = {"best": ft.sha256("\n".join(ft.sha256(p) for p in plain))[:16],
                      "best+form": ft.sha256("\n".join(ft.sha256(note_first(p)) for p in plain))[:16],
                      "chars_mean": round(sum(len(p) for p in plain) / len(plain), 1)}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode.

    Args: argv: arguments; `sys.argv` when None.
    Returns: 0 on success; 1 when a stop rule ended the run or the summary is not complete.
    Raises: SystemExit on bad arguments.
    Side Effects: --dev and --run call PSNC; --score writes the summary.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--plan-only", "--dev", "--run", "--score"):
        mode.add_argument(name, action="store_true")
    args = parser.parse_args(argv)
    ctx = afc.load()
    if args.plan_only or args.run:
        world = rt.corpus_world(rt.load(), None)
        found = afc.finder_facts(ctx, engine.latest(afc.CALLS_A))
    if args.plan_only:
        print(json.dumps({"provider": "PSNC", "model": engine.MODEL, "arms": ARMS, "units": len(ctx["ids"]) * REPETITIONS,
                          "request": engine.REQUEST, **fingerprints(ctx, world, found)}, indent=2))
        return 0
    if args.score:
        summary = summarise(ctx, rt.load())
        SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for arm, row in summary.get("arms", {}).items():
            print(f"{arm:14} F1 {row['f1_mean']:.4f} sd {row['f1_sd']:.4f} invalid {row['invalid']}")
        for arm, row in summary.get("against_best", {}).items():
            print(f"{arm} - best: {row['gain']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
        print("latency:", json.dumps(summary["latency"]), "\nfull response p95:", summary["full_response_p95"], "\nretry:", summary["retry"])
        return 0 if summary["complete"] else 1

    async def go() -> int:
        """Open PSNC's transport, run the asked part, always close the client.

        Args: none.
        Returns: 1 when a stop rule ended the work, else 0.
        Raises: SystemExit when the key is missing.
        Side Effects: PSNC calls and the calls files.
        """
        guard = si.Guard()
        transport, client = walk.chat_transport(concurrency=UNITS_AT_ONCE * 4)
        try:
            if args.dev:
                for row in await dev(transport, guard):
                    print(row["arm"], row["name"], "usable" if row["usable"] else "UNUSABLE", f"F1 {row['f1']:.3f}",
                          f"{row['tokens']} tokens {row['latency']}s |", row["answer"].replace("\n", " ")[:120])
                return 1 if guard.reason else 0
            ctx2 = rt.load()
            done = {k for k, r in engine.latest(CALLS).items() if r.get("ok")}
            units = [(run, v) for (_, run, v) in engine.schedule(("best",), ctx["ids"], REPETITIONS, SEED)
                     if not all((arm, run, v) in done for arm in (*SAMPLES, "best+form", "best+retry"))]
            queue: asyncio.Queue = asyncio.Queue()
            for u in units:
                queue.put_nowait(u)
            lock = asyncio.Lock()

            async def worker() -> None:
                """Take units until the queue is empty or a stop rule fires.

                Args: none.
                Returns: nothing.
                Raises: nothing.
                Side Effects: PSNC calls; appends records under the lock.
                """
                while not guard.reason:
                    try:
                        run, v = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    prompt = afc.prompt_b(ctx, world, ("rag-25+finder-matrix", run, v), found)
                    records = await unit(ctx2, prompt, run, v, transport)
                    async with lock:
                        with CALLS.open("a", encoding="utf-8") as handle:
                            for rec in records:
                                guard.record(rec)
                                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

            await asyncio.gather(*(worker() for _ in range(UNITS_AT_ONCE)))
            print(json.dumps({"units": len(units), "stopped": guard.reason}))
            return 1 if guard.reason else 0
        finally:
            await client.aclose()

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
