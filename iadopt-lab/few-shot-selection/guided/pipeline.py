#!/usr/bin/env python
"""Step 3, end to end: walk, fill, check, repair, fall back, against today's setup.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/GUIDED_DECOMPOSITION_20261119.md` (private repository).

Arms, all on Qwen3.8-27B with the study's settings:

- `base`: today's prompt, one call.
- `base+retry`: today's prompt; an answer the lab's validator rejects goes back once with the
  validator's complaints. This is the control for the repair mechanism itself.
- `guided`: today's prompt plus guidance from the model's own stored walk, one call.
- `guided+check`: as `guided`; the answer is checked against the schema and against the form the
  guidance states; a failure goes back once with the complaints; if it still fails, a fresh
  answer from today's plain prompt is used instead.

The walk is done once per variable at temperature 0 by `walk.py` and reused in every repetition.

    ./pipeline.py --name e2e --style steps --features form,names,types --plan-only
    ./pipeline.py --name e2e --style steps --features form,names,types --run
    ./pipeline.py --name e2e --style steps --features form,names,types --score
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
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
import route_key  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402

ARMS = ("base", "base+retry", "guided", "guided+check")
BASELINE = "base"
REPETITIONS = 15
SCHEDULE_SEED = 20261120
REPAIR_INTRO = "Your answer has these problems:"
REPAIR_OUTRO = ("Return the corrected JSON object only, matching the schema, without prose, Markdown "
                "fences, quotation wrappers, or explanations.")

Transport = Callable[[Mapping[str, Any]], Awaitable[dict]]


def guided_prompt(ctx: Mapping[str, Any], definition: str, facts: Mapping[str, Any],
                  features: frozenset[str]) -> str:
    """Today's prompt with guidance from a walk appended to the definition.

    Args: ctx: from `load`; definition: the target definition; facts: the walk's facts;
        features: guidance features.
    Returns: the full prompt.
    Raises: ValueError for an unknown feature.
    Side Effects: none. Deterministic.
    """
    block = hints.render(dict(facts), features, ctx["patterns"], ctx["types"])
    return ft.prompt_for(ctx["inputs"], "B", definition + block)


def stated_form(ctx: Mapping[str, Any], facts: Mapping[str, Any] | None, features: frozenset[str]) -> dict[str, Any]:
    """The form the guidance states, which is all the check may demand.

    Args: ctx: from `load`; facts: a stored walk's facts, or None; features: guidance features.
    Returns: a subset of {"object", "matrix", "context", "statistical_modifier",
        "constraint_types"}; empty when there is no guidance.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if facts is None:
        return {}
    form = hints.expected_form(facts, ctx["patterns"], ctx["types"])
    out: dict[str, Any] = {}
    if "form" in features:
        out.update({k: form[k] for k in ("matrix", "context", "statistical_modifier")})
        if form["object"]:
            out["object"] = form["object"]
    if "types" in features:
        out["constraint_types"] = form["constraint_types"]
    return out


def check(ctx: Mapping[str, Any], answer: str | None, form: Mapping[str, Any]) -> tuple[dict | None, list[str]]:
    """Check an answer against the schema and against the stated form.

    Args: ctx: from `load`; answer: the model's reply, or None for a failed call; form: from
        `stated_form`, or {} to check the schema only.
    Returns: (valid prediction or None, complaints); no complaints means the answer passes.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    from iadopt_lab.generation.extractor import extract_json
    from iadopt_lab.validation import validate_prediction

    extracted = extract_json(answer or "")
    if not extracted.success:
        return None, ["The answer is not exactly one JSON object."]
    result = validate_prediction(extracted.candidate, ctx["inputs"]["schema"])
    if not result.valid:
        return None, [f"{error.get('pointer') or 'the answer'}: {error.get('message') or error.get('code')}"
                      for error in result.errors][:6]
    prediction = extracted.candidate
    out: list[str] = []
    if "object" in form and route_key.shape(prediction["hasObjectOfInterest"]) != form["object"]:
        out.append(f"hasObjectOfInterest must be {hints.SHAPE_TEXT[form['object']]}")
    if "matrix" in form and route_key.shape(prediction["hasMatrix"]) != form["matrix"]:
        out.append('hasMatrix must be "".' if form["matrix"] == "empty"
                   else f"hasMatrix must be {hints.SHAPE_TEXT[form['matrix']]}")
    if "context" in form and bool(prediction["hasContextObject"]) != form["context"]:
        out.append("hasContextObject must be one term." if form["context"] else 'hasContextObject must be "".')
    if "statistical_modifier" in form and bool(prediction["hasStatisticalModifier"]) != form["statistical_modifier"]:
        out.append("hasStatisticalModifier must be given." if form["statistical_modifier"]
                   else 'hasStatisticalModifier must be "".')
    if "constraint_types" in form:
        got = sorted({route_key.constraint_type(c["label"]) for c in prediction["hasConstraint"]})
        if got != list(form["constraint_types"]):
            out.append("hasConstraint must be []." if not form["constraint_types"] else
                       "hasConstraint must have one or more constraints of each of these types and of no "
                       f"other type: {'; '.join(form['constraint_types'])}.")
    return prediction, out


def repair_messages(prompt: str, answer: str | None, complaints: Sequence[str]) -> list[dict[str, str]]:
    """The conversation that sends an answer back with its problems.

    Args: prompt: the first prompt; answer: the first reply; complaints: from `check`.
    Returns: three chat messages: the prompt, the reply, the complaints.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    listed = "\n".join(f"- {text}" for text in complaints)
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer or ""},
            {"role": "user", "content": f"{REPAIR_INTRO}\n{listed}\n\n{REPAIR_OUTRO}"}]


async def attempt(ctx: Mapping[str, Any], job: engine.Job, transport: Transport, features: frozenset[str],
                  walks: Mapping[str, Mapping[str, Any]], guard: engine.Guard) -> dict[str, Any]:
    """Answer one job under its arm's rules, with at most three calls.

    Args: ctx: from `load`; job: (arm, repetition, variable_id); transport: sends one request;
        features: guidance features; walks: stored walks by variable id; guard: the stop rules.
    Returns: a call record whose "answer" is the final answer, with what happened on the way.
    Raises: KeyError for an unknown arm.
    Side Effects: one to three provider calls.
    """
    arm, repetition, variable_id = job
    definition = ctx["records"][variable_id]["definition"]
    plain = ft.prompt_for(ctx["inputs"], "B", definition)
    facts = walks.get(variable_id, {}).get("facts") if arm.startswith("guided") else None
    prompt = plain if facts is None else guided_prompt(ctx, definition, facts, features)
    form = stated_form(ctx, facts, features) if arm == "guided+check" else {}
    record: dict[str, Any] = {"arm": arm, "repetition": repetition, "variable_id": variable_id,
                              "started_at": ft.now(), "guided": facts is not None, "repaired": False,
                              "fell_back": False, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                              "latency_seconds": 0.0, "reasoning_chars": 0, "prompt_sha256": ft.sha256(prompt)}

    async def call(messages: list[dict[str, str]]) -> dict:
        """Send one request and add its cost to the record.

        Args: messages: the chat messages.
        Returns: the transport's result.
        Raises: nothing.
        Side Effects: one provider call; updates the record and the guard.
        """
        result = await transport(dict(engine.REQUEST, model=engine.MODEL, messages=messages))
        reasoning = result.get("reasoning") or ""
        record["calls"] += 1
        record["prompt_tokens"] += result.get("prompt_tokens") or 0
        record["completion_tokens"] += result.get("completion_tokens") or 0
        record["latency_seconds"] = round(record["latency_seconds"] + (result.get("latency") or 0.0), 3)
        record["reasoning_chars"] += len(reasoning)
        guard.record(dict(result, reasoning_chars=len(reasoning)))
        record.update(ok=bool(result.get("ok")), status=result.get("status"), error=result.get("error"),
                      served_model=result.get("served_model"), finish_reason=result.get("finish_reason"))
        return result

    first = await call([{"role": "user", "content": prompt}])
    answer = first.get("answer") if first.get("ok") else None
    record["first_answer"] = answer
    if arm in ("base+retry", "guided+check") and first.get("ok"):
        _, complaints = check(ctx, answer, form)
        record["first_complaints"] = complaints
        if complaints:
            second = await call(repair_messages(prompt, answer, complaints))
            record["repaired"] = True
            if second.get("ok"):
                answer = second.get("answer")
            _, still = check(ctx, answer, form) if second.get("ok") else (None, complaints)
            record["repair_complaints"] = still
            if still and arm == "guided+check":
                third = await call([{"role": "user", "content": plain}])
                record["fell_back"] = True
                if third.get("ok"):
                    answer = third.get("answer")
    record["answer"] = answer
    return record


async def run(ctx: Mapping[str, Any], jobs: Sequence[engine.Job], transport: Transport, calls_path: Path,
              features: frozenset[str], walks: Mapping[str, Mapping[str, Any]],
              concurrency: int = engine.CONCURRENCY, max_jobs: int | None = None) -> dict[str, Any]:
    """Answer every job not already answered, appending one record per job.

    Args: ctx: from `load`; jobs: from `engine.schedule`; transport: sends one request;
        calls_path: the records file; features, walks: as for `attempt`; concurrency: workers;
        max_jobs: cap on new jobs.
    Returns: {"issued", "skipped", "stopped", "list_price_usd"}.
    Raises: nothing from the provider.
    Side Effects: provider calls; appends to `calls_path`. Resumable: answered jobs are not
        repeated and failed ones are retried.
    """
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    guard = engine.Guard()
    answered = {key for key, record in engine.latest(calls_path).items() if record.get("ok")}
    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        if job not in answered:
            queue.put_nowait(job)
    lock = asyncio.Lock()
    counts = Counter()
    budget = [max_jobs if max_jobs is not None else len(jobs)]

    async def worker() -> None:
        """Take jobs until the queue is empty, honouring the stop rules and the budget.

        Args: none.
        Returns: nothing.
        Raises: nothing.
        Side Effects: provider calls; appends records under the lock.
        """
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if guard.reason or budget[0] <= 0:
                counts["skipped"] += 1
                continue
            budget[0] -= 1
            record = await attempt(ctx, job, transport, features, walks, guard)
            counts["issued"] += 1
            async with lock:
                with calls_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return {"issued": counts["issued"], "skipped": counts["skipped"], "stopped": guard.reason,
            "list_price_usd": round(guard.spend, 4)}


def paths(name: str) -> dict[str, Path]:
    """The files of one end-to-end comparison.

    Args: name: file prefix.
    Returns: {"calls", "scores", "summary"} paths under the ignored output folder.
    Raises: nothing.
    Side Effects: none.
    """
    return {kind: engine.OUTPUT / f"{name}-{kind}.{'json' if kind == 'summary' else 'jsonl'}"
            for kind in ("calls", "scores", "summary")}


def behaviour(calls: Mapping[engine.Job, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """How often each arm was guided, repaired and fell back, and its calls per answer.

    Args: calls: from `engine.latest`.
    Returns: {arm: {"guided", "repaired", "fell_back": shares, "calls_per_answer": mean}}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out: dict[str, dict[str, float]] = {}
    for arm in sorted({key[0] for key in calls}):
        rows = [record for key, record in calls.items() if key[0] == arm]
        out[arm] = {name: sum(bool(r.get(name)) for r in rows) / len(rows)
                    for name in ("guided", "repaired", "fell_back")}
        out[arm]["calls_per_answer"] = sum(r.get("calls") or 1 for r in rows) / len(rows)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode: plan, run or score.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status, 0 on success.
    Raises: SystemExit on bad arguments, a missing PSNC key or missing stored walks.
    Side Effects: --run calls the provider; --run and --score write output files.
    """
    parser = argparse.ArgumentParser(description="End-to-end guided decomposition (exploratory).")
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--plan-only", "--run", "--score"):
        mode.add_argument(flag, action="store_true")
    parser.add_argument("--name", required=True)
    parser.add_argument("--style", choices=walk.STYLES, required=True)
    parser.add_argument("--features", required=True, help="comma-separated guidance features")
    parser.add_argument("--population", choices=("practice", "held-out"), default="practice")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args(argv)
    ctx = walk.load()
    features = frozenset(args.features.split(","))
    arms = args.arms.split(",")
    variables = ctx["practice"] if args.population == "practice" else sorted(ctx["inputs"]["targets"])
    walk_name = "route" if args.population == "practice" else "route-held-out"
    walks = walk.stored(walk.results_path(args.style, walk_name))
    missing = [v for v in variables if v not in walks]
    if missing and any(arm.startswith("guided") for arm in arms):
        raise SystemExit(f"{len(missing)} variables have no stored {args.style} walk")
    jobs = engine.schedule(arms, variables, REPETITIONS, SCHEDULE_SEED)
    files = paths(args.name)

    if args.plan_only:
        guided = [guided_prompt(ctx, ctx["records"][v]["definition"], walks[v]["facts"], features)
                  for v in variables if walks.get(v, {}).get("facts")]
        print(json.dumps({"population": args.population, "variables": len(variables), "arms": arms,
                          "jobs": len(jobs), "style": args.style, "features": sorted(features),
                          "variables_with_facts": len(guided),
                          "pipeline.py": ceiling.file_sha256(HERE / "pipeline.py"),
                          "guided_prompts_sha256": ft.sha256("\n".join(ft.sha256(p) for p in guided)),
                          "walks_sha256": ft.sha256(json.dumps({v: walks[v]["facts"] for v in variables if v in walks},
                                                               sort_keys=True))}, indent=2))
        return 0
    if args.score:
        scores = engine.score(files["calls"], files["scores"], ctx["records"], ctx["inputs"]["schema"])
        calls = engine.latest(files["calls"])
        summary = engine.summarise(arms, BASELINE, variables, REPETITIONS, calls, scores)
        summary["behaviour"] = behaviour(calls)
        summary["settings"] = {"style": args.style, "features": sorted(features), "population": args.population}
        files["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for arm, cell in summary["cells"].items():
            how = summary["behaviour"].get(arm, {})
            print(f"{arm:14} reps {cell['repetitions']:2}  F1 {cell['close_f1_official_mean']:.4f} "
                  f"(SD {cell['close_f1_official_sd'] or 0:.4f})  invalid {cell['invalid']:3}  "
                  f"repaired {how.get('repaired', 0):.2f}  fell back {how.get('fell_back', 0):.2f}  "
                  f"calls/answer {how.get('calls_per_answer', 1):.2f}  ${cell['list_price_usd']:.2f}")
        for arm, result in summary["comparisons"].items():
            print(f"{arm:14} vs {BASELINE}: {result['difference']:+.4f} [{result['ci_low']:+.4f}, "
                  f"{result['ci_high']:+.4f}] holm p {result['p_holm']:.2g} -> {result['verdict']}")
        return 0

    async def go() -> int:
        """Open the transport, run the grid and always close the client.

        Args: none.
        Returns: the exit status.
        Raises: SystemExit when the PSNC key is missing.
        Side Effects: provider calls and output files.
        """
        transport, client = ft.psnc_transport()
        try:
            print(json.dumps(await run(ctx, jobs, transport, files["calls"], features, walks,
                                       max_jobs=args.max_jobs), indent=2))
            return 0
        finally:
            await client.aclose()

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
