#!/usr/bin/env python
"""Step 1, the ceiling test: how much would perfect pattern guidance add?

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/GUIDED_DECOMPOSITION_20261119.md` (private repository).

Every arm is today's best prompt (the example-selection study's 25 examples in the output-shaped
layout). The guided arms append a block after the target definition that states what a perfect
tree walk would have landed on, read from the expert answer key. If even that adds little, no
tree walk can add more.

    ./ceiling.py --plan-only    # fingerprints and call counts, no provider calls
    ./ceiling.py --dev          # the guidance wording on invented variables only
    ./ceiling.py --run          # the grid, resumable
    ./ceiling.py --score        # score the stored answers and write the summary

Population: the practice set, the 38 search variables plus the 15 pool variables that are not
among the 25 examples. The 24 held-out variables are never called here.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import design  # noqa: E402
import engine  # noqa: E402
import hints  # noqa: E402
import route_key  # noqa: E402
import rulebook  # noqa: E402
from engine import ft  # noqa: E402

ARMS = {"base": frozenset(), "form": frozenset({"form"}), "form+names": frozenset({"form", "names"}),
        "form+types": frozenset({"form", "types"}),
        "form+names+types": frozenset({"form", "names", "types"}),
        "full": frozenset({"form", "names", "types", "texts"})}
BASELINE = "base"
REPETITIONS = 15
SCHEDULE_SEED = 20261119
GATE = 0.05
CALLS = engine.OUTPUT / "ceiling-calls.jsonl"
SCORES = engine.OUTPUT / "ceiling-scores.jsonl"
SUMMARY = engine.OUTPUT / "ceiling-summary.json"
DEV_CALLS = engine.OUTPUT / "ceiling-dev-calls.jsonl"
DEV_ARMS = ("form", "form+names+types", "full")
DEV_REPETITIONS = 2

# Invented variables for developing the guidance wording. None is a corpus variable. The facts
# are the right answers, written down before the first call: an answer is right when it has the
# form these facts imply.
INVENTED = (
    {"id": "dev-1", "definition": "Number of bicycles passing a counting station per hour",
     "property": [124], "object": [33], "matrix": [], "context": [], "constraint": [92]},
    {"id": "dev-2", "definition": "Mass fraction of microplastic fibres in beach sand",
     "property": [110], "object": [35], "matrix": [], "context": [], "constraint": []},
    {"id": "dev-3", "definition": "Daily maximum of the noise annoyance index in a residential district",
     "property": [132], "object": [33], "matrix": [44], "context": [], "constraint": [83],
     "statistical_modifier": True},
    {"id": "dev-4", "definition": "Mass flux of fine sediment from the river into the delta",
     "property": [130], "object": [33], "matrix": [45], "context": [], "constraint": []},
    {"id": "dev-5", "definition": "Temperature of the coolant water in a reactor cooling loop",
     "property": [109], "object": [33], "matrix": [42], "context": [], "constraint": []},
    {"id": "dev-6", "definition": "Angle between the roof ridge line of a building and true north",
     "property": [116], "object": [34], "matrix": [], "context": [], "constraint": []},
    {"id": "dev-7", "definition": "Mass concentration of dissolved nitrate in groundwater in an agricultural catchment",
     "property": [128], "object": [35], "matrix": [], "context": [50], "constraint": [76]},
    {"id": "dev-8", "definition": "Colour of the petals of an orchid flower",
     "property": [101], "object": [33], "matrix": [], "context": [], "constraint": [94]},
)


def practice_ids(records: dict[str, dict[str, Any]]) -> list[str]:
    """The practice set: search variables plus the pool variables that are not examples.

    Args: records: corpus records by id.
    Returns: 53 sorted variable ids, none of them an example or a held-out variable.
    Raises: ValueError when the split does not give 53 variables or overlaps the examples.
    Side Effects: none. Deterministic.
    """
    split = design.corpus_split(list(records.values()))
    ids = sorted(set(split["E"]) | (set(split["P"]) - set(ft.EXAMPLE_IDS)))
    if len(ids) != 53 or set(ids) & set(ft.EXAMPLE_IDS) or set(ids) & set(split["C"]):
        raise ValueError("the practice set is not the expected 53 variables")
    return ids


def load() -> dict[str, Any]:
    """Everything the ceiling test needs. No network, no secrets.

    Args: none.
    Returns: {"inputs", "records", "practice", "patterns", "types", "key"}.
    Raises: ValueError from the input checks; OSError when the data has not been built.
    Side Effects: reads local files.
    """
    inputs = ft.load_inputs()
    return {"inputs": inputs, "records": inputs["records"], "practice": practice_ids(inputs["records"]),
            "patterns": rulebook.load("patterns")["patterns"],
            "types": rulebook.load("constraint_types")["types"], "key": route_key.load()}


def guided_prompt(ctx: dict[str, Any], definition: str, facts: dict[str, Any], arm: str) -> str:
    """Today's prompt with one arm's guidance appended to the definition.

    Args: ctx: from `load`; definition: the target definition; facts: what the walk landed on;
        arm: an arm name.
    Returns: the full prompt; for the baseline arm it is today's prompt, byte for byte.
    Raises: KeyError for an unknown arm.
    Side Effects: none. Deterministic.
    """
    block = hints.render(facts, ARMS[arm], ctx["patterns"], ctx["types"])
    return ft.prompt_for(ctx["inputs"], "B", definition + block)


def prompt_of(ctx: dict[str, Any], job: engine.Job) -> str:
    """The prompt of one grid call, with facts from the expert answer key.

    Args: ctx: from `load`; job: (arm, repetition, variable_id).
    Returns: the full prompt.
    Raises: KeyError for an unknown arm or variable.
    Side Effects: none. Deterministic.
    """
    arm, _, variable_id = job
    facts = hints.facts_from_key(ctx["key"][variable_id])
    return guided_prompt(ctx, ctx["records"][variable_id]["definition"], facts, arm)


def file_sha256(path: Path) -> str:
    """Hash a file.

    Args: path: the file.
    Returns: its SHA-256 hex digest.
    Raises: OSError when it cannot be read.
    Side Effects: reads the file.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(ctx: dict[str, Any]) -> dict[str, Any]:
    """Hash what the test sends: code, rulebook data, population and every prompt per arm.

    Args: ctx: from `load`.
    Returns: hex digests and counts.
    Raises: nothing.
    Side Effects: reads this folder's files. Deterministic.
    """
    out: dict[str, Any] = {name: file_sha256(HERE / name)
                           for name in ("ceiling.py", "engine.py", "hints.py", "rulebook.py", "route_key.py")}
    out.update({f"data/{name}": file_sha256(rulebook.DATA / name)
                for name in ("trees.json", "patterns.json", "constraint_types.json", "route_key.json")})
    out["practice_sha256"] = ft.sha256("\n".join(ctx["practice"]))
    for arm in ARMS:
        prompts = [prompt_of(ctx, (arm, 1, v)) for v in ctx["practice"]]
        out[f"prompts[{arm}]"] = ft.sha256("\n".join(ft.sha256(p) for p in prompts))
        out[f"prompt_chars_mean[{arm}]"] = round(sum(len(p) for p in prompts) / len(prompts), 1)
    return out


def conformance_rates(ctx: dict[str, Any], scores: dict[engine.Job, dict]) -> dict[str, dict[str, float]]:
    """How often each arm's answers have the form the answer key demands.

    Args: ctx: from `load`; scores: from `engine.score`.
    Returns: {arm: {part: share of answers}}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    totals: dict[str, Counter] = {arm: Counter() for arm in ARMS}
    for (arm, _, variable_id), row in scores.items():
        if arm not in totals or variable_id not in ctx["key"]:
            continue
        form = hints.expected_form(hints.facts_from_key(ctx["key"][variable_id]), ctx["patterns"], ctx["types"])
        for part, ok in hints.conformance(row["prediction"], form).items():
            totals[arm][part] += ok
        totals[arm]["n"] += 1
    return {arm: {part: count / c["n"] for part, count in c.items() if part != "n"}
            for arm, c in totals.items() if c["n"]}


async def dev(ctx: dict[str, Any], transport: engine.Transport) -> list[dict]:
    """Try the guidance wording on the invented variables.

    Args: ctx: from `load`; transport: sends one request.
    Returns: one row per call with the answer and its conformance to the invented facts.
    Raises: nothing from the provider.
    Side Effects: provider calls for invented definitions only; appends to the dev calls file.
    """
    from iadopt_lab.generation.extractor import extract_json
    from iadopt_lab.validation import validate_prediction

    items = {item["id"]: dict(item, statistical_modifier=item.get("statistical_modifier", False), names={})
             for item in INVENTED}
    jobs = engine.schedule(DEV_ARMS, list(items), DEV_REPETITIONS, SCHEDULE_SEED)
    await engine.run_jobs(jobs, lambda job: guided_prompt(ctx, items[job[2]]["definition"], items[job[2]], job[0]),
                          transport, DEV_CALLS, concurrency=8)
    rows = []
    for key, record in sorted(engine.latest(DEV_CALLS).items()):
        prediction = None
        extracted = extract_json(record.get("answer") or "")
        if extracted.success and validate_prediction(extracted.candidate, ctx["inputs"]["schema"]).valid:
            prediction = extracted.candidate
        form = hints.expected_form(items[key[2]], ctx["patterns"], ctx["types"])
        rows.append({"key": key, "prediction": prediction, "conformance": hints.conformance(prediction, form)})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode: plan, dev, run or score.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status, 0 on success.
    Raises: SystemExit on bad arguments or a missing PSNC key.
    Side Effects: --dev and --run call the provider; --dev, --run and --score write output files.
    """
    parser = argparse.ArgumentParser(description="Ceiling test for pattern guidance (exploratory).")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--plan-only", "--dev", "--run", "--score"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--max-calls", type=int, default=None, help="issue at most this many new calls")
    args = parser.parse_args(argv)
    ctx = load()
    jobs = engine.schedule(list(ARMS), ctx["practice"], REPETITIONS, SCHEDULE_SEED)

    if args.plan_only:
        print(json.dumps({"practice_variables": len(ctx["practice"]), "arms": list(ARMS), "calls": len(jobs),
                          **fingerprints(ctx)}, indent=2))
        return 0
    if args.score:
        scores = engine.score(CALLS, SCORES, ctx["records"], ctx["inputs"]["schema"])
        summary = engine.summarise(list(ARMS), BASELINE, ctx["practice"], REPETITIONS, engine.latest(CALLS), scores)
        summary["conformance"] = conformance_rates(ctx, scores)
        SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for arm, cell in summary["cells"].items():
            print(f"{arm:18} reps {cell['repetitions']:2}  F1 {cell['close_f1_official_mean']:.4f} "
                  f"(SD {cell['close_f1_official_sd'] or 0:.4f})  invalid {cell['invalid']:3}  "
                  f"form ok {summary['conformance'].get(arm, {}).get('all', 0):.2f}  "
                  f"prompt {cell['prompt_tokens_mean']:.0f} tok  ${cell['list_price_usd']:.2f}")
        for arm, result in summary["comparisons"].items():
            print(f"{arm:18} vs {BASELINE}: {result['difference']:+.4f} [{result['ci_low']:+.4f}, "
                  f"{result['ci_high']:+.4f}] holm p {result['p_holm']:.2g} -> {result['verdict']}")
        return 0

    async def go() -> int:
        """Open the transport, run the dev check or the grid, and always close the client.

        Args: none.
        Returns: the exit status.
        Raises: SystemExit when the PSNC key is missing.
        Side Effects: provider calls and output files.
        """
        transport, client = ft.psnc_transport()
        try:
            if args.dev:
                for row in await dev(ctx, transport):
                    flags = "".join("Y" if ok else "n" for ok in row["conformance"].values())
                    print(row["key"], flags, json.dumps(row["prediction"], ensure_ascii=False))
                return 0
            result = await engine.run_jobs(jobs, lambda job: prompt_of(ctx, job), transport, CALLS,
                                           max_calls=args.max_calls)
            print(json.dumps(result, indent=2))
            return 0
        finally:
            await client.aclose()

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
