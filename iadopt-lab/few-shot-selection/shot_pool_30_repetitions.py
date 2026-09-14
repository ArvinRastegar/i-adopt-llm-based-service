#!/usr/bin/env python
"""Repeated measurements at 20/25/30 shots — EXPLORATORY, not the official experiment.

The single-repetition thirty-shot run put 20, 25 and 30 shots within 0.004 Close F1 of each
other, which one repetition cannot separate from sampling noise. This repeats each of those
three levels ten times so the spread can be measured instead of guessed.

Method
------
Ten repetitions per level, all collected identically: same model, same configuration, same
thirty-variable pool, same 72 evaluation variables. No seed is sent, so the variation is the
provider's own sampling at temperature 0.5 - which is exactly the quantity an error bar on
these numbers should describe.

The earlier single run is deliberately NOT reused as repetition 1. Ten observations gathered
the same way cost three extra minutes and remove any question about whether the first one was
collected under different conditions.

Reported metric is micro Close F1, aggregated as the official experiment aggregates. Both
scoring rules are reported: `as-run` drops predictions that fail validation, `official-style`
scores them as an explicit empty prediction and keeps the variable in the population, which is
what the official pipeline does. The two differ by up to 0.013 here.

Isolation
---------
Same as its siblings: lives in `few-shot-selection/`, which no artifact collector walks; writes no
campaign, run or task row; writes only `few-shot-selection/output/`; cannot enter an official ranking.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "few-shot-selection"))

import httpx  # noqa: E402

# Reused so the pool, the prompt bytes and the evaluation population cannot drift.
from shot_count_ablation import call, render, shot_pool  # noqa: E402
from shot_pool_30_ablation import CONFIG, MODEL_ID, extend_pool  # noqa: E402

from iadopt_eval import evaluate_item  # noqa: E402
from iadopt_eval.core import aggregate_items  # noqa: E402
from iadopt_lab.configuration import load_runtime_secrets  # noqa: E402
from iadopt_lab.corpus.ingestion import load_canonical_records  # noqa: E402
from iadopt_lab.generation.extractor import extract_json  # noqa: E402
from iadopt_lab.prompting.renderer import load_prompt_version  # noqa: E402
from iadopt_lab.validation import load_schema_bytes, validate_prediction  # noqa: E402
from iadopt_lab.validation.lexical import empty_prediction  # noqa: E402

EXPERIMENT = "shot-pool-30-repetitions-v1"
OUT_DIR = ROOT / "few-shot-selection" / "output"
RESULTS = OUT_DIR / "shot-pool-30-repetitions-results.jsonl"
SHOT_LEVELS = (20, 25, 30)
REPETITIONS = 10

CONCURRENCY = 8
TIMEOUT = 120.0


def done_keys() -> set:
    """(repetition, shots, variable_id) triples already scored, so nothing is called twice."""
    if not RESULTS.exists():
        return set()
    return {(row["repetition"], row["shot_count"], row["variable_id"])
            for row in (json.loads(line)
                        for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip())}


def _interval(values: list[float]) -> tuple[float, float, float]:
    """Mean, sample standard deviation and standard error for one level's repetitions.

    Args: values: per-repetition micro Close F1, at least one.
    Returns: (mean, sd, sem); sd and sem are 0.0 for a single observation.
    Side Effects: none.
    """
    if len(values) < 2:
        return (values[0], 0.0, 0.0)
    sd = statistics.stdev(values)
    return (statistics.mean(values), sd, sd / len(values) ** 0.5)


def summarise() -> None:
    """Print per-repetition and aggregated micro Close F1 under both scoring rules."""
    if not RESULTS.exists():
        return
    sys.path.insert(0, str(ROOT / "src"))
    from iadopt_lab.cli import _configuration, _similarity_backend
    similarity = _similarity_backend(ROOT, _configuration(ROOT).data["evaluation"])
    gold = {row["variable_id"]: row["gold"] for row in load_canonical_records(ROOT)}

    rows = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["shot_count"], row["repetition"])].append(row)

    as_run, official = defaultdict(list), defaultdict(list)
    print(f"\nper-repetition micro Close F1 ({len({r['variable_id'] for r in rows})} evaluation variables)\n")
    print(f"{'shots':>6}{'rep':>5}{'scored':>9}{'as-run':>10}{'official':>10}")
    for shots in SHOT_LEVELS:
        for rep in range(1, REPETITIONS + 1):
            batch = grouped.get((shots, rep))
            if not batch:
                continue
            scored = [r for r in batch if r.get("evaluation")]
            a = aggregate_items([r["evaluation"] for r in scored],
                                expected_variable_ids=[r["variable_id"] for r in scored])
            evidence = [r["evaluation"] for r in scored]
            ids = [r["variable_id"] for r in scored]
            for r in batch:
                if not r.get("evaluation"):
                    evidence.append(evaluate_item(gold[r["variable_id"]], empty_prediction(), similarity,
                                                  metadata={"variable_id": r["variable_id"]},
                                                  similarity_identity=similarity.identity))
                    ids.append(r["variable_id"])
            f = aggregate_items(evidence, expected_variable_ids=ids)
            x = a["close"]["metrics"]["f1"]["value"]
            y = f["close"]["metrics"]["f1"]["value"]
            as_run[shots].append(x)
            official[shots].append(y)
            print(f"{shots:>6}{rep:>5}{len(scored):>4}/{len(batch):<4}{x:>10.4f}{y:>10.4f}")

    for name, series in (("as-run", as_run), ("official-style", official)):
        print(f"\n{name} micro Close F1 across repetitions\n")
        print(f"{'shots':>6}{'n':>4}{'mean':>9}{'sd':>9}{'sem':>9}{'min':>9}{'max':>9}{'95% CI':>20}")
        for shots in SHOT_LEVELS:
            values = series[shots]
            if not values:
                continue
            mean, sd, sem = _interval(values)
            lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
            print(f"{shots:>6}{len(values):>4}{mean:>9.4f}{sd:>9.4f}{sem:>9.4f}"
                  f"{min(values):>9.4f}{max(values):>9.4f}{f'[{lo:.4f}, {hi:.4f}]':>20}")


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = list(load_canonical_records(ROOT))
    pool = extend_pool(records, shot_pool(records))
    reserved = {row["variable_id"] for row in pool}
    targets = [row for row in records if row["variable_id"] not in reserved]
    schema = load_schema_bytes(ROOT)
    schema_text = schema.decode("utf-8")

    planned = REPETITIONS * len(SHOT_LEVELS) * len(targets)
    already = done_keys()
    print(f"{EXPERIMENT}: {MODEL_ID}, shots {SHOT_LEVELS}, {REPETITIONS} repetitions, "
          f"{len(targets)} evaluation variables")
    print(f"planned: {REPETITIONS} x {len(SHOT_LEVELS)} x {len(targets)} = {planned}")
    print(f"already complete: {len(already)}   remaining: {planned - len(already)}")
    if "--plan-only" in sys.argv:
        return 0
    if "--summary-only" in sys.argv:
        summarise()
        return 0

    secrets = load_runtime_secrets(str(ROOT.parent / ".env"),
                                   ["PSNC_API_KEY", "PSNC_API_BASE_URL"], os.environ)
    base_url = (secrets.get("PSNC_API_BASE_URL") or "https://llm.hpc.psnc.pl").rstrip("/")
    url = base_url + "/chat/completions"
    headers = {"Authorization": "Bearer " + secrets.get("PSNC_API_KEY"), "Content-Type": "application/json"}

    from iadopt_lab.cli import _configuration, _similarity_backend
    similarity = _similarity_backend(ROOT, _configuration(ROOT).data["evaluation"])

    template = load_prompt_version(CONFIG["prompt_variant"], ROOT)
    body_extra = {"temperature": CONFIG["temperature"], "top_p": CONFIG["top_p"],
                  "max_tokens": CONFIG["max_output_tokens"], **CONFIG["reasoning_fields"]}
    semaphore = asyncio.Semaphore(CONCURRENCY)
    written = 0
    print()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for rep in range(1, REPETITIONS + 1):
            for shots in SHOT_LEVELS:
                examples = pool[:shots]
                pending = [row for row in targets if (rep, shots, row["variable_id"]) not in already]
                if not pending:
                    print(f"  rep {rep:>2}  {shots:>2}-shot  already complete", flush=True)
                    continue
                started = time.monotonic()
                prompts = [render(template, schema_text, examples, row["definition"]) for row in pending]
                results = await asyncio.gather(*(
                    call(client, semaphore, url, headers, MODEL_ID, body_extra, prompt) for prompt in prompts))
                scored = 0
                with RESULTS.open("a", encoding="utf-8") as handle:
                    for row, result in zip(pending, results):
                        record = {"experiment": EXPERIMENT, "provider": "psnc", "model_id": MODEL_ID,
                                  "repetition": rep, "shot_count": shots,
                                  "variable_id": row["variable_id"], "label": row.get("label"),
                                  "category": row.get("category"),
                                  "reasoning_mode": CONFIG["reasoning_mode"],
                                  "prompt_variant": CONFIG["prompt_variant"],
                                  "temperature": CONFIG["temperature"],
                                  "latency_seconds": round(result.get("latency", 0.0), 3),
                                  "prompt_tokens": result.get("prompt_tokens"),
                                  "completion_tokens": result.get("completion_tokens"),
                                  "finish_reason": result.get("finish_reason"),
                                  "ok": result.get("ok", False), "error": result.get("error"),
                                  "answer": result.get("answer")}
                        if result.get("ok"):
                            extracted = extract_json(result["answer"])
                            valid = bool(extracted.success
                                         and validate_prediction(extracted.candidate, schema).valid)
                            record["valid_prediction"] = valid
                            if valid:
                                record["evaluation"] = evaluate_item(
                                    row["gold"], extracted.candidate, similarity,
                                    metadata={"variable_id": row["variable_id"]},
                                    similarity_identity=similarity.identity)
                                scored += 1
                        else:
                            record["valid_prediction"] = False
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
                failed = sum(1 for r in results if not r.get("ok"))
                print(f"  rep {rep:>2}  {shots:>2}-shot  {len(pending):>3} calls  "
                      f"{time.monotonic()-started:6.1f}s  scored {scored:>3}  failed {failed}", flush=True)
    print(f"\nwrote {written} evaluations to {RESULTS.relative_to(ROOT)}")
    summarise()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
