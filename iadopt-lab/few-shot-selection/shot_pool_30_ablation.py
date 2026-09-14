#!/usr/bin/env python
"""Thirty-shot extension of the shot-count ablation — EXPLORATORY, not the official experiment.

Question: the ten-shot ablation left `Qwen3.8-27B` still climbing at its ceiling. Where does
it actually stop? This carries the same model and configuration out to thirty examples.

What changed from `shot_count_ablation.py`
------------------------------------------
* One model only (`Qwen3.8-27B`), at the identical configuration.
* Shot levels 0/1/3/5/7/10/15/20/25/30.
* The reserved pool grows from ten variables to thirty, so the evaluation population shrinks
  from 92 to 72.

That last point makes this **not comparable to the ten-shot run's numbers**, even at the shot
levels they share: a different set of variables is being scored. Only the trend within this
run is meaningful. The results therefore go to their own file rather than extending the
existing one, whose resume key would otherwise silently mix two populations.

Pool construction
-----------------
Positions 1-10 are the frozen ten of the earlier ablation, in their order, so the prompts at
0/1/3/5/7/10 are byte-identical to that run. The twenty additions are stratified by domain so
the completed thirty mirrors the corpus distribution (Life 9, Natural 9, Social 6, Technical
6 against a corpus of 29.4/28.4/21.6/20.6%).

The frozen ten are badly skewed on their own — five of ten are Social Sciences against a
corpus share of 21.6%, and only one is Natural Sciences against 28.4% — so the twenty are
allocated to *correct* that rather than to be proportional among themselves. Being
proportional among themselves would carry the skew through to the final pool.

Their order is chosen greedily: at each step the domain that leaves the cumulative pool
closest to the corpus distribution wins, ties broken alphabetically and within a domain by
sorted `variable_id`. That keeps the intermediate prefixes at 15, 20 and 25 shots
domain-balanced too, instead of feeding the model a run of one domain.

Isolation
---------
Identical to the ten-shot ablation: lives in `few-shot-selection/`, which no artifact collector
walks; writes no campaign, run or task row; reads `parameters.yml` for nothing; writes only
`few-shot-selection/output/`. Its results cannot enter an official ranking.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "few-shot-selection"))

import httpx  # noqa: E402

# Reused so the first ten entries and the prompt bytes cannot drift from the earlier run.
from shot_count_ablation import call, render, shot_pool  # noqa: E402

from iadopt_eval import evaluate_item  # noqa: E402
from iadopt_eval.core import aggregate_items  # noqa: E402
from iadopt_lab.configuration import load_runtime_secrets  # noqa: E402
from iadopt_lab.corpus.ingestion import load_canonical_records  # noqa: E402
from iadopt_lab.generation.extractor import extract_json  # noqa: E402
from iadopt_lab.prompting.renderer import load_prompt_version  # noqa: E402
from iadopt_lab.validation import load_schema_bytes, validate_prediction  # noqa: E402

EXPERIMENT = "shot-pool-30-ablation-v1"
OUT_DIR = ROOT / "few-shot-selection" / "output"
RESULTS = OUT_DIR / "shot-pool-30-ablation-results.jsonl"
SHOT_LEVELS = (0, 1, 3, 5, 7, 10, 15, 20, 25, 30)
POOL_SIZE = 30

MODEL_ID = "Qwen3.8-27B"
CONFIG = {
    "prompt_variant": "matrix-decomposition", "temperature": 0.5, "top_p": 1.0,
    "max_output_tokens": 16000, "reasoning_mode": "disabled",
    "reasoning_fields": {"chat_template_kwargs": {"enable_thinking": False}},
    "source": "campaign 5cdc9417, rank 1, Close F1 0.3912",
}

CONCURRENCY = 8
TIMEOUT = 120.0


def _quota(total: int, counts: Counter) -> dict[str, int]:
    """Apportion `total` slots across domains by largest remainder.

    Args: total: slots to hand out; counts: domain -> corpus frequency.
    Returns: domain -> integer slots, summing exactly to `total`.
    Side Effects: none. Deterministic.
    """
    size = sum(counts.values())
    exact = {domain: counts[domain] / size * total for domain in counts}
    base = {domain: int(value) for domain, value in exact.items()}
    short = total - sum(base.values())
    for domain in sorted(exact, key=lambda d: (-(exact[d] - base[d]), d))[:short]:
        base[domain] += 1
    return base


def extend_pool(records: list[dict], base: list[dict]) -> list[dict]:
    """Grow the frozen ten-variable pool to thirty, domain-representative at every prefix.

    Args: records: all canonical corpus rows; base: the frozen ten, in their order.
    Returns: thirty rows - `base` unchanged at the front, then the twenty additions.
    Raises: ValueError when a domain cannot supply its allocation.
    Side Effects: none. Deterministic, so the pool is reproducible without storing it.
    """
    corpus = Counter(row["category"] for row in records)
    share = {domain: corpus[domain] / sum(corpus.values()) for domain in corpus}
    quota = _quota(POOL_SIZE, corpus)
    held = Counter(row["category"] for row in base)
    # A domain already at or over its final quota contributes nothing further; the frozen ten
    # are fixed, so an over-share can only be diluted, never removed.
    need = {domain: max(0, quota[domain] - held.get(domain, 0)) for domain in quota}

    reserved = {row["variable_id"] for row in base}
    available = {domain: sorted((row for row in records
                                 if row["category"] == domain and row["variable_id"] not in reserved),
                                key=lambda row: row["variable_id"])
                 for domain in quota}
    for domain, wanted in need.items():
        if wanted > len(available[domain]):
            raise ValueError(f"{domain} needs {wanted} additions but only {len(available[domain])} remain")

    chosen: list[dict] = []
    counts, remaining = Counter(held), dict(need)
    while len(chosen) < POOL_SIZE - len(base):
        size = len(base) + len(chosen) + 1
        best_domain, best_deviation = None, None
        for domain in sorted(remaining):
            if remaining[domain] <= 0:
                continue
            trial = Counter(counts)
            trial[domain] += 1
            deviation = sum(abs(trial[other] / size - share[other]) for other in share)
            if best_deviation is None or deviation < best_deviation - 1e-12:
                best_domain, best_deviation = domain, deviation
        chosen.append(available[best_domain][need[best_domain] - remaining[best_domain]])
        counts[best_domain] += 1
        remaining[best_domain] -= 1
    return base + chosen


def done_keys() -> set:
    """(shots, variable_id) pairs already scored, so nothing is called twice."""
    if not RESULTS.exists():
        return set()
    return {(row["shot_count"], row["variable_id"])
            for row in (json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip())}


def summarise() -> None:
    """Print micro Close F1 per shot level, aggregated as the official experiment aggregates."""
    if not RESULTS.exists():
        return
    rows = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[int, list] = {}
    for row in rows:
        grouped.setdefault(row["shot_count"], []).append(row)
    print(f"\n{MODEL_ID}  micro Close F1 by shot count "
          f"({len({r['variable_id'] for r in rows})} evaluation variables)\n")
    print(f"{'shots':>6}{'scored':>9}{'closeF1':>10}{'closeP':>9}{'closeR':>9}{'exactF1':>10}")
    for shots in SHOT_LEVELS:
        batch = grouped.get(shots)
        if not batch:
            continue
        scored = [r for r in batch if r.get("evaluation")]
        if not scored:
            continue
        agg = aggregate_items([r["evaluation"] for r in scored],
                              expected_variable_ids=[r["variable_id"] for r in scored])
        read = lambda mode, metric: agg[mode]["metrics"][metric]["value"]  # noqa: E731
        print(f"{shots:>6}{len(scored):>4}/{len(batch):<4}{read('close','f1'):>10.4f}"
              f"{read('close','precision'):>9.4f}{read('close','recall'):>9.4f}{read('exact','f1'):>10.4f}")


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = list(load_canonical_records(ROOT))
    pool = extend_pool(records, shot_pool(records))
    reserved = {row["variable_id"] for row in pool}
    targets = [row for row in records if row["variable_id"] not in reserved]
    schema = load_schema_bytes(ROOT)
    schema_text = schema.decode("utf-8")

    corpus = Counter(row["category"] for row in records)
    print(f"{EXPERIMENT}: {len(records)} corpus variables, {len(pool)} reserved as shot pool, "
          f"{len(targets)} evaluated at every level\n")
    print(f"{'domain':<20}{'corpus':>8}{'corpus%':>9}{'pool':>6}{'pool%':>8}{'eval':>6}")
    pool_by = Counter(row["category"] for row in pool)
    eval_by = Counter(row["category"] for row in targets)
    for domain in sorted(corpus, key=lambda d: -corpus[d]):
        print(f"{domain:<20}{corpus[domain]:>8}{corpus[domain]/len(records)*100:>8.1f}%"
              f"{pool_by[domain]:>6}{pool_by[domain]/len(pool)*100:>7.1f}%{eval_by[domain]:>6}")

    print("\nshot pool order (position -> first shot level that uses it):")
    for position, row in enumerate(pool, 1):
        first = next(s for s in SHOT_LEVELS if s >= position)
        origin = (f"official demo #{row['demonstration_position']}" if row["demonstration_position"]
                  else "frozen ten" if position <= 10 else "added, domain-stratified")
        print(f"  {position:>2}  @{first:<3} [{row['category']:<18}] {(row.get('label') or '')[:46]:<46} {origin}")

    planned = len(SHOT_LEVELS) * len(targets)
    already = done_keys()
    print(f"\nplanned: {len(SHOT_LEVELS)} shot levels x {len(targets)} variables = {planned}")
    print(f"already complete: {len(already)}   remaining: {planned - len(already)}")
    if "--plan-only" in sys.argv:
        return 0

    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

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
        for shots in SHOT_LEVELS:
            examples = pool[:shots]
            pending = [row for row in targets if (shots, row["variable_id"]) not in already]
            if limit is not None:
                pending = pending[:limit]
            if not pending:
                print(f"  {shots:>2}-shot  already complete")
                continue
            started = time.monotonic()
            prompts = [render(template, schema_text, examples, row["definition"]) for row in pending]
            results = await asyncio.gather(*(
                call(client, semaphore, url, headers, MODEL_ID, body_extra, prompt) for prompt in prompts))
            scored = 0
            with RESULTS.open("a", encoding="utf-8") as handle:
                for row, result in zip(pending, results):
                    record = {"experiment": EXPERIMENT, "provider": "psnc", "model_id": MODEL_ID,
                              "shot_count": shots, "variable_id": row["variable_id"],
                              "label": row.get("label"), "category": row.get("category"),
                              "reasoning_mode": CONFIG["reasoning_mode"],
                              "prompt_variant": CONFIG["prompt_variant"],
                              "temperature": CONFIG["temperature"],
                              "latency_seconds": round(result.get("latency", 0.0), 3),
                              "prompt_tokens": result.get("prompt_tokens"),
                              "completion_tokens": result.get("completion_tokens"),
                              "finish_reason": result.get("finish_reason"),
                              "ok": result.get("ok", False), "error": result.get("error"),
                              # Retained deliberately: the ten-shot run discarded it, which left
                              # its 35 invalid responses impossible to diagnose after the fact.
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
            print(f"  {shots:>2}-shot  {len(pending):>3} calls  {time.monotonic()-started:6.1f}s  "
                  f"scored {scored:>3}  failed {failed}", flush=True)
    print(f"\nwrote {written} evaluations to {RESULTS.relative_to(ROOT)}")
    summarise()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
