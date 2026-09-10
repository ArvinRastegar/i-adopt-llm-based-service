#!/usr/bin/env python
"""Shot-count ablation — an EXPLORATORY side analysis, not part of the official experiment.

Question: with everything else held at each model's best known reasoning-off configuration,
how does performance change as the number of few-shot examples goes 0, 1, 3, 5, 7, 10?

Why this does not use the official campaign machinery
-----------------------------------------------------
It cannot, without contaminating the official experiment. Three guards block it, and every
way around them changes something the official plans are hashed against:

* `select_demonstrations` rejects any shot count outside (0, 1, 3, 5).
* `render_base_prompt` requires the exact official five-variable demonstration prefix.
* The frozen demonstration pool has five members; this needs ten.

Relaxing those means editing `src/`, `prompts/` or `data/manifests/`, all of which feed the
official `implementation` artifact hash. Changing it invalidates every official plan and
would force ~21,000 completed, paid tasks to be re-run. So this is a standalone runner that
reuses the official renderer and scorer as libraries while writing nowhere the official
experiment reads.

Isolation
---------
* Lives in `experiments/`, which no artifact collector walks.
* Reads `parameters.yml` for nothing; the configuration is fixed in this file.
* Writes only `experiments/output/`. It never touches the campaign database, `outputs/`,
  or any manifest.
* Results carry no campaign, run or task identity, so they cannot enter official rankings.

Reproducibility
---------------
The shot pool and its nesting are deterministic: the five official demonstrations in their
official order, then five more chosen by sorted `variable_id`. Lower shot counts are
prefixes of that same ordering, never independently resampled, so the only thing that
changes between conditions is how many examples the prompt carries.

Resumability
------------
Every scored evaluation is appended to a JSONL file as it completes. Re-running skips any
(model, shot count, variable) triple already present, so an interruption costs nothing and
no provider call is repeated.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from iadopt_eval import evaluate_item  # noqa: E402
from iadopt_lab.configuration import load_runtime_secrets  # noqa: E402
from iadopt_lab.corpus import DEMONSTRATION_PATHS  # noqa: E402
from iadopt_lab.corpus.ingestion import load_canonical_records  # noqa: E402
from iadopt_lab.generation.extractor import extract_json  # noqa: E402
from iadopt_lab.prompting.renderer import _json as canonical_json  # noqa: E402
from iadopt_lab.prompting.renderer import load_prompt_version  # noqa: E402
from iadopt_lab.validation import load_schema_bytes, validate_prediction  # noqa: E402

EXPERIMENT = "shot-count-ablation-v1"
OUT_DIR = ROOT / "experiments" / "output"
RESULTS = OUT_DIR / "shot-count-ablation-results.jsonl"
SHOT_LEVELS = (0, 1, 3, 5, 7, 10)
POOL_SIZE = 10

# Each model's best reasoning-OFF configuration from official campaign 5cdc9417, taken from
# the stored ranking rather than chosen here. Everything except shot count is held fixed.
MODELS = {
    "Qwen3.8-27B": {
        "prompt_variant": "matrix-decomposition", "temperature": 0.5, "top_p": 1.0,
        "max_output_tokens": 16000, "reasoning_mode": "disabled",
        "reasoning_fields": {"chat_template_kwargs": {"enable_thinking": False}},
        "source": "campaign 5cdc9417, rank 1, Close F1 0.3912",
    },
    "DeepSeek-V4-Flash": {
        "prompt_variant": "matrix-decomposition", "temperature": 0.5, "top_p": 1.0,
        "max_output_tokens": 16000, "reasoning_mode": "not_applicable",
        "reasoning_fields": {},
        "source": "campaign 5cdc9417, rank 2, Close F1 0.3899",
    },
}

CONCURRENCY = 8      # PSNC reasoning-off is 0.7-3.3s per call; 24 was validated, 8 is ample
TIMEOUT = 120.0      # reasoning is off, so calls never approach this


def shot_pool(records):
    """Freeze the ten shot examples and their nesting order, deterministically.

    The five official demonstrations come first, in their official order, so the 1-, 3- and
    5-shot conditions use exactly the examples the official experiment used and the 5-shot
    point stays comparable to it. The remaining five are chosen by sorted `variable_id`,
    which is reproducible and independent of anything measured.
    """
    by_path = {row["source_path"]: row for row in records}
    official = [by_path[path] for path in DEMONSTRATION_PATHS]
    rest = sorted((row for row in records if not row["demonstration_position"]),
                  key=lambda row: row["variable_id"])
    return official + rest[: POOL_SIZE - len(official)]


def evaluation_set(records, pool):
    """The same 92 variables for every shot level, so population never confounds the trend."""
    reserved = {row["variable_id"] for row in pool}
    return [row for row in records if row["variable_id"] not in reserved]


def render(template, schema_text, examples, definition):
    """Rebuild the official prompt exactly, minus the 0/1/3/5 prefix guard.

    Mirrors `render_base_prompt`: same template, same canonical JSON for the demonstration
    payload, same single substitution pass. Only the guard is absent, because this
    experiment deliberately uses shot counts and examples the official grid forbids.
    """
    payload = [{"demonstration": position, "definition": example["definition"],
                "decomposition": example["gold"]}
               for position, example in enumerate(examples, 1)]
    values = {"schema": schema_text, "demonstrations": canonical_json(payload),
              "target_definition": definition}
    return re.sub(r"\{\{(schema|demonstrations|target_definition)\}\}",
                  lambda match: values[match.group(1)], template.text)


async def call(client, semaphore, url, headers, model, body_extra, prompt):
    async with semaphore:
        started = time.monotonic()
        try:
            response = await client.post(url, headers=headers, json={
                "model": model, "messages": [{"role": "user", "content": prompt}],
                "stream": False, **body_extra})
            latency = time.monotonic() - started
            if response.status_code != 200:
                return {"ok": False, "error": f"HTTP {response.status_code}", "latency": latency}
            payload = response.json()
            choice = (payload.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = payload.get("usage") or {}
            return {"ok": True, "latency": latency,
                    "answer": message.get("content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens")}
        except Exception as error:  # noqa: BLE001 - a failed evaluation is recorded, not raised
            return {"ok": False, "error": type(error).__name__, "latency": time.monotonic() - started}


def done_keys():
    """(model, shots, variable_id) triples already scored, so nothing is paid for twice."""
    if not RESULTS.exists():
        return set()
    keys = set()
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((row["model_id"], row["shot_count"], row["variable_id"]))
    return keys


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = list(load_canonical_records(ROOT))
    pool = shot_pool(records)
    targets = evaluation_set(records, pool)
    schema = load_schema_bytes(ROOT)
    schema_text = schema.decode("utf-8")

    print(f"{EXPERIMENT}: {len(records)} corpus variables, {len(pool)} reserved as shot pool, "
          f"{len(targets)} evaluated at every level")
    planned = len(MODELS) * len(SHOT_LEVELS) * len(targets)
    already = done_keys()
    print(f"planned evaluations: {len(MODELS)} models x {len(SHOT_LEVELS)} shot levels x "
          f"{len(targets)} variables = {planned}")
    print(f"already complete: {len(already)}   remaining: {planned - len(already)}\n")
    if "--plan-only" in sys.argv:
        return 0

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    secrets = load_runtime_secrets(str(ROOT.parent / ".env"),
                                   ["PSNC_API_KEY", "PSNC_API_BASE_URL"], os.environ)
    base = (secrets.get("PSNC_API_BASE_URL") or "https://llm.hpc.psnc.pl").rstrip("/")
    url = base + "/chat/completions"
    headers = {"Authorization": "Bearer " + secrets.get("PSNC_API_KEY"),
               "Content-Type": "application/json"}

    from iadopt_lab.cli import _configuration, _similarity_backend
    similarity = _similarity_backend(ROOT, _configuration(ROOT).data["evaluation"])

    semaphore = asyncio.Semaphore(CONCURRENCY)
    written = 0
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for model_id, config in MODELS.items():
            template = load_prompt_version(config["prompt_variant"], ROOT)
            body_extra = {"temperature": config["temperature"], "top_p": config["top_p"],
                          "max_tokens": config["max_output_tokens"], **config["reasoning_fields"]}
            for shots in SHOT_LEVELS:
                examples = pool[:shots]
                pending = [row for row in targets
                           if (model_id, shots, row["variable_id"]) not in already]
                if limit is not None:
                    pending = pending[:limit]
                if not pending:
                    print(f"  {model_id:20} {shots:>2}-shot  already complete")
                    continue
                started = time.monotonic()
                prompts = [render(template, schema_text, examples, row["definition"])
                           for row in pending]
                results = await asyncio.gather(*(
                    call(client, semaphore, url, headers, model_id, body_extra, prompt)
                    for prompt in prompts))
                scored = 0
                with RESULTS.open("a", encoding="utf-8") as handle:
                    for row, result in zip(pending, results):
                        record = {"experiment": EXPERIMENT, "provider": "psnc",
                                  "model_id": model_id, "shot_count": shots,
                                  "variable_id": row["variable_id"], "label": row.get("label"),
                                  "category": row.get("category"),
                                  "reasoning_mode": config["reasoning_mode"],
                                  "prompt_variant": config["prompt_variant"],
                                  "temperature": config["temperature"],
                                  "latency_seconds": round(result.get("latency", 0.0), 3),
                                  "prompt_tokens": result.get("prompt_tokens"),
                                  "completion_tokens": result.get("completion_tokens"),
                                  "finish_reason": result.get("finish_reason"),
                                  "ok": result.get("ok", False), "error": result.get("error")}
                        if result.get("ok"):
                            extracted = extract_json(result["answer"])
                            valid = bool(extracted.success
                                         and validate_prediction(extracted.candidate, schema).valid)
                            record["valid_prediction"] = valid
                            if valid:
                                evaluation = evaluate_item(
                                    row["gold"], extracted.candidate, similarity,
                                    metadata={"variable_id": row["variable_id"]},
                                    similarity_identity=similarity.identity)
                                record["evaluation"] = evaluation
                                scored += 1
                        else:
                            record["valid_prediction"] = False
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
                elapsed = time.monotonic() - started
                failed = sum(1 for r in results if not r.get("ok"))
                print(f"  {model_id:20} {shots:>2}-shot  {len(pending):>3} calls  "
                      f"{elapsed:6.1f}s  scored {scored:>3}  failed {failed}", flush=True)
    print(f"\nwrote {written} evaluations to {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
