"""Candidate evaluation against a fixed evaluation set — see contracts/harness.md.

Owns the only provider access in this experiment and the call cache that makes every stage
resumable. The cache is keyed per call, not per candidate, so an interruption costs at most one
in-flight request rather than a whole candidate.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "few-shot-selection"))

import httpx  # noqa: E402

# Reused so prompt bytes stay identical to the sibling side experiments.
from shot_count_ablation import render  # noqa: E402

from iadopt_eval import evaluate_item  # noqa: E402
from iadopt_eval.core import aggregate_items  # noqa: E402
from iadopt_lab.generation.extractor import extract_json  # noqa: E402
from iadopt_lab.validation import validate_prediction  # noqa: E402
from iadopt_lab.validation.lexical import empty_prediction  # noqa: E402

CANDIDATE_SIZE = 25
CONCURRENCY = 24
TIMEOUT = 120.0

MODEL_ID = "Qwen3.8-27B"
CONFIG = {
    "prompt_variant": "matrix-decomposition", "temperature": 0.5, "top_p": 1.0,
    "max_output_tokens": 16000, "reasoning_mode": "disabled",
    "reasoning_fields": {"chat_template_kwargs": {"enable_thinking": False}},
}


@dataclass
class HarnessContext:
    """Everything one evaluation needs, assembled once: corpus, prompt, scorer, credentials."""

    records: Mapping[str, dict]
    template: Any
    schema: bytes
    schema_text: str
    similarity: Any
    url: str
    headers: Mapping[str, str]
    cache: dict
    calls_path: Path
    concurrency: int = CONCURRENCY
    # Injectable so the harness is testable without a provider: the invalid-prediction and
    # transport-failure paths cannot be provoked deliberately against a live endpoint.
    transport: Callable[..., Awaitable[dict]] | None = None
    lock: Any = field(default_factory=asyncio.Lock)
    # One client for the whole run. Opening one per call costs a TLS handshake and a fresh
    # pool every time, which measured 1.7 calls/s against 3.7 for a shared client.
    client: Any = None


def candidate_hash(candidate: Sequence[str]) -> str:
    """Compute a stable order-independent identity for a candidate set.

    Args: candidate: variable ids forming the example set.
    Returns: 16 hex characters, SHA-256 over the sorted ids.
    Raises: nothing.
    Side Effects: none. Deterministic across processes and machines.
    """
    joined = "\n".join(sorted(candidate)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def load_cache(path: Path) -> dict:
    """Rebuild the per-call cache from the calls JSONL.

    Args: path: the calls JSONL, which need not exist.
    Returns: {(candidate_hash, repetition, variable_id): record}; empty when the file is absent.
    Raises: nothing; a malformed trailing line is skipped rather than failing a resume.
    Side Effects: reads the file.
    """
    if not path.exists():
        return {}
    cache: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn trailing line from a kill must not block the resume
        # Failed calls are recorded for audit but deliberately not cached, so a resume retries
        # them instead of inheriting a transport blip as a permanent result.
        if row.get("ok"):
            cache[(row["candidate_hash"], row["repetition"], row["variable_id"])] = row
    return cache


def build_context(root: Path, concurrency: int = CONCURRENCY) -> HarnessContext:
    """Assemble corpus, prompt template, schema, scorer backend and PSNC credentials.

    Args: root: the iadopt-lab root; concurrency: provider concurrency ceiling.
    Returns: a HarnessContext ready for `evaluate`.
    Raises: LabError when PSNC credentials are absent.
    Side Effects: reads local files and the .env; opens no network connection.
    """
    import os

    from iadopt_lab.cli import _configuration, _similarity_backend
    from iadopt_lab.configuration import load_runtime_secrets
    from iadopt_lab.corpus.ingestion import load_canonical_records
    from iadopt_lab.prompting.renderer import load_prompt_version
    from iadopt_lab.validation import load_schema_bytes

    secrets = load_runtime_secrets(str(root.parent / ".env"),
                                   ["PSNC_API_KEY", "PSNC_API_BASE_URL"], os.environ)
    base = (secrets.get("PSNC_API_BASE_URL") or "https://llm.hpc.psnc.pl").rstrip("/")
    schema = load_schema_bytes(root)
    calls_path = root / "few-shot-selection" / "example_selection" / "output" / "example-selection-calls.jsonl"
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    return HarnessContext(
        records={row["variable_id"]: row for row in load_canonical_records(root)},
        template=load_prompt_version(CONFIG["prompt_variant"], root),
        schema=schema, schema_text=schema.decode("utf-8"),
        similarity=_similarity_backend(root, _configuration(root).data["evaluation"]),
        url=base + "/chat/completions",
        headers={"Authorization": "Bearer " + secrets.get("PSNC_API_KEY"),
                 "Content-Type": "application/json"},
        cache=load_cache(calls_path), calls_path=calls_path, concurrency=concurrency,
        client=httpx.AsyncClient(timeout=TIMEOUT,
                                 limits=httpx.Limits(max_connections=concurrency * 2,
                                                     max_keepalive_connections=concurrency)))


async def evaluate(candidate: Sequence[str], targets: Sequence[str], repetition: int,
                   context: HarnessContext) -> dict:
    """Score one candidate example-set against one evaluation set.

    Args: candidate: 25 example ids; targets: evaluation ids; repetition: 1-based; context: harness context.
    Returns: close_f1_official, close_f1_as_run, precision, recall, scored, invalid, failed_calls, latency, tokens.
    Raises: ValueError when the candidate is not size 25 or intersects `targets`.
    Side Effects: issues provider calls for uncached targets and appends them to the calls JSONL.
        Idempotent: a repeat call for the same key issues no request.
    """
    if len(set(candidate)) != CANDIDATE_SIZE:
        raise ValueError(f"candidate must be exactly {CANDIDATE_SIZE} distinct ids, got {len(set(candidate))}")
    overlap = set(candidate) & set(targets)
    if overlap:
        # The leak the whole design exists to prevent: an example that is also scored.
        raise ValueError(f"candidate overlaps the evaluation set at {sorted(overlap)}")

    digest = candidate_hash(candidate)
    examples = [context.records[v] for v in candidate]
    transport = context.transport or functools.partial(_psnc_transport, client=context.client)
    semaphore = asyncio.Semaphore(context.concurrency)
    body_base = {"model": MODEL_ID, "stream": False,
                 "temperature": CONFIG["temperature"], "top_p": CONFIG["top_p"],
                 "max_tokens": CONFIG["max_output_tokens"], **CONFIG["reasoning_fields"]}

    async def one(variable_id: str) -> dict:
        key = (digest, repetition, variable_id)
        if key in context.cache:
            return context.cache[key]
        target = context.records[variable_id]
        prompt = render(context.template, context.schema_text, examples, target["definition"])
        body = dict(body_base, messages=[{"role": "user", "content": prompt}])
        async with semaphore:
            started = time.monotonic()
            result = await transport(context.url, context.headers, body)
        record = {"candidate_hash": digest, "repetition": repetition, "variable_id": variable_id,
                  "ok": bool(result.get("ok")), "error": result.get("error"),
                  "answer": result.get("answer"),
                  "latency_seconds": round(result.get("latency", time.monotonic() - started), 3),
                  "prompt_tokens": result.get("prompt_tokens"),
                  "completion_tokens": result.get("completion_tokens"),
                  "finish_reason": result.get("finish_reason"), "valid_prediction": False}
        if record["ok"]:
            extracted = extract_json(record["answer"] or "")
            if extracted.success and validate_prediction(extracted.candidate, context.schema).valid:
                record["valid_prediction"] = True
                record["evaluation"] = evaluate_item(
                    target["gold"], extracted.candidate, context.similarity,
                    metadata={"variable_id": variable_id},
                    similarity_identity=context.similarity.identity)
            context.cache[key] = record
        async with context.lock:
            with context.calls_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    records = await asyncio.gather(*(one(v) for v in targets))
    return _summarise(records, targets, context)


async def _psnc_transport(url: str, headers: Mapping[str, str], body: Mapping[str, Any],
                          client: Any = None) -> dict:
    """Issue one PSNC chat-completion request over a shared client.

    Args: url: endpoint; headers: auth headers; body: request body; client: shared AsyncClient.
    Returns: {"ok", "latency", and on success "answer", "finish_reason", token counts}.
    Raises: nothing; a transport error is reported as ok=False so one call cannot kill a stage.
    Side Effects: one outbound HTTPS request on the shared connection pool.
    """
    started = time.monotonic()
    owned = client is None
    if owned:
        client = httpx.AsyncClient(timeout=TIMEOUT)
    try:
        response = await client.post(url, headers=headers, json=body)
        latency = time.monotonic() - started
        if response.status_code != 200:
            return {"ok": False, "error": f"HTTP {response.status_code}", "latency": latency}
        payload = response.json()
        choice = (payload.get("choices") or [{}])[0]
        usage = payload.get("usage") or {}
        return {"ok": True, "latency": latency,
                "answer": (choice.get("message") or {}).get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}
    except Exception as error:  # noqa: BLE001 - a failed call is recorded, never raised
        return {"ok": False, "error": type(error).__name__, "latency": time.monotonic() - started}
    finally:
        if owned:
            await client.aclose()


def _summarise(records: Sequence[dict], targets: Sequence[str], context: HarnessContext) -> dict:
    """Aggregate one candidate's per-call records under both scoring rules.

    Args: records: per-target call records; targets: evaluation ids; context: for gold and scorer.
    Returns: the result mapping described in contracts/harness.md.
    Raises: nothing.
    Side Effects: none. `denominator_official` is always len(targets), by construction.
    """
    scored = [r for r in records if r.get("evaluation")]
    as_run = _aggregate([r["evaluation"] for r in scored], [r["variable_id"] for r in scored])

    evidence = [r["evaluation"] for r in scored]
    ids = [r["variable_id"] for r in scored]
    for record in records:
        if not record.get("evaluation"):
            target = context.records[record["variable_id"]]
            evidence.append(evaluate_item(target["gold"], empty_prediction(), context.similarity,
                                          metadata={"variable_id": record["variable_id"]},
                                          similarity_identity=context.similarity.identity))
            ids.append(record["variable_id"])
    official = _aggregate(evidence, ids)
    latencies = [r["latency_seconds"] for r in records if r.get("latency_seconds")]
    prompt_tokens = [r["prompt_tokens"] for r in records if r.get("prompt_tokens")]
    return {
        "close_f1_official": official["f1"], "close_f1_as_run": as_run["f1"],
        "close_precision": official["precision"], "close_recall": official["recall"],
        "scored": len(scored), "invalid": len(records) - len(scored),
        "failed_calls": sum(1 for r in records if not r.get("ok")),
        "denominator_official": len(targets),
        "mean_latency": statistics.mean(latencies) if latencies else 0.0,
        "prompt_tokens": statistics.mean(prompt_tokens) if prompt_tokens else 0.0,
    }


def _aggregate(evidence: Sequence[dict], ids: Sequence[str]) -> dict:
    """Micro-aggregate evaluation records exactly as the official reporting path does."""
    if not evidence:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    result = aggregate_items(list(evidence), expected_variable_ids=list(ids))
    return {name: result["close"]["metrics"][name]["value"] for name in ("f1", "precision", "recall")}
