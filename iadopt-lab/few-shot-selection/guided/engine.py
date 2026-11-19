"""Run, score and compare prompt arms on Qwen3.8-27B (PSNC, reasoning off).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.

An "arm" is one way of building the prompt for a variable. Every arm is answered by the same
model with the settings of the example-selection study, scored with that study's extraction,
validation and scorer, and compared with a baseline arm over repeated runs. The transport, the
prompt layout, the statistics and the price list are reused from `example_selection/format_test.py`
unchanged. Outputs go to the gitignored `few-shot-selection/output/guided/`.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, ROOT / "src"):
    sys.path.insert(0, str(_path))

import format_test as ft  # noqa: E402
from harness import _aggregate  # noqa: E402

MODEL = "Qwen3.8-27B"
REQUEST = dict(ft.REQUEST)
CONCURRENCY = 24
OUTPUT = HERE.parent / "output" / "guided"
EARLY_WINDOW, EARLY_FAILURES, THINKING_LIMIT = 48, 5, 3

Transport = Callable[[Mapping[str, Any]], Awaitable[dict]]
Job = tuple[str, int, str]


class Guard:
    """Stop rules checked by every worker before every call. Arvin set no spending cap."""

    def __init__(self) -> None:
        """Start with no stop reason and no counts.

        Args: none.
        Returns: nothing.
        Raises: nothing.
        Side Effects: none.
        """
        self.reason: str | None = None
        self.calls = self.failures = self.thinking = 0
        self.spend = 0.0

    def record(self, result: Mapping[str, Any]) -> None:
        """Apply the stop rules to one finished call.

        Args: result: the stored call record.
        Returns: nothing.
        Raises: nothing.
        Side Effects: updates counts and the list-price total; may set the stop reason.
        """
        self.calls += 1
        if result.get("status") in (401, 403):
            self.reason = self.reason or f"credential refused (HTTP {result['status']})"
        if not result.get("ok"):
            self.failures += 1
            if self.calls <= EARLY_WINDOW and self.failures > EARLY_FAILURES:
                self.reason = self.reason or f"{self.failures} of the first {self.calls} calls failed"
            return
        self.spend += ft.list_price_usd(MODEL, result)
        if result.get("reasoning_chars"):
            self.thinking += 1
            if self.thinking > THINKING_LIMIT:
                self.reason = self.reason or "answers carry reasoning text despite the switch"


def schedule(arms: Sequence[str], variables: Sequence[str], repetitions: int, seed: int) -> list[Job]:
    """List every call, with all arms for one variable side by side.

    Args: arms: arm names; variables: variable ids; repetitions: runs per arm; seed: shuffle seed.
    Returns: (arm, repetition, variable_id) jobs, each exactly once.
    Raises: nothing.
    Side Effects: none. Deterministic for a given seed.
    """
    import random

    rng = random.Random(seed)
    jobs: list[Job] = []
    for repetition in range(1, repetitions + 1):
        order = list(variables)
        rng.shuffle(order)
        for variable_id in order:
            shuffled = list(arms)
            rng.shuffle(shuffled)
            jobs.extend((arm, repetition, variable_id) for arm in shuffled)
    return jobs


def key_of(record: Mapping[str, Any]) -> Job:
    """Identify one call.

    Args: record: a call record.
    Returns: (arm, repetition, variable_id).
    Raises: KeyError when a field is missing.
    Side Effects: none.
    """
    return (record["arm"], record["repetition"], record["variable_id"])


def latest(path: Path) -> dict[Job, dict]:
    """Keep one record per call: the first successful one, else the last failure.

    Args: path: a calls file.
    Returns: {call key: record}.
    Raises: OSError when the file exists but cannot be read.
    Side Effects: reads the file.
    """
    out: dict[Job, dict] = {}
    for record in ft.read_jsonl(path):
        key = key_of(record)
        if key not in out or not out[key].get("ok"):
            out[key] = record
    return out


async def run_jobs(jobs: Sequence[Job], prompt_of: Callable[[Job], str], transport: Transport,
                   calls_path: Path, concurrency: int = CONCURRENCY, max_calls: int | None = None,
                   guard: Guard | None = None, request: Mapping[str, Any] | None = None) -> dict:
    """Issue every job not already answered, appending one record per call.

    Args: jobs: from `schedule`; prompt_of: builds a job's prompt; transport: sends one request;
        calls_path: the calls file; concurrency: workers; max_calls: cap on new calls; guard: the
        stop rules; request: request settings, the study's by default.
    Returns: {"issued", "skipped", "stopped", "list_price_usd"}.
    Raises: nothing from the provider; a failed call is recorded.
    Side Effects: provider calls; appends to `calls_path`. Resumable: answered calls are not
        repeated and failed ones are retried.
    """
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    guard = guard or Guard()
    settings = dict(request or REQUEST)
    answered = {key for key, record in latest(calls_path).items() if record.get("ok")}
    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        if job not in answered:
            queue.put_nowait(job)
    lock = asyncio.Lock()
    counts = Counter()
    budget = [max_calls if max_calls is not None else len(jobs)]

    async def worker() -> None:
        """Take jobs until the queue is empty, honouring the stop rules and the call budget.

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
            prompt = prompt_of(job)
            body = dict(settings, model=MODEL, messages=[{"role": "user", "content": prompt}])
            started_at = ft.now()
            result = await transport(body)
            reasoning = result.get("reasoning") or ""
            record = {"arm": job[0], "repetition": job[1], "variable_id": job[2],
                      "started_at": started_at, "ok": bool(result.get("ok")),
                      "status": result.get("status"), "error": result.get("error"),
                      "answer": result.get("answer"), "reasoning_chars": len(reasoning),
                      "reasoning": reasoning or None, "finish_reason": result.get("finish_reason"),
                      "prompt_tokens": result.get("prompt_tokens"),
                      "completion_tokens": result.get("completion_tokens"),
                      "latency_seconds": round(result.get("latency") or 0.0, 3),
                      "served_model": result.get("served_model"), "prompt_sha256": ft.sha256(prompt)}
            counts["issued"] += 1
            guard.record(record)
            async with lock:
                with calls_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return {"issued": counts["issued"], "skipped": counts["skipped"], "stopped": guard.reason,
            "list_price_usd": round(guard.spend, 4)}


def score(calls_path: Path, scores_path: Path, records: Mapping[str, Mapping[str, Any]],
          schema: bytes) -> dict[Job, dict]:
    """Score every call's answer once; an unusable or missing answer scores as empty.

    Args: calls_path: the calls file; scores_path: the score cache; records: corpus records by
        id, for the gold; schema: the lexical schema bytes.
    Returns: {call key: {"key", "answer_sha256", "valid_prediction", "prediction", "evaluation"}}.
    Raises: nothing for a bad answer; it scores as `empty_prediction()`.
    Side Effects: loads the local similarity model when an answer is new; appends new scores.
        Same extraction, validation and scorer as the study's harness. Idempotent per answer.
    """
    from iadopt_eval import evaluate_item
    from iadopt_lab.cli import _configuration, _similarity_backend
    from iadopt_lab.generation.extractor import extract_json
    from iadopt_lab.validation import validate_prediction
    from iadopt_lab.validation.lexical import empty_prediction

    cached = {(tuple(row["key"]), row["answer_sha256"]): row for row in ft.read_jsonl(scores_path)}
    similarity = None
    out: dict[Job, dict] = {}
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("a", encoding="utf-8") as handle:
        for key, record in latest(calls_path).items():
            answer = record.get("answer") if record.get("ok") else None
            digest = ft.sha256(answer) if answer is not None else "no-answer"
            if (key, digest) in cached:
                out[key] = cached[(key, digest)]
                continue
            if similarity is None:
                similarity = _similarity_backend(ROOT, _configuration(ROOT).data["evaluation"])
            prediction = None
            if answer is not None:
                extracted = extract_json(answer)
                if extracted.success and validate_prediction(extracted.candidate, schema).valid:
                    prediction = extracted.candidate
            evaluation = evaluate_item(records[key[2]]["gold"], prediction or empty_prediction(), similarity,
                                       metadata={"variable_id": key[2]},
                                       similarity_identity=similarity.identity)
            row = {"key": list(key), "answer_sha256": digest, "valid_prediction": prediction is not None,
                   "prediction": prediction, "evaluation": evaluation}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            out[key] = row
    return out


def repetition_rows(arm: str, variables: Sequence[str], repetitions: int, calls: Mapping[Job, dict],
                    scores: Mapping[Job, dict]) -> tuple[list[dict], list[int]]:
    """Score one arm per repetition; a repetition missing a call is left out.

    Args: arm: arm name; variables: the population; repetitions: runs; calls: from `latest`;
        scores: from `score`.
    Returns: (per-repetition rows, numbers of the incomplete repetitions).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    rows, incomplete = [], []
    for repetition in range(1, repetitions + 1):
        keys = [(arm, repetition, v) for v in variables]
        if not all(key in calls and key in scores for key in keys):
            incomplete.append(repetition)
            continue
        valid = [key for key in keys if scores[key]["valid_prediction"]]
        answered = [calls[key] for key in keys if calls[key].get("ok")]
        rows.append({
            "repetition": repetition,
            "close_f1_official": _aggregate([scores[k]["evaluation"] for k in keys], [k[2] for k in keys])["f1"],
            "close_f1_as_run": _aggregate([scores[k]["evaluation"] for k in valid], [k[2] for k in valid])["f1"],
            "invalid": len(keys) - len(valid), "failed_calls": len(keys) - len(answered),
            "prompt_tokens": ft._mean(r.get("prompt_tokens") for r in answered),
            "completion_tokens": ft._mean(r.get("completion_tokens") for r in answered),
            "latency_seconds": ft._mean(r.get("latency_seconds") for r in answered),
            "list_price_usd": sum(ft.list_price_usd(MODEL, r) for r in answered),
            "reasoning_answers": sum(1 for key in keys if calls[key].get("reasoning_chars"))})
    return rows, incomplete


def summarise(arms: Sequence[str], baseline: str, variables: Sequence[str], repetitions: int,
              calls: Mapping[Job, dict], scores: Mapping[Job, dict]) -> dict[str, Any]:
    """Per-arm statistics and each arm's comparison with the baseline.

    Args: arms: all arm names; baseline: the arm compared against; variables, repetitions, calls,
        scores: as for `repetition_rows`.
    Returns: {"generated_at", "variables", "cells": {arm: {...}}, "comparisons": {arm: Welch
        result with "p_holm" and "verdict"}}; Holm runs over the non-baseline arms.
    Raises: nothing.
    Side Effects: none.
    """
    cells: dict[str, Any] = {}
    for arm in arms:
        rows, incomplete = repetition_rows(arm, variables, repetitions, calls, scores)
        if not rows:
            continue
        f1 = [row["close_f1_official"] for row in rows]
        cells[arm] = {"repetitions": len(rows), "incomplete_repetitions": incomplete,
                      "close_f1_official_mean": statistics.mean(f1),
                      "close_f1_official_sd": statistics.stdev(f1) if len(f1) > 1 else None,
                      "close_f1_as_run_mean": statistics.mean(r["close_f1_as_run"] for r in rows),
                      "invalid": sum(r["invalid"] for r in rows),
                      "failed_calls": sum(r["failed_calls"] for r in rows),
                      "reasoning_answers": sum(r["reasoning_answers"] for r in rows),
                      "prompt_tokens_mean": ft._mean(r["prompt_tokens"] for r in rows),
                      "completion_tokens_mean": ft._mean(r["completion_tokens"] for r in rows),
                      "latency_seconds_mean": ft._mean(r["latency_seconds"] for r in rows),
                      "list_price_usd": sum(r["list_price_usd"] for r in rows), "per_repetition": rows}

    def series(arm: str) -> list[float]:
        """Return one arm's per-repetition Close F1.

        Args: arm: arm name.
        Returns: the scores; empty when the arm did not run.
        Raises: nothing.
        Side Effects: none.
        """
        return [r["close_f1_official"] for r in cells[arm]["per_repetition"]] if arm in cells else []

    comparisons = {arm: ft.welch(series(arm), series(baseline)) for arm in arms
                   if arm != baseline and len(series(arm)) > 1 and len(series(baseline)) > 1}
    adjusted = ft.holm({arm: result["p"] for arm, result in comparisons.items()})
    for arm, result in comparisons.items():
        result["p_holm"] = adjusted[arm]
        better = "better" if result["difference"] > 0 else "worse"
        result["verdict"] = f"{better} than {baseline}" if adjusted[arm] < 0.05 else "cannot tell apart"
    return {"generated_at": ft.now(), "variables": len(variables), "cells": cells, "comparisons": comparisons}
