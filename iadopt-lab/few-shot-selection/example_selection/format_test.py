#!/usr/bin/env python
"""Example-layout test: the study's one-line examples against output-shaped examples.

EXPLORATORY side experiment, 2026-11-18. Not part of the official experiment: nothing it
produces may enter an official ranking, campaign, results table or the paper. The plan and the
results are recorded in the private paper repository,
`paper/tracking/EXAMPLE_FORMAT_TEST_20261118.md`.

Question. The example-selection study shows its 25 examples as one compact JSON line in which
each answer comes before its definition and the fields are in alphabetical order, the reverse
of the order the model is asked to answer in. Does showing each example as its definition
followed by the exact JSON answer, in the field order and layout of the expected output,
decompose better?

Grid: {Qwen3.8-27B, GLM-5.2} x {A: the study's layout, B: output-shaped layout}, the study's
winning `top25` examples, the 24 held-out confirmation variables `C`, 15 repetitions per cell.
Only the examples block differs between A and B, and A is byte-identical to the prompt the
study measured. GLM-5.3 was planned but failed the reasoning gate (with thinking off it never
produced an answer); Arvin approved GLM-5.2 in its place on 2026-11-18.

    ./format_test.py --plan-only    # fingerprints and call counts, no provider calls
    ./format_test.py --probe        # reasoning-control probe on an invented target
    ./format_test.py --run          # the grid, resumable; failed calls are retried on resume
    ./format_test.py --score        # score the stored answers and write the summary

Isolation as for the sibling runners: lives in `few-shot-selection/`, which no artifact
collector walks; writes no campaign, run, task or evaluation row; writes only
`few-shot-selection/example_selection/output/format-test-*`.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "few-shot-selection"))

import design  # noqa: E402
import httpx  # noqa: E402
from harness import _aggregate, candidate_hash  # noqa: E402

# Reused so format A stays byte-identical to the prompt the study measured.
from shot_count_ablation import render as render_a  # noqa: E402

MODELS = ("Qwen3.8-27B", "GLM-5.2")
FORMATS = ("A", "B")
REPETITIONS = 15
CONCURRENCY = 24
TIMEOUT = 120.0
SCHEDULE_SEED = 20261118
# The study's measured settings, with the lab's reasoning switch for both models.
REQUEST = {"stream": False, "temperature": 0.5, "top_p": 1.0, "max_tokens": 16000,
           "chat_template_kwargs": {"enable_thinking": False}}

# The study's winning examples (`top25`, candidate 2d56af0e65d42748) in the sorted order the
# study sent them. Written out so this test does not depend on the gitignored study output.
EXAMPLE_CANDIDATE = "2d56af0e65d42748"
EXAMPLE_IDS = tuple("urn:iadopt-lab:variable:" + suffix for suffix in (
    "0f3e9954aef6497d1eaeeb026c643c23dc0e24127c4f4a26938827039cabe944",
    "12b00c359adcf48b4a1dbef25c0e123e0dd41230bc29efcdf160b201be399f0a",
    "12d0bb86102fa82e6c55f1b3f178abd9fd14fb55a497ab442c94ed3ecd41e539",
    "139c80cce9c15999097bed86c8562e543eed77f8528d532f01a00f90778c0043",
    "17b44234089af8f003abef095347985e0d50b8cc418721ed2058c26b2dff1556",
    "1a039be29796c65ca05176b0d2752323eeae0711597ee9b9fb034a72e08f56e5",
    "1e8966f5353f9625d602da4a30952ce6b4e93d27cd2b697cbc207f5e2412edd3",
    "21171d68ac4530e2bd43a74c04c52ee72c7a7ba44ffd0f826b0cc7d47361c4c5",
    "269247ba0afcd49573d5f3890800dbe57cbab4de0abe23e35363dd4c408eeab7",
    "2cb06286f2a57367046baecdc253e1cc9d0b87e3ff744ece77467dbc2b209cd3",
    "33c4eb606b35dd5cbba886825fab3368a2ccec7e8648f0ea739ffdaa8e7ac381",
    "3a580edca2d246b8cacb94c1a25f05c414f9bf30ebb89ad44d7d3243ddefd5ab",
    "3f70359c18e6b25d95bd9792608b167905fb117d9dfe9119e2a8e82e96298cd8",
    "417e2fc4ad6d614f1aa6fbffac30a659c313db707818b36fe36e1636ceda31b0",
    "4468d192503a3df852da9a2e6b2232d2cf888db8e57a183bb36b150f90a4dfb3",
    "5195f6d2757865fa53c1a58080743c04f77f21da30e35252076d7011836494cc",
    "5382e8c971d5accb754b9b3c8c5052ac2a94ddc4c18bf7c14b05a8986546d509",
    "5aa2c10675a4b91f044861bac4875458821f234a139c770757ad4a880072dfdc",
    "608502045635ff2f6185e327ed3dabb4915ae7fcf480bf5aaeea346544d85dfb",
    "60c22a22493fe14d9b49056deb34f93814298c43443f3d47bd63346ea2042820",
    "661c259486249b392894bc315cc14457401c22ed1abb25e42110c733dc2a6128",
    "7a725c87db996829ebc171a000be0a2d49f868ec0778ab96908b28a38ba03be0",
    "878908f20228224335acf3157a9fc32acc26f6f85629e31c18e1cc006e1005a7",
    "8a816547d513e441cc011c55b73bc5051ee086c3a1b888e4a7c8d41c7b4f3531",
    "8b4f1caa8c7add1f5a3d21c7cc9dc6ca911b9abe84e3a583725215f55d662aed",
))

# PCSS AI HUB list prices in US dollars per million tokens (input, output). The hub page Arvin
# pasted on 2026-11-18 gives them in its local currency; converted at the National Bank of
# Poland mid rate for that day (table 224/A/NBP/2026, 1 USD = 4.0170). Used for the spend guard
# and the cost report only: the lab records PSNC access as owner-reported no-charge, and this
# test does not re-establish billing. The caps are the pre-specified ones at the same rate.
LIST_PRICE_USD = {"Qwen3.8-27B": (0.089121, 0.178491), "GLM-5.2": (6.134927, 12.269853),
                  "GLM-5.3": (6.134927, 12.269853)}
SPEND_CAP_USD = {"Qwen3.8-27B": 2.49, "GLM-5.2": 29.87, "GLM-5.3": 29.87}
EARLY_WINDOW = 48  # a model is stopped when more than EARLY_FAILURES of its first
EARLY_FAILURES = 5  # EARLY_WINDOW calls fail,
THINKING_LIMIT = 3  # or when more than THINKING_LIMIT answers carry reasoning text.

OUTPUT = HERE / "output"
CALLS = OUTPUT / "format-test-calls.jsonl"
SCORES = OUTPUT / "format-test-scores.jsonl"
SUMMARY = OUTPUT / "format-test-summary.json"
PROBES = OUTPUT / "format-test-probe.jsonl"
STUDY_OBSERVATIONS = OUTPUT / "example-selection-observations.jsonl"

# The order of the expected output: the schema's `required` list, which is also the order the
# model writes its answer in. Nested objects follow their own `required` lists.
TOP_KEYS = ("hasStatisticalModifier", "hasProperty", "hasObjectOfInterest", "hasMatrix",
            "hasContextObject", "hasConstraint")
NESTED_KEYS = ("SymmetricSystem", "hasPart", "AsymmetricSystem", "hasSource", "hasTarget",
               "hasNumerator", "hasDenominator", "label", "on")

# Invented for the reasoning-control probe; deliberately not a corpus variable.
PROBE_DEFINITION = ("Maximum hourly number of bicycles passing a city-centre counting station "
                    "on weekdays")
PROBE_UNSWITCHED_MAX_TOKENS = 1024

Transport = Callable[[Mapping[str, Any]], Awaitable[dict]]


# --- prompts -----------------------------------------------------------------------------

def _ordered(value: Any, order: Sequence[str]) -> Any:
    """Re-key nested dicts into the given order; unknown keys follow, sorted.

    Args: value: a JSON value; order: key order for this level (deeper levels use NESTED_KEYS).
    Returns: an equal value whose dict keys follow the order.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if isinstance(value, dict):
        keys = [k for k in order if k in value] + sorted(k for k in value if k not in order)
        return {k: _ordered(value[k], NESTED_KEYS) for k in keys}
    if isinstance(value, list):
        return [_ordered(item, NESTED_KEYS) for item in value]
    return value


def answer_text(gold: Mapping[str, Any]) -> str:
    """Render a gold decomposition exactly as the model is expected to return it.

    Args: gold: the six-field gold decomposition.
    Returns: indent-2 JSON in the schema's field order, non-ASCII characters kept.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return json.dumps(_ordered(gold, TOP_KEYS), ensure_ascii=False, indent=2)


def examples_block_b(examples: Sequence[Mapping[str, Any]]) -> str:
    """Render layout B: each example's definition followed by the answer it should produce.

    Args: examples: corpus records with `definition` and `gold`, in prompt order.
    Returns: the examples block, one `Example n` section per record.
    Raises: KeyError when a record lacks `definition` or `gold`.
    Side Effects: none. Deterministic.
    """
    return "\n\n".join(f"Example {n}\nDefinition: {example['definition']}\nOutput:\n"
                       f"{answer_text(example['gold'])}"
                       for n, example in enumerate(examples, 1))


def render_b(template: Any, schema_text: str, examples: Sequence[Mapping[str, Any]],
             definition: str) -> str:
    """Render a layout-B prompt with format A's template and single substitution pass.

    Args: template: the prompt version; schema_text: the schema JSON; examples: the 25 records;
        definition: the target definition.
    Returns: the full prompt; only the examples block differs from format A.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    values = {"schema": schema_text, "demonstrations": examples_block_b(examples),
              "target_definition": definition}
    return re.sub(r"\{\{(schema|demonstrations|target_definition)\}\}",
                  lambda match: values[match.group(1)], template.text)


RENDERERS = {"A": render_a, "B": render_b}


def examples_block(prompt: str) -> str:
    """Cut the examples block out of a rendered prompt, between the two template headings.

    Args: prompt: a prompt rendered from the matrix-decomposition template.
    Returns: the text between "ORDERED DEMONSTRATIONS" and "TARGET DEFINITION".
    Raises: IndexError when the prompt lacks the first heading.
    Side Effects: none. Deterministic.
    """
    return prompt.split("ORDERED DEMONSTRATIONS\n", 1)[1].split("\n\nTARGET DEFINITION", 1)[0]


def sha256(text: str) -> str:
    """Hash text for the fingerprints and the stored prompt identities.

    Args: text: any string.
    Returns: the SHA-256 hex digest of its UTF-8 bytes.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- inputs ------------------------------------------------------------------------------

def load_inputs(root: Path = ROOT) -> dict[str, Any]:
    """Load corpus, held-out targets, examples, template and schema. No network, no secrets.

    Args: root: the iadopt-lab root.
    Returns: {"records", "targets", "examples", "template", "schema", "schema_text"}.
    Raises: ValueError when the examples are not the study's top25 or overlap the targets.
    Side Effects: reads local files only. Deterministic for a given corpus.
    """
    from iadopt_lab.corpus.ingestion import load_canonical_records
    from iadopt_lab.prompting.renderer import load_prompt_version
    from iadopt_lab.validation import load_schema_bytes

    records = {row["variable_id"]: row for row in load_canonical_records(root)}
    targets = list(design.corpus_split(list(records.values()))["C"])
    if candidate_hash(EXAMPLE_IDS) != EXAMPLE_CANDIDATE:
        raise ValueError("the example ids are not the study's top25 candidate")
    if set(EXAMPLE_IDS) & set(targets):
        raise ValueError("an example is also an evaluation target")
    schema = load_schema_bytes(root)
    return {"records": records, "targets": targets,
            "examples": [records[v] for v in EXAMPLE_IDS],
            "template": load_prompt_version("matrix-decomposition", root),
            "schema": schema, "schema_text": schema.decode("utf-8")}


def prompt_for(inputs: Mapping[str, Any], fmt: str, definition: str) -> str:
    """Render the prompt that one call sends.

    Args: inputs: from `load_inputs`; fmt: "A" or "B"; definition: the target definition.
    Returns: the full prompt text.
    Raises: KeyError for an unknown layout.
    Side Effects: none. Deterministic.
    """
    return RENDERERS[fmt](inputs["template"], inputs["schema_text"], inputs["examples"],
                          definition)


def fingerprints(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Hash what the test sends: runner, template, schema, examples blocks and every prompt.

    Args: inputs: from `load_inputs`.
    Returns: hex digests, plus the mean prompt length per layout.
    Raises: nothing.
    Side Effects: reads this file. Deterministic.
    """
    result: dict[str, Any] = {
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "template_sha256": sha256(inputs["template"].text),
        "schema_sha256": hashlib.sha256(inputs["schema"]).hexdigest(),
        "example_candidate": candidate_hash(EXAMPLE_IDS),
        "targets_sha256": sha256("\n".join(sorted(inputs["targets"]))),
    }
    for fmt in FORMATS:
        prompts = [prompt_for(inputs, fmt, inputs["records"][v]["definition"])
                   for v in inputs["targets"]]
        result[f"examples_block_{fmt}_sha256"] = sha256(examples_block(prompts[0]))
        result[f"prompts_{fmt}_sha256"] = sha256("\n".join(sha256(p) for p in prompts))
        result[f"prompt_{fmt}_chars_mean"] = round(statistics.mean(len(p) for p in prompts), 1)
    return result


def schedule(targets: Sequence[str], models: Sequence[str] = MODELS,
             repetitions: int = REPETITIONS, seed: int = SCHEDULE_SEED) -> list[tuple]:
    """List every call of the grid, with A and B for one target and model side by side.

    Args: targets: evaluation ids; models: model ids; repetitions: per cell; seed: shuffle seed.
    Returns: (model, layout, repetition, variable_id) tuples, each exactly once.
    Raises: nothing.
    Side Effects: none. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    jobs: list[tuple] = []
    for repetition in range(1, repetitions + 1):
        order = list(targets)
        rng.shuffle(order)
        for variable_id in order:
            for model in models:
                formats = list(FORMATS)
                rng.shuffle(formats)
                jobs.extend((model, fmt, repetition, variable_id) for fmt in formats)
    return jobs


# --- provider ----------------------------------------------------------------------------

def psnc_transport(concurrency: int = CONCURRENCY) -> tuple[Transport, Any]:
    """Build a PSNC chat-completions transport over one shared client; stores no headers.

    Args: concurrency: size of the connection pool.
    Returns: (transport, client); the caller closes the client.
    Raises: SystemExit when PSNC_API_KEY is not set.
    Side Effects: reads the repository's .env through the lab's allowlisted loader; opens no
        connection until the first call.
    """
    from iadopt_lab.configuration import load_runtime_secrets

    secrets = load_runtime_secrets(str(ROOT.parent / ".env"),
                                   ["PSNC_API_KEY", "PSNC_API_BASE_URL"], os.environ)
    key = secrets.get("PSNC_API_KEY")
    if not key:
        raise SystemExit("PSNC_API_KEY is not set")
    # Same base and path rule as harness.build_context, so the endpoint matches the study.
    url = (secrets.get("PSNC_API_BASE_URL") or "https://llm.hpc.psnc.pl").rstrip("/")
    url += "/chat/completions"
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    client = httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False,
                               limits=httpx.Limits(max_connections=concurrency * 2,
                                                   max_keepalive_connections=concurrency))

    async def transport(body: Mapping[str, Any]) -> dict:
        """Send one request; a failure comes back as data, never as an exception.

        Args: body: the chat-completions request body.
        Returns: {"ok", "status", "latency"}, plus answer, reasoning and tokens on success.
        Raises: nothing.
        Side Effects: one HTTPS request on the shared client.
        """
        started = time.monotonic()
        try:
            response = await client.post(url, headers=headers, json=body)
        except Exception as error:  # noqa: BLE001 - a failed call is recorded, never raised
            return {"ok": False, "status": None, "error": type(error).__name__,
                    "latency": time.monotonic() - started}
        latency = time.monotonic() - started
        if response.status_code != 200:
            return {"ok": False, "status": response.status_code,
                    "error": f"HTTP {response.status_code}", "latency": latency}
        try:
            payload = response.json()
        except ValueError:
            return {"ok": False, "status": 200, "error": "body is not JSON", "latency": latency}
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = payload.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return {"ok": True, "status": 200, "latency": latency,
                "answer": message.get("content") or "",
                "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
                "reasoning_tokens": details.get("reasoning_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "served_model": payload.get("model")}

    return transport, client


def now() -> str:
    """Timestamp a call record.

    Args: none.
    Returns: the current UTC time, ISO 8601 with milliseconds.
    Raises: nothing.
    Side Effects: reads the clock. Operational only; never enters a hash.
    """
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def list_price_usd(model: str, record: Mapping[str, Any]) -> float:
    """Price one call at the hub's list price.

    Args: model: model id; record: a call record with token counts.
    Returns: US dollars; a missing token count counts as zero.
    Raises: KeyError for a model without a price.
    Side Effects: none. Deterministic.
    """
    price_in, price_out = LIST_PRICE_USD[model]
    return ((record.get("prompt_tokens") or 0) * price_in
            + (record.get("completion_tokens") or 0) * price_out) / 1e6


class Guard:
    """Stop rules, checked by every worker before every call."""

    def __init__(self) -> None:
        """Start with no stop reason, no spend and no counts.

        Args: none.
        Returns: nothing.
        Raises: nothing.
        Side Effects: none.
        """
        self.reasons: dict[str, str] = {}
        self.spend: dict[str, float] = defaultdict(float)
        self.calls: Counter = Counter()
        self.failures: Counter = Counter()
        self.thinking: Counter = Counter()

    def blocked(self, model: str) -> str | None:
        """Say why calls for this model must stop, if they must.

        Args: model: model id.
        Returns: the stop reason (one under "*" stops every model), or None.
        Raises: nothing.
        Side Effects: none.
        """
        return self.reasons.get("*") or self.reasons.get(model)

    def add_spend(self, model: str, record: Mapping[str, Any]) -> None:
        """Add one call's list price and stop the model once its cap is passed.

        Args: model: model id; record: a successful call record.
        Returns: nothing.
        Raises: KeyError for a model without a price or cap.
        Side Effects: updates the spend; may set a stop reason.
        """
        self.spend[model] += list_price_usd(model, record)
        if self.spend[model] > SPEND_CAP_USD[model]:
            self.reasons.setdefault(model, f"list-price spend ${self.spend[model]:.2f} "
                                           f"passed the ${SPEND_CAP_USD[model]:.2f} cap")

    def record(self, model: str, result: Mapping[str, Any]) -> None:
        """Apply the stop rules to one finished call.

        Args: model: model id; result: the stored call record.
        Returns: nothing.
        Raises: nothing.
        Side Effects: updates counts and spend; may stop the model, or every model on 401/403.
        """
        self.calls[model] += 1
        if result.get("status") in (401, 403):
            self.reasons.setdefault("*", f"credential refused (HTTP {result['status']})")
        if not result.get("ok"):
            self.failures[model] += 1
            if self.calls[model] <= EARLY_WINDOW and self.failures[model] > EARLY_FAILURES:
                self.reasons.setdefault(model, f"{self.failures[model]} of its first "
                                               f"{self.calls[model]} calls failed")
            return
        self.add_spend(model, result)
        if result.get("reasoning_chars"):
            self.thinking[model] += 1
            if self.thinking[model] > THINKING_LIMIT:
                self.reasons.setdefault(model, "answers carry reasoning text despite the switch")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file, skipping blank and torn lines.

    Args: path: the file, which need not exist.
    Returns: the parsed rows; empty when the file is absent.
    Raises: OSError when the file exists but cannot be read; a malformed line is skipped.
    Side Effects: reads the file.
    """
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn trailing line from a kill must not block a resume
    return rows


def call_key(record: Mapping[str, Any]) -> tuple:
    """Identify one call of the grid.

    Args: record: a call record.
    Returns: (model, layout, repetition, variable_id).
    Raises: KeyError when a field is missing.
    Side Effects: none.
    """
    return (record["model"], record["format"], record["repetition"], record["variable_id"])


def latest_records(path: Path) -> dict[tuple, dict]:
    """Keep one record per call: the first successful one, else the last failure.

    Args: path: the calls file.
    Returns: {call key: record}.
    Raises: as `read_jsonl`.
    Side Effects: reads the file.
    """
    latest: dict[tuple, dict] = {}
    for record in read_jsonl(path):
        key = call_key(record)
        if key not in latest or (record.get("ok") and not latest[key].get("ok")):
            latest[key] = record
        elif not latest[key].get("ok"):
            latest[key] = record
    return latest


async def run_grid(inputs: Mapping[str, Any], jobs: Sequence[tuple], transport: Transport,
                   calls_path: Path = CALLS, concurrency: int = CONCURRENCY,
                   max_calls: int | None = None, guard: Guard | None = None) -> dict:
    """Issue every job not already answered, appending one record per call.

    Args: inputs: from `load_inputs`; jobs: from `schedule`; transport: sends one request;
        calls_path: the calls file; concurrency: workers; max_calls: cap on new calls;
        guard: the stop rules, fresh by default.
    Returns: {"issued", "skipped", "stopped", "spend_usd"}.
    Raises: nothing from the provider; a failed call is recorded.
    Side Effects: provider calls; appends to `calls_path`. Resumable: answered calls are not
        repeated, failed ones are retried, and earlier spend counts toward the caps.
    """
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    guard = guard or Guard()
    answered = set()
    for key, record in latest_records(calls_path).items():
        if record.get("ok"):
            answered.add(key)
            guard.add_spend(key[0], record)
    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        if job not in answered:
            queue.put_nowait(job)
    lock = asyncio.Lock()
    issued: Counter = Counter()
    skipped: Counter = Counter()
    budget = [max_calls if max_calls is not None else math.inf]

    async def worker() -> None:
        """Take jobs until the queue is empty, honouring the stop rules and the call budget.

        Args: none.
        Returns: nothing.
        Raises: nothing.
        Side Effects: provider calls; appends records under the lock.
        """
        while True:
            try:
                model, fmt, repetition, variable_id = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if guard.blocked(model) or budget[0] <= 0:
                skipped[model] += 1
                continue
            budget[0] -= 1
            prompt = prompt_for(inputs, fmt, inputs["records"][variable_id]["definition"])
            body = dict(REQUEST, model=model, messages=[{"role": "user", "content": prompt}])
            started_at = now()
            result = await transport(body)
            reasoning = result.get("reasoning") or ""
            record = {"model": model, "format": fmt, "repetition": repetition,
                      "variable_id": variable_id, "started_at": started_at,
                      "ok": bool(result.get("ok")), "status": result.get("status"),
                      "error": result.get("error"), "answer": result.get("answer"),
                      "reasoning_chars": len(reasoning), "reasoning": reasoning or None,
                      "reasoning_tokens": result.get("reasoning_tokens"),
                      "finish_reason": result.get("finish_reason"),
                      "prompt_tokens": result.get("prompt_tokens"),
                      "completion_tokens": result.get("completion_tokens"),
                      "latency_seconds": round(result.get("latency") or 0.0, 3),
                      "served_model": result.get("served_model"),
                      "prompt_sha256": sha256(prompt)}
            issued[model] += 1
            guard.record(model, record)
            async with lock:
                with calls_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return {"issued": dict(issued), "skipped": dict(skipped), "stopped": dict(guard.reasons),
            "spend_usd": {m: round(v, 4) for m, v in guard.spend.items()}}


async def probe(inputs: Mapping[str, Any], models: Sequence[str], transport: Transport,
                path: Path = PROBES) -> list[dict]:
    """Check each model's reasoning switch on an invented target, before any grid call.

    Args: inputs: from `load_inputs`; models: model ids; transport: sends one request;
        path: the probe file.
    Returns: one row per call: two switched calls per model (layouts A and B), plus one
        unswitched call capped at 1,024 tokens for GLM-5.3, which the lab had not measured.
    Raises: nothing from the provider.
    Side Effects: provider calls; appends the rows to `path`.
    """
    rows = []
    for model in models:
        variants = [("A", True), ("B", True)] + ([("A", False)] if model == "GLM-5.3" else [])
        for fmt, switched in variants:
            prompt = prompt_for(inputs, fmt, PROBE_DEFINITION)
            body = dict(REQUEST, model=model, messages=[{"role": "user", "content": prompt}])
            if not switched:
                body.pop("chat_template_kwargs")
                body["max_tokens"] = PROBE_UNSWITCHED_MAX_TOKENS
            started_at = now()
            result = await transport(body)
            reasoning = result.get("reasoning") or ""
            rows.append({"model": model, "format": fmt, "switched": switched,
                         "started_at": started_at, "ok": bool(result.get("ok")),
                         "status": result.get("status"), "error": result.get("error"),
                         "answer": result.get("answer"), "reasoning_chars": len(reasoning),
                         "reasoning_tokens": result.get("reasoning_tokens"),
                         "finish_reason": result.get("finish_reason"),
                         "prompt_tokens": result.get("prompt_tokens"),
                         "completion_tokens": result.get("completion_tokens"),
                         "latency_seconds": round(result.get("latency") or 0.0, 3),
                         "served_model": result.get("served_model")})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def probe_passed(rows: Sequence[Mapping[str, Any]], model: str) -> bool:
    """Apply the gate: both switched calls succeed, with no reasoning and under 1,000 tokens.

    Args: rows: from `probe`; model: model id.
    Returns: True when the model may enter the grid.
    Raises: nothing.
    Side Effects: none.
    """
    switched = [r for r in rows if r["model"] == model and r["switched"]]
    return len(switched) == 2 and all(
        r["ok"] and r["reasoning_chars"] == 0 and (r["completion_tokens"] or 0) < 1000
        for r in switched)


# --- scoring and summary -----------------------------------------------------------------

def score_calls(inputs: Mapping[str, Any], calls_path: Path = CALLS,
                scores_path: Path = SCORES) -> dict[tuple, dict]:
    """Score every call's answer once; an unusable or missing answer scores as empty.

    Args: inputs: from `load_inputs`; calls_path: the calls file; scores_path: the score cache.
    Returns: {call key: {"key", "answer_sha256", "valid_prediction", "evaluation"}}.
    Raises: nothing for a bad answer; it scores as `empty_prediction()`.
    Side Effects: loads the local similarity model when an answer is new; appends new scores.
        Same extraction, validation and scorer as the study's harness. Idempotent per answer.
    """
    from iadopt_eval import evaluate_item
    from iadopt_lab.cli import _configuration, _similarity_backend
    from iadopt_lab.generation.extractor import extract_json
    from iadopt_lab.validation import validate_prediction
    from iadopt_lab.validation.lexical import empty_prediction

    cached = {(tuple(row["key"]), row["answer_sha256"]): row for row in read_jsonl(scores_path)}
    similarity = None
    scores: dict[tuple, dict] = {}
    with scores_path.open("a", encoding="utf-8") as handle:
        for key, record in latest_records(calls_path).items():
            answer = record.get("answer") if record.get("ok") else None
            digest = sha256(answer) if answer is not None else "no-answer"
            if (key, digest) in cached:
                scores[key] = cached[(key, digest)]
                continue
            if similarity is None:
                similarity = _similarity_backend(ROOT, _configuration(ROOT).data["evaluation"])
            valid, prediction = False, empty_prediction()
            if answer is not None:
                extracted = extract_json(answer)
                if extracted.success and validate_prediction(extracted.candidate,
                                                             inputs["schema"]).valid:
                    valid, prediction = True, extracted.candidate
            gold = inputs["records"][record["variable_id"]]["gold"]
            evaluation = evaluate_item(gold, prediction, similarity,
                                       metadata={"variable_id": record["variable_id"]},
                                       similarity_identity=similarity.identity)
            row = {"key": list(key), "answer_sha256": digest, "valid_prediction": valid,
                   "evaluation": evaluation}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            scores[key] = row
    return scores


def welch(b: Sequence[float], a: Sequence[float]) -> dict[str, float]:
    """Run Welch's two-sided t-test of mean(b) - mean(a), with a 95% confidence interval.

    Args: b, a: per-repetition scores, at least two in each.
    Returns: difference, t, df, p, ci_low, ci_high; p is 1 or 0 when both variances are zero.
    Raises: statistics.StatisticsError with fewer than two values in either group.
    Side Effects: none. Deterministic.
    """
    from scipy import stats

    diff = statistics.mean(b) - statistics.mean(a)
    var_b, var_a = statistics.variance(b) / len(b), statistics.variance(a) / len(a)
    se = math.sqrt(var_b + var_a)
    if se == 0:
        return {"difference": diff, "t": math.nan, "df": math.nan,
                "p": 1.0 if diff == 0 else 0.0, "ci_low": diff, "ci_high": diff}
    df = (var_b + var_a) ** 2 / (var_b ** 2 / (len(b) - 1) + var_a ** 2 / (len(a) - 1))
    t = diff / se
    half = float(stats.t.ppf(0.975, df)) * se
    return {"difference": diff, "t": t, "df": df, "p": float(2 * stats.t.sf(abs(t), df)),
            "ci_low": diff - half, "ci_high": diff + half}


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Adjust p-values with Holm's step-down correction.

    Args: pvalues: {name: p}.
    Returns: {name: adjusted p}, capped at 1 and monotone in rank.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    adjusted, running = {}, 0.0
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def _cell_rows(model: str, fmt: str, inputs: Mapping[str, Any], latest: Mapping[tuple, dict],
               scores: Mapping[tuple, dict]) -> tuple[list[dict], list[int]]:
    """Score one model and layout per repetition; a repetition missing a call is left out.

    Args: model, fmt: the cell; inputs: from `load_inputs`; latest: from `latest_records`;
        scores: from `score_calls`.
    Returns: (per-repetition rows, numbers of the incomplete repetitions).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    rows, incomplete = [], []
    for repetition in range(1, REPETITIONS + 1):
        keys = [(model, fmt, repetition, v) for v in inputs["targets"]]
        if not all(key in latest and key in scores for key in keys):
            incomplete.append(repetition)
            continue
        ids = [key[3] for key in keys]
        valid = [key for key in keys if scores[key]["valid_prediction"]]
        records = [latest[key] for key in keys]
        answered = [r for r in records if r.get("ok")]
        rows.append({
            "repetition": repetition,
            "close_f1_official": _aggregate([scores[k]["evaluation"] for k in keys], ids)["f1"],
            "close_f1_as_run": _aggregate([scores[k]["evaluation"] for k in valid],
                                          [k[3] for k in valid])["f1"],
            "invalid": len(keys) - len(valid),
            "failed_calls": len(records) - len(answered),
            "prompt_tokens": _mean(r.get("prompt_tokens") for r in answered),
            "completion_tokens": _mean(r.get("completion_tokens") for r in answered),
            "latency_seconds": _mean(r.get("latency_seconds") for r in answered),
            "list_price_usd": sum(list_price_usd(model, r) for r in answered),
            "reasoning_answers": sum(1 for r in records if r.get("reasoning_chars")),
        })
    return rows, incomplete


def _mean(values: Any) -> float | None:
    """Average the values that are present.

    Args: values: numbers, possibly with None among them.
    Returns: the mean, or None when nothing is present.
    Raises: nothing.
    Side Effects: none.
    """
    present = [v for v in values if v is not None]
    return statistics.mean(present) if present else None


def summarise(inputs: Mapping[str, Any], scores: Mapping[tuple, dict],
              calls_path: Path = CALLS) -> dict[str, Any]:
    """Compute per-cell statistics and the pre-specified comparisons.

    Args: inputs: from `load_inputs`; scores: from `score_calls`; calls_path: the calls file.
    Returns: {"generated_at", "cells", "primary", "secondary"}: primary is B against A per
        model with Holm over the models; secondary compares the models and checks drift.
    Raises: nothing.
    Side Effects: reads the calls file and, when present, the study's observations.
    """
    latest = latest_records(calls_path)
    cells: dict[str, Any] = {}
    for model in MODELS:
        for fmt in FORMATS:
            rows, incomplete = _cell_rows(model, fmt, inputs, latest, scores)
            if not rows:
                continue
            f1 = [row["close_f1_official"] for row in rows]
            cells[f"{model}|{fmt}"] = {
                "repetitions": len(rows), "incomplete_repetitions": incomplete,
                "close_f1_official_mean": statistics.mean(f1),
                "close_f1_official_sd": statistics.stdev(f1) if len(f1) > 1 else None,
                "close_f1_as_run_mean": statistics.mean(r["close_f1_as_run"] for r in rows),
                "invalid": sum(r["invalid"] for r in rows),
                "failed_calls": sum(r["failed_calls"] for r in rows),
                "reasoning_answers": sum(r["reasoning_answers"] for r in rows),
                "prompt_tokens_mean": _mean(r["prompt_tokens"] for r in rows),
                "completion_tokens_mean": _mean(r["completion_tokens"] for r in rows),
                "latency_seconds_mean": _mean(r["latency_seconds"] for r in rows),
                "list_price_usd": sum(r["list_price_usd"] for r in rows),
                "per_repetition": rows,
            }

    def series(model: str, fmt: str) -> list[float]:
        """Return one cell's per-repetition Close F1.

        Args: model, fmt: the cell.
        Returns: the scores; empty when the cell did not run.
        Raises: nothing.
        Side Effects: none.
        """
        cell = cells.get(f"{model}|{fmt}")
        return [r["close_f1_official"] for r in cell["per_repetition"]] if cell else []

    primary = {model: welch(series(model, "B"), series(model, "A")) for model in MODELS
               if len(series(model, "A")) > 1 and len(series(model, "B")) > 1}
    adjusted = holm({model: result["p"] for model, result in primary.items()})
    for model, result in primary.items():
        result["p_holm"] = adjusted[model]
        if adjusted[model] < 0.05:
            result["verdict"] = "B better" if result["difference"] > 0 else "A better"
        else:
            result["verdict"] = "cannot tell apart"
    secondary = {f"{glm} vs Qwen3.8-27B, format {fmt}":
                 welch(series(glm, fmt), series("Qwen3.8-27B", fmt))
                 for glm in MODELS[1:] for fmt in FORMATS
                 if len(series(glm, fmt)) > 1 and len(series("Qwen3.8-27B", fmt)) > 1}
    study = [row["result"]["close_f1_official"] for row in read_jsonl(STUDY_OBSERVATIONS)
             if row.get("stage") == "confirm" and row.get("name") == "top25"]
    if len(study) > 1 and len(series("Qwen3.8-27B", "A")) > 1:
        secondary["drift: Qwen3.8-27B format A now vs the study (2026-09-12)"] = dict(
            welch(series("Qwen3.8-27B", "A"), study), study_mean=statistics.mean(study),
            study_repetitions=len(study))
    return {"generated_at": now(), "cells": cells, "primary": primary, "secondary": secondary}


# --- entry point -------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode: plan, probe, run or score.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status, 0 on success.
    Raises: SystemExit on bad arguments or a missing PSNC key.
    Side Effects: --probe and --run call the provider; --run and --score write output files.
    """
    parser = argparse.ArgumentParser(description="Example-layout test (exploratory).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--score", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(LIST_PRICE_USD),
                        default=list(MODELS))
    parser.add_argument("--max-calls", type=int, default=None,
                        help="issue at most this many new calls (smoke test)")
    args = parser.parse_args(argv)
    inputs = load_inputs()

    if args.plan_only:
        jobs = schedule(inputs["targets"], args.models)
        print(json.dumps({"targets": len(inputs["targets"]), "examples": len(EXAMPLE_IDS),
                          "calls": len(jobs), "per_cell": REPETITIONS * len(inputs["targets"]),
                          **fingerprints(inputs)}, indent=2))
        return 0
    if args.score:
        scores = score_calls(inputs)
        summary = summarise(inputs, scores)
        SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
        for name, cell in summary["cells"].items():
            print(f"{name:16} reps {cell['repetitions']:2}  F1 {cell['close_f1_official_mean']:.4f}"
                  f" (SD {cell['close_f1_official_sd'] or 0:.4f})  invalid {cell['invalid']:3}"
                  f"  prompt {cell['prompt_tokens_mean']:.0f} tok"
                  f"  {cell['latency_seconds_mean']:.2f} s  ${cell['list_price_usd']:.2f}")
        for name, result in {**summary["primary"], **summary["secondary"]}.items():
            print(f"{name}: diff {result['difference']:+.4f} "
                  f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] p {result['p']:.4f}"
                  + (f" holm {result['p_holm']:.4f} -> {result['verdict']}"
                     if "verdict" in result else ""))
        return 0

    async def go() -> int:
        """Open the transport, run the probe or the grid, and always close the client.

        Args: none.
        Returns: the exit status.
        Raises: SystemExit when the PSNC key is missing.
        Side Effects: provider calls and output files.
        """
        transport, client = psnc_transport()
        try:
            if args.probe:
                rows = await probe(inputs, args.models, transport)
                for row in rows:
                    print(json.dumps({k: row[k] for k in (
                        "model", "format", "switched", "ok", "status", "error",
                        "reasoning_chars", "reasoning_tokens", "finish_reason",
                        "prompt_tokens", "completion_tokens", "latency_seconds",
                        "served_model")}))
                for model in args.models:
                    print(model, "passes" if probe_passed(rows, model) else "FAILS", "the probe")
                return 0
            result = await run_grid(inputs, schedule(inputs["targets"], args.models), transport,
                                    max_calls=args.max_calls)
            print(json.dumps(result, indent=2))
            return 0
        finally:
            await client.aclose()

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
