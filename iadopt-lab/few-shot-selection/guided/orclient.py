#!/usr/bin/env python
"""OpenRouter access for the pattern-retrieval stages: one pinned host, real costs, a hard cap.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/PATTERN_RETRIEVAL_20261119.md` (private repository).

Steps 1 to 5 ran on PSNC. On 2026-11-19 that gateway began refusing every chat request of the
team's key (HTTP 429, "Budget has been exceeded"), and Arvin moved the tests to OpenRouter with the
same model and a cap of 50 US dollars. This module is the only place that talks to OpenRouter:

- one model, `qwen/qwen3.8-27b`, served by one pinned host with fallbacks off, so the answering
  machine never changes inside a comparison; reasoning is switched off, as on PSNC;
- every reply's cost, as OpenRouter reports it, goes to one ledger file, and no request is sent
  that could take the ledger's total over the cap. A request whose cost is not known (a reply
  without a cost, a timeout, a dropped connection) is booked at an upper bound, so the ledger
  can only count too much. One process at a time may use the ledger;
- the key is read by name through the lab's allowlisted loader and is used for the
  `Authorization` header only. No title, referrer or account detail is sent; the HTTP library
  adds its own standard headers.

Results on this host are not interchangeable with the PSNC results of steps 1 to 5: another
machine, another precision of the weights, other sampling defaults. Every comparison therefore
carries its own baseline arm, run here.

    ./orclient.py --status      # the key's credit as OpenRouter states it, and the ledger total
    ./orclient.py --hosts       # the hosts that pass the pre-specified rule, cheapest first
    ./orclient.py --probe       # invented prompts only: does a host answer as the runs need?
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import engine  # noqa: E402
from engine import ft  # noqa: E402

API = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen3.8-27b"
KEY_NAME = "OPENROUTER_API_KEY"
CAP_USD = 50.0                      # Arvin, 2026-11-19: "have a cap of 50 dollars"
OUTPUT = HERE.parent / "output" / "retrieval"
LEDGER = OUTPUT / "openrouter-ledger.jsonl"
PROBE = OUTPUT / "openrouter-probe.jsonl"
CONCURRENCY = 16
TIMEOUT = 180.0
MAX_ATTEMPTS = 3
BACKOFF = (3.0, 12.0)
MAX_RETRY_AFTER = 60.0
# Transient: worth another attempt. Everything else will not change by asking again.
RETRYABLE = frozenset({408, 425, 429, 500, 502, 503, 504, 524, 529})
# The credential or the account's credit is the problem, so every other request fails the same way.
FATAL = frozenset({401, 402, 403})
EARLY_WINDOW, EARLY_FAILURES, THINKING_LIMIT = engine.EARLY_WINDOW, engine.EARLY_FAILURES, engine.THINKING_LIMIT
LATE_WINDOW, LATE_FAILURES = 50, 25     # after the early window: stop when half of the last 50 calls failed
# Error bodies from OpenRouter echo the account's user id; it identifies the account holder. The
# id can sit inside a JSON string nested in another, where its quotes are escaped.
_USER_ID = re.compile(r'(\\*"user_id\\*"\s*:\s*\\*")[^"\\]*')
_TOKEN = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")

# The host rule, written down before any host was tried (record, "Provider"): a host must declare
# one of the floating-point precisions below (8 bits or more), must list every parameter the
# stages use, and must have answered at least 99.9% of the last day's requests. Hosts are tried
# cheapest first for a typical call.
MIN_PRECISION = ("fp8", "fp16", "bf16", "fp32")
NEEDED_PARAMETERS = ("temperature", "top_p", "max_tokens", "seed", "reasoning", "logprobs", "top_logprobs")
MIN_UPTIME_DAY = 99.9
TYPICAL_CALL = (4300, 90)           # prompt and answer tokens of one fill call in steps 1 to 5

# The pinned host, chosen by the rule above and the probe on 2026-11-19. Prices are US dollars
# per million tokens as listed that day; they bound the cost of a call, the ledger holds the
# real cost.
HOST = {"slug": "near-ai", "name": "Near AI", "quantization": "fp8",
        "input_usd_per_m": 0.04, "output_usd_per_m": 1.35}

Transport = Callable[[Mapping[str, Any]], Awaitable[dict]]


def scrub(text: str, key: str) -> str:
    """Remove the key, anything shaped like a token and the account's user id from stored text.

    Args: text: any message; key: the API key.
    Returns: the text with those values replaced. Scrub before truncating.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return _USER_ID.sub(r"\1<redacted>", _TOKEN.sub("<token>", text.replace(key, "<redacted>")))


def api_key() -> str:
    """Read the OpenRouter key by name, without ever showing it.

    Args: none.
    Returns: the key.
    Raises: SystemExit when the key is not set.
    Side Effects: reads the repository's .env through the lab's allowlisted loader.
    """
    from iadopt_lab.configuration import load_runtime_secrets

    secrets = load_runtime_secrets(str(engine.ROOT.parent / ".env"), [KEY_NAME], os.environ)
    key = secrets.get(KEY_NAME)
    if not key:
        raise SystemExit(f"{KEY_NAME} is not set")
    return key


def worst_call_usd(max_tokens: int, prompt_tokens: int = 12000, host: Mapping[str, Any] = HOST) -> float:
    """An upper bound on the cost of one call, for the cap's reservation.

    Args: max_tokens: the request's answer limit; prompt_tokens: a bound on the prompt, about
        twice the longest prompt of steps 1 to 5; host: the pinned host's listed prices.
    Returns: US dollars.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return (prompt_tokens * host["input_usd_per_m"] + max_tokens * host["output_usd_per_m"]) / 1e6


def booked(row: Mapping[str, Any]) -> float:
    """What one ledger row counts towards the cap.

    Args: row: a ledger row.
    Returns: its "booked_usd" (the reported cost, or an upper bound where the cost is not known);
        for a row written before that field existed, its reported cost.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    value = row.get("booked_usd")
    return float(value if value is not None else row.get("cost_usd") or 0.0)


class Spend:
    """The cap. Counts what the ledger says was spent and reserves room for calls in flight."""

    def __init__(self, ledger: Path = LEDGER, cap: float = CAP_USD, limit: float | None = None,
                 purpose: str | None = None, allowance: float | None = None) -> None:
        """Read the ledger and set the limit of this run.

        Args: ledger: the ledger file; cap: the overall cap; limit: a lower limit as a total over
            the whole ledger; purpose, allowance: the most the ledger rows of one purpose may add
            up to, over every launch of that purpose. The limit is never above the cap.
        Returns: nothing.
        Raises: OSError when the ledger exists but cannot be read.
        Side Effects: reads the ledger.
        """
        self.ledger, self.cap, self.total_limit = ledger, cap, limit
        self.purpose, self.allowance = purpose, allowance
        self.reserved = 0.0
        self.refused = 0
        self.failure: str | None = None
        self.spent = self.limit = 0.0
        self.refresh()

    def refresh(self) -> None:
        """Read the ledger again and work out the limit.

        Args: none.
        Returns: nothing.
        Raises: OSError when the ledger exists but cannot be read.
        Side Effects: reads the ledger; resets the spent total and the limit.
        """
        rows = list(ft.read_jsonl(self.ledger))
        self.spent = sum(booked(row) for row in rows)
        self.limit = self.cap
        if self.total_limit is not None:
            self.limit = min(self.limit, self.total_limit)
        if self.allowance is not None:
            here = sum(booked(row) for row in rows if row.get("purpose") == self.purpose)
            self.limit = min(self.limit, self.spent - here + self.allowance)

    def reserve(self, amount: float) -> bool:
        """Set room aside for one call, if the limit allows it.

        Args: amount: an upper bound on the call's cost.
        Returns: True when the call may be sent.
        Raises: nothing.
        Side Effects: raises the reservation, or counts a refusal.
        """
        if self.failure or self.spent + self.reserved + amount > self.limit:
            self.refused += 1
            return False
        self.reserved += amount
        return True

    def settle(self, amount: float, row: Mapping[str, Any]) -> None:
        """Release one call's reservation and write what it cost.

        Args: amount: the reservation made for the call; row: the ledger row, with "booked_usd".
        Returns: nothing.
        Raises: nothing. A ledger that cannot be written sets `failure`, after which nothing more
            is sent: money that is not on the ledger cannot be counted by a later run.
        Side Effects: appends one line to the ledger.
        """
        self.reserved = max(0.0, self.reserved - amount)
        self.spent += booked(row)
        try:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as error:
            self.failure = self.failure or f"the ledger cannot be written ({type(error).__name__})"

    def refusal(self) -> str:
        """Why `reserve` said no.

        Args: none.
        Returns: the ledger failure, else "spend cap reached".
        Raises: nothing.
        Side Effects: none.
        """
        return self.failure or "spend cap reached"

    def room(self) -> float:
        """What may still be spent under this run's limit.

        Args: none.
        Returns: US dollars, never negative.
        Raises: nothing.
        Side Effects: none.
        """
        return max(0.0, self.limit - self.spent - self.reserved)


def translate(body: Mapping[str, Any], host: Mapping[str, Any] = HOST) -> dict[str, Any]:
    """Turn a request body of steps 1 to 5 into the body OpenRouter gets.

    Args: body: a chat request as the PSNC runners build it; host: the pinned host.
    Returns: the OpenRouter body: this module's model, the caller's messages and sampling settings,
        reasoning off, one host with fallbacks off, and cost accounting on.
    Raises: KeyError when the body has no messages.
    Side Effects: none. Deterministic.
    """
    out: dict[str, Any] = {"model": MODEL, "messages": list(body["messages"]), "stream": False}
    for name in ("temperature", "top_p", "max_tokens", "seed", "logprobs", "top_logprobs"):
        if name in body:
            out[name] = body[name]
    out["reasoning"] = {"enabled": False}
    out["provider"] = {"only": [host["slug"]], "allow_fallbacks": False, "require_parameters": True,
                       "quantizations": [host["quantization"]]}
    out["usage"] = {"include": True}
    return out


def read(status: int, text: str, key: str) -> dict[str, Any]:
    """Read one HTTP reply into the result the runners store.

    Args: status: HTTP status; text: the body; key: the API key, for scrubbing only.
    Returns: {"ok", "status", "error"} for a failure; for an answer also "answer", "reasoning",
        "reasoning_tokens", "finish_reason", "prompt_tokens", "completion_tokens", "cached_tokens",
        "cost_usd", "served_model", "host", "generation_id" and "top" (first-token alternatives).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if status != 200:
        return {"ok": False, "status": status, "error": f"HTTP {status}: " + scrub(text, key)[:300]}
    try:
        payload = json.loads(text)
    except ValueError:
        return {"ok": False, "status": 200, "error": "body is not JSON"}
    choices = payload.get("choices") if isinstance(payload, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    problem = (payload.get("error") if isinstance(payload, dict) else None) or (choice or {}).get("error")
    if problem or choice is None:
        code = problem.get("code") if isinstance(problem, dict) else None
        return {"ok": False, "status": code if isinstance(code, int) else 200,
                "error": "error in body: " + scrub(json.dumps(problem, ensure_ascii=False), key)[:300]}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    content = (choice.get("logprobs") or {}).get("content") or []
    return {"ok": True, "status": 200, "error": None, "answer": message.get("content") or "",
            "reasoning": message.get("reasoning") or message.get("reasoning_content") or "",
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "cost_usd": usage.get("cost"), "served_model": payload.get("model"),
            "host": payload.get("provider"), "generation_id": payload.get("id"),
            "top": [{"token": t.get("token"), "logprob": t.get("logprob")}
                    for t in (content[0].get("top_logprobs") or [])] if content else []}


def book(result: Mapping[str, Any], bound: float, host: Mapping[str, Any] = HOST) -> tuple[float, bool]:
    """What one sent attempt counts towards the cap.

    Args: result: from `read`, or the failure of a request that was sent; bound: the worst case
        reserved for the call; host: the pinned host's listed prices.
    Returns: (US dollars, whether the amount is an estimate): the reported cost where there is
        one; for an answer without a cost its tokens at the listed prices (the bound when the
        tokens are missing too); the bound for an attempt whose outcome is not known (no HTTP
        status, or a 200 reply that is no answer), because it may have been billed; nothing for
        an attempt refused or failed with another HTTP status.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if result.get("cost_usd") is not None:
        return float(result["cost_usd"]), False
    if result.get("ok"):
        tokens_in, tokens_out = result.get("prompt_tokens"), result.get("completion_tokens")
        if tokens_in is None or tokens_out is None:
            return bound, True
        return (tokens_in * host["input_usd_per_m"] + tokens_out * host["output_usd_per_m"]) / 1e6, True
    if result.get("status") in (None, 200):
        return bound, True
    return 0.0, False


def lock_ledger(ledger: Path) -> Any:
    """Take the ledger for this process alone.

    Two processes that each read the ledger once could together spend past the limit, so a second
    one is refused while the first holds the lock.

    Args: ledger: the ledger file.
    Returns: the open lock file; closing it releases the lock.
    Raises: SystemExit when another process holds the lock.
    Side Effects: creates `<ledger>.lock` beside the ledger.
    """
    ledger.parent.mkdir(parents=True, exist_ok=True)
    handle = (ledger.parent / (ledger.name + ".lock")).open("w", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit("another process is using the spending ledger; wait for it to finish") from None
    return handle


def open_transport(spend: Spend, purpose: str, concurrency: int = CONCURRENCY,
                   host: Mapping[str, Any] = HOST, client: Any = None, key: str | None = None,
                   sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> tuple[Transport, Any]:
    """Build the OpenRouter transport over one shared client.

    Args: spend: the cap; purpose: a label for the ledger rows; concurrency: size of the
        connection pool; host: the host every request is pinned to; client, key, sleep: for
        tests; by default an httpx client, the key read by name and `asyncio.sleep`.
    Returns: (transport, client); the caller ends both with `close_transport`. The transport
        returns failures as data, retries transient ones at most `MAX_ATTEMPTS` times, and
        refuses to send (a result with "refused") when the limit has no room or the ledger
        cannot be written.
    Raises: SystemExit when the key is not set or another process holds the ledger.
    Side Effects: reads the key; locks the ledger and reads it again; opens no connection until
        the first call.
    """
    key = key or api_key()
    held = lock_ledger(spend.ledger)
    spend.refresh()
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    if client is None:
        import httpx

        client = httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False,
                                   limits=httpx.Limits(max_connections=concurrency * 2,
                                                       max_keepalive_connections=concurrency))
    lock = asyncio.Lock()

    async def transport(body: Mapping[str, Any]) -> dict:
        """Send one request, with bounded retries; a failure comes back as data.

        Args: body: a chat request as the runners build it.
        Returns: the result of `read`, plus "latency" and "attempts"; a refusal has "refused".
        Raises: nothing.
        Side Effects: HTTPS requests; one ledger row per attempt that was sent.
        """
        request = translate(body, host)
        # Half a token per character is more than any prompt of these stages needs, so the bound
        # stays an upper bound for prompts longer than the 12,000 tokens assumed by default.
        chars = sum(len(str(message.get("content") or "")) for message in request["messages"])
        bound = worst_call_usd(int(request.get("max_tokens") or 16000), max(12000, chars // 2), host)
        started = time.monotonic()
        result: dict[str, Any] = {"ok": False, "status": None, "error": "not sent"}
        attempts = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            async with lock:
                allowed = spend.reserve(bound)
            if not allowed:
                result = {"ok": False, "status": None, "error": spend.refusal(), "refused": True}
                break
            attempts = attempt
            wait = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
            try:
                response = await client.post(API + "/chat/completions", headers=headers, json=request)
                result = read(response.status_code, response.text, key)
                retry_after = response.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    wait = min(float(retry_after), MAX_RETRY_AFTER)
            except Exception as error:  # noqa: BLE001 - a failed call is recorded, never raised
                result = {"ok": False, "status": None, "error": type(error).__name__}
            amount, estimated = book(result, bound, host)
            async with lock:
                spend.settle(bound, {"at": ft.now(), "purpose": purpose, "ok": result["ok"],
                                     "status": result.get("status"), "attempt": attempt,
                                     "prompt_tokens": result.get("prompt_tokens"),
                                     "completion_tokens": result.get("completion_tokens"),
                                     "cached_tokens": result.get("cached_tokens"),
                                     "cost_usd": result.get("cost_usd"), "booked_usd": amount,
                                     "estimated": estimated, "host": result.get("host")})
            transient = result.get("status") in RETRYABLE or result.get("status") is None
            if result["ok"] or not transient or attempt == MAX_ATTEMPTS:
                break
            await sleep(wait)
        return dict(result, latency=time.monotonic() - started, attempts=attempts)

    transport.ledger_lock = held  # type: ignore[attr-defined]
    return transport, client


async def close_transport(transport: Transport, client: Any) -> None:
    """Close the client and give the ledger back.

    Args: transport, client: from `open_transport`.
    Returns: nothing.
    Raises: nothing for a client without `aclose`.
    Side Effects: closes connections; releases the ledger lock.
    """
    if hasattr(client, "aclose"):
        await client.aclose()
    held = getattr(transport, "ledger_lock", None)
    if held is not None:
        held.close()


class Guard:
    """Stop rules, checked by every worker before every call."""

    def __init__(self, spend: Spend, host: Mapping[str, Any] = HOST) -> None:
        """Start with no stop reason and no counts.

        Args: spend: the cap; host: the host every answer must come from.
        Returns: nothing.
        Raises: nothing.
        Side Effects: none.
        """
        self.spend, self.host = spend, host
        self.reason: str | None = None
        self.calls = self.failures = self.thinking = 0
        self.recent: list[bool] = []

    def stop(self, reason: str) -> None:
        """Stop the run; the first reason given is kept.

        Args: reason: why.
        Returns: nothing.
        Raises: nothing.
        Side Effects: sets the stop reason.
        """
        self.reason = self.reason or reason

    def record(self, record: Mapping[str, Any]) -> None:
        """Apply the stop rules to one finished call.

        Args: record: the call record, as the host answered it.
        Returns: nothing.
        Raises: nothing.
        Side Effects: updates the counts; may set the stop reason.
        """
        self.calls += 1
        answered = bool(record.get("ok"))
        self.recent = [*self.recent, answered][-LATE_WINDOW:]
        if record.get("status") in FATAL:
            self.stop(f"credential or credit refused (HTTP {record['status']})")
        if not answered:
            self.failures += 1
            if self.calls <= EARLY_WINDOW and self.failures > EARLY_FAILURES:
                self.stop(f"{self.failures} of the first {self.calls} calls failed")
            if len(self.recent) == LATE_WINDOW and self.recent.count(False) > LATE_FAILURES:
                self.stop(f"{self.recent.count(False)} of the last {LATE_WINDOW} calls failed")
            return
        if record.get("host") != self.host["name"]:
            self.stop(f"an answer came from {record.get('host')!r}, not the pinned host")
        if record.get("reasoning_chars") or record.get("reasoning_tokens"):
            self.thinking += 1
            if self.thinking > THINKING_LIMIT:
                self.stop("answers carry reasoning despite the switch")


async def run_jobs(jobs: Sequence[engine.Job], prompt_of: Callable[[engine.Job], str], transport: Transport,
                   calls_path: Path, guard: Guard, concurrency: int = CONCURRENCY,
                   request: Mapping[str, Any] | None = None, max_calls: int | None = None) -> dict:
    """Issue every job not already answered, appending one record per call.

    Args: jobs: from `engine.schedule`; prompt_of: builds a job's prompt; transport: from
        `open_transport`; calls_path: the calls file; guard: the stop rules; concurrency: workers;
        request: sampling settings, the study's by default; max_calls: cap on new calls.
    Returns: {"issued", "skipped", "stopped", "spent_usd", "room_usd"}.
    Raises: nothing from the provider; a failed call is recorded.
    Side Effects: provider calls; appends to `calls_path`. Resumable: answered calls are not
        repeated and failed ones are retried. The records are those of `engine.run_jobs` plus the
        cost, the cached tokens, the host, the attempts and the generation id, so
        `engine.latest` and `engine.score` read them unchanged. An answer from another host than
        the pinned one is stored as a failed call, so it is never scored and is asked again. A
        call the cap refused is not stored.
    """
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in dict(request or engine.REQUEST).items() if k != "chat_template_kwargs"}
    answered = {key for key, record in engine.latest(calls_path).items() if record.get("ok")}
    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        if job not in answered:
            queue.put_nowait(job)
    lock = asyncio.Lock()
    counts: Counter = Counter()
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
            started_at = ft.now()
            result = await transport(dict(settings, messages=[{"role": "user", "content": prompt}]))
            reasoning = result.get("reasoning") or ""
            record = {"arm": job[0], "repetition": job[1], "variable_id": job[2],
                      "started_at": started_at, "ok": bool(result.get("ok")),
                      "status": result.get("status"), "error": result.get("error"),
                      "answer": result.get("answer"), "reasoning_chars": len(reasoning),
                      "reasoning": reasoning or None, "reasoning_tokens": result.get("reasoning_tokens"),
                      "finish_reason": result.get("finish_reason"),
                      "prompt_tokens": result.get("prompt_tokens"),
                      "completion_tokens": result.get("completion_tokens"),
                      "cached_tokens": result.get("cached_tokens"), "cost_usd": result.get("cost_usd"),
                      "latency_seconds": round(result.get("latency") or 0.0, 3),
                      "attempts": result.get("attempts"), "served_model": result.get("served_model"),
                      "host": result.get("host"), "generation_id": result.get("generation_id"),
                      "prompt_sha256": ft.sha256(prompt)}
            if result.get("refused"):
                guard.stop(str(result.get("error") or "spend cap reached"))
                counts["skipped"] += 1
                continue                      # nothing was sent, so nothing is stored
            guard.record(record)
            if record["ok"] and record["host"] != guard.host["name"]:
                record["ok"], record["error"] = False, f"served by another host: {record['host']}"
            counts["issued"] += 1
            async with lock:
                with calls_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return {"issued": counts["issued"], "skipped": counts["skipped"], "stopped": guard.reason,
            "spent_usd": round(guard.spend.spent, 4), "room_usd": round(guard.spend.room(), 4)}


def eligible(endpoints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The hosts that pass the pre-specified rule, cheapest first for a typical call.

    Args: endpoints: the "endpoints" list of OpenRouter's model listing.
    Returns: rows with the host's name, tag, precision, listed prices in US dollars per million
        tokens, last-day uptime and the cost of a typical call.
    Raises: nothing; a host with unreadable prices is left out.
    Side Effects: none. Deterministic.
    """
    rows = []
    for endpoint in endpoints:
        supported = set(endpoint.get("supported_parameters") or [])
        uptime = endpoint.get("uptime_last_1d")
        try:
            price_in = float(endpoint["pricing"]["prompt"]) * 1e6
            price_out = float(endpoint["pricing"]["completion"]) * 1e6
        except (KeyError, TypeError, ValueError):
            continue
        if (endpoint.get("quantization") not in MIN_PRECISION or not set(NEEDED_PARAMETERS) <= supported
                or uptime is None or uptime < MIN_UPTIME_DAY):
            continue
        rows.append({"name": endpoint.get("provider_name"), "tag": endpoint.get("tag"),
                     "quantization": endpoint.get("quantization"),
                     "input_usd_per_m": round(price_in, 6), "output_usd_per_m": round(price_out, 6),
                     "uptime_last_1d": uptime,
                     "typical_call_usd": (TYPICAL_CALL[0] * price_in + TYPICAL_CALL[1] * price_out) / 1e6})
    return sorted(rows, key=lambda row: (row["typical_call_usd"], row["name"] or ""))


async def listed_hosts() -> list[dict[str, Any]]:
    """Fetch the model's host listing. Public; sends no credential.

    Args: none.
    Returns: the "endpoints" list.
    Raises: httpx.HTTPError when the listing cannot be read.
    Side Effects: one HTTPS request.
    """
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(f"{API}/models/{MODEL}/endpoints")
        response.raise_for_status()
        return response.json()["data"]["endpoints"]


async def status() -> dict[str, Any]:
    """What OpenRouter says about the key's credit, next to the ledger. Prints no identifier.

    Args: none.
    Returns: {"key": usage, limit and remaining credit of the key, "credits": the account's total
        credits and usage when the key may read them, "ledger_usd", "cap_usd"}.
    Raises: SystemExit when the key is not set.
    Side Effects: two HTTPS requests; reads the ledger.
    """
    import httpx

    headers = {"Authorization": "Bearer " + api_key()}
    out: dict[str, Any] = {"ledger_usd": round(Spend().spent, 6), "cap_usd": CAP_USD}
    async with httpx.AsyncClient(timeout=60.0) as client:
        for name, path, fields in (("key", "/key", ("usage", "limit", "limit_remaining", "is_free_tier",
                                                     "usage_daily", "usage_monthly", "limit_reset")),
                                   ("credits", "/credits", ("total_credits", "total_usage"))):
            try:
                response = await client.get(API + path, headers=headers)
                data = response.json().get("data") if response.status_code == 200 else None
            except (httpx.HTTPError, ValueError) as error:
                out[name] = {"error": type(error).__name__}
                continue
            out[name] = ({field: data.get(field) for field in fields} if isinstance(data, dict)
                         else {"http_status": response.status_code})
    return out


async def probe(hosts: Sequence[Mapping[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Ask up to `limit` hosts the same invented questions and report how each answers.

    Args: hosts: from `eligible`; limit: how many hosts to try, cheapest first.
    Returns: one row per host: whether every check passed, and each check.
    Raises: SystemExit when the key is not set.
    Side Effects: provider calls for invented definitions only (the fill prompt holds the study's
        25 examples, as every dev run of steps 1 to 5 did); appends to the probe file and the ledger.
    """
    import ceiling

    from iadopt_lab.generation.extractor import extract_json
    from iadopt_lab.validation import validate_prediction

    inputs = ft.load_inputs()
    fills = [ft.prompt_for(inputs, "B", item["definition"]) for item in ceiling.INVENTED[:2]]
    choice = ("Answer with the number of the right option only.\nQuestion: Is a bicycle a vehicle?\n"
              "1 yes\n2 no\nAnswer:")
    settings = {k: v for k, v in engine.REQUEST.items() if k != "chat_template_kwargs"}
    rows = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for row in list(hosts)[:limit]:
        host = {"slug": (row["tag"] or "").split("/")[0], "name": row["name"], "quantization": row["quantization"],
                "input_usd_per_m": row["input_usd_per_m"], "output_usd_per_m": row["output_usd_per_m"]}
        spend = Spend(limit=Spend().spent + 0.50)
        transport, client = open_transport(spend, f"probe:{host['slug']}", concurrency=2, host=host)
        try:
            answers = [await transport(dict(settings, messages=[{"role": "user", "content": p}]))
                       for p in (fills[0], fills[1], fills[0])]
            one = await transport({"temperature": 0.0, "top_p": 1.0, "max_tokens": 1, "logprobs": True,
                                   "top_logprobs": 5, "messages": [{"role": "user", "content": choice}]})
        finally:
            await close_transport(transport, client)
        valid = []
        for answer in answers:
            extracted = extract_json(answer.get("answer") or "")
            valid.append(bool(answer.get("ok")) and extracted.success
                         and validate_prediction(extracted.candidate, inputs["schema"]).valid)
        checks = {"answered": all(a.get("ok") for a in answers) and bool(one.get("ok")),
                  "valid_json": all(valid),
                  "no_reasoning": not any(a.get("reasoning") or a.get("reasoning_tokens") for a in answers),
                  "served_by_host": all(a.get("host") == host["name"] for a in answers if a.get("ok")),
                  "cost_reported": all(a.get("cost_usd") is not None for a in answers if a.get("ok")),
                  "first_token_alternatives": len(one.get("top") or []) >= 2}
        out = {"host": host, "passed": all(checks.values()), "checks": checks,
               "statuses": [a.get("status") for a in answers] + [one.get("status")],
               "errors": [a.get("error") for a in (*answers, one) if a.get("error")],
               "hosts_seen": sorted({str(a.get("host")) for a in (*answers, one)}),
               "prompt_tokens": [a.get("prompt_tokens") for a in answers],
               "completion_tokens": [a.get("completion_tokens") for a in answers],
               "cached_tokens": [a.get("cached_tokens") for a in answers],
               "cost_usd": [a.get("cost_usd") for a in (*answers, one)],
               "latency": [round(a.get("latency") or 0.0, 2) for a in (*answers, one)],
               "one_token": {"answer": one.get("answer"), "top": (one.get("top") or [])[:3]},
               "first_answer": (answers[0].get("answer") or "")[:400], "at": ft.now()}
        with PROBE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
        rows.append(out)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode: status, hosts or probe.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status, 0 on success.
    Raises: SystemExit on bad arguments or a missing key.
    Side Effects: --status and --probe use the key; --probe calls the provider for invented prompts.
    """
    parser = argparse.ArgumentParser(description="OpenRouter access for the pattern-retrieval stages.")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--status", "--hosts", "--probe"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--limit", type=int, default=3, help="hosts to probe, cheapest first")
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(asyncio.run(status()), indent=2))
        return 0
    hosts = eligible(asyncio.run(listed_hosts()))
    if args.hosts:
        for row in hosts:
            print(json.dumps(row))
        return 0
    for row in asyncio.run(probe(hosts, args.limit)):
        print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
