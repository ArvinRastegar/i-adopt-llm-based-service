"""One-attempt task advancement, durable resume and provider-fair bounded execution."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import psycopg

from iadopt_eval import evaluate_item

from .artifacts import verify_bundle
from .domain import LabError, ProviderResult
from .generation.extractor import extract_json
from .persistence import (
    BudgetError,
    EvidenceConflict,
    PersistenceError,
    ProviderIneligible,
    RateLimitError,
    Repository,
)
from .prompting.renderer import load_prompt_version, render_base_prompt, render_correction
from .providers.base import build_request
from .validation import empty_prediction, load_schema_bytes, validate_prediction


def synthetic_similarity(left: str, right: str) -> float:
    """Compare normalized labels in an explicitly non-scientific offline fixture.

    Args: left/right: Evaluator-normalized labels.
    Returns: 1.0 for equality, 0.0 otherwise; not MiniLM cosine similarity.
    Raises: None.
    Side Effects: None; no embedding loading or network.
    """
    return float(left == right)


class SyntheticAdapter:
    """Three deterministic response cases, using fixture gold only in synthetic mode."""

    def __init__(self, task: dict, population: list[str]) -> None:
        """Bind a labelled fixture to durable task/attempt identities.

        Args: task: Synthetic leased task with gold; population: ordered three-target fixture.
        Returns: None.
        Raises: ValueError if target is not in the fixture population.
        Side Effects: None; no real client/key is allocated.
        """
        self.task = task
        self.case = population.index(task["variable"]["variable_id"]) % 3

    async def send_once(self, body: dict) -> ProviderResult:
        """Return valid, corrected-on-attempt2, or always-invalid offline evidence.

        Args: body: Full provider-neutral request whose evidence is retained unchanged.
        Returns: Synthetic ProviderResult; raw envelope is a normal Chat Completions fixture.
        Raises: None for a correctly planned fixture.
        Side Effects: In-memory fixture evaluation only; zero HTTP requests and zero monetary cost.
        """
        number = self.task["attempt_count"]
        valid = self.case == 0 or (self.case == 1 and number >= 2)
        assistant = json.dumps(self.task["gold"], ensure_ascii=False) if valid else '{"invalid_fixture":true}'
        envelope = {"id": "synthetic-" + self.task["id"] + "-" + str(number),
                    "model": "synthetic-fixture-v1",
                    "choices": [{"message": {"role": "assistant", "content": assistant}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "synthetic": True}
        now = datetime.now(UTC).isoformat()
        return ProviderResult(provider=self.task["provider"], requested_model=body["model"], request=body,
            raw_response=json.dumps(envelope, ensure_ascii=False), assistant_text=assistant, status_code=200,
            outcome="response_received", delivery="response_received", started_at=now, finished_at=now,
            latency_seconds=0.0, response_json=envelope, usage=envelope["usage"],
            returned_model=body["model"], finish_reason="stop", request_id=envelope["id"])


class SlidingWindowGate:
    """Process-local conservative RPM/TPM accounting; database leases guard task ownership."""

    def __init__(self, requests: int, tokens: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Initialize a provider's rolling60-second request/token reservation window.

        Args: requests/tokens: Positive frozen limits; clock: injectable monotonic clock.
        Returns: None.
        Raises: LabError for invalid limits.
        Side Effects: Allocates local counters, never sleeps or accesses a database.
        """
        if requests < 1 or tokens < 1:
            raise LabError("Provider RPM and TPM must be positive")
        self.requests, self.tokens, self.clock = requests, tokens, clock
        self.events: deque[tuple[float, int]] = deque()
        self.next_ready = 0.0

    def reserve(self, tokens: int) -> float:
        """Reserve one request now or return a delay without allocating an attempt.

        Args: tokens: Conservative all-in request token bound.
        Returns: Zero after reservation, otherwise nonnegative seconds until retrying this gate.
        Raises: LabError when one request exceeds the provider's entire minute allowance.
        Side Effects: Updates process-local quota receipts only; caller must release leases before waiting.
        """
        if tokens < 0 or tokens > self.tokens:
            raise LabError("One request exceeds the frozen per-minute token allowance")
        now = self.clock()
        while self.events and self.events[0][0] <= now - 60:
            self.events.popleft()
        if len(self.events) >= self.requests or sum(event[1] for event in self.events) + tokens > self.tokens:
            self.next_ready = self.events[0][0] + 60
            return max(0.0, self.next_ready - now)
        self.events.append((now, tokens))
        self.next_ready = now
        return 0.0


@dataclass
class Services:
    """Explicit runtime dependencies; never serialized or mixed into scientific identity."""

    repository: Repository
    root: Path
    bundle: dict
    similarity: Callable[[str, str], float]
    adapters: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, SlidingWindowGate] = field(default_factory=dict)
    checkpoint: Callable[[str, dict], None] | None = None
    cost_policy: Callable[[dict, dict], dict] | None = None
    token_bound: Callable[[dict, dict], int] | None = None
    settlement_policy: Callable[[dict, dict], dict] | None = None
    # In-flight evidence preservation. A commit registers itself here so the campaign can
    # wait for it even when the task that started it was cancelled; see `_commit_response`.
    preservation: set = field(default_factory=set)


async def _db(services: Services, method: str, *args: Any, **kwargs: Any) -> Any:
    """Call one short synchronous repository transaction outside the async event loop.

    Args: services: Injected repository; method: existing repository method; args/kwargs: its inputs.
    Returns: Repository method result after its transaction commits.
    Raises: Original repository errors; never retries a failed mutation automatically.
    Side Effects: Only the named repository operation; no transaction spans an HTTP await.
    """
    return await asyncio.to_thread(getattr(services.repository, method), *args, **kwargs)


def _checkpoint(services: Services, stage: str, task: dict) -> None:
    """Invoke an optional offline fault-injection hook after a durable boundary.

    Args: services: Injected hook; stage: committed boundary name; task: durable task identity.
    Returns: None.
    Raises: Hook exception intentionally propagates to emulate process interruption.
    Side Effects: Only explicitly injected test behavior; production hook is absent.
    """
    if services.checkpoint:
        services.checkpoint(stage, task)


def _base_prompt(task: dict, services: Services) -> Any:
    """Render one task from verified artifacts without exposing target gold.

    Args: task: Leased run and canonical variable; services: frozen demos/root.
    Returns: Original uncorrected one-user-message prompt.
    Raises: Artifact or renderer errors if the frozen protocol is violated.
    Side Effects: Read-only prompt/schema files; target gold is not passed to the renderer.
    """
    return render_base_prompt(load_prompt_version(task["run"]["prompt_variant"], services.root),
        task["variable"]["definition"], load_schema_bytes(services.root),
        services.bundle["demonstrations"][:task["run"]["shot_count"]],
        target_id=task["variable"]["variable_id"],
        control_suffix=task["run"].get("reasoning_prompt_suffix") or "")


# Roughly two minutes of scheduled waiting. Long enough to ride out a database restart
# or a brief pool exhaustion; short enough that a genuine outage still surfaces.
# Task states from which no further work is possible. `operational_failed` covers a
# non-retryable provider outcome such as truncation at the output ceiling;
# `ambiguous_delivery` covers a dispatch whose fate is unknown and which must never be
# resent. Both are terminal outcomes to be reported, not scored.
_TERMINAL_TASK_STATES = ("complete", "operational_failed", "ambiguous_delivery")
_IDLE_POLLS_BEFORE_STOP = 6
# Outcomes that say the DEPLOYMENT cannot serve requests, and so justify pausing every
# task queued against that provider. Everything absent from this set is a property of
# one task and must fail only that task. The distinction has to be stated positively:
# as a list of exceptions it silently mis-sorted every outcome nobody thought to add,
# and `output_truncated` - an HTTP 200 whose only fault is that this prompt and model
# reached the configured ceiling - paused a provider and stranded 16,167 queued tasks.
_DEPLOYMENT_FAILURES = frozenset({
    "provider_error",            # non-transient HTTP status: credentials, model, route
    "html_response",             # a gateway error page instead of a completion envelope
    "provider_error_envelope",   # HTTP 200 carrying an error object and no answer
    "unparsable_envelope",       # body is not the provider's JSON envelope
    "invalid_envelope",          # envelope present but the completion structure is missing
})

_COMMIT_BACKOFF_SECONDS = (0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 40.0, 45.0)


async def _commit_response(services: Services, attempt: dict, payload: dict) -> Any:
    """Commit exact provider evidence, retrying a transient storage failure.

    A response that has reached this process may already have been paid for and cannot
    be reproduced: re-sending would be a second provider request against a task whose
    three-request budget is fixed. A single failed insert must therefore not discard it.
    The commit is idempotent on attempt identity, so repeating it is safe.

    The loop runs as a task registered on `services.preservation` and is awaited through
    `asyncio.shield`, because the outage that fails a commit usually also fails the lease
    heartbeat, and a failing heartbeat cancels this task's supervisor. Shielding only the
    individual database await left the surrounding retry loop cancellable, so the process
    could stop trying to store an answer it had already been charged for while waiting out
    a recoverable outage. Cancelling the caller now stops further dispatch and leaves
    preservation running; `run_campaign` waits for it before returning.

    Args: services: injected repository; attempt: owning attempt row; payload: exact evidence.
    Returns: The stored response record.
    Raises: The final PersistenceError if every attempt to store the evidence fails.
    Side effects: Retries the same durable write with linear backoff; never re-dispatches.
    """

    async def preserve() -> Any:
        last: Exception | None = None
        for delay in _COMMIT_BACKOFF_SECONDS:
            if delay:
                await asyncio.sleep(delay)
            try:
                return await _db(services, "store_response", attempt, payload)
            except EvidenceConflict:
                # Two different payloads claiming one immutable identity. Waiting cannot
                # resolve that, and retrying would only delay a real integrity failure.
                raise
            except (PersistenceError, psycopg.Error) as error:
                # psycopg.Error covers connection loss and pool timeouts, which the
                # repository does not wrap: catching only PersistenceError left exactly the
                # transient outage this loop exists for able to bypass it.
                last = error
        raise last if last else PersistenceError("Response evidence was not stored")

    task = asyncio.ensure_future(preserve())
    services.preservation.add(task)
    task.add_done_callback(services.preservation.discard)
    return await asyncio.shield(task)


async def _heartbeat(lease: dict, services: Services, stopped: asyncio.Event) -> None:
    """Renew one active lease periodically while a single task operation runs.

    Args: lease: Current fenced ownership; services: repository/config; stopped: completion signal.
    Returns: None when stopped.
    Raises: Lost-lease or database error, cancelling safe continuation at the wrapper boundary.
    Side Effects: Short operational database writes only; no model requests or waiting transaction.
    """
    config = services.bundle["configuration"]["execution"]
    while not stopped.is_set():
        try:
            await asyncio.wait_for(stopped.wait(), timeout=config["heartbeat_seconds"])
        except TimeoutError:
            await _db(services, "heartbeat", lease, config["task_lease_seconds"])


async def run_task(lease: dict, services: Services) -> dict:
    """Advance a fenced task through local checkpoints and at most ONE provider request.

    Args: lease: Claimed task record; services: explicit frozen/runtime boundaries.
    Returns: Current durable task after release, preserving valid/invalid/operational distinctions.
    Raises: Integrity/storage errors or injected interruption; no automatic fourth request exists.
    Side Effects: Prompt/request, raw response, validation, prediction and score transactions;
        zero or one authorized send. SDK retries are independently disabled.
    """
    stopped = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat(lease, services, stopped))
    work = asyncio.create_task(_advance_task(lease, services))
    try:
        done, _ = await asyncio.wait({work, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
        if heartbeat in done and heartbeat.exception() is not None:
            work.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work
            raise heartbeat.exception()
        return await work
    finally:
        stopped.set()
        if not work.done():
            work.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat


async def _advance_task(lease: dict, services: Services) -> dict:
    """Execute one state-machine advancement under a wrapper-managed heartbeat.

    Args: lease: Current token/fence and task evidence; services: injected dependencies.
    Returns: Released next checkpoint. Invalid content is data, never an exception or silent repair.
    Raises: LabError/integrity errors; cancellation records ambiguous dispatch when necessary.
    Side Effects: Short repository writes and at most one send; raw commits before validation.
    """
    task = await _db(services, "get_task", lease["id"])
    if task["prediction"] is not None:
        return await _score(task, lease, services)
    attempts = task["attempts"]
    latest = attempts[-1] if attempts else None
    if latest and latest["response"] and latest["response"]["delivery"] == "response_received" and task["state"] != "retry_pending":
        return await _validate_and_select(task, lease, services)
    if task["state"] not in {"queued", "retry_pending", "request_persisted"}:
        return await _db(services, "release", lease)
    if task["state"] == "request_persisted":
        attempt, body = latest, latest["request_body"]
    else:
        prompt = _base_prompt(task, services)
        if latest and latest["validation"] and latest["validation"]["content_invalid"]:
            prompt = render_correction(prompt, latest["response"]["assistant_text"] or "",
                                       latest["validation"]["errors"], latest["attempt_number"])
        body = build_request(task["run"], list(prompt.messages))
        token_count = (services.token_bound(task, body) if services.token_bound else
                       len(prompt.content.encode("utf-8")) + task["run"]["max_output_tokens"])
        profiles = services.bundle["configuration"]["providers"][task["provider"]]["models"]
        model = next(model for model in profiles if model["id"] == task["model_id"])
        if token_count > model["context_window_tokens"]:
            return await _db(services, "release", lease, "paused_configuration",
                             {"code": "context_bound_exceeded", "tokens": token_count})
        gate = services.gates.get(task["provider"])
        try:
            if gate and gate.reserve(token_count) > 0:
                return await _db(services, "release", lease)
        except LabError:
            return await _db(services, "release", lease, "paused_configuration", {"code": "request_exceeds_tpm"})
        request = {"messages": list(prompt.messages), "prompt": prompt.content, "body": body,
                   "prompt_evidence": prompt.to_dict(), "scientific_parameters": task["run"]["configuration"],
                   "attempt_number": task["attempt_count"] + 1,
                   "correction_parent": latest["id"] if latest and latest["validation"] and latest["validation"]["content_invalid"] else None,
                   "token_bound": token_count,
                   "cost": services.cost_policy(task, body) if services.cost_policy else {}}
        try:
            attempt = await _db(services, "start_attempt", lease, request)
        except BudgetError:
            return await _db(services, "release", lease, "paused_budget", {"code": "optional_cap_or_cost_evidence"})
        except RateLimitError as error:
            # Admission is checked before any attempt row is allocated, so no request is
            # consumed. Hold the provider's gate for the advised wait and return the task
            # to the queue. Letting this propagate would reach the pool's failure path and
            # cancel unrelated in-flight work over ordinary, expected throttling.
            # The advised delay is an attribute; args[0] is the human-readable message.
            wait = float(getattr(error, "retry_after_seconds", 1.0))
            gate = services.gates.get(task["provider"])
            if gate is not None:
                gate.next_ready = max(gate.next_ready, time.monotonic() + wait)
            # `retry_pending` asserts a previous attempt exists and is being retried, and
            # the repository refuses that state without the attempt evidence to support it.
            # Admission runs BEFORE any attempt row is allocated, so a first-attempt
            # throttle has none: a task that has consumed nothing belongs back in the queue
            # it came from. Releasing it as a retry raised AttemptLimitError out of the
            # worker, through the pool, into the campaign's cancel-everything path - so
            # ordinary, expected throttling failed the campaign and left unrelated
            # in-flight requests recorded as ambiguous deliveries.
            resting = "retry_pending" if task["attempt_count"] else "queued"
            return await _db(services, "release", lease, resting, {"code": "rate_limited"})
        except ProviderIneligible:
            # A provider cooldown set by ONE worker while others are already past the
            # eligibility check. That is normal at any concurrency above one: a single
            # transient blip anywhere puts the provider into a few seconds of backoff, and
            # every worker mid-dispatch then finds it ineligible. Letting that propagate
            # meant one blip in 4,745 calls reached the pool's failure path and cancelled
            # every other in-flight request, ending the campaign. No attempt row was
            # allocated, so nothing is consumed; the task waits with the provider.
            resting = "retry_pending" if task["attempt_count"] else "queued"
            return await _db(services, "release", lease, resting, {"code": "provider_cooldown"})
        _checkpoint(services, "request_persisted", task)
    dispatched = await _db(services, "mark_dispatched", attempt, lease)
    if not dispatched["dispatch_allowed"]:
        return await _db(services, "release", lease, "ambiguous_delivery", {"code": "already_dispatched"})
    _checkpoint(services, "dispatch_started", task)
    current = await _db(services, "get_task", lease["id"])
    adapter = SyntheticAdapter(current, services.bundle["plan"]["population"]) if services.bundle["plan"]["mode"] == "synthetic" else services.adapters[task["provider"]]
    try:
        response = await adapter.send_once(body)
    except asyncio.CancelledError:
        # Cancellation after the write-ahead marker may have reached the provider.
        await asyncio.shield(_db(services, "store_response", attempt,
            {"raw_body": None, "delivery": "ambiguous_delivery", "outcome": "interrupted_after_dispatch"}))
        await asyncio.shield(_db(services, "release", lease, "ambiguous_delivery"))
        raise
    payload = response.to_dict()
    if services.settlement_policy is not None:
        # Accounting is attached before the commit because the repository writes the
        # response row and its settlement in one transaction, and the response row is
        # immutable afterwards. Interpreting an unfamiliar billing field must therefore
        # never be able to prevent the raw evidence being stored: a failure here is
        # recorded as unresolved accounting, which is recoverable, rather than losing a
        # response that has already been paid for, which is not.
        try:
            payload["cost"] = services.settlement_policy(task, payload)
        except Exception as error:  # noqa: BLE001 - accounting must not block evidence
            payload["cost"] = {"state": "unavailable",
                               "basis": "settlement policy failed: " + type(error).__name__,
                               "error": str(error)[:500]}
    await _commit_response(services, attempt, payload)
    _checkpoint(services, "response_stored", task)
    current = await _db(services, "get_task", lease["id"])
    if response.delivery == "response_received":
        return await _validate_and_select(current, lease, services)
    if response.outcome == "classified_transient_provider_error" and response.delivery in {"not_dispatched", "rejected"} and current["attempt_count"] < 3:
        seconds = response.retry_after_seconds if response.retry_after_seconds is not None else 2 ** current["attempt_count"]
        await _db(services, "set_provider_state", task["campaign_id"], task["provider"], "cooldown",
                  {"code": response.outcome}, datetime.now(UTC) + timedelta(seconds=seconds))
        return await _db(services, "release", lease, "retry_pending", {"code": response.outcome})
    # Pausing the provider stops the whole campaign, so it is reserved for failures that
    # say something is wrong with the deployment - see `_DEPLOYMENT_FAILURES`. A task that
    # merely exhausted its own attempts, or whose answer overran the output ceiling, is
    # that task's failure and must not hold back the rest of the population.
    if (response.delivery != "ambiguous_delivery"
            and response.outcome in _DEPLOYMENT_FAILURES):
        await _db(services, "set_provider_state", task["campaign_id"], task["provider"], "paused",
                  {"code": response.outcome, "http_status": response.status_code})
    return await _db(services, "release", lease,
                      "ambiguous_delivery" if response.delivery == "ambiguous_delivery" else "operational_failed",
                      {"code": response.outcome})


async def _validate_and_select(task: dict, lease: dict, services: Services) -> dict:
    """Continue extraction/validation from the committed raw response without re-sending.

    Args: task: Task with delivered response; lease: active ownership; services: schema/store.
    Returns: Released retry/failed/complete task depending on all three-attempt evidence.
    Raises: Integrity errors; ordinary JSON/schema errors are retained structured outcomes.
    Side Effects: Local validation/prediction/evaluation writes only, never a provider request.
    """
    latest = task["attempts"][-1]
    if not latest["validation"]:
        extraction = extract_json(latest["response"]["assistant_text"] or "")
        validity = validate_prediction(extraction.candidate) if extraction.success else None
        evidence = {"valid": bool(validity and validity.valid), "content_invalid": not bool(validity and validity.valid),
            "candidate": extraction.candidate, "extraction": extraction.to_dict(),
            "validation": validity.to_dict() if validity else None,
            "errors": list(validity.errors if validity else extraction.errors)}
        await _db(services, "record_validation", latest, evidence)
        _checkpoint(services, "validated", task)
        task = await _db(services, "get_task", task["id"])
        latest = task["attempts"][-1]
    evidence = latest["validation"]["evidence"]
    if evidence["valid"]:
        prediction = evidence["validation"]["canonical_prediction"]
        await _db(services, "select_prediction", lease, prediction, latest)
    elif len(task["attempts"]) == 3 and all(attempt["validation"] and attempt["validation"]["content_invalid"]
                                         for attempt in task["attempts"]):
        await _db(services, "select_prediction", lease, empty_prediction(), latest, terminal_invalid=True)
    elif task["attempt_count"] < 3:
        return await _db(services, "release", lease, "retry_pending", {"code": "content_invalid"})
    else:
        return await _db(services, "release", lease, "operational_failed", {"code": "attempt_budget_exhausted_with_operational_error"})
    _checkpoint(services, "prediction_ready", task)
    return await _score(await _db(services, "get_task", task["id"]), lease, services)


async def _score(task: dict, lease: dict, services: Services) -> dict:
    """Score one stored canonical prediction once and release its completed checkpoint.

    Args: task: Prediction-ready task; lease: current token/fence; services: identified similarity/store.
    Returns: Complete released task; existing score is reused unchanged.
    Raises: Integrity errors only; a scorer rejection fails this task alone.
    Side Effects: One immutable evaluation and normalized metric set; no provider calls.
    """
    if not task["evaluation"]:
        try:
            result = await asyncio.to_thread(evaluate_item, task["gold"], task["prediction"]["canonical"],
                services.similarity, metadata={"variable_id": task["variable"]["variable_id"],
                    "run_id": task["run"]["id"], "mode": services.bundle["plan"]["mode"],
                    "category": task["variable"]["category"], "subcategory": task["variable"]["subcategory"]},
                similarity_identity=services.bundle["similarity_identity"])
        except (ValueError, TypeError, KeyError, ArithmeticError) as error:
            # Data-shaped failures only. Scoring is a pure computation over one gold and
            # one prediction, so these say something about that pair; an infrastructure
            # fault still propagates and stops the campaign, which is what it should do.
            # The validator is supposed to reject anything the scorer will not accept, so
            # reaching here means the two disagree about this prediction. That is a defect
            # worth fixing, but it is still one task's prediction: letting it propagate
            # aborted a live campaign after 995 tasks, and because the offending response
            # was already durable, every resume replayed it into the same exception and
            # the run could never advance again. Fail the task, keep the campaign, and
            # record the disagreement so it can be found and fixed.
            return await _db(services, "release", lease, "operational_failed",
                             {"code": "scorer_rejected_validated_prediction",
                              "detail": str(error)[:500]})
        await _db(services, "store_evaluation", lease, result)
        _checkpoint(services, "scored", task)
    return await _db(services, "release", lease, "complete")


def build_observations(tasks: list[dict]) -> list[dict[str, Any]]:
    """Project stored task evidence into the rows the reporting layer consumes.

    Args: tasks: full task records from the repository, in any order.
    Returns: One observation per task carrying run/variable identity, category
        provenance, terminal validity and its evaluation record.
    Raises: KeyError if a task record is incomplete.
    Side effects: None; this is a pure projection of already-durable evidence.
    """
    rows = []
    for task in tasks:
        variable = task["variable"]
        rows.append({
            # Reporting joins against the frozen plan. The database keeps its own
            # campaign-scoped run fingerprint and its own UUID, neither of which the plan
            # knows; the original plan run is retained verbatim as the run evidence, so
            # that is the identity to project.
            "run_id": task["run"]["id"], "variable_id": variable["variable_id"],
            "category": variable["category"], "subcategory": variable["subcategory"],
            "category_path": variable["category_path"],
            "terminal_invalid": bool((task["prediction"] or {}).get("terminal_invalid", False)),
            "attempts": [_attempt_projection(attempt) for attempt in task["attempts"]],
            "evaluation": task["evaluation"],
        })
    return rows


def _attempt_projection(attempt: dict) -> dict[str, Any]:
    """Reduce one stored attempt to the JSON-safe fields the reporting layer reads.

    Database rows carry datetimes, Decimals and raw bytes, none of which survive the
    canonical-JSON boundary that reporting validates against. Passing the row through
    whole fails; this selects the outcome, usage, latency and validation errors that
    the attempt summary actually consumes, and nothing else.

    Args: attempt: stored attempt record including its response and validation rows.
    Returns: JSON-compatible attempt projection.
    Raises: None; absent evidence is represented as absent, never invented.
    Side effects: None.
    """
    response = attempt.get("response") or {}
    evidence = response.get("evidence") if isinstance(response.get("evidence"), dict) else {}
    validation = attempt.get("validation") or {}
    latency = response.get("latency_seconds")
    return {
        "attempt_number": attempt.get("attempt_number"),
        "response": {
            "outcome": evidence.get("outcome", response.get("delivery", "unavailable")),
            "usage": response.get("usage") if isinstance(response.get("usage"), dict) else None,
            "latency_seconds": float(latency) if latency is not None else None,
        },
        "validation": {"errors": validation.get("errors") or []},
    }


async def finalize_campaign(campaign_id: str, services: Services) -> dict[str, Any]:
    """Rank the completed campaign, persist its evidence and mark it complete.

    Finishing every task is not finishing the campaign: without this step the database
    keeps a campaign whose tasks are all complete in the `running` state, with no stored
    ranking and no report lineage, while the caller sees a successful exit.

    Each step is idempotent, so an interruption during finalization is recovered by
    running it again: repeating an identical ranking or report returns the existing row,
    and completing an already complete campaign is safe.

    Args: campaign_id: frozen campaign; services: verified bundle and repository;
    Returns: Ranking summary with the stored identities and final campaign state.
    Raises: LabError if tasks are incomplete; persistence errors on evidence conflict.
    Side effects: Writes ranking, report and campaign state; optionally one export file.
    """
    from .reporting import build_configuration_ranking, export_report

    plan = services.bundle["plan"]
    tasks = await asyncio.to_thread(services.repository.list_tasks, campaign_id)
    unfinished = [task["task_id"] for task in tasks
                  if task["state"] not in _TERMINAL_TASK_STATES]
    if unfinished:
        raise LabError(f"Cannot finalize: {len(unfinished)} task(s) can still advance")
    # Terminal failures are reported, not silently dropped and not scored. A configuration
    # missing any of its population fails `require_complete_population` and is ranked
    # nowhere; the ranking already records that as an explicit not-rankable reason.
    failed = [task["task_id"] for task in tasks if task["state"] != "complete"]

    # Take the scorer version and threshold from the frozen evaluator constants, not from
    # an evaluated task: reading them out of the evidence being checked would make the
    # check agree with whatever produced that evidence.
    from iadopt_eval.core import CLOSE_THRESHOLD, SCORER_VERSION

    # The scorer hash is recomputed from this checkout's evaluator source and the loaded
    # backend rather than copied out of the plan. Copying it compared the plan to itself,
    # which holds for any checkout and so could not detect a report produced by different
    # evaluator source or a different similarity backend from the one the plan froze.
    from .artifacts import scorer_identity_hash

    scorer_identity = {
        "scorer_version": SCORER_VERSION,
        "similarity_identity": services.bundle["similarity_identity"],
        "close_threshold": CLOSE_THRESHOLD,
        "plan_scorer": scorer_identity_hash(services.root, services.bundle["similarity_identity"]),
    }
    categories = {task["variable"]["variable_id"]: {
        "category": task["variable"]["category"],
        "category_path": task["variable"]["category_path"]} for task in tasks}

    report = build_configuration_ranking(plan, build_observations(tasks),
                                         scorer_identity=scorer_identity,
                                         population_categories=categories)
    ranking = {"policy_version": report["policy_version"], "configurations": [
        {"configuration_id": row["configuration_id"], "provider": row["provider"],
         "model_id": row["model_id"], "rank": row["rank"],
         "primary": {"numerator": row["primary"]["numerator"],
                     "denominator": row["primary"]["denominator"]} if row["primary"] else None,
         "reason": row["reason"]} for row in report["configurations"]]}
    stored = await asyncio.to_thread(services.repository.save_ranking, campaign_id, ranking)

    # Re-exporting identical content is a no-op; only genuinely different bytes at the
    # same destination raise, which keeps finalization safe to repeat. Exporting is not
    # optional: `save_report` requires a non-empty file list, so the former `outputs=False`
    # branch could only ever produce a manifest the repository rejects. No caller used it,
    # and a switch that cannot succeed is worse than no switch.
    written = await asyncio.to_thread(
        export_report, report, "json", f"ranking-{campaign_id}.json",
        outputs_root=services.root / "outputs")
    files = list(written["files"])
    manifest = {"schema_version": "1.0", "campaign_id": campaign_id,
                "kind": "configuration-ranking", "files": files, "final": report["complete"]}
    await asyncio.to_thread(services.repository.save_report, campaign_id, manifest)

    state = "paused"
    if report["complete"]:
        state = (await asyncio.to_thread(services.repository.complete_campaign, campaign_id))["state"]
    return {"campaign_id": campaign_id, "state": state, "ranking_id": stored["id"],
            "policy_version": report["policy_version"],
            "configurations": report["configuration_count"],
            "rankable": report["rankable_count"], "complete": report["complete"],
            "operationally_failed_tasks": len(failed)}


async def run_campaign(campaign_id: str, services: Services, *, stop_after: int | None = None) -> dict:
    """Continue all selected providers through durable completion or an explicit safe stop.

    Args: campaign_id: Explicit frozen campaign; services: verified bundle/adapters/store;
        stop_after: Optional offline test limit on task advancements, never scientific parameters.
    Returns: State counts, stop reason and cumulative call count. Reporting is a separate finalizer.
    Raises: Integrity/runtime drift or shared storage failure. Live work requires stored authorization.
    Side Effects: Provider-fair bounded task work, short DB transactions and interruptible cooldown waits.
    """
    verify_bundle(services.root, services.bundle)
    mode = services.bundle["plan"]["mode"]
    if mode == "live" and (services.token_bound is None or services.cost_policy is None
                           or services.settlement_policy is None):
        raise LabError("Live execution needs token-bound, price and settlement services")
    if mode == "live" and stop_after is not None:
        raise LabError("A live canary needs its own separately estimated and authorized campaign")
    if stop_after is not None and stop_after < 1:
        raise LabError("Stop-after must be positive")
    campaign = await _db(services, "get_campaign", campaign_id)
    if mode != campaign["mode"] or campaign["configuration"].get("lab_plan_sha256") != services.bundle["plan"]["sha256"]:
        raise LabError("Campaign/bundle scientific identity mismatch")
    if mode == "live" and (not campaign["authorizations"] or not services.bundle["configuration"]["campaign"]["live_calls_enabled"]):
        raise LabError("Live calls require enabled configuration and separate plan-bound authorization")
    if campaign["state"] == "complete":
        return {"campaign_id": campaign_id, "state": "complete", "states": campaign["states"], "stop_reason": "already_complete"}
    await _db(services, "reconcile", campaign_id)
    await _db(services, "set_campaign_state", campaign_id, "running")
    config = services.bundle["configuration"]
    selected = config["campaign"]["providers"]
    worker_count = config["execution"]["worker_count"]
    services.gates = services.gates or {name: SlidingWindowGate(config["providers"][name]["requests_per_minute"],
        config["providers"][name]["tokens_per_minute"]) for name in selected}
    worker, advancements, rotation = "worker-" + str(uuid.uuid4()), 0, 0
    stop_reason = "blocked"
    # Consecutive idle polls tolerated before concluding nothing can advance. Each is
    # followed by a short backoff, so this rides out a cooldown expiring mid-check without
    # spinning if the campaign really is stuck.
    idle_polls = 0
    # Streaming worker pool: a finished task frees its slot immediately instead of the
    # whole batch waiting on its slowest member. Provider fairness, per-provider
    # concurrency and the rate gates are unchanged; only the wait boundary moves.
    inflight: dict[asyncio.Task, str] = {}
    counts = {name: 0 for name in selected}

    async def reap(*, drain: bool) -> int:
        """Wait for in-flight work and release the slots it held.

        Args: drain: wait for every in-flight task instead of the first to finish.
        Returns: number of task advancements completed.
        Raises: the original exception of any failed task. asyncio.wait would
            otherwise discard it, turning a real failure into a silent stall.
        Side effects: releases per-provider concurrency slots for finished tasks.
        """
        if not inflight:
            return 0
        mode = asyncio.ALL_COMPLETED if drain else asyncio.FIRST_COMPLETED
        done, _ = await asyncio.wait(set(inflight), return_when=mode)
        for finished in done:
            counts[inflight.pop(finished)] -= 1
        for finished in done:
            finished.result()
        return len(done)

    try:
        while True:
            while len(inflight) < worker_count:
                if stop_after is not None and advancements + len(inflight) >= stop_after:
                    break
                claimed = None
                for offset in range(len(selected)):
                    name = selected[(rotation + offset) % len(selected)]
                    if counts[name] >= config["providers"][name]["max_concurrency"] or services.gates[name].next_ready > time.monotonic():
                        continue
                    claims = await _db(services, "claim_tasks", campaign_id, worker, 1,
                                       config["execution"]["task_lease_seconds"], name)
                    if claims:
                        claimed = (claims[0], name)
                        rotation = (selected.index(name) + 1) % len(selected)
                        break
                if claimed is None:
                    break
                lease, name = claimed
                idle_polls = 0
                inflight[asyncio.create_task(run_task(lease, services))] = name
                counts[name] += 1
            if inflight:
                advancements += await reap(drain=False)
                if stop_after is not None and advancements >= stop_after:
                    advancements += await reap(drain=True)
                    stop_reason = "requested_offline_checkpoint"
                    break
                continue
            campaign = await _db(services, "get_campaign", campaign_id)
            states = campaign["states"]
            total = services.bundle["plan"]["counts"]["tasks"]
            if states.get("complete", 0) == total:
                stop_reason = "tasks_complete"
                break
            # Some failures are terminal by design: a response truncated at the output
            # ceiling is an operational failure that must NOT be retried (the same request
            # would truncate again) and must NOT be scored as a model-quality zero. Such a
            # task can never reach `complete`, so waiting for a fully complete population
            # meant one truncation anywhere denied results for the entire campaign. When
            # nothing remains that could still advance, that is a finished campaign with
            # recorded failures, not a stalled one.
            if sum(states.get(name, 0) for name in _TERMINAL_TASK_STATES) == total:
                stop_reason = "tasks_terminal"
                break
            # A paused provider hands out no queued task and contributes no due time, so
            # it produced neither work nor a delay and the loop fell through to six idle
            # polls and the opaque `no_claimable_work`. When nothing is left that could
            # serve a request, say so: the reason names the condition, and reconciliation
            # on the next resume clears the pause, so a supervisor recovers by itself.
            if all(profile["state"] in {"paused", "failed"} for profile in campaign["providers"]):
                stop_reason = "providers_paused"
                break
            delays = [gate.next_ready - time.monotonic() for gate in services.gates.values() if gate.next_ready > time.monotonic()]
            for profile in campaign["providers"]:
                if profile["state"] == "cooldown" and profile["cooldown_until"]:
                    # psycopg returns timestamptz as a datetime and the row helper keeps it
                    # that way; only serialized evidence arrives as a string. Accept both.
                    until = profile["cooldown_until"]
                    if isinstance(until, str):
                        until = datetime.fromisoformat(until)
                    remaining = (until - datetime.now(UTC)).total_seconds()
                    if remaining > 0:
                        delays.append(remaining)
            if delays:
                idle_polls = 0
                await asyncio.sleep(min(30.0, max(0.01, min(delays))))
                continue
            # Nothing was claimable and no wait is known, yet work remains. Usually that is
            # a race rather than a dead end: a cooldown that expired between the claim
            # attempt and this check leaves no delay to report while the tasks it was
            # holding are now claimable again. Breaking immediately abandoned 9,929 queued
            # tasks after one upstream rate-limit. Back off briefly and look again, and
            # give up only after several consecutive polls find nothing at all.
            idle_polls += 1
            if idle_polls >= _IDLE_POLLS_BEFORE_STOP:
                stop_reason = "no_claimable_work"
                break
            await asyncio.sleep(min(5.0, 0.5 * idle_polls))
    finally:
        # A failure must not leave leases held by orphaned coroutines.
        for pending in inflight:
            pending.cancel()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        # Cancelling a worker stops it dispatching; it must not stop it storing an answer
        # already received. Those commits are shielded, so they outlive their task and are
        # awaited here instead - otherwise the loop closes on a write still in flight and
        # paid-for evidence is lost at exactly the moment the campaign is failing.
        if services.preservation:
            await asyncio.gather(*tuple(services.preservation), return_exceptions=True)
    # A campaign that stopped with terminal failures is finished, not paused: pausing it
    # would invite a resume that can never make progress.
    if stop_reason not in {"tasks_complete", "tasks_terminal"}:
        await _db(services, "set_campaign_state", campaign_id, "paused")
    campaign = await _db(services, "get_campaign", campaign_id)
    return {"campaign_id": campaign_id, "state": campaign["state"], "states": campaign["states"],
            "stop_reason": stop_reason, "advancements": advancements}
