"""Regressions for the execution/reporting boundaries fixed after the 2026-09-09 review.

Each test pins one behaviour that previously failed silently: a broken transport being
scored as a model answer, a report accepting the wrong scorer, a category claiming
completeness from the rows it was handed, and a saved manifest escaping verification.
"""

import asyncio
import contextlib
import json
import shutil
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import yaml
from test_provider_transport import _exchange

from iadopt_lab.corpus.ingestion import load_canonical_records
from iadopt_lab.domain import LabError
from iadopt_lab.reporting import build_category_summary, build_configuration_ranking
from iadopt_lab.workflow import _attempt_projection, build_observations

ROOT = Path(__file__).resolve().parents[2]


def _envelope(content, finish="stop"):
    """Build a minimal well-formed chat-completions envelope for a fixture response."""
    return {"id": "fixture", "model": "m", "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}]}


@pytest.mark.parametrize("body,status,delivery,outcome", [
    (b"<html><body>gateway error</body></html>", 200, "rejected", "html_response"),
    (b"not json at all", 200, "rejected", "unparsable_envelope"),
    (b'["a list, not an envelope"]', 200, "rejected", "unparsable_envelope"),
])
def test_broken_transport_is_operational_not_model_content(body, status, delivery, outcome):
    """An HTTP-200 body that is not a provider envelope must never reach content scoring.

    Routing these into validation let three gateway error pages exhaust a task's three
    attempts and be recorded as an explicit empty prediction, lowering a model's F1 on
    infrastructure noise.
    """
    result, _ = _exchange("openrouter", lambda request: httpx.Response(status, content=body))
    assert (result.delivery, result.outcome) == (delivery, outcome)


def test_truncated_generation_is_a_capacity_failure():
    """An answer that never started because the budget ran out is not a bad answer."""
    envelope = _envelope("", finish="length")
    result, _ = _exchange("psnc", lambda request: httpx.Response(200, json=envelope))
    assert (result.delivery, result.outcome) == ("rejected", "output_truncated")


def test_genuinely_empty_completion_remains_a_model_outcome():
    """A well-formed envelope whose completion is empty is still the model's answer."""
    result, _ = _exchange("psnc", lambda request: httpx.Response(200, json=_envelope("")))
    assert (result.delivery, result.outcome) == ("response_received", "empty_response")


def test_attempt_projection_is_json_safe():
    """Database rows carry datetimes, Decimals and bytes; reporting requires plain JSON."""
    attempt = {"attempt_number": 1, "created_at": datetime.now(UTC),
               "response": {"raw_body": b"\x00\x01", "latency_seconds": Decimal("1.25"),
                            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                            "delivery": "response_received", "created_at": datetime.now(UTC),
                            "evidence": {"outcome": "response_received"}},
               "validation": {"errors": [{"code": "schema"}], "created_at": datetime.now(UTC)}}
    projected = _attempt_projection(attempt)
    json.dumps(projected)  # must not raise
    assert projected["response"]["outcome"] == "response_received"
    assert projected["response"]["latency_seconds"] == 1.25
    assert projected["validation"]["errors"] == [{"code": "schema"}]


def test_build_observations_uses_the_plan_run_identity():
    """Reporting joins on the plan's run id, not the database fingerprint or UUID."""
    task = {"run_id": "db-uuid", "run": {"id": "plan-run-id"},
            "run_record": {"fingerprint": "campaign-scoped-hash"},
            "variable": {"variable_id": "v1", "category": "C", "subcategory": "S",
                         "category_path": "C/S"},
            "prediction": {"terminal_invalid": False}, "attempts": [], "evaluation": None}
    assert build_observations([task])[0]["run_id"] == "plan-run-id"


def _observation(variable_id, category="C", subcategory="S"):
    """Build one minimal scored observation for reporting-boundary tests."""
    return {"run_id": "r1", "variable_id": variable_id, "category": category,
            "subcategory": subcategory, "category_path": f"{category}/{subcategory}",
            "terminal_invalid": False, "attempts": [], "evaluation": None}


def test_category_completeness_needs_frozen_membership():
    """A category missing members must not report complete just because rows agree."""
    rows = [_observation("v1"), _observation("v2")]
    without = build_category_summary(rows)
    assert all(row["coverage_basis"] == "observed-rows-only" for row in without)

    frozen = {f"v{n}": {"category": "C", "category_path": "C/S"} for n in (1, 2, 3)}
    with_population = build_category_summary(rows, frozen)
    assert all(row["coverage_basis"] == "frozen-population" for row in with_population)
    # v3 is in the frozen membership but was never supplied, so it must show as missing
    # and the category must not claim completeness.
    assert all("v3" in row["missing_variable_ids"] for row in with_population)
    assert not any(row["complete"] for row in with_population)


def test_ranking_rejects_observations_scored_by_another_backend():
    """A report must belong to the scorer its plan froze, not merely be self-consistent.

    `_observations` only proves the evaluations agree with one another, so a campaign
    scored end to end with a synthetic backend previously satisfied every check and
    could be published as a scientific result.
    """
    from iadopt_lab.artifacts import collect_input_artifacts
    from iadopt_lab.configuration import load_parameters, resolve_configuration
    from iadopt_lab.planning import expand_campaign, synthetic_configuration

    resolved = synthetic_configuration(resolve_configuration(load_parameters(ROOT / "parameters.yml")))
    records = list(load_canonical_records(ROOT))
    targets = [row for row in records if not row["demonstration_position"]]
    identity = {"backend": "synthetic-equality-only-v1", "synthetic": True}
    inputs = collect_input_artifacts(ROOT, records, targets, identity)
    plan = expand_campaign(resolved, targets[:3], artifact_identities=inputs["identities"],
                           mode="synthetic")

    with pytest.raises(ValueError, match="does not belong to this plan"):
        build_configuration_ranking(
            plan, [], scorer_identity={"scorer_version": "january-derived-member-credit-v1",
                                       "similarity_identity": identity, "close_threshold": 0.8,
                                       "plan_scorer": "a-hash-from-another-plan"})
    # The same call with the plan's own scorer identity is accepted.
    report = build_configuration_ranking(
        plan, [], scorer_identity={"scorer_version": "january-derived-member-credit-v1",
                                   "similarity_identity": identity, "close_threshold": 0.8,
                                   "plan_scorer": plan["artifact_identities"]["scorer"]})
    assert report["scorer_binding"] == "verified-against-plan"


def _isolated_lab(tmp_path):
    """Build a lab root that shares the real corpus but owns its manifests.

    The corpus is large, so everything is symlinked except `data/manifests`, which is
    copied and therefore writable. Tampering used to be done to the repository's own
    manifest and undone in a `finally`; a killed process or a parallel run left the real
    file short of a member, which is a corrupted checkout rather than a failed test.
    """
    root = tmp_path / "lab"
    root.mkdir()
    for entry in ROOT.iterdir():
        if entry.name != "data":
            (root / entry.name).symlink_to(entry)
    (root / "data").mkdir()
    for entry in (ROOT / "data").iterdir():
        if entry.name != "manifests":
            (root / "data" / entry.name).symlink_to(entry)
    shutil.copytree(ROOT / "data/manifests", root / "data/manifests")
    return root


def test_verify_detects_a_tampered_population_manifest(tmp_path):
    """Deleting or editing a saved manifest must fail verification, not pass silently."""
    root = _isolated_lab(tmp_path)
    manifest = root / "data/manifests/evaluation-population-v2.0.1.yml"
    assert len(load_canonical_records(root)) == 102

    tampered = yaml.safe_load(manifest.read_bytes())
    tampered["members"] = tampered["members"][:-1]
    manifest.write_text(yaml.safe_dump(tampered, sort_keys=True, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation-population"):
        load_canonical_records(root)
    # The repository's own manifest was never touched, so no restore step can be skipped.
    assert len(load_canonical_records(ROOT)) == 102


# --- D-039 regressions: evidence preservation and classification -----------------

@pytest.mark.parametrize("envelope,expected_delivery,expected_outcome", [
    ({}, "rejected", "invalid_envelope"),
    ({"error": {"message": "gateway failure"}}, "rejected", "provider_error_envelope"),
    ({"id": "x", "choices": [{"finish_reason": "stop"}]}, "rejected", "invalid_envelope"),
    ({"id": "x", "choices": [{"finish_reason": "stop", "message": "oops"}]}, "rejected", "invalid_envelope"),
    ({"id": "x", "choices": [{"finish_reason": "length",
                              "message": {"content": '{"hasProp'}}]}, "rejected", "output_truncated"),
    ({"id": "x", "choices": [{"finish_reason": "length",
                              "message": {"content": ""}}]}, "rejected", "output_truncated"),
    ({"id": "x", "choices": [{"finish_reason": "stop",
                              "message": {"content": ""}}]}, "response_received", "empty_response"),
    ({"id": "x", "choices": [{"finish_reason": "stop",
                              "message": {"content": '{"ok":true}'}}]}, "response_received", "response_received"),
])
def test_only_a_well_formed_completion_reaches_content_scoring(envelope, expected_delivery, expected_outcome):
    """Everything a provider can return at HTTP 200 that is not the model's answer is operational.

    Otherwise three gateway faults or three truncations exhaust a task's attempts and are
    recorded as an empty prediction, lowering the measured score on infrastructure noise.
    """
    result, _ = _exchange("openrouter", lambda request: httpx.Response(200, json=envelope))
    assert (result.delivery, result.outcome) == (expected_delivery, expected_outcome)


class _FakeRepository:
    """Repository stub that fails a fixed number of times before succeeding."""

    def __init__(self, failures):
        self.failures, self.calls = list(failures), 0

    def store_response(self, attempt, payload):
        """Raise the next scripted failure, or record the successful commit."""
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"stored": True, "payload": payload}


def _services(repository):
    """Build the minimal Services object `_commit_response` needs."""
    from iadopt_lab.workflow import Services
    return Services(repository=repository, root=ROOT, bundle={}, similarity=lambda a, b: 0.0)


def test_commit_retries_transient_database_errors(monkeypatch):
    """psycopg connection failures must be retried, not allowed to discard a paid response."""
    import psycopg

    from iadopt_lab import workflow

    monkeypatch.setattr(workflow, "_COMMIT_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    repository = _FakeRepository([psycopg.OperationalError("connection lost")])
    result = asyncio.run(workflow._commit_response(_services(repository), {"id": "a"}, {"raw": 1}))
    assert result["stored"] and repository.calls == 2


def test_commit_does_not_retry_permanent_conflicts(monkeypatch):
    """An immutable-identity conflict cannot be resolved by waiting, so it must not be retried."""
    from iadopt_lab import workflow
    from iadopt_lab.persistence import EvidenceConflict

    monkeypatch.setattr(workflow, "_COMMIT_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    repository = _FakeRepository([EvidenceConflict("two payloads, one identity")])
    with pytest.raises(EvidenceConflict):
        asyncio.run(workflow._commit_response(_services(repository), {"id": "a"}, {"raw": 1}))
    assert repository.calls == 1


def test_rate_limit_delay_is_read_from_the_typed_attribute():
    """The advised delay lives on the exception, not in its message string."""
    from iadopt_lab.persistence import RateLimitError

    error = RateLimitError(30.0)
    assert error.retry_after_seconds == 30.0
    assert not isinstance(error.args[0], (int, float))  # args[0] is prose, never the delay


def test_ranking_requires_the_identity_to_name_its_plan():
    """A self-consistent scorer identity must still be tied to the plan it claims."""
    from iadopt_lab.artifacts import collect_input_artifacts
    from iadopt_lab.configuration import load_parameters, resolve_configuration
    from iadopt_lab.planning import expand_campaign, synthetic_configuration

    resolved = synthetic_configuration(resolve_configuration(load_parameters(ROOT / "parameters.yml")))
    records = list(load_canonical_records(ROOT))
    targets = [row for row in records if not row["demonstration_position"]][:3]
    identity = {"backend": "synthetic-equality-only-v1", "synthetic": True}
    inputs = collect_input_artifacts(ROOT, records, targets, identity)
    plan = expand_campaign(resolved, targets, artifact_identities=inputs["identities"], mode="synthetic")

    with pytest.raises(ValueError, match="does not belong to this plan"):
        build_configuration_ranking(plan, [], scorer_identity={
            "scorer_version": "january-derived-member-credit-v1",
            "similarity_identity": identity, "close_threshold": 0.8})  # no plan_scorer


def test_synthetic_artifacts_describe_the_planned_population():
    """The artifact called `population` must hash the IDs the plan actually contains."""
    from iadopt_lab.artifacts import collect_input_artifacts
    from iadopt_lab.canonical import content_hash

    records = list(load_canonical_records(ROOT))
    targets = [row for row in records if not row["demonstration_position"]][:3]
    inputs = collect_input_artifacts(ROOT, records, targets,
                                     {"backend": "synthetic-equality-only-v1", "synthetic": True})
    assert inputs["identities"]["population"] == content_hash(sorted(r["variable_id"] for r in targets))


# --- Fixes from the 2026-09-09 read-only audit -----------------------------------


@pytest.mark.parametrize("message,expected", [
    ({}, ("rejected", "invalid_envelope")),
    ({"role": "assistant"}, ("rejected", "invalid_envelope")),
    ({"content": 42}, ("rejected", "invalid_envelope")),
    ({"content": ["a", "list"]}, ("rejected", "invalid_envelope")),
    ({"content": None}, ("response_received", "empty_response")),
    ({"content": ""}, ("response_received", "empty_response")),
])
def test_a_message_object_is_not_itself_a_completion(message, expected):
    """`{"message": {}}` is a missing completion structure, not an empty answer.

    Treating any dictionary as well-formed meant three malformed envelopes exhausted a
    task's three attempts and were scored as the model answering nothing.
    """
    envelope = {"choices": [{"finish_reason": "stop", "message": message}]}
    result, _ = _exchange("psnc", lambda request: httpx.Response(200, json=envelope))
    assert (result.delivery, result.outcome) == expected


def test_an_error_object_beside_an_empty_completion_is_a_provider_error():
    envelope = {"error": {"message": "gateway failure"},
                "choices": [{"finish_reason": "stop", "message": {"content": ""}}]}
    result, _ = _exchange("psnc", lambda request: httpx.Response(200, json=envelope))
    assert (result.delivery, result.outcome) == ("rejected", "provider_error_envelope")


def test_a_real_answer_survives_a_stray_error_field():
    envelope = {"error": None, "choices": [
        {"finish_reason": "stop", "message": {"content": '{"hasProperty": []}'}}]}
    result, _ = _exchange("psnc", lambda request: httpx.Response(200, json=envelope))
    assert result.delivery == "response_received" and result.assistant_text


def test_reasoning_is_retained_even_from_a_rejected_envelope():
    """Reasoning is evidence; a malformed completion must not discard it."""
    envelope = {"choices": [{"finish_reason": "stop",
                             "message": {"reasoning_content": "thinking out loud"}}]}
    result, _ = _exchange("psnc", lambda request: httpx.Response(200, json=envelope))
    assert result.outcome == "invalid_envelope"
    assert result.to_dict()["reasoning_text"] == "thinking out loud"


def test_a_commit_survives_cancellation_of_its_supervisor():
    """The outage that fails a commit usually fails the heartbeat that cancels it too."""
    from iadopt_lab.persistence import PersistenceError
    from iadopt_lab.workflow import _commit_response

    attempts, stored = {"count": 0}, {}

    class SlowRepository:
        def store_response(self, attempt, payload):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise PersistenceError("connection lost")
            stored.update(payload)
            return {"id": "response-1"}

    services = _services(SlowRepository())

    async def scenario():
        commit = asyncio.ensure_future(
            _commit_response(services, {"id": "attempt-1"}, {"raw_body": b"paid bytes"}))
        await asyncio.sleep(0.05)
        commit.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await commit
        # Cancelling the caller stops dispatch; the shielded preservation task continues.
        await asyncio.gather(*tuple(services.preservation), return_exceptions=True)

    asyncio.run(scenario())
    assert stored["raw_body"] == b"paid bytes"
    assert attempts["count"] == 2


def test_absent_categories_are_represented_not_omitted():
    """A run that scored nothing for a whole category must not look like no such category."""
    rows = [_observation("v1", category="Physics", subcategory="Optics")]
    summary = build_category_summary(rows, population_categories={
        "v1": {"category": "Physics", "category_path": "Physics/Optics"},
        "v9": {"category": "Biology", "category_path": "Biology/Cells"}})
    biology = [row for row in summary if row["category"] == "Biology"]
    assert biology, "a category with no observations disappeared from the summary"
    assert all(row["observed_variables"] == 0 and not row["complete"] for row in biology)
    assert all(row["missing_variable_ids"] == ["v9"] for row in biology)


def test_scorer_hash_is_recomputed_from_source_not_copied_from_the_plan():
    """Comparing a plan's hash to itself holds for any checkout and proves nothing."""
    from iadopt_lab.artifacts import scorer_identity_hash

    backend = {"backend": "sentence-transformers/all-MiniLM-L6-v2", "revision": "abc"}
    first = scorer_identity_hash(ROOT, backend)
    assert first == scorer_identity_hash(ROOT, backend)
    # A different similarity backend must produce a different scorer identity.
    assert first != scorer_identity_hash(ROOT, {**backend, "revision": "different"})


def test_a_first_attempt_throttle_returns_the_task_to_the_queue():
    """Admission runs before an attempt row exists, so a throttle there has no evidence.

    Releasing such a task as `retry_pending` asserts a retry of an attempt that was never
    made; the repository refuses it, and the resulting AttemptLimitError propagated out of
    the worker into the campaign's cancel-everything path. Ordinary throttling then failed
    the whole campaign and left unrelated in-flight requests as ambiguous deliveries.
    """
    from iadopt_lab.persistence import RateLimitError
    from iadopt_lab.workflow import _advance_task

    released = []

    class ThrottlingRepository:
        def get_task(self, task_id):
            return {"id": task_id, "state": "queued", "prediction": None, "attempts": [],
                    "attempt_count": 0, "provider": "psnc", "model_id": "M",
                    "variable": {"variable_id": "v1", "definition": "d", "category": "C",
                                 "subcategory": "S", "category_path": "C/S"},
                    "run": {"model_id": "M", "prompt_variant": "strict-minimal", "shot_count": 0,
                            "temperature": 0.5, "top_p": 1.0, "max_output_tokens": 16,
                            "reasoning_fields": {}, "configuration": {"model_id": "M"}}}

        def start_attempt(self, lease, request):
            raise RateLimitError(5.0)

        def release(self, lease, state=None, detail=None):
            released.append((state, detail))
            return {"state": state}

    services = _services(ThrottlingRepository())
    services.bundle = {
        "demonstrations": (), "plan": {"mode": "live", "population": ["v1"]},
        "configuration": {"providers": {"psnc": {"models": [
            {"id": "M", "context_window_tokens": 1048576}]}}}}
    services.token_bound = lambda task, body: 10
    services.cost_policy = lambda task, body: {"reservation_amount": "0", "bounded": True}

    asyncio.run(_advance_task({"id": "t1", "provider": "psnc"}, services))
    # Queued, not retry_pending: nothing was consumed, so there is no retry to pend.
    assert released == [("queued", {"code": "rate_limited"})]


def test_finalization_proceeds_with_terminal_operational_failures():
    """One truncation must not deny results for an entire campaign.

    A response truncated at the output ceiling is an operational failure that is
    deliberately not retryable — the same request would truncate again — and is
    deliberately not scored as a model-quality zero. Its task therefore can never reach
    `complete`. Refusing to finalize while any task is incomplete meant a single such
    task discarded every result the campaign did produce; at temperature 2 that is not
    hypothetical, it is routine.
    """
    from iadopt_lab.workflow import _TERMINAL_TASK_STATES, finalize_campaign

    assert set(_TERMINAL_TASK_STATES) == {"complete", "operational_failed", "ambiguous_delivery"}

    class Repo:
        def __init__(self, states):
            self.states = states

        def list_tasks(self, campaign_id):
            return [{"task_id": f"t{i}", "state": s} for i, s in enumerate(self.states)]

    def finalize(states):
        services = _services(Repo(states))
        services.bundle = {"plan": {}, "similarity_identity": {}}
        return asyncio.run(finalize_campaign("c1", services))

    # Still-advancing work blocks finalization, as before.
    with pytest.raises(LabError, match="can still advance"):
        finalize(["complete", "queued"])
    with pytest.raises(LabError, match="can still advance"):
        finalize(["complete", "retry_pending"])

    # Terminal failures get past the gate: the failure here is the *next* step needing a
    # real plan, which proves the incompleteness check no longer rejects them.
    for terminal in ("operational_failed", "ambiguous_delivery"):
        with pytest.raises(Exception) as caught:
            finalize(["complete", terminal])
        assert "can still advance" not in str(caught.value)


def test_a_provider_cooldown_returns_the_task_instead_of_failing_the_campaign():
    """One transient blip must not cancel every other in-flight request.

    A cooldown is set by whichever worker saw the blip, while the others are already past
    the eligibility check. At concurrency 24 that is the normal case, not a rare race: the
    next worker to allocate an attempt finds the provider ineligible. Letting that
    propagate ended a 10,476-task campaign after one failure in 4,745 calls.
    """
    from iadopt_lab.persistence import ProviderIneligible
    from iadopt_lab.workflow import _advance_task

    released = []

    def repo_for(attempt_count):
        class Repo:
            def get_task(self, task_id):
                return {"id": task_id, "state": "queued" if not attempt_count else "retry_pending",
                        "prediction": None, "attempts": [], "attempt_count": attempt_count,
                        "provider": "psnc", "model_id": "M",
                        "variable": {"variable_id": "v1", "definition": "d", "category": "C",
                                     "subcategory": "S", "category_path": "C/S"},
                        "run": {"model_id": "M", "prompt_variant": "strict-minimal",
                                "shot_count": 0, "temperature": 0.5, "top_p": 1.0,
                                "max_output_tokens": 16, "reasoning_fields": {},
                                "configuration": {"model_id": "M"}}}

            def start_attempt(self, lease, request):
                raise ProviderIneligible("This provider is not eligible for new dispatch")

            def release(self, lease, state=None, detail=None):
                released.append((state, detail))
                return {"state": state}
        return Repo()

    for attempts, expected in ((0, "queued"), (1, "retry_pending")):
        released.clear()
        services = _services(repo_for(attempts))
        services.bundle = {
            "demonstrations": (), "plan": {"mode": "live", "population": ["v1"]},
            "configuration": {"providers": {"psnc": {"models": [
                {"id": "M", "context_window_tokens": 1048576}]}}}}
        services.token_bound = lambda task, body: 10
        services.cost_policy = lambda task, body: {"reservation_amount": "0", "bounded": True}
        asyncio.run(_advance_task({"id": "t1", "provider": "psnc"}, services))
        assert released == [(expected, {"code": "provider_cooldown"})]


def test_an_exhausted_transient_error_fails_the_task_not_the_provider():
    """A 429 that runs out of retries is that task's failure, not the deployment's.

    Pausing the provider stops every queued task. One rate-limited task exhausting its
    three attempts blocked 10,348 others; only a failure that says something is wrong with
    the deployment itself should pause it.
    """
    from iadopt_lab.workflow import _advance_task

    calls = []

    class Repo:
        def get_task(self, task_id):
            return {"id": task_id, "state": "request_persisted", "prediction": None,
                    "attempt_count": 3, "provider": "openrouter", "model_id": "M",
                    "attempts": [{"id": "a3", "attempt_number": 3, "validation": None,
                                  "response": None, "request_body": {"model": "M"},
                                  "delivery": "not_dispatched"}],
                    "variable": {"variable_id": "v1", "definition": "d", "category": "C",
                                 "subcategory": "S", "category_path": "C/S"},
                    "run": {"model_id": "M", "prompt_variant": "strict-minimal", "shot_count": 0,
                            "temperature": 0.5, "top_p": 1.0, "max_output_tokens": 16,
                            "reasoning_fields": {}, "configuration": {"model_id": "M"}}}

        def mark_dispatched(self, attempt, lease=None):
            return {"dispatch_allowed": True}

        def store_response(self, attempt, payload):
            return {"id": "r1"}

        def set_provider_state(self, *args, **kwargs):
            calls.append(("set_provider_state", args[2]))
            return {}

        def release(self, lease, state=None, detail=None):
            calls.append(("release", state))
            return {"state": state}

    class Adapter:
        async def send_once(self, body):
            from iadopt_lab.providers.base import ProviderResult
            return ProviderResult(
                provider="openrouter", requested_model="M", request=body, raw_response="429",
                raw_response_base64="", assistant_text=None, status_code=429,
                outcome="classified_transient_provider_error", delivery="rejected",
                started_at="", finished_at="", latency_seconds=0.0)

    services = _services(Repo())
    services.bundle = {"demonstrations": (), "plan": {"mode": "live", "population": ["v1"]}}
    services.adapters = {"openrouter": Adapter()}
    asyncio.run(_advance_task({"id": "t1", "provider": "openrouter"}, services))

    assert ("set_provider_state", "paused") not in calls, "a 429 must not pause the provider"
    assert ("release", "operational_failed") in calls


def test_an_empty_poll_does_not_abandon_queued_work(monkeypatch):
    """Queued tasks must survive a moment when nothing happens to be claimable.

    A provider cooldown that expires between the claim attempt and the delay calculation
    leaves no work to claim and no wait to report. Breaking on that first empty poll
    abandoned 9,929 queued tasks mid-campaign. The loop must back off and look again, and
    give up only after several consecutive polls find nothing.
    """
    from iadopt_lab import workflow
    from iadopt_lab.workflow import _IDLE_POLLS_BEFORE_STOP, run_campaign

    polls = {"claims": 0}

    class Repo:
        def get_campaign(self, campaign_id):
            # Work remains, but none of it is claimable right now.
            return {"id": campaign_id, "mode": "synthetic", "state": "running",
                    "configuration": {"lab_plan_sha256": "plan-hash"},
                    "authorizations": [], "providers": [{"provider": "psnc", "state": "ready",
                                                         "cooldown_until": None}],
                    "states": {"queued": 500, "complete": 10}}

        def reconcile(self, campaign_id):
            return {"repaired": [], "ambiguous": []}

        def set_campaign_state(self, campaign_id, state):
            return {"state": state}

        def claim_tasks(self, *args, **kwargs):
            polls["claims"] += 1
            return []

    services = _services(Repo())
    services.bundle = {"plan": {"mode": "synthetic", "sha256": "plan-hash",
                                "counts": {"tasks": 510}},
                       "configuration": {"campaign": {"live_calls_enabled": False,
                                                      "providers": ["psnc"]},
                                         "execution": {"worker_count": 2, "task_lease_seconds": 300,
                                                       "heartbeat_seconds": 30},
                                         "providers": {"psnc": {"max_concurrency": 2,
                                                                "requests_per_minute": 60,
                                                                "tokens_per_minute": 100000}}}}
    monkeypatch.setattr(workflow, "verify_bundle", lambda *a, **k: None)
    # Collapse the back-off waits so the test is fast; bind the real sleep first, or the
    # replacement calls itself.
    real_sleep = asyncio.sleep
    monkeypatch.setattr(workflow.asyncio, "sleep", lambda *a, **k: real_sleep(0))

    result = asyncio.run(run_campaign("c1", services))

    # It gave up eventually, but only after repeated looks, and it said why.
    assert result["stop_reason"] == "no_claimable_work"
    assert polls["claims"] >= _IDLE_POLLS_BEFORE_STOP, (
        f"gave up after {polls['claims']} polls; must retry at least {_IDLE_POLLS_BEFORE_STOP}")


@pytest.mark.parametrize(
    ("outcome", "status", "should_pause"),
    [
        ("output_truncated", 200, False),
        ("classified_transient_provider_error", 429, False),
        ("provider_error", 401, True),
        ("html_response", 200, True),
    ],
)
def test_only_a_deployment_failure_pauses_the_provider(outcome, status, should_pause):
    """Pausing must follow the deployment's health, not one task's content.

    `output_truncated` is an HTTP 200 whose only fault is that this prompt and model
    reached the configured output ceiling: the deployment answered perfectly. Pausing
    over it stopped a live campaign after 1,291 of 17,460 tasks and stranded the other
    16,167, because nothing but this decision distinguishes the two cases.
    """
    from iadopt_lab.workflow import _advance_task

    calls = []

    class Repo:
        def get_task(self, task_id):
            return {"id": task_id, "state": "request_persisted", "prediction": None,
                    "attempt_count": 3, "provider": "openrouter", "model_id": "M",
                    "campaign_id": "c1",
                    "attempts": [{"id": "a3", "attempt_number": 3, "validation": None,
                                  "response": None, "request_body": {"model": "M"},
                                  "delivery": "not_dispatched"}],
                    "variable": {"variable_id": "v1", "definition": "d", "category": "C",
                                 "subcategory": "S", "category_path": "C/S"},
                    "run": {"model_id": "M", "prompt_variant": "strict-minimal", "shot_count": 0,
                            "temperature": 0.5, "top_p": 1.0, "max_output_tokens": 16,
                            "reasoning_fields": {}, "configuration": {"model_id": "M"}}}

        def mark_dispatched(self, attempt, lease=None):
            return {"dispatch_allowed": True}

        def store_response(self, attempt, payload):
            return {"id": "r1"}

        def set_provider_state(self, *args, **kwargs):
            calls.append(("set_provider_state", args[2]))
            return {}

        def release(self, lease, state=None, detail=None):
            calls.append(("release", state))
            return {"state": state}

    class Adapter:
        async def send_once(self, body):
            from iadopt_lab.providers.base import ProviderResult
            return ProviderResult(
                provider="openrouter", requested_model="M", request=body, raw_response="x",
                raw_response_base64="", assistant_text=None, status_code=status,
                outcome=outcome, delivery="rejected",
                started_at="", finished_at="", latency_seconds=0.0)

    services = _services(Repo())
    services.bundle = {"demonstrations": (), "plan": {"mode": "live", "population": ["v1"]}}
    services.adapters = {"openrouter": Adapter()}
    asyncio.run(_advance_task({"id": "t1", "provider": "openrouter", "campaign_id": "c1"}, services))

    paused = ("set_provider_state", "paused") in calls
    assert paused is should_pause, (
        f"{outcome} {'must' if should_pause else 'must not'} pause the provider")
    assert ("release", "operational_failed") in calls, "the task itself always fails"


@pytest.mark.parametrize("error_type", [ValueError, TypeError, KeyError, ZeroDivisionError])
def test_a_scorer_rejection_fails_the_task_not_the_campaign(error_type):
    """A prediction the scorer refuses must not be able to abort the run.

    The validator is meant to reject anything unscorable, so reaching the scorer with a
    bad prediction is a defect - but it is still one task's. Letting the exception
    propagate ended a live campaign at 995 of 17,460 tasks, and since the offending
    response was already durable every resume replayed it and died at the same point.
    """
    from iadopt_lab.workflow import _score

    calls = []

    class Repo:
        def store_evaluation(self, lease, result):
            calls.append("store_evaluation")
            return {}

        def release(self, lease, state=None, detail=None):
            calls.append(("release", state, (detail or {}).get("code")))
            return {"state": state}

    def explode(*args, **kwargs):
        raise error_type("prediction/hasContextObject: whitespace-only entity is invalid")

    services = _services(Repo())
    services.bundle = {"plan": {"mode": "live"}, "similarity_identity": {}}
    task = {"evaluation": None, "gold": {}, "prediction": {"canonical": {}},
            "variable": {"variable_id": "v1", "category": "C", "subcategory": "S"},
            "run": {"id": "r1"}}

    import iadopt_lab.workflow as workflow

    original = workflow.evaluate_item
    workflow.evaluate_item = explode
    try:
        asyncio.run(_score(task, {"id": "t1"}, services))
    finally:
        workflow.evaluate_item = original

    assert ("release", "operational_failed", "scorer_rejected_validated_prediction") in calls
    assert "store_evaluation" not in calls, "a rejected prediction must not be scored"
