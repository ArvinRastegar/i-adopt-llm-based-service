"""Regressions for the execution/reporting boundaries fixed after the 2026-09-09 review.

Each test pins one behaviour that previously failed silently: a broken transport being
scored as a model answer, a report accepting the wrong scorer, a category claiming
completeness from the rows it was handed, and a saved manifest escaping verification.
"""

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import yaml
from test_provider_transport import _exchange

from iadopt_lab.corpus.ingestion import load_canonical_records
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


def test_verify_detects_a_tampered_population_manifest(tmp_path):
    """Deleting or editing a saved manifest must fail verification, not pass silently."""
    manifest = ROOT / "data/manifests/evaluation-population-v2.0.1.yml"
    original = manifest.read_bytes()
    try:
        tampered = yaml.safe_load(original)
        tampered["members"] = tampered["members"][:-1]
        manifest.write_text(yaml.safe_dump(tampered, sort_keys=True, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match="evaluation-population"):
            load_canonical_records(ROOT)
    finally:
        manifest.write_bytes(original)
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
