"""Crash-checkpoint tests against the explicitly selected isolated PostgreSQL DB."""

from __future__ import annotations

import os
import uuid

import pytest
from psycopg.conninfo import conninfo_to_dict

from iadopt_lab.persistence import Repository, StaleLeaseError

pytestmark = pytest.mark.integration


@pytest.fixture
def repo():
    """Open only the dedicated test database named in the test environment."""
    value = os.environ.get("IADOPT_LAB_TEST_DATABASE_URL")
    if not value:
        pytest.skip("Explicit isolated PostgreSQL test DSN is not configured")
    if conninfo_to_dict(value).get("dbname") != "iadopt_lab_test":
        pytest.fail("Recovery tests require iadopt_lab_test")
    with Repository(value) as repository:
        yield repository


def prepare(repo):
    """Create one unique resumable fixture; return its campaign ID and task lease."""
    import hashlib
    import json

    gold = {
        "hasStatisticalModifier": "",
        "hasProperty": "temperature",
        "hasObjectOfInterest": "water",
        "hasMatrix": "",
        "hasContextObject": "",
        "hasConstraint": [],
    }
    raw = b"recovery fixture"
    variable = {
        "variable_id": "recovery:" + uuid.uuid4().hex,
        "source_path": "Life/Biology/fixture.ttl",
        "definition": "water temperature",
        "gold": gold,
        "category": "Life",
        "subcategory": "Biology",
        "category_path": "Life/Biology",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "gold_sha256": hashlib.sha256(
            json.dumps(gold, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_content": raw,
    }
    corpus = repo.register_corpus({"commit": uuid.uuid4().hex, "expected_count": 1}, [variable])
    campaign = repo.register_campaign(
        {
            "campaign": {"name": uuid.uuid4().hex, "providers": ["psnc"]},
            "providers": {
                "psnc": {
                    "billing": {"mode": "non_billed", "basis": "test"},
                    "models": [{"id": "fixture"}],
                }
            },
        }
    )
    run = {
        "configuration_id": "fixture",
        "provider": "psnc",
        "model_id": "fixture",
        "reasoning_mode": "not_applicable",
        "prompt_variant": "strict-minimal",
        "shot_count": 0,
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 64,
        "repetition": 1,
    }
    repo.plan_tasks(campaign, [run], corpus["variables"])
    return campaign, repo.claim_tasks(campaign, "recovery-a")[0]


def attempt(repo, lease):
    """Persist one request without dispatching it; return its attempt evidence."""
    return repo.start_attempt(
        lease,
        {
            "attempt_number": 1,
            "messages": [{"role": "user", "content": "decompose"}],
            "prompt": "decompose",
            "body": {"model": "fixture", "temperature": 0, "top_p": 1, "max_tokens": 64},
        },
    )


def expire(repo, lease):
    """Simulate process loss by expiring this test lease using the migration role."""
    with repo.pool.connection() as conn:
        conn.execute(
            "UPDATE task SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=%s",
            (lease["id"],),
        )


def test_raw_response_checkpoint_resumes_without_provider_request(repo):
    """A stored raw response survives worker loss and resumes only local validation."""
    campaign, lease = prepare(repo)
    request = attempt(repo, lease)
    repo.mark_dispatched(request)
    repo.store_response(
        request,
        {
            "raw_body": '  {"x": 1} \n',
            "assistant_text": '  {"x": 1} \n',
            "delivery": "response_received",
        },
    )
    expire(repo, lease)
    report = repo.reconcile(campaign)
    assert report["ambiguous_tasks"] == []
    newer = repo.claim_tasks(campaign, "recovery-b")[0]
    assert newer["state"] == "response_stored" and len(newer["attempts"]) == 1
    assert newer["attempts"][0]["response"]["raw_body"] == b'  {"x": 1} \n'
    with pytest.raises(StaleLeaseError):
        repo.heartbeat(lease)
    assert not repo.mark_dispatched(request, lease=newer)["dispatch_allowed"]
    repo.release(newer)
    assert repo.reconcile(campaign)["repaired_tasks"] == []


def test_dispatch_without_response_remains_ambiguous(repo):
    """A crash after dispatch intent cannot allocate a replacement request on resume."""
    campaign, lease = prepare(repo)
    request = attempt(repo, lease)
    repo.mark_dispatched(request)
    expire(repo, lease)
    result = repo.reconcile(campaign)
    assert result["ambiguous_tasks"] == [lease["id"]]
    assert repo.get_task(lease["id"])["state"] == "ambiguous_delivery"
    assert repo.claim_tasks(campaign, "recovery-b") == []
    assert len(repo.list_attempts(lease["id"])) == 1


def test_before_dispatch_checkpoint_preserves_same_attempt(repo):
    """A known un-dispatched request is claimable with its original attempt identity."""
    campaign, lease = prepare(repo)
    request = attempt(repo, lease)
    expire(repo, lease)
    repo.reconcile(campaign)
    newer = repo.claim_tasks(campaign, "recovery-b")[0]
    assert newer["state"] == "request_persisted"
    assert newer["attempts"][0]["id"] == request["id"]
    assert repo.mark_dispatched(request, lease=newer)["dispatch_allowed"]
    assert len(repo.list_attempts(lease["id"])) == 1
