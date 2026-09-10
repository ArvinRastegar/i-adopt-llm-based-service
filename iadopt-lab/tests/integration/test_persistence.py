"""Real PostgreSQL checks; use only the explicitly designated isolated test DB."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from iadopt_eval.core import evaluate_item
from iadopt_lab.persistence import (
    AttemptLimitError,
    BudgetError,
    EvidenceConflict,
    PersistenceError,
    RateLimitError,
    Repository,
    StaleLeaseError,
    migrate,
)

pytestmark = pytest.mark.integration


def canonical(value):
    """Encode fixture JSON canonically; return exact bytes with no I/O."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    """Return SHA-256 of fixture bytes/JSON without writing any evidence."""
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


@pytest.fixture(scope="module")
def dsn():
    """Read only the explicit isolated test DSN; skip safely when it is absent."""
    value = os.environ.get("IADOPT_LAB_TEST_DATABASE_URL")
    if not value:
        pytest.skip("Set IADOPT_LAB_TEST_DATABASE_URL for dedicated PostgreSQL tests")
    if conninfo_to_dict(value).get("dbname") != "iadopt_lab_test":
        pytest.fail("Persistence tests require the dedicated iadopt_lab_test database")
    migrate(value)
    return value


@pytest.fixture
def repo(dsn):
    """Open a repository for this test; close its pool without deleting evidence."""
    with Repository(dsn, max_size=8) as repository:
        yield repository


def fixture_campaign(
    repo,
    *,
    providers=("psnc",),
    count=3,
    mode="synthetic",
    provider_limits=None,
    billing_mode="non_billed",
    provider_cap=None,
    global_cap=None,
):
    """Create uniquely named synthetic fixtures and return campaign/runs/variables.

    Inputs allow a provider union, population size and explicit live-gate tests.
    Writes only dedicated test-database records; no provider calls or deletions.
    """
    unique = uuid.uuid4().hex
    gold = {
        "hasStatisticalModifier": "",
        "hasProperty": "temperature",
        "hasObjectOfInterest": "water",
        "hasMatrix": "",
        "hasContextObject": "",
        "hasConstraint": [],
    }
    variables = []
    for number in range(count):
        raw = f"fixture source {unique}/{number}".encode()
        variables.append(
            {
                "variable_id": f"test:{unique}:{number}",
                "source_path": f"Life Sciences/Biology/{number}.ttl",
                "label": f"variable {number}",
                "definition": "temperature of water",
                "source_iri": "same:upstream-iri",
                "category": "Life Sciences",
                "subcategory": "Biology",
                "category_path": "Life Sciences/Biology",
                "gold": gold,
                "gold_sha256": digest(gold),
                "source_sha256": digest(raw),
                "source_content": raw,
            }
        )
    corpus = repo.register_corpus(
        {
            "repository": "synthetic",
            "release": "test",
            "commit": unique,
            "tree": unique,
            "expected_count": count,
        },
        variables,
    )
    config = {
        "campaign": {"name": unique, "providers": list(providers)},
        "providers": {
            p: {
                **(provider_limits or {}),
                "billing": {
                    "mode": billing_mode,
                    "basis": "explicit offline test account",
                    "maximum_provider_cost": {"amount": provider_cap, "currency": "EUR"},
                },
                "models": [{"id": "same-model", "enabled": True}],
            }
            for p in providers
        },
        "execution": {"maximum_campaign_cost": {"amount": global_cap, "currency": "EUR"}},
    }
    campaign_id = repo.register_campaign(config, mode=mode)
    runs = [
        {
            "configuration_id": f"{p}-configuration",
            "provider": p,
            "model_id": "same-model",
            "reasoning_mode": "not_applicable",
            "reasoning_fields": {},
            "prompt_variant": "strict-minimal",
            "shot_count": 0,
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": 64,
            "repetition": 1,
            "configuration": {"fixture": True},
        }
        for p in providers
    ]
    repo.plan_tasks(campaign_id, runs, corpus["variables"])
    return campaign_id, runs, corpus["variables"]


def request_for(lease, number=1):
    """Build a frozen fixture request for one lease/number; return JSON evidence."""
    prompt = f"Return decomposition; attempt {number}"
    messages = [{"role": "user", "content": prompt}]
    return {
        "attempt_number": number,
        "messages": messages,
        "prompt": prompt,
        "body": {
            "model": lease["model_id"],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 64,
            "messages": messages,
        },
        "scientific_parameters": {
            "model": lease["model_id"],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 64,
        },
    }


def delivered(repo, lease, number=1, valid=True):
    """Record one mock dispatch/raw response/validation; return its attempt record."""
    attempt = repo.start_attempt(lease, request_for(lease, number))
    assert repo.mark_dispatched(attempt)["dispatch_allowed"]
    repo.store_response(
        attempt,
        {
            "raw_body": "```json\n{}\n```",
            "assistant_text": "```json\n{}\n```",
            "delivery": "response_received",
            "http_status": 200,
            "usage": None,
        },
    )
    repo.record_validation(
        attempt,
        {
            "valid": valid,
            "errors": []
            if valid
            else [
                {
                    "stage": "schema",
                    "code": "required",
                    "pointer": "/hasProperty",
                    "message": "missing required key",
                }
            ],
            "candidate": lease["gold"] if valid else {},
            "validator_version": "fixture-v1",
        },
    )
    return attempt


def finish(repo, lease):
    """Complete one fixture task with real pure scoring and durable exact receipts."""
    attempt = delivered(repo, lease)
    repo.select_prediction(lease, lease["gold"], attempt)
    result = evaluate_item(
        lease["gold"],
        lease["gold"],
        lambda a, b: float(a == b),
        metadata={"variable_id": lease["variable"]["variable_id"]},
        similarity_identity={"kind": "test-exact"},
    )
    evaluation = repo.store_evaluation(lease, result)
    repo.release(lease)
    return evaluation


def test_registration_planning_and_union_are_idempotent(repo):
    """Verify provider-owned grids, categories, and repeated planning remain distinct."""
    campaign, runs, variables = fixture_campaign(repo, providers=("psnc", "openrouter"))
    first = repo.plan_tasks(campaign, runs, variables)
    second = repo.plan_tasks(campaign, list(reversed(runs)), list(reversed(variables)))
    assert first["fingerprint"] == second["fingerprint"]
    assert first["task_count"] == 6
    assert {row["provider"]: row["task_count"] for row in first["providers"]} == {
        "psnc": 3,
        "openrouter": 3,
    }
    tasks = repo.list_tasks(campaign)
    assert len({task["task_id"] for task in tasks}) == 6
    assert all(task["variable"]["category_path"] == "Life Sciences/Biology" for task in tasks)
    assert all(task["variable"]["source_iri"] == "same:upstream-iri" for task in tasks)
    with pytest.raises(EvidenceConflict):
        repo.plan_tasks(campaign, runs, variables[:2])


def test_concurrent_claims_and_stale_fence(repo):
    """Two real transactions cannot claim one active task; old leases lose authority."""
    campaign, _, _ = fixture_campaign(repo)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(
            pool.map(
                lambda worker: repo.claim_tasks(campaign, worker, limit=3), ["worker-a", "worker-b"]
            )
        )
    leases = [lease for batch in claims for lease in batch]
    assert len(leases) == 3 and len({lease["id"] for lease in leases}) == 3
    old = leases[0]
    repo.release(old)
    new = repo.claim_tasks(campaign, "worker-new", limit=3)[0]
    assert new["id"] == old["id"] and new["fence"] > old["fence"]
    with pytest.raises(StaleLeaseError):
        repo.heartbeat(old)
    repo.release(new)


def test_raw_bytes_idempotency_and_database_immutability(repo):
    """Preserve non-UTF8 response bytes and forbid conflicting/SQL-level mutation."""
    campaign, _, _ = fixture_campaign(repo)
    lease = repo.claim_tasks(campaign, "raw-worker")[0]
    attempt = repo.start_attempt(lease, request_for(lease))
    assert repo.start_attempt(lease, request_for(lease))["id"] == attempt["id"]
    assert repo.mark_dispatched(attempt)["dispatch_allowed"]
    assert not repo.mark_dispatched(attempt)["dispatch_allowed"]
    raw = b'\xff\x00  {"choices":[]}\r\n'
    result = {
        "raw_response": raw.decode(errors="replace"),
        "raw_response_base64": base64.b64encode(raw).decode(),
        "assistant_text": None,
        "delivery": "response_received",
        "usage": None,
    }
    stored = repo.store_response(attempt, result)
    assert repo.store_response(attempt, result)["id"] == stored["id"]
    assert repo.list_attempts(lease["id"])[0]["response"]["raw_body"] == raw
    with pytest.raises(EvidenceConflict):
        repo.store_response(attempt, {**result, "assistant_text": "changed"})
    with pytest.raises(psycopg.IntegrityError), repo.pool.connection() as conn:
        conn.execute("UPDATE response SET assistant_text='changed' WHERE id=%s", (stored["id"],))
    repo.release(lease)


def test_three_invalid_attempts_and_exact_empty_provenance(repo):
    """Every invalid call is durable, attempt four fails, and only three justify empty."""
    campaign, _, _ = fixture_campaign(repo, count=1)
    empty = {
        "hasStatisticalModifier": "",
        "hasProperty": "",
        "hasObjectOfInterest": "",
        "hasMatrix": "",
        "hasContextObject": "",
        "hasConstraint": [],
    }
    for number in range(1, 4):
        lease = repo.claim_tasks(campaign, "retry-worker")[0]
        attempt = delivered(repo, lease, number, valid=False)
        if number < 3:
            with pytest.raises(PersistenceError):
                repo.select_prediction(lease, empty, attempt, terminal_invalid=True)
            repo.release(lease, "retry_pending", {"reason": "content_invalid"})
    with pytest.raises(AttemptLimitError):
        repo.start_attempt(lease, request_for(lease, 4))
    prediction = repo.select_prediction(lease, empty, attempt, terminal_invalid=True)
    assert prediction["terminal_invalid"] and prediction["canonical"] == empty
    assert len(repo.list_attempts(lease["id"])) == 3
    assert all(a["validation"]["errors"] for a in repo.list_attempts(lease["id"]))
    repo.release(lease)


def test_provider_pause_isolation_and_stored_response_progress(repo):
    """Pause new calls at one provider while another provider and local stages advance."""
    campaign, _, _ = fixture_campaign(repo, providers=("psnc", "openrouter"))
    lease = repo.claim_tasks(campaign, "pause-worker", provider="psnc")[0]
    delivered(repo, lease)
    repo.release(lease)
    repo.set_provider_state(
        campaign,
        "psnc",
        "cooldown",
        {"reason": "rate_limit"},
        datetime.now(UTC) + timedelta(minutes=1),
    )
    local = repo.claim_tasks(campaign, "local-worker", limit=9, provider="psnc")
    assert len(local) == 1 and local[0]["state"] == "validated"
    assert len(repo.claim_tasks(campaign, "other-worker", limit=9, provider="openrouter")) == 3


def test_complete_ranking_requires_all_results_and_retains_exact_metrics(repo):
    """Verify real scores, rank ties, all-provider coverage, and final report gates."""
    campaign, runs, _ = fixture_campaign(repo, providers=("psnc", "openrouter"))
    with pytest.raises(PersistenceError):
        repo.complete_campaign(campaign)
    for lease in repo.claim_tasks(campaign, "completion-worker", limit=20):
        finish(repo, lease)
    rows = [
        {
            "configuration_id": run["configuration_id"],
            "provider": run["provider"],
            "model_id": run["model_id"],
            "rank": 1,
            "primary": {"numerator": 1, "denominator": 1},
            "reason": None,
        }
        for run in runs
    ]
    ranking = {"policy_version": "mean-repetition-micro-close-f1-v1", "configurations": rows}
    first = repo.save_ranking(campaign, ranking)
    assert repo.save_ranking(campaign, ranking)["id"] == first["id"]
    with pytest.raises(EvidenceConflict):
        repo.save_ranking(
            campaign, {**ranking, "configurations": [{**row, "rank": 2} for row in rows]}
        )
    with pytest.raises(PersistenceError):
        repo.complete_campaign(campaign)
    report = {
        "schema_version": "1.0",
        "campaign_id": campaign,
        "kind": "fixture-report",
        "files": [{"path": "outputs/fixture.json", "sha256": digest(b"{}"), "byte_length": 2}],
        "final": True,
    }
    repo.save_report(campaign, report)
    assert repo.complete_campaign(campaign)["state"] == "complete"
    with repo.pool.connection() as conn:
        n = conn.execute(
            "SELECT count(*) AS n FROM evaluation_facts WHERE campaign_id=%s", (campaign,)
        ).fetchone()["n"]
    assert n == 6 * 2 * 7 * 7  # tasks × modes × (six components + variable) × metrics/counts


def test_live_admission_requires_estimate_disclosure_and_authorization(repo):
    """A real-mode plan without approval cannot allocate a generation attempt."""
    campaign, _, _ = fixture_campaign(repo, count=97, mode="live")
    lease = repo.claim_tasks(campaign, "live-gate-worker")[0]
    with pytest.raises(PersistenceError, match="estimate"):
        repo.start_attempt(lease, request_for(lease))
    assert repo.list_attempts(lease["id"]) == []
    repo.release(lease)


def test_application_role_can_write_but_cannot_mutate_evidence():
    """Exercise actual app-role privileges without changing migrations or old facts."""
    app_dsn = os.environ.get("IADOPT_LAB_TEST_APP_DATABASE_URL")
    if not app_dsn:
        pytest.skip("Set IADOPT_LAB_TEST_APP_DATABASE_URL for worker role checks")
    if conninfo_to_dict(app_dsn).get("dbname") != "iadopt_lab_test":
        pytest.fail("App-role tests require the dedicated test database")
    with Repository(app_dsn) as repo:
        campaign, _, _ = fixture_campaign(repo, count=1)
        lease = repo.claim_tasks(campaign, "app-role-worker")[0]
        finish(repo, lease)
        assert repo.get_task(lease["id"])["state"] == "complete"
        with pytest.raises(psycopg.Error), repo.pool.connection() as conn:
            conn.execute(
                "DELETE FROM response WHERE attempt_id=%s",
                (repo.list_attempts(lease["id"])[0]["id"],),
            )


def test_shared_concurrency_and_rate_admission(repo):
    """Concurrent workers obey database-wide provider slots and rolling rate windows."""
    campaign, _, _ = fixture_campaign(
        repo,
        provider_limits={"max_concurrency": 1, "requests_per_minute": 1, "tokens_per_minute": 100},
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        leases = [
            lease
            for group in pool.map(
                lambda worker: repo.claim_tasks(campaign, worker, limit=3), ["rate-a", "rate-b"]
            )
            for lease in group
        ]
    assert len(leases) == 1
    first = leases[0]
    request = {**request_for(first), "token_bound": 50}
    repo.start_attempt(first, request)
    repo.release(first, "request_persisted")
    second = repo.claim_tasks(campaign, "rate-c")[0]
    if second["id"] == first["id"]:
        # The persisted unsent attempt is held as a legitimate occupied slot; expire
        # it for this rate test and select a different target explicitly via SQL.
        repo.release(second, "paused_configuration", {"reason": "fixture selects another target"})
        second = repo.claim_tasks(campaign, "rate-d")[0]
    with pytest.raises(RateLimitError) as rejected:
        repo.start_attempt(second, {**request_for(second), "token_bound": 50})
    assert 0 < rejected.value.retry_after_seconds <= 61
    assert repo.list_attempts(second["id"]) == []
    repo.release(second)


def approve_fixture(repo, campaign):
    """Record explicitly synthetic test receipts to exercise live database gates only."""
    plan = repo.get_campaign(campaign)["plan"]
    estimate = {
        "plan_fingerprint": plan["fingerprint"],
        "policy_version": "pre-run-estimate-v1",
        "usable": True,
        "price_evidence": {"kind": "offline-test-price"},
    }
    receipt_hash = digest(estimate)
    return repo.record_live_authorization(
        campaign,
        estimate,
        {"estimate_hash": receipt_hash, "disclosed_at": "2026-09-08T00:00:00+00:00"},
        {
            "estimate_hash": receipt_hash,
            "plan_fingerprint": plan["fingerprint"],
            "authorized_at": "2026-09-08T00:00:01+00:00",
            "actor": "offline integration fixture; no external call",
            "explicit": True,
        },
    )


def test_atomic_optional_cap_and_uncapped_settlement(repo):
    """Concurrent paid reservations cannot overspend a cap; null caps remain uncapped."""
    campaign, _, _ = fixture_campaign(
        repo,
        providers=("openrouter",),
        count=97,
        mode="live",
        billing_mode="metered",
        provider_cap="1.00",
        global_cap="1.00",
    )
    approve_fixture(repo, campaign)
    leases = repo.claim_tasks(campaign, "accounting-worker", limit=2)
    cost = {
        "reservation_amount": "0.75",
        "bounded": True,
        "price_evidence": {"version": "test-price"},
    }

    def reserve(lease):
        """Return admission status for one thread; catch only the expected cap error."""
        try:
            return repo.start_attempt(lease, {**request_for(lease), "cost": cost})
        except BudgetError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = list(pool.map(reserve, leases))
    assert sum(item is not None for item in attempts) == 1
    state = repo.get_campaign(campaign)
    assert str(state["reserved_cost"]) == "0.75"
    admitted = next(item for item in attempts if item)
    repo.mark_dispatched(admitted)
    response = {
        "raw_body": "{}",
        "assistant_text": "{}",
        "delivery": "response_received",
        "cost": {"amount": "0.50", "state": "actual"},
    }
    repo.store_response(admitted, response)
    repo.store_response(admitted, response)
    state = repo.get_campaign(campaign)
    assert state["reserved_cost"] == 0 and str(state["spent_cost"]) == "0.50"
    uncapped, _, _ = fixture_campaign(
        repo, providers=("openrouter",), count=97, mode="live", billing_mode="metered"
    )
    approve_fixture(repo, uncapped)
    lease = repo.claim_tasks(uncapped, "uncapped-worker")[0]
    accepted = repo.start_attempt(
        lease,
        {**request_for(lease), "cost": {**cost, "reservation_amount": "1000", "bounded": False}},
    )
    assert accepted["attempt_number"] == 1


def test_sql_identity_and_attempt_counter_cannot_be_rewritten(repo):
    """Database triggers protect task identity and attempt count from direct mutation."""
    campaign, _, _ = fixture_campaign(repo)
    lease = repo.claim_tasks(campaign, "identity-worker")[0]
    repo.start_attempt(lease, request_for(lease))
    with pytest.raises(psycopg.IntegrityError), repo.pool.connection() as conn:
        conn.execute("UPDATE task SET attempt_count=0 WHERE id=%s", (lease["id"],))
    with pytest.raises(psycopg.IntegrityError), repo.pool.connection() as conn:
        conn.execute("UPDATE task SET fingerprint=%s WHERE id=%s", ("f" * 64, lease["id"]))
    assert repo.get_task(lease["id"])["attempt_count"] == 1


def test_provider_metadata_never_blocks_raw_preservation(repo):
    """A field the provider happened to name `password` must not discard a paid answer.

    Rejecting it rolled back the whole transaction, taking the raw response with it. The
    value is replaced and never stored; the substitution is recorded as evidence.
    """
    campaign, _, _ = fixture_campaign(repo)
    lease = repo.claim_tasks(campaign, "redaction-worker")[0]
    attempt = repo.start_attempt(lease, request_for(lease))
    assert repo.mark_dispatched(attempt)["dispatch_allowed"]

    raw = b'{"choices": [{"message": {"content": "{}"}}]}'
    stored = repo.store_response(attempt, {
        "raw_response": raw.decode(),
        "raw_response_base64": base64.b64encode(raw).decode(),
        "assistant_text": "{}",
        "delivery": "response_received",
        "provider_detail": {"password": "hunter2", "region": "eu"},
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}})

    assert repo.list_attempts(lease["id"])[0]["response"]["raw_body"] == raw
    evidence = stored["evidence"]
    assert evidence["provider_detail"]["password"] == "[redacted: credential-shaped evidence]"
    assert evidence["provider_detail"]["region"] == "eu"
    assert evidence["evidence_redactions"] == ["provider_detail.password"]
    repo.release(lease)
