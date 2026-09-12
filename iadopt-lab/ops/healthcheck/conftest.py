"""One consistent read of the live campaign, shared by every health check.

These are not unit tests. They assert that a campaign running right now is behaving the
way the two completed campaigns did, and they are deliberately outside `testpaths`
(`tests/`) so a normal `pytest` run never collects them.

Every threshold here is derived from stored evidence of campaigns `5cdc9417` (PSNC,
10,476 tasks) and `844df00e` (OpenRouter, 10,332 tasks), never guessed, and each is set
well outside the observed range: the job is to detect breakage, not to grade quality.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Two files, deliberately. STATE is held back so progress is measured against something
# old enough to have moved; LATEST is overwritten every run so the notification reports
# what is true now. Reporting from STATE made every summary lag by up to five minutes
# and understate progress.
STATE = ROOT / ".runtime/campaign/health-state.json"
LATEST = ROOT / ".runtime/campaign/health-latest.json"
LOG = ROOT / ".runtime/campaign/supervisor.log"
LOCK_PID = ROOT / ".runtime/campaign/supervisor.lock/pid"

# Observed in the two completed campaigns; a check fires only well outside these.
#                        PSNC 5cdc9417   OpenRouter 844df00e
BASELINE = {
    "valid_json_rate": (0.953, 0.820),
    "close_f1_mean": (0.269, 0.226),
    "zero_f1_share": (0.375, 0.470),
    "empty_prediction_rate": (0.0046, 0.0446),
    "http_error_rate": (0.0, 0.0457),
    "truncation_rate": (0.0, 0.0),
    "latency_mean_s": (2.06, 3.15),
    "cost_per_task_usd": (0.0, 0.000197),
    "has_property_filled": (0.985, 0.948),
    "has_object_filled": (0.951, 0.936),
}
# Enough finished work for a rate to mean anything. Below this a check skips rather
# than firing on three unlucky responses in the first minute.
MIN_SAMPLE = 60


def _sql(conn, query: str, *args):
    return conn.execute(query, args).fetchone()


@pytest.fixture(scope="session")
def health() -> dict:
    """Read the live campaign once, so all twenty checks judge the same instant."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    import psycopg

    from iadopt_lab.local_database import local_dsn

    with psycopg.connect(local_dsn(ROOT, role="app"), connect_timeout=10) as conn:
        conn.read_only = True
        conn.execute("SET search_path TO iadopt_lab, public")
        # IADOPT_HEALTH_CAMPAIGN pins a specific campaign instead of the newest live one.
        # It exists so these checks can be validated against campaigns already known to
        # have failed, rather than being trusted merely because they pass today.
        pinned = os.environ.get("IADOPT_HEALTH_CAMPAIGN")
        row = _sql(conn, "SELECT id,state,spent_cost,maximum_cost,created_at FROM campaign "
                         "WHERE id::text LIKE %s ORDER BY created_at DESC LIMIT 1",
                   pinned + "%") if pinned else _sql(
            conn, "SELECT id,state,spent_cost,maximum_cost,created_at FROM campaign "
                  "WHERE mode='live' ORDER BY created_at DESC LIMIT 1")
        if row is None:
            pytest.skip("no live campaign has been registered")
        cid = str(row[0])
        states = dict(conn.execute(
            "SELECT state,count(*) FROM task WHERE campaign_id=%s GROUP BY state", (cid,)).fetchall())
        data = {
            "campaign_id": cid, "campaign_state": row[1],
            "spent": float(row[2] or 0), "cap": float(row[3]) if row[3] else None,
            "started": row[4], "states": states,
            "total": sum(states.values()),
            "complete": states.get("complete", 0),
            "ambiguous": states.get("ambiguous_delivery", 0),
            "failed": states.get("operational_failed", 0),
            "leases": _sql(conn, "SELECT count(*) FROM task WHERE campaign_id=%s "
                                 "AND lease_expires_at>clock_timestamp()", cid)[0],
            "provider": dict(conn.execute(
                "SELECT provider,state FROM campaign_provider WHERE campaign_id=%s", (cid,)).fetchall()),
            "cooldown_until": _sql(conn, "SELECT max(cooldown_until) FROM campaign_provider "
                                         "WHERE campaign_id=%s", cid)[0],
            "pause_events": _sql(conn, "SELECT count(*) FROM provider_event "
                                       "WHERE campaign_id=%s AND state='paused'", cid)[0],
            # Recent, not cumulative. "The cause is not going away" is a statement about
            # now; counting every pause the campaign ever had meant one resolved incident
            # left the check failing for the rest of the run, which is how a person learns
            # to ignore it and misses the next real one.
            "pauses_recent": _sql(conn, "SELECT count(*) FROM provider_event "
                                        "WHERE campaign_id=%s AND state='paused' "
                                        "AND created_at > now() - interval '60 minutes'", cid)[0],
            "first_attempt_at": _sql(conn, "SELECT min(a.created_at) FROM task t "
                                           "JOIN attempt a ON a.task_id=t.id "
                                           "WHERE t.campaign_id=%s", cid)[0],
        }
        data["lost"] = data["ambiguous"] + data["failed"]
        data["settled"] = data["complete"] + data["lost"]

        responses = _sql(conn, """
            SELECT count(*),
                   count(*) FILTER (WHERE s.http_status>=400)::float/NULLIF(count(*),0),
                   count(*) FILTER (WHERE s.finish_reason='length')::float/NULLIF(count(*),0),
                   avg(s.latency_seconds), max(s.latency_seconds)
            FROM task t JOIN attempt a ON a.task_id=t.id JOIN response s ON s.attempt_id=a.id
            WHERE t.campaign_id=%s""", cid)
        data.update(responses=responses[0], http_error_rate=responses[1] or 0.0,
                    truncation_rate=responses[2] or 0.0,
                    latency_mean=float(responses[3] or 0), latency_max=float(responses[4] or 0))

        validations = _sql(conn, """
            SELECT count(*), avg(CASE WHEN v.valid THEN 1.0 ELSE 0.0 END)
            FROM task t JOIN attempt a ON a.task_id=t.id JOIN validation_event v ON v.attempt_id=a.id
            WHERE t.campaign_id=%s""", cid)
        data.update(validations=validations[0], valid_rate=float(validations[1] or 0))

        predictions = _sql(conn, """
            SELECT count(*), avg(CASE WHEN p.terminal_invalid THEN 1.0 ELSE 0.0 END),
                   avg(CASE WHEN p.canonical->>'hasProperty' <> '' THEN 1.0 ELSE 0.0 END),
                   avg(CASE WHEN p.canonical->>'hasObjectOfInterest' <> '' THEN 1.0 ELSE 0.0 END)
            FROM task t JOIN prediction p ON p.task_id=t.id WHERE t.campaign_id=%s""", cid)
        data.update(predictions=predictions[0], empty_rate=float(predictions[1] or 0),
                    property_filled=float(predictions[2] or 0),
                    object_filled=float(predictions[3] or 0))

        scores = _sql(conn, """
            SELECT count(*), avg((e.evidence->'close'->'metrics'->'f1'->>'value')::float),
                   min((e.evidence->'close'->'metrics'->'f1'->>'value')::float),
                   max((e.evidence->'close'->'metrics'->'f1'->>'value')::float),
                   count(*) FILTER (WHERE (e.evidence->'close'->'metrics'->'f1'->>'value')::float=0)
                       ::float/NULLIF(count(*),0)
            FROM task t JOIN evaluation_item e ON e.task_id=t.id WHERE t.campaign_id=%s""", cid)
        data.update(scored=scores[0], f1_mean=float(scores[1] or 0), f1_min=float(scores[2] or 0),
                    f1_max=float(scores[3] or 0), zero_f1_share=float(scores[4] or 0))

        data["per_model"] = {
            model: {"complete": complete, "valid": float(valid or 0), "responses": responses_n}
            for model, complete, valid, responses_n in conn.execute("""
                SELECT r.model_id,
                       count(DISTINCT t.id) FILTER (WHERE t.state='complete'),
                       -- FILTER, not CASE: with a LEFT JOIN, `CASE WHEN v.valid` maps a
                       -- NULL from an unattempted task to 0.0, so thousands of queued
                       -- tasks counted as invalid answers and every model looked broken.
                       avg(CASE WHEN v.valid THEN 1.0 ELSE 0.0 END) FILTER (WHERE v.id IS NOT NULL),
                       count(v.id)
                FROM task t JOIN resolved_run r ON r.id=t.run_id
                LEFT JOIN attempt a ON a.task_id=t.id
                LEFT JOIN validation_event v ON v.attempt_id=a.id
                WHERE t.campaign_id=%s GROUP BY r.model_id""", (cid,)).fetchall()}

    data["checked_at"] = datetime.now(UTC).isoformat()
    data["failure_streak"] = _failure_streak()
    data["supervisor_alive"] = _supervisor_alive()
    data["previous"] = _load_previous(cid)
    yield data
    _save(data)


def _supervisor_alive() -> bool:
    try:
        pid = int(LOCK_PID.read_text().strip())
    except (OSError, ValueError):
        return False
    return subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0


def _failure_streak() -> int:
    """Consecutive supervisor failures, read from the supervisor's own log."""
    try:
        lines = LOG.read_text().splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        if "consecutive failures:" in line:
            return int(line.rsplit("consecutive failures:", 1)[1].strip(" )."))
        if "Attempt" in line and "exited" not in line:
            return 0
    return 0


def _load_previous(cid: str) -> dict | None:
    """Return the last snapshot, but only if it describes this same campaign."""
    try:
        previous = json.loads(STATE.read_text())
    except (OSError, ValueError):
        return None
    return previous if previous.get("campaign_id") == cid else None


def _save(data: dict) -> None:
    """Keep a baseline that is old enough to measure progress against.

    Overwriting on every invocation meant an ad-hoc run reset the baseline to "now", so
    the next scheduled run had nothing meaningful to compare with and the progress check
    silently stopped doing its job.
    """
    if os.environ.get("IADOPT_HEALTH_CAMPAIGN"):
        return  # a pinned diagnostic run must not disturb the live campaign's files
    current = {key: data[key] for key in
               ("campaign_id", "checked_at", "settled", "complete", "lost", "spent",
                "responses", "total", "cap", "valid_rate", "f1_mean", "leases")}
    # Carry the baseline this run was compared against. The summary used to re-read both
    # files from disk, but by then STATE had already been refreshed to this same instant,
    # so the delta was always zero and the ETA never appeared.
    previous_snapshot = data.get("previous")
    if previous_snapshot:
        current["prev_checked_at"] = previous_snapshot["checked_at"]
        current["prev_settled"] = previous_snapshot["settled"]
    if data.get("first_attempt_at"):
        current["first_attempt_at"] = data["first_attempt_at"].isoformat()
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(current, indent=2))
    previous = data.get("previous")
    if previous is not None:
        age = (datetime.now(UTC) - datetime.fromisoformat(previous["checked_at"])).total_seconds()
        if age < 300:
            return
    keep = {key: data[key] for key in
            ("campaign_id", "checked_at", "settled", "complete", "lost", "spent", "responses")}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(keep, indent=2))
