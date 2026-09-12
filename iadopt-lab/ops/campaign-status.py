"""Read-only progress report for the live campaign. Safe to run at any time.

Opens a read-only transaction, prints task states, permanent losses, spend against the
cap and a throughput-based estimate. It writes nothing and takes no locks that could
interfere with a running campaign.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402

from iadopt_lab.local_database import local_dsn  # noqa: E402

# States a task can still leave. Everything else is finished, for better or worse.
ACTIVE = ("queued", "retry_pending", "request_persisted", "response_stored",
          "validated", "prediction_ready", "dispatch_started")
# Finished but unusable. These are never retried: the request may already have been
# billed, so a second call is refused. Each one can cost a whole configuration,
# because ranking requires all 97 variables of a population.
LOST = ("ambiguous_delivery", "operational_failed")


def _bar(done: int, total: int, width: int = 44) -> str:
    filled = 0 if not total else round(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main() -> int:
    with psycopg.connect(local_dsn(ROOT, role="app"), connect_timeout=5) as conn:
        conn.read_only = True
        conn.execute("SET search_path TO iadopt_lab, public")
        leased = conn.execute(
            "SELECT count(*) FROM task WHERE lease_expires_at > clock_timestamp()"
        ).fetchone()[0]
        print(f"Active task leases: {leased}"
              f"{'  (nothing is executing)' if not leased else ''}")
        campaigns = conn.execute(
            "SELECT id,state,spent_cost,maximum_cost,currency,updated_at "
            "FROM campaign WHERE mode='live' ORDER BY updated_at DESC"
        ).fetchall()
        if not campaigns:
            print("No live campaign has been registered yet.")
            return 0

        current, history = campaigns[0], campaigns[1:]
        _report(conn, current, leased, detailed=True)
        if history:
            print("\nEarlier campaigns (none of these are executing):")
            for campaign in history:
                _report(conn, campaign, leased, detailed=False)
    return 0


def _report(conn, campaign, leased: int, *, detailed: bool) -> None:
    """Print one campaign, distinguishing a live run from a stale database label."""
    cid, state, spent, cap, cur, updated = campaign
    cid = str(cid)
    states = dict(conn.execute(
        "SELECT state,count(*) FROM task WHERE campaign_id=%s GROUP BY state", (cid,)
    ).fetchall())
    total = sum(states.values())
    if not total:
        return
    complete = states.get("complete", 0)
    lost = sum(states.get(name, 0) for name in LOST)
    active = sum(states.get(name, 0) for name in ACTIVE)
    # `state` is a column, not a process. A run killed abruptly never gets to write its
    # final state, so the row keeps saying `running` forever. Only a live lease proves
    # that something is actually executing.
    label = state
    if state == "running" and not leased:
        age = datetime.now(UTC) - updated
        label = f"running (STALE - no process since {updated:%Y-%m-%d %H:%M}, {age.days}d ago)"

    if not detailed:
        print(f"  {cid[:8]}  {complete:>6}/{total:<6} done  {lost:>4} lost  "
              f"{float(spent or 0):>7.4f} {cur}  {label}")
        return

    print(f"\n=== CURRENT: campaign {cid[:8]}  state={label} ===")
    print(f"{_bar(complete + lost, total)} {complete + lost}/{total} "
          f"({100 * (complete + lost) / total:.1f}% settled)")
    print(f"  complete   {complete:>6}")
    print(f"  remaining  {active:>6}")
    if lost:
        print(f"  LOST       {lost:>6}   " + ", ".join(
            f"{name}={states[name]}" for name in LOST if states.get(name)))
    pct = f" ({100 * float(spent or 0) / float(cap):.1f}% of cap)" if cap else ""
    print(f"  spend      {float(spent or 0):.4f} {cur} of {cap} {cur}{pct}")

    for provider, pstate, until in conn.execute(
        "SELECT provider,state,cooldown_until FROM campaign_provider WHERE campaign_id=%s", (cid,)
    ):
        # A cooldown is the healthy back-off path and clears itself; only a pause or a
        # failure holds the queue open-endedly, and since D-046 even that is released by
        # the next resume. Flagging a cooldown as blocking reads as an outage when it is
        # ordinary throttling.
        blocking = pstate in {"paused", "failed"}
        flag = "   <-- blocks every queued task until the next resume" if blocking else ""
        # Stored as UTC; every other time in this report is local, so convert rather
        # than print two clocks side by side. An elapsed cooldown is shown as expired.
        when = ""
        if until:
            local = until.astimezone()
            when = (f" until {local:%H:%M:%S}" if local > datetime.now().astimezone()
                    else f" (last cooldown expired {local:%H:%M:%S})")
        print(f"  provider   {provider}: {pstate}{when}{flag}")

    if not active:
        return
    # Rate over the window the campaign has actually been running, not a fixed hour: a
    # job two minutes old has almost no attempts inside the last hour, and dividing by
    # that produced a 232-hour estimate for a run that was going fine.
    row = conn.execute(
        "SELECT count(*), min(a.created_at) FROM attempt a JOIN task t ON t.id=a.task_id "
        "WHERE t.campaign_id=%s AND a.created_at > now() - interval '1 hour'", (cid,)
    ).fetchone()
    seen, since = row[0], row[1]
    minutes = (datetime.now(UTC) - since).total_seconds() / 60 if since else 0
    if not (seen and minutes >= 1 and leased):
        print("  throughput not measurable yet" if leased else "  throughput  -  not running")
        return
    per_hour = seen / minutes * 60
    hours = active / per_hour
    warming = "  (window: %.0f min so far)" % minutes if minutes < 20 else ""
    done_at = datetime.now(UTC).timestamp() + hours * 3600
    print(f"  throughput {per_hour:,.0f} attempts/hour{warming}")
    print(f"  ETA        ~{hours:.1f}h left, finishing about "
          f"{datetime.fromtimestamp(done_at):%a %H:%M}")


if __name__ == "__main__":
    raise SystemExit(main())
