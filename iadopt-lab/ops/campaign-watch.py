"""Emit one line per noteworthy campaign event; exit when there is nothing left to watch.

Written for an unattended multi-hour run: it stays quiet through normal progress except
at ten-percent milestones, and speaks up for the things worth interrupting someone over
- a paused provider, a jump in permanently lost tasks, the supervisor dying, or the
campaign finishing. Silence therefore means "running normally", and every way the run
can end produces a line, so silence can never be mistaken for a stalled job.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402

from iadopt_lab.local_database import local_dsn  # noqa: E402

POLL_SECONDS = 120
LOST = ("ambiguous_delivery", "operational_failed")
LOCK_PID = ROOT / ".runtime/campaign/supervisor.lock/pid"
LOG = ROOT / ".runtime/campaign/supervisor.log"


def say(message: str) -> None:
    print(f"[{datetime.now():%H:%M}] {message}", flush=True)


def supervisor_alive() -> bool:
    try:
        pid = int(LOCK_PID.read_text().strip())
    except (OSError, ValueError):
        return False
    return subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0


def supervisor_failures() -> int:
    """Read the supervisor's current consecutive-failure streak from its own log."""
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


def snapshot(conn, cid: str) -> dict:
    states = dict(conn.execute(
        "SELECT state,count(*) FROM task WHERE campaign_id=%s GROUP BY state", (cid,)
    ).fetchall())
    campaign = conn.execute(
        "SELECT state,spent_cost,maximum_cost FROM campaign WHERE id=%s", (cid,)
    ).fetchone()
    provider = conn.execute(
        "SELECT state FROM campaign_provider WHERE campaign_id=%s", (cid,)
    ).fetchone()[0]
    total = sum(states.values())
    return {"complete": states.get("complete", 0),
            "lost": sum(states.get(name, 0) for name in LOST),
            "total": total, "state": campaign[0], "spent": float(campaign[1] or 0),
            "cap": campaign[2], "provider": provider}


def main() -> int:
    dsn = local_dsn(ROOT, role="app")
    with psycopg.connect(dsn) as conn:
        conn.read_only = True
        conn.execute("SET search_path TO iadopt_lab, public")
        cid = str(conn.execute(
            "SELECT id FROM campaign WHERE mode='live' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()[0])
    say(f"watching campaign {cid[:8]}; milestones every 10%, plus anything that needs you")

    milestone, last_lost, warned_paused, started = 0, None, False, time.monotonic()
    # Liveness is not progress. The first version of this watcher only asked whether the
    # supervisor process existed, so it stayed silent for two hours while that process
    # retried a deterministically failing campaign ten times. Stalls are now detected by
    # the settled count standing still, which catches every cause rather than the ones
    # anticipated here, and the supervisor's own failure streak is read from its log.
    last_settled, stalled_polls, warned_failures = -1, 0, 0
    while True:
        try:
            with psycopg.connect(dsn, connect_timeout=10) as conn:
                conn.read_only = True
                conn.execute("SET search_path TO iadopt_lab, public")
                now = snapshot(conn, cid)
        except Exception as error:  # noqa: BLE001 - a database blip must not end the watch
            say(f"WARNING: cannot read the database ({type(error).__name__}); retrying")
            time.sleep(POLL_SECONDS)
            continue

        settled = now["complete"] + now["lost"]
        pct = 100 * settled / now["total"] if now["total"] else 0
        elapsed = (time.monotonic() - started) / 3600

        if last_lost is not None and now["lost"] > last_lost:
            say(f"WARNING: permanently lost tasks rose to {now['lost']} "
                f"(+{now['lost'] - last_lost}); these are never retried")
        last_lost = now["lost"]

        streak = supervisor_failures()
        if streak >= 3 and streak > warned_failures:
            warned_failures = streak
            say(f"WARNING: {streak} consecutive supervisor failures. Backoff is now up to "
                f"30 min. Read .runtime/campaign/attempt-*.log for the reason.")
        elif streak == 0:
            warned_failures = 0

        if settled == last_settled:
            stalled_polls += 1
            if stalled_polls in (3, 15) or (stalled_polls > 15 and stalled_polls % 15 == 0):
                say(f"WARNING: no progress for {stalled_polls * POLL_SECONDS // 60} min, "
                    f"stuck at {settled:,}/{now['total']:,}. Supervisor "
                    f"{'alive' if supervisor_alive() else 'GONE'}, provider {now['provider']}.")
        else:
            if stalled_polls >= 3:
                say(f"recovered: progress resumed at {settled:,}/{now['total']:,}")
            stalled_polls = 0
        last_settled = settled

        if now["provider"] in {"paused", "failed"} and not warned_paused:
            say(f"WARNING: provider is {now['provider']}. The supervisor should clear this "
                "on its next resume; no action needed unless it repeats.")
            warned_paused = True
        elif now["provider"] == "ready":
            warned_paused = False

        if now["state"] == "complete":
            say(f"FINISHED. {now['complete']:,} complete, {now['lost']} lost, "
                f"${now['spent']:.2f} of ${now['cap']} spent, {elapsed:.1f}h elapsed.")
            return 0

        if not supervisor_alive():
            tail = ""
            try:
                lines = [ln for ln in LOG.read_text().splitlines() if ln[:2].isdigit()]
                tail = lines[-1] if lines else ""
            except OSError:
                pass
            say(f"SUPERVISOR STOPPED at {settled:,}/{now['total']:,} "
                f"({pct:.1f}%), campaign state={now['state']}. Last log line: {tail}")
            return 1

        if pct >= milestone + 10:
            milestone = int(pct // 10) * 10
            rate = settled / elapsed if elapsed > 0.05 else 0
            left = (now["total"] - settled) / rate if rate else 0
            say(f"{milestone}% - {settled:,}/{now['total']:,} settled, {now['lost']} lost, "
                f"${now['spent']:.2f} spent"
                + (f", ~{left:.1f}h remaining" if left else ""))

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
