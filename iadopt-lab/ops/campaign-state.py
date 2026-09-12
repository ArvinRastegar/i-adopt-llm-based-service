"""Print `<campaign_state> <settled> <total>` for the newest live campaign. Read-only."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402

from iadopt_lab.local_database import local_dsn  # noqa: E402

with psycopg.connect(local_dsn(ROOT, role="app"), connect_timeout=10) as conn:
    conn.read_only = True
    conn.execute("SET search_path TO iadopt_lab, public")
    row = conn.execute("SELECT id,state FROM campaign WHERE mode='live' "
                       "ORDER BY created_at DESC LIMIT 1").fetchone()
    if row is None:
        print("none 0 0")
        raise SystemExit(0)
    states = dict(conn.execute(
        "SELECT state,count(*) FROM task WHERE campaign_id=%s GROUP BY state",
        (str(row[0]),)).fetchall())
    settled = (states.get("complete", 0) + states.get("ambiguous_delivery", 0)
               + states.get("operational_failed", 0))
    print(row[1], settled, sum(states.values()))
