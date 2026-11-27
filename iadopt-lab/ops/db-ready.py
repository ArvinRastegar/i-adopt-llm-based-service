"""Exit 0 when the lab database accepts a connection. Read-only; writes nothing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import psycopg

    from iadopt_lab.local_database import local_dsn

    root = Path(__file__).resolve().parents[1]
    with psycopg.connect(local_dsn(root, role="app"), connect_timeout=5) as conn:
        conn.execute("SELECT 1")
except Exception as exc:  # noqa: BLE001 - any failure means "not ready yet"
    print(f"database not ready: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1) from None
raise SystemExit(0)
