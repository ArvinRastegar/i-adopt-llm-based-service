"""One compact line describing the campaign right now, for the health-check notification.

Reads the current snapshot the checks just took, and uses the deliberately older
baseline only to derive a rate. Reporting straight from the baseline made every
notification lag by up to five minutes and understate progress.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    now = json.loads((ROOT / ".runtime/campaign/health-latest.json").read_text())
except (OSError, ValueError):
    print("no reading yet")
    raise SystemExit(0) from None

total, settled = now["total"], now["settled"]
pct = 100 * settled / total if total else 0.0

# Rate over the window this run was actually compared against, carried in the reading
# itself. Re-reading the baseline file here gave a zero-length window, because the check
# run had already refreshed it to the current instant.
eta = ""
try:
    minutes = (datetime.fromisoformat(now["checked_at"])
               - datetime.fromisoformat(now["prev_checked_at"])).total_seconds() / 60
    moved = settled - now["prev_settled"]
    if 5 <= minutes <= 45 and moved > 0:
        eta = f" eta~{(total - settled) / (moved / minutes * 60):.1f}h"
except (KeyError, ValueError, ZeroDivisionError):
    pass

print(f"{now['campaign_id'][:8]} {settled:,}/{total:,} ({pct:.1f}%) lost={now['lost']} "
      f"${now['spent']:.2f} valid={now['valid_rate']:.0%} f1={now['f1_mean']:.3f}{eta}")
