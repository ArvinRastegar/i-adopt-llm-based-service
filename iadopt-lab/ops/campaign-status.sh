#!/bin/bash
# Read-only status: supervisor liveness, campaign progress, recent log tail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/.runtime/campaign"

if [[ -f "$STATE/supervisor.lock/pid" ]] && kill -0 "$(cat "$STATE/supervisor.lock/pid")" 2>/dev/null; then
  pid=$(cat "$STATE/supervisor.lock/pid")
  echo "Supervisor: RUNNING (pid $pid, up $(ps -o etime= -p "$pid" | tr -d ' '))"
else
  echo "Supervisor: not running"
fi
echo "Sleep guard: $(pmset -g assertions 2>/dev/null | grep -c 'PreventUserIdleSystemSleep.*1') active assertion(s)"
echo "Power:       $(pmset -g batt 2>/dev/null | head -1 | sed "s/Now drawing from //;s/'//g")"

"$ROOT/.venv/bin/python" "$ROOT/ops/campaign-status.py" || echo "(database unreachable)"

if [[ -f "$STATE/supervisor.log" ]]; then
  echo
  echo "=== last 12 supervisor lines ==="
  grep -E '^[0-9]{4}-' "$STATE/supervisor.log" | tail -12
fi
