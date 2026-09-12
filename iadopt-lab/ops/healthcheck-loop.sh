#!/bin/bash
# Run the twenty health checks every ten minutes and emit one line per run.
# Ends by itself when the campaign reaches a terminal state.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${IADOPT_HEALTH_INTERVAL:-600}"

while true; do
  "$ROOT/ops/healthcheck.sh"
  read -r state settled total < <("$ROOT/.venv/bin/python" "$ROOT/ops/campaign-state.py" 2>/dev/null || echo "unknown 0 0")
  # Settled, not `state == complete`. A campaign that ends with recorded operational
  # failures finishes as `tasks_terminal` and its row stays `paused`, because not every
  # task reached `complete` - so waiting on that state never ended the watch even though
  # the run was over, its ranking written and its supervisor exited.
  if [[ "$state" == "complete" ]] || { [[ -n "${total:-}" ]] && (( total > 0 )) && (( settled >= total )); }; then
    echo "FINISHED: ${settled}/${total} settled (campaign state: ${state})."
    exit 0
  fi
  sleep "$INTERVAL"
done
