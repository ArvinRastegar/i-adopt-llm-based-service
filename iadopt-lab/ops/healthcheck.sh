#!/bin/bash
# Run the twenty live-campaign health checks and print one compact summary line.
# Read-only: it opens a read-only transaction and never touches the running campaign.
# Exit 0 = all good, 1 = at least one check failed.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

OUT=$(.venv/bin/python -m pytest ops/healthcheck -q --no-header -p no:cacheprovider 2>&1)
CODE=$?

# The summary is derived from the same snapshot the checks used, so the numbers in the
# notification always agree with the verdict beside them.
SUMMARY=$(.venv/bin/python ops/healthcheck-summary.py)

if (( CODE == 0 )); then
  # BSD sed has no \+ in a basic regex, so grep -oE does the counting.
  PASSED=$(grep -oE '[0-9]+ passed' <<<"$OUT" | grep -oE '[0-9]+' | tail -1)
  SKIPPED=$(grep -oE '[0-9]+ skipped' <<<"$OUT" | grep -oE '[0-9]+' | tail -1)
  echo "OK  $SUMMARY  [${PASSED:-0} checks passed${SKIPPED:+, ${SKIPPED} skipped}]"
else
  echo "FAIL  $SUMMARY"
  # Only the assertion messages, which are written to be actionable on their own.
  grep -E "^(FAILED|E  *assert|E  *[A-Za-z].*)" <<<"$OUT" | head -20
fi
exit $CODE
