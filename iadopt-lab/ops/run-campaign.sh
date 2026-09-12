#!/bin/bash
# Unattended supervisor for the official OpenRouter campaign.
#
# Why this exists: a task that is in flight when its process dies is recorded as
# `ambiguous_delivery`, which is terminal and never retried, because the request may
# already have been billed. Loss is therefore permanent, and because ranking needs a
# complete 97-variable population, scattered losses destroy whole configurations. This
# database already holds 44 tasks lost that way to earlier interruptions. The job here
# is to make abrupt death rare, not merely survivable.
#
# Usage:  ops/run-campaign.sh            run or resume until complete
#         ops/run-campaign.sh --status   read-only progress, safe at any time
#         ops/run-campaign.sh --stop     stop gracefully (costs the in-flight tasks)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
STATE="$ROOT/.runtime/campaign"
LOCK="$STATE/supervisor.lock"
STOPFILE="$STATE/stop-requested"
AUTHORIZED="$STATE/authorized"
LOG="$STATE/supervisor.log"

ESTIMATE="${IADOPT_ESTIMATE:-outputs/estimate-final.json}"
ENV_FILE="${IADOPT_ENV_FILE:-$ROOT/../.env}"
ACTOR="${IADOPT_ACTOR:-$(git -C "$ROOT" config user.name 2>/dev/null || echo "$USER")}"
# Entry point, overridable only so the supervisor loop can be tested against a stub.
CLI="${IADOPT_CLI:-ops/campaign-entry.py}"
# Overridable so the loop logic can be exercised offline without waiting.
read -r -a BACKOFFS <<< "${IADOPT_BACKOFFS:-30 60 120 300 600 900 1800}"
MAX_DB_WAIT=1800
# How long the campaign is given to store answers it has already received before the
# supervisor stops waiting. Cancellation itself is fast; this only covers the shielded
# database commits that outlive it.
DRAIN_TIMEOUT=180

mkdir -p "$STATE"
log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
notify() {
  osascript -e "display notification \"$2\" with title \"I-ADOPT campaign\" subtitle \"$1\"" \
    >/dev/null 2>&1 || true
}

case "${1:-}" in
  --status) exec "$ROOT/ops/campaign-status.sh" ;;
  --stop)
    if [[ -f "$LOCK/pid" ]] && kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
      pid=$(cat "$LOCK/pid")
      touch "$STOPFILE"
      kill -TERM "$pid" 2>/dev/null
      echo "Stopping supervisor $pid gracefully."
      echo "It forwards SIGINT to the campaign so answers already received are stored."
      echo "Tasks still awaiting a reply become ambiguous_delivery and are never retried."
      echo "Watch it finish with:  ops/campaign-status.sh"
    else
      echo "No supervisor is running."
    fi
    exit 0 ;;
esac

# ---------------------------------------------------- keep the machine awake, once
# -d display, -i idle sleep, -m disk sleep, -s system sleep (honoured on AC power).
# Re-exec so both the terminal path and the launchd path are covered.
if [[ -z "${IADOPT_CAFFEINATED:-}" ]]; then
  export IADOPT_CAFFEINATED=1
  exec /usr/bin/caffeinate -dims "${BASH_SOURCE[0]}" "$@"
fi

# ------------------------------------------------------------------ single instance
if ! mkdir "$LOCK" 2>/dev/null; then
  if [[ -f "$LOCK/pid" ]] && kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
    echo "A supervisor is already running (pid $(cat "$LOCK/pid")). Use --status." >&2
    exit 3
  fi
  log "Removing stale lock from pid $(cat "$LOCK/pid" 2>/dev/null || echo unknown)"
  rm -rf "$LOCK" && mkdir "$LOCK" || { echo "Cannot acquire lock" >&2; exit 3; }
fi
echo $$ > "$LOCK/pid"
rm -f "$STOPFILE"

CHILD=""
trap 'rm -rf "$LOCK"' EXIT
# A background job inherits SIGINT set to SIG_IGN, so neither this script nor the
# campaign can act on a forwarded Ctrl-C; that is why TERM is trapped and TERM is what
# gets forwarded. `ops/campaign-entry.py` turns SIGTERM into the KeyboardInterrupt the
# runner already handles, so its `finally` blocks run and answers already received are
# still committed. Sending SIGTERM to the interpreter directly would skip all of that.
forward() {
  touch "$STOPFILE"
  if [[ -z "$CHILD" ]] || ! kill -0 "$CHILD" 2>/dev/null; then
    log "Stop signal received while idle; exiting now."
    exit 0
  fi
  log "Stop signal received; asking the campaign to store what it has."
  if true; then
    kill -TERM "$CHILD" 2>/dev/null
    for _ in $(seq "$DRAIN_TIMEOUT"); do
      kill -0 "$CHILD" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$CHILD" 2>/dev/null; then
      log "Campaign still draining after ${DRAIN_TIMEOUT}s; leaving it to finish."
    else
      log "Campaign drained cleanly."
    fi
  fi
}
trap forward INT TERM

if [[ ! -x "$PY" ]]; then log "FATAL: no virtualenv at $PY"; exit 2; fi
if [[ ! -f "$ROOT/$ESTIMATE" ]]; then log "FATAL: estimate not found: $ROOT/$ESTIMATE"; exit 2; fi
if [[ ! -f "$ENV_FILE" ]]; then log "FATAL: credential file not found: $ENV_FILE"; exit 2; fi
if [[ -z "$ACTOR" ]]; then log "FATAL: set IADOPT_ACTOR to the person authorizing this run"; exit 2; fi

wait_for_database() {
  local waited=0
  if ! docker info >/dev/null 2>&1; then
    log "Docker is not responding; starting Docker Desktop."
    open -a Docker >/dev/null 2>&1 || true
  fi
  until "$PY" "$ROOT/ops/db-ready.py" >/dev/null 2>&1; do
    if (( waited >= MAX_DB_WAIT )); then
      log "FATAL: database still unreachable after ${MAX_DB_WAIT}s."
      return 1
    fi
    (( waited == 0 )) && log "Waiting for Docker and PostgreSQL..."
    sleep 10
    waited=$(( waited + 10 ))
  done
  (( waited > 0 )) && log "Database ready after ${waited}s."
  return 0
}

# Conditions no retry can resolve. Looping on these would burn days and hide the cause.
FATAL_PATTERNS=(
  "planned from different inputs"
  "scientific identity mismatch"
  "frozen plan cannot change membership"
  "Campaign fingerprint conflicts"
  "exceeds the campaign cost cap"
  "BudgetError"
  "is not available for provider"
  "Invalid runtime setting allowlist"
  "No module named"
)

cd "$ROOT" || exit 2
attempt=0
failures=0
started_at=$(date +%s)
log "==================================================================="
log "Supervisor starting. root=$ROOT actor=$ACTOR estimate=$ESTIMATE"
log "Plan: 17,460 tasks / 180 configurations / 5 OpenRouter models."
log "Stop with 'ops/run-campaign.sh --stop' or Ctrl-C. Never kill -9: it skips"
log "evidence preservation and turns every in-flight task into a permanent loss."
log "==================================================================="

while true; do
  if [[ -f "$STOPFILE" ]]; then
    log "Stop requested; exiting without starting another attempt."
    exit 0
  fi
  wait_for_database || { notify "Blocked" "Database unreachable"; exit 2; }

  attempt=$(( attempt + 1 ))
  run_log="$STATE/attempt-$(printf '%04d' "$attempt")-$(date '+%Y%m%dT%H%M%S').log"

  if [[ -f "$AUTHORIZED" ]]; then
    log "Attempt $attempt: resume  (log: ${run_log##*/})"
    set -- resume --env-file "$ENV_FILE"
  else
    log "Attempt $attempt: run --authorize  (log: ${run_log##*/})"
    set -- run --authorize --estimate "$ESTIMATE" --actor "$ACTOR" --env-file "$ENV_FILE"
    # The campaign and its authorization receipt are registered before any dispatch,
    # so mark it now: a crash after that point must resume, not re-authorize.
    touch "$AUTHORIZED"
  fi

  # Started directly, not in a subshell, so $! is the interpreter and a forwarded
  # SIGINT reaches the code that preserves evidence.
  "$PY" "$CLI" --json "$@" >"$run_log" 2>&1 &
  CHILD=$!
  wait "$CHILD"
  code=$?
  # `wait` returns as soon as a trap fires, which can be before the child has finished
  # draining. Wait again for the real exit status.
  if kill -0 "$CHILD" 2>/dev/null; then wait "$CHILD"; code=$?; fi
  CHILD=""

  tail -n 40 "$run_log" >> "$LOG"

  if (( code == 0 )); then
    reason=$(grep -o '"stop_reason": *"[a-z_]*"' "$run_log" | tail -1 | sed 's/.*"\([a-z_]*\)"$/\1/')
    elapsed=$(( $(date +%s) - started_at ))
    log "Campaign finished. stop_reason=${reason:-unknown} attempts=$attempt elapsed=$(( elapsed / 3600 ))h$(( (elapsed % 3600) / 60 ))m"
    log "Review permanent losses before reporting:  ops/campaign-status.sh"
    notify "Complete" "stop_reason=${reason:-unknown} after $attempt attempt(s)"
    exit 0
  fi

  if [[ -f "$STOPFILE" ]]; then
    log "Campaign stopped on request (exit $code)."
    exit 0
  fi

  # Authorization missing means the first attempt died before recording the receipt.
  # Clear the marker so the next attempt authorizes instead of resuming a campaign
  # that cannot dispatch.
  if grep -q "Live calls require enabled configuration" "$run_log"; then
    log "Authorization receipt absent; next attempt will re-authorize."
    rm -f "$AUTHORIZED"
  else
    for pattern in "${FATAL_PATTERNS[@]}"; do
      if grep -qF "$pattern" "$run_log"; then
        log "FATAL: unrecoverable condition -- $pattern"
        log "See $run_log. Completed work is safe; fix the cause and start again."
        notify "Stopped" "Unrecoverable: $pattern"
        exit 2
      fi
    done
  fi

  failures=$(( failures + 1 ))
  idx=$(( failures - 1 )); (( idx >= ${#BACKOFFS[@]} )) && idx=$(( ${#BACKOFFS[@]} - 1 ))
  delay=${BACKOFFS[$idx]}
  log "Attempt $attempt exited $code. Retrying in ${delay}s (consecutive failures: $failures)."
  (( failures == 3 )) && notify "Retrying" "$failures consecutive failures; still trying"
  # Backgrounded and waited on, never a plain `sleep`: bash defers a trap until the
  # current foreground command returns, so a plain sleep swallowed --stop for the whole
  # backoff - up to half an hour of a supervisor that had already been told to quit.
  # `wait` is the one builtin a trapped signal interrupts immediately.
  sleep "$delay" & NAP=$!
  wait "$NAP" 2>/dev/null
  [[ -f "$STOPFILE" ]] && { log "Stop requested during backoff; exiting."; exit 0; }
done
