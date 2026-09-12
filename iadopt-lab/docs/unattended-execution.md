# Running the official campaign unattended

The remaining official work is one OpenRouter campaign: **17,460 tasks, 180
configurations, 5 models**, expected cost **$13.71** against a **$40** cap. At the
measured throughput it runs for roughly **12–36 hours**, so it has to survive a night,
a lunch break and probably a commute.

This document explains what can go wrong, why one particular failure is worse than it
looks, and how to launch the job so it does not happen.

## Why not just run it from the Claude session

Right — don't. A process started inside an assistant session is tied to that session's
lifetime and to a tool call that can time out, be backgrounded, or be summarised away.
That already went wrong once here: a campaign was reported as stopped when it was in
fact still running, because the recorded PID belonged to the wrong process.

A separate terminal is the correct instinct, but on its own it is not enough — closing
the terminal or quitting VS Code sends `SIGHUP`, and the machine sleeping breaks every
open connection.

## The failure that actually matters

Most interruptions here are harmless. One is not.

When a request has been sent and no reply has arrived yet, the task is recorded as
`ambiguous_delivery`. That state is **terminal and never retried**, deliberately: the
provider may already have billed the call, so the runner refuses to pay for it twice.

```
src/iadopt_lab/workflow.py
    except asyncio.CancelledError:
        # Cancellation after the write-ahead marker may have reached the provider.
        await asyncio.shield(_db(services, "store_response", attempt, {...
            "delivery": "ambiguous_delivery", "outcome": "interrupted_after_dispatch"}))
```

`claim_tasks` will not pick such a task up again, so **the loss is permanent**, and it
is disproportionately expensive: `ranking.require_complete_population` needs all 97
variables of a configuration, so a handful of scattered losses can invalidate whole
configurations. This is measured, not theoretical — in campaign `844df00e`, 144 failures
spread thinly across `qwen/qwen3-8b` left only **3 of its 36 configurations rankable**.
Two abandoned campaigns in this database still carry 20 and 24 `ambiguous_delivery`
tasks from exactly this cause.

With `worker_count: 8`, **every abrupt death costs up to 8 tasks.** The goal is therefore
to minimise the *number* of interruptions, not merely to make restarting possible.

### What is and is not dangerous

| Event | Outcome | Cost |
|---|---|---|
| Wi-Fi drops *before* a request goes out | `ConnectError` → transient → retried | none |
| Provider 429 / 5xx | transient → retried, up to 3 attempts | none |
| Wi-Fi drops *after* a request goes out | `ambiguous_delivery` | up to 8 tasks |
| Machine sleeps | every open socket dies | up to 8 tasks |
| Terminal or VS Code closed | `SIGHUP` | up to 8 tasks |
| `kill -9`, power loss, panic | no cleanup at all | up to 8 tasks |
| `kill` (SIGTERM) sent to the campaign directly | Python skips `finally` | up to 8 tasks |
| Graceful stop via `--stop` | answers already received are committed | up to 8 tasks |

Note the last row: even a clean stop costs the in-flight tasks. There is no lossless
pause. **Start it once and let it finish.**

## How to run it

```bash
cd iadopt-lab
ops/run-campaign.sh
```

That is the whole command. It holds the terminal; leave it open. It:

1. re-executes itself under `caffeinate -dims`, so the machine will not sleep;
2. takes a single-instance lock, so a second copy cannot double-run;
3. starts Docker Desktop if needed and waits for PostgreSQL (up to 30 min);
4. runs `run --authorize` the first time, then `resume` for every later attempt —
   `resume` is idempotent by design and never repeats completed work;
5. restarts with backoff (30s → 30 min) after a recoverable failure, indefinitely;
6. **stops immediately** on conditions no retry can fix — plan drift, a changed
   population, a budget-cap breach, missing credentials — and says which;
7. exits 0 when the campaign reports `tasks_complete`, `tasks_terminal` or
   `already_complete`, and posts a macOS notification.

Logs: `.runtime/campaign/supervisor.log` plus one file per attempt.

### Surviving reboot and logout as well

The terminal command above survives a closed terminal only if you also detach it. For
full unattended operation across a reboot, install the LaunchAgent instead:

```bash
cp iadopt-lab/ops/com.iadopt.campaign.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.iadopt.campaign.plist
```

`RunAtLoad` starts it now and again at every login; `KeepAlive`/`SuccessfulExit=false`
restarts it only when it exits non-zero, so it stops by itself once the campaign is
finished. To remove it:

```bash
launchctl unload ~/Library/LaunchAgents/com.iadopt.campaign.plist
rm ~/Library/LaunchAgents/com.iadopt.campaign.plist
```

## Watching it

Safe to run at any time, from any terminal; it opens a read-only transaction:

```bash
iadopt-lab/ops/campaign-status.sh
```

It reports supervisor liveness, the sleep assertion, whether you are on battery, task
states, **permanent losses**, spend against the cap, and an ETA from the last hour's
throughput.

## Stopping it

```bash
iadopt-lab/ops/run-campaign.sh --stop
```

This is the only correct way. It signals the supervisor, which forwards `SIGTERM` to
`ops/campaign-entry.py`; that shim converts it into the `KeyboardInterrupt` the runner
already handles, so shielded evidence commits complete before exit. Restarting later
picks up exactly where it left off.

Never `kill -9`, and never signal the Python process directly — `main.py` does not
install the shim's handlers, so a plain `kill` skips every `finally` block.

## Before you launch: the three things that actually bite

1. **Mains power, lid open.** `caffeinate -s` only holds off system sleep on AC power,
   and nothing in userspace can prevent lid-close sleep. `campaign-status.sh` prints
   the current power source; at the time of writing this machine was on battery.
   For lid-closed operation you would need `sudo pmset -c disablesleep 1` (and
   `sudo pmset -c disablesleep 0` afterwards) — that is yours to run, not the script's.
2. **Wired network if you have one.** A drop after dispatch is unrecoverable; a drop
   before dispatch is free. Ethernet removes most of the difference.
3. **Docker Desktop set to start at login**, if you use the LaunchAgent. The supervisor
   will launch it and wait, but only if the user session is logged in.

## If it stops early

Check the provider line in `campaign-status.sh` first. A provider that is not `ready`
blocks every queued task, and the status output flags it. Since D-046 a resume releases
the pause automatically, so the supervisor recovers on its own; a provider that pauses
again immediately is a genuine deployment problem — read the reason in `provider_event`.

Stop reasons and what they mean:

| `stop_reason` | Meaning |
|---|---|
| `tasks_complete` | every task finished; the supervisor exits 0 |
| `tasks_terminal` | finished, with recorded operational failures; also exit 0 |
| `already_complete` | nothing left to do |
| `providers_paused` | no provider can serve work; the next resume clears it |
| `no_claimable_work` | nothing claimable and no known wait; retry |

A run that fails repeatedly in ~15 seconds is failing at startup, not in dispatch. Read
`.runtime/campaign/attempt-*.log` — the whole reason is usually its single line.

## Running the tests

`IADOPT_LAB_TEST_DATABASE_URL` in `../.env` is **empty**, which silently skipped the 15
recovery and integration tests. They need the `iadopt_lab_test` database and the
`migrator` role:

```bash
cd iadopt-lab
IADOPT_LAB_TEST_DATABASE_URL="$(.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from iadopt_lab.local_database import local_dsn
print(local_dsn(Path('.'), role='migrator').replace('dbname=iadopt_lab','dbname=iadopt_lab_test'))")" \
  .venv/bin/python -m pytest tests -q
```

With it set the suite is 403 passed, 1 skipped. Without it, 388 passed and 16 skipped —
and the skips are exactly the tests that cover crash recovery.

## Preconditions, all currently satisfied

| Check | Value |
|---|---|
| `preflight` | 0 issues |
| Plan hash matches the frozen estimate | `d16bee78…` ✓ (moved by D-046, then D-047) |
| Estimate | `outputs/estimate-final.json`, `ready: true` |
| Credentials | `OPENROUTER_API_KEY` in `../.env` (gitignored) |
| Database | `iadopt-lab-postgres-1`, healthy, `restart: unless-stopped` |
| Cap | $40 USD, against $13.71 expected |

`ops/` is outside every hashed artifact path (`src/`, `schemas/`, `prompts/`,
`migrations/`, `data/manifests/`, `uv.lock`, `pyproject.toml`, `.python-version`), so
adding this tooling leaves the plan hash and the frozen estimate valid — verified before
and after.
