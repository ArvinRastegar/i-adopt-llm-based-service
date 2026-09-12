# Live campaign health checks

Twenty assertions that a campaign running *right now* is behaving the way the two
finished ones did. Read-only: they open a read-only transaction and never touch the
running campaign.

```bash
ops/healthcheck.sh                       # one run, one summary line, exit 1 on failure
ops/healthcheck-loop.sh                  # every 10 min until the campaign finishes
IADOPT_HEALTH_CAMPAIGN=844df00e ops/healthcheck.sh   # judge a past campaign instead
```

They live outside `testpaths` (`tests/`), so a normal `pytest` run never collects them.

## Why these thresholds

Every threshold comes from stored evidence of `5cdc9417` (PSNC, 10,476 tasks) and
`844df00e` (OpenRouter, 10,332 tasks), and each sits well outside the observed range.
A model scoring badly is a *result* this experiment exists to measure; a campaign that
has stopped scoring at all is an incident. These detect the second.

| Signal | 5cdc9417 | 844df00e | Fires at |
|---|---|---|---|
| Valid-JSON rate | 0.953 | 0.820 | < 0.50 |
| Per-model valid rate | — | — | < 0.05 |
| Zero-F1 share | 0.375 | 0.470 | > 0.90 |
| Empty predictions | 0.5% | 4.5% | > 25% |
| HTTP ≥ 400 | 0.0% | 4.6% | > 20% |
| Truncation | 0% | 0% | > 2% |
| Mean latency | 2.1s | 3.2s | > 60s (timeout is 120s) |
| hasProperty / hasObjectOfInterest filled | .985/.951 | .948/.936 | < 0.70 |
| Cost per task | free | $0.000197 | > $0.0040 |
| Permanent losses | 0% | 1.37% | > 1% |

## The checks

**Liveness (1-5)** supervisor alive; work actually leased; settled count advanced since
the last check; throughput in a workable band; no supervisor failure streak.

**Provider (6-9)** not paused; pauses not recurring; HTTP error rate; latency not
collapsing toward the timeout.

**Outputs (10-15)** valid-JSON rate; every model still usable; truncation negligible;
the two universally-present gold fields being filled; empty predictions rare; scores in
range and not degenerate.

**Cost (16-18)** spend under cap; cost per task matches the estimate; projected total
under cap.

**Integrity (19-20)** permanent losses under the rankability line; every model
progressing.

## Validated against known failures

A check nobody has seen fail is not evidence of anything. Pinned against real campaigns:

| Campaign | What was wrong | Result |
|---|---|---|
| `5cdc9417` | nothing | all pass |
| `844df00e` | 144 scattered losses cost 33 of 36 configurations | **losses check fires** (1.37%) |
| `fb58adbb` | stalled two hours on the whitespace deadlock | **liveness + losses fire** |
| `c4c118d7` | provider paused by one truncation (D-046) | **provider + liveness fire** |

## Notes

- `health-state.json` is the comparison baseline and is deliberately held back at least
  five minutes; `health-latest.json` is the current reading and is what the summary
  line reports. Keeping one file for both made every notification lag.
- Checks skip rather than fail while a sample is under 60, so a fresh campaign does not
  alarm anyone in its first minute.
- A pinned (`IADOPT_HEALTH_CAMPAIGN`) run never writes either file.
