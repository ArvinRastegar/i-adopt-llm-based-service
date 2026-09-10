# Campaign log

The record of campaigns actually executed, why each one exists, and what it produced.

This file exists because the evidence does not survive in Git. Ranking exports live under
`outputs/`, which `.gitignore` excludes, and the full evidence lives in PostgreSQL, which
is not in the repository either. Without this log a reader months from now can see the code
that ran campaigns but has no way to learn which campaigns ran or what they found.

Campaign identifiers are deterministic: the same frozen inputs and plan always resolve to
the same UUID. Changing `parameters.yml` or any implementation file changes the plan hash
and therefore the campaign, which is why the list below contains several short-lived
entries — each code fix during development invalidated the plan in flight.

## Completed campaigns

### `5cdc9417` — PSNC full grid (the headline result)

The main PSNC result. 36 configurations per model over the full grid, reasoning off.

| | |
|---|---|
| Provider / models | PSNC: `GLM-5.2`, `Qwen3.8-27B`, `DeepSeek-V4-Flash` |
| Grid | 3 prompts × 4 shot counts `[0,1,3,5]` × 3 temperatures `[0,0.5,1]` |
| Scope | 108 configurations × 97 variables = **10,476 tasks** |
| Outcome | 10,476 complete, **108/108 rankable**, 0 operational failures |
| Wall time / cost | 28.6 minutes at concurrency 24 / $0 (non-billed) |
| Export | `outputs/ranking-5cdc9417-….json` (gitignored) |

Best configuration: **Qwen3.8-27B, matrix-decomposition, 5 shots, T=0.5 — Close F1 0.391**.

Findings worth keeping:

- **Shot count dominates.** Mean Close F1 by shots: 0 → 0.244, 1 → 0.272, 3 → 0.339,
  5 → 0.353. The curve has not flattened at 5, so 5 is a grid boundary rather than an
  optimum. More demonstrations is the obvious next experiment, but it is blocked: the
  corpus has exactly 5 demonstrations, and `select_demonstrations` rejects any shot count
  outside `(0,1,3,5)`. Going higher means promoting evaluation variables into the
  demonstration pool, which shrinks the 97-variable population and breaks comparability
  with everything here.
- **Prompt variant matters modestly.** matrix-decomposition 0.321 mean against
  strict-minimal 0.291 and constraint-decomposition 0.295; it takes 10 of the top 12 slots.
- **Temperature barely matters** across 0–1: 0.307 / 0.305 / 0.294.
- **The three models are within noise.** Means 0.310 / 0.300 / 0.296, and the two best
  single configurations differ by 0.001. With one repetition per configuration (D-029)
  that ordering is not distinguishable. What separates them is cost, not quality:
  DeepSeek-V4-Flash averaged 0.68s per call against GLM-5.2's 3.33s for the same score.

### `d6d7da21` — PSNC three models, single configuration

Scoped precursor to the full grid: strict-minimal, 5 shots, T=0.5. 291 tasks, all complete,
97 seconds, 0 failures. Close F1 GLM-5.2 0.372 / Qwen3.8-27B 0.368 / DeepSeek-V4-Flash
0.348. Superseded by `5cdc9417`; kept because it is the clean three-model comparison at one
configuration.

### `ee138ed5` — PSNC GLM-5.2 alone

The first reasoning-off campaign, run to validate D-041. 97 tasks, Close F1 **0.387**,
against qwen3-32b's 0.316 measured earlier *with* reasoning on. This is the evidence that
disabling reasoning is not a quality trade-off.

## Interrupted campaign

### `844df00e` — OpenRouter full grid

Same grid as `5cdc9417`, on OpenRouter, metered. Stopped by the operator partway.

| | |
|---|---|
| Provider / models | OpenRouter: `openai/gpt-4o-mini`, `qwen/qwen3-32b`, `qwen/qwen3-8b` |
| Scope | 10,476 tasks planned |
| Progress at stop | 7,906 complete, 144 operational failures, ~2,400 outstanding |
| Cost | **$1.55 spent** of a $25 hard cap; projected $2.05 for the full run |
| State | resumable — `iadopt-lab resume` continues it if `parameters.yml` is unchanged |

Two things a reader needs to know about this campaign:

**No configuration finished.** Task claim order is by fingerprint (a content hash), so the
scheduler spreads work uniformly across the grid rather than completing configurations one
at a time. At the stop, per-configuration completion ranged from 55 to 80 of 97 with a
median of 68 — every one of the 108 configurations was short. Under
`ranking.require_complete_population` that means the 7,906 finished tasks currently yield
**zero rankable configurations**. The work is durable and resume picks it up, but a partial
stop here is not a partial result.

**All 144 failures are `qwen/qwen3-8b`**, rate-limited upstream by Alibaba
(`"temporarily rate-limited upstream … add your own key to accumulate your rate limits"`).
The other two models had zero failures. At a 6.8% task failure rate spread evenly, 31 of
its 36 configurations already contained at least one failure, so qwen3-8b is expected to be
largely unrankable even after a full resume. That is a provider account limit, not
something the code can fix.

Resuming requires `parameters.yml` untouched. Adding a model changes the configuration
hash, hence the plan hash, hence the campaign identity — `resume` will refuse with
`Frozen inputs resolve to campaign X, not Y` rather than silently continuing. To add models
without discarding this work, finish this campaign first and run the new models as a
separate campaign on the identical grid; per-configuration scores remain comparable because
the corpus, prompts, scorer and population are unchanged.

## Superseded and abandoned campaigns

Short-lived entries exist in the `campaign` table from development. They hold real evidence
but no complete population, and each was invalidated by a code fix that changed the plan
hash. They are retained rather than deleted because the design keeps evidence:
`3ffac9b8` (planned only), `8893ef5c` and `f8114e69` (13,968-task grids including
temperature 2.0, abandoned under D-043), `fd88566b` (10,476, superseded), `2264f8fc`
(291, killed by the price-card blocker), `843165da` and `90ed32f9` (OpenRouter, killed by
the provider-pause and idle-exit bugs), plus several early single-model and synthetic runs.

## How to inspect a campaign

Campaign state, task counts and spend, via the read-only role:

```sql
SELECT c.id, c.mode, c.state, c.spent_cost, c.currency FROM campaign c ORDER BY c.created_at;

SELECT t.state, count(*) FROM task t WHERE t.campaign_id = '<uuid>' GROUP BY 1;

SELECT rr.model_id, rr.prompt_variant, rr.shot_count, rr.temperature, t.state, count(*)
FROM task t JOIN resolved_run rr ON rr.id = t.run_id
WHERE t.campaign_id = '<uuid>' GROUP BY 1,2,3,4,5;
```

Per-call operational evidence, including latency, tokens and outcome:

```sql
SELECT a.model_id,
       count(*)                                                   AS calls,
       avg((rs.evidence->>'latency_seconds')::numeric)            AS avg_latency,
       avg((rs.evidence->'usage'->>'completion_tokens')::numeric) AS avg_out_tokens,
       count(*) FILTER (WHERE rs.delivery <> 'response_received') AS failures
FROM attempt a
JOIN task t     ON t.id = a.task_id
JOIN response rs ON rs.attempt_id = a.id
WHERE t.campaign_id = '<uuid>'
GROUP BY 1;
```

The schema is `iadopt_lab`, not `public`. Connection details and roles are in
[Local database setup](local-database.md).
