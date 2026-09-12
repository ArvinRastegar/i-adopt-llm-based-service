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

### `844df00e` — OpenRouter full grid

The main OpenRouter result. Same grid as `5cdc9417`, metered, reasoning off. It ran to
completion; an earlier revision of this log described it as interrupted, which was wrong.

| | |
|---|---|
| Provider / models | OpenRouter: `openai/gpt-4o-mini`, `qwen/qwen3-32b`, `qwen/qwen3-8b` |
| Scope | 108 configurations x 97 variables = **10,476 tasks** |
| Outcome | 10,332 complete, 144 operational failures, **75/108 rankable** |
| Cost | **$2.03** against a $25 hard cap |
| Export | `ranking_id 7bdfd213`; JSON under `outputs/` (gitignored) |

| Model | Rankable | Mean Close F1 | Best |
|---|---|---|---|
| `qwen/qwen3-32b` | 36/36 | 0.291 | **0.371** |
| `qwen/qwen3-8b` | 3/36 | 0.347 | 0.362 |
| `openai/gpt-4o-mini` | 36/36 | 0.225 | 0.294 |

Best configuration: **qwen/qwen3-32b, constraint-decomposition, 3 shots, T=0.0 — 0.371**.

**All 144 failures are `qwen/qwen3-8b`**, rate-limited upstream by Alibaba. The other two
models had zero. `ranking.require_complete_population` needs all 97 variables, so a 6.8%
task failure rate spread evenly left only 3 of its 36 configurations rankable — its mean
above is over 3 configurations, not 36, and is not comparable with the other rows.

**This campaign only produced results because of the `tasks_terminal` fix (D-042).**
Finalization previously refused while any task was incomplete, and a non-retryable
operational failure can never reach `complete`. Without that change these 144 failures
would have discarded all 10,332 successful results.

Against PSNC on the same grid: best PSNC 0.391 versus OpenRouter 0.371, and PSNC ran free.

### `79b5ec5f` — OpenRouter continuation grid (reasoning dimension)

The final official campaign. Five OpenRouter models x 3 prompts x 4 shot counts x 3
temperatures = 180 configurations x 97 variables. Ran 8h40m and finished
`tasks_terminal`: every task reached a terminal state, 130 of them a failure.

| | |
|---|---|
| Models | `z-ai/glm-5.2`, `qwen/qwen3-32b`, `qwen/qwen3-8b`, `meta-llama/llama-3.1-8b-instruct`, `mistralai/ministral-8b-2512` |
| Scope | 180 configurations x 97 variables = **17,460 tasks** |
| Outcome | 17,330 complete, 130 lost, **115/180 rankable** |
| Cost | **$19.73** against a $40 cap |
| Export | `ranking_id c5e9333e`; plan `d16bee78` |

**Best configuration overall, across the whole experiment:**
`z-ai/glm-5.2`, matrix-decomposition, **5 shots, T=0.5, reasoning enabled — Close F1 0.4206**.
That beats the best PSNC result (0.391, `5cdc9417`) and the earlier OpenRouter grid
(0.371, `844df00e`). `glm-5.2` takes the top eight places outright.

| Model | Rankable | Mean Close F1 | Best |
|---|---|---|---|
| `z-ai/glm-5.2` | 18/36 | 0.349 | **0.4206** |
| `qwen/qwen3-32b` | 22/36 | 0.280 | 0.3618 |
| `mistralai/ministral-8b-2512` | 34/36 | 0.267 | 0.3615 |
| `qwen/qwen3-8b` | 8/36 | 0.282 | 0.3093 |
| `meta-llama/llama-3.1-8b-instruct` | 33/36 | 0.120 | 0.2226 |

Mean columns are over different numbers of configurations and are not comparable
across rows; see D-048.

**Shot count remains the strongest controlled factor**, and still has not plateaued at 5:

| Shots | 0 | 1 | 3 | 5 |
|---|---|---|---|---|
| Mean Close F1 | 0.200 | 0.195 | 0.265 | 0.295 |

That matches the shot-count ablation, which found `Qwen3.8-27B` still climbing at 10.

Prompt variant is a weak effect: matrix-decomposition 0.253, constraint-decomposition
0.237, strict-minimal 0.233.

**The reasoning contrast in this campaign is NOT a controlled comparison.** Reasoning is
`enabled` for exactly the three models that support it (`glm-5.2`, `qwen3-32b`,
`qwen3-8b`) and `not_applicable` for the two that do not (`llama-3.1-8b`,
`ministral-8b`). The 0.307 vs 0.194 gap between those groups therefore confounds
reasoning with model identity and must not be read as an effect of reasoning. The only
controlled reasoning evidence in this experiment is PSNC campaign `5cdc9417`, where the
same model ran both ways.

**Losses** (130, 0.74%) were concentrated and are explained:

| Model | Lost | Cause |
|---|---:|---|
| `qwen/qwen3-8b` | 59 | upstream HTTP 429 exhausting three attempts (D-048) |
| `z-ai/glm-5.2` | 50 | answers exceeding the 8,000-token output ceiling with reasoning on |
| `qwen/qwen3-32b` | 16 | mixed, including 6 ambiguous deliveries |
| others | 5 | the 403 outage below |

`glm-5.2`'s truncations are a distinct cause from `qwen3-8b`'s throttling and cost it
half its configurations: the model that produced the best result is also the one whose
coverage suffered most from the ceiling. Worth revisiting if the grid is ever re-run.

**One outage.** At 37.5% the OpenRouter key hit its configured monthly spend limit and
returned HTTP 403 to all five models. A 403 is non-retryable, so each one failed its task
outright; the supervisor's retry loop destroyed ~8 tasks per cycle until it was stopped.
17 tasks were lost this way. The owner raised the key limit and the campaign resumed from
37.5% with no completed work re-run. The health checks detected it within one 10-minute
interval and named the provider as the cause.

## Planned continuation campaign

All remaining work is **one OpenRouter campaign**. Everything left shares a provider,
ceiling, timeout and concurrency, so it runs as a single resumable unattended job rather
than several. See [Continuation strategy](#continuation-strategy) for why it cannot extend
the finished campaigns instead.

| Model | Reasoning | Measured out tok | Measured latency | Tasks | Expected |
|---|---|---:|---:|---:|---|
| `meta-llama/llama-3.1-8b-instruct` | `not_applicable` | 125 | 3.9s | 3,492 | $0.25 |
| `mistralai/ministral-8b-2512` | `not_applicable` | 46 | 1.5s | 3,492 | $0.67 |
| `qwen/qwen3-8b` | `enabled` | 1,187 | 29.9s | 3,492 | $2.39 |
| `qwen/qwen3-32b` | `enabled` | 1,065 | 32.4s | 3,492 | $1.38 |
| `z-ai/glm-5.2` | `enabled` | 772 | 14.0s | 3,492 | $3.81 |

**17,460 tasks.** Realistic cost **$8.26** from per-model measured usage ($9.92 with a 20%
retry allowance); the disclosed
estimate is **$13.71** because it conservatively applies 1,200 output tokens to every
model, including ones measured at 46. Cap **$40**, roughly 3x the conservative figure.

Settings: concurrency 8, timeout 120s, ceiling 8,000. Every measured output is at least
6.7x below the ceiling, and 8,000 stays under `qwen/qwen3-8b`'s published 8,192 cap.
Concurrency 8 is measured, not assumed: both reasoning-on Qwen models completed 16/16
under a 120s timeout at concurrency 8, and maximum latency *fell* from concurrency 4 to 8
on both, so the gateway is not saturating.

### What is deliberately excluded

| Excluded | Why |
|---|---|
| PSNC `GLM-5.2` reasoning-on | 233s per call, 12/16 timeouts at 120s (D-044) |
| PSNC `Qwen3.8-27B` reasoning-on | ~150s median, and 2/5 exceeded 120s even at concurrency 1 with a 20s cooldown (D-045) |
| OpenRouter `qwen/qwen3.8-27b` | $37.86 for its 3,492 tasks, against a $5 per-model budget (D-045) |
| OpenRouter `openai/gpt-4o-mini` reasoning | Emits no reasoning; its `not_applicable` arm is complete in `844df00e` |

The PSNC exclusions were tested rather than assumed. Cooldown was the specific hypothesis —
that PSNC throttles burst-like traffic — and it was disproven: at concurrency 1 with 20s
spacing, latency still tracked output length at ~60 tokens/sec, ranging 10.9s to 274.7s.
There is no queue to drain. D-045 has the measurements.

## Reading the results: the GLM reasoning confound

**This caveat is required whenever GLM reasoning results are reported, quoted or plotted.**

GLM-5.2 appears in the results with both reasoning arms, as it should:

| provider | model_id | reasoning_mode |
|---|---|---|
| `psnc` | `GLM-5.2` | `disabled` |
| `openrouter` | `z-ai/glm-5.2` | `enabled` |

Both rows are legitimate and both stay in the rankings, tables, exports and plots. The
reasoning dimension is a real experimental parameter and is displayed normally.

**But for GLM-5.2 specifically, reasoning state is confounded with deployment.** The
disabled arm ran on PSNC and the enabled arm runs on OpenRouter, because PSNC's GLM reasons
for 3,111 output tokens over 233 seconds and cannot fit the 120s timeout, while the same
model family through OpenRouter reasons for 772 tokens in 14 seconds (D-044, D-045). Two
things changed between those arms, not one, so a difference between them cannot be
attributed to reasoning alone.

The exports make this checkable rather than requiring trust: `provider` and
`reasoning_mode` are adjacent columns in every CSV and Excel export, and the two arms even
carry different `model_id` values, so they never silently collapse into a single "GLM-5.2"
row. A reader who wants to treat them as one model has to do so deliberately.

The other reasoning-capable models are **not** affected. `qwen/qwen3-8b` and
`qwen/qwen3-32b` run both arms on OpenRouter, so for those two the comparison is clean and
reasoning is the only thing that changed.

**Practical note when assembling the comparison:** the two GLM arms live in *different
campaigns*, so they never appear in a single ranking export. The disabled arm is in
`5cdc9417` (PSNC, 3,492 completed tasks, the authoritative one); the enabled arm will be in
the new OpenRouter campaign. Putting them side by side means joining two exports on
`model_id` + `reasoning_mode`, which is also the moment to carry the caveat above across.
Note that `GLM-5.2 disabled` also appears in several superseded campaigns (`ee138ed5`,
`d6d7da21`, `fd88566b`, `2264f8fc`); use `5cdc9417` and ignore the rest.


### Continuation strategy

The existing campaigns cannot be extended with new models or a reasoning dimension. Task
identity is `_hash({campaign, run, variable, source, gold})` and the run fingerprint itself
contains `campaign_id`, so identity is doubly campaign-scoped and there is no cross-campaign
deduplication. Adding a model to `parameters.yml` changes the configuration hash, hence the
plan hash, hence the campaign — every task would be re-created as `queued` and the ~21,000
already completed and paid for would run again.

The continuation therefore scopes each new campaign to only the combinations that have no
valid result, using `enabled:` flags and the `reasoning_profiles` list in `parameters.yml`.
Nothing already completed appears in any of the three plans.

Adding `reasoning = enabled` is safe for the same reason it is useful: `reasoning_mode` was
always part of the run parameters, so it already participates in `configuration_id` and the
task fingerprint. Adding an `enabled` profile to a model creates additional configurations
and leaves every existing `disabled` identity byte-identical. `tests/unit/test_reasoning_dimension.py`
pins exactly that, including that the grid doubles rather than shifts.

### Historical reasoning labels

One integrity note. Five `GLM-5.2` runs carry `reasoning_mode: not_applicable` but actually
reasoned: they predate D-041, when GLM was believed to have no controllable reasoning. They
hold 23 completed tasks, all in abandoned development campaigns, and no reported result
depends on them. They are left unmodified rather than relabelled, because rewriting stored
evidence to match a later understanding is worse than recording the discrepancy here.

Conversely, `d4296797` correctly records `reasoning_mode: enabled` for `qwen/qwen3-32b` with
97 completed tasks — the reasoning-on baseline that D-041 compares against. It used
`max_output_tokens: 5000` against the current plan's 8,000, so it is a different
configuration. Those 97 tasks are the one place the new plan overlaps completed work by
model/prompt/shots/temperature (1 of 180 combinations, 0.56%, about $0.04). Re-running
them is deliberate: mixing two output ceilings inside one model's 36-configuration grid
would be inconsistent, and that campaign was never finalized.

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
