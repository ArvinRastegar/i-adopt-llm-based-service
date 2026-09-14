# 30-shot extension and repetition study — results

**Exploratory side experiments.** Not part of the official experiment; these numbers cannot
enter an official ranking, campaign or results table. Siblings of
[shot-count-ablation.md](shot-count-ablation.md), which covers the earlier 0–10 run.

This file exists because the raw output does not survive in Git: `few-shot-selection/output/` is
gitignored, so without a written record a reader can see the runners but never learn what they
found. Run 2026-09-12, `Qwen3.8-27B` on PSNC, matrix-decomposition, T=0.5, top_p 1.0, 16,000
max output tokens, reasoning disabled.

## What was run

| Runner | Design | Calls | Outcome |
|---|---|---:|---|
| `shot_pool_30_ablation.py` | shots 0–30, 30-variable domain-stratified pool, 72 evaluation variables | 720 | 0 failed, 16 invalid |
| `shot_pool_30_repetitions.py` | 10 repetitions at 20/25/30 shots, same pool and population | 2,160 | 0 failed |

Neither is comparable to the earlier 0–10 ablation even at shared shot counts: that experiment
evaluated 92 variables, these evaluate 72, so a different set is being scored. Only the trend
within each run is meaningful.

## The curve, and where it stops

Micro Close F1, official-style invalid handling (invalid scored as an explicit empty
prediction, variable kept in the population):

| Shots | 0 | 1 | 3 | 5 | 7 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Close F1 | 0.224 | 0.238 | 0.347 | 0.373 | 0.365 | 0.378 | 0.414 | 0.451 | 0.455 | 0.456 |
| Step | — | +0.014 | +0.110 | +0.025 | −0.008 | +0.013 | +0.036 | +0.037 | **+0.004** | **+0.001** |

**The plateau is at roughly 20 examples.** Gains collapse by two orders of magnitude after 20.
This answers the question the earlier ablation left open — `Qwen3.8-27B` does not keep improving
indefinitely, and the official grid's ceiling of 5 was far below the useful range.

## The noise floor, which turned out to matter more

Ten repetitions at each of 20/25/30 shots:

| Shots | Mean | SD | 95% CI |
|---:|---:|---:|---|
| 20 | 0.4561 | 0.0164 | [0.4443, 0.4678] |
| 25 | 0.4510 | 0.0180 | [0.4381, 0.4639] |
| 30 | 0.4682 | 0.0168 | [0.4562, 0.4801] |

Pairwise Welch tests, Holm-corrected over three comparisons: 30 vs 25 `p = 0.123`, 30 vs 20
`p = 0.242`, 25 vs 20 `p = 0.519`. **Nothing is significant — 20, 25 and 30 are statistically
indistinguishable.**

**Pooled within-level SD is 0.019**, so a single measurement resolves only ~0.052 Close F1. This
number governs any further work on this corpus: it is why the example-selection optimization
uses randomized-subset attribution rather than greedy or evolutionary search, both of which
compare individual candidates at differences far below this floor. See
[example_selection/README.md](example_selection/README.md), decision DS-4.

## What a single repetition got wrong

The one-shot 30-variable run, read alone, showed a peak at 25 and a decline at 30. With ten
repetitions the ordering reverses:

| Shots | Single run (as-run) | 10-rep mean | Error |
|---:|---:|---:|---:|
| 20 | 0.4548 | 0.4620 | −0.007 |
| 25 | **0.4672** — looked like the peak | 0.4607 | +0.007 |
| 30 | 0.4603 — looked like a dip | **0.4768** — actually highest | −0.017 |

It drew a lucky-high 25 and an unlucky-low 30, both well inside the ±0.019 band. The apparent
peak-then-decline did not exist. Recorded because it is the clearest available argument for
replication on this corpus.

## Practical consequence

Use **20 examples**. It costs ~3,031 prompt tokens against 30-shot's ~3,979 — 24% cheaper for
no measurable loss.

## Isolation

Both runners live in `few-shot-selection/`, which no artifact collector walks; they write no campaign,
run, task or evaluation row, read nothing from `parameters.yml`, and write only to
`few-shot-selection/output/`. The official implementation hash is unaffected.
