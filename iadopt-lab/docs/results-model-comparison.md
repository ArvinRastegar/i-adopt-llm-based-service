# Every model at its best configuration

One row per model deployment, showing the single highest-scoring configuration it produced anywhere. **Reasoning is treated as a parameter like any other**: where a model was measured both ways, the row shows whichever setting won and the other is reported as a delta.

Pooled from the three full grids — `5cdc9417` (PSNC), `844df00e` and `79b5ec5f` (OpenRouter) — each 36 configurations per model over the same **97 variables**. The generator verifies they are comparable before reporting anything: identical population hash `86dcc62f505f…`, scorer `january-derived-member-credit-v1`, Close threshold 0.8. Every figure is recomputed from per-variable receipts and cross-checked against the stored ranking.

## Best configuration per model

| Model | Provider | Prompt | Shots | T | Reasoning | Close F1 | Exact F1 | Accuracy | Cost |
|---|---|---|--:|--:|---|--:|--:|--:|--:|
| `z-ai/glm-5.2` | openrouter | matrix-decomposition | 5 | 0.5 | enabled | **0.4206** | 0.3579 | 0.5134 | $0.3769 |
| `Qwen3.8-27B` | psnc | matrix-decomposition | 5 | 0.5 | disabled | **0.3912** | 0.3275 | 0.4759 | not billed |
| `DeepSeek-V4-Flash` | psnc | matrix-decomposition | 5 | 0.5 | not_applicable | **0.3899** | 0.3390 | 0.5000 | not billed |
| `GLM-5.2` | psnc | matrix-decomposition | 5 | 0 | disabled | **0.3752** | 0.3115 | 0.4661 | not billed |
| `qwen/qwen3-32b` | openrouter | constraint-decomposition | 3 | 0 | **disabled** | **0.3708** | 0.3113 | 0.4811 | $0.0165 |
| `qwen/qwen3-8b` | openrouter | constraint-decomposition | 5 | 1 | **disabled** | **0.3616** | 0.3098 | 0.4671 | $0.0230 |
| `mistralai/ministral-8b-2512` | openrouter | matrix-decomposition | 5 | 0 | not_applicable | **0.3615** | 0.3219 | 0.4590 | $0.0106 |
| `openai/gpt-4o-mini` | openrouter | matrix-decomposition | 3 | 0.5 | not_applicable | **0.2942** | 0.2384 | 0.4512 | $0.0159 |
| `meta-llama/llama-3.1-8b-instruct` | openrouter | strict-minimal | 5 | 0 | not_applicable | **0.2226** | 0.1979 | 0.4138 | $0.0067 |

A **bold** reasoning setting marks a model where both settings were actually measured and this one won. Every other row had only one setting available, so no choice was made — see the coverage table below.

## Where reasoning was a real choice

| Model | Winner | Best | Mean | n | Runner-up | Best | Mean | n | Delta (best) |
|---|---|--:|--:|--:|---|--:|--:|--:|--:|
| `qwen/qwen3-32b` | disabled | 0.3708 | 0.2907 | 36 | enabled | 0.3618 | 0.2804 | 22 | **+0.0090** |
| `qwen/qwen3-8b` | disabled | 0.3616 | 0.3474 | 3 | enabled | 0.3093 | 0.2822 | 8 | **+0.0523** |

Only these models were run both ways, on the same gateway and the same grid, so these are the experiment's only controlled reasoning comparisons. Everything else in the table above is a model's single measured setting.

**The two verdicts are not equally trustworthy.** `qwen/qwen3-32b` had 36 rankable configurations with reasoning off against 22 with it on, and off wins on both the maximum and the mean — that is a credible result. `qwen/qwen3-8b` had only **3 of 36** configurations survive with reasoning off, because upstream rate limiting destroyed the rest (D-048); its off arm is a maximum over three survivors and its higher mean reflects which configurations happened to complete, not which setting is better. Treat the qwen3-8b delta as unusable and the qwen3-32b one as the experiment's actual reasoning finding.

## Coverage — how much evidence each row rests on

A configuration is rankable only if all 97 of its variables completed. Losses are permanent, so a low count means the row's winner was chosen from a smaller field and its margin is correspondingly less certain.

| Model | Reasoning settings measured | Rankable configurations | Winner drawn from | Campaign |
|---|---|--:|--:|---|
| `z-ai/glm-5.2` | enabled | 18 | 18 | `79b5ec5f` |
| `Qwen3.8-27B` | disabled | 36 | 36 | `5cdc9417` |
| `DeepSeek-V4-Flash` | not_applicable | 36 | 36 | `5cdc9417` |
| `GLM-5.2` | disabled | 36 | 36 | `5cdc9417` |
| `qwen/qwen3-32b` | disabled, enabled | 58 | 36 | `844df00e` |
| `qwen/qwen3-8b` | disabled, enabled | 11 | 3 | `844df00e` |
| `mistralai/ministral-8b-2512` | not_applicable | 34 | 34 | `79b5ec5f` |
| `openai/gpt-4o-mini` | not_applicable | 36 | 36 | `844df00e` |
| `meta-llama/llama-3.1-8b-instruct` | not_applicable | 33 | 33 | `79b5ec5f` |

## Fine-grained: Close F1 per I-ADOPT field

Each model's winning configuration, scored one field at a time over the same 97 variables. Computed from each field's own contributions, so these neither average nor sum to the overall column.

| Model | Prop | Obj | Matrix | Ctx | StatMod | Constr |
|---|--:|--:|--:|--:|--:|--:|
| `z-ai/glm-5.2` | 0.472 | 0.512 | 0.247 | 0.273 | 0.545 | 0.384 |
| `Qwen3.8-27B` | 0.508 | 0.474 | 0.209 | 0.108 | 0.632 | 0.350 |
| `DeepSeek-V4-Flash` | 0.496 | 0.484 | 0.230 | 0.160 | 0.800 | 0.256 |
| `GLM-5.2` | 0.460 | 0.486 | 0.198 | 0.133 | 0.414 | 0.332 |
| `qwen/qwen3-32b` | 0.472 | 0.446 | 0.222 | 0.148 | 0.667 | 0.277 |
| `qwen/qwen3-8b` | 0.472 | 0.454 | 0.255 | 0.091 | 0.632 | 0.210 |
| `mistralai/ministral-8b-2512` | 0.508 | 0.494 | 0.145 | 0.111 | 0.500 | 0.248 |
| `openai/gpt-4o-mini` | 0.383 | 0.446 | 0.129 | 0.190 | 0.625 | 0.076 |
| `meta-llama/llama-3.1-8b-instruct` | 0.252 | 0.281 | 0.227 | 0.143 | 0.467 | 0.024 |

## Reading this table safely

**Provider is not a nuisance variable here, it is a confound.** The three PSNC rows ran
on different hardware under a different serving stack from the six OpenRouter rows. A
PSNC row scoring above an OpenRouter row does not establish that the model is better —
only that this deployment of it scored better on this corpus. `GLM-5.2` (PSNC) and
`z-ai/glm-5.2` (OpenRouter) are the same model family served two ways, and the gap
between those two rows mixes deployment with reasoning; it is not a reasoning result.

**Cost is not comparable across providers.** PSNC was non-billed on the owner's own
hardware, so its rows read `not billed` rather than `$0.0000`. That is an absence of
metering, not a price. Only the OpenRouter rows can be compared with each other on cost.

**Reasoning-off was never measured for `z-ai/glm-5.2`.** Its row is its only setting.
PSNC reasoning-enabled runs were excluded from the experiment because they could not fit
the 120-second timeout (D-044, D-045), so no controlled on/off comparison exists for the
GLM family at all.

**A model's own best row is a maximum over 36 configurations**, so it is optimistically
biased relative to that model's typical behaviour — the more configurations a model had
rankable, the more the maximum is favoured. The coverage table is there to make that
visible: `qwen/qwen3-8b` picked its reasoning-off winner from a field of 3.

## How the metrics are calculated

Identical to the per-configuration report, which documents the scoring in full:
[results-top-configurations.md](results-top-configurations.md). In brief — each of the
six I-ADOPT fields contributes exactly one unit of confusion mass per variable; TP, FP,
FN and TN are summed across all 97 variables as exact fractions, then:

```
precision = TP / (TP + FP)          F1       = 2·TP / (2·TP + FP + FN)
recall    = TP / (TP + FN)          accuracy = (TP + TN) / (TP + FP + FN + TN)
```

**Close** accepts a cosine similarity of 0.80 or better; **Exact** requires identical
normalised strings. Accuracy credits true negatives, and 39% of this corpus's field mass
is fields that should be left empty, so it rewards caution rather than skill — F1 is the
ranked metric and the one to read.
