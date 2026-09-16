# Top configurations — campaign `79b5ec5f`

Generated from stored evaluation receipts by `ops/report-top-configurations.py`. Every figure below is recomputed from the per-variable evidence with exact rational arithmetic; the overall Close F1 is checked against the value the ranking stored, and the generator refuses to write this file if they disagree.

Campaign total: **$19.73** over 17,460 tasks. Each configuration below was evaluated on the same **97 variables**.

## The ten best configurations

| # | Model | Prompt | Shots | T | Reasoning | Close F1 | Exact F1 | Accuracy | Cost |
|--:|---|---|--:|--:|---|--:|--:|--:|--:|
| 1 | `z-ai/glm-5.2` | matrix-decomposition | 5 | 0.5 | enabled | **0.4206** | 0.3579 | 0.5134 | $0.3769 |
| 2 | `z-ai/glm-5.2` | matrix-decomposition | 5 | 1 | enabled | **0.4097** | 0.3524 | 0.5025 | $0.3882 |
| 3 | `z-ai/glm-5.2` | matrix-decomposition | 3 | 1 | enabled | **0.3978** | 0.3348 | 0.4898 | $0.3559 |
| 4 | `z-ai/glm-5.2` | strict-minimal | 5 | 1 | enabled | **0.3978** | 0.3190 | 0.4989 | $0.3613 |
| 5 | `z-ai/glm-5.2` | matrix-decomposition | 3 | 0.5 | enabled | **0.3923** | 0.3273 | 0.4921 | $0.3367 |
| 6 | `z-ai/glm-5.2` | constraint-decomposition | 5 | 1 | enabled | **0.3779** | 0.3199 | 0.4847 | $0.3610 |
| 7 | `z-ai/glm-5.2` | strict-minimal | 5 | 0.5 | enabled | **0.3658** | 0.3054 | 0.4706 | $0.3279 |
| 8 | `z-ai/glm-5.2` | strict-minimal | 3 | 1 | enabled | **0.3647** | 0.3015 | 0.4767 | $0.3443 |
| 9 | `qwen/qwen3-32b` | strict-minimal | 3 | 1 | enabled | **0.3618** | 0.3065 | 0.4792 | $0.0774 |
| 10 | `mistralai/ministral-8b-2512` | matrix-decomposition | 5 | 0 | not_applicable | **0.3615** | 0.3219 | 0.4590 | $0.0106 |

Precision and recall behind the same Close F1 figures:

| # | Model | Close precision | Close recall | Close F1 |
|--:|---|--:|--:|--:|
| 1 | `z-ai/glm-5.2` | 0.3335 | 0.5694 | 0.4206 |
| 2 | `z-ai/glm-5.2` | 0.3281 | 0.5451 | 0.4097 |
| 3 | `z-ai/glm-5.2` | 0.3167 | 0.5350 | 0.3978 |
| 4 | `z-ai/glm-5.2` | 0.3185 | 0.5298 | 0.3978 |
| 5 | `z-ai/glm-5.2` | 0.3118 | 0.5289 | 0.3923 |
| 6 | `z-ai/glm-5.2` | 0.3004 | 0.5093 | 0.3779 |
| 7 | `z-ai/glm-5.2` | 0.2872 | 0.5039 | 0.3658 |
| 8 | `z-ai/glm-5.2` | 0.2851 | 0.5060 | 0.3647 |
| 9 | `qwen/qwen3-32b` | 0.2871 | 0.4891 | 0.3618 |
| 10 | `mistralai/ministral-8b-2512` | 0.3019 | 0.4504 | 0.3615 |

## Fine-grained: Close F1 per I-ADOPT field

Same configurations, scored one field at a time. A field's score is computed from that field's own contributions across all 97 variables — these are **not** averages of the overall column, and they do not sum to it.

| # | Model | Prop | Obj | Matrix | Ctx | StatMod | Constr |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | `z-ai/glm-5.2` | 0.472 | 0.512 | 0.247 | 0.273 | 0.545 | 0.384 |
| 2 | `z-ai/glm-5.2` | 0.496 | 0.538 | 0.148 | 0.200 | 0.545 | 0.355 |
| 3 | `z-ai/glm-5.2` | 0.496 | 0.510 | 0.121 | 0.129 | 0.571 | 0.380 |
| 4 | `z-ai/glm-5.2` | 0.472 | 0.502 | 0.205 | 0.160 | 0.571 | 0.343 |
| 5 | `z-ai/glm-5.2` | 0.460 | 0.519 | 0.107 | 0.154 | 0.545 | 0.384 |
| 6 | `z-ai/glm-5.2` | 0.448 | 0.484 | 0.189 | 0.190 | 0.480 | 0.320 |
| 7 | `z-ai/glm-5.2` | 0.460 | 0.468 | 0.200 | 0.143 | 0.435 | 0.298 |
| 8 | `z-ai/glm-5.2` | 0.484 | 0.466 | 0.097 | 0.148 | 0.455 | 0.317 |
| 9 | `qwen/qwen3-32b` | 0.460 | 0.448 | 0.182 | 0.138 | 0.667 | 0.279 |
| 10 | `mistralai/ministral-8b-2512` | 0.508 | 0.494 | 0.145 | 0.111 | 0.500 | 0.248 |

`n/a` means the field contributed no true positives, false positives or false negatives anywhere in the population — gold and prediction were both empty for every variable, so F1 is undefined rather than zero.

## What the cost buys

Cost per configuration covers 97 variables including retries. The right-hand column scales that to 1,000 variables, which is the number worth comparing against if this is ever run at corpus scale.

| # | Model | Close F1 | Cost (97 vars) | Relative | Per 1,000 vars |
|--:|---|--:|--:|--:|--:|
| 1 | `z-ai/glm-5.2` | 0.4206 | $0.3769 | 35.7x | $3.89 |
| 2 | `z-ai/glm-5.2` | 0.4097 | $0.3882 | 36.8x | $4.00 |
| 3 | `z-ai/glm-5.2` | 0.3978 | $0.3559 | 33.7x | $3.67 |
| 4 | `z-ai/glm-5.2` | 0.3978 | $0.3613 | 34.2x | $3.73 |
| 5 | `z-ai/glm-5.2` | 0.3923 | $0.3367 | 31.9x | $3.47 |
| 6 | `z-ai/glm-5.2` | 0.3779 | $0.3610 | 34.2x | $3.72 |
| 7 | `z-ai/glm-5.2` | 0.3658 | $0.3279 | 31.0x | $3.38 |
| 8 | `z-ai/glm-5.2` | 0.3647 | $0.3443 | 32.6x | $3.55 |
| 9 | `qwen/qwen3-32b` | 0.3618 | $0.0774 | 7.3x | $0.80 |
| 10 | `mistralai/ministral-8b-2512` | 0.3615 | $0.0106 | 1.0x | $0.11 |

The spread matters more than the absolute figures. `mistralai/ministral-8b-2512` at rank 10 reaches **86%** of the best score for **1/36 of the cost** — 0.0591 F1 for 36x the spend is the trade the top of this table is making. Which side of it is right depends on whether the output is reviewed by a person afterwards; at these accuracies it will be.

## How each metric is calculated

Scorer `january-derived-member-credit-v1`, similarity MiniLM-L6-v2 cosine over
NFC-normalised, case-folded, trimmed strings.

### The unit of measurement

A decomposition has six fields. **Each field contributes exactly one unit of confusion
mass per variable**, split between true positives, false positives, false negatives and
true negatives. One variable therefore carries 6 units, and a 97-variable population
carries 582. That fixed mass is what makes the fields comparable with each other and the
configurations comparable with one another.

How a field's unit is divided depends on what is being compared:

| Situation | Branch | Result |
|---|---|---|
| Both gold and prediction empty | `january-empty` | TN = 1 |
| Gold empty, prediction present | `january-empty` | FP = 1 |
| Gold present, prediction empty | `january-empty` | FN = 1 |
| Both plain strings | `january-scalar` | similarity ≥ threshold → TP = 1, else **FP = 1** |
| Either side a nested system | `system-member-credit-v1` | unit split across members by assignment |
| `hasConstraint` | `january-constraints-greedy` | unit split across constraints, see below |

**Note the scalar row.** A present-but-wrong answer scores FP = 1 and FN = 0. It is
counted as a wrong thing said, not as a right thing missed. This is why recall sits
above precision almost everywhere in the tables above: recall is only reduced by fields
left *empty* that should have been filled.

### hasConstraint

Each gold constraint carries two units of `1/(2·n_gold)` — one for its `label`, one for
its `on` target — so the field still totals one unit. Gold and predicted constraints are
paired greedily, highest mean similarity first, row-major on ties. Within a pair, label
and target are scored separately: at or above threshold adds its unit to TP, below adds
it to FP. Unmatched gold constraints add `2·unit` to FN; unmatched predicted constraints
add `2·unit` to FP. A final correction rescales so the field's mass is exactly 1.

### Nested systems

A ratio such as `lactate / blood`, or a flux such as `vegetation → soil`, is compared
member by member. Asymmetric systems only allow matches in the same role — a numerator
may not be credited against a denominator. Each member carries an equal share of the
field's unit, so getting one of two roles right earns half credit rather than none.

### Close versus Exact

Identical machinery, one number different:

- **Close** — cosine similarity ≥ **0.80**. "amount of substance concentration" against
  "concentration" scores as a hit.
- **Exact** — threshold **1.0**, i.e. the normalised strings must be identical.

Exact is always the harsher of the two, and the gap between the columns is a direct
measure of how much a configuration is getting *nearly* right.

### Aggregating to a configuration

Micro, not macro: TP, FP, FN and TN are summed across all 97 variables as exact
fractions, and the metric is computed once from those totals.

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2·TP / (2·TP + FP + FN)
accuracy  = (TP + TN) / (TP + FP + FN + TN)
```

Micro-averaging means every variable carries the same weight regardless of how many
constraints it happens to have. Averaging the 97 per-variable F1 scores instead (macro)
would give a different, generally higher number; the ranking uses micro, and so does
every figure here.

**Accuracy is the weakest of these four**, and is reported only because it was asked
for. It is the one metric that credits true negatives, and in this corpus true negatives
are abundant. Counting empty gold fields across the 97-variable population:

| Field | Gold present | Gold empty |
|---|--:|--:|
| `hasProperty` | 97 | 0 |
| `hasObjectOfInterest` | 97 | 0 |
| `hasConstraint` | 97 | 0 |
| `hasMatrix` | 47 | 50 |
| `hasContextObject` | 10 | 87 |
| `hasStatisticalModifier` | 8 | 89 |

**226 of the 582 mass units — 39% — are fields that should be left empty.** A model that
emits nothing at all for `hasContextObject` and `hasStatisticalModifier` banks 176 units
of true negative before saying anything correct. Accuracy therefore rewards correct
silence as much as correct extraction and rises with caution rather than skill, which is
why F1, which ignores TN entirely, is the ranked metric. Read the accuracy column as a
sanity check, not as a score.

### Fine-grained field scores

Each field's F1 is computed from that field's own contributions summed over the
population — the same micro formula, restricted to one of the six. They are independent
views, not a decomposition of the overall figure: the overall score is computed from
total mass across all six fields at once, so the six field scores neither average nor
sum to it.

`n/a` appears where a field generated no TP, FP or FN anywhere — gold and prediction
were both empty for all 97 variables, leaving only true negatives and an undefined F1.

### Cost

Per-attempt amounts from `cost_settlement`, summed over every attempt belonging to the
configuration, in USD. This is settled spend from the provider's reported token usage,
not an estimate, and it includes retries: a configuration whose answers failed
validation twice before succeeding paid for all three calls. Attempts whose settlement
is `unavailable` or `ambiguous` contribute nothing, so a configuration's cost is a
slight underestimate where those occurred; the per-row attempt counts are in the
generator's output if you need the exact coverage.
