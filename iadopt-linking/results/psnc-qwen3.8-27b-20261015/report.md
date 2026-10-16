# Qwen3.8-27B (reasoning off, PSNC) over the baseline's pooled Wikidata, QUDT and OM candidates

Run `psnc-qwen3.8-27b-20261015`, report built 2026-10-16T17:03:13+00:00. Requests sent 2026-10-15T18:35:06+00:00 to 2026-10-15T18:40:50+00:00: 1144 requests answered (565,176 input tokens, 1,144 output tokens; 340.7 s of request time, 0.298 s on average, 1.467 s at most) over 1144 stored responses, sent one request at a time. The service is not billed to the owner. Corpus I-ADOPT v2.0.1 (2598bf91fa). Candidates and Qwen3 scores come from baseline `baseline-20261014` (scores `ced730904872`), not from new retrieval. Evidence code `a342a3b64359`, all code `22b92f5cf681`, prompt `20d69e9a52e2`. Model requested `Qwen3.8-27B`, served as `Qwen3.8-27B` by psnc.

Qwen3.8-27B is a chat model served by PSNC; reasoning is switched off, the temperature is 0 and one token is generated. Each slot's pooled candidates, exactly as the Qwen3 baseline saw them (label and description capped at 300 characters, no source tag), were put to it as a numbered list with an extra option 0, `none of these`, and it was asked to reply with one label. Its probabilities for the options are read from the log-probabilities of that first token. The primary result, fixed in the run's plan and in DECISIONS L-016 and L-017 before the first request (no commit predates the run), is the first pass (a seeded candidate order) with the forced choice: the candidate with the highest probability, `none_of_these` ignored, ties decided as in the baseline. The same pool was also asked, to see how much the result depends on order and on chance: an identical repeat of the primary request; the reverse order; another candidate order (shuffle-b). The comparison with Jev on the same slots, also by gold vocabulary, is pre-specified too; everything else, the rows by gold vocabulary against the baseline and every reading that counts a QUDT kind and its equivalent items as one entity included, is exploratory.

## Result

| Population | Slots | Distinct terms | Qwen3.8-27B top-1 | Qwen3.8-27B mean per variable | Qwen3 top-1 | Qwen3.8-27B minus Qwen3 | McNemar p |
|---|---|---|---|---|---|---|---|
| Gold in Wikidata, QUDT or OM (primary) | 284 | 145 | 206/284 = 0.725 | 0.705 | 194/284 = 0.683 | +0.042 (95% interval -0.073 to +0.172) | 0.1882 |
| All controlled-vocabulary slots | 294 | 153 | 206/294 = 0.701 | 0.681 | 194/294 = 0.660 | +0.041 (95% interval -0.069 to +0.171) | 0.1882 |
| Gold in Wikidata | 218 | 116 | 169/218 = 0.775 | 0.752 | 138/218 = 0.633 | +0.142 (95% interval +0.008 to +0.300) | <0.0001 |
| Gold in QUDT | 61 | 31 | 32/61 = 0.525 | 0.525 | 51/61 = 0.836 | -0.311 (95% interval -0.522 to -0.138) | <0.0001 |
| Gold in OM | 5 | 1 | 5/5 = 1.000 | 1.000 | 5/5 = 1.000 | +0.000 (95% interval +0.000 to +0.000) | 1.0000 |
| Gold in another vocabulary (not searchable here) | 10 | 8 | 0/10 = 0.000 | 0.000 | 0/10 = 0.000 | - | - |

Both systems were judged by the same code on the same slots and the same candidates. The interval comes from resampling whole terms (153 terms ask 294 slots, so slots are not independent evidence); McNemar's p treats slots as independent and is therefore optimistic. Mean per variable is the manuscript's style of measure.

## Qwen3.8-27B against Qwen3 on the reachable slots

|  | Qwen3 right | Qwen3 wrong |
|---|---|---|
| Qwen3.8-27B right | 165 | 41 |
| Qwen3.8-27B wrong | 29 | 49 |

Strict identity: Qwen3.8-27B minus Qwen3 +0.042 (95% interval -0.073 to +0.172); exact McNemar p 0.1882.
Counted once per term instead of once per slot (exploratory: slots cluster by term): Qwen3.8-27B better on 19 of 145 terms, Qwen3 better on 16, equal on 110 (exact sign test p 0.7359; term-averaged difference +0.012).
If a QUDT kind and its `wikidataMatch` item counted as one entity: Qwen3.8-27B minus Qwen3 +0.056 (95% interval -0.053 to +0.183); Qwen3.8-27B better on 16 of 145 terms, Qwen3 better on 11, equal on 118 (exact sign test p 0.4421; term-averaged difference +0.024). (An undecided question of the project; strict identity is the headline.)


## Qwen3.8-27B against Jev on the same slots (pre-specified)

Jev is the primary pass of the finished run `jev-20261015` (read from its published, hash-verified outcomes), asked about the same candidates. Differences are in top-1 correct slots (Qwen3.8-27B minus Jev); the interval resamples whole terms.

| Population | Slots | Qwen3.8-27B top-1 | Qwen3 top-1 | Jev top-1 | Qwen3.8-27B minus Jev | McNemar p |
|---|---|---|---|---|---|---|
| Gold in Wikidata, QUDT or OM (primary) | 284 | 206/284 = 0.725 | 194/284 = 0.683 | 213/284 = 0.750 | -0.025 (95% interval -0.082 to +0.034) | 0.3489 |
| All controlled-vocabulary slots | 294 | 206/294 = 0.701 | 194/294 = 0.660 | 213/294 = 0.724 | -0.024 (95% interval -0.081 to +0.032) | 0.3489 |
| Gold in Wikidata | 218 | 169/218 = 0.775 | 138/218 = 0.633 | 174/218 = 0.798 | -0.023 (95% interval -0.086 to +0.039) | 0.4244 |
| Gold in QUDT | 61 | 32/61 = 0.525 | 51/61 = 0.836 | 34/61 = 0.557 | -0.033 (95% interval -0.196 to +0.107) | 0.8036 |
| Gold in OM | 5 | 5/5 = 1.000 | 5/5 = 1.000 | 5/5 = 1.000 | +0.000 (95% interval +0.000 to +0.000) | 1.0000 |

On the 284 reachable slots both are right on 189, only Qwen3.8-27B on 17, only Jev on 24 and neither on 54.
Strict identity on the reachable slots: Qwen3.8-27B minus Jev -0.025 (95% interval -0.082 to +0.034); exact McNemar p 0.3489.
Counted once per term: Qwen3.8-27B better on 13 of 145 terms, Jev better on 12, equal on 120 (exact sign test p 1.0000; term-averaged difference -0.013).
Exploratory, because whether such twins count is an open question of the project: if a QUDT kind and its `wikidataMatch` item counted as one entity, Qwen3.8-27B minus Jev -0.067 (95% interval -0.127 to -0.007); Qwen3.8-27B better on 8 of 145 terms, Jev better on 15, equal on 122 (exact sign test p 0.2100; term-averaged difference -0.060).

## Where the misses come from (reachable slots)

| Outcome | Qwen3.8-27B slots | Qwen3.8-27B share | Qwen3 slots | Qwen3 share |
|---|---|---|---|---|
| correct | 206 | 0.725 | 194 | 0.683 |
| wrong within source | 35 | 0.123 | 55 | 0.194 |
| wrong source | 34 | 0.120 | 26 | 0.092 |
| gold not retrieved | 3 | 0.011 | 3 | 0.011 |
| no candidates | 6 | 0.021 | 6 | 0.021 |

## Diagnostics (reachable slots)

| Question | Qwen3.8-27B | Qwen3 |
|---|---|---|
| Is the gold term among the pooled candidates? | 275/284 = 0.968 | 275/284 = 0.968 |
| Does the scorer pick it when it is? | 206/275 = 0.749 | 194/275 = 0.705 |
| Is the chosen candidate from the gold's own source? (slots with any candidate) | 244/278 = 0.878 | 252/278 = 0.906 |
| Top-1 if the source were known (best score within the gold's source) | 237/284 = 0.835 | 216/284 = 0.761 |
| Top-1 of the gold source's own first hit (no scorer) | 204/284 = 0.718 | 204/284 = 0.718 |
| Top-1 if a QUDT kind and its `wikidataMatch` item counted as the same entity | 218/284 = 0.768 | 202/284 = 0.711 |
| Ceiling for a system that sees only the term (majority gold per term) | 268/284 = 0.944 | 268/284 = 0.944 |

On the Wikidata-gold slots (exploratory), picking the best-scoring Wikidata candidate only (for Qwen3.8-27B the probabilities of the pooled question, restricted to Wikidata; for Qwen3 the manuscript's "reranker alone" setting): Qwen3.8-27B 172/218 = 0.789, Qwen3 153/218 = 0.702.

## Which source wins (reachable slots with any candidate)

Rows are the gold's source, columns the source of the selected candidate.

**Qwen3.8-27B**

| Gold in \ chosen from | wikidata | qudt | om |
|---|---|---|---|
| wikidata | 207 | 3 | 2 |
| qudt | 16 | 32 | 13 |
| om | 0 | 0 | 5 |

**Qwen3**

| Gold in \ chosen from | wikidata | qudt | om |
|---|---|---|---|
| wikidata | 193 | 19 | 0 |
| qudt | 6 | 54 | 1 |
| om | 0 | 0 | 5 |

## Passes, order and chance

| Pass | Arm | Order | Top-1 reachable | Top-1 all | Same selection as primary | Minus Qwen3 (reachable) |
|---|---|---|---|---|---|---|
| `choice-a1` | choice | shuffle-a | 206/284 = 0.725 | 206/294 = 0.701 | - | +0.042 (95% interval -0.073 to +0.172) |
| `choice-a1-repeat` | choice | shuffle-a | 206/284 = 0.725 | 206/294 = 0.701 | 286/286 = 1.000 | +0.042 (95% interval -0.073 to +0.172) |
| `choice-a2-reversed` | choice | reverse-a | 202/284 = 0.711 | 202/294 = 0.687 | 220/286 = 0.769 | +0.028 (95% interval -0.084 to +0.156) |
| `choice-a3` | choice | shuffle-b | 201/284 = 0.708 | 201/294 = 0.684 | 244/286 = 0.853 | +0.025 (95% interval -0.092 to +0.156) |

Identical request sent twice (`choice-a1` and `choice-a1-repeat`), 286 slots: the same candidate was selected for 286/286 = 1.000; every candidate's probability was identical for 286/286 = 1.000; the largest difference in a candidate's probability was 0.00 (mean 0.0000).

Accuracy totals hide offsetting changes (exploratory). Reachable slots whose correctness differs from the primary pass: `choice-a1-repeat` changes 0 of 278 (0 become right, 0 wrong); `choice-a2-reversed` changes 40 of 278 (18 become right, 22 wrong); `choice-a3` changes 21 of 278 (8 become right, 13 wrong). All the Choice passes select the same candidate on 207 of 278 slots and are equally right or wrong on 232.

Does Qwen3.8-27B favour the first option it is shown? Share of slots with at least two candidates in which the selected candidate was shown first, against the share if position played no part (the mean of 1/n):

| Pass | Slots | Selected candidate shown first | Expected number if position played no part |
|---|---|---|---|
| `choice-a1` | 281 | 27 | 31.5 |
| `choice-a1-repeat` | 281 | 27 | 31.5 |
| `choice-a2-reversed` | 281 | 24 | 31.5 |
| `choice-a3` | 281 | 24 | 31.5 |

Exploratory: averaging the probabilities of the Choice passes (`choice-a1`, `choice-a1-repeat`, `choice-a2-reversed`, `choice-a3`) and selecting from the average gives 202/284 = 0.711 on the reachable slots; minus Qwen3 +0.028 (95% interval -0.088 to +0.160). An identical repeat counts as a pass here, so its order weighs twice.

## Qwen3.8-27B's own `none_of_these` (primary pass, reachable slots)

| Question | Answer |
|---|---|
| Slots asked | 278 |
| `none_of_these` was Qwen3.8-27B's choice although the gold was among the candidates | 15/275 = 0.055 |
| `none_of_these` was Qwen3.8-27B's choice when the gold was not among the candidates (a right abstention) | 2/3 = 0.667 |
| Slots Qwen3.8-27B answered with a candidate | 261/278 = 0.939 |
| Precision of the forced choice on those | 199/261 = 0.762 |
| Forced choice correct on the slots where Qwen3.8-27B chose `none_of_these` | 7/17 = 0.412 |
| Refusals made at an own-choice probability of 0.9 or more | 5 of 17 |

The headline always uses the forced choice, as the baseline always picks its top candidate; this table shows what honouring Qwen3.8-27B's refusals would change. On the 8 asked slots whose gold lies in a vocabulary the sources cannot reach, Qwen3.8-27B chose `none_of_these` for 5; they are left out above because a pool may hold the same concept under another identifier.

## What the confidence is worth (reachable slots)

Any number that says how much to trust a selection can decide when to hand a slot to a stronger model. AUROC is the chance that a right answer is more confident than a wrong one (0.5 is no information). The calibration gap and Brier score treat the value as a probability of being right. All of this is exploratory and uses the data it describes; choose a threshold on held-out terms before relying on one.

| Signal | Slots answered | AUROC | Calibration gap | Brier |
|---|---|---|---|---|
| Qwen3.8-27B: probability of the selected candidate | 278 | 0.860 | 0.080 | 0.136 |
| Qwen3.8-27B: probability of its own choice (`none_of_these` included) | 278 | 0.846 | 0.070 | 0.142 |
| Qwen3.8-27B: top-two probability margin | 273 | 0.867 | - | - |
| Qwen3.8-27B: probability not given to `none_of_these` | 278 | 0.784 | - | - |
| Qwen3.8-27B: share of the Choice passes that agree | 278 | 0.759 | - | - |
| Qwen3: score of the selected candidate | 278 | 0.587 | 0.283 | 0.279 |
| Qwen3: top-two score margin | 273 | 0.604 | - | - |

The model has no confidence value of its own; its own-choice probability is the probability of the option it answered with, `none_of_these` included. For 17 reachable slots that choice was `none_of_these` (5 of them at 0.9 or more), so for those slots it measures confidence in the refusal, not in the forced candidate whose correctness is scored here.

Coverage and precision when only selections at or above a threshold are answered (reachable slots; the rest would go to a stronger model):

| Threshold | Qwen3.8-27B probability: answered | precision | Qwen3.8-27B own-choice probability: answered | precision | Qwen3 score: answered | precision |
|---|---|---|---|---|---|---|
| 0.000 | 278/284 = 0.979 | 206/278 = 0.741 | 278/284 = 0.979 | 206/278 = 0.741 | 278/284 = 0.979 | 194/278 = 0.698 |
| 0.100 | 273/284 = 0.961 | 204/273 = 0.747 | 278/284 = 0.979 | 206/278 = 0.741 | 278/284 = 0.979 | 194/278 = 0.698 |
| 0.200 | 268/284 = 0.944 | 203/268 = 0.757 | 278/284 = 0.979 | 206/278 = 0.741 | 278/284 = 0.979 | 194/278 = 0.698 |
| 0.300 | 263/284 = 0.926 | 198/263 = 0.753 | 277/284 = 0.975 | 205/277 = 0.740 | 277/284 = 0.975 | 194/277 = 0.700 |
| 0.400 | 249/284 = 0.877 | 194/249 = 0.779 | 264/284 = 0.930 | 200/264 = 0.758 | 275/284 = 0.968 | 194/275 = 0.705 |
| 0.500 | 227/284 = 0.799 | 186/227 = 0.819 | 243/284 = 0.856 | 192/243 = 0.790 | 275/284 = 0.968 | 194/275 = 0.705 |
| 0.600 | 202/284 = 0.711 | 177/202 = 0.876 | 216/284 = 0.761 | 182/216 = 0.843 | 275/284 = 0.968 | 194/275 = 0.705 |
| 0.700 | 176/284 = 0.620 | 163/176 = 0.926 | 187/284 = 0.658 | 168/187 = 0.898 | 274/284 = 0.965 | 194/274 = 0.708 |
| 0.800 | 157/284 = 0.553 | 151/157 = 0.962 | 164/284 = 0.577 | 153/164 = 0.933 | 271/284 = 0.954 | 194/271 = 0.716 |
| 0.900 | 135/284 = 0.475 | 131/135 = 0.970 | 140/284 = 0.493 | 133/140 = 0.950 | 270/284 = 0.951 | 194/270 = 0.719 |
| 0.950 | 118/284 = 0.415 | 115/118 = 0.975 | 120/284 = 0.423 | 115/120 = 0.958 | 262/284 = 0.923 | 192/262 = 0.733 |
| 0.990 | 76/284 = 0.268 | 75/76 = 0.987 | 76/284 = 0.268 | 75/76 = 0.987 | 226/284 = 0.796 | 167/226 = 0.739 |
| 0.999 | 29/284 = 0.102 | 29/29 = 1.000 | 29/284 = 0.102 | 29/29 = 1.000 | 91/284 = 0.320 | 67/91 = 0.736 |

Handing the slots below a threshold on Qwen3.8-27B's selected probability to a stronger model (the split into fixable and unfixable errors is exploratory). Of the escalated slots, the right ones are work a stronger model repeats; the wrong ones with the gold among the candidates are errors it could fix; the wrong ones without are beyond any scorer. The ceiling assumes the stronger model fixes every fixable error, so it is an upper bound, not a prediction:

| Threshold | Escalated | Right among them | Wrong, fixable | Wrong, unfixable | Precision of the rest | Ceiling |
|---|---|---|---|---|---|---|
| 0.000 | 6/284 = 0.021 | 0 | 0 | 6 | 206/278 = 0.741 | 206/284 = 0.725 |
| 0.100 | 11/284 = 0.039 | 2 | 2 | 7 | 204/273 = 0.747 | 208/284 = 0.732 |
| 0.200 | 16/284 = 0.056 | 3 | 5 | 8 | 203/268 = 0.757 | 211/284 = 0.743 |
| 0.300 | 21/284 = 0.074 | 8 | 5 | 8 | 198/263 = 0.753 | 211/284 = 0.743 |
| 0.400 | 35/284 = 0.123 | 12 | 15 | 8 | 194/249 = 0.779 | 221/284 = 0.778 |
| 0.500 | 57/284 = 0.201 | 20 | 29 | 8 | 186/227 = 0.819 | 235/284 = 0.827 |
| 0.600 | 82/284 = 0.289 | 29 | 45 | 8 | 177/202 = 0.876 | 251/284 = 0.884 |
| 0.700 | 108/284 = 0.380 | 43 | 57 | 8 | 163/176 = 0.926 | 263/284 = 0.926 |
| 0.800 | 127/284 = 0.447 | 55 | 64 | 8 | 151/157 = 0.962 | 270/284 = 0.951 |
| 0.900 | 149/284 = 0.525 | 75 | 66 | 8 | 131/135 = 0.970 | 272/284 = 0.958 |
| 0.950 | 166/284 = 0.585 | 91 | 67 | 8 | 115/118 = 0.975 | 273/284 = 0.961 |
| 0.990 | 208/284 = 0.732 | 131 | 69 | 8 | 75/76 = 0.987 | 275/284 = 0.968 |
| 0.999 | 255/284 = 0.898 | 177 | 69 | 9 | 29/29 = 1.000 | 275/284 = 0.968 |

Reliability of Qwen3.8-27B's selected probability (bins of width 0.1):

| Bin | Slots | Right | Mean probability |
|---|---|---|---|
| 0.000-0.100 | 5 | 2 | 0.030 |
| 0.100-0.200 | 5 | 1 | 0.151 |
| 0.200-0.300 | 5 | 5 | 0.255 |
| 0.300-0.400 | 14 | 4 | 0.367 |
| 0.400-0.500 | 22 | 8 | 0.458 |
| 0.500-0.600 | 25 | 9 | 0.553 |
| 0.600-0.700 | 26 | 14 | 0.652 |
| 0.700-0.800 | 19 | 12 | 0.756 |
| 0.800-0.900 | 22 | 20 | 0.843 |
| 0.900-1.000 | 135 | 131 | 0.983 |

Probabilities are read from the first token's log-probabilities, so they are continuous, not rounded; they are estimated from the 20 most probable tokens, and the notes say how much of the first token those left out.

## By slot, role and term length

| Group | Slots | Qwen3.8-27B top-1 | Qwen3 top-1 | Gold in pool |
|---|---|---|---|---|
| role:member | 68 | 59/68 = 0.868 | 46/68 = 0.676 | 64/65 = 0.985 |
| role:top-level | 226 | 147/226 = 0.650 | 148/226 = 0.655 | 211/219 = 0.963 |
| slot:hasContextObject | 11 | 11/11 = 1.000 | 5/11 = 0.455 | 11/11 = 1.000 |
| slot:hasMatrix | 55 | 36/55 = 0.655 | 32/55 = 0.582 | 55/55 = 1.000 |
| slot:hasObjectOfInterest | 118 | 95/118 = 0.805 | 79/118 = 0.669 | 107/113 = 0.947 |
| slot:hasProperty | 101 | 56/101 = 0.554 | 70/101 = 0.693 | 94/96 = 0.979 |
| slot:hasStatisticalModifier | 9 | 8/9 = 0.889 | 8/9 = 0.889 | 8/9 = 0.889 |
| words:1 | 182 | 127/182 = 0.698 | 117/182 = 0.643 | 176/178 = 0.989 |
| words:2 | 94 | 63/94 = 0.670 | 62/94 = 0.660 | 81/88 = 0.920 |
| words:3+ | 18 | 16/18 = 0.889 | 15/18 = 0.833 | 18/18 = 1.000 |

## Most repeated terms

A few terms are asked many times, so their outcome weighs on every slot-weighted figure above.

| Term | Slots | Qwen3.8-27B wrong | Qwen3 wrong |
|---|---|---|---|
| `person` | 26 | 20 | 14 |
| `water` | 12 | 0 | 12 |
| `count` | 11 | 0 | 0 |
| `urban area` | 11 | 0 | 10 |
| `atmosphere` | 9 | 0 | 1 |

## Examples of misses

Up to 6 per outcome, first by slot id. Gold is vocabulary:identifier.

**wrong within source** (35 slots)

- `enclosure` (C14_FeralfreeEnclosureArea.ttl) -> 'Enclosure' (wikidata:Q139730832, 0.59); gold wikidata:Q858177
- `oxygen` (C16OxygenMetabolicRate.ttl) -> 'oxygen' (wikidata:Q629, 0.98); gold wikidata:Q5203615
- `ground` (C8_CWDFraction.ttl) -> 'army' (wikidata:Q37726, 0.13); gold wikidata:Q36133
- `ground` (C9_FoliageCover.ttl) -> "Earth's surface" (wikidata:Q1349417, 0.55); gold wikidata:Q36133
- `person` (BMI.ttl) -> 'Person' (wikidata:Q105964318, 0.68); gold wikidata:Q5
- `person` (Bloodtype.ttl) -> 'person' (wikidata:Q215627, 0.38); gold wikidata:Q5

**wrong source** (34 slots)

- `mass` (C16OxygenMetabolicRate.ttl) -> 'mass' (om:Mass, 0.38); gold qudt:Mass
- `height` (C7_TrunkCharHeight.ttl) -> 'height' (om:Height, 0.57); gold qudt:Height
- `mass fraction` (CONCVAR.ttl) -> 'mass fraction' (om:MassFraction, 0.44); gold qudt:MassFraction
- `index` (BMI.ttl) -> 'Body Mass Index' (qudt:BodyMassIndex, 0.34); gold wikidata:Q1738991
- `pressure` (C21_BloodPressure.ttl) -> 'pressure' (wikidata:Q39552, 0.69); gold qudt:Pressure
- `time` (HRV.ttl) -> 'time' (om:Time, 0.69); gold qudt:Time

**gold not retrieved** (3 slots)

- `char` (C7_TrunkCharHeight.ttl) -> 'char' (wikidata:Q478251, 0.13); gold wikidata:Q11567121
- `standard deviation` (HRV.ttl) -> 'standard deviation' (wikidata:Q159375, 0.99); gold wikidata:Q175199
- `state` (StateOfPipeConnection.ttl) -> 'state' (wikidata:Q7275, 0.06); gold wikidata:Q813912

**no candidates** (6 slots)

- `nanomaterial particle` (C24_NanoparticleCount.ttl) -> nothing; gold wikidata:Q967847
- `hexanol (C6H14O)` (COMVAR.ttl) -> nothing; gold wikidata:Q76933
- `benzene (C6H6)` (DENSVAR.ttl) -> nothing; gold wikidata:Q2270
- `naphthalene (C10H8)` (MPVAR.ttl) -> nothing; gold wikidata:Q179724
- `naphthalene (C10H8)` (SLVAR.ttl) -> nothing; gold wikidata:Q179724
- `quality flag` (GDOP_QC.ttl) -> nothing; gold wikidata:Q137642315

## Run health

| Check | Result | Detail |
|---|---|---|
| inputs identical to the baseline | pass | 294 slots and 3473 candidate documents rebuilt and equal to the baseline's; 7 baseline files re-hashed; the baseline scored 294 slots in 294 records |
| every request answered | pass | 1144/1144 requests answered (0 needed an HTTP retry; 0 stored responses were rejected as invalid answers and asked again) |
| stored requests regenerate from the plan | pass | 1144/1144 stored requests equal the ones this tree builds |
| stored response hashes verify | pass | 1144 stored responses re-hashed |
| every response is a valid answer | pass | 1144 responses read strictly; at least 0.9512 of every first token's probability sits on option labels (required: 0.5), no response held reasoning text |
| one model snapshot served | pass | served as Qwen3.8-27B by build vllm-0.26.0-1c0daee9 (requested Qwen3.8-27B) |
| token usage fully accounted | pass | 566,320 tokens used over 1144 answers and 0 rejected responses, every stored response included; not billed to the owner |
| primary pass complete | pass | 286/286 requests of choice-a1 answered |

## Notes

- The model saw, for every candidate, the label and the description (capped at 300 characters): the information the Qwen3 reranker saw, with no vocabulary, identifier, rank or score. The term and the variable's definition are the baseline's, and the candidates were shown in the same seeded orders the Jev run used.
- 8 slots have no candidate; they were not asked and count as misses, as in the baseline.
- 17 label-only nodes and 2 non-vocabulary IRIs are outside the population, as in the baseline.
- 10 slots have their gold in a vocabulary these sources cannot reach; they count as misses in the "All controlled-vocabulary slots" row and are left out of the primary population (the reachable slots).
- Reasoning is switched off with `chat_template_kwargs.enable_thinking=false`, the control the lab measured for Qwen3.8-27B on this gateway; the parser refuses any response that carries reasoning text, and none did.
- A probability is the model's probability of that label given that its first token is a label, estimated from the 20 most probable tokens the gateway reports: the tokens that spell a label are summed and divided by the total over the labels found, so the options sum to 1, and a label outside the reported tokens counts as 0. The share of the first token's probability on option labels was at least 0.9512 in every response (median 0.9995), the rest sitting on other tokens, which the division leaves out; between 2 and 20 labels were reported per response, and what lay outside the reported tokens bounds the error of any probability at 0.0019.
- 1 of 281 primary selections that have a runner-up tie with it and were decided by the baseline's rule (source order, then rank).
- Gold terms are the corpus labels of the gold nodes; for QUDT slots they are often QUDT's own labels, so QUDT accuracy on these terms may overstate accuracy on LLM-predicted wording.
- The service is shared and its answers can differ slightly with what else its server is processing: in development, identical requests sent concurrently gave different probabilities and identical requests sent one after another gave identical ones, so the experiment sends one request at a time unless told otherwise (the header says how this run was sent). The repeated pass (an identical request, sent later in the same run) selected the same candidate for 286/286 = 1.000 and gave identical probabilities for 286/286 = 1.000. A later run, or one made while the server is busier, may differ, and a later build of the model server is a different system.

Contains data retrieved from Wikidata (CC0), QUDT (CC BY 4.0, attribution to QUDT.org) and the Ontology of units of Measure (CC BY 4.0) via the EBI Ontology Lookup Service, and answers from Qwen3.8-27B served by PSNC.
