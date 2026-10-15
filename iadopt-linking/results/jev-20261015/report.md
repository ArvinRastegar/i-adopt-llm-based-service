# Jev (typesafe/jev-1.13) over the baseline's pooled Wikidata, QUDT and OM candidates

Run `jev-20261015`, report built 2026-10-15T12:19:18+00:00. Requests sent 2026-10-15T10:33:21+00:00 to 2026-10-15T10:35:21+00:00: 1430 requests answered (1,685,133 input tokens), $0.070812294 reported by OpenRouter in `usage.cost` for all 1431 stored responses (not reconciled with the account's billing). Corpus I-ADOPT v2.0.1 (2598bf91fa). Candidates and Qwen3 scores come from baseline `baseline-20261014` (scores `ced730904872`), not from new retrieval. Evidence code `8a5e2e78f408`, all code `1bf445186a53`, prompt `a6042e8c2d20`. Model requested `typesafe/jev-1.13`, served as `typesafe/jev-1.13-20260917` by TypeSafe.

Jev is a typed-decision model: it answers a question with a probability for each option instead of generating text. Each slot's pooled candidates, exactly as the Qwen3 baseline saw them (label and description capped at 300 characters, no source tag), were put to Jev as one Choice question with an extra `none_of_these` option. The primary result, fixed in the run's plan and in DECISIONS L-012 before the first request (no commit predates the run), is the first Choice pass (a seeded candidate order) with the forced choice: the candidate with the highest probability, `none_of_these` ignored, ties decided as in the baseline. The same pool was also asked, to see how much the result depends on order and on chance: an identical repeat of the primary request; the reverse order; another candidate order (shuffle-b); one yes/no question per candidate (TypeSafe's re-ranking pattern). Everything except the primary result is exploratory.

## Result

| Population | Slots | Distinct terms | Jev top-1 | Jev mean per variable | Qwen3 top-1 | Jev minus Qwen3 | McNemar p |
|---|---|---|---|---|---|---|---|
| Gold in Wikidata, QUDT or OM (primary) | 284 | 145 | 213/284 = 0.750 | 0.737 | 194/284 = 0.683 | +0.067 (95% interval -0.040 to +0.182) | 0.0248 |
| All controlled-vocabulary slots | 294 | 153 | 213/294 = 0.724 | 0.713 | 194/294 = 0.660 | +0.065 (95% interval -0.038 to +0.179) | 0.0248 |
| Gold in Wikidata | 218 | 116 | 174/218 = 0.798 | 0.771 | 138/218 = 0.633 | +0.165 (95% interval +0.054 to +0.295) | <0.0001 |
| Gold in QUDT | 61 | 31 | 34/61 = 0.557 | 0.557 | 51/61 = 0.836 | -0.279 (95% interval -0.483 to -0.104) | 0.0005 |
| Gold in OM | 5 | 1 | 5/5 = 1.000 | 1.000 | 5/5 = 1.000 | +0.000 (95% interval +0.000 to +0.000) | 1.0000 |
| Gold in another vocabulary (not searchable here) | 10 | 8 | 0/10 = 0.000 | 0.000 | 0/10 = 0.000 | - | - |

Both systems were judged by the same code on the same slots and the same candidates. The interval comes from resampling whole terms (153 terms ask 294 slots, so slots are not independent evidence); McNemar's p treats slots as independent and is therefore optimistic. Mean per variable is the manuscript's style of measure.

## Jev against Qwen3 on the reachable slots

|  | Qwen3 right | Qwen3 wrong |
|---|---|---|
| Jev right | 171 | 42 |
| Jev wrong | 23 | 48 |

Strict identity: Jev minus Qwen3 +0.067 (95% interval -0.040 to +0.182); exact McNemar p 0.0248.
Counted once per term instead of once per slot (post hoc: added after the first results, because slots cluster by term): Jev better on 17 of 145 terms, Qwen3 better on 16, equal on 112 (exact sign test p 1.0000; term-averaged difference +0.025).
If a QUDT kind and its `wikidataMatch` item counted as one entity: Jev minus Qwen3 +0.123 (95% interval +0.033 to +0.229); Jev better on 17 of 145 terms, Qwen3 better on 5, equal on 123 (exact sign test p 0.0169; term-averaged difference +0.084). (An undecided question of the project; strict identity is the headline.)

## Where the misses come from (reachable slots)

| Outcome | Jev slots | Jev share | Qwen3 slots | Qwen3 share |
|---|---|---|---|---|
| correct | 213 | 0.750 | 194 | 0.683 |
| wrong within source | 29 | 0.102 | 55 | 0.194 |
| wrong source | 33 | 0.116 | 26 | 0.092 |
| gold not retrieved | 3 | 0.011 | 3 | 0.011 |
| no candidates | 6 | 0.021 | 6 | 0.021 |

## Diagnostics (reachable slots)

| Question | Jev | Qwen3 |
|---|---|---|
| Is the gold term among the pooled candidates? | 275/284 = 0.968 | 275/284 = 0.968 |
| Does the scorer pick it when it is? | 213/275 = 0.775 | 194/275 = 0.705 |
| Is the chosen candidate from the gold's own source? (slots with any candidate) | 244/278 = 0.878 | 252/278 = 0.906 |
| Top-1 if the source were known (best score within the gold's source) | 245/284 = 0.863 | 216/284 = 0.761 |
| Top-1 of the gold source's own first hit (no scorer) | 204/284 = 0.718 | 204/284 = 0.718 |
| Top-1 if a QUDT kind and its `wikidataMatch` item counted as the same entity | 237/284 = 0.835 | 202/284 = 0.711 |
| Ceiling for a system that sees only the term (majority gold per term) | 268/284 = 0.944 | 268/284 = 0.944 |

On the Wikidata-gold slots (post hoc, L-014), picking the best-scoring Wikidata candidate only (for Jev the probabilities of the pooled question, restricted to Wikidata; for Qwen3 the manuscript's "reranker alone" setting): Jev 181/218 = 0.830, Qwen3 153/218 = 0.702.

## Which source wins (reachable slots with any candidate)

Rows are the gold's source, columns the source of the selected candidate.

**Jev**

| Gold in \ chosen from | wikidata | qudt | om |
|---|---|---|---|
| wikidata | 204 | 8 | 0 |
| qudt | 25 | 35 | 1 |
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
| `choice-a1` | choice | shuffle-a | 213/284 = 0.750 | 213/294 = 0.724 | - | +0.067 (95% interval -0.040 to +0.182) |
| `choice-a1-repeat` | choice | shuffle-a | 212/284 = 0.746 | 212/294 = 0.721 | 281/286 = 0.983 | +0.063 (95% interval -0.044 to +0.178) |
| `choice-a2-reversed` | choice | reverse-a | 212/284 = 0.746 | 212/294 = 0.721 | 252/286 = 0.881 | +0.063 (95% interval -0.049 to +0.186) |
| `choice-a3` | choice | shuffle-b | 210/284 = 0.739 | 210/294 = 0.714 | 259/286 = 0.906 | +0.056 (95% interval -0.042 to +0.161) |
| `noul-n1` | noul | shuffle-a | 177/284 = 0.623 | 177/294 = 0.602 | 184/286 = 0.643 | -0.060 (95% interval -0.174 to +0.053) |

Identical request sent twice (`choice-a1` and `choice-a1-repeat`), 286 slots: the same candidate was selected for 281/286 = 0.983; every candidate's probability was identical for 78/286 = 0.273; the largest difference in a candidate's probability was 0.12 (mean 0.0033).

Accuracy totals hide offsetting changes (post hoc, L-014). Reachable slots whose correctness differs from the primary pass: `choice-a1-repeat` changes 3 of 278 (1 become right, 2 wrong); `choice-a2-reversed` changes 31 of 278 (15 become right, 16 wrong); `choice-a3` changes 23 of 278 (10 become right, 13 wrong). All the Choice passes select the same candidate on 232 of 278 slots and are equally right or wrong on 236.

Does Jev favour the first option it is shown? Share of slots with at least two candidates in which the selected candidate was shown first, against the share if position played no part (the mean of 1/n):

| Pass | Slots | Selected candidate shown first | Expected number if position played no part |
|---|---|---|---|
| `choice-a1` | 281 | 31 | 31.5 |
| `choice-a1-repeat` | 281 | 32 | 31.5 |
| `choice-a2-reversed` | 281 | 28 | 31.5 |
| `choice-a3` | 281 | 27 | 31.5 |

Exploratory: averaging the probabilities of the Choice passes (`choice-a1`, `choice-a1-repeat`, `choice-a2-reversed`, `choice-a3`) and selecting from the average gives 213/284 = 0.750 on the reachable slots; minus Qwen3 +0.067 (95% interval -0.040 to +0.183). An identical repeat counts as a pass here, so its order weighs twice.

Exploratory: the pointwise arm (`noul-n1`, one yes/no question per candidate) against the primary Choice pass: -0.127 (95% interval -0.188 to -0.073) (pointwise minus primary); it disagrees with the primary pass on 68 slots.

## Jev's own `none_of_these` (primary pass, reachable slots)

| Question | Answer |
|---|---|
| Slots asked | 278 |
| `none_of_these` was Jev's choice although the gold was among the candidates | 16/275 = 0.058 |
| `none_of_these` was Jev's choice when the gold was not among the candidates (a right abstention) | 2/3 = 0.667 |
| Slots Jev answered with a candidate | 260/278 = 0.935 |
| Precision of the forced choice on those | 201/260 = 0.773 |
| Forced choice correct on the slots where Jev chose `none_of_these` | 12/18 = 0.667 |
| Refusals made at a `confidence` of 0.9 or more | 2 of 18 |

The headline always uses the forced choice, as the baseline always picks its top candidate; this table shows what honouring Jev's refusals would change. On the 8 asked slots whose gold lies in a vocabulary the sources cannot reach, Jev chose `none_of_these` for 4; they are left out above because a pool may hold the same concept under another identifier.

## What the confidence is worth (reachable slots)

Any number that says how much to trust a selection can decide when to hand a slot to a stronger model. AUROC is the chance that a right answer is more confident than a wrong one (0.5 is no information). The calibration gap and Brier score treat the value as a probability of being right. All of this is exploratory and uses the data it describes; choose a threshold on held-out terms before relying on one.

| Signal | Slots answered | AUROC | Calibration gap | Brier |
|---|---|---|---|---|
| Jev: probability of the selected candidate | 278 | 0.812 | 0.045 | 0.144 |
| Jev: its own `confidence` value | 278 | 0.783 | 0.060 | 0.155 |
| Jev: top-two probability margin | 273 | 0.834 | - | - |
| Jev: probability not given to `none_of_these` | 278 | 0.567 | - | - |
| Jev: share of the Choice passes that agree | 278 | 0.625 | - | - |
| Qwen3: score of the selected candidate | 278 | 0.587 | 0.283 | 0.279 |
| Qwen3: top-two score margin | 273 | 0.604 | - | - |

TypeSafe documents Jev's `confidence` as (p_max - 1/n) / (1 - 1/n) over its n options. That holds within 0.01 for 1144/1144 = 1.000 of the Choice responses (largest difference 0.005; this check is post hoc, L-014), so where it holds `confidence` is a function of the top probability and the number of options and carries no other information.

Jev's `confidence` describes its own choice. For 18 reachable slots that choice was `none_of_these` (2 of them at 0.9 or more), so for those slots it measures confidence in the refusal, not in the forced candidate whose correctness is scored here.

Coverage and precision when only selections at or above a threshold are answered (reachable slots; the rest would go to a stronger model):

| Threshold | Jev probability: answered | precision | Jev `confidence`: answered | precision | Qwen3 score: answered | precision |
|---|---|---|---|---|---|---|
| 0.000 | 278/284 = 0.979 | 213/278 = 0.766 | 278/284 = 0.979 | 213/278 = 0.766 | 278/284 = 0.979 | 194/278 = 0.698 |
| 0.100 | 276/284 = 0.972 | 213/276 = 0.772 | 277/284 = 0.975 | 212/277 = 0.765 | 278/284 = 0.979 | 194/278 = 0.698 |
| 0.200 | 273/284 = 0.961 | 211/273 = 0.773 | 276/284 = 0.972 | 212/276 = 0.768 | 278/284 = 0.979 | 194/278 = 0.698 |
| 0.300 | 270/284 = 0.951 | 210/270 = 0.778 | 271/284 = 0.954 | 208/271 = 0.768 | 277/284 = 0.975 | 194/277 = 0.700 |
| 0.400 | 255/284 = 0.898 | 203/255 = 0.796 | 251/284 = 0.884 | 200/251 = 0.797 | 275/284 = 0.968 | 194/275 = 0.705 |
| 0.500 | 229/284 = 0.806 | 188/229 = 0.821 | 227/284 = 0.799 | 185/227 = 0.815 | 275/284 = 0.968 | 194/275 = 0.705 |
| 0.600 | 205/284 = 0.722 | 175/205 = 0.854 | 198/284 = 0.697 | 169/198 = 0.854 | 275/284 = 0.968 | 194/275 = 0.705 |
| 0.700 | 182/284 = 0.641 | 163/182 = 0.896 | 176/284 = 0.620 | 156/176 = 0.886 | 274/284 = 0.965 | 194/274 = 0.708 |
| 0.800 | 157/284 = 0.553 | 146/157 = 0.930 | 152/284 = 0.535 | 141/152 = 0.928 | 271/284 = 0.954 | 194/271 = 0.716 |
| 0.900 | 116/284 = 0.408 | 112/116 = 0.966 | 116/284 = 0.408 | 110/116 = 0.948 | 270/284 = 0.951 | 194/270 = 0.719 |
| 0.950 | 87/284 = 0.306 | 85/87 = 0.977 | 82/284 = 0.289 | 81/82 = 0.988 | 262/284 = 0.923 | 192/262 = 0.733 |
| 0.990 | 52/284 = 0.183 | 52/52 = 1.000 | 50/284 = 0.176 | 50/50 = 1.000 | 226/284 = 0.796 | 167/226 = 0.739 |
| 0.999 | 28/284 = 0.099 | 28/28 = 1.000 | 28/284 = 0.099 | 28/28 = 1.000 | 91/284 = 0.320 | 67/91 = 0.736 |

Handing the slots below a threshold on Jev's selected probability to a stronger model (the split into fixable and unfixable errors is post hoc, L-014). Of the escalated slots, the right ones are work a stronger model repeats; the wrong ones with the gold among the candidates are errors it could fix; the wrong ones without are beyond any scorer. The ceiling assumes the stronger model fixes every fixable error, so it is an upper bound, not a prediction:

| Threshold | Escalated | Right among them | Wrong, fixable | Wrong, unfixable | Precision of the rest | Ceiling |
|---|---|---|---|---|---|---|
| 0.000 | 6/284 = 0.021 | 0 | 0 | 6 | 213/278 = 0.766 | 213/284 = 0.750 |
| 0.100 | 8/284 = 0.028 | 0 | 0 | 8 | 213/276 = 0.772 | 213/284 = 0.750 |
| 0.200 | 11/284 = 0.039 | 2 | 1 | 8 | 211/273 = 0.773 | 214/284 = 0.754 |
| 0.300 | 14/284 = 0.049 | 3 | 3 | 8 | 210/270 = 0.778 | 216/284 = 0.761 |
| 0.400 | 29/284 = 0.102 | 10 | 11 | 8 | 203/255 = 0.796 | 224/284 = 0.789 |
| 0.500 | 55/284 = 0.194 | 25 | 22 | 8 | 188/229 = 0.821 | 235/284 = 0.827 |
| 0.600 | 79/284 = 0.278 | 38 | 33 | 8 | 175/205 = 0.854 | 246/284 = 0.866 |
| 0.700 | 102/284 = 0.359 | 50 | 43 | 9 | 163/182 = 0.896 | 256/284 = 0.901 |
| 0.800 | 127/284 = 0.447 | 67 | 51 | 9 | 146/157 = 0.930 | 264/284 = 0.930 |
| 0.900 | 168/284 = 0.592 | 101 | 58 | 9 | 112/116 = 0.966 | 271/284 = 0.954 |
| 0.950 | 197/284 = 0.694 | 128 | 60 | 9 | 85/87 = 0.977 | 273/284 = 0.961 |
| 0.990 | 232/284 = 0.817 | 161 | 62 | 9 | 52/52 = 1.000 | 275/284 = 0.968 |
| 0.999 | 256/284 = 0.901 | 185 | 62 | 9 | 28/28 = 1.000 | 275/284 = 0.968 |

Reliability of Jev's selected probability (bins of width 0.1):

| Bin | Slots | Right | Mean probability |
|---|---|---|---|
| 0.000-0.100 | 2 | 0 | 0.035 |
| 0.100-0.200 | 3 | 2 | 0.150 |
| 0.200-0.300 | 3 | 1 | 0.243 |
| 0.300-0.400 | 15 | 7 | 0.359 |
| 0.400-0.500 | 26 | 15 | 0.450 |
| 0.500-0.600 | 24 | 13 | 0.544 |
| 0.600-0.700 | 23 | 12 | 0.649 |
| 0.700-0.800 | 25 | 17 | 0.747 |
| 0.800-0.900 | 41 | 34 | 0.848 |
| 0.900-1.000 | 116 | 112 | 0.968 |

Jev reports probabilities rounded to two decimals, so a threshold moves in steps of 0.01 and many slots share a value near 1.

## By slot, role and term length

| Group | Slots | Jev top-1 | Qwen3 top-1 | Gold in pool |
|---|---|---|---|---|
| role:member | 68 | 58/68 = 0.853 | 46/68 = 0.676 | 64/65 = 0.985 |
| role:top-level | 226 | 155/226 = 0.686 | 148/226 = 0.655 | 211/219 = 0.963 |
| slot:hasContextObject | 11 | 9/11 = 0.818 | 5/11 = 0.455 | 11/11 = 1.000 |
| slot:hasMatrix | 55 | 40/55 = 0.727 | 32/55 = 0.582 | 55/55 = 1.000 |
| slot:hasObjectOfInterest | 118 | 93/118 = 0.788 | 79/118 = 0.669 | 107/113 = 0.947 |
| slot:hasProperty | 101 | 63/101 = 0.624 | 70/101 = 0.693 | 94/96 = 0.979 |
| slot:hasStatisticalModifier | 9 | 8/9 = 0.889 | 8/9 = 0.889 | 8/9 = 0.889 |
| words:1 | 182 | 132/182 = 0.725 | 117/182 = 0.643 | 176/178 = 0.989 |
| words:2 | 94 | 65/94 = 0.691 | 62/94 = 0.660 | 81/88 = 0.920 |
| words:3+ | 18 | 16/18 = 0.889 | 15/18 = 0.833 | 18/18 = 1.000 |

## Most repeated terms

A few terms are asked many times, so their outcome weighs on every slot-weighted figure above.

| Term | Slots | Jev wrong | Qwen3 wrong |
|---|---|---|---|
| `person` | 26 | 14 | 14 |
| `water` | 12 | 0 | 12 |
| `count` | 11 | 1 | 0 |
| `urban area` | 11 | 2 | 10 |
| `atmosphere` | 9 | 1 | 1 |

## Examples of misses

Up to 6 per outcome, first by slot id. Gold is vocabulary:identifier.

**wrong within source** (29 slots)

- `enclosure` (C14_FeralfreeEnclosureArea.ttl) -> 'enclosure' (wikidata:Q5375483, 0.65); gold wikidata:Q858177
- `oxygen` (C16OxygenMetabolicRate.ttl) -> 'oxygen' (wikidata:Q629, 0.87); gold wikidata:Q5203615
- `ground` (C8_CWDFraction.ttl) -> "Earth's surface" (wikidata:Q1349417, 0.38); gold wikidata:Q36133
- `ground` (C9_FoliageCover.ttl) -> "Earth's surface" (wikidata:Q1349417, 0.79); gold wikidata:Q36133
- `person` (BMI.ttl) -> 'person' (wikidata:Q215627, 0.44); gold wikidata:Q5
- `person` (Bloodtype.ttl) -> 'person' (wikidata:Q215627, 0.11); gold wikidata:Q5

**wrong source** (33 slots)

- `mass` (C16OxygenMetabolicRate.ttl) -> 'mass' (om:Mass, 0.37); gold qudt:Mass
- `height` (C7_TrunkCharHeight.ttl) -> 'height' (wikidata:Q208826, 0.61); gold qudt:Height
- `mass fraction` (CONCVAR.ttl) -> 'mass fraction' (wikidata:Q899138, 0.91); gold qudt:MassFraction
- `index` (BMI.ttl) -> 'Body Mass Index' (qudt:BodyMassIndex, 0.51); gold wikidata:Q1738991
- `blood` (C21_BloodPressure.ttl) -> 'Systolic Blood Pressure' (qudt:SystolicBloodPressure, 0.45); gold wikidata:Q7873
- `pressure` (C21_BloodPressure.ttl) -> 'pressure' (wikidata:Q39552, 0.54); gold qudt:Pressure

**gold not retrieved** (3 slots)

- `char` (C7_TrunkCharHeight.ttl) -> 'char' (wikidata:Q478251, 0.05); gold wikidata:Q11567121
- `standard deviation` (HRV.ttl) -> 'standard deviation' (wikidata:Q159375, 0.68); gold wikidata:Q175199
- `state` (StateOfPipeConnection.ttl) -> 'State of Charge' (qudt:StateOfCharge, 0.02); gold wikidata:Q813912

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
| every request answered | pass | 1430/1430 requests answered (0 needed an HTTP retry; 1 stored responses were rejected as invalid answers and asked again) |
| stored requests regenerate from the plan | pass | 1430/1430 stored requests equal the ones this tree builds |
| stored response hashes verify | pass | 1431 stored responses re-hashed |
| every response is a valid answer | pass | 1430 responses read strictly; probabilities sum to 1 within 0.010 at worst |
| one model snapshot served | pass | served by TypeSafe as typesafe/jev-1.13-20260917 (requested typesafe/jev-1.13) |
| spend fully accounted and within the cap | pass | $0.070812294 spent over 1430 answers and 1 rejected responses, every stored response included; cap $2 |
| primary pass complete | pass | 286/286 requests of choice-a1 answered |

## Notes

- Jev saw, for every candidate, the label and the description (capped at 300 characters): the information the Qwen3 reranker saw, with no vocabulary, identifier, rank or score. The term and the variable's definition are the baseline's.
- 8 slots have no candidate; they were not asked and count as misses, as in the baseline.
- 17 label-only nodes and 2 non-vocabulary IRIs are outside the population, as in the baseline.
- 10 slots have their gold in a vocabulary these sources cannot reach; they count as misses in the "All controlled-vocabulary slots" row and are left out of the primary population (the reachable slots).
- Jev reports probabilities rounded to two decimals: 1 of 281 primary selections that have a runner-up tie with it and were decided by the baseline's rule (source order, then rank).
- Gold terms are the corpus labels of the gold nodes; for QUDT slots they are often QUDT's own labels, so QUDT accuracy on these terms may overstate accuracy on LLM-predicted wording.
- The service answers live: another run would not give identical probabilities (see the repeated pass), and a later release of the model is a different model.
- 1 stored response(s) were rejected by the validity rule and the request was asked again, as the protocol says (1 x the choice is not the option with the highest probability); rejected responses stay in responses.jsonl.

Contains data retrieved from Wikidata (CC0), QUDT (CC BY 4.0, attribution to QUDT.org) and the Ontology of units of Measure (CC BY 4.0) via the EBI Ontology Lookup Service, and answers from TypeSafe's Jev model obtained through OpenRouter.
