# Baseline: Qwen3-Reranker-0.6B over pooled Wikidata, QUDT and OM candidates

Run `baseline-20261014`, report built 2026-10-14T14:09:11+00:00. Evidence retrieved 2026-10-14T13:42:09+00:00 to 2026-10-14T13:57:04+00:00. Corpus I-ADOPT v2.0.1 (2598bf91fa). Evidence code `c88a30e9c074`, all code `8a8ccc3bb83d`. Reranker `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` (revision `6a5829f507`), scored on mps (294 slots).

For every gold slot whose IRI names a controlled-vocabulary term, the slot's term was searched in Wikidata, QUDT quantity kinds and OM quantities (at most 10 candidates each), the pooled candidates were scored by the existing Qwen3 reranker with the variable's definition as context, and the best-scoring candidate was compared with the gold identity (vocabulary and local identifier).

## Result

| Population | Slots | Distinct terms | Top-1 accuracy | Mean per variable |
|---|---|---|---|---|
| All controlled-vocabulary slots | 294 | 153 | 194/294 = 0.660 | 0.641 |
| Gold in Wikidata, QUDT or OM | 284 | 145 | 194/284 = 0.683 | 0.661 |
| Gold in Wikidata | 218 | 116 | 138/218 = 0.633 | 0.603 |
| Gold in QUDT | 61 | 31 | 51/61 = 0.836 | 0.836 |
| Gold in OM | 5 | 1 | 5/5 = 1.000 | 1.000 |
| Gold in another vocabulary (not searchable here) | 10 | 8 | 0/10 = 0.000 | 0.000 |

Mean per variable is the manuscript's style of measure (accuracy per variable, then averaged); slots repeat terms, so the distinct-term column is the number of independent questions asked.

## Comparison with the manuscript

The manuscript's entity-linking table reports URI accuracy as the average accuracy per variable, over cases that have a Wikidata ground-truth link. The first two rows below use that measure on every Wikidata-gold slot of this report, members of ratios and source/target parts included; whether the manuscript counted exactly these cases is not established here.

| Measure (Wikidata gold) | Slots correct | Mean per variable | Manuscript |
|---|---|---|---|
| First Wikidata hit, no reranker (naive) | 138/218 = 0.633 | 0.627 | 0.656 |
| Qwen3 reranker on the Wikidata candidates alone | 153/218 = 0.702 | 0.677 | 0.759 |
| Qwen3 reranker on all three sources pooled (this baseline) | 138/218 = 0.633 | 0.603 | - |

The manuscript's figures are quoted, not reproduced: their inputs are not in this repository. A difference is not explained by this run. Candidates here come from the live services on the run date, Wikidata is asked for ten candidates where the API default is seven, and the slot population and prompts may differ from the manuscript's.

## Where the misses come from (reachable slots)

| Outcome | Slots | Share |
|---|---|---|
| correct | 194 | 0.683 |
| wrong within source | 55 | 0.194 |
| wrong source | 26 | 0.092 |
| gold not retrieved | 3 | 0.011 |
| no candidates | 6 | 0.021 |

## Diagnostics (reachable slots)

| Question | Answer |
|---|---|
| Is the gold term among the pooled candidates? | 275/284 = 0.968 |
| Does the reranker pick it when it is? | 194/275 = 0.705 |
| Is the chosen candidate from the gold's own source? (slots with any candidate) | 252/278 = 0.906 |
| Top-1 if the source were known (best score within the gold's source) | 216/284 = 0.761 |
| Top-1 of the gold source's own first hit (no reranker) | 204/284 = 0.718 |
| Top-1 if a QUDT kind and its `wikidataMatch` item counted as the same entity | 202/284 = 0.711 |
| Ceiling for a system that sees only the term (majority gold per term), reachable slots | 268/284 = 0.944 |
| Ceiling for such a system over all slots | 278/294 = 0.946 |
| Selections decided by a top-two score margin below 1e-3 (slots with 2+ candidates) | 53/281 = 0.189 |

## Which source wins (reachable slots with any candidate)

Rows are the gold's source, columns the source of the selected candidate.

| Gold in \ chosen from | wikidata | qudt | om |
|---|---|---|---|
| wikidata | 193 | 19 | 0 |
| qudt | 6 | 54 | 1 |
| om | 0 | 0 | 5 |

## Confidence threshold (all slots)

A selection counts as an answer only if its score reaches the threshold. Scores saturate near 1, so the grid continues above 0.9.

| Threshold | Answered | Precision when answered |
|---|---|---|
| 0.000 | 286/294 = 0.973 | 194/286 = 0.678 |
| 0.100 | 286/294 = 0.973 | 194/286 = 0.678 |
| 0.200 | 285/294 = 0.969 | 194/285 = 0.681 |
| 0.300 | 284/294 = 0.966 | 194/284 = 0.683 |
| 0.400 | 282/294 = 0.959 | 194/282 = 0.688 |
| 0.500 | 281/294 = 0.956 | 194/281 = 0.690 |
| 0.600 | 281/294 = 0.956 | 194/281 = 0.690 |
| 0.700 | 280/294 = 0.952 | 194/280 = 0.693 |
| 0.800 | 277/294 = 0.942 | 194/277 = 0.700 |
| 0.900 | 276/294 = 0.939 | 194/276 = 0.703 |
| 0.950 | 268/294 = 0.912 | 192/268 = 0.716 |
| 0.990 | 230/294 = 0.782 | 167/230 = 0.726 |
| 0.999 | 93/294 = 0.316 | 67/93 = 0.720 |

## By slot, role and term length

| Group | Slots | Top-1 | Gold in pool |
|---|---|---|---|
| role:member | 68 | 46/68 = 0.676 | 64/65 = 0.985 |
| role:top-level | 226 | 148/226 = 0.655 | 211/219 = 0.963 |
| slot:hasContextObject | 11 | 5/11 = 0.455 | 11/11 = 1.000 |
| slot:hasMatrix | 55 | 32/55 = 0.582 | 55/55 = 1.000 |
| slot:hasObjectOfInterest | 118 | 79/118 = 0.669 | 107/113 = 0.947 |
| slot:hasProperty | 101 | 70/101 = 0.693 | 94/96 = 0.979 |
| slot:hasStatisticalModifier | 9 | 8/9 = 0.889 | 8/9 = 0.889 |
| words:1 | 182 | 117/182 = 0.643 | 176/178 = 0.989 |
| words:2 | 94 | 62/94 = 0.660 | 81/88 = 0.920 |
| words:3+ | 18 | 15/18 = 0.833 | 18/18 = 1.000 |

## Most repeated terms

A few terms are asked many times, so their outcome weighs on every slot-weighted figure above.

| Term | Slots | Wrong |
|---|---|---|
| `person` | 26 | 14 |
| `water` | 12 | 12 |
| `count` | 11 | 0 |
| `urban area` | 11 | 10 |
| `atmosphere` | 9 | 1 |

## Examples of misses

Up to 6 per outcome, first by slot id. Gold is vocabulary:identifier.

**wrong within source** (55 slots)

- `enclosure` (C14_FeralfreeEnclosureArea.ttl) -> 'enclosure' (wikidata:Q5375483, 1.00); gold wikidata:Q858177
- `oxygen` (C16OxygenMetabolicRate.ttl) -> 'Oxygen' (wikidata:Q133864131, 0.97); gold wikidata:Q5203615
- `ground` (C8_CWDFraction.ttl) -> 'Ground' (wikidata:Q36999948, 0.91); gold wikidata:Q36133
- `ground` (C9_FoliageCover.ttl) -> 'ground' (wikidata:Q6501125, 0.78); gold wikidata:Q36133
- `person` (BMI.ttl) -> 'person' (wikidata:Q215627, 1.00); gold wikidata:Q5
- `person` (Bloodtype.ttl) -> 'person' (wikidata:Q215627, 1.00); gold wikidata:Q5

**wrong source** (26 slots)

- `mass` (C16OxygenMetabolicRate.ttl) -> 'mass' (om:Mass, 0.98); gold qudt:Mass
- `water` (ConcCarbon.ttl) -> 'Mass Concentration of Water' (qudt:MassConcentrationOfWater, 1.00); gold wikidata:Q283
- `index` (BMI.ttl) -> 'Body Mass Index' (qudt:BodyMassIndex, 1.00); gold wikidata:Q1738991
- `blood` (C21_BloodPressure.ttl) -> 'Diastolic Blood Pressure' (qudt:DiastolicBloodPressure, 1.00); gold wikidata:Q7873
- `mass` (weight without clothes.ttl) -> 'Mass' (qudt:Mass, 0.98); gold wikidata:Q11423
- `cloud` (C3_CloudCover.ttl) -> 'Amount of Cloud Cover' (qudt:AmountOfCloudCover, 1.00); gold wikidata:Q8074

**gold not retrieved** (3 slots)

- `char` (C7_TrunkCharHeight.ttl) -> 'char' (wikidata:Q478251, 0.97); gold wikidata:Q11567121
- `standard deviation` (HRV.ttl) -> 'standard deviation' (wikidata:Q159375, 0.99); gold wikidata:Q175199
- `state` (StateOfPipeConnection.ttl) -> 'state' (wikidata:Q7275, 0.93); gold wikidata:Q813912

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
| every linked node accounted for | pass | 313 linked nodes in the gold: 294 scored, 17 label only, 2 non-vocabulary IRI |
| every key retrieved from every source | pass | 153 keys x 3 sources, all answered; 0 failed requests were repeated successfully and 15 needed an HTTP retry |
| every source answered for some keys | pass | wikidata 142/153, qudt 56/153, om 36/153 keys returned candidates (floor 5%) |
| stored response hashes verify | pass | 459 stored responses re-hashed |
| source releases recorded | pass | OM and QUDT release data probed (1 + 1 probes) |
| every slot scored | pass | 294 of 294 slots |
| scores finite and in [0, 1] | pass | 3473 scored candidates |
| at most 10 candidates per source | pass | pool size min 0, max 30 |
| one scoring configuration | pass | tomaarsen/Qwen3-Reranker-0.6B-seq-cls revision 6a5829f507, description cap 300; mps 294 slots |
| device consistency | pass | 63 slots rescored on cpu (53 with a top-two margin below 0.001, 10 others): max |difference| 1.59e-05 (tolerance 0.0001); rescoring on mps differed by at most 0.0e+00; 0 selections change on cpu (correct among the checked: 34 on mps, 34 on cpu) |

## Notes

- Retrieval is live: the candidates in candidates.jsonl are what the services returned between 2026-10-14T13:42:09+00:00 and 2026-10-14T13:57:04+00:00. They will drift, so a later run need not match; rescoring and reporting on the stored candidates is exactly repeatable.
- Descriptions are cut to 300 characters when a document is built; the longest retrieved description is 1544 characters for QUDT and 222 for Wikidata.
- OLS returned no description for any of the 110 OM candidates, so the reranker scores them from their label alone.
- Gold terms are the corpus labels of the gold nodes; for QUDT slots they are often QUDT's own labels, so QUDT accuracy on these terms may overstate accuracy on LLM-predicted wording.
- 10 slots have gold in a vocabulary these sources cannot reach (bioportal, envthes, iobc, iupac-goldbook, nalt, nerc); they count as misses in the first row of the result table and are left out of the others.
- Gold IRIs are compared by vocabulary and local identifier, never as strings. Read by pattern: 2 malformed IRIs, 1 in a variant path form.
- QUDT's endpoint also serves community-extension kinds: 2 of the 225 distinct kinds retrieved are defined outside the core quantity-kind vocabulary.
- 53 of 281 selections were decided by a top-two margin below 1e-3 and could change with the device or the batch. The cross-device check rescored 53 near-ties and 10 other slots on cpu: 0 selections changed; over those 63 slots the correct count was 34 on mps and 34 on cpu.

Contains data retrieved from Wikidata (CC0), QUDT (CC BY 4.0, attribution to QUDT.org) and the Ontology of units of Measure (CC BY 4.0) via the EBI Ontology Lookup Service.
