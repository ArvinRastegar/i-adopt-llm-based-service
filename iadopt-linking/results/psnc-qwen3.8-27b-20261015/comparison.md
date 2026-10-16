# Qwen3.8-27B against the Qwen3 baseline: psnc-qwen3.8-27b-20261015

Primary pass `choice-a1` of run `psnc-qwen3.8-27b-20261015` against baseline `baseline-20261014`, on the same slots and the same candidates. Differences are in top-1 correct slots (Qwen3.8-27B minus Qwen3); intervals resample whole terms.

## Paired counts by population

| Population | Slots | Both right | Only Qwen3.8-27B | Only Qwen3 | Both wrong | Qwen3.8-27B minus Qwen3 | McNemar p |
|---|---|---|---|---|---|---|---|
| reachable | 284 | 165 | 41 | 29 | 49 | +0.042 (95% interval -0.073 to +0.172) | 0.1882 |
| all | 294 | 165 | 41 | 29 | 59 | +0.041 (95% interval -0.069 to +0.171) | 0.1882 |
| gold:wikidata | 218 | 130 | 39 | 8 | 41 | +0.142 (95% interval +0.008 to +0.300) | <0.0001 |
| gold:qudt | 61 | 30 | 2 | 21 | 8 | -0.311 (95% interval -0.522 to -0.138) | <0.0001 |
| gold:om | 5 | 5 | 0 | 0 | 0 | +0.000 (95% interval +0.000 to +0.000) | 1.0000 |

## Counted once per term (exploratory; reachable and by gold vocabulary)

| Population | Terms | Qwen3.8-27B better | Qwen3 better | Equal | Sign-test p | Term-averaged difference |
|---|---|---|---|---|---|---|
| reachable | 145 | 19 | 16 | 110 | 0.7359 | +0.012 |
| all | 153 | 19 | 16 | 118 | 0.7359 | +0.012 |
| gold:wikidata | 116 | 17 | 2 | 97 | 0.0007 | +0.096 |
| gold:qudt | 31 | 2 | 14 | 15 | 0.0042 | -0.306 |
| gold:om | 1 | 0 | 0 | 1 | 1.0000 | +0.000 |

## How slots moved between categories (Qwen3 to Qwen3.8-27B)

| Qwen3 > Qwen3.8-27B | Slots |
|---|---|
| correct > correct | 165 |
| correct > wrong source | 21 |
| correct > wrong within source | 8 |
| gold not retrieved > gold not retrieved | 3 |
| no candidates > no candidates | 6 |
| unreachable vocabulary > unreachable vocabulary | 10 |
| wrong source > correct | 17 |
| wrong source > wrong source | 9 |
| wrong within source > correct | 24 |
| wrong within source > wrong source | 4 |
| wrong within source > wrong within source | 27 |

## Every pass against Qwen3 (reachable slots)

| Pass | Strict top-1 | With QUDT/Wikidata equivalence |
|---|---|---|
| `choice-a1` | 206/284 = 0.725 | 218/284 = 0.768 |
| `choice-a1-repeat` | 206/284 = 0.725 | 218/284 = 0.768 |
| `choice-a2-reversed` | 202/284 = 0.711 | 215/284 = 0.757 |
| `choice-a3` | 201/284 = 0.708 | 210/284 = 0.739 |
| Qwen3 baseline | 194/284 = 0.683 | 202/284 = 0.711 |

## Slots only Qwen3.8-27B got right (41)

3 of these 41 slots would not be misses for Qwen3 if a QUDT kind and its `wikidataMatch` Wikidata item counted as one entity (exploratory).

- `water` (ConcCarbon.ttl): gold wikidata:Q283; Qwen3.8-27B 'water' (wikidata:Q283, 0.99); Qwen3 'Mass Concentration of Water' (qudt:MassConcentrationOfWater, 1.00)
- `blood` (C21_BloodPressure.ttl): gold wikidata:Q7873; Qwen3.8-27B 'blood' (wikidata:Q7873, 0.99); Qwen3 'Diastolic Blood Pressure' (qudt:DiastolicBloodPressure, 1.00)
- `status` (C22_SmokingStatus.ttl): gold wikidata:Q813912; Qwen3.8-27B 'condition' (wikidata:Q813912, 0.20); Qwen3 'status' (wikidata:Q11424100, 0.99)
- `cloud` (C3_CloudCover.ttl): gold wikidata:Q8074; Qwen3.8-27B 'cloud' (wikidata:Q8074, 0.81); Qwen3 'Amount of Cloud Cover' (qudt:AmountOfCloudCover, 1.00)
- `atmospheric boundary layer` (C4_ABLHeight.ttl): gold wikidata:Q1757268; Qwen3.8-27B 'planetary boundary layer' (wikidata:Q1757268, 0.26); Qwen3 'Atmospheric boundary layer top height in South Africa: measurements with lidar and radiosonde compared to three atmospheric models' (wikidata:Q58212201, 0.98)
- `optical thickness` (C5_OpticalThickness.ttl): gold wikidata:Q890809; Qwen3.8-27B 'optical depth' (wikidata:Q890809, 0.83); Qwen3 'Optical Thickness and Size Distribution of Dust Aerosol Particles in Inner Mongolia' (wikidata:Q106333828, 1.00)
- `water` (IceConcentration.ttl): gold wikidata:Q283; Qwen3.8-27B 'water' (wikidata:Q283, 0.88); Qwen3 'Mass Concentration of Water' (qudt:MassConcentrationOfWater, 1.00)
- `atmosphere` (PMER.ttl): gold wikidata:Q8104; Qwen3.8-27B 'atmosphere' (wikidata:Q8104, 0.39); Qwen3 'standard atmosphere' (wikidata:Q177974, 1.00)
- `ground` (PrecRate.ttl): gold wikidata:Q36133; Qwen3.8-27B 'soil' (wikidata:Q36133, 0.53); Qwen3 'ground' (wikidata:Q6501125, 0.92)
- `water` (PrecRate.ttl): gold wikidata:Q283; Qwen3.8-27B 'water' (wikidata:Q283, 0.92); Qwen3 'Mass Fraction of Water' (qudt:MassFractionOfWater, 0.35)
- `earth` (C10_PeakGroundAcceleration.ttl): gold wikidata:Q2; Qwen3.8-27B 'Earth' (wikidata:Q2, 0.87); Qwen3 'earthquake' (wikidata:Q7944, 0.97)
- `water` (GLVAR.ttl): gold wikidata:Q283; Qwen3.8-27B 'water' (wikidata:Q283, 0.99); Qwen3 'Water Solubility' (qudt:WaterSolubility, 1.00)

## Slots only Qwen3 got right (29)

8 of these 29 slots would not be misses for Qwen3.8-27B if a QUDT kind and its `wikidataMatch` Wikidata item counted as one entity (exploratory).

- `height` (C7_TrunkCharHeight.ttl): gold qudt:Height; Qwen3.8-27B 'height' (om:Height, 0.57); Qwen3 'Height' (qudt:Height, 1.00)
- `mass fraction` (CONCVAR.ttl): gold qudt:MassFraction; Qwen3.8-27B 'mass fraction' (om:MassFraction, 0.44); Qwen3 'Mass Fraction' (qudt:MassFraction, 1.00)
- `person` (C21_BloodPressure.ttl): gold wikidata:Q215627; Qwen3.8-27B 'Person' (wikidata:Q105964318, 0.44); Qwen3 'person' (wikidata:Q215627, 1.00)
- `pressure` (C21_BloodPressure.ttl): gold qudt:Pressure; Qwen3.8-27B 'pressure' (wikidata:Q39552, 0.69); Qwen3 'Pressure' (qudt:Pressure, 1.00)
- `time` (HRV.ttl): gold qudt:Time; Qwen3.8-27B 'time' (om:Time, 0.69); Qwen3 'Time' (qudt:Time, 1.00)
- `person` (lactate.ttl): gold wikidata:Q215627; Qwen3.8-27B 'Person' (wikidata:Q105964318, 0.33); Qwen3 'person' (wikidata:Q215627, 1.00)
- `amount of substance concentration` (lactate.ttl): gold qudt:AmountOfSubstanceConcentration; Qwen3.8-27B 'amount of substance concentration' (om:AmountOfSubstanceConcentration, 0.59); Qwen3 'Amount of Substance of Concentration' (qudt:AmountOfSubstanceConcentration, 1.00)
- `person` (weight without clothes.ttl): gold wikidata:Q215627; Qwen3.8-27B 'Person' (wikidata:Q105964318, 0.50); Qwen3 'person' (wikidata:Q215627, 0.99)
- `volume fraction` (BPFOS.ttl): gold qudt:VolumeFraction; Qwen3.8-27B 'volume fraction' (wikidata:Q909482, 0.34); Qwen3 'Volume Fraction' (qudt:VolumeFraction, 1.00)
- `mass` (C23_NanoparticleDose.ttl): gold qudt:Mass; Qwen3.8-27B 'mass' (om:Mass, 0.53); Qwen3 'Mass' (qudt:Mass, 0.99)
- `mass fraction` (ConcEndosulfaneOstrea.ttl): gold qudt:MassFraction; Qwen3.8-27B 'mass fraction' (wikidata:Q899138, 0.44); Qwen3 'Mass Fraction' (qudt:MassFraction, 1.00)
- `mass fraction` (DehpNail.ttl): gold qudt:MassFraction; Qwen3.8-27B 'mass fraction' (om:MassFraction, 0.58); Qwen3 'Mass Fraction' (qudt:MassFraction, 1.00)

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

