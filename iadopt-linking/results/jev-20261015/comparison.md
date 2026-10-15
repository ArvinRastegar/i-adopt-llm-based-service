# Jev against the Qwen3 baseline: jev-20261015

Primary pass `choice-a1` of run `jev-20261015` against baseline `baseline-20261014`, on the same slots and the same candidates. Differences are in top-1 correct slots (Jev minus Qwen3); intervals resample whole terms.

## Paired counts by population

| Population | Slots | Both right | Only Jev | Only Qwen3 | Both wrong | Jev minus Qwen3 | McNemar p |
|---|---|---|---|---|---|---|---|
| reachable | 284 | 171 | 42 | 23 | 48 | +0.067 (95% interval -0.040 to +0.182) | 0.0248 |
| all | 294 | 171 | 42 | 23 | 58 | +0.065 (95% interval -0.038 to +0.179) | 0.0248 |
| gold:wikidata | 218 | 135 | 39 | 3 | 41 | +0.165 (95% interval +0.054 to +0.295) | <0.0001 |
| gold:qudt | 61 | 31 | 3 | 20 | 7 | -0.279 (95% interval -0.483 to -0.104) | 0.0005 |
| gold:om | 5 | 5 | 0 | 0 | 0 | +0.000 (95% interval +0.000 to +0.000) | 1.0000 |

## Counted once per term (post hoc; reachable and by gold vocabulary)

| Population | Terms | Jev better | Qwen3 better | Equal | Sign-test p | Term-averaged difference |
|---|---|---|---|---|---|---|
| reachable | 145 | 17 | 16 | 112 | 1.0000 | +0.025 |
| all | 153 | 17 | 16 | 120 | 1.0000 | +0.024 |
| gold:wikidata | 116 | 15 | 1 | 100 | 0.0005 | +0.109 |
| gold:qudt | 31 | 2 | 15 | 14 | 0.0023 | -0.309 |
| gold:om | 1 | 0 | 0 | 1 | 1.0000 | +0.000 |

## How slots moved between categories (Qwen3 to Jev)

| Qwen3 > Jev | Slots |
|---|---|
| correct > correct | 171 |
| correct > wrong source | 19 |
| correct > wrong within source | 4 |
| gold not retrieved > gold not retrieved | 3 |
| no candidates > no candidates | 6 |
| unreachable vocabulary > unreachable vocabulary | 10 |
| wrong source > correct | 14 |
| wrong source > wrong source | 12 |
| wrong within source > correct | 28 |
| wrong within source > wrong source | 2 |
| wrong within source > wrong within source | 25 |

## Every pass against Qwen3 (reachable slots)

| Pass | Strict top-1 | With QUDT/Wikidata equivalence |
|---|---|---|
| `choice-a1` | 213/284 = 0.750 | 237/284 = 0.835 |
| `choice-a1-repeat` | 212/284 = 0.746 | 235/284 = 0.827 |
| `choice-a2-reversed` | 212/284 = 0.746 | 235/284 = 0.827 |
| `choice-a3` | 210/284 = 0.739 | 233/284 = 0.820 |
| `noul-n1` | 177/284 = 0.623 | 180/284 = 0.634 |
| Qwen3 baseline | 194/284 = 0.683 | 202/284 = 0.711 |

## Slots only Jev got right (42)

0 of these 42 slots would not be misses for Qwen3 if a QUDT kind and its `wikidataMatch` Wikidata item counted as one entity (post hoc, L-014).

- `water` (ConcCarbon.ttl): gold wikidata:Q283; Jev 'water' (wikidata:Q283, 0.98); Qwen3 'Mass Concentration of Water' (qudt:MassConcentrationOfWater, 1.00)
- `status` (C22_SmokingStatus.ttl): gold wikidata:Q813912; Jev 'condition' (wikidata:Q813912, 0.65); Qwen3 'status' (wikidata:Q11424100, 0.99)
- `person` (Rhesusform.ttl): gold wikidata:Q5; Jev 'human' (wikidata:Q5, 0.33); Qwen3 'person' (wikidata:Q215627, 1.00)
- `person` (BPFOS.ttl): gold wikidata:Q5; Jev 'human' (wikidata:Q5, 0.54); Qwen3 'person' (wikidata:Q215627, 1.00)
- `cloud` (C3_CloudCover.ttl): gold wikidata:Q8074; Jev 'cloud' (wikidata:Q8074, 0.93); Qwen3 'Amount of Cloud Cover' (qudt:AmountOfCloudCover, 1.00)
- `atmospheric boundary layer` (C4_ABLHeight.ttl): gold wikidata:Q1757268; Jev 'planetary boundary layer' (wikidata:Q1757268, 0.56); Qwen3 'Atmospheric boundary layer top height in South Africa: measurements with lidar and radiosonde compared to three atmospheric models' (wikidata:Q58212201, 0.98)
- `optical thickness` (C5_OpticalThickness.ttl): gold wikidata:Q890809; Jev 'optical depth' (wikidata:Q890809, 0.85); Qwen3 'Optical Thickness and Size Distribution of Dust Aerosol Particles in Inner Mongolia' (wikidata:Q106333828, 1.00)
- `water` (IceConcentration.ttl): gold wikidata:Q283; Jev 'water' (wikidata:Q283, 0.53); Qwen3 'Mass Concentration of Water' (qudt:MassConcentrationOfWater, 1.00)
- `water` (PrecRate.ttl): gold wikidata:Q283; Jev 'water' (wikidata:Q283, 0.86); Qwen3 'Mass Fraction of Water' (qudt:MassFractionOfWater, 0.35)
- `earth` (C10_PeakGroundAcceleration.ttl): gold wikidata:Q2; Jev 'Earth' (wikidata:Q2, 0.92); Qwen3 'earthquake' (wikidata:Q7944, 0.97)
- `plane angle` (C11_StrikeBedding.ttl): gold qudt:PlaneAngle; Jev 'Plane Angle' (qudt:PlaneAngle, 0.42); Qwen3 'plane angle' (wikidata:Q2849635, 1.00)
- `water` (GLVAR.ttl): gold wikidata:Q283; Jev 'water' (wikidata:Q283, 1.00); Qwen3 'Water Solubility' (qudt:WaterSolubility, 1.00)

## Slots only Qwen3 got right (23)

16 of these 23 slots would not be misses for Jev if a QUDT kind and its `wikidataMatch` Wikidata item counted as one entity (post hoc, L-014).

- `height` (C7_TrunkCharHeight.ttl): gold qudt:Height; Jev 'height' (wikidata:Q208826, 0.61); Qwen3 'Height' (qudt:Height, 1.00)
- `mass fraction` (CONCVAR.ttl): gold qudt:MassFraction; Jev 'mass fraction' (wikidata:Q899138, 0.91); Qwen3 'Mass Fraction' (qudt:MassFraction, 1.00)
- `pressure` (C21_BloodPressure.ttl): gold qudt:Pressure; Jev 'pressure' (wikidata:Q39552, 0.54); Qwen3 'Pressure' (qudt:Pressure, 1.00)
- `count` (HeartRate.ttl): gold qudt:Count; Jev 'Count Rate' (qudt:CountRate, 0.53); Qwen3 'Count' (qudt:Count, 1.00)
- `amount of substance concentration` (lactate.ttl): gold qudt:AmountOfSubstanceConcentration; Jev 'amount-of-substance concentration' (wikidata:Q672821, 0.80); Qwen3 'Amount of Substance of Concentration' (qudt:AmountOfSubstanceConcentration, 1.00)
- `volume fraction` (BPFOS.ttl): gold qudt:VolumeFraction; Jev 'volume fraction' (wikidata:Q909482, 0.36); Qwen3 'Volume Fraction' (qudt:VolumeFraction, 1.00)
- `mass` (C23_NanoparticleDose.ttl): gold qudt:Mass; Jev 'mass' (wikidata:Q11423, 0.67); Qwen3 'Mass' (qudt:Mass, 0.99)
- `mass fraction` (ConcEndosulfaneOstrea.ttl): gold qudt:MassFraction; Jev 'mass fraction' (wikidata:Q899138, 0.64); Qwen3 'Mass Fraction' (qudt:MassFraction, 1.00)
- `mass fraction` (DehpNail.ttl): gold qudt:MassFraction; Jev 'mass fraction' (wikidata:Q899138, 0.53); Qwen3 'Mass Fraction' (qudt:MassFraction, 1.00)
- `thermodynamic temperature` (C2_AirDailyMaximumTemperature.ttl): gold qudt:ThermodynamicTemperature; Jev 'thermodynamic temperature' (wikidata:Q264647, 0.38); Qwen3 'Thermodynamic Temperature' (qudt:ThermodynamicTemperature, 1.00)
- `irradiance` (RadDown.ttl): gold qudt:Irradiance; Jev 'irradiance' (wikidata:Q830654, 0.37); Qwen3 'Irradiance' (qudt:Irradiance, 1.00)
- `speed` (sfcWindmax.ttl): gold qudt:Speed; Jev 'speed' (wikidata:Q3711325, 0.48); Qwen3 'Speed' (qudt:Speed, 1.00)
