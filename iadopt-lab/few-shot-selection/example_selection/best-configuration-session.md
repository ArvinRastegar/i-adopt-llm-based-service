# Best-known configuration — complete session record

**Exploratory side experiment, not an official result.** The configuration below won a
held-out comparison inside [the example-selection study](README.md); it carries no campaign
identity and must not enter an official ranking or results table. It is recorded here so the
exact prompt, the exact 25 examples and the exact request body can be reused elsewhere —
for instance by a web service — without re-deriving them from the run logs.

Generated from `output/example-selection-calls.jsonl` and the canonical corpus by
[`build_best_config_doc.py`](build_best_config_doc.py). Candidate hash `2d56af0e65d42748`.

## 1. What is being claimed, and what is not

Micro Close F1 on the 24-variable confirmation set `C`, which never informed selection.
15 repetitions per candidate.

| Candidate | Close F1 | SD | 95% CI |
|---|---:|---:|---|
| **`top25`** | 0.4890 | 0.0252 | [0.4763, 0.5018] |
| `refined` | 0.4805 | 0.0281 | [0.4663, 0.4947] |
| `random-1` | 0.4757 | 0.0247 | [0.4633, 0.4882] |
| `random-2` | 0.4464 | 0.0257 | [0.4334, 0.4594] |
| `random-3` | 0.4349 | 0.0378 | [0.4157, 0.4540] |
| `bottom25` | 0.3981 | 0.0258 | [0.3850, 0.4111] |
| `stratified` | 0.3952 | 0.0246 | [0.3827, 0.4076] |

**What holds.** This set beats a deliberately-bad selection (`bottom25`, +0.091,
Holm `p < 0.0001`) and a domain-stratified selection (+0.094, `p < 0.0001`), and beats the
pooled random-25 reference (+0.037, `p = 0.0002`). The attribution model learned something
real and reproducible.

**What does not hold.** It is *not* significantly better than the best of three random
draws (`random-1` at 0.4757, difference +0.0133, `p = 0.156`). Treat this as a reliably
good selection, not a proven optimum. Two further limits are worth carrying forward:

- The 25 were chosen from a 40-variable pool `P`, so **8 of the 25 carry negative
  coefficients** — they are included because 25 of 40 must be taken, not because they help.
- The coefficients were fitted on `E`. On `E` the top-25/bottom-25 gap is +0.254; on `C` it
  is +0.091. Roughly two thirds of the fitted signal is specific to the set that fitted it.

Absolute accuracy is modest: **Close F1 0.4890** means a little under half of the
component mass is recovered. This is a research-grade configuration, not a solved task.

## 2. The 25 examples

Attribution fit over 600 random 25-subsets: **r² = 0.5513**,
**split-half stability = 0.8928** (coefficients had stabilised).
Coefficients are *relative* marginal contributions; only their ordering and differences
mean anything, never their absolute level.

> **Ordering matters for byte-exact reproduction.** The prompt carries these 25 in sorted
> `variable_id` order — *not* in coefficient order. The table is shown by coefficient rank
> for readability; the `prompt #` column is the order actually sent.

| prompt # | rank | coefficient | domain | label |
|---:|---:|---:|---|---|
| 20 | 1 | +0.0240 | Social Sciences | Date of birth of mother |
| 10 | 2 | +0.0170 | Natural Sciences | Daily maximum hourly precipitation rate |
| 1 | 3 | +0.0145 | Social Sciences | Overnight stays in 3-star hotel near the sea shore |
| 12 | 4 | +0.0130 | Life Sciences | Blood lactate concentration |
| 4 | 5 | +0.0123 | Life Sciences | Observation of bloodtype in blood |
| 9 | 6 | +0.0121 | Life Sciences | BPM Mean heart rate |
| 2 | 7 | +0.0100 | Social Sciences | Average number of persons without high school diploma during the last three years within a statistical unit |
| 17 | 8 | +0.0075 | Social Sciences | Total Number of people living in a statistical unit |
| 3 | 9 | +0.0072 | Life Sciences | Standard metabolic rate in mg of Oxygen per hour |
| 23 | 10 | +0.0066 | Natural Sciences | Strike of bedding |
| 6 | 11 | +0.0050 | Technical Sciences | solubility of benzene in water |
| 15 | 12 | +0.0045 | Life Sciences | Foliage projective cover in the lower canopy strata |
| 18 | 13 | +0.0036 | Natural Sciences | Daily Maximum Near-Surface Wind Speed of Gust |
| 11 | 14 | +0.0034 | Life Sciences | Concentration of dissolved organic carbon in water |
| 25 | 15 | +0.0018 | Natural Sciences | solubility of molecular oxygen from air in water |
| 24 | 16 | +0.0017 | Natural Sciences | Surface runoff |
| 19 | 17 | +0.0011 | Technical Sciences | I129 deposition from Ringhals of Sweden |
| 22 | 18 | -0.0002 | Social Sciences | Count of tropic nights |
| 13 | 19 | -0.0004 | Technical Sciences | boiling point of benzene |
| 21 | 20 | -0.0014 | Technical Sciences | density of benzene |
| 16 | 21 | -0.0017 | Life Sciences | Cellular dose in vitro of nanomaterials (particle / cell) |
| 7 | 22 | -0.0018 | Social Sciences | Fraction of flooded area around building |
| 5 | 23 | -0.0018 | Natural Sciences | DRVA Direction of radial velocity of water current relative to instrument and to True North by high frequency radar |
| 8 | 24 | -0.0018 | Life Sciences | Docosahexaenoic acid content per dry weight (DHA content/ C22:6 n-3 content) |
| 14 | 25 | -0.0025 | Natural Sciences | Northward velocity uncertainty of water current in the water body by high frequency radar |

The 25 `variable_id`s, in prompt order:

```json
[
  "urn:iadopt-lab:variable:0f3e9954aef6497d1eaeeb026c643c23dc0e24127c4f4a26938827039cabe944",
  "urn:iadopt-lab:variable:12b00c359adcf48b4a1dbef25c0e123e0dd41230bc29efcdf160b201be399f0a",
  "urn:iadopt-lab:variable:12d0bb86102fa82e6c55f1b3f178abd9fd14fb55a497ab442c94ed3ecd41e539",
  "urn:iadopt-lab:variable:139c80cce9c15999097bed86c8562e543eed77f8528d532f01a00f90778c0043",
  "urn:iadopt-lab:variable:17b44234089af8f003abef095347985e0d50b8cc418721ed2058c26b2dff1556",
  "urn:iadopt-lab:variable:1a039be29796c65ca05176b0d2752323eeae0711597ee9b9fb034a72e08f56e5",
  "urn:iadopt-lab:variable:1e8966f5353f9625d602da4a30952ce6b4e93d27cd2b697cbc207f5e2412edd3",
  "urn:iadopt-lab:variable:21171d68ac4530e2bd43a74c04c52ee72c7a7ba44ffd0f826b0cc7d47361c4c5",
  "urn:iadopt-lab:variable:269247ba0afcd49573d5f3890800dbe57cbab4de0abe23e35363dd4c408eeab7",
  "urn:iadopt-lab:variable:2cb06286f2a57367046baecdc253e1cc9d0b87e3ff744ece77467dbc2b209cd3",
  "urn:iadopt-lab:variable:33c4eb606b35dd5cbba886825fab3368a2ccec7e8648f0ea739ffdaa8e7ac381",
  "urn:iadopt-lab:variable:3a580edca2d246b8cacb94c1a25f05c414f9bf30ebb89ad44d7d3243ddefd5ab",
  "urn:iadopt-lab:variable:3f70359c18e6b25d95bd9792608b167905fb117d9dfe9119e2a8e82e96298cd8",
  "urn:iadopt-lab:variable:417e2fc4ad6d614f1aa6fbffac30a659c313db707818b36fe36e1636ceda31b0",
  "urn:iadopt-lab:variable:4468d192503a3df852da9a2e6b2232d2cf888db8e57a183bb36b150f90a4dfb3",
  "urn:iadopt-lab:variable:5195f6d2757865fa53c1a58080743c04f77f21da30e35252076d7011836494cc",
  "urn:iadopt-lab:variable:5382e8c971d5accb754b9b3c8c5052ac2a94ddc4c18bf7c14b05a8986546d509",
  "urn:iadopt-lab:variable:5aa2c10675a4b91f044861bac4875458821f234a139c770757ad4a880072dfdc",
  "urn:iadopt-lab:variable:608502045635ff2f6185e327ed3dabb4915ae7fcf480bf5aaeea346544d85dfb",
  "urn:iadopt-lab:variable:60c22a22493fe14d9b49056deb34f93814298c43443f3d47bd63346ea2042820",
  "urn:iadopt-lab:variable:661c259486249b392894bc315cc14457401c22ed1abb25e42110c733dc2a6128",
  "urn:iadopt-lab:variable:7a725c87db996829ebc171a000be0a2d49f868ec0778ab96908b28a38ba03be0",
  "urn:iadopt-lab:variable:878908f20228224335acf3157a9fc32acc26f6f85629e31c18e1cc006e1005a7",
  "urn:iadopt-lab:variable:8a816547d513e441cc011c55b73bc5051ee086c3a1b888e4a7c8d41c7b4f3531",
  "urn:iadopt-lab:variable:8b4f1caa8c7add1f5a3d21c7cc9dc6ca911b9abe84e3a583725215f55d662aed"
]
```

## 3. The model call

OpenAI-compatible chat completion against PSNC. Model `Qwen3.8-27B`, reasoning disabled —
that is the condition the configuration was measured in, and `enable_thinking: true` would
be a different configuration with none of the evidence above behind it.

```http
POST {PSNC_API_BASE_URL}/chat/completions
Authorization: Bearer {PSNC_API_KEY}
Content-Type: application/json
```

```json
{
  "model": "Qwen3.8-27B",
  "messages": [
    {
      "role": "user",
      "content": "<the prompt from section 4>"
    }
  ],
  "stream": false,
  "temperature": 0.5,
  "top_p": 1.0,
  "max_tokens": 16000,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

A single user message carries the whole prompt; there is no system message. Default base
URL is `https://llm.hpc.psnc.pl`. Measured cost over this candidate's 550 recorded
calls: **3,736 prompt tokens** on average (3,720–3,797 — the spread is the
target definition alone), against 99 completion tokens (53–224), 3.35 s mean latency,
and zero transport failures. The 16,000-token output ceiling is never approached; it is
inherited from the measured configuration rather than chosen for this task.

## 4. The prompt

Template [`prompts/matrix-decomposition-v1.txt`](../../prompts/matrix-decomposition-v1.txt) with
three placeholders filled in one substitution pass:

- `{{schema}}` — [`schemas/lexical-decomposition.schema.json`](../../schemas/lexical-decomposition.schema.json), verbatim UTF-8.
- `{{demonstrations}}` — the 25 examples as one compact JSON array, each entry
  `{"demonstration": <1-based position>, "definition": <text>, "decomposition": <gold>}`.
- `{{target_definition}}` — the definition text of the variable being decomposed.

The demonstrations array is encoded with **sorted keys, no ASCII folding, and `(",", ":")`
separators**. That encoding is part of the measured prompt: re-serialising it with default
`json.dumps` spacing changes the prompt bytes, and nothing here was measured under those
bytes. In Python the exact call is:

```python
json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
```

Rendered length for the session below: **15,791 characters, 3,739 prompt tokens**. The block
below is byte-identical to what was sent, with one caveat: the prompt ends in a single
newline, and the fence adds one of its own. Strip exactly one trailing newline when you
copy it out. Full text, verbatim:

````text
Follow the JSON-Schema exactly. Do not infer or invent new concepts.

hasProperty = the main measurable property in the definition.
hasObjectOfInterest = the thing that has this property.
hasMatrix = the medium in which the object occurs. Never a method or location.

If a required key is not in the definition, output an empty string for it.

Output only the JSON object.

Six-field compatibility rules:
Return only hasStatisticalModifier, hasProperty, hasObjectOfInterest, hasMatrix, hasContextObject, and hasConstraint. Do not regenerate the definition, label, or comment. Every field is required: use "" for an unsupported scalar/entity value and [] for no constraints. Never use null.
hasStatisticalModifier is an explicitly stated statistical operation; hasContextObject is an explicitly necessary background entity other than Object of Interest or Matrix. Do not infer either.
An entity may be a simple string, a symmetric system of at least two distinct parts, an asymmetric source/target system, or an asymmetric numerator/denominator system. The schema defines the exclusive shapes; do not mix the role pairs or invent unstated members. System container labels are metadata and may be empty.
Constraint label preserves its explicit restriction text, including a prefix if present. Constraint on identifies an emitted Property, Statistical Modifier, entity, member, or whole system; it is not a constraint category. A whole system can be named by its container label or by sorted parts joined with " + ", source and target joined with " → ", or numerator and denominator joined with " / ". Do not target another constraint or invent a target.

Decision rules:

1. Identify hasProperty first.
2. Identify hasObjectOfInterest:
   → the entity that carries the property.
3. Identify hasMatrix only if the definition clearly states
   the medium or material the object is inside.
4. If a phrase describes a condition/state, not a medium:
   → put it in hasConstraint.
5. Never use methods, units, instruments, or locations.

LEXICAL JSON SCHEMA
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://iadopt-lab.local/schemas/lexical-decomposition-v1",
  "title": "I-ADOPT Lab six-field lexical decomposition v1",
  "type": "object",
  "additionalProperties": false,
  "required": ["hasStatisticalModifier", "hasProperty", "hasObjectOfInterest", "hasMatrix", "hasContextObject", "hasConstraint"],
  "properties": {
    "hasStatisticalModifier": {"type": "string"},
    "hasProperty": {"type": "string"},
    "hasObjectOfInterest": {"$ref": "#/$defs/entity"},
    "hasMatrix": {"$ref": "#/$defs/entity"},
    "hasContextObject": {"$ref": "#/$defs/entity"},
    "hasConstraint": {"type": "array", "items": {"$ref": "#/$defs/constraint"}}
  },
  "$defs": {
    "nonempty": {"type": "string", "minLength": 1, "pattern": "\\S"},
    "entity": {"oneOf": [
      {"type": "string"},
      {"type": "object", "additionalProperties": false,
       "required": ["SymmetricSystem", "hasPart"],
       "properties": {"SymmetricSystem": {"type": "string"}, "hasPart": {"type": "array", "minItems": 2, "uniqueItems": true, "items": {"$ref": "#/$defs/nonempty"}}}},
      {"type": "object", "additionalProperties": false,
       "required": ["AsymmetricSystem", "hasSource", "hasTarget"],
       "properties": {"AsymmetricSystem": {"type": "string"}, "hasSource": {"$ref": "#/$defs/nonempty"}, "hasTarget": {"$ref": "#/$defs/nonempty"}}},
      {"type": "object", "additionalProperties": false,
       "required": ["AsymmetricSystem", "hasNumerator", "hasDenominator"],
       "properties": {"AsymmetricSystem": {"type": "string"}, "hasNumerator": {"$ref": "#/$defs/nonempty"}, "hasDenominator": {"$ref": "#/$defs/nonempty"}}}
    ]},
    "constraint": {"type": "object", "additionalProperties": false,
      "required": ["label", "on"],
      "properties": {"label": {"$ref": "#/$defs/nonempty"}, "on": {"$ref": "#/$defs/nonempty"}}}
  }
}


ORDERED DEMONSTRATIONS
[{"decomposition":{"hasConstraint":[{"label":"rating: three star","on":"hotel"},{"label":"vicinity: near the sea shore","on":"hotel"}],"hasContextObject":"","hasMatrix":"hotel","hasObjectOfInterest":"night","hasProperty":"count","hasStatisticalModifier":""},"definition":"Number of nights in a 3-star hotel near the sea shore","demonstration":1},{"decomposition":{"hasConstraint":[{"label":"condition: registered as resident in the previous three years","on":"person"},{"label":"education: without high school diploma","on":"person"},{"label":"normalization: per statistical unit","on":"count"}],"hasContextObject":"urban area","hasMatrix":"","hasObjectOfInterest":"person","hasProperty":"count","hasStatisticalModifier":"arithmetic mean"},"definition":"Average number of persons without high school diploma during the last three years residing in a statistical unit which is defined as a grouping of homogeneous neighboring building blocks","demonstration":2},{"decomposition":{"hasConstraint":[{"label":"normalization: per hour","on":"mass"},{"label":"process: metabolism","on":"oxygen"},{"label":"state: inactive","on":"ectotherm organism"},{"label":"state: unstressed","on":"ectotherm organism"}],"hasContextObject":"","hasMatrix":"ectotherm organism","hasObjectOfInterest":"oxygen","hasProperty":"mass","hasStatisticalModifier":""},"definition":"The measure of an ectotherm animal's baseline metabolic rate (i.e. non-active, non-stressed) measured by the milligrams of oxygen consumed in a period of time of one hour.","demonstration":3},{"decomposition":{"hasConstraint":[],"hasContextObject":"","hasMatrix":"person","hasObjectOfInterest":"blood","hasProperty":"phenotype","hasStatisticalModifier":""},"definition":"Type of blood in a blood sample","demonstration":4},{"decomposition":{"hasConstraint":[{"label":"reference: True North","on":"plane angle"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":"radial velocity of sea water","hasProperty":"plane angle","hasStatisticalModifier":""},"definition":"The compass direction in degrees positive clockwise from TrueNorth of the component of water velocity in a water body along radial lines centered at the HFR antenna.","demonstration":5},{"decomposition":{"hasConstraint":[{"label":"condition: at 298.15 K  thermodynamic temperature","on":"mass concentration"},{"label":"purity: 99.82 (w/w)","on":"benzene"},{"label":"resistivity: > 1 MOhm cm","on":"water"},{"label":"state: dissolved","on":"benzene"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":{"AsymmetricSystem":"benzene / water","hasDenominator":"water","hasNumerator":"benzene"},"hasProperty":"mass concentration","hasStatisticalModifier":""},"definition":"solubility (g/L) of 99.82 (w/w) pure benzene in deionized water at phase equilibrium and 298.15 K","demonstration":6},{"decomposition":{"hasConstraint":[{"label":"condition: flooded > 30 cm above ground","on":"flooded area"},{"label":"spatial extent: 5 m buffer around buildings","on":"flooded area / total area"}],"hasContextObject":"","hasMatrix":"urban area","hasObjectOfInterest":{"AsymmetricSystem":"flooded area / total area","hasDenominator":"total area","hasNumerator":"flooded area"},"hasProperty":"area fraction","hasStatisticalModifier":""},"definition":"Fraction of flooded area above 30 cm water level within a 5 m buffer around an affected building in an urban area","demonstration":7},{"decomposition":{"hasConstraint":[{"label":"state: dry","on":"organism"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":{"AsymmetricSystem":"docosahexaenoic acid / organism","hasDenominator":"organism","hasNumerator":"docosahexaenoic acid"},"hasProperty":"amount of substance per mass","hasStatisticalModifier":""},"definition":"The amount of docosahexaenoic acid relative to dry weight in an organism.","demonstration":8},{"decomposition":{"hasConstraint":[{"label":"normalization: per minute","on":"count"}],"hasContextObject":"","hasMatrix":"person","hasObjectOfInterest":"heart beat","hasProperty":"count","hasStatisticalModifier":"mean"},"definition":"mean heart rate, beats per minute (ECG-derived)","demonstration":9},{"decomposition":{"hasConstraint":[{"label":"process: precipitation","on":"water"},{"label":"reporting frequency: daily","on":"maximum"},{"label":"reporting frequency: hourly","on":"maximum"}],"hasContextObject":"","hasMatrix":{"AsymmetricSystem":"atmosphere → ground","hasSource":"atmosphere","hasTarget":"ground"},"hasObjectOfInterest":"water","hasProperty":"mass flux density","hasStatisticalModifier":"maximum"},"definition":"Daily maximum hourly precipitation rate, measured in kg m⁻² s⁻¹","demonstration":10},{"decomposition":{"hasConstraint":[{"label":"state: dissolved","on":"organic carbon"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":{"AsymmetricSystem":"organic carbon / water","hasDenominator":"water","hasNumerator":"organic carbon"},"hasProperty":"mass concentration","hasStatisticalModifier":""},"definition":"Mass concentration of dissolved organic carbon in water","demonstration":11},{"decomposition":{"hasConstraint":[{"label":"criterium: lactate tolerance threshold ","on":"person"}],"hasContextObject":"","hasMatrix":"person","hasObjectOfInterest":{"AsymmetricSystem":"lactate / blood","hasDenominator":"blood","hasNumerator":"lactate"},"hasProperty":"amount of substance concentration","hasStatisticalModifier":""},"definition":"Blood lactate concentration at lactate tolerance threshold measured in mmol/l","demonstration":12},{"decomposition":{"hasConstraint":[{"label":"condition: at 1 atm pressure","on":"thermodynamic temperature"},{"label":"process: liquid to gas phase transition","on":"thermodynamic temperature"},{"label":"purity: 99.07 (w/w)","on":"benzene"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":"benzene","hasProperty":"thermodynamic temperature","hasStatisticalModifier":""},"definition":"thermodynamic temperature for the boiling point (at the liquid to gas phase transition) of a known purity 99.07 (w/w) liquid sample of benzene at atmospheric pressure","demonstration":13},{"decomposition":{"hasConstraint":[],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":"northward linear velocity of water","hasProperty":"confidence interval","hasStatisticalModifier":""},"definition":"Uncertainty expressed using confidence interval of the component of the sea surface current velocity that is directed northward.","demonstration":14},{"decomposition":{"hasConstraint":[{"label":"part: lower canopy strata","on":"foliage"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":{"AsymmetricSystem":"foliage / ground","hasDenominator":"ground","hasNumerator":"foliage"},"hasProperty":"area fraction","hasStatisticalModifier":""},"definition":"The proportion (percentage) of the ground area covered by foliage (or photosynthetic tissue) in the lower canopy strata.","demonstration":15},{"decomposition":{"hasConstraint":[{"label":"normalization: per cell","on":"count"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":"nanomaterial particle","hasProperty":"count","hasStatisticalModifier":""},"definition":"Count of nanomaterial particles internalised into the cell over the duration of the experiment, entity per cell","demonstration":16},{"decomposition":{"hasConstraint":[{"label":"condition: registered as resident","on":"person"},{"label":"normalization: per statistical unit","on":"count"}],"hasContextObject":"urban area","hasMatrix":"","hasObjectOfInterest":"person","hasProperty":"count","hasStatisticalModifier":""},"definition":"total number of people living within a statistical unit which is defined as a grouping of homogeneous neighboring building blocks in an urban area","demonstration":17},{"decomposition":{"hasConstraint":[{"label":"event: gust","on":"wind"},{"label":"part: near surface","on":"wind"},{"label":"reporting frequency: daily","on":"maximum"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":"wind","hasProperty":"speed","hasStatisticalModifier":"maximum"},"definition":"Daily Maximum Near-Surface Wind Speed of Gust, measured in m s⁻¹","demonstration":18},{"decomposition":{"hasConstraint":[{"label":"source: nuclear fuel reprocessing plant at Ringhals, Sweden","on":"Iodine-129"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":{"AsymmetricSystem":"Iodine-129 / sea water","hasDenominator":"sea water","hasNumerator":"Iodine-129"},"hasProperty":"number density per unit mass","hasStatisticalModifier":""},"definition":"Number of I129 atoms per mass  of sea water in the artic from discharges  of the nuclear fuel reprocessing plant at Ringhals of Sweden (I129se40) in atoms kg-1","demonstration":19},{"decomposition":{"hasConstraint":[{"label":"relationship: mother","on":"person"}],"hasContextObject":"","hasMatrix":"person","hasObjectOfInterest":"birth","hasProperty":"calendar date","hasStatisticalModifier":""},"definition":"date of birth of mother","demonstration":20},{"decomposition":{"hasConstraint":[{"label":"condition: at room temperature","on":"mass density"},{"label":"quality: ACS reagent grade (99.0 (v/v))","on":"benzene (C6H6)"},{"label":"volume: 10.00 mL","on":"benzene (C6H6)"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":"benzene (C6H6)","hasProperty":"mass density","hasStatisticalModifier":""},"definition":"mass density of a 10.00 mL sample of ACS reagent grade (99.0 (v/v)) benzene at a room temperature","demonstration":21},{"decomposition":{"hasConstraint":[{"label":"min temperature: > 20 ° C","on":"night"}],"hasContextObject":"","hasMatrix":"urban area","hasObjectOfInterest":"night","hasProperty":"count","hasStatisticalModifier":""},"definition":"Count of tropic nights, with min temperature > 20 ° Celsius in urban areas","demonstration":22},{"decomposition":{"hasConstraint":[],"hasContextObject":"","hasMatrix":"sedimentary rock","hasObjectOfInterest":{"SymmetricSystem":"bedding surface line + True North","hasPart":["bedding surface line","True North"]},"hasProperty":"plane angle","hasStatisticalModifier":""},"definition":"Geographic azimuth (relative to true north) at a point observation location, of a horizontal line contained in a sedimentary rock bedding surface. The bedding surface must not be horizontal; the azimuth is reported such that the dip direction of the inclined bedding is to the right when facing in the azimuth direction.","demonstration":23},{"decomposition":{"hasConstraint":[{"label":"condition: not soaking into the ground","on":"water"},{"label":"part: surface","on":"ground"}],"hasContextObject":"","hasMatrix":{"AsymmetricSystem":"ground → water body","hasSource":"ground","hasTarget":"water body"},"hasObjectOfInterest":"water","hasProperty":"mass flux density","hasStatisticalModifier":""},"definition":"Surface runoff measured in kg m⁻² s⁻¹","demonstration":24},{"decomposition":{"hasConstraint":[{"label":"condition: at local atmospheric pressure","on":"amount of substance per mass"},{"label":"condition: at local temperature","on":"amount of substance per mass"},{"label":"resistivity: 10.2 MΩ cm","on":"water"},{"label":"source: air","on":"molecular oxygen"}],"hasContextObject":"","hasMatrix":"","hasObjectOfInterest":{"AsymmetricSystem":"molecular oxygen / water","hasDenominator":"water","hasNumerator":"molecular oxygen"},"hasProperty":"amount of substance per mass","hasStatisticalModifier":""},"definition":"solubility of molecular oxygen from air in water with MilliQ (10.2MOhm) purity at local air conditions","demonstration":25}]

TARGET DEFINITION
number of persons receiving welfare in a statistical unit which is defined as a grouping of homogeneous neighboring building blocks in an urban area

Return exactly one JSON object matching the schema, without prose, Markdown fences, quotation wrappers, or explanations.

````

## 5. The raw response

Target variable: **Number of persons receiving welfare in a statistical unit** (`Social Sciences` / Demography),
held-out set `C`, repetition 1.

Definition sent:

```text
number of persons receiving welfare in a statistical unit which is defined as a grouping of homogeneous neighboring building blocks in an urban area
```

Raw `choices[0].message.content`, exactly as returned — `finish_reason: stop`,
113 completion tokens, 3.434 s:

````text
{
  "hasStatisticalModifier": "",
  "hasProperty": "count",
  "hasObjectOfInterest": "person",
  "hasMatrix": "",
  "hasContextObject": "urban area",
  "hasConstraint": [
    {
      "label": "condition: receiving welfare",
      "on": "person"
    },
    {
      "label": "normalization: per statistical unit",
      "on": "count"
    }
  ]
}
````

The model returned a bare JSON object with no fences and no prose, which is what the
template asks for. **Do not rely on that** — the extraction step exists because it is not
guaranteed; see section 7.

Gold decomposition for the same variable:

```json
{
  "hasConstraint": [
    {
      "label": "condition: receiving welfare",
      "on": "person"
    },
    {
      "label": "condition: registered as resident",
      "on": "person"
    },
    {
      "label": "normalization: per statistical unit",
      "on": "count"
    }
  ],
  "hasContextObject": "urban area",
  "hasMatrix": "",
  "hasObjectOfInterest": "person",
  "hasProperty": "count",
  "hasStatisticalModifier": ""
}
```

This item scored **Close F1 0.957**.

## 6. How the rest of the held-out set scored

All 24 confirmation targets at repetition 1, same 25 examples, same configuration. The
spread is the honest picture: perfect items and zeroes coexist at this accuracy level.

| Close F1 | completion tokens | target |
|---:|---:|---|
| 1.000 | 86 | Sheet resistance of layer of gold |
| 1.000 | 95 | Concentration of Al cations in water |
| 1.000 | 149 | Number of Uranium-236 atoms in sea water |
| 0.957 | 113 | Number of persons receiving welfare in a statistical unit ← featured above |
| 0.897 | 124 | Atmospheric boundary layer height defined by temperature inversion |
| 0.785 | 134 | Labile detrital nitrogen concentration in the seabed |
| 0.667 | 99 | Atmosphere optical thickness of particulate organic matter ambient aerosol |
| 0.667 | 60 | Determination of iron concentration in a soil sample |
| 0.621 | 91 | Peak ground acceleration |
| 0.615 | 94 | Mass concentration of tbph in breast milk |
| 0.531 | 100 | Radial velocity (away from) standard deviation over the coverage period of water current relative to instrument in the water body by high frequency radar |
| 0.421 | 137 | Global fallout of U‑236 (U236GF) |
| 0.400 | 90 | Ice water content in atmosphere |
| 0.400 | 61 | Aircraft Angle of Attack. |
| 0.400 | 89 | Fire risk per district |
| 0.286 | 94 | Body Mass Index |
| 0.154 | 127 | Percentage Branched Perfluorooctane sulfonic acid (BPFOS) |
| 0.000 | 57 | Biological sex |
| 0.000 | 58 | Mean sea level pressure |
| 0.000 | 58 | Current smoking status |
| 0.000 | 61 | Heat stress index |
| 0.000 | 90 | Flood risk index related to mobility and accessibility of an urban area |
| 0.000 | 94 | Northward Near-Surface Wind |
| 0.000 | 137 | Probablity of occurence of habitat |

3 of 24 exactly right, 7 of 24 scoring zero, the rest partial. A service built
on this should present decompositions as drafts for review, not as answers.

## 7. Adopting this in a service

Five things that are easy to get wrong and that invalidate the measurement if you do:

1. **Never let a target variable appear among the 25 examples.** The harness raises rather
   than score an overlap, because an example that is also scored leaks its own answer. A
   service decomposing user-supplied definitions is safe by construction; one decomposing
   these 102 corpus variables is not.
2. **Keep the demonstration encoding byte-exact** (section 4). Different spacing is a
   different prompt.
3. **Extract before you parse.** Responses are not guaranteed to be bare JSON.
   `iadopt_lab.generation.extractor.extract_json` runs a frozen ordered protocol — whole
   response, one unwrapped JSON string, complete fenced blocks, then balanced outermost
   objects — and reports ambiguity as failure rather than guessing. Reuse it.
4. **Validate against the schema and treat failure as an empty prediction**, keeping the
   variable in the population. Silently dropping unparseable output inflates every score.
   The repetition shown above happened to produce 0 invalid responses, but 9 of this
   candidate's 550 calls failed validation (1.6%), and 3.2% did across the whole run.
5. **`""` and `[]` are meaningful**, and `null` is never valid. An absent statistical
   modifier is the empty string, not a missing key.

Temperature is 0.5, so the same definition will not always give the same decomposition.
Measured run-to-run SD on an aggregate of this size is about 0.019 Close F1; per-item
variation is much larger. If a service needs stable output, that is an argument for
caching a decomposition once accepted, not for lowering the temperature — T=0.5 is the
value every number here was measured at.

## 8. Provenance

| | |
|---|---|
| Model | `Qwen3.8-27B` on PSNC |
| Prompt variant | `matrix-decomposition` (`prompts/matrix-decomposition-v1.txt`) |
| Sampling | T=0.5, top_p=1.0, max_tokens=16000 |
| Reasoning | disabled (`{"chat_template_kwargs": {"enable_thinking": false}}`) |
| Shot count | 25 |
| Candidate hash | `2d56af0e65d42748` |
| Corpus | 102 canonical variables, tag `v2.0.1` |
| Split | P=40 pool / E=38 search-eval / C=24 held out |
| Held-out score | Close F1 0.4890 ± 0.0252 over 15 repetitions |
| Run date | 2026-09-12 |
| Raw evidence | `output/example-selection-calls.jsonl` (396 MB, gitignored) |

Regenerate this document with `./build_best_config_doc.py`. It refuses to write if the
recomputed top-25 no longer matches the set the run actually evaluated, or if the featured
response no longer extracts and validates.

Fidelity was checked against the run logs when this was written: the prompt in section 4 is
byte-identical to the render the harness sent, and the response in section 5 is byte-
identical to the logged `choices[0].message.content`, re-extracting and re-validating
cleanly. The prompt is rebuilt from `prompts/`, `schemas/` and the corpus rather than
stored, so it stays correct only while those artifacts are unchanged — all three were
clean at commit `e341649` and predate the run.
