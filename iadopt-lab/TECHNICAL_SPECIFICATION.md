---
title: I-ADOPT Lab Technical Specification
version: 0.4.0
status: Implemented; live runs executed. See README and DECISIONS.md for current state.
corpus: I-ADOPT Corpus v2.0.1
---

# I-ADOPT Lab Technical Specification

## 1. Purpose and authority

This document is the authoritative technical specification for I-ADOPT Lab. It defines the scientific protocol, data contracts, provider behavior, persistence model, evaluator behavior, interruption recovery, tests, and reproducibility requirements before implementation begins.

If a later component document conflicts with this file, this file wins. `DECISIONS.md` records why the governing choices were made. A scientific change after execution begins creates a new version and campaign; it never rewrites completed evidence.

The intended result is a self-contained experiment platform that a different engineer can understand and reproduce without depending on undocumented behavior in the older repository.

## 2. Active scope

The system evaluates lexical decomposition of scientific-variable definitions. The active path is:

```text
immutable Corpus v2.0.1 Turtle
  -> deterministic lexical gold representation
  -> frozen 97-variable evaluation-population and demonstration manifests
  -> rendered prompt
  -> one provider/model request
  -> raw-response preservation
  -> JSON extraction
  -> lexical-schema and semantic validation
  -> validation-feedback attempt when required
  -> accepted or explicit empty prediction
  -> January-compatible corrected evaluator
  -> PostgreSQL facts
  -> reproducible summaries and exports
```

The experiment has no per-variable human decision point after a campaign passes preflight.

### 2.1 Explicitly excluded

The following are not implemented or executed in the active experiment:

- RDF generation
- JSON-LD generation
- SHACL validation
- Entity linking
- Wikidata candidate retrieval or reranking
- Ontology-version approval during generation
- Human correction or adjudication
- Deterministic non-LLM baseline

These exclusions remove stages that do not contribute to the agreed lexical evaluator. Their future contracts are documented separately and cannot be mistaken for active functionality.

### 2.2 Planned implementation structure

The complete planned tree and package ownership are defined in `docs/architecture.md`. In summary, the future implementation separates:

- A thin root `main.py` and CLI package
- Configuration and strict schemas
- Immutable Corpus source/canonical/manifests
- Prompt rendering
- One-call OpenRouter and PSNC adapters
- Extraction and lexical/semantic validation
- PostgreSQL repositories and migrations
- Grid planning, bounded parallel workers, leases, and resume
- The pure standalone `iadopt_eval` package
- Read-only reporting/export code
- Unit, property, parity, provider-contract, integration, recovery, and end-to-end tests

There is no general `config/` directory. The one human-edited experiment file is the root `parameters.yml`; schemas and frozen generated manifests have separate, non-configuration responsibilities.

## 3. Evidence and pinned sources

### 3.1 Corpus

The authoritative source is the local Git object for `https://github.com/i-adopt/Corpus`:

| Property | Value |
|---|---|
| Tag | `v2.0.1` |
| Commit | `2598bf91fa927b78a6529bae7864ef0f7d485b73` |
| Tree | `df665d32bb2433a60742c80a4a53908dc7ebde0c` |
| Turtle files | 102 |

D-030 moved this pin from v2.0.0. The release still holds 102 files with no addition, removal, or rename, so categories, the five demonstrations, and the 97-variable population keep their exact membership and order. Four files carry changed gold: `C12_HabitatProbability` (Matrix becomes Context Object), `SurfRunoff` (constraint targets exchanged), `NumChild` and `PersWelfare` (constraint label wording). `PersWelfare` is demonstration position 2, so every 3-shot and 5-shot rendered prompt changes; prompt template artifacts do not, because demonstrations are inserted at render time.

Because `variable_id` is derived from the commit and the release-relative path, all 102 identifiers change, not only those four. That behavior is confirmed rather than incidental: a variable's identity includes the snapshot it was projected from, so results cannot be joined across releases by identifier, and a cross-release comparison uses the stable release-relative path instead. This is a new corpus identity and therefore a new campaign. The migration is executed; `docs/migration-v2.0.1.md` records what changed and its acceptance results.

Implementation must materialize files from the immutable tag/tree, not from the mutable checkout. The copied snapshot, upstream license, Git blob identifiers, and SHA-256 values are stored under `data/` and PostgreSQL.

### 3.2 Historical evaluator

`benchmarking_example/randomShotsPhaseOne.py` at tag `V1.1-Experiment`, commit `b9683d2242aa5ca5b987440ea4b6f70bc1253c7e`, tree `35bf57411d7e8143a255415825f9bb0b644e8c6e`, is the historical behavioral reference. It is not imported at runtime. Accepted protocol `january-derived-member-credit-v1` preserves its unaffected branches and adds explicit system-container-label exclusion and D-022 fractional member contributions for simple/system and system/system comparisons. The matching, arithmetic, and before/after regression boundaries are specified below and in `docs/scorer-parity.md`.

The pre-submission SHA-256 is:

```text
2d4a6271fd86a076127dfb3f85589391cff9744efdfdee8ec570b52c74921df0
```

### 3.3 Historical Turtle projection

`benchmarking_example/gt_json_maker.py` is a semantic reference only. It is not suitable as runtime code because it has absolute paths, import-time filesystem writes, arbitrary graph selection, unstable blank-node identifiers, semantic fallbacks, ratio-role collapse, and incomplete provenance.

The pre-submission file at `V1.1-Experiment` has SHA-256:

```text
b8caf88bc39a1aa8d6b01bc7e0fe2ed6419de5b6450455f0d6dd5242d7ab3773
```

The current February 2026 variant has SHA-256 `4ae635676d79ec665b3a5de1f0d472f3c85b2426a0c6db62a10302a845be2322`; it is separate supporting evidence, not the January source.

### 3.4 PSNC provider contract

The PSNC wire behavior is based on the inspected `iadopt-variable-description-service` commit `0d1a4cdae362b5aa9a23ab7b24b9f55d4a270e8b`. Only its low-level chat-completions contract is reused conceptually. I-ADOPT Lab does not call or import the sibling service.

### 3.5 Historical repository and result audit

`docs/repository-audit.md` records the tagged source hashes, defective-but-unenforced historical schema, nested nine-request risk, prompt metadata leakage, 96-target grids, inconsistent 97–102-target December 30 workbooks (including demonstration overlap), and unresolved manuscript-lineage gaps. Historical results remain audit evidence only; they do not create an active legacy experiment track.

## 4. Terminology and execution grain

Clear execution grain is necessary for retries, database uniqueness, and reporting.

### 4.1 Campaign

A campaign is one immutable execution specification containing:

- A non-empty selected set of providers: PSNC, OpenRouter, or both
- One or more enabled provider-owned models for each selected provider
- One corpus version and complete evaluation population
- Prompt variants
- Shot counts and ordered demonstration set
- Temperatures and other sampling values
- Capability-aware reasoning modes
- Repetition policy
- One lexical output contract
- One evaluator version
- Execution limits, cost-estimate policy, and any optional monetary caps

One campaign may select PSNC only, OpenRouter only, or both, with a separate editable model list for each selected provider. Each task still owns exactly one provider/model. All complete configurations in a combined campaign are ranked together with provider identity retained; separate campaigns remain distinct execution evidence.

### 4.2 Resolved run

A resolved run is one model and one complete parameter combination applied to the complete evaluation population for one repetition. At minimum its identity includes:

```text
campaign
+ provider
+ model ID/revision
+ prompt version
+ shot count and example-order hash
+ temperature
+ reasoning mode and native mapping
+ other sampling parameters
+ repetition index
```

### 4.3 Task

A task is one resolved run applied to one target variable. Task identity additionally includes the variable source and gold hashes. A task is the retry and terminal-failure unit.

### 4.4 Attempt

An attempt is one actual provider request made for one task. Attempt numbers are `1`, `2`, or `3`. Every provider HTTP request counts. An adapter cannot perform an unrecorded inner generation retry.

### 4.5 Transport event

A transport event records request dispatch, HTTP status, timeout, rate limit, connection failure, ambiguous delivery, or response receipt. It does not create an extra retry budget. A new provider request is a new numbered attempt.

## 5. Corpus ingestion contract

### 5.1 Input

The corpus ingester receives an immutable source descriptor:

```text
repository URL
tag
expected commit
expected tree
expected Turtle count
release license
importer version and configuration hash
```

It enumerates `**/*.ttl` from the pinned tree and reads exact bytes. It does not read untracked files or the checkout state.

### 5.2 Per-file validation

Every source file must satisfy these preconditions:

1. It is valid Turtle.
2. It contains exactly one I-ADOPT `Variable` root.
3. The root has one label.
4. The exact model input definition is available from `rdfs:comment` for Corpus v2.0.1.
5. It has exactly one Property and exactly one ObjectOfInterest.
6. Optional singular roles do not have ambiguous cardinality.
7. Every referenced lexical component has a usable label.
8. Every Constraint has a label and target.
9. System role combinations are supported by the lexical contract.

An ambiguity is an import failure, not permission to choose the first RDF value or invent a value.

The documentation-phase read-only audit of the immutable tag established the following regression expectations:

| Corpus fact | Count |
|---|---:|
| Turtle files / Variable roots | 102 / 102 |
| Variables with one `rdfs:comment`, Property, and Object of Interest | 102 |
| Variables with Matrix | 52 |
| Variables with Context Object | 10 |
| Variables with Statistical Modifier | 9 |
| Variables with Constraints / total Constraints | 85 / 157 |
| Asymmetric systems | 36: 31 numerator/denominator and 5 source/target |
| Symmetric systems / total parts | 2 / 4 |

These documentation values do not replace implementation-time verification from the tag, manifest, and exact file bytes.

### 5.3 Category derivation

The release-relative path is parsed as:

```text
<category>/<subcategory>/<remaining path and file>.ttl
```

The database stores:

- Exact category
- Exact subcategory
- Exact full category path
- Exact release-relative source path
- Optional reporting alias stored separately

Source spellings are never overwritten by normalized display values. This includes `Technical Sciences` and any upstream typographical errors.

The user-supplied five-variable table described the fourth example as
`Engineering and Technology`, while the Corpus release places its file under
`Technical Sciences/Material Science`. The release path is authoritative; an
optional reporting alias can retain the supplied wording without changing
provenance or evaluation-population membership.

### 5.4 Gold lexical output

The projection produces one canonical record per variable:

```text
record schema version
stable variable IRI/identifier
label
exact definition
definition source predicate
category
subcategory
category path
source path
Git tag/commit/tree/blob
TTL SHA-256
six-field lexical gold decomposition
lexical schema version/hash
importer version/hash
canonical record hash
demonstration/evaluation role
```

URI-enrichment fields are not included in the six-field decomposition. Source IRIs may be retained as provenance columns but are not LLM outputs or evaluator inputs.

### 5.5 Deterministic collections

RDF graph iteration order is not stable. Set-like values are sorted using a documented Unicode-normalized lexical key before canonical JSON serialization. Role-bearing asymmetric values are never sorted across their roles.

Gold Constraints receive a deterministic order for serialization because RDF does not preserve one: versioned lexical sort key of `label`, then `on`, then canonical bytes as the final tie-break. Original labels, including meaningful prefixes and whitespace, are preserved in stored gold. Prediction order is retained as evidence; the evaluator follows the January greedy algorithm, including its documented row-major behavior when similarities tie.

Two Corpus constraints target an unlabeled asymmetric-system blank node rather than a labeled simple component. The projector resolves those targets to the deterministic role-derived system display label. It must never expose an RDF parser's random blank-node identifier as lexical gold.

### 5.6 Atomic output

Import occurs in a staging transaction. The corpus version becomes active only after all 102 records, manifests, hashes, category counts, demonstration mappings, and lexical-schema validations pass. A failed import cannot leave a partially active dataset.

### 5.7 Output locations

The same canonical records are:

- Stored authoritatively in PostgreSQL
- Exported deterministically beneath `data/canonical/<tag>/`
- Indexed by `data/manifests/corpus-<tag>.json`

The file export is reproducible evidence, not an alternative mutable source of truth.

Under D-031, each canonical record is accompanied by two derived projections of itself: a provenance **metadata file** and a human-readable **variable file** carrying `label`, `definition`, the six lexical fields in their canonical empty representations, and the Wikidata IRIs beside the fields that have them. Both are regenerated, never hand-edited, and verified to reproduce byte-identically from the canonical parent. Neither is a scoring input and neither participates in a hash chain that verification depends on; the single canonical record remains the authoritative artifact.

## 6. Demonstrations and evaluation-population contract

### 6.1 Ordered demonstration pool

The manifest contains these exact paths and positions:

1. `Natural Sciences/Atmospheric Science/C2_AirDailyMaximumTemperature.ttl`
2. `Social Sciences/Demography/PersWelfare.ttl`
3. `Life Sciences/Health Science/lactate.ttl`
4. `Technical Sciences/Material Science/CirculationMode-Water.ttl`
5. `Social Sciences/Disaster Risk Science/HeatStress.ttl`

Shot count selects the first `k` entries. It does not independently sample examples.

### 6.2 Leakage rule

All five demonstration variables are excluded from scoring for every shot count, including zero. A database constraint or preflight query must reject overlap with the evaluation population.

### 6.3 Complete evaluation population

There is no training, development, test, holdout, or stratified partition. The remaining 97 variables form one immutable evaluation population, and every parameter configuration is evaluated against every one of them.

The evaluation-population manifest records all 97 release-relative paths in exact UTF-8 byte order, variable/source/gold hashes, exclusion reason for each demonstration, corpus identity, manifest version, and manifest hash. A population change creates a new manifest and campaign. Ranking is performed only after results exist; it never changes membership or deletes lower-ranked results.

## 7. Lexical decomposition contract

### 7.1 Model-visible fields

The LLM returns exactly six keys:

| Key | Type | Empty value | Meaning |
|---|---|---|---|
| `hasStatisticalModifier` | string | `""` | Applied statistical measure |
| `hasProperty` | string | `""` | Characteristic being observed or derived |
| `hasObjectOfInterest` | entity or system | `""` | Entity carrying the Property |
| `hasMatrix` | entity or system | `""` | Containing medium/material |
| `hasContextObject` | entity or system | `""` | Additional background entity |
| `hasConstraint` | array of constraints | `[]` | Explicit restrictions and their targets |

All keys are required to eliminate missing-versus-empty ambiguity. Unknown top-level and nested properties are rejected.

The schema validates representation, not scientific quality. An all-empty but correctly shaped six-field response is accepted and scored as the model's prediction; it does not trigger a retry merely for being uninformative. The system-created explicit-empty prediction used after three invalid attempts has the same lexical shape but separate failure provenance.

The variable label, exact definition, comment, category, source path, and provenance are stored separately. The model does not echo or regenerate them.

### 7.2 Entity or system union

An entity role contains exactly one of:

1. A non-empty lexical string.
2. A symmetric-system object.
3. An asymmetric source/target object.
4. An asymmetric numerator/denominator object.

### 7.3 Symmetric system

```json
{
  "SymmetricSystem": "stable metadata label or empty compatibility value",
  "hasPart": ["part A", "part B"]
}
```

`hasPart` contains at least two non-empty strings. Order is not semantically meaningful. The canonical record sorts parts before deriving its stable metadata label. The raw extracted prediction retains provider order and original label.

Canonical display labels use the sorted, preserved part strings joined by the exact separator `" + "`. Sorting uses a versioned Unicode/case-folded comparison key, but the scored part strings themselves are not lowercased or rewritten. A stable machine identity separately hashes the typed canonical role record.

### 7.4 Asymmetric system

Flow/flux form:

```json
{
  "AsymmetricSystem": "stable metadata label or empty compatibility value",
  "hasSource": "source",
  "hasTarget": "target"
}
```

Ratio form:

```json
{
  "AsymmetricSystem": "stable metadata label or empty compatibility value",
  "hasNumerator": "numerator",
  "hasDenominator": "denominator"
}
```

The schema uses mutually exclusive alternatives. A system cannot mix the role pairs or omit one role. Role order is semantic and remains fixed during canonicalization and scoring.

The canonical source/target display label is `source + " → " + target`. The canonical ratio display label is `numerator + " / " + denominator`. These exact display labels are deterministic metadata; the stable machine identity hashes the typed role record. The model-supplied container label remains in raw/extracted evidence but is replaced by the role-derived label only in the canonical representation.

### 7.5 Constraint

```json
{
  "label": "explicit restriction",
  "on": "the constrained component label"
}
```

Each item requires only `label` and `on`, both non-empty strings. Extra properties are invalid. Constraint array order is not domain semantics, but the evaluator retains the January scorer's documented tie behavior. The semantic validator checks that a target can be resolved to an emitted Property, Entity/system member, or StatisticalModifier under the frozen normalization rules; it does not invent a target.

When a Constraint targets an entire system, `on` resolves to its deterministic role-derived display label. A prediction may identify that system through either its model-supplied container string or the derived display string; canonicalization records the derived string. The original container string may therefore act only as a cross-reference alias during validation. Its text is never compared by the evaluator; the scored Constraint target is the role-derived value.

### 7.6 Three representations of a prediction

The database distinguishes:

1. **Raw response:** exact provider text/envelope, untouched.
2. **Extracted candidate:** the JSON value obtained by the documented extractor, before semantic modification.
3. **Canonical prediction:** a validated deterministic record used by the evaluator, with stable serialization and derived unscored system metadata.

This separation makes robust extraction possible without hiding what the model actually returned.

## 8. Prompt contract

### 8.1 Prompt families

Stable IDs and display names are:

| ID | Display name |
|---|---|
| `strict-minimal` | Strict minimal |
| `constraint-decomposition` | Constraint decomposition |
| `matrix-decomposition` | Matrix decomposition |

Templates are immutable after freeze. A text change creates a new prompt version and hash.

D-021 preserves the historical no-interpretation/no-inference prompt policy with only minor changes needed for the agreed six-field schema, system shapes, prompt names, and validation-feedback workflow. The approved demonstrations contain several RDF-derived labels that are not literal definition spans, and the Matrix instructions conflict with some geographic Matrix examples. `docs/prompt-specification.md` inventories these accepted limitations. Do not change gold answers, broaden semantic instructions, or retry valid predictions to repair those limitations. Future template versions retain source hashes and a reviewable diff against the historical prompts; their final bytes/hashes are implementation artifacts still to be produced.

### 8.2 Render input

The prompt renderer receives:

- Prompt template ID/version/hash
- Exact lexical JSON Schema bytes/hash
- Ordered zero/one/three/five canonical demonstrations
- Exact target definition
- Rendering-engine version

It does not receive target gold fields, category, source filename, IRIs, linked entities, RDF, or SHACL.

### 8.3 Render output

The renderer returns a UTF-8 message array and a human-readable full prompt representation. It stores:

- Exact message roles and text
- Serialized byte length and character length
- Template, schema, example, definition, and final prompt hashes
- Demonstration IDs and order

The same rendered base prompt is the source for all attempts of one task.

### 8.4 Correction prompt

Attempts two and three preserve the base prompt and add a versioned correction section containing:

- Previous attempt number
- Exact previous raw model text
- Ordered extraction/schema/semantic error records
- Instruction to return only one corrected JSON object

The renderer must never include another variable’s response or error.

## 9. Experiment configuration

### 9.1 Single editable file

`parameters.yml` is the only human-edited experiment configuration. `schemas/parameters.schema.json` will reject unknown keys, wrong types, invalid ranges, duplicate model IDs, unsupported provider names, and incompatible combinations.

Before planning, the system stores:

1. Exact original YAML bytes and hash.
2. Parsed configuration.
3. Environment-independent resolved configuration.
4. Provider capability-resolution report.
5. Canonical JSON configuration and hash.

Secrets are referenced only by environment-variable names and are excluded from all snapshots.

### 9.2 Selected providers and editable model lists

Configuration schema version `2.1` uses `campaign.providers`: `[psnc]`, `[openrouter]`, or `[psnc, openrouter]`, and the D-027 pre-run estimate policy. It replaces the former scalar provider field; supplying both forms is invalid. The selected list is non-empty and unique. Each selected provider owns a `models` list; only its enabled entries participate. Concrete model IDs are unique within their provider, not across providers. Lists may have different lengths in later campaigns, with no fixed three-model limit.

The selected exact IDs are PSNC `GLM-5.2`, `Qwen3.8-27B`, `DeepSeek-V4-Flash`; OpenRouter `qwen/qwen3-8b`, `qwen/qwen3-32b`, `openai/gpt-4o-mini`. D-032 withdrew `inclusionai/ling-3.0-flash` in favour of `openai/gpt-4o-mini`; the count stays at six, so the grid arithmetic below is unchanged. `openai/gpt-4o-mini` is the only hosted closed-weight selection, making metered price evidence load-bearing. `docs/model-catalog.md` records source provenance, display order, SDK routing, and unresolved deployment capabilities. These models are selected in the draft, but no availability, reasoning-control, or live-readiness claim follows from their names.

Canonical provider/model set order determines scientific fingerprints; original YAML order remains display/provenance evidence. Disabled models and inactive providers never expand into tasks. A live-disabled draft may retain unknown capabilities for an enabled intended model, but exact plan freeze and live preflight reject that unresolved profile rather than skipping it or inventing a capability. Active changes after freeze create a new campaign; resume always uses its original stored selection.

### 9.3 Grid factors

The initial documented factors are:

```text
prompt = strict-minimal, constraint-decomposition, matrix-decomposition
shots = 0, 1, 3, 5
temperature = 0.0, 0.5, 1.0, 2.0
reasoning = capability-aware disabled/enabled
repetition = 1 at every temperature, for every selected provider/model (D-029)
```

Every resolved combination is materialized before execution. The planner reports target count, planned and maximum provider requests, and token/cost estimates with explicit assumptions. A conservative upper bound is labelled conditional on verified output, reasoning, context, and price limits; an unknown bound is never presented as a guaranteed number.

### 9.4 Grid-count invariants

The four temperatures contribute `1 + 1 + 1 + 1 = 4`
repetition-specific conditions. Therefore, for each applicable reasoning profile:

```text
ranked configurations = 3 prompts × 4 shots × 4 temperatures = 48
resolved runs          = 3 prompts × 4 shots × 4 repetition conditions = 48
tasks / initial calls  = 48 runs × 97 variables = 4,656
maximum provider calls = 4,656 tasks × 3 attempts = 13,968
```

A reasoning-capable model has two profiles, so it contributes 96 ranked
configurations, 96 resolved runs, 9,312 tasks, and at most 27,936 provider
calls. A model with `not_applicable` reasoning contributes the one-profile
counts. Campaign totals sum these values over each selected provider's enabled
models, never a Cartesian product of providers and all catalog models. With six
one-profile models, the totals are 288 configurations, 288 resolved runs,
27,936 tasks, and at most 83,808 requests; six two-profile models double these
counts. The actual count waits for verified model capabilities. Preflight shows
provider subtotals and campaign totals plus token and applicable monetary bounds.
Non-billed PSNC still requires explicit token/output/rate limits and documented
zero billing. Monetary caps are optional for every provider selection; no cap is required to create or execute an otherwise authorized plan.

For OpenRouter alone with its three selected models, the one-profile-per-model scenario is 13,968 initial and at most 41,904 total calls; two verified profiles on every model give 27,936 initial and at most 83,808 total calls. Both selected providers with two profiles on all six models give 55,872 initial and at most 167,616 total calls. These are prospective grid counts, not current executable readiness or provider capability verification.

D-029 reduces resolved runs/tasks/request bounds by 75% without removing parameter configurations. Both repetition fields remain configurable for future campaigns. Freeze the new plan and cost estimate; never change a stored campaign's repetition membership during resume. Content corrections and safe transient resends remain part of the same three-request task allowance, not independent repetitions.

### 9.5 Reasoning capability contract

Reasoning is a normalized experimental factor, not one universal API field.

For every model, configuration declares:

```text
supports_reasoning_control
normalized modes allowed
disabled provider request mapping
enabled provider request mapping
reasoning response/usage fields, if any
```

For a capable model, the planner creates both `disabled` and `enabled` runs. For an incapable model, it creates one `not_applicable` run and sends no reasoning control. Unsupported mappings fail preflight.

For PSNC Qwen-style models, mappings may use `enable_thinking` and `chat_template_kwargs.enable_thinking`. For OpenRouter models, mappings may use the provider’s reasoning object or another documented model-supported field. Exact enabled effort/level must be explicit; omission cannot be presented as a reproducible controlled value.

The database stores the normalized mode and exact serialized native request fields. “Enabled” across providers is reported as a configuration category, not assumed to represent equal reasoning compute.

### 9.6 Scientific versus operational settings

Scientific settings—including model, prompt, schema, examples, temperature, reasoning mode, explicit seed value when separately approved/supported, output limit, and evaluator—are included in run fingerprints. Seed is not part of the current documented grid; unsupported or unused seed control is recorded explicitly rather than invented.

Operational settings—including worker count, polling interval, and log verbosity—are recorded but do not change scientific identity unless they can affect provider behavior. Concurrency and provider routing are always retained for audit.

## 10. Provider interface

### 10.1 Provider-neutral request

The orchestration layer sends an adapter:

```text
task and attempt IDs
provider/model identity
exact message array
temperature and supported sampling values
reasoning mode and resolved native mapping
maximum output tokens
timeout
idempotency key, if supported
```

### 10.2 Provider result

The adapter returns one typed result containing:

```text
sanitized exact request body
untouched response body
assistant text
optional reasoning text
requested and returned model IDs
provider request/system fingerprint IDs
HTTP status
finish reason
all reported token categories
start/end/latency
error classification and sanitized error payload
delivery certainty
```

Authorization headers and secrets are never returned or persisted.

### 10.3 OpenRouter

OpenRouter uses its own client from the shared OpenAI Python library, with explicit `OPENROUTER_API_KEY`, SDK base URL `https://openrouter.ai/api/v1`, and Chat Completions path `/chat/completions`. Model names remain exact provider identifiers. Provider routing metadata is preserved when returned.

### 10.4 PSNC

PSNC uses:

```text
SDK base_url = https://llm.hpc.psnc.pl/v1
POST {base_url}/chat/completions
Authorization: Bearer ${PSNC_API_KEY}
Content-Type: application/json
```

Non-streaming response text is extracted from `choices[0].message.content`, with the documented compatible fallback only when required. Model-specific thinking fields come from the resolved configuration, not hard-coded assumptions.

Both adapters use the same pinned OpenAI Python SDK with separate provider-scoped clients, credentials, base URLs, and connection lifecycles. `client_library: openai` declares the transport library, not an additional selected inference provider. SDK base URLs include the version prefix once; do not produce `/v1/v1/chat/completions`. `PSNC_API_BASE_URL`, when supplied, overrides the complete SDK base URL. Use the raw-response interface and exception response body to retain full evidence before extracting assistant text. The owner-supplied example is routing/model evidence only; its toy prompts and generic system messages do not replace the frozen experiment prompts. See `docs/model-catalog.md` for evidence and SDK documentation.

### 10.5 Non-streaming experiment calls

Primary benchmark calls are non-streaming. This makes one response envelope, usage accounting, delivery status, extraction, and attempt boundaries easier to audit. A future streaming mode would require a separate protocol version.

### 10.6 No adapter retry

Adapters make at most one provider request per invocation (zero on local validation failure). Set SDK `max_retries=0` and disable transport retries in both provider clients. They classify the result but never call themselves again. Retry decisions belong exclusively to the task state machine; raw-response parsing and evidence recovery make no new request. SDK/transport versions and final wire behavior must be pinned and verified with offline fixtures.

## 11. Extraction, validation, and attempt protocol

### 11.1 Write-ahead evidence

Before dispatch, the system commits the task, attempt number, exact sanitized request body, prompt hash, parameters, and deterministic idempotency key. After receipt, it persists the untouched response bytes/envelope before parsing. A crash after receipt can therefore resume at extraction without paying for a replacement call.

### 11.2 Deterministic extraction order

The extractor receives the exact assistant-visible text; the untouched provider
response has already been stored. It applies the following protocol in order:

1. Remove leading and trailing whitespace and strictly parse the whole text. If
   the result is an object, accept it as `whole_response`.
2. If that whole-text parse instead returns a JSON string, unwrap exactly one
   string layer, record `json_string_unwrapped`, and restart this protocol on
   the decoded string. A second string wrapper is rejected. If the successfully
   parsed whole-text value is any other JSON type, reject it as
   `top_level_not_object`; do not mine objects from inside an array or scalar.
3. If whole-text parsing failed, enumerate Markdown fences labelled `json` or
   having no language label and strictly parse each complete fence body. Accept
   only when there is exactly one parsed top-level object and no competing
   parsed JSON value. Multiple parsed candidates are `ambiguous_json`; a sole
   parsed non-object is `top_level_not_object`.
4. Only when no eligible fence body parses as JSON, run a quote-aware and
   escape-aware balanced-brace scan over the current text. Strictly parse every
   complete outermost object span. Accept exactly one parsed object; reject
   multiple objects as `ambiguous_json` and zero as `no_valid_json_object`.

The protocol handles ordinary prose, Markdown fences, and a single quoted JSON
wrapper without attempting to repair the model's content. A greedy `\{.*\}`
regular expression, first-object selection, trailing-comma repair, quote
replacement, field insertion, field renaming, and type coercion are prohibited.

The extraction record includes the selected strategy, whether one string layer
was unwrapped, candidate count, candidate offsets and their coordinate space,
parse diagnostics for every inspected candidate, duration, safety-limit status,
and selected-candidate hash. Resource limits on response bytes, nesting depth,
fence count, and candidate count are frozen in the implementation manifest.

### 11.3 Validation stages

Validation occurs in this order:

```text
transport envelope
-> extraction
-> JSON parse
-> lexical JSON Schema
-> semantic cross-field checks
-> canonicalization
```

Every stage produces a typed pass/fail event. Errors contain a stable code, JSON Pointer where applicable, human-readable message, validator version, and offending value summary without secrets.

### 11.4 Attempt budget

Each task has `max_generation_attempts_per_task = 3`:

| Attempt | Prompt | Maximum provider requests |
|---:|---|---:|
| 1 | Frozen base prompt | 1 |
| 2 | Base plus attempt-1 response/errors | 1 |
| 3 | Base plus attempt-2 response/errors | 1 |

This is three total requests, not an initial request plus three retries. SDK automatic retries are configured to zero when the SDK permits. If a library cannot guarantee this boundary, it cannot be used for the primary experiment.

### 11.5 Retry eligibility

Another attempt may be scheduled only when budget remains and the preceding attempt ends in a registered retryable state, including:

- No usable JSON object
- JSON parse failure
- Lexical-schema failure
- Semantic-validation failure
- Empty or HTML response
- A classified transient provider failure for which issuing another request is permitted

A schema-valid but scientifically poor, incomplete, or all-empty decomposition is not retryable. It is the accepted prediction and is evaluated. No gold field or similarity score is consulted during validation or retry decisions.

A correction prompt is possible only when raw model output and validation errors exist. After a transport failure with no model output, the next attempt uses the unchanged base prompt and stores the transport reason.

Before first live dispatch, preflight validates all selected providers and verifies that the pre-run estimate was disclosed and the matching live scope separately authorized. During execution, authentication, permission, missing-model, and provider/model-specific configuration failures pause the affected scope; eligible work at other selected providers continues. If an optional monetary cap is configured, its exhaustion blocks positive-cost dispatch, not documented zero-cost PSNC work. Shared database/integrity failures or an explicit global stop pause all dispatch. Repeatedly sending the same invalid request is prohibited, and any unresolved scope prevents complete campaign status.

### 11.6 Ambiguous delivery

If the provider may have accepted a request but no response was durably received, the attempt becomes `ambiguous_delivery`. The system first uses provider request lookup or idempotency evidence when available. It does not blindly issue another paid request. Any approved replacement is a new attempt and consumes the remaining three-request budget.

### 11.7 Terminal invalid prediction

If three delivered model responses exhaust the task budget without yielding a valid canonical prediction because of extraction, schema, or semantic-validation failure, the task records:

```text
final status = terminal_invalid
all three attempt records
final error bundle
explicit empty six-field prediction
failure-policy version
```

The evaluator receives that explicit empty representation. The variable is not dropped, and all expected gold components contribute according to the frozen scorer policy.

If the task instead exhausts its request budget because of provider, authentication, infrastructure, budget, or unresolved-delivery failure, it records an operational terminal/paused state and no model-performance prediction. That target remains visible, and publication promotion is blocked until the operational gap is resolved under an approved campaign policy. Infrastructure failure is never scored as if the model returned an empty decomposition.

## 12. PostgreSQL persistence contract

### 12.1 Authority

PostgreSQL 16 is authoritative for relationships, state, raw evidence, and results. File exports are derivatives. Full prompts and raw response text are stored in PostgreSQL `text`/`jsonb` columns or a hash-verified database artifact table; they are not represented only by filesystem paths.

### 12.2 Conventions

- UTC `timestamptz` for timestamps
- UUIDv7 generated by the application where supported
- Lowercase 64-character SHA-256 with constraints
- `numeric` for money; exact integer numerator/denominator receipts for confusion contributions and metrics, with `numeric` decimal derivatives for display
- Explicit foreign keys and uniqueness constraints
- Append-only terminal evidence
- No cascade deletion of experiment evidence
- `jsonb` only for provider-specific secondary metadata; join/filter fields remain normalized columns

### 12.3 Corpus and registry tables

| Logical table | Required responsibility |
|---|---|
| `corpus_snapshot` | Repository, tag, commit, tree, manifest, record count, state |
| `source_file` | Relative path, Git blob, exact-byte hash, category path |
| `science_category` | Snapshot-scoped hierarchy and optional reporting alias |
| `variable` | Stable source ID, label, definition, source, category leaf |
| `gold_decomposition` | Canonical six-field JSON, schema/importer versions and hashes |
| `evaluation_population` / `member` | Exact 97-variable membership, corpus/gold hashes, exclusions, manifest identity |
| `demonstration_set` / `member` | Exact ordered five-item pool |
| `prompt_version` | Stable prompt family, bytes, hash, rendering contract |
| `schema_version` | Lexical-schema bytes, draft, semantic-rule version, hash |
| `scorer_version` | Evaluator code/config/model hashes and compatibility note |
| `provider` / `model_configuration` | Provider profile, exact model ID, capabilities, reasoning mappings |

Frozen version rows reject mutation. Corrections insert new versions.

### 12.4 Campaign and execution tables

| Logical table | Required responsibility |
|---|---|
| `campaign` | Original/resolved config, selected provider set, fingerprints, optional global cap, estimate/disclosure/authorization links, state |
| `campaign_provider` | Selected provider membership, provider-owned model/config links, billing/optional-cap/rate/cooldown state |
| `resolved_run` | One model/parameter/repetition combination |
| `task` | One run-variable unit, fingerprint, state, lease, attempt count |
| `attempt` | Number 1–3, exact request, status, delivery state, provider IDs |
| `transport_event` | Dispatch/HTTP/error/receipt events without hidden calls |
| `extraction_event` | Strategy, offsets, candidate, errors |
| `validation_event` | Ordered validator stage, version, result, complete errors |
| `prediction` | Accepted canonical or explicit empty terminal prediction |
| `task_event` | Append-only state transition history |
| `worker_session` | Worker identity/version, lease heartbeat, start/end |

Key constraints include:

```text
UNIQUE campaign(campaign_fingerprint)
UNIQUE resolved_run(run_fingerprint)
UNIQUE task(task_fingerprint)
UNIQUE attempt(task_id, attempt_number)
CHECK attempt_number BETWEEN 1 AND 3
UNIQUE prediction(task_id)
```

### 12.5 Attempt evidence

Every attempt stores:

- Complete rendered prompt and messages
- Sanitized exact provider body
- Untouched response text and JSON envelope
- Prompt/request/response hashes
- Attempt number and correction-parent attempt
- Provider/model requested and returned
- Normalized and native reasoning configuration
- All sampling parameters
- Provider request/system fingerprint IDs
- Token categories, finish reason, status
- Start, first byte/token when available, finish, latency
- Error class/status and sanitized error payload
- Delivery certainty and retry decision

Authorization headers, cookies, API keys, and database secrets are prohibited.

### 12.6 Evaluation tables

| Logical table | Required responsibility |
|---|---|
| `evaluation_run` | Prediction set plus scorer/code/environment fingerprint |
| `evaluation_item` | Variable, gold/prediction hashes, item/failure status |
| `match_record` | Component, structure/role, normalized values, similarity, decision |
| `confusion_contribution` | Exact TP/FP/FN/TN numerator/denominator, decimal derivative, member counts/normalization denominator when applicable, and rule ID |
| `metric_value` | Named metric, numerator, denominator, support, value |
| `configuration_rank` | Population-complete configuration identity, repetition metrics, primary rank value, shared rank, and rank-policy hash |
| `analysis_slice` / `member` | Category, subcategory, config, and other frozen memberships |

Only storing final F1 is prohibited. Component evidence must regenerate every aggregate.

### 12.7 Usage and cost

Attempt records preserve provider-reported raw usage and normalized token categories. Cost rows reference a versioned provider/model price card, original currency, quantity, rate, actual/estimated state, and adjustment history. Missing provider data remains null with an availability reason, never zero by assumption.

For the owner's PSNC access, record `providers.psnc.billing.mode: non_billed`, an explicit account-scoped billing basis, and zero monetary cost. Missing token counts still remain unknown; no-charge cost is justified by billing evidence rather than inferred from absent usage. OpenRouter is `metered` and requires frozen price evidence. The common cost manifest covers all selected providers. Missing billing or price information is not evidence of free access.

D-027 replaces mandatory spending ceilings with mandatory advance cost disclosure and separate explicit live authorization. `cost_accounting.require_pre_run_estimate: true` and `estimate_policy_version: pre-run-estimate-v1` require an immutable estimate tied to the resolved plan before a live execution request can be authorized. The report gives per-model, per-provider, and combined planned/maximum-three-attempt call counts; input, output, and reasoning-token assumptions; correction-prompt growth; frozen price-card and exchange-rate provenance; a labelled expected scenario/range; and a conservative upper-bound scenario conditional on its stated limits. Missing inputs are shown as unresolved, never fabricated. A paid calibration call is not permitted merely to obtain this estimate; an optional live canary needs its own disclosed estimate and separate authorization.

The optional `maximum_provider_cost.amount` and `maximum_campaign_cost.amount` default to `null`, meaning no enforced monetary cap. An explicit zero means a zero spending ceiling, not an unknown value; it permits evidenced zero-cost dispatch but blocks positive-cost reservations. Positive values set optional ceilings. Always persist cost estimates, reservations, settlement, and ambiguous exposure even when uncapped. Lock accounting rows consistently across workers and atomically check only non-null limits. Exhausting an explicitly configured paid limit cannot authorize overspend; evidenced non-billed dispatch reserves zero and may continue. An estimate is not a cap: actual cost may exceed the disclosed expected scenario, and the report must say so. Attempt, rate, concurrency, delivery, and integrity protections are unchanged.

A capped paid request requires a defensible reservation bound; an uncapped request still requires useful price-supported estimates and explicit uncertainty, but not a guaranteed upper bound on its final invoice. Distinguish conditional token-bound scenarios from provider-enforced billing bounds. Unknown prices or absent usable estimate evidence block live admission in either mode. A missing guaranteed bound alone is not a new spending-cap requirement for uncapped execution.

## 13. Planning, concurrency, and resume

### 13.1 Preflight

Before task creation, plan-preparation checks verify the following non-execution requirements. Live readiness additionally requires a disclosed estimate and explicit authorization tied to the resulting plan; preparing the plan or estimate does not dispatch requests:

- Corpus, manifest, and demonstration hashes/counts
- Exactly 97 evaluation-population members
- No demonstration overlap with the evaluation population
- Prompt/schema/scorer hashes
- Non-empty, unique selected provider set
- Non-empty enabled model list for every selected provider, with provider-scoped concrete ID uniqueness
- Model capability and reasoning-profile mappings
- Supported parameter combinations
- Credentials are present for live mode without reading them into evidence
- Selected-provider/model prices or explicit non-billed basis and valid optional monetary caps; unknown billing is rejected
- Database migration version
- Offline provider-contract fixtures

Any failure occurs before a provider request.

### 13.2 Deterministic expansion

For each selected provider, expand the following grid, then combine the resulting task sets:

```text
enabled model belonging to that provider
x model-applicable reasoning profile
x prompt variant
x shot count
x temperature
x applicable repetition
x target variable
```

Models without reasoning control receive one `not_applicable` profile. Capable models receive exactly the configured `disabled` and `enabled` profiles.

The plan shows logical tasks and a maximum of three requests per task. The user-visible summary includes the 97-variable population count, provider/model subtotals, combined calls/tokens, zero versus metered billing, the D-027 cost estimate and assumptions, and optional cap status (`uncapped` when null). A task has exactly one provider/model owner; there is no provider fallback or cross-provider model pairing.

### 13.3 Fingerprints

Task fingerprints hash canonical identities for corpus item, evaluation-population manifest, provider/model, prompt, schema, examples/order, all scientific parameters, reasoning mapping, repetition, retry protocol, and code/environment versions. Attempt number is a child identity, not a new scientific condition.

Changing unused provider catalog text changes complete YAML evidence but not the resolved selected-set fingerprint. Canonical selection ordering prevents incidental display reordering from changing scientific identity. Changing an active provider/model or scientific value creates a new campaign snapshot and campaign-scoped run/task identities. It never appends new YAML entries to an existing resumed campaign or implicitly reuses another campaign's requests.

### 13.4 State machine

Task states are equivalent to:

```text
planned -> queued -> leased -> generating -> response_stored
         -> extracting -> validating -> retry_pending
         -> prediction_ready -> scoring -> complete

After attempt 3 content failure -> terminal_invalid -> prediction_ready
Any active stage -> operational_failed | ambiguous_delivery
Any claim/dispatch boundary -> paused_budget | paused_configuration
```

Attempts have their own state progression from request persisted through selected or terminal failure. Every transition creates a task event.

### 13.5 Task claims

Workers claim tasks in short transactions using row locking such as `FOR UPDATE SKIP LOCKED`. A claim records worker, lease expiry, heartbeat, and row version. Provider calls never hold a database transaction open. `provider_fair` scheduling visits eligible provider queues without exhausting all worker slots on one provider's cooldown. Provider/model concurrency and rate gates remain independent under the global worker limit. Pre-dispatch waiting consumes neither a generation attempt nor a held worker/database transaction; persisted due-times let eligible work continue and survive restart.

### 13.6 Resume

Resume reconciles rather than restarts. It reports:

```text
expected/stored tasks
states and terminal counts
verified responses/predictions/scores
expired leases
ambiguous deliveries
remaining attempt budgets
actual/estimated cost and remaining optional cap (or uncapped)
orphan rows or artifacts
```

If a response is already stored, resume continues at extraction or validation. If a prediction is stored, it continues at scoring. Completed tasks are not called or scored twice. Configuration/hash mismatch blocks resume.

### 13.7 Graceful stop

Interrupt and optional-cap-stop handlers stop new claims, let safely in-flight work reach the next durable checkpoint, release or expire leases, record state, and exit with a status that distinguishes pause from failure. Exceeding an expected estimate alone is not a configured stop condition.

### 13.8 Run-to-completion behavior

With `continue_until_complete: true`, one separately authorized `run`/`resume` invocation advances all selected model grids, validation, scoring, aggregation, ranking, and required exports. With `continue_unaffected_providers: true`, a provider/model pause affects only its scope; eligible other work continues. Only explicitly configured monetary caps stop positive-cost reservations; no cap is required or enabled by default, and documented zero-cost work can continue. User interruption and shared integrity/database failures retain their global stop semantics. Resume preserves the original plan, estimate, and live-authorization references; it does not silently authorize a changed model/grid or a new live canary.

Recoverable cooldowns remain scheduled without resetting the task's three-request counter. A completed task, exhausted operational task, unresolved delivery, or credential failure cannot be recycled into another provider/model or a fourth request. If all remaining work requires external correction or renewed authority, persist the blocking reasons and return a resumable paused result instead of a busy loop or false success. A process restart still invokes `resume`; this documentation does not create a desktop automation or independent scheduler.

Campaign `complete` requires exact expected coverage across every selected provider/model/configuration/repetition, all terminal scored predictions, immutable aggregates/ranking, and required report manifests. Interim reports may expose completed configurations and unrankable gaps, but cannot claim full campaign completion. Completing PSNC alone does not complete a campaign that also selected unfinished OpenRouter work.

## 14. Evaluator contract

### 14.1 Isolation

`iadopt_eval` is a pure package. Its core imports no provider, database, network, filesystem-discovery, clock, or random-number service. Inputs and outputs are immutable serializable records. Database integration belongs to I-ADOPT Lab, not the evaluator.

### 14.2 Evaluated components

The six component slots are:

```text
hasStatisticalModifier
hasProperty
hasObjectOfInterest
hasMatrix
hasContextObject
hasConstraint
```

Label, definition, comment, category, URIs, and system container identifiers do not contribute to scores.

### 14.3 Exact string matching

Strings are lowercased and have leading/trailing whitespace removed, matching January behavior. No additional stemming, punctuation deletion, translation, or ontology lookup is silently added.

### 14.4 Close string matching

Non-exact strings are encoded using pinned `sentence-transformers/all-MiniLM-L6-v2` bytes and dependencies. Cosine similarity at threshold `0.80` determines close equivalence. Exact model revision, files, tokenizer, framework, device, and dtype are manifest entries. Every similarity score is retained.

Cached candidate revisions are tested against historical fixtures. The active revision is frozen only after parity evidence is documented.

### 14.5 Simple-component confusion

The January policy remains:

- Present correct prediction: `TP = 1`
- Present wrong prediction: `FP = 1`, without an additional FN
- Missing prediction for present gold: `FN = 1`
- Both absent: `TN = 1`
- Prediction present for absent gold: `FP = 1`

This is preserved for paper comparability and documented rather than silently “corrected.”

### 14.6 Entity/system member matching

Handle empty values with the historical FN/FP/TN rules before selecting the comparison branch. Non-empty scalar/scalar comparisons retain Section 14.5. For Object of Interest, Matrix, or Context Object with at least one non-empty system, D-022 expands the simple entity to one member, symmetric systems to their parts, and asymmetric systems to two ordered role occurrences. Container labels never enter a matcher. Preserve original lexical values and pointers separately from normalization and assignment evidence.

The accepted dispatch is:

| Comparison | Eligible pairs | Equivalence rule |
|---|---|---|
| Symmetric/symmetric | Unordered, one-to-one parts | Literal equality in both modes; January case/whitespace sensitivity and no embeddings retained |
| Asymmetric/asymmetric | Corresponding first/second slots only | Historical scalar Exact/Close from Sections 14.3–14.4 |
| Simple/system, either direction | Singleton to at most one member | Historical scalar Exact/Close |
| Symmetric/asymmetric, either direction | Unordered, one-to-one membership | Historical scalar Exact/Close; shared ordered-role evidence unavailable |

Asymmetric slot 1 is numerator or the January source fallback; slot 2 is denominator or target. Preserve the original role-pair names in evidence. When both sides provide ordered roles, cross-slot matches are prohibited. A simple or symmetric side provides membership without asserting numerator/denominator/source/target correctness. Mixed symmetric/asymmetric comparisons may receive full membership credit without establishing ordered-role equivalence; the representation and missing role evidence must remain visible in reports.

Qualifying unordered matches maximize cardinality first, then summed pair similarity, then use the lexicographically smallest sorted pair-index list. Canonical member indices use exact UTF-8 label order for symmetric parts, fixed first/second slots for asymmetric systems, and index zero for a simple entity. Keep original positions as separate provenance. One occurrence can appear in only one pair; normalization-equivalent labels or repeated role labels do not let one member fill several roles. This assignment is independent of symmetric input order and is not the historical greedy Constraint algorithm. Pair eligibility uses the branch's exact literal test or the inclusive scalar Close threshold, never a threshold over aggregate system similarity.

### 14.7 Fractional system contributions

Let `g` and `p` be the positive gold/predicted member counts and `m` the matching cardinality. Accepted D-022 defines:

```text
U  = g + p - m
TP = m / U
FP = (p - m) / U
FN = (g - m) / U
TN = 0
```

Every non-empty system component has total contribution one. Retain these fractions directly; do not apply January's former whole-component threshold to them. For gold `water + air`, a simple `water` prediction gives TP `1/2`, FN `1/2` and component F1 `2/3`; a symmetric `water + soil` prediction gives TP/FP/FN each `1/3` and component F1 `1/2`. A non-empty no-match system comparison has both FP and FN; this does not change the FP-only rule for wrong scalar/scalar predictions.

For literal symmetric/symmetric matching, TP equals Jaccard intersection-over-union, but is fractional evidence instead of a pass/fail threshold. Exact and Close continue to use the same literal part identity in that branch. The schema rejects exact duplicate parts; membership scoring never deduplicates additional normalized-equivalent occurrences. `docs/scorer-parity.md` includes full examples, matching tie rules, and before/after fixtures.

Persist `g`, `p`, `m`, `U`, eligible/selected/unmatched member evidence, actual role names, absent-role-evidence flags, branch identity, and exact fraction numerators/denominators. Finite decimal displays cannot exactly represent thirds and must not determine shared ranking ties. Finalized values from unchanged historical branches retain their exact source numerical ratio and trace; their computation and threshold behavior remain the regression target. Aggregation and ranking use the exact stored contribution receipts, with decimal presentation precision recorded separately.

### 14.8 Constraints

Candidate pairs use the January greedy matching behavior over the average of `label` and `on` similarity. Constraint strings are stripped, lowercased, and have internal whitespace collapsed. Before that normalization, `on` removes a prefix only when the text matches `<key>: <value>` and `<key>` is an exact case-sensitive member of the six `ONTO_KEYS` names.

The scorer repeatedly chooses the global maximum pair and masks its row and column. Equal maxima use first row-major position, matching the historical `numpy.argmax` behavior. Thus the procedure is generally insensitive to simple list reordering when the optimum is unique, but tied similarity matrices can be input-order-sensitive; the rewrite must preserve and fixture this edge behavior rather than claim stronger invariance.

For `n` gold Constraints, each matched `label` and `on` contributes `1/(2n)`. A field comparison meeting the mode threshold contributes that unit to TP; otherwise it contributes the unit to FP. Unmatched gold fields add FN and unmatched prediction fields add FP. The historical numerical correction is retained: a deficit larger than `1e-6` is added to FP, while a total larger than `1 + 1e-6` divides TP, FP, and FN by that total. Empty/empty returns TN `1`; empty gold with any prediction returns FP `1`. All details receive golden fixtures.

### 14.9 Aggregation

For each resolved configuration and repetition, the primary result sums fractional contributions over all six slots and all 97 evaluation variables, then calculates:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

Zero metric denominators produce `0`, preserving January's rule. Numerators, denominators, support, exact reduced rational values, and decimal presentation derivatives are stored with every metric. Sum exact contribution receipts before division; never rebuild an aggregate from rounded displayed values.

Every unrounded repetition-level metric and its TP/FP/FN/TN inputs are stored. Exact metrics, per-component values, per-variable contributions, validity rate, and category/subcategory slices are required outputs.

Under D-029, this campaign has one repetition per configuration at every temperature. The existing mean-repetition ranking therefore equals that single repetition's micro Close F1; no ranking policy/version change or per-variable macro averaging is introduced. Retain the repetition identity and generic mean calculation for future campaign configurability.

### 14.10 Configuration ranking

D-023 accepts the following primary/tie rules under ordering policy `mean-repetition-micro-close-f1-v1`. D-024 defers additional descriptive-statistics calculations while requiring complete retained evidence.

A **configuration** contains provider, model, model revision, reasoning profile,
prompt, shot count, temperature, output limit, supported sampling parameters,
schema, retry protocol, and scorer identity. It does not contain target-variable
or repetition identity.

For each configuration, the system computes micro Close F1 separately for every
declared repetition over the same 97 variables. The primary ranking value is the
arithmetic mean of those unrounded repetition-level micro Close F1 values. Each
repetition has equal weight within its configuration. This is neither the
arithmetic mean of 97 per-variable F1 values nor F1 recomputed after pooling
contributions from every repetition.

Configurations are ordered by the primary value descending. Exact numerical
ties share the same competition rank; secondary metrics do not silently break a
tie. Exact/Close Precision, Recall, F1, and TP/FP/FN/TN remain stored for every
variable, component, and repetition, alongside validity, category identities,
latency, tokens, attempts, and cost. Additional dispersion and distribution
summaries are deferred under D-024. A later report must identify whether it
summarizes the 97 variable values or repeated runs, and cannot silently substitute
one for the other.

A configuration is rankable only when every planned variable/repetition task
has a terminal scored result and no unresolved operational failure. Explicit
empty predictions produced by three content-invalid responses are scored and
remain in the denominator. Incomplete configurations are retained with a
machine-readable `not_rankable` reason. Every rank and all lower-ranked results
are stored under a versioned rank-policy and evaluation-population hash; ranking
never deletes or overwrites generation or evaluation evidence.

### 14.11 Failed tasks

A terminal-invalid content task is evaluated through the explicit empty prediction. It cannot disappear from denominators. Operational failures are reported separately, receive no model-performance score, and block publication promotion until resolved or covered by a frozen failure policy.

### 14.12 Versioning and parity

The scorer identity records:

```text
January reference commit/file hash
active evaluator package/code hash
normalization rules
matching and assignment rules
embedding model manifest
threshold
zero-denominator behavior
system-identifier exclusion decision
golden-fixture hash
```

Changing any semantic rule creates a new evaluation run and scorer version without repeating model calls or overwriting prior scores.

## 15. Reporting contract

Reporting is read-only with respect to experiment evidence. It derives reproducible tables from explicit campaign and evaluation IDs, never an implicit “latest” result.

Required dimensions include:

- Provider and exact model
- Reasoning profile and native value
- Prompt family
- Shot count
- Temperature and repetition
- Evaluation-population and ranking-policy identity
- Complete-configuration rank and rankability status
- Science category, subcategory, and full path
- Component
- Exact versus close mode
- Validity/final-failure class

Required summaries include:

- Unrounded Exact/Close TP/FP/FN/TN, Precision, Recall, F1, denominators, and support
- Per-variable, per-component, and overall metrics
- Every repetition-level metric and complete-configuration ranking
- Schema-validity and terminal-invalid rates
- Attempt and validation-error distributions
- Mean repetition-level micro Close F1 used for the accepted ranking
- Tokens, latency, calls, retries, and cost
- Category/subcategory counts next to every sliced metric

CSV, canonical JSON, and optional XLSX exports contain source campaign/evaluation IDs and hashes. PostgreSQL remains authoritative.

Additional statistical analysis is deferred until the database is populated (D-024). Later versioned queries can calculate arithmetic mean, median, variance, mode, standard deviation, minimum/maximum, range, and IQR over the 97 individual variable scores or other explicitly selected scopes. Record whether the query groups by repetition, category, or configuration and its sample/population variance, quantile, mode/tie, and any uncertainty conventions. None of these additional statistics is a prerequisite for finishing generation, evaluation, or ranking. All necessary source values and counts must already be retained, so analysis requires no new LLM calls.

With D-029's singleton repetition, within-configuration run-to-run variability is not estimated. Do not report zero stochastic variance or repetition-based confidence intervals as observed evidence. Dispersion across different variables or configurations is not a replacement for independent repeated runs. Collecting those runs in the future would require new generation and separate authorization.

## 16. Reproducibility contract

### 16.1 Frozen evidence

Every campaign freezes or records hashes for:

- Code commit, tree state, and approved dirty diff if any
- Python/dependency lock and container image
- Corpus source and canonical projection
- Evaluation-population and demonstration manifests
- Prompt templates and rendered prompts
- Lexical schema and semantic rules
- Provider/model IDs and capabilities
- Complete original/resolved configuration
- Reasoning mappings
- Retry protocol
- Evaluator and embedding artifacts
- Price cards

### 16.2 Environment

`pyproject.toml` and `uv.lock` will pin Python dependencies. `.python-version`, `Dockerfile`, and `compose.yaml` define a reproducible local runtime and PostgreSQL 16 service. An external PostgreSQL DSN remains supported through an environment variable.

Locale, timezone, CPU/GPU type, library determinism flags, and embedding device/dtype are recorded. Provider-hosted generation cannot be made bitwise deterministic, so returned provider fingerprints, access timestamps, seeds where supported, and repetition identities remain essential evidence. D-029 collects only one repetition per configuration; retaining its identity does not establish run-to-run variability.

### 16.3 Secrets

`.env.example` contains only variable names. `.env` is ignored. The owner already has `OPENROUTER_API_KEY` and `PSNC_API_KEY` in the repository-root `.env`; implementation must support explicit `--env-file` selection of that existing file without copying it into `iadopt-lab/`. An optional runtime loader reads only allowlisted environment names referenced by the selected providers and `DATABASE_URL`; already-set process environment values win. It never shell-sources the file, evaluates expressions, interpolates variables, or imports unrelated keys. Runtime secrets remain in memory and are excluded from configuration/evidence: do not store `.env` content, its hash, copied files, credentials, or a secret-bearing DSN. Persist only required names and safe presence/source-kind facts. Sanitization occurs before request/header persistence. A release scan checks API keys, bearer tokens, private filesystem paths, email addresses, and provider-restricted data.

### 16.4 Documentation and code synchronization

Every executable module has a component contract defining inputs, outputs, state, failures, and tests. Public and private functions receive typed docstrings with summary, Args, Returns/Yields, Raises, Side Effects, and determinism/idempotency when relevant. Comments explain non-obvious scientific or concurrency reasons.

## 17. Testing strategy

### 17.1 Static and documentation checks

- Formatter and lint checks
- Strict type checking
- Docstring coverage and style
- YAML and JSON Schema validation
- Broken-link checks for local documentation
- Configuration examples validated against the parameter schema

### 17.2 Corpus tests

- Exactly 102 pinned Turtle source files
- Commit/tree/file hashes match the manifest
- One root, Property, and ObjectOfInterest per file
- Required definitions and labels present
- Category/subcategory counts regenerated from paths
- Every gold decomposition passes the lexical and semantic contracts
- Blank-node parsing yields byte-identical canonical JSON across repeated processes
- Demonstration paths and order match exactly
- No demonstration appears in the 97-variable evaluation population
- Every non-demonstration Corpus variable appears exactly once in that population

### 17.3 Schema tests

- Valid simple entity
- Valid symmetric system with at least two parts
- Invalid one-part symmetric system
- Valid source/target asymmetric system
- Valid numerator/denominator asymmetric system
- Invalid mixed or incomplete asymmetric system
- Unknown property rejection
- Explicit optional empties
- Valid and invalid Constraints
- Missing required top-level key

### 17.4 Extraction and retry tests

- Pure JSON response
- Fenced JSON
- Prose before/after one object
- Quoted JSON string
- Double-wrapped JSON string rejection
- Top-level array/scalar rejection without nested-object mining
- Braces inside escaped strings
- Multiple ambiguous objects
- Malformed/truncated JSON
- Invalid then corrected response
- Three invalid responses and terminal empty prediction
- Exactly three provider invocations, including transport failures
- No SDK/adapter hidden retries
- Correction contains only the preceding response/errors

### 17.5 Provider contract tests

Recorded fixtures verify exact OpenRouter and PSNC request construction, auth redaction, reasoning mappings, response parsing, usage preservation, error classification, and one-request-per-adapter-invocation behavior. Live tests are opt-in and excluded from normal test runs.

### 17.6 Evaluator golden tests

Hand-calculated and historical fixtures cover simple correct/wrong/missing values, empty gold, close threshold boundaries, asymmetric role reversal, symmetric reordering, system-identifier exclusion, constraints and fractional scores, both directions of structural mismatch (after the open policy is decided), terminal empty prediction, and micro aggregation.

All cases outside explicitly approved corrections must match the January reference. The system-label exclusion and every finalized D-022 partial-credit change require focused before/after fixtures proving their exact scope and contribution values.

### 17.7 PostgreSQL and concurrency tests

- Fresh migration succeeds
- Duplicate campaign/run/task fingerprints are rejected or return existing rows
- Two workers cannot own the same active lease
- Expired lease recovery is safe
- Selected attempt belongs to its task
- Attempt number cannot exceed three
- Terminal evidence cannot be overwritten or cascade-deleted
- Frozen registry rows reject mutation
- Same prediction/scorer fingerprint returns existing evaluation run

### 17.8 Crash matrix

Recovery is tested after interruption:

- Before attempt insert
- After request persistence, before dispatch
- After dispatch, before response storage
- After response storage, before extraction
- After extraction, before validation
- After validation, before prediction commit
- After prediction, before task completion
- During scoring
- During aggregate/export generation

The expected result is no duplicate verified call, no evidence loss, and continuation from the earliest incomplete durable stage.

## 18. Offline dry run

The first end-to-end execution uses PostgreSQL and a deterministic mock provider with a small set of real corpus variables. It performs no OpenRouter or PSNC call.

Required cases are:

1. Direct valid JSON on attempt one.
2. Invalid/fenced output followed by a valid correction.
3. Three invalid outputs followed by terminal empty prediction.
4. Deliberate interruption after a response is stored, then resume without another provider invocation.

Acceptance requires complete lineage from source TTL to report, correct category/subcategory records, exact attempt counts, full prompt/response/error retention, scorer outputs, and idempotent re-execution.

## 19. Live execution gates

Live mode remains blocked until all of the following are non-placeholder and frozen:

- Selected provider set and exact provider-owned model lists
- Exact enabled model list and revisions
- Per-model reasoning capability and exact enabled/disabled request values
- Evaluation-population manifest count and hash
- Ranking-policy version and hash
- Sampling and output-token values
- Provider concurrency/rate limits
- Credentials available through environment variables
- Selected billing evidence, frozen prices, and valid optional provider/global caps (null means no enforced cap)
- Disclosed pre-run cost estimate with token/retry/price assumptions and separate explicit live authorization for that frozen plan
- Provider request contract fixtures
- Corpus/schema/prompt/scorer hashes
- Successful offline dry run and resume test

The planner presents exact planned and maximum call counts. Because every task may use three requests, the conservative scenario includes all three attempts and their growing correction prompts. Expected cost and any conditional upper bound are estimates, not spending caps or a guarantee of actual billing. No live generation is authorized by credentials, implementation approval, or estimate creation alone.

## 20. Documentation-first implementation order

1. Complete and review this specification, decision log, component contracts, and future-work boundaries.
2. Materialize and hash the pinned corpus snapshot and legacy references.
3. Implement and validate boundary schemas and typed configuration.
4. Implement forward PostgreSQL migrations and constraints.
5. Implement deterministic corpus projection and manifests.
6. Implement prompts and ordered demonstrations.
7. Implement one-call provider adapters and fixture tests.
8. Implement extraction, validation, and the three-attempt state machine.
9. Implement the isolated evaluator and regression parity suite.
10. Implement planner, worker leases, concurrency, and resume.
11. Implement reporting and reproducible exports.
12. Pass all unit, integration, concurrency, recovery, and reproducibility tests.
13. Run and verify the offline dry run.
14. Present the pre-run cost estimate and stop for explicit live execution authorization.

## 21. Platform acceptance criteria

Implementation is ready for a live campaign only when:

1. A fresh container/environment can create PostgreSQL through migrations.
2. Corpus v2.0.1 imports as exactly 102 verified records.
3. The five demonstrations are excluded and all other 97 variables appear exactly once in the evaluation population.
4. Prompt schema bytes equal runtime validator schema bytes.
5. Provider adapters make exactly one request per invocation.
6. Each task can make no more than three total provider requests.
7. Every invalid attempt, prompt, response, and error remains queryable.
8. Terminal-invalid variables receive explicit empty predictions and scores.
9. System container labels do not affect either exact or close scores.
10. Every evaluator behavior designated for preservation after D-022 passes January regression fixtures, and every approved correction has explicit before/after fixtures.
11. Same configuration planning is idempotent.
12. Interruption after response storage resumes without another provider call.
13. Category and subcategory reports regenerate from database facts.
14. Configuration/model/reasoning/billing or optional-cap incompatibilities fail preflight; absent mandatory estimate disclosure or separate live authorization prevents dispatch.
15. No secrets are present in tracked files, evidence, logs, or exports.
16. Every complete configuration is ranked from all declared repetitions, and every ranked or unranked result remains queryable.

## 22. Deferred work boundaries

### 22.1 Deterministic ontology-derived baseline

The future baseline uses the same definition input and lexical output schema. The ontology supplies structural semantics, while a separately authored, versioned lexical rule bundle supplies natural-language extraction rules. It must record a decision trace and disclose whether benchmark gold influenced rule authorship. Its development and comparison protocol requires separate approval because this experiment has no hidden subset. No active baseline package is created now.

### 22.2 Entity linking and RDF

Future entity linking consumes a frozen decomposition through a separate experiment stage. If later approved, deterministic RDF conversion and SHACL validation occur after lexical evaluation and cannot change the stored prediction. JSON-LD generation is not part of the planned stage; the supplied JSON-LD context remains reference evidence only. None of these stages is an active dependency now.

## 23. Decisions and campaign values still to freeze

The architecture and scientific decisions are settled: historical conservative prompts (D-021), fractional member scoring (D-022), ranking (D-023), and complete score retention with deferred additional statistics (D-024). The scorer protocol is `january-derived-member-credit-v1`. Source code, tests, schema/migration artifacts, dependencies, and the offline dry run remain implementation deliverables, not unresolved scientific choices.

The following values are deliberately deferred until live campaign preparation and must be recorded in `parameters.yml` and PostgreSQL:

- Availability/revision evidence for the already supplied OpenRouter and PSNC model lists; future list changes require a new frozen campaign
- Exact enabled reasoning effort/value for each capable model
- Provider-supported sampling fields; seed is not a grid factor unless separately approved and supported
- Maximum output tokens
- Concurrency and rate limits
- Price cards, token/retry assumptions, disclosed cost estimate, and separate live execution authorization (spending caps are optional)
- Optional live-canary variables and authorization
