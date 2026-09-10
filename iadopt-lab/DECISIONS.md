# I-ADOPT Lab Decision Record

This file is the concise index of experiment decisions. `TECHNICAL_SPECIFICATION.md` contains their complete technical consequences. A change to a settled scientific decision requires a new decision entry, configuration fingerprint, and—after execution has begun—a new campaign rather than mutation of existing results.

## Settled decisions

### D-001 — Isolated directory

**Status:** Accepted

All new documentation, code, data snapshots, migrations, tests, and generated-output conventions belong under the repository-root directory `iadopt-lab/`. Existing benchmark code remains historical evidence and is not imported at runtime.

### D-002 — New experiment campaign

**Status:** Accepted

The platform will run a new Corpus v2.0.0 experiment. It will not rerun the old corpus as a separate active track. Historical code and results are used only to reconstruct and regression-test intended behavior.

**Superseded in part by D-030:** the release pin is now `v2.0.1`. The rest of this decision — one new campaign, no rerun of the old corpus, historical material used only for regression — stands unchanged.

### D-003 — Authoritative corpus

**Status:** Accepted

Corpus tag `v2.0.0`, commit `8097662ca323771fd977d22cdb8c3e58b7b7d64a`, and tree `bd9cf247d22c5b8345796572e3e55f333460c6ce` are the source snapshot. **Superseded by D-030:** the pin is now tag `v2.0.1`, commit `2598bf91fa927b78a6529bae7864ef0f7d485b73`, tree `df665d32bb2433a60742c80a4a53908dc7ebde0c`. Everything else in this decision stands. The tag contains exactly 102 Turtle files. Release-relative directory names are the authoritative category, subcategory, and full category path.

The supplied five-variable table called the fourth category `Engineering and Technology`; the release path is `Technical Sciences/Material Science`. The exact release value is stored as authoritative provenance. A reporting alias may preserve the alternative wording, but it cannot replace or silently normalize the source category.

### D-004 — Fixed demonstration pool

**Status:** Accepted

The ordered demonstrations are:

1. `Natural Sciences/Atmospheric Science/C2_AirDailyMaximumTemperature.ttl`
2. `Social Sciences/Demography/PersWelfare.ttl`
3. `Life Sciences/Health Science/lactate.ttl`
4. `Technical Sciences/Material Science/CirculationMode-Water.ttl`
5. `Social Sciences/Disaster Risk Science/HeatStress.ttl`

Shot settings use ordered prefixes of this list: zero, first one, first three, or all five. All five are excluded from every independent score, including zero-shot scores.

### D-005 — Lexical JSON only

**Status:** Accepted

The model produces the six scored lexical fields. `label`, `definition`, and `comment` are immutable corpus metadata, not model-generated fields. No RDF, JSON-LD, SHACL, stable RDF identifier, or ontology-version check is part of the active pipeline.

### D-006 — Corrected lexical schema

**Status:** Accepted

A new strict JSON Schema will match the decomposition structure consumed by the evaluator. It will allow either source/target or numerator/denominator asymmetric systems, never require both pairs, require at least two symmetric parts, require explicit empty values, and reject unknown properties.

### D-007 — System identifiers are unscored metadata

**Status:** Accepted; intentional evaluator correction

Both symmetric and asymmetric systems receive deterministic canonical identifiers for stable storage. Symmetric identifiers derive from sorted part labels. Asymmetric identifiers derive from role-preserving source/target or numerator/denominator labels. Container identifiers are excluded from scoring; symmetric parts and asymmetric roles remain scored.

The January evaluator already ignored asymmetric identifiers but included symmetric identifiers. Preserve January behavior outside explicitly approved corrections. Accepted D-022 adds fractional member credit in simple/system and system/system comparisons; its versioned contract excludes dictionary-string comparison and whole-component thresholds that erase partial matches.

### D-008 — Exactly three provider requests per task

**Status:** Accepted

One task is one target variable under one fully resolved provider, model, prompt, shot count, temperature, reasoning mode, and repetition. A task can issue at most three provider requests total: the initial request and at most two subsequent attempts. Every actual provider request counts. Provider adapters have no hidden retry loop.

When a response is invalid, the next prompt contains the immediately previous raw response and its exact extraction/schema/semantic validation errors. Model and sampling parameters remain fixed. After three delivered content-invalid model responses, the task receives an explicit terminal-invalid empty prediction and remains in evaluation. Authentication, infrastructure, budget, and unresolved-delivery failures are recorded as operational failures and are not converted into model-performance errors.

### D-009 — Original single-provider restriction

**Status:** Superseded by D-025

The original design restricted a campaign to one provider and its model list. The owner subsequently requested PSNC-only, OpenRouter-only, or both providers in one execution. D-025 is now authoritative. Each individual task still belongs to exactly one provider and model; the same model family through two providers remains two configurations.

### D-010 — Capability-aware reasoning comparison

**Status:** Accepted

Models that expose a reproducible reasoning/thinking control are evaluated in both normalized modes: `disabled` and `enabled`. Each model maps those normalized modes to exact provider request fields recorded in `parameters.yml`. Models that do not support the control run once with reasoning marked `not_applicable`; the platform never sends an unsupported parameter.

Reasoning mode is a scientific experiment factor, is included in fingerprints, and is reported as a separate dimension. Provider-specific meanings of “enabled” are not assumed to be quantitatively equivalent.

### D-011 — Provider implementations

**Status:** Accepted

OpenRouter and PSNC are separate adapters behind one provider-neutral interface. PSNC follows the audited OpenAI/LiteLLM-compatible `/v1/chat/completions` wire contract from `iadopt-variable-description-service`. The experiment does not invoke that service’s higher-level decomposition endpoint.

### D-012 — PostgreSQL authority

**Status:** Accepted

PostgreSQL 16 is the authoritative metadata and result store. It retains full rendered prompts, sanitized request bodies, untouched raw responses, extracted candidates, validation errors, predictions, scorer inputs and outputs, per-component contributions, aggregate scores, usage, timing, cost, and state transitions.

### D-013 — Resumable and idempotent execution

**Status:** Accepted

Deterministic fingerprints, unique constraints, leases, heartbeats, write-ahead attempt records, and state reconciliation prevent duplicate logical work. Completed tasks are immutable. Resume reclaims only unfinished work and never silently repeats a verified successful request.

### D-014 — Complete-population evaluation and ranking

**Status:** Accepted; exact ordering confirmed under D-023 on 2026-09-06

There is no training, development, test, holdout, or stratified split. After the five demonstrations are removed, all remaining 97 Corpus variables form one immutable evaluation population, from the release pinned by D-030. Every parameter configuration is evaluated against every variable in that population, subject to its declared repetitions.

The system automatically ranks all complete configurations and retains every result. A configuration excludes repetition identity and includes provider, model, reasoning profile, prompt, shot count, temperature, and every other frozen scientific parameter.

The accepted ordering policy calculates micro Close F1 separately for each repetition and ranks by the arithmetic mean of those unrounded repetition-level values, descending. Exact ties receive shared competition ranks with no secondary tie-break. Exact/Close Precision, Recall, F1, and their supporting contributions remain stored at variable, component, and repetition levels. Validity, latency, tokens, cost, and category identities remain available for analysis and never silently change the primary rank. Descriptive statistics over the 97 individual variable scores are deferred under D-024; their arithmetic mean is not the micro F1 used for ranking.

Every attempt, raw response, prediction, item/component contribution, repetition metric, aggregate, configuration rank, and unranked failure state remains stored. A configuration with missing expected results or an unresolved operational failure is visible but not rankable. Three content-invalid responses still produce the explicit empty prediction, so those variables remain scored and do not disappear from the ranking denominator.

### D-015 — Structured-output policy

**Status:** Accepted

Provider-native structured-output enforcement is disabled for the main cross-model experiment unless a later campaign explicitly establishes equivalent support for every selected model. All primary comparisons use the same text-schema, extraction, validation, and retry protocol.

### D-016 — Documentation before code

**Status:** Accepted

The README, technical specification, decisions, configuration contract, database contract, component contracts, test plan, and future-work boundaries are written before executable implementation.

### D-017 — Deferred deterministic baseline

**Status:** Accepted

No baseline code is created now. A future deterministic method will use structural rules grounded in the official I-ADOPT ontology plus an independently authored, versioned lexical rule bundle and produce the same lexical output contract and decision trace. Its development and comparison protocol must be approved separately; this experiment does not reserve a hidden benchmark subset for it.

### D-018 — Deferred entity linking and RDF work

**Status:** Accepted

Entity linking, RDF conversion, and SHACL validation are documentation-only future work in this build. JSON-LD generation is not planned; the supplied JSON-LD context may be preserved only as reference evidence. None of these files or stages is an active runtime input.

### D-019 — Dry-run policy

**Status:** Accepted

The implementation phase first performs a zero-cost offline dry run with a deterministic mock provider and PostgreSQL. A live canary is a separate, explicitly authorized activity with a frozen provider, model list, and disclosed cost estimate. D-027 makes monetary caps optional.

### D-020 — Historical evidence anchor

**Status:** Accepted as an audit boundary

Tag `V1.1-Experiment` at commit `b9683d2242aa5ca5b987440ea4b6f70bc1253c7e` is the immutable pre-submission code anchor. Post-submission files and local ignored workbooks are separately identified by hash and are never silently treated as the tagged experiment.

The audit freezes scorer behavior but does not claim complete manuscript-table reproduction. Historical 96-target grids and the later inconsistent 97–102-target single-configuration workbooks remain distinct evidence sets; four of the latter include demonstration paths in their stored tested population. Resolving old table lineage later would be a separately approved audit, not an active old-corpus experiment track.

## Audit follow-up decisions

The owner has resolved the prompt, ranking, partial-credit, and flexible provider-execution choices. D-021 through D-026 are accepted. These records distinguish the approved protocol from implementation artifacts and live campaign values still to be produced. These updates remain documentation work.

### D-021 — Prompt-to-gold lexical policy

**Status:** Accepted on 2026-09-06; preserve historical conservative prompts

Keep the same historical prompt strategies and no-interpretation/no-inference policy, with only minor changes required by the agreed six-field lexical schema, system representations, prompt names, and validation-feedback workflow. Do not introduce permission for semantic inference or broaden Matrix rules to improve agreement with demonstration gold. Preserve the benchmark answers unchanged.

Record the known limitations, including non-literal gold labels and Matrix instructions that conflict with geographic Matrix examples, without repairing them in this experiment. They are scientific interpretation limitations, not JSON validation failures or reasons for corrective generation. Future templates must retain a reviewable diff against the historical source; this decision settles the policy, not yet-created template bytes/hashes.

### D-022 — Fractional entity/system member scoring

**Status:** Accepted; protocol `january-derived-member-credit-v1`

Partial credit applies in both simple/system directions and to system/system comparisons in Object of Interest, Matrix, and Context Object. Arbitrary symmetric/asymmetric container labels are excluded. A simple entity may earn membership credit against an asymmetric system without claiming that it identified an ordered role. Both explicitly asymmetric systems retain corresponding-role comparisons and January's numerator/source and denominator/target slot compatibility; roles cannot be exchanged.

Let `g` and `p` be gold and predicted member counts and `m` the number of one-to-one qualifying matches. With `U = g + p - m`, the non-empty system branch contributes `TP = m/U`, `FP = (p-m)/U`, `FN = (g-m)/U`, and `TN = 0`. Each component retains total contribution one. Missing and extra members both count; no whole-component threshold discards fractional credit. Gold `water + air` versus `water` gives TP `1/2`, FN `1/2`, component F1 `2/3`; versus `water + soil` gives TP/FP/FN each `1/3`, component F1 `1/2`.

Preserve January's member-identity rules where they already exist: symmetric/symmetric uses literal part equality in both modes, whereas asymmetric and newly supported mixed/simple comparisons use the existing normalized scalar Exact or Close matcher. Unordered comparisons maximize the number of eligible one-to-one matches, then summed similarity, with canonical tie handling; ordered asymmetric comparisons restrict eligibility to corresponding slots. Mixed symmetric/asymmetric comparisons record membership-only evidence rather than inventing missing roles. `docs/scorer-parity.md` specifies the complete dispatch, matching, arithmetic, and evidence contract.

Empty-value behavior, scalar/scalar FP-only wrong predictions, and the historical Constraint branch remain unchanged. Store member counts and exact fraction numerators/denominators so non-terminating fractions such as thirds are reproducible; decimal displays are derivatives. All scientific choices are settled; scorer code, regression fixtures, and artifact hashes remain implementation deliverables.

### D-023 — Exact configuration-ranking formula

**Status:** Accepted on 2026-09-06

Rank by the mean of unrounded repetition-level micro Close F1, descending, with shared competition rank for exact ties and no secondary tie-break. The versioned ordering policy is `mean-repetition-micro-close-f1-v1`. Each repetition first sums scorer contributions across the same 97 variables and six components, then computes its Precision, Recall, and F1. The configuration ranking averages those repetition-level Close F1 values, not the 97 per-variable F1 values. All Exact/Close Precision, Recall, F1, category metadata, attempts, and underlying results remain retained.

### D-024 — Retain scores now; descriptive statistics later

**Status:** Accepted on 2026-09-06

Persist unrounded per-variable, per-component, and per-repetition Exact/Close Precision, Recall, F1 and TP/FP/FN/TN contributions with identities, support, and denominators. Preserve every configuration and repetition so later analysis can use all results without new LLM calls.

After the database is populated, ad hoc analysis may calculate arithmetic mean, median, variance, mode, standard deviation, minimum/maximum and range, and interquartile range (IQR) over the 97 variable-level scores, as well as explicitly selected repetition or category scopes. These descriptive statistics are not required during experiment execution and do not alter the accepted ranking rule. Population versus sample variance, quantile interpolation, mode/tie conventions, and uncertainty methods are decisions for that later analysis; store its query, scope, and formulas when it is performed.

### D-025 — Flexible provider and model selection

**Status:** Accepted; supersedes D-009's campaign restriction

One campaign and one `run`/`resume` invocation can use PSNC only, OpenRouter only, or both. Configuration version `2.0` replaces scalar `campaign.provider` with a non-empty unique `campaign.providers` list. Each selected provider owns its editable `models` list. The owner supplied the initial exact IDs: PSNC `GLM-5.2`, `Qwen3.8-27B`, `DeepSeek-V4-Flash`; OpenRouter `inclusionai/ling-3.0-flash`, `qwen/qwen3-32b`, `qwen/qwen3-8b`. All six are selected in the draft, while live execution remains disabled and deployment/parameter capabilities await verification. Three per provider is not a schema limit. `docs/model-catalog.md` records the source and full ownership/transport table.

Use the shared OpenAI Python library behind separate provider-scoped clients/adapters. PSNC's SDK base is `https://llm.hpc.psnc.pl/v1`; OpenRouter's is `https://openrouter.ai/api/v1`. Both append `/chat/completions` once and use their own named API key. Disable SDK and transport retries; preserve full raw responses. The supplied examples establish selected IDs and intended routing, not experimental prompt content, reasoning-control support, or successful live availability.

Expand and combine the per-provider grids; do not pair an OpenRouter model with PSNC or cross-multiply providers by all catalog models. Each resolved run/task has one provider/model owner. Concrete model IDs must be unique within a provider, not globally across providers. All selected configurations share the 97-variable population and frozen prompt/schema/scorer protocol and can appear in one ranking with the provider dimension retained.

Model IDs, list lengths, and selected providers may be edited before a campaign is frozen. Thereafter `resume` uses its immutable snapshot; changing an active selection creates a new campaign and its own run/task identities, with no implicit reuse of old provider calls. Canonical set ordering excludes incidental provider/catalog display order from scientific identity; original YAML order and bytes remain evidence. An inactive provider's catalog does not require credentials, capabilities, or prices for live readiness of a selected provider.

### D-026 — Complete execution with independent provider limits and billing

**Status:** Accepted; mandatory monetary-cap requirement superseded by D-027

The future `run` and `resume` workflows continue through all selected provider/model tasks, validation, scoring, aggregation, ranking, and required reports without per-model manual handoffs. Completion means every frozen expected variable/configuration/repetition task is scored (including explicit-empty content-invalid predictions) and final ranking/report evidence exists. A blocked provider, missing task, or unresolved delivery remains visible as incomplete; it cannot be silently dropped.

Use fair scheduling and independent provider/model concurrency, rate, cooldown, and optional monetary-cap gates. A provider-specific outage or explicitly configured paid limit pauses its affected work while other eligible selected work continues. Pre-dispatch rate waits do not hold worker slots or consume attempts. Persist recovery state; respect the existing maximum of three actual provider requests per task across restarts. Never substitute another provider/model or create a fourth request to force completion. Shared database/integrity failures or an explicit global stop stop all dispatch; safely retryable cooldowns can resume automatically, while conditions requiring changed credentials or authority return a resumable paused result after eligible work is drained.

The owner reports free PSNC access for this experiment. Record it as a provider-scoped `non_billed` billing basis with explicit zero rates/cost; retain tokens, requests, timing, and quotas. This is not a general claim about all PSNC access. OpenRouter is configured as `metered`, requiring frozen price evidence. Unknown prices are never zero. The original design required positive metered caps and a zero PSNC cap; D-027 replaces that requirement with disclosed pre-run estimates and optional caps. Retain every reservation, adjustment, and actual/estimated cost source even when no ceiling is enforced. If a cap is explicitly set, atomically check it before dispatch; exhausted paid caps do not prohibit documented zero-cost PSNC work.

### D-027 — Disclose costs before live execution; no mandatory spending cap

**Status:** Accepted from the owner's latest instruction; supersedes D-026's mandatory caps

The owner does not want to choose a spending cap, but must be told the estimated cost before experiments run. Parameter schema version `2.1` uses `cost_accounting.require_pre_run_estimate: true` and policy `pre-run-estimate-v1`. Both provider amounts and the campaign amount default to null, meaning no enforced monetary ceiling, not missing approval or zero billing. Non-null optional caps remain supported; zero is an actual zero ceiling. No automatic stop is inferred from the estimate itself.

Before any live canary or scientific execution, present the frozen selection/grid, task and initial/maximum-three request counts, per-model/provider and total estimated cost, input/output/reasoning-token assumptions, correction-prompt growth, price-card and currency-conversion provenance, and scenario range plus conservative upper estimate where defensible. Label unknowns and conditional assumptions; do not invent an expected retry rate, output length, model price, or guaranteed maximum. Paid calibration is not authorized merely to produce this report. Model capabilities, output limits, and pricing must be resolved to produce a useful estimate.

Store the estimate receipt and hash, its configuration/price evidence and calculation version, presentation timestamp, and the separately required live-authorization record. Changing the grid, prices, or estimate assumptions invalidates that disclosure for a new live plan. Existing safe resume reuses its frozen evidence and does not ask for per-variable approval. Actual billing may differ from the estimate and remains fully recorded. Monetary caps, if deliberately configured in a future campaign, are checked atomically; no cap weakens the three-request maximum, rate limits, immutable campaign, or operational-failure rules.

### D-028 — Existing credential source and local database preparation

**Status:** Credential source accepted; local PostgreSQL setup documented as a proposal, not provisioned

The owner supplied the repository-root `.env` as the existing source of `OPENROUTER_API_KEY` and `PSNC_API_KEY`. The planned CLI may use an explicitly supplied `--env-file` path to load allowlisted runtime settings, with process-environment values taking precedence. Never execute the file as shell code, interpolate commands/variables, copy it, hash its contents into evidence, print secret values, or persist secret-bearing DSNs. An unrelated service's additional entries are ignored. Only selected-provider settings and the required database connection are used.

PostgreSQL access means a reachable database server and authorized connection details, not an AWS account. A local PostgreSQL 16 service with persistent storage and DBeaver access is the recommended implementation target, documented in `docs/local-database.md`. The latest request authorizes documentation cleanup only; it does not start Docker, provision a database, modify `.env`, or begin script implementation.

### D-029 — One repetition at every temperature

**Status:** Accepted from the owner's explicit instruction; supersedes the initial five-repetition nonzero-temperature plan

Use one repetition at each of `0.0`, `0.5`, `1.0`, and `2.0` for every selected provider, model, reasoning profile, prompt, and shot count. Set both `parameter_grid.repetitions.temperature_zero` and `nonzero_temperature` to `1`. The owner chose this reduction to limit total request volume. It applies to PSNC as well as OpenRouter; the preceding OpenRouter-only cost calculation was not a provider-specific repetition policy.

The set of scientific parameter configurations is unchanged, but planned runs/tasks and maximum request counts fall by 75% compared with the former `1 + 5 + 5 + 5` temperature schedule. Per model and applicable reasoning profile: 48 configurations, 48 resolved runs, 4,656 variable tasks/initial requests, and at most 13,968 provider requests. Three OpenRouter models therefore require 13,968 initial requests with one profile each, or 27,936 if all three have verified reasoning-off/on profiles; the respective three-attempt maxima are 41,904 and 83,808. Counts remain conditional on verified reasoning support.

Retain the repetition dimension in configuration, identifiers, database records, exports, and generic aggregation code so future campaigns can explicitly choose more repetitions. D-023's ranking formula and policy version do not change: the mean of one repetition's micro Close F1 is that same value. All 97 variables, Exact/Close Precision/Recall/F1, partial-credit rules, and full evidence retention remain unchanged. Retry attempts are not repetitions; each variable/configuration task still allows at most three total provider requests.

This campaign cannot estimate within-configuration run-to-run variability from independent repetitions. Do not report an observed zero stochastic variance or a repetition-based confidence interval from a singleton. Later descriptive statistics across the 97 variable scores remain possible but measure a different distribution. More independent repetitions would require newly authorized generation; they cannot be reconstructed from retries or different configurations.

No live campaign exists yet. The new repetition values change the resolved plan/configuration identity and require fresh counts and a fresh disclosed cost estimate before live execution; never mutate a frozen older campaign or treat its five-repetition estimate as current. The parameter schema shape stays at `2.1` because only values changed.

### D-030 — Corpus release v2.0.1 replaces v2.0.0

**Status:** Accepted from the owner's explicit instruction; supersedes D-002's and D-003's release pin

The authoritative dataset becomes I-ADOPT Corpus release `v2.0.1`, commit `2598bf91fa927b78a6529bae7864ef0f7d485b73`, tree `df665d32bb2433a60742c80a4a53908dc7ebde0c`. The release still contains exactly 102 Turtle files, and no file is added, removed, or renamed, so the category/subcategory hierarchy, the five-item demonstration pool, and the 97-variable evaluation population keep their exact membership and order. `expected_turtle_files` stays 102 and `102 - 5 = 97` is unchanged.

Four files differ from v2.0.0 and every one of them changes gold content:

| Release-relative path | Change | Role |
|---|---|---|
| `Life Sciences/Ecology/C12_HabitatProbability.ttl` | `hasMatrix` becomes `hasContextObject`; constrained entity `Q11081619` becomes `Q3622002` | scored |
| `Natural Sciences/Hydrology/SurfRunoff.ttl` | two Constraint targets exchanged | scored |
| `Social Sciences/Demography/NumChild.ttl` | constraint label `registered as residents` becomes `registered as resident` | scored |
| `Social Sciences/Demography/PersWelfare.ttl` | constraint label shortened to `condition: registered as resident` | demonstration, position 2 |

The demonstration change is the consequential one. `PersWelfare` occupies position 2 of the ordered pool, so it appears in every 3-shot and 5-shot prompt. Its gold text is inserted verbatim into those prompts, so all 3-shot and 5-shot rendered-prompt bytes and hashes change. Prompt *template* bytes and their registry hashes do not change, because demonstrations are inserted at render time and are not part of a template artifact.

This is a new corpus identity and therefore a new campaign. Reimporting produces new source, canonical, manifest, population, and demonstration hashes; a resolved plan built on v2.0.0 evidence must not be resumed against v2.0.1 evidence. No live campaign exists, so nothing is invalidated in flight.

Variable identifiers stay commit-derived. `variable_id` is `sha256(commit + "\n" + release-relative path)`, so this upgrade renumbers all 102 variables, not only the four whose content changed. That is the accepted behavior: a variable's scientific identity includes the exact source snapshot it was projected from, and one identifier must never denote two different gold answers. The cost is that results cannot be joined across corpus releases by identifier; a before/after comparison of the same upstream variable must be made through its release-relative path, which is stable and retained in every record and manifest. This was confirmed explicitly rather than inherited by accident.

The upgrade is a coordinated change, not a configuration edit. `parameters.yml` cannot move to v2.0.1 on its own, because `schemas/parameters.schema.json` pins release, commit, and tree as JSON Schema `const` values; changing only the YAML makes the file fail its own validation. `docs/migration-v2.0.1.md` lists every artifact that must change together and is the authorization checklist for that step.

### D-031 — Derived per-variable files and readable predictions

**Status:** Accepted from the owner's explicit instruction

The single canonical record per variable remains the authoritative scientific artifact. Its `record_sha256`, the corpus manifest, ingestion verification, and all scoring continue to read it unchanged. Ingestion additionally writes two derived views of that same record, so a human can read a variable without parsing provenance:

- A **metadata file** carrying provenance only: repository, tag, commit, tree, source path, Git blob id, byte length, source and gold hashes, importer and schema identities, category path, variable id, and demonstration position.
- A **readable variable file** in the shape the earlier TTL-to-JSON conversion produced: `label`, `definition`, the six lexical fields, and the Wikidata IRIs beside the fields that carry them (`hasPropertyURI`, `hasObjectOfInterestURI`, `hasMatrixURI`, `hasContextObjectURI`, `hasStatisticalModifierURI`). The IRIs are already present in the source Turtle; retaining them costs nothing now and is the input a future entity-linking stage would need.

Both files are deterministic projections of the canonical record. They are regenerated, never hand-edited, and a mismatch between a derived file and its canonical parent is an integrity failure. Neither file is a scoring input, and neither replaces the canonical record in any hash chain.

Stored predictions gain the same readability. The model continues to return **exactly the six evaluated fields**; the lexical schema keeps `additionalProperties: false` and its six required keys, and `iadopt_eval` continues to reject any decomposition that is not exactly those six. When a prediction is persisted, the pipeline attaches the target's real `label` and `definition`, copied from the canonical corpus record, to the readable prediction view. They are provenance for the reader, never model output and never scored.

The model is deliberately not asked to produce them. It never sees the target label, so a generated label would be invented rather than recalled, and re-emitting the definition on every one of the planned tasks reintroduces the truncation and paraphrase failures that D-021's adaptation removed, at a real output-token cost, for two fields that carry no score. Copying the corpus values yields a strictly more accurate artifact for free. D-021's no-interpretation policy and the approved historical prompt diffs stand unchanged.

### D-032 — Revised OpenRouter model list

**Status:** Accepted from the owner's explicit instruction; revises D-025's initial selection

The OpenRouter selection becomes `qwen/qwen3-8b`, `qwen/qwen3-32b`, and `openai/gpt-4o-mini`. `inclusionai/ling-3.0-flash` is withdrawn; the two Qwen entries are unchanged. The PSNC selection is untouched: `GLM-5.2`, `Qwen3.8-27B`, `DeepSeek-V4-Flash`.

The campaign therefore still comprises six selected models across two providers, so D-029's grid arithmetic is unchanged: 48 configurations, 48 resolved runs, 4,656 tasks, and at most 13,968 requests per model per applicable reasoning profile.

`openai/gpt-4o-mini` differs in kind from the other five, which are open-weight deployments. It is a hosted closed-weight model reached through OpenRouter's metered billing, so complete price-card evidence and the disclosed pre-run estimate become load-bearing for this campaign rather than a formality; an unknown price is still never treated as zero. Selection asserts nothing about availability, context window, reasoning-control support, or a pinnable revision. All three OpenRouter entries keep `capabilities: null` and empty `reasoning_profiles`, and preflight continues to refuse a frozen plan while any capability is unverified.

### D-033 — Capability declares ability; reasoning profiles declare scope

**Status:** Accepted from the owner's explicit instruction; refines D-025

`capabilities.reasoning_control` states what a deployment *can* do. `reasoning_profiles` states which of those modes a campaign *tests*. Previously the two were conflated: a reasoning-capable model was required to declare exactly `{disabled, enabled}`, which made a deliberate single-mode campaign impossible to express on a capable model. That is a legitimate scientific scope, not a contradiction.

A declared profile list must still be non-empty, free of duplicates, and a subset of what the capability supports. A profile for a mode the model cannot control remains an error, so the change cannot be used to assert reasoning support that does not exist. Grid arithmetic is unchanged: the number of configurations follows the declared profiles, so a one-profile campaign is 48 per model and a two-profile campaign is 96.

### D-034 — Reasoning cannot be disabled on the OpenRouter qwen3-32b route

**Status:** Accepted from measurement during the first live run

The first live attempt declared `reasoning_control: true` with a `disabled` profile sending `reasoning: {enabled: false}`, and `max_output_tokens: 1000`. It failed: 41 of 55 responses still contained reasoning text, and 15 returned a completely empty answer after spending the whole 1000-token budget on thinking. Those 15 were the run's `no_valid_json_object` failures.

Four request shapes were then probed directly against `qwen/qwen3-32b`, one call each:

| Request field | Reasoning still emitted |
|---|---:|
| none (baseline) | 1327 characters |
| `reasoning: {enabled: false}` | 1112 characters |
| `chat_template_kwargs: {enable_thinking: false}` | 687 characters |
| `reasoning: {max_tokens: 0}` | 565 characters |

None suppresses generation. The control exists in OpenRouter's API surface and is accepted without error, but the upstream routes serving this model (DeepInfra, SiliconFlow) always produce thinking tokens, which are billed as output.

This campaign therefore declares only an `enabled` profile under D-033, and raises `max_output_tokens` to 5000 so the ceiling covers thinking plus the answer. Measured completions then averaged 1005 tokens with a maximum of 2450, every response finished with `stop`, and truncation disappeared entirely.

The wider consequence is for D-025, which treats reasoning on/off as a controllable experimental factor across the grid. On this provider and model family it is not. Before the full campaign is frozen, each selected model needs the same empirical probe rather than a declaration inferred from its documentation; a model whose reasoning cannot be disabled can only contribute the reasoning-enabled half of that comparison. Record the probe result as capability evidence.

### D-035 — Streaming worker pool replaces batch-gather scheduling

**Status:** Accepted from the owner's explicit instruction

`run_campaign` claimed up to `worker_count` leases, awaited all of them with `asyncio.gather`, and only then claimed more. Every batch therefore ran at the speed of its slowest member. With reasoning enabled, per-response latency ranged from about 10 to 92 seconds, so slots sat idle for most of each batch and measured throughput fell to roughly 1.4 tasks a minute.

The scheduler now keeps the pool saturated: it claims while free slots exist, waits for the first task to finish, releases that one slot, and immediately claims again. Measured throughput rose to about 20 tasks a minute on the same configuration, with slot occupancy pinned at the configured concurrency.

Nothing scientific changes. Provider-fair rotation, per-provider concurrency, the rate gates, the three-request budget, lease and heartbeat handling, evidence, scoring and resume are all per task and untouched. Two safety properties are explicit in the implementation: every finished task's result is retrieved so a failure is re-raised rather than silently discarded, which `asyncio.wait` would otherwise do; and a `finally` block cancels and drains any in-flight task so a failure cannot leave leases held by orphaned coroutines.

Because `workflow.py` is part of the implementation index, this change alters the plan fingerprint and a campaign planned under the previous scheduler cannot be resumed against it. That is the drift guard behaving correctly; the in-flight campaign was discarded and replanned.

### D-036 — GLM-5.2 on PSNC: reasoning is always on and not controllable

**Status:** SUPERSEDED by D-041. The conclusion below is wrong: the probe behind it tested
only `enable_thinking: true` and never the false case. Retained unedited as the record of
what was believed before the second live run, and of how the error was made.

PCSS publishes no per-model capability table. Its documentation describes a LiteLLM-compatible OpenAI-shaped API at `https://llm.hpc.psnc.pl`, lists models through `/v1/models`, and states token quotas by user type, but gives no context window, rate limit, or parameter support matrix. Capabilities were therefore established by direct probe rather than declaration.

| Probe | Result |
|---|---|
| `temperature`, `top_p` | Accepted; normal completion |
| `seed` | Accepted; normal completion |
| `response_format: {type: json_object}` | HTTP 200 with a **null** content field; native structured output does not work |
| `reasoning: {enabled: true}` | HTTP 200 with an **empty** response; sending this field breaks generation |
| `chat_template_kwargs: {enable_thinking: true}` | Accepted; changes prose style only |
| Baseline, trivial prompt | 417 characters of reasoning returned in `reasoning_content` |
| 103,840-token prompt | Accepted and reported by the server without error |

GLM-5.2 therefore always reasons and offers no way to stop it, which mirrors D-034's finding for `qwen/qwen3-32b` on OpenRouter from the opposite direction: there the reasoning-disable field was accepted and ignored, here it is accepted and destructive. `reasoning_control` is declared `false` with a single `not_applicable` profile, `structured_output` `false`, and `context_window_tokens` a conservative 100,000 recorded explicitly as a verified lower bound rather than a published maximum.

Two models on two providers have now been probed and neither can produce a reasoning-disabled condition. D-025 treats reasoning on/off as a controllable grid factor; on this evidence that factor may not be available at all for the current selection. Probe each remaining model the same way before freezing the full campaign, and if none supports disabling, drop reasoning from the grid rather than carrying a dimension that cannot vary.

### D-037 — Provider reasoning channels have different names

**Status:** Accepted from a defect found during D-036 probing

The shared adapter read only `choices[].message.reasoning`. OpenRouter populates that field, but vLLM deployments such as PSNC return `reasoning_content`. GLM-5.2 responses were therefore arriving with their reasoning silently discarded and stored as null, while `usage.completion_tokens` still counted those tokens, so the evidence would have understated what the model produced and what was billed.

The adapter now reads `reasoning`, then `reasoning_content`, in that fixed order so the selection stays deterministic if a provider ever returns both. This is an evidence-completeness fix required by the `complete-v1` policy, not a scientific change: no prompt, parameter, prediction or score is affected. It was caught only because the raw response body was inspected directly; the parsed field alone looked plausibly empty.

### D-038 — Reasoning is uncapped; the output ceiling is set high enough that truncation cannot recur

**Status:** SUPERSEDED by D-040 for the ceiling value, and by D-041 for its premise that
neither model can stop reasoning. The reasoning below was sound given D-036; the premise
was not. Retained unedited as dated history.

`max_output_tokens` is a single budget shared by a model's private reasoning and its visible answer, not a limit on the answer. Neither selected model can stop reasoning (D-034, D-036), so whenever reasoning exhausted the budget, generation halted before the answer began. The result was an empty `content` field, indistinguishable in the evidence from a model that produced nothing useful. Four GLM-5.2 responses show the pattern exactly: each stopped at the 5,000 ceiling having emitted 17,635-20,510 characters of reasoning and **zero characters of answer**.

Reasoning length varies far too widely to size a tight ceiling against. Across successful responses, GLM-5.2 averaged 7,545 characters of reasoning with a maximum of 24,114, and qwen3-32b averaged 4,832 with a maximum of 21,844 — a three- to fivefold spread, uncorrelated with any visible property of the target.

The owner has chosen not to cap reasoning. `max_output_tokens` is therefore set to `100,000`, roughly eighteen times the largest completion observed to date (5,613 tokens), so the ceiling cannot realistically be reached. This is a ceiling, not a reservation: billing and latency follow tokens actually generated, so ordinary responses are unaffected.

Three consequences are accepted deliberately rather than overlooked:

1. **It is no longer a runaway guard.** The context window becomes the only bound on a pathological generation. That is the intended trade: a guaranteed-complete answer is worth more than protection against a hypothetical runaway.
2. **It is grid-wide, so it reaches metered providers too.** On OpenRouter a worst-case runaway would bill up to 100,000 output tokens per call, moving that campaign's conditional maximum from about $1 to roughly $17. Expected cost is unchanged, because expectation follows real usage.
3. **Operational limits must scale with it.** The conservative admission bound now includes the full ceiling, about 103,000 tokens per request. Both providers' `tokens_per_minute` were raised to 3,000,000, without which the rate gate would have throttled dispatch to five requests a minute regardless of concurrency, and both timeouts raised to 900 seconds, without which long generations reproduce the ambiguous deliveries seen on PSNC. Any future change to the ceiling must revisit these three values together.

Supporting evidence for the ceiling fitting: LiteLLM's `/model/info` reports a context window of 1,048,576 for GLM-5.2 on PSNC, and OpenRouter documents 131,072 for `qwen/qwen3-32b`. Both leave ample room for a 100,000-token completion alongside a prompt of roughly 1,500 tokens.

The classification defect this created was addressed separately in D-039: `finish_reason: "length"` is now an operational `output_truncated` result whether the answer is empty or partial, so a capacity failure no longer consumes content retries or scores as a model error. Note also that no finite ceiling can *guarantee* a complete answer; a sufficiently long generation can still reach any limit, and a completed generation can still be invalid JSON. The ceiling makes truncation improbable, not impossible.

### D-039 — Evidence preservation and classification hardening after the second review

**Status:** Accepted; implements the resolutions to `docs/read-only-review-2026-09-09.md`

A second read-only review found that the previous round of fixes had the right shape but
incomplete substance. This decision records what changed and the principles behind it.

**Raw evidence outranks its interpretation.** A response that has reached the process may
already have been paid for and cannot be reproduced within a task's fixed three-request
budget. Two defects broke that ordering. Accounting ran before the raw commit, so an
unfamiliar billing field could raise and destroy the very evidence needed to diagnose it;
settlement failures are now caught and recorded as unresolved accounting instead. And the
commit retry caught only `PersistenceError` while the repository lets `psycopg` errors
propagate unwrapped, so exactly the transient outage it existed for bypassed it; it now
also catches `psycopg.Error`, explicitly re-raises `EvidenceConflict` because a permanent
identity conflict cannot be resolved by waiting, and backs off over roughly two minutes.

**Only a well-formed completion may be scored.** Anything else a provider can return at
HTTP 200 is an operational failure, or three of them exhaust a task and are recorded as an
empty prediction, lowering a model's measured score on infrastructure noise. The adapter
now rejects a non-object body, a provider error object, a missing or malformed choice
message, and any `finish_reason: "length"` — whether the answer was empty or partial,
because an incomplete answer reflects the configured ceiling rather than the model's
ability. A well-formed envelope whose completion is genuinely empty remains a model
outcome and is still scored. Classification is separate from permission to retry; the
three-request rule is unchanged.

**Scientific checks belong to the boundary, not the caller.** `build_configuration_ranking`
now requires a supplied scorer identity to name the plan it belongs to, and `iadopt-lab
report` supplies both scorer identity and frozen category membership, so a report's
integrity no longer depends on which entry point produced it. Finalization takes the
scorer version and threshold from the frozen evaluator constants rather than reading them
out of the evidence under examination, which would have made the check agree with whatever
produced that evidence.

**Smaller corrections.** The rate-limit handler reads `retry_after_seconds` instead of a
message string, so an advised delay is honoured. Reservations and provider-reported costs
apply the same `fx_to_reporting` as usage-based settlement, and all three record their
currency. Artifact identities are collected for the population actually planned, so a
synthetic plan no longer hashes 97 identifiers while planning three. Resume verifies the
requested campaign before registering anything, and treats an already-complete campaign as
success while still finalizing, so an interruption between the last task and the ranking
write is recovered by running resume again.

**Token-bound conditions are now stated.** One token per UTF-8 byte is an upper bound only
for byte-level BPE tokenizers, with bounded chat-template overhead and no server-side
expansion. Those three conditions are documented at the call site, and a deployment that
violates them requires the bound to be re-established.

## Values to freeze before live execution

### D-040 — The output ceiling is 16,000 tokens, not 100,000

**Status:** Accepted from measurement, superseding D-038's ceiling

D-038 raised `max_output_tokens` to 100,000 on the premise that reasoning could not be
stopped, so the only defence against truncation was headroom. D-041 removes that premise.
With reasoning disabled the ceiling stops being a reasoning budget and becomes what it was
always meant to be: a bound on a runaway.

A ceiling is free in billing terms but not in time. At 100,000 the runner had no way to
stop a request that would never finish: 18 of 31 GLM-5.2 requests were killed by the 900s
timeout, each holding a worker slot for fifteen minutes and producing nothing.

16,000 is sized from evidence rather than chosen for comfort. Every successful GLM-5.2
response recorded fits well inside it — the largest is 8,605 completion tokens, and the
campaign that followed peaked at 197 with reasoning off. At the observed 20-28 tokens per
second, 16,000 caps a single request near 800 seconds, inside the timeout, so a runaway is
truncated and classified rather than left to hang.

The ceiling is now taken from the plan rather than from current parameters wherever it is
used as evidence, so a stale plan cannot borrow a number it was never expanded with.

### D-041 — GLM-5.2 and Qwen3.8-27B reasoning IS controllable on PSNC

**Status:** Accepted from measurement, superseding D-036

D-036 concluded that GLM-5.2's reasoning could not be disabled. That conclusion came from
probing `chat_template_kwargs.enable_thinking: true` and inferring the false case, which
was never sent. Measured directly on one real 5-shot prompt:

| Request | Latency | Completion tokens | Reasoning | Answer |
|---|---|---|---|---|
| No reasoning control | 367.8s | 16,000 (truncated) | 66,697 chars | **0 chars** |
| `enable_thinking: false` | 2.7s | 81 | 0 chars | 356 chars |

136 times faster, and the difference between no answer and a complete one. The full
97-variable campaign that followed scored **Close F1 0.387** with reasoning off, ahead of
qwen3-32b's 0.316 with reasoning fully on, so this is not a quality trade.

`enable_thinking` is the only control this deployment honours. `reasoning.enabled: false`
returns an empty response; `reasoning.effort` at `minimal` and `low`, the flat
`reasoning_effort` field, and `reasoning.max_tokens` were each measured and none holds a
call under 30 seconds. The effort scale runs backwards — `minimal` produced 5,798
characters of thinking against `low`'s 1,773 — and a 512-token budget overshot 24-fold at
12,428 characters, which is how we know the `reasoning` object is not interpreted here.

Qwen3.8-27B honours the same switch (111 characters of thinking uncontrolled, 0 with it).
DeepSeek-V4-Flash emits no reasoning in either probe, so it declares `not_applicable`: that
records the absence of a control, **not** a measurement that the model never reasons.

D-034's matching claim for qwen3-32b on OpenRouter rests on the same flawed method and has
not been re-tested. It should not be relied on until it is.

### D-042 — Remediation of the 2026-09-09 read-only audit

**Status:** Accepted; implemented in this pass

Thirteen findings were independently verified against source before any change. Twelve were
real. The load-bearing ones and what changed:

- **Price-card coverage** blocked nothing and would have crashed the three-model campaign
  on its first non-GLM task, because `cost_policy` raises for an uncovered model *before*
  reading the billing mode, from inside the worker pool where the failure cancels unrelated
  in-flight work. The card now covers every enabled model, coverage is checked once against
  the whole plan before dispatch, and the estimate reads that same card instead of deriving
  its own — which is why a "ready" $0 estimate could previously precede that crash.
- **Probe verdicts** treated a *failed* baseline as evidence a model never reasons, and
  would have written `not_applicable` for a model that reasons — sending no switch at all.
  Reasoning observed under the switch is now decisive first; an unusable or truncated
  baseline is `inconclusive` rather than favourable.
- **Received evidence** could be discarded by the outage that caused the failure: a failing
  heartbeat cancelled the commit-retry loop mid-backoff. Commits now run as shielded tasks
  the campaign awaits, so cancelling a worker stops dispatch without stopping preservation.
- **Envelope classification** accepted any message dictionary as a completion, so
  `{"message": {}}` was scored as the model answering nothing. A completion now requires a
  `content` field of string or null type; reasoning is retained even from rejected
  envelopes.
- **Report binding** compared a plan's scorer hash to itself, which holds for any checkout.
  It is now recomputed from evaluator source plus the loaded backend. Category denominators
  came from all 102 records, marking demonstration categories incomplete in a fully scored
  97-target report; they now come from the planned population, and wholly absent categories
  are represented rather than omitted.
- **The freeze bundle** was never persisted despite the documented contract; one
  `experiment-bundle` artifact, including the original `parameters.yml` bytes, is now
  registered and linked before work becomes dispatchable.
- **Authorization** asserted an approval nothing expressed and named a `--authorize` flag
  that did not exist. That flag now exists and is required for live dispatch, `--actor` is
  required, and the estimate's own hash and model coverage are verified before the receipt
  is written.
- **Resume** could not recover a durably stored transient rejection; `reconcile` now
  re-derives the runner's retry decision on exactly its condition.

Two findings were judged overstated and are recorded rather than acted on as defects:
treating `probe-models` as an unauthorized live entry point (three short calls, no scored
generation), and the criticism of `not_applicable` accounting, which was nonetheless
reworded to stop implying an unmeasured absence of reasoning.

### D-043 — Temperature 2.0 is dropped from the PSNC full-grid campaign

**Status:** Accepted from measurement; narrows D-029's temperature set for this campaign

D-029 sets one repetition at every temperature in `[0, 0.5, 1, 2]`. This campaign runs
`[0, 0.5, 1]`. The reason is measured, not preference.

With reasoning disabled, temperature 2.0 flattens the sampling distribution far enough that
the stop token rarely wins, so generation runs toward the output ceiling instead of
terminating. Measured on 24 real 5-shot prompts per model against the 16,000-token ceiling:

| Temperature | Latency per call | Completion tokens | Truncated |
|---|---|---|---|
| 0.0-1.0, all three models | 0.7-6.3s | 55-133 | 0 of 32 |
| 2.0, DeepSeek-V4-Flash | up to 172s | avg 6,997, max 16,000 | ~40% |
| 2.0, GLM-5.2 | up to 134s | avg 3,801 | observed |
| 2.0, Qwen3.8-27B | up to 31s | avg 284 | not observed |

The consequence is not merely slowness. A response truncated at the ceiling is a
non-retryable operational failure by design: resending the identical request would truncate
again, and scoring it as a model-quality zero would blame the model for a capacity stop. So
its task never reaches `complete`, and under `ranking.require_complete_population` a single
such task makes its whole configuration unrankable. At a ~40% per-call truncation rate over
97 variables, effectively every temperature-2.0 configuration is unrankable. That quarter of
the grid would have consumed the large majority of the runtime to produce almost no rankable
result.

This is a scope decision, not a finding that temperature 2.0 is uninteresting. That the
models fail to terminate at 2.0 is itself a result worth reporting. Restoring 2.0 requires
first settling whether a runaway generation with reasoning disabled is an operational
failure or a model-quality outcome — with reasoning off it is arguably the latter, since
nothing but the model's own distribution is producing those tokens. That question is
deliberately left open rather than resolved by whichever classification is convenient.

Related: D-040 fixed the ceiling at 16,000; D-042 made a campaign able to finalize with
terminal operational failures recorded, so a truncation no longer denies results for every
other configuration.

### D-044 — PSNC reasoning-enabled is excluded; the 120s timeout is the binding constraint

**Status:** Accepted from measurement; narrows the reasoning dimension added in this pass

The reasoning dimension was to cover four models: `GLM-5.2` and `Qwen3.8-27B` on PSNC,
`qwen/qwen3-8b` and `qwen/qwen3-32b` on OpenRouter. The two PSNC models are excluded.

The owner set a 120-second request timeout for the unattended run. PSNC reasoning-enabled
generation does not fit inside it. Measured with reasoning on, 16 real 5-shot prompts per
level, at exactly that timeout:

| Model | Concurrency | Succeeded | Timed out |
|---|---:|---:|---:|
| GLM-5.2 | 2 | 4 of 16 | **12** |
| Qwen3.8-27B | 4 | 7 of 16 | **9** |
| Qwen3.8-27B | 8 | 8 of 16 | **8** |

Lowering concurrency does not help, and that is the important part. Going from 4 to 8 on
Qwen3.8-27B *improved* the result slightly (7 to 8 successes), which shows the failures are
not queuing behind a saturated server: they are single generations that need longer than
120 seconds to emit ~3,100-3,500 reasoning tokens. An earlier sequential measurement put
GLM-5.2 at 233s per call with nothing competing. No concurrency setting makes a 233-second
generation finish in 120.

The consequence of running them anyway would not be slowness but emptiness. A timeout is a
transient failure consuming one of a task's three attempts, so a ~50-75% per-attempt
failure rate leaves most tasks operationally failed, and under
`ranking.require_complete_population` a configuration with one failed task is unrankable.
Essentially every PSNC reasoning-enabled configuration would produce no rankable result
while consuming days of wall time. This is the same trap as temperature 2.0 in D-043.

OpenRouter reasoning-enabled is unaffected and stays in scope. At the same 120s timeout it
completed 16 of 16 at both concurrency 4 and 8, with p50 26-41s and maximum 94-117s.
Maximum latency *fell* moving from 4 to 8 on both models, so concurrency 8 is used.

This narrows the experiment rather than settling a question. Whether reasoning helps these
two PSNC models is still open; answering it needs a timeout near 600-900s, which the owner
has excluded for an unattended run. The measurements above are recorded so the decision can
be revisited without repeating them.

Supersedes the reasoning-enabled scope sketched for campaign B1; see docs/campaign-log.md.

### D-045 — PSNC reasoning latency is generation length, not throttling; GLM-5.2 moves to OpenRouter

**Status:** Accepted from measurement; completes D-044

D-044 excluded PSNC reasoning-enabled work because it timed out at 120s. The open question
was *why*: if PSNC were throttling traffic that looks like a burst, spacing calls out would
fix it and the exclusion would be wrong. It was tested directly and it is not throttling.

`Qwen3.8-27B`, reasoning on, **concurrency 1 with a 20-second cooldown between calls** —
maximally spaced, nothing competing, 900s timeout:

| Call | Latency | Output tokens |
|---|---:|---:|
| 1 | 33.3s | 2,026 |
| 2 | 10.9s | 660 |
| 3 | 243.0s | 14,751 |
| 4 | 97.2s | 5,943 |
| 5 | 274.7s | 16,000 (truncated at the ceiling) |

Latency is a straight function of output length at roughly 60 tokens per second, and
nothing else. Two of five calls exceeded 120s with zero contention, and one hit the 16,000
ceiling. Cooldown cannot help because there is no queue to drain: the model simply decides
how much to reason, and sometimes that is 16,000 tokens. This also explains the earlier
confusion where concurrency 1 looked fast (10-33s) while concurrency 8 showed a 157s
median — that was sampling variance in output length, not load.

The same reading resolves the concurrency question. Median latency was flat from
concurrency 8 (157.4s) to 16 (146.6s), which is what a server that is *not* saturating
looks like. Concurrency was never the problem and lowering it was never the fix.

**GLM-5.2 moves to OpenRouter as `z-ai/glm-5.2`.** The same model reasons far more briefly
there: 772 output tokens in 14.0s, against 3,111 tokens and 233s on PSNC, with 3/3 valid
JSON. That fits the 120s timeout comfortably and costs $3.81 for its 3,492 tasks.

`qwen/qwen3.8-27b` on OpenRouter was evaluated as the equivalent move for the other PSNC
model and **rejected on cost**: measured 3,391 output tokens per call at $3.00 per million
output tokens gives $37.86 for 3,492 tasks, against a $5 per-model budget. It is not in the
plan. The reasoning arm for that model therefore has no home in this experiment, and that
gap is deliberate rather than overlooked.

One consequence worth recording: the reasoning-enabled arm now uses `z-ai/glm-5.2` while
the reasoning-disabled arm used PSNC `GLM-5.2`. These are the same model family on
different deployments, so the two arms are NOT a controlled comparison of reasoning alone.
Any reading of that pair has to treat deployment as a confound.

## Still to freeze

These operational or campaign-specific values must still be frozen for a live campaign:

- Availability and exposed deployment revision/backend evidence for the six selected model IDs under D-032; record limitations where a provider cannot pin revisions
- Completed v2.0.1 migration under D-030: coordinated schema/ingestion constants, re-import, and regenerated corpus, demonstration, and population manifests
- Exact provider request mapping for each model’s enabled and disabled reasoning modes
- Exact 97-variable evaluation-population manifest and hash
- Maximum input/output tokens and other sampling parameters; unsupported seed controls are recorded as unavailable and are not silently added as a grid factor
- Provider concurrency and rate limits
- Price-card versions, token/FX assumptions, and a disclosed pre-run cost estimate; monetary caps are optional
- Live-canary scope and authorization

The planner must refuse a live run while any required value remains a placeholder.
