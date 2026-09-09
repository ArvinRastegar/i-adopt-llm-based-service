# Database and Persistence Design

## 1. Purpose and authority

This document defines how I-ADOPT Lab will persist corpus provenance, campaign configuration, provider evidence, retry state, predictions, evaluator evidence, usage, cost, and reports. It refines section 12 of `TECHNICAL_SPECIFICATION.md`; that specification remains authoritative if the two documents ever conflict.

PostgreSQL 16 is the authoritative store for experiment relationships, mutable workflow state, immutable evidence, and evaluation results. Files under `data/` are reproducible corpus/manifests, and files under `outputs/` are derived exports. Neither is a substitute for a PostgreSQL backup.

AWS is not required. The proposed local deployment and DBeaver connection are described in [Local database setup](local-database.md): a dedicated PostgreSQL 16 instance, loopback-only host access, persistent storage, separate roles, and verified backups. No database is provisioned during this documentation step.

The database design has four non-negotiable goals:

1. Reconstruct exactly what was sent to and returned by a provider.
2. Resume after interruption without silently repeating completed work.
3. Regenerate every reported metric from item-level evidence.
4. Prevent a new configuration, corpus, prompt, schema, or scorer from being mistaken for an existing campaign.

This document is the design contract. The migrations that implement it are `migrations/0001_initial.sql` and `migrations/0002_coordination_integrity.sql`, applied with `iadopt-lab database migrate`; this file contains no SQL itself and the migrations remain authoritative for structure.

## 2. Responsibility and boundaries

The persistence layer is responsible for:

- PostgreSQL 16 compatibility and forward-only schema migrations.
- Typed reads and writes for registry, execution, evidence, and evaluation records.
- Transactional state changes, ownership checks, uniqueness, and row-version checks.
- Short-lived worker leases and append-only state history.
- Full in-database retention of rendered prompts, sanitized request bodies, raw provider responses, extracted candidates, and validation errors.
- Integrity and reconciliation reports.

It is not responsible for:

- Parsing Turtle or deciding corpus semantics.
- Rendering prompt text.
- Sending provider requests or deciding whether an error is retryable.
- Extracting or repairing JSON.
- Computing matching or metric formulas.
- Treating CSV, JSON, or spreadsheet exports as authoritative state.

## 3. Inputs

All writes receive validated, typed records. Persistence never accepts an unvalidated free-form dictionary as a substitute for a domain record.

| Input class | Required content |
|---|---|
| Corpus evidence | Repository, release, commit, tree, file path, Git blob identifier, exact-byte SHA-256, category path, canonical gold decomposition, and version hashes |
| Configuration evidence | Exact original YAML bytes, original hash, parsed form, resolved configuration for all selected providers, resolved hash, capability report, pre-run estimate receipt/disclosure, explicit live-authorization evidence, and preflight result |
| Registry versions | Exact prompt/schema/scorer bytes or database-contained artifact, semantic version, content hash, provenance, and freeze state |
| Planned execution | Campaign, resolved-run and task fingerprints; nonempty unique selected-provider set; one provider and its model per resolved run/task; all scientific parameters |
| Attempt evidence | Task/attempt identity, full rendered messages, sanitized exact request body, native reasoning fields, timing, provider metadata, and delivery state |
| Response evidence | Exact response body, parsed envelope when available, assistant/reasoning text, status, usage, finish reason, errors, and hashes |
| Validation evidence | Extraction candidate, strategy and offsets, every validation stage, complete ordered errors, validator version, and content hashes |
| Prediction evidence | Accepted canonical six-field prediction or the versioned explicit-empty terminal prediction |
| Evaluation evidence | Gold/prediction hashes, member/role assignments and similarities, exact rational confusion receipts and decimal derivatives, unrounded Exact/Close Precision/Recall/F1 at every scope, metric numerators/denominators/support, and scorer fingerprint |
| Operational evidence | Worker identity, lease, heartbeat, task/provider/campaign transitions, provider and global cost reservations/settlements, any optional caps, usage, billing basis, price-card version when billed, and cost calculation state |

Credentials are not inputs to persistence. Only environment-variable names and non-secret provider configuration may appear in stored configuration.

## 4. Outputs

Persistence operations return typed records or explicit conflict/failure results. Their durable outputs are:

- Immutable corpus and protocol registry versions.
- One campaign row for each resolved scientific campaign fingerprint.
- Deterministically identified resolved runs and variable tasks.
- At most three numbered attempt rows per task.
- Append-only transport, extraction, validation, and task events.
- At most one selected prediction per task.
- Immutable item-, component-, and aggregate-level evaluation records.
- Queryable usage, cost, category, subcategory, provider, model, reasoning, prompt, shot, temperature, repetition, and failure dimensions.
- Reconciliation facts sufficient to determine the next safe resume stage.

An idempotent repeat with byte-identical content returns the existing record. A repeat with the same fingerprint but different content is an integrity failure; it must never overwrite the existing row.

## 5. PostgreSQL 16 conventions

- All instants use UTC timestamps with time-zone semantics.
- Application-generated UUIDv7 identifiers are used where supported by the implementation libraries.
- SHA-256 values are lowercase 64-character hexadecimal strings with validation constraints.
- Monetary amounts use exact decimal storage. Confusion contributions, aggregate metrics, and rank inputs retain exact integer numerator/denominator pairs (positive denominator, reduced form for values) plus decimal presentation derivatives; thirds cannot be authoritative finite decimals. Integer numerator/denominator columns use arbitrary-precision integral `numeric` with integrality checks. No display rounding determines a rank or tie.
- Provider-specific secondary metadata may use JSON-compatible database values, while fields needed for joins, constraints, filters, and grouping remain normalized columns.
- Exact raw text or bytes are stored separately from parsed JSON-compatible copies because a parsed JSON representation does not preserve whitespace, key order, or original bytes.
- Foreign keys are explicit. Experiment evidence uses restrictive deletion behavior; cascade deletion must not remove it.
- Terminal evidence is append-only. Corrections create a new version or evaluation run rather than rewriting the original.
- Application and database migration versions are recorded for every campaign execution.

## 6. Logical data model

The names below are logical responsibilities. Implementation may refine physical names only if the mapping is documented and all constraints remain equivalent.

### 6.1 Corpus and scientific registries

| Logical record | Responsibility and essential relationships |
|---|---|
| `corpus_snapshot` | Immutable repository/tag/commit/tree identity, expected and imported counts, manifest hash, license reference, and staged/active/failed state |
| `source_file` | Snapshot-scoped release-relative path, Git blob identifier, exact-byte hash, and exact category path |
| `science_category` | Snapshot-scoped hierarchy, exact upstream name, parent, depth, and optional separate reporting alias |
| `variable` | Stable source identifier, exact label/definition, definition predicate, source file, and category leaf |
| `gold_decomposition` | Canonical six-field JSON, lexical schema/importer identities, and canonical content hash |
| `evaluation_population` and member | Exact 97-variable membership, canonical order, source/gold hashes, demonstration exclusions, manifest version/hash |
| `demonstration_set` and member | Exact ordered five-item pool and position; shot count selects only an ordered prefix |
| `prompt_version` | Prompt-family ID, exact template bytes, rendering contract, and content hash |
| `schema_version` | Exact lexical schema, draft, semantic-rule version, and content hash |
| `scorer_version` | Evaluator code/configuration/similarity-model hashes and the system-identifier-exclusion compatibility note |
| `provider` and `model_configuration` | Provider adapter profile, exact provider model ID/revision, capability declaration, and normalized/native reasoning mappings |

Registry rows become immutable when frozen. A reviewed correction creates a new version and content hash.

### 6.2 Campaign and execution

| Logical record | Responsibility and essential relationships |
|---|---|
| `campaign` | Original and resolved configuration, hashes, optional global cost cap, expected combined counts, pre-run estimate/disclosure/authorization links, preflight evidence, and campaign state |
| `campaign_provider` | One unique `(campaign_id, provider_id)` membership, frozen provider profile/billing basis and optional cost cap, expected provider counts, dispatch state, pause reason, and reconciliation evidence |
| `resolved_run` | Exactly one campaign-provider membership and provider-owned model, applicable reasoning profile, prompt, shots, temperature, sampling values, repetition, shared evaluation population, and run fingerprint |
| `task` | One resolved run applied to one target variable, task fingerprint, state, attempt count, lease, heartbeat, and row version |
| `attempt` | Attempt number 1, 2, or 3; correction parent; full request evidence; status; delivery certainty; and provider IDs |
| `transport_event` | Append-only dispatch, HTTP, timeout, connection, rate-limit, ambiguous-delivery, and receipt facts |
| `extraction_event` | Extractor version, ordered strategy, candidate offsets/content/hash, diagnostics, and outcome |
| `validation_event` | Validator stage/version, pass/fail, JSON Pointer where relevant, and complete ordered error data |
| `prediction` | One accepted canonical prediction or one versioned explicit-empty terminal prediction |
| `task_event` | Append-only previous/next state, cause, actor, time, and related attempt/evidence identity |
| `worker_session` | Worker/code/environment identity, start/end, health, and lease-heartbeat audit fields |

### 6.3 Evaluation, usage, and reporting

| Logical record | Responsibility and essential relationships |
|---|---|
| `evaluation_run` | Frozen prediction set and scorer/code/environment fingerprint |
| `evaluation_item` | Variable, gold/prediction hashes, item status, and terminal-failure policy |
| `match_record` | Component/role, original and normalized member references, every permitted pair similarity/eligibility, selected/unmatched occurrences, assignment mode/tie evidence, absent-role-evidence flags, and rule ID |
| `confusion_contribution` | Exact TP/FP/FN/TN numerators/denominators, decimal derivatives, source match IDs, branch/rule ID, and system counts `g`, `p`, `m`, `U` where applicable |
| `metric_value` | Exact/Close mode, explicit Precision/Recall/F1 name, numerator, denominator, support, unrounded value, variable/component/repetition or aggregate scope, and source-contribution links |
| `ranking_run` | Evaluation-population/evaluation identities plus accepted policy `mean-repetition-micro-close-f1-v1` frozen for that run, completeness check, and immutable ranking fingerprint |
| `configuration_rank` | Provider/model configuration identity without repetition, all repetition-metric links, primary mean, shared rank across the selected providers or explicit `not_rankable` reason, and analysis columns |
| `analysis_slice` and member | Frozen category, subcategory, configuration, failure, and other report memberships |
| Usage evidence | Provider-reported raw usage, normalized token categories, availability state, and source attempt |
| Cost evidence | Frozen provider billing mode/basis, versioned price card when billed, original/reporting currencies, quantities, rates, confirmed-zero/actual/estimated/unavailable state, and adjustment history |
| Pre-run cost estimate | Immutable `pre-run-estimate-v1` receipt/hash, resolved configuration/plan and price-card/FX hashes, dated sources, token/count/scenario assumptions, per-model/provider and combined totals, disclosure receipt, and separately linked explicit live authorization |
| Cost reservation and settlement | Attempt-scoped exact-decimal estimated exposure with bound status/assumptions; a defensible upper bound for capped admission; checks against every non-null provider/campaign cap, idempotent settlement/release, unreconciled exposure, and audit events even when uncapped |

Only storing an aggregate Precision, Recall, F1, or rank is prohibited. Item-level match and confusion evidence must reproduce every aggregate, and stored unrounded repetition aggregates must reproduce every configuration rank. Lower-ranked, tied, incomplete, and failed configurations remain queryable.

For `january-derived-member-credit-v1`, validate `U = g + p - m`, `0 <= m <= min(g,p)`, one-to-one member assignment, role eligibility, and exact system contribution total one. Preserve numerator `m`, `p-m`, and `g-m` with denominator `U`, plus reduced value ratios. Other branches retain their historical computation and exact finalized source-number receipt. Aggregate and rank from exact receipts; do not reconstruct fractions from rounded decimals. Member arithmetic and source identities are checked by typed boundary validation and cross-record integrity checks as well as SQL constraints.

Each scored variable retains its own Exact and Close Precision, Recall, and F1, its component metrics, and the contributions/numerators/denominators/support needed to recompute them. Mean, median, variance, mode, standard deviation, range, and IQR over these per-variable scores are deferred until the database is populated, along with further uncertainty analyses. These future analyses use separate versioned query definitions; they do not require another generation run and are not mandatory evaluation rows now.

## 7. Full evidence storage

### 7.1 Prompts and requests

Every attempt stores the complete message sequence and a human-readable rendered prompt, including correction material for attempts two and three. It also stores the exact sanitized provider request body that was prepared for dispatch, its hash, the selected model, normalized reasoning profile, exact native reasoning fields, all supported sampling values, output limit, and deterministic idempotency key.

Sanitization removes only secrets and prohibited headers. It must not remove scientific request content. The redaction procedure and version are stored.

### 7.2 Raw responses

The exact provider response body is retained before extraction or validation. When a provider returns a JSON envelope, the database stores both an exact textual or binary representation and a parsed queryable representation. Assistant content, optional reasoning content, usage, returned model ID, provider request/system identifiers, finish reason, and timing remain linked to that exact body.

A filesystem pathname, log line, spreadsheet cell, or parsed JSON value alone is not sufficient raw-response evidence.

### 7.3 Errors and decisions

Transport, extraction, lexical-schema, semantic-validation, budget, and orchestration errors are stored in full, ordered form. Each error includes a stable code, stage, validator/classifier version, human-readable message, JSON Pointer when applicable, sanitized offending-value summary, time, and retry decision.

The database records both the classification produced by a component and the task-state decision made from it. This prevents later code from rewriting the historical reason for a retry or terminal failure.

### 7.4 Hash verification

Exact evidence receives a content hash at the application boundary and is rechecked when read for resume, evaluation, export, or backup verification. A parsed derivative has its own hash and cannot replace the raw-artifact hash.

## 8. Required integrity constraints

The implementation must enforce, through database constraints plus preflight/transactional checks where a cross-row rule cannot be expressed locally:

- One campaign fingerprint identifies one immutable scientific campaign specification.
- A campaign selects a nonempty unique subset of `psnc` and `openrouter`; each selected provider has at least one enabled, fully specified model before live execution. The initial intention of three models per provider is configurable, not a database cardinality constraint.
- Composite foreign keys enforce each resolved run's `(campaign_id, provider_id)` membership and `(provider_id, model_configuration_id)` ownership. The same model-ID text on two providers is not the same provider-model identity.
- Every task, attempt, prediction, score, usage/cost fact, and rank has an unambiguous provider/model lineage; denormalized provider columns must agree with that lineage.
- The campaign plan is the union of each selected provider's model grid. It contains no cross-provider model pairings and uses the same 97-variable population, demonstrations, prompt/schema/scorer, and ranking protocol throughout.
- Reasoning-capable models resolve to the frozen `disabled` and `enabled` profiles; incapable models resolve only to `not_applicable` and send no reasoning field.
- Run and task fingerprints are unique.
- The evaluation population contains exactly the 97 non-demonstration Corpus variables, each once.
- A task belongs to exactly one resolved run and target variable.
- Attempt identity is unique within a task, and attempt numbers are limited to 1, 2, and 3.
- A correction parent belongs to the same task and immediately precedes the correction attempt.
- One attempt represents at most one outbound provider invocation. Provider adapters cannot create hidden attempts.
- A selected valid attempt belongs to its task and has passed every required validation stage.
- A task has at most one prediction; the explicit-empty form is permitted only under the frozen terminal-failure policy.
- Demonstration variables cannot be members of the evaluation population.
- Every resolved run has exactly one planned task for every evaluation-population member.
- A completed task has one prediction and the required evaluation linkage.
- A configuration rank can be assigned only when every variable/repetition result is present and no unresolved operational failure remains; otherwise an explicit `not_rankable` record is required.
- Campaign completion requires every planned task across every selected provider to be scored, every planned configuration to be covered by the final combined ranking, and required reports to be durably recorded. A paused provider prevents whole-campaign completion without deleting completed work from another provider.
- Terminal attempt, prediction, and evaluation evidence cannot be updated or cascade-deleted.
- Missing usage or billing information remains unavailable/null with a reason; it is never assumed to be zero. A recorded owner-reported no-charge billing basis may explicitly establish zero financial cost for that provider/account context; it does not establish zero token usage or universally free access.
- Live dispatch requires a valid pre-run estimate bound to the frozen plan, evidence that it was disclosed, and separate explicit live authorization. An estimate is not a spending ceiling or authorization by itself.
- Each dispatched attempt has one reservation or explicit no-charge zero-cost receipt. Accounting is atomic even when uncapped. Billed reservations must fit every configured non-null provider/global cap, including all existing unreconciled reservations; concurrent workers cannot independently spend the same remaining allowance.
- Under configuration `2.1`, a null cap means no cap, zero is a real zero ceiling, and negative amounts are invalid. All default cap amounts are null, including PSNC; no-charge billing is recorded independently from cap settings.

## 9. Transaction boundaries and actions

### 9.1 Corpus activation

Corpus ingestion writes into a staging snapshot. One final transaction verifies the 102 records, source/manifests hashes, category counts, demonstration mapping, the exact 97-member evaluation population, and gold-schema validity before marking the snapshot active. Failure leaves no partially active corpus.

### 9.2 Campaign registration and planning

Registration stores exact configuration evidence before task creation. Planning inserts resolved runs and tasks by deterministic fingerprint in bounded transactions. Repeating a byte-identical planning operation is idempotent; conflicting content fails.

Registration creates all selected `campaign_provider` memberships together. Planning validates per-provider and combined counts before making the complete union dispatchable. Under D-029's one repetition at every temperature, let `S` be the sum of applicable reasoning-profile counts over all enabled models of all selected providers: expected counts are `48 × S` ranked configurations, `48 × S` repetition runs, `4,656 × S` variable tasks/initial calls, and at most `13,968 × S` dispatched requests. These counts derive from the frozen lists, not an assumed six-model constant. Retain repetition identities and general membership constraints; singleton repetitions do not remove that database dimension or justify run-to-run variance claims.

Before live dispatch, persist the estimate receipt with `estimate_policy_version: pre-run-estimate-v1` and its hash. It references the frozen resolved configuration/plan, price and billing evidence, reporting-currency conversion provenance and effective dates, and explicit assumptions for expected and bounded scenarios. Store planned one-attempt and maximum-three-attempt call counts, prompt/output/reasoning token quantities and bounds, correction-message growth assumptions, and monetary subtotals for every model/provider and the campaign. Unknowns remain explicit and block a misleading definite estimate. Record when and what was disclosed to the owner and the separate live-authorization evidence; do not store credentials or infer consent from providing an API key. A changed resolved campaign or price/FX basis requires a new estimate and disclosure before its live execution.

### 9.3 Task claim

A worker claims eligible tasks in a short transaction using skip-locked row-claiming semantics. The claim writes worker identity, lease expiry, heartbeat, and a fencing row version. New dispatch claims check both campaign-wide and provider-specific eligibility; a provider pause does not prevent durable local processing or eligible work for another provider. No database transaction remains open during provider I/O.

### 9.4 Write-ahead attempt evidence

Before dispatch, a transaction commits the attempt number, full rendered prompt/messages, sanitized exact request body, request hash, scientific parameters, correction parent, deterministic idempotency key, and provider/global reservation or explicit no-charge receipt. A request must not be sent if this commit fails.

Dispatch and receipt are represented by append-only transport events. After a response arrives, the exact raw body and response metadata are committed before extraction begins. Extraction, validation, prediction selection, and evaluation each write idempotently against their evidence fingerprints.

Cost admission locks campaign and provider accounting rows in a consistent order, validates the estimated exposure and its bound status, checks every non-null cap using a defensible upper-bound reservation, and creates one reservation linked to the attempt in the same transaction as dispatch authorization. Response accounting settles or releases that reservation idempotently; ambiguous delivery retains its exposure until reconciled. Stored billing evidence distinguishes provider-reported amounts, versioned price-card estimates, and explicit no-charge zero cost. PSNC is initially owner-reported no-charge in the user's access context; OpenRouter is metered and requires a price card, not a mandatory spending cap. Null caps allow uncapped dispatch after the required estimate/disclosure and live authorization; exceeding the estimate alone does not pause work. Unknown pricing or absent usable estimate evidence blocks billed dispatch; an unsupported guaranteed upper bound blocks capped admission, not an otherwise documented uncapped estimate with explicit uncertainty. If an optional provider cap is configured, exhaustion blocks that provider's positive-cost dispatch; if an optional global cap is configured, exhaustion blocks positive-cost dispatch across the campaign. A configured zero cap permits only zero-cost work. Explicitly no-charge work may continue with its zero-cost receipt, subject to all other eligibility checks. Unresolved positive-cost work still prevents final campaign completion.

### 9.5 Prediction and evaluation

Prediction selection and its task transition occur atomically. An evaluation run is keyed by the frozen prediction-set and scorer fingerprints. Repeating evaluation with identical inputs returns the existing evaluation; changing a scorer or input creates a new run.

### 9.6 Configuration ranking

Ranking first records a completeness result for every configuration from the union of the selected provider grids. Under accepted D-023 policy `mean-repetition-micro-close-f1-v1`, it links each declared repetition's unrounded micro Close F1, calculated from summed contributions over all 97 variables, and takes their arithmetic mean. Eligible configurations are ranked together descending with exact ties assigned shared competition rank; incomplete or operationally unresolved cases receive an explicit `not_rankable` reason under the ranking-run fingerprint. Provider/model identity is retained on every ranking row. Provider-filtered analysis views have explicit filters and do not replace the complete campaign ranking. Ranking does not average the 97 per-variable F1 values or recompute F1 from averaged Precision and Recall. Repeating the same ranking is idempotent; a policy or input change creates a new ranking run. No ranking transaction deletes, updates, or filters generation or evaluation evidence.

## 10. Task state, leases, and concurrency

Task state follows the state machine in `TECHNICAL_SPECIFICATION.md`. Every transition produces a task event and checks the current state, lease owner, lease expiry, and row version.

Leases are operational coordination, not ownership of evidence. An expired lease allows another worker to reconcile the task, but it does not authorize a new provider request when an existing attempt may have been dispatched. Stale workers are prevented from committing a conflicting state by their fencing version.

Heartbeats extend only an active, matching lease. They do not alter scientific identity or attempt budget. Workers release leases after a safe durable checkpoint; abrupt termination leaves them to expire.

## 11. Failure and recovery behavior

| Failure point | Required database behavior |
|---|---|
| Before campaign/task commit | No provider activity is permitted; retrying registration/planning is idempotent |
| Before attempt evidence commit | Do not dispatch; the task remains safely resumable |
| After attempt commit but before confirmed dispatch | Reconcile transport state; dispatch only when non-dispatch is certain |
| After dispatch but before response commit | Treat as possible ambiguous delivery; never infer that no generation occurred |
| Response received while PostgreSQL is temporarily unavailable | Do not parse, retry, or call again; keep retrying the same evidence commit while the process lives |
| Process loss with an uncommitted received response | Reconciliation sees a possible dispatch without durable response and applies ambiguous-delivery policy |
| After raw response commit | Resume extraction from stored evidence with no provider call |
| After extraction or validation commit | Continue from the earliest incomplete durable stage |
| After prediction commit | Continue scoring; do not regenerate |
| During evaluation | Insert by evaluation fingerprint so restart cannot duplicate or overwrite results |
| Hash/content mismatch | Quarantine/pause the affected campaign and emit an integrity failure |
| Migration or PostgreSQL-major mismatch | Fail preflight before task claims or provider requests |
| Provider outage, authentication/configuration error, or provider-specific limit | Persist the affected provider's pause/backoff reason; continue eligible work for other selected providers; preserve exact task/provider ownership |
| Configured optional global monetary cap boundary | Pause positive-cost dispatch across providers; explicitly no-charge work may continue; retain in-flight evidence and reservations |
| Shared database/integrity failure or explicit user stop | Pause all new provider dispatch; retain in-flight evidence and reservations for reconciliation |

PostgreSQL unavailability never justifies a filesystem-only authoritative fallback. If evidence cannot be made durable, the workflow pauses safely.

The database preserves per-provider progress while the orchestrator continues eligible work until all selected tasks and final reports are complete. Recovery never moves a task to another provider/model or resets its three-attempt allowance. Recoverable provider backoff can expire automatically under the frozen policy; unresolved credentials, exhausted limits, or ambiguous delivery remain explicit pauses rather than a false completed campaign.

## 12. Ambiguous delivery constraint

Exactly-once execution against an external provider cannot be inferred after a connection loss unless the provider offers a reliable idempotency or request-lookup mechanism. Therefore, the database preserves the distinction between safely-not-dispatched, confirmed-response, confirmed-rejected, and ambiguous-delivery states.

An ambiguous attempt retains its attempt number and request evidence. Resume first tries the provider-supported lookup or idempotency reconciliation recorded for that adapter. It does not blindly create another outbound call. If the response cannot be recovered and the protocol authorizes a replacement, that replacement is a new numbered attempt and consumes the remaining three-attempt budget. Without a safe resolution, the task remains paused with explicit evidence rather than risking an unrecorded duplicate paid generation.

## 13. Security, retention, and access

- API keys, authorization headers, cookies, database passwords, and complete connection strings are prohibited from evidence columns.
- Raw responses are untrusted data and are never rendered as executable markup by database/reporting tools.
- Full prompts and responses remain available for reproducibility, so database access and backups require experiment-level authorization.
- Logs default to identifiers, hashes, states, and sanitized summaries; they do not duplicate full raw evidence.
- Evidence deletion is not part of normal workflow. Any future retention policy requires a new reviewed protocol and an auditable tombstone process.
- The database role used by workers receives only the permissions required for planned execution; migration and backup roles are separate.

## 14. Backup, restore, and reproducibility

A reproducible campaign archive consists of a PostgreSQL backup plus the pinned repository revision, corpus/manifests, dependency lock, and environment record. Backups are versioned, encrypted according to deployment policy, and verified through restore tests and sampled content hashes.

After restore, an integrity report must verify registry hashes, campaign/run/task counts, attempt counts, terminal-state constraints, raw-evidence hashes, prediction/evaluation linkages, metric regeneration, configuration coverage, and ranking regeneration. Derived exports may be recreated only after this verification succeeds.

## 15. Failures exposed to callers

Persistence operations return typed failures for unavailable database, unsupported PostgreSQL version, migration mismatch, constraint violation, fingerprint/content conflict, stale lease, invalid transition, attempt-budget violation, evidence hash mismatch, immutable-row mutation, missing parent evidence, and backup/integrity failure.

A caller may retry an unavailable-database transaction only when doing so is idempotent. It may not translate a database failure into another provider request.

## 16. Acceptance criteria

- A clean PostgreSQL 16 instance migrates to the documented schema version.
- The database can regenerate the complete union of selected provider grids and reconcile every provider subtotal and combined campaign count from stored configuration.
- Composite ownership constraints reject unselected providers and cross-provider model/attempt/score links.
- Concurrent dispatch admission and settlement retain exact-decimal accounting with null caps and respect every configured non-null cap; no-charge zero receipts remain distinct from unknown usage/cost.
- Missing, undisclosed, or configuration-mismatched estimates and missing explicit live authorization block live dispatch; a disclosed estimate alone never becomes a hard ceiling.
- A provider-specific pause permits healthy-provider progress. Optional global monetary cap exhaustion blocks positive-cost dispatch while explicitly no-charge work can continue; shared persistence/integrity failure or explicit stop blocks all new dispatch. Whole-campaign completion requires all selected work and reports.
- Full base/correction prompts, sanitized requests, exact raw responses, extracted candidates, and complete validation errors survive backup and restore.
- Real concurrent planning and claiming tests prove unique campaigns, runs, tasks, attempts, and active task ownership.
- A fourth attempt cannot be inserted or dispatched.
- Terminal rows cannot be overwritten or removed through cascade deletion.
- Stale workers cannot commit over a newer lease owner.
- Every crash checkpoint in `docs/retry-and-resume.md` resumes from the earliest safe durable stage.
- Ambiguous delivery never causes an automatic blind duplicate request.
- Item-level match and confusion records reproduce every stored aggregate metric.
- Every scored variable, component, and repetition retains unrounded Exact/Close Precision, Recall, and F1 with the contributions and denominator/support that reproduce them.
- Repetition aggregates reproduce every stored primary ranking value and shared rank while retaining all configurations.
- Repeated planning, response storage, prediction selection, evaluation, resume, and export are idempotent.
- Sampled raw and canonical content hashes verify after backup/restore.
