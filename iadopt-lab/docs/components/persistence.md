# PostgreSQL Persistence Contract

## Responsibility

Implement migrations, transactions, typed repositories, immutable evidence, task leases, configuration/corpus registries, and queryable evaluator facts in PostgreSQL 16.

Persistence does not implement scoring formulas, provider protocols, prompt text, or retry policy. It enforces their identities and constraints.

## Inputs

- Validated typed registry/execution/evaluation records
- Exact evidence text/JSON and hashes
- Transaction and expected row-version information
- PostgreSQL DSN from `DATABASE_URL`

## Outputs

- Durable rows with generated IDs and timestamps
- Idempotent existing-record results for matching fingerprints
- Atomic claim/transition results
- Reconciliation and integrity reports
- Immutable evaluation-population, repetition-metric, and configuration-ranking records

## Core guarantees

- Full prompts, raw responses, and validation errors are stored.
- Terminal records are immutable.
- Attempt number is constrained to 1–3.
- One task has at most one selected prediction.
- A campaign has a nonempty unique selected-provider set. Composite foreign keys bind each run/task to exactly one selected provider and one model owned by it; same-text IDs on different providers remain distinct.
- Selected provider grids are combined by union, with one shared 97-variable population/protocol and no model pairing across providers.
- Selected attempts must belong to their task.
- Evidence tables use restrictive foreign keys and do not cascade-delete.
- Money uses exact decimal storage. Confusion contributions and metric/ranking values retain exact integer numerator/denominator receipts plus decimal derivatives; display rounding never determines rank ties.
- Every mutable state transition uses transaction and row-version checks.
- All dispatches retain cost/reservation records; atomic ceiling checks apply only to non-null provider/global caps. Null means uncapped, explicit zero means a zero ceiling. Explicit owner-reported no-charge zero cost retains its billing basis; unavailable usage or price is not replaced by zero.
- The frozen plan's versioned pre-run estimate, disclosure evidence, and subsequent separate explicit live authorization remain immutable and linked; actual cost never overwrites the estimate. An estimate is not a cap.
- Provider-specific pauses preserve eligible work for healthy providers. An optional non-null global monetary cap's exhaustion blocks positive-cost dispatch while explicitly no-charge work can continue; shared database/integrity failures or explicit stop block all new dispatch. Campaign completion checks all selected provider tasks and required final reports.
- Every attempt, prediction, item/component score, repetition aggregate, lower-ranked result, tie, and unranked reason remains queryable.

## Configuration keys consumed

- `database.*`
- `execution.task_lease_seconds` and `execution.heartbeat_seconds`
- `campaign.providers`, `providers.*.billing`, `cost_accounting.*`, and optional provider/global cost caps
- Resolved identities derived from `dataset`, `demonstrations`, `evaluation_population`, `ranking`, `providers`, `parameter_grid`, `generation`, and `evaluation`

## Planned repository operations

### `register_campaign(configuration) -> CampaignRecord`

- **Input:** One schema-valid resolved configuration with exact original/canonical bytes and hashes, all referenced frozen artifact identities, approval/preflight evidence, and non-secret actor/context metadata.
- **Action:** In one transaction, verify every registry reference and selected provider/model/billing identity, insert campaign and unique campaign-provider memberships by scientific fingerprint, or byte-compare and return existing records; record original YAML separately from the canonical resolved union. Inactive catalogs do not enter the resolved fingerprint; changing selected membership or active profiles creates a new campaign.
- **Output:** Immutable campaign record with database ID, fingerprints, artifact foreign keys, freeze/readiness state, and created-or-existing disposition.
- **Raises:** Missing/unfrozen reference, failed preflight, fingerprint/content conflict, secret-bearing value, migration mismatch, or transaction failure.
- **Side effects:** PostgreSQL registry write only; no task planning or provider call.
- **Idempotency:** Identical content returns the same campaign; same fingerprint with different bytes never overwrites.

### `plan_tasks(campaign, resolved_runs, variables) -> PlanningResult`

- **Input:** One frozen campaign, complete deterministic resolved-run expansion, and exact ordered 97-member evaluation population with source/gold hashes.
- **Action:** Verify per-provider and combined configuration/repetition/task counts and composite provider/model ownership, then insert the union of run and variable-task identities in bounded transactions under campaign-scoped unique constraints; no dispatchable task is created from an incomplete plan and no task is shared across campaigns.
- **Output:** Per-provider and combined counts/IDs for existing/created configurations, repetition runs, and tasks; expected/max request bounds; shared population coverage proof; and plan fingerprint.
- **Raises:** Campaign not runnable, count/coverage mismatch, duplicate/conflicting fingerprint, population/demo overlap, foreign-key mismatch, or transaction failure.
- **Side effects:** PostgreSQL planning rows/events only; zero provider calls.
- **Idempotency:** Replanning byte-identical inputs creates no duplicate logical row and returns the same plan identity.

### `claim_tasks(worker, limit, lease) -> tuple[TaskLease, ...]`

- **Input:** Registered worker/session identity, positive claim limit, current database time, lease duration, campaign/filter scope that cannot alter scientific membership, and eligibility state set.
- **Action:** In a short transaction, select eligible tasks with `FOR UPDATE SKIP LOCKED` (or documented equivalent), checking campaign-wide and provider-specific dispatch gates, apply fencing token/expiry/row version, append claim events, and commit before returning. A provider dispatch pause must not block eligible other-provider work or already durable local processing.
- **Output:** Zero or more typed leases containing task identity, owner, fencing token, acquired/expiry times, row version, and next durable stage.
- **Raises:** Unknown worker/campaign, invalid limit/duration/filter, migration failure, or database conflict; ordinary contention returns fewer leases.
- **Side effects:** PostgreSQL lease/state-event writes only; never holds a transaction across provider or embedding work.
- **Concurrency:** A task has one valid active owner; stale tokens cannot mutate it after reclaim.

### `start_attempt(lease, request_evidence) -> AttemptRecord`

- **Input:** Valid current task lease/fencing token and complete sanitized request evidence: attempt number, prompt/messages/hashes, native fields, idempotency key, provider/model/adapter/profile identities, frozen billing basis, attempt cost estimate or explicit no-charge receipt, plan-bound estimate disclosure/live-authorization references, and provider/global accounting state with optional caps.
- **Action:** Verify task state, composite ownership, next consecutive number, `1..3` cap, parent lineage, parameter invariance, estimate/authorization scope, and evidence hashes; lock global and provider accounting rows in consistent order, check only non-null remaining caps (requiring a defensible reservation bound for capped paid dispatch), then insert the unique reservation/no-charge receipt, immutable attempt/request rows, and dispatch-pending event atomically. Uncapped execution still retains estimates, actual settlement, and ambiguous exposure.
- **Output:** Durable attempt record, linked provider/global reservation or zero-cost receipt, and proof that all reconstructable request evidence committed before dispatch.
- **Raises:** Stale lease, wrong owner/state/number, attempt four, missing correction parent, parameter drift, absent/mismatched estimate disclosure or live authorization, configured provider/global cap exhaustion, missing billing/estimate evidence or defensible bound for capped paid dispatch, secret detection, hash conflict, or transaction failure. A null cap is never an error.
- **Side effects:** PostgreSQL write only; this operation never dispatches HTTP.
- **Idempotency:** Identical retry returns the existing attempt; conflicting content under the same task/number fails.

### `store_response(attempt, response) -> AttemptRecord`

- **Input:** Existing attempt identity/fencing context plus exact provider result: raw bytes/text/envelope, assistant/reasoning content, status, usage, IDs, timing, error/delivery classification, sanitized hashes, and receipt facts.
- **Action:** Verify ownership/request lineage, persist untouched raw evidence separately from parsed/indexed fields, append delivery/receipt events, settle or retain the linked reservation from frozen billing/usage evidence idempotently, hash-verify the write, and advance only to the documented local-processing or ambiguous state. Unknown delivery retains its exposure; absent usage stays unavailable even under a no-charge billing basis.
- **Output:** Updated immutable-evidence view of the attempt with response IDs/hashes, delivery state, and next safe processing stage.
- **Raises:** Unknown/wrong attempt, stale conflicting writer, duplicate response with different bytes, secret-bearing evidence, invalid delivery transition, hash mismatch, or transaction failure.
- **Side effects:** PostgreSQL response/event write only; no extraction, retry, or provider call.
- **Idempotency:** Byte-identical response replay returns the same stored evidence; content conflict is never overwritten.

### `store_prediction_and_complete(task, prediction) -> TaskRecord`

- **Input:** Current task/fencing identity, selected valid or versioned terminal-empty prediction with source attempt/failure lineage, validation/canonical hashes, and expected row version.
- **Action:** Verify prediction belongs to the task and allowed terminal path, enforce one selected prediction, insert immutable raw/extracted/canonical links, append state events, and move to `prediction_ready` (not silently past scoring) atomically.
- **Output:** Task record with selected prediction identity/status and next stage, plus created-or-existing disposition.
- **Raises:** Stale lease/version, wrong attempt/task, invalid terminal-empty provenance, duplicate conflicting prediction, hash/schema conflict, or transaction failure.
- **Side effects:** PostgreSQL prediction/state writes only; no evaluation occurs inside the repository.
- **Idempotency:** Identical prediction selection returns the same record; any different second selection fails.

### `store_evaluation(records) -> EvaluationRun`

- **Input:** Complete compatible evaluator records, explicit prediction-set/population/scorer/environment identities, expected coverage, and canonical hashes.
- **Action:** Verify item uniqueness/coverage and contribution/aggregate reconciliation, register or resolve the evaluation fingerprint, insert item/component/match/contribution/metric facts, then mark the run complete only after integrity queries pass.
- **Output:** Immutable evaluation run with IDs, coverage, exact/close aggregates, item/component counts, integrity status, and created-or-existing disposition.
- **Raises:** Missing/duplicate item, incompatible scorer/prediction/population, contribution mismatch, hash conflict, immutable-run conflict, or transaction failure.
- **Side effects:** PostgreSQL evaluation writes only; never calls the evaluator or provider.
- **Idempotency:** Identical complete records return the same evaluation run; changed scorer/content creates a distinct fingerprint.

### `store_configuration_ranking(records) -> RankingRun`

- **Input:** Explicit campaign/evaluation run/population/ranking-policy identities plus every selected provider/model configuration's repetition aggregates, per-provider and combined expected coverage, rankability decision/reason, unrounded primary value, secondary values, and source hashes.
- **Action:** Recompute/verify the full frozen provider union, population and repetition coverage, arithmetic means, combined ordering, shared-rank/tie outcomes, and all `not_rankable` reasons; insert a new immutable ranking namespace without modifying generation/evaluation facts. Retain provider/model on every row; incomplete selected-provider work prevents final whole-campaign completion.
- **Output:** Ranking run containing all eligible ranks, ties, every lower-ranked configuration, every ineligible configuration/reason, ranking inputs, policy/version hash, and integrity report.
- **Raises:** Missing configuration/result, wrong population, inconsistent repetition mean/order/tie, hidden failure, source hash conflict, policy mismatch, or transaction failure.
- **Side effects:** PostgreSQL derived-ranking writes only; no provider request, rescoring, deletion, or overwrite.
- **Idempotency:** Identical inputs/policy return the same ranking; changed policy creates a separate ranking run.

## Acceptance tests

- Empty database migrates to current version
- Real concurrent duplicate planning
- Lease race and stale-worker rejection
- Attempt-four rejection
- Response/prediction idempotency
- Immutable terminal rows
- FK ownership rules
- PSNC-only, OpenRouter-only, and combined selected-provider memberships and union totals
- Concurrent accounting/admission with null, explicit-zero, and positive optional caps; settlement replay and ambiguous reservation retention
- Pre-run estimate/disclosure/live authorization survive resume and cannot be overwritten by actual costs
- No credential, `.env` content/hash, or secret-bearing DSN enters persisted evidence
- Explicit no-charge zero cost versus unavailable usage/billing evidence
- Provider pause isolation and all-provider campaign/report completion
- Evaluation re-run fingerprints
- Exact 97-member population coverage and demonstration exclusion
- Ranking regeneration, shared ties, incomplete-status retention, and no result deletion
- Backup/restore and sample hash verification

## Initial implementation mapping

The implementation exposes a synchronous `Repository` backed by `psycopg_pool`.
The asynchronous runner invokes its short operations through `asyncio.to_thread`;
no connection or transaction is retained while a provider or embedding model runs.
`migrate(dsn, migrations_dir=None)` applies hash-checked forward SQL migrations
under a PostgreSQL advisory lock and requires PostgreSQL major version 16.

`register_artifact(kind, content, metadata)` retains exact bytes and their hash.
`register_corpus(snapshot, variables)` stores immutable corpus/source/category/gold
facts. `register_campaign(configuration, mode, artifact_refs)` stores the complete
resolved configuration and provider-owned model membership; `plan_tasks` freezes
the complete run/variable union before making it claimable. Synthetic campaigns
may use a declared three-variable subset; live campaigns require exactly the 97
non-demonstration targets. UUID5 identifiers derived from immutable fingerprints
provide deterministic IDs on Python 3.12, which has no standard-library UUID7.

Run mappings contain `configuration_id`, `provider`, `model_id`,
`reasoning_mode`, `reasoning_fields`, `prompt_variant`, `shot_count`, `temperature`,
`top_p`, `max_output_tokens`, `repetition`, and `configuration`. Variable mappings
contain `variable_id`, `source_path`, `definition`, `gold`, `category`,
`subcategory`, `category_path`, `source_sha256`, and `gold_sha256`; optional source
bytes and corpus metadata preserve additional provenance. Every mapping passes
explicit boundary checks before insertion. Essential joins/grouping dimensions
are columns; full input mappings are retained as immutable JSON evidence.

`claim_tasks` returns a lease containing the full run and variable records plus
`task_id`, `worker_id`, `lease_token`, and monotonically increasing `fence`.
`start_attempt` stores `messages`, `prompt`, sanitized `body`, frozen
`scientific_parameters`, and consecutive attempt identity before any dispatch.
`mark_dispatched` must commit immediately before the provider call. A stored
dispatch without a durable response is ambiguous on resume, even when the process
may have stopped immediately before sending. `store_response` preserves raw bytes
and normalized response facts atomically and permits idempotent late response
evidence without giving a stale worker permission to replace another owner's state.

`record_validation` persists the complete extraction/schema/semantic receipt.
`select_prediction` requires valid evidence or exactly three delivered invalid
receipts; it moves only to `prediction_ready`. `store_evaluation` retains the full
exact-rational evaluator receipt and normalized per-mode/component metrics before
marking a task complete. `release`, `heartbeat`, `set_provider_state`, and
`reconcile` implement operational coordination; they never reset attempts.
`save_ranking`, `save_report`, and `complete_campaign` preserve all configurations
and require complete coverage before recording final completion. Read operations
`get_campaign`, `get_task`, `list_tasks`, and `list_attempts` verify stored content
hashes and return sufficient evidence for the runner's next safe stage.

Live admission additionally requires immutable estimate, disclosure, and explicit
authorization receipts bound to the campaign's frozen plan. These can be recorded
with `record_live_authorization` after planning; synthetic mode cannot be promoted
to live. All monetary caps remain optional (`null` is uncapped), and every attempt
has an accounting receipt even when confirmed non-billed. Database constraints
and immutable-row triggers supplement application validation; integrations test
the database directly using only an explicitly designated isolated test database.
