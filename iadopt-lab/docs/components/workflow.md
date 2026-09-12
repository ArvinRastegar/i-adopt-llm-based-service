# Planning, Runner, and Resume Contract

## Responsibility

Expand a validated campaign deterministically, disclose its cost estimate before separate live authorization, claim tasks safely, coordinate bounded parallel execution, advance durable state, enforce provider limits and any optional monetary caps, and resume from the earliest incomplete durable stage.

The workflow layer does not contain provider wire details, JSON extraction algorithms, evaluator formulas, or reporting calculations.

## Inputs

- Frozen campaign/configuration and preflight report
- Non-empty selected provider set and separate provider-owned model lists
- Immutable 97-variable evaluation-population manifest
- PostgreSQL repositories
- Prompt, generation, validation, and evaluator services
- Worker limit, per-provider concurrency/rate limits, billing evidence, and optional provider/global cost caps
- Versioned pre-run estimate and separate live authorization bound to the frozen plan

## Outputs

- Resolved run and task plan with fingerprints
- Exact task/request counts, expected token/cost scenarios, conditional conservative bounds, and disclosed assumptions/unknowns
- Worker sessions and task-event history
- Completed, terminal-invalid, paused, or failed task states
- Resume/reconciliation report
- Complete configuration-ranking input set and ranking-build status

## Planning expansion

For each selected provider, expand only its enabled owned models across applicable reasoning profiles, prompts, shots, temperatures, repetitions, and all 97 targets in the immutable evaluation population. The campaign is the union of those provider-specific grids. It is never the product of all providers with all model IDs. Every resolved run and task fixes exactly one provider/model pair. A capable model yields enabled and disabled reasoning configurations. An incapable model yields one not-applicable configuration.

D-029 sets both temperature-zero and nonzero repetition counts to one for this campaign. Per model/reasoning profile, expand 48 configurations and 48 runs over 97 variables: 4,656 tasks/initial calls and at most 13,968 total requests. Keep repetition identity and configurable repetition expansion for future campaigns, but do not infer extra repetitions from nonzero temperatures. The three-request retry allowance is independent of these repetition counts.

The initial selection is the owner's three PSNC models plus three OpenRouter models recorded in `docs/model-catalog.md`. Exact IDs are supplied; deployment and reasoning capabilities still need verification. Later campaigns may select either provider or both and change either list before freezing. The planner shows provider/model subtotals and their sum. Exact plan freeze fails if a selected model list is empty or an enabled profile remains unresolved; missing capability is never interpreted as zero tasks. An unselected provider contributes zero tasks and requires no credential for this campaign.

All selected providers must pass initial preflight before any live dispatch. Plan preparation and cost estimation use local/frozen evidence without paid calibration calls; they must be possible before live authorization. D-027 requires showing the owner a plan-bound estimate and obtaining separate explicit live authorization. Credentials or implementation approval alone cannot authorize dispatch. Campaign-scoped run/task identities remain distinct when selection changes create a new campaign; earlier campaigns are not silently reused as completed tasks.

## Configuration keys consumed

- `campaign.*`
- `evaluation_population.*`
- `ranking.*`
- Every selected entry under `providers.*`, including per-provider `billing.*`
- `parameter_grid.*`
- `generation.max_generation_attempts_per_task`
- `cost_accounting.*`
- `execution.*`

## Concurrency

The runner uses a bounded global worker pool plus separate semaphore, rate limiter, cooldown, and cost reservation scopes for each selected provider. `execution.scheduler: provider_fair` interleaves eligible providers before taking the next canonical eligible task within a provider, preventing one large model list or slow provider from starving another. Scheduling does not change task identity or scientific parameters. Tasks are independently claimed through leases. Output completion order cannot change fingerprints, scoring, or report order.

With `execution.continue_until_complete: true`, `run` keeps advancing the frozen plan through generation, local scoring, ranking, and final reports until all expected tasks and final artifacts are complete, a user interrupt occurs, or safe progress is blocked. Temporary, safely retryable provider cooldowns are persisted and resumed automatically when due. Waiting does not consume an attempt; an actual resend still consumes the next of the task's maximum three requests. A provider outage, credential problem, or provider budget stop pauses the affected scope. `execution.continue_unaffected_providers: true` lets other selected providers continue; no task is rerouted to another provider/model.

The scheduler checks provider eligibility before claiming dispatch work. Rate/cooldown waits hold no worker slot, task lease for a new dispatch, or database transaction and allocate no attempt number. A response already received is still checkpointed and processed locally while that provider's new dispatches are paused.

Provider-local caps apply only to their provider and all cap amounts default to null (uncapped), including metered OpenRouter. PSNC's documented `non_billed` declaration supports evidenced zero monetary rates while preserving token, call, latency, error, and billing evidence. Always record costs and reservations; atomically check only non-null provider/campaign caps. Explicit zero is a zero spending ceiling, not a missing value. Exhausting an optional OpenRouter or campaign cap blocks further positive-cost reservations; evidenced non-billed PSNC requests reserve zero and remain eligible under other gates. Exceeding an expected estimate does not create an undeclared cap or stop condition. A global integrity/database failure or user stop halts all new dispatches. If unfinished work has no safe automatically scheduled continuation, record an incomplete paused/failed campaign and return a truthful summary; never report success by omitting that work.

`main.py` accepts worker/selection arguments but does not implement the worker pool; this module remains independently testable.

## Resume

Resume inspects durable evidence:

- No attempt record: begin next permitted attempt.
- Request persisted but not safely dispatched: resolve dispatch state.
- Ambiguous delivery: reconcile; do not blind retry.
- Raw response stored: continue extraction.
- Candidate stored: continue validation.
- Prediction stored: continue scoring.
- Score stored: mark/reconcile completion.

Completed tasks are never repeated. Hash/config mismatch blocks continuation into a different campaign. A resume command reconciles all selected providers and then runs the same continuation loop. Reconciliation releases a paused provider as part of that repair, because nothing else clears a pause and an uncleared one makes every later resume claim nothing. Provider/model list changes require a new frozen campaign. If only a provider subset is temporarily dispatched, all other planned tasks remain part of the completion requirement.

After all expected tasks are complete and scored, a deterministic reporting operation ranks every fully resolved configuration under accepted D-023 policy `mean-repetition-micro-close-f1-v1`: calculate micro Close F1 over the same 97 variables in each repetition, average the unrounded repetition values, rank descending, and assign exact ties a shared competition rank. All raw attempts, predictions, per-variable/component/repetition Exact/Close Precision, Recall, F1 and contributions, aggregates, and lower-ranked configurations remain queryable. Additional descriptive statistics and confidence intervals are deferred analyses and do not block workflow completion.

## Planned public functions

These planning signatures describe the decomposed responsibilities and the names
used while the component was designed. The implemented interface, including every
name and signature that differs, is recorded under *Implementation interface
(version 1)* at the end of this document.

### `expand_campaign(configuration, targets) -> ExperimentPlan`

- **Input:** One plan-ready resolved configuration containing the selected provider set and each provider's owned enabled models, plus the exact ordered 97-member evaluation population. Live authorization is not needed to build a plan.
- **Action:** Expand each selected provider's owned model grid across applicable reasoning profiles × prompts × shots × temperatures × temperature-specific repetitions × all targets, then take the union; create stable provider-bound configuration/run/task fingerprints and exact initial/max request counts by provider and in total. Supply immutable inputs to the separate estimator; planning never treats estimates as exact actual token/cost counts.
- **Output:** Immutable experiment plan containing every configuration, repetition run, variable task, canonical order, count breakdown, denominator, maximum request bounds, and plan hash.
- **Raises:** Empty/placeholder model list, incompatible capability/grid value, manifest/hash mismatch, wrong population count, duplicate fingerprint, unresolved planning input, or arithmetic overflow/inconsistency.
- **Side effects:** None; planning persistence is a repository operation and there are zero provider requests.
- **Determinism:** Input collection order cannot change canonical plan order, fingerprints, or counts.

### `estimate_campaign_cost(plan, prompt_artifacts, billing_evidence, assumptions) -> CostEstimate`

- **Input:** One immutable experiment plan; exact frozen prompt/schema/demo and correction-template artifacts; per-model tokenizer or documented token-estimation evidence; frozen provider/model price cards, explicit non-billed bases and exchange-rate provenance; and versioned assumptions about output/reasoning tokens and retry incidence under `pre-run-estimate-v1`.
- **Action:** Reconcile per-model/provider and combined initial/maximum-three-attempt call counts, estimate input/output/reasoning-token quantities without double-counting provider billing categories, include previous-response/error growth in correction prompts, and calculate exact-decimal monetary scenarios from frozen prices. Produce a labelled expected scenario/range and a conservative upper-bound scenario conditional on documented token/context/output/reasoning limits. Mark unsupported quantities and unbounded assumptions explicitly; never infer missing prices are zero or invent a reliable total.
- **Output:** Immutable plan-bound estimate with policy/hash, per-model/provider/total call-token-cost breakdowns, currency/price/FX provenance, assumptions and unresolved-input reasons, expected range, conditional bound status, and the warning that actual cost may exceed estimates. Disclosure and subsequent explicit live authorization are separate evidence records referring to this identity.
- **Raises:** Conflicting plan/prompt/billing identities, invalid negative prices/quantities, inconsistent token billing categories, invalid assumptions, or arithmetic inconsistency. Ordinary unresolved input produces a not-ready estimate with explicit reasons instead of a fabricated number.
- **Side effects:** None; no provider calls, paid calibration, secret reads, database writes, or authorization. Persisting and presenting the result belongs to command/repository services. A future live canary requires its own estimate and authorization.
- **Determinism:** Identical frozen inputs and assumptions yield identical estimate content/hash; actual usage is recorded later and never overwrites the pre-run estimate.

### `run_campaign(campaign_id, execution_options) -> ExecutionSummary`

- **Input:** Explicit planned campaign ID plus operational-only worker/concurrency/rate/batch/stop options that may narrow execution order but cannot alter scientific configuration or population.
- **Action:** Verify the frozen selection, billing evidence, disclosed estimate, separate live authorization and any optional caps; start registered workers, fairly claim eligible tasks across selected providers, dispatch `run_task` within independent provider limits, heartbeat leases, reconcile outcomes/cost, and automatically revisit safe cooldowns. Continue unaffected providers when a local scope pauses; only an explicitly configured monetary cap blocks further positive-cost reservations, while evidenced non-billed work stays eligible. Stop all dispatch on user interrupt or fatal shared integrity/database failure. Once every planned task is scored, run ranking and final reporting idempotently.
- **Output:** Execution summary containing per-provider and total counts by durable state, actual calls/usage/cost and billing basis, stop reason, outstanding/ambiguous work, cooldowns, lease health, final ranking/report identities when complete, and next safe command otherwise. A complete outcome requires every planned task to be terminal — `complete`, `operational_failed` or `ambiguous_delivery` under D-042 — and every final artifact to be produced, across every planned model. The named stop reasons are `tasks_complete`, `tasks_terminal`, `already_complete`, `providers_paused`, `no_claimable_work`, `requested_offline_checkpoint` and `blocked`; the first three are successful exits. `docs/unattended-execution.md` tabulates what each means for an operator. A campaign that stopped with terminal failures is finished rather than paused, because pausing it would invite a resume that can never make progress.
- **Raises:** Unknown/unplanned campaign, absent or mismatched estimate disclosure/live authorization, incompatible execution option, shared database/integrity failure, or unrecoverable worker-manager error; provider-local and task outcomes remain recorded in their affected scope rather than aborting healthy providers.
- **Side effects:** PostgreSQL workflow/evidence writes and authorized provider calls through task services only. It never changes scientific parameters or ranks partial results.
- **Resume behavior:** A rerun delegates to durable state and does not recreate completed work.

### `run_task(lease, services) -> TaskOutcome`

- **Input:** One current fenced task lease and injected prompt, attempt, validation, persistence, evaluation, budget, clock, and provider services bound to the frozen task identities.
- **Action:** Inspect the last durable checkpoint, perform only the next eligible local step or one numbered attempt, persist evidence before advancing, renew/release the lease correctly, and stop at valid prediction, terminal-invalid prediction, operational pause/failure, or completed score.
- **Output:** Typed task outcome with prior/next state, work performed, request count, prediction/evaluation IDs when created, retry eligibility, operational reason, and evidence references.
- **Raises:** Stale lease, hash/config drift, illegal state, attempt-four path, evidence conflict, or unrecoverable persistence invariant. Model-controlled invalid output is an ordinary outcome.
- **Side effects:** Durable PostgreSQL transitions; zero or one provider call per invocation depending on checkpoint. No hidden loop may create multiple calls.
- **Idempotency:** Re-entry from the same committed checkpoint either returns the existing result or advances once without duplicating verified work.

### `resume_campaign(campaign_id) -> ReconciliationReport`

- **Input:** Explicit campaign ID, current database time, lease/retry policy, per-provider and global budget/cooldown state, and optional non-scientific scope filter.
- **Action:** Verify campaign/artifact hashes and immutable selected-provider/model membership; compare planned versus stored runs/tasks/attempts/predictions/scores across every selected provider; inspect expired leases and ambiguous delivery; requeue only states whose next action is provably safe; leave complete work untouched. The command service passes the reconciled plan to `run_campaign` for continued execution.
- **Output:** Reconciliation report with per-provider and total expected/stored counts, verified checkpoints, reclaimed/requeued work, blocked ambiguities/conflicts, next eligible times, remaining attempts/cost, and post-reconciliation state totals.
- **Raises:** Campaign/config/artifact mismatch, corrupt evidence, impossible transition, unresolved migration problem, or database transaction failure.
- **Side effects:** PostgreSQL lease/state repair allowed by the frozen protocol; provider calls occur only later through normal task execution.
- **Idempotency:** Repeated reconciliation converges to the same state and creates no duplicate task, attempt, prediction, or score.

### `heartbeat(lease) -> LeaseStatus`

- **Input:** Active task lease ID, owner/fencing token, expected row version, current database time, and configured extension duration.
- **Action:** Atomically verify ownership/state/non-expiry policy and extend the expiry while recording minimal operational heartbeat evidence.
- **Output:** Lease status containing renewed expiry/row version or a typed lost/stale/not-active result.
- **Raises:** Database failure or malformed lease identity; ordinary lease loss is returned, not hidden.
- **Side effects:** PostgreSQL operational lease update only; no scientific evidence, task parameter, or provider request changes.
- **Concurrency:** A stale worker can never revive or mutate a lease reclaimed under a newer fencing token.

## Acceptance tests

- Exact expansion counts for mixed reasoning capabilities
- PSNC-only, OpenRouter-only, and both-provider plans with independently sized model lists
- Per-provider grid counts sum exactly; no model is called through another provider
- Provider-fair scheduling and separate concurrency/rate/cost gates
- A provider-local failure or cooldown permits healthy providers to finish
- Optional non-null provider/global cap exhaustion blocks positive-cost work while evidenced non-billed PSNC continues
- A shared database/integrity stop or user interrupt halts all new dispatches
- Non-billed zero rates remain evidenced; metered calls accept null caps and still record reservations/actual cost
- Null means uncapped, explicit zero blocks positive-cost dispatch, and estimate overrun alone does not stop execution
- Estimate includes model/provider totals, reasoning-token accounting, correction growth and all-three-attempt bounds with price/FX provenance
- Unknown estimation inputs remain explicit; no paid calibration or live dispatch before separate authorization
- Resume verifies original plan/estimate/authorization without authorizing changed selections or a live canary
- Run/resume continues through final scoring, ranking, and reporting without stopping after one provider
- Incomplete blocked provider work prevents campaign completion and is never skipped or rerouted
- Stable fingerprints under ordering/concurrency changes
- Two-worker claim race
- Graceful interrupt and budget pause
- All 97 targets planned once per resolved configuration
- Ranking covers every completed resolved configuration without filtering stored results
- Full crash matrix at every durable checkpoint
- No duplicate request after stored response
- Repeated resume is idempotent

## Implementation interface (version 1)

`expand_campaign(configuration, targets, ...)` lives in `planning.py` and
`estimate_campaign_cost(plan, prompt_artifacts, billing_evidence, assumptions)` in
`costing.py`; both are pure. The rest of this contract is `workflow.py`, whose functions
are `async` and take an injected `Services` record rather than loose option arguments:
`run_campaign(campaign_id, services, *, stop_after=None)` and `run_task(lease,
services)`. `finalize_campaign(campaign_id, services)` is the separate idempotent step
that ranks, exports and marks the campaign complete once every task is terminal;
finishing the tasks is not finishing the campaign. `build_observations(tasks)` projects
stored task evidence into the rows `reporting.py` consumes.

`resume_campaign(campaign_id)` does not exist as one function, and there is no
`ReconciliationReport` type. Its responsibilities are split three ways:
`Repository.reconcile(campaign_id)` performs the database-side reconciliation and
returns `campaign_id`, `repaired_tasks`, `ambiguous_tasks`, `released_providers` and the
campaign record; `cli.cmd_resume` re-prepares the frozen campaign and verifies that the
checkout's inputs still resolve to the same campaign ID; and continued execution is
ordinary `run_campaign`. `docs/architecture.md` section 2.1 records the consolidation of
the planned `workflow/{planner,runner,state,resume}.py` into `planning.py` plus
`workflow.py`.

`heartbeat(lease)` is `Repository.heartbeat(lease, lease_seconds=300)`, a persistence
operation rather than a workflow function; `workflow._heartbeat` is the private renewal
loop that calls it while a task is in flight.
