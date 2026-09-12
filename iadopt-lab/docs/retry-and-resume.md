# Retry, Interruption, and Resume Design

## 1. Purpose and authority

This document defines the attempt controller, correction protocol, durable checkpoints, worker recovery, and ambiguous-delivery behavior for I-ADOPT Lab. It refines sections 11 and 13 of `TECHNICAL_SPECIFICATION.md`; that specification remains authoritative if the documents conflict.

The governing rule is exact and applies independently to every fully resolved variable task:

> A task may cause at most three outbound provider requests in total. Attempt 1 is the initial generation. Attempts 2 and 3 are possible follow-up attempts. There is no fourth request.

The three-attempt budget is not shared across variables and is not reset after an interruption. It belongs to one task fingerprint, which fixes the variable, selected provider, model, reasoning profile/native mapping, prompt and demonstrations, schema, shot count, temperature, sampling values, repetition, retry protocol, corpus evidence, and relevant code/environment versions.

Provider SDK retries, HTTP-client retries that resend a request, adapter recursion, and nested generation loops are disabled. Every actual provider request must have one durable numbered attempt.

This is a behavioral design, not executable code.

## 2. Responsibility and boundaries

The retry/resume subsystem is responsible for:

- Advancing one task from its earliest incomplete durable stage.
- Committing complete request evidence before dispatch.
- Ensuring one adapter invocation can produce at most one outbound request.
- Preserving full raw response/error evidence before extraction or validation.
- Deciding whether a typed outcome is successful, retryable, terminal, paused, or campaign-failing.
- Building correction attempts from only the immediately preceding invalid response and its complete errors.
- Enforcing the task's remaining three-attempt budget.
- Reconciling expired leases and ambiguous delivery without blind duplicate calls.
- Continuing extraction, validation, prediction, scoring, and reporting without regenerating when durable evidence already exists.

It does not:

- Change the provider, model, reasoning mode, prompt family, examples, temperature, repetition, output limit, schema, or scorer between attempts.
- Repair, fill, rename, or coerce model-generated semantic fields.
- Hide a provider call inside an adapter or SDK.
- Drop a variable after invalid output.
- Treat a filesystem log as authoritative evidence.

## 3. Inputs

### 3.1 Task input

The controller receives one immutable planned task containing:

- Task, resolved-run, campaign, variable, and fingerprint identities.
- Selected provider and exact model ID/revision.
- Applicable normalized reasoning profile and exact native request fields.
- Frozen base prompt/messages and their hashes.
- Prompt template, lexical schema, semantic validator, demonstrations/order, and hashes.
- Temperature, supported sampling fields, output limit, repetition, and seed when supported.
- Current task/attempt states, durable evidence identities, and remaining attempt budget.
- Current lease owner, fencing row version, heartbeat, and expiry.
- Provider adapter, timeout, rate/optional-cap gates, required estimate/disclosure/authorization evidence, and idempotency/lookup capabilities.

### 3.2 Previous-attempt input

An output-correction attempt requires exactly the immediately preceding attempt's:

- Untouched assistant response text.
- Extraction diagnostics.
- Complete ordered lexical-schema and semantic-validation errors.
- Attempt number and content hashes.

Another variable's output or an older attempt's errors must never enter the correction section.

When a retry follows a classified transport failure with no model output, the next request uses the unchanged base prompt and records the transport reason. It does not fabricate a correction message.

## 4. Outputs

For each task, the subsystem produces:

- Zero to three durable attempt records, each representing at most one outbound provider request.
- Complete rendered messages, sanitized request body, prompt/request hashes, native reasoning fields, and parameters for every attempt.
- Append-only transport, extraction, validation, retry-decision, lease, and task-transition evidence.
- Untouched raw provider response text/envelope and complete sanitized errors when available.
- One accepted canonical prediction from the first fully valid attempt; or
- After three delivered content-invalid model responses, one versioned explicit-empty six-field prediction with terminal-invalid status and final error bundle; or
- A paused/failed task when safe automatic continuation is impossible, campaign-level configuration is invalid, a configured optional cap is exhausted, or required estimate/disclosure/live authorization is absent.
- A reconciliation report showing verified work, remaining work, attempt budgets, expired leases, ambiguous deliveries, and exceptions.

## 5. Execution grain and attempt identity

The experiment hierarchy is campaign, resolved run, task, then attempt. Retries occur only at the task level.

A campaign may select PSNC only, OpenRouter only, or both through `campaign.providers`. Its plan is the union of each selected provider's owned model grid. Every resolved run and task still has exactly one provider/model identity. The initial three models from each provider are listed with exact IDs in `docs/model-catalog.md`; model counts are not hard-coded. Changes before freezing are supported; changes to the selection after freezing require a new campaign and new campaign-scoped tasks.

For example, the same variable with the same model but a different reasoning profile is a different task and receives its own three-attempt budget. The same is true for a different prompt, shot count, temperature, or repetition. Conversely, restarting a worker or the whole application does not create a new task or reset its budget.

Attempt number is a child of task identity. It is not a scientific grid dimension. Request and response hashes, timestamps, delivery state, and retry decisions distinguish the evidence produced within the task.

## 6. Attempt protocol

| Attempt | Request content | Maximum outbound requests | Possible next step |
|---:|---|---:|---|
| 1 | Frozen base prompt | 1 | Select valid prediction, schedule attempt 2, pause, or fail campaign/task |
| 2 | Base prompt plus attempt-1 response and errors, or unchanged base after an eligible transport failure | 1 | Select valid prediction, schedule attempt 3, pause, or fail campaign/task |
| 3 | Base prompt plus attempt-2 response and errors, or unchanged base after an eligible transport failure | 1 | Select valid prediction, create terminal-invalid empty prediction for content-invalid exhaustion, or preserve an operational pause/failure |

The first valid attempt is selected. No later attempt is created. Attempt three is the hard request boundary. Failure causes remain typed: only three delivered content-invalid model responses create a scored empty prediction; operational exhaustion remains an unscored operational state.

### 6.1 Before dispatch

The scheduler checks the task provider's rate, cooldown, and billing/cost gates plus shared campaign gates before assigning dispatch work. Live eligibility includes a disclosed valid pre-run estimate and separate explicit authorization. An ineligible provider waits without consuming a worker slot, holding a database transaction, or allocating an attempt number. Once eligible, the worker must hold a valid lease and atomically record its cost reservation, checking every configured non-null provider/global cap. Null caps mean uncapped, not missing billing evidence. PostgreSQL then commits:

- Task and attempt number.
- Complete rendered message sequence and human-readable prompt.
- Sanitized exact provider request body.
- Provider/model identity and exact normalized/native reasoning configuration.
- All sampling values and output limit.
- Request/prompt hashes and deterministic idempotency key.
- Correction-parent identity when applicable.
- A state indicating that request evidence is durable but dispatch is not yet confirmed.

If this commit fails, no provider request may be sent.

### 6.2 Dispatch and receipt

The adapter executes at most one non-streaming provider request and returns a typed result. It performs no retry. Dispatch, provider/HTTP status, timeout or connection outcome, delivery certainty, receipt, and timing become append-only transport events.

When a response is received, its untouched body/envelope is committed before JSON extraction starts. The system must not send another request merely because parsing or validation has not yet run.

### 6.3 Extraction and validation

Resume-safe processing applies the frozen deterministic stages:

1. Run the frozen extraction protocol: whole-response object; one optional JSON-string unwrap; one unambiguous parsed JSON/unlabelled fence; then one unambiguous strictly parsed object found by quote-aware balanced scanning.
2. Parse JSON without adding or coercing fields.
3. Apply the strict six-field lexical JSON Schema.
4. Apply semantic cross-field validation.
5. Canonicalize a valid candidate, deriving only deterministic unscored system metadata.

Each stage writes its version, input/output hashes, complete diagnostics, and pass/fail event. Repeating a stage with identical evidence returns or verifies the existing result.

### 6.4 Selection or next decision

A valid canonical prediction is selected transactionally and the task advances to scoring. An invalid result is classified against the frozen retry policy. If eligible and an attempt remains, the task becomes retry-pending. Otherwise it becomes paused, failed, or terminal-invalid according to the rules below.

## 7. Retry eligibility

A new numbered request is allowed only when all of the following are true:

1. The preceding attempt is durably recorded.
2. Its delivery/result state has been reconciled sufficiently to allow another request.
3. Its outcome has a registered retryable classification.
4. The task has fewer than three outbound attempts.
5. The task's owning provider remains within credential, capability, rate, and provider-cost gates, and shared campaign cost/integrity gates permit dispatch.
6. The worker owns a valid lease and row version.

The documented retryable outcomes are:

- Empty response.
- HTML response instead of the expected model result.
- No unambiguous JSON object.
- JSON parse failure.
- Lexical-schema failure.
- Semantic-validation failure.
- A specifically classified transient provider failure for which another outbound request is safe and permitted.

A schema-valid but scientifically poor, incomplete, or all-empty decomposition is not a retryable outcome. It is selected and scored. The validator and attempt controller do not inspect gold values or evaluator similarity when deciding whether to retry.

Output extraction, lexical-schema, and semantic-validation failures count as content-invalid model responses. Provider, authentication, infrastructure, budget, and ambiguous-delivery outcomes remain operational even when a safe retry consumes another numbered attempt. If operational outcomes prevent the task from receiving three content-invalid responses before the request budget is exhausted, the task does not receive an empty model prediction.

The following are not generation retries:

- Missing or invalid credential.
- Authentication or permission rejection.
- Unknown provider/model.
- Placeholder or unsupported capability/reasoning mapping.
- Invalid campaign configuration.
- Schema, prompt, corpus, scorer, or task fingerprint mismatch.
- Exhausted configured optional cost cap, absent valid price/billing evidence, or missing/undisclosed/plan-mismatched pre-run estimate or explicit live authorization.
- Unsupported PostgreSQL/migration version.
- Unresolved ambiguous delivery.

These outcomes pause or fail the appropriate scope. Repeatedly sending the same invalid request is prohibited.

Every selected provider must pass initial preflight before the first live request. After execution begins, most failures are scoped to the task, not the provider. Under D-046 a provider is paused only by an outcome saying the deployment cannot serve requests at all: a non-transient HTTP status (`provider_error`), a gateway HTML page instead of a completion envelope (`html_response`), an HTTP 200 carrying an error object and no answer (`provider_error_envelope`), a body that is not the provider's JSON envelope (`unparsable_envelope`), or an envelope whose completion structure is missing (`invalid_envelope`). Everything else fails only its own task.

A classified transient failure — a rate limit or a 5xx — sets a provider *cooldown* and returns the task to the queue or to retry-pending; the runner revisits it automatically when due. A task that exhausts its three attempts fails alone. A truncated answer (`output_truncated`) is a property of one prompt and model and likewise fails only its task. Stating the rule as a list of exceptions instead is what let one truncation pause a provider and strand 16,167 queued tasks; the rule is now a positive list.

Exhausting an optional provider or global monetary cap releases the affected task as `paused_budget` and blocks further paid reservations, while evidenced non-billed PSNC requests reserve zero and can continue; it does not pause the provider. A shared database/integrity failure or user interrupt halts all new dispatches. Failure isolation never changes provider/model identity or routes a failed PSNC task to OpenRouter, or vice versa.

An explicit `non_billed` PSNC profile uses zero financial rates backed by the owner's recorded billing statement; it still passes rate, context, request-count, and evidence gates. Metered OpenRouter work requires price-card evidence but not a spending cap. Under configuration `2.1`, all default cap amounts are null, including PSNC: null means no cap, zero means a genuine zero ceiling, and negative amounts are invalid. Accounting and reservations remain durable and atomic in uncapped mode. Unknown or missing cost evidence is not a free-service declaration.

`require_pre_run_estimate: true` and `estimate_policy_version: pre-run-estimate-v1` require a hashed estimate bound to the resolved plan and dated price/FX evidence, with per-model/provider/combined planned and maximum-three-attempt call counts, prompt/output/reasoning token scenarios, and correction-prompt bounds. The estimate must be disclosed before separate explicit live authorization. Changing a resolved campaign or price/FX basis requires a replacement estimate/disclosure; resuming the same frozen plan retains its evidence. An estimate is informative and is never silently enforced as a monetary cap, so an authorized uncapped campaign does not pause merely because actual cost exceeds the estimate.

## 8. Correction-prompt contract

Attempts two and three preserve every scientific setting and the frozen base prompt. For an invalid model response, a versioned correction section adds only:

- The immediately preceding attempt number.
- The exact previous raw assistant text.
- The ordered extraction/schema/semantic errors, including stable codes and JSON Pointers where available.
- An instruction to return one corrected JSON object matching the existing schema.

The correction renderer records the full resulting prompt/messages and hashes. It must not summarize away an error, silently fix the candidate, expose another target's data, or add a new demonstration.

Because correction text necessarily changes the request, each correction has its own rendered-prompt and request hashes while remaining a child of the same task/scientific condition.

## 9. Terminal-invalid content behavior

When three delivered model responses are content-invalid and the third exhausts the request budget, the controller atomically records:

- Task status `terminal_invalid`.
- All three attempt and transport records.
- The final extraction/validation error bundle.
- The failure-policy version.
- One explicit empty prediction containing all six required fields with their contract-defined empty values.

The evaluator scores this explicit-empty prediction. The variable remains in all applicable denominators; it is never omitted because generation failed.

An operationally exhausted task instead retains every attempt and error, receives no model-performance prediction or score, and blocks publication promotion. The campaign cannot hide it by reducing the denominator. Resolution requires evidence recovery or an explicitly approved new execution/campaign policy; it does not create attempt four.

System identifier labels, including deterministic symmetric and asymmetric metadata labels, remain unscored in successful predictions. This rule does not change during retry.

## 10. Durable state and checkpoints

Task stages are equivalent to planned, queued, leased, generating, response stored, extracting, validating, retry pending, prediction ready, scoring, and complete, with explicit paused and final-failure states. Every transition has an append-only event.

The durable checkpoint determines resume behavior, not an in-memory worker state:

| Last verified checkpoint | Resume action |
|---|---|
| Task planned, no attempt | Claim task and start attempt 1 |
| Attempt request evidence stored; non-dispatch is certain | Dispatch that recorded attempt once |
| Dispatch may have occurred; no durable response | Reconcile ambiguous delivery before any new request |
| Raw response stored | Continue extraction; do not call provider |
| Extraction stored | Continue required validation; do not re-extract unless integrity verification fails |
| Validation failure stored and retry allowed | Build the next numbered request from durable preceding evidence |
| Valid canonical prediction stored | Continue scoring; do not regenerate |
| Evaluation item stored | Reconcile/complete aggregates without regenerating or rescoring the item unnecessarily |
| Task complete | Verify and skip |

Configuration, corpus, prompt, schema, scorer, model/reasoning mapping, or evidence-hash mismatch blocks continuation into the existing campaign.

## 11. Worker leases and concurrency

Workers claim tasks in short PostgreSQL transactions using skip-locked row-claiming semantics. A claim records worker identity, lease expiry, heartbeat, and fencing row version. No transaction remains open during provider I/O.

A heartbeat may extend only a matching active lease. An expired lease allows another worker to reconcile the task, but it does not imply that an in-flight provider request never occurred. The new worker inspects attempt and transport evidence before choosing a stage.

A stale worker cannot commit over a newer owner because every transition checks the fencing version. If the stale worker returns with a provider response, the system attempts an idempotent evidence commit against the existing attempt; it never discards a unique response silently or starts a replacement call.

Separate provider-level semaphores, rate limits, cooldowns, and cost reservations bound concurrent calls within the global worker limit. A provider-fair scheduler revisits each eligible selected provider without allowing one provider's queued work or cooldown to occupy all worker slots. Completion order cannot alter fingerprints, selected predictions, scoring, or deterministic report order.

Temporary, safely retryable cooldowns store their reason, affected scope, and next eligible time. The active runner resumes eligible work automatically when due, preserving the next allowed attempt number. Waiting creates no provider request or attempt. Exhausting three actual requests remains an operational pause/failure where applicable; elapsed time does not grant a fourth request.

## 12. Crash and interruption matrix

| Interruption point | Safe recovery behavior |
|---|---|
| Before task/attempt insert | Nothing was callable; normal planning/claim resumes idempotently |
| After attempt evidence commit, before dispatch | Dispatch only if transport evidence establishes that it was not sent |
| During dispatch or after send, before durable response | Mark/reconcile possible ambiguous delivery; do not blind retry |
| After response receipt, while PostgreSQL is temporarily unavailable | Keep persisting the same response; do not parse or make another call |
| Process dies after receipt but before response commit | Treat the recorded dispatch as ambiguous because the response cannot be reconstructed locally |
| After raw response commit, before extraction | Re-run/continue deterministic extraction from stored raw evidence |
| After extraction commit, before validation | Validate the stored candidate |
| After failed validation, before retry task transition | Reconcile the stored decision; create at most the next numbered attempt |
| After valid prediction commit, before scoring | Score the existing prediction |
| During scoring | Resume by evaluation fingerprint and item-level uniqueness |
| During aggregation/export | Regenerate from immutable database facts |
| User interrupt | Stop new claims; allow safe checkpointing; release or let leases expire; record a paused exit |
| Provider cooldown or local service/credential failure | Persist the affected scope and next eligible time when known; continue eligible work at other selected providers |
| Configured optional provider cap boundary | Stop that provider's positive-cost dispatches, preserve evidence, record `paused_budget`, and continue other eligible providers |
| Configured optional global monetary cap boundary | Stop additional paid reservations; evidenced non-billed PSNC remains eligible and all received evidence is preserved |
| Disclosed estimate exceeded with no configured cap | Continue otherwise eligible authorized work and retain actual-cost evidence; an estimate is not a stop threshold |
| Shared integrity/database failure | Stop all new dispatches, preserve received evidence, and record the global pause/failure |

Resume is itself idempotent. Running it repeatedly without new evidence produces no provider call and no duplicate prediction, score, or task transition.

## 13. Ambiguous delivery

Ambiguous delivery means a request may have been accepted by the provider but no response was durably recorded. Common causes include connection loss after upload, timeout after provider acceptance, process termination during the call, or loss of PostgreSQL connectivity after receipt followed by process failure.

No client can prove exactly-once provider execution in this situation unless the provider supports a reliable idempotency or request-lookup contract. I-ADOPT Lab therefore follows this order:

1. Preserve the original attempt number, exact request, idempotency key, dispatch evidence, and uncertainty classification.
2. Use provider request lookup or idempotency reconciliation when that adapter has a verified mechanism.
3. If the original response is recovered, store it under the original attempt and continue locally.
4. If the provider definitively confirms non-acceptance, apply the registered retry policy; any replacement request is a new numbered attempt.
5. If acceptance cannot be resolved safely, pause the task as ambiguous instead of blindly issuing another generation, including when the provider is non-billed.

An approved replacement always consumes the next remaining attempt. Ambiguity never creates attempt 4 and never resets the budget. A paused ambiguity is an explicit reproducibility fact, not a terminal-empty prediction unless a separately versioned failure policy later says so.

## 14. Graceful stop and campaign resume

On a user interrupt, shared fatal failure, or controlled shutdown, the runner:

1. Stops claiming new tasks.
2. Stops dispatching new provider requests.
3. Lets safely in-flight work reach the next durable checkpoint when possible.
4. Stores response/error evidence already received.
5. Records worker/task state and releases leases or allows them to expire.
6. Exits with a status that distinguishes pause from failure.

A provider-local optional-cap or operational pause applies these checkpointing rules to the affected scope while other providers continue. A configured global monetary cap stops further paid reservations while evidenced non-billed PSNC requests remain eligible. Already received responses may still be extracted, validated, and scored locally. A temporary known cooldown is revisited automatically; a missing credential, unresolved delivery, or exhausted request allowance is not resolved by blind waiting or extra calls.

Campaign resume first produces a reconciliation report containing expected/stored counts per provider and in total, terminal states, verified responses/predictions/scores, expired leases, ambiguous deliveries, remaining attempt allowances, provider/global spent and reserved cost, optional caps, billing basis, estimate/disclosure/live-authorization validity, cooldowns/next eligible times, and orphan/integrity exceptions. Only then are eligible tasks queued and the same continued-execution loop runs across all selected providers.

Reconciliation also releases any paused provider, recording a `released_by_reconcile` provider event and returning the released names. Nothing else clears a pause, and a queued task is only handed out when its provider is ready or past its cooldown, so a pause left in place made every subsequent resume claim nothing and report `no_claimable_work` — permanently. Resuming *is* the explicit act of retrying, so the pause is cleared and its cause is allowed to reassert itself rather than being assumed to persist. A run that finds every provider paused or failed stops with the named reason `providers_paused`, so the next resume recovers without operator intervention.

The frozen evaluation-population and campaign rules remain in force after resume. A resume operation cannot change the 97 target identities, select a new provider/model list, or modify scientific parameters. Such a change creates a new campaign.

With `execution.continue_until_complete: true`, normal `run` and `resume` progress through all selected providers, local scoring, ranking, and final reports. A diagnostic dispatch filter does not remove omitted work from campaign completeness. If unresolved work has no safe automatic continuation, record the precise incomplete state and exit with a paused/failed summary.

Success requires every planned task to be **terminal** and every required final artifact to be produced. Under D-042 terminal means `complete`, `operational_failed` or `ambiguous_delivery`: a task that failed operationally can never reach `complete`, so waiting for a wholly complete population let one failure anywhere deny results for the entire campaign. A run whose population is entirely terminal stops with `tasks_terminal`, finalizes, and exits zero. Finalization still refuses to proceed while any task can still advance.

Terminal failures are reported, never silently dropped and never scored as model quality. A configuration missing any member of its population fails `ranking.require_complete_population` and is ranked nowhere, carrying an explicit not-rankable reason into the report. Completion at one provider never hides unfinished work at another.

## 15. Failure outputs and observability

The subsystem exposes typed outcomes for success, queued, retry pending, terminal invalid, paused budget, paused configuration, ambiguous delivery, non-retryable provider error, stale lease, evidence conflict, integrity/hash mismatch, and internal failure.

Two operational failures are recorded against the task rather than the run. `attempt_budget_exhausted_with_operational_error` is a task whose three attempts were spent on operational failures, so it never received three content-invalid responses and gets no empty model prediction. `scorer_rejected_validated_prediction` is a prediction the validator accepted and the evaluator then refused: under D-047 a data-shaped scoring failure (`ValueError`, `TypeError`, `KeyError`, `ArithmeticError`) fails that task and the campaign continues, because scoring is a pure computation over one gold and one prediction and such a failure says something about that pair. An infrastructure exception during scoring still propagates and stops the run. The validator is expected to reject exactly what the evaluator rejects, so reaching this outcome records a disagreement between them that is a defect to be fixed — at the cost of one task rather than a campaign.

Operational logs contain task/attempt IDs, hashes, state, timing, and sanitized summaries. PostgreSQL contains the full rendered prompt, sanitized request, raw response, and complete errors. Logs must not become a second incomplete evidence system and must not expose credentials.

For every attempted call, operators can answer:

- Which exact task and scientific configuration caused it?
- Which provider/model/reasoning controls were sent?
- Which attempt number consumed the call?
- What complete prompt and sanitized request were used?
- Was dispatch and response receipt certain?
- What exact raw content or transport error returned?
- Which extraction and validation errors occurred?
- Why was another request allowed or prohibited?
- What stage will resume execute next?

## 16. Acceptance criteria

- Mixed reasoning-capable and incapable models expand into the documented task identities.
- PSNC-only, OpenRouter-only, and combined plans preserve provider-owned model identities through every retry and restart.
- Every task can invoke the provider at most three times across any number of process restarts.
- Attempt 1 uses the base prompt; attempts 2 and 3 use only the permitted immediately preceding evidence.
- Provider SDK and adapters are proven to make no hidden retry.
- The scientific/provider parameters are byte-equivalent across attempts except for the versioned correction content and attempt/idempotency identity.
- Raw responses and complete errors are durable before downstream processing.
- Invalid-then-valid and three-invalid scenarios select the correct prediction and attempt count.
- Operational exhaustion remains distinct from content-invalid exhaustion and is never scored as an empty model response.
- A transport failure consumes an attempt once dispatch begins; an eligible next request uses the next number.
- A fourth attempt cannot be planned, inserted, or dispatched.
- Expired leases and two-worker races do not duplicate a confirmed provider request.
- Every crash checkpoint resumes from the earliest incomplete durable stage.
- A stored response prevents a replacement call.
- A stored prediction prevents regeneration.
- Ambiguous delivery is reconciled or paused, never blindly resent.
- Repeated resume and reconciliation operations are idempotent.
- Provider-local failures, caps, and cooldowns leave healthy selected providers eligible.
- Cooldown/rate waits consume no attempt or worker slot, and active execution resumes safe work when due.
- Initial preflight must pass for all selected providers; runtime isolation cannot skip an unready provider at startup.
- Configured optional provider/global monetary cap exhaustion blocks paid work while evidenced non-billed PSNC remains eligible.
- Null caps permit otherwise eligible authorized billed work with complete atomic accounting; exceeding a disclosed estimate is not a hidden cap failure.
- Missing/undisclosed/plan-mismatched estimates or missing explicit live authorization block live dispatch before allocating an attempt number.
- Shared database/integrity failure or user interruption halts all new dispatches.
- Non-billed zero-cost evidence remains distinct from absent price or unknown cost.
- Completion requires every selected provider's work plus scoring, ranking, and final reports.
- Terminal-invalid variables remain present and are evaluated through the explicit-empty policy.
