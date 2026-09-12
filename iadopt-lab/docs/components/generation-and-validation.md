# Generation, Extraction, Validation, and Retry Contract

## Responsibility

Manage the maximum-three-attempt lifecycle for one task after planning: persist request evidence, invoke the selected adapter, preserve raw output, extract one JSON object, validate it, construct correction prompts, and finalize an accepted prediction, a content-invalid empty prediction, or an explicit operational state.

This layer does not expand parameter grids, switch providers/models, alter sampling values between attempts, score predictions, or silently repair semantic content.

It also does not use gold data or evaluator similarity to judge answer quality. A schema-valid, cross-reference-valid but incomplete, incorrect, or all-empty response is accepted and scored without another attempt.

## Inputs

- One immutable planned task and remaining attempt budget
- Frozen rendered base prompt
- Provider adapter and model profile
- Task-bound provider identity, provider-local eligibility, and global budget/integrity gates
- Lexical JSON Schema and semantic-rule versions
- Previous attempt only when building attempts two or three

## Outputs

- One to three durable attempt records
- Transport, extraction, and validation events
- One accepted canonical prediction, one terminal-invalid empty prediction after three content-invalid model responses, or one operational failure/pause without a model-performance prediction
- Task state transitions and final failure reason

## Attempt behavior

Each actual provider request consumes one of three attempt numbers. Attempt one uses the base prompt. A validation failure may produce attempt two; attempt two’s failure may produce attempt three. Parameters, provider, model, demonstrations, and base prompt remain unchanged.

This rule applies separately to every task across all selected providers in a combined campaign. The scheduler may advance another provider's tasks during a local cooldown, but this task never switches provider/model. Waiting for rate capacity or a cooldown before dispatch consumes no attempt; no worker slot or database transaction is held for that wait.

A transport failure with no response can consume an attempt and schedule the unchanged base prompt when policy permits. Invalid model output produces a correction section containing the immediately previous raw output and complete errors.

## Extraction

The extractor first strictly parses the trimmed whole response. A top-level
object is accepted. A top-level JSON string may be unwrapped exactly once and
processed again; a second wrapper is invalid. A successfully parsed array,
number, Boolean, or null is rejected without mining an object from inside it.

If whole-response parsing fails, complete `json` or unlabelled Markdown fence
bodies are parsed. Exactly one object with no competing parsed JSON value is
accepted. If none parses, a quote-aware and escape-aware balanced-brace scanner
enumerates complete outermost object spans, which are then strictly parsed.
Exactly one may succeed. Multiple candidates are rejected as ambiguous.

Candidate offsets, hashes, parse diagnostics, selected method, wrapper status,
and safety-limit status are recorded. The extractor never selects the first of
several objects and never repairs syntax or inserts, removes, renames, or coerces
fields.

## Validation

The candidate must pass JSON parse, strict lexical schema, and semantic cross-field rules. Validation errors have stable codes and JSON Pointers. Canonicalization after validation may derive unscored system metadata labels and deterministic ordering but preserves the raw/extracted representations separately.

Semantic validation here means objective representation invariants such as a complete exclusive system shape and resolvable Constraint target. It is not expert grading or semantic-quality filtering.

The gate must reject exactly what the evaluator rejects. A prediction this component accepts is selected, stored and handed to `iadopt_eval`; if the evaluator then refuses it, the two disagree about durable content, and before D-047 that ended a campaign and deadlocked every resume, because the stored response was replayed into the same exception. The one rule the two did not share was whitespace: the JSON schema accepts any string, while the evaluator refuses a string that is non-empty but all whitespace. All five string-valued fields — `hasStatisticalModifier`, `hasProperty`, and the three entity fields — now fail semantic validation with code `whitespace_only_text`, so such an answer is ordinary invalid content: retried, and scored as an empty prediction if it never improves.

Any future disagreement is still contained. Under D-047 a data-shaped failure raised by the evaluator is recorded as `scorer_rejected_validated_prediction` against that task alone and the campaign continues; only an infrastructure exception stops the run.

## Configuration keys consumed

- `generation.output_schema`
- `generation.max_generation_attempts_per_task`
- `generation.provider_sdk_auto_retries`
- `generation.correction_attempt.*`
- `generation.retryable_results`
- `evaluation.terminal_invalid_prediction`
- `database.evidence_policy`

## Terminal behavior

The first valid attempt is selected and later attempts do not exist. When three delivered model responses are content-invalid, a six-field empty prediction is created under a versioned failure policy and sent to evaluation. An exhausted operational/provider failure is retained without a model-performance prediction and blocks campaign promotion; it is never converted into an empty model answer. In both cases, the target remains visible.

A task/provider-local pause reports its affected scope and next eligible time when known so that the workflow can continue healthy selected providers. Stored responses still proceed through local processing while new calls for their provider are paused. A blocked task remains part of campaign completeness; completing the other provider never hides or substitutes it.

## Planned public functions

These planning signatures describe the decomposed responsibilities and the names
used while the component was designed. *Boundary implementation interfaces (version 1)* below records the implemented
interface, including every name and signature that differs.

### `extract_json(raw_text) -> ExtractionResult`

- **Input:** The exact assistant-visible response text plus frozen extractor version and explicit byte/depth/candidate safety limits.
- **Action:** Apply the documented ordered strategies: whole-text parse, one-level JSON-string unwrap, complete fence candidates, then quote/escape-aware outermost balanced-object spans. Reject parsed non-objects and ambiguity; never repair or coerce JSON.
- **Output:** A typed result containing success/failure code, selected parsed object when unique, exact candidate text/byte offsets/hash, strategy, wrapper status, all candidate diagnostics, and safety-limit evidence.
- **Raises:** No raw parser exception escapes. Only programmer-level contract violations raise typed errors; malformed/model-controlled content is returned as data.
- **Side effects:** None.
- **Determinism:** Identical UTF-8 text, extractor version, and limits yield byte-identical canonical evidence.

### `validate_candidate(candidate, contract) -> ValidationResult`

- **Input:** One extracted JSON object preserving its source representation plus exact lexical-schema bytes/hash, JSON Schema engine/version, and semantic cross-field policy/version.
- **Action:** Validate schema first, then run semantic checks only when shape permits; order errors by stable stage/path/code rules and create sanitized summaries without consulting gold quality or scorer similarity.
- **Output:** Typed validity result, schema/semantic engine identities, complete ordered error records with JSON Pointers, candidate hash, and stage-level evidence.
- **Raises:** Schema artifact/hash/engine incompatibility or internal contract corruption; ordinary validation failures are returned, not raised.
- **Side effects:** None; the input object is not changed and no correction is attempted.
- **Determinism:** Identical candidate and validator artifacts produce identical error ordering and result hash.

### `canonicalize_prediction(valid_candidate, policy) -> CanonicalPrediction`

- **Input:** A schema- and semantic-valid candidate plus frozen canonicalization, lexical ordering, system-display-label, and terminal-provenance policies.
- **Action:** Preserve scoreable lexical values, derive stable unscored system metadata, resolve accepted whole-system Constraint aliases, sort only collections declared set-like for canonical storage, and calculate canonical bytes/hash while retaining links to raw/extracted forms.
- **Output:** Immutable six-field evaluator input, derived metadata/trace, representation links, policy identities, canonical bytes, and hash.
- **Raises:** Post-validation invariant violation, ambiguous target alias, duplicate/unsupported value, or canonicalization/hash failure. It never invents a scored value.
- **Side effects:** None.
- **Determinism:** Identical valid input and policy produce identical canonical output/hash.

### `execute_attempt(task, attempt_number, services) -> AttemptOutcome`

- **Input:** One valid task lease, requested attempt number `1..3`, frozen task/base-or-correction prompt, its owning provider adapter, evidence repository, extractor/validator/canonicalizer services, and provider-local plus global budget/delivery guards.
- **Action:** Verify lease and next attempt number, commit write-ahead request evidence, invoke the adapter at most once, commit response/error evidence before local processing, then extract, validate, and canonicalize as far as durable evidence permits.
- **Output:** Typed outcome identifying delivered/ambiguous/not-dispatched state; every durable attempt/transport/response/extraction/validation record; optional valid canonical prediction; and the facts needed by `advance_task`.
- **Raises:** Stale lease, attempt conflict/fourth attempt, evidence/hash conflict, persistence failure, or internal invariant error. Provider/model-controlled failures are recorded in the outcome.
- **Side effects:** PostgreSQL evidence writes and at most one provider request. It performs no retry, scoring, ranking, or parameter change.
- **Idempotency:** Re-entry resumes from verified durable evidence and must not repeat a stored response or create a duplicate logical request.

### `advance_task(task, outcome) -> TaskDecision`

- **Input:** Current versioned task state, one persisted `AttemptOutcome`, remaining total-request allowance, and the frozen retry/failure policy.
- **Action:** Apply the state table exactly once: select a valid prediction; schedule only the next numbered eligible attempt; create explicit-empty prediction only after three delivered content-invalid responses; or pause/fail operational/ambiguous cases without a model-performance prediction.
- **Output:** Typed decision with next state, reason/error code, affected task/model/provider/global scope, next eligible time when known, next attempt number and correction parent when applicable, prediction action, publication impact, and transition evidence.
- **Raises:** Illegal current state, outcome/task ownership mismatch, stale row version, non-consecutive attempt, attempt-four request, or evidence conflict.
- **Side effects:** No provider call. The repository applies the returned decision transactionally; a pure decision core is separately testable.
- **Determinism:** Identical valid state/outcome/policy produce the same decision.

## Acceptance tests

## Boundary implementation interfaces (version 1)

`execute_attempt` and `advance_task` are not defined in this component. Only the
workflow layer may schedule another numbered attempt, so both were implemented
there: `workflow.run_task(lease, services)` performs one attempt against a fenced
lease, and the private `workflow._advance_task(lease, services)` applies the state
table. There is no `AttemptOutcome` or `TaskDecision` type, and no pure decision
core separable from the repository — `_advance_task` decides and persists in the
same async function. `docs/architecture.md` section 2.1 records the folding of the
planned `generation/attempts.py` into `workflow.py`.

`extract_json(raw_text)` returns `ExtractionResult` with `success`, `candidate`,
`errors`, strategy, candidate evidence, and `to_dict()`. Limits are 1 MiB UTF-8,
64 nested containers, 64 fences, and 64 balanced candidates. Strict JSON rejects
duplicate object keys and non-finite numbers. Byte offsets identify either the
original response or the once-decoded wrapper string. Timing is intentionally an
operational envelope supplied by the caller, not part of deterministic evidence.

`validate_prediction(candidate, schema_bytes=None)` (also `validate_candidate`)
returns `ValidationResult` with `valid`, ordered `errors`, `canonical_prediction`,
schema/candidate hashes and `to_dict()`. Invalid model content is data, not an
exception. `canonicalize_prediction(candidate)` is available separately and
rejects invalid input. Canonicalization derives system labels and sorts symmetric
parts but preserves Constraint order. `empty_prediction()` creates the six-field
empty shape; workflow provenance distinguishes its reason from valid empty output.

Target matching uses `NFC(text).strip().casefold()` and resolves to one distinct
original lexical label. Duplicate symmetric parts under that key are invalid.
Equivalent lexical labels reused by different component roles remain valid;
aliases that resolve to distinct preserved labels are rejected as ambiguous.

- All extraction forms and malformed inputs
- Braces/escapes inside JSON strings
- Multiple-object ambiguity
- Full path-addressed validation errors
- Invalid then valid correction
- Structurally valid all-empty response selected without quality retry
- Exactly three invalid attempts
- Three safe transient provider failures produce an operational failure with no model-performance score
- Every request counts, with no fourth call
- Parameters identical across attempts
- Combined campaigns preserve each task's provider and model through every correction/resume
- Provider cooldowns consume no attempt and do not prevent another provider's eligible work
- Raw/extracted/canonical representations distinct
- Stable system IDs derived without changing scoreable values
