# Verification and Test Plan

## 1. Objective

The test program must prove that I-ADOPT Lab implements the documented scientific protocol, preserves complete evidence, never exceeds three outbound requests per task, resumes safely, and regenerates every score and report. Passing a few happy-path examples is not sufficient.

No normal test performs a live OpenRouter or PSNC call, including no-charge access. Provider behavior is tested with pure payload tests, recorded sanitized fixtures, and local mock HTTP servers. A live canary is a separate, explicitly authorized activity.

## 2. Test layers

| Layer | Purpose | External dependencies |
|---|---|---|
| Documentation/static | Ensure contracts, types, schemas, links, formatting, and secret rules remain synchronized | None |
| Unit | Prove deterministic functions and edge cases | None |
| Property-based | Explore parser, canonicalization, fingerprint, and state-machine invariants | None |
| Golden regression | Lock preserved January scorer behavior and every explicitly approved correction | Frozen local embedding artifact |
| Provider contract | Verify exact PSNC/OpenRouter mapping and one-request boundary | Mock server/recorded fixtures only |
| PostgreSQL integration | Prove migrations, constraints, transactions, leases, and evidence persistence | PostgreSQL 16 test instance |
| End-to-end dry run | Exercise source-to-report flow and recovery | PostgreSQL 16 + deterministic mock provider |
| Live canary | Verify current provider/model wire capabilities and accounting | Disclosed cost estimate, explicit authorization, and bounded call plan; monetary caps optional |

## 3. Traceability requirement

Each normative requirement in `TECHNICAL_SPECIFICATION.md`, `DECISIONS.md`, `docs/components/`, `docs/database.md`, and `docs/retry-and-resume.md` must map to at least one test ID. The implementation phase will maintain a machine-checkable traceability table with:

```text
requirement ID
source document and heading
test ID/path
test layer
expected evidence
last pass environment
```

A protocol-changing pull request updates the specification, decision record, tests, and scorer/configuration version together.

## 4. Static and documentation tests

Verify:

- Formatting, linting, and strict type checking.
- Complete typed docstrings for every public and private function: inputs, action, output/yield, raised failures, side effects, and determinism/idempotency when relevant.
- No undocumented executable module or CLI command.
- YAML duplicate-key rejection and schema validation of every example.
- JSON Schema meta-validation under Draft 2020-12.
- Internal Markdown links and planned-module/contract mapping.
- No credentials, private keys, authorization headers, database URLs, personal absolute paths, or copied secrets in tracked files or fixtures.
- License/provenance records for copied Corpus and reference artifacts.

Protocol-critical modules—configuration, corpus projection, prompting, provider request mapping, extraction, validation, retry/resume, persistence, and scoring—must have complete branch coverage for every documented decision path. Overall coverage targets are secondary to this requirement and cannot excuse an untested failure branch.

## 5. Configuration and grid tests

### Valid cases

- PSNC selected with several PSNC models.
- OpenRouter selected with several OpenRouter models.
- Both selected with three fully specified mock models each; also exercise one, two, and four models per selected provider so the initial three-model intention cannot become a hard-coded limit.
- The combined plan is the union of provider-owned grids: no OpenRouter model is dispatched through PSNC, no PSNC model through OpenRouter, and no cross-provider model pair is a configuration.
- Identical model-ID text on different providers creates distinct provider/model identities and payload routing.
- A single execution progresses through both provider queues, scoring, combined ranking, and required reports before final completion.
- Reasoning-capable model expands to exactly `disabled` and `enabled`.
- Incapable model expands to exactly `not_applicable` and no native reasoning field.
- Mixed capability models in one provider list produce the exact expected grid count.
- One reasoning profile produces 48 ranked configurations, 48 repetition-specific runs, 4,656 variable tasks/initial calls, and a 13,968-call upper bound.
- One reasoning-capable two-profile model doubles those counts exactly.
- For `S` applicable reasoning profiles summed over every selected model/provider, totals are exactly `48 × S` configurations, `48 × S` repetition runs, `4,656 × S` tasks, and a `13,968 × S` request upper bound; per-provider subtotals sum to those totals. Six one-profile mock models yield `288`, `288`, `27,936`, and `83,808`, respectively. Six two-profile models double these values.
- All four temperatures receive exactly one repetition under D-029 for both providers. Three OpenRouter-only one-profile models yield 13,968 initial and at most 41,904 total calls; two profiles each yield 27,936 initial and at most 83,808 total calls. No extra repetition is inferred from a nonzero temperature.
- Future configurability remains tested with a separate synthetic multi-repetition fixture; changing repetition counts changes the frozen plan identity and requires a new estimate, without changing or reusing an earlier campaign's tasks.
- Shot counts select ordered prefixes of length 0, 1, 3, and 5.
- The evaluation population contains exactly all 97 non-demonstration variables.
- Ranking excludes repetition from configuration identity and retains every configuration/result.
- Original-YAML hash and resolved selected-provider-union fingerprint follow their separate rules: provider-list reordering preserves the resolved hash; changing selected membership or active model profiles changes it; editing inactive catalogs does not.
- Frozen resume uses the stored selection/model list after YAML edits; another campaign has separate execution/run/task identities and does not silently reuse task rows.
- Configuration `2.1` accepts null provider/global caps for PSNC-only, OpenRouter-only, and combined selections; null means no cap, while all cost/reservation evidence remains recorded.
- Explicit owner-reported PSNC no-charge billing supports zero financial cost independently of cap settings while retaining unknown usage as unavailable. A configured zero provider/global cap permits zero-cost requests only.
- Combined no-charge PSNC and metered OpenRouter can run uncapped with billing bases, metered price cards, a valid disclosed estimate, and separate explicit live authorization retained.
- `require_pre_run_estimate: true` and `estimate_policy_version: pre-run-estimate-v1` produce hashed estimate receipts tied to the exact resolved plan, price and FX provenance/effective dates, and scenario assumptions.
- Estimates include per-model/provider/combined planned and maximum-three-attempt requests, prompt/output/reasoning token scenarios and correction-prompt bounds. Verify no doubled reasoning billing and no unpriced category silently treated as zero.
- Exceeding a disclosed estimate does not stop an uncapped authorized mock campaign; configured optional caps remain independently enforced.

### Invalid cases

- Unknown/duplicate YAML key, custom tag, wrong type, invalid range, or duplicate model ID.
- No selected provider, duplicate provider, unsupported provider, obsolete scalar `campaign.provider`, or model belonging to another provider.
- Empty enabled model list for any selected provider or unresolved active placeholder; a valid other-provider list cannot conceal the error.
- A reasoning-capable model missing either mode or an exact native value.
- A non-capable model with a reasoning request field.
- A mandatory-reasoning model incorrectly declared controllably disabled.
- Unsupported temperature, seed, top-p, output limit, or structured-output combination.
- Live mode without the evaluation-population/ranking-policy manifests, model revision policy, per-provider rate/concurrency limits, credential name, or explicit billing/optional-cap policy.
- Metered access without a usable price card/estimate, or capped paid admission without a defensible request-cost bound; negative caps, non-finite monetary values, or positive-cost admission against a configured zero cap. Uncapped mode must not invent a guaranteed upper bound as a prerequisite when uncertainty is documented in a usable disclosed estimate.
- Missing estimate, disabled required-estimate policy, undisclosed estimate, mismatched resolved-plan/price/FX hashes, unbounded or undocumented cost assumptions, or missing explicit live authorization. Configuration or pricing/FX changes invalidate prior estimates; providing an API key does not establish authorization.
- No-charge access inferred from missing price/usage data or lacking explicit owner/account-context billing provenance.
- Attempt limit other than the protocol value `3`.

Preflight failure occurs before campaign/task mutation or any provider request.

## 6. Corpus ingestion tests

### Verified Corpus v2.0.1 expectations

These expected facts are regression assertions enforced by the importer before it materializes anything:

| Fact | Expected |
|---|---:|
| Turtle files / Variable roots | 102 / 102 |
| Variables with exactly one `rdfs:comment` | 102 |
| Variables with exactly one Property | 102 |
| Variables with exactly one Object of Interest | 102 |
| Variables with Matrix | 51 |
| Variables with Context Object | 11 |
| Variables with Statistical Modifier | 9 |
| Variables with Constraints / total Constraints | 85 / 157 |
| Asymmetric systems | 36: 31 numerator/denominator and 5 source/target |
| Symmetric systems / total parts | 2 / 4 |

The implementation regenerates these values from tag `v2.0.1`; documentation numbers never replace programmatic verification.

Under v2.0.0 the Matrix and Context Object rows read 52 and 10. D-030's upstream edit moved `C12_HabitatProbability` from Matrix to Context Object, and that single change accounts for the whole difference: every other row is identical across the two releases. The importer refuses to materialize when the computed counts disagree with this table, and it reports the computed values so any divergence is immediately attributable.

### Per-file and determinism cases

- Tag, commit, tree, file count, ordered paths, Git blobs, byte lengths, and SHA-256 values match the frozen manifest.
- One root, label, definition, Property, and Object of Interest per file.
- Optional singular predicates have supported cardinality.
- All entity/system roles and constraint targets resolve without arbitrary graph iteration.
- The two constraints targeting unlabeled asymmetric-system blank nodes resolve to deterministic role-derived system labels, not parser-generated blank-node IDs.
- Exact category, subcategory, and full category path come from the release-relative path.
- All five demonstrations resolve uniquely and in exact order.
- No demonstration is an evaluation-population member.
- Every other Corpus variable is an evaluation-population member exactly once, for a total of 97.
- Different RDF triple/blank-node iteration orders produce identical canonical JSON and manifest hashes.
- A synthetic ambiguous/malformed file aborts atomic activation and leaves no runnable partial snapshot.

## 7. Lexical schema and semantic validation tests

Test at minimum:

- Complete valid simple decomposition.
- Every optional field at its explicit empty value.
- Missing/extra/null/wrong-type top-level fields.
- Empty and non-empty Property/Object-of-Interest values, including a structurally valid all-empty model response that is accepted and scored without correction.
- Valid symmetric system with two and more unique parts; invalid zero/one-part and duplicate-part cases as defined by the frozen schema.
- Valid source/target and numerator/denominator asymmetric systems.
- Mixed, incomplete, or unknown system shapes.
- Simple and structured alternatives for Object of Interest, Matrix, and Context Object.
- Empty and multiple Constraints; missing/empty/extra constraint fields.
- Constraint `on` resolving to Property, Statistical Modifier, entity, system member, or complete derived system label under the frozen rules.
- Unknown or ambiguous constraint targets.
- Separation between model-output validation and the internal explicit-empty terminal prediction policy.
- Deterministic canonical system labels without modifying raw/extracted records.

Every validation failure must match a stable error code, JSON Pointer where applicable, validator identity, ordered error record, and sanitized value summary.

## 8. Prompt tests

- Snapshot every base prompt family at 0, 1, 3, and 5 shots.
- Confirm single-message role structure and deterministic section order.
- Confirm exact schema bytes/hash match runtime validation.
- Confirm demonstrations contain exact definitions and canonical six-field outputs in approved order.
- Verify D-021 preserves the historical no-interpretation instructions and unchanged demonstration gold, including the documented non-literal-label and geographic-Matrix tensions. A known retained semantic limitation does not fail a snapshot or trigger generation correction; an unauthorized semantic repair or undocumented prompt change does.
- Confirm the target receives only its exact definition—no label, path, category, URI, RDF, gold fields, or linking answer.
- Confirm strict-minimal, constraint-decomposition, and matrix-decomposition differ only in their documented instruction block.
- Exercise Unicode, braces, quotes, backticks, and newline content without changing section boundaries.
- Confirm correction attempt 2 contains only attempt 1 raw visible response/errors, and attempt 3 only attempt 2 evidence.
- Confirm hidden/provider reasoning is stored when available but never inserted into correction prompts.
- Confirm all scientific request parameters remain identical across a task's attempts.

## 9. Response extraction tests

Cover:

- Whole response is one JSON object.
- Exactly one `json` fence or plain Markdown fence.
- Prose before/after one balanced object.
- Whole response parses to a string containing one JSON object, decoded once.
- A second JSON-string wrapper is rejected.
- A successfully parsed top-level array or scalar is rejected without mining a nested object.
- Braces and escaped quotes inside JSON strings.
- Nested objects/arrays and Unicode escapes.
- One valid and one malformed fence, multiple parsed fences, multiple plausible balanced objects, empty, HTML, malformed, and truncated responses.
- Huge/deep inputs at documented safety limits.

The extractor records strategy and byte offsets but never inserts, removes, renames, coerces, or semantically repairs fields. Property-based tests generate balanced/unbalanced structures and prove termination and deterministic candidate selection.

## 10. Provider contract tests

For both adapters, assert:

- Exact non-streaming URL, method, headers, timeout, messages, model, sampling, output, reasoning, and optional idempotency mapping.
- Authorization secret is present only in-memory/on-wire and absent from results, logs, fixtures, hashes, and database evidence.
- One adapter invocation causes zero requests on local validation failure or exactly one outbound request after dispatch.
- SDK/client automatic retry count is zero.
- Enabled, disabled, and not-applicable reasoning payloads match the exact selected model profile.
- Unsupported/mandatory reasoning and sampling combinations fail before dispatch.
- Complete raw response, assistant text, optional reasoning, returned model/provider IDs, usage, finish reason, timing, HTTP/error body, and delivery certainty are preserved.
- Authentication, permission, unknown model, validation 4xx, rate limit, transient 5xx, DNS/connect, timeout-before/after possible acceptance, malformed envelope, empty content, and HTML content are classified correctly.
- Adapter routing always follows the task's frozen provider/model identity; a provider error cannot invoke the other adapter as fallback.
- Both adapters share the scientific prompt/schema/scorer protocol while retaining distinct native capability maps, credentials, accounting, rate limits, and response provenance.

Mock servers count actual requests so nested or transport-library retries cannot escape detection.

## 11. Retry and state-machine tests

Use an invocation counter and durable PostgreSQL attempts to prove:

- Valid on attempt 1: exactly one provider request.
- Invalid then valid: exactly two requests and one correction parent.
- Invalid, invalid, valid: exactly three requests.
- Three content-invalid results: exactly three requests, terminal-invalid status, explicit-empty prediction, and score.
- Three safely retryable provider/transport failures: exactly three requests, operational-failure status, no model-performance prediction, and publication gate failure.
- A schema-valid but scientifically poor or all-empty response: accepted on that attempt, with no quality-triggered retry.
- Each transport failure consumes its dispatched attempt number under the frozen policy.
- Authentication/configuration/budget failures do not trigger repeated invalid requests.
- No adapter, SDK, worker restart, or repeated resume command can create attempt 4.
- All full prompts, raw responses, extraction candidates, errors, decisions, timings, usage, and cost remain queryable.
- The first valid prediction is selected once; later attempts do not exist.
- A PSNC-specific outage, rate limit, or authentication/configuration pause leaves eligible OpenRouter work running, and vice versa; temporary backoff resumes under the frozen provider policy without manual per-task continuation.
- Provider cap exhaustion blocks that provider's positive-cost dispatch; global monetary cap exhaustion blocks positive-cost dispatch across providers while explicitly no-charge PSNC work can continue. Shared persistence/integrity failure or explicit user stop blocks all new dispatch. Every case retains in-flight results and reservations for safe completion/reconciliation.
- No provider pause resets attempts, changes a task's provider/model, drops unfinished work, or permits a false completed campaign. Exhausted/ambiguous tasks remain explicit unresolved evidence.
- The orchestrator continues all eligible selected-provider work through scoring and required reports; process restart/resume reuses durable stages and final completion requires the whole frozen union.

Transition-model tests generate valid and invalid event sequences and prove that invalid transitions, stale leases, duplicate selected predictions, and content conflicts are rejected.

## 12. January scorer regression tests

The fixture bundle must be hand-calculated and, where unaffected, executed against `benchmarking_example/randomShotsPhaseOne.py` at its recorded hash.

Cover:

- Scalar normalization, exact equality, Close cosine threshold immediately below/at/above `0.80`, and missing values.
- Historical wrong non-empty behavior: FP only, no additional FN.
- Asymmetric numerator/denominator and source/target fallback, role reversal, partial role match, and arbitrary container labels.
- Accepted D-022 in both simple/system directions and all system/system combinations: one/full/no member match, more than two members, equivalent occurrences, canonical ties, asymmetric slot restrictions, and membership-only evidence where ordered roles are absent.
- Gold `water + air` against `water`: TP `1/2`, FN `1/2`, component F1 `2/3`; against `water + soil`: TP/FP/FN each `1/3`, component F1 `1/2`. Prove fractional credit survives without a whole-component threshold.
- Verify `U = g+p-m`, exact normalized contribution sum one, no member reuse, and exact stored rational metrics/rank inputs. A greatest-similarity greedy assignment must not replace maximum-cardinality system matching; use the documented counterexample and permutation/tie fixtures.
- Symmetric container-label exclusion, reordered parts, exact/partial/no part-set overlap, duplicate validation boundary, and the fact that the January part-set similarity does not use embeddings.
- Constraint normalization, recognized case-sensitive component-key prefix handling, all empty cases, greedy global assignment, equal-score tie behavior, fractional scaling, unmatched values, and numerical correction.
- Six-slot per-item aggregation and campaign micro aggregation from summed contributions.
- Explicit-empty terminal predictions.

Every intended difference from January must be attributable to the approved system-container-label exclusion or the explicitly finalized D-022 partial-credit correction. The scorer parity document and fixture manifest record the before/after expected values and prove preservation outside the final change boundary.

## 13. PostgreSQL integration tests

Against a fresh PostgreSQL 16 instance:

- Apply all forward migrations and verify the recorded revision.
- Import and atomically activate exactly one verified Corpus snapshot.
- Enforce unique campaign/run/task fingerprints and idempotent identical writes.
- Reject same fingerprint with conflicting content.
- Enforce nonempty unique campaign-provider memberships and composite foreign keys for `(campaign_id, provider_id)` and `(provider_id, model_configuration_id)`; reject a run or evidence record with mismatched provider ownership.
- Regenerate per-provider and combined plan totals; adding a provider to a new campaign does not transfer or reuse the existing campaign's task rows.
- Enforce demonstration/evaluation-population non-overlap and exact 97-member coverage.
- Enforce attempts 1–3, correct parent ownership/order, one selected prediction, and immutable terminal evidence.
- Preserve exact raw bytes/text separately from parsed queryable data.
- Use exact decimal money and exact rational confusion/metric/ranking receipts with separate decimal derivatives. Round-trip thirds and confirm mathematically equal ranks stay tied regardless of display precision.
- Race concurrent billed workers with null caps: atomic reservations/settlements retain all exposure without rejecting valid work for absent caps. Then configure provider and global caps: admission locks and reserves both allowances atomically; reject overspending even when each worker's initial read would fit individually.
- Persist and restore the required estimate receipt/hash, dated source/FX/scenario provenance, disclosure record, and separate explicit live authorization; reject live admission when any required link is missing or refers to a different frozen plan.
- Replay reservation/settlement writes idempotently, retain ambiguous-delivery exposure, and release unused allowance only when evidence supports it. Explicit no-charge zero cost remains distinct from unknown token usage or pricing.
- Prevent cascading deletion of scientific evidence.
- Verify short skip-locked claims, lease fencing, heartbeat, expiry, stale-worker rejection, and no transaction held during simulated network delay.
- Regenerate metrics from match/contribution records.
- Back up, restore, and verify sampled/all hashes and relationships.

## 14. Ranking and complete-result retention tests

- Build known repetition-level micro metrics and verify the primary value is their unrounded arithmetic mean, not a pooled-variable weighting across repetitions.
- Verify the current singleton repetition's ranking mean equals its own micro Close F1 and retains all Precision/Recall/contribution evidence. Use a separately labelled synthetic multi-repetition fixture to test generic averaging, with distinct campaign/plan identities and no current-grid expansion.
- Do not treat retry attempts as independent repetitions or report singleton run-to-run variance/confidence intervals as measured evidence; variable-level descriptive statistics remain a separate deferred analysis.
- Sort primary values descending and give exact ties the same competition rank.
- Rank the combined selected-provider configuration union using the unchanged shared population/scorer/policy; retain provider/model on each row and permit exact ties across providers.
- Provider-filtered views declare their scope; incomplete work for one provider remains visible in the campaign report and blocks final campaign completion, while completed other-provider configurations can retain interim ranks.
- Confirm secondary metrics are reported but do not silently break a primary tie.
- Preserve unrounded per-variable, per-component, and per-repetition Exact/Close Precision, Recall, F1 and supporting counts/denominators so later descriptive statistics need no generation rerun. Do not require deferred variance, median, mode, IQR, or uncertainty calculations for a completed ranking.
- Count terminal-invalid explicit-empty predictions in the 97-variable denominator.
- Mark missing or operationally unresolved configurations `not_rankable` without deleting their completed results.
- Preserve every raw attempt, prediction, item/component score, repetition metric, lower-ranked configuration, tied configuration, and failure reason.
- Reject a rank when expected coverage is not exactly `97 × declared repetitions`.
- Rebuild the same rank idempotently; a changed ranking policy creates a new immutable ranking run.

## 15. Failure-injection and crash matrix

Terminate or fault the worker:

- Before attempt insertion.
- After request evidence commit but before dispatch.
- During dispatch and after possible acceptance.
- After response receipt but before durable response commit.
- After response commit but before extraction.
- After extraction but before schema/semantic validation.
- After failed validation but before retry transition.
- After valid prediction commit but before scoring.
- During item scoring and aggregate insertion.
- During export creation.
- At a user interrupt and configured optional hard-cap boundary; separately exceed only an informational estimate and prove continued execution when uncapped.
- During concurrent provider/global reservation admission or response settlement.
- While one provider is paused/backing off and another is processing, including process restart before final combined ranking/report creation.

Expected invariants are no lost committed evidence, no duplicate verified call, conservative ambiguous-delivery pause, no fourth attempt, no overwritten prediction/score, and convergence to the same known database state when the external outcome is known.

## 16. Offline end-to-end dry run

The dry run uses three real non-demonstration Corpus variables selected to cover a simple entity and at least one structured system/constraint. Deterministic mock endpoints exercise the four required scenarios in `docs/runbook.md`, plus PSNC-only, OpenRouter-only, and combined selection/routing. Exact three-model-per-provider expansion is validated separately without increasing the three-variable smoke dataset.

Verify end-to-end:

- Source TTL → canonical gold → evaluation-population/category provenance.
- Prompt/schema/demo hashes → planned task identity.
- Request → raw response → extraction → validation → correction/terminal policy.
- Prediction → exact/close match facts → aggregate metrics.
- Usage/timing/cost availability → reports and manifests.
- Interruption → resume from stored response with unchanged invocation count.
- Second full invocation → no duplicate logical work.
- One provider pauses → healthy-provider work completes → stored pause clears under the supported recovery policy → remaining work and combined reports complete without changing task ownership or retry allowance.
- Explicit no-charge and metered fixture accounting → separate provider subtotals and reconciled global reservations with null caps and configured caps; no synthetic usage or cost is represented as live evidence.
- Frozen plan and price/FX fixtures → estimate receipt/report → disclosure/authorization fixture → permitted mock dispatch; missing estimate or changed plan → rejection before dispatch.

No dry-run artifact may be selectable by a live report.

## 17. Performance and scale tests

Before the full grid, measure without paid generation:

- Planner expansion time/memory at the exact campaign size.
- Database insertion, claiming, heartbeat, and report-query behavior under expected worker concurrency.
- Maximum prompt and response sizes supported by storage/extraction limits.
- Embedding batching/cache correctness and deterministic result tolerances.
- Backup size, restore time, and evidence-hash verification time.

Performance optimization must not change canonical ordering, task identity, request content, evaluator results, attempt accounting, or evidence durability.

## 18. Acceptance gates

| Gate | Required outcome |
|---|---|
| T0 Documentation | All contracts complete, consistent, linked, and approved before code |
| T1 Static/unit | All deterministic boundaries and documented error branches pass |
| T2 Corpus/schema/prompt | 102 records, five ordered demos, no leakage, all contracts/hashes pass |
| T3 Scorer | Accepted `january-derived-member-credit-v1` passes unchanged-branch January fixtures and explicit before/after correction, matching, and rational arithmetic fixtures |
| T4 Persistence/recovery | Migrations, concurrency, attempt cap, evidence, crash/resume, backup/restore pass |
| T5 Offline end-to-end | Mock dry run and idempotent resume pass with zero provider calls |
| T6 Live freeze | Selected provider set and exact per-provider models/reasoning/population/ranking/sampling/rates/billing bases/price cards/optional-cap policy frozen; matching cost estimate disclosed and separate explicit live authorization recorded |
| T7 Live canary | Current wire behavior, accounting, validation, and retry boundary verified |
| T8 Scientific campaign | Every selected provider subtotal and combined task/denominator count reconciles; all tasks scored, combined ranking complete, and required reports recorded/regenerable |

A failure in any gate blocks the next one. A test waiver affecting scientific semantics requires a new documented decision and cannot be hidden as an operational exception.
