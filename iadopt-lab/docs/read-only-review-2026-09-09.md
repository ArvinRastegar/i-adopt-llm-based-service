# I-ADOPT Lab — current source audit, 9 September 2026

## Purpose, scope and limitations

This review asks whether the current implementation faithfully carries out the approved lexical-decomposition experiment and preserves enough evidence to reproduce and interpret it. Repository: `/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service`; experiment: `iadopt-lab/`. All links below are relative to this report.

**Assessment: the core scientific pipeline is substantially implemented, but the current system does not yet satisfy the evidence-preservation, reporting-identity, and durable-freeze contracts.** Corrected mechanisms stay closed; the historical claim that all ten findings are resolved remains too broad. First priorities are received-response preservation (1–2), artifact retention (11), and scientific attribution (3–4, 13). The newly enabled three-model PSNC scope also lacks matching live price-card coverage (6). Exported report artifacts read during this pass turn that cost-projection defect from a source-derived trigger into an observed one (6), and the committed corpus bundle carries one derived projection its own verifier would reject (14). No particular recorded score is established as wrong. This review does not certify live readiness.

Only this report was changed. Inspection used local file reads, source searches, file hashing, and read-only Git inspection. No project modules were imported; no tests, experiment commands, database operations, service operations, dependency installation, downloads, or model calls were performed. Credential files were not opened. No applicable `AGENTS.md` was found in the repository or inspected ancestor locations. The existing dirty/untracked worktree was treated as user-owned.

Test bodies, fixtures, migrations, and historical scorer source were read as evidence of implementation and coverage, not executed. Statements about 299/312 passing tests, lint, preflight, corpus verification, dry runs, and live campaigns are **reported evidence from the existing author log or project documents**, not independently verified outcomes. Failure examples below are source-derived triggers, not reproduced failures. Deployment behavior and actual stored campaign contents remain unverified.

This revision uses focused edits to the current assessment. The pre-existing historical appendix, including author annotations and the appended resolution log, is retained verbatim; the previous source snapshot and revision note also remain below as dated history. Historical statuses and phrases such as “remaining problems” are not current findings: use the table below. An earlier review reported that its original log was absent; no missing log has been reconstructed.

**Concurrent-work limitation:** `parameters.yml` changed through another writer during inspection. The first read selected only GLM-5.2 with reasoning disabled and a 16,000-token ceiling; the later snapshot enables all three PSNC models. The latter snapshot governs the scope stated below. Its hash and the earlier observed hash are recorded in the revision record. No inference is made about who changed it or which commands they ran. Only this review’s operations are covered by the no-execution statement. **This report file was also revised by another writer during the same pass:** it was read at SHA-256 `5b436d9a1d222f939d759bcd74c59cc355c3cc66784546b4ccb2d1094ded5dac` and re-read at `34473e61f126930751e8732599d13d415883cf50435a443fe552fa9e3b4fa449` before editing. The present revision extends that later version by focused edit and deletes none of it; every earlier finding, note and hash table below is retained.

## Intended experiment and current scope

Corpus **v2.0.1 is intentional**. Its 102 scientific variables are converted from pinned Turtle into deterministic six-field lexical ground truth. Release identity, source hashes, categories and subcategories accompany the records. Five demonstrations form a fixed ordered pool: `C2_AirDailyMaximumTemperature`, `PersWelfare`, `lactate`, `CirculationMode-Water`, and `HeatStress`. All other 97 variables form the evaluation population. There is no training, development, test, holdout, or stratified split.

A task combines one target with a provider-owned model, prompt family, demonstration prefix, temperature, reasoning profile, and repetition. Strict minimal, Constraint decomposition, and Matrix decomposition retain historical no-inference instructions with documented six-field/schema adaptations. Only the target definition goes to the model, alongside the schema and chosen demonstrations; target gold and provenance are not prompt inputs. Validated lexical predictions are scored; RDF, JSON-LD, SHACL, entity linking, deterministic baseline, and human review remain deferred.

Each task permits at most **three total provider requests**, including operational retries. Corrections keep the scientific settings and include the immediately preceding invalid assistant response and validation errors. Three delivered content-invalid responses may produce an explicit empty prediction. Provider, capacity, infrastructure, and ambiguous-delivery failures must remain distinct and must not become model-quality zeros.

The scorer preserves January scalar, empty-value, and Constraint behavior with D-022 system-member corrections. Arbitrary system names are unscored. Partial matches retain exact TP/FP/FN contributions; missing members reduce Recall and extra members reduce Precision. Ranking sums contributions over the full population for each repetition, computes MICRO Close F1, then averages repetition values. It uses unrounded fractions and shared exact ties; incomplete configurations remain visible and unranked.

The broad grid and the current scoped selection must not be conflated. D-029 establishes one repetition at every temperature; D-032 changes the OpenRouter catalog; D-033 permits deliberately selected reasoning-profile subsets. In the latest inspected [parameters](../parameters.yml), PSNC alone is selected, with **DeepSeek-V4-Flash (`not_applicable`), GLM-5.2 (`disabled`), and Qwen3.8-27B (`disabled`)** enabled. Both disabled profiles send `chat_template_kwargs.enable_thinking=false`. Strict minimal, five shots, temperature 0.5, top_p 1.0, one repetition and a **16,000-token output ceiling** give three configurations and **291 target tasks, at most 873 requests** by source-derived grid arithmetic, not an executed plan. The enabled OpenRouter catalog entry remains inactive because OpenRouter is not selected.

The comments report direct capability measurements, but D-036 still concludes GLM reasoning cannot be disabled, and D-038 still records a 100,000-token ceiling. The current parameters cite missing D-040 for the reduction; an earlier version read during this review also cited missing D-041 for the reasoning reversal. These are unresolved decision/evidence discrepancies (10), not proof the reported measurements are false. A `not_applicable` control means no reasoning switch is sent; it does not establish absence of reasoning. No finite output ceiling guarantees a complete or valid answer. No mandatory spending cap is required; an estimate must be disclosed and actual, estimated, unavailable and evidenced-zero costs must stay distinguishable.

## Current status of earlier findings

“Addressed in source; execution unverified” means the original mechanism is corrected in these bytes. Historical successful checks remain attributed reports rather than new verification.

| ID | Current status | Reassessment |
|---|---|---|
| 1 | Partially addressed | psycopg/pool failures now enter retries; immutable conflicts escape immediately. Heartbeat/campaign cancellation can still abandon the retry lifecycle. |
| 2 | Partially addressed | Exceptions from `settlement_policy` are caught. Metadata normalization and settlement inside `store_response` can still prevent raw persistence. |
| 3 | Partially addressed | All eight cases in the appended log have corresponding source branches. A dictionary message without a valid content field still counts as a completion. |
| 4 | Partially addressed | CLI supplies identities/categories and finalization uses constants. Binding is asserted by copying a hash, and CLI category denominators include demonstrations. |
| 5 | Addressed in source; execution unverified | Runner reads `retry_after_seconds`; original message-field mismatch is closed. |
| 6 | Partially addressed | FX multiplication added; current three-model selection exceeds the live card’s coverage and nothing checks it before dispatch, currencies are not bound, and reports discard cost evidence — the last now observed in an exported live report, not only deduced. |
| 7 | Addressed in source; execution unverified | Both CLI planners select three synthetic targets before collecting the population hash. |
| 8 | Partially addressed | Original exit-code and pre-registration mismatch defects are addressed in source. A stored transient rejection can strand work after interruption; see residual finding 8. |
| 9 | Partially addressed | Conditional admission assumptions are documented; deployment evidence is not enforced, and the new evidence command can bind current-input assumptions to an old plan. No observed undercount is established. |
| 10 | Partially addressed; unresolved evidence discrepancy | Old front matter and D-039 reference are fixed. Current reasoning/ceiling claims conflict with the decision record; D-040/D-041 are absent; the two new public commands are undocumented. |
| 11 | Open | CLI still does not persist or retrieve its complete freeze bundle. |
| 12 | Open | Estimate identity/disclosure validation remains incomplete; the actor default is now `operator`, not an empty string. |
| 13 | Open (new) | Probe verdicts can promote inconclusive or failed observations into campaign capabilities. |
| 14 | Open (new) | The committed corpus bundle holds one derived projection that `verify` would reject; the repair exists only as an uncommitted worktree edit. |

No whole earlier numbered finding is withdrawn as historically incorrect. The incoming assessment’s current-scope description (always-on GLM, 100,000 ceiling), untracked-lab claim, and empty-string actor-default claim are superseded by current bytes; their historical truth is not adjudicated. Several old **current-tense** claims are retired: exception mismatch (1), uncaught settlement-policy exception (2), the four original malformed/truncated examples (3), omitted CLI optional arguments (4), wrong delay field (5), missing FX multiplication (6), synthetic population mismatch (7), completed-resume failure/pre-check registration (8), and obsolete specification front matter (10). Residual findings below do not reopen those exact mechanisms.

## Detailed current findings, in practical priority order

### 1. Response-storage retries can be cancelled by their supervisor

**High — partially addressed; confirmed control-flow defect.** Expected: a temporary database failure stops new dispatch while already received evidence continues toward durable storage without another model call.

[workflow.py](../src/iadopt_lab/workflow.py), `_commit_response` (174–207), now catches `PersistenceError` and `psycopg.Error`, excludes `EvidenceConflict`, and schedules 122.5 seconds of backoff across eight writes. The previous review reported that the installed pool’s `PoolTimeout` derives from psycopg `OperationalError`; this pass rechecked the project’s catches and its `psycopg.OperationalError` fixture, not installed dependency internals. The old psycopg exception-mismatch mechanism is addressed in source. However, `run_task` (226–254) cancels `_advance_task` if `_heartbeat` raises. The configured heartbeat interval is 30 seconds. A database outage can therefore affect both the commit and heartbeat, cancelling work during a retry sleep before storage recovers. `asyncio.shield` protects an individual database await, not the surrounding retry loop. `run_campaign.reap` propagates worker failures, and the campaign `finally` cancels other inflight tasks (597–605, 651–656).

**Trigger/consequence:** receive an answer, fail its writes, then fail a heartbeat while the commit loop waits. The process may remain alive but cease trying to store those paid bytes. A dispatch marker preserves ambiguity, not the answer. The exact timing depends on connection/pool timeouts; no runtime failure was reproduced. A hard process death is separately documented as inherently ambiguous and is not presented as a newly fixable exactly-once guarantee.

**Future completion criterion:** define ownership of received evidence across lease loss, shutdown, and shared failures; stop further dispatch without cancelling preservation. Proposed acceptance: exercise `run_task`/`run_campaign`, not just `_commit_response`, with simultaneous heartbeat and storage failures and later recovery; assert exact raw bytes survive and provider count stays one. Classify permanent non-conflict storage errors separately rather than retrying every psycopg error indiscriminately.

### 2. Raw persistence still depends on metadata and transactional accounting

**High — partially addressed; confirmed remaining evidence-loss paths.** Expected: uninterpretable provider metadata must not destroy the raw evidence that explains it.

The policy exception guard in [workflow.py](../src/iadopt_lab/workflow.py), `_advance_task` (331–345), fixes the original nonnumeric `usage.cost` case by recording unavailable accounting. It does not cover downstream validation in [repository.py](../src/iadopt_lab/persistence/repository.py), `store_response`: `_clean(metadata)` precedes any transaction (1325); the response insert (1366) and monetary interpretation/settlement (1424–1460) share one transaction.

**Concrete triggers:** (a) on a metered route, `usage.prompt_tokens=-1` and `completion_tokens=0` pass `_live_services.settlement_policy`'s integer check ([cli.py](../src/iadopt_lab/cli.py), 443–449), producing a negative estimate. Repository `_money` rejects it (154–168, 1431), rolling back the response insert; retrying the same payload cannot fix it. (b) envelope parsing uses ordinary `json.loads` ([providers/base.py](../src/iadopt_lab/providers/base.py), 139–154), so non-finite numeric metadata or an escaped unpaired surrogate can survive parsing but fail canonical metadata serialization before raw storage. Even a harmless provider metadata key named `password` is rejected recursively by `_clean` (124–140). These examples do not require opening or exposing a credential.

**Impact/scope:** a received response can remain undurable despite the new policy guard. Metered negative usage is not an observed PSNC charge; metadata failures can affect either provider. The adapter's base64 raw-byte receipt is useful, but it cannot bypass a transaction that never commits.

**Future completion criterion:** preserve raw receipt independently of fallible interpretation; retain safe diagnostic evidence and explicit unresolved accounting. The historical repair-boundary note remains valid: immutable response/settlement rows require a designed append-only correction path, not merely reordered calls. Proposed acceptance: run malformed usage and metadata through adapter → actual repository → retry supervision, verify exact bytes and unresolved state, and verify later accounting reconciliation without rewriting evidence or settling twice.

### 11. CLI preparation does not persist the artifact bundle it claims to freeze

**High — open; incomplete wiring, linked to 8.** Expected: PostgreSQL retains the frozen plan, source/schema/prompt/scorer/runtime evidence needed for reconstruction and explicit resume.

[artifacts.py](../src/iadopt_lab/artifacts.py), `collect_input_artifacts` (48–91), constructs file bytes, a file-hash index, runtime versions, and identities. [cli.py](../src/iadopt_lab/cli.py), `_prepare_campaign` (512–596), uses those identities but keeps the bundle only in `Services`. It calls `register_campaign` without `artifact_refs` and never calls `register_artifact`. Repository supports both artifact registration/linking and retrieval (`register_artifact`, `register_campaign`, `get_campaign_artifact`), but the CLI does not wire them. `plan_tasks` stores a database-specific plan and full resolved runs, not the original bundle/file index/runtime payload. The original YAML bytes held by `Configuration.original_bytes` are also not included in this CLI registration, and `collect_input_artifacts` does not collect `parameters.yml`; preserving resolved JSON alone does not meet the original-YAML evidence contract.

**Trigger/consequence:** create a campaign through the public CLI, then lose/change the checkout or environment. The plan hash can reject drift but cannot recover its underlying bytes. `_prepare_campaign` reconstructs from current files and instructs the user to restore the old checkout; it does not retrieve a bundle from PostgreSQL as [execution-evidence.md](execution-evidence.md) claims. Resolved configuration, task/run evidence, source TTL, gold, and dispatched prompts do remain stored; this is not total loss of provenance. Manually registered artifacts or external backups might exist, but were not queried.

**Future completion criterion:** persist and link a complete retrievable freeze before work becomes dispatchable, and compare against that stored identity on resume. Require an explicit campaign selection (`--campaign` is currently optional at CLI 887). Proposed acceptance: prepare through the public CLI, inspect the complete database artifact lineage, and restore a matching workspace from it; a changed workspace must reject before mutation, while a restored one resumes completed generation locally.

### 4. Public reports still overstate scorer binding and category coverage

**High for scientific attribution; Medium for category summaries — partially addressed.** Expected: every report verifies the scorer, target gold, run association, and selected population against independently frozen evidence.

`cmd_report` now supplies scorer and category arguments ([cli.py](../src/iadopt_lab/cli.py), 206–244); finalization uses evaluator constants ([workflow.py](../src/iadopt_lab/workflow.py), 498–515). Those earlier omissions are fixed. However, `plan_scorer` is copied directly from the supplied plan, then compared back to that same hash ([reporting.py](../src/iadopt_lab/reporting.py), 347–366). The hash actually binds evaluator source files plus similarity identity (`collect_input_artifacts`, 84–85); reporting never recomputes that binding. A current different backend plus observations from that backend can agree with one another and still be labelled `verified-against-plan` for an old plan. The unit “wrong backend” case supplies **no evaluations** and tests a wrong copied string, not this chain.

Observation checks bind `metadata.variable_id` and consistent gold hashes across rows, but not `metadata.run_id` to the outer run or gold to the canonical corpus (`_observations`, 123–136; ranking 369–376). Misassigning an intact evaluation to another configuration's outer run can therefore misattribute scores without altering that evaluation's hash. This concerns externally supplied reporting inputs; normal workflow constructs the correct run and persistence checks its own evidence.

Category metadata passed by `cmd_report` covers **all 102 records** (235–237), not `plan.population`. `build_category_summary` (264–288) uses that mapping unchanged. Thus a fully scored 97-target report can mark a category incomplete because its demonstration members were never meant to be scored; a three-target synthetic report inherits unrelated corpus denominators. Entirely absent categories are omitted because groups are created only from observations. Supplied row categories are not compared against the expected mapping.

**Future completion criterion:** derive and check source/backend, gold, run and category bindings independently; filter expected metadata to the selected population and represent wholly missing categories. Proposed acceptance: invoke the actual report command with old-plan/new-backend evidence, a swapped run, wrong gold/category, all 97 targets, and a synthetic subset. Valid inputs must have correct denominators; inconsistent inputs must fail before export. Whole-configuration missing-target checks and rational ranking already work and must remain.

### 3. A dictionary message is still mistaken for a valid completion structure

**High — partially addressed; confirmed classification defect.** Expected: only an established model completion can enter content validation.

[providers/base.py](../src/iadopt_lab/providers/base.py), `send_once` (155–201), sets `well_formed_choice=True` whenever `message` is a dictionary. It does not require a `content` field or validate its type. `{"choices":[{"finish_reason":"stop","message":{}}]}` and a message with numeric `content` become delivered `empty_response`. Three such results can produce the scored terminal-empty prediction through `_validate_and_select` ([workflow.py](../src/iadopt_lab/workflow.py), 371–393). A provider `error` object is also rejected only when no dictionary message was found (184).

The original `{}`, error-only envelope, absent/non-dictionary message, and empty/nonempty `finish_reason="length"` cases are now operational. Their historical resolution is preserved. A genuine explicit empty completion must retain its approved treatment; deployed null-content/reasoning-only cases need an explicit envelope contract rather than accidental dictionary acceptance.

**Impact/scope:** malformed API structures can still lower measured quality. No affected stored task is established. **Future completion criterion:** validate the supported completion shape and conflicting error/completion fields before content routing. Proposed acceptance: extend transport cases to missing content, wrong content types, valid empty/null cases under the chosen contract, error-plus-message, and a normal six-field answer; then verify three malformed envelopes never create a scored empty prediction.

### 13. Capability probes promote inconclusive evidence into scientific configuration

**High — open; confirmed classification defect and unsupported capability inference.** Expected: capability evidence distinguishes a successful supported control from failed, truncated, ignored, or unobserved behavior and remains tied to the actual deployment.

In [probing.py](../src/iadopt_lab/probing.py), `_probe_one` (123–144) computes `reasons_by_default` as false both for a successful baseline with no exposed reasoning **and for a failed baseline**. After any switched answer, `elif not reasons_by_default` assigns `never_reasons`, `usable=True` before checking whether the switched call itself emitted reasoning. `models_block` (239–255) converts that into `reasoning_control:false` and `not_applicable` with **no switch** for real generation. Even two successful trivial probes cannot establish the universal claim “never reasons”; an unexposed reasoning channel is not proof of no reasoning generation.

**Concrete trigger:** baseline HTTP 503, followed by a successful switched response containing an answer and 100 characters of reasoning. The generated configuration says the model emits no reasoning and enables it for the nominally reasoning-off scope. The test fixture only fails all calls together; it does not cover this mixed outcome. `structured_ok` (141) similarly means nonempty answer, not valid JSON or successful native enforcement; `finish_reason=length` is recorded but ignored. `models_block` writes temperature/top_p/seed support true based on accepted requests, without establishing whether those controls were honored. These assertions can pass `resolve_configuration` because the generated shape and evidence strings are valid.

A related deployment-binding gap is visible in callers: `cmd_probe_models` resolves the configured base-URL environment override (731–739), whereas `_prepare_campaign` loads API-key names only and invokes `psnc.create_adapter(profile, key)` without its supported `base_url_override` (CLI 560–568; [psnc.py](../src/iadopt_lab/providers/psnc.py), `create_adapter`). Thus probes may describe an override deployment while generation uses the default URL. The latest generated evidence strings name the host without `/v1`, while generation defaults to `/v1`; equivalence of those gateway routes was not established by this read-only review.

**Impact/scope:** the experimental reasoning labels and deployment capabilities may overstate what was demonstrated. This is not proof that the latest GLM or Qwen switch measurements are wrong, nor a reason to reject deliberate D-033 single-mode scope. Probe outputs are not scored benchmark predictions. Their reduced, non-durable receipts are separately covered by 12.

**Future completion criterion:** require usable baseline and controlled observations, classify contradictory/missing/truncated evidence as inconclusive, validate structured content, state the limits of acceptance-only sampling checks, and bind probes and execution to one frozen effective deployment. Proposed acceptance: baseline failure plus switched reasoning, baseline no-reasoning plus switched reasoning, missing reasoning exposure, malformed envelope types, non-JSON structured answer, partial `length` answer, and a differing base-URL override must never yield an unsupported usable capability. A verified supported profile must survive configuration resolution and emit the exact verified request fields at the generation boundary. All proposed probes/tests require separate execution authorization; none ran here.

### 10. Documentation corrections are real, but operating and acceptance claims remain mixed

**High for reasoning/scope attribution; Medium for stale status text — partially addressed, with an unresolved evidence discrepancy.** Expected: current behavior, historical results, and future requirements are clearly separated.

The disputed [technical-specification](../TECHNICAL_SPECIFICATION.md) line 4 now reads `status: Implemented; live runs executed. See README and DECISIONS.md for current state.` Its current hash is in the snapshot below. The author's appended explanation resolves the **historical** search discrepancy; the stale line must no longer be presented as current. README's PostgreSQL-not-run sentence and missing D-039 reference were also corrected.

**Unresolved scientific evidence discrepancy:** the latest parameter snapshot declares GLM reasoning control supported and selects its disabled profile, while D-036 (334–352) explicitly states the opposite. It sets 16,000 tokens (parameters 303) while D-038 records 100,000 (decisions 362–380). D-040 is referenced but absent; D-041 appeared in the earlier parameter snapshot read during this pass and is also absent from the decision record, which ends at D-039. Current comments and generated probe summaries are reported measurements, not independent verification or a recovered decision entry. They may support a legitimate refinement, but the authoritative supersession, scope and linked evidence must be recorded before describing these as settled scientific decisions. No change to either configuration or decision history is proposed by this review.

Other contradictions include the [runbook](runbook.md), §1, saying “No provider request has been made” despite README’s documented live campaigns, and README's final “Approval gates” paragraph saying source/dependencies still need implementation despite its opening implemented/live status; [implementation-progress.md](implementation-progress.md) mixing old 274/285/299 totals and “Two gaps remain open” with statements that those gaps have passed; and [execution-evidence.md](execution-evidence.md) promising database bundle retrieval, per-model token evidence, and separately supplied authorization receipts that the CLI does not implement (11, 9, 12). D-038's heading and “guaranteed-complete answer” language still conflict with its corrected final caveat. The two new public commands are also wholly undocumented: `probe-models` and `evidence` appear in no README, runbook, specification or decision text. Searching `README.md`, `TECHNICAL_SPECIFICATION.md`, `DECISIONS.md` and `docs/` for `probe-models`, and for the invocation `iadopt-lab evidence`, returns no hit outside this report; the runbook and README command lists (runbook §1; README 20–26) still end at `estimate`, `report` and `database`. That matters because one of the two makes live provider requests and rewrites `parameters.yml` in place, and the other produces the document whose `ceiling_verified` flag gates live dispatch. README's current 299 total differs from the appended log's reported 312; these may be different revisions and are not evidence of a failing suite.

**Impact/scope:** an engineer cannot use these documents alone to distinguish an acceptance target from an exercised public workflow. **Future completion criterion:** retain dated historical results, label them with code identity and commands, and reconcile current operating instructions with source and recorded decisions. Proposed inspection: each current completion/test claim names its evidence and limits; obsolete claims remain only as clearly dated history. This review performs no new test-count verification.

### 6. Cost conversion is improved, but units and report projection remain inconsistent

**High for current campaign completion; Medium for units/reporting — partially addressed; confirmed boundary defects.** Expected: estimates, reservations, settlement, and summaries identify common units and retain actual/estimated/unavailable/confirmed-zero distinctions.

`_live_services` now applies `fx_to_reporting` to reservations, provider costs, and usage estimates ([cli.py](../src/iadopt_lab/cli.py), 392–452). The missing multiplication claim is closed. But [parameters.yml](../parameters.yml) declares reporting USD (358–363), while `maximum_campaign_cost.currency` is EUR even with a null cap (388–390). `register_campaign` derives the database currency from that cap field ([repository.py](../src/iadopt_lab/persistence/repository.py), 470–471), and `start_attempt` writes that currency on reservations (1144–1152), without comparing it to the cost receipt/card. The current cards declare USD. This mismatch exists now; today's non-billed PSNC zero has no numerical FX loss, but a metered selection can store USD-derived numbers as EUR. Neither provider-reported billing currency nor estimate/card consistency is independently validated.

Reporting has a separate broken boundary under the same accounting finding: `_attempts` (2011–2048) does not retrieve settlement rows, and `_attempt_projection` ([workflow.py](../src/iadopt_lab/workflow.py), 444–469) drops cost metadata already present in response evidence. `_attempt_summary` ([reporting.py](../src/iadopt_lab/reporting.py), 190–217) requires `attempt.cost` with `mode`, `amount`, `currency`, `basis`, and `kind`, a shape the projection never supplies. Normal campaign reports therefore classify all attempts' money as unavailable, including known zero. Raw settlement rows are not lost. This confirms the appended author's “Still open” note and identifies the additional projection/shape issue.

**Observed in a produced artifact, not only in source.** The exported live report [`outputs/ranking-ee138ed5-15c8-52b9-8499-c887a4521649.json`](../outputs/ranking-ee138ed5-15c8-52b9-8499-c887a4521649.json) was read during this pass. It is `mode: live`, `synthetic: false`, 97 targets, one configuration (`psnc/GLM-5.2`, `reasoning_mode: disabled`, strict-minimal, 5 shots, T=0.5), `complete: true`, `scorer_binding: verified-against-plan`. Every one of its **26 cost summaries** — the repetition summary and each category/subcategory slice — reports `available: false`, `reported_attempts: 0`, `groups: []`, totalling **315 attempt slots with no monetary evidence**, on a provider whose billing mode is `non_billed` and whose correct value is therefore a confirmed zero. The repetition summary shows `attempt_records: 105` with `outcomes: {"response_received": 105}`, so the attempts themselves are present and classified; only their money is dropped. This is direct output evidence of the projection defect above. It does **not** establish what the database holds: no `cost_settlement` row was queried, and the raw settlement evidence is expected to be intact.

**Nothing enforces price-card coverage before dispatch.** `parameters.yml` declares `require_complete_price_evidence_before_live: true` (364), but that key appears nowhere in `src/` — it is never read. `configuration.py` (208) only checks that `price_card_manifest` is non-null, not that the card covers the enabled models, and `validate_live_readiness` has no card check. `cost_policy` is invoked at [workflow.py](../src/iadopt_lab/workflow.py) 300 inside the `request` mapping literal, outside any `try`, so its `LabError` leaves `_advance_task`, is re-raised by `reap` (`finished.result()`), and reaches `cmd_run` through `run_campaign`'s cancelling `finally`. The staged [`outputs/estimate.json`](../outputs/estimate.json) reports `ready: true`, `issues: []` and expected cost `0` for all three models because `evidence.billing_cards` reads `providers.psnc.billing`, while execution reads the card file — two different sources that no boundary compares.

**New current-scope blocker:** the latest parameters enable three PSNC models, but the configured [price card](../data/manifests/price-card-glm-5.2-psnc.yml) contains only `psnc/GLM-5.2`. `_live_services.cost_policy` (CLI 395–400) looks up the exact model key and raises **before** its non-billed branch. `evidence.billing_cards` (94–122) can generate non-billed estimate entries for all selected models from the provider declaration, but neither `cmd_evidence` nor `_authorize` synchronizes or compares that evidence with the card actually used by execution. Thus a ready zero-cost estimate does not establish live coverage. The first task for either other model raises before `start_attempt`; the campaign failure path may cancel unrelated in-flight work (1). No full run or failure was executed here, and task order determines whether any GLM request precedes that failure.

**Future completion criterion:** establish one authoritative reporting currency, validate conversion evidence across all boundaries, and project durable settlements into a versioned report shape. Proposed acceptance: a metered USD→EUR non-unit-FX campaign plus actual, estimated, unavailable and non-billed fixtures must reconcile estimate, database balances and exported subtotals; repeated finalization must not double count. Add a public preparation case with the current three selected models and a one-model card: reject the incomplete card before **any** provider dispatch, or consume a complete frozen non-billed card consistently. Preserve separate state labels rather than turning missing money into zero.

### 12. The live CLI constructs disclosure and approval evidence rather than consuming it

**High — open; operational evidence/contract mismatch.** Expected: an estimate is disclosed before live dispatch and approval provenance reflects an explicit recorded action.

`cmd_run` requires an estimate file; `_prepare_campaign` then calls `_authorize` automatically when one is provided ([cli.py](../src/iadopt_lab/cli.py), 591–595, 652–673). `_authorize` (458–489) checks `ready`, issues and the plan hash, but does not recompute the estimate's own hash, bind its price evidence to the live card, or preserve the full assumptions/prompt evidence. It constructs both disclosure and authorization timestamps, sets `explicit=True`, and records channel `iadopt-lab run --authorize`. No `--authorize` option exists (parser 879–884), and `--actor` is optional, defaulting downstream to the generic `operator` (595). The incoming report’s empty-string default is no longer current. Invoking `run` may represent an operator’s deliberate authorization; it still does not prove that the estimate was shown or that the recorded channel/timestamps describe a separate disclosure. The repository requires a non-null actor, not independently supplied identity evidence.

**Trigger/consequence:** a file marked ready with a matching plan hash is enough for `run` to manufacture those receipts without displaying the estimate or receiving the separate receipts described in [execution-evidence.md](execution-evidence.md). A stale/edited estimate can retain a claimed hash and be accepted. This does **not** prove that previously documented live calls lacked real user approval; such approval may have occurred outside the CLI. The defect is the provenance asserted by the public path and its insufficient input validation, not the absence of a mandatory cap.

**The declared live gate is neither satisfiable nor consulted.** `validate_live_readiness` ([configuration.py](../src/iadopt_lab/configuration.py), 187–213) fails unless the caller injects `artifacts_verified`, `database_verified`, `credentials_present`, `estimate_disclosed` and `live_authorized`. Its only production caller is `cmd_preflight`, which passes no facts at all ([cli.py](../src/iadopt_lab/cli.py), 89–97), so `iadopt-lab preflight --live` reports all five gates unmet and returns `EXIT_FAILED` on every input — while [README](../README.md) line 20 documents `--live` as applying “the full dispatch gate”. Only `tests/unit/test_configuration_planning.py` (282–284) ever supplies the facts. `cmd_run` and `_prepare_campaign` never call the function, so the live path enforces its own narrower conditions (draft issues, credentials, a `ready` estimate) and the five named gates constrain nothing.

The new `cmd_probe_models` (700–781) is another live entry point: it fetches a catalog and makes three short calls per selected catalog model without an estimate/disclosure gate or durable probe receipt. `_one_call` retains only reduced fields (answer at most 200 characters, selected usage and reason lengths), not full requests/raw responses or complete usage/cost evidence. This is pre-campaign capability work, **not** a fourth decomposition attempt or benchmark scoring. Nevertheless, deliberate invocation is not itself cost disclosure, and repeat-to-apply instructions run the probes again. A separately defined, explicitly scoped probe contract should retain its own evidence and disclose expected calls/cost, including confirmed zero, before dispatch. No claim is made that an external user failed to authorize any actual probe.

**Future completion criterion:** define and implement an explicit CLI authorization/disclosure contract consistent with D-027, verify the full estimate identity and live billing assumptions, and retain original evidence. Proposed acceptance: the public command with a missing disclosure, missing actor/approval under that contract, edited estimate, or mismatched card must make zero provider requests; a valid separately evidenced authorization must remain traceable to the actual disclosed estimate.

### 9. Token-bound assumptions are documented but not enforced per deployment

**Medium — partially addressed; missing verification and documentation mismatch, not a demonstrated undercount.** Expected: admission assumptions are frozen and defensible for each selected deployment, including protocol overhead and all-in output accounting.

`_live_services.token_bound` ([cli.py](../src/iadopt_lab/cli.py), 366–391) now states byte-level tokenizer, bounded overhead, and no server-side expansion conditions. Its implementation nevertheless uses the same message-byte count plus `_MESSAGE_OVERHEAD_TOKENS` and output ceiling for every model. It does not read or require `tokenization_by_model`. [execution-evidence.md](execution-evidence.md), “Cost-estimate input,” says those deployment-specific records are mandatory and that missing evidence blocks dispatch. Configuration/readiness and estimate code contain no such check. D-039 documents conditions but supplies no enforcement or measured overhead bound.

The new [evidence.py](../src/iadopt_lab/evidence.py) measures rendered **content** with one local tokenizer (`prompt_tokens_by_run`, 61–90), defaults to Qwen3-32B, shares counts across models, and does not apply each deployment’s chat template or freeze tokenizer bytes/revision. These are reproducible only relative to an unstated local tokenizer snapshot, and are not measured provider input counts. `build_cost_evidence` (159–175) takes its ceiling/billing from **current parameters**, attaches the supplied plan hash, and never checks that parameters or prompt artifacts match that plan. An old 100,000-token plan plus current 16,000-token parameters can therefore acquire a nominally plan-bound smaller ceiling; `estimate_campaign_cost` checks the copied plan hash and evidence presence, not each run’s output limit. `--ceiling-verified` can supply a generic assertion without a specific basis. `reasoning_accounting` (138–140) also incorrectly equates `not_applicable` with no expected reasoning, contrary to D-036’s meaning of that mode. The function’s fallback sentence does retain the rule against double-counting observed reasoning.

These are source-confirmed evidence-generation limitations, not demonstrated tokenizer undercounts or charges. Future acceptance must reject old-plan/current-input mixtures, freeze tokenizer identity and deployment overhead, distinguish control absence from reasoning absence, and validate the all-in ceiling against every planned run before calling the result ready.

**Trigger/impact:** a newly selected deployment can be admitted on model capability declarations and price evidence without establishing these conditions. Context/TPM and reservation bounds are then conditional assumptions. The large configured headroom and reported successful runs do not prove a universal upper bound; this review does not assert an observed overflow or incorrect tokenizer-family claim.

**Future completion criterion:** either freeze and enforce evidence for every supported deployment or explicitly reconcile the governing operational contract with the accepted scoped assumption. Proposed acceptance: missing or incompatible per-model evidence blocks before attempt creation, while documented supported profiles calculate a traceable all-in bound, including correction prompts.

### 8. Resume does not finish classifying a stored transient rejection

**Medium — partially addressed; confirmed recovery gap.** Expected: after a crash, a durably received, explicitly safe transient failure with remaining budget resumes the same retry decision without resending its old attempt.

`store_response` sets a rejected response’s task to `operational_failed` ([repository.py](../src/iadopt_lab/persistence/repository.py), 1400–1418). The workflow only subsequently sets cooldown and `retry_pending` ([workflow.py](../src/iadopt_lab/workflow.py), 346–355). `reconcile` (repository 2088–2101) recovers delivered content, ambiguous dispatch, and requests never dispatched, but has no branch for an already stored rejected transient response. `claim_tasks` (876–882) excludes `operational_failed`.

**Trigger/consequence:** attempt 1 receives HTTP 429, its raw rejection commits, and the process stops before the retry transition. After lease expiry, resume retains the receipt but cannot claim that task; a safely recoverable throttling event blocks the campaign even with two requests left. This does not lose raw evidence or produce a model-quality zero. It applies to that interrupted boundary, not uninterrupted cooldown handling. Earlier completed-campaign exit-code and explicit mismatch-before-registration fixes remain addressed in source; no general regression of those fixes is claimed.

**Future completion criterion:** persist or deterministically recover the post-response operational decision and due time, retaining original attempt number and remaining budget. Proposed acceptance: interrupt immediately after a mock 429 receipt commits, resume, wait the prescribed cooldown and allocate attempt 2 exactly once; a 408/ambiguous receipt or exhausted attempt 3 must remain blocked. Also inject interruption after ranking/export/manifest writes. The public default finalizer has idempotent publication support, but its internal `outputs=False` option supplies `files=[]`, which `save_report` rejects (2258–2261); either define a supported no-export contract or remove that misleading option. No current CLI caller sets it false.

### 14. The committed corpus bundle contains a derived projection its own verifier would reject

**Medium — open (new); confirmed artifact inconsistency in the committed snapshot, repaired only in the untracked worktree.** Expected: the corpus bundle recorded in Git regenerates byte-identically from its canonical parents, so `iadopt-lab verify` passes on any clean checkout of that commit.

Each canonical record has two derived siblings. `verify_derived_projections` ([corpus/ingestion.py](../src/iadopt_lab/corpus/ingestion.py), 374–394) rebuilds both with `_metadata_bytes`/`_readable_bytes` and compares **exact bytes**, raising `ValueError` on any difference. The metadata projection emits the fixed field order in `_METADATA_FIELDS` (280–284), which includes `definition_predicate` between `record_sha256` and `importer_version`, and serializes with `sort_keys=False` (350–359).

At Git HEAD `cf5ec8af6ad9b61fa2ef2b856249033a2784aae9` (“experiment dry run passed”), `data/canonical/v2.0.1/Social Sciences/Demography/highest education.meta.json` omits `definition_predicate` entirely, while its canonical parent `highest education.json` carries `"definition_predicate": "http://www.w3.org/2000/01/rdf-schema#comment"` and all 101 other committed metadata files carry the key. The regenerated bytes therefore cannot equal the committed bytes for that one file. The working tree adds exactly that line, in exactly that position — the sole difference in `git diff` for the whole `data/canonical/` subtree — so the current worktree is self-consistent and the committed snapshot is not.

**Trigger/consequence:** clone or check out `cf5ec8a` cleanly and run `iadopt-lab verify`; it is expected to raise “Derived projection does not match its canonical parent” on that path and return non-zero, because the untracked repair is not in the commit. Anyone reproducing from the tagged history therefore cannot pass the corpus gate that every planning command depends on. Nothing about the **science** is affected: `load_canonical_records` reads only the `.json` records and never the `.meta.json` siblings ([ingestion.py](../src/iadopt_lab/corpus/ingestion.py), 743), the record and gold hashes are unchanged, and `record_sha256` is identical on both sides of the diff. This is a provenance and reproducibility defect, not a corpus, gold or dataset-version error.

**Scope and uncertainty:** established by byte comparison of the committed file, the working file, the canonical record and the projection field list. `verify` was **not executed**, here or against the commit; the failure is deduced from those bytes and the exact-comparison code, not reproduced. Why the committed file lacked the field, and whether the worktree line was hand-written or regenerated, is not established — the worktree is user-owned and was not altered.

**Future completion criterion:** the corpus bundle recorded in version control must regenerate byte-identically from its own canonical parents. A future fix must land the repaired projection (or a regenerated bundle) in the commit, so a clean checkout verifies. Proposed acceptance, not performed here: check out the commit into a scratch clone, run `iadopt-lab verify`, and require `records: 102` with `derived_files_verified: 204`; additionally assert in CI that every `.meta.json` regenerates from its parent before the corpus is treated as frozen.

## Working behavior and acceptance boundaries to preserve

| Workflow boundary | Current source evidence and interpretation |
|---|---|
| Configuration → planning | `configuration.load_parameters` rejects duplicate YAML mappings/aliases and validates the strict schema; resolution filters selected providers/enabled models and permits D-033 reasoning subsets. `planning.expand_campaign` enumerates provider-owned conditions and one-repetition settings with deterministic identities. |
| Corpus → gold/population | `ingestion.enumerate_tag_files` checks pinned tag/commit/tree and reads Git blobs; `parse_variable` enforces cardinalities and explicit labels/targets. `project_gold` validates lexical structures. `load_canonical_records` checks source inventory, schema/record/gold hashes and saved population reconstruction. The source contains no train/test split. No full corpus re-execution was performed here. |
| Prompts → requests | `prompting.renderer.load_prompt_version` checks registry hashes; `render_base_prompt` verifies ordered prefixes and gold hashes, inserts only definition/schema/demonstrations; `render_correction` preserves the base and immediate prior answer/errors. `providers.base.build_request` sends one textual user message with explicit sampling/native reasoning fields. |
| Dispatch → evidence | `start_attempt` and `mark_dispatched` provide write-ahead numbering and dispatch guards. Schema uniqueness, attempt checks and migration 0002 prevent resetting the counter or reopening terminal delivery. SDK `max_retries=0`, HTTP transport retries zero and redirects disabled implement the no-hidden-retry policy at this boundary. Limitations are findings 1–3. |
| Extraction → validation | `extract_json` uses strict parse, one string unwrap, eligible fences, then balanced objects, with duplicate-key/non-finite/depth/size checks and evidence. `validate_prediction` enforces six-field schema, system shapes, member uniqueness and unambiguous emitted Constraint targets. It does not compare against target gold or add external lookup. |
| Prediction → scoring | `_validate_and_select` selects a valid prediction or requires three content-invalid receipts for explicit empty; operational exhaustion remains unscored. `_score` reuses a stored prediction/evaluation. Repository selection independently checks terminal-empty eligibility. |
| Scoring → ranking | `iadopt_eval.core` preserves scalar wrong-nonempty FP-only behavior, literal symmetric/symmetric equivalence, role-aware asymmetric matching, maximum-cardinality member assignment, historical greedy Constraint ties and numerical correction. Contributions/metrics use fraction receipts; reporting averages repetition micro scores and assigns shared competition ranks. These unusual rules are accepted scientific choices. |
| Finalization → recovery | `_advance` finalizes both successful stop reasons. `finalize_campaign` persists ranking, exports, saves a report manifest, then completes. `export_report` uses immutable atomic publication and accepts identical files; a missing sibling manifest after publication can be recreated. `reconcile` reuses delivered content/predictions and leaves unresolved dispatch ambiguous; stored operational decisions have the gap in 8. Source supports these paths; reported idempotent dry-run evidence is not a fault-injected recovery verification. |

The retained January source was independently **hashed**, not executed; it matches the documented SHA-256 `2d4a6271fd86a076127dfb3f85589391cff9744efdfdee8ec570b52c74921df0`. The embedding adapter verifies explicit local artifact files and uses local-only CPU/float32 loading. Neither actual embedding bytes nor numerical parity on a live campaign were executed in this review.

### Scientific clarification: ambiguity across extraction stages

The extractor returns as soon as one eligible fence parses; it does not inspect another unfenced object outside that fence (`generation/extractor.py`, `extract_json`, fence-success return). Thus a fenced valid six-field object followed by a conflicting unfenced six-field object can select the fenced answer. Multiple objects within the balanced-scan stage or multiple parsed fences are rejected.

The specification §11.2 explicitly prioritizes fences before balanced scanning, but also requires no competing parsed JSON value; the overarching experiment requirement rejects ambiguous JSON. Whether uniqueness is stage-local or response-wide is not unambiguously settled by that wording. This is a **scientific protocol clarification**, not an assertion that the implementation disobeys its explicitly ordered stages. Preserve the existing behavior until the intended scope is clarified and version any resulting change. A meaningful proposed acceptance case contains one valid fenced object and one conflicting unfenced object; another contains malformed outer JSON with an embedded valid object. Record the intended decision and extraction evidence for each rather than silently choosing a convenient answer.

## Test evidence and gaps

The appended log reports **312 passed, no skips**, clean lint, corpus verification, zero preflight issues, and repeat dry-run/resume with the same ranking ID. Those reports are preserved exactly below. They are useful reported acceptance evidence but do not establish fault recovery or every live boundary in this snapshot.

Source inspection distinguishes the following:

- [workflow/reporting tests](../tests/unit/test_workflow_reporting.py) exercise the eight envelope cases, isolated `_commit_response` with a fake repository, observation projection, category missing-member behavior, supplied hash checks, and synthetic artifact collection. The delay test checks the exception attribute, not the runner's scheduling. The population test calls the helper, not both public CLI paths. The “wrong backend” test has empty observations.
- [Persistence integration tests](../tests/integration/test_persistence.py) exercise actual repository registration, concurrent claims, immutability, three-invalid selection, provider isolation, admission, settlement and complete ranking with fixtures. They do not call the complete campaign loop or live services. [Recovery tests](../tests/recovery/test_repository_recovery.py) manually establish database checkpoints and reconcile them; they do not simulate HTTP receipt plus heartbeat failure.
- [Scorer parity tests](../tests/scorer_regression/test_january_parity.py) isolate hash-verified January AST functions and compare synthetic cosines/Constraint cases. [Core tests](../tests/unit/test_eval_core.py) include hand-calculated member examples and an exhaustive small assignment oracle. This is meaningful algorithm coverage, distinct from frozen-model numerical parity.
- [Extractor tests](../tests/unit/test_extractor.py), [validation tests](../tests/unit/test_validation.py), [prompt tests](../tests/unit/test_prompting.py), and [corpus tests](../tests/unit/test_corpus.py) cover their local contracts. Their existence is not proof of complete workflow recovery.

Highest-value missing acceptance cases are received-response preservation during supervisor cancellation; malformed metadata/negative usage through the real transaction; CLI artifact reconstruction; public reporting with swapped provenance and correct population denominators; end-to-end accounting units/projection; explicit estimate/disclosure validation; and interruption after each finalization write, including data-file-without-manifest recovery. No direct automated `run_campaign`/`finalize_campaign` integration call was found in the inspected test tree. A source-reviewed helper or reported happy-path dry run should not be described as that missing integration coverage.

The new [probe tests](../tests/unit/test_probing.py) exercise mock transport verdicts and generated YAML resolution; they omit a failed baseline followed by a successful switched call, actual structured-JSON validation, and probe-to-generation endpoint binding. The new [evidence tests](../tests/unit/test_evidence.py) verify helper output, but some load a real locally cached tokenizer without injection; successful execution depends on that undeclared local prerequisite. They do not verify stale-plan rejection or per-deployment counts. The test for `reasoning_accounting` checks that mode names appear, not the erroneous claim about `not_applicable`.

A test isolation limitation also deserves preservation: `test_verify_detects_a_tampered_population_manifest` in the workflow/reporting test file (140–153) writes the actual repository manifest inside try/finally, despite accepting `tmp_path`. A normal completion restores it, but process death or concurrent execution can leave/interleave changes. Proposed verification should use an isolated copied corpus; no test or manifest mutation was performed here.

### Artifact evidence read during this pass

These are files produced by earlier runs and inspected read-only here. They are stronger than source deduction for the points they cover and weaker than a reproduced execution for everything else; none was regenerated, and no database was queried to corroborate them.

| Artifact | What it establishes | What it does not establish |
|---|---|---|
| [`outputs/ranking-ee138ed5-…json`](../outputs/) | A **completed live** ranking exists: `mode: live`, 97 targets, one configuration `psnc/GLM-5.2` at `reasoning_mode: disabled`, `complete: true`, `scorer_binding: verified-against-plan`, primary mean-repetition micro Close F1 `0.3865546218487395` retained as an exact unrounded fraction. 105 attempt records, all `response_received`. Confirms finding 6's cost projection in produced output (26 summaries, 315 attempt slots, no monetary evidence). | That the stored campaign, its settlement rows or its scores are correct; that this configuration is the one now planned. |
| [`outputs/plan.json`](../outputs/) | The currently staged live plan: `experiment-plan-v1`, `mode: live`, 97 population, 3 configurations, 291 tasks, 291 initial and at most 873 requests. | That it has been registered, authorized or run. |
| [`outputs/evidence.json`](../outputs/) | The `evidence` command has been exercised: one prompt shape (strict-minimal, 5 shots), tokenizer `Qwen/Qwen3-32B`, ceiling 16,000, `ceiling_verified: true` with the operator basis “GLM-5.2 with reasoning off peaked at 197 completion tokens against a 16,000 ceiling”. That basis is an operator assertion recorded by `--ceiling-verified`, and it independently corroborates that the reasoning switch changes generation on this deployment. | That the assertion was independently measured, or that the tokenizer counts match PSNC's own accounting (finding 9). |
| [`outputs/estimate.json`](../outputs/) | A `ready: true`, `issues: []`, USD, zero-cost estimate covering all three models — the input `_authorize` would accept. | That the models it prices are dispatchable; the execution price card covers only `psnc/GLM-5.2` (finding 6). |
| Two `ranking-*.json` files at 3 targets | `mode: synthetic`, `synthetic: true`, `complete: true` — the labelled dry runs. | Anything scientific; they are fixtures. |

Each ranking carries a sibling `.manifest.json`. The two newer manifests record `exporter_code_sha256` `9e990cec…`, which equals the current `src/iadopt_lab/reporting.py` hash; the oldest records `a8bcf8d7…`. Export lineage is therefore traceable to the exporter bytes, and the newest exports were produced by the reporting code reviewed here. That is an observation about lineage, not a defect.

## Suggested future repair and verification order

This is a proposed sequence only; no implementation or execution is authorized by this report.

1. Make received evidence survive metadata/accounting failures and supervisor cancellation (1–2); verify with real persistence and controlled provider fixtures.
2. Persist complete campaign artifacts and establish explicit stored-snapshot resume/reconstruction (11).
3. Close malformed completion and report-attribution boundaries (3–4), and validate probe-derived capability claims and the unresolved decision history (13, 10); agree on cross-stage extraction ambiguity before changing the scientific protocol.
4. Reconcile selected-model price coverage, currency units and settlement reporting (6), then verify disclosure and plan-bound token/ceiling evidence at actual CLI entry points, including probes (12, 9).
5. Repair interrupted transient-response decision recovery (8); exercise full offline generation-to-finalization recovery, both providers' isolation, and no-regeneration/no-fourth-call invariants. Reconcile current documentation and dated acceptance records (10), and land the corpus-projection repair in version control so a clean checkout verifies (14).

Do not reissue generation merely to repair reports or accounting. Existing database evidence, where available, should support local reconciliation; this review did not inspect individual stored campaigns to determine what can be recovered.

## Previous source identity and revision record — retained history

The following snapshot and note belong to the incoming review. Their statement that the lab was wholly untracked describes that earlier account, not the current Git state. Current identity and this pass’s corrections follow below.

Git HEAD was `c2ff4414fd85760184199089f1dab78204edf2e9`. The entire `iadopt-lab/` was untracked, so HEAD alone does **not** identify this implementation. Initial Git status also showed unrelated changes in the parent repository and an external submodule; none were altered. The following SHA-256 values identify current reviewed bytes, relative to `iadopt-lab/`:

| File | SHA-256 |
|---|---|
| `src/iadopt_lab/workflow.py` | `1ab8904bf07302441742d74df018f0e167e345f9a0edfa5fc16165e45be941fa` |
| `src/iadopt_lab/cli.py` | `e5fca57926450d3db2e4f39b679678cdce44c78e705fee3586ef2942cbc6c351` |
| `src/iadopt_lab/providers/base.py` | `bf960c8363bd119a66bfff5a415c82f06bcd43ed01cb3e591403cbb630ef5536` |
| `src/iadopt_lab/reporting.py` | `9e990cecdd92821b73c4d5d9d8a4d171d7eccc31192f11c888f081ae73d2a55a` |
| `src/iadopt_lab/persistence/repository.py` | `db8ae78b044ed835c43e8c3ce70bd38945227f3a3d8d3f0aabc1780983abcee3` |
| `src/iadopt_lab/artifacts.py` | `077ad21957059e90b4632cb06dc2a63dbacfc173ff17b657b545f8cf27711cd7` |
| `parameters.yml` | `0576be9562e6fe4130d6970e59230afd97fda4382f1f918b8c614c1b2d4fcad8` |
| `TECHNICAL_SPECIFICATION.md` | `f2d2ee2b2d33ecea4998c0100c59233a874973dea9e0b5a154b6d412b18fc3df` |
| `tests/unit/test_workflow_reporting.py` | `c83faa14f675d1693629bcd4c3bf8bdf443f6174325fbef3a8a5e078f0462bbd` |

**Revision note — 9 September 2026:** reread the prior report and its appended author log; reassessed findings 1–10 against current source; closed their corrected mechanisms, retained residual issues under stable identifiers, and added 11–12 for previously unreported public-path gaps. Resolved the specification-front-matter account against current bytes without rewriting the historical explanation. Preserved all incoming report bytes below, including reported demonstrations and acceptance results. No implementation or scientific decision was changed.

## Current source identity and revision note — 9 September 2026

Current Git HEAD: `cf5ec8af6ad9b61fa2ef2b856249033a2784aae9`. The lab and this report are now tracked; the initial worktree already had modified CLI, parameters and one derived metadata file, untracked evidence/probing modules and their tests, and a dirty external submodule. Those are user-owned. This review edited only this report. The metadata diff adds `definition_predicate`, which agrees with `_METADATA_FIELDS`; it is not evidence of changed gold or a dataset-version error.

The parameter file was observed at SHA-256 `9552345b9ce364898705d2179a608fc7f920f51a2ffa0a7ee32d600bedb39d4d` (single GLM scope) and subsequently `4199448ec074237d2fe557e988632306cb2cb476492d85e4331ea81401717629` (three-model scope). These changes were not made by this review. Hashes identify bytes; no executed plan, model capability, database content or test success follows from hashing.

| File, relative to `iadopt-lab/` | SHA-256 |
|---|---|
| `src/iadopt_lab/workflow.py` | `1ab8904bf07302441742d74df018f0e167e345f9a0edfa5fc16165e45be941fa` |
| `src/iadopt_lab/cli.py` | `6d9346e36ba25572cea910fc5ae082b5aa46f3d761f9213d6b4f4dd4e19be937` |
| `src/iadopt_lab/providers/base.py` | `bf960c8363bd119a66bfff5a415c82f06bcd43ed01cb3e591403cbb630ef5536` |
| `src/iadopt_lab/reporting.py` | `9e990cecdd92821b73c4d5d9d8a4d171d7eccc31192f11c888f081ae73d2a55a` |
| `src/iadopt_lab/persistence/repository.py` | `db8ae78b044ed835c43e8c3ce70bd38945227f3a3d8d3f0aabc1780983abcee3` |
| `src/iadopt_lab/artifacts.py` | `077ad21957059e90b4632cb06dc2a63dbacfc173ff17b657b545f8cf27711cd7` |
| `src/iadopt_lab/evidence.py` | `f83b2e387cd70682cf33cbbfdb8e7c6f417af72921c4fc38edc4b25b65a2e05d` |
| `src/iadopt_lab/probing.py` | `df36fbe50061d2bf66c3cfc50f233c793c35dc171828def381f566ff46380d22` |
| `src/iadopt_lab/costing.py` | `e678f08e66f1f645505ebf99da6ca52d764d59858d4adb93d18fcae0cd4dd02d` |
| `parameters.yml` | `4199448ec074237d2fe557e988632306cb2cb476492d85e4331ea81401717629` |
| `TECHNICAL_SPECIFICATION.md` | `f2d2ee2b2d33ecea4998c0100c59233a874973dea9e0b5a154b6d412b18fc3df` |
| `DECISIONS.md` | `32a321e0b1ae96b88be627f2c6ed0b428e6e3d6c5aabace622d8c21e23a5d9d1` |
| `tests/unit/test_workflow_reporting.py` | `c83faa14f675d1693629bcd4c3bf8bdf443f6174325fbef3a8a5e078f0462bbd` |
| `data/manifests/price-card-glm-5.2-psnc.yml` | `404395a20b83f76377362d55560113e1affe6bf916fdbca6547620cae7a54526` |
| `tests/unit/test_evidence.py` | `a05806445296a3ea6647f9d33376900e68e91947ed40f59b4099ef54b1b4fbec` |
| `tests/unit/test_probing.py` | `9466f97e306cc552f6737afacda777b5d0d6677f20e14761ad4357d85d8ea0b9` |
| `src/iadopt_lab/configuration.py` | `69bf69dc83be692b2d03b11832539ce40f955a3f29cad6501042c0d44f74ee5b` |
| `src/iadopt_lab/corpus/ingestion.py` | `1d8f53a8b39bb8fe8a8bc52b03cceea04baefa3fe30e5cfc8f9304a32ee89907` |
| `README.md` | `3e4e70501af3e39df4c7045750b2914bf57d0ff493a8a7c1a392d1c29f46ce2b` |
| `docs/runbook.md` | `abcba1c8768c6132412634658885d3f7f814ba54b0316ab78ade407750cb39d7` |
| `docs/execution-evidence.md` | `ddb469fcf66fa2cfe09af917b779c885331e6cbdb9d45841252f78d247ec80a7` |

The five rows above were added by the later pass described in the second revision note; the rows before them are unchanged.

**Revision note (first pass this date):** independently reread the incoming report and author log, current pipeline and new CLI helpers. Retained identifiers 1–12, added 13 for probe-capability inference, and linked the new transient-rejection recovery gap to 8. Retired obsolete current claims about scope, actor default and Git tracking. Distinguished the D-040/D-041 absence from the resolved historical specification-front-matter discrepancy. Kept received-evidence risks foremost, added current price-card coverage and stale-plan estimate checks, and preserved all existing author resolution evidence. No implementation, test, configuration, data, service, database or Git-state change was made by this reviewer; no model call was performed.

**Revision note (second pass, 9 September 2026) — additions only.** A further read-only pass re-verified the findings above against the same bytes rather than accepting them, and extended the report by focused edit; nothing was removed. Independent rechecks confirmed the load-bearing claims of the first pass, including the `_probe_one` ordering defect in 13 (a failed baseline reaches the `never_reasons` branch and is written `usable: true`), the missing `base_url_override` at `psnc.create_adapter` in 13, the absent `reconcile` branch for a stored rejected transient response and the `claim_tasks` state filter in 8, and the `save_report` empty-`files` rejection under `finalize_campaign(outputs=False)`. None is withdrawn.

Four things are added. Finding **6** gains observed artifact evidence — the exported live ranking records no monetary evidence in any of its 26 cost summaries — plus the reason nothing intercepts the price-card gap: `require_complete_price_evidence_before_live` is declared in `parameters.yml` and read nowhere in `src/`, and `cost_policy` raises outside any `try` at `workflow.py` 300. Finding **12** gains the live-gate leg: `validate_live_readiness` is called only by `cmd_preflight`, always without facts, so `preflight --live` can never pass, and `cmd_run` never calls it. Finding **10** gains the undocumented `probe-models` and `evidence` commands. Finding **14** is new: the committed corpus bundle at `cf5ec8a` holds one derived projection that `verify_derived_projections` would reject, repaired only in the untracked worktree. A new *Artifact evidence* table records what the files under `outputs/` do and do not establish, and separates that grade of evidence from source deduction and from execution.

The concurrency limitation stated at the top of this report applies to this note as well: `parameters.yml` and this report file both changed through another writer during the combined pass, and the hashes recorded here identify the bytes actually inspected. No test, experiment command, database operation, service operation, model call, dependency install or Git-state change was performed by this reviewer, and only this report was written.

## Historical appendix — preserved incoming report and author resolution log

**Archive boundary. Everything after this notice is the incoming report verbatim, not the current assessment.** Its SHA-256 before this revision was `8565588309b281c2f0520bf242e0d10f9126aa6185a4280c888431d717332056`. In particular, both the old open-finding prose and the later “Resolved” table remain historical accounts; use the current table above for present status.

---

# I-ADOPT Lab: updated implementation review

Review date: 9 September 2026. Follow-up revision incorporating the supplied verification notes and a fresh read of the local source. Finding numbers are retained so earlier discussion and resolution notes remain traceable.

## Purpose and conclusion

This report explains the intended experiment, the problems addressed since the previous review, and the problems still present in the inspected code. It is written to be understood without the project conversation.

**The implementation has improved substantially, but it is not yet fully verified against the experiment requirements.** Finalization, accounting, and verification now have connections that were previously absent. Remaining issues concern response preservation, failure classification, reporting checks, and reliable resumption.

**The first repair priorities are findings 2 and 1.** Billing interpretation can fail before a received response is stored, and ordinary database outages bypass the response-storage retry handler. Both can leave an already received, potentially paid answer without durable evidence. Classification and reporting integrity follow because they affect whether scores mean what the experiment claims.

This was a source review. No tests, model calls, database queries, or service operations were performed. Only this report was rewritten. Failure examples describe consequences of code paths; they do not establish that a particular stored experiment was affected. Test totals and live-run descriptions in project documents were not independently reproduced.

Links are relative to this document for portability. Function names identify the relevant source logic.

### Evidence and revision history

The supplied follow-up verification confirms the implementation issues and reports demonstrations of four malformed/truncated response cases. Those demonstrations are recorded here as **reported verification**, not as tests executed by this review. The source was independently reread to check the corresponding branches.

The supplied notes dispute one documentation statement: whether the technical specification still says implementation has not started. The file in this checkout does contain that exact statement in its front matter. Finding 10 records the path, line, quote, and hash rather than assuming either account applies to every checkout. This discrepancy does not change the response-preservation or scoring findings.

The previous report's table of addressed and partially addressed issues is retained below. No separate original resolution log was present in the file at the start of this revision; this report does not claim to reconstruct a missing log verbatim. The newly supplied verification notes are preserved in substance in the findings and status record. Future revisions should append resolution evidence and change finding statuses without erasing that history.

| Finding | Follow-up status | Evidence qualification |
|---|---|---|
| 1–2: response preservation | Open; urgent | Source confirms both exception mismatch and billing-before-storage ordering; supplied verification agrees. |
| 3: classification | Open; partially repaired | Source confirms four remaining cases; supplied notes report demonstrations. |
| 4–8: reporting, scheduling, accounting, identity, resume | Open | Source rechecked; supplied verification agrees. |
| 9: token bound | Open evidence requirement | No undercount is claimed; deployment and overhead assumptions remain insufficiently established. |
| 10: documentation | Open; specification subclaim disputed across accounts | Current local bytes support the quoted front-matter finding; other inconsistencies are also confirmed in the supplied notes. |

## What the experiment should do

The experiment uses **Corpus v2.0.1**, converts its 102 Turtle variables into deterministic lexical JSON ground truth, and retains release hashes and science categories. Five ordered variables are demonstrations; the other 97 form the evaluation population. There is no train/test split.

Each task identifies a variable, provider/model, prompt, demonstration count, temperature, reasoning profile, and repetition. Providers own their model lists. The original broad campaign uses one repetition at each temperature; later decisions also describe smaller campaigns and deployment-specific reasoning limitations.

The model returns six lexical fields. Full requests, raw responses, extraction results, validation errors, and retry history must be preserved. A task permits at most three provider requests. Exhausted content-invalid model responses produce an explicit empty prediction; operational failures must remain distinguishable from model mistakes.

Scoring retains January-derived behavior with approved fractional system-member credit and excludes arbitrary system names. Exact and Close Precision, Recall, F1, and their underlying counts are retained. Complete configurations are ranked by mean repetition-level micro Close F1 across the same 97 variables, with unrounded values and shared ties. Interrupted work must be resumable through final reporting.

RDF, JSON-LD, SHACL, entity linking, and the deterministic baseline are deferred. Their absence is intentional. v2.0.1 is confirmed and is not an unresolved decision.

## What changed since the previous review

These are changes observed in the source, not changes made during this review.

| Earlier problem | Current assessment |
|---|---|
| Execution did not finalize reports | The normal path now calls finalization, saves rankings/reports, and marks completion. |
| Cooldown datetime caused a type error | Corrected: strings are parsed and datetime objects are accepted directly. |
| Rate-limit exception escaped the runner | A handler exists, but reads the wrong retry-delay field. |
| Response storage was attempted only once | A retry helper exists, but does not catch ordinary database connection errors. |
| HTML gateway responses were scored as model content | Corrected for HTML, invalid JSON, and non-object JSON. Malformed object envelopes still pass. |
| Half a token per byte was called an upper bound | Replaced by one token per byte plus overhead. Deployment-specific justification remains incomplete. |
| Cost settlement was unwired | Settlement is connected; currency handling and its position before raw storage need attention. |
| Reporting lacked identity and category checks | Optional checks exist; the standalone report command omits them. |
| Saved manifests were not verified | Both saved hashes are checked, and population membership is compared with canonical reconstruction. |
| No workflow/reporting test file existed | A new file covers selected helper regressions. It does not execute the complete campaign/recovery path. |

Resolved findings should not continue to be presented as though these improvements had not happened.

## Remaining problems

### 1. Received responses are still vulnerable to database outages

**Priority: High. Confirmed exception mismatch.**

Expected behavior is to retain a received response and retry transient storage failures without making another model request.

The new `_commit_response` catches only the project's `PersistenceError`. However, `store_response` uses psycopg directly, and `_db` propagates original exceptions. Ordinary psycopg connection errors and pool timeouts are not automatically converted into that custom exception, so they can bypass the retry loop. The inverse is also a defect: `EvidenceConflict` inherits from `PersistenceError`, so the same permanent integrity conflict can be pointlessly retried four times. Waiting will not resolve two different payloads claiming the same immutable identity.

Even a caught error is retried only four times, with 7.5 seconds of scheduled waiting in total. There is no durable fallback for an outage that lasts longer.

**Why this is a problem:** a paid response can disappear from application memory before its raw bytes, usage, and validation evidence reach the database. A dispatch marker helps prevent an unsafe duplicate request but cannot reconstruct the answer.

**Evidence:** [workflow.py](../src/iadopt_lab/workflow.py), `_db` and `_commit_response`; [repository.py](../src/iadopt_lab/persistence/repository.py), exception definitions and `store_response`.

**Verification needed:** inject actual database/pool failure types after a mock response arrives, including a longer outage. Prove that the original bytes are retained and provider call count remains one. Treat permanent integrity errors separately.

### 2. Billing interpretation can prevent raw-response storage

**Priority: High. Confirmed ordering risk.**

The runner calls `settlement_policy` before `_commit_response`. The policy parses provider cost fields and pricing data and can raise an exception. For example, a nonnumeric `usage.cost` can fail decimal conversion before raw persistence is attempted.

**Why this is a problem:** the evidence needed to diagnose an unfamiliar billing response can itself be lost because interpreting that response failed. Raw preservation should not depend on successful billing interpretation.

**Evidence:** [workflow.py](../src/iadopt_lab/workflow.py), `_advance_task`; [cli.py](../src/iadopt_lab/cli.py), `_live_services.settlement_policy`.

**Verification needed:** malformed billing fields must still leave exact raw response evidence stored, with accounting marked unresolved rather than silently discarded.

**Repair boundary:** preserving the raw response first will require a compatible accounting-persistence design. Simply moving one function call is not sufficient if the existing immutable response row can no longer accept later settlement evidence. A future fix should preserve the original raw receipt and append traceable accounting outcomes without rewriting it or double-counting expenditure.

### 3. Some invalid API envelopes and truncated answers still enter content scoring

**Priority: High. Partially corrected classification defect.**

The adapter rejects HTML, unparsable JSON, and JSON arrays. However, a dictionary such as `{}` or `{"error":{"message":"gateway failure"}}` passes the envelope-type check. With no valid completion, it becomes a delivered `empty_response`. Three such responses can be scored as an empty model prediction.

Also, `finish_reason="length"` is operational only when assistant content is empty. A nonempty partial JSON answer truncated at the same limit still enters content-invalid retries.

The supplied verification reports these four cases, all consistent with the inspected branches:

| Response body or condition | Current routing | Why that routing is problematic |
|---|---|---|
| `{}` | Delivered empty response → content validation | No actual completion envelope has been established. |
| `{"error":{"message":"gateway failure"}}` | Delivered empty response → content validation | A provider error is treated as model output. |
| A choice with no `message` | Delivered empty response → content validation | A missing completion structure is treated as a genuine empty answer. |
| `finish_reason="length"` with nonempty partial JSON | Delivered response → content validation | An explicit capacity stop is handled as a JSON-generation mistake. |

These cases do not imply that every truncated response should trigger an extra generation request. The maximum-three-provider-request rule still applies; classification and permission to retry are separate decisions.

**Why this is a problem:** gateway faults and capacity failures can reduce the measured model score. A genuinely empty, well-formed completion is a separate case and should retain its documented treatment. Raising the output ceiling reduces the likelihood of truncation but does not fix classification.

**Evidence:** [providers/base.py](../src/iadopt_lab/providers/base.py), `send_once`; [workflow.py](../src/iadopt_lab/workflow.py), `_validate_and_select`.

**Verification needed:** distinguish missing choices, error objects, malformed messages, valid empty completions, and nonempty truncated JSON.

### 4. The standalone report command bypasses new scientific checks

**Priority: High for identity; Medium for coverage.**

`build_configuration_ranking` accepts optional scorer identity and population-category metadata. The public `cmd_report` supplies neither. It therefore checks evaluations for agreement with each other without binding them to a supplied frozen backend, and builds category summaries from observed rows.

The supplied-identity path also permits a missing `plan_scorer`. A self-consistent identity is therefore not necessarily proven to belong to the plan. Normal finalization improves this by supplying bundle similarity evidence, but takes scorer version and threshold from an evaluated task.

**Example:** a category expected to contain ten variables can appear complete when only eight scored rows are provided. The new `observed-rows-only` label helps disclose the basis but does not establish full category coverage. Entirely absent categories produce no rows.

**Why this is a problem:** scientific checks depend on the entry point used. Whole-configuration ranking still checks the full plan population; this finding does not claim that missing configurations are ranked as complete.

**Evidence:** [cli.py](../src/iadopt_lab/cli.py), `cmd_report`; [reporting.py](../src/iadopt_lab/reporting.py), `build_configuration_ranking` and `build_category_summary`; [workflow.py](../src/iadopt_lab/workflow.py), `finalize_campaign`.

**Verification needed:** exercise the actual standalone command with wrong-backend evidence, missing category members, and wholly absent categories. Scientific checks should be required at every public reporting boundary.

### 5. The runner ignores the actual rate-limit delay

**Priority: Medium. Confirmed field mismatch.**

`RateLimitError` stores the delay in `retry_after_seconds`. Its `args[0]` contains a descriptive string. The runner looks for a number in `args[0]`, then falls back to one second, so it never uses the supplied delay.

**Why this is a problem:** a provider needing a longer cooldown can be repeatedly reclaimed and rejected, producing unnecessary database work and state transitions. Repository admission still protects the limit; this is not evidence that requests exceed it.

**Evidence:** [repository.py](../src/iadopt_lab/persistence/repository.py), `RateLimitError.__init__`; [workflow.py](../src/iadopt_lab/workflow.py), rate-limit handler.

**Verification needed:** a 30-second advised delay should schedule approximately that delay, consume no attempt, and leave other providers eligible.

### 6. Cost paths handle currency conversion inconsistently

**Priority: Medium. Confirmed issue for configurable currencies.**

Usage-based settlement applies `fx_to_reporting`. Pre-dispatch reservations do not. Provider-reported actual costs are also stored without explicit currency conversion or validation.

**Why this is a problem:** if provider amounts are in USD and reporting is in EUR, reservation and settlement values can represent different units. USD 1 at a conversion factor of 0.9 should not become EUR 1.

The inspected price cards use USD and factor 1, and current reporting is USD. This is therefore a defect for supported future configurations, not a demonstrated error in the current card's totals.

**Evidence:** [cli.py](../src/iadopt_lab/cli.py), `_live_services.cost_policy` and `settlement_policy`; [parameters.yml](../parameters.yml); [price manifests](../data/manifests/).

**Verification needed:** use a non-unit exchange rate and explicit provider billing currency. Estimates, reservations, actual costs, and settlements must agree on stored units and retain conversion evidence.

### 7. Synthetic plans contain conflicting population identities

**Priority: Medium. Confirmed provenance mismatch.**

Both `cmd_plan` and `_prepare_campaign` collect artifact identities using all 97 targets, then pass only three targets into the synthetic planner. The artifact called `population` hashes 97 IDs while the explicit synthetic plan population contains three.

**Why this is a problem:** the selected test population and its full-benchmark source should have distinct identities. Conflicting meanings weaken the dry run's ability to exercise the scientific identity contract. This does not mean 97 synthetic targets are dispatched.

**Evidence:** [cli.py](../src/iadopt_lab/cli.py), `cmd_plan` and `_prepare_campaign`; [artifacts.py](../src/iadopt_lab/artifacts.py), `collect_input_artifacts`.

**Verification needed:** the selected-population hash must match the three planned IDs. Preserve full-benchmark lineage in a separately named field if needed.

### 8. Completed-campaign resume reports failure, and mismatched resume can create records

**Priority: Medium. Confirmed control-flow mismatches.**

The runner returns `already_complete` for a completed campaign. `cmd_resume` returns success only for `tasks_complete`, so successfully recognizing completion produces a failure exit code.

Resume also calls `_prepare_campaign` before checking the requested campaign ID. That reconstructs and registers a campaign from current files. With changed configuration, a rejected resume can create another campaign's registration/planning records before reporting the mismatch.

**Why this is a problem:** automation can treat completed work as failed, and attempting to resume one campaign can create another campaign's metadata. These branches alone do not establish duplicate paid requests.

**Evidence:** [cli.py](../src/iadopt_lab/cli.py), `cmd_resume` and `_prepare_campaign`; [workflow.py](../src/iadopt_lab/workflow.py), `run_campaign`.

**Verification needed:** completed resume should succeed without new work. A mismatched current configuration should either use the stored snapshot or reject before registering another campaign.

### 9. The revised token bound still needs deployment-specific justification

**Priority: Medium. Evidence limitation, not a demonstrated undercount.**

One token per UTF-8 byte plus fixed overhead is more conservative than the previous rule and can be justified for suitable byte-level tokenizers. However, it is applied across configurable deployments while the PSNC tokenizer is described as unpublished. Actual chat-template and special-token overhead also need a documented basis.

**Why this is a problem:** a tokenizer-family property does not independently establish the provider's full request accounting. Admission bounds should state the conditions under which they hold. This review does not claim the new formula undercounted an observed request.

**Evidence:** [cli.py](../src/iadopt_lab/cli.py), `_live_services.token_bound`; [execution-evidence.md](execution-evidence.md), per-model bound requirements.

**Verification needed:** freeze the evidence and overhead assumptions for each supported deployment; check them when adding models.

### 10. Documentation and acceptance claims still conflict

**Priority: Medium. Confirmed inconsistencies.**

The README now acknowledges documented live runs and reports 299 passing tests with PostgreSQL. The earlier claim that it denied all live calls is closed. Remaining contradictions include:

- The README database section still says PostgreSQL tests have not run.
- The supplied verification says the technical-specification status was already fixed. However, a fresh read of `/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service/iadopt-lab/TECHNICAL_SPECIFICATION.md`, line 4, returns exactly `status: Documentation baseline; implementation not started`. The SHA-256 of the inspected file is `f0ed6d05f0bbb841ea192e4af5f619e7299e213b4b2cb367fd73213c3283e09b`. This is a confirmed stale statement in these local bytes and a discrepancy with the supplied account, not proof that another version was never corrected. No cause for that discrepancy was established.
- The implementation record retains older acceptance/coverage claims.
- README refers to D-039, while the inspected decision headings end at D-038.
- D-038 describes truncation as wholly uncorrected, although empty truncation now has special handling. It also implies a high finite ceiling guarantees an answer; such a ceiling can still be reached and cannot guarantee valid JSON.

The new workflow/reporting test file covers selected helpers. Its inspected cases do not execute the campaign loop, finalization recovery, response-storage retries, or live accounting services. A passing count alone does not prove these paths.

**Why this is a problem:** another engineer cannot reliably distinguish current operating instructions, historical evidence, and unverified expectations.

**Evidence:** [README](../README.md); [technical specification](../TECHNICAL_SPECIFICATION.md); [implementation record](implementation-progress.md); [decisions](../DECISIONS.md); [workflow/reporting tests](../tests/unit/test_workflow_reporting.py).

**Verification needed:** reconcile status with dated commands, code identity, results, and limitations. Add meaningful acceptance cases for the remaining boundaries. No test counts were independently verified during this review.

### Source snapshot for this follow-up

These hashes identify the main implementation files reread for this revision. They are evidence of which bytes were inspected, not execution results. Paths are relative to `iadopt-lab/`.

| File | SHA-256 |
|---|---|
| `src/iadopt_lab/workflow.py` | `e8e523d049d0a6538a6c90c4050416fcf9cd2e6a09b133ce76d997db9dc0935c` |
| `src/iadopt_lab/cli.py` | `44a955a2e388db55a726ba1f8f85bcb8dad1db8c5e27c04f16d034f42f075c58` |
| `src/iadopt_lab/providers/base.py` | `2da8ce33d36079ee6bf35ba770e7cf13f468989abce7ec25aeb988e149177636` |
| `src/iadopt_lab/reporting.py` | `1da69bb33d8ec9961e61674887dbd0a332ceab6fccab3c6ae2900b4bbf8f532b` |
| `src/iadopt_lab/persistence/repository.py` | `db8ae78b044ed835c43e8c3ce70bd38945227f3a3d8d3f0aabc1780983abcee3` |

## Decisions and working parts to preserve

Inspection supports the v2.0.1 pin, ordered demonstrations, category retention, provider-owned models, three-request structure, disabled SDK retries, approved member credit, excluded system labels, exact ranking, and deferred RDF/linking scope.

D-032 records replacing Ling with GPT-4o mini. This report treats it as the repository's recorded decision rather than treating missing conversational context as proof of an unauthorized change. Approval provenance can be audited separately if required.

Later decisions describe deployment-specific reasoning restrictions and scoped live runs. Those campaigns should remain clearly distinguishable from the original full grid. Provider measurements were not independently verified here.

## Suggested repair and acceptance order

This is a proposed sequence, not permission to implement or run experiments.

1. Preserve raw responses through storage and accounting failures.
2. Complete API-envelope and truncation classification.
3. Require scientific identity and coverage checks for every reporting entry point.
4. Correct retry-delay and resume behavior.
5. Correct synthetic identities and configurable currency handling.
6. Establish token-bound evidence and verify the complete offline workflow, including interrupted finalization.
7. Reconcile documentation with implementation and actual acceptance evidence.

Only this report was rewritten. No claim is made that a prior scientific result is invalid; determining that would require examining the relevant campaign's stored evidence.


---

# Resolution log — appended, not overwriting

Per this report's own guidance, resolutions are appended and finding statuses updated
without erasing history. Findings above are retained verbatim.

## Verification outcome

**All ten findings were confirmed against the source before any change was made.**

Finding 10's disputed subclaim is resolved **in this report's favour**. The previous
verification searched for "implementation *has* not started"; the actual text omits "has"
and sits in YAML front matter that was never opened. `TECHNICAL_SPECIFICATION.md` line 4
read exactly `status: Documentation baseline; implementation not started`, and the quoted
SHA-256 `f0ed6d05…` matched byte for byte. The earlier "none found" was wrong twice over.
Citing path, line, quote and hash is what made that resolvable, and is worth keeping.

The report's new **repair boundary** on finding 2 was also confirmed and changed the fix.
`store_response` writes the response row and its settlement in one call (repository.py
lines 1366 and 1449), `response.attempt_id` is `UNIQUE`, and re-storing different metadata
raises `EvidenceConflict`. Moving the settlement call after the commit would therefore have
failed. Settlement is instead made non-raising, so ordering cannot destroy evidence.

## Status after this pass

| Finding | Status | Evidence |
|---|---|---|
| 1 | Resolved | Retry now catches `psycopg.Error` as well as `PersistenceError`, re-raises `EvidenceConflict` unretried, and backs off over roughly two minutes. Tested both directions. |
| 2 | Resolved | Settlement failures are caught and recorded as unresolved accounting; raw preservation can no longer be blocked by billing interpretation. |
| 3 | Resolved | All eight envelope cases verified: non-object body, provider error object, missing choice, malformed message, and truncation whether empty or partial are operational; a genuinely empty completion and a valid answer still reach scoring. |
| 4 | Resolved | `plan_scorer` is now required, and `iadopt-lab report` supplies scorer identity and frozen category membership. Finalization reads the scorer version and threshold from frozen constants rather than from the evidence under examination. |
| 5 | Resolved | The handler reads `retry_after_seconds` instead of the message string. |
| 6 | Resolved | Reservations and provider-reported costs apply the same `fx_to_reporting` as settlement, and all three record their currency. |
| 7 | Resolved | Artifacts are collected for the planned population, so a synthetic plan's `population` identity matches its three IDs. |
| 8 | Resolved | Resume verifies the requested campaign before registering anything, treats `already_complete` as success, and finalizes on that path so an interrupted finalization is recovered by running resume again. |
| 9 | Resolved as an evidence requirement | The three conditions under which one token per byte is an upper bound are stated at the call site, with a note that a deployment violating them requires the bound to be re-established. |
| 10 | Resolved | Specification front matter, README PostgreSQL claim, README's dangling D-039 reference, D-038's truncation note, and the implementation record's acceptance claims all corrected. D-039 now exists and records this pass. |

## Evidence

`tests/unit/test_workflow_reporting.py` grew to 23 cases covering the eight-way envelope
classification, transient-versus-permanent commit retries, the typed rate-limit delay, the
required plan binding, and synthetic population identity.

Suite: **312 passed, no skips** with the PostgreSQL test database configured.
`ruff check src tests` clean. `iadopt-lab verify` reports the corpus verified;
`preflight` reports zero issues. A dry run finalizes to `state: complete`, and repeating
it through `resume` returns the same `ranking_id`, demonstrating idempotency.

## Still open

The report's observation that the test file does not execute the full campaign loop under
fault injection remains true. Coverage now includes the classification, retry, binding and
identity boundaries, but not a fault-injected end-to-end campaign. Also unchanged from the
previous pass: `cost_settlement` rows are written but `get_task` does not expose them, so
the reporting layer's monetary summary still counts those attempts as cost-unavailable.
