# Documentation and Logic Audit

## 1. Status and boundary

**Original evidence audit:** 2026-09-03; decision follow-up: 2026-09-06

**Boundary note:** this audit ran against Corpus **v2.0.0**, before any code existed. Both conditions have since changed: D-030 moved the pin to v2.0.1, and the implementation was subsequently written, so findings such as "no executable implementation" and "no code exists" describe the repository at audit time and are no longer true of it. The findings are retained as written rather than restated, because an audit record must report what was actually checked. Current state is recorded in `README.md` and `docs/implementation-progress.md`; re-verification against v2.0.1 is in `docs/migration-v2.0.1.md`.

**Scope:** `iadopt-lab/` documentation/configuration plus read-only source evidence

**Implementation status:** Not started

This audit checks whether the proposed experiment matches the user's decisions, whether every document agrees with the others, and whether factual claims can be reproduced from the available source files. It does not treat a written plan as proof of working software.

The decision follow-up records accepted conservative prompts (D-021), accepted fractional member scoring in simple/system and system/system comparisons (D-022), accepted ranking (D-023), and deferred descriptive statistics with full score retention (D-024). All four scientific choices are resolved. Historical corpus/workbook findings below belong to the original evidence audit; this follow-up does not claim a new experiment or a rerun of those workbook analyses.

The provider-selection follow-up adds D-025/D-026: one campaign can run PSNC, OpenRouter, or both, using editable provider-owned model lists and continuing through completion with independent provider/billing gates. It supersedes the original D-009 single-provider restriction. The owner's later Python attachment supplies all six initial exact IDs and the shared-SDK routing evidence in `docs/model-catalog.md`. Availability and reasoning controls remain preflight evidence to collect, not missing model choices.

No experiment code, schema artifact, migration, prompt template, corpus copy, database, provider call, entity linker, RDF converter, SHACL execution, or live/dry experiment was created or run. At the end of this audit, the directory still contains only Markdown and the documentation-phase `parameters.yml`.

Status labels in this document mean:

- **Pass:** internally consistent and supported by the current evidence/decisions.
- **Conditional:** technically specified, but one named owner decision must be resolved before implementation.
- **Draft-safe:** intentionally contains null/disabled values so it cannot authorize live execution.
- **Deferred:** intentionally outside the active experiment and not an implementation dependency.

## 2. Method

The audit was performed in this order:

1. Reconstructed a requirement ledger from the redesign input and every later user correction.
2. Enumerated every file in `iadopt-lab/` and confirmed there is no executable implementation.
3. Read the root overview, decision index, full technical specification, configuration, each operational/scientific document, every component contract, and both future-work documents.
4. Rechecked immutable Git evidence for the January runner, historical converter, schema, prompts, tag commit, and tree.
5. Imported the relevant historical XLSX workbooks read-only and recalculated their sheet row counts, configuration dimensions, target-list lengths, demonstration overlap, and stored F1 values.
6. Independently inspected Corpus v2.0.0 from its exact tag/commit/tree and parsed all 102 Turtle files.
7. Rechecked the five approved demonstrations against their definitions and RDF-derived lexical gold.
8. Revalidated the supplied JSON/JSON Schema/Turtle reference files without copying or repairing them.
9. Tested YAML parsing/duplicate-key behavior, grid arithmetic, Markdown structure, function-contract completeness, secret patterns, prohibited personal paths, and obsolete split/stratification language.
10. Corrected every confirmed documentation defect and recorded decisions that cannot be made honestly without the owner.

## 3. Requirement traceability ledger

| ID | Required outcome | Current authority | Audit result |
|---|---|---|---|
| R-001 | One isolated root directory named `iadopt-lab/` | D-001; architecture | Pass |
| R-002 | Documentation and decisions before any code | D-016; README; runbook | Pass; no code exists |
| R-003 | Every planned function documented with full input/action/output/failure/side-effect behavior before code | Component index/contracts | Pass; 51/51 planned public functions have all required fields |
| R-004 | New experiment only; historical work is evidence, not an active track | D-002; repository audit | Pass |
| R-005 | Corpus v2.0.0 exact release, commit/tree, 102 Turtle files | D-003; data contract | Pass; independently verified |
| R-006 | Deterministic Turtle-to-six-field lexical JSON gold | D-005/D-006; corpus/schema contracts | Pass as specification; code/schema not implemented |
| R-007 | LLM receives the variable description, schema, selected prompt, and ordered demonstrations | D-021; prompt contracts | Pass; historical no-interpretation policy retained, limitations documented |
| R-008 | Three prompt families named Strict minimal, Constraint decomposition, and Matrix decomposition | D-004/config/prompt docs | Pass |
| R-009 | Exact five demonstrations in the approved order; prefixes for 0/1/3/5 shots | D-004; data/prompt docs | Pass; paths independently verified |
| R-010 | Demonstrations never scored, including zero-shot; all other 97 variables evaluated; no train/dev/test/holdout/stratification | D-014; population contracts | Pass |
| R-011 | PSNC only, OpenRouter only, or both in one campaign/execution, with editable provider-owned model lists | D-025 supersedes D-009; config/provider/workflow docs | Pass as specification; each task still has one provider/model owner |
| R-012 | OpenRouter and PSNC adapters; sibling service is evidence only | D-011; provider/reference docs | Pass as design; exact live profiles remain unset |
| R-013 | Reasoning enabled and disabled only for models with a controllable parameter; otherwise not applicable | D-010; config/provider docs | Pass as design; exact native mappings remain live placeholders |
| R-014 | Config-driven prompts, shots, temperatures `[0, 0.5, 1, 2]`, repetitions, models, and operational gates | `parameters.yml`; configuration docs | Draft-safe; values are explicit and unset live fields are null |
| R-015 | One orchestrator owns three total provider requests per fully resolved variable/parameter/repetition task; no nested retries | D-008; retry/provider/workflow docs | Pass |
| R-016 | On content-invalid output, retain raw response/errors and give the immediately previous response/errors to the next correction prompt | D-008; prompt/retry docs | Pass |
| R-017 | After three content-invalid model responses, create an explicit-empty prediction and score it; operational failure remains separate | D-008; retry/evaluation docs | Pass |
| R-018 | Store every prompt, sanitized request, raw response, error, attempt, prediction, score, result, usage, cost, and state transition | D-012/D-014; database/evidence contracts | Pass as schema/DB design; no DB exists yet |
| R-019 | Store corpus version plus exact category, subcategory, and full category path for every variable | D-003; data/database/reporting docs | Pass |
| R-020 | PostgreSQL is authoritative and work can resume from durable checkpoints | D-012/D-013; database/retry/reproducibility docs | Pass as design |
| R-021 | Rewrite the January scorer without importing it; exclude system container identifiers and award the explicitly approved partial credit | D-007/D-022; scorer parity | Pass as specification; accepted fractional member policy, role handling, exact receipts, and preserved January boundaries |
| R-022 | Automatically rank configurations and retain every result for later analysis | D-014/D-023/D-024; reporting/database docs | Pass; mean repetition-level micro Close F1 and shared ties accepted; retain Precision/Recall/F1 and counts, defer additional statistics |
| R-023 | No active entity linking, RDF, JSON-LD, SHACL, deterministic baseline, or human-in-the-loop step | D-017/D-018; future docs | Pass; all are clearly deferred |
| R-024 | Offline mock dry run and full verification only after implementation approval; live calls require a later gate | D-019; runbook/test plan | Pass |
| R-025 | Reproducible source/config/prompt/schema/model/scorer/evidence identities and rebuildable reports | Reproducibility/database/output docs | Pass as design |
| R-026 | Thin `main.py`; algorithms, providers, parallel execution, retry, SQL, scoring, and reporting stay in separate modules | Architecture/component contracts | Pass as planned structure |
| R-027 | Record the exact three PSNC and three OpenRouter models supplied by the owner; permit future model/list changes | D-025; model catalog; parameters | Pass as selection; provider availability/capability verification remains preflight work |
| R-028 | One execution advances every selected model through scoring/ranking/reports and resumes checkpoints | D-026; workflow/recovery | Pass as specification; blocked scopes remain incomplete and healthy work continues |
| R-029 | No-charge PSNC access is scoped and evidenced; disclose costs before live work without requiring spending caps | D-026/D-027; billing/database/contracts | Pass as specification; null means uncapped, unknown cost is not zero |
| R-030 | One repetition at every temperature for all selected providers/models; retain retry, ranking, and evidence contracts | D-029; parameters/grid/reporting/reproducibility docs | Pass as specification; singleton ranking and absent run-to-run variability are explicit |

## 4. Independent evidence checks

### 4.1 Source and supplied files

| Evidence | Rechecked result |
|---|---|
| Redesign Markdown | 130,419 bytes; 2,814 lines; SHA-256 `dc66ed2053b2b2796446ccca4a1315a498e156959f59951dde77411e3d5c68d1` |
| January tag | `V1.1-Experiment` → commit `b9683d2242aa5ca5b987440ea4b6f70bc1253c7e`, tree `35bf57411d7e8143a255415825f9bb0b644e8c6e` |
| January runner | SHA-256 `2d4a6271fd86a076127dfb3f85589391cff9744efdfdee8ec570b52c74921df0`; current tracked bytes are identical |
| Tagged gold converter | SHA-256 `b8caf88bc39a1aa8d6b01bc7e0fe2ed6419de5b6450455f0d6dd5242d7ab3773` |
| Tagged prompt schema | SHA-256 `c099f2cebe91e495c22506b8accd592c209059a69c60110b12e7a263a7ab7d9a` |
| Sibling PSNC evidence | Commit `0d1a4cdae362b5aa9a23ab7b24b9f55d4a270e8b`; confirms the low-level bearer-auth `/v1/chat/completions` shape and Qwen disable-thinking fields, not an approved live model profile |
| Supplied Variable schema | 5,759 bytes; recorded hash matches; valid JSON and Draft 2020-12 meta-schema |
| Supplied JSON-LD context | 1,950 bytes; recorded hash matches; valid JSON syntax |
| Supplied `iadopt.sh.ttl` | 7,326 bytes; recorded hash matches; valid Turtle with 203 triples |
| Supplied `iadopt-llm.sh.ttl` | 3,878 bytes; recorded hash matches; Turtle parse fails at line 21 as documented |

No credential value from the sibling service was copied into `iadopt-lab/`; only environment-variable names and non-secret interface facts are allowed.

### 4.2 Corpus v2.0.0

An independent exact-tag checkout confirmed:

- tag `v2.0.0`, commit `8097662ca323771fd977d22cdb8c3e58b7b7d64a`, tree `bd9cf247d22c5b8345796572e3e55f333460c6ce`;
- 102/102 Turtle files parse and each has one Variable root, one Property, one Object of Interest, and one `rdfs:comment`;
- 52 variables have Matrix, 10 Context Object, 9 Statistical Modifier;
- 85 variables contain 157 Constraints;
- 36 asymmetric systems: 31 numerator/denominator and 5 source/target;
- 2 symmetric systems containing 4 total parts;
- all 102 records can be projected into the planned six-field structural shape without unresolved labels/targets;
- all five demonstration paths exist, are distinct, and leave exactly 97 other variables;
- categories/subcategories derive from 21 exact two-level release paths;
- two Constraints target whole unlabeled asymmetric systems and therefore require the documented deterministic role-derived display label.

The official release record identifies v2.0.0 as CC-BY-4.0 with version DOI `10.5281/zenodo.22011435` and concept DOI `10.5281/zenodo.18101358`. The Git tag has no standalone `LICENSE` file, so the implementation must retain the explicit release metadata/attribution rather than inventing a file.

### 4.3 Historical XLSX correction

The read-only spreadsheet audit confirmed that the large December 26 and 28 grids have the documented 192/48/48 Summary rows and exactly 96 stored tested paths per row. It found and corrected a more serious error in the earlier documentation: the six December 30 one-row workbooks do **not** share a 97-target population.

Their exact stored unique target counts are `98, 98, 97, 98, 102, 97`; their demonstration-path overlaps are `0, 4, 4, 4, 5, 0`. Therefore:

- they are not six repetitions of one condition;
- four contain demonstration leakage in the stored tested population;
- their F1 values cannot be averaged as a stochastic estimate;
- they remain useful only as separately identified historical evidence.

`repository-audit.md` and D-020 now state this accurately.

### 4.4 Logic/static checks

| Check | Result |
|---|---|
| Directory boundary | Only `.md` and one `.yml`; no executable code/schema/migration/data output |
| YAML syntax and duplicate-key-aware parse | Pass; 14 top-level sections after cost-evidence addition |
| Live safety | `live_calls_enabled: false`; six owner-selected models enabled only as intended selection; unknown capabilities/revisions/limits, price manifest, embedding revision, worker count, and missing estimate disclosure/live authorization block execution; D-027 makes null caps valid and independent of PSNC's zero billing |
| Grid arithmetic | Under D-029, per model/reasoning profile: 48 configurations, 48 repetition runs, 4,656 tasks/initial calls, maximum 13,968 requests; two profiles double each count |
| Population arithmetic | `102 - 5 = 97`; no split/stratification/seeded partition |
| Function contracts | 51 planned public functions; 0 missing Input/Action/Output/Raises/Side effects fields |
| Markdown fences | 0 unbalanced fences |
| Tabs/trailing whitespace | 0 / 0 |
| Secret-pattern scan within `iadopt-lab/` | 0 hits |
| Personal absolute paths within `iadopt-lab/` | 0 hits |
| Active split/stratification language | 0; remaining occurrences explicitly say there is no split or refer to separating workflow stages |
| Internal terminology | Provider union versus individual task ownership, prompt IDs, 102/5/97 counts, three-attempt cap, and deferred stages are specified consistently |

The initial 2026-09-06 follow-up checked the 35-file Markdown/YAML-only boundary, duplicate-aware YAML parsing (14 sections), accepted ranking and disabled-live invariants, all 51 function contracts, and Markdown whitespace/fences. Five then-proposed partial-credit examples also passed their arithmetic checks. The owner subsequently approved the broader system/system scope and fractional TP/FP/FN policy.

After that approval, the final documentation checks passed again: 35 Markdown/YAML files, 14 configuration sections, all 51 function contracts, zero whitespace/fence defects, accepted scorer/ranking protocol IDs, and disabled live execution. In-memory exact-fraction checks covered 23,740 valid `(g,p,m)` combinations for member counts 1–40, proving unit contribution mass, bounds, Precision `m/p`, Recall `m/g`, and component F1 `2m/(g+p)`. The maximum-cardinality counterexample, threshold boundary, and exact fraction/ranking-tie examples passed. These are arithmetic and specification checks, not tests of a future executable evaluator or PostgreSQL integration. Stale scorer-choice, prompt-snapshot, normalization, and immediate-statistics wording was corrected.

The subsequent provider/model update passed these additional checks:

- Current boundary: 36 Markdown/YAML files after adding `docs/model-catalog.md`, 14 version-2 YAML sections, and all 51 function contracts; zero whitespace/fence defects.
- AST-only inspection of the owner's attachment matched all six model IDs and its SHA-256; the example itself was not executed.
- PSNC-only, OpenRouter-only, and combined selections each produced the correct provider-owned model union. One/two hypothetical reasoning profiles per model yielded the documented conditional totals. Actual profiles remain unverified and cannot be inferred from this arithmetic.
- The locally installed OpenAI SDK `2.43.0`, with an in-memory mock HTTP transport and a dummy credential, constructed the correct final endpoint for all six IDs and preserved raw responses. One synthetic HTTP 429 per provider produced exactly one request with `max_retries=0`, retaining the error body. There were eight mock HTTP exchanges and zero external provider requests.
- At that earlier check, six models were selected, live mode was false, and unknown capabilities/revisions remained explicit; PSNC then had a zero provider cap while OpenRouter/global paid caps were unset. D-027 subsequently superseded those mandatory-cap gates: all default caps are now null and valid, while price evidence, estimate disclosure, and separate live authorization remain required.

The SDK probe verifies interface assumptions in the inspected local version; it is not a dependency lock, a live model-availability test, or an end-to-end experiment dry run. The official SDK reference is linked in `docs/model-catalog.md`. No implementation scripts, database mutations, or generation calls were added.

## 5. File-by-file review result

### 5.1 Root and directory-policy documents

| File | Result | What was checked |
|---|---|---|
| `README.md` | Pass | Plain-language goal/scope, 12-step flow, no split, full retention, ranking, deferred stages, approval gates |
| `DECISIONS.md` | Pass | D-021–D-024 accepted; historical denominator correction and live values remain explicit |
| `TECHNICAL_SPECIFICATION.md` | Pass as specification | End-to-end authority with accepted member dispatch, fractional arithmetic, ranking, and retained/deferred boundaries |
| `THIRD_PARTY_NOTICES.md` | Pass | Corpus commit/DOIs/license exact; no invented Git license file; deferred sources remain inactive |
| `parameters.yml` | Draft-safe | Version 2.1 selected provider list, six supplied IDs, unknown capability gates, mandatory pre-run estimate and optional null caps, continued execution, unchanged scientific grid and scoring |
| `data/README.md` | Pass | Source/canonical/manifest responsibilities, 102/5/97 counts, categories, stable system labels, deterministic ordering, price/scorer provenance |
| `prompts/README.md` | Pass | Accepted historical prompt policy, minor adaptations, limitations, versioning/render/retry boundary |
| `schemas/README.md` | Pass | Correct lexical schema versus supplied JSON-LD schema, parameter/evidence/manifest schemas, relational versus JSON validation boundary |
| `reference/README.md` | Pass | Immutable hashes/provenance, January/current separation, PSNC evidence, supplied-file syntax findings, no runtime imports |
| `outputs/README.md` | Pass | Derived-only reports, complete rankings/results, explicit denominators/lineage, sensitive evidence stays in PostgreSQL |

### 5.2 Core design and evidence documents

| File | Result | What was checked |
|---|---|---|
| `docs/README.md` | Pass | Correct navigation and accepted/proposed distinction |
| `docs/architecture.md` | Pass | Independent package tree, thin CLI, pure evaluator, persistence/provider/workflow boundaries, no active deferred package |
| `docs/model-catalog.md` | Pass as selection/contract | Exact six IDs and attachment hash, provider endpoints/keys, shared SDK with no automatic retries, unresolved deployment controls, editable lists and completion semantics |
| `docs/database.md` | Pass | Complete evidence tables, categories/version, population, repetitions/ranking, constraints, leases, recovery, immutable/decimal storage |
| `docs/documentation-audit.md` | Pass as decision record | Original evidence findings distinguished from 2026-09-06 decision updates; documentation checks do not establish implementation correctness |
| `docs/prompt-specification.md` | Pass | Accepted historical conservative policy and minor changes; non-literal gold and geographic-Matrix tensions retained as documented limitations |
| `docs/repository-audit.md` | Pass | Tagged hashes, actual schema/retry behavior, workbook grids, corrected December 30 counts/leakage, uncertainty labels |
| `docs/reproducibility.md` | Pass | Original versus resolved config, source/model/scorer/evidence hashes, backup/restore, re-evaluation without generation |
| `docs/retry-and-resume.md` | Pass | Exactly three numbered requests across restarts, correction lineage, operational/content distinction, leases/ambiguity/crash matrix |
| `docs/runbook.md` | Pass as specification | Correct command order and accepted scientific protocol; implementation and live configuration/artifact verification remain separate stages |
| `docs/scorer-parity.md` | Pass as specification | Accepted member scoring, preserved literal symmetric identity and scalar/Constraint branches, ordered-role rules, exact fractions, and before/after fixture requirements |
| `docs/test-plan.md` | Pass | Static/unit/property/golden/provider/PostgreSQL/recovery/dry-run/scale gates; tests cover all newly found edge cases |
| `docs/decisions/README.md` | Pass | ADR statuses/template/immutability and activation rules |

### 5.3 Component contracts

| File | Result | What was checked |
|---|---|---|
| `docs/components/README.md` | Pass | Required pre-code and implemented-docstring format; module ownership map |
| `docs/components/shared-foundations.md` | Pass | Typed domain/canonical JSON/hash/sort/redaction boundaries |
| `docs/components/cli-and-configuration.md` | Pass | Strict version-2.1 YAML/preflight, selected-provider union, provider-owned model lists, unknown-capability gate, estimate/disclosure/authorization separation, optional caps, explicit private env-file input |
| `docs/components/corpus-ingestion.md` | Pass | Exact Git blobs, strict RDF projection, atomic activation, population/category/system/Constraint determinism |
| `docs/components/prompting.md` | Pass | Complete four-function contracts with accepted conservative policy and historical reference/diff evidence |
| `docs/components/providers.md` | Pass | One invocation/one request, no retry, secret-free payload/result, explicit capability validation |
| `docs/components/generation-and-validation.md` | Pass | Robust one-object extraction, strict layered validation, no quality retry/coercion, three-attempt outcomes |
| `docs/components/persistence.md` | Pass | Full transaction/repository contracts, idempotency, fencing, response-before-parse, immutable scores/ranks |
| `docs/components/workflow.md` | Pass | Deterministic expansion, bounded concurrency, stage-level resume, attempt cap, ranking handoff |
| `docs/components/evaluation.md` | Pass as specification | Pure evaluator contracts implement accepted D-022 with typed match evidence and exact fraction receipts |
| `docs/components/reporting.md` | Pass | Accepted ordering, complete Precision/Recall/F1 retention, and deferred additional descriptive statistics |

### 5.4 Deferred documents

| File | Result | What was checked |
|---|---|---|
| `docs/future/deterministic-ontology-baseline.md` | Deferred | Structural ontology rules distinguished from lexical heuristics; no hidden split; no package/code now |
| `docs/future/entity-linking-and-rdf.md` | Deferred | No active linking/RDF/JSON-LD/SHACL; future target is RDF/Turtle; supplied-file defects/approvals stay isolated |

## 6. Confirmed corrections made during this audit

1. Removed every train/development/test/holdout/stratification design. The experiment now has one exact 97-variable evaluation population.
2. Made automatic ranking and complete lower-ranked/raw/item/component/repetition/result retention explicit across configuration, database, workflow, reporting, tests, and runbook.
3. Changed unapproved live values—timeouts, concurrency, worker count, output limit, top-p, revisions, context limits, rate limits, price manifest, and budget—to null/disabled gates. Accepted ranking and scorer policies now have explicit version identifiers; their future executable/artifact hashes remain distinct.
4. Added price-card/billing provenance to configuration/data contracts.
5. Expanded all planned public functions to full pre-code contracts; the mechanical check now reports 51/51 complete.
6. Added exact Corpus version DOI, concept DOI, and CC-BY-4.0 evidence while recording that the Git tag has no standalone license file.
7. Added deterministic path/Constraint ordering and whole-unlabeled-system target handling verified against Corpus v2.0.0.
8. Corrected the historical December 30 workbook claim: populations are inconsistent and some include demonstrations.
9. Documented the verified prompt-to-gold inconsistency instead of freezing contradictory prompt text.
10. Documented the January scorer's directional structural-mismatch quirk instead of incorrectly calling every mismatch zero.
11. Initially separated automatic ranking/all-result retention from the proposed exact formula; D-023 now accepts that formula.
12. Recorded the owner's choice to preserve historical no-interpretation prompts and document their gold/Matrix limitations without semantic repairs.
13. Initially recorded partial credit in principle; the owner's subsequent approval now extends fractional matched/missing/extra-member contributions to both simple/system and system/system comparisons.
14. Made Exact/Close Precision, Recall, F1 and supporting counts explicit at all score scopes; deferred additional variable-score statistics until the database is populated.
15. Specified complete member matching/role dispatch and exact rational receipts, preventing finite decimal approximations of thirds from changing ranking ties. Retained January literal symmetric member identity rather than silently adding normalization or embeddings to that established branch.
16. Superseded D-009 with provider-union execution, complete expected membership and independent provider scheduling/billing; retained the three-request and immutable-resume rules.
17. Replaced old sample/placeholder model IDs with the owner's six supplied IDs, added source/SDK routing documentation, and left unverified reasoning capabilities explicit.

## 7. Settled scientific policy and deferred live values

No scientific policy among D-021 through D-024 remains open:

**D-022 — Fractional member credit.** Accepted protocol `january-derived-member-credit-v1` applies one-to-one member matches in simple/system and system/system comparisons. With `U=g+p-m`, TP is `m/U`, FP is `(p-m)/U`, and FN is `(g-m)/U`. Arbitrary labels are excluded, ordered roles remain significant when both sides provide them, and membership-only comparisons record absent role evidence. Symmetric/symmetric literal identity is preserved. Exact fractions and complete assignment evidence make the policy reproducible.

D-021 is settled: keep historical conservative prompts with minor workflow/schema changes and unchanged gold; record limitations only. D-023 is settled: rank by the unrounded mean of repetition-level micro Close F1, descending, with shared competition ranks and no secondary tie-break. D-024 requires all Exact/Close Precision/Recall/F1 evidence now and defers mean/median/variance/mode/standard-deviation/range/IQR analysis over the 97 individual variable scores. That later arithmetic mean is not the primary micro metric.

These live-run values are intentionally not decisions for this documentation review and may remain unset while the architecture is implemented: available provider/model revision evidence, native reasoning mappings, similarity-model artifact revision, top-p/output tokens, context/rate/concurrency/worker limits, price card, disclosed cost estimate, and live-canary authorization. Preflight must block live use until required evidence is frozen. The six IDs are already supplied; monetary caps are optional under D-027 and null caps are not unresolved placeholders.

Implementation-level library choices—exact Python version, database driver/migration library, dependency-lock versions, test libraries, and container digests—are also not silently fixed by this audit. They are selected, documented, locked, and tested after implementation approval without changing the scientific contract.

## 8. Step-by-step owner review plan

Review the documents in this order; each step has one bounded purpose and exit condition.

### Step 1 — Confirm the experiment in plain language

Read `README.md`, then the requirement ledger in this document. Confirm that the input, output, 97-variable population, three-attempt behavior, stored evidence, ranking purpose, and deferred work match the intended experiment.

**Exit:** no scope item is missing or unexpectedly active.

### Step 2 — Verify the accepted scoring contract

Use D-022 and `scorer-parity.md` as the accepted implementation contract. Check every representation combination, matching restriction, fractional contribution, preserved January branch, and evidence field against its specified fixture. Do not reopen the settled scientific approvals.

**Exit:** fractional score contributions and the boundary of the January correction are unambiguous.

### Step 3 — Approve the authoritative end-to-end protocol

Read `TECHNICAL_SPECIFICATION.md` in order: scope, corpus, lexical representation, prompt, configuration, provider, extraction/validation, persistence, orchestration, evaluator, reporting, tests, and gates.

**Exit:** every stage has defined input, action, output, failure, and ownership; no section conflicts with accepted decisions.

### Step 4 — Review the only editable experiment configuration

Read `parameters.yml` beside `docs/components/cli-and-configuration.md`. Confirm which values are scientific grid factors, which are capability mappings, which are operational, and which nulls intentionally block live execution.

**Exit:** selected-provider union, provider-owned editable lists, reasoning expansion, 3×4×4 grid, repetitions, 97 targets, attempt cap, complete ranking/retention, zero-cost and metered gates are understandable and schema-testable.

### Step 5 — Review source data and historical evidence

Read `data/README.md`, `reference/README.md`, `THIRD_PARTY_NOTICES.md`, and `repository-audit.md`. Pay special attention to exact commits/hashes, category paths, five demonstrations, deterministic gold projection, historical nine-call risk, and the corrected inconsistent December 30 populations.

**Exit:** source-of-truth and historical-versus-new boundaries are accepted.

### Step 6 — Review the model boundary

Read `prompts/README.md`, `docs/prompt-specification.md`, `schemas/README.md`, and `docs/components/prompting.md` plus `generation-and-validation.md`.

**Exit:** exact model-visible content, robust JSON extraction, strict schema/semantic validation, feedback retry, and valid-poor-response behavior are approved.

### Step 7 — Review provider, database, and recovery behavior

Read `docs/components/providers.md`, `docs/database.md`, `docs/components/persistence.md`, `docs/retry-and-resume.md`, and `docs/components/workflow.md`.

**Exit:** every call/evidence transaction, maximum-three rule, lease/fencing rule, ambiguous-delivery behavior, and resume checkpoint is testable and safe.

### Step 8 — Review scoring, ranking, and reports

Read `docs/components/evaluation.md`, `docs/scorer-parity.md`, `docs/components/reporting.md`, and `outputs/README.md`.

**Exit:** each scored/unscored field, January exception, aggregation denominator, repetition summary, ranking input, not-rankable state, and retained output is explicit.

### Step 9 — Review verification and reproducibility

Read `docs/test-plan.md`, `docs/reproducibility.md`, and `docs/runbook.md`.

**Exit:** every requirement maps to a future test, no normal test makes paid calls, the mock dry run covers success/correction/exhaustion/resume, and an interrupted campaign/report can be reconstructed.

### Step 10 — Confirm deferred boundaries and authorize or reject implementation

Read both `docs/future/` documents and `docs/architecture.md`. Confirm that no baseline/linking/RDF/SHACL code is included and that the planned module tree is appropriately isolated. Then explicitly authorize implementation if all earlier exits pass.

**Exit:** a clear written implementation approval or a list of documentation changes; silence or document edits alone do not authorize code.

## 9. Audit conclusion

The scientific and provider-execution decisions are resolved, including D-022 member credit, D-025/D-026 flexible provider/model selection and completion, D-027 cost disclosure without mandatory caps, and D-029's one repetition at every temperature. The six model IDs are supplied and recorded. D-028 records the existing private credential source and proposed local PostgreSQL/DBeaver setup, not completed provisioning. Documentation specifies the inputs, behavior, outputs, evidence, and tests needed for implementation. Verified deployment capabilities, live operational limits, price/estimate evidence, and future artifact hashes remain to be frozen before execution. Documentation and in-memory checks do not substitute for the future implementation tests, PostgreSQL integration, or offline dry run.

## 10. Documentation-only readiness cleanup

The follow-up corrected stale model-ID and separate-provider-campaign sentences, clarified that classified safe transient resends share the same three-request allowance as content corrections, and propagated D-027's no-mandatory-cap policy through configuration, database, workflow, reporting, recovery, and test contracts. D-028 and `local-database.md` explain the existing private credential source and proposed PostgreSQL/DBeaver setup. At that checkpoint the technical document was version `0.3.0-documentation`; parameter schema was `2.1`.

Post-edit checks passed: duplicate-key-aware YAML parsing with 14 sections; 37 Markdown/YAML files and no implementation artifacts; 53 complete public-function contracts, including the cost estimator and private runtime-secret loader; zero trailing-whitespace or code-fence defects; unchanged six model IDs, scientific grid, scorer/ranking identifiers, 97-variable denominator, and three-request maximum. All monetary caps are null, pre-run estimation is required, and live mode remains false. Five in-memory null/zero/positive-cap arithmetic cases passed. These are documentation/policy checks, not executable workflow or database integration tests.

Read-only host checks found Docker and Compose clients but an unreachable Docker engine; no service was started. The existing root `.env` exists, is ignored and untracked, and was not read, copied, modified, or used for a provider request. The subsequent cost question initially requested a calculation only, so the five-repetition setting was preserved at that checkpoint. The owner's later explicit instruction is now accepted as D-029 and replaces it with one repetition at every temperature.

## 11. Accepted single-repetition update

D-029 sets both repetition fields to `1` for every selected provider/model. The technical specification is now `0.3.1-documentation`; the unchanged parameter schema remains `2.1`. Updated the plain-language README, decision context, grid counts, database planning, workflow/runbook, model catalog, test expectations, ranking/output notes, and reproducibility limitations. Preserved historical workbook facts and kept generic repetition support for future campaigns.

Verification passed: duplicate-key-aware YAML parsing; both repetition values equal one; six conditional PSNC-only/OpenRouter-only/combined one-/two-profile count scenarios; 75% reduction versus the superseded temperature schedule; exact singleton mean identity; 37 documentation/configuration files with 53 complete function contracts; zero format defects; no stale active five-repetition defaults or old numeric grid totals. Model IDs, provider selection, all 97 targets, scorer/ranking policy IDs, three-request limit, null monetary caps, required cost disclosure, and disabled live mode remain unchanged. These are in-memory specification checks, not tests of an implemented experiment runner. No scripts, database operations, credential reads, or live model calls were performed.
