# Historical Repository and Result Audit

## Status and purpose

**Boundary note:** this audit was performed while Corpus **v2.0.0** was the authoritative release. D-030 later moved the pin to v2.0.1. The findings are retained as written, because an audit record must report what was actually examined; where the text says "the new Corpus v2.0.0 campaign", read the campaign pinned by D-030.

This is a read-only reconstruction of the decomposition experiment code and the latest locally available result artifacts before the January 2026 paper submission. It records what the repository proves, what can be inferred from multiple artifacts, and what remains unproven. It does not authorize a legacy rerun, modify historical files, or make historical artifacts runtime dependencies of I-ADOPT Lab.

The audit has two purposes:

1. Identify the January scorer behavior, the approved system-container-label correction, and any additional edge requiring an explicit preservation/correction decision.
2. Prevent historical configuration, corpus, schema, retry, and denominator ambiguities from being copied into the new Corpus v2.0.0 campaign.

This document is narrower than a complete manuscript-table reconciliation. No table is declared reproduced unless every value can be joined to immutable inputs, exact code, exact configuration, and raw run evidence.

## Executive finding

The repository contains a credible immutable code anchor and useful result evidence, but it does **not** contain enough joined provenance to replay the paper experiment blindly or to prove which individual workbook supplied every manuscript table cell.

The most important findings are:

- The repository tag `V1.1-Experiment` is the strongest immutable pre-submission code anchor.
- The January-facing scorer file at that tag is byte-identical to the current file and is suitable as a behavioral regression reference.
- The schema embedded in historical prompts contains the known impossible asymmetric-system definition, but the runner did not execute JSON Schema validation at all.
- Historical JSON extraction used a greedy brace expression and the retry design could make as many as nine provider requests for one target/configuration.
- The large December 26 and 28 grids recorded 96 evaluated targets per configuration because the runner capped the test set at 96.
- Six completed December 30 local runs were separate one-repetition workbooks and did not share one target population. Their stored target lists contain 98, 98, 97, 98, 102, and 97 unique paths; four workbooks include four or five of the five demonstration paths.
- Historical examples included an internal `__path` field in model-visible JSON.
- Raw assistant responses, per-attempt errors, provider request IDs, exact model revisions, seeds, prices, and run-to-code fingerprints were not preserved together.

These findings justify the new strict schema, one-controller/three-total-request policy, immutable PostgreSQL evidence model, leakage-safe prompt renderer, and deterministic manifests. They do not change the accepted decision to run only a new Corpus v2.0.0 campaign.

## Evidence labels

This audit uses three labels:

- **Confirmed:** directly observable in an immutable Git object or the exact bytes of a named artifact.
- **Reconstructed:** supported by multiple consistent artifacts but missing one authoritative join, such as an invocation manifest.
- **Unresolved:** cannot be selected among plausible explanations from the available evidence.

File names and modification times are not treated as sufficient provenance. Git object IDs and SHA-256 values identify immutable evidence; SHA-256 identifies local-only evidence.

## Immutable pre-submission anchor

The strongest repository anchor is:

| Property | Value |
|---|---|
| Git tag | `V1.1-Experiment` |
| Commit | `b9683d2242aa5ca5b987440ea4b6f70bc1253c7e` |
| Tree | `35bf57411d7e8143a255415825f9bb0b644e8c6e` |
| Commit time | `2025-12-31T14:18:57+01:00` |
| Commit subject | `Scripts and results of the error analysis` |

The tag points to a commit, not an annotated tag object. Its history contains the `a251e1fe63bda1d4def5164fcf9f133f97850612` release-preparation checkpoint from earlier on December 31. The audited runner, schema, prompt, and historical converter bytes are the same at that checkpoint and `V1.1-Experiment`.

This tag is called the **pre-submission code anchor**, not proof that every manuscript number was produced from its exact working tree. The workbooks do not carry a Git commit fingerprint, and several December 30 artifacts are intentionally ignored by Git.

The current repository `HEAD` contains February 2026 changes. Current files must therefore not be used as substitutes for the tagged bytes when reconstructing January behavior.

## Pre-submission source hash inventory

All hashes in this table are calculated from `V1.1-Experiment:<path>`:

| Artifact | SHA-256 | Audit role |
|---|---|---|
| `benchmarking_example/randomShotsPhaseOne.py` | `2d4a6271fd86a076127dfb3f85589391cff9744efdfdee8ec570b52c74921df0` | Authoritative historical generation/evaluation behavior |
| `benchmarking_example/gt_json_maker.py` | `b8caf88bc39a1aa8d6b01bc7e0fe2ed6419de5b6450455f0d6dd5242d7ab3773` | Historical Turtle-to-lexical-JSON behavior |
| `benchmarking_example/data/Json_schema.json` | `c099f2cebe91e495c22506b8accd592c209059a69c60110b12e7a263a7ab7d9a` | Schema text embedded in historical prompts |
| `benchmarking_example/data/prompts/strict_minimal.txt` | `c184ec3edcdac4eee348fd315835110e7b9ce6f42b4229d09ad6501f0b8afedc` | Historical Strict minimal instructions |
| `benchmarking_example/data/prompts/constraint_tree.txt` | `e7355417a4153999287df4cbe199de47f33e2119840df644e06d2dc0efb12a42` | Historical constraint instructions |
| `benchmarking_example/data/prompts/matrix_tree.txt` | `f77aba03472c117b15aa2e3ac27d76519687629f7eb4d4fffb89e60c88237229` | Historical Matrix instructions |

The scorer hash is the fixed behavioral reference. The historical converter and prompt/schema bytes are evidence, not implementation templates.

## Post-submission variants that must not be confused with January

The repository later changed some of the same paths:

| Artifact state | SHA-256 or change | Consequence |
|---|---|---|
| Current `gt_json_maker.py` | `4ae635676d79ec665b3a5de1f0d472f3c85b2426a0c6db62a10302a845be2322` | February converter preserves directory structure and constraint prefixes; it is not the pre-submission converter. |
| Current tracked `Json_schema.json` at `HEAD` | `42d34405c7cd1439881bdc81386729cac1ce175eeb64de5b41e37ff58e52aaa2` | Post-submission schema edit; the working tree also contains a separate user edit and is excluded from historical conclusions. |
| Current `constraint_tree.txt` | `c3d8bf1b9f6a46341a90ce2e6a085c43aae33df551da30cd4c7db2431695c617` | February text adds an `allowed hasConstraint.on` category list that conflicts with component-target semantics. |
| Current strict prompt path | renamed to `strict_minimum.txt`, bytes unchanged | The January path was `strict_minimal.txt`; the new stable family is `strict-minimal`. |

I-ADOPT Lab records source identity by hash and provenance, never by a mutable path alone.

## Historical pipeline reconstructed from code

### Inputs

For each historical target, the runner consumed:

- One JSON gold record from `benchmarking_example/data/Json_preferred/test_set`.
- `definition`, falling back to `comment` when necessary.
- One prompt text file discovered from the historical prompt directory.
- The complete historical JSON Schema serialized into the prompt.
- Zero, one, three, or five JSON demonstration records.
- Provider model ID, temperature, mode, test-set cap, and worker count.

### Prompt construction

The runner appended instructions, schema text, demonstrations, the target definition, and an output-only heading into one user message. It serialized each full in-memory demonstration dictionary.

The loader had already inserted `__path` into those dictionaries. As a result, internal demonstration paths were model-visible. The new renderer intentionally exposes only the demonstration definition and reviewed six-field decomposition; category, path, IRI, and other provenance stay outside the prompt.

### Provider call and extraction

The historical runner called OpenRouter's OpenAI-compatible chat-completions endpoint. It did not record an immutable routed provider/backend revision.

JSON extraction removed Markdown fence markers and applied the greedy regular expression `\{.*\}` over the complete response. It then called `json.loads` on that span. Multiple brace-delimited objects or explanatory brace text could therefore be merged into one invalid candidate.

After parsing, the runner inserted the target `definition`, filled missing evaluated fields with empty values, and coerced a non-list Constraint value to an empty list. This was coercion, not JSON Schema validation.

### Historical retry count

The transport/model helper attempted up to three calls. The outer JSON-extraction helper invoked that helper up to three times. In the worst case, one variable/configuration could therefore cause:

```text
3 outer extraction attempts × 3 inner provider attempts = 9 provider requests
```

The prompt was not corrected with the prior response or exact validation error. The new experiment has one owner for the attempt lifecycle and permits exactly three provider requests total.

### Evaluation and persistence

The script immediately evaluated the processed prediction with its in-process scorer and wrote logs and XLSX workbooks. It did not create an independently versioned evaluation run, database lineage, leases, idempotency keys, or resumable task state.

Logs contain complete rendered prompts, gold JSON, and processed predicted JSON for completed targets. They do not provide the untouched provider response as a separately hashed artifact for each actual call. Workbooks store the selected processed prediction and scores, not a lossless attempt history.

## Historical schema audit

### Formal defect

The pre-submission schema's asymmetric object required:

```text
AsymmetricSystem
hasSource
hasTarget
hasNumerator
hasDenominator
```

Its `properties` block defined only `AsymmetricSystem`, `hasSource`, and `hasTarget`, while `additionalProperties` was false. Numerator and denominator were therefore simultaneously required and forbidden. Formally, neither a flow-shaped nor ratio-shaped asymmetric system could validate.

Other notable differences from the new contract include:

- Top-level model output included `label`, `definition`, and `comment`, although the evaluator scores none of them.
- Several evaluated top-level fields were optional rather than always present.
- A present Constraint array required at least one item, rather than permitting the explicit empty array used by the new six-field contract.
- A symmetric system permitted one part; the corrected contract requires at least two distinct non-empty parts.

### Actual runtime effect

The runner imported no JSON Schema validator and never called one. Therefore the defective schema affected the model-visible instructions but did not mechanically reject, retry, or drop asymmetric outputs.

This corrects a claim in the initial redesign input: the repository does **not** confirm that only schema-valid historical outputs were retained. Historical retries were driven by transport/empty/HTML failures or JSON extraction/parsing failure.

### Historical ratio projection

The pre-submission `gt_json_maker.py` read either `hasSource` or `hasNumerator` into the output key `hasSource`, and either `hasTarget` or `hasDenominator` into `hasTarget`. That collapsed ratio roles into flow-shaped lexical roles. The new deterministic importer must preserve the ontology's actual source/target versus numerator/denominator alternative.

### New-system decision

The new workflow does not reproduce these defects. It uses a corrected six-field lexical schema for both prompt text and runtime validation, followed by objective cross-field validation. Schema-valid but incorrect or all-empty content is accepted and scored; semantic quality never triggers a retry.

## Historical demonstrations and evaluation set

The historical ordered demonstration paths were:

1. `test_set/sfcWindmax.json`
2. `test_set/DetritalNitrogenConc.json`
3. `test_set/HeartRate.json`
4. `test_set/SoilMoist.json`
5. `test_set/SurfRunoff.json`

These are not the approved Corpus v2.0.0 demonstrations. The five new release-relative Turtle paths in `data/README.md` replace them for the new campaign.

The historical fixed mode excluded all five reserved paths even at zero shots. That behavior is preserved conceptually: the new five demonstrations are excluded from every score, and all other 97 Corpus v2.0.0 variables form the one evaluation population.

## Reconstructed late-December execution matrix

The following inventory describes the exact workbook Summary sheets that were inspected. A row count here means configuration rows in `Summary`, not target variables.

| Evidence | Status | Summary rows | Reconstructed grid | Targets per row |
|---|---|---:|---|---:|
| Local `randomShotsPhaseOne20251226_192545.xlsx` | Local-only/ignored artifact | 192 | 4 models × 3 prompts × 4 shots × 4 temperatures | 96 |
| Tagged `randomShotsPhaseOne20251228_061438.xlsx` | Immutable at `V1.1-Experiment` | 48 | Qwen3-32B × 3 prompts × 4 shots × 4 temperatures | 96 |
| Local `randomShotsPhaseOne20251228_103558.xlsx` | Local-only/ignored artifact | 48 | Qwen3-8B × 3 prompts × 4 shots × 4 temperatures | 96 |
| Six completed `randomShotsPhaseOne20251230_*.xlsx` files | Local-only/ignored artifacts | 1 each | Qwen3-32B × Strict minimal × 5 shots × temperature 0.5 × repetition 1 | Inconsistent: 97–102 |

The four models in the December 26 workbook are:

- `meta-llama/llama-3-8b-instruct`
- `mistralai/mistral-7b-instruct`
- `openai/gpt-4o-mini`
- `qwen/qwen3-32b`

The 96-target result is explained by the historical `--test-per-set` cap of 96: fixed mode selected a deterministic prefix after excluding the demonstrations. It must not be relabeled as a 97-variable evaluation.

The later tagged runner changed the default cap to 105 and restricted active constants to Qwen3-32B, five shots, and temperature 0.5. That does **not** establish one 97-variable December 30 population: the workbook metadata proves that the input directory/membership changed between invocations or that demonstration exclusion was inconsistently applied. The exact cause is not reconstructable from the retained artifacts.

There is an unresolved invocation-state discrepancy: fixed mode in the tagged CLI discovers all prompt files, but each December 30 workbook contains only Strict minimal. Possible explanations include direct function invocation, a temporary prompt directory, or an uncommitted runner state. The artifacts do not identify which explanation is correct.

## Latest completed December 30 runs

Each workbook below contains one `Summary` row and `Repetition = 1`, but its stored `TestedPaths` population differs:

| Workbook | Unique tested paths | Demo paths also tested | Exact F1 | Close F1 | SHA-256 |
|---|---:|---:|---:|---:|---|
| `randomShotsPhaseOne20251230_134942.xlsx` | 98 | 0 | 0.403 | 0.441 | `8c9efd31f1632b1660dfb7b945a9cee8502e7adde764e0fa63b23f109513e70d` |
| `randomShotsPhaseOne20251230_135722.xlsx` | 98 | 4 | 0.421 | 0.458 | `cfcef28f0af028c2145c167644c2ea796503e45a1c629c840d4ec868e0e7711c` |
| `randomShotsPhaseOne20251230_140332.xlsx` | 97 | 4 | 0.428 | 0.474 | `3e2d905bc4846fe0a9d39c4410d3398942cdf80b968959b5cb5df63e07921bf2` |
| `randomShotsPhaseOne20251230_142938.xlsx` | 98 | 4 | 0.412 | 0.453 | `0f19d05750c5167eeda9809b6fa3d59fe639d9687ca26a5ac49c28ad62e66f30` |
| `randomShotsPhaseOne20251230_143411.xlsx` | 102 | 5 | 0.438 | 0.461 | `f589a8c68fd4d44f3bbca6d6edb0dba37a00c65ba056add372ff200744aee605` |
| `randomShotsPhaseOne20251230_144925.xlsx` | 97 | 0 | 0.371 | 0.404 | `b1ec776673d9d55205db43da626399d52a22319b225bcbf95f371a6fa150a2a0` |

The spreadsheet audit imported each workbook read-only and parsed the exact JSON lists stored in `ExamplePaths`, `TestedPaths`, and `TestedLabels`. Within each workbook, tested paths are unique and the label/path counts agree. The demonstration-overlap count is exact string-set intersection with the five stored `ExamplePaths` values.

These scores must not be averaged or presented as a prespecified six-repetition estimate or as stochastic variation under one condition. They are separate executions without a shared population, registered repetition identity, seed record, immutable code/config fingerprint, or explicit selection rule. The new campaign now uses one repetition at every temperature under D-029 to reduce request volume, superseding the initial prospective five-repetition nonzero-temperature proposal. Neither proposal is justified by treating these incomparable historical files as a variance sample; their archived facts remain unchanged.

The log `randomShotsPhaseOne20251230_143758.log` contains initialization only and has no matching result workbook. It is an interrupted/aborted run, not a seventh completed repetition.

## Relationship to submitted tables

The artifacts are consistent with major parts of the paper-facing experiment design: OpenRouter model IDs, three prompt families, shot counts `[0, 1, 3, 5]`, temperatures `[0, 0.5, 1, 2]`, fixed demonstrations, and component-level Exact/Close scoring.

They are not sufficient to assert a complete table lineage because:

- Workbooks do not store the Git commit, source hash, schema hash, prompt hash, or exact command.
- Provider model revision/routing, seeds, and complete native parameters are absent.
- Some important workbooks are local ignored files rather than immutable tag contents.
- The 96-target grids and the inconsistent 97–102-target latest runs are not the same evaluation population; four latest workbooks also leak demonstration paths into scoring membership.
- The latest six runs do not identify which result, if any, was selected for a paper table.
- The tagged runner constants and the one-prompt December 30 workbooks do not fully explain one another.
- Raw provider-call attempts are not independently retained, so retry counts and response transformations cannot be reconstructed exactly.

Accordingly, the historical audit is adequate to freeze scorer behavior and identify architectural defects. It is not yet a proof that every manuscript table cell is reproducible. A later, separately approved legacy-table audit would need a frozen mapping from manuscript cells to exact workbooks/logs and would report irrecoverable fields explicitly.

## Consequences for I-ADOPT Lab

| Historical risk | New-system control |
|---|---|
| Mutable paths used as identity | SHA-256, Git blob/commit/tree, and immutable registry versions |
| Schema shown but not executed | Same frozen schema bytes in prompt and runtime validator, with hash equality preflight |
| Greedy JSON extraction | Deterministic single-object candidate extraction with ambiguity errors and fixtures |
| Silent coercion | Layered syntax/schema/cross-field validation; no invented repairs |
| Up to nine hidden calls | One task controller, three outbound requests total, no adapter retries |
| No correction feedback | Immediately previous raw response plus exact deterministic error bundle |
| Demonstration metadata leakage | Model-visible demonstration allowlist and snapshot tests |
| 96-target and inconsistent 97–102-target membership | Immutable 97-variable evaluation-population manifest, demonstration-exclusion constraint, and expected/observed count gates |
| One-row workbooks used as repetitions | Explicit repetition dimension in configuration, fingerprints, database, and reports |
| Processed prediction without raw-attempt lineage | Full sanitized request, untouched response, errors, usage, timing, and selection lineage in PostgreSQL |
| Model/container labels affect representation | Deterministic container display for storage; both system labels excluded from scoring |
| Post-hoc scorer changes | Pure versioned evaluator and append-only evaluation runs |

## Inputs and outputs of this audit

### Inputs

- Immutable Git objects under `V1.1-Experiment`.
- Current repository history used only to distinguish later changes.
- Historical `randomShotsPhaseOne` logs and workbooks.
- The submitted redesign Markdown as a claim/checklist source.
- User-approved decisions recorded in `DECISIONS.md`.

### Outputs

- The immutable pre-submission source/hash inventory above.
- The reconstructed pipeline and retry behavior.
- The formal-versus-actual schema determination.
- The late-December configuration/denominator inventory.
- An explicit list of evidence gaps and the controls required in the new design.

No historical file was modified, no provider was contacted, and no database was written during this audit.
