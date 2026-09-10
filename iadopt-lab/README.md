# I-ADOPT Lab

I-ADOPT Lab is the isolated experiment system for evaluating how well language models decompose scientific-variable definitions into the lexical components used by I-ADOPT.

The goal is deliberately narrow: take the exact definition of each variable in Corpus v2.0.1, ask a configured language model for a structured decomposition, validate the returned JSON, and calculate reproducible Precision, Recall, and F1 scores. Every prompt, model response, validation failure, retry, prediction, and score will be retained in PostgreSQL so a result can be traced back to the exact evidence that produced it.

## Current status

**Implemented, and live runs have been executed.** The evaluator, corpus ingestion, prompt rendering, extraction, validation, provider adapters, planning, workflow, persistence, migrations and reporting are implemented. The suite passes at **263 unit tests without a database**, and more with `IADOPT_LAB_TEST_DATABASE_URL` configured, which adds the PostgreSQL integration and recovery modules. Counts move as tests are added; treat the command output as authoritative over any number quoted in prose.

`campaign.live_calls_enabled` is `true`. Real provider calls have been made and scored: a complete 97-variable campaign against `qwen/qwen3-32b` on OpenRouter, and partial campaigns against `GLM-5.2` on PSNC. Those results are experimental output from a scoped single-configuration campaign, not the full parameter grid.

The Corpus v2.0.1 migration (D-030), the derived per-variable files (D-031) and the revised OpenRouter selection (D-032) are executed and verified. Decisions D-033 to D-039 record what the live runs established, including that neither selected model can disable reasoning.

Known limitations are listed in `docs/read-only-review-2026-09-09.md` together with their resolutions.

## Commands

```bash
iadopt-lab preflight            # draft readiness; --live applies the full dispatch gate
iadopt-lab ingest --source-repository <clone>
iadopt-lab verify               # re-check every materialized artifact
iadopt-lab probe-models --provider psnc --write   # measure model capabilities; rewrites parameters.yml
iadopt-lab plan --synthetic     # expand the grid; omit --synthetic for the real population
iadopt-lab evidence --plan <p> --out <e> --ceiling-basis "<observation>"
iadopt-lab estimate --plan <p> --evidence <e>
iadopt-lab report --plan <p> --observations <o> --format csv --out <name>
iadopt-lab database prepare|start|init|migrate
```

Two of these need care. **`probe-models` makes live provider requests** — three short throwaway
calls per model, never a corpus prompt — and with `--write` it rewrites the
`providers.<name>.models` block of `parameters.yml` in place, leaving the rest of the file
byte-identical. It exists because capability fields gate live execution and must not be inferred
from a model name: it measures each model's default reasoning, whether the off-switch changes it,
and whether structured output is enforced, and writes back disabled any model whose evidence is
contradictory or inconclusive. **`evidence`** produces the document `estimate` consumes, tokenizing
every planned prompt and reading the same frozen price card the runner reserves against. Its
`--ceiling-basis` marks the output ceiling verified. Without it a metered campaign is blocked; a fully evidenced non-billed one proceeds with a recorded warning, since zero cost does not bound tokens or runtime.

## Plain-language workflow

1. Read the 102 Turtle files from the pinned Corpus `v2.0.1` release (D-030).
2. Convert each variable deterministically into the lexical JSON used as ground truth, and write two derived views of it — a provenance metadata file and a human-readable variable file carrying label, definition, the six fields and their source IRIs (D-031).
3. Keep five approved variables, in a fixed order, as the demonstration pool.
4. Build a prompt from one prompt strategy, the output schema, zero to five demonstrations, and the target definition.
5. Send the prompt to its configured model on PSNC or OpenRouter. One campaign can include either provider or both, each with its own model list.
6. Store the full request and untouched response.
7. Extract and validate one JSON object.
8. If invalid, send the previous response and exact validation errors back to the same model, with the same experiment parameters.
9. Stop after at most three provider requests for that variable and configuration.
10. Score the valid prediction, or score an explicit empty prediction if three delivered model responses remain content-invalid. Operational/provider failures remain separate and block completion instead of being scored as model errors.
11. Retain every attempt, prediction, item score, component score, repetition, and aggregate result for later analysis.
12. Rank every complete parameter configuration by the mean of its repetition-level micro Close F1 scores over the same 97-variable evaluation population.

## Important rules

- Choose PSNC only, OpenRouter only, or both in one execution. Each provider has its own editable model list, and every task stays with its chosen provider/model.
- The selection is PSNC: GLM-5.2, Qwen3.8 27B, DeepSeek V4 Flash; OpenRouter: Qwen3 8B, Qwen3 32B, GPT-4o mini (D-032). Exact API IDs are recorded in `parameters.yml`; the lists can change before a new campaign is frozen. GPT-4o mini is the only hosted closed-weight selection, so OpenRouter price evidence is mandatory before live use.
- One `run` or `resume` invocation advances all selected work through generation, scoring, ranking, and reports. Provider-specific pauses leave other eligible work running; unfinished work remains resumable and is never called complete.
- Your reported no-charge PSNC access is recorded with zero monetary cost and normal usage tracking. Before live work, you receive a cost estimate for every selected model and the whole campaign. No spending cap is required; actual costs remain fully tracked and may differ from the estimate.
- A model that supports reasoning controls is tested with reasoning disabled and enabled. Models without that capability are run once without a reasoning parameter.
- Use one repetition at every temperature for both providers. The repetition field remains recorded, but this campaign does not measure run-to-run variability. With one repetition, the ranking mean is simply that run's micro Close F1 over all 97 variables.
- Every provider request counts toward the maximum of three attempts for one variable/configuration task. Provider adapters cannot add hidden retries.
- Only exhaustion by content-invalid model responses creates a scored empty prediction. Authentication, infrastructure, budget, and unresolved delivery failures remain operational failures.
- Symmetric- and asymmetric-system container labels are stored deterministically but are not scored. Matching members earn fractional credit when a simple entity is compared with a system or when two systems partly match. Missing and extra members contribute to Recall and Precision; asymmetric role evidence remains explicit.
- Demonstration variables are never included in an independent evaluation score.
- There is no training, development, test, holdout, or stratified split. Every configuration is evaluated on all 97 non-demonstration variables.
- Ranking uses unrounded values, gives exact ties a shared rank, and retains lower-ranked results. Incomplete configurations remain visible but unranked.
- Keep the historical prompts' no-interpretation policy with only minor workflow/schema changes. Known tensions with benchmark answers are documented and left unchanged.
- Retain Exact and Close Precision, Recall, F1, and their supporting counts for each variable, component, and repetition. Additional descriptive statistics over variable scores will be calculated later from the populated database.
- Completed work is resumable and is not repeated after interruption.
- API keys and database passwords are read from environment variables and are never written to configuration, prompts, logs, or result artifacts.

## Planned active scope

The active experiment includes Corpus ingestion, lexical JSON generation, prompt construction, OpenRouter and PSNC model calls, response extraction, JSON validation, validation-feedback retries, evaluation, PostgreSQL persistence, resumable execution, and reporting.

The following are intentionally deferred:

- RDF or JSON-LD generation
- SHACL validation
- Entity linking
- A deterministic non-LLM baseline
- Human-in-the-loop review

The deferred work is documented under `docs/future/` so that it can be designed later without being confused with functionality available now.

## Configuration

`parameters.yml` is the only human-edited experiment configuration. Set `campaign.providers` to `[psnc]`, `[openrouter]`, or `[psnc, openrouter]`, then edit each provider's `models` list. It also defines prompts, shots, temperatures, reasoning modes, repetitions, the evaluation population, ranking, execution limits, and cost-estimate policy. Optional spending-cap amounts are null by default, meaning no enforced ceiling. Change selections before starting a new campaign; resume always uses the original frozen selection. The existing repository-root `.env` supplies the named provider credentials through the planned explicit `--env-file` option; secrets never enter configuration snapshots or results.

The effective configuration will be validated, normalized, hashed, and stored with every campaign before any task is planned.

## Database access

AWS is not required. The local setup runs PostgreSQL 16 on this computer; the experiment writes its evidence there and DBeaver connects to the same database for queries and analysis. Persistent storage and backups protect the results. Local execution needs this computer and the database service to stay running; after interruption, the workflow resumes from stored checkpoints. See `docs/local-database.md` for setup and connection details, and use `iadopt-lab database prepare|start|init|migrate`. Local preparation has been run and its credentials live in the ignored `.runtime/` directory. The PostgreSQL suite passes: set `IADOPT_LAB_TEST_DATABASE_URL` (and `IADOPT_LAB_TEST_APP_DATABASE_URL`) to a dedicated test database and the integration and recovery tests run instead of skipping.

## Documentation map

- `TECHNICAL_SPECIFICATION.md`: authoritative experiment and implementation specification
- `DECISIONS.md`: settled scientific decisions, intentional corrections, and live-execution values still to freeze
- `docs/migration-v2.0.1.md`: authorization checklist for the Corpus v2.0.1 upgrade and the derived per-variable files
- `docs/architecture.md`: planned independent directory/package structure and why configuration is organized this way
- `docs/model-catalog.md`: the six selected model IDs, provider endpoints, shared SDK, and how to change provider/model selections
- `docs/prompt-specification.md`: exact model-visible content and correction-message contract
- `docs/database.md`: PostgreSQL records, evidence, constraints, and transaction boundaries
- `docs/retry-and-resume.md`: three-request lifecycle and interruption recovery
- `docs/scorer-parity.md`: January evaluator compatibility and the accepted system-label and fractional member-credit corrections
- `docs/repository-audit.md`: what the pre-submission code and late-December result artifacts actually prove
- `docs/documentation-audit.md`: requirement-by-requirement and file-by-file consistency review, evidence checks, and decision history
- `docs/test-plan.md`: verification layers, fixtures, failure injection, and acceptance gates
- `docs/runbook.md`: future prepare/preflight/dry-run/canary/campaign/resume procedure
- `docs/reproducibility.md`: hashing, backups, reconstruction, and re-evaluation contract
- `docs/components/`: exact input/output and behavior contracts for every planned executable component
- `docs/future/`: explicitly deferred experiment stages
- `data/README.md`: source, canonical, manifest, and demonstration data policy
- `prompts/README.md`: prompt versioning and rendering policy
- `schemas/README.md`: JSON Schema responsibilities
- `reference/README.md`: legacy evidence and deferred reference-file policy
- `outputs/README.md`: generated export policy

## Approval gates

The work is split into two explicit gates:

1. **Documentation gate:** complete and review every input, output, invariant, failure mode, and reproducibility decision.
2. **Implementation gate:** write code, migrations, schemas, tests, and the offline dry run only after separate approval.

Both gates are passed. The scientific and flexible provider-execution policies in D-021–D-026 are accepted; D-027 replaces mandatory monetary caps with pre-run cost disclosure, D-028 records the credential source and local database preparation, and D-029 sets one repetition at every temperature. The scorer protocol is `january-derived-member-credit-v1`. Source, dependencies and embedding artifacts are implemented, locked and verified, and campaigns have run against both providers.

What still gates each *new* live campaign, rather than the project as a whole: capability evidence measured per model by `iadopt-lab probe-models`, a price card covering every enabled model, a disclosed estimate bound to the frozen plan, and explicit `--authorize` with a named `--actor`. These are per-campaign preconditions, not outstanding implementation work.
