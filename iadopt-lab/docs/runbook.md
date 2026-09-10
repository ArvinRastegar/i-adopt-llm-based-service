# Experiment Runbook

## 1. Status and purpose

This runbook describes how I-ADOPT Lab is prepared, verified, executed, interrupted, resumed, and reported.

The offline stages are executable today through the `iadopt-lab` command. The execution stages are implemented but gated: `run` and `resume` need a prepared PostgreSQL database, and live dispatch additionally needs verified model capabilities, frozen sampling values, price evidence covering every planned model, a disclosed estimate, and explicit `--authorize` with a named `--actor`. Live campaigns have been executed against both providers; see D-039 to D-042 for what was run and measured.

```bash
iadopt-lab preflight                                  # what is still unfrozen
iadopt-lab ingest --source-repository <clone>         # materialize the pinned corpus
iadopt-lab verify                                     # recheck every hash
iadopt-lab probe-models --provider psnc --write        # LIVE: measure model capabilities
iadopt-lab plan --synthetic                           # expand the grid offline
iadopt-lab database prepare | start | init | migrate  # isolated local PostgreSQL
iadopt-lab evidence --plan <p> --out <e>              # derive the cost-estimate evidence
iadopt-lab estimate --plan <p> --evidence <e>         # pre-run cost disclosure
iadopt-lab report --plan <p> --observations <o>       # rank and export
iadopt-lab run --estimate <e> --authorize --actor <name>   # LIVE: execute the campaign
iadopt-lab resume                                     # continue an interrupted campaign
```

`probe-models` and `run` are the only commands that contact a provider. `probe-models` sends
three short throwaway calls per model and, with `--write`, rewrites the models block of
`parameters.yml`; it is capability work, not scored generation, and produces no benchmark
prediction. `run` refuses to dispatch without `--authorize` and a named `--actor`, and refuses a
plan whose price card does not cover every planned model.

The runbook separates three activities that must never be confused:

1. **Offline verification:** deterministic tests and a mock-provider dry run; no OpenRouter or PSNC request.
2. **Live canary:** a small, separately authorized provider test used only to verify the frozen wire contract and cost recording.
3. **Scientific campaign:** the complete parameter grid over all 97 non-demonstration variables, executed only from a frozen campaign record.

## 1a. The actual command sequence

This is the order that was used for the campaigns in [campaign log](campaign-log.md). Every
step gates the next, so a failure stops before anything downstream runs. Run from
`iadopt-lab/`.

```bash
# 0. Database, once per machine. Both are idempotent.
iadopt-lab database start
iadopt-lab database migrate

# 1. Measure model capabilities and write them into parameters.yml. LIVE: a few short
#    throwaway calls per model. Omit --write to review the verdicts first.
iadopt-lab probe-models --provider psnc --env-file ../.env --write

# 2. Configuration must resolve with zero issues before planning.
iadopt-lab preflight

# 3. Expand the frozen grid.
iadopt-lab plan --out outputs/plan.json

# 4. Derive the estimate's inputs. --ceiling-basis is the observation establishing that the
#    output ceiling bounds generation; without it a metered campaign is blocked and a
#    non-billed one carries a recorded warning.
iadopt-lab evidence --plan outputs/plan.json --out outputs/evidence.json   --ceiling-basis "<the measurement that justifies the ceiling>"

# 5. Disclose the cost. Note --json is a GLOBAL flag and precedes the subcommand.
iadopt-lab --json estimate --plan outputs/plan.json --evidence outputs/evidence.json   > outputs/estimate.json

# 6. Execute. LIVE. --authorize and --actor are both required.
iadopt-lab run --env-file ../.env --estimate outputs/estimate.json --authorize --actor <name>

# Interrupted? Resume continues from durable checkpoints, provided parameters.yml and the
# implementation are unchanged.
iadopt-lab resume --env-file ../.env
```

Four things about this sequence are easy to get wrong:

- **`preflight --live` cannot pass from the command line.** Its five gates
  (`artifacts_verified`, `database_verified`, `credentials_present`, `estimate_disclosed`,
  `live_authorized`) are runtime facts injected by the runner, and the CLI passes none, so
  they always report missing. Use plain `preflight` to check the configuration; `run`
  enforces the live gates itself.
- **`--json` is global**, so `iadopt-lab --json estimate …`, not `estimate … --json`.
- **Changing anything re-plans.** The campaign identity covers `parameters.yml` and the
  implementation file index, so any edit to either produces a different campaign and
  `resume` will refuse the old one rather than silently continuing it.
- **Switching provider** means editing `campaign.providers` and pointing
  `cost_accounting.price_card_manifest` at that provider's card. The cards are
  `price-card-psnc.yml` (non-billed) and `price-card-openrouter.yml` (metered).

## 2. Roles and approval boundaries

| Role | Responsibility |
|---|---|
| Experiment owner | Approves the scientific configuration, evaluation population, selected providers and their model lists, reasoning profiles, disclosed cost estimate/live execution, and promotion of results |
| Operator | Runs preflight, planning, execution, resume, integrity checks, backups, and reports without changing scientific settings |
| Implementation maintainer | Maintains code, migrations, schemas, provider adapters, tests, and versioned documentation |
| Database administrator | Provides PostgreSQL 16, backup/restore, access control, and operational monitoring |

One person may hold several roles, but the database must retain which actor performed each approval or operation. No per-variable human choice occurs during a campaign.

The database administrator role does not require an AWS service or another person. A local PostgreSQL 16 instance is prepared with `iadopt-lab database prepare` and inspected with DBeaver; see `local-database.md`. Preparation has been run; its generated role credentials live in the ignored `.runtime/` directory and are never committed.

The experiment owner must explicitly approve the implementation gate before executable code is added. Live canary and full live execution each require later, separate authorization because they incur external calls and cost.

## 3. Phase A — documentation review

Before implementation begins, review:

- `README.md` for the plain-language experiment.
- `DECISIONS.md` for settled scientific choices and live values still to freeze. D-021 preserves historical no-interpretation prompt wording with minor necessary edits; D-022 accepts fractional member credit; D-023 accepts the ranking formula; D-024 defers additional statistics while retaining all score evidence.
- `TECHNICAL_SPECIFICATION.md` for the authoritative protocol.
- `docs/components/` for planned module/function boundaries.
- `docs/database.md` and `docs/retry-and-resume.md` for durable evidence and recovery.
- `docs/scorer-parity.md` for the January reference metric and accepted `january-derived-member-credit-v1` protocol, including unscored system labels, fractional member contributions, and explicit asymmetric role evidence.
- `docs/prompt-specification.md` and `schemas/README.md` for the model boundary.
- `docs/test-plan.md` and `docs/reproducibility.md` for verification and reconstruction.

Documentation approval authorizes implementation only if the user says so explicitly. Editing documentation does not imply permission to call a provider.

## 4. Phase B — implementation preparation

After implementation approval, create the planned independent runtime beneath `iadopt-lab/` and complete these steps in order:

1. Add the Python project and lock its dependencies.
2. Add forward-only PostgreSQL 16 migrations.
3. Add the strict parameter and lexical decomposition schemas.
4. Materialize the immutable Corpus v2.0.1 source snapshot from the tagged Git object.
5. Generate the file manifest and deterministic gold projection.
6. Add and freeze the three prompt templates and ordered demonstration manifest.
7. Implement one-request OpenRouter and PSNC adapters with recorded fixtures.
8. Implement extraction, validation, correction attempts, task planning, leases, resume, the evaluator, and reports.
9. Pass every static, unit, integration, parity, concurrency, and recovery test. Scorer tests must preserve unaffected January behavior and verify each accepted member-credit and container-label correction separately.

The implementation must not import the historical benchmark scripts or the sibling variable-description service at runtime. Hashed copies or manifests may be retained only as reference evidence.

## 5. Phase C — data preparation

### 5.1 Verify the source lock

The ingester must verify all of these values before materialization:

| Property | Required value |
|---|---|
| Repository | `https://github.com/i-adopt/Corpus` |
| Tag | `v2.0.1` |
| Commit | `2598bf91fa927b78a6529bae7864ef0f7d485b73` |
| Tree | `df665d32bb2433a60742c80a4a53908dc7ebde0c` |
| Turtle files | `102` |

The implementation must read the immutable Git object, not whichever files happen to be checked out in a mutable worktree.

### 5.2 Build and validate the canonical corpus

For every file, verify Turtle syntax, one variable root, required cardinalities, labels, exact `rdfs:comment`, system shapes, constraints, and release-relative category information. Project the six lexical gold fields deterministically, write hashes, then validate every record against the lexical and semantic contracts.

Activation is atomic. If one record cannot be represented without guessing, the snapshot remains non-runnable and the complete import report is reviewed before any prompt is rendered.

### 5.3 Freeze demonstrations and evaluation population

Resolve the exact five demonstration paths in their approved order. Exclude all five from scoring, including zero-shot scoring.

Materialize one evaluation-population manifest containing every other Corpus v2.0.1 variable exactly once: 97 variables. Store canonical order, source/gold hashes, Corpus identity, exclusions, version, and manifest hash. There is no training, development, test, holdout, or stratified partition.

## 6. Phase D — configure one campaign

Copy the documentation example in `parameters.yml` into a campaign-specific reviewed configuration while preserving its schema. Do not store credentials in it.

For each campaign:

1. Select PSNC only, OpenRouter only, or both using the non-empty unique list `campaign.providers` under parameter schema version `2.1`.
2. Enable one or more exact model IDs inside each selected provider's own model list.
3. Record exact model revisions/backend constraints where the provider exposes them.
4. For every controllable reasoning model, freeze one explicit `disabled` mapping and one explicit `enabled` mapping/effort.
5. For a model without a controllable reasoning parameter, use only `not_applicable` and send no reasoning field.
6. Freeze supported sampling fields, output limit, D-029's one repetition at every temperature, per-provider concurrency/rate limits, billing basis/price evidence, and cost-estimation assumptions. Leave monetary-cap amounts null unless a cap is deliberately requested.
7. Keep provider-native structured output disabled for the primary comparison.
8. Use continued execution and provider-fair scheduling so one `run` or `resume` advances all selected work through scoring, ranking, and final reports, with provider-local runtime failures isolated from healthy providers.

The initial intended selection is the three PSNC and three OpenRouter model IDs supplied by the owner and recorded in `model-catalog.md` and `parameters.yml`. All six are selected in the draft, but live calls remain disabled until deployment capabilities and the live gates are verified. Three per provider is not a fixed limit. Later campaigns may select either provider or both, with different model lists. Models from an unselected provider are catalog data only and are not planned or called.

Provider/model selections are editable before a campaign is frozen. After freezing, adding, removing, or renaming a selected provider/model requires a new campaign with new campaign-scoped task identities. Resume uses the original frozen records rather than rereading a changed model list as instructions to modify pending work.

PSNC is `non_billed` based on the experiment owner's statement that their access is free. Record that account-specific billing basis and explicit zero rates while still collecting every call, token, retry, latency, and usage record. OpenRouter is `metered`: freeze complete price-card evidence and disclose the pre-run estimate under D-027. No monetary ceiling is required: null means uncapped, whereas an explicitly configured zero means a zero ceiling. Missing pricing or cost evidence must never be interpreted as free access. Track costs and reservations in both modes and enforce only explicitly configured non-null caps.

The existing repository-root `.env` is the owner's credential source. The `--env-file` option can load this explicit path without shell execution or interpolation; process environment takes precedence. Read only allowlisted active-provider/database settings, ignore unrelated service settings, and never print, copy, hash, or persist the file or secret values. Supplying credentials is not live-execution authorization. Database connection details will be prepared separately; see `local-database.md`.

## 7. Phase E — preflight

The planned `preflight` command is read-only with respect to providers. It must finish successfully before planning. It verifies:

- Parameter YAML syntax, duplicate keys, strict schema, and semantic combinations.
- No unresolved placeholder in any selected live field.
- A non-empty unique selected provider list and a non-empty owned enabled model list for each selected provider.
- Current provider/model capability evidence and exact reasoning controls.
- Corpus, canonical records, demonstrations, the 97-variable evaluation population, prompts, schemas, scorer, ranking policy, embedding model, and all hashes.
- PostgreSQL major/migration version and required privileges.
- Secret availability by name without storing secret values.
- Provider fixtures, no hidden SDK retries, timeouts, concurrency, rate limits, explicit billing modes/bases, price cards where metered, and valid optional caps if present. Null caps pass this check.
- Exact logical task count and maximum request count (`tasks × 3`).

Preflight produces a stored, human-readable report. Preparation checks precede planning; the final live gate additionally verifies the planned estimate's disclosure and separate live authorization. All selected providers must pass before the first live dispatch. A failure blocks the applicable stage; it never silently drops a provider, model, or parameter. Continuing a healthy provider after another develops a runtime issue does not bypass this startup requirement.

## 8. Phase F — deterministic planning

The planned `plan` command freezes a campaign and expands:

```text
for each selected provider:
  each enabled model owned by that provider
× each model-applicable reasoning profile
× three prompt variants
× shot counts 0, 1, 3, and 5
× configured temperatures
× applicable repetitions
× all 97 variables in the evaluation population

campaign plan = union of these provider-owned grids
```

Every task/resolved run fixes exactly one provider and owned model. Never multiply all providers by all model IDs. Under D-029, every configured temperature (`0.0`, `0.5`, `1.0`, `2.0`) has exactly one repetition for every selected provider/model. Repetition indexes remain materialized, not created ad hoc by workers. Future changes require a new frozen plan and cost estimate; retries remain separate from repetition membership.

Before approval, planning displays task counts, expected and maximum calls, token bounds, estimated cost, target denominators, and a breakdown by provider/model/reasoning/prompt/shot/temperature. Replanning identical inputs returns the same identities and adds no duplicate work.

The mandatory `pre-run-estimate-v1` receipt includes per-model/provider and total costs, price-card date/hash and original currency, any FX source/date, initial and maximum-three request scenarios, output/reasoning tokens, and the larger retry prompts containing prior responses/errors. Expected ranges must state their assumptions; a conservative upper estimate is conditional on verified bounds and prices, never a guaranteed bill or an enforced cap. If estimates are not defensible, show the missing inputs and resolve them before live authorization. No paid calibration call is implicit. Persist the receipt/hash, configuration/price identities, disclosure timestamp, and explicit authorization. A changed grid/price/assumption set needs a new estimate disclosure for that new plan.

## 9. Phase G — offline dry run

The planned `dry-run` command uses PostgreSQL 16, three real Corpus variables, and a deterministic mock provider. It makes no network request. The fixtures must exercise:

1. A valid JSON object on attempt 1.
2. An invalid response followed by a valid correction.
3. Three invalid responses followed by a terminal-invalid explicit-empty prediction.
4. A deliberate interruption after raw-response persistence, followed by resume without another provider invocation.
5. Two logical provider fixtures with separate model lists: one enters a temporary cooldown while the other advances, then both finish without cross-provider routing or extra attempts.
6. Uncapped execution retaining all cost evidence, plus an explicitly enabled optional-cap fixture where paid work pauses and non-billed PSNC continues; contrast with shared database/integrity failure stopping all dispatches. Verify missing estimate/disclosure blocks live dispatch even when caps are null.

Accept the dry run only if all prompts, raw responses, validation errors, attempt counts, category metadata, predictions, component contributions, aggregate metrics, state transitions, hashes, and report artifacts reconcile with the plan.

Dry-run records are permanently marked synthetic/test and cannot be selected by a live scientific report.

## 10. Phase H — live canary

A live canary is not part of the offline test suite and requires its own disclosed cost estimate and explicit authorization. It uses a small frozen subset, with no mandatory monetary ceiling, to verify:

- The exact provider URL and model IDs.
- Request/response wire compatibility.
- Reasoning disabled/enabled behavior for controllable models.
- Temperature/output-limit support.
- Non-streaming usage, finish reason, provider IDs, latency, and cost capture.
- Text extraction, validation feedback, and the one-request adapter boundary.

Canary evidence belongs to a clearly named campaign. It is not silently merged with the scientific campaign unless it was planned as part of that campaign before any call.

## 11. Phase I — full evaluation grid

After canary verification, disclose the full campaign estimate and obtain full live-execution authorization before invoking `run`. Canary approval alone does not authorize the full grid. It processes the complete union of all selected provider/model grids over all 97 evaluation variables and continues through scoring, ranking, and final reports. Workers use a bounded global pool, fair provider scheduling, independent provider concurrency/rate/cooldown gates, leases, heartbeats, and provider/global cost accounting. A rate or cooldown wait allocates no attempt number and occupies no worker slot or database transaction.

Each task is one variable plus one fully resolved configuration. It receives at most three outbound requests total. A content-invalid response may cause a correction attempt containing the immediately previous raw output and exact errors. The first valid prediction ends generation. Three delivered content-invalid responses create an explicit-empty scored prediction.

Operational failures and unresolved ambiguous deliveries pause or fail their affected task/model/provider scope; they are not converted to empty predictions even when the request allowance is exhausted. Healthy selected providers continue. Known safe transient cooldowns are persisted and revisited automatically by the active runner. No task switches provider/model and no fourth request is created. If an optional monetary cap is configured, exhausting it blocks further paid reservations while evidenced non-billed PSNC requests reserve zero and remain eligible. With null caps there is no monetary-cap stop; an estimate is not a ceiling. Shared database/integrity failure or user interruption stops all new dispatches. Resume follows durable evidence, not operator memory.

Every repetition and every configuration remains stored. The runner does not stop early because one configuration appears worse or discard results after a provisional ranking.

If all remaining work is blocked without a safe scheduled automatic continuation, the process records an incomplete paused/failed campaign and explains the remaining scope. Otherwise it keeps advancing the frozen plan. Completion requires every selected provider's expected tasks and required final artifacts; finishing only PSNC or only OpenRouter in a combined campaign is partial progress.

## 12. Phase J — ranking and complete-result verification

Before ranking, reconcile the expected task count for every configuration across all selected providers and owned models: 97 variables multiplied by the declared repetitions. A content-invalid terminal empty prediction is a scored result. A missing task or unresolved operational failure remains visible and makes that configuration `not_rankable` until resolved. The campaign-wide ranking includes all eligible configurations from the combined selection; provider-filtered reports preserve those ranks and clearly record the filter.

Apply accepted D-023 policy `mean-repetition-micro-close-f1-v1`: compute micro Close F1 separately for each repetition by summing contributions over all 97 variables, then rank complete configurations by the arithmetic mean of those unrounded repetition-level values, descending. Exact ties share competition rank. This differs from averaging 97 per-variable F1 values. Store every variable/component/repetition Exact/Close Precision, Recall, F1 and contribution, ranking input, rank, tie, lower-ranked configuration, and unranked reason. Mean repetition micro Exact F1, validity, category results, latency, usage, and cost remain analysis columns and do not change the primary rank.

Mean, median, variance, mode, standard deviation, range, and IQR over per-variable scores are deferred until the database is populated, along with further variability and uncertainty analyses. Their eventual definitions must state scope and formulas. Completing these analyses is not a prerequisite for generation, scoring, ranking, or retaining results.

For this one-repetition campaign, the ranking mean equals the single repetition's micro Close F1. Run-to-run variability within a configuration cannot be estimated from this singleton; do not report zero stochastic variance or repetition-based confidence intervals. Statistics across the 97 different variables describe a different distribution and remain available for later analysis.

Ranking is a derived, versioned database result. It never triggers another provider request, changes a parameter, selects a smaller population, or deletes any evidence.

## 13. Monitoring and controlled stop

Monitor counts rather than mutable output files:

- Planned, leased, in-flight, response-stored, retry-pending, valid, terminal-invalid, operational-failure, ambiguous, and complete tasks.
- Provider/model calls and remaining attempt allowances.
- Per-provider eligibility, cooldown reason/next eligible time, and paused scopes.
- Token usage, latency, rate-limit responses, billing basis, and actual/estimated/unavailable cost.
- Per-provider and total costs/reservations against the disclosed estimate; remaining cap only when an optional cap is configured, otherwise explicitly `uncapped`.
- Lease health, worker heartbeat, database availability, and evidence-commit failures.

On interrupt or shared fatal failure, stop all new claims and dispatches, allow safe checkpointing, persist evidence already received, record the pause, and exit. A provider-local optional-cap boundary or runtime failure pauses that scope while eligible work at the other provider continues. Exhausting an explicitly configured global monetary cap blocks new paid reservations while evidenced non-billed PSNC stays eligible; null caps impose no such stop. Never change scientific settings to make an interrupted campaign easier to finish.

## 14. Resume procedure

The planned `resume` command first generates a reconciliation report covering every selected provider. It verifies expected/stored counts, expired leases, incomplete stages, ambiguous deliveries, remaining attempt allowances, evidence hashes, provider cooldowns, and provider/global spent/reserved/remaining cost, then continues all safe unfinished work through the normal execution loop. The report is retained as evidence; ordinary safe continuation does not require a new per-step human choice.

Resume actions follow the last verified durable checkpoint:

- Stored raw response: extract and validate locally; no provider call.
- Stored valid prediction: score locally; no generation.
- Stored score: complete/rebuild aggregates idempotently.
- Retry-pending content failure: create only the next numbered attempt.
- Ambiguous delivery: reconcile provider evidence; do not blind retry.
- Complete task: verify and skip.

Repeated resume commands must converge without duplicate logical tasks, attempts, predictions, scores, or verified provider requests.

An optional diagnostic dispatch filter cannot remove the other provider's tasks from the frozen plan or make the campaign complete. Changed provider/model selections create a new campaign; they are never applied to an existing campaign through `resume`.

## 15. Reporting and promotion

Reports always name explicit campaign, evaluation-population, evaluation, and ranking-policy IDs. Generate:

- Per-variable and per-component Exact/Close Precision, Recall, F1, and underlying match/confusion evidence.
- Micro Precision, Recall, and F1 with numerators, denominators, and support.
- Validity and terminal-invalid rates.
- Retry/error distributions.
- Provider/model/reasoning/prompt/shot/temperature/repetition summaries.
- A complete configuration table with rank or explicit `not_rankable` reason.
- Category, subcategory, and full-category-path summaries with counts.
- Token, latency, call, and actual/estimated/unavailable cost summaries, including evidenced non-billed zero cost and billing basis.
- Reproducibility and integrity manifests.

Promote a result for paper use only when every expected task is complete, no operational failure is hidden, denominators reconcile, hashes verify, a backup restore has been tested, and the report can be regenerated from PostgreSQL facts.

## 16. Command surface

Every command below exists and is implemented. `score` and `rank`, which earlier revisions
of this table listed as separate commands, were never built as such: scoring happens inside
the workflow as each task terminates, and ranking happens inside `finalize_campaign` and
`report`. They are listed here as the boundaries they became, not as pending work.

| Command | Input | Main output | Provider calls |
|---|---|---|---:|
| `ingest` | Frozen Corpus source descriptor | Active verified corpus and manifests | 0 |
| `preflight` | Parameters and frozen artifacts | Pass/fail and count/cost report | 0 generation calls |
| `plan` | Successful preflight | Immutable campaign/runs/tasks | 0 |
| `dry-run` | Mock fixture manifest | Synthetic end-to-end campaign | 0 |
| `run` | Planned campaign ID with one or both providers | Continued progress through all planned work, scoring, ranking, and final reports; explicit incomplete status if blocked | Yes, only for authorized live campaigns |
| `resume` | Existing campaign ID with immutable provider/model selection | Reconciliation and the same continued execution over all safe unfinished work | Only for eligible unfinished tasks |
| `report` | Frozen plan and observations | Ranking plus hashed derived export | 0 |
| `evidence` | Frozen plan and the price card | Cost-estimate input document | 0 |
| `estimate` | Plan and evidence document | Disclosed pre-run cost estimate | 0 |
| `probe-models` | Provider and model IDs | Measured capabilities written to `parameters.yml` | **Yes** — short throwaway probes |
| `database` | Action name | Local PostgreSQL prepared, started, initialized or migrated | 0 |

The root `main.py` only routes these commands. Scientific algorithms, concurrency, retry policy, provider mappings, SQL, and scoring remain in their dedicated modules.
