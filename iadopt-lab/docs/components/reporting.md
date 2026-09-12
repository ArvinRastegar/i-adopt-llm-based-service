# Reporting Contract

## Responsibility

Read immutable PostgreSQL facts for explicit campaign/evaluation IDs, calculate the accepted aggregate metrics and configuration ranking without changing evidence, and export reproducible human- and machine-readable results. Additional descriptive statistics and uncertainty analyses are deferred until the database is populated.

Reporting does not change scorer behavior, repair predictions, discard lower-ranked configurations, or query an implicit latest campaign.

## Inputs

- Explicit campaign, resolved-run, evaluation-run, scorer, and evaluation-population IDs
- Versioned query/report definitions
- Frozen repetition membership and accepted ranking policy
- Requested output formats and destination under `outputs/`

## Outputs

- Per-variable, per-component, and aggregate Exact/Close Precision, Recall, and F1, with confusion contributions, numerators, denominators, and support
- Complete per-configuration results over the same 97-variable population
- Deterministic ranking of every eligible fully resolved configuration
- Provider/model/prompt/shot/temperature/reasoning/repetition comparisons
- Category and subcategory summaries with counts
- Validity, attempts, validation failures, tokens, latency, and cost summaries
- CSV, canonical JSON, and optional XLSX exports
- Report manifest linking every output to source IDs/query/code/hash

## Required rules

- Every metric is explicitly named; no unlabeled `F1` or `accuracy`.
- Every percentage and metric includes denominator/support.
- Reasoning enabled/disabled/not-applicable is always visible.
- Provider/model identity remains explicit in every configuration, repetition, and summary. A combined campaign ranks all its eligible configurations together under the same accepted policy; provider-filtered views preserve that campaign-wide rank. Cross-campaign comparisons require explicit compatible identities and never merge separate runs as identical work.
- Every ranked configuration must cover the same immutable 97-variable evaluation population and satisfy completeness gates.
- Ranking uses accepted policy `mean-repetition-micro-close-f1-v1` from D-023: sum confusion contributions over the same 97 variables for each repetition, calculate its micro Close F1, then average the unrounded repetition values and rank descending. Exact ties share competition rank without secondary tie-breaking.
- D-029 gives the current campaign one repetition at every temperature, so each primary mean equals that single micro Close F1. Preserve the repetition record and generic averaging function; do not add implicit repeat calls or change to averaging per-variable F1.
- A mean of 97 per-variable F1 values is a different, macro statistic and is not the ranking input.
- Every configuration remains in result tables and exports regardless of rank.
- Expected configuration coverage includes every selected provider and each frozen owned model. A blocked provider remains visible with its incomplete configurations; other providers' completion does not make the whole campaign complete.
- Monetary summaries distinguish evidenced `non_billed` zero cost from metered actual/estimated cost and unavailable evidence. PSNC's owner-declared free access remains attributed to its frozen billing basis; usage, latency, retries, and calls remain reportable regardless of monetary cost.
- Unrounded database values are used; rounding occurs only for display.
- Generated exports are replaceable derivatives, while their manifests remain evidence.

## Deferred statistical analysis

Under accepted D-024, after the database is populated, separate versioned analyses may compute mean, median, variance, mode, standard deviation, range, and interquartile range over the 97 per-variable F1 values, or examine distributions across repetitions and categories. These statistics are not required for initial experiment completion or ranking. Confidence intervals are also deferred. Their later report definitions must identify the metric (Exact or Close, Precision/Recall/F1), scope, membership, formulas, variance convention, mode/tie handling, and quantile method; those choices need not be decided now. Retaining every unrounded metric and its underlying contributions makes this possible without generating predictions again.

That reuse applies only to observations actually collected. D-029's one-repetition campaign cannot estimate within-configuration run-to-run variability or repetition-based confidence intervals; do not label singleton variance as observed zero stochastic variation. Distributions across the 97 variables or across different configurations are not independent repeated runs of one configuration. Additional repetitions would require new authorized generation, even though the database and generic reporting contracts already support them.

## Configuration keys consumed

- `ranking.*`
- `reporting.*`
- `evaluation.*` identities for metric naming and provenance
- `cost_accounting.*`
- `database.evidence_policy` for required lineage/retention checks

## Planned public functions

These planning signatures describe the decomposed responsibilities and the names
used while the component was designed. *First implementation interfaces and evidence
shape* below records the implemented interface, including every name and signature
that differs.

### First implementation interfaces and evidence shape

Reporting reads no database. Every function takes *observations* — the rows
`workflow.build_observations(tasks)` projects from durable task evidence — so the
planned query-object inputs (`query`, `runs`, `evaluation_ids`, `evaluation_id`) do not
exist; the caller reads the evidence and passes the rows in. `build_run_summary` and
`build_repetition_summary` are not implemented as named functions: per-attempt and
per-scope summarization are the private helpers `_attempt_summary` and `_scope_summary`,
and repetition means are calculated inside the ranking pass below. The full implemented
signatures are `build_configuration_ranking(plan, observations, *,
scorer_identity=None, population_categories=None)` and
`build_category_summary(observations, population_categories=None)`.

`build_configuration_ranking(plan, observations)` is pure. `plan` is the frozen
`experiment-plan-v1` mapping from the planner, including its population,
configurations, and repetition runs. Each observation contains a planned
`run_id`, `variable_id`, the full evaluator `evaluation` record, an explicit
`terminal_invalid` boolean, source `category`, `subcategory`, `category_path`,
and an `attempts` list. The evaluator metadata must contain the same variable
identity. An explicit unresolved operational observation may use `evaluation:
null`; it is retained and blocks rankability, never converted to a model failure.
Missing tasks are also retained as named coverage gaps. Synthetic plans use
their declared small population; only live plans enforce the 97-member gate.

The report retains the full observations, every planned configuration, all
repetition aggregates, Exact/Close mean repetition Precision/Recall/F1 receipts,
and incomplete reasons. The primary ranking is the exact rational mean of
repetition micro Close F1 only. Display ordering within a shared competition
rank uses configuration identity without changing the tie. Category summaries
group within a run, preserving provider/configuration/repetition ownership, then
retain both top-category and exact subcategory-path scopes.

`build_category_summary(observations)` is also pure and requires compatible
scorer/backend identities plus unambiguous source classification. Attempt
summaries use available `usage` and `latency_seconds` from an attempt or its
`response` object. Prompt/input and completion/output token aliases are accepted;
reasoning tokens remain separate and are never added again to completion tokens.
An optional attempt `cost` receipt contains `mode`, `amount`, `currency`, `basis`,
and `kind` (`actual` or `estimated`). A declared `non_billed` amount must be zero.
Missing cost or usage is explicitly unavailable, not inferred as zero. Every
original attempt and error remains in the canonical report regardless of which
summary fields are available.

`export_report(report, format, destination, *, outputs_root)` renders canonical
JSON or CSV only; XLSX is optional and not implemented. CSV retains one row per
configuration, fraction receipt columns, and full canonical configuration/source
observation JSON columns. All text cells are formula-injection escaped. Export
paths are constrained to the explicit outputs root and reject symlink escapes.
Bytes are written atomically, verified, and accompanied by a deterministic
manifest. Re-exporting identical bytes is idempotent; a different existing
artifact/manifest is not overwritten and requires another destination name.
No database writes, provider requests, prediction repairs, or deferred statistics
are performed by these functions.

### `build_run_summary(query) -> RunSummary`

- **Input:** Typed query naming explicit campaign, evaluation, scorer, population and optional resolved-run/configuration IDs plus approved provider/model and other grouping/filter dimensions; frozen selected-provider/model membership and billing evidence are available for coverage and cost checks.
- **Action:** Validate identity compatibility and read immutable facts, calculate named quality/validity/retry/usage/latency/cost summaries from stored unrounded values, and attach denominators/support and unavailable reasons.
- **Output:** Timestamp-free canonical summary with source IDs/query version, metrics, counts, lineage hashes, and integrity warnings; no implicit latest selection.
- **Raises:** Unknown/incompatible ID, unsupported filter/group, missing required lineage, incomplete/corrupt evidence, or query-version mismatch.
- **Side effects:** Read-only PostgreSQL access; no evidence mutation or provider/evaluator execution.
- **Determinism:** Same database snapshot, explicit IDs, and query version yield identical canonical summary bytes.

### `build_repetition_summary(runs) -> RepetitionSummary`

- **Input:** Complete compatible repetition-level Exact/Close Precision, Recall, and F1 aggregate records for explicitly named configurations and accepted ranking policy `mean-repetition-micro-close-f1-v1`.
- **Action:** Verify expected repetition membership and 97-variable coverage; retain each repetition's metrics/contributions and calculate the arithmetic mean of unrounded repetition micro Close F1 for ranking, plus the mean repetition micro Exact F1 analysis column. Preserve Precision and Recall separately without deriving F1 from averaged Precision/Recall.
- **Output:** Per-configuration primary/secondary means with every source repetition ID, unrounded Exact/Close Precision/Recall/F1 value, denominator/support, formula/policy identity, and hashes. Additional descriptive statistics and confidence intervals are absent until a separately versioned analysis defines them.
- **Raises:** Mixed configuration/population/scorer identities, missing/duplicate repetition, non-finite metric, incorrect denominator/support, or unknown ranking policy.
- **Side effects:** None beyond read-only source retrieval by the caller.
- **Determinism:** Input ordering does not affect results or canonical output order.

### `build_category_summary(evaluation_ids) -> tuple[CategorySummary, ...]`

- **Input:** Explicit compatible evaluation/population IDs and requested exact category, subcategory, full-path, component, and repetition dimensions.
- **Action:** Join evaluation items to immutable source-path classification, verify every member maps once, sum contributions within each scope, and calculate named metrics with counts/support.
- **Output:** Canonically ordered category summaries retaining exact upstream spelling, optional separate reporting alias, population coverage, metric numerators/denominators, and lineage hashes.
- **Raises:** Incompatible evaluations, missing/ambiguous category mapping, duplicate item, unsupported grouping, or contribution/coverage mismatch.
- **Side effects:** Read-only database access.
- **Determinism:** Exact category-path UTF-8 ordering and frozen metric rules determine output order/values.

### `build_configuration_ranking(evaluation_id, protocol) -> ConfigurationRanking`

- **Input:** One explicit evaluation ID, exact 97-member population identity, and accepted ranking protocol `mean-repetition-micro-close-f1-v1` containing configuration identity fields, primary metric, repetition aggregation, direction, tie, completeness, and retention rules.
- **Action:** Enumerate every planned configuration across every selected provider and its owned models; verify `97 × repetitions` coverage; derive each repetition's micro Close F1 from summed contributions and take the arithmetic mean of those unrounded values; mark incomplete/operationally unresolved cases not rankable; order eligible values descending and assign shared competition ranks without secondary tie-breaking. Preserve provider/model identities and never collapse matching model names across providers.
- **Output:** Immutable complete ranking containing all eligible and ineligible configurations, every repetition/input value, ranks/ties, secondary analysis values, failure reasons, source IDs/hashes, and policy hash.
- **Raises:** Population/scorer mismatch, hidden/missing configuration, duplicate result, incorrect coverage/mean, unknown policy, non-finite primary value, or ranking invariant failure.
- **Side effects:** No provider request or evidence mutation. Persistence is performed separately by the ranking repository.
- **Determinism:** Same explicit evaluation facts and protocol produce identical ranks and canonical bytes.

### `export_report(report, format, destination) -> ExportManifest`

- **Input:** One validated immutable report object, approved format (`canonical-json`, `csv`, or optional `xlsx`), explicit destination constrained under `outputs/`, and versioned exporter policy.
- **Action:** Render stable columns/order/number formatting and untrusted text escaping, write through a temporary file with atomic replacement, hash the final bytes, then generate a manifest binding output to report/query/source/code identities.
- **Output:** Export manifest containing format/path, byte length/hash, row/sheet counts, column schema, source lineage, exporter version, and generation status.
- **Raises:** Unsupported format, destination escape, unsafe/unrenderable value, overwrite-policy violation, write/atomic-rename failure, or post-write hash mismatch.
- **Side effects:** Writes only the named derived artifact/manifest under `outputs/`; never changes PostgreSQL evidence.
- **Idempotency:** Same report/exporter policy produces semantically identical content; formats with unavoidable package metadata must define and test canonicalization before being called reproducible.

## Acceptance tests

- Recompute every displayed metric from stored contributions
- Exact/Close Precision, Recall, and F1 remain available at variable, component, and repetition scopes
- Ranking averages repetition micro Close F1 rather than per-variable F1 or F1 recomputed from averaged Precision/Recall
- Additional descriptive statistics and confidence intervals are not prerequisites for campaign completion
- Counts accompany category/subcategory values
- Reasoning/provider dimensions retained
- Combined-campaign ranking includes all selected providers and models with no identity collisions
- Provider-filtered reports retain campaign-wide rank and record the filter
- Blocked provider configurations remain visible and prevent complete-campaign promotion
- Explicit non-billed zero cost is distinguishable from unavailable cost, with all usage retained
- Ranking population and configuration coverage reconcile exactly
- Lower-ranked and tied configurations remain exportable with all supporting results
- Stable output ordering and hashes
- CSV/JSON/XLSX cross-format reconciliation
- No implicit latest-run query
