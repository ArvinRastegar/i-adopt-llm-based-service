# CLI and Configuration Contract

## Responsibility

The CLI exposes user commands and exit codes. Configuration loading reads, validates, resolves, canonicalizes, hashes, and freezes `parameters.yml`.

The CLI does not contain corpus parsing, provider calls, retry logic, concurrency mechanics, scoring, SQL queries, or scientific decisions. Root `main.py` is only a compatibility entry point calling `iadopt_lab.cli.main()`.

## Inputs

- Command name: `ingest`, `preflight`, `plan`, `run`, `resume`, `score`, `rank`, `report`, or `dry-run`
- Path to `parameters.yml`, defaulting to the directory-root file
- Optional execution filters that can narrow already planned work without changing scientific settings or the 97-variable population
- Environment variable names referenced by configuration
- Optional explicitly selected `--env-file` path; no implicit directory discovery or shell sourcing
- PostgreSQL DSN through `DATABASE_URL`

## Outputs

- Typed command result and deterministic nonzero exit code on failure
- Human-readable summary containing campaign/task counts and next action
- Exact original configuration artifact and hash
- Resolved selected-provider set, provider-owned model lists, and campaign fingerprint
- Preflight compatibility report

No command prints a credential or full raw response by default.

## Processing contract

1. Read exact YAML bytes.
2. Reject duplicate YAML keys and unsafe/custom tags.
3. Validate against `schemas/parameters.schema.json` with unknown properties forbidden.
4. Resolve `campaign.providers` as a non-empty, duplicate-free selection of `psnc`, `openrouter`, or both.
5. Resolve each selected provider's enabled model list; retain each `(provider, model ID)` pair as a distinct identity.
6. Validate model capabilities and reasoning profiles.
7. Validate the exact 97-variable evaluation-population manifest and ranking policy.
8. Reject live placeholders, missing credential names, incomplete billing/estimate evidence, or invalid optional caps for any selected provider; require separate live authorization before dispatch.
9. Canonically serialize the resolved non-secret configuration.
10. Calculate original-YAML and resolved-campaign SHA-256 values.
11. Persist the configuration before planning or execution.

Edits to an inactive provider remain visible in the original-YAML hash but do not change the selected scientific fingerprint. Provider-selection order does not change canonical identities. A model ID need only be unique within its provider; the same native ID offered by two providers represents two distinct configurations.

The initial campaign selects the three exact PSNC IDs and three exact OpenRouter IDs supplied by the owner and recorded in `docs/model-catalog.md` and `parameters.yml`. Three models per provider is an initial selection, not a schema limit: later campaigns may select either provider or both and any non-empty owned model list. Provider/model selections are editable before freezing and immutable within a frozen campaign; changes create a new campaign rather than modifying work under resume.

This selection and cost-disclosure contract uses parameter schema version `2.1`; the old scalar `campaign.provider` is rejected rather than silently migrated. With live mode disabled, an enabled intended model may have `capabilities: null` and `reasoning_profiles: []` while verification is pending. Draft loading reports unknown fields; exact plan freeze and live readiness reject them without skipping the selected model or assuming `not_applicable`. A null model ID is permitted only in a disabled draft catalog slot. Every selected enabled model needs concrete ID, capability, and reasoning evidence before execution.

`run` and `resume` use the complete frozen selection by default. Optional diagnostic filters may prioritize or temporarily narrow dispatch, but omitted planned tasks remain pending and cannot be excluded from the completion count. The workflow owns continued execution, safe provider-specific cooldowns, scoring, ranking, and final reporting.

Billing resolution uses `providers.<id>.billing.mode`, `basis`, and optional `maximum_provider_cost`, together with campaign-wide cost accounting. `non_billed` requires recorded billing evidence and explicit zero monetary rates; the PSNC declaration is the owner's account-specific statement that access is free. `metered` requires complete price-card evidence, not a mandatory spending cap. Both provider and campaign cap amounts default to null (uncapped); explicit zero is a zero ceiling that prevents positive-cost dispatch, and a positive amount enables that ceiling. Missing or unknown cost is never inferred to be zero. D-027 requires `cost_accounting.require_pre_run_estimate: true`, `estimate_policy_version: pre-run-estimate-v1`, advance disclosure of the plan's estimate, and separate explicit live authorization. An estimate is not a cap and actual cost may exceed it.

Runtime credentials may be supplied by process environment or an explicitly selected `.env` file. The owner has an existing repository-root `.env`; do not copy it into this experiment. The loader only accepts allowlisted names referenced by active provider configuration plus `DATABASE_URL`, with process values taking precedence. No shell execution, interpolation, expression evaluation, unrelated-key import, content/hash persistence, or secret-bearing DSN logging is permitted. Offline commands do not require live provider credentials.

## Configuration keys consumed

This is the only component that reads the complete `parameters.yml`. It validates every top-level section and emits narrower resolved records so downstream components never parse YAML independently.

## Reasoning expansion

- Capable model: exactly `disabled` and `enabled` profiles, each with explicit native fields.
- Incapable model: exactly `not_applicable`, with an empty native field object.
- Placeholder or unsupported native field: preflight failure.
- A model's enabled effort/value is part of the scientific fingerprint.

## State and side effects

Configuration validation is read-only until the user invokes a command that registers a campaign. Registration writes immutable configuration records to PostgreSQL. It never modifies `parameters.yml`.

## Failures

Named failures include YAML syntax, duplicate key, YAML alias, a parameter file over the 2 MB limit, schema violation, non-finite or otherwise uncanonicalizable value, unknown/empty/duplicate provider selection, duplicate model ID within one provider, no enabled model for a selected provider in live mode, unsupported capability, invalid reasoning mapping, missing manifest, unresolved placeholder, absent credential, unsupported PostgreSQL version, and billing/budget gate failure.

## Planned public functions

These planning signatures describe the decomposed responsibilities and the names
used while the component was designed. The implemented public interface, including
every name and signature that differs, is recorded under *Implementation interface
(version 1)* at the end of this document.

### `load_configuration(path) -> RawConfiguration`

- **Input:** One explicit filesystem path to a YAML file; the caller is responsible for selecting the path, while this function owns reading its exact bytes.
- **Action:** Read the file once, reject a file over 2 MB, decode it as UTF-8, reject duplicate keys, aliases and unsafe/custom YAML tags, parse supported YAML values, validate the complete parameter schema, and calculate the canonical and source bytes. Under D-049 the schema gate lives here rather than in resolution, so a structurally invalid file cannot reach any later stage.
- **Output:** An immutable record containing the exact original bytes, the canonical bytes and the source hash. No environment values are resolved here.
- **Raises:** Typed unreadable-file, oversize-file, non-UTF-8, YAML-syntax, duplicate-key, alias, unsafe-tag, schema-violation, unsupported-value, or hashing errors.
- **Side effects:** Filesystem read only; no database, environment mutation, provider request, or output write.
- **Determinism:** Identical file bytes produce identical parsed content and hash independent of current directory.

### `resolve_configuration(raw, environment_capabilities) -> ResolvedConfiguration`

- **Input:** One `RawConfiguration` plus typed, non-secret runtime/capability facts such as supported PostgreSQL major version and provider/model feature declarations.
- **Action:** Take the already schema-valid snapshot and resolve the unique non-empty selected provider set and each provider's enabled owned models, expand applicable reasoning profiles and repetition rules, validate billing modes, mandatory estimate policy and optional caps, normalize artifact paths relative to the project root, validate the 97-member population and ranking contract, and canonicalize the selected scientific configuration. Expansion is the union of each provider's model grid, never a cross-product between every provider and every model.
- **Output:** An immutable campaign configuration containing the selected provider set, fully resolved provider-owned models/profiles/grid values, provider billing and execution limits, normalized artifact identities, original-YAML hash, canonical selected-configuration bytes/hash, and a list of non-secret environment-variable names required later.
- **Raises:** Typed schema, unknown-field, provider/model ownership, duplicate-ID, capability, reasoning-profile, manifest, path, placeholder, scientific-combination, or canonicalization error.
- **Side effects:** None. It does not read credential values, write PostgreSQL, or contact a provider.
- **Determinism:** Identical raw bytes and capability facts yield the same resolved record and scientific fingerprint; inactive-provider edits affect only the original-YAML identity.

### `validate_live_readiness(resolved) -> PreflightReport`

- **Input:** A `ResolvedConfiguration` and read-only facts supplied by artifact registries, database inspection, credential-presence checks, capability fixtures, price cards, the disclosed plan-bound cost estimate, and explicit live authorization.
- **Action:** Evaluate every live gate without short-circuiting so the owner receives one complete report: artifacts/hashes, each selected provider's enabled models, native reasoning mappings, sampling support, context limits, database/migration state, secret presence by name, timeouts, concurrency, task/call counts, billing evidence, price versions, estimate disclosure/authorization lineage, and optional provider/campaign caps. Null caps are valid and impose no monetary gate; explicit zero caps prevent positive-cost dispatch. Plan preparation and estimate creation do not themselves require live authorization or send requests.
- **Output:** A typed report containing each gate ID, status, evidence reference, blocking reason, exact planned counts, estimates/conditional bounds and unresolved assumptions, cap status or `uncapped`, and overall readiness. Secret values and authorization headers are never included.
- **Readiness boundary:** Every selected provider must pass before the first live dispatch. Runtime isolation of a later provider failure never permits bypassing initial preflight or silently dropping an unready selected provider.
- **Raises:** Only integrity failures that prevent producing a trustworthy report; ordinary readiness failures are returned as failed gates.
- **Side effects:** Read-only filesystem/database/environment inspection as explicitly injected; zero generation requests and no campaign/task mutation.
- **Idempotency:** Repeating against unchanged resolved inputs and readiness facts yields an equivalent report aside from an operational inspection timestamp outside its scientific hash.

### `main(arguments=None) -> int`

- **Input:** An optional explicit argument sequence; when omitted, the process argument vector is used. Environment access occurs only through delegated configuration/command services.
- **Action:** Parse the documented command and options, construct/inject the appropriate application service, invoke it once, render its safe summary, and translate typed outcomes into stable exit codes.
- **Output:** Integer process exit code plus user-facing stdout/stderr owned by the presentation layer. Structured command results remain available to tests without parsing terminal text.
- **Raises:** Unexpected programmer defects may propagate to the top-level error boundary; all documented user/configuration/operational failures become typed messages and nonzero exit codes.
- **Side effects:** Only those explicitly owned by the selected command service. `main` itself performs no SQL, provider call, scientific calculation, retry, or worker scheduling.
- **Testability:** Argument parsing and exit-code mapping must be testable with injected services and no network/database dependency.

### `load_runtime_secrets(env_file, required_names, process_environment) -> RuntimeSecrets`

- **Input:** An optional explicitly selected `.env` path, an allowlist of configured active-provider environment names and `DATABASE_URL`, and an injected process-environment mapping. No default file search occurs.
- **Action:** Read the selected file only when provided, parse supported literal key/value syntax without interpolation or evaluation, select only allowlisted names, and overlay already-set process values. Reject malformed selected values without echoing them; distinguish missing from present values.
- **Output:** An in-memory, non-serializable secret resolver plus safe name/presence/source-kind facts for preflight. Secret values, file contents/hashes, and secret-bearing DSNs never enter returned evidence, configuration snapshots, exception messages, or logs.
- **Raises:** Typed unreadable-file, unsupported selected-value syntax, or invalid allowlist errors with secret-free diagnostics; missing optional/unselected provider names are not errors.
- **Side effects:** Read-only selected-file access; no process-environment mutation, shell sourcing, interpolation, network, database, copying, or persisted artifacts.
- **Determinism:** Identical selected literal values, allowlist, and process mapping yield identical runtime resolution; process environment always wins.

## Acceptance tests

- Valid PSNC-only, OpenRouter-only, and combined examples
- Empty, duplicate, or unknown provider selection rejected
- Initially three models per provider and later list sizes handled without hard-coded counts
- Union of provider-owned model grids; no accidental cross-provider calls
- Same native model ID across providers retained as distinct identities
- Non-billed and metered selections accept null caps; explicit zero/positive limits have distinct enforcement semantics, and missing-cost evidence is rejected
- Estimate disclosure and separate live authorization required; producing an estimate makes no provider calls and does not authorize dispatch
- Optional env-file selection imports only active allowlisted names, process values win, and no file content/hash or secret-bearing DSN is persisted
- Shell commands/interpolation are never evaluated and unrelated environment keys are never imported
- Exactly 97 non-demonstration evaluation members and no split fields
- Complete ranking policy with all-result retention enabled
- Inactive-provider models ignored by expansion
- Frozen provider/model selections cannot be edited by resume
- Reasoning-capable and incapable expansion
- Placeholder blocks live mode
- Secret values absent from resolved snapshot
- Equivalent YAML produces the documented canonical fingerprint
- Unknown keys and unsupported capability combinations fail

## Implementation interface (version 1)

`load_parameters(path, schema_path=None)` is the implemented reader; the contract above
calls it `load_configuration`. It returns a `Configuration`, the single byte-backed
snapshot type that also stands in for the planned `RawConfiguration` and
`ResolvedConfiguration`: it exposes `canonical_bytes`, `original_bytes`, `issues`, a
`data` property returning an independent parsed copy, and a `sha256` property.

`resolve_configuration(raw, environment_capabilities=None)` returns another
`Configuration` whose `issues` carry the plan-readiness paths rather than raising for
them. It does not re-validate the parameter schema: under D-049 that gate runs once, inside
`load_parameters`, so the same input is rejected one call earlier than the planning
signatures described. `resolve_configuration` still raises for ownership, duplicate,
model and mode inconsistencies it alone can see. `validate_live_readiness(resolved, facts=None)` returns a plain dictionary with
`ready`, `issues`, `configuration_sha256` and `live_calls_enabled` in place of the
planned `PreflightReport` type; injected `facts` supply the artifact, database,
credential, estimate and authorization evidence. `main(argv=None)` is the entry point
the contract calls `main(arguments=None)`; `load_runtime_secrets(env_file,
required_names, process_environment)` is unchanged.

Per-model capability checking has no adapter-side function. Declared capabilities are
checked against the grid inside `resolve_configuration`, and `probing.py` establishes
those declarations from live observation before a plan is frozen.
