# Implementation Record

## Authorization and scope

The owner explicitly authorized implementation after D-029. Implement only the lexical decomposition experiment inside `iadopt-lab/`, including PostgreSQL and offline tests. The six selected live models remain disabled for execution until deployment profiles, artifact identities, a disclosed cost estimate, and separate live authorization are ready. No existing service files or root `.env` are implementation targets.

## Delivery order and concrete boundaries

1. **Foundation and installation.** Input: accepted contracts and Python environment. Output: isolated installable Python 3.12 project, dependency lock, strict canonicalization, typed request/result records, parameter schema and private env-file loader. `uv` locks dependencies; tests run in the new project environment, not the old service environment. Scientific metadata uses canonical versioned JSON and SHA-256; operational timestamps are separate.
2. **Pure evaluation.** Input: validated gold/prediction mappings plus an explicitly identified similarity callable. Output: serializable exact fractional component counts, Precision/Recall/F1 and match evidence. Regression fixtures cover unaffected January behavior and all accepted member-credit changes. Embedding artifacts are an optional local-only dependency, never implicitly downloaded by scoring.
3. **Corpus and prompt boundary.** Input: immutable Corpus Git objects, historical prompt sources, and schema contracts. Output: 102 hash-verified source/canonical records, five ordered demonstrations, 97 targets, frozen prompt/schema artifacts, deterministic extraction/validation evidence. No runtime import from legacy service code.
4. **PostgreSQL.** Input: isolated server connection and forward migrations. Output: constrained registry/task/attempt/evidence/metric tables and short transactional repositories. Use psycopg 3; the async runner delegates synchronous database work without holding a transaction during HTTP or scoring. Raw responses commit before interpretation. Fenced leases, uniqueness and immutable evidence protect resume.
5. **Provider/configuration/planning.** Input: selected-provider union, frozen capabilities and prompts. Output: deterministic run/task plan, conditional cost estimate, one-request async SDK adapters and full response/error evidence. Native reasoning fields are explicit profile mappings. SDK/transport retries are zero. Mock adapters require no API key.
6. **Workflow and reports.** Input: frozen plan and durable repositories. Output: bounded provider-fair execution, exactly three requests maximum per task, checkpoint recovery, exact shared ranking and regenerable reports. Synthetic dry-run campaigns use a labelled three-variable subset; real scientific campaigns require all 97 targets. Synthetic scores cannot masquerade as scientific results.
7. **Acceptance.** Run unit, provider-contract, scorer, PostgreSQL, recovery, and offline end-to-end checks. Record the actual commands, counts, artifacts, and unverified boundaries here before handoff. No live calibration or inference is part of acceptance.

## Local infrastructure and secrets

The planned database uses a separate Compose project, loopback-only port, persistent volume, and distinct bootstrap/migration/app/reader roles. Bootstrap generates new random database secrets in ignored owner-only runtime files; it does not reuse API keys or edit the existing parent `.env`. Setup never deletes an existing volume or database. DBeaver uses the reader account. Offline testing uses a separately named test database and cannot drop a user database.

## Implementation details to lock

The project targets Python 3.12, with exact dependency resolution recorded by `uv.lock`; runtime evidence records the actual interpreter build. Shared SDK interfaces are checked against the [official Python SDK reference](https://developers.openai.com/api/reference/python) and local/mock transport behavior. Exact container digest and runtime/dependency evidence are recorded when installation succeeds. SDK examples do not override scientific prompts or silently fill live capability fields.

## Status

Implementation is complete for the scoped campaigns described below, and live runs have been executed. The offline PostgreSQL dry run passes end to end, including finalization. Remaining limitations are tracked in `docs/read-only-review-2026-09-09.md`.

### Verified state as of the D-030/D-031/D-032 documentation pass

The offline suite passes: 285 passed with 14 skipped, and 299 passed with no skips when the PostgreSQL test database is configured. All 14 skips are the PostgreSQL integration and recovery tests, which require `IADOPT_LAB_TEST_DATABASE_URL` and did not run in this environment; the largest single module is therefore unexercised here. `resolve_configuration` against the current `parameters.yml` reports eight draft-readiness issues — six missing model capability declarations, `top_p`, and `max_output_tokens` — which is the expected pre-freeze state.

### Executed after that pass

`src/iadopt_lab/cli.py` and the root `main.py` now exist, so the declared `iadopt-lab` console script works. The CLI parses arguments and delegates; it contains no loop, provider call, retry decision, SQL, scoring, or worker management, per `docs/architecture.md` section 4. Commands: `preflight`, `ingest`, `verify`, `plan`, `estimate`, `report`, `database`.

The v2.0.1 migration is complete and verified; `docs/migration-v2.0.1.md` records the executed steps, four unanticipated findings, and the acceptance results. `variable_id` remains commit-derived by explicit owner confirmation, recorded in D-030. The suite is 274 passed / 14 skipped, and the corpus verifies with 204 derived files byte-identical to their canonical parents.

Two gaps remain open:

1. **Delivery step 7 (Acceptance) has run offline.** `iadopt-lab dry-run` completes three synthetic tasks, finalizes, stores the ranking and report, and marks the campaign complete; repeating it is idempotent. A live end-to-end campaign has also run against `qwen/qwen3-32b`.
2. **The PostgreSQL suite passes** when `IADOPT_LAB_TEST_DATABASE_URL` and `IADOPT_LAB_TEST_APP_DATABASE_URL` are set.

The D-032 OpenRouter selection (`qwen/qwen3-8b`, `qwen/qwen3-32b`, `openai/gpt-4o-mini`) is applied in `parameters.yml` and validates. The PSNC selection is unchanged.

### Concrete configuration/provider boundary

`load_parameters(path)` returns validated original data/bytes; `resolve_configuration` returns a canonical selected-provider snapshot and explicit draft-readiness issues. It does not replace unknown live model fields with mock facts. The strict schema uses reasoning-profile records `{mode, request_fields}` and a capability declaration with required boolean `temperature`, `seed`, `structured_output`, and `reasoning_control`; optional bounded sampling/capability evidence is explicit. `expand_campaign` accepts a resolved snapshot and canonical variables, returning immutable JSON-compatible run/task records; only a separately labelled synthetic fixture may use fewer than 97 targets.

Exact money is serialized as decimal strings and exact score fractions as numerator/denominator receipts. Canonical JSON accepts finite JSON numbers but rejects implicit Decimal serialization; callers must encode exact quantities explicitly. Original YAML and canonical snapshots retain separate hashes. The env-file loader never implicitly searches or modifies `.env` and allows only explicitly requested names.

Local installation succeeded with Python 3.12.1, `uv.lock`, and an isolated `.venv`; the SDK resolves to OpenAI Python 2.54.0, psycopg 3.3.5, and pytest 8.4.2. The owner subsequently started Docker; its server reports 28.4.0. Database integration uses a new isolated project, without changing the existing variable-description service containers.

### Local release input confirmed

The owner originally supplied `Corpus-2.0.0` at `/Users/rastegar-a/Documents/GitHub/I-ADOPT-Variables`. All 102 release-relative TTL paths and exact Git blob hashes matched commit `8097662ca323771fd977d22cdb8c3e58b7b7d64a`, tree `bd9cf247d22c5b8345796572e3e55f333460c6ce`, with no missing or extra TTL files. That import produced the snapshot currently on disk.

Under D-030 the read-only source became `Corpus-2.0.1` in the same clone, tag `v2.0.1`, commit `2598bf91fa927b78a6529bae7864ef0f7d485b73`, tree `df665d32bb2433a60742c80a4a53908dc7ebde0c`, also exactly 102 TTL files with no path added, removed, or renamed. Both tags are present in that clone. The re-import has been executed and verified; see `docs/migration-v2.0.1.md`. This directory is read-only input, not an implementation target. Ingestion verifies it against the bundled release lock and copies exact TTL bytes into the lab before deriving canonical records. Its operator-local path is provenance, not the portable scientific identity; another verified copy yields the same corpus identity. No source clone or network access is required during ingestion.
