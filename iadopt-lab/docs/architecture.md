# Planned Repository Architecture

## 1. Design goal

`iadopt-lab/` is a self-contained experiment project inside the repository. Its runtime does not import code, prompts, schemas, generated JSON, or mutable outputs from the older benchmark directories. Historical files are copied or described only as hash-verified evidence under `reference/`.

The structure separates scientific contracts, configuration, immutable inputs, application services, the pure evaluator, database migrations, tests, and derived outputs. This keeps the thin CLI understandable while making provider concurrency and resume behavior independently testable.

This document describes the layout as built. Where it diverges from the layout originally planned, section 2.1 records the divergence and its reason.

## 2. Actual tree

This is the layout as built. It is flatter than the tree originally planned here: several contracts that were sketched as separate modules were implemented as one cohesive module. Section 2.1 records those consolidations so a reader can still map every component contract in `docs/components/` onto real code.

```text
iadopt-lab/
├── README.md                            # status, workflow, commands
├── DECISIONS.md                         # D-001 … D-050
├── TECHNICAL_SPECIFICATION.md
├── THIRD_PARTY_NOTICES.md
├── parameters.yml                       # the only human-edited configuration
├── main.py                              # thin entry; delegates to cli.main
├── pyproject.toml · uv.lock · .python-version · .env.example · .gitignore
├── compose.yml                          # isolated local PostgreSQL 16
├── docs/                                # specification, contracts, audits, migration record
├── schemas/                             # 6 JSON Schemas
├── prompts/                             # 3 frozen templates, v1
├── data/
│   ├── corpus/<tag>/                    # immutable TTL bytes, v2.0.0 and v2.0.1 retained
│   ├── canonical/<tag>/                 # per variable: record, .meta.json, .readable.json
│   └── manifests/                       # corpus, source lock, demonstrations, population,
│                                        #   prompts, and one frozen price card per provider
├── migrations/                          # 0001_initial.sql, 0002_coordination_integrity.sql
├── src/
│   ├── iadopt_lab/
│   │   ├── cli.py                       # command parsing and delegation only
│   │   ├── configuration.py             # YAML, schema, resolution, live gate, secrets
│   │   ├── canonical.py                 # canonical JSON and hashing
│   │   ├── domain.py                    # shared record types and errors
│   │   ├── artifacts.py                 # input bundle and drift verification
│   │   ├── planning.py                  # deterministic grid expansion
│   │   ├── costing.py                   # pre-run cost estimate
│   │   ├── evidence.py                  # derives the estimate's inputs from plan + price card
│   │   ├── probing.py                   # measures model capabilities; rewrites parameters.yml
│   │   ├── workflow.py                  # attempt advancement, resume, bounded execution
│   │   ├── reporting.py                 # ranking, summaries, exports
│   │   ├── local_database.py            # local PostgreSQL preparation
│   │   ├── corpus/ingestion.py          # source, parse, project, manifests, derived files
│   │   ├── prompting/                   # renderer.py, provenance.py
│   │   ├── providers/                   # base.py, openrouter.py, psnc.py
│   │   ├── generation/extractor.py      # deterministic JSON extraction
│   │   ├── validation/lexical.py        # schema, semantics, canonicalization, readable view
│   │   └── persistence/repository.py    # transactions, evidence, leases, migrations
│   └── iadopt_eval/                     # core.py (pure scorer), embeddings.py
├── tests/
│   ├── unit/                            # 11 modules
│   ├── scorer_regression/               # January parity
│   ├── integration/                     # PostgreSQL; skipped without a test DSN
│   └── recovery/                        # lease and resume; skipped without a test DSN
├── reference/january/                   # retained hash-verified January scorer copy
├── ops/                                 # unattended supervision; outside every hashed artifact path
└── experiments/                         # exploratory side analyses and their own output/
```

### 2.0 Modules added after the first live runs

Two modules exist that the original plan did not anticipate. Both were added because live
execution exposed a gap the offline design could not have seen.

`probing.py` measures what a deployment actually supports. `parameters.yml` gates live
execution on per-model capability fields and forbids inferring them from a model name, but
nothing filled them in, so they were being typed by hand. It probes each model and writes
the result back, so those fields carry an observation instead of an assumption. It is one
of only two commands that contact a provider.

`evidence.py` derives the cost-estimate document `costing.estimate_campaign_cost` consumes.
That document was previously hand-built, which does not scale past one model and let the
estimate be priced from different evidence than the runner reserves against. It now reads
the same frozen price card and takes the output ceiling from the plan.

### 2.1 Consolidations against the original plan

| Originally planned | As built | Why |
|---|---|---|
| `corpus/{source,parser,projector,manifest}.py` | `corpus/ingestion.py` | One verify-parse-project-materialize pipeline with a single atomic activation boundary |
| `validation/{schema,semantics,canonicalizer}.py` | `validation/lexical.py` | The three stages share the schema bytes and run as one ordered pass |
| `persistence/{database,registry,execution,evaluation,ranking}.py` | `persistence/repository.py` | One transaction boundary; splitting it would spread transaction control across modules |
| `workflow/{planner,runner,state,resume}.py` | `planning.py` + `workflow.py` | Planning is pure and testable offline; the rest is one runner owning leases and state |
| `reporting/{queries,statistics,ranking,exporters}.py` | `reporting.py` | Ranking and export read the same immutable rows |
| `iadopt_eval/{models,normalization,similarity,confusion,aggregation}.py` | `iadopt_eval/core.py` | The scorer is one frozen protocol; splitting it invites divergent partial versions |
| `generation/attempts.py` | folded into `workflow.py` | Only the workflow layer may schedule another numbered attempt |

Two consequences to keep in view. `persistence/repository.py` is by far the largest module and is the natural first candidate if it grows further. And every module still maps to one or more contracts in `docs/components/`, but the mapping is now many-contracts-to-one-module rather than one-to-one, so a contract change must name its module explicitly.

Directories from the original plan that do not exist: `tests/property/`, `tests/provider_contract/`, `tests/end_to_end/` (property-based cases live inside `tests/unit/test_eval_core.py`), `data/fixtures/`, `docs/decisions/` detailed ADRs, and `Dockerfile`.

## 3. Why there is no general `config/` directory

The experiment intentionally has one obvious human-edited configuration file at the project root: `parameters.yml`. This answers “which file do I edit before a campaign?” without requiring a user to combine several partially overlapping files.

The adjacent directories have different responsibilities:

- `schemas/` validates configuration and model-output shapes; it is not user configuration.
- `data/manifests/` contains frozen generated identities such as corpus files, demonstrations, the complete evaluation population, and embedding artifacts; these are reviewed evidence, not casual settings.
- `prompts/` contains immutable versioned scientific inputs.
- PostgreSQL stores the original and resolved configuration for each campaign.

If future deployment-only settings become numerous, a narrowly named directory such as `deployment/` may be introduced. Scientific parameters must remain visible in `parameters.yml` and may not be scattered among environment variables or code constants.

## 4. `main.py` and parallel execution

The root `main.py` is deliberately small. It imports the CLI entry function, accepts commands, and returns an exit code. It does not implement loops, provider requests, retry decisions, SQL, scoring, or thread/process management.

The `run` and `resume` commands invoke `workflow.runner`, which owns bounded parallel execution of the union of selected PSNC/OpenRouter model grids. One invocation continues through scoring, ranking, and required reports, with fair provider scheduling, independently bounded rates/cooldowns, and atomic cost accounting. Monetary caps are optional; null means uncapped and zero is a real zero ceiling. A pre-run estimate must be disclosed and explicit live authorization recorded before live dispatch; the estimate is not a hard cap. Every task still uses one owning provider/model. An affected provider can pause while eligible other work continues; completion requires the entire frozen plan. Adapters use separate provider-scoped clients from the shared OpenAI Python SDK with automatic retries disabled. Only the workflow layer can schedule another numbered attempt.

The initial database deployment is local PostgreSQL 16 with DBeaver access; AWS is not required. Its proposed persistent-volume, loopback connection, least-privilege roles, and backup boundaries are documented in [Local database setup](local-database.md). Deployment remains separate from scientific configuration and is not created during documentation work.

## 5. Package boundaries

### `iadopt_lab`

Owns experiment configuration, corpus ingestion, prompting, providers, validation, persistence, orchestration, and reporting. Side effects are explicit at adapter/repository boundaries.

### `iadopt_eval`

Owns only pure evaluation under accepted protocol `january-derived-member-credit-v1`. It preserves January behavior outside the approved member-credit and system-label corrections and returns explicit member/role evidence with fractional contributions. It cannot import provider, workflow, PostgreSQL, clock, filesystem discovery, or random services. This lets the same stored predictions be re-evaluated under a new metric version without another LLM request.

### Migrations

Own only database structure and forward changes. Application repositories do not create tables opportunistically. Migration identity is checked during preflight.

### Tests

Mirror package boundaries and retain provider/scorer/recovery fixtures. A source module cannot be accepted without its corresponding contract and tests.

## 6. Data flow and ownership

```text
data/corpus tagged bytes
        │
        ▼
corpus parser/projector ──► PostgreSQL corpus + gold registry
        │                              │
        ▼                              ▼
prompt renderer ─────────► PostgreSQL frozen prompt/request evidence
                                       │
                                       ▼
workflow runner ─► one-call provider adapter ─► raw response evidence
                                       │
                                       ▼
                         extractor/schema/semantic validator
                                       │
                                       ▼
                              canonical prediction
                                       │
                                       ▼
                           pure iadopt_eval records
                                       │
                                       ▼
                         PostgreSQL evaluation facts
                                       │
                                       ▼
                         complete configuration ranking
                                       │
                                       ▼
                           outputs/ derived reports
```

PostgreSQL is authoritative after ingestion and planning. `outputs/` can be deleted and regenerated; raw provider evidence cannot.

## 7. Deferred code boundary

No package for entity linking, RDF conversion, SHACL validation, or the deterministic baseline is created in this implementation phase. JSON-LD generation is not planned. Their documentation under `docs/future/` does not create an inactive runtime dependency or imply that those stages ran.

## 8. Architecture acceptance rules

- No active import crosses from `iadopt-lab/` into historical repository code or the sibling service.
- Every source module maps to a pre-code component contract.
- Every function receives accurate typed input/action/output/raises/side-effect documentation.
- Provider adapters perform at most one network request per invocation.
- Only workflow modules schedule parallel tasks or a later attempt.
- Only persistence modules own PostgreSQL transactions.
- Only `iadopt_eval` defines scientific matching/confusion formulas.
- Ranking covers every complete configuration and never removes lower-ranked or failed evidence.
- Prompt/schema/corpus/scorer/configuration changes produce new immutable identities.
- No secret appears in configuration, database evidence, fixtures, logs, or derived exports.
