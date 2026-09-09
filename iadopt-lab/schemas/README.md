# Schema Directory

## Current status

The implementation includes the tested lexical decomposition, corpus manifest,
and evaluation-population Draft 2020-12 schemas. Other configuration/evidence/report
schemas belong to their corresponding implementation components; their presence
and executable tests, not this inventory alone, establish implementation status.

## Schemas

All six exist and are exercised by the suite.

| File | Responsibility |
|---|---|
| `parameters.schema.json` | Validate the complete human-edited `parameters.yml` contract |
| `lexical-decomposition.schema.json` | Validate model output and canonical six-field lexical decompositions |
| `corpus-manifest.schema.json` | Validate immutable corpus and file-level provenance manifests |
| `evaluation-population.schema.json` | Validate the exact ordered 97-variable population and five demonstration exclusions |
| `experiment-evidence.schema.json` | Validate portable typed campaign/run/task/attempt/prediction/evaluation/ranking records |
| `report-manifest.schema.json` | Validate exported-result lineage, hashes, dimensions, denominators, artifact inventory, and cost-estimate/disclosure evidence |

Additional internal record schemas may be added only when their inputs, outputs, versioning, and database relationship are documented first.

## Lexical decomposition schema

The model-visible top level contains exactly:

```text
hasStatisticalModifier
hasProperty
hasObjectOfInterest
hasMatrix
hasContextObject
hasConstraint
```

All six fields are required. Unsupported scalar/entity positions use an empty string; unsupported constraints use an empty array. `null`, omitted fields, and unknown fields are invalid.

Empty strings are representation-valid in every scalar/entity position, including Property and Object of Interest. This preserves the experiment rule that schema validation does not judge semantic quality. A correctly shaped all-empty model response is accepted and scored, while a system-created terminal-empty prediction is distinguished by provenance rather than by a different JSON shape.

Entity-bearing positions accept a simple lexical string or one structured system:

- Symmetric system with at least two parts
- Asymmetric source/target system
- Asymmetric numerator/denominator system

The two asymmetric role pairs are alternatives. A schema that requires source, target, numerator, and denominator together is incorrect.

System container labels remain in the canonical representation for stable identification, but deterministic canonicalization supplies them and the evaluator excludes them. Symmetric parts and ordered asymmetric roles remain scored.

Constraints contain exactly a non-empty `label` and `on`. The JSON Schema validates shape; a separate semantic validator confirms that `on` refers to an allowed extracted component.

## One schema for prompt and runtime

The exact schema inserted into a prompt must be the same schema used by runtime validation. The implementation must store and compare:

- Schema version
- Exact or canonical schema bytes
- SHA-256
- Prompt-embedded schema hash
- Runtime-validator schema hash

A mismatch blocks preflight. Provider-specific response schemas must not silently replace the shared lexical contract.

## Validation layers

Schema validation is only one layer:

1. JSON candidate extraction and syntax parsing
2. JSON Schema validation
3. Semantic cross-reference validation
4. Deterministic canonicalization

The pipeline stores layer-specific error codes and JSON Pointer paths. It does not repair invalid syntax or invent semantic values.

## Parameter schema

`parameters.schema.json` must reject unknown keys and incompatible combinations. It must enforce, among other requirements:

- Parameter-schema version `2.1`, with nonempty unique `campaign.providers` selecting `psnc`, `openrouter`, or both; the earlier scalar `campaign.provider` is invalid
- A nonempty enabled model list under every selected `providers.<id>.models` before live execution; three models per provider is the initial intention, not a fixed allowed count, and unresolved IDs block live preflight
- Provider-specific model ownership and separate identities for the same model-ID text on different providers; each resolved run/task has exactly one provider/model
- A deterministic union of the selected provider model grids using the shared population/protocol, with no cross-provider model pairing
- Reasoning-enabled and reasoning-disabled profiles for capable models
- `not_applicable` with no reasoning request field for incapable models
- At most three generation attempts per task
- Valid prompt, shot, temperature, repetition, concurrency, and optional cost-cap values
- D-029's current campaign sets both repetition counts to one and records the singleton run identity. The generic schema still supports positive repetition counts for explicitly configured future campaigns; changing them changes plan identity and estimate requirements, not the three-attempt retry limit
- Explicit per-provider rate/concurrency limits, model context limits, billing mode/basis, `require_pre_run_estimate: true`, and `estimate_policy_version: pre-run-estimate-v1` for live mode
- Optional provider/global caps: null means no cap, zero is a real zero ceiling, and a finite positive value is a hard ceiling; negative values are invalid. Every default cap amount is null, including PSNC. A zero cap is not equivalent to unavailable cost information and does not declare a provider free
- For metered providers, versioned price-card and token/cost-bound evidence without requiring caps; for explicit no-charge access, retained owner/account-context provenance and zero financial cost. Missing usage/billing evidence is not a declaration of zero cost
- A valid estimate receipt bound to the resolved campaign/plan and pricing/FX basis, evidence of its disclosure, and separately recorded explicit live authorization before live dispatch. The estimate is not a cap or authorization
- Exactly one 97-variable evaluation-population manifest and no train/development/test fields
- Ranking enabled with all-result retention and accepted D-023 policy `mean-repetition-micro-close-f1-v1`: descending mean of unrounded repetition micro Close F1, shared competition ranks on exact ties, and complete 97-variable coverage for every expected repetition
- No credential values in configuration

Static validation is followed by semantic preflight against model capabilities and frozen manifests.

The selected-provider list is set-like for canonical hashing. Its YAML order is preserved in original-file evidence, while resolved serialization sorts provider IDs. Selected membership and active profile edits create a new campaign after freeze; inactive provider catalog edits change only the original YAML hash. Schema validation must not invent model IDs, infer unavailable capabilities, or append later YAML changes to a stored campaign on resume.

Schema version `2.1` declares `client_library: openai` and full versioned SDK base URLs plus relative `/chat/completions` paths. Its documentation draft permits unknown capabilities/reasoning profiles on intended enabled models only while live mode is false; exact plan freeze rejects unresolved profiles. A null model ID is allowed only in a disabled draft slot. The current six IDs are known and selected, but capabilities/revisions remain unresolved. `execution.continue_until_complete`, `continue_unaffected_providers`, and `scheduler: provider_fair` define the continued-execution policy; provider billing objects carry mode, basis, and optional monetary cap. Estimate evidence is checked during live preflight, not invented by static YAML validation.

## Metadata and result-record validation

PostgreSQL migrations and constraints are authoritative for stored relational
state, but database rows are never assembled from unvalidated free-form mappings.
Versioned typed domain models validate every persistence input. The portable
`experiment-evidence.schema.json` mirrors those boundary records so a campaign
archive can be validated independently of the application.

It covers corpus/configuration identity, the complete selected-provider set and
each record's provider/model/reasoning settings,
prompt and sanitized request evidence, raw-response metadata and hashes,
extraction/validation events, retry lineage, predictions, item/component
contributions, unrounded Exact/Close Precision, Recall, and F1 at variable,
component, and repetition scopes with numerators/denominators/support, ranking
inputs/ranks, usage/cost and its billing basis, provider/global reservations and optional caps,
pre-run estimate/disclosure/authorization receipts and their hashes/provenance,
task state, provider-specific pauses, and campaign completion. Cross-row
rules—such as exactly 97 population members, one task per member/run, at most
three attempts, complete selected-provider union/ranking/report coverage,
composite campaign-provider/model ownership, atomic cost admission and
settlement, and immutable terminal evidence—are enforced by PostgreSQL and
verified again by integrity queries. JSON Schema alone is not treated as a
substitute for relational constraints.

The evidence schema also covers accepted D-022 member/role matching: original
and normalized member references, pair eligibility/similarity, one-to-one
assignment and canonical tie evidence, missing role-evidence flags, `g/p/m/U`
counts, and exact contribution/metric numerator-denominator receipts with
decimal derivatives. A system container label cannot be a scored member.
Relational and typed integrity checks validate assignment uniqueness, role
eligibility, count consistency, and exact fraction sums. Scorer protocol
`january-derived-member-credit-v1` is accepted; its future executable artifact
hash and embedding revision remain separate required evidence.

Descriptive statistics such as mean, median, variance, mode, standard deviation,
range, and IQR over per-variable scores, plus uncertainty analyses, are deferred.
They are not required fields for a complete initial evidence archive. The archive
must preserve all inputs required to calculate them later under an explicit,
versioned analysis definition.

## Supplied validation files

The supplied `Variable.schema.json`, `Variable.context.jsonld`, `iadopt.sh.ttl`, and `iadopt-llm.sh.ttl` are deferred RDF/JSON-LD/SHACL references. They are not substitutes for `lexical-decomposition.schema.json` and are not active runtime inputs.

## Historical schema finding

The schema embedded by the pre-submission runner has SHA-256 `c099f2cebe91e495c22506b8accd592c209059a69c60110b12e7a263a7ab7d9a`. Its asymmetric branch simultaneously required numerator/denominator fields and forbade them through `additionalProperties: false` because only source/target fields were declared.

The runner did not actually invoke JSON Schema validation, so this defect affected the prompt contract but did not mechanically reject or retry historical responses. I-ADOPT Lab intentionally does not preserve either defect: the new alternative shapes are valid by construction and the exact prompt schema is enforced at runtime. See `docs/repository-audit.md` for the code evidence.

## Versioning and tests

Every schema uses an explicit version and JSON Schema Draft 2020-12 where applicable. A schema change after a campaign is planned creates a new schema identity and campaign.

Tests must cover:

- Schema meta-validation
- Every valid entity/system alternative
- Every empty-value form
- Missing and extra fields
- Invalid asymmetric mixtures
- Invalid symmetric part counts
- Invalid constraint shapes
- Exact equality between prompt-embedded and runtime schema hashes
