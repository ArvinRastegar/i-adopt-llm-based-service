# Evaluator Contract

## Responsibility

The standalone `iadopt_eval` package transforms immutable gold/prediction cases into component match records, fractional confusion contributions, and named aggregate metrics.

It is pure: no provider, network, PostgreSQL, clock, filesystem discovery, or task-state imports. Database integration serializes its inputs and outputs outside the package.

## Inputs

Each evaluation case contains:

- Schema version and variable ID
- Input-definition hash
- Canonical six-field gold decomposition and hash
- Canonical six-field prediction and hash
- Terminal-failure status when applicable
- Category and representation structure as analysis metadata
- Scorer bundle and embedding backend identity

## Outputs

- Normalization traces
- Component/role-level exact and close match records
- Similarity scores and thresholds
- Pairing/assignment decisions
- Member counts, normalization denominator, match mode, and availability of asymmetric-role evidence
- Exact rational TP/FP/FN/TN receipts and decimal presentation derivatives
- Per-item and aggregate metric records with numerators/denominators/support
- Timestamp-free canonical result hashes

## Accepted scorer behavior

The accepted protocol is `january-derived-member-credit-v1`. It preserves January paper behavior outside the explicitly documented system-member correction in D-022. Its implementation code hash and frozen embedding revision will be established and regression-tested during implementation; the protocol identifier is already decided.

The following rules remain unchanged:

- Lowercase/strip exact string normalization
- Wrong non-empty simple prediction contributes FP only
- `all-MiniLM-L6-v2`, cosine threshold `0.80` for close matching
- Greedy fractional Constraint matching
- Six-slot micro contribution aggregation and harmonic F1

`AsymmetricSystem` and `SymmetricSystem` identifier strings are never matcher inputs. Their ordered roles or unordered parts remain inputs.

### System-member matching and partial contributions

For a non-empty entity-bearing component where either side is a system, expand a simple value into one member, a symmetric system into its parts, and an asymmetric system into its two role values. Count members before matching: `g` gold members and `p` predicted members. Preserve separate members even if normalization makes their labels equivalent; validation and matching have separate responsibilities.

Symmetric/symmetric member equivalence preserves January literal equality in both Exact and Close, including case/whitespace sensitivity and no embeddings. Other system-member comparisons use the January scalar matcher: Exact equality after lowercase/strip normalization, or Close equality/similarity at the inclusive cosine threshold `0.80`. D-022 changes fractional accounting, not the established symmetric label-identity rule. Neither system labels nor serialized dictionaries enter similarity calculations.

If both sides are asymmetric systems, only corresponding first and second role slots may match. The January fallback treats numerator/source as the first slot and denominator/target as the second. Cross-slot matches are prohibited. If either side lacks ordered roles, compare membership without a role claim and record asymmetric-role evidence as unavailable when an asymmetric system is involved.

For unordered membership, choose a one-to-one assignment that first maximizes the number of qualifying pairs, then their summed similarity, then the canonical lexical pair ordering. No member may be reused. Preserve original member references and canonical tie evidence so symmetric input order cannot change scores. Constraint assignment remains its separate historical greedy algorithm.

Let `m` be the matched-pair count and `U = g + p - m`. The component contributes:

```text
TP = m / U
FP = (p - m) / U
FN = (g - m) / U
TN = 0
```

The non-empty component has total contribution mass one. Store integer counts and each exact numerator/denominator with the decimal derivative; no second whole-component threshold may discard partial credit. For gold `water + air` and prediction `water`, the contributions are TP `1/2`, FN `1/2`, FP `0`, and component F1 `2/3`. For gold `water + air` and prediction `water + soil`, TP, FP, and FN are each `1/3`, and component F1 is `1/2`.

This branch covers both simple/system directions and system/system comparisons, including mixed symmetric/asymmetric representations. Empty-component cases, simple/simple scoring, and Constraint scoring retain January behavior. The branch decision and whether an approved correction applies are included in the result evidence; this scorer must not be described as unchanged January scoring. `docs/scorer-parity.md` defines the complete before/after compatibility boundary.

## Configuration keys consumed

- `evaluation.*`

The pure evaluator receives a resolved scorer bundle rather than reading YAML itself.

## Design responsibilities

These planning signatures describe the decomposed responsibilities. The concrete
first-implementation public interface is documented below; internal helpers
fulfil normalization, component matching, and contribution responsibilities.

### `normalize_case(case, specification) -> NormalizedCase`

- **Input:** One immutable gold/prediction case and exact scorer specification, including mode, six component names, January normalization rules, system policy, threshold, and embedding identity.
- **Action:** Validate input ownership/hashes, prepare the historical per-branch string/Constraint forms, preserve structured roles/parts and original evidence, and record every transformation. It does not apply a match outcome.
- **Output:** Typed normalized case with original-to-normalized trace, component structures, scorer identity, and timestamp-free content hash.
- **Raises:** Schema/version/hash mismatch, unsupported structure, missing scorer artifact, or unresolved/unrecognized scorer policy.
- **Side effects:** None; no model loading/network/database access in the normalization function.
- **Determinism:** Identical case and scorer artifacts produce identical normalization records.

### `compare_component(gold, prediction, context) -> tuple[MatchRecord, ...]`

- **Input:** One normalized gold component, corresponding prediction component, and immutable match context containing Exact/Close mode, threshold, similarity backend, and branch policy.
- **Action:** Select the frozen simple/system/Constraint branch, calculate permitted member-pair similarities, and exclude both system container identifiers. For non-empty entity comparisons involving a system, apply the accepted ordered-role or unordered maximum-cardinality matching rule above. Retain role-evidence availability separately from membership evidence; preserve the historical greedy assignment only for Constraints.
- **Output:** Ordered match records containing branch and correction scope, original/normalized member references, raw similarities, threshold/eligibility decisions, selected and unmatched members, matching mode, assignment/tie evidence, role-evidence availability, and scorer/version identity.
- **Raises:** Unsupported structural combination, unknown scorer policy, unavailable/mismatched embedding artifact, or malformed assignment evidence.
- **Side effects:** None. The supplied local similarity backend may cache immutable embeddings but may not perform network retrieval.
- **Determinism:** Identical operands/backend bytes/device-dtype policy yield values within the frozen numerical tolerance and the same decisions.

### `confusion_contributions(matches, policy) -> tuple[Contribution, ...]`

- **Input:** Complete ordered match records for one component plus the `january-derived-member-credit-v1` confusion/weighting/numerical-correction policy.
- **Action:** Retain January empty/simple/Constraint behavior. For the system-member branch, validate `g`, `p`, and one-to-one matched count `m`, derive `U = g + p - m`, and return TP `m/U`, FP `(p-m)/U`, FN `(g-m)/U`, TN `0`. Do not average asymmetric-role similarities or apply a second whole-component threshold to this branch.
- **Output:** Ordered contribution records with exact numerator/denominator receipts, decimal derivatives, raw member counts, explicit totals/support, correction-scope identity, and the applicable mass invariant/tolerance.
- **Raises:** Missing/duplicate match fact, negative/non-finite value, unsupported branch, assignment inconsistency, or mass/tolerance violation.
- **Side effects:** None.
- **Determinism:** Exact rational system-member arithmetic for identical records/policy; unchanged historical branches retain their specified numerical calculation and exact result receipts.

### `evaluate_case(case, bundle) -> EvaluationRecord`

- **Input:** One valid evaluation case and complete frozen scorer bundle (code/protocol, schema, embedding artifacts, thresholds, normalization, zero-denominator and correction identities).
- **Action:** Normalize, compare, and derive contributions for all six slots in fixed order for Exact and Close; calculate item totals/metrics; attach terminal-failure/category metadata without changing scientific values.
- **Output:** Immutable per-variable evaluation record containing all traces, match facts, similarities, contributions, TP/FP/FN/TN totals, named metrics/numerators/denominators/support, hashes, and scorer identity.
- **Raises:** Case/bundle incompatibility, artifact failure, invalid contribution invariant, or deterministic-hash conflict.
- **Side effects:** None; persistence and batching are external.
- **Determinism:** Same case/bundle/environment contract yields the same timestamp-free record/hash.

### `aggregate(records, specification) -> tuple[MetricValue, ...]`

- **Input:** A complete explicitly scoped collection of compatible item/component records plus aggregate scope (variable population, repetition, category/component filters) and scorer/metric specification.
- **Action:** Verify uniqueness/coverage and common identities, sum exact rational contribution receipts, calculate micro Precision/Recall/F1 under the frozen zero-denominator rule, and retain support and source-record lineage. It never averages per-variable F1 for the primary micro metric or uses display-rounded values.
- **Output:** Ordered named metric values with TP/FP/FN/TN, numerator, denominator, support, availability/reason, scope/population hash, unrounded value, and aggregate hash.
- **Raises:** Missing/duplicate item, incompatible scorer/population, incomplete required scope, arithmetic invariant failure, or unknown metric.
- **Side effects:** None.
- **Determinism:** Input order cannot change aggregate bytes or values; canonical record order is part of the contract.

## Implemented interface (first implementation)

The first executable interface exposes `iadopt_eval.evaluate_item(gold,
prediction, similarity, *, metadata=None, similarity_identity=None)` and
`iadopt_eval.aggregate_items(records, *, expected_variable_ids=None)`. The six
lexical keys must be present; representation validation is repeated defensively
without requiring the experiment package or a database. The supplied similarity
callable accepts two normalized strings and returns a finite cosine value. Its
explicit identity distinguishes an offline test fixture from the scientific
embedding backend.

Each item contains `exact` and `close` records, each with `components`, `totals`,
and `metrics`. Contribution and metric receipts contain integer `numerator`,
positive integer `denominator`, and a derived floating-point `value`. Each
component contains its branch and full matching/normalization evidence. Item
metadata and hashes preserve lineage but never affect scoring. Aggregate input
must share scorer and similarity identities; optional expected variable IDs
enforce complete population coverage, and duplicate identified variables fail.

The core uses the Python standard library only. Unordered member assignment uses
successive shortest augmenting paths with exact rational similarity objectives
and a canonical pair-order tie objective, rather than greedy matching. Constraint
arithmetic intentionally executes the January binary floating-point operations
before recording their exact numerical ratios. The optional embedding adapter
loads only an explicitly supplied local artifact directory after verifying its
manifest; it never discovers or downloads a model. Scientific embedding parity
remains a separate gate until a real revision and its files have been frozen.

`iadopt_eval.embeddings.load_local_similarity(artifact_directory, manifest)` is a
separate optional I/O adapter. Its callable result exposes a detached `.identity`
mapping. The manifest requires `model_id`, an immutable 40-hex `revision`, complete
relative-path-to-SHA-256 `files`, `device: cpu`, `dtype: float32`, and exact
`dependencies` versions for sentence-transformers, torch, transformers, numpy,
and tokenizers. Missing, extra, changed, or symlinked artifact files are rejected.
Construction hashes local bytes; first use rechecks bytes and dependency versions
before loading with `local_files_only=True` and `trust_remote_code=False`.
No automatic model downloads or cache discovery exist.

## Implementation verification

The evaluator tests execute isolated pure function definitions extracted from
the retained immutable January source copy only after verifying its recorded SHA-256.
The historical module itself is never imported. Actual NumPy greedy assignment
and binary floating-point Constraint correction match for all 64 combinations
of zero through seven gold/predicted Constraints in both modes. Scalar/empty
fixtures also match exactly; D-022 tests explicitly assert the intended
before/after differences. Cosines in these tests are named offline fixtures,
so this demonstrates scorer logic parity, not frozen MiniLM artifact parity.

The test suite additionally checks every documented partial-credit example,
role reversal and role-name fallback, normalization-equivalent occurrences,
maximum-cardinality matching against 100 generated exhaustive small-graph
oracles, literal symmetric equivalence, historical Constraint tie sensitivity,
exact micro aggregation and coverage, corrupt evidence, artifact hash failures,
and mocked local-only loading/caching. No live provider calls or actual embedding
downloads are involved.

## Determinism

Given identical canonical inputs, scorer bundle, embedding artifact, dependencies, device/dtype, and numerical tolerance, evaluator record hashes must match. Timestamps belong only to the database envelope.

## Acceptance tests

- Hand-calculated simple/empty/wrong cases
- Below/at/above close threshold
- Asymmetric role reversal
- Symmetric part reorder and container-label changes
- Constraint reorder/partial/unmatched cases, plus historical row-major behavior for tied greedy matches
- Both simple/system directions and system/system partial matching, including mixed system representations, no/full/partial match, empty boundaries, tied/duplicate-equivalent members, absent asymmetric-role evidence, and exact normalization receipts
- Gold `water + air` versus prediction `water`: TP `1/2`, FN `1/2`, component F1 `2/3`; versus `water + soil`: TP/FP/FN each `1/3`, component F1 `1/2`
- Maximum-cardinality matching counterexample where greedy highest-similarity selection loses a qualifying pair; summed-similarity and canonical-lexical tie rules
- Literal symmetric/symmetric identity in both modes; scalar-normalized asymmetric and mixed/simple comparisons; symmetric permutation invariance, ordered asymmetric slot reversal, and no member reuse
- Preserved empty/simple/Constraint behavior and explicit exclusion of system branches from January whole-component thresholds
- Terminal empty prediction
- Micro aggregate regeneration
- January golden parity outside the accepted D-022 boundary and explicit before/after receipts for every correction
- Pure result serialization contains no time, random ID, or machine path
