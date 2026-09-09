# January Scorer Parity Contract

## Status

This document defines the implemented evaluator's compatibility boundary.
`src/iadopt_eval/` contains the pure evaluator and a separate optional local-only
embedding adapter. The retained historical source under
`reference/january/randomShotsPhaseOne.py.txt` matches the SHA-256 below exactly.

The accepted evaluator protocol is `january-derived-member-credit-v1`. It preserves January behavior outside the documented system-label exclusion and fractional member-scoring correction. D-022 covers both simple/system directions and system/system comparisons. Executable offline regression tests now verify scalar/empty parity, original NumPy Constraint matching and numerical correction, and explicit before/after D-022 receipts. These tests use named synthetic cosine fixtures. Real embedding bytes, immutable revision, dependencies, and artifact-level parity must still be frozen and verified before scientific scoring; offline logic parity is not a claim of embedding parity.

## Authoritative historical evidence

The behavioral reference is the file at immutable tag `V1.1-Experiment`, commit `b9683d2242aa5ca5b987440ea4b6f70bc1253c7e`:

```text
benchmarking_example/randomShotsPhaseOne.py
```

Its pre-submission SHA-256 is:

```text
2d4a6271fd86a076127dfb3f85589391cff9744efdfdee8ec570b52c74921df0
```

The pre-submission Turtle-to-JSON helper is supporting evidence only:

```text
benchmarking_example/gt_json_maker.py
SHA-256: b8caf88bc39a1aa8d6b01bc7e0fe2ed6419de5b6450455f0d6dd5242d7ab3773
```

The current February 2026 helper has SHA-256 `4ae635676d79ec665b3a5de1f0d472f3c85b2426a0c6db62a10302a845be2322`; it cannot replace the tagged helper in parity fixtures. Neither helper will be imported by the new runtime. Copies retained under `reference/` are evidence, not dependencies. The complete source/result distinction is documented in `repository-audit.md`.

## Evaluator input

The evaluator receives two schema-valid six-field lexical decompositions:

1. The deterministic Corpus gold decomposition, from the release pinned in `parameters.yml`
2. A terminal canonical model prediction, including the explicit empty prediction used after three content-invalid attempts

The six evaluated keys are:

1. `hasStatisticalModifier`
2. `hasProperty`
3. `hasObjectOfInterest`
4. `hasMatrix`
5. `hasContextObject`
6. `hasConstraint`

Variable labels, definitions, comments, RDF identifiers, source paths, category metadata, URIs, and system container labels are not separate scored fields.

## Evaluator output

For both Exact and Close modes, evaluation returns:

- Per-component TP, FP, FN, and TN contributions
- Aggregate TP, FP, FN, and TN across the six components
- Precision, Recall, and F1
- Evaluator version and configuration hash
- Frozen similarity-model identity for Close evaluation

The pure evaluator does not write to PostgreSQL. Persistence is a separate component so a new evaluator version can create a new evaluation run without changing earlier results.

## Behavior preserved from January

Outside the approved system-label and D-022 member-scoring corrections below, preserve the paper-facing January behavior and demonstrate it with golden fixtures. In particular, retain scalar/scalar matching, empty-value handling, and Constraint scoring.

### Scalar matching

- Values are lowercased and trimmed before exact comparison.
- Equal normalized strings receive similarity `1.0`.
- Close comparison otherwise uses the frozen `all-MiniLM-L6-v2` sentence-embedding model.
- The Close threshold is `0.80`, inclusive.
- The exact model revision and downloaded artifact hash must be frozen before scientific scoring.

### Simple confusion behavior

The historical behavior is intentionally retained:

| Gold value | Prediction | Contribution |
|---|---|---|
| present | matching non-empty | TP |
| present | wrong non-empty | FP only |
| present | empty | FN |
| empty | non-empty | FP |
| empty | empty | TN |

The wrong-non-empty case does not add an FN. This is unusual, but changing it would be a metric change rather than a rewrite of the January evaluator.

### Historical system behavior superseded by D-022

The January function chooses its comparison branch from the **gold value only**:

- Structured gold plus a simple prediction enters the structured-system matcher, which returns similarity `0.0` for the wrong representation.
- Simple gold plus a structured prediction enters the generic scalar matcher. That matcher applies `str()` to the prediction mapping; Close mode can therefore embed a Python dictionary representation instead of returning an unconditional structural mismatch.

The second branch depends on Python mapping representation/key order and can expose the arbitrary system label. January also thresholds an entire system after averaging role similarities or calculating part overlap, so a similarity of `0.5` yields FP only. Both behaviors are superseded: members now earn fractional contributions directly, without a final whole-component threshold.

### Accepted D-022 scope and input dispatch

Apply member scoring to `hasObjectOfInterest`, `hasMatrix`, and `hasContextObject` whenever both values are non-empty and at least one is a system. Handle absent values first with the January FN/FP/TN rules. Ordinary scalar/scalar comparisons and Constraints retain their historical branches.

Expand a simple string into one member, a symmetric system into its `hasPart` occurrences, and an asymmetric system into two ordered role occurrences. Exclude container labels completely. Preserve original strings and source pointers; matching normalization is a traceable derivative, not a mutation of predictions.

| Non-empty comparison | Eligible member pairs | Member equivalence |
|---|---|---|
| Simple/simple | Historical scalar branch; not member scoring | Historical normalized Exact or Close |
| Symmetric/symmetric | Unordered one-to-one parts | Literal string equality in both modes, preserving January part identity |
| Asymmetric/asymmetric | First slot with first; second with second only | Historical normalized scalar Exact or Close |
| Simple with either system, in either direction | Singleton with at most one member | Historical normalized scalar Exact or Close |
| Symmetric/asymmetric, in either direction | Unordered one-to-one membership | Historical normalized scalar Exact or Close; absent shared role evidence recorded |

Asymmetric slot 1 is `hasNumerator` or the historical `hasSource` fallback; slot 2 is `hasDenominator` or `hasTarget`. Preserve this January compatibility even when the two sides use different role-pair names, and record both actual names. Never match slot 1 to slot 2 when both sides specify ordered roles. When a side has no ordered roles, award membership credit without asserting role correctness. A mixed symmetric/asymmetric comparison may therefore achieve full membership credit while providing no shared role evidence; retain that limitation in match metadata.

Symmetric/symmetric deliberately remains case/whitespace-sensitive and does not use embeddings in Close mode. Changing this established label-equivalence rule was not requested. Other member branches lowercase/strip for Exact equality and use normalized equality or frozen cosine similarity at least `0.80` in Close. Fractional accounting is common across branches even though these historical equivalence rules differ.

### One-to-one assignment and determinism

Build the eligible-pair graph after applying role restrictions and the branch-specific equivalence test. Every gold and predicted member can appear in at most one selected pair. Keep distinct occurrences even when labels become equal after normalization; one simple entity cannot fill both asymmetric roles.

For unordered comparisons, select a matching with maximum cardinality first, then maximum sum of stored pair similarities. Resolve any remaining tie by the lexicographically smallest sorted pair-index list. Define those indices from canonical member order: symmetric parts use exact UTF-8 label order, asymmetric members use the fixed first/second slots, and a simple value has index zero. Source positions remain separate provenance. This makes symmetric array permutations irrelevant to scores and canonical pairing evidence. Stored finite similarities determine the secondary objective; do not apply an undocumented tolerance when deciding ties.

Do not use greedy highest-similarity matching here: for eligible similarities `[[0.95, 0.85], [0.84, 0.10]]` at `0.80`, selecting `0.95` first loses a valid second match, while selecting `0.85` and `0.84` matches both members. The historical greedy Constraint matcher is unaffected by this new system-member assignment.

### Fractional TP/FP/FN and exact receipts

Let `g` and `p` be gold and predicted member counts, and `m` the selected matching cardinality. Both counts are positive in this branch. Define:

```text
U  = g + p - m
TP = m / U
FP = (p - m) / U
FN = (g - m) / U
TN = 0
```

The contribution total is exactly one, retaining one component unit regardless of system size. Incorrect predicted members add FP and unmatched gold members add FN; a non-empty no-match system comparison therefore contributes both. The historical FP-only rule remains for wrong scalar/scalar predictions. Do not threshold `m/U` again. In general it is not `m/max(g,p)`; those coincide for a non-empty singleton/system match, but not for two systems that contain missing and extra members.

Store `g`, `p`, `m`, `U`, the selected and rejected pair evidence, role-evidence availability, and each integer contribution numerator/denominator. These fractions are authoritative; finite decimal approximations of thirds must not drive exact ties. Preserve the finalized numerical behavior of unchanged historical branches and its source values. Aggregate exact contribution receipts for micro metrics, storing reduced rational metric and ranking values alongside display decimals. An existing finite historical branch result can be represented by its exact numerical ratio without changing its branch decision. Decimal serialization precision is versioned separately from ranking arithmetic.

Hand-calculated examples (`+` below denotes a symmetric system, not an input JSON literal):

| Gold | Prediction | TP | FP | FN | Precision | Recall | Component F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| water + air | water | 1/2 | 0 | 1/2 | 1 | 1/2 | 2/3 |
| water | water + air | 1/2 | 1/2 | 0 | 1/2 | 1 | 2/3 |
| water + air | water + soil | 1/3 | 1/3 | 1/3 | 1/2 | 1/2 | 1/2 |
| water + air | air + water | 1 | 0 | 0 | 1 | 1 | 1 |
| water + air + soil | water | 1/3 | 0 | 2/3 | 1 | 1/3 | 1/2 |
| water + air | rock | 0 | 1/3 | 2/3 | 0 | 0 | 0 |
| water + air | soil + rock | 0 | 1/2 | 1/2 | 0 | 0 | 0 |

Half of a component's TP contribution is not F1 `0.5`. The first example has component F1 `2/3`; a variable's F1 additionally includes the other five components, and a repetition's micro F1 includes all 97 variables. Stored records name these scopes explicitly. Zero metric denominators retain the January result `0`.

### Constraints

- Constraint `label` and `on` values are trimmed, lowercased, and have internal whitespace collapsed.
- A component-key prefix on `on` is removed only when it has the form `<key>: <value>` and `<key>` is an exact case-sensitive member of the historical six-key list.
- Constraint pairs are matched greedily using their combined label/target similarity.
- The global maximum is selected repeatedly and its row/column are masked. Equal maxima use the first row-major position, matching historical `numpy.argmax`; tied matrices can therefore be input-order-sensitive even though ordinary unique-optimum cases are order-independent.
- Each gold constraint contributes two equal fractional units: one for `label`, one for `on`.
- Unmatched gold contributions become FN; unmatched prediction contributions become FP.
- For `n` gold Constraints, each field unit is `1/(2n)`. A matched field at or above the mode threshold adds that unit to TP and a field below it adds the unit to FP.
- The historical numerical correction is retained: a deficit above `1e-6` is added to FP; an excess above `1e-6` divides TP, FP, and FN by their total.

### Aggregation

TP, FP, FN, and TN are summed across the six fields before calculating micro Precision, Recall, and F1. Campaign micro metrics are calculated from summed contributions, not by averaging per-variable F1 values.

## Approved correction: system container labels

Both symmetric and asymmetric container labels are deterministically created for stable storage, but neither contributes to similarity or confusion values.

For asymmetric systems, this agrees with the January intent already documented in the historical scorer.

For symmetric systems, January combined container-label similarity with a part-set comparison. The active evaluator removes the container term and applies D-022's member contributions. With literal symmetric part identity, TP equals intersection-over-union, but it is retained as fractional credit rather than thresholded at `1.0` or `0.80`. Parts remain literal and unordered in this branch. The schema rejects exact duplicate symmetric parts before scoring.

The deterministic container-label policies are still applied outside the scorer:

- Symmetric: sorted part labels joined with a fixed separator
- Source/target asymmetric: role-preserving source-to-target label
- Numerator/denominator asymmetric: role-preserving numerator-over-denominator label

Changing a container label while preserving the same parts or ordered roles must never change an active score.

## Required parity fixtures

Before the evaluator may be used, fixtures must cover at least:

- Exact scalar match
- Close scalar match above and below `0.80`
- Present gold with wrong non-empty prediction producing FP only
- Present gold with empty prediction producing FN
- Empty gold and empty prediction producing TN
- Before/after historical structural-mismatch examples, both simple/system directions, and system/system partial matching
- One matched member among two/three members, no matched member, empty scalar, scalar normalization, and Close below/at/above `0.80`
- At-most-one-member assignment, normalization-equivalent occurrences, maximum-cardinality versus greedy examples, deterministic tied candidates, and asymmetric role-evidence metadata
- Exact component TP/FP/FN, Precision, Recall, and F1 for every partial-credit example; never equate partial TP with F1
- The common fractional system policy and preserved scalar/empty/Constraint boundaries
- Source/target asymmetric comparison
- Numerator/denominator asymmetric comparison
- Changed asymmetric container label with unchanged roles
- Changed symmetric container label with unchanged parts
- Symmetric literal exact match, partial/no overlap, case/whitespace differences unchanged between Exact/Close, and mixed symmetric/asymmetric membership
- Constraint reordering
- Multiple constraints with greedy fractional matching
- Equal-score Constraint ties in different list orders, preserving the historical row-major choice
- Empty prediction after the third invalid attempt
- Micro aggregation across all six keys

Fixtures on unaffected branches must match the historical script. System-label and accepted D-022 fixtures explicitly record every expected difference; exact fraction sums and ranking ties are checked separately from retained historical numerical calculations.

## Parity acceptance gate

The evaluator is ready only when:

1. The historical source hash is recorded.
2. The active evaluator source and configuration are versioned and hashed.
3. The embedding model revision and local artifact hash are frozen.
4. Every unaffected golden fixture matches January behavior.
5. Every system-label and D-022 difference matches this accepted protocol, including its complete representation dispatch and exact fractional receipts.
6. Repeated evaluation of the same inputs produces identical stored contributions.
7. A new metric version can be run without overwriting an older evaluation run.
