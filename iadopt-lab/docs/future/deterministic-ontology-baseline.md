# Future Deterministic Ontology-Guided Baseline

## Status: deferred

This baseline is not part of the active lexical experiment and no baseline code is created in the current build. Its absence must remain visible in reports and manuscript claims; entity linking is not a substitute for a decomposition baseline.

The future design will be grounded in the public [I-ADOPT ontology specification](https://i-adopt.github.io/ontology/), but ontology structure alone is not a complete natural-language extraction method.

## Future objective

The baseline will provide a deterministic, non-LLM comparison that receives the same natural-language variable definition and returns the same six-field lexical decomposition contract as a model.

It is intended to answer whether model-based decomposition outperforms a transparent rule-based method. It does not need to be strong enough to replace the LLM workflow.

## Important distinction

Ontology axioms can define structural requirements such as:

- A Variable has a Property and ObjectOfInterest.
- A SymmetricSystem contains parts.
- An AsymmetricSystem has ordered roles.
- A Constraint constrains another component.

Those axioms do not determine which phrase in an arbitrary natural-language definition is a Property, Matrix, ObjectOfInterest, or Constraint. The future baseline therefore requires two separately versioned layers:

1. **Ontology-derived structural rules**, traceable to public ontology terms and constraints
2. **Lexical extraction rules**, authored under a separately approved and fully disclosed protocol

The documentation must never describe lexical heuristics as if they were ontology entailments.

## Proposed future input

- Exact Corpus variable definition
- Frozen ontology release or commit
- Frozen ontology-derived structural-rule version
- Frozen lexical-rule bundle
- Optional deterministic lexical resources whose licenses and versions are recorded
- The same corrected lexical JSON Schema used by the active experiment

This experiment has no hidden development/test partition. Before a future baseline is compared, its protocol must state exactly whether Corpus gold influenced rule authorship and what scientific claims remain justified. The rule bundle must be frozen before its recorded comparison run; later changes create a new baseline version rather than overwriting the earlier result.

## Proposed future output

The baseline will return:

- One six-field lexical decomposition
- Schema- and semantic-validation status
- A deterministic decision trace identifying every fired rule
- Rule-bundle and ontology hashes
- Runtime and resource-usage metadata
- An explicit empty value for every unsupported field

The output is evaluated by the same scorer version and the same complete 97-variable evaluation population as the model predictions. It receives one deterministic repetition and makes no provider calls.

## Future design requirements

- No LLM, embedding-generation API, or hidden learned classifier
- Same definition input as the LLM experiment
- Same lexical output contract
- Deterministic result for identical inputs and rule versions
- Ordered, inspectable rule precedence
- No fallback that copies or looks up the gold decomposition
- No undisclosed tuning against benchmark scores; every rule change creates a new version and comparison record
- Explicit ambiguity and no-match behavior
- Complete per-field provenance in the decision trace
- Unit fixtures for every rule and conflict-resolution branch
- Category-independent core rules unless category-specific behavior is declared and justified

Entity-linking dictionaries may not be used to turn this into an entity-linking baseline. If lexicons are used for phrase recognition, their role, version, origin, and test-set leakage risk must be documented separately.

## Decisions required before future implementation

The future phase must approve:

1. Exact ontology version and artifact hash
2. Structural rules derived from each relevant ontology axiom
3. Lexical-rule authoring protocol
4. Allowed external lexical resources
5. Benchmark-gold exposure policy and the resulting limits on comparison claims
6. Rule conflict and ambiguity policy
7. Whether rules may use part-of-speech or deterministic parsing libraries
8. Baseline versioning and trace schema
9. Acceptance tests and minimum reporting requirements

Only after those decisions are documented may an executable `baseline` package be added.
