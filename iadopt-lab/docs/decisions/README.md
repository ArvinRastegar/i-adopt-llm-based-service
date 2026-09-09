# Decision Records

## Authority

The root `DECISIONS.md` is the current concise index of accepted, proposed, superseded, and rejected decisions. This directory is reserved for expanded architecture or scientific decision records when a choice requires more context than the index can provide.

No expanded record is required merely to duplicate an entry already explained completely in `DECISIONS.md` and `TECHNICAL_SPECIFICATION.md`.

## When to add a record

Add a new record before making a change that affects:

- Scientific protocol or evaluated denominator
- Corpus, demonstration-set, or evaluation-population identity
- Lexical output contract
- Prompt family or retry behavior
- Provider/model capability treatment
- Reasoning-mode mapping
- Evaluator or aggregation behavior
- Persistence, lineage, or resume guarantees
- Security or credential handling
- Activation of deferred baseline, linking, RDF, or SHACL work

Ordinary refactoring that preserves all documented behavior does not need a scientific decision record.

## Naming convention

Use:

```text
D-<three-digit-number>-<short-kebab-title>.md
```

Numbers must match the root decision index. Records are never renumbered.

## Record template

```markdown
# D-NNN — Decision title

## Status

Proposed | Accepted | Superseded | Rejected

## Date and approver

Record the UTC date and approving role/person when required.

## Context

Describe the problem, evidence, constraints, and why a decision is needed.

## Decision

State the exact normative choice.

## Alternatives considered

List credible alternatives and why they were not selected.

## Consequences

Describe data, code, database, reproducibility, cost, and reporting effects.

## Migration or campaign impact

State whether existing evidence remains valid and whether a new campaign,
schema, scorer, or migration is required.

## Verification

List documentation, tests, manifests, and acceptance gates that prove the
decision was implemented correctly.
```

## Immutability policy

Accepted decision history is append-only. If a settled choice changes:

1. Add a new decision record.
2. Mark the older decision superseded rather than rewriting history.
3. Update the root index and technical specification.
4. Create new configuration, schema, scorer, or campaign identities as required.
5. Preserve all earlier database evidence and derived-report provenance.

Deferred work is not activated by documentation alone. It requires an explicit accepted decision and its own implementation gate.
