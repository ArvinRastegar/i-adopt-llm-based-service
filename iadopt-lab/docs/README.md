# Documentation Guide

Start with the root `README.md` for the experiment in plain language. The root `TECHNICAL_SPECIFICATION.md` is authoritative when a shorter document omits detail.

## Design and operation

| Document | Question it answers |
|---|---|
| `architecture.md` | Where will every type of code, data, configuration, test, and output live? |
| `model-catalog.md` | Which six model IDs are selected, which provider/client owns each, and how can future provider/model lists change? |
| `prompt-specification.md` | Exactly what information reaches the model, and how do correction attempts work? |
| `database.md` | What does PostgreSQL store and what integrity/transaction rules apply? |
| `local-database.md` | Can the database run locally, how will DBeaver connect, and how are existing `.env` credentials kept private? |
| `retry-and-resume.md` | How are three total calls enforced and how does interruption recovery work? |
| `scorer-parity.md` | Which January rules are preserved and how do the accepted fractional system-member corrections work? |
| `repository-audit.md` | What do the pre-submission code and latest archived result files prove, and what remains unresolved? |
| `documentation-audit.md` | Does every user requirement map to consistent documentation/configuration, and which decisions still block implementation? |
| `test-plan.md` | How will every requirement and failure mode be verified? |
| `reproducibility.md` | Which hashes/evidence/backups make a campaign reconstructable? |
| `runbook.md` | How will an operator prepare, dry-run, canary, run, resume, and report a campaign? |

## Component contracts

`components/` defines planned module boundaries and public-function inputs, actions, outputs, failures, side effects, determinism, and acceptance tests before implementation. An executable module cannot be added without a matching contract.

## Decisions

The root `DECISIONS.md` is the decision index: it records accepted choices, superseded restrictions, and live campaign values still to freeze. `decisions/` is reserved for detailed future architecture decision records; adding a later decision never rewrites the history of an executed campaign.

## Future work

`future/` documents inactive work so it cannot be mistaken for part of the current experiment:

- Deterministic ontology-derived baseline
- Entity linking, future RDF conversion, and SHACL validation (JSON-LD generation is not planned)

Those documents are planning boundaries, not implemented features or runtime dependencies.
