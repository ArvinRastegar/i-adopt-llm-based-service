# Component Contract Index

These documents define every planned executable boundary before implementation. A source module cannot be added until its contract exists here and is consistent with `TECHNICAL_SPECIFICATION.md`.

## Required contract format

Every component document specifies:

1. Responsibility and explicit non-responsibilities
2. Typed inputs and their sources
3. Outputs, files, database records, and exit behavior
4. Ordered processing and scientific invariants
5. State and side effects
6. Named failures and caller behavior
7. Idempotency, concurrency, and resume behavior
8. Exact `parameters.yml` keys consumed
9. Secret handling and provenance
10. Planned public functions and their contracts
11. Acceptance tests

Every implemented public or private function must then receive a truthful typed docstring with summary, Args, Returns/Yields, Raises, Side Effects, and determinism/idempotency where relevant.

## Planned components

| Contract | Planned implementation owner |
|---|---|
| `shared-foundations.md` | `iadopt_lab.domain`, `iadopt_lab.canonical` |
| `cli-and-configuration.md` | `main.py`, `iadopt_lab.cli`, `iadopt_lab.configuration` |
| `corpus-ingestion.md` | `iadopt_lab.corpus` |
| `prompting.md` | `iadopt_lab.prompting` |
| `providers.md` | `iadopt_lab.providers` |
| `generation-and-validation.md` | `iadopt_lab.generation`, `iadopt_lab.validation` |
| `persistence.md` | `iadopt_lab.persistence`, migrations |
| `workflow.md` | `iadopt_lab.workflow` |
| `evaluation.md` | standalone `iadopt_eval` package |
| `reporting.md` | `iadopt_lab.reporting` |

Function names are provisional until implementation review. Their inputs, outputs, invariants, and side-effect boundaries are normative.
