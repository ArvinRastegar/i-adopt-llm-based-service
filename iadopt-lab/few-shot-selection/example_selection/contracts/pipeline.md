# Contract — `pipeline`

Part of the [example-selection optimization](../README.md) side experiment.

Planned implementation owner: `pipeline.py` and the `run.py` entry point, beside this contract.

## 1. Responsibility and explicit non-responsibilities

Orchestrates the four stages — attribution, construction, refinement, confirmation — enforces
the budget, protects the held-out set, and writes the report.

**Not responsible for:** how a candidate is scored (`harness`), how sets are built (`design`),
or how contributions are estimated (`attribution`).

## 2. Typed inputs and their sources

| Input | Type | Source |
|---|---|---|
| `n_subsets` | `int` | stage-A sample count, default 600 |
| `search_reps` | `int` | repetitions per constructed finalist on `E`, default 5 |
| `confirm_reps` | `int` | repetitions per finalist on `C`, default 15 |
| `--stage` | `str` | optional; run a single stage and stop |
| `--plan-only` | flag | print the budget and exit without any provider call |

## 3. Outputs, files, database records, and exit behavior

Writes, all under `output/`:

| File | Content |
|---|---|
| `example-selection-calls.jsonl` | every provider call (owned by `harness`) |
| `example-selection-observations.jsonl` | one line per evaluated `(candidate, repetition, eval-set)` |
| `example-selection-report.md` | the human-readable result |

Writes no database record. Exits 0 on success, non-zero only on a contract violation.

## 4. Ordered processing and scientific invariants

- **INV-1** `C` is not evaluated before the confirmation stage. The pipeline asserts that no
  observation against `C` exists until stage D begins, and refuses to start stage D twice with
  different finalists.
- **INV-2** Finalists are chosen using `E` observations only. `C` never informs selection.
- **INV-3** The headline claim is the `C` comparison. Any `E` score reported for a finalist is
  labelled as selection-biased and is never presented as the result.
- **INV-4** Stages run in order A → B → C → D; a later stage refuses to run while an earlier
  one is incomplete.
- **INV-5** Baselines are evaluated on `C` under exactly the same protocol and repetition count
  as the finalists, in the same session.
- **INV-6** The falsification check is mandatory and has two arms, because neither alone is
  sufficient. The bottom-25 by coefficient shares 10 of its 25 members with the top-25
  (design INV-9), so it is a *weakened* control: only 15 members differ, and a real effect will
  show up attenuated. The random-25 reference distribution therefore carries equal weight. If
  the top-25 does not beat **both** on `C`, the report states that the attribution model failed
  to learn, rather than presenting the winner as meaningful.
- **INV-7** The budget is enforced before dispatch: the pipeline computes planned calls and
  refuses to exceed the declared ceiling.

## 5. State and side effects

All durable state is the JSONL files, which make every stage resumable. Makes provider calls
through `harness` only.

## 6. Named failures and caller behavior

| Condition | Behavior |
|---|---|
| stage D requested while stage A incomplete | `RuntimeError` naming what is missing (INV-4) |
| an observation against `C` exists before stage D | `RuntimeError` (INV-1) |
| planned calls exceed the ceiling | `RuntimeError` before any dispatch (INV-7) |
| provider unreachable | stages already written stay valid; re-running resumes |

## 7. Idempotency, concurrency, and resume behavior

Re-running any stage performs only the work not already recorded. Re-running a complete
pipeline performs zero provider calls and rewrites the report from stored observations.

## 8. Exact `parameters.yml` keys consumed

None.

## 9. Secret handling and provenance

Delegates all secret handling to `harness`. The report records the corpus hash, the split
sizes, the stage budgets and the scorer version, so a reader can tell what produced it.

## 10. Planned public functions and their contracts

```python
def planned_calls(n_subsets: int, search_reps: int, confirm_reps: int,
                  n_candidates: int, n_eval: int, n_confirm: int) -> int:
    """Total provider calls a full run would issue, across all four stages."""

async def stage_attribution(context, split, n_subsets: int = 600) -> dict:
    """Evaluate random subsets on E and fit the contribution model."""

async def stage_construct(context, split, coefficients: Mapping[str, float],
                          reps: int = 5) -> dict:
    """Build finalists and baselines, evaluate each on E with `reps` repetitions."""

async def stage_refine(context, split, candidates, reps: int = 5) -> dict:
    """Replicated swap search around the best constructed candidate."""

async def stage_confirm(context, split, finalists, reps: int = 15) -> dict:
    """Evaluate every named set in `finalists` on the held-out C; the only headline evidence.

    Baselines are passed as entries of `finalists`; there is no separate `baselines`
    parameter, so every set is guaranteed the identical protocol (INV-5).
    """

def write_report(path: Path, results: Mapping[str, Any]) -> None:
    """Render the report: C comparison with CIs, falsification check, per-example table."""
```

## 11. Acceptance tests

1. `--plan-only` prints a budget and makes zero provider calls.
2. Stage D raises `RuntimeError` when stage A has no observations (INV-4).
3. The pipeline raises if any `C` observation exists before stage D (INV-1).
4. A planned call count above the ceiling raises before any dispatch (INV-7).
5. Re-running a completed stage issues zero provider calls (resume).
6. Finalist selection reads only `E` observations — verified by a test that corrupts all `C`
   observations and confirms the chosen finalists are unchanged (INV-2).
7. `write_report` labels every `E` figure as selection-biased (INV-3).
8. `write_report` includes both falsification arms and states the failure interpretation when
   the top-25 fails to beat either the bottom-25 or the random-25 distribution (INV-6).
10. `write_report` records that the bottom-25 control is attenuated by the 10-member overlap,
    so a reader does not over-read a small margin.
9. Baselines in the report carry the same repetition count as the finalists (INV-5).
