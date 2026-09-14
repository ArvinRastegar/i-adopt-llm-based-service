# Contract — `harness`

Part of the [example-selection optimization](../README.md) side experiment. Not an official
component; `docs/components/` is reserved for `src/` modules.

Planned implementation owner: `harness.py`, beside this contract.

## 1. Responsibility and explicit non-responsibilities

Evaluates one candidate example-set against one evaluation set: renders the 25-shot prompt per
target, calls PSNC, extracts, validates, scores, and returns micro Close F1 under both scoring
rules. Owns the on-disk cache that makes the whole experiment resumable.

**Not responsible for:** choosing candidates, fitting models, deciding how many repetitions to
run, or any reporting. It answers "what does this set score", nothing else.

## 2. Typed inputs and their sources

| Input | Type | Source |
|---|---|---|
| `candidate` | `Sequence[str]` | `design` module; exactly 25 variable ids |
| `targets` | `Sequence[str]` | `design.corpus_split`, the `E` or `C` list |
| `repetition` | `int` | caller; 1-based |
| `records` | `Mapping[str, dict]` | canonical corpus rows by `variable_id` |
| `template`, `schema_text`, `schema` | prompt/schema artifacts | `iadopt_lab.prompting`, `iadopt_lab.validation` |
| `similarity` | scorer backend | `iadopt_lab.cli._similarity_backend` |
| `transport` | `Callable[..., Awaitable[dict]]` | the context; defaults to the real PSNC caller |
| `client` | `httpx.AsyncClient` | the context; one shared client for the whole run |
| PSNC key and base URL | `str` | `../../.env` via `load_runtime_secrets` |

Reused as libraries from `src/`, never modified: `render` and `shot_pool` semantics come from
`shot_count_ablation`, so prompt bytes stay identical to the earlier side experiments.

## 3. Outputs, files, database records, and exit behavior

Returns a result mapping: `close_f1_official`, `close_f1_as_run`, `close_precision`,
`close_recall`, `scored`, `invalid`, `failed_calls`, `denominator_official`, `mean_latency`,
`prompt_tokens`. `denominator_official` is always `len(targets)` and is returned explicitly so
a caller can assert the denominator never moved (INV-3) rather than infer it.

Appends one JSON line per *provider call* to `output/example-selection-calls.jsonl`, carrying
the candidate hash, repetition, target id, raw answer, usage and the scored evaluation. Writes
nowhere else. Never exits the process.

Writes no campaign, run, task, evaluation or any other database row, and never opens a
database connection.

## 4. Ordered processing and scientific invariants

- **INV-1** The prompt carries the 25 candidate examples in the order given, then the target
  definition, rendered by the same substitution pass the official renderer uses.
- **INV-2** No target variable ever appears among the examples. The harness asserts
  `set(candidate) & set(targets) == set()` and raises if violated. This is the guard against
  the leak the whole design exists to prevent.
- **INV-3** `close_f1_official` scores a prediction that fails validation as
  `empty_prediction()` and keeps the variable in the population, so its denominator is exactly
  `len(targets)` for every candidate.
- **INV-4** `close_f1_as_run` drops invalid predictions, matching the earlier side
  experiments. It is recorded for comparability and is never the optimization target.
- **INV-5** Aggregation uses `iadopt_eval.aggregate_items`, so the number means what the
  official rankings mean.
- **INV-6** A call that fails at the transport level is recorded with `ok: false` and counted
  in `failed_calls`; it is never retried within a repetition and never silently dropped.
- **INV-7** Reasoning stays disabled: every request carries
  `chat_template_kwargs.enable_thinking = false`.

**Amendment 2 (Phase 4).** `HarnessContext` holds **one** `httpx.AsyncClient` reused across
every call. The first implementation opened a client per request, which meant a fresh TLS
handshake and connection pool for each of ~28,000 calls; measured throughput was 1.7 calls/s
against the 3.7 calls/s a single shared client achieves, turning a 90-minute run into a
4.6-hour one. The sibling `shot_count_ablation.py` already had this right.

**Amendment 1 (Phase 3).** `HarnessContext` carries an injectable `transport` callable.
Without it the module cannot be tested at all without a live provider, which would mean the
acceptance tests below could only run by spending real calls — and the invalid-handling and
transport-failure cases could not be provoked deliberately at all. The default is the real PSNC
caller, so production behaviour is unchanged.

## 5. State and side effects

Holds an in-process cache keyed by `(candidate_hash, repetition, variable_id)`, seeded on first
use from the JSONL. Appends to that JSONL. Makes outbound HTTPS requests to PSNC.

## 6. Named failures and caller behavior

| Condition | Behavior |
|---|---|
| `len(candidate) != 25` | `ValueError` |
| candidate and targets intersect | `ValueError` naming the offending ids (INV-2) |
| PSNC key absent | `LabError` from `load_runtime_secrets`; the runner aborts before any call |
| individual call fails | recorded, counted, not raised |
| every call in a batch fails | returns a result with `scored == 0`; the caller decides |

## 7. Idempotency, concurrency, and resume behavior

Evaluating the same `(candidate, repetition, targets)` twice performs zero provider calls the
second time: every scored call is read back from the cache. An interrupted run resumes at call
granularity, not candidate granularity, so no completed call is ever paid for twice.

Concurrency is bounded by a semaphore at 24, the validated PSNC ceiling. The JSONL is appended
under a lock so concurrent writes cannot interleave a partial line.

## 8. Exact `parameters.yml` keys consumed

None. Model, prompt variant, temperature, top_p, max output tokens and reasoning fields are
fixed constants in the runner, so `parameters.yml` cannot drift this experiment.

## 9. Secret handling and provenance

Reads `PSNC_API_KEY` and `PSNC_API_BASE_URL` from `../../.env` through `load_runtime_secrets`.
The key is held only in the request header and never written to the JSONL, a log or a result.

## 10. Planned public functions and their contracts

```python
def candidate_hash(candidate: Sequence[str]) -> str:
    """Stable 16-hex identity of a candidate set, order-independent (sorted, SHA-256)."""

async def evaluate(candidate: Sequence[str], targets: Sequence[str], repetition: int,
                   context: "HarnessContext") -> dict:
    """Score one candidate on one evaluation set, reusing cached calls."""

def load_cache(path: Path) -> dict:
    """Rebuild the call cache from the JSONL; returns {} when the file is absent."""

def build_context(root: Path, concurrency: int = 24) -> "HarnessContext":
    """Assemble corpus, prompt template, schema, scorer and PSNC credentials once."""
```

## 11. Acceptance tests

1. `candidate_hash` is order-independent and stable across processes.
2. `candidate_hash` differs for sets differing by one member.
3. `evaluate` raises `ValueError` when the candidate and targets intersect (INV-2).
4. `evaluate` raises `ValueError` when the candidate is not exactly 25 members.
5. With a stubbed provider returning a fixed valid prediction, `close_f1_official` equals the
   value `aggregate_items` produces for the same records (INV-5).
6. With a stubbed provider returning unparseable text for one target, the official denominator
   is still `len(targets)` and `as_run` excludes that target (INV-3, INV-4).
7. A second `evaluate` on the same arguments issues zero provider calls (INV-7 resume).
8. A transport failure is recorded with `ok: false`, counted in `failed_calls`, and does not
   raise (INV-6).
9. Every request body contains `enable_thinking: false` (INV-7).
10. The JSONL contains exactly one line per provider call and each line parses as JSON.
11. The API key appears in no written line.
