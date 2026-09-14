# Contract — `design`

Part of the [example-selection optimization](../README.md) side
experiment. Not an official component; `docs/components/` is reserved for `src/` modules.

Planned implementation owner: `design.py`, beside this contract

## 1. Responsibility and explicit non-responsibilities

Constructs every *set of variables* the experiment uses, deterministically and without any
network or provider access: the three-way corpus split, the random subsets for attribution,
and the constructed candidate and baseline sets.

**Not responsible for:** evaluating anything, calling a provider, scoring, fitting models,
or reading `parameters.yml`.

## 2. Typed inputs and their sources

| Input | Type | Source |
|---|---|---|
| `records` | `list[dict]` | `iadopt_lab.corpus.ingestion.load_canonical_records`, all 102 rows |
| `seed` | `int` | caller; fixed constants in the runner |
| `coefficients` | `Mapping[str, float]` | the `attribution` module, for `top_k_candidate` |
| `embeddings` | `Mapping[str, Sequence[float]]` | local `all-MiniLM-L6-v2`, for the diversity baseline |

## 3. Outputs, files, database records, and exit behavior

Pure in-memory values. Writes no file, no database record, and never exits the process.

## 4. Ordered processing and scientific invariants

- **INV-1** The split is a partition: `P`, `E` and `C` are pairwise disjoint and their union is
  all 102 `variable_id`s.
- **INV-2** `len(P) == 40`. `E` and `C` take the remainder, sized by proportional allocation.
- **INV-3** Each split is domain-stratified: within every `category`, members are allocated to
  P/E/C by largest remainder on that domain's own count, so per-domain totals reconcile with
  the corpus exactly. Independent per-split allocation does not reconcile and is forbidden.
- **INV-4** The split depends only on `records`; it takes no seed and is identical on every
  call and every machine.
- **INV-5** Within a domain, members are ordered by sorted `variable_id` before allocation.
- **INV-6** Every sampled subset satisfies `len(subset) == 25` and `subset ⊆ P`.
- **INV-7** Sampling is reproducible from `seed` alone and yields no duplicate subsets.
- **INV-8** No function in this module reads or returns a `gold`, a score, or any measured
  quantity, except `top_k_candidate`, which consumes already-fitted coefficients.
- **INV-9** `top_k_candidate(..., invert=False)` and `top_k_candidate(..., invert=True)`
  necessarily **overlap** by `2 * size - len(pool)` members — 10 of 25 for a 40-member pool.
  Disjointness is arithmetically impossible whenever `2 * size > len(pool)`. The two selections
  differ in `len(pool) - size` members, 15 here, and that is the real contrast available.

**Amendment 1 (Phase 4).** INV-9 and acceptance test 9 replace an earlier claim that the
top-25 and bottom-25 selections are disjoint. That claim was false by arithmetic and was caught
by the test written against it. See DS-7 in the plan for the consequence.

## 5. State and side effects

Stateless and free of side effects. No I/O of any kind.

## 6. Named failures and caller behavior

| Condition | Raises |
|---|---|
| `records` is not exactly 102 rows | `ValueError("design requires the complete 102-row corpus")` |
| a domain cannot supply its allocation | `ValueError` naming the domain and the shortfall |
| `n_subsets` exceeds the number of distinct 25-subsets obtainable | `ValueError` |
| `coefficients` does not cover all of `P` | `ValueError` |

All are programming errors; callers do not catch them.

## 7. Idempotency, concurrency, and resume behavior

Every function is a pure deterministic function of its arguments, so it is idempotent, safe to
call concurrently, and needs no resume support.

## 8. Exact `parameters.yml` keys consumed

None. The experiment's configuration is fixed in its runner.

## 9. Secret handling and provenance

Handles no secrets. Provenance is the corpus records themselves.

## 10. Planned public functions and their contracts

```python
def corpus_split(records: list[dict]) -> dict[str, list[str]]:
    """Partition the corpus into pool/search-eval/confirmation, domain-stratified."""
    # Returns {"P": [...40 ids...], "E": [...], "C": [...]}, each sorted.

def sample_subsets(pool: list[str], n_subsets: int, size: int = 25,
                   seed: int = 0) -> list[tuple[str, ...]]:
    """Draw `n_subsets` distinct sorted 25-subsets of `pool`, reproducibly from `seed`."""

def top_k_candidate(pool: list[str], coefficients: Mapping[str, float],
                    size: int = 25, invert: bool = False) -> tuple[str, ...]:
    """The `size` pool members with the highest (or, if `invert`, lowest) coefficient."""
    # Ties broken by sorted variable_id so the result is deterministic.

def stratified_baseline(records: list[dict], pool: list[str],
                        size: int = 25) -> tuple[str, ...]:
    """A domain-proportional selection from `pool`, by largest remainder then sorted id."""

def diversity_baseline(pool: list[str], embeddings: Mapping[str, Sequence[float]],
                       size: int = 25) -> tuple[str, ...]:
    """Facility-location greedy selection maximising embedding coverage of `pool`."""
    # Deterministic: seeded with the medoid, ties broken by sorted variable_id.

def random_baselines(pool: list[str], n: int, size: int = 25,
                     seed: int = 0) -> list[tuple[str, ...]]:
    """`n` independent random selections, for the random-25 reference distribution."""

def perturb(candidate: Sequence[str], pool: list[str], n: int,
            swaps: int = 1, seed: int = 0) -> list[tuple[str, ...]]:
    """`n` neighbours of `candidate`, each differing by exactly `swaps` members."""
```

## 11. Acceptance tests

1. `corpus_split` returns three pairwise-disjoint lists whose union is all 102 ids (INV-1).
2. `corpus_split` returns exactly 40 ids in `P` (INV-2).
3. Per-domain counts summed across P/E/C equal the corpus per-domain counts exactly (INV-3).
4. `corpus_split` called twice on the same records returns identical output (INV-4).
5. `corpus_split` raises `ValueError` when given 101 records.
6. `sample_subsets` returns `n_subsets` subsets, each of size 25 and a subset of `P` (INV-6).
7. `sample_subsets` returns no duplicates, and the same seed reproduces the same list (INV-7).
8. Two different seeds produce different subset lists.
9. `top_k_candidate` returns the 25 highest-coefficient members; with `invert=True`, the 25
   lowest; the two overlap in exactly `2 * size - len(pool)` members and differ in
   `len(pool) - size` (INV-9).
10. `top_k_candidate` breaks coefficient ties by sorted `variable_id`.
11. `stratified_baseline` returns 25 ids whose domain proportions are within one of the pool's.
12. `diversity_baseline` returns 25 distinct pool members and is identical across calls.
13. `perturb` returns neighbours that each differ from the input in exactly `swaps` members
    and remain size-25 subsets of `P`.
14. Every function raises rather than silently truncating when the pool is too small.
