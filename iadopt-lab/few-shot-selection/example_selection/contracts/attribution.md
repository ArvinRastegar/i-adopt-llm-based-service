# Contract — `attribution`

Part of the [example-selection optimization](../README.md) side experiment.

Planned implementation owner: `attribution.py`, beside this contract.

## 1. Responsibility and explicit non-responsibilities

Estimates each pool example's marginal contribution to Close F1 by ridge-regressing observed
subset scores on binary inclusion indicators, and reports whether that estimate has stabilised.

**Not responsible for:** sampling subsets, evaluating them, or choosing the final candidate.

## 2. Typed inputs and their sources

| Input | Type | Source |
|---|---|---|
| `pool` | `list[str]` | `design.corpus_split()["P"]`, 40 ids |
| `observations` | `list[tuple[tuple[str, ...], float]]` | `(subset, close_f1_official)` from the harness |
| `alpha` | `float` | ridge penalty; default 1.0 |

## 3. Outputs, files, database records, and exit behavior

Returns `{"coefficients": {variable_id: float}, "intercept": float, "r2": float,
"split_half": float, "n": int}`. Writes no file and no database record.

## 4. Ordered processing and scientific invariants

- **INV-1** The design matrix has one column per pool member, in sorted `variable_id` order.
- **INV-2** Because every subset has exactly 25 of 40 members, the rows sum to a constant and
  the raw columns are collinear with the intercept. Columns are therefore **centred** by
  subtracting each column's mean before fitting, and ridge regularisation (`alpha > 0`) is
  applied, so the coefficients are identified. Fitting uncentred columns without a penalty is
  forbidden — it yields an arbitrary solution.
- **INV-3** `coefficients` covers every pool member exactly once.
- **INV-4** Coefficients are *relative* contributions; only their ordering and differences are
  meaningful, never their absolute level.
- **INV-5** `split_half` is the Pearson correlation between coefficients fitted on two random
  halves of the observations. It is the stability check that decides whether more sampling is
  needed; it is reported, never silently acted on.
- **INV-6** The fit is deterministic given the same observations and alpha, including the
  split-half partition, which is seeded.

## 5. State and side effects

Stateless and pure. No I/O.

## 6. Named failures and caller behavior

| Condition | Raises |
|---|---|
| fewer observations than pool members | `ValueError("attribution needs at least len(pool) observations")` |
| a subset contains an id outside the pool | `ValueError` naming the id |
| `alpha <= 0` | `ValueError` (INV-2) |
| any score is NaN or outside [0, 1] | `ValueError` |

## 7. Idempotency, concurrency, and resume behavior

Pure function; idempotent, concurrency-safe, no resume concept.

## 8. Exact `parameters.yml` keys consumed

None.

## 9. Secret handling and provenance

Handles no secrets. Provenance is the observation list, itself traceable to the harness JSONL.

## 10. Planned public functions and their contracts

```python
def fit(pool: list[str], observations: list[tuple[tuple[str, ...], float]],
        alpha: float = 1.0, seed: int = 0) -> dict:
    """Ridge-fit centred inclusion indicators to subset scores; report r2 and split-half."""

def ranking(coefficients: Mapping[str, float]) -> list[tuple[str, float]]:
    """Pool members ordered by coefficient descending, ties by sorted variable_id."""
```

## 11. Acceptance tests

1. On synthetic data where a known subset of examples each add a fixed amount, `fit` recovers
   their ordering exactly.
2. `fit` returns a coefficient for every pool member and no others (INV-3).
3. `fit` is deterministic: two calls on identical input give identical coefficients (INV-6).
4. `fit` raises when an observation contains an id outside the pool.
5. `fit` raises when given fewer observations than pool members.
6. `fit` raises when `alpha <= 0` (INV-2).
7. Centring is applied: fitting a constant-score dataset yields coefficients that are all
   approximately zero rather than an arbitrary split.
8. `split_half` is near 1.0 on noiseless synthetic data and materially lower when the scores
   are pure noise (INV-5).
9. `ranking` orders by coefficient descending and breaks ties by sorted `variable_id`.
