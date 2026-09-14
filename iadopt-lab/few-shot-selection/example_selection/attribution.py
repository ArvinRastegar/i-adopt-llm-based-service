"""Per-example contribution estimation — see contracts/attribution.md.

The design matrix is rank-deficient by construction: every subset has exactly 25 of 40 members,
so the inclusion columns sum to a constant and are collinear with the intercept. Columns are
centred and ridge-regularised so the coefficients are identified; fitting raw columns without a
penalty returns an arbitrary point on a solution ray.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np


def fit(pool: list[str], observations: Sequence[tuple[tuple[str, ...], float]],
        alpha: float = 1.0, seed: int = 0) -> dict:
    """Ridge-fit centred inclusion indicators to observed subset scores.

    Args: pool: candidate pool ids; observations: (subset, close_f1) pairs; alpha: ridge penalty; seed: split-half seed.
    Returns: {"coefficients": {id: float}, "intercept": float, "r2": float, "split_half": float, "n": int}.
    Raises: ValueError for too few observations, an id outside the pool, alpha <= 0, or a score outside [0, 1].
    Side Effects: none. Deterministic given the same observations, alpha and seed.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive; the design matrix is rank-deficient")
    columns = sorted(pool)
    if len(observations) < len(columns):
        raise ValueError("attribution needs at least len(pool) observations")
    index = {variable_id: position for position, variable_id in enumerate(columns)}
    matrix = np.zeros((len(observations), len(columns)))
    scores = np.empty(len(observations))
    for row, (subset, score) in enumerate(observations):
        if not 0.0 <= score <= 1.0 or score != score:
            raise ValueError(f"score {score!r} is outside [0, 1]")
        for variable_id in subset:
            if variable_id not in index:
                raise ValueError(f"observation names {variable_id!r}, which is outside the pool")
            matrix[row, index[variable_id]] = 1.0
        scores[row] = score
    coefficients = _ridge(matrix, scores, alpha)
    fitted = matrix @ coefficients + scores.mean() - (matrix.mean(axis=0) @ coefficients)
    residual = float(((scores - fitted) ** 2).sum())
    total = float(((scores - scores.mean()) ** 2).sum())
    return {"coefficients": {v: float(coefficients[index[v]]) for v in columns},
            "intercept": float(scores.mean()),
            "r2": 1.0 - residual / total if total > 0 else 1.0,
            "split_half": _split_half(matrix, scores, alpha, seed),
            "n": len(observations)}


def ranking(coefficients: Mapping[str, float]) -> list[tuple[str, float]]:
    """Order pool members by estimated contribution, descending.

    Args: coefficients: per-id fitted contribution.
    Returns: (id, coefficient) pairs, highest first, ties broken by sorted variable_id.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return sorted(coefficients.items(), key=lambda item: (-item[1], item[0]))


def _ridge(matrix: "np.ndarray", scores: "np.ndarray", alpha: float) -> "np.ndarray":
    """Solve centred ridge regression.

    Args: matrix: n x p binary inclusion design; scores: n observed values; alpha: penalty.
    Returns: p coefficients.
    Raises: nothing; the penalty guarantees the normal equations are non-singular.
    Side Effects: none. Deterministic.

    Centring is not cosmetic. Every subset has exactly the same number of members, so the raw
    columns sum to a constant along each row and are collinear with the intercept; the uncentred
    normal equations are singular and the solver would return an arbitrary point on a ray.
    """
    centred = matrix - matrix.mean(axis=0)
    target = scores - scores.mean()
    gram = centred.T @ centred + alpha * np.eye(centred.shape[1])
    return np.linalg.solve(gram, centred.T @ target)


def _split_half(matrix: "np.ndarray", scores: "np.ndarray", alpha: float, seed: int) -> float:
    """Correlate coefficients fitted on two random halves, as a stability statistic.

    Args: matrix: design; scores: observations; alpha: penalty; seed: partition seed.
    Returns: Pearson correlation of the two coefficient vectors; 0.0 when either is constant.
    Raises: nothing.
    Side Effects: none. Deterministic given `seed`.
    """
    order = list(range(matrix.shape[0]))
    random.Random(seed).shuffle(order)
    cut = len(order) // 2
    if cut < 2 or len(order) - cut < 2:
        return 0.0
    first = _ridge(matrix[order[:cut]], scores[order[:cut]], alpha)
    second = _ridge(matrix[order[cut:]], scores[order[cut:]], alpha)
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])
