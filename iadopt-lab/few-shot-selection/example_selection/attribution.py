"""Per-example contribution estimation — see contracts/attribution.md."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np


def fit(pool: list[str], observations: Sequence[tuple[tuple[str, ...], float]],
        alpha: float = 1.0, seed: int = 0) -> dict:
    """Ridge-fit centred inclusion indicators to observed subset scores."""
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
    """Order pool members by estimated contribution, descending."""
    return sorted(coefficients.items(), key=lambda item: (-item[1], item[0]))


def _ridge(matrix: "np.ndarray", scores: "np.ndarray", alpha: float) -> "np.ndarray":
    """Solve centred ridge regression."""
    centred = matrix - matrix.mean(axis=0)
    target = scores - scores.mean()
    gram = centred.T @ centred + alpha * np.eye(centred.shape[1])
    return np.linalg.solve(gram, centred.T @ target)


def _split_half(matrix: "np.ndarray", scores: "np.ndarray", alpha: float, seed: int) -> float:
    """Correlate coefficients fitted on two random halves, as a stability statistic."""
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
