"""Acceptance tests for contracts/attribution.md."""

from __future__ import annotations

import random

import pytest
from attribution import fit, ranking


def _pool(n=40):
    return [f"urn:v:{i:03d}" for i in range(n)]


def _synthetic(pool, weights, n_obs=400, noise=0.0, seed=0):
    """Subsets scored as 0.3 + sum of member weights, optionally with gaussian noise."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_obs):
        subset = tuple(sorted(rng.sample(pool, 25)))
        score = 0.3 + sum(weights[v] for v in subset) + (rng.gauss(0, noise) if noise else 0.0)
        out.append((subset, min(1.0, max(0.0, score))))
    return out


def test_recovers_known_ordering():
    """AT-1: examples that genuinely add more must rank higher."""
    pool = _pool()
    weights = {v: (0.01 if i < 10 else 0.0) for i, v in enumerate(pool)}
    result = fit(pool, _synthetic(pool, weights))
    top10 = {v for v, _ in ranking(result["coefficients"])[:10]}
    assert top10 == set(pool[:10])


def test_covers_every_pool_member(_pool_fixture=None):
    """AT-2: a coefficient for every pool member and nothing else (INV-3)."""
    pool = _pool()
    result = fit(pool, _synthetic(pool, dict.fromkeys(pool, 0.0)))
    assert set(result["coefficients"]) == set(pool)


def test_is_deterministic():
    """AT-3: identical input gives identical coefficients (INV-6)."""
    pool = _pool()
    obs = _synthetic(pool, dict.fromkeys(pool, 0.0))
    assert fit(pool, obs)["coefficients"] == fit(pool, obs)["coefficients"]


def test_rejects_id_outside_pool():
    """AT-4: an observation naming an unknown id is a programming error."""
    pool = _pool()
    obs = _synthetic(pool, dict.fromkeys(pool, 0.0), n_obs=50)
    obs.append((tuple(sorted(pool[:24] + ["urn:v:999"])), 0.5))
    with pytest.raises(ValueError):
        fit(pool, obs)


def test_rejects_too_few_observations():
    """AT-5: fewer observations than parameters cannot identify the model."""
    pool = _pool()
    with pytest.raises(ValueError):
        fit(pool, _synthetic(pool, dict.fromkeys(pool, 0.0), n_obs=10))


def test_rejects_non_positive_alpha():
    """AT-6: alpha <= 0 leaves the rank-deficient system unidentified (INV-2)."""
    pool = _pool()
    with pytest.raises(ValueError):
        fit(pool, _synthetic(pool, dict.fromkeys(pool, 0.0)), alpha=0.0)


def test_constant_scores_give_near_zero_coefficients():
    """AT-7: with no signal, centring must leave coefficients at ~0, not an arbitrary split."""
    pool = _pool()
    obs = [(s, 0.5) for s, _ in _synthetic(pool, dict.fromkeys(pool, 0.0))]
    coefficients = fit(pool, obs)["coefficients"]
    assert max(abs(c) for c in coefficients.values()) < 1e-6


def test_split_half_separates_signal_from_noise():
    """AT-8: the stability statistic is high on clean signal and low on pure noise (INV-5)."""
    pool = _pool()
    weights = {v: (0.01 if i < 10 else 0.0) for i, v in enumerate(pool)}
    clean = fit(pool, _synthetic(pool, weights, n_obs=600))["split_half"]
    noisy = fit(pool, _synthetic(pool, dict.fromkeys(pool, 0.0), n_obs=600, noise=0.05, seed=7))["split_half"]
    assert clean > 0.9
    assert noisy < clean


def test_ranking_orders_and_breaks_ties():
    """AT-9: descending by coefficient, ties by sorted variable_id."""
    coefficients = {"urn:v:002": 1.0, "urn:v:000": 1.0, "urn:v:001": 2.0}
    assert [v for v, _ in ranking(coefficients)] == ["urn:v:001", "urn:v:000", "urn:v:002"]
