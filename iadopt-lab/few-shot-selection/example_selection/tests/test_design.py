"""Acceptance tests for contracts/design.md."""

from __future__ import annotations

from collections import Counter

import pytest
from design import (
    CANDIDATE_SIZE,
    POOL_SIZE,
    corpus_split,
    diversity_baseline,
    perturb,
    random_baselines,
    sample_subsets,
    stratified_baseline,
    top_k_candidate,
)


def test_split_is_a_partition(records, split):
    """AT-1: P, E and C are pairwise disjoint and cover every corpus id (INV-1)."""
    p, e, c = set(split["P"]), set(split["E"]), set(split["C"])
    assert p & e == set() and p & c == set() and e & c == set()
    assert p | e | c == {row["variable_id"] for row in records}


def test_pool_is_exactly_forty(split):
    """AT-2: the candidate pool is exactly 40 (INV-2)."""
    assert len(split["P"]) == POOL_SIZE


def test_domain_totals_reconcile_with_corpus(records, split):
    """AT-3: per-domain counts summed over P/E/C equal the corpus exactly (INV-3)."""
    category = {row["variable_id"]: row["category"] for row in records}
    total = Counter()
    for name in ("P", "E", "C"):
        total.update(Counter(category[v] for v in split[name]))
    assert total == Counter(row["category"] for row in records)


def test_split_is_deterministic(records):
    """AT-4: the split takes no seed and is identical on repeated calls (INV-4)."""
    assert corpus_split(records) == corpus_split(records)


def test_split_rejects_incomplete_corpus(records):
    """AT-5: an incomplete corpus is a programming error, not a silent partial split."""
    with pytest.raises(ValueError):
        corpus_split(records[:-1])


def test_sampled_subsets_are_valid(pool):
    """AT-6: each subset is size 25 and a subset of the pool (INV-6)."""
    subsets = sample_subsets(pool, 50, seed=1)
    assert len(subsets) == 50
    assert all(len(s) == CANDIDATE_SIZE and set(s) <= set(pool) for s in subsets)


def test_sampling_is_distinct_and_reproducible(pool):
    """AT-7: no duplicates, and the same seed reproduces the same list (INV-7)."""
    subsets = sample_subsets(pool, 50, seed=1)
    assert len(set(subsets)) == len(subsets)
    assert subsets == sample_subsets(pool, 50, seed=1)


def test_different_seeds_differ(pool):
    """AT-8: sampling actually depends on the seed."""
    assert sample_subsets(pool, 50, seed=1) != sample_subsets(pool, 50, seed=2)


def test_top_and_bottom_overlap_by_arithmetic(pool):
    """AT-9: the two selections overlap by 2*size-len(pool) and differ in len(pool)-size (INV-9).

    They cannot be disjoint: 25 + 25 exceeds a 40-member pool. The contract originally claimed
    disjointness and this test is what caught it.
    """
    coefficients = {v: float(i) for i, v in enumerate(sorted(pool))}
    top = top_k_candidate(pool, coefficients)
    bottom = top_k_candidate(pool, coefficients, invert=True)
    assert len(top) == len(bottom) == CANDIDATE_SIZE
    assert len(set(top) & set(bottom)) == 2 * CANDIDATE_SIZE - len(pool)
    assert len(set(top) - set(bottom)) == len(pool) - CANDIDATE_SIZE
    assert set(top) == set(sorted(pool, key=lambda v: -coefficients[v])[:CANDIDATE_SIZE])


def test_top_k_breaks_ties_by_id(pool):
    """AT-10: equal coefficients resolve by sorted variable_id, so the result is stable."""
    flat = dict.fromkeys(pool, 1.0)
    assert top_k_candidate(pool, flat) == tuple(sorted(pool)[:CANDIDATE_SIZE])


def test_top_k_requires_full_coverage(pool):
    """A coefficient map missing a pool member is a programming error."""
    partial = {v: 0.0 for v in sorted(pool)[:-1]}
    with pytest.raises(ValueError):
        top_k_candidate(pool, partial)


def test_stratified_baseline_matches_pool_proportions(records, pool):
    """AT-11: domain proportions track the pool to within one member."""
    category = {row["variable_id"]: row["category"] for row in records}
    chosen = stratified_baseline(records, pool)
    assert len(chosen) == CANDIDATE_SIZE
    got = Counter(category[v] for v in chosen)
    for domain, count in Counter(category[v] for v in pool).items():
        assert abs(got[domain] - count * CANDIDATE_SIZE / len(pool)) <= 1.0


def test_diversity_baseline_is_deterministic(pool):
    """AT-12: 25 distinct pool members, identical across calls."""
    embeddings = {v: [float(i), float(i * i % 7), 1.0] for i, v in enumerate(sorted(pool))}
    first = diversity_baseline(pool, embeddings)
    assert len(set(first)) == CANDIDATE_SIZE and set(first) <= set(pool)
    assert first == diversity_baseline(pool, embeddings)


def test_perturb_changes_exactly_the_requested_members(pool):
    """AT-13: each neighbour differs in exactly `swaps` members and stays legal."""
    base = sample_subsets(pool, 1, seed=3)[0]
    for neighbour in perturb(base, pool, 5, swaps=1, seed=4):
        assert len(neighbour) == CANDIDATE_SIZE
        assert set(neighbour) <= set(pool)
        assert len(set(base) - set(neighbour)) == 1


def test_random_baselines_are_reproducible(pool):
    """Random baselines are a seeded reference distribution, not ad-hoc draws."""
    assert random_baselines(pool, 10, seed=5) == random_baselines(pool, 10, seed=5)


def test_functions_raise_when_pool_too_small():
    """AT-14: nothing silently truncates when the pool cannot supply the request."""
    with pytest.raises(ValueError):
        sample_subsets(["a", "b"], 1, size=CANDIDATE_SIZE)
    with pytest.raises(ValueError):
        random_baselines(["a", "b"], 1, size=CANDIDATE_SIZE)
