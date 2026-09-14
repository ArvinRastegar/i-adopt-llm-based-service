"""Set construction for the example-selection optimization — see contracts/design.md.

Pure and deterministic: every function here is a function of its arguments alone, performs no
I/O, and touches no provider. The split in particular takes no seed, because a partition that
could vary between runs would make two stages of the same experiment incomparable.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

POOL_SIZE = 40
CANDIDATE_SIZE = 25
SPLITS = ("P", "E", "C")


def corpus_split(records: list[dict]) -> dict[str, list[str]]:
    """Partition the corpus into candidate pool, search-eval and confirmation sets.

    Args: records: all 102 canonical corpus rows.
    Returns: {"P": [...], "E": [...], "C": [...]}, disjoint, each sorted, union = all ids.
    Raises: ValueError when the corpus is not 102 rows or a domain cannot be allocated.
    Side Effects: none. Deterministic and seedless, so it is identical on every machine.
    """
    if len(records) != 102:
        raise ValueError("design requires the complete 102-row corpus")
    by_domain = defaultdict(list)
    for row in records:
        by_domain[row["category"]].append(row["variable_id"])
    out: dict[str, list[str]] = {name: [] for name in SPLITS}
    # Allocate inside each domain rather than per split. Allocating each split independently
    # over the whole corpus does not reconcile - it returns 28 Natural Sciences where the
    # corpus has 29 - because three independent roundings need not sum to the original.
    for domain in sorted(by_domain):
        members = sorted(by_domain[domain])
        for name, count in zip(SPLITS, _apportion(len(members), _SPLIT_WEIGHTS)):
            taken, members = members[:count], members[count:]
            out[name].extend(taken)
    if len(out["P"]) != POOL_SIZE:
        raise ValueError(f"pool must be {POOL_SIZE}, apportionment produced {len(out['P'])}")
    return {name: sorted(ids) for name, ids in out.items()}


def sample_subsets(pool: list[str], n_subsets: int, size: int = CANDIDATE_SIZE,
                   seed: int = 0) -> list[tuple[str, ...]]:
    """Draw distinct sorted subsets of `pool`, reproducibly from `seed`.

    Args: pool: candidate pool ids; n_subsets: how many; size: members per subset; seed: RNG seed.
    Returns: `n_subsets` distinct sorted tuples, each a size-`size` subset of `pool`.
    Raises: ValueError when `size` exceeds the pool or `n_subsets` exceeds what is obtainable.
    Side Effects: none. Deterministic given `seed`.
    """
    if size > len(pool):
        raise ValueError(f"cannot draw {size} from a pool of {len(pool)}")
    rng = random.Random(seed)
    seen: set[tuple[str, ...]] = set()
    # Bounded because the caller may ask for more subsets than the pool can supply distinctly.
    attempts, ceiling = 0, max(1000, n_subsets * 200)
    while len(seen) < n_subsets:
        if attempts >= ceiling:
            raise ValueError(f"could only draw {len(seen)} distinct subsets of {n_subsets}")
        seen.add(tuple(sorted(rng.sample(pool, size))))
        attempts += 1
    return sorted(seen)


def top_k_candidate(pool: list[str], coefficients: Mapping[str, float],
                    size: int = CANDIDATE_SIZE, invert: bool = False) -> tuple[str, ...]:
    """Select the highest- (or lowest-) scoring pool members by fitted coefficient.

    Args: pool: candidate pool; coefficients: per-id contribution; size: how many; invert: take lowest.
    Returns: sorted tuple of `size` ids.
    Raises: ValueError when `coefficients` does not cover `pool`.
    Side Effects: none. Ties break by sorted variable_id, so the result is deterministic.
    """
    missing = set(pool) - set(coefficients)
    if missing:
        raise ValueError(f"coefficients missing {len(missing)} pool members")
    sign = 1 if invert else -1
    ordered = sorted(pool, key=lambda v: (sign * coefficients[v], v))
    return tuple(sorted(ordered[:size]))


def stratified_baseline(records: list[dict], pool: list[str],
                        size: int = CANDIDATE_SIZE) -> tuple[str, ...]:
    """Select a domain-proportional baseline from the pool.

    Args: records: canonical rows, for the category of each id; pool: candidate pool; size: how many.
    Returns: sorted tuple of `size` ids, domain proportions matching the pool to within one.
    Raises: ValueError when a domain cannot supply its allocation.
    Side Effects: none. Deterministic.
    """
    category = {row["variable_id"]: row["category"] for row in records}
    by_domain = defaultdict(list)
    for variable_id in sorted(pool):
        by_domain[category[variable_id]].append(variable_id)
    domains = sorted(by_domain)
    quota = _apportion(size, [len(by_domain[d]) for d in domains])
    chosen: list[str] = []
    for domain, count in zip(domains, quota):
        if count > len(by_domain[domain]):
            raise ValueError(f"{domain} needs {count} but has {len(by_domain[domain])}")
        chosen.extend(by_domain[domain][:count])
    return tuple(sorted(chosen))


def diversity_baseline(pool: list[str], embeddings: Mapping[str, Sequence[float]],
                       size: int = CANDIDATE_SIZE) -> tuple[str, ...]:
    """Select a maximally covering baseline by facility-location greedy on embeddings.

    Args: pool: candidate pool; embeddings: unit vectors per id; size: how many.
    Returns: sorted tuple of `size` ids.
    Raises: ValueError when `embeddings` does not cover `pool`.
    Side Effects: none. Seeded at the medoid with ties by sorted id, so it is deterministic.
    """
    missing = set(pool) - set(embeddings)
    if missing:
        raise ValueError(f"embeddings missing {len(missing)} pool members")
    if size > len(pool):
        raise ValueError(f"cannot select {size} from a pool of {len(pool)}")
    ordered = sorted(pool)
    sim = {a: {b: _cosine(embeddings[a], embeddings[b]) for b in ordered} for a in ordered}
    # Facility location: each step adds whoever most improves the pool's coverage, measured as
    # the sum over every pool member of its similarity to the nearest selected member.
    start = max(ordered, key=lambda a: (sum(sim[a].values()), a))
    chosen = [start]
    best = dict(sim[start])
    while len(chosen) < size:
        remaining = [v for v in ordered if v not in chosen]
        gain = {v: sum(max(best[u], sim[v][u]) for u in ordered) for v in remaining}
        pick = max(remaining, key=lambda v: (gain[v], v))
        chosen.append(pick)
        best = {u: max(best[u], sim[pick][u]) for u in ordered}
    return tuple(sorted(chosen))


def random_baselines(pool: list[str], n: int, size: int = CANDIDATE_SIZE,
                     seed: int = 0) -> list[tuple[str, ...]]:
    """Draw independent random selections for the random-25 reference distribution.

    Args: pool: candidate pool; n: how many baselines; size: members each; seed: RNG seed.
    Returns: `n` sorted tuples; duplicates are permitted, unlike `sample_subsets`.
    Raises: ValueError when `size` exceeds the pool.
    Side Effects: none. Deterministic given `seed`.
    """
    if size > len(pool):
        raise ValueError(f"cannot draw {size} from a pool of {len(pool)}")
    rng = random.Random(seed)
    return [tuple(sorted(rng.sample(pool, size))) for _ in range(n)]


def perturb(candidate: Sequence[str], pool: list[str], n: int, swaps: int = 1,
            seed: int = 0) -> list[tuple[str, ...]]:
    """Generate neighbours of `candidate` differing by exactly `swaps` members.

    Args: candidate: the set to perturb; pool: legal members; n: how many; swaps: members exchanged; seed: RNG seed.
    Returns: up to `n` distinct sorted tuples, each a size-|candidate| subset of `pool`.
    Raises: ValueError when `candidate` is not a subset of `pool` or `swaps` is not feasible.
    Side Effects: none. Deterministic given `seed`.
    """
    current = set(candidate)
    if not current <= set(pool):
        raise ValueError("candidate contains ids outside the pool")
    outside = sorted(set(pool) - current)
    if swaps > min(len(current), len(outside)):
        raise ValueError(f"cannot swap {swaps} members")
    rng = random.Random(seed)
    seen: set[tuple[str, ...]] = set()
    for _ in range(n * 50):
        if len(seen) >= n:
            break
        trial = set(current)
        for out, into in zip(rng.sample(sorted(current), swaps), rng.sample(outside, swaps)):
            trial.discard(out)
            trial.add(into)
        if len(trial) == len(current):
            seen.add(tuple(sorted(trial)))
    return sorted(seen)


# Proportional allocation to P/E/C. The weights make the pool exactly 40 on this corpus and
# split the remaining 62 as 38/24; `corpus_split` asserts the pool size rather than trusting it.
_SPLIT_WEIGHTS = (40, 38, 24)


def _apportion(total: int, weights: Sequence[int]) -> list[int]:
    """Divide `total` across `weights` by largest remainder.

    Args: total: units to hand out; weights: relative shares, at least one non-zero.
    Returns: integer counts summing exactly to `total`, in the order of `weights`.
    Raises: ValueError when every weight is zero.
    Side Effects: none. Deterministic; ties resolve toward the earlier index.
    """
    if sum(weights) <= 0:
        raise ValueError("apportionment needs a positive weight")
    exact = [total * w / sum(weights) for w in weights]
    base = [int(v) for v in exact]
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - base[i]), i))
    for i in order[: total - sum(base)]:
        base[i] += 1
    return base


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, 0.0 when either has zero norm."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
