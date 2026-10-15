# Replication of the baseline

The first full run of this baseline (`baseline-20261014-v1`: the code before the hardening of
decision L-009, superseded and not committed) retrieved its candidates on 2026-10-14 between 07:05
and 07:20 UTC. The committed run (`baseline-20261014`) retrieved them between 13:42 and 13:57 UTC
the same day, about six and a half hours later, with the same model revision on the same device
(MPS). This comparison reads only the stored evidence of both runs and calls nothing.

294 slots compared (0 only in baseline-20261014-v1, 0 only in baseline-20261014).

Correct: baseline-20261014-v1 194/294, baseline-20261014 194/294. The selected candidate changed for 0 slots.

| Source | Queries | Identical lists | Same set, reordered | Different | Unavailable |
|---|---|---|---|---|---|
| wikidata | 153 | 149 | 3 | 1 | 0 |
| qudt | 153 | 153 | 0 | 0 | 0 |
| om | 153 | 153 | 0 | 0 | 0 |

Candidates present in both runs: 3472; score difference mean 0.00e+00, max 0.00e+00.

Reading: the services returned the same candidates for 455 of the 459 queries; the four that
differ are Wikidata lists (three reordered, one with different members), and they changed no
selection. The scores of the candidates present in both runs are identical. This shows that
retrieval is stable over hours; Wikidata is edited continuously, so it says nothing about weeks.
