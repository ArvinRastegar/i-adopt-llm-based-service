# Example-selection optimization — result

**Exploratory side experiment.** Not part of the official experiment; these numbers
cannot enter an official ranking. See [the plan](../README.md).

Split: pool 40, search-eval 38, confirmation 24.

## Held-out confirmation (the result)

Measured on `C`, which never informed selection. This is the only evidence here.

| Candidate | Close F1 | SD | 95% CI |
|---|---:|---:|---|
| `top25` | 0.4890 | 0.0252 | [0.4763, 0.5018] |
| `refined` | 0.4805 | 0.0281 | [0.4663, 0.4947] |
| `random-1` | 0.4757 | 0.0247 | [0.4633, 0.4882] |
| `random-2` | 0.4464 | 0.0257 | [0.4334, 0.4594] |
| `random-3` | 0.4349 | 0.0378 | [0.4157, 0.4540] |
| `bottom25` | 0.3981 | 0.0258 | [0.3850, 0.4111] |
| `stratified` | 0.3952 | 0.0246 | [0.3827, 0.4076] |

## Search scores (selection-biased — not the result)

Every figure below was measured on `E`, the set used to *choose* these candidates.
Picking the maximum of a noisy objective inflates it, so these are **selection-biased**
and are reported only to show the search worked, never as the finding.

- `bottom25`: 0.2683 (selection-biased)
- `random-1`: 0.4044 (selection-biased)
- `random-2`: 0.4842 (selection-biased)
- `random-3`: 0.4007 (selection-biased)
- `stratified`: 0.4055 (selection-biased)
- `top25`: 0.5218 (selection-biased)

## Falsification check

The top-25 beat both the attenuated bottom-25 control and the random-25
reference, so the attribution model learned something real.

The bottom-25 control is **attenuated**: it necessarily shares 10 of its 25
members with the top-25 (design INV-9), so a real effect appears reduced. Do not
over-read a small margin against it.

## Significance on the held-out set

15 repetitions per candidate. Welch tests, Holm-corrected over four comparisons.

| top25 vs | Difference | t | Holm p | Verdict |
|---|---:|---:|---:|---|
| `bottom25` | +0.0909 | +9.77 | <0.0001 | significant |
| `stratified` | +0.0938 | +10.33 | <0.0001 | significant |
| all randoms pooled | +0.0367 | +4.45 | 0.0002 | significant |
| **best random draw** | **+0.0133** | **+1.46** | **0.156** | **not significant** |

**The attribution model learned something real.** It beats both falsification arms decisively,
and the margin over the attenuated `bottom25` control (+0.091) is large despite that control
sharing 10 of its 25 members with the winner.

**But it does not beat a lucky random draw.** `random-1` reached 0.4757 against the optimized
0.4890, and that gap is inside the noise. Random 25-subsets vary widely on `C` — 0.435 to 0.476
across just three draws — so drawing a few at random and keeping the best is a real competitor
to this whole procedure. The honest claim is "reliably better than an average or a bad
selection", not "better than anything else you could do".

**Selection bias behaved exactly as DS-2 predicted.** `top25` measured 0.5218 on `E`, the set
that chose it, and 0.4890 on `C`, which never saw it: an inflation of **+0.033**. Reporting the
`E` figure as the result would have overstated the method by a quarter of its own effect. This
is why `C` was held out.

**Refinement contributed nothing.** The swap search returned `refined` at 0.4805, *below* the
unrefined `top25` at 0.4890. It compared individual candidates at differences far under the
noise floor, which is precisely the failure mode DS-4 predicted for greedy and hill-climbing
methods. The stage should be dropped or given far heavier replication.

**The stratified baseline is the worst performer**, statistically tied with the deliberately-bad
`bottom25` (0.3952 against 0.3981). Domain-proportional selection is intuitive and principled
and it is worth no more here than an adversarially chosen set. That finding applies directly to
the sibling 30-shot experiment, whose pool was built by domain stratification.
