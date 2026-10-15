# iadopt-linking

Reproducible experiments measuring how well entity linkers map I-ADOPT slot values to
controlled-vocabulary identifiers: Wikidata first, then QUDT and OM for the Property slot.

It is a sibling of [`iadopt-lab`](../iadopt-lab), which runs the lexical-decomposition
experiments and owns the corpus. Linking lives apart on purpose: its own environment, its own
evaluation and its own results, so nothing here can change a decomposition result.

## The baseline

For every gold slot whose IRI names a controlled-vocabulary term (294 of the 313 linked nodes in
the 102 corpus variables, 153 distinct terms), the slot's term is searched live in Wikidata, QUDT
quantity kinds and OM quantities (at most 10 candidates each). The pooled candidates are scored by
the existing Qwen3-Reranker-0.6B with the variable's definition as context, and the best-scoring
candidate is compared with the gold identity.

Result of [`results/baseline-20261014`](results/baseline-20261014/report.md): 194 of 294 slots
correct (0.660) and 194 of the 284 slots whose gold the three sources can reach (0.683); the 10
slots in other vocabularies cannot be reached. By gold vocabulary: Wikidata 138/218, QUDT 51/61,
OM 5/5. The gold entity is among the pooled candidates for 275 of the 284, so most misses are
ranking and routing: knowing the source would give 216/284 (0.761), QUDT wins 19 of the
Wikidata-gold slots, and a Wikidata-only pipeline gets 153/218 (0.702) on the Wikidata gold
against 138/218 pooled. Three terms carry 36 of the 100 wrong slots: `person` (14 of 26),
`water` (12 of 12) and `urban area` (10 of 11).

On the manuscript's measure (Wikidata-gold cases, accuracy per variable) the first Wikidata hit
scores 0.627 against the manuscript's 0.656, and the reranker on the Wikidata candidates alone
0.677 against 0.759. The manuscript's figures are not reproduced: their inputs are not in this
repository, and this run does not explain the gap.

The headline does not depend on the device: 53 selections were decided by a top-two margin below
1e-3, all 53 were rescored on the CPU (scores within 1.6e-05 of the GPU's), and none changed. A
first run six and a half hours earlier changed no selection
([`replication.md`](results/baseline-20261014/replication.md)).

## How the two projects share work

| Shared thing | Owner | How this project uses it |
|---|---|---|
| Corpus Turtle (v2.0.1) and its frozen manifests | `iadopt-lab/data/` | read in place after the lab's own verification; never copied |
| Corpus loading and hash checks | `iadopt_lab.corpus.ingestion.load_canonical_records` | imported |
| Canonical JSON and SHA-256 identities | `iadopt_lab.canonical` | to be imported when a snapshot needs an identity |
| Lexical scorer | `iadopt_eval` | to be imported if end-to-end runs need it |
| Link gold, including ratio and source/target members | `iadopt_linking.gold` | here, because the lab's parser keeps IRIs only for top-level slots |
| Link evaluation | `iadopt_linking_eval` | here, standalone so the lab can import it later |

The dependency is one-way. `iadopt-lab` is hash-frozen, so this project never edits it and the
lab never imports this project. Where both read the same data, a contract test in `tests/`
fails if their readings diverge.

## Layout

```text
src/iadopt_linking/        gold, population, sources, http, recorder, retrieve, rerank, score,
                           health, devices, provenance, artifacts, errors, experiment, report,
                           compare, cli
src/iadopt_linking_eval/   pure link scorer (standard library only): identity, metrics
tests/                     offline tests, contract tests against iadopt-lab, result-integrity tests
results/<run>/             committed evidence and report of a run
outputs/                   gitignored scratch: logs, pilots, superseded runs
DECISIONS.md               why things are the way they are
```

A results folder holds `candidates.jsonl` (every retrieval, with request, time, attempts and
their causes, the response text and its hash), `scores.jsonl` (every scored slot with its
document, score and the model, revision, device and description cap that made it), `plan.json`
(what the folder is bound to), `provenance.json` (a journal of every invocation), `summary.json`,
`outcomes.csv` (one row per slot), `report.md` and `manifest.json` (code, corpus, model, source
releases, health checks, the cross-device check and the hash of every file). Retrieval is live, so
the stored candidates are the result; scoring and reporting repeat exactly from them.

## Commands

```bash
uv sync --extra rerank
uv run pytest
uv run ruff check .
uv run iadopt-linking plan
uv run iadopt-linking run --out results/<name>
uv run iadopt-linking compare --first results/a --second results/b
```

`run` resumes where it stopped and stops with a named reason on any failed request, failed health
check, missing or impossible score. A full run took 27 minutes: 15 for retrieval (Wikidata
rate-limits with 429, which the client waits out and records), 3.4 for scoring on the GPU and 8.7
for the CPU cross-check of 63 slots. Rerunning `run` in a finished folder rebuilds the report
without a service call or a model load. `--skip-device-check` and `--recheck-devices` control the
cross-check. The model is loaded from the local Hugging Face cache only.

Until `uv sync` has been run, the lab's interpreter runs everything read-only:
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=iadopt-linking/src iadopt-lab/.venv/bin/python -m iadopt_linking.cli ...`
from the repo root. The rules for working here are in [`AGENTS.md`](AGENTS.md).

## Status

The baseline is done and its evidence is in `results/`. Not yet decided: the evaluation population
for comparing the reranker with Jev (TypeSafe's constrained-output model, the candidate
replacement), whether QUDT's `wikidataMatch` makes a QUDT kind and its Wikidata item equally
correct (it would add 8 correct slots), whether the ten slots in other vocabularies should get a
source, and how to read the gap to the manuscript's table. QUDT accuracy on gold terms is close to
a label lookup (59 of 61 corpus terms are QUDT's own label), so it will not carry over to
LLM-predicted wording.
