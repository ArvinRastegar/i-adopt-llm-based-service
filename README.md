# i-adopt-llm-based-service

Experiments on turning the definition of a scientific variable into an [I-ADOPT](https://w3id.org/iadopt/) description with
language models. Two questions are measured apart: how well a model takes a definition apart into six fields
(**decomposition**), and how well it links the parts to entries in Wikidata, QUDT and OM (**linking**). The web service that uses the
results is [`iadopt-variable-description-service`](https://github.com/ArvinRastegar/iadopt-variable-description-service).

[![DOI](https://zenodo.org/badge/946537030.svg)](https://doi.org/10.5281/zenodo.18108687)

The DOI's latest version is from 13 October 2026; the linking experiments were added on 15 and 16 October and are not in it.

## Results

| What was measured | Result | Read it in |
|---|---|---|
| 298 configurations that completed, of 396 planned (9 model deployments), 97 variables, one run each | best Close F1 0.4206: `z-ai/glm-5.2`, 5 examples, temperature 0.5, reasoning on; 18 of its 36 configurations completed | [`configuration-results.csv`](configuration-results.csv), [campaign log](iadopt-lab/docs/campaign-log.md) |
| More examples in the prompt, `Qwen3.8-27B`, 72 variables | one run each: 0.224 with none, 0.451 with 20, 0.456 with 30; ten repetitions each at 20, 25 and 30: 0.456, 0.451, 0.468, no difference significant | [`shot-pool-30-results.md`](iadopt-lab/few-shot-selection/shot-pool-30-results.md) |
| Choosing 25 examples, 24 variables, 15 repetitions | 0.4890 (SD 0.0252); +0.0367 over three random sets pooled (p = 0.0002), +0.0133 over the best of them (p = 0.156, not significant) | [`example-selection-report.md`](iadopt-lab/few-shot-selection/example_selection/output/example-selection-report.md) |
| Linking, 284 reachable slots, one pass each | reranker 194, chat model 206, `jev-1.13` 213 right; neither difference from the reranker can be told from zero, the gains are in Wikidata and the losses in QUDT | [`iadopt-linking`](iadopt-linking/README.md) |

The rankings of the main experiment and the exploratory studies under `few-shot-selection/` are kept apart. Providers are confounded with models
([how to read the tables](iadopt-lab/docs/results-model-comparison.md)), and rows of different studies are not comparable: the official runs ask an
invalid answer again, the studies count it as empty.

## What is where

| Folder | What |
|---|---|
| [`iadopt-lab/`](iadopt-lab/README.md) | The decomposition experiments: corpus, prompts, providers, scorer, PostgreSQL store, reports |
| [`iadopt-lab/few-shot-selection/`](iadopt-lab/few-shot-selection/example_selection/README.md) | Exploratory studies on examples in the prompt; `guided/` has code and data of a later one, its results are not public |
| [`iadopt-linking/`](iadopt-linking/README.md) | The linking experiments, with the raw answers of each run in `results/` |
| [`configuration-results.csv`](configuration-results.csv) | One row per configuration of the main experiment |
| `error_analysis/` | Analysis scripts |
| `Expert-Assessment/` | A spreadsheet with expert evaluations of the corpus and of model outputs, see the [note in the folder](Expert-Assessment/Corpus%20Analysis%20-%20readme.md) |
| `misc/` | Older experiments and the first pipeline, kept as they were |

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12; no key and no database. Both projects run their tests offline.

```bash
git clone https://github.com/ArvinRastegar/i-adopt-llm-based-service.git
cd i-adopt-llm-based-service/iadopt-lab
uv sync
uv run pytest -q -rs
uv run iadopt-lab verify
```

```bash
cd ../iadopt-linking
uv sync
uv run pytest -q
```

Database tests skip without a dedicated test database; `-rs` lists what skipped. `verify` checks the hashes of the corpus files. Asking a
model needs a provider key and an explicit authorisation, see the [runbook](iadopt-lab/docs/runbook.md).

The older scripts of the first experiments are in `misc/benchmarking_example/`.

## Contributors

In the history of `main` up to 20 November 2026 (`4c56487`): Arvin Rastegar, Barbara Magagna, Cristian Berrio Aroca, esgg and Saurav Kumar. The corpus is
the work of the [I-ADOPT Corpus](https://github.com/i-adopt/Corpus) contributors.

## Licence

[MIT](LICENSE). The corpus files are CC BY 4.0 ([provenance](iadopt-lab/data/corpus/LICENSE-PROVENANCE.md),
[notices](iadopt-lab/THIRD_PARTY_NOTICES.md)).
