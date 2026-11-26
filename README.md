# i-adopt-llm-based-service

How well can language models describe scientific variables? This repository measures it, in two steps: taking a definition apart into the parts of the [I-ADOPT framework](https://w3id.org/iadopt/), and linking those parts to controlled vocabularies. The results go into the [I-ADOPT Variable Description Service](https://github.com/ArvinRastegar/iadopt-variable-description-service).

[![DOI](https://zenodo.org/badge/946537030.svg)](https://doi.org/10.5281/zenodo.18108687)

## Highlights

- Compares 8 language models and several prompts on one benchmark of 102 scientific variables
- The score doubles when the prompt shows 20 worked examples instead of none ([results](iadopt-lab/few-shot-selection/shot-pool-30-results.md))
- A method to choose the best examples, which does better than random choices on variables it has not seen ([report](iadopt-lab/few-shot-selection/example_selection/output/example-selection-report.md))
- Picking the 25 worked examples most similar to each variable, instead of one fixed set of 25, raises the score from 0.59 to 0.68 (exploratory, [results](iadopt-lab/few-shot-selection/pattern-retrieval-results.md))
- Compares a chat model with a small reranker for linking ([experiments](iadopt-linking/README.md))
- Built to be reproduced: a fixed scorer, stored prompts, answers and settings, campaigns that can be resumed, and more than 1,100 tests that run offline

## What is where

| Folder | What |
|---|---|
| [`iadopt-lab/`](iadopt-lab/README.md) | The decomposition experiments: prompts, providers, scorer, PostgreSQL store, campaigns, reports |
| [`iadopt-lab/few-shot-selection/`](iadopt-lab/few-shot-selection/example_selection/README.md) | Studies on the examples in the prompt, including [picking the most similar ones](iadopt-lab/few-shot-selection/pattern-retrieval-results.md) |
| [`iadopt-linking/`](iadopt-linking/README.md) | The linking experiments, with the answers of every run in `results/` |
| [`configuration-results.csv`](configuration-results.csv) | One row per model configuration |
| `Expert-Assessment/` | Expert evaluations of the corpus and of model outputs, see the [note in the folder](Expert-Assessment/Corpus%20Analysis%20-%20readme.md) |
| `error_analysis/` | Analysis scripts |
| `misc/` | Older experiments and the first pipeline |

## Quick start

You need [uv](https://docs.astral.sh/uv/) and Python 3.12. The tests need no key and no database.

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

Tests that need a database are skipped without one. `verify` checks the corpus files. To ask a model you need a provider key, see the [runbook](iadopt-lab/docs/runbook.md).

## Reading the numbers

Scores are F1 over six fields. Numbers of different studies are not directly comparable, and the same model can score differently on different providers ([how to read the tables](iadopt-lab/docs/results-model-comparison.md)).

## Contributors

Arvin Rastegar, Barbara Magagna, Cristian Berrio Aroca, esgg and Saurav Kumar. The corpus is the work of the [I-ADOPT Corpus](https://github.com/i-adopt/Corpus) contributors.

## Licence

[MIT](LICENSE). The corpus files are CC BY 4.0 ([provenance](iadopt-lab/data/corpus/LICENSE-PROVENANCE.md), [notices](iadopt-lab/THIRD_PARTY_NOTICES.md)).
