# i-adopt-llm-based-service

<!-- Link to [I-Adopt workshop notes](https://docs.google.com/document/d/1eY9UGJv_YMGi1iYKPb-6SLo8GjZWMFy1oD4wWv8MsKY/edit?usp=sharing).

Link to [I-Adopt workshop Summary](https://docs.google.com/document/d/1YmfNovC-78ECMQCWaqeEndo3LW47REKYvqPEr9vMtK0/edit?usp=sharing). -->

## Latest results

**[`configuration-results.csv`](configuration-results.csv) holds the latest experiment
results for now.** It is a plain CSV — no database or setup needed, and GitHub renders it
as a sortable table.

Every one of the 298 model configurations that completed the full 97-variable evaluation,
sorted by Close F1 descending:

| Column | Meaning |
|---|---|
| `model_id`, `provider` | which model, on which provider |
| `prompt`, `shots`, `temperature`, `reasoning` | the configuration that was run |
| `close_precision`, `close_recall`, `close_f1` | scored with similarity ≥ 0.8 |
| `exact_precision`, `exact_recall`, `exact_f1` | scored on identical text |

Best configuration is `z-ai/glm-5.2` with the matrix-decomposition prompt, 5 examples,
temperature 0.5 and reasoning enabled, at **0.4206** Close F1.

The experiment that produced these numbers lives in [`iadopt-lab/`](iadopt-lab/); the
commentary is in [`iadopt-lab/docs/results-top-configurations.md`](iadopt-lab/docs/results-top-configurations.md)
and [`results-model-comparison.md`](iadopt-lab/docs/results-model-comparison.md).

## Getting started

1. Install Dependencies

From the root of the repository, install all required Python packages:

pip install -r requirements.txt

2. Configure Environment Variables

Create a .env file in the root directory of the project and add your OpenRouter API key:

OPENROUTER_API_KEY=your_api_key_here


Make sure the .env file is located at the project root so it can be detected by the scripts.

3. Phase One: Variable Decomposition

To reproduce the variable decomposition output, run the following script:

python3 ./benchmarking_example/randomShotsPhaseOne.py --mode fixed-examples-grid


This script performs the decomposition of variables using a fixed example grid configuration.

4. Phase One (Merged): Decomposition + Vocabulary Linking

To reproduce both:

the variable decomposition, and

the linking of controlled vocabularies to Wikidata entities,

run the merged script:

python3 ./benchmarking_example/phaseOneThreeMerged.py


This script assumes:

all dependencies are installed,

the .env file is present, and

the OPENROUTER_API_KEY is correctly set.


[![DOI](https://zenodo.org/badge/946537030.svg)](https://doi.org/10.5281/zenodo.18108687)
