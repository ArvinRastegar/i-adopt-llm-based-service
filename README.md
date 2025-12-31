# i-adopt-llm-based-service

Link to [I-Adopt workshop notes](https://docs.google.com/document/d/1eY9UGJv_YMGi1iYKPb-6SLo8GjZWMFy1oD4wWv8MsKY/edit?usp=sharing).

Link to [I-Adopt workshop Summary](https://docs.google.com/document/d/1YmfNovC-78ECMQCWaqeEndo3LW47REKYvqPEr9vMtK0/edit?usp=sharing).

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