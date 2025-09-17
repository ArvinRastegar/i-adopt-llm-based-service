# i-adopt-llm-based-service

Link to [I-Adopt workshop notes](https://docs.google.com/document/d/1eY9UGJv_YMGi1iYKPb-6SLo8GjZWMFy1oD4wWv8MsKY/edit?usp=sharing).

Link to [I-Adopt workshop Summary](https://docs.google.com/document/d/1YmfNovC-78ECMQCWaqeEndo3LW47REKYvqPEr9vMtK0/edit?usp=sharing).


## Wikidata linking experiments

Requirement: Conda

```bash
conda create -n i_adopt python=3.13
conda activate i_adopt
pip install -r requirements.txt
cd benchmarking_example/
```

### Naive approach

For the naive approach, we use the wikidata search function, and select as predicted entity the first result. This is used as baseline.

```bash
python experiment_design.py --shot 0 --workers 1 --approach naive
```

At the end, an excel file with the results will be generated, it will be placed at benchmarking_outputs/iadopt_metrics_YYYYMMDD_HHMMSS.xlsx, if we open the file, we will see the following results:

|     F_exact    |     P_exact    |     R_exact    |
|----------------|----------------|----------------|
|     0.618      |     0.501      |     0.96       |

### Embedding-based approach

For the embedding-based approach, we compute the embedding of the sentence 'Definition of "{term}" in context: "{context}"', where as context it is used the variable that we are decomposing e.g. "Distance to nearest neighbour habitat patch". We then compute the embeddings for the candidate entities, for each candidate we embedd the sentence 'label: "{label}", description: "{description}"', where label is the Wikidata label e.g. "Euclidean distance", and description is the Wikidata description e.g. "conventional distance in mathematics and physics". We then compute the cosine similarity between the "query" embedding and each of the candidate embeddings, and we use as final prediction the candidate with the highest cosine similarity.

```bash
python experiment_design.py --shot 0 --workers 1 --approach embedding
```

Results:
|     F_exact    |     P_exact    |     R_exact    |
|----------------|----------------|----------------|
|     0.562      |     0.43       |     0.96       |

### Cross-encoder-based approach

For the cross-encoder approach, we pass both the query and the candidate to the model, and it returns a score. In this case we pass: 'Definition of "{term}" in context: "{context}"' + 'label: "{label}", description: "{description}"'

This has to be passed for each candidate. More information about bi-encoder vs cross-encoder can be found in the [sentence-transformer documentation](https://sbert.net/examples/cross_encoder/applications/README.html).

```bash
python experiment_design.py --shot 0 --workers 1 --approach cross-encoder --model_name cross-encoder/ms-marco-MiniLM-L6-v2
```

|     F_exact    |     P_exact    |     R_exact    |
|----------------|----------------|----------------|
|     0.618      |     0.502      |     0.92       |

```bash
python experiment_design.py --shot 0 --workers 1 --approach cross-encoder --model_name cross-encoder/ms-marco-MiniLM-L12-v2
```

|     F_exact    |     P_exact    |     R_exact    |
|----------------|----------------|----------------|
|     0.678      |     0.559      |     0.96       |


For the Qwen3 reranker model, we use the following prompt template: 

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Given a web search query, retrieve relevant passages that answer the query 
<Query>: Definition of "{term}" in context: "{context}"
<Document>: label: "{label}", description: "{description}"<|im_end|>
<|im_start|>assistant
<think>

</think>
```

```bash
python experiment_design.py --shot 0 --workers 1 --approach cross-encoder --model_name tomaarsen/Qwen3-Reranker-0.6B-seq-cls
```
[log file](benchmarking_example/benchmarking_logs/iadopt_run_20250905_124920.log)

|     F_exact    |     P_exact    |     R_exact    |
|----------------|----------------|----------------|
|     0.737      |     0.618      |     1          |

```bash
python experiment_design.py --shot 0 --workers 1 --approach cross-encoder --model_name tomaarsen/Qwen3-Reranker-0.6B-seq-cls --threshold 0.5
```
[log file](benchmarking_example/benchmarking_logs/iadopt_run_20250910_111218.log)

|     F_exact    |     P_exact    |     R_exact    |
|----------------|----------------|----------------|
|     0.788      |     0.671      |     1          |

```bash
python experiment_design.py --shot 0 --workers 1 --approach cross-encoder --model_name tomaarsen/Qwen3-Reranker-0.6B-seq-cls --threshold 0.9
```
[log file](benchmarking_example/benchmarking_logs/iadopt_run_20250910_153120.log)

|     F_exact    |     P_exact    |     R_exact    |
|----------------|----------------|----------------|
|     0.815      |     0.709      |     1          |

### Evaluation only when there is a correct link

For the previous results, each time there is not a link, the same term is used, however, we might be interested in evaluate only when there is a link. To do this, we can use the "--none_if_no_link" flag.

```bash
python experiment_design.py --shot 0 --workers 1 --approach cross-encoder --model_name tomaarsen/Qwen3-Reranker-0.6B-seq-cls --none_if_no_link
```

|     F_exact    |     P_exact    |     R_exact    |
|----------------|----------------|----------------|
|     0.631      |     0.497      |     0.92       |

```bash
python experiment_design.py --shot 0 --workers 1 --approach cross-encoder --model_name tomaarsen/Qwen3-Reranker-0.6B-seq-cls --threshold 0.5 --none_if_no_link
```

|     F_exact    |     P_exact    |     R_exact    |
|----------------|----------------|----------------|
|     0.639      |     0.504      |     0.92       |

```bash
python experiment_design.py --shot 0 --workers 1 --approach cross-encoder --model_name tomaarsen/Qwen3-Reranker-0.6B-seq-cls --threshold 0.9 --none_if_no_link
```

|     F_exact    |     P_exact    |     R_exact    |
|----------------|----------------|----------------|
|     0.66       |     0.534      |     0.92       |