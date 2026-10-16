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

## Jev on the same candidates

[`results/jev-20261015`](results/jev-20261015/report.md) asks TypeSafe's Jev (`typesafe/jev-1.13`
through OpenRouter) to choose among exactly the candidates the baseline scored, so only the scorer
differs (DECISIONS L-010 to L-015). Jev answers a typed question with a probability for each option.
Each slot is one Choice question over its pooled candidates plus `none_of_these`, asked in four
Choice passes (three distinct candidate orders; one pass repeats the primary request exactly) and
once more as one yes/no question per candidate. The 1,430 requests (one of them asked a second time
after a rejected answer, L-014) took two minutes and cost $0.07 as OpenRouter reports it. The prompt
was written before any corpus slot was sent and developed on invented slots only (L-011); the primary
analysis was written down before the first request (L-012), pre-specified rather than registered
anywhere outside this repository.

On the 284 reachable slots Jev's first Choice pass gets 213 right (0.750) against Qwen3's 194
(0.683). That +0.067 cannot be told from zero once a term asked many times counts once: the interval
that resamples terms is -0.040 to +0.182, and (post hoc, L-014) Jev does better on 17 of 145 terms,
Qwen3 on 16. It is two opposite effects. Where the gold is in Wikidata, Jev gets 174/218 against
138/218 (post hoc: better on 15 terms, worse on 1). Where it is in QUDT, 34/61 against 51/61 (better
on 2, worse on 15), because Jev often picks the Wikidata item that QUDT lists as the twin of the same
concept (post hoc: 16 of the 23 slots only Qwen3 got right are such twins). Counting a QUDT kind and its twin as
one entity, an open decision, gives 237/284 = 0.835 against 202/284 = 0.711.

The total hides how much individual slots move. The four Choice passes get 210 to 213 right, and an
identical repeat selects the same candidate for 98% of slots (though only 27% of slots get identical
probabilities). Another order changes the selection for 9 to 12% of slots and flips whether the answer
is right for 23 to 31 of the 278 asked reachable slots, in both directions (post hoc); all four passes
select the same candidate on 232 of them. No pass favours the first option shown. The pointwise arm
is worse (177/284).

Confidence is where Jev differs most. The probability of its selected candidate separates right from
wrong (AUROC 0.81, calibration gap 0.045) where Qwen3's score hardly does (0.59, 0.28). Answering only
at probability 0.8 or more covers 157 of 284 slots at 93% precision, and at 0.9 or more 116 slots at
97%. Jev's own `confidence` matches (p_max - 1/n) / (1 - 1/n) in all 1,144 Choice responses, so it
adds nothing to the probability and the option count. `outcomes.csv`, `confidence.csv` and
`probabilities.csv` hold every figure per slot and candidate for choosing thresholds later, and the
report's escalation table counts the errors a stronger model could and could not fix. A threshold
read off these same data needs held-out terms before it is trusted.

## A PSNC chat model on the same candidates

[`results/psnc-qwen3.8-27b-20261015`](results/psnc-qwen3.8-27b-20261015/report.md) asks
`Qwen3.8-27B`, a chat model on PSNC's gateway with reasoning switched off, the question Jev was asked
about exactly the same candidates, in the same four candidate orders (DECISIONS L-016 to L-018). A
chat model answers in text, so the options are named by one-token labels, the model replies with one
label, and the probability of each option is read from the log-probabilities of that token. The 1,144
requests were sent one at a time, took under six minutes and are not billed to the owner.

On the 284 reachable slots its first pass gets 206 right (0.725), against Qwen3-0.6B's 194 (0.683) and
Jev's 213 (0.750). Neither difference can be told from zero: against the baseline +0.042 (term-resampled
95% interval -0.073 to +0.172; better on 19 of 145 terms, worse on 16), against Jev -0.025 (-0.082 to
+0.034; better on 13 terms, worse on 12). The total hides the same two effects as in Jev's run. On
Wikidata gold it gets 169/218 against 138/218 (better on 17 terms, worse on 2); on QUDT gold 32/61
against 51/61 (better on 2, worse on 14), because it often picks the Wikidata (16 slots) or OM (13)
item that duplicates a QUDT kind. It gets `water` and `urban area` right in every slot, where the
baseline misses all 12 and 10 of 11, and `person` wrong in 20 of 26 (the baseline 14), where the gold
itself names two different items.

Its probabilities are informative. The probability of the selected candidate separates right from
wrong answers (AUROC 0.86; Jev 0.81; the Qwen3 score 0.59), though it is somewhat over-confident
(calibration gap 0.080). Answering only at 0.9 or more covers 135 of 284 slots at 97% precision. With
one request at a time the answers were reproducible: the identical repeat selected the same candidate
and gave identical probabilities for 286 of 286 slots, and a complete run made minutes before it (a
shakedown, kept outside the repository in `outputs/`; DECISIONS L-017) gave the same top-20
log-probabilities in 1,143 of 1,144 responses (largest difference in any token's probability 0.0011). The candidate order is not neutral: reversing it changes whether 40 of 278 slots
are right (18 become right, 22 wrong). `report.md` also compares the two on every vocabulary and by
term; `outcomes.csv`, `confidence.csv` and `probabilities.csv` have the same layout as Jev's, with
`llm_` in place of `jev_`.

## How the two projects share work

| Shared thing | Owner | How this project uses it |
|---|---|---|
| Corpus Turtle (v2.0.1) and its frozen manifests | `iadopt-lab/data/` | read in place after the lab's own verification; never copied |
| Corpus loading and hash checks | `iadopt_lab.corpus.ingestion.load_canonical_records` | imported |
| Canonical JSON and SHA-256 identities | `iadopt_lab.canonical` | imported (`content_hash`, for the Jev prompt fingerprint) |
| Credential loading | `iadopt_lab.configuration.load_runtime_secrets` | imported by `openrouter.api_key_from`; it keeps only the allowlisted `OPENROUTER_API_KEY` |
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
                           compare, cli; for Jev: jev, openrouter, jev_inputs, jev_health,
                           jev_analysis, jev_report, jev_experiment (the engine both experiments
                           run on); for a PSNC chat model: chat, chat_choice, chat_redact, chat_experiment
src/iadopt_linking_eval/   pure link scorer (standard library only): identity, metrics, paired,
                           confidence
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

A Jev folder (`results/jev-<date>`) holds no new retrieval. It has `responses.jsonl` (every request,
its response text and hash, the served model and the cost OpenRouter reported), `plan.json`,
`provenance.json`, `summary.json`, `report.md`, `comparison.md` (paired counts, counts per term, how
slots moved between outcome categories, and the slots only one of the two scorers got right), and
three CSV files for later threshold analysis: `outcomes.csv` (one row per slot: the
primary pass's selection, probability, margin and confidence beside the pointwise arm and Qwen3),
`confidence.csv` (one row per slot and pass, with every confidence signal and the outcome) and
`probabilities.csv` (one row per candidate and pass: Jev's probability, whether it is the gold and
whether it was selected). Group the CSV rows by `query_key`, not by `term`, when splitting terms into
folds: three terms differ only in case, spacing or hyphen form. `manifest.json` hashes all of it.

## Commands

```bash
uv sync --extra rerank
uv run pytest
uv run ruff check .
uv run iadopt-linking plan
uv run iadopt-linking run --out results/<name>
uv run iadopt-linking compare --first results/a --second results/b
uv run iadopt-linking jev-plan --baseline results/baseline-20261014
uv run iadopt-linking jev-run --baseline results/baseline-20261014 --out results/jev-<date> \
    --max-cost-usd 2
uv run iadopt-linking psnc-plan --baseline results/baseline-20261014
uv run iadopt-linking psnc-run --baseline results/baseline-20261014 \
    --out results/psnc-qwen3.8-27b-<date> --max-requests 1500 --reference results/jev-20261015
```

`jev-plan` builds every Jev request offline, prints the cost estimate and one request of each kind,
and calls nothing. `jev-run` is billed, so it refuses to start without a spending cap and without an
estimate that fits under it. It reads `OPENROUTER_API_KEY` from `../.env` (or `--env-file`) only when
requests are pending and never prints or stores it. It resumes, stops with a named reason on a
refused credential, a passed cap or the first ten requests all failing, and rerunning it in a finished folder
rebuilds the report without a service call. Both commands take `--passes` (a comma-separated subset of
the five passes, which must include the primary `choice-a1`; a folder keeps the list it started with),
`--only` (variable file stems, for a pilot on a few variables) and `--lab-root`; `jev-run` also takes
`--workers` (parallel requests, default 4), `--run-id` and `--env-file`.

`psnc-plan` builds every PSNC request offline and prints the size of the run and one request.
`psnc-run` asks a chat model on PSNC's gateway (`--model`, default `Qwen3.8-27B`, reasoning off) the
same question as Jev about the same candidates. PSNC access is free to the owner, so the run is limited
by a number of requests (`--max-requests`, required) and sends them one at a time by default
(`--workers 1`): identical requests sent concurrently gave different probabilities. It reads
`PSNC_API_KEY` (and optionally `PSNC_API_BASE_URL`, which must be PSNC's https host) from `../.env`
or `--env-file`, only when requests are pending, and never prints or stores it. `--reference` names a
finished Jev run on the same candidates, which the report then compares with; a rebuild needs the same
`--reference`. It takes the same `--passes` (the four Choice passes, which must include `choice-a1`),
`--only`, `--lab-root` and `--run-id` as `jev-run`.

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

The baseline, the Jev run and the PSNC run are done and their evidence is in `results/`. The results
decide nothing yet: no success criterion for replacing the reranker has been chosen. Not yet decided
(see `DECISIONS.md`): that criterion; whether to repeat the Jev run with the client hardening listed in
L-015; which other PSNC models to run (the gateway lists 23, among them `Qwen3.5-397B-A17B`) and
whether to add the one-yes/no-per-candidate arm, which would compare model size with the 0.6B
reranker like for like; whether QUDT's `wikidataMatch` makes a QUDT kind and its Wikidata item equally
correct (it moves Jev's result more than Qwen3's: +24 slots against +8, and it does not cover the OM
twins that the PSNC model often picks), whether the ten slots in other
vocabularies should get a source, how to read the gap to the manuscript's table, whether a response
whose `choice` differs from its highest probability by a rounding step should be accepted, and which
confidence threshold, if any, should send a slot to a larger model, a PSNC-hosted one being the stated
candidate. QUDT accuracy on gold terms is close to a label lookup (59 of 61 corpus terms are QUDT's own
label), so it will not carry over to LLM-predicted wording.
