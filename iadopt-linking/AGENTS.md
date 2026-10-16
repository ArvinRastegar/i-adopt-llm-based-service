# iadopt-linking

Reproducible experiments measuring how well entity linkers map I-ADOPT slot values to
controlled-vocabulary identifiers. Sibling of `iadopt-lab` (lexical decomposition), which owns
the corpus and is hash-frozen; this project only reads from it. Results here are scientific
evidence, so most rules exist to stop a change quietly invalidating work already paid for.

## Commands

```bash
uv sync --extra rerank   # own environment; never reuse or modify iadopt-lab/.venv
uv run pytest            # includes contract tests against iadopt-lab's parser
uv run ruff check .      # lint; this is the gate
uv run iadopt-linking plan                               # population and queries, calls nothing
uv run iadopt-linking run --out results/<name>           # retrieve, score, check, report; resumable
uv run iadopt-linking compare --first results/a --second results/b
uv run iadopt-linking jev-plan --baseline results/baseline-20261014                  # offline, shows requests and cost
uv run iadopt-linking jev-run --baseline results/baseline-20261014 --out results/jev-<date> --max-cost-usd 2
uv run iadopt-linking psnc-plan --baseline results/baseline-20261014                 # offline, shows one request and the size
uv run iadopt-linking psnc-run --baseline results/baseline-20261014 --out results/psnc-<model>-<date> \
    --max-requests 1500 --reference results/jev-20261015
```

Python 3.12 only, `uv` for everything. Never `pip install` into this project. Until `uv sync` has
been run, the lab's interpreter can run the tests and the experiment read-only (it already holds
torch and sentence-transformers):
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=iadopt-linking/src iadopt-lab/.venv/bin/python -m iadopt_linking.cli ...`
from the repo root. Do not `uv run` anything inside `iadopt-lab`.

## Boundary with iadopt-lab

- **One-way dependency.** This project imports public APIs from `iadopt-lab` (path dependency in
  `pyproject.toml`). `iadopt-lab` never imports anything from here; a test enforces it.
- **Never edit the lab from here.** In `iadopt-lab`, `uv.lock`, `pyproject.toml`, `.python-version`,
  `src/**/*.py`, `schemas/`, `prompts/`, `migrations/` and `data/manifests/` feed the implementation
  hash, so touching one forces a new campaign and about 21,000 paid tasks. From the repo root,
  `git diff --stat -- iadopt-lab` must be empty before you commit.
- **One home per thing, no copies.** Corpus bytes and manifests stay in `iadopt-lab/data/` and are
  read in place after the lab's own verification (`load_canonical_records`). Hashing comes from
  `iadopt_lab.canonical`, the lexical scorer from `iadopt_eval`. If the lab lacks something this
  project needs, write it here; never paste a lab function.
- **Agreement is tested, not assumed.** Where this project re-reads what the lab also reads (the
  Turtle corpus, the legacy reranker prompt), a contract test in `tests/` fails if the two diverge.
- **Shareable evaluation stays standalone.** `iadopt_linking_eval` is standard library only with no
  I/O, like `iadopt_eval`, so the lab can import it later without inheriting this project's
  dependencies. Using it for lab results needs a new decision there (D-018) and a new protocol
  version; it is not a mutation of existing results.

## Results discipline

- `results/<run>/` is committed evidence: `candidates.jsonl` (every retrieval, with the response
  text beside its hash), `scores.jsonl` (every scored slot), `plan.json` (what the folder is
  bound to), `provenance.json` (a journal of every invocation), `summary.json`, `outcomes.csv`,
  `report.md`, `manifest.json` (written last, with the hash of each of those files).
  `replication.md` compares two runs and is not hashed. `outputs/` is gitignored scratch (logs,
  pilots, superseded runs). A Jev folder holds `responses.jsonl` in place of `candidates.jsonl` and
  `scores.jsonl`, plus `comparison.md`, `confidence.csv` and `probabilities.csv` (see below).
- **A results folder is bound to the code that made it.** `plan.json` records a hash of the
  evidence-producing modules (`EVIDENCE_FILES` in `provenance.py`), the variables, queries,
  configuration and corpus; a rerun that differs in any of them is refused, as is a folder with
  evidence but no plan or journal. `tests/test_results.py` fails if committed evidence no longer
  matches this tree or its own manifest hashes. Changing `gold`, `http`, `population`, `recorder`,
  `rerank`, `retrieve`, `score`, `sources` or `iadopt_linking_eval/identity.py` therefore means a
  new run in a new folder. Report-side code (`metrics`, `report`, `compare`, `cli`, `experiment`,
  `health`, `devices`, `provenance`, `artifacts`, `errors`) may change; rerunning `run` in the same
  folder then rebuilds the report from the stored evidence with no retrieval, no scoring, no
  service call and no model load (a passed device check is reused for the same score bytes).
  `test_results.py` recomputes the summary, CSV and report from the stored scores, so a metric or
  wording change shows up there until the report is rebuilt.
- **The manifest describes the evidence, not the last invocation.** `provenance.json` journals each
  invocation (code hash, git base, environment, seconds per phase, what it did, why it stopped), so
  an interrupted and resumed run is described as one; `run` removes an earlier `manifest.json` at
  once, so a folder that is being worked on never claims to be finished.
- Do not edit `src/` or `pyproject.toml` while a run is in progress; each invocation records the
  code it started with, and a mid-run edit would split the run across two code versions.
- Read the report before believing it: recompute the headline independently from `scores.jsonl`,
  and check every denominator against the population it claims.

## The Jev experiment

`jev-run` asks TypeSafe's Jev (`typesafe/jev-1.13`, OpenRouter's Decisions API) to choose among the
candidates a finished baseline folder already holds (DECISIONS L-010 to L-015). The rules that keep it
evidence:

- **Nobody reads the credential.** `OPENROUTER_API_KEY` is read from the repository's `.env` (or
  `--env-file`) by the lab's allowlisted loader, called in `openrouter.api_key_from`; a value in the
  process environment wins over the file. `DecisionsClient` holds the key privately. Never `cat`,
  `grep` or print `.env` (exclude `.env*` from repo-wide searches); to check connectivity run code
  that prints a whitelist of fields. No test can know the real key: one runs the pipeline with a
  made-up key and searches every file it writes, another searches the committed results for key-,
  header- and account-id-shaped text. Error bodies are scrubbed of the key and of the account's
  `user_id` before they are stored, but only in the forms `_scrub` knows (L-015).
- **Do not reuse the client for another service yet.** `openrouter.py` and `jev.py` are evidence
  modules with known gaps that a fix would only close in a new folder and a paid re-run: urllib's
  default redirect handling re-sends `Authorization`, the scrubbing covers few forms, and a cost
  field is parsed twice with different strictness (L-015, Open).
- **A Jev folder.** `responses.jsonl` (every request, its response text and hash, the served model and
  the cost OpenRouter reported), `plan.json`, `provenance.json`, `summary.json`, `report.md`,
  `comparison.md`, `outcomes.csv` (one row per slot), `confidence.csv` (slot and pass),
  `probabilities.csv` (candidate and pass) and `manifest.json`. Group CSV rows by `query_key`, not
  `term`: three terms differ only in case, spacing or hyphen form.
- **Billed calls need a cap, an estimate and the owner's go.** `jev-run` refuses without
  `--max-cost-usd` and without an upper-bound estimate for the pending requests under it. Smoke tests
  and prompt development use invented slots only, never corpus slots.
- **The prompt is frozen.** `jev.CHOICE_INSTRUCTIONS`, `NOUL_INSTRUCTIONS`, `ORDER_SEED` and the passes
  are pinned by a fingerprint test. Changing one is a new experiment in a new folder with a new
  decision; never tune it on corpus slots or on anything seen in a result.
- **Evidence and report-side modules.** A Jev folder is bound to `JEV_EVIDENCE_FILES` in
  `jev_experiment.py`. `jev_health`, `jev_analysis`, `jev_report`, `jev_experiment`, `paired` and
  `confidence` are report-side: change them and rebuild by rerunning `jev-run` in the same folder (no
  service call, no key). `tests/test_jev_results.py` re-derives every published file and fails until
  the report is rebuilt.
- **Pin the model.** Send `typesafe/jev-1.13`, store the snapshot the service reports and fail on a
  second one; `~typesafe/jev-latest` moves.
- **Leave the baseline alone.** Jev reads `results/baseline-20261014` after re-verifying its hashes,
  evidence code, queries and documents, and never writes to it.

## The PSNC chat-model experiment

`psnc-run` asks a chat model served by PSNC (`Qwen3.8-27B`, reasoning off) the same question about the
same frozen candidates in Jev's four candidate orders, and compares it with the baseline and with the
Jev run (DECISIONS L-016, L-017). It runs on the same engine as Jev's: `jev_experiment.Experiment`
describes what differs (requests, how an answer is read, checks, notes, voice), `run_experiment` does
the rest. The rules:

- **Nobody reads the credential.** `PSNC_API_KEY` (and optionally `PSNC_API_BASE_URL`) are read from the
  repository's `.env` by the lab's allowlisted loader in `chat.credentials_from`; the process
  environment wins. `RedactingChatClient` (a `ChatClient` that also redacts what it keeps of a failure,
  `chat_redact.py`) holds the key privately, follows no redirect, ignores proxies, sends it only to an
  https host in `chat.ALLOWED_HOSTS`, and stores no response header (the gateway's carry a backend
  address and the key's running spend). Same rules as Jev for `.env`: never open, grep or print it;
  check connectivity with code that prints a whitelist of fields.
- **Not billed, but limited.** PSNC access is the owner's free access, so `psnc-run` refuses without
  `--max-requests` and without that many requests being enough for what is pending. The limit is per
  invocation and counts requests, not attempts: the client retries a transient failure up to three
  more times, so a failing gateway can see up to four times as many calls.
- **One request at a time.** Identical requests sent concurrently gave different probabilities, ones
  sent one after another did not (L-016), so the default is `--workers 1`; do not raise it for a run
  that is meant to be compared.
- **The prompt is frozen.** `chat_choice.SYSTEM_PROMPT`, `ANSWER_LINE`, the labels, the request
  parameters, `REASONING_OFF` and the thresholds are pinned by a fingerprint test and by a golden
  request hash. A model joins `REASONING_OFF` only after its way of turning reasoning off has been
  measured (the lab's `probe-models`), and a test compares the entry with the lab's `parameters.yml`.
- **Evidence and report-side modules.** A PSNC folder is bound to `PSNC_EVIDENCE_FILES` (Jev's nine
  files and `chat.py`, `chat_choice.py`). Changing any of Jev's nine files therefore also invalidates
  PSNC folders, on purpose: `jev.py` and `jev_inputs.py` decide the slots, candidates and orders both
  experiments share, and `openrouter.py` supplies the request record and status sets the chat client
  uses. `chat_experiment`, `chat_redact`, `jev_health`, `jev_analysis`, `jev_report`,
  `jev_experiment`, `paired` and `confidence` are report-side. `tests/test_chat_results.py` re-derives the committed folder.
- **Probabilities are estimates from 20 tokens.** They come from the first token's top 20
  log-probabilities, summed over a label's spellings and divided by the total over the option labels
  found, so they sum to 1; a label outside the 20 counts as 0. `label_mass` and `top_mass` are not
  stored: recompute them from a response's `response_text` with `chat_choice.parse_chat_choice` (the
  report gives their extremes and the resulting bound on any probability, 0.0019 in the committed run).
- **The reference run is read, not rerun.** `--reference results/jev-<date>` is verified by its
  manifest hashes and baseline identity, must cover exactly the same slots and must sit in the same
  folder as the baseline (the plan names it by folder name and the report phase looks it up there); a
  rebuild needs the same `--reference`. A run without one says so in its report.

## Traps

**The lab's readable view is not the full link gold.** `*.readable.json` and
`parse_variable()["iris"]` keep IRIs only for top-level slots. IRIs of ratio, source/target and
part members exist only in the Turtle; use `iadopt_linking.gold`. Counting from the readable files
undercounts Wikidata links by about a third.

**Gold IRIs are compared by identity, never as strings.** The corpus mixes `https://` and
`http://` forms, the vocabularies' own IRIs are `http://` (Wikidata, QUDT, OM), and two gold IRIs
are malformed. `iadopt_linking_eval.identity.classify` reads them and the run counts every
non-canonical reading. One term can have two gold identifiers (`person` is Q5 in 14 slots and
Q215627 in 12), which caps any system that sees only the term; QUDT's `wikidataMatch` is not yet
treated as an equivalence (open decision).

**A QUDT kind and its Wikidata twin are one concept to Jev.** Given `mass` it may pick Wikidata's
`mass` where the corpus linked QUDT's `Mass`; strict identity counts that as a miss (16 of the 23
slots only Qwen3 got right in `jev-20261015`). Read the equivalence row beside the strict one, and
remember that whether it should count is an open decision.

**Jev's `choice` can differ from its probabilities.** In one near-tie it named an option one
hundredth below another's (1 of 1,431 responses). The validity rule stores such a response as a failure
and asks again; the forced choice never uses `choice`. Its `confidence` is a function of the top
probability and the option count, so analyse the probabilities.

**Slots are not independent.** 153 terms ask 294 slots; `water` alone is 12 slots. A slot-level p-value
or interval is optimistic, and the headline of `jev-20261015` changes when each term counts once.

**A shared inference server answers slightly differently under load.** Identical PSNC requests sent
8 at a time gave four different probability vectors (the probability of `none_of_these` ranged from
0.22 to 0.39 for one near-tied slot); sent one after another they gave one. Prefix caching was not the
cause (`cached_tokens` was always 0). Compare runs through their repeated pass, not by expecting equal
numbers, and never raise `--workers` for a run meant for comparison.

**The PSNC gateway leaks its backend in headers and error texts.** Responses carry `x-litellm-*`
headers (a private backend address, the key's running spend) and failures can name that address;
`RedactingChatClient` stores no header and redacts private and cluster-internal addresses, and
`test_chat_results.py` checks that nothing of the sort reached a committed file. A new client must do
the same. A failing LiteLLM gateway also echoes a masked copy of a rejected key (`sk-...90zz`) and its
hash, `User=` and `Spend=`: `chat.scrub` misses those, `chat_redact` does not.

**A chat model's probabilities come from 20 tokens.** The gateway returns at most 20 log-probabilities
per token, so a pool of 30 options shows 17 to 20 labels; the rest count as 0. What the 20 leave out
(1 - `top_mass`) bounds the error of every probability: at most 0.0019 over the 1,144 responses of the
committed run. They are not lower bounds: they are divided by `label_mass`, so a reported value is up
to 1/`label_mass` (1.05) times its token's own probability.

**The workflow retrieves; it does not download.** Candidates come from each source's query
service at run time, never from a local copy of a vocabulary dump. Every response is recorded
(request, time, attempts, retry causes, status, hash, parsed candidates) so scorers are compared
on identical candidates and a run can be replayed offline; a scorer itself never calls a retrieval
API. A dump may be downloaded for a survey, never as an input to the workflow.

**Wikimedia rejects python-httpx.** Its edge answers httpx with a generic "respect our robot
policy" 403 although curl and urllib succeed with the same user agent, so `http.py` uses urllib.
It also rate-limits with 429 even at one request per second; the client honours `Retry-After`
(capped at 120 s, because the blocks last about a minute) and records every retry cause, the
header and the seconds waited. A full retrieval takes about 15 minutes for that reason.

**The sources have quirks.** QUDT's SPARQL endpoint has no text index (`text:query` returns an
empty solution), so retrieval is a word-containment filter; its labels may carry only `en-US`; it
also serves community-extension kinds. OLS does not expose OM's `om:alternativeLabel` synonyms.
OM's own IRI host fails TLS validation, so never dereference OM IRIs.

**A source with no candidate is a miss, not "not applicable".** Diagnostics that compare with the
headline (source known, first hit, Wikidata only) count a source that returned nothing as wrong;
only the routing share is conditional on a selection and says so.

## Non-negotiables

- **Live calls are gated.** No billed call (Jev or any other) without a recorded estimate and the
  owner's explicit authorization. Pin `typesafe/jev-1.13` (OpenRouter's slug; TypeSafe's own API
  calls it `jev-1.13.0`), never a `latest` alias, and store the `model` field the API returns with
  every response. Free public services are called at most one request at a time per host, at least a
  second apart. The PSNC gateway is not one of them: it is the owner's authenticated access, limited by
  `--max-requests` and used one request at a time.
- **Raw responses are stored before interpretation,** and a failure is stored as a failure, never
  turned into an empty result. A parser refuses a response that contradicts itself (OM `numFound`
  or status, Wikidata `success`, a QUDT row of the wrong shape); a bug in a parser is not recorded
  as a failed retrieval. A run stops, with a named reason, on any failed retrieval, missing or
  impossible score, or failed health check. Checks are tri-state (pass, fail, not run); only the
  device cross-check may be "not run", and never in a committed result.
- **Exact quantities stay exact.** Metrics are numerator/denominator receipts, never a float.
- **Models load from the local cache only** (`HF_HUB_OFFLINE=1`); a missing model is an error,
  not a download.
- **Results from here do not enter the lab's results, rankings or the paper** without an explicit
  decision.

## Conventions

- Every public and private function carries a typed docstring: one-line summary, then terse
  `Args:`, `Returns:`, `Raises:` and `Side effects:` lines, as in the lab.
- Comments explain *why*. A comment restating the code is noise.
- `ruff format` is not enforced; do not reformat files.
- Decisions that change behaviour are recorded in `DECISIONS.md` and superseded by a new entry,
  never edited.
