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
  pilots, superseded runs).
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
  owner's explicit authorization. Pin `jev-1.13.0`, never `jev-latest`, and store the `model`
  field the API returns with every response. Free public services are called at most one request
  at a time per host, at least a second apart.
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
