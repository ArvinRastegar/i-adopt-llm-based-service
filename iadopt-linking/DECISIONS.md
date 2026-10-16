# Decisions

A settled decision is superseded by a new entry, never edited. Entries L-001 to L-009 were made on
2026-10-13 and 2026-10-14 under Arvin's instruction to build and run the baseline without stopping
to ask; each is open to review and reversal by a new entry.

## L-001 — Linking lives in its own project, one-way dependent on iadopt-lab

**Status:** Accepted (Arvin asked for a separate folder with no copied code).

`iadopt-linking/` is a sibling of `iadopt-lab` with its own environment. It imports the lab's public
APIs (`load_canonical_records`, later `iadopt_lab.canonical` and `iadopt_eval`); the lab never imports
it. The lab's implementation hash covers its `uv.lock`, `pyproject.toml`, `src/`, schemas, prompts,
migrations and manifests, so nothing in the lab is edited from here. Where this project re-reads
what the lab reads (the Turtle corpus), a contract test fails if the readings diverge. Evaluation
code that the lab may want later lives in the standalone `iadopt_linking_eval`.

## L-002 — The baseline experiment

**Status:** Accepted (Arvin's design).

For every gold slot whose IRI names a controlled-vocabulary term, search the slot's term in Wikidata,
QUDT quantity kinds and OM quantities (at most 10 candidates each), pool the candidates (at most 30),
score them with the existing Qwen3-Reranker-0.6B using the variable's definition as context, and take
the best score. The question is how well the reranker the project already has performs when it must
also choose the vocabulary.

## L-003 — Retrieval is live and recorded, not downloaded

**Status:** Accepted (Arvin: "the actual workflow of the experiment should retrieve the data and not
download").

Candidates come from the services at run time: Wikidata's `wbsearchentities`, QUDT's public SPARQL
endpoint and EBI OLS4 search (OM restricted to descendants of `om:Quantity`). Every response is
written to `candidates.jsonl` with its request, time, attempts, retry causes, status and a hash, so
scorers are compared on identical candidates and scoring and reporting repeat exactly offline.
Vocabulary dumps may be downloaded for a survey, never as a workflow input.

Client: the standard library, one request at a time per host, at least 1 s between requests to a
host, three attempts at most with backoff, `Retry-After` honoured up to 30 s. httpx is not used:
Wikimedia's edge answers it with a generic "respect our robot policy" 403 although curl and urllib
succeed with the same user agent. The user agent identifies the project but carries no contact
address; add one if Wikimedia asks.

## L-004 — Evaluation protocol

**Status:** Accepted.

- **Population:** every linked node whose gold IRI is a controlled-vocabulary IRI, including ratio,
  source/target and part members (294 slots). Label-only nodes (17) and non-vocabulary IRIs (2) are
  counted and left out.
- **Correct:** the selected candidate's (vocabulary, local identifier) equals the gold's. IRI strings
  are never compared: the vocabularies' own IRIs are `http://`, the corpus mostly writes `https://`,
  and two gold Wikidata IRIs are malformed (`hhttps://`, `/enity/`) and read by pattern. Every such
  reading is counted in the manifest.
- **Reported:** all 294 slots (the stated population; the 10 slots in vocabularies none of the three
  sources can reach count as misses) and the 284 reachable slots, each micro and as the paper's mean
  per variable. Diagnostics separate retrieval (gold in pool), ranking, and routing between sources.
- **Metrics are numerator/denominator receipts;** decimals appear only in the rendered report.
- **No acceptance threshold:** the top-scoring candidate is the answer; a coverage-versus-precision
  table at thresholds 0 to 0.9 is reported instead.

## L-005 — The reranker configuration

**Status:** Accepted.

`tomaarsen/Qwen3-Reranker-0.6B-seq-cls` from the local Hugging Face cache only (`HF_HUB_OFFLINE=1`,
`local_files_only`), the legacy prompt verbatim (a contract test compares it with
`misc/benchmarking_example/phaseOneThreeMerged.py`), query `Definition of "<term>" in context:
"<definition>"`, document `label: "<label>", description: "<description>"`, no source tag. The
description is capped at 300 characters at a word boundary because QUDT's reach 1,500 while
Wikidata's stay near 40. Scoring runs on MPS (45 ms per pair against 334 ms on CPU); each run
rescoring a spread sample of 10 slots on CPU must give the same top-1 and scores within 1e-3, and
rescoring on MPS must reproduce its scores, or the run stops.

## L-006 — Retrieval queries

**Status:** Accepted.

Wikidata: English items, `limit=10`. QUDT: kinds whose English label (`en` or `en-US`) contains every
word of the key, ignoring `of the a an and`, active kinds only, ordered by label length so an exact
label comes first, ten kinds. OM: OLS4 search on the `om` ontology, `allChildrenOf` the OM Quantity
class, ten rows. Queries use the normalised, lower-cased corpus label; plurals and synonyms are not
expanded, so wording that differs from the vocabulary's label can miss.

## L-007 — A results folder is bound to the code that produced its evidence

**Status:** Accepted.

Live retrieval cannot be reproduced later, so the committed evidence (`candidates.jsonl`,
`scores.jsonl`) is the result and must not drift from the code that made it. `plan.json` records a
hash of the evidence-producing modules (`EVIDENCE_FILES`); a rerun with different ones, or for a
different set of variables, is refused. The manifest records that hash, a hash over all source
files, the corpus identity, model revision, device, platform, per-phase timings, the client's
settings, what each service reports about its data (OM version and load time, QUDT graph
modification time; Wikidata has none) and the SHA-256 of every output file. `tests/test_results.py`
fails when a committed folder no longer matches its manifest or this tree. Report-side code may
change, and the report is then rebuilt from the stored evidence by rerunning `run` in the folder.

## L-008 — A source with no candidate is a miss in every comparison

**Status:** Accepted. Found by reading the report of the first run (2026-10-13), which was
discarded and redone.

The diagnostics "top-1 if the source were known", "gold source's own first hit" and "Wikidata-only
pipeline" first left out slots where that source returned nothing, so they read higher than the
headline they sit beside (the paper-style row showed 153/212 where the comparable figure is
153/218). They now count such slots as wrong, over the same population as the headline. Only the
routing share stays conditional on a selection, and its table says so. Also added: which source
wins when the choice is wrong, and accuracy by term length.

## L-009 — Hardening after review: stored responses, a provenance journal, a stronger device check

**Status:** Accepted (2026-10-14). Two independent reviews of the first committed run found that its
evidence and manifest claimed more than they proved. The evidence format changed, so the baseline
was run again; the first run is superseded and kept only in the gitignored `outputs/`.

Supersedes: L-003's `Retry-After` cap of 30 s (now 120 s, because Wikimedia's blocks last about a
minute and a 36 s header was being cut short); L-005's cross-check (10 spread slots, scores within
1e-3, a different top-1 fatal); L-007's manifest, which described only the last invocation; and the
reporting list of L-004 where it called the all-vocabulary mean per variable "the paper's".

- **Evidence.** The text of every 200 response is stored beside its SHA-256 (the first run hashed
  text it did not store, so no hash could be checked). Every attempt records its cause, the
  `Retry-After` it was given and the seconds waited; a non-200 keeps the service's explanation;
  service time is kept apart from wall time. A parser refuses a response that contradicts itself
  or has the wrong inner shape (OM `numFound` or status, Wikidata `success`, a QUDT row), and that
  becomes a recorded failure; a bug in a parser is not recorded as one.
- **The store.** A torn final line is ignored on read and removed on the next append; a complete
  last record without a newline gets one instead of being fused with the next.
- **Scores.** Each record stores the model, revision, device, batch size and description cap that
  made it. Scores are replayed from a twin only for the same variable and the exact query text (the
  first run keyed on the lower-cased term although the query keeps case). Resuming with another cap
  or model revision is refused.
- **Binding.** `plan.json`, written atomically, binds code, variables, queries, configuration and
  corpus; evidence without a plan or a journal is refused, not guessed at.
- **Provenance.** `provenance.json` journals every invocation: code hash and git base, environment,
  client settings, seconds per phase, what it did and why it stopped. Release data of OM and QUDT is
  probed when retrieval ran or none was recorded, with its time; a rebuild makes no service call.
  The manifest is built from the journal and the evidence (evidence period, summed timings, failed
  and retried requests), so a resumed or rebuilt run is described as what it was.
- **Checks** are named, tri-state and required: linked nodes accounted for (counted independently
  of the population loop), every key retrieved from every source, every source answering for at
  least 5% of keys, response hashes verifying, release data recorded, every slot scored, scores in
  range, at most 10 candidates per source, one scoring configuration, device consistency. A failed
  check stops the run before the next phase. "Not run" is allowed only for the device check, and a
  committed result must pass all ten (`tests/test_results.py`, which also recomputes the summary,
  CSV and report from the stored scores).
- **Device check.** The headline depends on near-ties: 53 of 281 selections in the first run had a
  top-two margin below 1e-3 while MPS and CPU differ by about 1e-5. The check now rescores every such
  slot plus 10 others on the other device; scores must agree within 1e-4 and rescoring on the same
  device must reproduce, or the run stops. Selections that change on the other device are reported
  with their effect on the correct count: they are a property of the headline, not a fault. A check
  that rescored nothing fails, and a passed check is reused only for the same score bytes.
- **Corrections.** L-005 said Wikidata descriptions stay near 40 characters; that is their typical
  length, the longest retrieved is 222 (QUDT's longest is 1,544). L-004's all-vocabulary "mean per
  variable" is not the manuscript's number: the manuscript's measure covers Wikidata-gold cases
  only, and the report compares on that population.
- **Report.** Adds distinct terms per population, the manuscript comparison (Wikidata gold, accuracy
  per variable, the reranker alone and the first hit, quoted with the statement that they are not
  reproduced), ceilings under the population they cover, the most repeated terms, margin counts and
  notes computed from this run's numbers (OM candidates carry no description).

## L-010 — Jev is a second scorer, evaluated on the baseline's frozen candidates

**Status:** Accepted (Arvin, 2026-10-15: keep the analysis of the other vocabularies against Wikidata
and the 284 slots to evaluate; use the OpenRouter key in `.env` without showing it to the assistant;
use `typesafe/jev-1.13`; record Jev's confidence for every choice; make it commit-ready).

For each slot, Jev chooses among the same pooled candidates the Qwen3 baseline scored. Nothing is
retrieved again: candidates, documents and queries are read from `results/baseline-20261014` after
its manifest hashes, its evidence-code hash and every query and candidate document have been
re-verified (`jev_inputs.load_baseline`). Only the scorer differs.

- **Model and route.** `typesafe/jev-1.13` through OpenRouter's Decisions API
  (`POST /api/alpha/decisions`), fallback to other providers off. The served snapshot
  (`typesafe/jev-1.13-20260917` when first seen) is stored with every response and must be one
  snapshot per run. TypeSafe released jev-1.14 on 2026-10-14; 1.13 is pinned because it was asked for
  and because thresholds tuned on one version do not carry over (TypeSafe's own advice).
- **Population.** The 286 slots that have candidates are asked (8 have none and are not asked); the
  primary population is the 284 reachable slots; the 10 slots whose gold the sources cannot reach count
  as misses in the "all" row, as in the baseline. The comparison of the three vocabularies stays:
  source confusion, routing and "source known" diagnostics are reported for Jev beside Qwen3.
- **Credential.** `OPENROUTER_API_KEY` is read from the repository's `.env` by the lab's allowlisted
  loader (`load_runtime_secrets`, called in `openrouter.api_key_from`; a value in the process
  environment wins over the file) and held privately by the client. It is not in settings, errors,
  representations, journals or results. The assistant never opened `.env` and the key does not appear
  in the session transcript. Error bodies are scrubbed of the key and of the account's `user_id`
  before they are stored. No test can know the real key, so the tests check the mechanism: a run with
  a made-up key searches every file it writes for that key
  (`test_the_credential_never_reaches_a_file`), and the committed results are searched for key-,
  header- and account-id-shaped text (`test_no_committed_result_holds_a_key_or_an_account_id`).
  One live error response (a deliberately malformed request in the smoke test on invented slots,
  before the `user_id` scrubbing existed) put the account id, not the key, into the session
  transcript; it was copied into a test fixture for a while and is now in no repository file.
- **Spend.** Authorised for this experiment under an explicit cap given to `jev-run` (no cap, no billed
  call). TypeSafe charges $0.042 per million input tokens and nothing for output; the whole run is
  estimated, as an upper bound, at $0.13 (`plan.json` records $0.132).
- **No pass/fail criterion (the assistant's reversible default, not a decision of Arvin's).** None was
  given, so the result is descriptive and nothing in it decides adoption. What would count as
  success or as a reason to adopt Jev (a margin over the baseline, a precision at a coverage, a cost
  per slot) is still open (see Open). What Jev's confidence is worth is recorded for a later
  decision about thresholds and about a larger model (a PSNC-hosted one is the stated candidate); no
  larger model is run here.
- **Sources** (all retrieved 2026-10-15, 09:51 to 09:59 UTC; the pages move, and jev-1.14 is newer).
  TypeSafe: the complete documentation as text, `https://docs.typesafe.ai/llms-full.txt` (index at
  `/llms.txt`), from which the prompting guidance of L-011, the re-ranking pattern, the definition
  of `confidence` and the jev-1.13 jaggedness page (`/model-jaggedness/jev-1.13`) were taken. OpenRouter:
  `https://openrouter.ai/docs/guides/community/jev.md`, `.../jev-tutorial.md`,
  `.../multimodal-decisions.md`, `.../typesafe-sdk.md` and the API reference
  `https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request.md`;
  the model list and the endpoint list of `typesafe/jev-1.13` under `https://openrouter.ai/api/v1/models`.
  The price ($0.042 per million input tokens, output free) is the vendor's; this project's figures use
  the cost OpenRouter reports in each response.
- **Confidence.** Every answer's full distribution and Jev's own `confidence` are stored in
  `responses.jsonl` and flattened into `outcomes.csv`, `confidence.csv` and `probabilities.csv`. Jev
  documents `confidence` as `(p_max - 1/n) / (1 - 1/n)` over its n options, a function of the top
  probability and the option count; the report checks that against every response and keeps the
  probability of the selected candidate as the main signal.

## L-011 — The Jev prompt, and how it was developed

**Status:** Accepted (frozen; fingerprint `a6042e8c2d20`).

*What Jev sees.* The state is the normalised `term` and the variable's `variable_definition`, the two
parts of the baseline's query. One Choice question has one option per pooled candidate, each
`{label, description}` (the two parts of the Qwen3 document, description capped at 300 characters),
plus `none_of_these`. Option ids are neutral (`c01`, ...) and sort in presentation order. No
vocabulary, identifier, rank, retrieval order or score is shown. A second arm asks one yes/no question
per candidate (TypeSafe's re-ranking pattern), each carrying only its own candidate.

*Why this shape.* TypeSafe's guidance: narrow, literal instructions that name state fields in
backticks; criteria for every option; a catch-all option; only the state a question needs; option names
are visible to the model, so they are neutral; the same fields in every structured description. Its
jaggedness page for jev-1.13 says the model reads instructions literally and leans toward the first
option of a Choice, which is why the same pool is asked in several orders (L-012).

*Development.* Draft 0 asked which candidate denotes the concept `term` names "as used in" the
definition. It was tried on five invented slots, with the right answers written down before any
call (`term`: definition, then the right answer):

- `cello`: "Number of cellos in the string section of an orchestra." The instrument (not a village, a
  footballer or the shoulder cello).
- `violin`: "Price of a violin in euros." The instrument (not a family name, a concerto or the viola).
- `bridge`: "Length of a suspension bridge in metres." The structure (not the card game, the dental
  prosthesis or the narrower "suspension bridge").
- `length`: the same definition. "length" in Wikidata, QUDT or OM (not "Bridge Length").
- `croissant`: "Mass of one croissant in grams." `none_of_these`: the pool holds a village, a moon
  phase, "Mass Density" and "Angle".

Each pool also held a whole-variable distractor such as "Cello Count" or "Bridge Length". The script
lived in the session scratchpad and is not kept; the slots and answers are recorded here instead.
Draft 0 was right in 8 of 15 Choice runs (three orders each) and 1 of 5 pointwise runs: it kept
choosing the quantity of the whole variable ("Cello Count", "Bridge Length").
Draft 1 says what the pieces are: `term` is a component of the variable, the definition is used only to
tell which meaning of `term` is meant, and a concept for the whole variable does not match. It was
right in 15 of 15 and 4 of 5 (the miss: "suspension bridge" 0.55 against "bridge" 0.50). Draft 1 is
frozen and was not iterated further. The five slots were written around the failure being fixed, so
15/15 says the wording does what it says, not how well it will do on the corpus. No corpus slot was
sent to Jev before this entry; the invented slots cost well under $0.01.

Changing an instruction, the order seed or the passes changes the fingerprint (a test pins it) and is a
new experiment in a new results folder.

## L-012 — Pre-specified analysis

**Status:** Accepted. Written on 2026-10-15 between 10:29 and 10:31 UTC (the times of the edits to
this file) and before the first request that names a corpus slot, which `responses.jsonl` stamps
10:33:21 UTC; the same plan is stored in `plan.json` of the results folder when the first request is
sent. "Pre-specified" and not "pre-registered": nothing was filed with an outside registry and no
commit predates the run, so the dated records are this file's edit times and `plan.json`, both
written by the session that ran the experiment. A *term* below is the slot's `query_key` (the term
normalised to NFKC with dashes unified, lower-cased), the key the baseline groups its retrieval by.

- **Primary result:** pass `choice-a1` (one Choice per slot, candidates in a seeded order), forced
  choice (the candidate with the highest probability, `none_of_these` ignored, ties by the baseline's
  rule: source order, then rank), strict identity (vocabulary and local identifier), on the 284
  reachable slots.
- **Estimand and interval:** top-1 accuracy of Jev minus the Qwen3 baseline on the same slots and
  candidates; percentile bootstrap over terms (10,000 resamples, seed 20261015, 95%). Exact McNemar on
  slots is reported as descriptive only: 153 terms ask 294 slots, so slots are not independent.
- **Everything else is exploratory:** the other passes, the consensus of the Choice passes, Jev's own
  refusals, position bias, subpopulations by gold vocabulary, the QUDT/Wikidata equivalence (reported
  beside the strict number, which stays the headline; whether it should count is still open), and every
  confidence analysis (AUROC, calibration, coverage and precision, escalation). Signals: the selected
  candidate's probability, Jev's `confidence`, the top-two margin, the probability not given to
  `none_of_these`, and the share of the four Choice passes that agree. Thresholds are read off the data
  they describe, so a chosen threshold needs held-out terms.
- **Passes.** Candidate order is a hash of (seed, slot, candidate identity), never of score, gold or
  retrieval rank. `choice-a1` and `choice-a1-repeat` send byte-identical requests, so their difference
  is Jev's own variability; `choice-a2-reversed` is `choice-a1` backwards; `choice-a3` is a second
  independent shuffle; `noul-n1` is the pointwise arm.
- **No gate:** nothing in the result decides the adoption of Jev.

## L-013 — Run protocol and safeguards

**Status:** Accepted.

- A Jev results folder (`results/jev-<date>`) is bound by `plan.json` to a hash of `JEV_EVIDENCE_FILES`
  (the modules that decide what is asked and how it is read), the baseline's file hashes, the prompt
  fingerprint, the passes, the model pin and this analysis plan; a rerun that differs in any is refused.
  Report-side modules may change and the report be rebuilt without a service call.
- Every response is stored with its SHA-256 before it is judged. A response that is not a valid answer
  (exact option set, finite probabilities in [0, 1] summing to 1 within 0.05, a `choice` that has the
  highest probability, a `confidence`) is stored as a failure. Stored responses that no longer match
  their hash stop the run before anything is asked.
- Three stop rules, applied by the workers under a lock so they do not depend on timing: the service
  refusing the credential or the account (401, 402, 403), total spend passing the cap, and the first ten
  requests all failing. An upper-bound estimate for the pending requests must fit under the cap.
- A committed result passes every check: inputs identical to the baseline, every request answered,
  stored requests regenerate from the plan, stored response hashes verify, every response is a valid
  answer, one model snapshot served, spend fully accounted and within the cap, primary pass complete.
- `tests/test_results.py` applies its baseline checks only to baseline folders; Jev folders are checked
  by `tests/test_jev_results.py`.
- A bug found after the run in an evidence module means a new folder; paid responses are never edited.

## L-014 — The Jev run and what changed in its analysis after the first results

**Status:** Accepted (2026-10-15). Results: `results/jev-20261015`.

*Run history.* The first invocation (10:33:21 to 10:35:10 UTC) sent 1,430 requests with 4 workers and
stored 1,430 responses. One was rejected by the validity rule of L-013: Jev named `c01` as its choice
with probability 0.33 while its own probabilities gave `c05` 0.34 (its documented `confidence` matches
the larger one). It was stored as a failure, the run stopped with a named reason, and the second
invocation (10:35:21) asked that one request again, as the protocol says. The rule was not changed
after seeing this: 1 of 1,431 stored responses is a rate low enough to leave the pre-specified rule
alone, and the rejected response stays in `responses.jsonl` and its cost is counted. Total spend
$0.0708 as reported by OpenRouter over the 1,431 stored responses, against an upper-bound estimate of
$0.132 and a cap of $2. Five later invocations rebuilt the report from the stored responses with
revised analysis code and made no service call; the evidence-code hash is the same in all seven
entries of the journal. The first entry has no `asking` counters because that field was added later;
what it sent is in `responses.jsonl`.

*What the first report said, and what changed.* The report built at 10:35:21 UTC, before any of the
additions below, showed the same primary figures as the final one (213/284 against 194/284, +0.067,
term-resampled interval -0.040 to +0.182, exact McNemar p 0.0248, equivalence view +0.123). It differed
in wording ("pre-registered", "billed by OpenRouter") and in its cost ($0.070775586, which left out
the rejected response). That report and the earlier versions of the analysis code were not kept; the
folder holds only the final report, which is rebuilt from the stored responses and checked by
`tests/test_jev_results.py`.

*Added after the first results (post hoc and exploratory; the report labels each one).*
- A comparison counted once per term (sign test over terms and the unweighted mean difference),
  because the slot-level interval and McNemar's p treat 153 terms asking 294 slots as independent
  evidence. It changes how the headline reads (see the report), which is why it is listed here.
- The escalation table separates errors a better scorer could fix (the gold is among the candidates)
  from errors no scorer could fix; its first version counted both in its ceiling.
- The Wikidata-only diagnostic row, a count of disagreements that are QUDT/Wikidata twins of the same
  concept, and the check of Jev's `confidence` against its documented formula (announced in L-010).
- Per-slot order effects (how many selections each pass changes against the primary pass and in which
  direction), the abstention table restricted to the 284 reachable slots, exact decimal margins
  between passes instead of float differences, and the `query_key` column in the CSV files.
- Report wording: an unsupported claim that equal probabilities are "common" (1 of the 281 primary
  selections that have a runner-up is tied) was removed, hard-coded term counts and pass lists became
  computed, "pre-registered" became "pre-specified", "billed" became "reported", and the rejected
  response is disclosed in the health table and the notes.

The pre-specified primary result and analyses (L-012) were not changed.

## L-015 — Review of the Jev work: what was fixed, what was left, what stays open

**Status:** Accepted (2026-10-15, before the first commit).

Three read-only reviewers (correctness; truth, secrecy and request fit; cleanup and conventions) read
the uncommitted work after the run. An earlier attempt with eleven narrower reviewers ended without
findings because the service limit was reached, and was redone. Their 32 findings overlap; each was
checked against the code before anything was changed. A review is critique, not proof.

*Fixed before the commit* (report-side code, tests and documents; the report was rebuilt from the
stored responses with no service call, and the evidence-code hash is unchanged):
- A paid response is stored by the worker, under the lock, before it is judged; any exception while
  judging becomes a stored failure, and reading a cost cannot raise (a body nested too deeply, or a
  number of 5,000 digits, is simply unpriced); spend is totalled over every stored response,
  including the rejected one, whose cost the first report had left out.
- The journal records what each invocation asked and why it stopped; `--passes` is validated before
  any request (no duplicates, the primary pass present).
- Abstention figures are limited to the 284 reachable slots; margins and probability mass are exact
  decimals, not float differences; agreement and repeatability are exact receipts; the evidence
  period is taken from every stored record.
- The report states what the run did (pass list, description cap, which row holds the unreachable
  slots) instead of fixed text; "pre-registered" became "pre-specified" and "billed" became
  "reported"; post hoc analyses are labelled; the CSV files carry `query_key`, the term key the
  statistics group by; the confidence tables say that `jev_confidence` is Jev's confidence in its
  own choice, which was `none_of_these` for 18 of 278 asked reachable slots.
- `CHECK_INPUTS` can fail (it compares with the baseline's slot and record counts), `CHECK_ANSWERED`
  counts parsed answers, the Jev result suite fails when no Jev result exists, a test requires every
  `manifest.json` to belong to a suite, the committed results are searched for key-, header- and
  account-id-shaped text, and loopback tests exercise the real transport.
- A test fixture that held a real account id was replaced by an invented one. L-010's cost estimate
  ($0.11, now $0.13 as `plan.json` records it) and L-012's time of writing (first given as after the
  run began) were corrected.

*Left unchanged because the fix is in an evidence module.* Changing one means a new results folder and
a paid run of about $0.07 (L-013). None of it changes the numbers in the report, but the client should
not be reused for another service or run before the first is closed.
- `openrouter.urllib_post` uses urllib's default redirect handling, which re-sends the
  `Authorization` header to the target of a 301, 302 or 303. All 1,431 stored responses are
  well-formed answers that name the expected model, which a redirect to another service would not
  produce, but the exposure exists in the code.
- `_scrub` removes the exact key and a literal `"user_id": "..."` pair. An id inside an escaped JSON
  string, under another key name, inside a message, or a masked echo of the key would be stored as
  received. No stored response is an error body.
- The message for a probability sum below 1 prints `1.0 + drift`, so it reads as a sum above 1. No
  stored record carries it; the one rejection is the highest-probability rule.
- The cost field is read twice with different strictness (`jev._envelope_fields` rejects a malformed
  cost, `jev_health.reported_cost` returns none), so a malformed cost on a valid answer would reject
  it and leave it unpriced.
- `SCORER_VERSION` and `iadopt_linking_eval.metrics`, which decide the headline, are report-side and
  not bound by `plan.json`. `tests/test_jev_results.py` re-derives the published files, so a change
  there fails until the report is rebuilt, but the folder does not refuse it.

*Looked at and kept.* The rule that rejects a response whose `choice` is not its highest probability
(one response in 1,431, in a non-primary pass; Open). The consensus and agreement signals that count
`choice-a1-repeat` as a pass, although its request is byte-identical to the primary pass (the report
says so; exploratory). Six of the nine `JEV_EVIDENCE_FILES` repeat the baseline's list, which costs
nothing. The Jev folder, paths, plan-binding and journal helpers stay near-copies of the baseline's
because merging them would change the baseline's report-side modules; only identical code was reused.

## L-016 — A PSNC chat model on the same candidates: what it is asked and how it was developed

**Status:** Accepted (Arvin, 2026-10-15). He asked to run the experiment against `Qwen3.8-27B` from
PSNC with reasoning off, to see whether it performs better using only models available from PSNC, and
wrote: "I think you can use the same workflow by just changing the calls to psnc provider with the new
model". The design below is the assistant's reading of that; it was not put to him before the run
(L-019 lists what he may want to change).

*Scope.* One model, `Qwen3.8-27B` on PSNC's gateway (`https://llm.hpc.psnc.pl/v1`, LiteLLM in front of
vLLM), asked about exactly the candidates the Qwen3-0.6B baseline scored, read from
`results/baseline-20261014` after the same re-verification as in L-010, and compared with the
baseline and with the finished Jev run. Other PSNC models (the gateway lists 23 on 2026-10-15,
including `Qwen3.5-397B-A17B`, `GLM-5.2` and `DeepSeek-V4-Flash`) are not run. Reasoning is off:
`chat_template_kwargs.enable_thinking=false`, the control the lab measured (`parameters.yml`); a test
compares the request with the lab's file. Reasoning on was excluded by the lab (its D-044: it does not
fit the 120 s timeout).

*Why this is not only a change of calls.* Jev's API returns a probability for every option; a chat
model returns text. So (1) the options are named by one-character labels (`1`-`9`, then the capital
letters without `I` and `O`, 33 in all; a pool holds at most 30; `0` is "none of these"), so that
each option is one token; (2) the model is asked to reply with the label alone, one token is
generated at temperature 0; (3) the probability of each option is read from the log-probabilities of
that first token (the gateway reports at most 20), summed over the label's spellings and normalised
over the labels found, with 0 for a label not reported, so a probability is the model's probability
of the label given that its first token is a label, estimated from the reported tokens (L-019 says how
far it can be off); the share on labels (`label_mass`) and on all reported tokens (`top_mass`) are
computed when a response is read, not stored per response (the report gives their extremes). No
confidence is asked for in words. The question is Jev's (L-011), the candidates are shown as
`"label": description` (description capped at 300 characters as for the baseline), and the candidate
orders are Jev's (`jev.presentation_order` with Jev's seed; a test compares both builders slot by
slot).

*A response is refused* (stored as a failure and asked again on the next invocation) if it names no
model, finished abnormally, carries reasoning text, has no log-probabilities for the first token,
puts less than 0.5 of that token's probability on option labels, starts with something that is not a
label, or whose own label is not the most probable of the labels found. Nothing is repaired. A
response that fails again is not scored as a miss behind the owner's back: the run does not finish
and the owner is told (the prompt is not changed).

*Listwise only.* Jev's pointwise arm (one yes/no per candidate) would need one request per candidate
here (3,473 for the 294 slots), not one per slot. It is left out; it is the like-for-like test of
model size against the 0.6B reranker, which scores each candidate alone, and can be added.

*The engine.* The Jev engine was generalised so that one run function serves both scorers:
`jev_experiment.Experiment` describes what differs (the requests, how an answer is read, the checks and
notes, the wording, whether a request costs money) and `run_experiment` does the rest; Jev is the `JEV`
instance and `psnc-run` the `chat_experiment()` one. Jev's analysis and report modules gained a `key`
and a `Voice` (internal names `jev_*` became `own_*`). Five source modules changed, none of them
evidence code of Jev's folder, and that folder is unchanged: `git diff` of `results/jev-20261015` is
empty and `tests/test_jev_results.py` re-derives it and reruns it in a copy, which reproduces every
data file byte for byte. This was the assistant's way of honouring "the same workflow".

*The client* (`chat.py`) closes the gaps listed for Jev's in L-015 for this endpoint: it follows no
redirect and ignores proxies, sends the key only to an allowlisted https host, scrubs the key,
bearer tokens, key-shaped strings, account ids in the JSON forms of an error body and private
(RFC 1918) addresses, and stores no response header (the gateway's carry a backend address and the
key's running spend); L-019 widens the redaction to the other forms a LiteLLM gateway uses. The key (`PSNC_API_KEY`, optionally `PSNC_API_BASE_URL`) is read only by the lab's allowlisted
loader, as in L-010; the assistant never opened `.env`.

*Spend.* PSNC access is the owner's free access (the lab records it as `non_billed`), so a run is
limited by a number of requests, not by money: 1,144 are planned (4 passes of 286) and the limit is
set when the run starts.

*Development.* On invented slots only, with the right answers written down first and the acceptance
criteria set before the first call: every response valid, the forced choice right on at least 90% of
the answerable runs, at least 0.8 of the first token's probability on labels in every run. The 23
slots are the five of L-011 (`bridge`, `cello`, `violin`, `length`, `croissant`) and `guitar`,
`lighthouse`, `glacier`, `temperature`, `forest`, `Plants` (plural and capital), `wavelength`,
`torque`, `serendipity coefficient` (right answer: none), `oak` with only a broader and a narrower
entry (none), `oak` with the genus present, `bank` (river bank, among a financial bank and others),
`car` (right: the entry named "automobile"), `age`, `DNA`, `pentane (C5H12)`, `sheep` (with a "Sheep
Count" distractor) and `frequency` with a pool of 30 near-duplicates. None of the 23 terms is among
the corpus's 153 query terms. Each was asked in the three orders (69 runs) with the first draft of
the prompt: all 69 responses were valid; the forced choice was right on 60 of 60 answerable runs; the
model's own choice was right on 67 of 69 (both misses a near-tie between the right candidate and
`none_of_these`, 0.394 against 0.394 and 0.245 against 0.216); the share on labels was at least
0.9976 (median 0.9996); with 30 candidates 17 or 18 labels were reported and at least 0.9991 of the
probability was on labels. A second draft with three invented worked examples was written but not
run, because the first met the criteria. The prompt is frozen (fingerprint `20d69e9a52e2`, pinned by a
test). The slots were not written around a failure, so this says the prompt works, not how well it
will do on the corpus.

*What development showed about the service.* Identical requests sent one after another gave identical
probabilities (30 in a row, three slots); 40 identical requests sent 8 at a time gave four different
probability vectors (the probability of `none_of_these` ranged from 0.22 to 0.39 for one near-tied
slot), all from one backend, with `cached_tokens` always 0. So the answer depends slightly on what
else the server is processing, and the run sends one request at a time. Another run, or the same one
with other traffic on the server, will not reproduce the probabilities exactly; the repeated pass
measures it.

*Sources.* The lab's `docs/model-catalog.md`, `docs/components/providers.md` and `parameters.yml`
(PSNC base URL, the measured reasoning control, the free-access basis, concurrency and timeout), the
gateway's own `/models` list and response bodies, 2026-10-15. Probing used invented text only.

## L-017 — The PSNC run: pre-specified analysis and protocol

**Status:** Accepted. Written on 2026-10-15 between 18:27 and 18:29 UTC (the times of the edits to
this file), before the first request that names a corpus slot; the structured part is stored in
`plan.json` when that request is sent. Pre-specified,
not pre-registered: nothing was filed outside this repository and no commit predates the run (L-012
explains the difference). A *term* is the slot's `query_key`, as in L-012.

- **Primary result:** pass `choice-a1` (candidates in Jev's first seeded order), forced choice (the
  candidate with the highest probability, `none_of_these` ignored, ties by the baseline's rule),
  strict identity, on the 284 reachable slots. Estimand: top-1 accuracy of the model minus the
  Qwen3-0.6B baseline's on the same slots and candidates; percentile bootstrap over terms (10,000
  resamples, seed 20261015, 95%) and exact McNemar on slots as a descriptive test.
- **Secondary result (pre-specified):** the same comparison against the primary pass of the finished
  Jev run `jev-20261015`, from its published `outcomes.csv` after its manifest hashes and its baseline
  identity have been verified, on every gold vocabulary, with the term-level comparison. The plan
  stores that run's identity: the hashes of its responses, its plan and its per-slot outcomes (not of
  its manifest, which a rebuilt report changes).
- **Pass or fail:** no criterion (the assistant's default, as in L-010; what would count as
  "performs better" or as a reason to adopt a scorer is still open).
- **Everything else is exploratory:** the other passes, the consensus of the passes, the model's own
  `none_of_these`, position bias, the confidence signals (the probability of the selected candidate,
  of its own choice, the top-two margin, the probability not given to `none_of_these`, the share of
  passes that agree), thresholds and escalation, the QUDT/Wikidata equivalence, subpopulations.
  Thresholds are read off the data they describe and need held-out terms.
- **Passes:** `choice-a1`, `choice-a1-repeat` (a byte-identical request), `choice-a2-reversed`,
  `choice-a3`: Jev's four Choice passes.
- **Protocol:** `psnc-run`, one request at a time, a request limit of 1,500 for the invocation (1,144
  planned, the rest for asking failures again; a request counts once however often the client retries
  it, which it does up to three more times after a transient failure), stop rules applied by the
  workers (the service
  refusing the credential, the limit, the first ten requests all failing), responses stored with their
  SHA-256 before they are judged. The folder is bound by `plan.json` to the evidence code
  (`PSNC_EVIDENCE_FILES`: Jev's nine files and the two chat modules), the baseline, the prompt, the
  passes, the analysis plan and the reference run; report-side modules may change and the report be
  rebuilt without a service call.
- **A committed result passes every check:** inputs identical to the baseline, every request
  answered, stored requests regenerate from the plan, stored response hashes verify, every response a
  valid answer, one served model and one server build, token usage fully accounted, primary pass
  complete.
- **After the results:** whatever is added is labelled post hoc and listed in a later entry. A bug found
  in an evidence module afterwards means a new folder; stored responses are never edited.
- **Deviation (decided before the second run, which started at 18:35 UTC).** The run described above
  was started at 18:28 UTC and finished cleanly (1,144 of 1,144 requests answered, exit code 0). Before
  any accuracy figure of it was read (only validity, label mass and latency had been looked at), a
  flaw in this harness was found: the plan pinned the reference run by the hash of its
  `manifest.json`, which changes whenever that run's report is merely rebuilt, so a later rebuild of
  either folder would have been refused. The pin was changed to the reference's evidence (responses,
  plan, outcomes) and the run repeated in a fresh folder with the same prompt, the same evidence
  code and the same plan otherwise. The first run's folder was moved to `outputs/` (gitignored) as
  a shakedown and its report read afterwards, to check that the report renders on real data; its
  headline is reported next to the second run's in the entry that holds the results, as a
  replicate, not replaced by it. The decision to repeat did not depend on any accuracy figure.

## L-018 — The PSNC run: what it shows and what it does not

**Status:** Accepted (2026-10-15). Results: `results/psnc-qwen3.8-27b-20261015`, the second complete
run of L-017's deviation note.

*Runs.* Two complete runs of 1,144 requests each (4 passes of 286 slots), one request at a time, 18:28
to 18:34 UTC (run 1, the shakedown, in `outputs/`, gitignored) and 18:35 to 18:41 UTC (run 2, the
committed one). In each: no request failed, no response was refused, 342 and 341 s of request time (0.30 s
on average; the slowest request took 0.95 s and 1.47 s; run 1 first in each pair), 565,176 input and 1,144 output tokens, served
as `Qwen3.8-27B` by server build `vllm-0.26.0-1c0daee9`, not billed. Both runs give 206/284 on the primary pass and 206, 202 and 201
on the other three; 1,143 of the 1,144 responses have identical top-20 log-probabilities in the two runs,
the largest difference in any token's probability is 0.0011 and no selection changed. So the headline is
that of both runs, and run 1 is a replicate, not a discarded run. The committed folder's report was
rebuilt twice after run 2 (no service call) to correct wording. First: the analyses that Jev's report
gained after its first results (L-014) are labelled "exploratory" here, not "post hoc", because this
run's code had them from the start, and two notes say what the data show (whether reasoning text
occurred, how the repeated pass compared). Second, after the review of L-019: which readings are
exploratory, what a probability is, how the requests were sent. No number changed.

*Pre-specified results.* Primary: the first pass gets 206 of the 284 reachable slots right (0.725),
Qwen3-0.6B 194 (0.683): +0.042, term-resampled 95% interval -0.073 to +0.172, exact McNemar p 0.188
(descriptive); counted once per term the model is better on 19 of 145 terms and worse on 16. Secondary:
against Jev's 213 (0.750) it is -0.025 (-0.082 to +0.034; p 0.349), better on 13 terms and worse on 12;
by gold vocabulary against Jev it is -0.023 on Wikidata and -0.033 on QUDT, neither distinguishable
from zero. So the data do not show that this model performs better than the 0.6B reranker, nor worse
than Jev.

*Exploratory results (L-017 lists them).* Every interval of this run that excludes zero is exploratory.
By gold vocabulary against the baseline, Wikidata 169/218 against 138/218 (+0.142, +0.008 to +0.300; 17
terms better, 2 worse) and QUDT 32/61 against 51/61 (-0.311, -0.522 to -0.138; 2 better, 14 worse): the
pattern Jev has. On QUDT gold the model picks the Wikidata item for 16 slots and the OM item for 13:
QUDT's `wikidataMatch` equivalence covers 11 of the 16 and none of the 13, so counting twins as one
entity gives the model 218/284 against Jev's 237/284 (-0.067, -0.127 to -0.007), a gap that comes from
which twins the equivalence covers. It gets `water` (12 slots) and `urban area` (11) right in
every slot, where the baseline gets none and one, and `person` wrong in 20 of 26 against the baseline's
14, where the gold names two items (Q5 in 14 slots, Q215627 in 12). The probability of the selected
candidate has AUROC 0.86 (Jev 0.81, Qwen3 score 0.59), calibration gap 0.080 (Jev 0.045) and Brier score
0.136; answering only at 0.9 or more covers 135 slots at 131/135 = 97.0%, at 0.8 or more 157 at 96.2%.
The model chose `none_of_these` for 17 reachable slots (5 at 0.9 or more), 15 of them with the gold
among the candidates. The identical repeat selected the same candidate and gave identical probabilities
for 286 of 286 slots; reversing the order changes whether 40 of 278 slots are right (18 become right, 22
wrong), another shuffle 21 (8 and 13), and the four passes select the same candidate on 207 of 278.

*Checks.* The run passed every health check; `tests/test_chat_results.py` re-derives the published
files from the stored responses, reads the raw responses and the baseline's and Jev's per-slot outcomes
with no project code (the probability and choice rules of L-016 and L-017) to the same pass totals and
the same paired counts against the baseline and against Jev, and reruns the folder in a copy, as the
documented rebuild does, to the same data files; nothing secret or internal was published (the stored
text is a fixed point of the scrubber, no header or address).

*What is not concluded.* Nothing about other PSNC models, about the one-yes/no-per-candidate
formulation, or about an adoption rule (none was given). The prompt was developed on easy invented
slots, a probability is estimated from at most 20 log-probabilities per token (L-019), the pool and gold are
the baseline's, and the equivalence question decides who leads on QUDT slots.

## L-019 — Review of the PSNC work: what was corrected and what stays

**Status:** Accepted (2026-10-16). An independent read-only review of L-016 to L-018 (three reviewers:
correctness, truth and secrecy, cleanup and tests; the first attempt stopped on a usage limit and was
repeated) found no defect in a published number: about 230 quoted figures were recomputed with separate
code and matched, Jev's folder reruns in a copy to the same data files, and no key, account id, header
or spend figure is in any file. It found the following, corrected without a service call and without
touching an evidence module (`PSNC_EVIDENCE_FILES`), so `results/psnc-qwen3.8-27b-20261015` is still
bound to the code that made it. Its report was rebuilt (L-018).

- *Statements that were false or loose.* A probability is not a lower bound: it is the token's
  probability divided by `label_mass`, so a reported label's value is up to 1/`label_mass` times its
  token probability (1.0513 at most in the run), and the exact label-conditional probability is within
  0.0019 of it (what the 20 tokens leave out, over `label_mass`); a label outside the 20 counts as 0.
  `label_mass` and `top_mass` are not stored per response; they are recomputed from `response_text`.
  With 30 candidates the run reported 17 to 20 labels, not 17 or 18 (that was development). The
  per-vocabulary rows against the baseline and the equivalence reading against Jev, the only intervals
  that exclude zero, were exploratory but sat under "pre-specified"; the report and L-018 now say so.
  Smaller: two request times transposed, 153 query terms (not 154), the owner's words altered inside
  quotation marks, the replicate placed after the committed run (it ran first), the engine refactor
  missing from this log, and `--max-requests` is a limit per invocation on requests, each retried up to
  three more times. All nine Jev evidence files, `openrouter.py` among them, invalidate a PSNC folder
  when changed, not only `jev.py` and `jev_inputs.py`.
- *Behaviour.* A reference run outside the baseline's folder passed planning, cost every request, then
  failed in the report phase; it is now refused at planning. The usage check's token total skipped
  rejected responses (none occurred), the note on the service hard-coded how the run was sent, a repeat
  difference under 0.005 printed as 0.00, and a misspelled plan field would have skipped its check.
- *Redaction (a gap; nothing leaked).* `chat.scrub` did not cover a masked key (`sk-...90zz`), the hash
  of a rejected key, `User=`, `Spend=`/`Budget=`, a Python-dict `'user_id'`, loopback, link-local and
  carrier-grade addresses or cluster-internal names, all forms a LiteLLM gateway uses; the committed run
  stored no error text. `chat_redact.RedactingChatClient`, which a real run now builds, redacts them
  before an error is cut. A test had copied the backend address the gateway names in a response header;
  an invented one replaces it.
- *Tests.* Two that could not fail (scrubbing before the cut, proxies from the environment), one that
  stopped on the circuit breaker and never reached the usage check, and tautologies were replaced; the
  eight checks, the plan fields `corpus` and `run_id`, the forwarding of every `psnc-run` option, the
  reference's location and identity, the notes and the documented rebuild of both committed folders
  gained tests. The summary test compares floats to a relative 1e-9, because they derive from `exp()`
  and another platform's C library may round it differently. 22 one-line mutations are each caught.
- *Left in evidence modules (a fix means a new folder).* `chat_choice.py` still calls the probabilities
  lower bounds in a docstring, and `chat.py` promises "account ids in any of the forms"; `estimate`,
  `_require_permutation` and `ChatClient.post` copy their Jev counterparts; the validity rule compares
  the sum over a label's spellings with the others', so a response whose own token loses to another
  label's sum would be refused every time (none of 1,144 was; the run would stop loudly). In report-side
  code, about fourteen defaults (`exp or JEV`, `key="jev"`, `voice=None`) make a forgotten argument fall
  back to Jev's parser or names instead of failing; the PSNC tests would show it, a new scorer's might not.
- *Process.* The design departed from "just changing the calls" (labels and log-probabilities, listwise
  only, the engine refactor) and a second complete run of 1,144 requests was made as a shakedown;
  neither was put to the owner first. The decisions below are his.

## Open

- Which other PSNC models to run on the same candidates (the gateway lists 23 on 2026-10-15, among
  them `Qwen3.5-397B-A17B`, `GLM-5.2` and `DeepSeek-V4-Flash`), each needing its reasoning control
  measured first, and whether to add the one-yes/no-per-candidate arm (3,473 requests), which would
  compare model size with the 0.6B reranker like for like.
- Whether QUDT's OM twins (`Mass`, `Height`, `Time` exist in both vocabularies) should count as the same
  entity as the QUDT kind, as `wikidataMatch` makes the Wikidata items count (the PSNC model picks them
  for 13 QUDT-gold slots).
- What counts as success for replacing the reranker with Jev (the first item of the earlier list,
  which Arvin's request only partly settled: it fixed the 284-slot population and the OpenRouter
  route, not a criterion). L-010 and L-012 record "no criterion" as the assistant's default, so the
  Jev results decide nothing until a margin, a precision at a coverage or a cost per slot is chosen.
- Whether QUDT's `wikidataMatch` should make a QUDT kind and its Wikidata item equally correct. The
  Jev report shows both and keeps strict identity as the headline.
- Whether the 10 slots in vocabularies other than Wikidata, QUDT and OM should get a source. They are
  outside the primary population (the 284 reachable slots).
- How to read the gap to the manuscript's entity-linking table: its inputs are not in this
  repository, so the population, candidate sets and prompts behind 0.656 and 0.759 cannot be compared
  with this run's (Arvin to decide whether that needs chasing before any claim).
- Whether a response whose `choice` differs from its highest probability by a rounding step should be
  accepted instead of asked again (1 of 1,431 responses; the rule is unchanged).
- Which confidence threshold, if any, should send a slot to a larger model, and which model (a
  PSNC-hosted one is the stated candidate). The Jev folder records the evidence; a threshold needs
  held-out terms.
- Whether to repeat the Jev run with the evidence-module fixes of L-015 (a no-redirect transport,
  wider scrubbing, a corrected sum message, one cost parser). It would be a replicate in a new folder
  (about $0.07 and two minutes) reported beside this one: Jev's probabilities vary from run to run,
  so its figures would differ slightly and neither run replaces the other. Until then the client
  should not be reused for another service. Changing any of Jev's nine evidence files (`openrouter.py`
  among them) also ends the binding of the PSNC folder, so that fix would mean running the PSNC
  experiment again as well (free, about six minutes).
- Which statistics must be exact receipts. AGENTS.md says metrics are numerator/denominator receipts,
  never a float, while the Brier score, the calibration gap, the mean confidence and the bootstrap
  bounds are floats (their docstrings say so). Either the rule names them as exceptions or the code
  changes.
