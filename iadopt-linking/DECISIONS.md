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

## Open

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
  should not be reused for another service.
- Which statistics must be exact receipts. AGENTS.md says metrics are numerator/denominator receipts,
  never a float, while the Brier score, the calibration gap, the mean confidence and the bootstrap
  bounds are floats (their docstrings say so). Either the rule names them as exceptions or the code
  changes.
