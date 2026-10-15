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

## Open

- What counts as success for replacing the reranker with Jev (TypeSafe's constrained-output
  model), which evaluation population is primary, and whether to spend on a TypeSafe key
  (Arvin to decide).
- Whether QUDT's `wikidataMatch` should make a QUDT kind and its Wikidata item equally correct.
- Whether the 10 slots in vocabularies other than Wikidata, QUDT and OM should get a source.
- How to read the gap to the manuscript's entity-linking table: its inputs are not in this
  repository, so the population, candidate sets and prompts behind 0.656 and 0.759 cannot be
  compared with this run's (Arvin to decide whether that needs chasing before any claim).
