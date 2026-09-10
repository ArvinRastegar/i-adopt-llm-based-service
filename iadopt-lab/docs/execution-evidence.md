# Executable preparation and live gates

The entry point delegates to separate modules: configuration validates parameters,
corpus verifies/imports TTL, planning expands the owned grids, artifacts freezes
inputs and code, costing estimates cost, workflow advances tasks, reporting
calculates and exports results, and persistence owns PostgreSQL transactions.
The synthetic dry run uses three real corpus targets with **synthetic responses**,
an equality-only similarity fixture and no network. Its scores are not scientific results.

## Bundle and identities

Preparation takes the validated parameters, all 102 verified canonical records,
the five ordered demonstrations, the exact population and an explicit similarity
identity. It returns a portable bundle containing the original selected configuration,
full expanded plan, corpus/demo evidence, implementation file-hash index and exact
runtime package versions. PostgreSQL stores each source TTL, canonical gold, source
category/subcategory, file artifact, bundle, request, response, validation and result.
The campaign identity also binds the plan hash, so changed files or parameters
cannot accidentally reuse a previous campaign's completed work.

Preparation registers the complete freeze as one `experiment-bundle` artifact and links it
to the campaign before any work becomes dispatchable: the plan, artifact identities and file
index, runtime versions, similarity identity, the original `parameters.yml` bytes, and every
collected file. Resume reconstructs from the current checkout and verifies its
code/dependency/input hashes against the frozen identities, rejecting drift before mutation;
the stored bundle is what makes a lost or changed checkout recoverable, read back with
`get_campaign_artifact`. Resume then advances the earliest incomplete durable checkpoint.
It never uses an implicit “latest” campaign or silently accepts code drift.
After a crash, an unacknowledged dispatch is ambiguous: automatic retry is forbidden.
Durable raw evidence is processed locally even while a provider is paused.

## Cost-estimate input

An explicitly supplied JSON file provides:

- prompt_artifacts: plan_sha256, tokenizer/estimation evidence, and input_tokens_by_run,
  mapping each planned run hash to an ordered list of token counts for its complete
  population. These cover full rendered prompts, including schema and demonstrations.
- billing_evidence: reporting currency and models, keyed as provider/model_id.
  A card contains mode, basis, provenance, input_per_million, output_per_million,
  and fx_to_reporting. Non-billed needs an evidenced basis; unknown prices are not zero.
- assumptions: expected_output_tokens, attempt2_fraction, attempt3_fraction,
  correction_error_tokens, total_output_token_ceiling, ceiling_verified, and
  reasoning_accounting_evidence. Billed reasoning tokens are included in output,
  not counted twice. Each correction includes the base and immediately prior answer/errors.

Admission uses one bound for every deployment rather than per-model records: complete
message UTF-8 bytes times one token per byte, plus an explicit message-overhead allowance,
plus the all-in output ceiling, checked against context capacity, TPM and cost reservation.
D-039 states the three conditions under which that is an upper bound - a byte-level BPE
tokenizer, chat-template overhead inside the allowance, and no server-side expansion - and
they are stated at the call site, not enforced in code. **Adding a deployment that violates
them requires re-establishing the bound before it can be trusted; nothing checks this for
you.** The generous headroom is what makes the shared bound safe for the deployments in use,
not a proof that it holds universally.

Prompt token counts in the evidence document are measured with one locally cached tokenizer,
whose id, class, vocabulary size and library version are recorded so a count can be
reproduced or refuted. They cover rendered message content only; protocol overhead is the
separate allowance above. No tokenizer downloads, calibration requests or fabricated
capability facts occur automatically.

All monetary calculations use Decimal and serialize amounts as strings. Actual
provider usage and costs are retained separately. An estimate does not impose a
cap; optional configured caps remain null unless the owner chooses otherwise.
An all-in output ceiling must cover reasoning billing and deployment behavior;
a nominal max_tokens field alone is not evidence of that bound.

## Separate authorization

The estimate command writes a plan-bound envelope carrying its own hash and prints its
monetary summary. `run` then requires the estimate file, an explicit `--authorize`, and a
named `--actor`, and writes the disclosure and authorization receipts from that invocation:
estimate_hash, plan_fingerprint, authorized_at, actor, explicit: true, and the exact channel.

Before writing them it re-derives what it is about to assert. The estimate's own hash is
recomputed from its contents, so a file edited after it was produced cannot keep a matching
plan hash and a `ready` flag while its numbers say something else. The models it prices must
be exactly the models the plan runs, so a campaign cannot be authorized against price
evidence for a different set of models. An estimate reporting any issue is refused.

`--authorize` is the approval; the command does not infer one from the presence of a file,
and it does not treat implementation approval as spending permission. Live execution
additionally requires enabled configuration, capability evidence measured per model, a price
card covering every planned model, frozen local MiniLM artifacts and a usable estimate.
Credentials alone never bypass these gates.

## Operational limits

The async process rotates provider claims, bounds workers and per-provider
concurrency, and checks separate rate gates. PostgreSQL admission serializes provider
reservations across processes. Waiting allocates no generation attempt and holds no
open transaction. A task advancement sends at most once; the orchestrator alone owns
the three-request total. Safe transient transport failures consume an attempt but
are not schema failures. Only three delivered invalid answers produce the explicit
empty prediction; exhausted operational failures remain unrankable.

The CLI catches expected boundary failures without printing DSNs or credentials.
Database setup generates private local credentials and never modifies the parent
.env. Reports use explicit campaign IDs and retain incomplete configurations,
all underlying results and Exact/Close Precision, Recall and F1.

## Non-billed campaign readiness clarification (2026-09-09)

For a plan whose every selected model has explicit `non_billed` billing evidence
with a basis and provenance, an unverified output ceiling is a disclosed warning,
not a monetary-estimate blocker. The estimate retains `ceiling_verified: false`
and reports zero expected and maximum monetary cost on that account-specific basis;
it does not assert bounded token consumption or runtime. Metered, mixed, missing,
or unknown billing retains the verified-ceiling requirement. All other required
evidence, live authorization, timeouts, request limits and scientific policies remain.

The estimator takes the existing plan, prompt evidence, billing cards and assumptions
and returns its existing hashed estimate plus `warnings`. Only complete non-billed
coverage permits the warning. Regression checks must cover free-only readiness,
mixed/paid rejection, and missing provenance rejection without changing assumptions.
