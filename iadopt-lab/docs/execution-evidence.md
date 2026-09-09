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

Resume retrieves the bundle from PostgreSQL, checks its code/dependency/input hashes
against the current checkout, and advances the earliest incomplete durable checkpoint.
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

For live admission, assumptions additionally need tokenization_by_model, with
one record per provider/model key: maximum_tokens_per_utf8_byte,
message_overhead_tokens, and evidence. These must be verified conservative bounds
for the actual deployed tokenizer/protocol, not inferred from a model name.
The runner multiplies complete message UTF-8 bytes by that bound, adds explicit
message overhead and all-in output ceiling, then checks context capacity, TPM and
cost reservation. Missing evidence blocks live dispatch. No tokenizer downloads,
calibration requests or fabricated capability facts occur automatically.

All monetary calculations use Decimal and serialize amounts as strings. Actual
provider usage and costs are retained separately. An estimate does not impose a
cap; optional configured caps remain null unless the owner chooses otherwise.
An all-in output ceiling must cover reasoning billing and deployment behavior;
a nominal max_tokens field alone is not evidence of that bound.

## Separate authorization

The estimate command writes a plan-bound envelope with its hash and prints its
monetary summary. A later explicit authorization JSON contains estimate_hash,
plan_fingerprint, authorized_at, actor, and explicit: true; a separate disclosure
receipt contains estimate_hash and disclosed_at. The authorization command records
these supplied receipts; it never invents approval or treats implementation approval
as spending permission. Live execution requires enabled configuration, verified
provider capabilities, frozen local MiniLM artifacts, a usable estimate and these
receipts. Credentials alone never bypass these gates.

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
