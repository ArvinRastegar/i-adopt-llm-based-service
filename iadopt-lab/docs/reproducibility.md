# Reproducibility and Provenance Contract

## 1. Reproducibility claim

I-ADOPT Lab targets **procedural and evidential reproducibility**: another authorized operator must be able to reconstruct the exact inputs, software, decisions, requests, responses, validations, predictions, scores, and reports for a campaign.

Remote LLM providers may change model weights, serving stacks, routing, or nondeterministic execution without exposing a byte-identical revision. Therefore the project must not promise that a future provider rerun will reproduce identical model text. Instead it records every provider identifier and returned fingerprint available, retains the original response and repetition identity, and marks unavailable revision facts as `unknown` rather than inventing them. D-029 selects one repetition at every temperature, including stochastic temperatures, for all selected providers/models. This limits request volume but cannot estimate within-configuration run-to-run variability; additional independent repetitions require new authorized generation, not reinterpretation of retries.

Local deterministic stages—corpus projection, configuration resolution, prompt rendering, extraction, schema/semantic validation, canonicalization, scoring with a frozen embedding artifact, aggregation, and report generation—must reproduce their hashes from identical inputs.

## 2. Reproducibility units

| Unit | Stable identity includes |
|---|---|
| Corpus snapshot | Repository, tag, commit, tree, ordered file manifest, exact file bytes/hashes, importer, lexical schema |
| Demonstration set | Five variable identities, exact order, canonical decomposition hashes |
| Evaluation population | Corpus identity, all 97 ordered variable/source/gold identities, five demonstration exclusions, count, manifest version/hash |
| Prompt version | Family, exact template bytes, message/rendering contract, schema hash |
| Model profile | Provider, exact model ID/revision, capability evidence, reasoning profiles/native fields |
| Campaign | Canonically ordered selected-provider set, selected provider-owned model lists, shared protocol, frozen billing/optional-cap and pre-run-estimate policy, and resolved configuration fingerprint |
| Resolved run | Campaign-scoped identity for exactly one selected provider/model/reasoning plus prompt, shots, temperature, sampling, repetition, evaluation population, scorer protocol |
| Task | Resolved run plus target variable, definition hash, gold hash, retry protocol |
| Attempt | Task plus attempt number, exact rendered messages/request, dispatch/result evidence |
| Evaluation run | Frozen prediction set plus scorer, embedding model, dependencies, device/dtype, tolerance |
| Configuration ranking | Evaluation population/run, complete configuration identities, repetition metrics, ranking policy, rankability, shared ranks, hash |
| Report | Explicit campaign/evaluation IDs plus query/report code, settings, format, and source hashes |

Identifiers use canonical JSON serialization and SHA-256. Display formatting and database-generated IDs do not replace content fingerprints.

## 3. Required campaign manifest

Every campaign must retain or reference immutable records for:

### 3.1 Source and code

- I-ADOPT Lab Git commit and tree.
- Whether the worktree was clean.
- If not clean, a reviewed diff/patch hash and reason.
- Implementation package versions and source hashes.
- Database migration revision.
- Historical reference file hashes used for parity.

### 3.2 Runtime environment

- Python version and executable implementation.
- Exact dependency lockfile and hash.
- Operating system, architecture, locale, and timezone.
- Container image name and digest when containers are used.
- PostgreSQL major/server version and configured extensions.
- CPU/GPU identity relevant to local embedding inference.
- Numerical library versions, determinism settings, embedding device/dtype, and tolerance.

### 3.3 Data and protocol

- Corpus repository, tag, commit, tree, Git blob IDs, exact file SHA-256 values, category paths, and canonical record hashes.
- Demonstration and 97-variable evaluation-population manifests.
- Original `parameters.yml` bytes/hash and canonical resolved configuration/hash for the complete nonempty `campaign.providers` set and each selected provider's enabled model list. The initial six exact IDs are supplied in [the model catalog](model-catalog.md); later campaigns may select either provider or both with different positive model counts. Current configuration version is `2.1`.
- Prompt templates, complete rendered messages, schema, semantic rules, and hashes.
- Provider/model IDs, capability evidence, endpoints/region/routing policy where known, exact reasoning controls, sampling parameters, repetition, seeds, and retry policy.
- Evaluator code/configuration, January parity bundle, each explicit correction and its scope, similarity threshold, embedding model revision/files/hashes, and zero-denominator policy. Accepted protocol `january-derived-member-credit-v1` identifies member/role matching, fractional TP/FP/FN normalization, container-label exclusion, and missing-role evidence. The protocol name is not an artifact hash; source, dependencies, and embedding artifacts must be frozen and hashed before evaluation.
- Per-provider billing mode/basis, account-context provenance, effective date, currency, optional provider/global cost caps, reservation/settlement policy, price cards for metered access, and confirmed-zero/actual/estimated/unavailable cost status. Null means no cap, zero means a true zero ceiling, and all default cap amounts are null. The user's PSNC access is initially recorded as owner-reported no-charge independently of these cap settings; this is a scoped billing assumption, not a claim about every PSNC user/model. Missing token usage remains unknown even when financial cost is explicitly zero.
- Required `pre-run-estimate-v1` receipt/hash referencing the resolved campaign/plan and price-card/FX provenance, source dates and assumptions, planned and maximum-three-attempt request counts, prompt/output/reasoning token scenarios and correction-prompt bounds, per-model/provider/combined monetary totals, and estimate completeness. Retain the disclosure receipt and separate explicit live-authorization evidence. The estimate is informative, not a hard cap; exceeding it does not by itself pause an uncapped campaign.

### 3.4 Attempt and result evidence

- Every full prompt/message sequence and sanitized outgoing body.
- Every untouched raw response body/envelope and assistant-visible text.
- Optional returned reasoning fields and token categories where exposed.
- Transport events, delivery certainty, timestamps, latency, provider IDs, usage, and finish reason.
- Extraction candidates/offsets/strategy and complete validation errors.
- Retry decisions and correction-parent linkage.
- Canonical or explicit-empty prediction.
- Component and member match records, similarities, original member/role positions, missing-role evidence, TP/FP/FN/TN contributions, aggregate metric numerators/denominators/support, and report lineage.
- Every repetition-level metric, configuration-ranking input, shared rank, lower-ranked result, and `not_rankable` reason.
- Provider/model lineage on every result, per-provider and combined planned/completed counts, provider-specific pause/backoff events, campaign-wide pause events, and final report/completion receipts.

## 4. Canonical hashing rules

The implementation must publish one canonicalization specification before hashes become scientific identities. At minimum:

- UTF-8 encoding without platform-dependent transcoding.
- Canonical JSON with deterministic object-key ordering, number representation, and no insignificant whitespace.
- Arrays retain semantic order; set-like collections are sorted only where the data contract explicitly declares them unordered.
- `campaign.providers` is a set serialized in canonical provider-ID order; changing only its YAML order leaves the resolved identity unchanged. The ordered demonstration pool is not a set and must never be sorted this way.
- Source files and raw provider bodies are hashed as exact bytes, not parsed or pretty-printed values.
- Prompt templates and rendered messages have separate hashes.
- Original YAML bytes and resolved canonical JSON have separate hashes.
- Secrets are removed before a sanitized request hash is computed; the sanitization-policy version is included.
- Generated timestamps, UUIDs, filesystem paths, and worker completion order are excluded from deterministic scientific content hashes unless the contract explicitly makes them evidence fields.

A hash mismatch is an integrity failure that pauses the campaign. It is never corrected by overwriting prior evidence.

## 5. Two configuration hashes

I-ADOPT Lab stores:

1. **Original configuration hash:** exact `parameters.yml` bytes, proving what the operator supplied.
2. **Resolved campaign fingerprint:** canonical non-secret values that affect the selected campaign.

Editing an inactive provider catalog changes the original configuration hash but does not change the resolved fingerprint. The resolved configuration contains only the selected provider set and its active model/profile entries, alongside shared settings. Editing selected membership, an active model ID/profile, reasoning value, prompt, shot, temperature, repetition, supported provider seed, output limit, evaluation population, ranking policy, schema, retry policy, scorer, or other resolved setting creates a new fingerprint and campaign. The preserved original YAML still records inactive catalog entries without allowing them to alter an already frozen plan.

Resume reads the stored frozen campaign, including its complete provider/model union, billing basis, optional caps, and estimate/disclosure/authorization links. It does not absorb later YAML additions or rename a pending task's model. Execution/run/task identities remain campaign-scoped; changing the selection creates new execution identities, not implicit reuse or transfer of another campaign's task rows. A changed resolved campaign or pricing/FX basis requires a new estimate and disclosure before live use. A separate comparison key may identify matching scientific conditions across campaigns, but it is not authorization to skip generation or merge evidence.

## 6. Reconstructing an existing campaign

An authorized reconstruction proceeds without new provider calls:

1. Restore a verified PostgreSQL backup into a compatible PostgreSQL 16 environment.
2. Check migration, registry, campaign/provider memberships, per-provider and combined run/task counts, attempt, prediction, evaluation, cost-reservation, pre-run estimate/disclosure/authorization, and configuration-ranking counts.
3. Obtain the exact source revision, dependency lock, corpus/manifests, schemas, prompts, scorer, and embedding artifacts named by the campaign.
4. Verify every stored content hash and foreign-key relationship.
5. Re-render sampled or all prompts from frozen inputs and compare hashes.
6. Re-run extraction and validation against stored raw responses in a new audit namespace; compare candidates, errors, and predictions without overwriting originals.
7. Re-run the same scorer as a new verification evaluation or compare against the immutable evaluation records.
8. Rebuild the combined selected-provider ranking from stored unrounded repetition metrics, verify coverage/shared ranks and provider/model dimensions, then regenerate reports from explicit IDs and compare canonical output hashes. Verify final completion against the entire frozen plan rather than the subset whose provider finished first.
9. Record the reconstruction environment, differences, and pass/fail result.

This audit is different from regenerating model responses. It proves that the recorded scientific result follows from the retained evidence.

## 7. Re-evaluation without regeneration

Metric logic is isolated so stored predictions can be evaluated under a new scorer version without making another LLM call. A re-evaluation must:

- Create a new immutable scorer/evaluation identity.
- Retain original and new contributions side by side.
- State every semantic difference and affected cases.
- Never relabel the old result as if it used the new metric.
- Never alter the prediction or raw response.

The initial scorer uses accepted protocol `january-derived-member-credit-v1` and preserves January behavior outside the approved member-credit and system-label corrections. Member credit applies both when a simple entity is compared with a system and when two systems partly match. Matched members contribute fractional TP, missing members FN, and extra members FP; container labels are excluded. Symmetric/symmetric pairs retain literal equality in both modes; other member comparisons use the existing lowercase/strip Exact matcher and inclusive Close threshold `0.80`. Two asymmetric systems preserve ordered slots; comparisons without corresponding role evidence record that limitation instead of claiming role identification. Retain member counts and exact contribution/metric/ranking fractions, with decimal displays as derivatives. The full contract is in `scorer-parity.md`. Any later normalization, matching, threshold, assignment, contribution-weighting, or aggregation change creates a different scorer version.

## 8. Repeating generation later

A future provider rerun is a new campaign even when its intended parameters match. It must record the new access time, selected providers and current model lists/capability metadata, routing/revision information, billing basis and price cards where applicable, software, and environment. Results may be compared using shared logical labels, but request evidence is never merged with the earlier campaign.

Reasoning `enabled` and `disabled` are comparable experimental labels only when exact native fields are known and supported for that model at that time. A mandatory-reasoning model cannot be placed in a false disabled condition; preflight marks the comparison unavailable.

## 9. Backup and restore

The minimum durable archive is:

- An encrypted, verified PostgreSQL backup containing complete experiment evidence.
- The exact repository revision and any approved dirty patch.
- The dependency lock and container digest.
- The Corpus source snapshot and manifests.
- Frozen prompt/schema/scorer/embedding artifacts not already byte-contained in PostgreSQL.
- Frozen price/billing and FX evidence, pre-run estimate receipts, disclosure records, and explicit live-authorization records not already byte-contained in PostgreSQL.
- A machine-readable top-level archive manifest with hashes.

Restore tests must verify sampled raw-body hashes plus all registry, campaign, task, attempt, prediction, and evaluation constraints. A backup is not considered valid merely because its file exists.

## 10. Secret and privacy boundary

Reproducibility does not justify storing credentials. The archive excludes API keys, bearer/authorization headers, database passwords, cookies, complete environment dumps, and secret-bearing connection strings.

Full prompts and raw responses are retained because they are scientific evidence; database and backup access must reflect that sensitivity. Reports use sanitized derived fields and must render raw model content as untrusted text, never executable markup.

## 11. Reproducibility acceptance tests

- Repeated corpus projection in separate processes produces byte-identical canonical records and manifests.
- Repeated planning with identical inputs produces identical run/task fingerprints and no duplicates.
- Prompt rendering produces identical messages/hashes independent of machine path and worker order.
- Extraction, validation, and canonicalization reproduce prior event hashes from stored raw responses.
- The frozen evaluator reproduces every contribution and aggregate within its declared numerical tolerance.
- Ranking reproduces every complete configuration's primary value and shared rank without dropping any result.
- Backup/restore preserves counts, links, exact raw evidence, and hashes.
- Reports regenerate from explicit IDs with stable canonical JSON/CSV contents.
- Provider-selection reordering changes only the original YAML hash; PSNC-only, OpenRouter-only, and combined selections create distinct resolved campaign identities. An inactive-provider edit changes only the original YAML hash, while an active catalog or selected scientific change creates a new campaign fingerprint.
- Frozen resume preserves provider/model ownership and does not import later YAML edits or reuse another campaign's task rows.
- Restore preserves explicit no-charge provenance, unavailable usage, provider/global reservations and optional-cap semantics, estimate/disclosure/authorization evidence, provider-specific pauses, and all-provider completion evidence without making new requests.
- An uncapped campaign retains complete actual-cost accounting; changing its resolved plan invalidates its prior estimate, while merely exceeding an estimate does not trigger a cap pause.
- No tracked artifact or stored request contains a credential.
- A rerun with unavailable provider revision data records `unknown` and does not claim bitwise model reproducibility.
