# Shared Domain, Canonicalization, and Hashing Contract

## Responsibility

The shared foundation defines immutable typed records, canonical JSON serialization, content hashes, lexical sort keys, UTC/time envelopes, and secret-safe request evidence used across I-ADOPT Lab.

It does not parse Turtle, render prompts, call providers, validate model semantics, access PostgreSQL, choose task state, or calculate evaluator similarity.

## Inputs

- Already typed primitive/domain values
- A named canonicalization contract/version
- Exact bytes to hash
- A typed provider request plus a versioned redaction allowlist
- Lexical strings used only as deterministic sort/display-label inputs

Inputs containing unsupported runtime objects, non-finite numbers, unordered anonymous mappings, naive datetimes, or secret-bearing unknown fields are rejected rather than serialized heuristically.

## Outputs

- Immutable records with explicit schema/protocol versions
- UTF-8 canonical JSON bytes
- Lowercase 64-character SHA-256 values
- Deterministic lexical comparison keys that do not replace scored source strings
- Sanitized request evidence plus redaction-policy identity
- Typed UTC timestamp envelopes for operational evidence

## Processing invariants

- Canonical JSON fixes key ordering, number representation, UTF-8, escaping, and whitespace.
- Array order is preserved unless a data contract explicitly marks the collection as set-like.
- Exact source/raw bytes are hashed before parsing; parsed/canonical derivatives receive separate hashes.
- Set sorting uses a documented Unicode normalization/case-folded comparison key with a stable original-string tie-break. It does not mutate the retained lexical value.
- Provider request redaction is allowlist-based. Authorization, cookies, API keys, DSNs, and secret-like unknown headers are never returned.
- Scientific content hashes contain no clock, random UUID, worker order, or machine-local path unless that field is explicitly part of provenance.
- No hash function silently converts a path to its file contents or a string to normalized text.

## Configuration keys consumed

This foundation reads no YAML directly. Callers pass already resolved canonicalization, redaction, and evidence-policy versions. Their originating configuration/protocol identities remain in the returned records.

## Planned public functions

### `canonical_json_bytes(value, contract) -> bytes`

- **Input:** A supported typed value and exact canonicalization version.
- **Action:** Validate representability and serialize deterministically.
- **Output:** UTF-8 bytes with no platform-dependent formatting.
- **Raises:** Unsupported type/value, non-finite number, ambiguous unordered value, or unknown contract.
- **Side effects:** None.
- **Determinism:** Byte-identical for identical typed inputs and contract.

### `sha256_bytes(data) -> str`

- **Input:** Exact immutable bytes.
- **Action:** Calculate SHA-256 without decoding or normalization.
- **Output:** Lowercase 64-character hexadecimal digest.
- **Raises:** Type error for non-bytes input.
- **Side effects:** None.
- **Determinism:** Pure.

### `build_content_identity(kind, version, payload) -> ContentIdentity`

- **Input:** Registered artifact kind/version and typed content payload.
- **Action:** Canonically serialize the versioned envelope and hash it.
- **Output:** Identity containing kind, version, byte length, hash, and canonical bytes/reference.
- **Raises:** Unknown kind/version or canonicalization failure.
- **Side effects:** None; persistence belongs to repositories.
- **Idempotency:** Identical envelope returns the identical identity.

### `lexical_sort_key(text, policy) -> LexicalSortKey`

- **Input:** Original lexical string and versioned sorting policy.
- **Action:** Build the Unicode-normalized/case-folded comparison key and original-string tie-break.
- **Output:** Comparable key; original input remains unchanged.
- **Raises:** Non-string or unsupported policy.
- **Side effects:** None.
- **Scientific boundary:** This key orders set-like data and constructs metadata labels; it is not a scorer normalization function.

### `sanitize_provider_request(request, policy) -> SanitizedRequest`

- **Input:** Typed outbound request before authentication plus explicit allowed evidence fields.
- **Action:** Remove/prohibit credentials and secret headers, retain all scientific fields, and hash the sanitized result.
- **Output:** Sanitized typed request, redaction report, policy version, and hash.
- **Raises:** Unknown secret-like field, incomplete allowlist, or prohibited evidence value.
- **Side effects:** None; it does not send or persist the request.
- **Security:** Error messages cannot echo rejected secret values.

## Failures and caller behavior

A canonicalization or sanitization failure blocks the caller before provider/database mutation. Hash/content conflicts are integrity errors. Callers cannot fall back to `str()`, default JSON serialization, dropped fields, or an unversioned redactor.

## Acceptance tests

- Stable canonical bytes across processes, locale, timezone, and dictionary insertion order
- Exact array order and explicitly sorted set-like collections
- Unicode normalization/sort edge cases without changing retained scored strings
- Finite decimal/integer handling and rejection of NaN/infinity
- Raw-byte hash differs appropriately from parsed/canonical derivative hash
- Known digest fixtures and type rejection
- Secret/header/DSN redaction, unknown-field fail-closed behavior, and non-secret scientific-field retention
- Timestamp/random/machine-path exclusion from scientific identities
