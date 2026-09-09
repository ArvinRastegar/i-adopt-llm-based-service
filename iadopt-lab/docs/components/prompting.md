# Prompting Contract

## Responsibility

Load immutable prompt templates and render exact model messages from one target definition, the lexical schema, and an ordered demonstration prefix. Render correction prompts from the same base plus the immediately previous response/errors.

It never reads target gold fields, chooses model parameters, calls providers, parses responses, or scores predictions.

## Inputs

- Prompt ID/version/template bytes/hash
- Exact lexical-schema bytes/hash
- Shot count in `0`, `1`, `3`, `5`
- Approved ordered demonstration records
- Target variable ID and exact definition
- Optional immediately previous attempt response/error bundle

## Outputs

- Provider-neutral ordered message array
- Human-readable full rendered prompt
- Template/schema/example/definition/rendered hashes
- Character/byte/token estimate metadata
- Demonstration IDs and order

## Invariants

- The five approved demonstration variables are the only source pool.
- Shot count selects an ordered prefix.
- Target cannot also be a demonstration.
- Definition is copied byte-for-byte from the canonical corpus record.
- Schema in the prompt is the runtime validator schema.
- Templates retain the historical no-interpretation/no-inference policy under accepted D-021. Only reviewed minor six-field/schema, workflow/correction, and display-name adaptations are permitted.
- Historical prompt hashes and reference-to-template diffs remain part of prompt-version evidence. The renderer cannot rewrite instructions or broaden their extraction policy.
- Known non-literal demonstration gold and geographic-Matrix tensions are documented limitations; demonstration gold stays unchanged, and rendering never repairs those tensions or adds correctness hints.
- Every absent lexical field uses its explicit schema-defined empty representation; no field is omitted or set to null.
- Correction attempt includes only its own immediate predecessor.
- No category, filename, URI, RDF, SHACL, or target gold decomposition leaks into the target section.

## Configuration keys consumed

- `demonstrations.manifest`
- `parameter_grid.prompt_variants`
- `parameter_grid.shot_counts`
- `generation.output_schema`
- `generation.correction_attempt.*`

## Planned public functions

### `load_prompt_version(prompt_id) -> PromptVersion`

- **Input:** One exact registered prompt-family/version identifier; callers cannot request an implicit latest version.
- **Action:** Resolve the approved template artifact, read exact UTF-8 bytes, verify its registry hash/protocol metadata and recorded historical-reference/minor-change review, and reject aliases that are provenance-only historical names.
- **Output:** Immutable prompt version containing stable ID, display name, version, exact bytes, byte length, hash, renderer protocol, historical reference hash, reviewed diff identity, and review/freeze status.
- **Raises:** Unknown/unfrozen ID, missing artifact, non-UTF-8 bytes, registry/hash mismatch, or incompatible renderer protocol.
- **Side effects:** Read-only artifact/registry access; no template mutation, prompt render, database write, or fallback.
- **Determinism:** One frozen artifact identity always resolves to identical bytes and metadata.

### `select_demonstrations(pool, shot_count) -> tuple[Demonstration, ...]`

- **Input:** The verified five-item ordered demonstration pool and requested shot count.
- **Action:** Validate pool identity/order/uniqueness, allow only `0`, `1`, `3`, or `5`, and select the exact prefix without randomization or target-aware choice.
- **Output:** Immutable tuple of zero, one, three, or five demonstration records in approved order, retaining their source/gold hashes for prompt lineage.
- **Raises:** Invalid shot count, wrong pool/version/order, duplicate/missing demonstration, or invalid demonstration decomposition.
- **Side effects:** None.
- **Determinism:** Pure prefix selection; it never uses a seed, category, target, or score.

### `render_base_prompt(request) -> RenderedPrompt`

- **Input:** Frozen prompt version, exact lexical-schema bytes/hash, selected demonstration tuple, exact target definition/hash, renderer version, and context-window/tokenizer facts supplied by preflight.
- **Action:** Validate artifact agreement and target/demo non-overlap, render sections in the frozen order, insert schema and demonstrations verbatim under the defined escaping rules, and calculate component/final message hashes and sizes.
- **Output:** Exact provider-neutral one-user-message array, human-readable equivalent, all component/final hashes, demonstration identities/order, byte/character counts, and deterministic token estimate when a frozen tokenizer is available.
- **Raises:** Artifact/hash mismatch, invalid demo/target relationship, unresolved template field, non-UTF-8 value, unsafe delimiter/escaping failure, or context-window overflow. It never truncates.
- **Side effects:** None; no database write or provider request.
- **Determinism:** Identical request inputs and renderer version produce byte-identical message content and hashes.

### `render_correction_prompt(base, previous_attempt) -> RenderedPrompt`

- **Input:** The exact frozen base rendering and the immediately preceding content-invalid attempt's number, untouched assistant-visible response, response hash/length, and complete ordered extraction/schema/semantic errors.
- **Action:** Verify parent/evidence hashes, JSON-string escape the prior response, append exactly one versioned correction section, and prove every base/scientific component remains unchanged.
- **Output:** One-user-message corrected rendering with base/correction/final hashes, parent attempt identity, error IDs/pointers, sizes, and inherited scientific lineage.
- **Raises:** Missing/non-immediate parent, non-content failure, evidence/hash conflict, empty/incomplete errors, attempt outside `2..3`, escaping failure, or context overflow.
- **Side effects:** None; it neither updates task state nor invokes a provider.
- **Determinism:** The same base and previous-attempt evidence yield identical corrected bytes/hash.

## Acceptance tests

## Implementation interface (version 1)

`load_prompt_version(prompt_id, project_root=None)` reads a frozen template and
hash registry. `select_demonstrations(pool, shot_count)` validates the exact five
source paths and selects the approved prefix. `render_base_prompt(template,
target_definition, schema, demonstrations=(), *, target_id=None)` accepts no
target gold or metadata. Schema is exact UTF-8 bytes. It returns a `RenderedPrompt`
with `messages`, `content`, `metadata`, `sha256`, and a serializable `to_dict()`.
`render_correction(base, raw_response, errors, previous_attempt_number=1)` returns
the unchanged base plus one escaped canonical feedback object. Calling it with
an already corrected base is rejected to prevent accumulating past failures.

Context/token capacity requires the caller's frozen tokenizer and provider facts;
the renderer records exact bytes/characters but does not invent a token count.
Templates preserve historical instruction wording except removal of regenerated
definition/comment instructions and documented six-field compatibility additions.
The prompt registry retains historical bytes and exact unified diffs as JSON
evidence, with review classification `approved-compatibility-policy-D021`.

- Snapshot tests for all three prompts and four shot counts
- Exact demonstration order
- No target leakage
- Schema hash equality
- Reviewed historical-reference diffs contain only approved minor adaptations and preserve the no-interpretation/no-inference instructions
- Unchanged demonstration gold and documented semantic limitations survive rendering without repairs or correctness feedback
- Unicode and whitespace determinism
- Correct previous response/error isolation
