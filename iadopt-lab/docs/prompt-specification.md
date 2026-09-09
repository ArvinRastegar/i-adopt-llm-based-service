# Prompt Protocol Specification

## 1. Purpose

This document defines the model-visible prompt protocol before prompt-template files or Python rendering code are created. The accepted policy is to retain the historical prompts and their no-interpretation/no-inference instructions, with only minor changes required by the new six-field schema, workflow, correction feedback, and display names. The sections below specify those adaptations and summarize the retained behavior; they do not authorize a semantic rewrite of the historical prompts. Every future template change must be recorded in a reviewed diff, versioned, and hashed.

The three prompt families are:

| Stable ID | Display name | Experimental difference |
|---|---|---|
| `strict-minimal` | Strict minimal | General conservative extraction instructions |
| `constraint-decomposition` | Constraint decomposition | Adds an ordered constraint-identification and target-resolution procedure |
| `matrix-decomposition` | Matrix decomposition | Adds an ordered Matrix-versus-condition/location/method disambiguation procedure |

They are not called decision trees in new reports. Historical filenames/IDs remain provenance aliases only.

## 2. Historical prompt evidence and required changes

The immutable pre-submission runner at tag `V1.1-Experiment` loaded these files:

| Historical file | SHA-256 | New family |
|---|---|---|
| `benchmarking_example/data/prompts/strict_minimal.txt` | `c184ec3edcdac4eee348fd315835110e7b9ce6f42b4229d09ad6501f0b8afedc` | `strict-minimal` |
| `benchmarking_example/data/prompts/constraint_tree.txt` | `e7355417a4153999287df4cbe199de47f33e2119840df644e06d2dc0efb12a42` | `constraint-decomposition` |
| `benchmarking_example/data/prompts/matrix_tree.txt` | `f77aba03472c117b15aa2e3ac27d76519687629f7eb4d4fffb89e60c88237229` | `matrix-decomposition` |

Those bytes remain the immutable reference and semantic baseline. Future templates may differ only through the documented minor compatibility changes below; implementation must retain an auditable diff against the reference for each family:

- The current working path `strict_minimum.txt` is a later rename. The current constraint prompt is also a post-submission variant whose appended category allowlist must not be attributed to the January run.

- The model now returns only the six evaluated fields; it no longer regenerates `label`, `definition`, or `comment`. D-031 reaffirms this. A readable stored prediction does carry `label` and `definition`, but the pipeline attaches them from the canonical corpus record when the prediction is persisted; they are never requested from the model, never reach the evaluator, and never enter the prediction hash. The model has deliberately never seen the target label, so a generated one would be invented rather than recalled, and re-emitting the definition on every task reintroduces the truncation and paraphrase failures this adaptation removed, at a real token cost, for two fields that carry no score.
- The corrected lexical schema allows simple, symmetric, source/target asymmetric, and numerator/denominator asymmetric entities.
- `hasContextObject` follows the same entity/system union as Object of Interest and Matrix.
- Constraint `on` names the extracted component or system/member being constrained. The historical constraint prompt's separate list of labels such as `size`, `state`, and `location` describes constraint-label categories and must not be used as the `on` value.
- Every optional field is present with an explicit empty value.
- Validation feedback is provided on attempts two and three; the old nested retry implementation is not copied.

These adaptations must not introduce interpretation, inference, or permission to add unstated information. Known disagreements between the retained instructions and Corpus demonstration gold are recorded in Section 4.1 and deliberately remain unresolved for this experiment. Following implementation approval, the three version-1 templates and their exact reference diffs are now stored under `prompts/` and `data/manifests/prompt-registry-v1.json`.

## 3. Message structure

Primary generation is non-streaming and uses exactly one provider-neutral user message. A base message contains sections in this fixed order:

1. Task and conservative extraction rules.
2. Six-field meaning and system/constraint rules.
3. Prompt-family-specific instruction block.
4. Exact lexical JSON Schema.
5. Zero, one, three, or five ordered demonstrations.
6. Exact target definition.
7. Final output-only instruction.

The implementation must not add an undocumented provider-specific system message, assistant prefill, tool, response-format schema, or hidden example. If a provider requires a transport wrapper, the sanitized exact request proves that the model-visible messages remain equivalent.

The one-user-message policy also applies to correction requests. The frozen base message is reproduced and one deterministic correction section is appended. This avoids relying on provider-specific multi-turn reasoning-block requirements and makes the complete input visible in one stored message.

## 4. Common task instruction

Every family must retain the historical conservative extraction rules while communicating the required six-field/schema adaptations. The following is a specification of that behavior, not a replacement prompt to substitute for the historical wording:

- Decompose exactly one scientific-variable definition into the six keys shown in the schema.
- Use only information explicitly supported by the definition; do not add background knowledge, ontology lookup, units, methods, instruments, locations, or inferred entities unless the definition states them in a role represented by the schema.
- Return exactly one JSON object and no prose, Markdown fence, quotation wrapper, explanation, or second candidate.
- Include every top-level key. Use `""` for any scalar/entity value that cannot be extracted without invention and `[]` for no constraints. Do not use `null` or omit a key.
- `hasProperty` is the characteristic being observed, counted, classified, or derived.
- `hasStatisticalModifier` is an explicitly stated statistical operation such as maximum, mean, sum, or count when modeled separately from Property.
- `hasObjectOfInterest` is the entity or system whose Property is observed.
- `hasMatrix` is the medium or material containing the Object of Interest. A method, instrument, geographic location, condition, state, time, threshold, or unit is not a Matrix merely because it follows a preposition.
- `hasContextObject` is another explicitly necessary background entity that is neither the Object of Interest nor Matrix.
- Use a symmetric system only when at least two parts have equivalent roles. Use source/target for a directional relation and numerator/denominator for a ratio. Never mix the two asymmetric role pairs.
- A system container string is metadata. The model may emit an empty or descriptive value accepted by the schema; the canonical record derives the reproducible value from parts/roles, and the evaluator never scores the model-supplied container string.
- A Constraint is an explicitly stated restriction. Its `label` preserves the meaningful restriction text, including a prefix such as `state:` or `reference:` when present. Its `on` identifies the emitted Property, Statistical Modifier, entity, system member, or whole derived system that the restriction limits.
- Do not make a Constraint point to another Constraint. Do not invent a target merely to make validation pass.

The template gives no correctness hints derived from the target gold record.

### 4.1 Accepted policy and documented prompt-to-gold limitations

The conservative wording above follows the historical prompts, but an independent check of the approved five Corpus demonstrations found that several gold labels are not literal spans of their `rdfs:comment` definitions:

| Demonstration | Definition evidence | Gold lexical value that creates tension |
|---|---|---|
| Air daily maximum temperature | Says “air” and “aboveground” | Matrix is `atmosphere` |
| Blood lactate concentration | Says “blood lactate concentration” | Matrix is `person` |
| Circulation method in pipe | Says “water” and “canalisation pipe” | Object is `water circulation`; Matrix is `sewer line` |
| Heat stress index | Says “per district” | Matrix is `urban area` |

The welfare demonstration also contains a gold Context Object `urban area`, but that phrase is literally present in its longer definition and therefore does not create the same conflict.

The historical Matrix instructions also exclude geographic locations, while the approved heat-stress demonstration assigns `urban area` to Matrix. That is a role-assignment tension in addition to the non-literal wording; “per district” does not establish that a district is urban.

Under accepted decision D-021, the prompts keep the historical no-interpretation/no-inference policy. They do not gain permission for faithful paraphrasing, role inference, external lookup, or unsupported additions. The Corpus gold answers and approved demonstration order remain unchanged. Only the minor schema/workflow/display adaptations in Section 2 are permitted.

Consequently, literal conservative extraction and matching the RDF-derived gold are not universally compatible. We record this as an experiment limitation, including the geographic-Matrix tension, without repairing the prompts' semantic rules or gold answers now. A disagreement of this kind is not a JSON-shape failure: it must not trigger a correctness retry, a gold-dependent hint, or a manual intervention. The ordinary scorer evaluates the accepted prediction against the unchanged gold.

The policy choice is resolved. Exact version-1 template bytes, reference-to-template
diffs and hashes have been materialized. Snapshot and limitation-preservation tests
verify the accepted policy; any future wording change requires a new version.

## 5. Family-specific instruction blocks

Only the retained family-specific instruction block varies between prompt families. The common schema/workflow adaptations, demonstrations, target, and final output rule are otherwise identical for a resolved task. The descriptions below document the historical procedures; they are not permission to rewrite them or resolve their known semantic limitations.

### 5.1 Strict minimal

The block requires one conservative pass:

1. Read the definition literally.
2. Extract Property and Object of Interest when supported by the text; keep the required keys even when a value must be empty.
3. Add optional roles only when directly supported.
4. Represent an explicitly described composite entity with the correct system shape.
5. Add only explicit constraints and resolve each target to an emitted component.
6. If uncertain about an optional value, use its empty representation.

It must not contain the specialized constraint ordering or extended Matrix disambiguation rules below.

### 5.2 Constraint decomposition

The block requires this order:

1. Extract Property and optional Statistical Modifier.
2. Extract Object of Interest, then Matrix and Context Object if stated.
3. Decompose any system into equivalent parts or ordered roles.
4. List the exact lexical values that may serve as Constraint targets.
5. Identify each explicit limiting phrase only after the core components exist.
6. For every Constraint, preserve a minimal faithful `label` and set `on` to the exact emitted target label or the documented whole-system display label.
7. Omit a proposed Constraint when it cannot be tied to an emitted target without invention.

The model does not output the intermediate list or reasoning; it outputs only the final JSON object.

### 5.3 Matrix decomposition

The block requires this order:

1. Extract Property and Object of Interest first.
2. Ask whether another noun phrase is a material/medium within which that Object occurs. Only then may it become Matrix.
3. Treat an explicitly necessary additional entity as Context Object when it is not a containing medium.
4. Treat condition, state, threshold, reference frame, temporal/spatial extent, method, instrument, unit, and location phrases as non-Matrix information. Represent an explicit restriction as a Constraint when it has a valid emitted target; otherwise do not force it into another role.
5. Re-check that Matrix is not a process, method, instrument, geographic location, condition, or state.

The model does not output this decision process.

## 6. Schema section

The prompt includes the exact bytes of the frozen `lexical-decomposition.schema.json` in a clearly delimited JSON section. The prompt-embedded SHA-256 and runtime-validator SHA-256 must match before planning.

The schema, not informal examples, is authoritative for shape. Prompt prose may explain semantics but cannot permit a value forbidden by the schema or require one the schema allows to be empty.

Provider-native structured-output enforcement remains disabled for the primary comparison. The same text extraction and validator process therefore applies to every provider/model.

## 7. Demonstration section

Each demonstration contains only:

- A stable demonstration position/identifier for audit.
- The exact Corpus `rdfs:comment` used as input.
- Its six-field model-visible lexical decomposition, serialized deterministically against the same schema.

It excludes variable label, file path, category, IRIs, RDF, linking data, and all other provenance from model-visible content. Those values remain in database evidence.

The five-path order is fixed in `data/README.md`; shot count selects the first `k`. The demonstration serializer uses stable key order and UTF-8. No demonstration is selected randomly, and all five demonstration variables are excluded from every score.

System container values in demonstrations follow the same canonical display policy used after prediction validation. Because the values are unscored metadata, a test must prove that changing them alone cannot affect evaluator output.

## 8. Target section

The target section contains exactly the canonical Corpus definition string and a fixed delimiter/heading. It does not contain target label, path, category, subcategory, issue number, IRI, RDF triples, decomposition, constraint targets from gold, or entity links.

The renderer stores the source definition hash, character count, UTF-8 byte count, and final message hash. It does not normalize the definition's punctuation, capitalization, or whitespace silently.

## 9. Correction section

A content-invalid attempt with remaining budget appends a correction section to the unchanged base message. The section contains one canonical JSON feedback object with:

```text
protocol version
previous attempt number
previous assistant-visible response as one JSON string
previous response SHA-256 and UTF-8 byte length
ordered extraction/schema/semantic validation errors
instruction to return exactly one corrected JSON object
```

JSON-string escaping preserves quotes, newlines, backticks, and braces deterministically and prevents the previous output from changing prompt delimiters. The database still stores the untouched raw response separately; the escaped prompt representation is not treated as the raw artifact.

Attempt 2 includes only attempt 1 feedback. Attempt 3 includes only attempt 2 feedback. Hidden reasoning fields, credentials, transport headers, another variable's output, and cumulative older failures are never inserted.

When a permitted retry follows a transport failure without model output, the next attempt uses the unchanged base message and stores the transport reason outside model-visible content.

## 10. Inputs, outputs, and failures

### Renderer inputs

- Frozen family/version/template bytes and hash.
- Frozen lexical schema bytes/hash.
- Exact target definition and hash.
- Approved demonstration manifest and requested shot count.
- Optional immediately preceding invalid-attempt evidence.
- Renderer/protocol version.

### Renderer outputs

- Exact one-message role/content array.
- Human-readable rendering of the same content.
- Template, schema, demonstration, target, correction, and final-message hashes.
- Ordered demonstration identities.
- Byte/character counts and deterministic token estimate when available.

### Failures

Rendering fails before provider dispatch on missing/mismatched artifacts, invalid shot count, demonstration/target overlap, invalid demonstration decomposition, schema-hash mismatch, target-definition mismatch, unresolved template field, non-UTF-8 content, oversized model context, missing correction parent, or evidence-hash conflict.

There is no fallback to another prompt family and no silent truncation of schema, demonstrations, definition, previous output, or validation errors. A context-window overflow is a preflight/planning failure for that model/configuration.

## 11. Prompt freeze and tests

Before live provider execution, the exact three template files must be reviewed, versioned, and hashed, with their historical reference hashes and an explicit minor-change diff retained as evidence. Required tests include:

- Snapshots for 3 families × 4 shot counts.
- Exact single-user-message structure and section order.
- Same common/schema/demo/target bytes across families except the documented family block.
- Exact demonstration prefixes and no target leakage.
- Reference-to-template diffs limited to the six-field/schema, workflow/correction, and display-name adaptations; the historical no-interpretation/no-inference instructions remain intact.
- Reviewed fixtures retain the unchanged demonstration gold and document every identified non-literal value and geographic-Matrix tension; they must not silently repair those tensions or add inference permission.
- Prompt schema hash equals runtime schema hash.
- Constraint `on` examples follow component-target semantics, not the historical category list.
- Systems cover both asymmetric alternatives and symmetric parts.
- Correction encoding round-trips the exact prior response and ordered errors.
- Braces, fences, quotes, Unicode, and adversarial-looking previous output cannot break section construction.
- Hidden reasoning is excluded from correction content.
- Byte-identical rerendering and hash equality across processes.

Any wording change after campaign planning creates a new prompt version and task/campaign fingerprints. It never mutates prompts already stored with attempts.
