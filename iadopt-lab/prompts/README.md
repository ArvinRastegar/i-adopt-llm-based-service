# Prompt Directory

## Current status

Three version-1 lexical decomposition templates are implemented and covered by
12 exact family/shot snapshot hashes. `data/manifests/prompt-registry-v1.json`
retains their original January prompt text, source hashes, template hashes and
complete unified diffs. Runtime loading verifies each artifact before rendering.

## Prompt families

| Stable ID | Display name | Purpose |
|---|---|---|
| `strict-minimal` | Strict minimal | Extract only explicitly supported components and leave unsupported fields empty |
| `constraint-decomposition` | Constraint decomposition | Extract core components before constraints and attach every constraint to an extracted component |
| `matrix-decomposition` | Matrix decomposition | Distinguish an actual medium/material from conditions, locations, processes, or methods |

The new experiment does not call these prompts “decision trees.”

## Base prompt input

The renderer receives:

- Exact frozen template bytes
- Exact corrected lexical JSON Schema and hash
- Shot count
- Ordered demonstration prefix
- Target variable definition
- Template and protocol versions

The target prompt does not receive its label, category, source path, URI, gold decomposition, or any linked-entity answer.

## Base prompt output

The renderer produces:

- Complete provider message array
- Human-readable full rendered prompt
- Template, schema, demonstration, definition, and final-prompt hashes
- Ordered demonstration IDs
- Rendering version

Rendering is deterministic. Every full rendered prompt is stored in PostgreSQL with the attempt that used it.

## Ordered demonstrations

Shot counts use one frozen list:

- `0`: no demonstration
- `1`: first approved demonstration
- `3`: first three approved demonstrations
- `5`: all five approved demonstrations

There is no random example selection. Demonstration order is part of the prompt hash and task fingerprint.

## Model output instruction

Every template asks for exactly one JSON object containing the six lexical fields. It instructs the model to use `""` or `[]` for unsupported information and not to invent values.

Accepted decision D-021 retains the historical prompts' no-interpretation/no-inference policy. Only minor changes needed for the six-field schema, workflow/correction feedback, and new display names are allowed. Future templates must preserve a reviewed diff against the immutable historical prompt bytes; they must not broaden the extraction policy or permit unstated information.

Several approved demonstration gold values are not literal definition substrings, and the geographic-Matrix rule conflicts with the heat-stress demonstration. `docs/prompt-specification.md` records these limitations. The gold answers and prompt semantic rules remain unchanged; the limitations do not trigger corrective prompt edits, correctness retries, or human intervention in this experiment. Tests explicitly retain those limitations and the original no-inference instruction.

The schema is supplied as text. Provider-native structured-output enforcement is disabled in the primary comparison so every selected model follows the same extraction, validation, and retry path.

## Correction prompt

A content-invalid response can lead to a correction attempt if the task has remaining allowance. The correction prompt contains:

- The unchanged base prompt material
- The immediately previous assistant-visible raw response
- Exact extraction, schema, or semantic-validation errors
- A request for one corrected JSON object

Provider-returned hidden reasoning is retained as evidence where available but is not copied into the next prompt.

Model, provider, prompt family, examples, temperature, reasoning mode, repetition, seed, and other scientific parameters remain unchanged. Only the validation feedback section differs.

There are at most three provider requests total per variable/configuration task. A provider adapter cannot add hidden retries.

## Versioning rules

- Prompt text is immutable after a campaign is planned.
- Any byte change creates a new template version and hash.
- Each template retains its historical source hash and reviewed diff showing only the approved minor compatibility changes.
- Display-name-only changes must still be recorded.
- Prompt and schema versions are independent but jointly fingerprinted.
- A rendered prompt must be reconstructable from stored inputs and must reproduce its stored hash.
- Templates never contain credentials, provider keys, or environment-specific paths.

Snapshot tests must cover every prompt/shot combination, preserve the accepted no-interpretation policy and unchanged demonstration gold, and confirm that target gold data cannot leak into the request. Known prompt-to-gold limitations are documented fixtures, not defects that the renderer repairs.

## Implemented modules

`iadopt_lab.prompting.renderer` loads versions, selects the fixed prefix, renders
one user message, and appends immediate-predecessor validation corrections.
`iadopt_lab.prompting.provenance` rebuilds the reference registry from the immutable
January Git commit when deliberately regenerating implementation artifacts. The
normal experiment runtime does not read or import historical service code.

The templates differ only in their historical family instruction block. Common
six-field compatibility instructions and schema/demo/target section bytes are
shared. Rendered input is roughly 4,200–6,400 characters for the snapshot target,
not a claimed 300-token prompt; provider/tokenizer facts are required for estimates.
