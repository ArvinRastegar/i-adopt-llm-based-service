# Reference Evidence

## Purpose

`reference/` is reserved for immutable evidence used to understand, audit, or regression-test the new design. Files retained here are not imported by the active runtime and do not become dependencies merely because they are present.

One reference artifact is retained: `reference/january/randomShotsPhaseOne.py.txt`, the pre-submission runner kept as provenance and regression evidence. It is never imported by the runtime.

## Planning-source evidence

The supplied `I-ADOPT_experiment_redesign_spec.md` is a 130,419-byte, 2,814-line planning input with SHA-256 `dc66ed2053b2b2796446ccca4a1315a498e156959f59951dde77411e3d5c68d1`. Its front matter identifies version `1.0`, dated `2026-08-18`, with author decision gates still pending.

The source is not the final protocol by itself. Later user decisions recorded in `DECISIONS.md` deliberately supersede parts of it, including the proposed two active tracks, immediate non-LLM baseline, active entity-linking/expert work, and RDF/JSON-LD/SHACL stages.

## Historical code evidence

The immutable pre-submission anchor is tag `V1.1-Experiment`, commit `b9683d2242aa5ca5b987440ea4b6f70bc1253c7e`, tree `35bf57411d7e8143a255415825f9bb0b644e8c6e`.

| Source at `V1.1-Experiment` | Role | Recorded SHA-256 |
|---|---|---|
| `benchmarking_example/randomShotsPhaseOne.py` | January paper-facing scorer behavior | `2d4a6271fd86a076127dfb3f85589391cff9744efdfdee8ec570b52c74921df0` |
| `benchmarking_example/gt_json_maker.py` | Pre-submission Turtle projection behavior | `b8caf88bc39a1aa8d6b01bc7e0fe2ed6419de5b6450455f0d6dd5242d7ab3773` |
| `benchmarking_example/data/Json_schema.json` | Schema embedded in historical prompts | `c099f2cebe91e495c22506b8accd592c209059a69c60110b12e7a263a7ab7d9a` |

The new evaluator will be rewritten as a pure package and regression-tested under accepted protocol `january-derived-member-credit-v1`. D-022 awards fractional TP for matched members, FN for missing members, and FP for extra members in simple/system and system/system comparisons, while excluding all system container labels. Asymmetric role evidence is retained explicitly. Unaffected January behavior is preserved; dictionary-string scoring and whole-component system thresholding are superseded only within the approved correction. Protocol acceptance does not establish implementation parity: source, dependency, and embedding artifact hashes will be recorded when those artifacts exist.

The new corpus importer is also rewritten. It does not preserve hard-coded paths, import-time writes, arbitrary RDF iteration, unstable blank-node identifiers, or semantic fallbacks from the historical helper.

The current `gt_json_maker.py` hash is `4ae635676d79ec665b3a5de1f0d472f3c85b2426a0c6db62a10302a845be2322`; that is a February 2026 variant and is retained only as additional design evidence. See `docs/repository-audit.md` for prompt hashes, the result inventory, and historical gaps.

## Provider evidence

The PSNC wire contract was inspected in `iadopt-variable-description-service` at commit:

```text
0d1a4cdae362b5aa9a23ab7b24b9f55d4a270e8b
```

Only the low-level OpenAI/LiteLLM-compatible chat-completions contract informs the independent adapter. I-ADOPT Lab does not import or call the sibling service’s higher-level decomposition workflow.

The owner later supplied a 294-line Python example with SHA-256 `901c0cdc1f774ad7cc0177586f77ff25548a9bc3813d629bfe6f48e2929569c1`. It supplies the selected six model IDs, the two SDK base URLs, and use of the OpenAI Python library with separate clients. `docs/model-catalog.md` records the exact mapping. The example was inspected without execution; no credentials, toy prompts, or live responses are copied. Its lack of explicit reasoning controls is not capability evidence, particularly for the mentioned but unverified Qwen `xhigh` option. Historical sibling-service model examples do not replace these owner-selected models.

## Deferred validation resources

These supplied files may later be retained under `reference/deferred-rdf-validation/`. Their documentation-phase source bytes were inspected read-only and have these identities:

| Supplied file | Bytes | SHA-256 | Deferred role |
|---|---:|---|---|
| `iadopt.sh.ttl` | 7,326 | `3a17b044587077e83604c14470c996eefeb4330a5a0c294a480791b1bf4a3895` | Structural I-ADOPT SHACL shapes |
| `iadopt-llm.sh.ttl` | 3,878 | `ef8479a9d4212dc55fc622c2cd126c51f6a64502bc5dde3dc041759deba25e4f` | Additional label/data-quality SHACL shapes |
| `Variable.context.jsonld` | 1,950 | `4759df64b174fd2ef8f5b28e86c548caee8dc068076bf1a6b62849070f0aa0f6` | JSON-LD term-to-ontology mapping reference |
| `Variable.schema.json` | 5,759 | `067c08cdccbf39981005b5be2118279b83a46320dd283080ff44604ceb7805e8` | Identifier-based JSON-LD Variable document schema |

The JSON Schema requires JSON-LD concepts such as `@context`, `@id`, `@type`, and identifier-bearing nested entities. It is therefore a different representation contract from the six-field lexical string/system output evaluated in this experiment.

Read-only syntax checks found:

- `Variable.schema.json` is valid JSON and passes JSON Schema Draft 2020-12 meta-schema checking.
- `Variable.context.jsonld` is valid JSON syntax.
- `iadopt.sh.ttl` parses as Turtle and contains 203 triples.
- `iadopt-llm.sh.ttl` does **not** parse as Turtle as supplied. The first parser failure is at line 21 because the preceding `shape:VariableShape` statement ends with `;` instead of a terminating `.`; later shape fragments must also be reviewed as part of repair.

Two additional candidate defects must be decided during that future review, not silently changed now:

- In `Variable.schema.json`, the `constrains` subschema uses the JSON-LD-looking keyword `@type` rather than JSON Schema's `type`. Unknown keywords are permitted by the meta-schema, so this does not enforce a string-valued `constrains` field.
- In `iadopt.sh.ttl`, `AsymmetricSystemShape_hasTarget` says a target requires a source but its second property path checks `iadopt:hasTarget` again. The likely `iadopt:hasSource` correction requires semantic approval.

No file has been copied or repaired yet. Before a future copy or activation, also record source/author, license, acquisition date, semantic approver, corrections, and purpose. Hashing and syntax-checking the supplied bytes do not approve their semantics or establish that they work correctly as a combined validator.

They are inactive in the lexical experiment. In particular:

- No RDF or JSON-LD is generated.
- No SHACL validation runs.
- `Variable.schema.json` is not the corrected lexical output schema.
- No ontology version is fetched or approved at runtime.

## Retention rules

Every retained artifact must include provenance and license information. Prefer exact copies over edited excerpts. If annotations are needed, store them in a separate Markdown file rather than modifying source evidence.

Reference files must never contain API keys, credentials, private provider responses, or mutable experiment results.
