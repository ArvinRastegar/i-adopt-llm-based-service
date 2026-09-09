# Future Entity Linking, RDF, and SHACL Work

## Status: deferred

Entity linking and generated RDF are not used by the active I-ADOPT Lab experiment. There is no RDF conversion after a valid lexical decomposition, and no SHACL validator runs during generation, retry, evaluation, or reporting.

This boundary is important: the current scorer consumes lexical JSON directly. Adding RDF would increase cost and complexity without changing the agreed lexical Precision, Recall, and F1 computation.

## Explicitly inactive stages

The current campaign does not perform:

- URI or vocabulary candidate generation
- Wikidata lookup
- NERC, QUDT, PATO, or other vocabulary linking
- Candidate ranking or acceptance
- Stable RDF-node assignment
- JSON-LD conversion
- RDF/Turtle serialization of model predictions
- SHACL validation
- Ontology-version approval
- Human semantic correction

Configuration, reports, and diagrams must not imply that any of these stages ran.

If this future phase is approved, the requested serialization target is RDF/Turtle, not JSON-LD. `Variable.context.jsonld` may help interpret the supplied validation bundle but does not establish a future JSON-LD output stage.

## Supplied reference resources

The following externally supplied resources may later be copied into `reference/deferred-rdf-validation/` with their original bytes, source locations, licenses, and SHA-256 values:

- `iadopt.sh.ttl`
- `iadopt-llm.sh.ttl`
- `Variable.context.jsonld`
- `Variable.schema.json`

Their current supplied-byte hashes and sizes are recorded in `reference/README.md`. That inventory is provenance only; it does not activate RDF conversion, JSON-LD output, or SHACL validation.

The same reference inventory records that `iadopt-llm.sh.ttl` is not Turtle-syntax-valid as supplied. A future repair must preserve the original bytes separately, create a new repaired artifact/hash, validate the combined shape graph, and obtain the required semantic approval before execution.

They are reference evidence only. In particular, `Variable.schema.json` is not the corrected lexical decomposition schema used in prompts or response validation.

No supplied file becomes an active dependency merely because it is retained under `reference/`.

## Possible future pipeline

If approved later, the work may be split into independent, versioned stages:

```text
valid canonical lexical decomposition
  -> optional entity-linking candidates
  -> optional accepted URI assignments
  -> deterministic RDF construction
  -> optional SHACL validation
  -> validation and linking reports
```

Each stage requires its own input/output contract. RDF construction must not modify the stored lexical prediction or its original evaluation.

## Future entity-linking contract topics

A future design must decide and document:

- Benchmark vocabularies and gold identifiers
- Candidate-source versions and retrieval dates
- Candidate generation, ranking, thresholds, and tie behavior
- Exact-match and semantic-match metrics
- Behavior for components without a gold URI
- Multiple acceptable identifiers
- Network caching and offline reproducibility
- Provider/API licensing and rate limits
- Whether a human may approve candidates
- Separation between automatic experiment and human-in-the-loop service use

The existing Wikidata-oriented code cannot be described as a multi-vocabulary decomposition baseline.

## Future RDF contract topics

A future deterministic converter must define:

- Exact ontology release and namespace map
- Input lexical-schema version
- URI policy for variables, entities, systems, and constraints
- Blank-node or skolemization policy
- Mapping for source/target and numerator/denominator systems
- Mapping for symmetric-system parts
- Constraint target resolution
- Serialization format and canonicalization
- RDF graph hash and provenance
- Behavior when lexical values are incomplete

The converter must never repair or enrich the lexical prediction silently.

The initially planned serialization format is RDF/Turtle. Adding JSON-LD output would require a separate later decision and contract.

## Future SHACL contract topics

Before SHACL execution is enabled, the project must establish:

- Which shapes file is authoritative
- Shapes-file version and SHA-256
- Which RDF graph is validated
- Validation engine and version
- Inference regime
- Severity policy
- Retry or correction policy
- Whether validation is structural, semantic, or both
- Storage format for complete validation reports

SHACL failures must not be retroactively described as failures from the active lexical experiment.

## Activation gate

None of this work may become active until a new decision record approves the scientific purpose, ontology and vocabulary versions, exact contracts, database migrations, tests, cost, and human-input policy. Activation creates a new protocol and campaign version; it does not mutate existing lexical results.
