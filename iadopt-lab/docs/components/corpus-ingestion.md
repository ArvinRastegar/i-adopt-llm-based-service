# Corpus Ingestion Contract

## Responsibility

Materialize the immutable pinned Corpus Git tree, verify exact source bytes, parse one variable from every Turtle file, project deterministic lexical ground truth, derive category metadata from paths, and activate the dataset atomically.

It does not generate prompts, infer missing semantics, link entities, convert predictions to RDF, or modify the upstream checkout.

## Inputs

- Repository URL, tag, expected commit/tree, expected count
- Exact Turtle bytes from the tag tree
- Corpus manifest, demonstration specification, and evaluation-population policy
- Lexical output schema and projection-policy version
- Importer code/environment version

## Outputs

- Verified source snapshot under `data/corpus/<tag>/`
- Deterministic canonical JSON under `data/canonical/<tag>/`
- Two derived projections per variable under the same canonical tree (D-031): a provenance metadata file and a human-readable variable file
- `data/manifests/corpus-<tag>.json`
- `data/manifests/demonstrations-<tag>.yml`
- `data/manifests/evaluation-population-<tag>.yml` containing exactly 97 variables
- `data/manifests/corpus-source-lock-<tag>.json`

All four manifest filenames carry the release, so a new release never overwrites the previous one's evidence. The `schema_version` field inside each file carries the independent schema version, which does not move with the release.
- PostgreSQL corpus/source/category/variable/gold rows
- Import audit report with counts, errors, and hashes

## Processing contract

The importer enumerates only `*.ttl` blobs from the pinned tree and orders release-relative paths by their exact UTF-8 bytes. It requires exactly 102 files and one Variable root per file. It reads the exact `rdfs:comment` as the v2 model-input definition and records that predicate. Required values must be unique; an RDF set is never resolved by arbitrary first value.

The first two release-relative directory components become category and subcategory. The complete parent path is stored as category path. Exact upstream spelling is immutable provenance.

Entity/system labels are projected deterministically. Symmetric parts are sorted only for canonical set representation. Asymmetric roles remain ordered. Stable system metadata labels are derived after validation and are not evaluator targets. Gold Constraints, which have no RDF list order, are sorted by the versioned lexical keys of `label`, then `on`, with canonical bytes as the final tie-break; prediction order remains untouched.

The five exact demonstration paths are verified and excluded. Every other source path is placed once in the evaluation-population manifest, producing exactly 97 members with no random sampling or partition.

Under D-031 the importer also retains the Wikidata IRI carried by each entity-bearing and property-bearing node, which the current gold projection discards, and writes two deterministic projections beside each canonical record: a provenance metadata file, and a readable variable file holding `label`, `definition`, the six lexical fields in their canonical empty representations, and a `*URI` key beside each field whose source node supplied an IRI. A field with no IRI omits its `*URI` key rather than setting it to `null`. Both projections are verified to regenerate byte-identically from the canonical parent; a mismatch is an integrity failure that blocks activation. Neither projection is a scoring input and neither joins the verification hash chain.

## Configuration keys consumed

- `dataset.*`
- `demonstrations.manifest` and `demonstrations.exclude_all_from_scoring`
- `evaluation_population.*`
- `generation.output_schema`

## Failures

Invalid Turtle, wrong file count/hash, missing/multiple roots, missing definition/label, ambiguous cardinality, unlabeled referenced component, incomplete role pair, invalid Constraint, unsupported nesting, schema failure, demonstration mismatch, or category-path failure aborts activation.

## Idempotency

Dataset identity includes tag, commit, tree, file-manifest hash, importer hash, and lexical-schema hash. Reimporting the same identity returns the existing active snapshot after verification. Different inputs create a new snapshot.

## Planned public functions

### `enumerate_tag_files(source) -> tuple[SourceBlob, ...]`

- **Input:** A frozen source descriptor containing repository URL, tag, commit, tree, allowed `.ttl` suffix, and expected file count.
- **Action:** Read blobs from the exact Git tree, reject symbolic/mutable checkout substitution, retain original release-relative paths and bytes, and sort paths by exact UTF-8 bytes.
- **Output:** An immutable tuple of source blobs containing path, Git blob ID, bytes, byte length, and raw SHA-256, plus verified source identity.
- **Raises:** Missing/ref-moved tag, commit/tree mismatch, inaccessible blob, invalid path/UTF-8 policy, duplicate path, unexpected file type, or count mismatch.
- **Side effects:** Read-only Git-object access; no checkout mutation, network fetch, filesystem materialization, or database write.
- **Determinism:** The same Git tree and enumeration policy produce the same ordered blob tuple.

### `parse_variable(ttl_bytes, source_identity) -> ParsedVariable`

- **Input:** Exact Turtle bytes and the owning source-file identity/path/category provenance.
- **Action:** Parse RDF, identify exactly one I-ADOPT Variable root, enforce documented predicate cardinalities and supported entity/system shapes, and resolve every referenced node and Constraint target without relying on graph iteration order.
- **Output:** An immutable typed graph projection retaining exact labels/definition, role structure, constraints, source node identities, and source provenance; no lexical sorting or metadata-label derivation occurs yet.
- **Raises:** Turtle syntax, wrong root count/type, missing/multiple required values, ambiguous optional values, unsupported nesting, incomplete/invalid system, unlabeled referenced component, invalid Constraint, or unresolved target.
- **Side effects:** None.
- **Determinism:** Equivalent RDF graphs produce equivalent typed projections even when triple or blank-node iteration order differs.

### `project_gold(parsed, policy) -> CanonicalGold`

- **Input:** One validated `ParsedVariable` and exact versioned projection, canonicalization, system-display, and collection-order policies.
- **Action:** Map RDF roles to the six lexical fields, derive stable unscored system metadata labels, resolve whole-system Constraint targets, sort only contract-declared set-like collections, validate the result, and hash its canonical representation.
- **Output:** Schema-valid six-field gold, retained source lexical/provenance fields, system metadata, projection trace, policy IDs, canonical bytes, and content hash.
- **Raises:** Unsupported representational case, non-unique lexical label, unresolved Constraint target, schema/semantic failure, or canonicalization conflict. It never guesses or silently drops a field.
- **Side effects:** None.
- **Determinism:** Identical parsed input and policy produce byte-identical gold and trace.

### `build_corpus_manifest(records) -> CorpusManifest`

- **Input:** The complete ordered source/projected record set plus frozen repository, importer, schema, license, and environment identities.
- **Action:** Verify completeness and uniqueness, calculate per-file and aggregate counts/hashes, and serialize a timestamp-free canonical manifest.
- **Output:** Manifest containing all 102 paths, blobs, byte hashes/sizes, category paths, variable/gold identities, aggregate regression counts, provenance, and manifest hash.
- **Raises:** Missing/duplicate record, path/order mismatch, count disagreement, hash conflict, incomplete license provenance, or non-canonical content.
- **Side effects:** None; materialization and database registration belong to `ingest_corpus`.
- **Determinism:** Input order is verified against the frozen ordering policy; equivalent complete inputs yield identical bytes/hash.

### `build_evaluation_population(records, demonstrations) -> EvaluationPopulationManifest`

- **Input:** The complete 102-record corpus inventory and the exact five-position demonstration manifest.
- **Action:** Verify demonstration membership/order/uniqueness, exclude all five regardless of shot setting, retain every other variable once in canonical path order, and bind each member to source/gold hashes.
- **Output:** A versioned, hashable manifest with 97 ordered members, five explicit exclusions, corpus identity, ordering policy, counts, and content hash.
- **Raises:** Unknown/duplicate demonstration, wrong demonstration order, member overlap, missing/duplicate corpus member, wrong `102 - 5 = 97` count, or hash mismatch.
- **Side effects:** None; no sampling, stratification, shuffling, or partitioning.
- **Determinism:** The same corpus and demonstration manifest produce the same population bytes/hash.

### `ingest_corpus(snapshot, repository) -> IngestionResult`

- **Input:** A verified complete snapshot bundle (source blobs, parsed/projected records, manifests, licenses, hashes) and a typed PostgreSQL corpus repository.
- **Action:** Materialize immutable source/canonical files through an atomic staging boundary, register all corpus/category/variable/gold/population/demo records in one activation workflow, verify persisted hashes/counts, then mark the snapshot active.
- **Output:** An ingestion result containing existing-or-created snapshot identity, paths, database IDs, verification counts, manifest hashes, activation status, and complete error report when unsuccessful.
- **Raises:** Evidence conflict, filesystem atomic-write failure, migration/transaction failure, database constraint violation, or post-write verification mismatch.
- **Side effects:** Writes only the documented immutable `data/` snapshot/materializations and PostgreSQL registry rows; never modifies upstream Git data. Failed staging is not active/runnable.
- **Idempotency:** Repeating the identical snapshot verifies and returns the existing identity; conflicting content under the same identity is a hard integrity failure.

## Acceptance tests

## Implementation boundary (version 1)

`ingest_corpus(project_root, source_repository=None, source_directory=None)` materializes and verifies a
complete filesystem bundle and returns a dictionary with `manifest`, `records`,
`demonstrations`, and `evaluation_population`. It does not activate database rows;
the workflow passes this verified bundle to the PostgreSQL repository transaction.
`source_repository` is a read-only local Git object database used to create or
cross-check `corpus-source-lock-<tag>.json`. The bundled lock permits a normal
directory/ZIP extraction through `source_directory`, requiring all 102 exact paths,
byte hashes, byte lengths and Git blob IDs without requiring a Git checkout.
When supplied, that directory is the primary read-only source of Turtle bytes.
Without either argument, the importer verifies the already materialized pinned
snapshot. No function downloads, fetches a mutable branch, or silently accepts a
working-tree file. Local operator paths do not enter scientific content identities.

`load_canonical_records(project_root)` verifies the manifest and each record hash,
returning ordered dictionaries. Each record exposes `variable_id`, `source_path`,
`definition`, `gold`, `category`, `subcategory`, `category_path`, `source_sha256`,
and `gold_sha256`, plus versioned provenance. `project_root` means `iadopt-lab/`.
Atomic activation is the corpus manifest published last: incomplete staging is
never readable as an activated dataset. Existing artifact bytes are checked and
never overwritten with conflicting content.

The version-1 lexical ordering key is `(NFC(text).casefold(), text)`; Constraint
target resolution uses `NFC(text).strip().casefold()`. Targets must resolve to one
distinct retained lexical value; collisions are rejected. Identical labels in
different roles are one lexical target because the six-field format cannot
distinguish them. System aliases resolving to a different component are ambiguous.

The historical importer is reference-only: the new projection preserves exact
comments and labels, rejects cardinality ambiguities, preserves ratio roles,
requires explicit Constraint targets, and never substitutes blank-node IDs.

### Source IRI collisions verified during implementation

The release contains six duplicated Variable IRIs, each reused in two different
files. An RDF IRI therefore cannot be the database primary identity. The importer
preserves it as `source_iri` and derives `variable_id` as
`urn:iadopt-lab:variable:` plus SHA-256 of UTF-8 `commit + "\n" + source_path`.
This retains all 102 source variables without changing upstream labels or gold.
Paths are release-relative, so relocating the source directory changes no identity.

- Exact tag/commit/tree and 102 blobs
- Every v2 file projects successfully
- Regenerated counts match 52 Matrix, 10 Context Object, 9 Statistical Modifier, 85 variables/157 Constraints, 36 asymmetric systems, and 2 symmetric systems
- Repeated processes yield identical canonical hashes
- Two symmetric and all asymmetric systems have stable metadata IDs
- Source path/category/subcategory retained exactly
- Demonstration paths resolve uniquely and in order
- Exactly 97 non-demonstration variables appear once in the evaluation population
- Ambiguous synthetic TTL fails without partial activation
