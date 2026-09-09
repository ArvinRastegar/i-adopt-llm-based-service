# Data Directory

## Current status

The authoritative release is **Corpus v2.0.1** under D-030: 102 verified Turtle
files, 102 canonical gold records with two derived companions each, the fixed five
demonstrations, the complete 97-variable evaluation population, and three
historical-compatible prompt versions. These are immutable inputs and
materializations, not live experiment results.

The superseded **v2.0.0** snapshot is retained beside it. Release-scoping every
path meant the older tree was never overwritten, so the four gold differences can
be inspected side by side. Its manifests keep their original `-v1` filenames;
v2.0.1 manifests carry the release in the filename instead.

Do not edit any materialization by hand. Regenerate it with `iadopt-lab ingest`
and re-check it with `iadopt-lab verify`.

The operator-supplied `I-ADOPT-Variables/Corpus-2.0.1` directory is the read-only
source for the migration, with every path, byte hash, length and Git blob ID checked
against the approved immutable release. No corpus download is needed.

## Planned layout

```text
data/
  corpus/
    <tag>/                  exact upstream Turtle snapshot and license
  canonical/
    <tag>/                  deterministic lexical gold JSON materialization
                            per variable, three adjacent files (D-031):
                              <Name>.json           canonical record, authoritative
                              <Name>.meta.json      provenance projection
                              <Name>.readable.json  label, definition, six fields, IRIs
  manifests/
    corpus-<tag>.json       Git and file-level provenance
    corpus-source-lock-<tag>.json
                            portable Git-derived exact source-byte inventory
    demonstrations-<tag>.yml
                            five examples and their exact order
    evaluation-population-<tag>.yml
                            all 97 scored variables in canonical order
                            (filenames carry the release; the schema_version
                             field inside carries the schema version)
    prompt-registry-v1.json  exact templates, historical source bytes/hashes and diffs
    scorer-model-v1.yml     close-similarity model provenance
    price-card-<campaign>.yml
                             all selected provider/model prices or explicit non-billed basis
    fx-<campaign>.yml        dated conversion basis when reporting currency differs
  fixtures/
    dry-run/                non-scientific mock-provider cases
    scorer-regression/      reviewed evaluator parity cases
```

The exact implementation may add a directory only after its input, output, ownership, and retention policy are documented.

## Authoritative corpus input

The source lock is:

| Property | Value |
|---|---|
| Repository | `https://github.com/i-adopt/Corpus` |
| Tag | `v2.0.1` |
| Commit | `2598bf91fa927b78a6529bae7864ef0f7d485b73` |
| Tree | `df665d32bb2433a60742c80a4a53908dc7ebde0c` |
| Concept DOI | `10.5281/zenodo.18101358` |
| License | `CC-BY-4.0` |
| Expected Turtle files | 102 |

The stored lock still records the v2.0.0 version DOI as its license source; see the open item in `THIRD_PARTY_NOTICES.md`.

The snapshot must be materialized from the immutable Git object, not copied from a mutable working checkout. Original release paths, filenames, file bytes, and license information are preserved.

A normal extracted source directory is also supported: its bytes must match the
bundled Git-derived source lock exactly. Merely matching a directory name or tag
string is insufficient. `ingest_corpus(..., source_directory=...)` performs that
verification before any activation; subsequent reads require no external Git clone.

The documentation-phase read-only tag audit found 102 valid Variable roots, 52 variables with Matrix, 10 with Context Object, 9 with Statistical Modifier, 85 variables containing 157 Constraints, 36 asymmetric systems (31 ratio and 5 source/target), and 2 symmetric systems containing 4 parts. These are implementation regression expectations, not a substitute for regenerating and hashing the manifest.

## Corpus manifest

Every Turtle entry records at least:

- Release tag, commit, and tree
- Git blob ID
- Release-relative path
- SHA-256 and byte length
- Category, subcategory, and complete category path
- Stable variable identity
- Import status and canonical-record hash

The release path is authoritative for science classification. For example:

```text
Life Sciences/Biology/C14_FeralfreeEnclosureArea.ttl
```

produces category `Life Sciences`, subcategory `Biology`, and category path `Life Sciences/Biology`. Source spelling is retained exactly; a reporting alias may be stored separately.

## Demonstration manifest

The fixed order is:

1. `Natural Sciences/Atmospheric Science/C2_AirDailyMaximumTemperature.ttl`
2. `Social Sciences/Demography/PersWelfare.ttl`
3. `Life Sciences/Health Science/lactate.ttl`
4. `Technical Sciences/Material Science/CirculationMode-Water.ttl`
5. `Social Sciences/Disaster Risk Science/HeatStress.ttl`

Shot counts `0`, `1`, `3`, and `5` select ordered prefixes. All five entries are excluded from scoring, including zero-shot scoring. The immutable evaluation-population manifest contains every other Corpus variable exactly once: 97 variables total. There is no train/development/test partition.

Corpus and evaluation-population paths use exact release-relative UTF-8 byte order. Gold Constraint arrays use the frozen lexical sort key for `label`, then `on`, and canonical bytes as the final tie-break because RDF provides no list order. These ordering rules affect reproducible serialization and hashes, not scientific meaning.

## Canonical gold data

Canonical JSON is a deterministic, reviewable materialization of the PostgreSQL gold record. It contains provenance metadata and the six lexical decomposition fields, but no model output and no entity-linking enrichment.

Random RDF blank-node identifiers are never treated as lexical labels. Symmetric and asymmetric system container labels are derived deterministically for stable storage and are excluded from scoring.

Six upstream Variable IRIs are each reused by two files (96 distinct IRIs across
102 files). `source_iri` preserves that evidence; the unique `variable_id` instead
hashes the frozen commit and exact source path. No variables are merged or removed.

Two v2 Constraints point to their complete unlabeled asymmetric-system blank node. Their canonical `on` value is derived from the ordered system roles, so neither gold JSON nor a report contains the parser's unstable blank-node identifier.

The original Turtle snapshot is attributed to the I-ADOPT Corpus contributors.
Its CC-BY-4.0 license basis is the frozen release DOI record, retained in both
source and corpus manifests. The Git tree contains no standalone license file;
`corpus/LICENSE-PROVENANCE.md` records this distinction rather than inventing an
upstream license artifact.

If any of the 102 files cannot be represented without guessing, the complete corpus import fails. A partial directory must not be treated as an experiment-ready corpus.

## Source of truth

PostgreSQL is authoritative for imported records, task relationships, predictions, and scores. Files under `data/` provide immutable source inputs, reviewed manifests, and deterministic materializations. They are not a second mutable experiment database.

Price manifests retain effective dates, source URLs or account-context statements, units, token categories including reasoning where billed, and exact model/provider ownership. FX evidence is required when currencies differ; it records dated rate/source, units, and conversion policy without inventing a rate. These inputs support the required `pre-run-estimate-v1` receipt stored in PostgreSQL and its derived pre-run disclosure report. The receipt binds their hashes to the resolved plan, scenario assumptions, call counts, and token bounds. A changed resolved plan or price/FX basis requires a new estimate/disclosure. A price manifest is not a mandatory spending cap: configuration `2.1` permits null caps, while retaining full accounting and explicit no-charge evidence independently.

## Data that does not belong here

Do not place the following under `data/`:

- API keys or environment files
- Database passwords or dumps containing secrets
- Ad hoc downloads without provenance
- Live raw provider responses
- Rendered prompts from individual attempts
- Mutable spreadsheets used as the only result store
- RDF or JSON-LD generated from predictions
- Entity-linking candidates or outputs in the active experiment

Full prompts, raw responses, validation events, and authoritative results belong in PostgreSQL. Sanitized derived exports belong under `outputs/`.
