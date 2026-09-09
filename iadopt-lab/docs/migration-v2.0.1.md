# Migration to Corpus v2.0.1 and derived per-variable files

## Purpose and status

**Executed.** Every step below was carried out under explicit owner authorization, together with `variable_id` remaining commit-derived. The suite passes at 274 passed / 14 skipped, the corpus verifies at tag `v2.0.1`, and all 204 derived files regenerate byte-identically from their canonical parents. This file is retained as the record of what changed and why, and as the template for the next release bump.

Four things surfaced during execution that the plan did not anticipate. Each is described in place below:

1. Editing the importer invalidated every canonical record, because `importer_sha256` is part of each record.
2. The three unscoped manifests would have been overwritten, so all four are now release-scoped.
3. Nested system keys serialized in two different orders depending on whether a record was in memory or reloaded.
4. `corpus-manifest.schema.json` pinned the regression counts as a `const`, in addition to the three known pin sites.

## Why this is not a configuration edit

`parameters.yml` declares the dataset, but it does not own it. Three artifacts must agree:

1. `parameters.yml` — the human-edited `dataset:` block
2. `schemas/parameters.schema.json` — pins release, commit, and tree as JSON Schema `const` values, so an unmatched YAML edit fails validation immediately
3. `src/iadopt_lab/corpus/ingestion.py` — module constants used to verify the Git tag, resolve blobs, and stamp every derived record

Editing only the YAML makes `load_parameters` raise. Editing only the constants makes ingestion write records that the manifest schema rejects. They change together or not at all.

## Target identity

| Field | Current (v2.0.0) | Target (v2.0.1) |
|---|---|---|
| Tag | `v2.0.0` | `v2.0.1` |
| Commit | `8097662ca323771fd977d22cdb8c3e58b7b7d64a` | `2598bf91fa927b78a6529bae7864ef0f7d485b73` |
| Tree | `bd9cf247d22c5b8345796572e3e55f333460c6ce` | `df665d32bb2433a60742c80a4a53908dc7ebde0c` |
| Turtle files | 102 | 102 |
| Demonstrations / population | 5 / 97 | 5 / 97, same members and order |

Both tags are present in the owner's local clone at `/Users/rastegar-a/Documents/GitHub/I-ADOPT-Variables`. That operator-local path is provenance for this migration, not portable scientific identity; any verified clone containing the tag yields the same corpus identity.

## Consequence that is easy to miss: every variable id changes

`ingestion.py` derives the stable variable identifier from the commit and the release-relative path:

```text
variable_id = "urn:iadopt-lab:variable:" + sha256(SOURCE_COMMIT + "\n" + source_path)
```

Because the commit is an input, **all 102 identifiers change**, not only the four variables whose content changed. Everything keyed by `variable_id` is therefore rebuilt: the population manifest, the demonstration manifest, every canonical record, and — once a campaign is planned — every task fingerprint. Nothing carries over from a v2.0.0 plan.

This is correct under the current identity policy, which treats a variable's identity as belonging to an exact source snapshot. It is worth a deliberate confirmation rather than an accident, because the alternative is available: deriving the identifier from the release-relative path alone would keep identifiers stable across corpus patch releases and make cross-release comparison possible, at the cost of letting one identifier denote two different gold answers. **Decide this before running the import**, since changing it later is another full re-identification. If the current behavior is kept, record it explicitly in D-030 rather than leaving it implicit.

## What actually changed

| Item | Before | After |
|---|---|---|
| Release identity | v2.0.0 / `8097662c` / `bd9cf247` | v2.0.1 / `2598bf91` / `df665d32` |
| Release paths | seven hardcoded `v2.0.0` literals | derived from `SOURCE_TAG` |
| Manifests | one release-scoped, three unscoped | all four release-scoped |
| Canonical record schema | `corpus-record-v1` | `corpus-record-v2`, retaining source IRIs |
| Files per variable | 1 | 3 (canonical, `.meta.json`, `.readable.json`) |
| Regression counts | matrix 52, context object 10 | matrix 51, context object 11 |
| Prompt snapshots | 12 | 6 unchanged (0/1-shot), 6 updated (3/5-shot) |

The prompt-snapshot split is the clearest confirmation that D-030's analysis was right: `PersWelfare` is demonstration position 2, so it enters only the 3-shot and 5-shot prefixes. The 0-shot and 1-shot hashes came back byte-identical to their v2.0.0 values across all three families, and exactly six of twelve moved.

## Step 1 — Ingestion constants

`src/iadopt_lab/corpus/ingestion.py`

- Lines 24-26: `SOURCE_COMMIT`, `SOURCE_TREE`, `SOURCE_TAG` to the v2.0.1 values above.
- Line 217: docstring says "repository path containing tag v2.0.0"; update the wording.
- Lines 438, 486, 487, 490, 504, 510, 516: hardcoded `data/corpus/v2.0.0`, `data/canonical/v2.0.0`, and `data/manifests/corpus-v2.0.0.json` path segments.

Prefer deriving those path segments from `SOURCE_TAG` rather than replacing one literal with another, so the next release is a one-line change and no stale path can survive a partial edit.

## Step 2 — Schema constants

- `schemas/parameters.schema.json` lines 46, 49, 52 — release, commit, tree.
- `schemas/corpus-manifest.schema.json` lines 4, 10, 11, 12 — title, tag, commit, tree.
- `schemas/evaluation-population.schema.json` line 9 — commit.

## Step 3 — Configuration

`parameters.yml` — the `dataset:` block, and remove the D-030 holding comment placed above it. `expected_turtle_files` stays `102`; `definition_predicate` and `category_from_release_path` are unchanged.

## Step 4 — Derived per-variable files (D-031)

Ingestion currently writes one canonical record per variable. It must additionally write two deterministic projections of that same record. The canonical record stays authoritative: it keeps `record_sha256`, it is what the manifest and scoring read, and the derived files never enter a hash chain that verification depends on.

**Metadata file** — provenance only: repository, tag, commit, tree, source path, Git blob id, source byte length, `source_sha256`, `gold_sha256`, `record_sha256`, importer version and hash, lexical schema hash, category, subcategory, category path, `variable_id`, `source_iri`, and `demonstration_position`.

**Readable variable file** — the shape of the earlier TTL-to-JSON conversion, pretty-printed rather than minified:

```json
{
  "label": "Feral-free enclosure area",
  "definition": "The total area (in ha) of feral-free enclosures. ...",
  "hasProperty": "area",
  "hasPropertyURI": "https://www.wikidata.org/entity/...",
  "hasObjectOfInterest": "enclosure",
  "hasObjectOfInterestURI": "https://www.wikidata.org/entity/...",
  "hasMatrix": "",
  "hasContextObject": "",
  "hasStatisticalModifier": "",
  "hasConstraint": [{"label": "condition: free of feral", "on": "enclosure"}]
}
```

All six lexical fields are always present, using the same empty representations as the gold record — `""` for an absent scalar or entity, `[]` for no constraints. A `*URI` key is emitted only beside a field whose source Turtle carried an IRI; it is omitted rather than set to `null` when the source has none. The IRIs are already parsed from the Turtle during ingestion but are currently discarded when projecting `gold`, so this step also requires retaining them.

Add a verification pass, alongside the existing hash checks, asserting that each derived pair regenerates byte-identically from its canonical parent. A mismatch is an integrity failure, not a warning.

Decide the file naming before implementing, and state it in `data/README.md`. The straightforward option is a suffix beside the canonical file — `C14_FeralfreeEnclosureArea.json` (canonical), `C14_FeralfreeEnclosureArea.meta.json`, `C14_FeralfreeEnclosureArea.readable.json` — which keeps a variable's three files adjacent when sorted.

## Step 5 — Readable predictions (D-031)

The model's contract is unchanged: exactly six fields, `additionalProperties: false`, and `iadopt_eval` continues to reject anything else. Do not add `label` or `definition` to `schemas/lexical-decomposition.schema.json`, to the prompt templates, or to `_validate_decomposition`.

Instead, when a prediction is persisted, attach the target's real `label` and `definition` — copied from the canonical corpus record — to the readable prediction view only. `validation/lexical.py::_canonicalize()` already sits between raw model output and the scored record and is the natural boundary. The scored six-field prediction must remain byte-identical to what it is today; the attachment is presentation, and it must not reach the evaluator, the prediction hash, or any metric.

### Executed note — importer hash invalidates the corpus

The first corrected re-import failed with `Immutable artifact conflicts with existing bytes` on a canonical record that had just been written. The cause is correct behaviour, not a bug: every record embeds `importer_sha256`, so editing `ingestion.py` changes the identity of every record it produces. Any change to the importer therefore requires clearing the importer-dependent artifacts — the canonical tree and the corpus manifest — and re-importing. The source snapshot, source lock, demonstration and population manifests do not depend on the importer hash and survive unchanged.

Treat this as the expected cost of touching the importer, and never work around it by relaxing the immutable-write guard.

## Step 6 — Re-import and regenerate

Re-run ingestion against the v2.0.1 clone. It rewrites:

- `data/corpus/<tag>/` — 102 exact Turtle byte copies
- `data/canonical/<tag>/` — 102 canonical records, plus the new metadata and readable files
- `data/manifests/corpus-<tag>.json`
- `data/manifests/corpus-source-lock-<tag>.json`
- `data/manifests/demonstrations-<tag>.yml`
- `data/manifests/evaluation-population-<tag>.yml`

Both releases are retained. Release-scoping every path meant the v2.0.0 tree and its manifests were never touched, so the four gold diffs can be inspected side by side. Three manifests had to be renamed to make this work — `corpus-source-lock-v1.json`, `demonstrations-v1.yml` and `evaluation-population-v1.yml` were not release-scoped and would have been overwritten by the immutable-write guard. Their filenames now carry the release while the `schema_version` field inside continues to carry the independent schema version. `parameters.yml` points at the new names.

The exact command executed was:

```bash
iadopt-lab --root iadopt-lab ingest --source-repository /Users/rastegar-a/Documents/GitHub/I-ADOPT-Variables
```

### Executed note — deterministic nested ordering

The first verification pass failed on a variable holding an asymmetric system. A record held in memory during ingestion keeps the insertion order produced by RDF projection, while the same record reloaded from its canonical file carries sorted keys, so the readable projection serialized two different byte sequences for identical content. `_ordered()` now sorts every nested mapping before serialization, which makes both paths agree. Verification caught this immediately, which is the reason the byte-identity check exists.

## Step 7 — Tests (executed)

`tests/unit/test_corpus.py` hardcodes `data/corpus/v2.0.0` and `data/manifests/corpus-v2.0.0.json` at lines 34, 36, 110, 124, and 131. Update these to the migrated paths, and add a regression asserting the four known v2.0.1 gold differences so a silent reversion to v2.0.0 content fails loudly:

- `C12_HabitatProbability` has `hasContextObject` and an empty `hasMatrix`
- `SurfRunoff` constraint targets match v2.0.1
- `NumChild` and `PersWelfare` use `condition: registered as resident`

Add coverage for the derived files (round-trip equality with the canonical parent, URI presence and omission, empty-field representation) and for readable predictions (attached `label`/`definition` are absent from the scored record and do not change its hash).

## Step 8 — Acceptance (results)

1. **Pass.** `pytest -q` reports 274 passed / 14 skipped. The three new tests cover the v2.0.1 gold differences, the derived projections, and readable predictions. The 14 skips are unchanged and remain the PostgreSQL integration and recovery tests.
2. **Pass.** `iadopt-lab preflight` reports the same eight draft-readiness issues as before the migration, now naming `openai/gpt-4o-mini` in place of the withdrawn model.
3. **Pass.** 102 corpus files, 5 demonstrations, 97 population members, 5 exclusions.
4. **Pass.** Every canonical record reports `tag: v2.0.1`, commit `2598bf91`, tree `df665d32`.
5. **Pass.** `iadopt-lab verify` checks 204 derived files, two per variable, all byte-identical to their parents.
6. **Pass.** The four changed variables match the upstream diff. `C12_HabitatProbability` now carries an empty `hasMatrix` and `hasContextObject: geographical area`, with `region: European Union` retargeted onto it; `SurfRunoff` splits its two constraints across `water` and `ground`; `NumChild` and `PersWelfare` both read `condition: registered as resident`.
7. **Pass.** `iadopt-lab plan --synthetic` expands 2 configurations, 2 runs, 6 tasks and an 18-request ceiling, exercising configuration, corpus, artifact and planning together with no database.

The regression-count guard proved its worth: the first import attempt refused with the exact computed counts, showing matrix 52→51 and context object 10→11 and every other value identical. That is the C12 change and nothing else, which is independent confirmation that the release diff is fully accounted for.

## Documents already updated for this migration

`README.md`, `DECISIONS.md` (D-030, D-031, D-032), `TECHNICAL_SPECIFICATION.md`, `docs/components/corpus-ingestion.md`, `docs/prompt-specification.md`, `docs/model-catalog.md`, `docs/components/providers.md`, and `data/README.md` describe the v2.0.1 release, the derived-file layout, and the revised OpenRouter selection, and now match the executed state. `THIRD_PARTY_NOTICES.md`, `data/corpus/LICENSE-PROVENANCE.md`, `data/README.md`, `docs/runbook.md`, `docs/test-plan.md`, `docs/architecture.md`, and `docs/database.md` were brought in line in a follow-up consistency pass. The audit records under `docs/documentation-audit.md` and `docs/repository-audit.md` keep their original v2.0.0 findings behind an explicit boundary note, because an audit must report what it actually examined.
