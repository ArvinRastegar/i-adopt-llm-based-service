# Third-Party Sources and Notices

This file records external materials copied or referenced by I-ADOPT Lab. The corpus snapshot below is materialized and hash-verified; the deferred items further down are not.

## I-ADOPT Corpus

- Source: `https://github.com/i-adopt/Corpus`
- Release: `v2.0.1` (D-030), superseding `v2.0.0`
- Commit: `2598bf91fa927b78a6529bae7864ef0f7d485b73`
- Tree: `df665d32bb2433a60742c80a4a53908dc7ebde0c`
- Concept DOI: `10.5281/zenodo.18101358`, taken from the release README badge. A concept DOI is release-independent and resolves to the latest version.
- License: Creative Commons Attribution 4.0 International (`CC-BY-4.0`)
- Local location: `data/corpus/v2.0.1/`, with the retained `v2.0.0/` snapshot alongside it
- Status: materialized. All 102 Turtle files are byte-verified against the pinned tree, and `iadopt-lab verify` rechecks every hash.

**Open item — version DOI.** The stored source lock records the license source as `https://doi.org/10.5281/zenodo.22011435`, which is the version DOI of the **v2.0.0** Zenodo record. It is hardcoded at `src/iadopt_lab/corpus/ingestion.py:450` and was not updated with the release pin, so the v2.0.1 lock and manifest currently cite a v2.0.0 version DOI. The concept DOI above is correct for either release. The v2.0.1 version DOI must be supplied and recorded before publication; it must not be guessed. Correcting it changes the importer, which changes every canonical record hash and requires a re-import.

The Git tag contains no standalone `LICENSE` file. The implementation must not invent one or infer licensing only from the README image; it uses the explicit Zenodo record and records the metadata retrieval date.

## I-ADOPT ontology

- Documentation: `https://i-adopt.github.io/ontology/`
- Active experiment use: none
- Deferred use: structural source for a future deterministic baseline

## Historical experiment code

- Source repository: the parent `i-adopt-llm-based-service` repository
- Intended local location: `reference/legacy/`
- Purpose: provenance and regression evidence only; never imported by the active runtime

## Deferred validation resources

The supplied Variable JSON Schema, JSON-LD context, and two SHACL Turtle files may be copied to `reference/deferred-rdf-validation/` during implementation. They remain outside the active lexical-generation and evaluation workflow.
