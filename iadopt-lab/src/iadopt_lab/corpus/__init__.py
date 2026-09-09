"""Pinned source ingestion, strict RDF projection and corpus provenance."""

from .ingestion import (
    DEMONSTRATION_PATHS,
    SOURCE_COMMIT,
    SOURCE_TREE,
    build_evaluation_population,
    ingest_corpus,
    load_canonical_records,
    parse_variable,
    project_gold,
)

__all__ = ["DEMONSTRATION_PATHS", "SOURCE_COMMIT", "SOURCE_TREE",
           "build_evaluation_population", "ingest_corpus", "load_canonical_records",
           "parse_variable", "project_gold"]
