"""Pinned-source, strict RDF and complete population regressions."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from iadopt_lab.corpus import (
    DEMONSTRATION_PATHS,
    build_evaluation_population,
    ingest_corpus,
    load_canonical_records,
    parse_variable,
)
from iadopt_lab.corpus.ingestion import SOURCE_TAG, readable_projection, verify_derived_projections

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data/corpus" / SOURCE_TAG
MANIFEST_PATH = ROOT / "data/manifests" / f"corpus-{SOURCE_TAG}.json"
LOCK_NAME = f"corpus-source-lock-{SOURCE_TAG}.json"
TTL = '''@prefix iop: <https://w3id.org/iadopt/ont/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<urn:test:v> a iop:Variable; rdfs:label "temperature"; rdfs:comment "  Exact definition.\\n " ;
iop:hasProperty <urn:test:p>; iop:hasObjectOfInterest <urn:test:o>.
<urn:test:p> rdfs:label "temperature". <urn:test:o> rdfs:label "water".
'''


def test_all_pinned_records_and_categories():
    """Check all102 immutable records and exact directory-derived classification.

    Args: none. Returns: None.
    Raises: AssertionError or verification errors on corruption. Side effects: corpus reads.
    """
    records = load_canonical_records(ROOT)
    assert len(records) == 102
    assert len({record["variable_id"] for record in records}) == 102
    assert len({record["source_iri"] for record in records}) == 96
    for record in records:
        path = Path(record["source_path"])
        assert (record["category"], record["subcategory"], record["category_path"]) == (path.parts[0], path.parts[1], path.parent.as_posix())
        raw = (CORPUS_DIR / path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["source_sha256"]
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    assert manifest["regression_counts"]["constraints"] == 157
    assert manifest["regression_counts"]["ratio_systems"] == 31


def test_complete_population_and_demo_order():
    """Require all97 non-demonstration variables with exact fixed exclusions.

    Args: none. Returns: None.
    Raises: AssertionError for population drift. Side effects: corpus reads.
    """
    records = load_canonical_records(ROOT)
    population = build_evaluation_population(records)
    assert population["member_count"] == 97
    assert tuple(item["source_path"] for item in population["exclusions"]) == DEMONSTRATION_PATHS
    assert not {item["source_path"] for item in population["members"]} & set(DEMONSTRATION_PATHS)
    with pytest.raises(ValueError):
        build_evaluation_population(records, reversed(DEMONSTRATION_PATHS))
    with pytest.raises(ValueError):
        build_evaluation_population(records[:-1])


def test_preserved_definition_and_ambiguous_cardinality():
    """Preserve original definition whitespace and reject multiple required RDF values.

    Args: none. Returns: None.
    Raises: AssertionError for lossy projection. Side effects: none.
    """
    parsed = parse_variable(TTL)
    assert parsed["definition"] == "  Exact definition.\n "
    for suffix in ('<urn:test:v> rdfs:comment "second".',
                   '<urn:test:v> iop:hasProperty <urn:test:p2>.',
                   '<urn:test:v2> a iop:Variable.'):
        with pytest.raises(ValueError):
            parse_variable(TTL + suffix)


def test_unlabeled_whole_system_targets_are_stable():
    """Resolve unlabeled RDF system targets without leaking random blank-node IDs.

    Args: none. Returns: None.
    Raises: AssertionError for unstable role or Constraint projection. Side effects: none.
    """
    ttl = TTL.replace('<urn:test:o> rdfs:label "water".', '''<urn:test:o> a iop:AsymmetricSystem;
    iop:hasNumerator _:a; iop:hasDenominator _:b.
    _:a rdfs:label "water". _:b rdfs:label "air".
    <urn:test:v> iop:hasConstraint _:c.
    _:c rdfs:label "reference: exact"; iop:constrains <urn:test:o>.''')
    first = parse_variable(ttl)
    second = parse_variable(ttl.replace("_:a", "_:xyz").replace("_:b", "_:zyx"))
    assert first["gold"] == second["gold"]
    assert first["gold"]["hasConstraint"] == [{"label": "reference: exact", "on": "water / air"}]
    assert "hasNumerator" in first["gold"]["hasObjectOfInterest"]


def test_missing_constraint_target_is_not_invented():
    """Reject constraints lacking explicit targets instead of using a legacy fallback.

    Args: none. Returns: None.
    Raises: AssertionError for accepted missing target. Side effects: none.
    """
    with pytest.raises(ValueError):
        parse_variable(TTL + '<urn:test:v> iop:hasConstraint _:c. _:c rdfs:label "wet".')


def test_ingest_idempotent_offline_source_directory(tmp_path):
    """Verify a fresh ordinary source directory and resume without Git or downloads.

    Args: pytest temporary writable directory. Returns: None.
    Raises: AssertionError on different identities or source mutation. Side effects: isolated fixture writes.
    """
    shutil.copytree(ROOT / "schemas", tmp_path / "schemas")
    (tmp_path / "data/manifests").mkdir(parents=True)
    shutil.copyfile(ROOT / "data/manifests" / LOCK_NAME, tmp_path / "data/manifests" / LOCK_NAME)
    result = ingest_corpus(tmp_path, source_directory=CORPUS_DIR)
    again = ingest_corpus(tmp_path)
    assert result["manifest"] == again["manifest"]
    assert len(load_canonical_records(tmp_path)) == 102
    assert result["evaluation_population"]["member_count"] == 97


def test_corrupt_source_fails_before_activation(tmp_path):
    """Reject one altered source byte without publishing an active corpus manifest.

    Args: pytest temporary directory. Returns: None.
    Raises: AssertionError if corruption is accepted. Side effects: isolated fixture copies/one-byte mutation.
    """
    shutil.copytree(ROOT / "schemas", tmp_path / "schemas")
    shutil.copytree(CORPUS_DIR, tmp_path / "source")
    (tmp_path / "data/manifests").mkdir(parents=True)
    shutil.copyfile(ROOT / "data/manifests" / LOCK_NAME, tmp_path / "data/manifests" / LOCK_NAME)
    first = sorted((tmp_path / "source").rglob("*.ttl"))[0]
    first.write_bytes(first.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Source bytes"):
        ingest_corpus(tmp_path, source_directory=tmp_path / "source")
    assert not (tmp_path / "data/manifests" / f"corpus-{SOURCE_TAG}.json").exists()


def test_v201_gold_differences_are_present():
    """Pin the four upstream gold changes that distinguish v2.0.1 from v2.0.0.

    Args: none. Returns: None.
    Raises: AssertionError if the corpus silently reverts to v2.0.0 content.
    Side effects: corpus reads.
    """
    records = {record["source_path"]: record for record in load_canonical_records(ROOT)}
    habitat = records["Life Sciences/Ecology/C12_HabitatProbability.ttl"]["gold"]
    assert habitat["hasMatrix"] == "" and habitat["hasContextObject"] == "geographical area"
    assert {"label": "region: European Union", "on": "geographical area"} in habitat["hasConstraint"]
    # v2.0.0 pointed both constraints at "ground"; v2.0.1 retargets the condition to "water".
    runoff = records["Natural Sciences/Hydrology/SurfRunoff.ttl"]["gold"]
    assert {"label": "condition: not soaking into the ground", "on": "water"} in runoff["hasConstraint"]
    assert {"label": "part: surface", "on": "ground"} in runoff["hasConstraint"]
    for path in ("Social Sciences/Demography/NumChild.ttl", "Social Sciences/Demography/PersWelfare.ttl"):
        labels = [item["label"] for item in records[path]["gold"]["hasConstraint"]]
        assert "condition: registered as resident" in labels
        assert not any(label.startswith("condition: registered as residents") for label in labels)


def test_derived_projections_match_their_canonical_parent():
    """Require both derived views per variable to regenerate byte-identically.

    Args: none. Returns: None.
    Raises: AssertionError or ValueError for a missing or divergent projection.
    Side effects: corpus reads.
    """
    records = load_canonical_records(ROOT)
    assert verify_derived_projections(ROOT, records) == 2 * len(records)
    readable = readable_projection(records[0])
    assert list(readable)[:2] == ["label", "definition"]
    for field in ("hasProperty", "hasStatisticalModifier", "hasObjectOfInterest",
                  "hasMatrix", "hasContextObject", "hasConstraint"):
        assert field in readable
    with_uri = next(row for row in records if "hasProperty" in (row.get("iris") or {}))
    projected = readable_projection(with_uri)
    assert projected["hasPropertyURI"] == with_uri["iris"]["hasProperty"]
    without = next(row for row in records if "hasMatrix" not in (row.get("iris") or {}))
    assert "hasMatrixURI" not in readable_projection(without)
