"""Materialize a verified 102-variable corpus without mutable upstream assumptions."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from rdflib import RDF, RDFS, Graph, Literal, Namespace, URIRef

from iadopt_lab.validation.lexical import (
    empty_prediction,
    lexical_order,
    load_schema_bytes,
    system_display_label,
    validate_prediction,
)

IOP = Namespace("https://w3id.org/iadopt/ont/")
SOURCE_REPOSITORY = "https://github.com/i-adopt/Corpus"
SOURCE_COMMIT = "2598bf91fa927b78a6529bae7864ef0f7d485b73"
SOURCE_TREE = "df665d32bb2433a60742c80a4a53908dc7ebde0c"
SOURCE_TAG = "v2.0.1"
IMPORTER_VERSION = "corpus-lexical-v1"
DEMONSTRATION_PATHS = (
    "Natural Sciences/Atmospheric Science/C2_AirDailyMaximumTemperature.ttl",
    "Social Sciences/Demography/PersWelfare.ttl",
    "Life Sciences/Health Science/lactate.ttl",
    "Technical Sciences/Material Science/CirculationMode-Water.ttl",
    "Social Sciences/Disaster Risk Science/HeatStress.ttl",
)


def _corpus_dir(root: Path) -> Path:
    """Locate the immutable Turtle snapshot directory for the pinned release."""
    return root / "data/corpus" / SOURCE_TAG


def _canonical_dir(root: Path) -> Path:
    """Locate the canonical record directory for the pinned release."""
    return root / "data/canonical" / SOURCE_TAG


def _manifest_path(root: Path) -> Path:
    """Locate the corpus manifest for the pinned release."""
    return root / "data/manifests" / f"corpus-{SOURCE_TAG}.json"


def _lock_path(root: Path) -> Path:
    """Locate the portable source-byte lock for the pinned release."""
    return root / "data/manifests" / f"corpus-source-lock-{SOURCE_TAG}.json"


def _demonstrations_path(root: Path) -> Path:
    """Locate the ordered demonstration manifest for the pinned release."""
    return root / "data/manifests" / f"demonstrations-{SOURCE_TAG}.yml"


def _population_path(root: Path) -> Path:
    """Locate the evaluation-population manifest for the pinned release."""
    return root / "data/manifests" / f"evaluation-population-{SOURCE_TAG}.yml"


def _canonical(value: Any) -> bytes:
    """Serialize provenance deterministically."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _hash(value: bytes) -> str:
    """Hash exact bytes without normalization."""
    return hashlib.sha256(value).hexdigest()


def _one(graph: Graph, node: Any, predicate: Any, required: bool = True) -> Any:
    """Read one RDF predicate with explicit cardinality enforcement."""
    values = set(graph.objects(node, predicate))
    if len(values) > 1 or (required and not values):
        raise ValueError(f"Invalid cardinality for {predicate}: {len(values)}")
    return next(iter(values)) if values else None


def _label(graph: Graph, node: Any) -> str:
    """Resolve a required exact lexical RDF label."""
    value = _one(graph, node, RDFS.label)
    if not isinstance(value, Literal) or not str(value).strip():
        raise ValueError("Required lexical label must be a nonempty literal")
    return str(value)


def _entity(graph: Graph, node: Any, targets: dict[Any, str]) -> Any:
    """Project one simple entity or supported flat system and register targets."""
    symmetric = (node, RDF.type, IOP.SymmetricSystem) in graph
    asymmetric = (node, RDF.type, IOP.AsymmetricSystem) in graph
    roles = (IOP.hasSource, IOP.hasTarget, IOP.hasNumerator, IOP.hasDenominator)
    if symmetric and asymmetric:
        raise ValueError("Entity cannot have both symmetric and asymmetric types")
    if not symmetric and not asymmetric:
        if any(graph.objects(node, IOP.hasPart)) or any(_one(graph, node, role, False) is not None for role in roles):
            raise ValueError("System predicates require an explicit supported system type")
        targets[node] = _label(graph, node)
        return targets[node]

    def member(term: Any) -> str:
        """Resolve a flat system member without recursive nesting."""
        if term is None or (term, RDF.type, IOP.SymmetricSystem) in graph or (term, RDF.type, IOP.AsymmetricSystem) in graph:
            raise ValueError("Missing or nested system member is unsupported")
        targets[term] = _label(graph, term)
        return targets[term]

    if symmetric:
        if any(_one(graph, node, role, False) is not None for role in roles):
            raise ValueError("Symmetric system has asymmetric role predicates")
        parts = set(graph.objects(node, IOP.hasPart))
        if len(parts) < 2:
            raise ValueError("Symmetric system requires at least two members")
        result = {"SymmetricSystem": "", "hasPart": sorted((member(part) for part in parts), key=lexical_order)}
    else:
        if any(graph.objects(node, IOP.hasPart)):
            raise ValueError("Asymmetric system cannot carry symmetric parts")
        present = [_one(graph, node, role, False) for role in roles]
        if present[0] is not None and present[1] is not None and present[2:] == [None, None]:
            result = {"AsymmetricSystem": "", "hasSource": member(present[0]), "hasTarget": member(present[1])}
        elif present[2] is not None and present[3] is not None and present[:2] == [None, None]:
            result = {"AsymmetricSystem": "", "hasNumerator": member(present[2]), "hasDenominator": member(present[3])}
        else:
            raise ValueError("Asymmetric system requires exactly one complete role pair")
    container = "SymmetricSystem" if symmetric else "AsymmetricSystem"
    # Original system labels are provenance-only; absence is permitted for systems.
    _one(graph, node, RDFS.label, False)
    result[container] = system_display_label(result)
    targets[node] = result[container]
    return result


def parse_variable(ttl_bytes: bytes | str, source_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse one Turtle variable with exact definition and objective invariants."""
    raw = ttl_bytes.encode("utf-8") if isinstance(ttl_bytes, str) else ttl_bytes
    graph = Graph().parse(data=raw.decode("utf-8"), format="turtle")
    roots = set(graph.subjects(RDF.type, IOP.Variable))
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one Variable root, received {len(roots)}")
    root = next(iter(roots))
    comment = _one(graph, root, RDFS.comment)
    if not isinstance(comment, Literal) or not str(comment).strip():
        raise ValueError("Variable definition must be a nonempty rdfs:comment literal")
    targets: dict[Any, str] = {}
    iris: dict[str, str] = {}
    gold = empty_prediction()
    for field in ("hasProperty", "hasStatisticalModifier"):
        node = _one(graph, root, IOP[field], field == "hasProperty")
        if node is not None:
            gold[field] = _label(graph, node)
            targets[node] = gold[field]
            if isinstance(node, URIRef):
                iris[field] = str(node)
    for field in ("hasObjectOfInterest", "hasMatrix", "hasContextObject"):
        node = _one(graph, root, IOP[field], field == "hasObjectOfInterest")
        if node is not None:
            gold[field] = _entity(graph, node, targets)
            # A system container is normally a blank node and has no single IRI.
            if isinstance(node, URIRef) and isinstance(gold[field], str):
                iris[field] = str(node)
    for constraint in set(graph.objects(root, IOP.hasConstraint)):
        target = _one(graph, constraint, IOP.constrains)
        if target not in targets:
            raise ValueError("Constraint must target an emitted component or system member")
        gold["hasConstraint"].append({"label": _label(graph, constraint), "on": targets[target]})
    gold["hasConstraint"].sort(key=lambda value: (lexical_order(value["label"]), lexical_order(value["on"]), _canonical(value)))
    identity = str(root) if isinstance(root, URIRef) else "urn:iadopt-lab:variable:" + _hash(raw)
    return {"variable_id": identity, "label": _label(graph, root), "definition": str(comment),
            "definition_predicate": str(RDFS.comment), "gold": gold,
            "iris": dict(sorted(iris.items())),
            "source_identity": dict(source_identity or {})}


def project_gold(parsed: dict[str, Any], schema_bytes: bytes | None = None) -> dict[str, Any]:
    """Validate the strict graph projection and bind its canonical gold hash."""
    result = validate_prediction(parsed["gold"], schema_bytes)
    if not result.valid:
        raise ValueError("Corpus gold fails lexical validation: " + json.dumps(result.errors))
    return {**parsed, "gold": result.canonical_prediction,
            "gold_sha256": _hash(_canonical(result.canonical_prediction)),
            "lexical_schema_sha256": result.schema_sha256}


_METADATA_FIELDS = (
    "schema_version", "variable_id", "source_iri", "source_path", "category", "subcategory",
    "category_path", "repository", "tag", "commit", "tree", "git_blob", "source_byte_length",
    "source_sha256", "gold_sha256", "record_sha256", "definition_predicate",
    "importer_version", "importer_sha256", "lexical_schema_sha256", "demonstration_position",
)
_READABLE_ORDER = ("hasProperty", "hasStatisticalModifier", "hasObjectOfInterest",
                   "hasMatrix", "hasContextObject", "hasConstraint")


def _ordered(value: Any) -> Any:
    """Normalize nested mapping order so a projection is byte-stable."""
    if isinstance(value, dict):
        return {key: _ordered(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_ordered(item) for item in value]
    return value


def metadata_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Project provenance-only metadata from one canonical record."""
    return {field: record[field] for field in _METADATA_FIELDS}


def readable_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Project the human-readable variable view from one canonical record."""
    iris = record.get("iris") or {}
    readable: dict[str, Any] = {"label": record["label"], "definition": record["definition"]}
    for field in _READABLE_ORDER:
        readable[field] = _ordered(record["gold"][field])
        if field in iris:
            readable[field + "URI"] = iris[field]
    return readable


def _readable_bytes(record: dict[str, Any]) -> bytes:
    """Serialize the readable projection deterministically for humans."""
    return (json.dumps(readable_projection(record), indent=2, ensure_ascii=False,
                       allow_nan=False, sort_keys=False) + "\n").encode("utf-8")


def _metadata_bytes(record: dict[str, Any]) -> bytes:
    """Serialize the metadata projection deterministically."""
    return (json.dumps(metadata_projection(record), indent=2, ensure_ascii=False,
                       allow_nan=False, sort_keys=False) + "\n").encode("utf-8")


def _derived_paths(root: Path, record: dict[str, Any]) -> tuple[Path, Path]:
    """Locate the two derived files that sit beside one canonical record."""
    stem = _canonical_dir(root) / Path(record["source_path"]).with_suffix("")
    return stem.with_suffix(".meta.json"), stem.with_suffix(".readable.json")


def verify_derived_projections(project_root: Path | str,
                               records: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> int:
    """Confirm every derived file regenerates byte-identically from its parent."""
    root = Path(project_root)
    checked = 0
    for record in records:
        metadata_path, readable_path = _derived_paths(root, record)
        for path, expected in ((metadata_path, _metadata_bytes(record)),
                               (readable_path, _readable_bytes(record))):
            if not path.is_file():
                raise ValueError(f"Derived projection is missing: {path}")
            if path.read_bytes() != expected:
                raise ValueError(f"Derived projection does not match its canonical parent: {path}")
            checked += 1
    return checked


def _git(repository: Path, *arguments: str) -> bytes:
    """Read immutable Git objects using argument-safe subprocess execution."""
    return subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True).stdout


def enumerate_tag_files(source: Path | str) -> tuple[dict[str, Any], ...]:
    """Read the pinned Turtle blobs from an existing Git object database."""
    repository = Path(source)
    commit = _git(repository, "rev-parse", SOURCE_TAG + "^{commit}").decode().strip()
    tree = _git(repository, "rev-parse", SOURCE_TAG + "^{tree}").decode().strip()
    if (commit, tree) != (SOURCE_COMMIT, SOURCE_TREE):
        raise ValueError("Corpus tag does not resolve to the frozen commit/tree")
    result = []
    for entry in _git(repository, "ls-tree", "-rz", SOURCE_TREE).split(b"\0"):
        if not entry:
            continue
        metadata, path_bytes = entry.split(b"\t", 1)
        path = path_bytes.decode("utf-8")
        if not path.endswith(".ttl"):
            continue
        mode, kind, blob = metadata.decode().split()
        if mode != "100644" or kind != "blob":
            raise ValueError("Source Turtle must be an ordinary Git blob")
        raw = _git(repository, "cat-file", "blob", blob)
        result.append({"source_path": path, "git_blob": blob, "source_sha256": _hash(raw),
                       "source_byte_length": len(raw), "raw_bytes": raw})
    result.sort(key=lambda value: value["source_path"].encode("utf-8"))
    if len(result) != 102:
        raise ValueError("Pinned source must contain exactly 102 Turtle files")
    return tuple(result)


def build_source_lock(source_repository: Path | str) -> dict[str, Any]:
    """Build portable source byte evidence from verified immutable Git objects."""
    files = [{key: value for key, value in blob.items() if key != "raw_bytes"}
             for blob in enumerate_tag_files(source_repository)]
    lock = {"schema_version": "corpus-source-lock-v1", "repository": SOURCE_REPOSITORY,
            "tag": SOURCE_TAG, "commit": SOURCE_COMMIT, "tree": SOURCE_TREE,
            "license": {"identifier": "CC-BY-4.0", "source": "https://doi.org/10.5281/zenodo.22011435"},
            "file_count": 102, "files": files}
    return {**lock, "manifest_sha256": _hash(_canonical(lock))}


def _check_manifest_hash(manifest: dict[str, Any]) -> None:
    """Verify a manifest's non-self-referential canonical content hash."""
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _hash(_canonical(content)):
        raise ValueError("Manifest content hash mismatch")


def _validate_source_lock(lock: dict[str, Any]) -> None:
    """Validate the frozen source descriptor and complete safe path inventory."""
    _check_manifest_hash(lock)
    if (lock.get("repository"), lock.get("tag"), lock.get("commit"), lock.get("tree")) != (SOURCE_REPOSITORY, SOURCE_TAG, SOURCE_COMMIT, SOURCE_TREE):
        raise ValueError("Source lock does not match approved Corpus release")
    paths = [file["source_path"] for file in lock["files"]]
    if len(paths) != 102 or len(set(paths)) != 102 or paths != sorted(paths, key=lambda value: value.encode("utf-8")):
        raise ValueError("Source lock inventory is not the ordered 102-file set")
    for path in paths:
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or len(parsed.parts) < 3 or not path.endswith(".ttl"):
            raise ValueError("Unsafe or category-less source path")


def _read_source_directory(directory: Path, lock: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Read ordinary source files only after exact release inventory verification."""
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*.ttl")}
    expected = {entry["source_path"] for entry in lock["files"]}
    if actual != expected:
        raise ValueError("Source directory Turtle inventory differs from the frozen release")
    result = []
    for entry in lock["files"]:
        path = directory / entry["source_path"]
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != directory.parent):
            raise ValueError("Symlinked source files/directories are not allowed")
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if (_hash(raw), len(raw), blob) != (entry["source_sha256"], entry["source_byte_length"], entry["git_blob"]):
            raise ValueError("Source bytes do not match frozen release: " + entry["source_path"])
        result.append({**entry, "raw_bytes": raw})
    return tuple(result)


def build_evaluation_population(records: list[dict[str, Any]] | tuple[dict[str, Any], ...], demonstrations: Any = None) -> dict[str, Any]:
    """Build the complete fixed evaluation population with five explicit exclusions."""
    paths = tuple(demonstrations) if demonstrations is not None else DEMONSTRATION_PATHS
    if paths != DEMONSTRATION_PATHS:
        raise ValueError("Demonstrations must use the approved five-path order")
    indexed = {record["source_path"]: record for record in records}
    if len(records) != 102 or len(indexed) != 102 or not set(paths) <= set(indexed):
        raise ValueError("Population requires 102 unique records and all demonstrations")
    members = []
    excluded = []
    for path in sorted(indexed, key=lambda value: value.encode("utf-8")):
        record = indexed[path]
        entry = {key: record[key] for key in ("variable_id", "source_path", "source_sha256", "gold_sha256")}
        if path in paths:
            excluded.append({**entry, "reason": "fixed_demonstration", "position": paths.index(path) + 1})
        else:
            members.append(entry)
    excluded.sort(key=lambda value: value["position"])
    manifest = {"schema_version": "evaluation-population-v1", "commit": SOURCE_COMMIT,
                "ordering": "source-path-utf8-v1", "member_count": len(members),
                "exclusion_count": len(excluded), "members": members, "exclusions": excluded}
    return {**manifest, "manifest_sha256": _hash(_canonical(manifest))}


def _regression_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count the independently audited structural Corpus facts."""
    result = {"variables": len(records), "matrix": 0, "context_object": 0,
              "statistical_modifier": 0, "variables_with_constraints": 0, "constraints": 0,
              "symmetric_systems": 0, "symmetric_parts": 0, "asymmetric_systems": 0,
              "ratio_systems": 0, "source_target_systems": 0}
    for record in records:
        gold = record["gold"]
        for field, counter in (("hasMatrix", "matrix"), ("hasContextObject", "context_object"), ("hasStatisticalModifier", "statistical_modifier"), ("hasConstraint", "variables_with_constraints")):
            result[counter] += bool(gold[field])
        result["constraints"] += len(gold["hasConstraint"])
        for field in ("hasObjectOfInterest", "hasMatrix", "hasContextObject"):
            entity = gold[field]
            if isinstance(entity, dict):
                if "SymmetricSystem" in entity:
                    result["symmetric_systems"] += 1
                    result["symmetric_parts"] += len(entity["hasPart"])
                else:
                    result["asymmetric_systems"] += 1
                    result["ratio_systems" if "hasNumerator" in entity else "source_target_systems"] += 1
    return result


def _write_immutable(path: Path, raw: bytes) -> None:
    """Atomically publish exact artifact bytes without overwriting conflicts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Immutable artifact conflicts with existing bytes: " + str(path))
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".iadopt-stage-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError("Concurrent immutable artifact conflict")
    finally:
        os.unlink(temporary)


def ingest_corpus(project_root: Path | str, source_repository: Path | str | None = None,
                  source_directory: Path | str | None = None) -> dict[str, Any]:
    """Verify and materialize a complete pinned corpus bundle for DB activation."""
    root = Path(project_root)
    lock_path = _lock_path(root)
    if source_repository is not None:
        lock = build_source_lock(source_repository)
        if lock_path.exists() and json.loads(lock_path.read_bytes()) != lock:
            raise ValueError("Bundled source lock differs from verified Git source")
    else:
        lock = json.loads(lock_path.read_bytes())
    _validate_source_lock(lock)
    if source_directory is not None:
        blobs = _read_source_directory(Path(source_directory), lock)
    elif source_repository is not None:
        blobs = enumerate_tag_files(source_repository)
    else:
        blobs = _read_source_directory(_corpus_dir(root), lock)
    schema_bytes = load_schema_bytes(root)
    importer_hash = _hash(Path(__file__).read_bytes())
    records = []
    for blob in blobs:
        path = PurePosixPath(blob["source_path"])
        projected = project_gold(parse_variable(blob["raw_bytes"]), schema_bytes)
        record = {key: value for key, value in projected.items() if key != "source_identity"}
        record["source_iri"] = record["variable_id"]
        record["variable_id"] = "urn:iadopt-lab:variable:" + _hash((SOURCE_COMMIT + "\n" + str(path)).encode("utf-8"))
        record.update({"schema_version": "corpus-record-v2", "source_path": str(path),
                       "category": path.parts[0], "subcategory": path.parts[1],
                       "category_path": str(path.parent), "source_sha256": blob["source_sha256"],
                       "source_byte_length": blob["source_byte_length"], "git_blob": blob["git_blob"],
                       "repository": SOURCE_REPOSITORY, "tag": SOURCE_TAG, "commit": SOURCE_COMMIT,
                       "tree": SOURCE_TREE, "importer_version": IMPORTER_VERSION,
                       "importer_sha256": importer_hash,
                       "demonstration_position": DEMONSTRATION_PATHS.index(str(path)) + 1 if str(path) in DEMONSTRATION_PATHS else None})
        record["record_sha256"] = _hash(_canonical(record))
        records.append(record)
    if len({record["variable_id"] for record in records}) != 102:
        raise ValueError("Derived variable identities must uniquely identify all corpus records")
    counts = _regression_counts(records)
    # v2.0.1 (D-030) moved C12_HabitatProbability from hasMatrix to hasContextObject.
    # That single upstream edit is the only structural difference from v2.0.0, whose
    # counts were matrix 52 / context_object 10 with every other value identical.
    expected = {"variables": 102, "matrix": 51, "context_object": 11, "statistical_modifier": 9,
                "variables_with_constraints": 85, "constraints": 157, "symmetric_systems": 2,
                "symmetric_parts": 4, "asymmetric_systems": 36, "ratio_systems": 31, "source_target_systems": 5}
    if counts != expected:
        raise ValueError("Corpus structural regression mismatch: " + json.dumps(counts))
    population = build_evaluation_population(records)
    demos = {"schema_version": "demonstrations-v1", "commit": SOURCE_COMMIT,
             "demonstrations": [{key: record[key] for key in ("variable_id", "source_path", "source_sha256", "gold_sha256", "demonstration_position")}
                                for path in DEMONSTRATION_PATHS for record in records if record["source_path"] == path]}
    demos["manifest_sha256"] = _hash(_canonical(demos))
    manifest = {"schema_version": "corpus-manifest-v1", "repository": SOURCE_REPOSITORY,
                "tag": SOURCE_TAG, "commit": SOURCE_COMMIT, "tree": SOURCE_TREE,
                "license": lock["license"], "source_lock_sha256": lock["manifest_sha256"],
                "importer_version": IMPORTER_VERSION, "importer_sha256": importer_hash,
                "lexical_schema_sha256": _hash(schema_bytes), "regression_counts": counts,
                "file_count": 102, "demonstrations_sha256": demos["manifest_sha256"],
                "evaluation_population_sha256": population["manifest_sha256"],
                "files": [{key: record[key] for key in ("source_path", "git_blob", "source_sha256", "source_byte_length", "variable_id", "category", "subcategory", "category_path", "gold_sha256", "record_sha256")}
                          for record in records]}
    manifest["manifest_sha256"] = _hash(_canonical(manifest))
    for name, value in (("corpus-manifest.schema.json", manifest), ("evaluation-population.schema.json", population)):
        Draft202012Validator(json.loads((root / "schemas" / name).read_bytes())).validate(value)
    # Complete validation precedes every materialization. The corpus manifest is last.
    _write_immutable(lock_path, _canonical(lock))
    for blob, record in zip(blobs, records, strict=True):
        _write_immutable(_corpus_dir(root) / blob["source_path"], blob["raw_bytes"])
        _write_immutable(_canonical_dir(root) / Path(record["source_path"]).with_suffix(".json"), _canonical(record))
        metadata_path, readable_path = _derived_paths(root, record)
        _write_immutable(metadata_path, _metadata_bytes(record))
        _write_immutable(readable_path, _readable_bytes(record))
    _write_immutable(_demonstrations_path(root), yaml.safe_dump(demos, allow_unicode=True, sort_keys=True).encode("utf-8"))
    _write_immutable(_population_path(root), yaml.safe_dump(population, allow_unicode=True, sort_keys=True).encode("utf-8"))
    _write_immutable(_manifest_path(root), _canonical(manifest))
    return {"manifest": manifest, "records": records, "demonstrations": demos, "evaluation_population": population}


def _verify_saved_manifest(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    """Read one saved manifest and prove it matches both itself and the corpus manifest."""
    if not path.is_file():
        raise ValueError(f"Saved {label} manifest is missing: {path}")
    saved = yaml.safe_load(path.read_bytes())
    if not isinstance(saved, dict) or "manifest_sha256" not in saved:
        raise ValueError(f"Saved {label} manifest is malformed: {path}")
    content = {key: value for key, value in saved.items() if key != "manifest_sha256"}
    if _hash(_canonical(content)) != saved["manifest_sha256"]:
        raise ValueError(f"Saved {label} manifest fails its own hash: {path}")
    if saved["manifest_sha256"] != expected_hash:
        raise ValueError(f"Saved {label} manifest differs from the activated corpus manifest")
    return saved


def load_canonical_records(project_root: Path | str) -> tuple[dict[str, Any], ...]:
    """Load an activated bundle while rechecking source, record and gold evidence."""
    root = Path(project_root)
    manifest = json.loads(_manifest_path(root).read_bytes())
    _check_manifest_hash(manifest)
    lock = json.loads(_lock_path(root).read_bytes())
    _validate_source_lock(lock)
    if manifest["source_lock_sha256"] != lock["manifest_sha256"]:
        raise ValueError("Corpus and source lock identities differ")
    _read_source_directory(_corpus_dir(root), lock)
    schema = load_schema_bytes(root)
    if manifest["lexical_schema_sha256"] != _hash(schema):
        raise ValueError("Corpus lexical schema identity differs from runtime")
    records = []
    for entry in manifest["files"]:
        record = json.loads((_canonical_dir(root) / Path(entry["source_path"]).with_suffix(".json")).read_bytes())
        content = {key: value for key, value in record.items() if key != "record_sha256"}
        if record["record_sha256"] != _hash(_canonical(content)) or record["record_sha256"] != entry["record_sha256"] or _hash(_canonical(record["gold"])) != record["gold_sha256"]:
            raise ValueError("Canonical corpus record hash mismatch")
        if not validate_prediction(record["gold"], schema).valid:
            raise ValueError("Canonical corpus gold is invalid")
        records.append(record)
    _verify_saved_manifest(_demonstrations_path(root), manifest["demonstrations_sha256"], "demonstration")
    saved_population = _verify_saved_manifest(
        _population_path(root), manifest["evaluation_population_sha256"], "evaluation-population")
    # The saved manifest must also agree with what the records themselves imply, so a
    # self-consistent but stale file cannot pass. Comparing the rebuilt population to
    # the stored one closes that gap; the rebuild alone was previously discarded.
    if build_evaluation_population(records) != saved_population:
        raise ValueError("Saved evaluation-population manifest disagrees with the canonical records")
    return tuple(records)
