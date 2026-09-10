"""Portable input/code/runtime receipts and atomic, confined derived artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path

from .canonical import canonical_json_bytes, content_hash, sha256_bytes
from .domain import LabError


def atomic_write(path: str | Path, content: bytes, *, root: str | Path, immutable: bool = True) -> dict:
    """Write a confined artifact atomically and verify the resulting bytes.

    Args: path: Explicit destination; content: complete bytes; root: allowed output subtree;
        immutable: refuse different existing bytes when True, replace derivatives otherwise.
    Returns: Relative path, exact SHA-256 and byte length.
    Raises: LabError for escape/conflicting immutable output; OSError for I/O failures.
    Side Effects: Creates parent directories and one file via fsync/atomic rename. No broad deletion.
    """
    boundary, destination = Path(root).resolve(), Path(path).resolve()
    if not destination.is_relative_to(boundary) or destination == boundary:
        raise LabError("Artifact destination must be inside its declared subtree")
    if destination.exists() and immutable and destination.read_bytes() != content:
        raise LabError("Immutable artifact already exists with different bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.read_bytes() != content:
        descriptor, temporary = tempfile.mkstemp(prefix=".iadopt-", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if destination.read_bytes() != content:
        raise LabError("Post-write artifact verification failed")
    return {"path": destination.relative_to(boundary).as_posix(), "sha256": sha256_bytes(content),
            "byte_length": len(content)}


def collect_input_artifacts(project_root: str | Path, records: list[dict] | tuple[dict, ...],
                            population: list[dict] | tuple[dict, ...], similarity_identity: dict) -> dict:
    """Read portable scientific artifacts and the exact implementation/runtime identity.

    Args: project_root: Lab root; records: all canonical corpus rows; population: selected targets;
        similarity_identity: Explicit scorer similarity identity, including synthetic status.
    Returns: Artifact byte records plus hashes required by planning and runtime verification.
    Raises: OSError for missing required files; LabError for an incomplete corpus or no source code.
    Side Effects: Local file/package metadata reads only. Secrets, absolute paths and timestamps excluded.
    """
    root = Path(project_root).resolve()
    if len(records) != 102 or not population:
        raise LabError("Artifacts require the complete 102-row corpus and a nonempty population")
    paths = [root / "uv.lock", root / "pyproject.toml", root / ".python-version"]
    for directory, pattern in (("src", "*.py"), ("schemas", "*.json"), ("prompts", "*.txt"),
                               ("migrations", "*.sql"), ("data/manifests", "*")):
        paths.extend(path for path in (root / directory).rglob(pattern) if path.is_file())
    artifacts = [{"kind": "workspace-file", "path": path.relative_to(root).as_posix(),
                  "content": path.read_bytes()} for path in sorted(set(paths))]
    index = {row["path"]: sha256_bytes(row["content"]) for row in artifacts}
    if not any(path.startswith("src/") for path in index):
        raise LabError("No executable source artifacts found")
    runtime = {"python": platform.python_version(), "implementation": platform.python_implementation(),
        "system": platform.system(), "machine": platform.machine(),
        "packages": dict(sorted((distribution.metadata["Name"].lower(), distribution.version)
                                for distribution in importlib.metadata.distributions()
                                if distribution.metadata["Name"])),
        "dependency_lock_sha256": index["uv.lock"]}
    runtime_bytes = canonical_json_bytes(runtime)
    artifacts.append({"kind": "runtime", "path": "runtime-v1.json", "content": runtime_bytes})
    # The original YAML is evidence in its own right: the resolved configuration stored in
    # the plan cannot reproduce the comments recording why each value was chosen. It is
    # appended after the index is built, and so stays out of the implementation identity -
    # a comment-only edit should preserve the bytes without invalidating a running
    # campaign's resume, which compares that identity.
    artifacts.append({"kind": "configuration", "path": "parameters.yml",
                      "content": (root / "parameters.yml").read_bytes()})
    identities = {
        "corpus": content_hash([{"variable_id": row["variable_id"], "source": row["source_sha256"],
                                "gold": row["gold_sha256"]} for row in sorted(records, key=lambda r: r["variable_id"])]),
        "population": content_hash(sorted(row["variable_id"] for row in population)),
        "prompts": content_hash({path: digest for path, digest in index.items() if path.startswith("prompts/")}),
        "schema": index["schemas/lexical-decomposition.schema.json"],
        "scorer": content_hash({"source": {path: digest for path, digest in index.items() if path.startswith("src/iadopt_eval/")},
                               "similarity": similarity_identity}),
        "runtime": sha256_bytes(runtime_bytes), "implementation": content_hash(index)}
    return {"identities": identities, "index": index, "runtime": runtime, "artifacts": artifacts,
            "similarity_identity": similarity_identity}


def scorer_identity_hash(project_root: str | Path, similarity_identity: dict) -> str:
    """Recompute the scorer artifact hash from current source and the loaded backend.

    A report claims to be a particular plan's scientific result. Copying that plan's own
    scorer hash and comparing it back to itself cannot support the claim: it holds whatever
    the current checkout and backend are. Recomputing the hash here and comparing it to the
    plan's frozen value is what actually detects a report produced by different evaluator
    source or a different similarity backend from the one the plan froze.

    Must stay byte-identical to the `scorer` identity in `collect_input_artifacts`.

    Args: project_root: Lab root; similarity_identity: identity of the loaded backend.
    Returns: The scorer artifact hash for this checkout and backend.
    Raises: LabError when no evaluator source is present.
    Side effects: Reads evaluator source files.
    """
    root = Path(project_root).resolve()
    source = {path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
              for path in sorted((root / "src/iadopt_eval").rglob("*.py")) if path.is_file()}
    if not source:
        raise LabError("No evaluator source found to identify the scorer")
    return content_hash({"source": source, "similarity": similarity_identity})


def verify_bundle(project_root: str | Path, bundle: dict) -> None:
    """Reject configuration, plan, input or implementation drift before continuation.

    Args: project_root: Current lab root; bundle: frozen plan/configuration/artifact index.
    Returns: None when every stored hash matches its canonical value and current source bytes.
    Raises: LabError for any drift; OSError for missing files.
    Side Effects: Read-only. It never upgrades or silently migrates scientific identity.
    """
    root = Path(project_root).resolve()
    plan = bundle["plan"]
    if content_hash({key: value for key, value in plan.items() if key != "sha256"}) != plan["sha256"]:
        raise LabError("Stored plan hash mismatch")
    if content_hash(bundle["configuration"]) != plan["configuration_sha256"]:
        raise LabError("Stored selected configuration hash mismatch")
    for relative, expected in bundle["artifact_index"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256_bytes(path.read_bytes()) != expected:
            raise LabError("Frozen input/implementation differs at: " + relative)
    if content_hash(bundle["artifact_index"]) != plan["artifact_identities"]["implementation"]:
        raise LabError("Frozen implementation index mismatch")
    # Runtime metadata is separate from scientific files, but is still an exact resume gate.
    runtime = json.loads(canonical_json_bytes(bundle["runtime"]))
    if sha256_bytes(canonical_json_bytes(runtime)) != plan["artifact_identities"]["runtime"]:
        raise LabError("Frozen runtime identity mismatch")
    if platform.python_version() != runtime["python"]:
        raise LabError("Python version differs from the frozen runtime")
    actual = {distribution.metadata["Name"].lower(): distribution.version
              for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]}
    if actual != runtime["packages"]:
        raise LabError("Installed package versions differ from the frozen runtime")
