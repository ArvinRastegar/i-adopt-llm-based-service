"""Portable input/code/runtime receipts."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

from .canonical import canonical_json_bytes, content_hash, sha256_bytes
from .domain import LabError


def collect_input_artifacts(project_root: str | Path, records: list[dict] | tuple[dict, ...],
                            population: list[dict] | tuple[dict, ...], similarity_identity: dict) -> dict:
    """Read portable scientific artifacts and the exact implementation/runtime identity."""
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

    Must stay byte-identical to the `scorer` identity in `collect_input_artifacts`."""
    root = Path(project_root).resolve()
    source = {path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
              for path in sorted((root / "src/iadopt_eval").rglob("*.py")) if path.is_file()}
    if not source:
        raise LabError("No evaluator source found to identify the scorer")
    return content_hash({"source": source, "similarity": similarity_identity})


def verify_bundle(project_root: str | Path, bundle: dict) -> None:
    """Reject configuration, plan, input or implementation drift before continuation."""
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
