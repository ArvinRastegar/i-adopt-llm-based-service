"""One-time, reproducible historical prompt evidence without legacy runtime imports."""

from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from iadopt_lab.corpus.ingestion import _write_immutable

HISTORICAL_COMMIT = "b9683d2242aa5ca5b987440ea4b6f70bc1253c7e"
PROMPTS = (
    ("strict-minimal", "Strict minimal", "strict_minimal.txt", "c184ec3edcdac4eee348fd315835110e7b9ce6f42b4229d09ad6501f0b8afedc"),
    ("constraint-decomposition", "Constraint decomposition", "constraint_tree.txt", "e7355417a4153999287df4cbe199de47f33e2119840df644e06d2dc0efb12a42"),
    ("matrix-decomposition", "Matrix decomposition", "matrix_tree.txt", "f77aba03472c117b15aa2e3ac27d76519687629f7eb4d4fffb89e60c88237229"),
)


def build_prompt_registry(repository: Path | str, project_root: Path | str) -> dict[str, Any]:
    """Verify historical Git bytes and record exact adapted-template unified diffs.

    Args: local experiment Git repository with immutable January commit and lab root
    containing the three already authored compatibility templates.
    Actions: read historical blobs only, assert known hashes and conservative text,
    record full historical bytes, template hashes, diff and adaptation classification.
    Returns: timestamp-free JSON-compatible registry with self-excluding content hash.
    Raises: ValueError for changed reference or missing no-inference instructions;
    subprocess/filesystem errors for inaccessible evidence.
    Side effects: read-only Git/template access; no legacy imports or network calls.
    """
    repo, root = Path(repository), Path(project_root)
    entries = []
    for family, display, filename, expected in PROMPTS:
        historical_path = "benchmarking_example/data/prompts/" + filename
        raw = subprocess.run(["git", "-C", str(repo), "show", HISTORICAL_COMMIT + ":" + historical_path], check=True, capture_output=True).stdout
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Historical prompt hash does not match January evidence")
        template_path = "prompts/" + family + "-v1.txt"
        adapted = (root / template_path).read_bytes()
        text, template = raw.decode("utf-8"), adapted.decode("utf-8")
        if "Do not infer or invent new concepts." not in template:
            raise ValueError("Accepted no-inference instruction must remain unchanged")
        entries.append({"prompt_id": family, "display_name": display, "version": "v1",
                        "template_path": template_path, "template_sha256": hashlib.sha256(adapted).hexdigest(),
                        "historical_commit": HISTORICAL_COMMIT, "historical_path": historical_path,
                        "historical_sha256": expected, "historical_text": text,
                        "review_status": "approved-compatibility-policy-D021",
                        "adaptations": ["Remove regenerated definition/comment fields", "Add shared six-field/system/constraint-target schema compatibility instructions", "Add fixed schema/demo/target rendering placeholders", "Add output-only wrapper instruction", "Constraint family: read rather than regenerate definition"],
                        "unified_diff": "".join(difflib.unified_diff(text.splitlines(keepends=True), template.splitlines(keepends=True), fromfile=historical_path, tofile=template_path))})
    registry = {"schema_version": "prompt-registry-v1", "renderer_version": "one-user-message-v1", "prompts": entries}
    raw = json.dumps(registry, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**registry, "manifest_sha256": hashlib.sha256(raw).hexdigest()}


def materialize_prompt_registry(repository: Path | str, project_root: Path | str) -> dict[str, Any]:
    """Publish a verified immutable prompt registry for runtime use.

    Args: historical local Git repository and iadopt-lab root with templates.
    Returns: exact registry mapping also written as canonical UTF-8 JSON.
    Raises: provenance verification errors or existing-artifact content conflicts.
    Side effects: creates only data/manifests/prompt-registry-v1.json; no provider calls.
    """
    registry = build_prompt_registry(repository, project_root)
    raw = json.dumps(registry, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _write_immutable(Path(project_root) / "data/manifests/prompt-registry-v1.json", raw)
    return registry
