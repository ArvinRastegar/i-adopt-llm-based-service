"""Explicit local-only embedding boundary, separate from the pure scoring core.

No default model directory, cache discovery, or network download is implemented.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import re
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, Mapping

_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_DEPENDENCIES = frozenset({"sentence-transformers", "torch", "transformers", "numpy", "tokenizers"})


def verify_local_artifact(artifact_directory: Path | str,
                          manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify complete model bytes and the explicitly frozen execution policy.

    Args:
        artifact_directory: Existing directory containing regular model files;
            symlinks and undeclared files are deliberately rejected.
        manifest: JSON-compatible model_id, immutable 40-hex revision, files
            mapping relative POSIX names to SHA-256, device='cpu', dtype='float32',
            and dependencies mapping exact versions of the five inference packages.
    Returns:
        Path-free backend identity containing the manifest and its SHA-256.
    Raises:
        ValueError: Wrong identity/policy, unsafe paths, missing/extra/changed files,
            malformed hashes, or incomplete dependency-version declarations.
        OSError: An explicitly supplied artifact file cannot be read.
        TypeError: Manifest values are not JSON-compatible.
    Side Effects:
        Reads and hashes only the explicit local directory. Never downloads,
        discovers caches, imports the model, or writes files.
    """
    frozen = json.loads(json.dumps(dict(manifest), sort_keys=True, allow_nan=False))
    if frozen.get("model_id") != _MODEL_ID or not re.fullmatch(r"[0-9a-f]{40}", str(frozen.get("revision", ""))):
        raise ValueError("Embedding manifest requires the approved model and immutable 40-hex revision")
    if frozen.get("device") != "cpu" or frozen.get("dtype") != "float32":
        raise ValueError("Embedding manifest must freeze cpu/float32 execution")
    dependencies = frozen.get("dependencies")
    if not isinstance(dependencies, dict) or not _DEPENDENCIES.issubset(dependencies):
        raise ValueError("Embedding manifest must pin inference dependency versions")
    if not all(isinstance(version, str) and version.strip() for version in dependencies.values()):
        raise ValueError("Every inference dependency must have an exact nonempty version")
    files = frozen.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Embedding manifest requires a nonempty complete file map")
    directory = Path(artifact_directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Embedding artifact must be a real local directory")
    for relative, digest in files.items():
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or "\\" in relative
                or str(path) != relative or not re.fullmatch(r"[0-9a-f]{64}", str(digest))):
            raise ValueError("Unsafe artifact path or malformed SHA-256 in manifest")
    actual_files: dict[str, Path] = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Embedding artifact symlinks are not immutable copied bytes")
        if path.is_file():
            actual_files[path.relative_to(directory).as_posix()] = path
        elif not path.is_dir():
            raise ValueError("Embedding artifact contains a non-regular filesystem entry")
    if set(actual_files) != set(files):
        raise ValueError("Embedding artifact file set differs from its manifest")
    for relative, path in actual_files.items():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != files[relative]:
            raise ValueError(f"Embedding artifact hash mismatch: {relative}")
    canonical = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"kind": "local-sentence-transformer", "manifest": frozen,
            "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


class LocalSentenceSimilarity:
    """Thread-safe lazy local embedding callable with immutable identity evidence."""

    def __init__(self, artifact_directory: Path | str, manifest: Mapping[str, Any]) -> None:
        """Verify and retain an explicit model location without loading its runtime.

        Args:
            artifact_directory: Explicit local copied model directory.
            manifest: Complete manifest accepted by verify_local_artifact.
        Returns:
            None; exposes path-free verified identity through the identity property.
        Raises:
            ValueError: Manifest or artifact verification fails.
            OSError: Required local files cannot be read.
        Side Effects:
            Reads/hashes model files; allocates only a local cache and lock.
        """
        self._directory = Path(artifact_directory)
        self._identity = verify_local_artifact(self._directory, manifest)
        self._model: Any = None
        self._util: Any = None
        self._torch: Any = None
        self._cache: dict[str, Any] = {}
        self._lock = RLock()

    @property
    def identity(self) -> dict[str, Any]:
        """Return a detached copy of the verified path-free backend identity.

        Args:
            None.
        Returns:
            JSON-compatible manifest and manifest hash for an evaluation record.
        Raises:
            None.
        Side Effects:
            None; callers cannot mutate the retained manifest through the result.
        """
        return json.loads(json.dumps(self._identity))

    def _load(self) -> None:
        """Lazily load verified local bytes using the exact declared dependencies.

        Args:
            None; uses the verified explicit directory and manifest.
        Returns:
            None; initializes the local model, cosine utility, and inference runtime.
        Raises:
            ValueError: Bytes changed or installed dependency versions mismatch.
            ImportError: Optional embedding dependencies are unavailable.
            Exception: Local model loading fails; there is no fallback or download.
        Side Effects:
            Reads verified model files, imports optional packages, and allocates
            CPU float32 model memory. Network retrieval is explicitly disabled.
        """
        if self._model is not None:
            return
        manifest = self._identity["manifest"]
        verify_local_artifact(self._directory, manifest)
        for name, expected in manifest["dependencies"].items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise ImportError(f"Missing embedding dependency {name}; install the locked embeddings extra") from exc
            if actual != expected:
                raise ValueError(f"Embedding dependency version mismatch for {name}: expected {expected}, got {actual}")
        sentence_transformers = importlib.import_module("sentence_transformers")
        self._torch = importlib.import_module("torch")
        self._util = sentence_transformers.util
        self._model = sentence_transformers.SentenceTransformer(
            str(self._directory), device="cpu", local_files_only=True, trust_remote_code=False)
        self._model.float()
        self._model.eval()

    def __call__(self, gold: str, prediction: str) -> float:
        """Encode two supplied strings and reproduce the January cosine operation.

        Args:
            gold: First already-normalized scalar string from the pure evaluator.
            prediction: Second already-normalized scalar string.
        Returns:
            Finite float from sentence_transformers.util.cos_sim(...).item().
        Raises:
            TypeError: Either operand is not a string.
            ValueError: Manifest/dependencies mismatch or cosine is non-finite.
            Exception: Optional dependency/model loading or encoding fails.
        Side Effects:
            On first use loads only verified local model bytes. Caches at most
            32768 string embeddings, serializes concurrent CPU inference, and never
            contacts a provider or downloads a model.
        """
        if not isinstance(gold, str) or not isinstance(prediction, str):
            raise TypeError("Local similarity operands must be strings")
        with self._lock:
            self._load()
            with self._torch.no_grad():
                vectors = []
                for text in (gold, prediction):
                    if text not in self._cache:
                        if len(self._cache) >= 32768:
                            self._cache.clear()
                        self._cache[text] = self._model.encode(text, convert_to_tensor=True)
                    vectors.append(self._cache[text])
                score = float(self._util.cos_sim(vectors[0], vectors[1]).item())
        if not math.isfinite(score):
            raise ValueError("Local embedding cosine is non-finite")
        return score


def load_local_similarity(artifact_directory: Path | str,
                          manifest: Mapping[str, Any]) -> LocalSentenceSimilarity:
    """Construct a verified, lazy, local-only scientific similarity backend.

    Args:
        artifact_directory: Explicit directory with copied immutable model files.
        manifest: Complete model/revision/file/dependency/cpu/float32 manifest.
    Returns:
        Callable LocalSentenceSimilarity exposing its frozen identity property.
    Raises:
        ValueError: Artifact bytes or manifest policy are invalid.
        OSError: Explicit artifact files cannot be read.
    Side Effects:
        Reads and hashes local files; does not import model packages or download.
    """
    return LocalSentenceSimilarity(artifact_directory, manifest)
