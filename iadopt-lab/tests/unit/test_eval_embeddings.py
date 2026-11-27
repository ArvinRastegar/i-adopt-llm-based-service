"""Local artifact verification and lazy embedding boundaries, without downloads."""

import hashlib
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest

from iadopt_eval.embeddings import load_local_similarity, verify_local_artifact


@pytest.fixture
def artifact(tmp_path):
    """Create a tiny non-model artifact solely for filesystem verification tests."""
    directory = tmp_path / "copied-model"
    directory.mkdir()
    content = b'{"test_fixture":true}'
    (directory / "config.json").write_bytes(content)
    manifest = {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2", "revision": "a" * 40,
        "device": "cpu", "dtype": "float32",
        "files": {"config.json": hashlib.sha256(content).hexdigest()},
        "dependencies": {name: "test-only-version" for name in
                         ("sentence-transformers", "torch", "transformers", "numpy", "tokenizers")},
    }
    return directory, manifest


def test_artifact_identity_is_path_free_detached_and_load_is_lazy(artifact, monkeypatch):
    """Verify artifact bytes without importing optional model packages."""
    def forbidden(name):
        """Fail on any optional model import during artifact-only verification."""
        raise AssertionError(f"Unexpected import {name}")

    monkeypatch.setattr("iadopt_eval.embeddings.importlib.import_module", forbidden)
    directory, manifest = artifact
    backend = load_local_similarity(directory, manifest)
    assert str(directory) not in str(backend.identity)
    changed = backend.identity
    changed["manifest"]["revision"] = "b" * 40
    assert backend.identity["manifest"]["revision"] == "a" * 40
    assert backend.identity == verify_local_artifact(directory, manifest)


@pytest.mark.parametrize("mutation", ["changed", "extra", "missing", "symlink", "revision", "dtype", "path", "dependencies"])
def test_artifact_rejects_unpinned_or_changed_inputs(artifact, mutation):
    """Reject incomplete or mutable artifact evidence before a backend can load."""
    directory, manifest = artifact
    manifest = deepcopy(manifest)
    if mutation == "changed":
        (directory / "config.json").write_bytes(b"changed")
    elif mutation == "extra":
        (directory / "extra.json").write_bytes(b"extra")
    elif mutation == "missing":
        (directory / "config.json").unlink()
    elif mutation == "symlink":
        (directory / "alias.json").symlink_to(directory / "config.json")
    elif mutation == "revision":
        manifest["revision"] = "main"
    elif mutation == "dtype":
        manifest["dtype"] = "float16"
    elif mutation == "path":
        manifest["files"] = {"../config.json": "a" * 64}
    elif mutation == "dependencies":
        manifest["dependencies"] = {}
    with pytest.raises(ValueError):
        verify_local_artifact(directory, manifest)


def test_mocked_loader_passes_local_only_controls_and_caches(artifact, monkeypatch):
    """Verify adapter call shape and caching with a fake model, never real inference."""
    directory, manifest = artifact
    calls = []

    class FakeModel:
        """Minimal model object exposing only methods the adapter is allowed to use."""

        def __init__(self, path, **kwargs):
            """Capture the explicit local-only construction arguments."""
            calls.append(("load", path, kwargs))

        def float(self):
            """Record the float32 conversion request."""
            calls.append(("float32",))
            return self

        def eval(self):
            """Record deterministic evaluation-mode selection."""
            calls.append(("eval",))
            return self

        def encode(self, text, **kwargs):
            """Return a synthetic embedding token while recording encoding options."""
            calls.append(("encode", text, kwargs))
            return text

    fake_sentence_transformers = SimpleNamespace(
        SentenceTransformer=FakeModel,
        util=SimpleNamespace(cos_sim=lambda a, b: SimpleNamespace(item=lambda: 0.875)))
    fake_torch = SimpleNamespace(no_grad=nullcontext)
    monkeypatch.setattr("iadopt_eval.embeddings.importlib.metadata.version", lambda name: "test-only-version")
    monkeypatch.setattr("iadopt_eval.embeddings.importlib.import_module",
                        lambda name: {"sentence_transformers": fake_sentence_transformers, "torch": fake_torch}[name])
    backend = load_local_similarity(directory, manifest)
    assert backend("water", "liquid") == 0.875
    assert backend("water", "liquid") == 0.875
    assert calls[0] == ("load", str(directory), {"device": "cpu", "local_files_only": True, "trust_remote_code": False})
    assert calls[1:3] == [("float32",), ("eval",)]
    assert [entry for entry in calls if entry[0] == "encode"] == [
        ("encode", "water", {"convert_to_tensor": True}), ("encode", "liquid", {"convert_to_tensor": True})]


def test_dependencies_and_post_construction_byte_change_are_rechecked(artifact, monkeypatch):
    """Fail closed if environment or artifact changes before first lazy inference."""
    directory, manifest = artifact
    backend = load_local_similarity(directory, manifest)
    monkeypatch.setattr("iadopt_eval.embeddings.importlib.metadata.version", lambda name: "wrong-version")
    with pytest.raises(ValueError, match="version mismatch"):
        backend("water", "liquid")
    (directory / "config.json").write_bytes(b"changed after construction")
    with pytest.raises(ValueError, match="hash mismatch"):
        backend("water", "liquid")
