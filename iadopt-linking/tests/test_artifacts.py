"""Published files are written in one step and hashed from their bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from iadopt_linking.artifacts import file_sha256, write_json, write_text


def test_a_write_replaces_the_file_and_leaves_no_temporary(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    write_text(path, "one")
    write_text(path, "two")
    assert path.read_text(encoding="utf-8") == "two"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt"]


def test_json_is_sorted_indented_and_ends_with_a_newline(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    write_json(path, {"b": [1, 2], "a": "é"})
    text = path.read_text(encoding="utf-8")
    assert text == '{\n  "a": "é",\n  "b": [\n    1,\n    2\n  ]\n}\n'
    assert json.loads(text) == {"a": "é", "b": [1, 2]}


def test_a_value_json_cannot_hold_leaves_the_old_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    write_json(path, {"ok": 1})
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json"]


def test_the_hash_is_of_the_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"\x00abc\r\n")
    assert file_sha256(path) == hashlib.sha256(b"\x00abc\r\n").hexdigest()
