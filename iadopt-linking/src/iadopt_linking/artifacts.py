"""Write and hash the files a run publishes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def write_text(path: Path, text: str) -> None:
    """Replace a file in one step, so a crash never leaves half a file.

    Args: path: destination; text: content, written as UTF-8.
    Returns: None.
    Raises: OSError on a filesystem failure.
    Side effects: writes ``<name>.tmp`` beside the destination, then renames it over it.
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, readable JSON.

    Args: path: destination; value: JSON-serialisable value.
    Returns: None.
    Raises: TypeError or ValueError for a value JSON cannot hold (NaN included).
    Side effects: replaces the file atomically.
    """
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                allow_nan=False) + "\n")


def file_sha256(path: Path) -> str:
    """Hash a file's exact bytes.

    Args: path: file to hash.
    Returns: hexadecimal SHA-256.
    Raises: OSError if unreadable.
    Side effects: filesystem read.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()
