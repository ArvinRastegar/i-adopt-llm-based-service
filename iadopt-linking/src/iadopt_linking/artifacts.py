"""Write and hash the files a run publishes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def write_text(path: Path, text: str) -> None:
    """Replace a file in one step, so a crash never leaves half a file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, readable JSON."""
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                allow_nan=False) + "\n")


def file_sha256(path: Path) -> str:
    """Hash a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
