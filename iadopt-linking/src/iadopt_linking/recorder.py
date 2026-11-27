"""Append-only JSON-lines store with resume support."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


class StoreError(ValueError):
    """An evidence file holds a line that is not a JSON object."""


class JsonlStore:
    """One JSON object per line; later lines for the same key supersede earlier ones."""

    def __init__(self, path: Path | str) -> None:
        """Bind the store to a file."""
        self.path = Path(path)

    def _parse(self, line: bytes, number: int) -> dict[str, Any]:
        """Decode one line into a record."""
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"{self.path}:{number}: not valid JSON ({exc})") from exc
        if not isinstance(record, dict):
            raise StoreError(f"{self.path}:{number}: not a JSON object")
        return record

    def read(self) -> list[dict[str, Any]]:
        """Load every complete record in file order."""
        if not self.path.is_file():
            return []
        parts = self.path.read_bytes().split(b"\n")
        tail = parts.pop()
        records = [self._parse(line, number) for number, line in enumerate(parts, 1)
                   if line.strip()]
        if tail.strip():
            try:
                records.append(self._parse(tail, len(parts) + 1))
            except StoreError:
                pass
        return records

    def _terminate_tail(self) -> None:
        """Make the file end on a line boundary before appending."""
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return
        with self.path.open("rb+") as handle:
            data = handle.read()
            if data.endswith(b"\n"):
                return
            cut = data.rfind(b"\n") + 1
            try:
                self._parse(data[cut:], 0)
            except StoreError:
                handle.truncate(cut)
            else:
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, record: dict[str, Any]) -> None:
        """Write one record durably."""
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._terminate_tail()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def latest(records: Iterable[dict[str, Any]],
           key: Callable[[dict[str, Any]], Any]) -> dict[Any, dict[str, Any]]:
    """Keep the last record for each key."""
    return {key(record): record for record in records}
