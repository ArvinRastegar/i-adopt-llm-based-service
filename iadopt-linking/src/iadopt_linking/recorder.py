"""Append-only JSON-lines store with resume support.

Every retrieval and every scored slot is written the moment it exists, so an interrupted run
loses nothing and a rerun skips what is already done. A write cut short by a kill leaves at most
one unterminated last line; reading ignores it (it was never completed, so it was never done)
and the next append removes it.
"""

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
        """Bind the store to a file.

        Args: path: JSONL file; its folder is created on first append.
        Returns: None.
        Raises: None.
        Side effects: none until append.
        """
        self.path = Path(path)

    def _parse(self, line: bytes, number: int) -> dict[str, Any]:
        """Decode one line into a record.

        Args: line: the line without its newline; number: 1-based line number for messages.
        Returns: the record.
        Raises: StoreError naming the file and line when the line is not a JSON object.
        Side effects: none.
        """
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"{self.path}:{number}: not valid JSON ({exc})") from exc
        if not isinstance(record, dict):
            raise StoreError(f"{self.path}:{number}: not a JSON object")
        return record

    def read(self) -> list[dict[str, Any]]:
        """Load every complete record in file order.

        Args: none.
        Returns: records; empty when the file does not exist. An unterminated last line that is
        not a whole record is ignored as the remains of an interrupted write.
        Raises: StoreError naming the line when any other line is not a JSON object.
        Side effects: filesystem read.
        """
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
        """Make the file end on a line boundary before appending.

        Args: none.
        Returns: None.
        Raises: OSError on a filesystem failure.
        Side effects: appends a newline to a complete unterminated last record, or truncates the
        unfinished fragment of an interrupted write.
        """
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
        """Write one record durably.

        Args: record: JSON-serialisable mapping.
        Returns: None.
        Raises: TypeError for a value JSON cannot hold.
        Side effects: repairs an unterminated tail, then appends a line, flushes and fsyncs.
        """
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._terminate_tail()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def latest(records: Iterable[dict[str, Any]],
           key: Callable[[dict[str, Any]], Any]) -> dict[Any, dict[str, Any]]:
    """Keep the last record for each key.

    Args: records: records in file order; key: function naming a record's identity.
    Returns: mapping from key to the most recent record.
    Raises: None.
    Side effects: none.
    """
    return {key(record): record for record in records}
