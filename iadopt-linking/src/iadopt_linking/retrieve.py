"""Retrieve candidates for every query key from every source, recording each response.

Bounded by construction: one pass over keys x sources, each request capped by the client's
attempt limit, an optional wall-clock deadline, and a resume rule that skips only records that
succeeded. A failure is written as a failure and counted; it is never turned into an empty list.
The raw response text is stored beside its hash, so the hash can be checked from the file alone.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .http import PoliteClient
from .recorder import JsonlStore, latest
from .sources import SourceError


class RetrievalTimeout(RuntimeError):
    """The wall-clock budget for retrieval ran out before every request was made."""


def is_ok(record: Mapping[str, Any] | None) -> bool:
    """Say whether a stored retrieval succeeded and parsed.

    Args: record: stored retrieval record, or None.
    Returns: True only when there is no error, no parse error and a candidate list.
    Raises: None.
    Side effects: none.
    """
    return bool(record) and (record["error"] is None and record["parse_error"] is None
                             and record["candidates"] is not None)


def response_hash_matches(record: Mapping[str, Any]) -> bool:
    """Check a record's stored response text against its stored hash.

    Args: record: stored retrieval record.
    Returns: True when there is no response (a skipped key) or the hash of the text matches.
    Raises: None.
    Side effects: none.
    """
    text = record.get("response_text")
    if text is None:
        return record.get("response_sha256") is None
    return hashlib.sha256(text.encode("utf-8")).hexdigest() == record.get("response_sha256")


def _utc_now() -> str:
    """Timestamp for a retrieval record.

    Args: none.
    Returns: UTC time to the second, ISO 8601.
    Raises: None.
    Side effects: reads the clock.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def retrieve(keys: Sequence[str], sources: Mapping[str, Any], client: PoliteClient,
             store: JsonlStore, *, deadline: float | None = None,
             clock: Callable[[], float] = time.monotonic, now: Callable[[], str] = _utc_now,
             progress: Callable[[int, int, dict[str, Any]], None] | None = None) -> Counter[str]:
    """Ask every source about every key that has no successful record yet.

    Args: keys: query keys; sources: name -> source with request() and parse(); client: HTTP
    client; store: retrieval JSONL; deadline: clock() value after which to stop; progress:
    callback(done, total, record) after each request.
    Returns: counters: requested, skipped, ok:<source>, failed:<source>, empty:<source>.
    Raises: RetrievalTimeout when the deadline passes with requests left.
    Side effects: network I/O; one appended, fsynced line per request.
    """
    recorded = latest(store.read(), lambda record: (record["source"], record["query_key"]))
    pending = [(key, name) for key in keys for name in sources
               if not is_ok(recorded.get((name, key)))]
    stats: Counter[str] = Counter(skipped=len(keys) * len(sources) - len(pending))
    for done, (key, name) in enumerate(pending, 1):
        if deadline is not None and clock() > deadline:
            raise RetrievalTimeout(f"deadline passed with {len(pending) - done + 1} requests left")
        source = sources[name]
        request = source.request(key)
        record: dict[str, Any] = {
            "source": name, "query_key": key, "retrieved_at": now(), "http_status": None,
            "attempts": 0, "attempt_errors": [], "latency_s": 0.0, "request_s": 0.0,
            "error": None, "parse_error": None, "response_sha256": None, "response_text": None,
            "candidates": None, "skipped_reason": None,
            "request": None if request is None else {"url": request.url, "params": request.params}}
        if request is None:
            record["candidates"] = []
            record["skipped_reason"] = "no searchable word in the key"
        else:
            fetched = client.get(request.url, request.params, request.headers)
            stats["requested"] += 1
            record.update(http_status=fetched.status, attempts=fetched.attempts,
                          attempt_errors=list(fetched.retries),
                          latency_s=round(fetched.latency_s, 3),
                          request_s=round(fetched.request_s, 3), error=fetched.error)
            if fetched.text is not None:
                record["response_text"] = fetched.text
                record["response_sha256"] = hashlib.sha256(fetched.text.encode("utf-8")).hexdigest()
                try:
                    record["candidates"] = [asdict(c) for c in source.parse(fetched.text)]
                except SourceError as exc:
                    record["parse_error"] = str(exc)
        store.append(record)
        if is_ok(record):
            stats[f"ok:{name}"] += 1
            if not record["candidates"]:
                stats[f"empty:{name}"] += 1
        else:
            stats[f"failed:{name}"] += 1
        if progress is not None:
            progress(done, len(pending), record)
    return stats
