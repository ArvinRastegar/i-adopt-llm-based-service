"""Typed neutral records at provider and preflight boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderResult:
    """One HTTP attempt's raw evidence and classified outcome; never contains credentials."""

    provider: str
    requested_model: str
    request: dict[str, Any]
    raw_response: str
    assistant_text: str | None
    status_code: int | None
    outcome: str
    delivery: str
    started_at: str
    finished_at: str
    latency_seconds: float
    response_headers: dict[str, str] = field(default_factory=dict)
    response_json: Any = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    returned_model: str | None = None
    request_id: str | None = None
    reasoning_text: str | None = None
    retry_after_seconds: float | None = None
    raw_response_base64: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Copy the complete result into persistence-safe JSON primitives.

        Args: None beyond this immutable record.
        Returns: Independent dictionary of all available response evidence.
        Raises: TypeError if a caller constructed a non-copyable field.
        Side Effects: None; no parsing or extra network request.
        """
        return asdict(self)


class LabError(Exception):
    """Expected operational failure with a deliberately credential-free message."""
