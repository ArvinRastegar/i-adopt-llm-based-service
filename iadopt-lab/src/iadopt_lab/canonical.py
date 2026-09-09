"""Versioned canonical JSON, content hashes and secret-safe request evidence."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

JSON_CONTRACT = "canonical-json-v1"


def _validate(value: Any) -> None:
    """Check JSON primitives recursively without coercing scientific values.

    Args: value: Candidate nested JSON value.
    Returns: None on success.
    Raises: TypeError for unsupported objects/keys; ValueError for non-finite values.
    Side Effects: None; deterministic.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical JSON rejects non-finite numbers")
        return
    if isinstance(value, Decimal):
        raise TypeError("Encode exact decimals as explicitly named strings/receipts")
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Canonical JSON keys must be strings")
            _validate(item)
        return
    raise TypeError("Unsupported canonical JSON value type")


def canonical_json_bytes(value: Any, contract: str = JSON_CONTRACT) -> bytes:
    """Serialize explicit JSON primitives with sorted keys and preserved array order.

    Args: value: JSON-compatible value; contract: exact serialization version.
    Returns: Compact UTF-8 JSON bytes, retaining original Unicode strings.
    Raises: ValueError for unknown contract/non-finite data; TypeError for unsupported data.
    Side Effects: None. Identical inputs produce identical bytes under Python 3.12.
    """
    if contract != JSON_CONTRACT:
        raise ValueError("Unknown canonical JSON contract")
    _validate(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Hash exact bytes without decoding or normalization.

    Args: data: Immutable source bytes.
    Returns: Lowercase SHA-256 hexadecimal digest.
    Raises: TypeError if data is not bytes.
    Side Effects: None; pure.
    """
    if not isinstance(data, bytes):
        raise TypeError("SHA-256 input must be bytes")
    return hashlib.sha256(data).hexdigest()


def content_hash(value: Any) -> str:
    """Hash a canonical JSON value with the project's fixed serialization protocol.

    Args: value: Supported JSON primitives.
    Returns: SHA-256 digest of canonical bytes.
    Raises: TypeError/ValueError if canonicalization fails.
    Side Effects: None; deterministic.
    """
    return sha256_bytes(canonical_json_bytes(value))


@dataclass(frozen=True)
class ContentIdentity:
    """Immutable artifact identity and the exact canonical bytes it describes."""

    kind: str
    version: str
    byte_length: int
    sha256: str
    canonical_bytes: bytes


def build_content_identity(kind: str, version: str, payload: Any) -> ContentIdentity:
    """Build a versioned artifact envelope without persisting it.

    Args: kind: Nonempty artifact kind; version: protocol version; payload: JSON value.
    Returns: Immutable identity containing envelope bytes, length and hash.
    Raises: ValueError for empty kind/version or invalid values; TypeError for non-JSON data.
    Side Effects: None; deterministic, no timestamps or random identities are inserted.
    """
    if not kind or not version:
        raise ValueError("Artifact kind and version are required")
    data = canonical_json_bytes({"kind": kind, "version": version, "payload": payload})
    return ContentIdentity(kind, version, len(data), sha256_bytes(data), data)


def lexical_sort_key(text: str, policy: str = "unicode-nfc-casefold-v1") -> tuple[str, str]:
    """Return a lexical ordering key without altering retained/scored source text.

    Args: text: Original lexical label; policy: supported sort contract.
    Returns: NFC/casefold comparison key and original-label tie-break.
    Raises: TypeError for non-string labels; ValueError for an unsupported policy.
    Side Effects: None; never used as scorer normalization.
    """
    if not isinstance(text, str):
        raise TypeError("Lexical sort input must be text")
    if policy != "unicode-nfc-casefold-v1":
        raise ValueError("Unknown lexical sorting policy")
    return unicodedata.normalize("NFC", text).casefold(), text


def sanitize_provider_request(request: Mapping[str, Any], policy: str = "request-allowlist-v1") -> dict:
    """Validate a credential-free outgoing body against its explicit evidence allowlist.

    Args: request: Chat-completions body before authentication; policy: redaction contract.
    Returns: Independent JSON body containing every admitted scientific field.
    Raises: ValueError for unknown fields/policy or secret-like field names; TypeError for non-JSON data.
    Side Effects: None. Error messages never echo rejected values.
    """
    allowed = {"model", "messages", "temperature", "top_p", "max_tokens", "max_completion_tokens",
               "stream", "reasoning", "reasoning_effort", "enable_thinking", "chat_template_kwargs",
               "seed", "provider", "include_reasoning"}
    if policy != "request-allowlist-v1" or set(request) - allowed:
        raise ValueError("Request contains an unsupported evidence field")
    def check_keys(value: Any) -> None:
        """Reject credential-named keys recursively while preserving message text.

        Args: value: Nested request value.
        Returns: None when credential key names are absent.
        Raises: ValueError for prohibited keys, without revealing values.
        Side Effects: None; message strings are not rewritten.
        """
        if isinstance(value, Mapping):
            for key, item in value.items():
                if any(token in key.lower() for token in ("api_key", "authorization", "password", "cookie", "database_url")):
                    raise ValueError("Secret-bearing request field prohibited")
                check_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_keys(item)
    check_keys(request)
    return json.loads(canonical_json_bytes(dict(request)))
