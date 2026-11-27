"""Wider redaction for the failure text a PSNC chat run stores (DECISIONS L-019)."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Any

from .chat import ChatClient
from .openrouter import Exchange

# A masked key: three characters of its start are replaced by dots and its end is kept.
_MASKED_KEY = re.compile(r"\b(?:sk|pk|rk)-\.{2,}[A-Za-z0-9_\-]{0,16}")
# LiteLLM prints the hash of a rejected key beside the masked key: 64 hexadecimal digits.
_DIGEST = re.compile(r"\b[0-9a-fA-F]{64}\b")
# `User=<id>`, `user: <id>`, `'user_id': '<id>'` and the same inside an escaped JSON string.
_USER = re.compile(
    r"(?i)((?:\\*['\"])?\buser(?:[_-]?id)?(?:\\*['\"])?\s*[:=]\s*(?:\\*['\"])?)[^\s'\",;}\\]+")
# The running spend and the budget of the key, which a budget error prints.
_BUDGET = re.compile(r"(?i)\b((?:max_)?(?:spend|budget)['\"]?\s*[:=]\s*)[0-9][0-9.,eE+-]*")
# Addresses and names that lead into the gateway's own network: loopback, link-local, carrier-grade
# NAT (100.64.0.0/10), `localhost`, `[::1]`, and cluster-internal host names.
_INTERNAL = re.compile(
    r"(?i)\b(?:127(?:\.\d{1,3}){3}|169\.254(?:\.\d{1,3}){2}"
    r"|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.\d{1,3}){2})(?::\d{1,5})?\b"
    r"|\[::1\](?::\d{1,5})?|\blocalhost(?::\d{1,5})?\b"
    r"|\b[a-z0-9-]{1,63}(?:\.[a-z0-9-]{1,63}){0,5}\.(?:svc|internal|local)"
    r"(?:\.[a-z0-9-]{1,63}){0,3}(?::\d{1,5})?\b")


def redact_more(text: str) -> str:
    """Remove the forms of key, account and backend that ``chat.scrub`` leaves."""
    text = _MASKED_KEY.sub("<redacted>", text)
    text = _DIGEST.sub("<redacted>", text)
    text = _USER.sub(r"\1<redacted>", text)
    text = _BUDGET.sub(r"\1<redacted>", text)
    return _INTERNAL.sub("<private-address>", text)


class RedactingChatClient(ChatClient):
    """A chat client that also redacts the forms ``redact_more`` knows from what it reports."""

    def _scrub(self, text: str) -> str:
        """Remove the key and every other secret-shaped string from text that may be stored."""
        return redact_more(super()._scrub(text))

    def post(self, body: Mapping[str, Any]) -> Exchange:
        """Send one request as the base class does, then redact the failure text it kept."""
        exchange = super().post(body)
        return dataclasses.replace(
            exchange, error=None if exchange.error is None else redact_more(exchange.error),
            retries=tuple(redact_more(note) for note in exchange.retries))


__all__ = ["RedactingChatClient", "redact_more"]
