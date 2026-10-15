"""A client for OpenRouter's Decisions API that keeps the credential out of every record.

The API key is read from a credential file or the process environment by the lab's allowlisted
loader and handed to the client, which holds it privately and uses it for exactly one thing: the
``Authorization`` header of a request. It is never part of a settings dictionary, an error message,
a representation or a stored exchange. Built on the standard library like the other clients here.

Unlike the retrieval clients, this one calls a paid API that expects parallel callers, so it has no
per-host spacing; it retries what is transient (rate limits, overloads, upstream failures,
connection errors) a bounded number of times and reports everything else as a failure with the
service's own explanation, without raising.
"""

from __future__ import annotations

import http.client
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from iadopt_lab.configuration import ConfigurationError, load_runtime_secrets

from .errors import RunError
from .http import MAX_RETRY_AFTER_S, snippet
from .jev import encode
from .provenance import now_utc

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
USER_AGENT = "iadopt-linking/0.1 (I-ADOPT corpus entity-linking benchmark; research use)"
KEY_NAME = "OPENROUTER_API_KEY"
# Error bodies from OpenRouter echo the account's user id; it identifies the account holder.
_USER_ID = re.compile(r'("user_id"\s*:\s*")[^"]*')
# Transient: worth another attempt. Everything else (a malformed request, no credit) will not
# change by asking again.
RETRYABLE = frozenset({408, 425, 429, 500, 502, 503, 504, 524, 529})
# The credential or the account is the problem, so every other request would fail the same way.
FATAL = frozenset({401, 402, 403})
Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, Mapping[str, str], bytes]]


def urllib_post(url: str, headers: Mapping[str, str], body: bytes,
                timeout_s: float) -> tuple[int, Mapping[str, str], bytes]:
    """Perform one POST with urllib.

    Args: url: full URL; headers: request headers; body: request body; timeout_s: socket timeout.
    Returns: status, lower-cased response headers and body bytes; HTTP error statuses are
    returned, not raised.
    Raises: OSError or http.client.HTTPException for connection-level failures.
    Side effects: network I/O.
    """
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return (response.status, {k.lower(): v for k, v in response.headers.items()},
                    response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


def api_key_from(env_file: str | None, environ: Mapping[str, str]) -> str:
    """Read the OpenRouter key without ever showing it.

    Args: env_file: credential file such as the repository's ``.env``, or None; environ: the
    process environment, whose value wins over the file's.
    Returns: the key.
    Raises: RunError, whose message never contains the key or any other setting, when the file
    cannot be read or parsed or no key is set.
    Side effects: reads the named file (only the allowlisted setting is kept).
    """
    try:
        secrets = load_runtime_secrets(env_file, [KEY_NAME], environ)
    except (OSError, ConfigurationError) as exc:
        raise RunError(f"the credential file cannot be used ({type(exc).__name__}); "
                       f"it must contain a single line {KEY_NAME}=...") from None
    key = secrets.get(KEY_NAME)
    if key is None:
        raise RunError(f"{KEY_NAME} is not set; pass --env-file with a file that defines it")
    return key


@dataclass(frozen=True)
class Exchange:
    """What one logical request ended with, after any retries.

    ``latency_s`` is the wall time including waits; ``request_s`` is the final transport call
    alone. ``text`` is set only for a 200 response. ``retries`` lists every failed attempt before
    the last, with the status, the Retry-After it was given and the wait.
    """

    status: int | None
    text: str | None
    error: str | None
    attempts: int
    retries: tuple[str, ...]
    latency_s: float
    request_s: float
    sent_at: str


class DecisionsClient:
    """Post Decisions requests with a bounded number of retries; hold the key privately."""

    def __init__(self, api_key: str, *, url: str = DECISIONS_URL, user_agent: str = USER_AGENT,
                 timeout_s: float = 90.0, max_attempts: int = 4,
                 backoff_s: Sequence[float] = (2.0, 6.0, 20.0), transport: Transport = urllib_post,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """Configure the client.

        Args: api_key: the OpenRouter key; url: endpoint; user_agent: sent on every request;
        timeout_s: per-attempt timeout; max_attempts: hard cap per logical request; backoff_s:
        waits between attempts; transport, clock and sleep: injectable for tests.
        Returns: None.
        Raises: ValueError for an empty key or a non-positive attempt cap (the message never
        includes the key).
        Side effects: none.
        """
        if not api_key:
            raise ValueError("an API key is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.__key = api_key
        self._url = url
        self._user_agent = user_agent
        self._timeout = timeout_s
        self._max_attempts = max_attempts
        self._backoff = tuple(backoff_s) or (2.0,)
        self._transport = transport
        self._clock = clock
        self._sleep = sleep

    def __repr__(self) -> str:
        """Describe the client without the key.

        Args: none.
        Returns: fixed redacted text.
        Raises: None.
        Side effects: none.
        """
        return "DecisionsClient(<redacted>)"

    def __reduce__(self) -> Any:
        """Prevent accidental serialization of the key.

        Args: none.
        Returns: never returns.
        Raises: TypeError always.
        Side effects: none.
        """
        raise TypeError("A client that holds a credential cannot be serialized")

    def settings(self) -> dict[str, object]:
        """Report how the client behaves, for the journal. Contains no credential.

        Args: none.
        Returns: endpoint, user agent, timeout, attempt cap, backoff and Retry-After cap.
        Raises: None.
        Side effects: none.
        """
        return {"url": self._url, "user_agent": self._user_agent, "timeout_s": self._timeout,
                "max_attempts": self._max_attempts, "backoff_s": list(self._backoff),
                "max_retry_after_s": MAX_RETRY_AFTER_S}

    def _scrub(self, text: str) -> str:
        """Remove the key and the account's user id from text that may be stored.

        Args: text: any message; scrub before truncating, so that a cut can never leave half a key.
        Returns: the text with every occurrence of the key and any ``user_id`` value replaced.
        Raises: None.
        Side effects: none.
        """
        return _USER_ID.sub(r"\1<redacted>", text.replace(self.__key, "<redacted>"))

    def post(self, body: Mapping[str, Any]) -> Exchange:
        """Send one request, retrying transient failures at most ``max_attempts`` times.

        Args: body: the request body, serialized exactly as ``jev.encode`` does.
        Returns: an Exchange; status 200 with the response text on success, otherwise the last
        status and an error that includes the start of the service's own explanation. Never
        raises for network or HTTP problems.
        Raises: TypeError or ValueError if the body cannot be serialized.
        Side effects: network I/O; sleeps between attempts.
        """
        payload = encode(body)
        headers = {"Authorization": f"Bearer {self.__key}", "Content-Type": "application/json",
                   "Accept": "application/json", "User-Agent": self._user_agent,
                   "X-Title": "iadopt-linking"}
        sent_at = now_utc()
        started = self._clock()
        status: int | None = None
        error: str | None = None
        request_s = 0.0
        failures: list[str] = []
        for attempt in range(1, self._max_attempts + 1):
            delay = self._backoff[min(attempt - 1, len(self._backoff) - 1)]
            began = self._clock()
            try:
                status, response_headers, raw = self._transport(
                    self._url, headers, payload, self._timeout)
            except (OSError, http.client.HTTPException) as exc:
                request_s = self._clock() - began
                status, error = None, self._scrub(f"{type(exc).__name__}: {exc}")
                note = error
            else:
                request_s = self._clock() - began
                if status == 200:
                    try:
                        return Exchange(200, raw.decode("utf-8"), None, attempt, tuple(failures),
                                        self._clock() - started, request_s, sent_at)
                    except UnicodeDecodeError as exc:
                        return Exchange(200, None, f"undecodable response: {exc}", attempt,
                                        tuple(failures), self._clock() - started, request_s,
                                        sent_at)
                reason = snippet(self._scrub(raw.decode("utf-8", "replace")).encode("utf-8"))
                error = f"HTTP {status}" + (f": {reason}" if reason else "")
                if status not in RETRYABLE:
                    return Exchange(status, None, error, attempt, tuple(failures),
                                    self._clock() - started, request_s, sent_at)
                header = response_headers.get("retry-after", "")
                if re.fullmatch(r"\d+", header):
                    delay = min(float(header), MAX_RETRY_AFTER_S)
                note = f"HTTP {status}" + (f", Retry-After {header}" if header else "")
            if attempt < self._max_attempts:
                failures.append(f"{note}; waited {delay:g} s")
                self._sleep(delay)
        return Exchange(status, None, error, self._max_attempts, tuple(failures),
                        self._clock() - started, request_s, sent_at)
