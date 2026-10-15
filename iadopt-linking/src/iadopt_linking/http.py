"""A polite HTTP client: one request at a time per host, bounded retries, no silent failure.

Built on the standard library on purpose. Wikimedia's edge answers python-httpx requests with a
generic "respect our robot policy" 403 even with an honest user agent and a trivial request
rate (diagnosed 2026-10-13: the same URL and user agent succeed through curl and urllib), so
httpx is not used here. The client still identifies itself truthfully, never runs requests in
parallel against one host and waits at least ``min_interval_s`` between them.
"""

from __future__ import annotations

import http.client
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

# Wikimedia blocks a client for about a minute at a time; a smaller cap spends an attempt waking
# up inside the block. The cap still bounds a hostile header, and the attempt count bounds the rest.
MAX_RETRY_AFTER_S = 120.0
SNIPPET_CHARS = 200
Transport = Callable[[str, Mapping[str, str], float], tuple[int, Mapping[str, str], bytes]]


def urllib_transport(url: str, headers: Mapping[str, str],
                     timeout_s: float) -> tuple[int, Mapping[str, str], bytes]:
    """Perform one GET with urllib.

    Args: url: full URL; headers: request headers; timeout_s: socket timeout.
    Returns: status, lower-cased response headers and body bytes; HTTP error statuses are
    returned, not raised.
    Raises: OSError or http.client.HTTPException for connection-level failures.
    Side effects: network I/O.
    """
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return (response.status, {k.lower(): v for k, v in response.headers.items()},
                    response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


def snippet(body: bytes) -> str:
    """Shorten an error body so the reason a service gave is not lost.

    Args: body: response bytes.
    Returns: the first SNIPPET_CHARS characters with whitespace collapsed; "" for an empty body.
    Raises: None.
    Side effects: none.
    """
    return re.sub(r"\s+", " ", body.decode("utf-8", "replace")).strip()[:SNIPPET_CHARS]


@dataclass(frozen=True)
class Fetched:
    """What one logical request ended with, after any retries.

    ``latency_s`` is the wall time including waits; ``request_s`` is the final transport call
    alone, which is what the service took.
    """

    url: str
    status: int | None
    text: str | None
    latency_s: float
    attempts: int
    error: str | None
    retries: tuple[str, ...] = ()
    request_s: float = 0.0


class PoliteClient:
    """Throttle per host, retry transient failures a bounded number of times, report the rest."""

    def __init__(self, user_agent: str, *, min_interval_s: float = 1.0, timeout_s: float = 30.0,
                 max_attempts: int = 3, backoff_s: Sequence[float] = (2.0, 5.0),
                 transport: Transport = urllib_transport,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """Configure the client.

        Args: user_agent: sent on every request; min_interval_s: minimum gap between requests to
        one host; timeout_s: per-attempt timeout; max_attempts: hard cap per logical request;
        backoff_s: waits between attempts; transport/clock/sleep: injectable for tests.
        Returns: None.
        Raises: ValueError for a non-positive attempt cap.
        Side effects: none.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._user_agent = user_agent
        self._interval = min_interval_s
        self._timeout = timeout_s
        self._max_attempts = max_attempts
        self._backoff = tuple(backoff_s) or (2.0,)
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}

    def settings(self) -> dict[str, object]:
        """Report how the client behaves, for the run manifest.

        Args: none.
        Returns: user agent, minimum interval, timeout, attempt cap, backoff and Retry-After cap.
        Raises: None.
        Side effects: none.
        """
        return {"user_agent": self._user_agent, "min_interval_s": self._interval,
                "timeout_s": self._timeout, "max_attempts": self._max_attempts,
                "backoff_s": list(self._backoff), "max_retry_after_s": MAX_RETRY_AFTER_S}

    def close(self) -> None:
        """Release resources; the stdlib transport holds none.

        Args: none.
        Returns: None.
        Raises: None.
        Side effects: none.
        """

    def _wait(self, host: str) -> None:
        """Sleep so consecutive requests to one host respect the minimum interval.

        Args: host: network location of the next request.
        Returns: None.
        Raises: None.
        Side effects: may sleep.
        """
        last = self._last.get(host)
        if last is not None:
            remaining = last + self._interval - self._clock()
            if remaining > 0:
                self._sleep(remaining)

    def get(self, url: str, params: Mapping[str, str] | None = None,
            headers: Mapping[str, str] | None = None) -> Fetched:
        """Fetch a URL, retrying 429, 5xx and connection errors at most max_attempts times.

        Args: url: endpoint; params: query parameters; headers: extra request headers.
        Returns: Fetched with status 200 and text on success; otherwise the last status and an
        error string that includes the start of the service's own explanation. ``retries`` lists
        every failed attempt before the last, with the Retry-After it was given and the wait.
        Never raises for network or HTTP problems.
        Raises: None for network or HTTP failures.
        Side effects: network I/O; sleeps between requests and attempts.
        """
        full_url = url + ("?" + urllib.parse.urlencode(params) if params else "")
        host = urlsplit(url).netloc
        sent = {"User-Agent": self._user_agent, **(headers or {})}
        started = self._clock()
        status: int | None = None
        error: str | None = None
        request_s = 0.0
        failures: list[str] = []
        for attempt in range(1, self._max_attempts + 1):
            self._wait(host)
            delay = self._backoff[min(attempt - 1, len(self._backoff) - 1)]
            began = self._clock()
            try:
                status, response_headers, body = self._transport(full_url, sent, self._timeout)
            except (OSError, http.client.HTTPException) as exc:
                self._last[host] = self._clock()
                request_s = self._clock() - began
                status, error = None, f"{type(exc).__name__}: {exc}"
                note = error
            else:
                self._last[host] = self._clock()
                request_s = self._clock() - began
                if status == 200:
                    try:
                        return Fetched(full_url, 200, body.decode("utf-8"),
                                       self._clock() - started, attempt, None, tuple(failures),
                                       request_s)
                    except UnicodeDecodeError as exc:
                        return Fetched(full_url, 200, None, self._clock() - started, attempt,
                                       f"undecodable response: {exc}", tuple(failures), request_s)
                reason = snippet(body)
                error = f"HTTP {status}" + (f": {reason}" if reason else "")
                if status != 429 and status < 500:
                    return Fetched(full_url, status, None, self._clock() - started, attempt,
                                   error, tuple(failures), request_s)
                header = response_headers.get("retry-after", "")
                if header.isdigit():
                    delay = min(float(header), MAX_RETRY_AFTER_S)
                note = f"HTTP {status}" + (f", Retry-After {header}" if header else "")
            if attempt < self._max_attempts:
                failures.append(f"{note}; waited {delay:g} s")
                self._sleep(delay)
        return Fetched(full_url, status, None, self._clock() - started, self._max_attempts,
                       error, tuple(failures), request_s)
