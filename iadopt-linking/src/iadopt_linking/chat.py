"""A chat-completions client for the PSNC gateway that holds its key privately and stores nothing secret.

The endpoint must be https on an allowlisted host, so a damaged settings file cannot send the key
elsewhere.

Response headers are never stored: the gateway's carry a backend address and the key's running
spend."""

from __future__ import annotations

import http.client
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from iadopt_lab.configuration import ConfigurationError, load_runtime_secrets

from .errors import RunError
from .http import MAX_RETRY_AFTER_S, snippet
from .jev import encode
from .openrouter import FATAL, RETRYABLE, Exchange, Transport
from .provenance import now_utc

KEY_NAME = "PSNC_API_KEY"
BASE_URL_NAME = "PSNC_API_BASE_URL"
DEFAULT_BASE_URL = "https://llm.hpc.psnc.pl/v1"
# The only hosts the key may be sent to. Adding one is a reviewed change to this line.
ALLOWED_HOSTS = ("llm.hpc.psnc.pl",)
CHAT_PATH = "/chat/completions"
USER_AGENT = "iadopt-linking/0.1 (I-ADOPT corpus entity-linking benchmark; research use)"

# Forms an account id or a secret takes in an error body: a JSON pair, the same pair inside an
# escaped JSON string, a bare token that looks like an id or a key, a bearer header, and a private
# address the gateway names when its backend fails.
_ID_PAIR = re.compile(r'(?i)((?:\\*")?user[_-]?id(?:\\*")?\s*[:=]\s*(?:\\*")?)[^"\\,}\s]*')
_ID_TOKEN = re.compile(r"\buser_[A-Za-z0-9]{8,}\b")
_KEY_TOKEN = re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{12,}")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{8,}")
_PRIVATE_ADDRESS = re.compile(
    r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?::\d{1,5})?\b")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow a redirect, so the Authorization header never reaches another host."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        """Decline every redirect."""
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def secure_post(url: str, headers: Mapping[str, str], body: bytes,
                timeout_s: float) -> tuple[int, Mapping[str, str], bytes]:
    """Perform one POST that follows no redirect and uses no proxy."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with _OPENER.open(request, timeout=timeout_s) as response:
            return (response.status, {k.lower(): v for k, v in response.headers.items()},
                    response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


def scrub(text: str, secrets: Sequence[str] = ()) -> str:
    """Remove everything from a message that must not be stored."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    text = _BEARER.sub("Bearer <redacted>", text)
    text = _KEY_TOKEN.sub("<redacted>", text)
    text = _ID_PAIR.sub(r"\1<redacted>", text)
    text = _ID_TOKEN.sub("<redacted>", text)
    return _PRIVATE_ADDRESS.sub("<private-address>", text)


def checked_base_url(base_url: str) -> str:
    """Accept only an https endpoint on an allowlisted host."""
    parts = urlparse(base_url)
    if (parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or parts.username
            or parts.password or parts.query or parts.fragment):
        raise RunError(f"the gateway URL must be https on one of {list(ALLOWED_HOSTS)} with no "
                       "credentials; refusing to send the key anywhere else")
    return base_url.rstrip("/")


def credentials_from(env_file: str | None, environ: Mapping[str, str]) -> tuple[str, str]:
    """Read the PSNC key and base URL without ever showing them."""
    try:
        secrets = load_runtime_secrets(env_file, [KEY_NAME, BASE_URL_NAME], environ)
    except (OSError, ConfigurationError) as exc:
        raise RunError(f"the credential file cannot be used ({type(exc).__name__}); it must "
                       f"contain lines {KEY_NAME}=... and optionally {BASE_URL_NAME}=...") from None
    key = secrets.get(KEY_NAME)
    if not key:
        raise RunError(f"{KEY_NAME} is not set; pass --env-file with a file that defines it")
    return key, checked_base_url(secrets.get(BASE_URL_NAME) or DEFAULT_BASE_URL)


class ChatClient:
    """Post chat-completions requests with a bounded number of retries; hold the key privately."""

    def __init__(self, api_key: str, *, base_url: str = DEFAULT_BASE_URL,
                 user_agent: str = USER_AGENT, timeout_s: float = 120.0, max_attempts: int = 4,
                 backoff_s: Sequence[float] = (2.0, 6.0, 20.0), transport: Transport = secure_post,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """Configure the client."""
        if not api_key:
            raise ValueError("an API key is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.__key = api_key
        self._base = checked_base_url(base_url)
        self._url = self._base + CHAT_PATH
        self._user_agent = user_agent
        self._timeout = timeout_s
        self._max_attempts = max_attempts
        self._backoff = tuple(backoff_s) or (2.0,)
        self._transport = transport
        self._clock = clock
        self._sleep = sleep

    def __repr__(self) -> str:
        """Describe the client without the key."""
        return "ChatClient(<redacted>)"

    def __reduce__(self) -> Any:
        """Prevent accidental serialization of the key."""
        raise TypeError("A client that holds a credential cannot be serialized")

    def settings(self) -> dict[str, object]:
        """Report how the client behaves, for the journal. Contains no credential."""
        return {"host": urlparse(self._base).hostname, "path": CHAT_PATH,
                "user_agent": self._user_agent, "timeout_s": self._timeout,
                "max_attempts": self._max_attempts, "backoff_s": list(self._backoff),
                "max_retry_after_s": MAX_RETRY_AFTER_S, "redirects": "not followed",
                "proxies": "ignored"}

    def _scrub(self, text: str) -> str:
        """Remove the key and every other secret-shaped string from text that may be stored.

        Scrubbing happens before truncating, so a cut can never leave half a key."""
        return scrub(text, (self.__key,))

    def post(self, body: Mapping[str, Any]) -> Exchange:
        """Send one request, retrying transient failures at most ``max_attempts`` times."""
        payload = encode(body)
        headers = {"Authorization": f"Bearer {self.__key}", "Content-Type": "application/json",
                   "Accept": "application/json", "User-Agent": self._user_agent}
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


__all__ = ["ALLOWED_HOSTS", "CHAT_PATH", "ChatClient", "FATAL", "Exchange", "credentials_from",
           "scrub", "secure_post"]
