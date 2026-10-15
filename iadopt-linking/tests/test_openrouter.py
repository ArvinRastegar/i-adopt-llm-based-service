"""The OpenRouter client: retries, failures, and above all that the key never leaves it."""

from __future__ import annotations

import pickle
import threading
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from iadopt_linking.errors import RunError
from iadopt_linking.openrouter import (
    FATAL,
    RETRYABLE,
    DecisionsClient,
    api_key_from,
    urllib_post,
)

KEY = "sk-or-v1-SECRET-DO-NOT-STORE-0123456789"
BODY = {"model": "m", "state": {"term": "t"}, "questions": {}}


class Script:
    """A transport that plays back a list of (status, headers, body) or exceptions."""

    def __init__(self, *steps: tuple[int, Mapping[str, str], bytes] | Exception) -> None:
        self.steps = list(steps)
        self.calls: list[tuple[str, dict[str, str], bytes, float]] = []

    def __call__(self, url, headers, body, timeout_s):
        self.calls.append((url, dict(headers), body, timeout_s))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def client(script: Script, sleeps: list[float] | None = None, **options) -> DecisionsClient:
    waited = [] if sleeps is None else sleeps
    return DecisionsClient(KEY, transport=script, sleep=waited.append, clock=iter(range(1000)).__next__,
                           **options)


def test_success_returns_the_text_and_sends_the_key_only_as_a_header() -> None:
    script = Script((200, {}, b'{"ok": 1}'))
    exchange = client(script).post(BODY)
    assert (exchange.status, exchange.text, exchange.error, exchange.attempts) == (
        200, '{"ok": 1}', None, 1)
    url, headers, body, _ = script.calls[0]
    assert url.endswith("/api/alpha/decisions")
    assert headers["Authorization"] == f"Bearer {KEY}" and headers["Content-Type"] == "application/json"
    assert KEY.encode() not in body


def test_nothing_the_client_reports_contains_the_key() -> None:
    instance = client(Script((200, {}, b"{}")))
    assert KEY not in repr(instance) and KEY not in str(instance.settings())
    assert KEY not in repr(instance.post(BODY))
    with pytest.raises(TypeError):
        pickle.dumps(instance)


def test_an_error_body_or_exception_that_echoes_the_key_is_scrubbed() -> None:
    echo = f"bad credential {KEY}".encode()
    exchange = client(Script((401, {}, echo))).post(BODY)
    assert exchange.status == 401 and KEY not in (exchange.error or "") and "<redacted>" in exchange.error
    failing = client(Script(*[OSError(f"connection to {KEY} failed")] * 4),
                     backoff_s=(0.0,)).post(BODY)
    assert failing.status is None and KEY not in (failing.error or "")
    assert all(KEY not in note for note in failing.retries)


def test_the_account_id_in_an_error_body_is_not_kept() -> None:
    body = b'{"error":{"message":"bad","code":400},"user_id":"user_EXAMPLE_NOT_A_REAL_ID"}'
    exchange = client(Script((400, {}, body))).post(BODY)
    assert "EXAMPLE_NOT_A_REAL_ID" not in exchange.error and "<redacted>" in exchange.error


def test_a_key_is_scrubbed_before_the_error_is_shortened_not_after() -> None:
    # 190 characters of padding put the key across the 200-character cut: a scrub after the cut
    # would see half a key and keep it.
    body = ("x" * 190 + KEY).encode()
    exchange = client(Script((400, {}, body))).post(BODY)
    assert "SECRET" not in exchange.error and "sk-or" not in exchange.error


def test_rate_limits_are_retried_after_the_wait_the_service_asked_for() -> None:
    sleeps: list[float] = []
    script = Script((429, {"retry-after": "7"}, b"slow down"), (200, {}, b"{}"))
    exchange = client(script, sleeps).post(BODY)
    assert exchange.status == 200 and exchange.attempts == 2 and sleeps == [7.0]
    assert exchange.retries == ("HTTP 429, Retry-After 7; waited 7 s",)


def test_a_hostile_retry_after_is_capped() -> None:
    sleeps: list[float] = []
    client(Script((429, {"retry-after": "99999"}, b""), (200, {}, b"{}")), sleeps).post(BODY)
    assert sleeps == [120.0]


@pytest.mark.parametrize("status", sorted(RETRYABLE))
def test_every_transient_status_is_retried_a_bounded_number_of_times(status: int) -> None:
    sleeps: list[float] = []
    script = Script(*[(status, {}, b"upstream")] * 4)
    exchange = client(script, sleeps, max_attempts=4).post(BODY)
    assert exchange.status == status and exchange.attempts == 4 and len(script.calls) == 4
    assert exchange.text is None and exchange.error.startswith(f"HTTP {status}")
    assert len(exchange.retries) == 3 and sleeps == [2.0, 6.0, 20.0]


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 422])
def test_a_request_that_asking_again_cannot_fix_is_not_repeated(status: int) -> None:
    script = Script((status, {}, b'{"error": {"message": "nope"}}'))
    exchange = client(script).post(BODY)
    assert exchange.attempts == 1 and len(script.calls) == 1
    assert exchange.status == status and "nope" in exchange.error


def test_the_credential_problems_are_the_ones_that_should_stop_a_run() -> None:
    assert FATAL == {401, 402, 403} and not FATAL & RETRYABLE


def test_connection_errors_are_retried_then_reported() -> None:
    exchange = client(Script(OSError("reset"), (200, {}, b"{}"))).post(BODY)
    assert exchange.status == 200 and exchange.attempts == 2 and "OSError: reset" in exchange.retries[0]
    gone = client(Script(OSError("down"), OSError("down")), max_attempts=2).post(BODY)
    assert gone.status is None and gone.error == "OSError: down" and gone.attempts == 2


def test_an_undecodable_success_is_a_failure_not_an_empty_answer() -> None:
    exchange = client(Script((200, {}, b"\xff\xfe"))).post(BODY)
    assert exchange.status == 200 and exchange.text is None and "undecodable" in exchange.error


def test_the_client_refuses_to_exist_without_a_key_or_attempts() -> None:
    with pytest.raises(ValueError, match="key"):
        DecisionsClient("")
    with pytest.raises(ValueError, match="max_attempts"):
        DecisionsClient(KEY, max_attempts=0)


def test_the_default_transport_is_the_standard_library_one() -> None:
    assert DecisionsClient(KEY)._transport is urllib_post


class Recorder(BaseHTTPRequestHandler):
    """A loopback endpoint that records what arrives and answers 200, or 429 on /busy."""

    seen: ClassVar[list[tuple[str, dict[str, str], bytes]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        type(self).seen.append((self.path, dict(self.headers), self.rfile.read(length)))
        payload = b'{"ok": true}'
        self.send_response(429 if self.path.endswith("/busy") else 200)
        self.send_header("Retry-After", "7")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def endpoint() -> Iterator[str]:
    Recorder.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def test_the_real_transport_posts_a_body_with_its_headers(endpoint: str) -> None:
    status, _, body = urllib_post(endpoint + "/x", {"Authorization": "Bearer fake",
                                                    "Content-Type": "application/json"},
                                  b'{"a":1}', 5.0)
    assert (status, body) == (200, b'{"ok": true}')
    path, headers, sent = Recorder.seen[0]
    assert path == "/x" and sent == b'{"a":1}' and headers["Authorization"] == "Bearer fake"


def test_the_real_transport_returns_an_error_status_with_lower_cased_headers(endpoint: str) -> None:
    status, headers, _ = urllib_post(endpoint + "/busy", {}, b"{}", 5.0)
    assert status == 429 and headers["retry-after"] == "7"


def test_the_client_over_the_real_transport_gives_up_after_its_attempts(endpoint: str) -> None:
    slept: list[float] = []
    exchange = DecisionsClient(KEY, url=endpoint + "/busy", max_attempts=2, sleep=slept.append,
                               timeout_s=5.0).post(BODY)
    assert exchange.status == 429 and exchange.attempts == 2 and slept == [7.0]
    assert len(Recorder.seen) == 2 and KEY not in repr(exchange)


# --- reading the key ---

def write_env(tmp_path: Path, text: str) -> str:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_key_is_read_from_the_file_and_nothing_else_is_kept(tmp_path: Path) -> None:
    env = write_env(tmp_path, f"# credentials\nOTHER_SETTING=do-not-read\nOPENROUTER_API_KEY={KEY}\n")
    assert api_key_from(env, {}) == KEY


def test_quotes_and_export_are_accepted(tmp_path: Path) -> None:
    assert api_key_from(write_env(tmp_path, f'export OPENROUTER_API_KEY="{KEY}"\n'), {}) == KEY


def test_the_process_environment_wins_over_the_file(tmp_path: Path) -> None:
    env = write_env(tmp_path, "OPENROUTER_API_KEY=from-file\n")
    assert api_key_from(env, {"OPENROUTER_API_KEY": "from-process"}) == "from-process"
    assert api_key_from(None, {"OPENROUTER_API_KEY": "from-process"}) == "from-process"


@pytest.mark.parametrize("text", ["", "OTHER=1\n", "OPENROUTER_API_KEY=\n"])
def test_a_missing_key_stops_with_a_message_that_shows_no_settings(tmp_path: Path, text: str) -> None:
    with pytest.raises(RunError, match="OPENROUTER_API_KEY is not set") as caught:
        api_key_from(write_env(tmp_path, text), {})
    assert "OTHER" not in str(caught.value)


def test_an_unusable_file_stops_without_echoing_its_content(tmp_path: Path) -> None:
    missing = str(tmp_path / "absent.env")
    with pytest.raises(RunError, match="cannot be used") as caught:
        api_key_from(missing, {})
    assert caught.value.__cause__ is None and KEY not in str(caught.value)
    with pytest.raises(RunError, match="cannot be used") as caught:
        api_key_from(write_env(tmp_path, f"OPENROUTER_API_KEY='{KEY} unquoted tail\n"), {})
    assert KEY not in str(caught.value) and caught.value.__cause__ is None
