"""The PSNC chat client: retries, failures, and above all that the key never leaves it."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from iadopt_linking.chat import (
    ALLOWED_HOSTS,
    CHAT_PATH,
    DEFAULT_BASE_URL,
    ChatClient,
    checked_base_url,
    credentials_from,
    scrub,
    secure_post,
)
from iadopt_linking.errors import RunError
from iadopt_linking.jev import encode
from iadopt_linking.openrouter import FATAL, RETRYABLE

KEY = "psnc-SECRET-DO-NOT-STORE-0123456789"
ACCOUNT = "user_ZZexampleACCOUNT0000001"
BODY = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "temperature": 0}


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


def client(script: Script, sleeps: list[float] | None = None, **options) -> ChatClient:
    waited = [] if sleeps is None else sleeps
    return ChatClient(KEY, transport=script, sleep=waited.append, clock=iter(range(1000)).__next__,
                      **options)


def test_success_returns_the_text_and_sends_the_key_only_as_a_header() -> None:
    script = Script((200, {}, b'{"ok": 1}'))
    exchange = client(script).post(BODY)
    assert (exchange.status, exchange.text, exchange.error, exchange.attempts) == (
        200, '{"ok": 1}', None, 1)
    url, headers, body, _ = script.calls[0]
    assert url == DEFAULT_BASE_URL + CHAT_PATH == "https://llm.hpc.psnc.pl/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {KEY}" and headers["Content-Type"] == "application/json"
    assert body == encode(BODY) and KEY.encode() not in body


def test_nothing_the_client_reports_contains_the_key() -> None:
    instance = client(Script((200, {}, b"{}")))
    assert KEY not in repr(instance) and KEY not in str(instance.settings())
    assert KEY not in repr(instance.post(BODY))
    with pytest.raises(TypeError):
        pickle.dumps(instance)


def test_the_settings_say_how_it_behaves_without_naming_the_endpoint_beyond_its_host() -> None:
    settings = client(Script()).settings()
    assert settings["host"] == "llm.hpc.psnc.pl" and settings["path"] == CHAT_PATH
    assert settings["redirects"] == "not followed" and settings["proxies"] == "ignored"
    assert "url" not in settings and KEY not in str(settings)


def test_an_error_body_or_exception_that_echoes_the_key_is_scrubbed() -> None:
    echo = f"bad credential {KEY}".encode()
    exchange = client(Script((401, {}, echo))).post(BODY)
    assert exchange.status == 401 and KEY not in (exchange.error or "") and "<redacted>" in exchange.error
    failing = client(Script(*[OSError(f"connection to {KEY} failed")] * 4),
                     backoff_s=(0.0,)).post(BODY)
    assert failing.status is None and KEY not in (failing.error or "")
    assert all(KEY not in note for note in failing.retries)


@pytest.mark.parametrize("body", [
    f'{{"error":{{"message":"bad"}},"user_id":"{ACCOUNT}"}}',
    f'{{"error":{{"metadata":{{"raw":"{{\\"user_id\\":\\"{ACCOUNT}\\"}}"}}}}}}',
    f'{{"userId": "{ACCOUNT}"}}', f"owner {ACCOUNT} was rate limited", f"user-id={ACCOUNT}",
    "Authorization: Bearer abcdefgh12345678", "key sk-FAKEKEY1234567890***", "key sk-or-v1-FAKEKEY123456",
])
def test_every_form_of_an_account_id_or_secret_in_an_error_body_is_scrubbed(body: str) -> None:
    exchange = client(Script((400, {}, body.encode()))).post(BODY)
    for fragment in (ACCOUNT, "abcdefgh12345678", "FAKEKEY"):
        assert fragment not in exchange.error
    assert "<redacted>" in exchange.error


def test_the_private_address_a_gateway_names_when_its_backend_fails_is_not_kept() -> None:
    # An invented address: the real one a gateway named must never be copied into this repository.
    text = scrub("Hosted_vllmException - Connection error at http://10.20.30.40:8005/v1/chat/completions")
    assert "10.20.30.40" not in text and "<private-address>" in text
    assert scrub("see 192.168.1.20 and 172.20.3.4:80") == "see <private-address> and <private-address>"
    assert scrub("version 10.2 of the 1.2.3.4 service") == "version 10.2 of the 1.2.3.4 service"


def test_a_key_is_scrubbed_before_the_error_is_shortened_not_after() -> None:
    # 190 characters of padding put the key across the 200-character cut: a scrub after the cut
    # would see half a key and keep it.
    body = ("x" * 190 + KEY).encode()
    exchange = client(Script((400, {}, body))).post(BODY)
    pieces = [KEY[i:i + 5] for i in range(len(KEY) - 4)]
    assert not [piece for piece in pieces if piece in exchange.error], (
        "even a part of the key survived: " + exchange.error[-20:])


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


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308, 400, 401, 402, 403, 404, 413, 422])
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
        ChatClient("")
    with pytest.raises(ValueError, match="max_attempts"):
        ChatClient(KEY, max_attempts=0)


# --- where the key may go ---

@pytest.mark.parametrize("url", [
    "http://llm.hpc.psnc.pl/v1", "https://example.com/v1", "https://llm.hpc.psnc.pl.evil.example/v1",
    "https://user:pw@llm.hpc.psnc.pl/v1", "https://llm.hpc.psnc.pl/v1?key=1",
    "https://llm.hpc.psnc.pl/v1#frag", "ftp://llm.hpc.psnc.pl/v1", ""])
def test_the_key_is_never_sent_to_a_host_that_is_not_allowed(url: str) -> None:
    with pytest.raises(RunError, match="refusing to send the key") as caught:
        checked_base_url(url)
    assert "pw" not in str(caught.value) and "evil" not in str(caught.value)
    with pytest.raises(RunError):
        ChatClient(KEY, base_url=url)


def test_the_allowed_hosts_are_the_published_gateway_only() -> None:
    assert ALLOWED_HOSTS == ("llm.hpc.psnc.pl",)
    assert checked_base_url("https://llm.hpc.psnc.pl/v1/") == "https://llm.hpc.psnc.pl/v1"


# --- the real transport ---

class Recorder(BaseHTTPRequestHandler):
    """A loopback endpoint that records what arrives; /busy answers 429, /moved answers 302."""

    seen: ClassVar[list[tuple[str, dict[str, str], bytes]]] = []
    moved_to: ClassVar[str] = ""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        type(self).seen.append((self.path, dict(self.headers), self.rfile.read(length)))
        payload = b'{"ok": true}'
        if self.path.endswith("/moved"):
            self.send_response(302)
            self.send_header("Location", type(self).moved_to)
        else:
            self.send_response(429 if self.path.endswith("/busy") else 200)
        self.send_header("Retry-After", "7")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST

    def log_message(self, *args: object) -> None:
        pass


def serve() -> tuple[HTTPServer, str]:
    httpd = HTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}"


@pytest.fixture
def endpoint() -> Iterator[str]:
    Recorder.seen = []
    httpd, url = serve()
    yield url
    httpd.shutdown()
    httpd.server_close()


def test_the_real_transport_posts_a_body_with_its_headers(endpoint: str) -> None:
    status, _, body = secure_post(endpoint + "/x", {"Authorization": "Bearer fake",
                                                    "Content-Type": "application/json"},
                                  b'{"a":1}', 5.0)
    assert (status, body) == (200, b'{"ok": true}')
    path, headers, sent = Recorder.seen[0]
    assert path == "/x" and sent == b'{"a":1}' and headers["Authorization"] == "Bearer fake"


def test_the_real_transport_returns_an_error_status_with_lower_cased_headers(endpoint: str) -> None:
    status, headers, _ = secure_post(endpoint + "/busy", {}, b"{}", 5.0)
    assert status == 429 and headers["retry-after"] == "7"


def test_a_redirect_is_returned_not_followed_and_the_key_never_reaches_its_target() -> None:
    elsewhere_log: list[tuple[str, dict[str, str], bytes]] = []

    class Elsewhere(Recorder):
        seen = elsewhere_log

    Recorder.seen = []
    target = HTTPServer(("127.0.0.1", 0), Elsewhere)
    threading.Thread(target=target.serve_forever, daemon=True).start()
    source, source_url = serve()
    Recorder.moved_to = f"http://127.0.0.1:{target.server_port}/stolen"
    try:
        status, headers, _ = secure_post(source_url + "/moved",
                                         {"Authorization": f"Bearer {KEY}"}, b"{}", 5.0)
        assert status == 302 and "location" in headers
    finally:
        source.shutdown()
        target.shutdown()
        source.server_close()
        target.server_close()
    assert elsewhere_log == [], "the redirect target must receive nothing"


def test_the_client_over_the_real_transport_does_not_follow_a_redirect(endpoint: str) -> None:
    client_ = ChatClient(KEY, sleep=lambda _: None)
    client_._url = endpoint + "/moved"          # the allowlist guards the constructor, not this seam
    Recorder.moved_to = endpoint + "/elsewhere"
    exchange = client_.post(BODY)
    assert exchange.status == 302 and exchange.attempts == 1 and exchange.text is None
    assert [path for path, _, _ in Recorder.seen] == ["/moved"]


def test_proxies_named_in_the_environment_are_ignored() -> None:
    # The opener is built when the module is imported, so the variables must exist at import: ask a
    # fresh interpreter. An opener without proxy handling registers no ProxyHandler at all; a default
    # opener built beside it is the control that shows the variables were seen.
    code = ("import urllib.request as r; from iadopt_linking import chat; "
            "mine = [h.proxies for h in chat._OPENER.handlers if isinstance(h, r.ProxyHandler)]; "
            "control = [h.proxies for h in r.build_opener().handlers if isinstance(h, r.ProxyHandler)]; "
            "print(mine, bool(control and control[0]))")
    environment = {**os.environ, "HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9",
                   "HTTPS_PROXY": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
                   "PYTHONPATH": os.pathsep.join(sys.path), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True,
                          text=True, timeout=120, check=False)
    assert done.stdout.strip() == "[] True", done.stdout + done.stderr[-300:]


def test_the_client_over_the_real_transport_gives_up_after_its_attempts(endpoint: str) -> None:
    slept: list[float] = []
    client_ = ChatClient(KEY, max_attempts=2, sleep=slept.append, timeout_s=5.0)
    client_._url = endpoint + "/busy"
    exchange = client_.post(BODY)
    assert exchange.status == 429 and exchange.attempts == 2 and slept == [7.0]
    assert len(Recorder.seen) == 2 and KEY not in repr(exchange)


def test_the_default_transport_is_the_one_that_follows_no_redirect() -> None:
    assert ChatClient(KEY)._transport is secure_post


# --- reading the key ---

def write_env(tmp_path: Path, text: str) -> str:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_key_is_read_from_the_file_and_nothing_else_is_kept(tmp_path: Path) -> None:
    env = write_env(tmp_path, f"# credentials\nOTHER_SETTING=do-not-read\nPSNC_API_KEY={KEY}\n")
    assert credentials_from(env, {}) == (KEY, DEFAULT_BASE_URL)


def test_the_base_url_may_be_named_but_only_if_it_is_allowed(tmp_path: Path) -> None:
    named = write_env(tmp_path, f"PSNC_API_KEY={KEY}\nPSNC_API_BASE_URL=https://llm.hpc.psnc.pl/v1/\n")
    assert credentials_from(named, {}) == (KEY, "https://llm.hpc.psnc.pl/v1")
    elsewhere = write_env(tmp_path, f"PSNC_API_KEY={KEY}\nPSNC_API_BASE_URL=https://example.com/v1\n")
    with pytest.raises(RunError, match="refusing to send the key") as caught:
        credentials_from(elsewhere, {})
    assert KEY not in str(caught.value) and "example.com" not in str(caught.value)


def test_quotes_and_export_are_accepted(tmp_path: Path) -> None:
    assert credentials_from(write_env(tmp_path, f'export PSNC_API_KEY="{KEY}"\n'), {})[0] == KEY


def test_the_process_environment_wins_over_the_file(tmp_path: Path) -> None:
    env = write_env(tmp_path, "PSNC_API_KEY=from-file\n")
    assert credentials_from(env, {"PSNC_API_KEY": "from-process"})[0] == "from-process"
    assert credentials_from(None, {"PSNC_API_KEY": "from-process"})[0] == "from-process"


@pytest.mark.parametrize("text", ["", "OTHER=1\n", "PSNC_API_KEY=\n"])
def test_a_missing_key_stops_with_a_message_that_shows_no_settings(tmp_path: Path, text: str) -> None:
    with pytest.raises(RunError, match="PSNC_API_KEY is not set") as caught:
        credentials_from(write_env(tmp_path, text), {})
    assert "OTHER" not in str(caught.value)


def test_an_unusable_file_stops_without_echoing_its_content(tmp_path: Path) -> None:
    missing = str(tmp_path / "absent.env")
    with pytest.raises(RunError, match="cannot be used") as caught:
        credentials_from(missing, {})
    assert caught.value.__cause__ is None and KEY not in str(caught.value)
    with pytest.raises(RunError, match="cannot be used") as caught:
        credentials_from(write_env(tmp_path, f"PSNC_API_KEY='{KEY} unquoted tail\n"), {})
    assert KEY not in str(caught.value) and caught.value.__cause__ is None
