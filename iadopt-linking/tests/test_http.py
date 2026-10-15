"""The polite client: retries are bounded, waits are respected, failures are reported."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from iadopt_linking.http import MAX_RETRY_AFTER_S, SNIPPET_CHARS, PoliteClient

UA = "test-agent/1.0"


class Script:
    """A scripted transport plus a fake clock that only advances when the client sleeps."""

    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps)
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.sleeps: list[float] = []
        self.now = 100.0

    def transport(self, url: str, headers: dict[str, str], timeout_s: float):
        self.calls.append((url, dict(headers)))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def client(self, **kwargs: Any) -> PoliteClient:
        return PoliteClient(UA, transport=self.transport, clock=self.clock, sleep=self.sleep,
                            **kwargs)


def ok(body: str = "{}") -> tuple[int, dict[str, str], bytes]:
    return 200, {}, body.encode("utf-8")


def test_success_sends_the_user_agent_and_the_query_string() -> None:
    script = Script(ok('{"a": 1}'))
    fetched = script.client().get("https://example.test/api", {"q": "mass flux", "n": "10"},
                                  {"Accept": "application/json"})
    assert (fetched.status, fetched.text, fetched.attempts, fetched.error) == (200, '{"a": 1}', 1, None)
    url, headers = script.calls[0]
    assert url == "https://example.test/api?q=mass+flux&n=10"
    assert headers == {"User-Agent": UA, "Accept": "application/json"}


def test_a_server_error_is_retried_after_the_backoff() -> None:
    script = Script((500, {}, b""), ok())
    fetched = script.client(backoff_s=(2.0, 5.0)).get("https://example.test/api")
    assert (fetched.status, fetched.attempts) == (200, 2)
    assert fetched.retries == ("HTTP 500; waited 2 s",)
    assert script.sleeps == [2.0]


def test_retry_after_is_honoured_and_capped() -> None:
    script = Script((429, {"retry-after": "7"}, b""), (429, {"retry-after": "999"}, b""), ok())
    fetched = script.client(max_attempts=3).get("https://example.test/api")
    assert fetched.status == 200
    assert script.sleeps == [7.0, MAX_RETRY_AFTER_S]
    assert fetched.retries == ("HTTP 429, Retry-After 7; waited 7 s",
                               f"HTTP 429, Retry-After 999; waited {MAX_RETRY_AFTER_S:g} s")


def test_a_retry_after_that_is_not_a_number_falls_back_to_the_backoff() -> None:
    script = Script((429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, b""), ok())
    script.client(backoff_s=(3.0,)).get("https://example.test/api")
    assert script.sleeps == [3.0]


def test_a_client_error_is_not_retried_and_keeps_the_services_explanation() -> None:
    script = Script((404, {}, b"missing\n  page"))
    fetched = script.client().get("https://example.test/api")
    assert (fetched.status, fetched.attempts, fetched.text) == (404, 1, None)
    assert fetched.error == "HTTP 404: missing page"
    assert script.sleeps == []


def test_a_long_explanation_is_shortened() -> None:
    fetched = Script((403, {}, b"x" * 5000)).client().get("https://example.test/api")
    assert fetched.error == "HTTP 403: " + "x" * SNIPPET_CHARS


def test_persistent_failure_stops_at_the_attempt_cap() -> None:
    script = Script(*[(503, {}, b"")] * 10)
    fetched = script.client(max_attempts=3).get("https://example.test/api")
    assert (fetched.status, fetched.error, fetched.attempts) == (503, "HTTP 503", 3)
    assert fetched.retries == ("HTTP 503; waited 2 s", "HTTP 503; waited 5 s")
    assert len(script.calls) == 3 and len(script.sleeps) == 2


def test_the_wall_time_includes_waits_and_the_request_time_does_not() -> None:
    script = Script((503, {}, b""), ok())
    fetched = script.client(backoff_s=(4.0,)).get("https://example.test/api")
    assert fetched.latency_s == pytest.approx(4.0)
    assert fetched.request_s == pytest.approx(0.0)


def test_the_settings_describe_how_the_client_behaves() -> None:
    settings = Script().client(min_interval_s=2.0, max_attempts=4).settings()
    assert settings["user_agent"] == UA and settings["min_interval_s"] == 2.0
    assert settings["max_attempts"] == 4 and settings["max_retry_after_s"] == MAX_RETRY_AFTER_S


def test_connection_errors_are_retried_and_then_reported() -> None:
    script = Script(*[ConnectionResetError("reset")] * 3)
    fetched = script.client(max_attempts=3).get("https://example.test/api")
    assert fetched.status is None and fetched.attempts == 3
    assert "ConnectionResetError" in fetched.error


def test_requests_to_one_host_are_spaced_but_other_hosts_are_not_delayed() -> None:
    script = Script(ok(), ok(), ok())
    client = script.client(min_interval_s=1.0)
    client.get("https://a.test/x")
    client.get("https://b.test/x")
    assert script.sleeps == []
    script.now += 0.25
    client.get("https://a.test/y")
    assert script.sleeps == [pytest.approx(0.75)]


def test_an_undecodable_body_is_an_error_not_garbled_text() -> None:
    fetched = Script((200, {}, b"\xff\xfe")).client().get("https://example.test/api")
    assert fetched.text is None and "undecodable" in fetched.error


def test_a_zero_attempt_cap_is_rejected() -> None:
    with pytest.raises(ValueError):
        PoliteClient(UA, max_attempts=0)


def test_close_is_a_no_op_that_exists() -> None:
    close: Callable[[], None] = Script().client().close
    close()
