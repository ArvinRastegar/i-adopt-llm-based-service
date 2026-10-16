"""The wider redaction of failure text: the forms a LiteLLM gateway uses, and nothing it should keep."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from iadopt_linking.chat import scrub
from iadopt_linking.chat_redact import RedactingChatClient, redact_more

KEY = "psnc-SECRET-DO-NOT-STORE-0123456789"
DIGEST = "0123456789abcdef" * 4
BODY = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "temperature": 0}


class Script:
    """A transport that plays back (status, headers, body) steps or raises the exceptions given."""

    def __init__(self, *steps: tuple[int, Mapping[str, str], bytes] | Exception) -> None:
        self.steps = list(steps)

    def __call__(self, url, headers, body, timeout_s):
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def client(script: Script, **options) -> RedactingChatClient:
    return RedactingChatClient(KEY, transport=script, sleep=lambda _: None,
                               clock=iter(range(1000)).__next__, **options)


@pytest.mark.parametrize(("text", "gone"), [
    (f"Authentication Error. Received API Key = sk-...90zz, Key Hash (Token) ={DIGEST}",
     ["90zz", DIGEST]),
    ("ExceededBudget: User=user-abc123-xyz over budget. Spend=0.0016, Budget=0.001",
     ["user-abc123-xyz", "0.0016", "0.001"]),
    ("{'user_id': 'f3b2-9c1d-4e5a', 'max_budget': 12.5}", ["f3b2-9c1d-4e5a", "12.5"]),
    ('{"detail": "{\\"user_id\\": \\"f3b2-9c1d\\"}"}', ["f3b2-9c1d"]),
    ("Connection error at http://127.0.0.1:8005/v1/chat/completions", ["127.0.0.1"]),
    ("could not reach localhost:4000", ["localhost"]),
    ("dial tcp [::1]:8000: refused", ["[::1]"]),
    ("metadata at 169.254.169.254", ["169.254.169.254"]),
    ("via 100.64.12.9:8080", ["100.64.12.9"]),
    ("no route to vllm-qwen.models.svc.cluster.local:8000", ["vllm-qwen", "cluster.local"]),
    ("host gpu-node-3.psnc.internal timed out", ["gpu-node-3"]),
])
def test_the_forms_that_scrub_leaves_are_redacted(text: str, gone: list[str]) -> None:
    redacted = redact_more(scrub(text))
    for fragment in gone:
        assert fragment not in redacted, f"{fragment!r} survived in {redacted!r}"


@pytest.mark.parametrize("text", [
    "HTTP 429: rate limit reached, retry later",
    "Hosted_vllmException - Connection error",
    "version 10.2 of the 1.2.3.4 service",
    "User-Agent: iadopt-linking/0.1",
    "username=bob and the username field",
    "the share was 0.9512 of 1.0",
    "see https://llm.hpc.psnc.pl/v1/chat/completions",
    "0123456789abcdef" * 3 + "0123456789abcde",
    "model Qwen3.8-27B served by build vllm-0.26.0-1c0daee9",
])
def test_ordinary_failure_text_is_left_alone(text: str) -> None:
    assert redact_more(text) == text


def test_redacting_twice_changes_nothing() -> None:
    once = redact_more("User=abc Spend=3.5 sk-...90zz 127.0.0.1 " + DIGEST)
    assert redact_more(once) == once and "<redacted>" in once and "<private-address>" in once


def test_a_very_long_run_of_name_characters_does_not_stall_the_patterns() -> None:
    # A gateway body is scrubbed before it is cut, so the patterns see all of it; an unbounded
    # host-name pattern would take minutes here instead of a moment.
    text = ("a-" * 400_000) + ".svc"
    redacted = redact_more(text)
    assert "<private-address>" in redacted and len(redacted) < len(text)


def test_a_rejected_key_echo_is_not_stored_by_the_client() -> None:
    body = (f"Authentication Error, Invalid proxy server token passed. Received API Key = "
            f"sk-...90zz, Key Hash (Token) ={DIGEST}").encode()
    exchange = client(Script((401, {}, body))).post(BODY)
    assert exchange.status == 401 and "90zz" not in exchange.error and DIGEST not in exchange.error
    assert exchange.error.startswith("HTTP 401")


def test_a_budget_error_names_no_user_and_no_spend() -> None:
    body = b"ExceededBudget: User=user-abc123-xyz over budget. Spend=0.0016, Budget=0.001"
    exchange = client(Script((400, {}, body))).post(BODY)
    assert "user-abc123-xyz" not in exchange.error and "0.0016" not in exchange.error


def test_a_network_error_that_names_the_backend_is_redacted_in_the_error_and_the_retries() -> None:
    failing = client(Script(*[OSError("connect to 127.0.0.1:8005 failed")] * 4), backoff_s=(0.0,))
    exchange = failing.post(BODY)
    assert exchange.status is None and "127.0.0.1" not in exchange.error
    assert exchange.retries and all("127.0.0.1" not in note for note in exchange.retries)


def test_a_retry_after_header_that_echoes_a_secret_is_not_kept() -> None:
    script = Script((429, {"retry-after": f"later, for {KEY[:5]}... User=abc"}, b"slow"),
                    (200, {}, b"{}"))
    exchange = client(script).post(BODY)
    assert exchange.status == 200 and exchange.retries
    assert "User=abc" not in exchange.retries[0] and KEY not in exchange.retries[0]


def test_a_form_is_redacted_before_the_error_is_cut_not_after() -> None:
    # 170 characters of padding and a label put the 64-digit hash across the 200-character cut: a
    # redaction after the cut sees too few digits to know it, and would keep 11 of them.
    body = ("x" * 170 + " Key Hash (Token) =" + DIGEST).encode()
    exchange = client(Script((401, {}, body))).post(BODY)
    pieces = [DIGEST[i:i + 8] for i in range(0, 16)]
    assert not [piece for piece in pieces if piece in exchange.error], exchange.error[-30:]


def test_the_key_itself_is_still_scrubbed_and_the_response_text_is_left_as_received() -> None:
    echo = f"bad credential {KEY}".encode()
    assert KEY not in client(Script((401, {}, echo))).post(BODY).error
    payload = f'{{"content": "User=abc 127.0.0.1 {DIGEST}"}}'
    ok = client(Script((200, {}, payload.encode()))).post(BODY)
    assert ok.text == payload and ok.error is None, "an answer is evidence, never redacted"


def test_the_client_still_refuses_to_be_serialized_or_to_show_its_key() -> None:
    import pickle

    instance = client(Script())
    assert KEY not in repr(instance) and KEY not in str(instance.settings())
    with pytest.raises(TypeError):
        pickle.dumps(instance)
