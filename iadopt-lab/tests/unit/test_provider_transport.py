"""One-exchange shared-SDK tests; all transport is local httpx.MockTransport.

The official OpenAI Python reference documents max_retries=0 and raw response
access: https://developers.openai.com/api/reference/python . These tests verify
the pinned local SDK rather than assuming third-party provider capabilities.
"""

import asyncio
import base64
import json

import httpx
import pytest

from iadopt_lab.canonical import sanitize_provider_request
from iadopt_lab.domain import LabError
from iadopt_lab.providers.base import OpenAICompatibleAdapter, build_request
from iadopt_lab.providers.openrouter import create_adapter as create_openrouter
from iadopt_lab.providers.psnc import create_adapter as create_psnc

MESSAGES = [{"role": "user", "content": "Artificial offline fixture. Return {}."}]
RUN = {"model_id": "synthetic-fixture-model", "temperature": 0.5, "top_p": 1.0,
       "max_output_tokens": 300, "reasoning_fields": {}}


def _exchange(provider, handler, body=None):
    """Execute one isolated mocked SDK call and close its resources.

    Args: provider ID, injected local HTTP handler and optional validated body.
    Returns: ProviderResult plus captured SDK retry setting.
    Raises: adapter/programmer errors; no real networking is configured.
    Side effects: one in-process mocked exchange and local event-loop allocation.
    """
    async def run():
        """Own the asynchronous client lifecycle for one local fixture.

        Args: captured test inputs. Returns: (result,max_retries) pair.
        Raises: transport/programmer errors. Side effects: closes the local mock client.
        """
        profile = {"base_url": "https://openrouter.ai/api/v1", "default_base_url": "https://llm.hpc.psnc.pl/v1", "timeout_seconds": 1}
        factory = create_psnc if provider == "psnc" else create_openrouter
        adapter = factory(profile, "offline-fake-" + provider, transport=httpx.MockTransport(handler))
        try:
            result = await adapter.send_once(body or build_request(RUN, MESSAGES))
            return result, adapter._client.max_retries
        finally:
            await adapter.close()
    return asyncio.run(run())


@pytest.mark.parametrize("provider,url", [
    ("psnc", "https://llm.hpc.psnc.pl/v1/chat/completions"),
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions"),
])
def test_endpoint_credentials_raw_envelope_usage_and_native_reasoning(provider, url):
    """Retain exact native fields and complete raw evidence through one scoped endpoint.

    Args: provider and expected complete URL. Returns: None.
    Raises: AssertionError on extra requests, mutated evidence or leaked header values.
    Side effects: one in-process mock HTTP exchange; no external requests or real keys.
    """
    seen = []
    envelope = {"id": "fixture-request", "model": "fixture-returned-model", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "{}", "reasoning": "hidden fixture reasoning", "reasoning_details": [{"type": "fixture"}]}}], "usage": {"prompt_tokens": 300, "completion_tokens": 300, "completion_tokens_details": {"reasoning_tokens": 100}}, "provider_extra": "preserve-me"}
    raw = json.dumps(envelope, indent=1).encode("utf-8")
    body = build_request({**RUN, "reasoning_fields": {"chat_template_kwargs": {"enable_thinking": True}}}, MESSAGES)

    def handler(request):
        """Capture one mocked wire request and supply untouched provider bytes.

        Args: httpx request. Returns: local success response.
        Raises: no provider errors. Side effects: appends request to fixture list only.
        """
        seen.append(request)
        return httpx.Response(200, content=raw, headers={"content-type": "application/json", "x-request-id": "header-request", "authorization": "must-not-return", "set-cookie": "must-not-return"})

    result, retries = _exchange(provider, handler, body)
    assert len(seen) == 1 and retries == 0
    assert str(seen[0].url) == url
    assert seen[0].headers["authorization"] == "Bearer offline-fake-" + provider
    sent = json.loads(seen[0].content)
    assert sent == body
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}
    assert "extra_body" not in sent and sent["stream"] is False
    assert result.provider == provider and result.outcome == "response_received"
    assert result.assistant_text == "{}" and result.reasoning_text == "hidden fixture reasoning"
    assert result.usage == envelope["usage"] and result.response_json == envelope
    assert base64.b64decode(result.raw_response_base64) == raw
    assert result.raw_response == raw.decode("utf-8")
    assert result.request == body and result.request_id == "fixture-request"
    assert result.returned_model == "fixture-returned-model"
    assert "must-not-return" not in json.dumps(result.to_dict())
    assert "offline-fake-" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("provider", ["psnc", "openrouter"])
@pytest.mark.parametrize("status,outcome,delivery", [
    (429, "classified_transient_provider_error", "rejected"),
    (500, "classified_transient_provider_error", "rejected"),
    (401, "provider_error", "rejected"),
    (408, "ambiguous_delivery", "ambiguous_delivery"),
])
def test_status_errors_do_not_trigger_hidden_sdk_retries(provider, status, outcome, delivery):
    """Prove 429/500 and other error responses consume exactly one adapter exchange.

    Args: provider and expected status classification. Returns: None.
    Raises: AssertionError on auto-retry or lost body/cooldown evidence.
    Side effects: one in-process mocked HTTP exchange; no sleeps or network calls.
    """
    calls = []
    raw = b'{"error":{"message":"fixture error","type":"fixture"}}'

    def handler(request):
        """Count and return one artificial rejected response.

        Args: local request. Returns: fixture status/body/cooldown response.
        Raises: none. Side effects: appends request to local list.
        """
        calls.append(request)
        return httpx.Response(status, content=raw, headers={"retry-after": "7", "content-type": "application/json"})

    result, retries = _exchange(provider, handler)
    assert len(calls) == 1 and retries == 0
    assert (result.outcome, result.delivery, result.status_code) == (outcome, delivery, status)
    assert base64.b64decode(result.raw_response_base64) == raw
    assert result.retry_after_seconds == 7


@pytest.mark.parametrize("exception,outcome,delivery", [
    (httpx.ReadTimeout, "ambiguous_delivery", "ambiguous_delivery"),
    (httpx.WriteError, "ambiguous_delivery", "ambiguous_delivery"),
    (httpx.ConnectTimeout, "classified_transient_provider_error", "not_dispatched"),
    (httpx.ConnectError, "classified_transient_provider_error", "not_dispatched"),
])
def test_timeout_connection_delivery_and_no_retries(exception, outcome, delivery):
    """Keep ambiguous read/write delivery distinct from known connect failures.

    Args: local transport exception class and expected classification. Returns: None.
    Raises: AssertionError on repeated requests or unsafe delivery certainty.
    Side effects: one in-process mocked failure, no external requests.
    """
    calls = []

    def handler(request):
        """Raise one synthetic transport failure after recording dispatch entry.

        Args: mocked httpx request. Returns: never normally.
        Raises: supplied transport exception. Side effects: local call-list append.
        """
        calls.append(request)
        raise exception("fixture failure", request=request)

    result, retries = _exchange("openrouter", handler)
    assert len(calls) == 1 and retries == 0
    assert (result.outcome, result.delivery) == (outcome, delivery)
    assert result.status_code is None and result.assistant_text is None
    assert result.raw_response == ""


@pytest.mark.parametrize("fields", [
    {}, {"enable_thinking": False}, {"enable_thinking": True},
    {"reasoning": {"enabled": False}}, {"reasoning": {"effort": "high"}},
    {"reasoning_effort": "none"}, {"reasoning_effort": "high"},
])
def test_native_reasoning_mapping_preserved(fields):
    """Pass frozen native reasoning controls unchanged, omitting them when not applicable.

    Args: declared local profile mapping. Returns: None.
    Raises: AssertionError on implicit defaults or field mutation. Side effects: none.
    """
    body = build_request({**RUN, "reasoning_fields": fields}, MESSAGES)
    for key, value in fields.items():
        assert body[key] == value
    assert body["temperature"] == 0.5 and body["max_tokens"] == 300
    assert "response_format" not in body
    if not fields:
        assert not ({"reasoning", "reasoning_effort", "enable_thinking", "chat_template_kwargs"} & body.keys())


def test_local_body_rejections_never_dispatch():
    """Reject unsupported model overrides, malformed messages and nested secret keys locally.

    Args: none. Returns: None.
    Raises: AssertionError on accepted invalid request construction. Side effects: none.
    """
    with pytest.raises(LabError):
        build_request({**RUN, "reasoning_fields": {"model": "override"}}, MESSAGES)
    with pytest.raises(LabError):
        build_request(RUN, [{"role": "system", "content": "not allowed"}])
    with pytest.raises(LabError):
        build_request({**RUN, "max_output_tokens": None}, MESSAGES)
    marker = "fake-secret-value"
    with pytest.raises(ValueError) as error:
        sanitize_provider_request({"model": "fixture", "reasoning": {"api_key": marker}})
    assert marker not in str(error.value)


@pytest.mark.parametrize("url", ["http://example.invalid/v1", "https://user:password@example.invalid/v1", "not-a-url"])
def test_unsafe_endpoint_rejected_without_transport(url):
    """Require explicit HTTPS endpoints without credentials in their authority.

    Args: invalid fixture URL. Returns: None.
    Raises: AssertionError on constructor acceptance. Side effects: no client/network created.
    """
    with pytest.raises(LabError):
        OpenAICompatibleAdapter("psnc", url, "fake", 1)
