"""Shared async SDK transport with exact raw evidence and no nested retries."""

from __future__ import annotations

import base64
import json
import math
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from ..canonical import sanitize_provider_request
from ..domain import LabError, ProviderResult

# Distinguishes an absent `content` field from an explicit null one.
_ABSENT = object()
NATIVE_FIELDS = {"reasoning", "reasoning_effort", "enable_thinking", "chat_template_kwargs", "include_reasoning"}
SAFE_HEADERS = {"content-type", "retry-after", "x-request-id", "request-id", "x-ratelimit-limit-requests",
                "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"}


def build_request(run: dict, messages: list[dict]) -> dict:
    """Map a resolved run and rendered messages to one explicit scientific request body.

    Args: run: Model/sampling/native-reasoning record; messages: frozen one-user-message prompt.
    Returns: Sanitized non-streaming Chat Completions body.
    Raises: LabError for unresolved fields, invalid messages, or scientific-field override.
    Side Effects: None; credentials and transport are not accessed.
    """
    if not run.get("model_id") or run.get("max_output_tokens") is None or run.get("top_p") is None:
        raise LabError("Provider request has unresolved model/sampling values")
    if len(messages) != 1 or messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str):
        raise LabError("Experiment request requires exactly one textual user message")
    native = run.get("reasoning_fields", {})
    if set(native) - NATIVE_FIELDS:
        raise LabError("Reasoning mapping contains unsupported or overriding fields")
    body = {"model": run["model_id"], "messages": messages, "temperature": run["temperature"],
            "top_p": run["top_p"], "max_tokens": run["max_output_tokens"], "stream": False, **native}
    return sanitize_provider_request(body)


def _retry_after(value: str | None) -> float | None:
    """Parse a provider cooldown hint without waiting or retrying.

    Args: value: Retry-After seconds or HTTP date.
    Returns: Nonnegative seconds, or None if absent/malformed.
    Raises: None for malformed provider input.
    Side Effects: Reads UTC only when parsing a date.
    """
    if value is None:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


class OpenAICompatibleAdapter:
    """Provider-scoped SDK client; exactly one HTTP exchange per send invocation."""

    def __init__(self, provider: str, base_url: str, api_key: str, timeout: float,
                 *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Create an explicit isolated client with both retry layers disabled.

        Args: provider: psnc/openrouter; base_url: SDK URL; api_key: private key;
            timeout: positive seconds; transport: optional offline fixture transport.
        Returns: None.
        Raises: LabError for invalid provider/URL/key/timeout.
        Side Effects: Allocates a connection pool; sends no network request.
        """
        parsed = urlparse(base_url)
        if provider not in {"psnc", "openrouter"} or parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise LabError("Invalid provider or credential-bearing/non-HTTPS base URL")
        if not api_key or timeout <= 0:
            raise LabError("Explicit provider credentials and positive timeout are required")
        self.provider = provider
        self._http = httpx.AsyncClient(transport=transport or httpx.AsyncHTTPTransport(retries=0),
                                      follow_redirects=False, timeout=timeout, trust_env=False)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                                   timeout=timeout, http_client=self._http)

    async def close(self) -> None:
        """Release this adapter's HTTP resources.

        Args: None beyond the adapter.
        Returns: None.
        Raises: Underlying transport errors during cleanup.
        Side Effects: Closes sockets; no generation or retry.
        """
        await self._client.close()

    async def send_once(self, body: dict[str, Any]) -> ProviderResult:
        """Send one request and preserve its full success/error response before parsing.

        Args: body: Validated credential-free Chat Completions request.
        Returns: Classified ProviderResult including raw bytes/text and all available usage.
        Raises: ValueError for unsafe local body; unexpected programmer errors are not retried.
        Side Effects: At most one HTTP request; no sleeping, persistence, or automatic resend.
        """
        body = sanitize_provider_request(body)
        started = datetime.now(UTC).isoformat()
        start = time.monotonic()
        raw = b""
        headers = {}
        status = None
        delivery = "response_received"
        outcome = "response_received"
        standard = {key: value for key, value in body.items() if key not in NATIVE_FIELDS and key != "provider"}
        extra = {key: value for key, value in body.items() if key in NATIVE_FIELDS or key == "provider"}
        try:
            response = await self._client.chat.completions.with_raw_response.create(**standard, extra_body=extra)
            raw = response.http_response.content
            status = response.status_code
            headers = {key.lower(): value for key, value in response.headers.items() if key.lower() in SAFE_HEADERS}
        except APIStatusError as exc:
            raw = exc.response.content
            status = exc.status_code
            headers = {key.lower(): value for key, value in exc.response.headers.items() if key.lower() in SAFE_HEADERS}
            outcome = "classified_transient_provider_error" if status in {409, 429} or status >= 500 else "provider_error"
            delivery = "rejected"
            if status == 408:
                outcome, delivery = "ambiguous_delivery", "ambiguous_delivery"
        except (APITimeoutError, APIConnectionError) as exc:
            if isinstance(exc.__cause__, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
                outcome, delivery = "classified_transient_provider_error", "not_dispatched"
            else:
                outcome, delivery = "ambiguous_delivery", "ambiguous_delivery"
        envelope = None
        assistant = reasoning = returned = finish = request_id = None
        usage = None
        try:
            envelope = json.loads(raw) if raw else None
        except (ValueError, UnicodeError):
            if outcome == "response_received":
                # A body that is not the provider's JSON envelope is a broken transport
                # result, not a model answer: an HTTP-200 gateway error page must never be
                # routed into content validation, where three of them would exhaust a task's
                # attempts and be scored as an empty prediction. Marking delivery `rejected`
                # keeps it an operational failure, which blocks completion visibly instead.
                outcome = "html_response" if raw.lstrip().startswith(b"<") else "unparsable_envelope"
                delivery = "rejected"
        well_formed_choice = False
        if isinstance(envelope, dict):
            returned = envelope.get("model")
            request_id = envelope.get("id")
            usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else None
            choices = envelope.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                finish = choices[0].get("finish_reason")
                # No default: a missing message must stay distinguishable from an empty
                # one, or `{}` is treated as a well-formed completion.
                message = choices[0].get("message")
                if isinstance(message, dict):
                    # Providers name the reasoning channel differently: OpenRouter
                    # returns `reasoning`, vLLM deployments such as PSNC return
                    # `reasoning_content`. Reading only one silently discards the
                    # other provider's reasoning evidence, which the evidence policy
                    # requires retaining. Order is fixed so the field chosen is
                    # deterministic when a provider ever returns both. This is read
                    # before the completion contract is checked, because reasoning is
                    # evidence worth keeping even from an envelope we go on to reject.
                    for field in ("reasoning", "reasoning_content"):
                        value = message.get(field)
                        if isinstance(value, str) and value:
                            reasoning = value
                            break
                    # A completion is established by a `content` field of the right type,
                    # not by the mere presence of a message object. `{"message": {}}` is a
                    # structure with no completion in it; treating it as a well-formed
                    # empty answer let a malformed envelope be scored as the model
                    # answering nothing. An explicit null stays well-formed: that is how a
                    # reasoning-only or genuinely empty completion is expressed.
                    content = message.get("content", _ABSENT)
                    if content is None or isinstance(content, str):
                        well_formed_choice = True
                        assistant = content if isinstance(content, str) else None
        if outcome == "response_received":
            # Only a well-formed completion may reach content validation. Everything a
            # provider can return at HTTP 200 that is *not* the model's answer has to be
            # an operational failure, or three of them exhaust a task's three attempts
            # and are scored as an empty prediction, lowering a model's F1 on
            # infrastructure noise. Order matters: the most specific cause wins.
            if not isinstance(envelope, dict):
                # Valid JSON that is not an object at all (a list, a number, a string).
                outcome, delivery = "unparsable_envelope", "rejected"
            elif isinstance(envelope.get("error"), (dict, str)) and not assistant:
                # A provider error object returned with a 200 status. It wins whenever no
                # answer text accompanies it: an envelope carrying an error and an empty
                # completion is the gateway reporting a failure, not the model replying.
                outcome, delivery = "provider_error_envelope", "rejected"
            elif not well_formed_choice:
                # No choices, a choice carrying no message object, or a message with no
                # `content` field or a non-string one: the completion structure is
                # missing, which is not the same as an empty answer.
                outcome, delivery = "invalid_envelope", "rejected"
            elif finish == "length":
                # The generation budget was exhausted. Whether it stopped before any
                # answer or mid-way through one, the answer is incomplete because of the
                # configured ceiling, not because the model reasoned badly. Classifying a
                # partial answer as content-invalid would spend retries on an identical
                # request that cannot succeed and would score the shortfall as model error.
                outcome, delivery = "output_truncated", "rejected"
            elif not assistant:
                # A well-formed envelope whose completion is genuinely empty IS a model
                # outcome, and stays deliverable so it can be validated and scored.
                outcome = "empty_response"
        return ProviderResult(provider=self.provider, requested_model=body["model"], request=body,
            raw_response=raw.decode("utf-8", errors="replace"), raw_response_base64=base64.b64encode(raw).decode("ascii"),
            assistant_text=assistant, status_code=status, outcome=outcome, delivery=delivery,
            started_at=started, finished_at=datetime.now(UTC).isoformat(), latency_seconds=time.monotonic()-start,
            response_headers=headers, response_json=envelope, usage=usage, finish_reason=finish,
            returned_model=returned, request_id=request_id or headers.get("x-request-id"), reasoning_text=reasoning,
            retry_after_seconds=_retry_after(headers.get("retry-after")))
