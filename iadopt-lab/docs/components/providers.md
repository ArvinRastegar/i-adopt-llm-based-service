# Provider Adapter Contract

## Responsibility

Translate one provider-neutral attempt request into exactly one OpenRouter or PSNC request, execute it, and return one typed result with raw evidence and normalized metadata.

Adapters do not retry, render prompts, validate JSON, choose another model, enrich entities, generate RDF, or decide task state.

## Inputs

- Attempt/task identifiers and idempotency key
- Exact ordered messages
- Provider and exact model ID/revision
- Temperature and other capability-approved sampling fields
- Normalized reasoning profile plus exact native mapping
- Output limit and timeout
- Credential retrieved from the in-memory runtime resolver at call time (process environment or explicitly selected allowlisted env-file; never persisted)

## Outputs

- Sanitized exact request body
- Untouched raw response body and assistant text
- Optional reasoning text
- Requested/returned model IDs and provider request identifiers
- HTTP status, finish reason, timing, token categories
- Delivery certainty
- Typed success or error classification

The output never contains an API key or Authorization header.

## Common behavior

- One adapter invocation equals at most one provider request.
- SDK automatic retries are disabled.
- Use the shared pinned OpenAI Python library with separate provider-scoped clients, explicit provider keys/base URLs, and `max_retries=0`. Neither client may fall back to an OpenAI-hosted endpoint or another provider's key. Capture final serialized requests and complete raw responses before extracting assistant content; do not use the attachment's printed-only output as evidence storage.
- Calls are non-streaming for the primary experiment.
- An unsupported field is a preflight/configuration error, not silently omitted.
- Exact outgoing reasoning fields are retained in the sanitized request.
- Provider raw usage is retained alongside normalized usage.
- A combined campaign injects the adapter belonging to this task's frozen provider/model pair. Adapter lookup never sends another selected provider's model ID or performs fallback routing across the two adapters.

## Configuration keys consumed

- The task's provider identity, verified as a member of `campaign.providers`
- That task provider's resolved entry under `providers.*`
- The selected model's `id`, `revision`, `capabilities`, and `reasoning_profiles`
- `parameter_grid.temperatures`, `parameter_grid.top_p`, and `parameter_grid.max_output_tokens`
- `generation.provider_native_structured_output` and `generation.provider_sdk_auto_retries`

## OpenRouter

Uses an OpenAI Python client configured with base `https://openrouter.ai/api/v1` and explicit `OPENROUTER_API_KEY`; the endpoint path is `/chat/completions`. Provider-native model names and routing/system fingerprints remain unchanged. Model-specific reasoning fields come exclusively from its frozen model profile.

## PSNC

Uses an OpenAI Python client configured with base `https://llm.hpc.psnc.pl/v1`, bearer `PSNC_API_KEY`, and endpoint path `/chat/completions`. The base already includes `/v1`; the final path must not duplicate it. `PSNC_API_BASE_URL` overrides the full SDK base. Non-stream response text normally comes from `choices[0].message.content`, derived only after preserving the entire raw envelope. Provider-native thinking fields may be sent through the SDK's request-extension interface only when that exact model profile declares verified support.

PSNC models are `GLM-5.2`, `Qwen3.8-27B`, and `DeepSeek-V4-Flash`; OpenRouter models are `qwen/qwen3-8b`, `qwen/qwen3-32b`, and `openai/gpt-4o-mini` under D-032. `docs/model-catalog.md` records the owner-supplied source and SDK evidence. Lists remain configurable; adapters impose no fixed count. Model selection does not prove availability or reasoning controls, including the attachment's unverified `xhigh` mention. PSNC's `non_billed` basis records the owner's account-specific free access; OpenRouter is metered and requires price evidence, not a mandatory cap. Both adapters preserve usage even when documented monetary cost is zero. Unknown usage remains unavailable rather than invented. The orchestrator verifies estimate disclosure and separate live authorization before invoking either adapter; possession of API keys does not authorize calls.

## Failures

Typed failures include missing credential, DNS/connect, timeout before known acceptance, rate limit, HTTP client/server error, authentication, permission, model not found, malformed provider envelope, empty/HTML content, and ambiguous delivery. The adapter classifies retryability but does not retry.

Each result identifies its provider and model and retains a sanitized `Retry-After` or equivalent cooldown hint when available. The orchestrator determines the affected scope. A runtime failure does not authorize task identity changes, a fourth attempt, or stopping healthy providers. Shared database/integrity failures stop all dispatch; optional non-null monetary caps restrict positive-cost requests while documented zero-cost requests remain eligible under other gates. An expected estimate is not a dispatch ceiling.

## Planned public interface

### `ProviderAdapter.invoke(request) -> ProviderResult`

- **Input:** One validated provider-neutral attempt request containing exact messages, provider/model/revision, sampling/output values, native reasoning mapping, timeout, idempotency key when supported, and a credential reference resolved only at call time.
- **Action:** Build and sanitize the native payload, add in-memory authentication, dispatch zero requests on local validation failure or exactly one non-streaming request otherwise, read the response once, and classify delivery/result without retrying.
- **Output:** Typed success/error result containing sanitized exact request, untouched response bytes/text/envelope when received, assistant/reasoning text, provider/status/finish/usage/model/request IDs, timings, delivery certainty, provider classification, and observed retry/cooldown hints when supplied.
- **Raises:** Programmer/interface violations may raise typed adapter errors; network/provider outcomes are returned as evidence-rich result variants so the orchestrator owns retry decisions.
- **Side effects:** At most one external provider request; no database write, log of secrets, retry, prompt mutation, extraction, validation, or scoring.
- **Security:** Credentials/authorization never enter returned records, exception text, hashes, fixtures, or logs.

### `ProviderAdapter.validate_model_profile(profile) -> CapabilityReport`

- **Input:** One selected provider/model profile, including exact ID/revision policy, declared capabilities, reasoning profiles/native fields, supported sampling/output controls, and provider adapter version.
- **Action:** Check ownership, unique normalized reasoning modes, required enabled/disabled pair or sole `not_applicable`, prohibited fields, placeholders, and compatibility of every grid value with the declared model interface.
- **Output:** Complete pass/fail capability report with stable gate IDs, normalized native request fragments, unsupported combinations, and evidence provenance.
- **Raises:** Integrity error only when the profile cannot be inspected safely; normal incompatibilities are returned as failed gates.
- **Side effects:** None and zero network requests. Current remote capability verification, if later authorized, is a separate preflight evidence operation.
- **Determinism:** Identical profile and capability evidence yield the same report.

### `build_openrouter_request(request) -> SanitizedProviderRequest`

- **Input:** A validated neutral request whose selected provider is OpenRouter and whose model-specific capability mapping has passed preflight.
- **Action:** Map messages, exact model ID, supported sampling/output fields, non-streaming flag, native reasoning object, and supported idempotency metadata to the frozen OpenRouter wire contract; reject rather than omit unsupported values.
- **Output:** Credential-free URL/method/body/header-name specification, canonical sanitized evidence, mapping/version identity, and hash.
- **Raises:** Wrong provider, placeholder model, unsupported/unknown field, invalid reasoning profile, missing required value, or sanitization failure.
- **Side effects:** None; it does not read the API key or send the request.
- **Determinism:** Identical neutral request and adapter contract produce identical sanitized evidence.

### `build_psnc_request(request) -> SanitizedProviderRequest`

- **Input:** A validated neutral request whose selected provider is PSNC, including exact base URL/path, model ID/revision policy, sampling/output values, and model-specific thinking-control fragment.
- **Action:** Map to the frozen LiteLLM-compatible chat-completions payload, set `stream: false`, preserve exact declared Qwen thinking fields when applicable, and reject unsupported controls rather than copying the sibling service's defaults blindly.
- **Output:** Credential-free URL/method/body/header-name specification, canonical sanitized evidence, mapping/version identity, and hash.
- **Raises:** Wrong provider, invalid URL/path, placeholder model, unsupported/unknown field, invalid reasoning mapping, missing required value, or sanitization failure.
- **Side effects:** None; it does not read `PSNC_API_KEY` or dispatch HTTP.
- **Determinism:** Identical neutral request and adapter contract produce identical sanitized evidence.

## Acceptance tests

- Recorded request/response fixtures for both providers
- Multiple models remain provider-scoped lists
- Combined campaign dispatches each task only through its owning provider adapter
- Matching native model IDs at different providers remain distinguishable in every result
- Enabled/disabled/not-applicable reasoning mappings
- No reasoning parameter for incapable models
- Exactly one HTTP invocation per adapter call
- Automatic SDK retries set to zero
- Separate SDK clients form exactly the specified endpoint URLs and cannot leak another provider's credentials or model IDs
- Raw success/error responses are preserved before typed parsing; reading them does not send another request
- Secret and header redaction
- Raw envelope/usage preservation
- Non-billed PSNC usage preserved alongside an explicit billing basis
- Provider-specific cooldown hints retained without adapter sleep or retry
- All error classifications and ambiguous delivery
