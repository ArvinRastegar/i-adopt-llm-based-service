# Provider and Model Selection

## Current selection and evidence

These are the exact identifiers supplied by the experiment owner. The OpenRouter list reflects the revision accepted in D-032; the PSNC list is unchanged from the owner's original supply. They are the intended first campaign models, not automatically verified provider availability or capability declarations.

| Provider | Position | Exact API model ID | Display name |
|---|---:|---|---|
| PSNC / PCSS | 1 | `GLM-5.2` | GLM-5.2 |
| PSNC / PCSS | 2 | `Qwen3.8-27B` | Qwen3.8 27B |
| PSNC / PCSS | 3 | `DeepSeek-V4-Flash` | DeepSeek V4 Flash |
| OpenRouter | 1 | `qwen/qwen3-8b` | Qwen3 8B |
| OpenRouter | 2 | `qwen/qwen3-32b` | Qwen3 32B |
| OpenRouter | 3 | `openai/gpt-4o-mini` | GPT-4o mini |

D-032 withdrew `inclusionai/ling-3.0-flash` and selected `openai/gpt-4o-mini` in its place; the two Qwen entries are unchanged. The count stays at six selected models across two providers, so D-029's grid arithmetic is unaffected.

`openai/gpt-4o-mini` is the one hosted closed-weight selection. The other five are open-weight deployments. Three consequences follow and none may be assumed away: OpenRouter meters it, so complete price-card evidence and the disclosed pre-run estimate are load-bearing rather than formalities; its exposed revision is controlled by the upstream vendor and may not be pinnable, which must be recorded as a limitation rather than left blank; and no reasoning-control support may be inferred from the name.

Source for the original supply: owner-supplied 294-line Python example, SHA-256 `901c0cdc1f774ad7cc0177586f77ff25548a9bc3813d629bfe6f48e2929569c1`. It was read, not executed. The D-032 revision was supplied directly as three OpenRouter identifiers and has no separate attachment. Credential-related lines were excluded from displayed evidence, and the raw attachment is not copied into the experiment directory. Its sample prompts and sample responses are not experiment artifacts.

All six catalog entries have `enabled: true` to record selection, while `campaign.live_calls_enabled` remains false. Their deployment revisions, capabilities, and exact reasoning controls remain unresolved. An enabled entry with a missing capability profile is valid only as an unfinished non-live configuration: the planner must report that missing information, never silently skip the model, assume reasoning is unsupported, or freeze an incomplete scientific grid.

## Choose one provider or both

Edit only `parameters.yml` before a new campaign:

| Desired execution | `campaign.providers` |
|---|---|
| PSNC only | `[psnc]` |
| OpenRouter only | `[openrouter]` |
| Both | `[psnc, openrouter]` |

Each provider owns its `models` list. Add, remove, disable, or rename entries there; three per provider is the current selection, not a program limit. Model IDs must be unique within a provider. Never translate a PSNC model ID into an OpenRouter slug or silently substitute a newer model. Changing an active selection after freezing requires a new campaign; resume uses the original database snapshot.

The planner combines each selected provider's own model grid. It does not cross-multiply the provider names with all six model IDs. Each task has exactly one owning provider/model and the existing three-request limit. Provider labels remain part of all scores, cost records, and rankings.

D-029 uses one repetition at every temperature for both providers. For the three OpenRouter models alone, one applicable reasoning profile per model gives 13,968 initial calls and at most 41,904 total requests; two verified profiles on all three models give 27,936 initial and at most 83,808 total requests. PSNC tasks are additional only when that provider is selected. Recalculate cost estimates from the new frozen plan; prior five-repetition estimates are not the current workload.

## Shared library, separate clients

| Provider | SDK base URL | Relative Chat Completions path | Credential environment variable |
|---|---|---|---|
| PSNC / PCSS | `https://llm.hpc.psnc.pl/v1` | `/chat/completions` | `PSNC_API_KEY` |
| OpenRouter | `https://openrouter.ai/api/v1` | `/chat/completions` | `OPENROUTER_API_KEY` |

Use the same pinned OpenAI Python library behind two provider-specific adapters and separate provider-scoped clients. The base URL already includes `/v1`; do not append it twice. Explicit client credentials prevent fallback to `OPENAI_API_KEY` or another provider's environment. No OpenAI-hosted generation endpoint is part of this campaign.

The SDK supports configurable base URLs, disabling retries with `max_retries=0`, and access to raw responses through `with_raw_response`. These are documented in the [official OpenAI Python library reference](https://developers.openai.com/api/reference/python). The independently inspected local SDK also exposes these interfaces. Its installed version is evidence only, not the new project's dependency lock.

The implementation must set retries to zero in both clients and any underlying transport. Configure timeout explicitly; use the appropriate async client interface for the concurrent workflow while retaining the same Chat Completions request semantics. Capture the final sanitized outgoing payload and the complete raw HTTP response before deriving assistant content. Preserve error response bodies/status/request IDs too. Reading or parsing a stored raw response cannot trigger another request. Record native provider-specific reasoning fields through the SDK's supported request-extension mechanism, and verify the serialized wire payload in offline fixtures.

Do not copy the example's generic system messages, toy questions, printed-only result handling, absent sampling settings, or default retry behavior. Experiment requests use the frozen one-user-message prompt protocol, schema, demonstrations, parameters, response evidence, and validation correction rules already specified.

## Reasoning and deployment readiness

The examples demonstrate intended routing and names, not enabled/disabled reasoning behavior. In particular, the attachment notes uncertainty about an `xhigh` control for `Qwen3.8-27B`; it does not establish such support. Do not invent `xhigh`, reuse a different Qwen deployment's thinking fields without verification, or equate an omitted field with reasoning disabled.

For every selected model, collect and freeze provider/deployment evidence for availability/revision policy, supported temperatures and sampling controls, context/output limits, and whether reproducible reasoning controls exist. Capable models receive both configured disabled/enabled profiles. Incapable models receive the documented `not_applicable` profile. Unknown capability remains unknown and blocks exact plan freeze. External capability verification is a separate preflight step; this documentation update made no provider call.

## Billing and finishing the execution

The owner reports no-charge PSNC access for this experiment. Store that account-specific `non_billed` basis, explicit zero monetary cost, and complete token/request/time evidence. OpenRouter is configured as `metered`; freeze its price evidence. D-027 requires a per-model/provider and total pre-run estimate with explicit token/retry/reasoning and price/FX assumptions before live authorization, but no mandatory spending cap. All provider/campaign amounts default to null, meaning uncapped, not zero cost. Unknown billing is not free. The estimate is not a guaranteed bill or a ceiling; actual costs remain recorded. No paid calibration call is implicit in preparing it.

One `run` or `resume` invocation processes all frozen selected tasks and advances through validation, scoring, aggregation, ranking, and required reports. Independent provider limits and fair scheduling allow healthy work to continue when another provider is paused. If an optional monetary cap is explicitly configured, its exhaustion does not block documented zero-cost PSNC dispatch. Null caps impose no monetary stop. A shared database/integrity failure or explicit global stop pauses all work. Unresolved delivery or blocked provider tasks remain visible; the campaign is complete only when all expected tasks and final outputs are complete. Resume never changes providers, invents replacement models, or resets request counters.

The owner supplied the repository-root `.env` as the existing source for `OPENROUTER_API_KEY` and `PSNC_API_KEY`. The planned explicit `--env-file` option loads allowlisted settings without shell execution or interpolation and without overriding process environment. Never copy or record the file, secret values, or secret-bearing connection strings in experiment evidence. Supplying a key is not permission to make live calls. See `local-database.md` for database connection preparation.

## Required checks before implementation acceptance

- Exact six IDs and provider ownership match this table and the source attachment.
- SDK URL construction reaches each expected `/v1/chat/completions` or `/api/v1/chat/completions` endpoint exactly once.
- No automatic SDK/transport retries, implicit provider fallback, or credential leakage.
- PSNC-only, OpenRouter-only, and combined configurations expand their own model lists correctly; future lists of different lengths also work.
- Selected but unknown capabilities fail planning with an explicit reason.
- Exact calls, costs, raw evidence, and restart counters remain complete across provider-specific pauses.
- Model/library/example evidence never changes the frozen scientific prompts or scorer.
