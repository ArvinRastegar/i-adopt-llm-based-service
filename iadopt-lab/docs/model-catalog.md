# Provider and Model Selection

## Current selection and evidence

These are the exact identifiers supplied by the experiment owner. The OpenRouter list reflects the revision accepted in D-032; the PSNC list is unchanged from the owner's original supply. Capabilities are no longer assumed: `iadopt-lab probe-models` measures each one against the live deployment and writes the result into `parameters.yml`. The measured state is recorded below.

| Provider | Position | Exact API model ID | Display name |
|---|---:|---|---|
| PSNC / PCSS | 1 | `GLM-5.2` | GLM-5.2 |
| PSNC / PCSS | 2 | `Qwen3.8-27B` | Qwen3.8 27B |
| PSNC / PCSS | 3 | `DeepSeek-V4-Flash` | DeepSeek V4 Flash |
| OpenRouter | 1 | `qwen/qwen3-8b` | Qwen3 8B |
| OpenRouter | 2 | `qwen/qwen3-32b` | Qwen3 32B |
| OpenRouter | 3 | `openai/gpt-4o-mini` | GPT-4o mini |
| OpenRouter | 4 | `mistralai/ministral-8b-2512` | Ministral 8B |
| OpenRouter | 5 | `meta-llama/llama-3.1-8b-instruct` | Llama 3.1 8B Instruct |

D-032 withdrew `inclusionai/ling-3.0-flash` and selected `openai/gpt-4o-mini` in its place; the two Qwen entries are unchanged. The count stays at six selected models across two providers, so D-029's grid arithmetic is unaffected.

`openai/gpt-4o-mini` is the one hosted closed-weight selection. The other five are open-weight deployments. Three consequences follow and none may be assumed away: OpenRouter meters it, so complete price-card evidence and the disclosed pre-run estimate are load-bearing rather than formalities; its exposed revision is controlled by the upstream vendor and may not be pinnable, which must be recorded as a limitation rather than left blank; and no reasoning-control support may be inferred from the name.

Source for the original supply: owner-supplied 294-line Python example, SHA-256 `901c0cdc1f774ad7cc0177586f77ff25548a9bc3813d629bfe6f48e2929569c1`. It was read, not executed. The D-032 revision was supplied directly as three OpenRouter identifiers and has no separate attachment. Credential-related lines were excluded from displayed evidence, and the raw attachment is not copied into the experiment directory. Its sample prompts and sample responses are not experiment artifacts.

`campaign.live_calls_enabled` is now true and all six models have measured capabilities. An
entry with a missing capability profile is still only valid as an unfinished non-live
configuration: the planner reports the gap rather than skipping the model or assuming
reasoning is unsupported.

### Reasoning as an experimental dimension

`reasoning` is a real experiment parameter, not a hidden request option. The experiment
states **what** it wants — `enabled` or `disabled` — and each model's `reasoning_profiles`
entry says **how** that provider expresses it. `reasoning_mode` participates in
`configuration_id` and the task fingerprint, so the two arms are distinct tasks and every
report separates them.

Only models with a measured, working control get both arms. A model that cannot reason gets
`not_applicable` and exactly one arm, because two identical arms would be a meaningless
doubling of cost.

| Provider | Model | disabled | enabled | Mechanism |
|---|---|:--:|:--:|---|
| PSNC | `GLM-5.2` | yes | **excluded** | `enable_thinking` false / true; enabled excluded by D-044 |
| PSNC | `Qwen3.8-27B` | yes | **excluded** | `enable_thinking` false / true; enabled excluded by D-044 |
| PSNC | `DeepSeek-V4-Flash` | — | — | `not_applicable`; emits no reasoning |
| OpenRouter | `qwen/qwen3-8b` | yes | yes | `reasoning.enabled` false / true |
| OpenRouter | `qwen/qwen3-32b` | yes | yes | **`/no_think` suffix to disable, `reasoning.enabled=true` to enable** |
| OpenRouter | `openai/gpt-4o-mini` | — | — | `not_applicable`; emits no reasoning |
| OpenRouter | `mistralai/ministral-8b-2512` | — | — | `not_applicable`; emits no reasoning |
| OpenRouter | `meta-llama/llama-3.1-8b-instruct` | — | — | `not_applicable`; emits no reasoning |

`qwen/qwen3-32b` is the reason the mechanism is per-model and per-direction rather than one
switch negated: disabling needs the prompt suffix because OpenRouter drops
`chat_template_kwargs`, while enabling works through the ordinary request field.

Measured cost of turning reasoning on, three real 5-shot prompts per model:

| Model | Latency off → on | Output tokens off → on | Valid JSON |
|---|---|---|---|
| `GLM-5.2` | 3.3s → **233.2s** | 80 → 3,111 | 3/3 → 2/3 |
| `Qwen3.8-27B` | 2.2s → 57.7s | 98 → 3,529 | 3/3 → 2/3 |
| `qwen/qwen3-8b` | ~1s → 20.1s | ~90 → 875 | 3/3 → 3/3 |
| `qwen/qwen3-32b` | 3.0s → 33.1s | 90 → 1,291 | 3/3 → 3/3 |

Reasoning is 6x to 70x slower and produces 9x to 39x more output tokens, which is what
drives the separate execution settings per campaign. The PSNC validity drop from 3/3 to 2/3
is a small sample and should not be read as a finding yet, but it is worth watching: if it
holds at scale, reasoning costs validity as well as time.

### Execution settings and why

Settings are frozen per campaign rather than per model, because the output ceiling is
grid-wide and campaigns are the unit of freezing. Every remaining arm turned out to need
the same settings, so the remaining work is **one** campaign rather than several.

| Setting | Value | Why |
|---|---:|---|
| Concurrency | 8 | Both reasoning-on Qwen models completed 16/16 under a 120s timeout at concurrency 8, and maximum latency FELL from 4 to 8 (105.6s and 94.6s versus 117.0s and 96.5s), so the gateway is not saturating. Not pushed higher because `qwen/qwen3-8b` is rate-limited upstream by Alibaba |
| Worker count | 8 | Matched to the provider gate so the pool can fill it |
| Timeout | 120s | Owner-set. Every included arm fits: measured maxima are 117.0s (qwen3-8b), 96.5s (qwen3-32b) and ~14s (z-ai/glm-5.2); the two non-reasoning models run 1.5-3.9s |
| Max output | 8,000 | Above every measured output by at least 6.7x, and under `qwen/qwen3-8b`'s published 8,192 cap, which is the binding limit across the five arms |

Excluded rather than given their own settings: **PSNC reasoning-on for `GLM-5.2` and
`Qwen3.8-27B`** (D-044, D-045) — at 120s, GLM timed out on 12/16 and Qwen3.8-27B on 8-9/16,
at every concurrency tried including 1 with a 20s cooldown, because latency tracks output
length at ~60 tokens/sec rather than server load.

An earlier revision of this table proposed concurrency 24 and a 900s timeout for PSNC
reasoning-on. That carried a number validated with reasoning *off*, where calls take 2.6s,
into a workload where they take 233s and hold a slot 90 times longer. Re-measuring at the
owner's 120s timeout showed the workload does not fit at any concurrency.

### Measured reasoning controls

Every model needed a different switch, and none could have been guessed from the name. Run
`iadopt-lab probe-models --provider <name> --write` to re-measure; it rewrites only the
`models` block.

| Provider | Model | Reasoning switch that works | Evidence |
|---|---|---|---|
| PSNC | `GLM-5.2` | `chat_template_kwargs.enable_thinking=false` | 355 chars of reasoning uncontrolled, 0 with it (D-041) |
| PSNC | `Qwen3.8-27B` | `chat_template_kwargs.enable_thinking=false` | 147 chars uncontrolled, 0 with it |
| PSNC | `DeepSeek-V4-Flash` | none needed | emits no reasoning in either probe, so `not_applicable` |
| OpenRouter | `qwen/qwen3-8b` | `reasoning.enabled=false` | 1,184 chars uncontrolled, 0 with it |
| OpenRouter | `qwen/qwen3-32b` | **`/no_think` prompt suffix** | 3,243 chars uncontrolled, 2 with it |
| OpenRouter | `openai/gpt-4o-mini` | none needed | emits no reasoning, so `not_applicable` |

The qwen3-32b row is the one worth remembering. `enable_thinking` is Qwen's documented
switch and it works on PSNC, because that is vLLM applying the chat template locally.
OpenRouter routes to upstream providers (DeepInfra, SiliconFlow) that forward the message
but drop `chat_template_kwargs` before the tokenizer, so the switch never arrives. Qwen's
`/no_think` travels inside the prompt, which routing cannot strip. Measured over 6 real
prompts: 37.2s and 1,353 completion tokens without it, 3.0s and 90 with it, same answer
validity. This is why reasoning profiles support `prompt_suffix` as well as `request_fields`.

Two traps the prober now guards against, both of which produced wrong answers first:

- **Route-dependent switches.** `reasoning.enabled=false` silences qwen3-32b on SiliconFlow
  and not on DeepInfra. A single probe call caught a lucky route and recorded a control the
  campaign would not get. Each candidate is now tried three times and accepted only if
  every attempt is clean.
- **Hidden but billed reasoning.** `include_reasoning=false` returns zero reasoning
  characters while still generating and charging for 1,069 completion tokens. Suppressing
  the field is not suppressing the cost, so a switch whose completion tokens far exceed its
  visible answer is rejected.

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
