"""Provider capability probing: establish per-model evidence before a plan is frozen.

`parameters.yml` gates live execution on per-model capability fields and states
plainly that they must not be inferred from a model name. This module measures
them against the live deployment so an operator fills those fields from
observation. It freezes nothing, scores nothing, and never sends a corpus
prompt: every probe is a short throwaway request with a small output ceiling.

The reasoning verdict needs two observations, not one. Sending the off-switch
and seeing no reasoning proves nothing on its own, because a model that never
reasons looks identical. So each model is probed without any reasoning field to
learn its default, then with each candidate switch to learn whether it changes
that default.

Three things make that verdict trustworthy rather than lucky. The probe prompt
asks for real decomposition work, because a trivial prompt provokes no reasoning
and makes every switch look effective. Each candidate is tried several times and
accepted only if every attempt is clean, because a routing gateway sends
successive requests to different upstream providers and they do not all honour
the same switch. And a switch that merely hides reasoning while still generating
and billing it is rejected, since suppressing the field is not suppressing the
cost.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import yaml

from .domain import LabError

# Modes in the LiteLLM catalog that cannot serve a chat-completions decomposition.
NON_CHAT_MODES = frozenset({"embedding", "rerank", "audio_transcription", "image_generation"})
# Catalog entries whose mode is unset but which are not general chat models.
NON_CHAT_NAME_MARKERS = ("whisper", "-OCR-", "Nanonets")
# The reasoning switch this experiment uses. Measured on GLM-5.2 (D-041); every
# other model has to earn the same verdict rather than inherit it.
REASONING_OFF_FIELDS: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
# Deployments disagree on how reasoning is switched off, so the candidates are tried in
# order and the one that actually works is recorded. `reasoning.exclude` is deliberately
# absent: it removes reasoning from the RESPONSE while the model still generates and bills
# it, which would read as a working switch while quietly costing full price.
# Each candidate is (name, request fields, prompt suffix). A switch is not always a
# request field: Qwen models accept `/no_think` in the prompt, and through a routing
# gateway that is the only channel that survives, because the gateway forwards the message
# but drops chat-template arguments before they reach the tokenizer.
REASONING_OFF_CANDIDATES: tuple[tuple[str, dict[str, Any], str], ...] = (
    ("chat_template_kwargs.enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}, ""),
    ("reasoning.enabled=false", {"reasoning": {"enabled": False}}, ""),
    ("reasoning.max_tokens=0", {"reasoning": {"max_tokens": 0}}, ""),
    ("prompt suffix /no_think", {}, "\n/no_think"),
)
# A switch that hides reasoning without stopping it leaves completion_tokens far above what
# the visible answer needs. Roughly four characters per token, tripled, plus slack for short
# answers: anything past that is generation we are paying for but cannot see.
_HIDDEN_REASONING_SLACK = 50
# A reasoning channel is "silent" below this many characters, not at exactly zero. Some
# deployments return a couple of whitespace characters in the field even when thinking is
# off, and demanding an exact zero rejected `/no_think` on qwen3-32b for a 2-character
# residue - after it had cut reasoning from 1,526 characters and 350 tokens to 2 and 35.
_RESIDUAL_REASONING_CHARS = 16
# Small enough that a runaway thinking model is truncated in seconds rather than
# holding the probe open, large enough that a normal answer completes.
_PROBE_MAX_TOKENS = 512
# The probe prompt has to provoke reasoning, or a model that would think hard on the real
# task looks quiet here and any switch appears to work. A two-word prompt measured nothing:
# this asks for the same shape of work the campaign does, while staying a throwaway that
# touches no corpus record and is never scored.
_PROBE_PROMPT = (
    "Decompose this variable definition into JSON with keys hasProperty and "
    "hasObjectOfInterest, and nothing else: "
    "\"Daily maximum air temperature measured two metres above ground level.\"")
# Each candidate is tried this many times. A routing gateway sends successive requests to
# different upstream providers, and they do not all honour the same switch: one attempt
# that happens to land on a compliant route reports a control the campaign will not get.
_CANDIDATE_ATTEMPTS = 3
_JSON_PROBE_PROMPT = 'Reply with exactly this JSON object and nothing else: {"ok": true}'


def chat_models(catalog: list[dict]) -> list[dict]:
    """Reduce a provider catalog to unique chat-capable models, from either shape.

    LiteLLM deployments answer /model/info with `model_name` plus a `model_info` block;
    OpenRouter answers /models with `id`, `context_length`, `pricing` and
    `supported_parameters`. Both are normalized here so probing does not care which
    provider it is talking to, and so pricing evidence travels with the model when the
    provider publishes it.

    Args: catalog: The `data` array from either endpoint.
    Returns: One entry per model id, sorted by id.
    Raises: Nothing.
    Side effects: None.
    """
    seen: dict[str, dict] = {}
    for entry in catalog:
        name = entry.get("model_name") or entry.get("id")
        if not name or name in seen:
            continue
        info = entry.get("model_info") or {}
        if info.get("mode") in NON_CHAT_MODES:
            continue
        if any(marker.lower() in name.lower() for marker in NON_CHAT_NAME_MARKERS):
            continue
        pricing = entry.get("pricing") or {}
        record = {"id": name,
                  "context_window": info.get("context_window") or entry.get("context_length"),
                  "supported_openai_params": (info.get("supported_openai_params")
                                              or entry.get("supported_parameters") or []),
                  "max_output_tokens": (info.get("max_output_tokens")
                                        or (entry.get("top_provider") or {}).get("max_completion_tokens")),
                  "price_per_input_token": pricing.get("prompt"),
                  "price_per_output_token": pricing.get("completion")}
        seen[name] = record
    return [seen[key] for key in sorted(seen)]


async def _one_call(client: httpx.AsyncClient, url: str, headers: dict, model_id: str,
                    prompt: str, extra: dict, suffix: str = "") -> dict:
    """Send one probe request and reduce the response to the facts a verdict needs.

    Args: client: open async client; url: chat-completions endpoint; headers: auth headers;
        model_id: exact provider model id; prompt: throwaway probe text; extra: additional body fields.
    Returns: Mapping with status, latency, reasoning/answer sizes and finish reason.
    Raises: Nothing; transport and protocol failures are returned as `error`.
    Side effects: One provider request. No corpus data leaves the machine.
    """
    body = {"model": model_id, "messages": [{"role": "user", "content": prompt + suffix}],
            "temperature": 0.5, "top_p": 1.0, "seed": 7,
            "max_tokens": _PROBE_MAX_TOKENS, "stream": False, **extra}
    started = time.monotonic()
    try:
        response = await client.post(url, headers=headers, json=body)
    except Exception as error:  # noqa: BLE001 - a probe records failure, it does not raise
        return {"ok": False, "error": type(error).__name__, "seconds": time.monotonic() - started}
    elapsed = time.monotonic() - started
    if response.status_code != 200:
        return {"ok": False, "error": f"HTTP {response.status_code}", "seconds": elapsed,
                "detail": response.text[:300]}
    try:
        payload = response.json()
    except ValueError:
        return {"ok": False, "error": "unparsable JSON envelope", "seconds": elapsed}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"ok": False, "error": "no choices in envelope", "seconds": elapsed,
                "detail": json.dumps(payload)[:300]}
    message = choices[0].get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    answer = message.get("content") or ""
    usage = payload.get("usage") or {}
    try:
        parsed_object = isinstance(json.loads(answer), dict)
    except ValueError:
        parsed_object = False
    return {"ok": True, "seconds": elapsed, "provider": payload.get("provider"),
            "reasoning_chars": len(reasoning),
            "answer_chars": len(answer), "answer": answer[:200],
            "answer_is_json_object": parsed_object,
            "finish_reason": choices[0].get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens")}


def _hidden_reasoning(probe: dict) -> bool:
    """Detect billed generation that the response does not show.

    A switch can suppress the reasoning FIELD while the model still generates and bills the
    tokens. That reads as a working switch and is not one: the cost is unchanged and the
    latency is unchanged. Completion tokens far above what the visible answer needs is the
    signature, since the answer is the only thing those tokens could otherwise be.

    Args: probe: One `_one_call` result.
    Returns: True when billed output greatly exceeds the visible answer.
    Raises: Nothing.
    Side effects: None.
    """
    produced = probe.get("completion_tokens")
    if not isinstance(produced, int):
        return False
    visible = probe.get("answer_chars", 0) / 4
    return produced > 3 * visible + _HIDDEN_REASONING_SLACK


async def _probe_one(client: httpx.AsyncClient, url: str, headers: dict, model: dict,
                     semaphore: asyncio.Semaphore) -> dict:
    """Probe one model's reasoning, structured output and sampling support.

    Deployments disagree on how reasoning is switched off, so every candidate is tried and
    the one that works is recorded rather than assumed. A candidate counts only if it both
    silences the reasoning channel AND stops the tokens being produced.

    Args: client: open async client; url: chat-completions endpoint; headers: auth headers;
        model: catalog entry from `chat_models`; semaphore: provider concurrency guard.
    Returns: Catalog facts, raw probe results, the working switch, and a verdict.
    Raises: Nothing.
    Side effects: Up to five short provider requests per model.
    """
    async with semaphore:
        baseline = await _one_call(client, url, headers, model["id"], _PROBE_PROMPT, {})
        attempts, switched = [], None
        switch_name = switch_fields = switch_suffix = None
        for name, fields, suffix in REASONING_OFF_CANDIDATES:
            tries = [await _one_call(client, url, headers, model["id"], _PROBE_PROMPT, fields, suffix)
                     for _ in range(_CANDIDATE_ATTEMPTS)]
            routes = sorted({probe.get("provider") for probe in tries if probe.get("provider")})
            worst = max(tries, key=lambda probe: (not probe.get("ok"),
                                                  probe.get("reasoning_chars") or 0,
                                                  probe.get("completion_tokens") or 0))
            attempts.append({"switch": name, "attempts": len(tries), "routes": routes, **worst})
            if switched is None or (worst.get("ok") and not switched.get("ok")):
                switched, switch_name, switch_fields, switch_suffix = worst, name, fields, suffix
            # EVERY attempt must be clean. Accepting the best of several would record a
            # control that only some routes honour, which is how a switch that works one
            # time in eight gets written into the configuration as if it always works.
            if all(probe.get("ok") and probe.get("answer_chars")
                   and (probe.get("reasoning_chars") or 0) <= _RESIDUAL_REASONING_CHARS
                   and not _hidden_reasoning(probe)
                   for probe in tries):
                switched, switch_name, switch_fields, switch_suffix = worst, name, fields, suffix
                break
        structured = await _one_call(client, url, headers, model["id"], _JSON_PROBE_PROMPT,
                                     {"response_format": {"type": "json_object"}})

    baseline_ok = bool(baseline.get("ok"))
    baseline_reasoning = int(baseline.get("reasoning_chars") or 0)
    switched = switched or {}
    switch_answers = bool(switched.get("ok") and switched.get("answer_chars"))
    switched_reasoning = int(switched.get("reasoning_chars") or 0)
    truncated = any(probe.get("finish_reason") == "length" for probe in (baseline, switched))

    # Order is the correctness argument. Reasoning observed under the switch refutes it
    # outright, whatever the baseline did, so it is tested first. Reasoning that is merely
    # hidden is treated the same way, because the bill and the latency are identical. Only
    # once the switched call is known clean does the baseline matter, and an unusable
    # baseline then leaves the question open rather than answering it favourably.
    if not switched.get("ok"):
        verdict, usable = "probe_failed", False
    elif not switch_answers:
        verdict, usable = "no_answer_with_switch", False
    elif switched_reasoning > _RESIDUAL_REASONING_CHARS:
        verdict, usable = "switch_ignored", False
    elif _hidden_reasoning(switched):
        verdict, usable = "reasoning_hidden_not_stopped", False
    elif truncated:
        verdict, usable = "inconclusive_truncated", False
    elif not baseline_ok:
        verdict, usable = "inconclusive_baseline", False
    elif baseline_reasoning > _RESIDUAL_REASONING_CHARS or _hidden_reasoning(baseline):
        verdict, usable = "switch_disables_reasoning", True
    else:
        verdict, usable = "never_reasons", True

    structured_ok = bool(structured.get("ok") and structured.get("answer_is_json_object"))
    declared = set(model.get("supported_openai_params") or ())
    accepted = bool(switched.get("ok"))
    sampling = {name: bool(accepted and name in declared)
                for name in ("temperature", "top_p", "seed")}
    return {**model, "baseline": baseline, "switched": switched, "structured": structured,
            "switch_attempts": attempts, "verdict": verdict, "usable": usable,
            "structured_output": structured_ok, "sampling": sampling, "truncated": truncated,
            "reasoning_switch": switch_name if verdict == "switch_disables_reasoning" else None,
            "reasoning_fields": switch_fields if verdict == "switch_disables_reasoning" else None,
            "reasoning_prompt_suffix": switch_suffix if verdict == "switch_disables_reasoning" else None,
            "reasoning_control": verdict == "switch_disables_reasoning"}


async def probe_provider(*, base_url: str, api_key: str, path: str, models: list[dict],
                         concurrency: int, timeout_seconds: float) -> list[dict]:
    """Probe every supplied model against one provider deployment.

    Args: base_url: provider root; api_key: bearer credential, never logged or stored;
        path: chat-completions path; models: catalog entries; concurrency: parallel models;
        timeout_seconds: per-request ceiling.
    Returns: One verdict mapping per model, in the supplied order.
    Raises: Nothing; per-model failures are reported in their own mapping.
    Side effects: Three short provider requests per model.
    """
    url = base_url.rstrip("/") + path
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    semaphore = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        return list(await asyncio.gather(
            *(_probe_one(client, url, headers, model, semaphore) for model in models)))


def fetch_catalog(*, base_url: str, api_key: str, timeout_seconds: float = 60.0) -> list[dict]:
    """Read the provider's own model catalog, from whichever endpoint it serves.

    LiteLLM deployments expose /model/info; OpenRouter and OpenAI-shaped gateways expose
    /models. Both are tried rather than assumed from the URL, because the endpoint is a
    property of the deployment and guessing it from a hostname is exactly the kind of
    inference this module exists to avoid.

    Args: base_url: provider root; api_key: bearer credential; timeout_seconds: request ceiling.
    Returns: The raw `data` array from the endpoint that answered.
    Raises: LabError when neither endpoint yields a catalog.
    Side effects: One or two provider requests.
    """
    root = base_url.rstrip("/")
    headers = {"Authorization": "Bearer " + api_key}
    failures = []
    for path in ("/model/info", "/models"):
        try:
            response = httpx.get(root + path, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json().get("data") or []
            if data:
                return data
            failures.append(f"{path}: empty")
        except Exception as error:  # noqa: BLE001 - each endpoint is optional
            failures.append(f"{path}: {type(error).__name__}")
    raise LabError("Could not read the provider model catalog (" + "; ".join(failures) + ")")


def _quote(value: str) -> str:
    """Render a string as a double-quoted YAML scalar.

    Args: value: Text to quote.
    Returns: The quoted scalar.
    Raises: Nothing.
    Side effects: None.
    """
    # Newlines are escaped, not emitted raw: a control suffix such as "\n/no_think" would
    # otherwise break the scalar across lines and silently change what gets sent.
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def models_block(results: list[dict], *, provider_label: str, base_url: str,
                 verified_at: str | None = None) -> str:
    """Render probe verdicts as the `models:` block of a provider in parameters.yml.

    Every capability written here is backed by an observation in `results`; nothing is
    inferred from a model name. A model whose reasoning could not be switched off is
    emitted disabled, with the measurement that disqualified it recorded beside it.

    Args: results: verdicts from `probe_provider`; provider_label: name for display strings;
        base_url: deployment probed, recorded as evidence; verified_at: ISO date of the probe.
    Returns: YAML text indented to sit under `providers.<name>:`.
    Raises: Nothing.
    Side effects: None.
    """
    when = verified_at or date.today().isoformat()
    source = (f"iadopt-lab probe-models against {base_url}: context_window and parameter support "
              "from the LiteLLM /model/info catalog; reasoning control, structured output and "
              "parameter acceptance from live chat/completions probes. Sampling capabilities record "
              "that the deployment ACCEPTED the parameter and that its catalog lists it, not that "
              "the value was shown to change the output.")
    lines = ["    models:"]
    for row in results:
        note = {
            "switch_disables_reasoning": (
                f"Reasoning switched off: {row['baseline'].get('reasoning_chars', 0)} characters of "
                f"thinking with no control, 0 with enable_thinking=false."),
            "never_reasons": (
                "Emitted no reasoning in either probe, so there is no control to declare. This "
                "records two clean observations, not a guarantee the model never reasons."),
            "switch_ignored": (
                f"DISABLED: enable_thinking=false did not stop reasoning "
                f"({row['switched'].get('reasoning_chars', 0)} characters still produced)."),
            "no_answer_with_switch": "DISABLED: returned an empty answer under enable_thinking=false.",
            "probe_failed": f"DISABLED: probe failed ({row['switched'].get('error', 'unknown')}).",
            "inconclusive_truncated": (
                "DISABLED: a probe stopped at the output ceiling, so neither observation "
                "establishes the model's reasoning behaviour."),
            "inconclusive_baseline": (
                f"DISABLED: the uncontrolled baseline failed "
                f"({row['baseline'].get('error', 'unknown')}), so a silent switched call cannot be "
                "told apart from a model that never reasons."),
            "reasoning_hidden_not_stopped": (
                f"DISABLED: the switch hid the reasoning channel but still billed "
                f"{row['switched'].get('completion_tokens')} completion tokens for a "
                f"{row['switched'].get('answer_chars')}-character answer. Suppressing the field "
                "is not suppressing the generation."),
        }[row["verdict"]]
        context = row.get("context_window")
        lines += [
            f"      - id: {_quote(row['id'])}",
            f"        display_name: {_quote(row['id'] + ' on ' + provider_label)}",
            f"        enabled: {'true' if row['usable'] else 'false'}",
            f"        # {note}",
            "        revision: null",
            f"        context_window_tokens: {context if isinstance(context, int) else 'null'}",
        ]
        if not row["usable"]:
            # An unusable model carries no capability claim: it was never established.
            lines += ["        capabilities: null", "        reasoning_profiles: []"]
            continue
        sampling = row.get("sampling") or {}
        lines += [
            "        capabilities:",
            f"          temperature: {'true' if sampling.get('temperature') else 'false'}",
            f"          top_p: {'true' if sampling.get('top_p') else 'false'}",
            f"          seed: {'true' if sampling.get('seed') else 'false'}",
            f"          structured_output: {'true' if row['structured_output'] else 'false'}",
            f"          reasoning_control: {'true' if row['reasoning_control'] else 'false'}",
        ]
        # The provider's own completion cap, when it publishes one. Recording it lets the
        # configuration reject a grid ceiling the model cannot honour, instead of the
        # provider clamping or refusing mid-campaign.
        if isinstance(row.get("max_output_tokens"), int):
            lines.append(f"          max_output_tokens: {row['max_output_tokens']}")
        lines += [
            "          evidence:",
            f"            source: {_quote(source)}",
            f"            verified_at: {_quote(when)}",
            "        reasoning_profiles:",
        ]
        if row["reasoning_control"]:
            # The switch that was measured to work on THIS deployment, not a default.
            lines += ['          - mode: "disabled"']
            if row["reasoning_fields"]:
                rendered = yaml.safe_dump(row["reasoning_fields"], default_flow_style=False,
                                          sort_keys=True).rstrip("\n").splitlines()
                lines += ["            request_fields:"]
                lines += ["              " + line for line in rendered]
            else:
                lines += ["            request_fields: {}"]
            if row.get("reasoning_prompt_suffix"):
                lines += [f"            prompt_suffix: {_quote(row['reasoning_prompt_suffix'])}"]
        else:
            lines += ['          - mode: "not_applicable"', "            request_fields: {}"]
    return "\n".join(lines) + "\n"


def replace_models_block(text: str, provider: str, block: str) -> str:
    """Substitute one provider's `models:` block, leaving the rest of the file byte-identical.

    parameters.yml is human-edited and carries the reasoning behind every value in its
    comments, so it is edited in place as text rather than round-tripped through a YAML
    serializer that would discard them.

    Args: text: current parameters.yml contents; provider: provider key under `providers:`;
        block: replacement block from `models_block`.
    Returns: The updated file text.
    Raises: LabError when the provider or its models block cannot be located unambiguously.
    Side effects: None; the caller decides whether to write.
    """
    lines = text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip("\n") == f"  {provider}:")
    except StopIteration:
        raise LabError(f"No providers.{provider} block in parameters.yml") from None
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip() and not lines[i].startswith("    ")), len(lines))
    try:
        models_at = next(i for i in range(start + 1, end) if lines[i].rstrip("\n") == "    models:")
    except StopIteration:
        raise LabError(f"No models: block under providers.{provider}") from None
    models_end = next((i for i in range(models_at + 1, end)
                       if lines[i].strip() and not lines[i].startswith("      ")), end)
    return "".join(lines[:models_at]) + block + "".join(lines[models_end:])


def write_models_block(path: Path, provider: str, block: str) -> None:
    """Rewrite one provider's models block in parameters.yml on disk.

    Args: path: parameters.yml; provider: provider key; block: replacement block.
    Returns: None.
    Raises: LabError when the block cannot be located.
    Side effects: Overwrites parameters.yml.
    """
    text = path.read_text(encoding="utf-8")
    path.write_text(replace_models_block(text, provider, block), encoding="utf-8")
