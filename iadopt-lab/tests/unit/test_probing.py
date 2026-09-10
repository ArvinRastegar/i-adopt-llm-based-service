"""Guards for capability probing and the parameters.yml models-block rewrite.

Two properties make the block rewrite safe, and both are pinned below: everything
outside the targeted block survives byte-identically, and whatever is written
still resolves under the real schema.

The verdict tests exist because a probe that cannot tell a working switch from an
absent one will write a configuration that sends no switch at all. The ordering
they pin is the correctness argument: reasoning seen under the switch refutes the
switch whatever the baseline did, and an unusable baseline leaves the question
open rather than answering it in the model's favour.
"""

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest

from iadopt_lab.domain import LabError
from iadopt_lab.probing import (
    REASONING_OFF_FIELDS,
    _probe_one,
    chat_models,
    models_block,
    replace_models_block,
)

ROOT = Path(__file__).resolve().parents[2]
_LIVE_PARAMS = (ROOT / "parameters.yml").read_text(encoding="utf-8")
# Pin the provider under test rather than inheriting whichever campaign is configured:
# these tests are about the block rewrite, not about today's selected provider.
PARAMS = re.sub(r"^  providers: \[.*\]$", "  providers: [psnc]", _LIVE_PARAMS, count=1, flags=re.M)


def _run(*, baseline=None, switched=None, structured=None,
         params=("temperature", "top_p", "seed"), catalog=None):
    """Drive one model through the probe with independently scripted responses.

    The handler dispatches on the request body, not on call order, because the probe now
    tries several candidate off-switches and stops at the first that works — so the number
    of calls, and the position of the structured probe, both vary by scenario.
    """
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        calls.append(body)
        if "response_format" in body:
            spec = structured or {}
        elif (any(key in body for key in ("chat_template_kwargs", "reasoning"))
              or "/no_think" in body["messages"][0]["content"]):
            spec = switched or {}
        else:
            spec = baseline or {}
        if spec.get("status", 200) != 200:
            return httpx.Response(spec["status"], text="upstream refused")
        message = {"content": spec.get("content", "ok")}
        if spec.get("reasoning"):
            message["reasoning_content"] = "t" * spec["reasoning"]
        return httpx.Response(200, json={
            "choices": [{"message": message, "finish_reason": spec.get("finish", "stop")}],
            "usage": {"completion_tokens": spec.get("completion_tokens", 4)}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _probe_one(client, "https://p/chat/completions", {},
                                    {"id": "M", "context_window": 4096,
                                     "supported_openai_params": list(params),
                                     **(catalog or {})},
                                    asyncio.Semaphore(1))

    return asyncio.run(go()), calls


def test_switch_that_silences_reasoning_is_a_declarable_control():
    result, calls = _run(baseline={"reasoning": 5000}, switched={})
    assert result["verdict"] == "switch_disables_reasoning"
    assert result["usable"] and result["reasoning_control"]
    # The switch must be absent from the baseline, or the comparison proves nothing.
    assert not any(k in calls[0] for k in ("chat_template_kwargs", "reasoning"))
    assert result["reasoning_switch"] and result["reasoning_fields"]


def test_model_that_never_reasons_declares_no_control():
    result, _ = _run(baseline={}, switched={})
    assert result["verdict"] == "never_reasons"
    # Usable for a reasoning-off campaign, but claiming the switch did something
    # would be a fabrication, so no control is asserted.
    assert result["usable"] and not result["reasoning_control"]


def test_reasoning_under_the_switch_refutes_it_whatever_the_baseline_did():
    """The ordering bug: a failed baseline used to make this read as `never_reasons`."""
    result, _ = _run(baseline={"status": 503}, switched={"reasoning": 100})
    assert result["verdict"] == "switch_ignored"
    assert not result["usable"] and not result["reasoning_control"]


def test_failed_baseline_alone_is_inconclusive_not_favourable():
    """A silent switched call and a model that never reasons look identical here."""
    result, _ = _run(baseline={"status": 503}, switched={})
    assert result["verdict"] == "inconclusive_baseline"
    assert not result["usable"]


def test_a_truncated_probe_settles_nothing():
    result, _ = _run(baseline={"reasoning": 500, "finish": "length"}, switched={})
    assert result["verdict"] == "inconclusive_truncated"
    assert not result["usable"]


def test_ignored_switch_disqualifies_the_model():
    result, _ = _run(baseline={"reasoning": 5000}, switched={"reasoning": 4800})
    assert result["verdict"] == "switch_ignored"
    assert not result["usable"]


def test_empty_answer_under_the_switch_disqualifies_the_model():
    result, _ = _run(baseline={"reasoning": 5000}, switched={"content": ""})
    assert result["verdict"] == "no_answer_with_switch"
    assert not result["usable"]


def test_transport_failure_is_recorded_not_raised():
    result, _ = _run(baseline={}, switched={"status": 503})
    assert result["verdict"] == "probe_failed"
    assert not result["usable"]


def test_structured_output_requires_a_parsed_json_object():
    """Prose in the answer is evidence the mode is NOT enforced."""
    prose, _ = _run(baseline={}, switched={}, structured={"content": "Sure! Here you go."})
    assert prose["structured_output"] is False
    real, _ = _run(baseline={}, switched={}, structured={"content": '{"ok": true}'})
    assert real["structured_output"] is True


def test_sampling_support_needs_both_acceptance_and_a_catalog_declaration():
    declared, _ = _run(baseline={}, switched={})
    assert declared["sampling"] == {"temperature": True, "top_p": True, "seed": True}
    # A deployment whose catalog lists nothing cannot have its controls asserted from
    # the mere fact that a request carrying them returned 200.
    undeclared, _ = _run(baseline={}, switched={}, params=())
    assert undeclared["sampling"] == {"temperature": False, "top_p": False, "seed": False}


def test_catalog_excludes_non_chat_modes_and_duplicates():
    catalog = chat_models([
        {"model_name": "Chat-A", "model_info": {"context_window": 1024}},
        {"model_name": "Chat-A", "model_info": {"context_window": 999}},
        {"model_name": "Embed", "model_info": {"mode": "embedding"}},
        {"model_name": "bge-reranker", "model_info": {"mode": "rerank"}},
        {"model_name": "whisper-large-v3", "model_info": {}},
        {"model_name": "Nanonets-OCR-s", "model_info": {}},
    ])
    assert [entry["id"] for entry in catalog] == ["Chat-A"]
    # The first entry wins, so a duplicate cannot silently change a context window.
    assert catalog[0]["context_window"] == 1024


def _fake(model_id, verdict, *, usable, reasoning_control, sampling=True, max_output=None):
    return {"id": model_id, "context_window": 4096, "max_output_tokens": max_output,
            "reasoning_switch": "chat_template_kwargs.enable_thinking=false" if reasoning_control else None,
            "reasoning_fields": {"chat_template_kwargs": {"enable_thinking": False}} if reasoning_control else None,
            "baseline": {"ok": True, "reasoning_chars": 900},
            "switched": {"ok": True, "reasoning_chars": 0, "answer_chars": 4, "seconds": 1.0},
            "structured": {"ok": True, "answer_is_json_object": True}, "verdict": verdict,
            "usable": usable, "structured_output": True, "reasoning_control": reasoning_control,
            "truncated": False,
            "sampling": {name: sampling for name in ("temperature", "top_p", "seed")}}


def test_generated_block_resolves_under_the_real_schema(tmp_path):
    from iadopt_lab.configuration import load_parameters, resolve_configuration

    block = models_block(
        [_fake("Good", "switch_disables_reasoning", usable=True, reasoning_control=True),
         _fake("Plain", "never_reasons", usable=True, reasoning_control=False),
         _fake("Bad", "switch_ignored", usable=False, reasoning_control=False),
         _fake("Unknown", "inconclusive_baseline", usable=False, reasoning_control=False)],
        provider_label="PSNC", base_url="https://p/v1")
    target = tmp_path / "parameters.yml"
    target.write_text(replace_models_block(PARAMS, "psnc", block), encoding="utf-8")

    resolved = resolve_configuration(load_parameters(target))
    # Only probed-usable models survive, and none raises a blocking issue.
    assert [m["id"] for m in resolved.data["providers"]["psnc"]["models"]] == ["Good", "Plain"]
    assert not [issue for issue in resolved.issues if "psnc" in issue]


def test_unestablished_sampling_support_blocks_the_configuration(tmp_path):
    """Capabilities that were never established must not silently pass live readiness."""
    from iadopt_lab.configuration import load_parameters, resolve_configuration

    block = models_block([_fake("Weak", "never_reasons", usable=True, reasoning_control=False,
                                sampling=False)], provider_label="PSNC", base_url="https://p/v1")
    target = tmp_path / "parameters.yml"
    target.write_text(replace_models_block(PARAMS, "psnc", block), encoding="utf-8")
    assert [issue for issue in resolve_configuration(load_parameters(target)).issues
            if "top_p" in issue or "temperature" in issue]


def test_rewrite_leaves_the_rest_of_the_file_byte_identical():
    block = models_block([_fake("Only", "never_reasons", usable=True, reasoning_control=False)],
                         provider_label="PSNC", base_url="https://p/v1")
    updated = replace_models_block(PARAMS, "psnc", block)
    # Everything from the next provider onwards, including its comments, is untouched.
    assert PARAMS.split("  openrouter:")[1] == updated.split("  openrouter:")[1]
    assert PARAMS.split("    models:")[0] == updated.split("    models:")[0]
    psnc_models = updated.split("    models:")[1].split("  openrouter:")[0]
    assert '- id: "Only"' in psnc_models and '- id: "GLM-5.2"' not in psnc_models


def test_rewrite_refuses_an_unknown_provider():
    with pytest.raises(LabError):
        replace_models_block(PARAMS, "nonexistent", "    models: []\n")


def test_reasoning_off_fields_stay_within_the_permitted_native_set():
    from iadopt_lab.providers.base import NATIVE_FIELDS

    assert set(REASONING_OFF_FIELDS) <= NATIVE_FIELDS


def test_a_switch_that_hides_reasoning_without_stopping_it_is_refused():
    """Suppressing the reasoning FIELD is not suppressing the billed generation.

    `reasoning.exclude` on OpenRouter removes reasoning from the response while the model
    still generates and bills those tokens. Counting reasoning characters alone would call
    that a working off-switch and be wrong about both cost and latency.
    """
    result, _ = _run(baseline={"reasoning": 5000, "completion_tokens": 4000},
                     switched={"content": "ok", "completion_tokens": 4000})
    assert result["verdict"] == "reasoning_hidden_not_stopped"
    assert not result["usable"] and not result["reasoning_control"]


def test_the_measured_switch_is_what_gets_written(tmp_path):
    """The block records the switch that worked here, never a hardcoded default."""
    from iadopt_lab.probing import REASONING_OFF_CANDIDATES

    # The vLLM switch is a no-op for this model; the second candidate is the one that works.
    fields = dict(REASONING_OFF_CANDIDATES[1][1])
    row = _fake("M", "switch_disables_reasoning", usable=True, reasoning_control=True)
    row["reasoning_fields"] = fields
    block = models_block([row], provider_label="OPENROUTER", base_url="https://p/v1")
    assert "reasoning:" in block and "enabled: false" in block
    assert "chat_template_kwargs" not in block


def test_a_published_output_cap_is_recorded():
    """A model capping completion below the grid ceiling must be visible to validation."""
    block = models_block([_fake("Small", "never_reasons", usable=True,
                                reasoning_control=False, max_output=8192)],
                         provider_label="OPENROUTER", base_url="https://p/v1")
    assert "max_output_tokens: 8192" in block
