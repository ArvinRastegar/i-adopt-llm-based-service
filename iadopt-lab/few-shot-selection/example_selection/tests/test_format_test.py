"""Offline tests for format_test.py, the example-layout side test. A stub replaces PSNC."""

from __future__ import annotations

import json

import format_test as ft
import pytest

DOCUMENTED_TARGET = "number of persons receiving welfare"


@pytest.fixture(scope="module")
def inputs():
    """Load the real corpus, targets, examples, template and schema once per module."""
    return ft.load_inputs()


def _documented_prompt() -> str:
    """Read the prompt the study recorded verbatim in best-configuration-session.md."""
    doc = (ft.HERE / "best-configuration-session.md").read_text(encoding="utf-8")
    return doc.split("````text\n", 1)[1].split("\n````", 1)[0]


def _target(inputs, prefix):
    """Find the corpus record whose definition starts with `prefix`."""
    return next(r for r in inputs["records"].values() if r["definition"].startswith(prefix))


def test_format_a_is_the_measured_prompt(inputs):
    """Format A must be byte-identical to the prompt the study recorded."""
    target = _target(inputs, DOCUMENTED_TARGET)
    assert ft.prompt_for(inputs, "A", target["definition"]) == _documented_prompt()


def test_only_the_examples_block_differs(inputs):
    """Everything outside the examples block is identical in layouts A and B."""
    definition = _target(inputs, DOCUMENTED_TARGET)["definition"]
    a, b = (ft.prompt_for(inputs, fmt, definition) for fmt in ft.FORMATS)
    block_a, block_b = ft.examples_block(a), ft.examples_block(b)
    assert block_a != block_b
    assert a.replace(block_a, "<EXAMPLES>") == b.replace(block_b, "<EXAMPLES>")


def test_format_b_shows_every_answer_as_the_expected_output(inputs):
    """Each B example: definition, then JSON that parses back to the gold, in schema order."""
    block = ft.examples_block_b(inputs["examples"])
    parts = block.split("\n\nExample ")
    assert len(parts) == len(ft.EXAMPLE_IDS) == 25
    schema = json.loads(inputs["schema_text"])
    for n, (part, example) in enumerate(zip(parts, inputs["examples"], strict=True), 1):
        header, answer = part.split("\nOutput:\n", 1)
        assert header.removeprefix("Example ").startswith(f"{n}\nDefinition: ")
        assert header.endswith(example["definition"])
        parsed = json.loads(answer)
        assert parsed == example["gold"]
        assert tuple(parsed) == tuple(schema["required"]) == ft.TOP_KEYS
        assert answer == json.dumps(parsed, ensure_ascii=False, indent=2)


def test_nested_order_follows_the_schema(inputs):
    """Systems and constraints are re-keyed in the order of the schema's `required` lists."""
    schema = json.loads(inputs["schema_text"])
    expected = []
    for option in schema["$defs"]["entity"]["oneOf"]:
        expected += [k for k in option.get("required", []) if k not in expected]
    expected += [k for k in schema["$defs"]["constraint"]["required"] if k not in expected]
    assert tuple(expected) == ft.NESTED_KEYS


def test_examples_are_the_study_winner_and_disjoint_from_targets(inputs):
    """The 25 examples are the study's top25, in its order, and never an evaluation target."""
    assert ft.candidate_hash(ft.EXAMPLE_IDS) == ft.EXAMPLE_CANDIDATE
    assert list(ft.EXAMPLE_IDS) == sorted(ft.EXAMPLE_IDS)
    assert len(inputs["targets"]) == 24
    assert not set(ft.EXAMPLE_IDS) & set(inputs["targets"])


def test_schedule_covers_the_grid_once(inputs):
    """Every call of the grid appears once, reproducibly, with each A/B pair adjacent."""
    jobs = ft.schedule(inputs["targets"])
    assert len(jobs) == len(set(jobs)) == 2 * 2 * 15 * 24
    assert ft.schedule(inputs["targets"]) == jobs  # seeded, so reproducible
    # A and B for one target and model are adjacent, so they meet the same server load.
    for first, second in zip(jobs[::2], jobs[1::2], strict=True):
        assert first[0] == second[0] and first[2:] == second[2:] and first[1] != second[1]


def _stub(answer='{"hasStatisticalModifier": ""}', *, status=200, reasoning="",
          prompt_tokens=4000, completion_tokens=100):
    """Build a stub transport that records every request and returns one fixed reply."""
    calls = []

    async def transport(body):
        """Record the request and return the fixed reply."""
        calls.append(body)
        if status != 200:
            return {"ok": False, "status": status, "error": f"HTTP {status}", "latency": 0.1}
        return {"ok": True, "status": 200, "latency": 0.1, "answer": answer,
                "reasoning": reasoning, "finish_reason": "stop",
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "served_model": body["model"]}

    return transport, calls


@pytest.mark.asyncio
async def test_run_records_every_call_and_resumes_without_repeats(inputs, tmp_path):
    """Each job is recorded once with the study's settings, and a resume sends nothing new."""
    jobs = ft.schedule(inputs["targets"][:2], repetitions=2)
    transport, calls = _stub()
    path = tmp_path / "calls.jsonl"
    result = await ft.run_grid(inputs, jobs, transport, calls_path=path, concurrency=4)
    assert result["issued"] == {"Qwen3.8-27B": 8, "GLM-5.2": 8}
    rows = ft.read_jsonl(path)
    assert {ft.call_key(r) for r in rows} == set(jobs)
    assert all(r["prompt_sha256"] and r["reasoning_chars"] == 0 for r in rows)
    body = calls[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.5 and body["max_tokens"] == 16000
    again = await ft.run_grid(inputs, jobs, transport, calls_path=path, concurrency=4)
    assert again["issued"] == {} and len(calls) == 16


@pytest.mark.asyncio
async def test_failed_calls_are_retried_on_resume(inputs, tmp_path):
    """A failed call is not treated as answered: the next run retries it."""
    jobs = ft.schedule(inputs["targets"][:1], models=["Qwen3.8-27B"], repetitions=1)
    path = tmp_path / "calls.jsonl"
    failing, _ = _stub(status=503)
    await ft.run_grid(inputs, jobs, failing, calls_path=path, concurrency=2)
    working, calls = _stub()
    await ft.run_grid(inputs, jobs, working, calls_path=path, concurrency=2)
    assert len(calls) == 2
    assert all(r["ok"] for r in ft.latest_records(path).values())


@pytest.mark.asyncio
async def test_credential_refusal_stops_everything(inputs, tmp_path):
    """One HTTP 401 stops every model after a single call."""
    jobs = ft.schedule(inputs["targets"], repetitions=1)
    transport, calls = _stub(status=401)
    result = await ft.run_grid(inputs, jobs, transport, calls_path=tmp_path / "c.jsonl",
                               concurrency=1)
    assert len(calls) == 1 and "*" in result["stopped"]


@pytest.mark.asyncio
async def test_spend_cap_stops_the_model(inputs, tmp_path):
    """The model stops as soon as its list-price spend passes the cap."""
    jobs = ft.schedule(inputs["targets"], models=["GLM-5.3"], repetitions=1)
    # 2.5M prompt tokens per call is about $15.34 at the list price: the second call passes
    # the $29.87 cap.
    transport, calls = _stub(prompt_tokens=2_500_000)
    result = await ft.run_grid(inputs, jobs, transport, calls_path=tmp_path / "c.jsonl",
                               concurrency=1)
    assert len(calls) == 2 and "GLM-5.3" in result["stopped"]


@pytest.mark.asyncio
async def test_early_failures_stop_the_model(inputs, tmp_path):
    """More than five failures among a model's first 48 calls stop that model."""
    jobs = ft.schedule(inputs["targets"], models=["Qwen3.8-27B"], repetitions=1)
    transport, calls = _stub(status=503)
    result = await ft.run_grid(inputs, jobs, transport, calls_path=tmp_path / "c.jsonl",
                               concurrency=1)
    assert len(calls) == ft.EARLY_FAILURES + 1 and "Qwen3.8-27B" in result["stopped"]


@pytest.mark.asyncio
async def test_reasoning_text_stops_the_model(inputs, tmp_path):
    """More than three answers carrying reasoning text stop that model."""
    jobs = ft.schedule(inputs["targets"], models=["GLM-5.3"], repetitions=1)
    transport, calls = _stub(reasoning="thinking...")
    result = await ft.run_grid(inputs, jobs, transport, calls_path=tmp_path / "c.jsonl",
                               concurrency=1)
    assert len(calls) == ft.THINKING_LIMIT + 1 and "GLM-5.3" in result["stopped"]


@pytest.mark.asyncio
async def test_probe_uses_an_invented_target_and_gates_on_thinking(inputs, tmp_path):
    """The probe names no corpus target, and reasoning text fails the gate."""
    transport, calls = _stub()
    rows = await ft.probe(inputs, ["Qwen3.8-27B", "GLM-5.3"], transport,
                          path=tmp_path / "probe.jsonl")
    assert len(calls) == 5
    assert all(ft.PROBE_DEFINITION in c["messages"][0]["content"] for c in calls)
    assert not any(r["definition"] == ft.PROBE_DEFINITION for r in inputs["records"].values())
    unswitched = [c for c in calls if "chat_template_kwargs" not in c]
    assert len(unswitched) == 1 and unswitched[0]["model"] == "GLM-5.3"
    assert ft.probe_passed(rows, "Qwen3.8-27B") and ft.probe_passed(rows, "GLM-5.3")
    thinking, _ = _stub(reasoning="hmm")
    rows = await ft.probe(inputs, ["GLM-5.3"], thinking, path=tmp_path / "probe.jsonl")
    assert not ft.probe_passed(rows, "GLM-5.3")


def test_holm_and_welch():
    """Holm adjusts a known pair correctly, and Welch recovers a known difference."""
    adjusted = ft.holm({"x": 0.01, "y": 0.04})
    assert adjusted == {"x": 0.02, "y": 0.04}
    result = ft.welch([0.5, 0.52, 0.51, 0.53], [0.45, 0.46, 0.44, 0.47])
    assert result["difference"] == pytest.approx(0.06)
    assert result["ci_low"] < 0.06 < result["ci_high"] and result["p"] < 0.001
