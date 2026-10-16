"""The chat prompt and how its answers are read: labels, requests, the parser's refusals, pinned text."""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from iadopt_linking import experiment
from iadopt_linking.chat_choice import (
    CHAT_PASSES,
    LABELS,
    MAX_TOKENS,
    MIN_LABEL_MASS,
    MODEL,
    NONE_LABEL,
    REASONING_OFF,
    SYSTEM_PROMPT,
    TOP_LOGPROBS,
    ChatAnswerError,
    build_tasks,
    chat_request,
    estimate,
    label_to_option,
    option_label,
    parse_chat_choice,
    prompt_fingerprint,
    prompt_plan,
    reported_cost,
    user_message,
)
from iadopt_linking.jev import NONE_KEY, PASSES, JevError, encode, request_sha256
from iadopt_linking.jev_inputs import build_tasks as jev_build_tasks
from iadopt_linking.jev_inputs import load_baseline

BASELINE = Path(__file__).resolve().parents[1] / "results" / "baseline-20261014"
LAB_PARAMETERS = Path(__file__).resolve().parents[2] / "iadopt-lab" / "parameters.yml"
PILOT = ["EWCT", "C8_CWDFraction"]
CACHED_GOLD = functools.cache(experiment.load_gold_links)
FINGERPRINT = "20d69e9a52e2dfe38c19250c0ce7fea2761b5352440b1b38bae4b63735ff6f24"
GOLDEN_SHA256 = "c4502acbad2111ea7ec7a529b567d53e4d8b8d58a8556ad54e1a901fd6028e49"


@pytest.fixture(autouse=True)
def read_the_corpus_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment, "load_gold_links", CACHED_GOLD)


def cand(vocabulary: str, local_id: str, label: str, description: str = "") -> dict[str, Any]:
    return {"source": vocabulary, "rank": 1, "iri": f"http://x/{local_id}", "vocabulary": vocabulary,
            "local_id": local_id, "label": label, "description": description, "extras": {}}


POOL = [cand("wikidata", "Q1", "bridge", "structure built to span a physical obstacle"),
        cand("qudt", "L", "Bridge Length"), cand("om", "M", "bridge", "x" * 400)]
DEFINITION = "Length of a suspension bridge in metres."


# --- labels ---

def test_the_labels_are_single_unambiguous_characters() -> None:
    assert len(LABELS) == 33 and len(set(LABELS)) == 33
    assert all(len(label) == 1 for label in LABELS)
    assert NONE_LABEL == "0" and NONE_LABEL not in LABELS
    assert "I" not in LABELS and "O" not in LABELS and LABELS.isascii()
    assert [option_label(p) for p in (1, 9, 10, 11, 33)] == ["1", "9", "A", "B", "Z"]


def test_a_pool_of_thirty_fits_and_one_more_than_the_labels_does_not() -> None:
    assert option_label(30) == "W"
    with pytest.raises(ValueError):
        option_label(0)
    with pytest.raises(ValueError):
        option_label(34)


def test_each_label_names_the_option_of_its_position() -> None:
    assert label_to_option(3) == {"1": "c01", "2": "c02", "3": "c03", "0": NONE_KEY}
    assert label_to_option(10)["A"] == "c10"


# --- the request ---

def test_the_user_message_is_exactly_this() -> None:
    expected = (
        "term: bridge\nvariable_definition: Length of a suspension bridge in metres.\n\n"
        "candidates:\n"
        f"1. \"bridge\": {'x' * 300}…\n"
        "2. \"bridge\": structure built to span a physical obstacle\n"
        "3. \"Bridge Length\"\n"
        "0. none of these: no candidate is the same concept as the term\n\n"
        "Answer with one label.")
    assert user_message("bridge", DEFINITION, POOL, [2, 0, 1], 300) == expected


def test_nothing_that_identifies_a_candidate_is_shown() -> None:
    text = user_message("bridge", DEFINITION, POOL, [2, 0, 1], 300)
    for hidden in ("wikidata", "qudt", "http", "Q1", "c01", "vocabulary", "local_id"):
        assert hidden not in text


def test_the_request_is_exactly_this() -> None:
    body = chat_request("bridge", DEFINITION, POOL, [2, 0, 1], max_description_chars=300)
    assert list(body) == ["model", "messages", "temperature", "max_tokens", "logprobs",
                          "top_logprobs", "stream", "chat_template_kwargs"]
    assert body["model"] == MODEL == "Qwen3.8-27B"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    assert (body["temperature"], body["max_tokens"], body["logprobs"], body["top_logprobs"],
            body["stream"]) == (0, MAX_TOKENS, True, TOP_LOGPROBS, False)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_same_request_always_has_the_same_bytes() -> None:
    body = chat_request("bridge", DEFINITION, POOL, [2, 0, 1], max_description_chars=300)
    assert request_sha256(body) == request_sha256(
        chat_request("bridge", DEFINITION, POOL, [2, 0, 1], max_description_chars=300))
    assert request_sha256(body) == GOLDEN_SHA256, request_sha256(body)
    assert encode(body).startswith(b'{"model":"Qwen3.8-27B","messages":[{"role":"system"')


def test_reasoning_is_switched_off_the_way_the_lab_measured() -> None:
    config = yaml.safe_load(LAB_PARAMETERS.read_text(encoding="utf-8"))
    models = {m["id"]: m for m in config["providers"]["psnc"]["models"]}
    profile = next(p for p in models[MODEL]["reasoning_profiles"] if p["mode"] == "disabled")
    assert REASONING_OFF[MODEL] == profile["request_fields"]
    assert config["providers"]["psnc"]["default_base_url"] == "https://llm.hpc.psnc.pl/v1"


def test_a_model_without_a_measured_reasoning_control_is_refused() -> None:
    with pytest.raises(ValueError, match="no measured control"):
        chat_request("t", "d", POOL, [0, 1, 2], max_description_chars=300, model="GLM-5.2")


@pytest.mark.parametrize(("pool", "order"), [
    ([], []), (POOL, [0, 1]), (POOL, [0, 0, 1]), (POOL, [0, 1, 3]),
    ([cand("wikidata", f"Q{i}", f"l{i}") for i in range(34)], list(range(34)))])
def test_an_impossible_order_or_pool_is_refused(pool: list[dict], order: list[int]) -> None:
    with pytest.raises(ValueError):
        chat_request("t", "d", pool, order, max_description_chars=300)


def test_the_prompt_is_frozen() -> None:
    assert prompt_fingerprint() == FINGERPRINT, prompt_fingerprint()
    plan = prompt_plan(300)
    assert plan["fingerprint"] == FINGERPRINT and plan["system_prompt"] == SYSTEM_PROMPT
    assert plan["request_parameters"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert plan["min_label_mass"] == MIN_LABEL_MASS == 0.5 and plan["max_description_chars"] == 300


def test_the_instructions_say_what_matters() -> None:
    for phrase in ("`term`", "`variable_definition`", "same concept", "broader, narrower or merely "
                   "related", "the whole variable", "option 0", "nothing else"):
        assert phrase in SYSTEM_PROMPT


# --- the same candidates in the same orders as the Jev run ---

def test_the_chat_requests_show_what_jev_was_shown_in_the_order_it_was_shown() -> None:
    baseline = load_baseline(BASELINE, only=PILOT)
    chat_tasks, chat_counts = build_tasks(baseline, CHAT_PASSES)
    jev_tasks, jev_counts = jev_build_tasks(baseline, [p for p in PASSES if p.arm == "choice"],
                                            session_id=None)
    assert chat_counts == jev_counts and len(chat_tasks) == len(jev_tasks) > 0
    for ours, theirs in zip(chat_tasks, jev_tasks, strict=True):
        assert (ours.key, ours.order, ours.shown, ours.offered) == (
            theirs.key, theirs.order, theirs.shown, theirs.offered)
        assert ours.sha256 == request_sha256(ours.body)


def test_only_choice_passes_are_built() -> None:
    baseline = load_baseline(BASELINE, only=PILOT)
    with pytest.raises(ValueError, match="listwise"):
        build_tasks(baseline, [p for p in PASSES if p.arm == "noul"])
    assert [p.name for p in CHAT_PASSES] == [
        "choice-a1", "choice-a1-repeat", "choice-a2-reversed", "choice-a3"]


def test_the_estimate_counts_requests_and_costs_nothing() -> None:
    baseline = load_baseline(BASELINE, only=PILOT)
    tasks, _ = build_tasks(baseline, CHAT_PASSES)
    summary = estimate(tasks)
    assert summary["requests"] == len(tasks) and summary["estimated_cost_usd"] == "0"
    assert summary["requests_by_pass"]["choice-a1"] == len(tasks) // 4
    assert summary["input_tokens_upper_estimate"] > summary["request_chars"] / 2
    assert reported_cost("anything") == reported_cost(None) == 0


# --- reading an answer ---

OPTIONS = ("c01", "c02", "c03", NONE_KEY)


def reply(top: list[tuple[str, float]], *, token: str | None = None, content: str | None = None,
          finish: str = "length", model: str = "Qwen3.8-27B", message: dict | None = None,
          usage: Any = None, logprobs: Any = "default", fingerprint: str | None = "vllm-0.26.0-x"
          ) -> str:
    first = {"token": top[0][0] if token is None else token, "bytes": [], "logprob": top[0][1],
             "top_logprobs": [{"token": t, "bytes": [], "logprob": lp} for t, lp in top]}
    body: dict[str, Any] = {
        "id": "chatcmpl-1", "model": model, "object": "chat.completion",
        "choices": [{"finish_reason": finish, "index": 0,
                     "message": {"content": top[0][0] if content is None else content,
                                 "role": "assistant", **(message or {})},
                     "logprobs": {"content": [first]} if logprobs == "default" else logprobs}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 1, "total_tokens": 121}
        if usage is None else usage}
    if fingerprint is not None:
        body["system_fingerprint"] = fingerprint
    return json.dumps(body)


def lp(p: float) -> float:
    return math.log(p)


GOOD = [("2", lp(0.6)), ("0", lp(0.2)), ("1", lp(0.1)), ("3", lp(0.05)), ("The", lp(0.04))]


def test_a_valid_answer_gives_a_distribution_over_the_options() -> None:
    answer = parse_chat_choice(reply(GOOD), OPTIONS)
    assert answer["choice"] == "c02" and answer["model"] == "Qwen3.8-27B"
    assert answer["provider"] == "psnc" and answer["cost_usd"] is None
    assert answer["system_fingerprint"] == "vllm-0.26.0-x"
    assert (answer["input_tokens"], answer["output_tokens"]) == (120, 1)
    mass = 0.6 + 0.2 + 0.1 + 0.05
    assert answer["label_mass"] == pytest.approx(mass) and answer["top_mass"] == pytest.approx(0.99)
    assert answer["labels_reported"] == 4
    p = answer["probabilities"]
    assert list(p) == list(OPTIONS)
    assert p["c02"] == pytest.approx(0.6 / mass) and p[NONE_KEY] == pytest.approx(0.2 / mass)
    assert sum(p.values()) == pytest.approx(1.0) and answer["probability_sum_drift"] < 1e-12
    assert answer["confidence"] == p["c02"]


def test_a_label_the_gateway_did_not_report_gets_zero_not_a_guess() -> None:
    answer = parse_chat_choice(reply([("1", lp(0.9)), ("0", lp(0.05))]), OPTIONS)
    assert answer["probabilities"]["c02"] == 0.0 and answer["probabilities"]["c03"] == 0.0
    assert answer["labels_reported"] == 2


def test_the_model_may_choose_none_of_these() -> None:
    answer = parse_chat_choice(reply([("0", lp(0.7)), ("2", lp(0.2))]), OPTIONS)
    assert answer["choice"] == NONE_KEY and answer["confidence"] == pytest.approx(0.7 / 0.9)


def test_a_leading_space_does_not_split_a_label() -> None:
    answer = parse_chat_choice(reply([("2", lp(0.5)), (" 2", lp(0.1)), ("0", lp(0.3))]), OPTIONS)
    assert answer["probabilities"]["c02"] == pytest.approx(0.6 / 0.9)
    assert answer["labels_reported"] == 2


def test_a_tie_between_labels_is_allowed() -> None:
    answer = parse_chat_choice(reply([("1", lp(0.4)), ("2", lp(0.4))]), OPTIONS)
    assert answer["choice"] == "c01"


def test_a_pool_of_thirty_reads_its_letters() -> None:
    options = tuple(f"c{i:02d}" for i in range(1, 31)) + (NONE_KEY,)
    answer = parse_chat_choice(reply([("W", lp(0.8)), ("V", lp(0.1)), ("0", lp(0.05))]), options)
    assert answer["choice"] == "c30" and answer["probabilities"]["c29"] == pytest.approx(0.1 / 0.95)


@pytest.mark.parametrize(("what", "text", "match"), [
    ("not json", "not json", "not JSON"),
    ("deeply nested", "[" * 100_000, "not JSON"),
    ("not an object", "[1]", "not a JSON object"),
    ("no model", reply(GOOD, model=""), "names no model"),
    ("two choices", json.dumps({"model": "m", "choices": [{}, {}], "usage": {}}), "exactly one choice"),
    ("filtered", reply(GOOD, finish="content_filter"), "did not finish normally"),
    ("no finish", reply(GOOD, finish=None), "did not finish normally"),
    ("null text", reply(GOOD, message={"content": None}), "no text"),
    ("reasoning text", reply(GOOD, message={"reasoning_content": "hmm"}), "reasoned"),
    ("reasoning text 2", reply(GOOD, message={"reasoning": "hmm"}), "reasoned"),
    ("no usage", reply(GOOD, usage={"x": 1}), "usage.prompt_tokens"),
    ("negative tokens", reply(GOOD, usage={"prompt_tokens": -1, "completion_tokens": 1}), "not a count"),
    ("boolean tokens", reply(GOOD, usage={"prompt_tokens": True, "completion_tokens": 1}), "not a count"),
    ("no logprobs", reply(GOOD, logprobs=None), "no log-probabilities"),
    ("empty logprobs", reply(GOOD, logprobs={"content": []}), "no log-probabilities"),
    ("no tops", reply(GOOD, logprobs={"content": [{"token": "2", "top_logprobs": []}]}), "no top"),
    ("positive logprob", reply([("2", 0.5)]), "above zero"),
    ("nan logprob", reply([("2", float("nan"))]), "not a finite"),
    ("string logprob", reply([("2", "x")]), "not a finite"),
    ("too little on labels", reply([("The", lp(0.6)), ("2", lp(0.3))], token="2"), "of the first token"),
    ("no labels", reply([("The", lp(0.9)), ("A", lp(0.05))]), "of the first token"),
    ("first token is chatter", reply([("2", lp(0.55)), ("The", lp(0.4))], token="The"),
     "not an option label"),
    ("own label is not the top label", reply([("1", lp(0.6)), ("2", lp(0.3))], token="2"), "most probable"),
])
def test_a_response_that_cannot_be_an_answer_is_refused(what: str, text: str, match: str | None) -> None:
    with pytest.raises(ChatAnswerError, match=match or ""):
        parse_chat_choice(text, OPTIONS)


def test_a_refused_answer_is_a_jev_error_so_the_run_stores_it_as_a_failure() -> None:
    assert issubclass(ChatAnswerError, JevError)


def test_the_offered_options_must_be_a_pool_plus_none() -> None:
    with pytest.raises(ChatAnswerError, match="options plus 'none'"):
        parse_chat_choice(reply(GOOD), ("c01", "c03", NONE_KEY))
    with pytest.raises(ChatAnswerError, match="options plus 'none'"):
        parse_chat_choice(reply(GOOD), ("c01", "c02", "c03"))


def test_exactly_the_minimum_label_mass_is_enough() -> None:
    answer = parse_chat_choice(reply([("2", lp(MIN_LABEL_MASS))]), OPTIONS)
    assert answer["choice"] == "c02" and answer["probabilities"]["c02"] == pytest.approx(1.0)
