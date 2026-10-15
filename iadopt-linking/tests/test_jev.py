"""The Jev prompt, its candidate orders and its strict reading of answers, with no network."""

from __future__ import annotations

import json
from typing import Any

import pytest

from iadopt_linking import jev
from iadopt_linking.jev import (
    JevError,
    choice_request,
    choice_scores,
    encode,
    noul_request,
    noul_scores,
    parse_choice,
    parse_noul,
    pass_spec,
    presentation_order,
    prompt_fingerprint,
    request_sha256,
    scored_record,
)

SLOT = "x.ttl#hasProperty#0"
LONG = "word " * 100


def candidate(vocabulary: str, local_id: str, label: str = "", description: str = "",
              rank: int = 1) -> dict[str, Any]:
    return {"source": vocabulary, "rank": rank, "iri": f"http://example.test/{local_id}",
            "vocabulary": vocabulary, "local_id": local_id, "label": label or local_id.lower(),
            "description": description, "extras": {}, "document": "baseline document",
            "score": 0.123456}


POOL = [candidate("wikidata", "Q1"), candidate("wikidata", "Q2"), candidate("wikidata", "Q3"),
        candidate("qudt", "Mass", description="physical quantity"), candidate("qudt", "Area"),
        candidate("om", "Mass")]


def answer(choice: str, probabilities: dict[str, float], confidence: float = 0.5,
           **envelope: Any) -> str:
    body = {"id": "gen-dec-1", "model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe",
            "answers": {"entity": {"type": "choice", "choice": choice, "confidence": confidence,
                                   "probabilities": probabilities}},
            "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992}, **envelope}
    return json.dumps(body)


OFFERED = ["c01", "c02", "c03", "none_of_these"]
GOOD = {"c01": 0.2, "c02": 0.7, "c03": 0.08, "none_of_these": 0.02}


def test_candidate_orders_are_frozen() -> None:
    # A change here changes what Jev is shown for every slot: it needs a new results folder.
    assert presentation_order(POOL, SLOT, "shuffle-a") == [3, 1, 4, 2, 0, 5]
    assert presentation_order(POOL, SLOT, "reverse-a") == [5, 0, 2, 4, 1, 3]
    assert presentation_order(POOL, SLOT, "shuffle-b") == [5, 3, 0, 2, 4, 1]


def test_orders_are_permutations_independent_of_scores_and_pool_order() -> None:
    first = presentation_order(POOL, SLOT, "shuffle-a")
    assert sorted(first) == list(range(len(POOL)))
    rescored = [{**c, "score": 1 - c["score"], "rank": 9} for c in POOL]
    assert presentation_order(rescored, SLOT, "shuffle-a") == first
    shuffled_pool = POOL[::-1]
    keys = [POOL[i]["local_id"] for i in first]
    assert [shuffled_pool[i]["local_id"]
            for i in presentation_order(shuffled_pool, SLOT, "shuffle-a")] == keys
    assert presentation_order(POOL, SLOT, "shuffle-a") != presentation_order(POOL, SLOT, "shuffle-b")


def test_an_unknown_order_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        presentation_order(POOL, SLOT, "alphabetical")


def test_the_choice_request_shows_each_candidate_once_then_none() -> None:
    order = presentation_order(POOL, SLOT, "shuffle-a")
    body = choice_request("mass", "A definition.", POOL, order, max_description_chars=300,
                          session_id="run-1")
    question = body["questions"]["entity"]
    assert body["model"] == "typesafe/jev-1.13" and body["session_id"] == "run-1"
    assert body["provider"] == {"allow_fallbacks": False}
    assert body["state"] == {"term": "mass", "variable_definition": "A definition."}
    assert question["type"] == "choice" and question["instructions"] == jev.CHOICE_INSTRUCTIONS
    options = list(question["criteria"])
    assert options == [f"c{i:02d}" for i in range(1, 7)] + ["none_of_these"]
    assert options == sorted(options)               # a server that sorts keys changes nothing
    assert [question["criteria"][o]["label"] for o in options[:-1]] == [
        POOL[i]["label"] for i in order]
    assert question["criteria"]["none_of_these"] == jev.NONE_DESCRIPTION


def test_the_request_carries_only_label_and_capped_description() -> None:
    long = candidate("wikidata", "Q9", label="Long one", description=LONG)
    body = choice_request("t", "d", [long], [0], max_description_chars=300)
    shown = body["questions"]["entity"]["criteria"]["c01"]
    assert set(shown) == {"label", "description"}
    assert len(shown["description"]) <= 301 and shown["description"].endswith("…")
    text = encode(body).decode()
    # nothing that identifies a candidate or reveals a score or the gold may be sent
    for secret in ("Q9", "example.test", "0.123456", "wikidata", "baseline document", "score"):
        assert secret not in text


def test_an_empty_description_stays_an_empty_string() -> None:
    body = choice_request("t", "d", [POOL[0]], [0], max_description_chars=300)
    assert body["questions"]["entity"]["criteria"]["c01"] == {"label": "q1", "description": ""}


def test_the_noul_request_asks_one_independent_question_per_candidate() -> None:
    order = presentation_order(POOL, SLOT, "shuffle-a")
    body = noul_request("mass", "A definition.", POOL, order, max_description_chars=300)
    questions = body["questions"]
    assert list(questions) == [f"c{i:02d}" for i in range(1, 7)]
    first = questions["c01"]
    assert first["type"] == "noul" and "criteria" not in first
    assert first["instructions"]["question"] == jev.NOUL_INSTRUCTIONS
    assert first["instructions"]["candidate"]["label"] == POOL[order[0]]["label"]
    assert "none_of_these" not in encode(body).decode()


def test_the_serialisation_keeps_insertion_order_and_identifies_the_bytes_sent() -> None:
    body = choice_request("t", "d", POOL[:2], [1, 0], max_description_chars=300)
    text = encode(body).decode()
    assert text.index('"label"') < text.index('"description"')
    assert text.index('"c01"') < text.index('"c02"') < text.index('"none_of_these"')
    swapped = choice_request("t", "d", POOL[:2], [0, 1], max_description_chars=300)
    assert request_sha256(body) != request_sha256(swapped)
    assert request_sha256(body) == request_sha256(json.loads(text))


@pytest.mark.parametrize("order", [[0, 0], [0, 2], [1], []])
def test_an_order_that_is_not_a_permutation_is_refused(order: list[int]) -> None:
    with pytest.raises(ValueError):
        choice_request("t", "d", POOL[:2], order, max_description_chars=300)
    with pytest.raises(ValueError):
        noul_request("t", "d", POOL[:2], order, max_description_chars=300)


def test_an_empty_pool_has_nothing_to_ask() -> None:
    with pytest.raises(ValueError, match="empty pool"):
        choice_request("t", "d", [], [], max_description_chars=300)


def test_the_prompt_of_the_committed_experiment_is_frozen() -> None:
    # Changing any instruction, the seed or the passes is a new experiment: give it a new results
    # folder and a new decision, then update this value.
    assert prompt_fingerprint() == (
        "a6042e8c2d203a7716bf5241008d4efa5ac33b44c28309fbcb3dca461ec565c5")


def test_the_fingerprint_follows_every_part_of_the_design(monkeypatch: pytest.MonkeyPatch) -> None:
    base = prompt_fingerprint()
    assert base == prompt_fingerprint() and len(base) == 64
    monkeypatch.setattr(jev, "CHOICE_INSTRUCTIONS", jev.CHOICE_INSTRUCTIONS + " ")
    assert prompt_fingerprint() != base
    monkeypatch.undo()
    monkeypatch.setattr(jev, "ORDER_SEED", "another")
    assert prompt_fingerprint() != base


def test_the_passes_are_named_and_the_repeat_sends_the_same_request() -> None:
    assert pass_spec(jev.PRIMARY_PASS).arm == "choice"
    assert pass_spec("choice-a1-repeat").order == pass_spec("choice-a1").order
    with pytest.raises(KeyError):
        pass_spec("nope")


def test_a_valid_choice_answer_is_read_in_full() -> None:
    parsed = parse_choice(answer("c02", GOOD, 0.67), OFFERED)
    assert parsed["choice"] == "c02" and parsed["confidence"] == 0.67
    assert parsed["probabilities"] == GOOD
    assert parsed["model"] == "typesafe/jev-1.13-20260917" and parsed["provider"] == "TypeSafe"
    assert (parsed["input_tokens"], parsed["output_tokens"]) == (476, 70)
    assert parsed["cost_usd"] == "0.000019992"       # exact, not 1.9992e-05
    assert parsed["probability_sum_drift"] == pytest.approx(0.0)


def test_a_tie_for_the_top_probability_is_accepted_either_way() -> None:
    tie = {"c01": 0.4, "c02": 0.4, "c03": 0.1, "none_of_these": 0.1}
    assert parse_choice(answer("c01", tie), OFFERED)["choice"] == "c01"
    assert parse_choice(answer("c02", tie), OFFERED)["choice"] == "c02"


def test_cost_is_optional() -> None:
    body = json.loads(answer("c02", GOOD))
    del body["usage"]["cost"]
    assert parse_choice(json.dumps(body), OFFERED)["cost_usd"] is None


def broken(change) -> str:
    body = json.loads(answer("c02", GOOD))
    change(body)
    return json.dumps(body)


@pytest.mark.parametrize(("text", "why"), [
    ("not json", "not JSON"),
    ("[1]", "not a JSON object"),
    (broken(lambda b: b.pop("model")), "no model"),
    (broken(lambda b: b.update(answers={})), "exactly the questions"),
    (broken(lambda b: b["answers"].update(extra={})), "exactly the questions"),
    (broken(lambda b: b["answers"]["entity"].update(type="noul")), "not a choice"),
    (broken(lambda b: b["answers"]["entity"]["probabilities"].pop("c03")), "exactly the offered"),
    (broken(lambda b: b["answers"]["entity"]["probabilities"].update(c01=1.5)), "outside"),
    (broken(lambda b: b["answers"]["entity"]["probabilities"].update(c01=0.7)), "sum to"),
    (broken(lambda b: b["answers"]["entity"].update(choice="c09")), "not offered"),
    (broken(lambda b: b["answers"]["entity"].update(choice=["c02"])), "not offered"),
    (broken(lambda b: b["answers"]["entity"].update(choice="c01")), "highest probability"),
    (broken(lambda b: b["answers"]["entity"].pop("confidence")), "confidence"),
    (broken(lambda b: b["answers"]["entity"].update(confidence=1.2)), "outside"),
    (broken(lambda b: b.pop("usage")), "no usage"),
    (broken(lambda b: b["usage"].update(input_tokens=True)), "not a count"),
    (broken(lambda b: b["usage"].update(cost=-1)), "outside"),
])
def test_a_response_that_contradicts_its_request_is_refused(text: str, why: str) -> None:
    with pytest.raises(JevError, match=why):
        parse_choice(text, OFFERED)


def test_a_non_finite_number_is_refused() -> None:
    text = answer("c02", GOOD).replace("0.7", "NaN")
    with pytest.raises(JevError, match="finite"):
        parse_choice(text, OFFERED)


def noul_answer(values: dict[str, float]) -> str:
    return json.dumps({
        "model": "typesafe/jev-1.13-20260917",
        "answers": {k: {"type": "noul", "noul": v} for k, v in values.items()},
        "usage": {"input_tokens": 10, "output_tokens": 2}})


def test_noul_answers_are_read_per_question() -> None:
    parsed = parse_noul(noul_answer({"c01": 0.9, "c02": 0.0}), ["c01", "c02"])
    assert parsed["nouls"] == {"c01": 0.9, "c02": 0.0} and parsed["cost_usd"] is None


@pytest.mark.parametrize("text", [
    noul_answer({"c01": 0.9}),                                  # a question went unanswered
    noul_answer({"c01": 0.9, "c02": 1.5}),                      # not a probability
    json.dumps({"model": "m", "usage": {"input_tokens": 1, "output_tokens": 1},
                "answers": {"c01": {"type": "choice"}, "c02": {"type": "noul", "noul": 0.5}}}),
])
def test_bad_noul_answers_are_refused(text: str) -> None:
    with pytest.raises(JevError):
        parse_noul(text, ["c01", "c02"])


def test_option_probabilities_are_mapped_back_to_the_pool() -> None:
    order = [3, 1, 4, 2, 0, 5]
    ids = [jev.option_id(p) for p in range(1, 7)] + ["none_of_these"]
    probabilities = {i: 0.0 for i in ids} | {"c02": 0.6, "c05": 0.3, "none_of_these": 0.1}
    parsed = parse_choice(answer("c02", probabilities), ids)
    scores, p_none, chosen = choice_scores(parsed, order, 6)
    assert scores[1] == 0.6 and scores[0] == 0.3 and p_none == 0.1   # c02 -> pool 1, c05 -> pool 0
    assert chosen == 1
    parsed = parse_choice(answer("none_of_these", {**probabilities, "c02": 0.1,
                                                   "none_of_these": 0.6}), ids)
    assert choice_scores(parsed, order, 6)[2] is None


def test_noul_scores_are_mapped_back_to_the_pool() -> None:
    parsed = parse_noul(noul_answer({"c01": 0.1, "c02": 0.8}), ["c01", "c02"])
    assert noul_scores(parsed, [1, 0], 2) == [0.8, 0.1]


def test_scored_record_swaps_in_the_new_scores_and_drops_the_old_provenance() -> None:
    base = {"slot_id": "s", "gold": {"vocabulary": "qudt", "local_id": "Mass"},
            "scoring": {"model": "qwen"}, "pool": POOL[:2]}
    record = scored_record(base, [0.25, 0.75])
    assert [c["score"] for c in record["pool"]] == [0.25, 0.75]
    assert "scoring" not in record and all("document" not in c for c in record["pool"])
    assert base["pool"][0]["score"] == 0.123456 and "document" in base["pool"][0]   # not mutated
    with pytest.raises(ValueError, match="scores for a pool"):
        scored_record(base, [0.5])
