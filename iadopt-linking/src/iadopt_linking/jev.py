"""Jev as a drop-in scorer for the linking baseline: the questions it is asked, the answers it may give.

Jev (TypeSafe's ``typesafe/jev-1.13``, reached through OpenRouter's Decisions API) is a typed
decision model, not a chat model: a request is a ``state`` plus typed ``questions``, and each answer
is a probability distribution over options the caller defined. The prompt follows TypeSafe's
published guidance: one narrow, literal question whose instruction names state fields in backticks;
neutral short option ids with structured ``{label, description}`` descriptions; an explicit
"none" option; nothing in the state that the question does not need. The same pool is also asked in
several orders, because TypeSafe documents that jev-1.13 leans toward the first option of a Choice.

The prompt says what an entity link is (the same concept as the term, not a broader, narrower or
related one, and not the whole variable) and what role the definition plays (to tell which meaning
of the term is meant). It was developed on invented slots only, in two drafts, before any corpus
slot was sent to Jev (DECISIONS L-011), and it is frozen: a changed prompt is a new experiment in a
new results folder.

Everything here is pure. Requests are built from recorded candidates, ordered by a seeded hash, and
answers are read strictly: a response that contradicts its request is refused, never repaired.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from iadopt_lab.canonical import content_hash

from .rerank import truncate_description

MODEL = "typesafe/jev-1.13"
PROMPT_VERSION = "jev-entity-link-v1"
ORDER_SEED = "iadopt-linking/jev-entity-link-v1/orders"
QUESTION_ID = "entity"
NONE_KEY = "none_of_these"
ARM_CHOICE = "choice"
ARM_NOUL = "noul"
# Jev reports probabilities rounded to two decimals, so their sum can drift from 1 by the sum of the
# rounding errors; this only has to catch an answer that is not a distribution at all.
SUM_TOLERANCE = 0.05
MAX_OPTIONS = 99
COST_QUANTUM = Decimal("0.000000000001")

CHOICE_INSTRUCTIONS = (
    "`term` is one component of the variable that `variable_definition` describes. Which candidate "
    "is the entry for the concept that `term` names? Use `variable_definition` only to tell which "
    "meaning of `term` is intended. A candidate matches only if it is the same concept as `term`: "
    "a broader, narrower or merely related concept does not match, and neither does a concept for "
    "the whole variable. Differences in capitalization, plural form or word order do not matter. "
    "Choose `none_of_these` if no candidate is the same concept.")
NONE_DESCRIPTION = "No candidate is the same concept as `term`."
NOUL_INSTRUCTIONS = (
    "`term` is one component of the variable that `variable_definition` describes. Is `candidate` "
    "the same concept as the one that `term` names? Use `variable_definition` only to tell which "
    "meaning of `term` is intended. A broader, narrower or merely related concept is not the same "
    "concept, and neither is a concept for the whole variable. Differences in capitalization, "
    "plural form or word order do not matter.")


@dataclass(frozen=True)
class PassSpec:
    """One way of asking the whole population once.

    ``order`` names how candidates are arranged in the request: ``shuffle-a`` and ``shuffle-b`` are
    two independent seeded orders and ``reverse-a`` is ``shuffle-a`` backwards. Two passes with the
    same arm and order send byte-identical requests, which measures the model's own variability.
    """

    name: str
    arm: str
    order: str


PASSES = (
    PassSpec("choice-a1", ARM_CHOICE, "shuffle-a"),
    PassSpec("choice-a1-repeat", ARM_CHOICE, "shuffle-a"),
    PassSpec("choice-a2-reversed", ARM_CHOICE, "reverse-a"),
    PassSpec("choice-a3", ARM_CHOICE, "shuffle-b"),
    PassSpec("noul-n1", ARM_NOUL, "shuffle-a"),
)
PRIMARY_PASS = "choice-a1"


class JevError(ValueError):
    """A response that cannot be a valid answer to the request it answers."""


def pass_spec(name: str) -> PassSpec:
    """Look a pass up by name.

    Args: name: a pass name from PASSES.
    Returns: its spec.
    Raises: KeyError for an unknown name.
    Side effects: none.
    """
    for spec in PASSES:
        if spec.name == name:
            return spec
    raise KeyError(f"No such pass: {name!r}")


def candidate_key(candidate: Mapping[str, Any]) -> str:
    """Name a candidate by its vocabulary and local identifier.

    Args: candidate: a pool entry.
    Returns: ``<vocabulary>:<local id>``.
    Raises: KeyError if the candidate lacks its identity fields.
    Side effects: none.
    """
    return f"{candidate['vocabulary']}:{candidate['local_id']}"


def option_id(position: int) -> str:
    """Name the option shown at a 1-based position; the ids also sort in presentation order.

    Args: position: 1-based place in the request.
    Returns: ``c01``, ``c02``, ...
    Raises: ValueError outside 1..MAX_OPTIONS.
    Side effects: none.
    """
    if not 1 <= position <= MAX_OPTIONS:
        raise ValueError(f"Option position {position} is outside 1..{MAX_OPTIONS}")
    return f"c{position:02d}"


def _hash_order(pool: Sequence[Mapping[str, Any]], slot_id: str, stream: str) -> list[int]:
    """Arrange a pool by a seeded hash of each candidate's identity.

    Args: pool: candidates in pool order; slot_id: the slot; stream: independent stream name.
    Returns: pool indices in presentation order; deterministic across Python versions and runs.
    Raises: KeyError if a candidate lacks its identity fields.
    Side effects: none.
    """
    keyed = [(hashlib.sha256(f"{ORDER_SEED}|{stream}|{slot_id}|{candidate_key(c)}".encode()).digest(),
              index) for index, c in enumerate(pool)]
    return [index for _, index in sorted(keyed)]


def presentation_order(pool: Sequence[Mapping[str, Any]], slot_id: str, order: str) -> list[int]:
    """Decide in which order a slot's candidates are shown.

    Args: pool: candidates in pool order; slot_id: the slot; order: shuffle-a, reverse-a or
    shuffle-b.
    Returns: pool indices in presentation order. The order depends on the candidates' identities
    and the slot, never on their scores or on whether one is the gold.
    Raises: ValueError for an unknown order; KeyError if a candidate lacks its identity fields.
    Side effects: none.
    """
    if order == "shuffle-a":
        return _hash_order(pool, slot_id, "a")
    if order == "reverse-a":
        return _hash_order(pool, slot_id, "a")[::-1]
    if order == "shuffle-b":
        return _hash_order(pool, slot_id, "b")
    raise ValueError(f"Unknown candidate order: {order!r}")


def _shown(candidate: Mapping[str, Any], max_description_chars: int | None) -> dict[str, str]:
    """Give the part of a candidate a linker may see: the label and the capped description.

    Args: candidate: a pool entry; max_description_chars: the baseline's description cap.
    Returns: {"label", "description"}, the same information the Qwen3 document carries.
    Raises: KeyError if the candidate lacks its label or description.
    Side effects: none.
    """
    return {"label": candidate["label"],
            "description": truncate_description(candidate["description"], max_description_chars)}


def _envelope(state: Mapping[str, Any], questions: Mapping[str, Any], session_id: str | None
              ) -> dict[str, Any]:
    """Wrap a state and questions into a Decisions request.

    Args: state: what is evaluated; questions: typed questions; session_id: OpenRouter log
    grouping label, or None.
    Returns: the request body. Fallback to other providers is off, so a request is served by the
    pinned model or fails.
    Raises: None.
    Side effects: none.
    """
    body: dict[str, Any] = {"model": MODEL, "state": dict(state), "questions": dict(questions),
                            "provider": {"allow_fallbacks": False}}
    if session_id:
        body["session_id"] = session_id
    return body


def choice_request(term: str, definition: str, pool: Sequence[Mapping[str, Any]],
                   order: Sequence[int], *, max_description_chars: int | None,
                   session_id: str | None = None) -> dict[str, Any]:
    """Build the Choice request: one question, one option per candidate, plus "none".

    Args: term: the slot's term as sent to the baseline; definition: the variable's definition;
    pool: candidates in pool order; order: pool indices in presentation order (a permutation);
    max_description_chars: description cap; session_id: log grouping label.
    Returns: the request body. Insertion order is part of the request: it is the order of the
    options, and the ids sort the same way, so a server that sorts keys changes nothing.
    Raises: ValueError if ``order`` is not a permutation of the pool or the pool is empty or too
    large; KeyError for a malformed candidate.
    Side effects: none.
    """
    _require_permutation(order, len(pool))
    options: dict[str, Any] = {
        option_id(place): _shown(pool[index], max_description_chars)
        for place, index in enumerate(order, start=1)}
    options[NONE_KEY] = NONE_DESCRIPTION
    question = {"type": "choice", "instructions": CHOICE_INSTRUCTIONS, "criteria": options}
    return _envelope({"term": term, "variable_definition": definition}, {QUESTION_ID: question},
                     session_id)


def noul_request(term: str, definition: str, pool: Sequence[Mapping[str, Any]],
                 order: Sequence[int], *, max_description_chars: int | None,
                 session_id: str | None = None) -> dict[str, Any]:
    """Build the pointwise request: one yes/no question per candidate, as in TypeSafe's re-ranking.

    Args: as choice_request.
    Returns: the request body. Each question carries only its own candidate, and Jev answers
    questions independently, so no candidate is judged against another.
    Raises: as choice_request.
    Side effects: none.
    """
    _require_permutation(order, len(pool))
    questions = {
        option_id(place): {"type": "noul",
                           "instructions": {"candidate": _shown(pool[index], max_description_chars),
                                            "question": NOUL_INSTRUCTIONS}}
        for place, index in enumerate(order, start=1)}
    return _envelope({"term": term, "variable_definition": definition}, questions, session_id)


def _require_permutation(order: Sequence[int], size: int) -> None:
    """Refuse an order that does not show every candidate exactly once.

    Args: order: pool indices; size: pool size.
    Returns: None.
    Raises: ValueError for an empty pool, an oversized pool or an order that is not a permutation.
    Side effects: none.
    """
    if size == 0:
        raise ValueError("There is nothing to ask about an empty pool")
    if size > MAX_OPTIONS - 1:
        raise ValueError(f"A pool of {size} candidates does not fit {MAX_OPTIONS - 1} options")
    if sorted(order) != list(range(size)):
        raise ValueError("The presentation order must show every candidate exactly once")


def encode(body: Mapping[str, Any]) -> bytes:
    """Serialize a request exactly as it is sent.

    Args: body: a request body.
    Returns: compact UTF-8 JSON in insertion order (not sorted: key order is the option order).
    Raises: TypeError or ValueError for a value JSON cannot hold.
    Side effects: none.
    """
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def request_sha256(body: Mapping[str, Any]) -> str:
    """Identify a request by the bytes that are sent.

    Args: body: a request body.
    Returns: hexadecimal SHA-256 of ``encode(body)``.
    Raises: as encode.
    Side effects: none.
    """
    return hashlib.sha256(encode(body)).hexdigest()


def prompt_fingerprint() -> str:
    """Identify the whole prompt design in one hash.

    Args: none.
    Returns: a hash over the model pin, the instructions, the order seed and the passes.
    Raises: None.
    Side effects: none.
    """
    return content_hash({
        "version": PROMPT_VERSION, "model": MODEL, "question_id": QUESTION_ID,
        "none_key": NONE_KEY, "choice_instructions": CHOICE_INSTRUCTIONS,
        "none_description": NONE_DESCRIPTION, "noul_instructions": NOUL_INSTRUCTIONS,
        "order_seed": ORDER_SEED,
        "passes": [[p.name, p.arm, p.order] for p in PASSES]})


def _number(value: Any, what: str, low: float | None = None, high: float | None = None) -> float:
    """Read a finite JSON number, optionally within bounds.

    Args: value: the parsed JSON value; what: its name for messages; low and high: bounds.
    Returns: the number as a float.
    Raises: JevError for a non-number, a boolean, a non-finite value or one out of bounds.
    Side effects: none.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise JevError(f"{what} is not a finite number: {value!r}")
    if (low is not None and value < low) or (high is not None and value > high):
        raise JevError(f"{what} is outside [{low}, {high}]: {value!r}")
    return float(value)


def _envelope_fields(text: str, expected_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read what every Decisions response carries.

    Args: text: the response body; expected_ids: the question ids that must be answered.
    Returns: the answers by id, and the envelope: model, id, provider, input and output tokens, and
    the cost in USD as a fixed-point string (None when absent).
    Raises: JevError for text that is not a JSON object, a missing or empty model, answers that
    are not exactly the expected questions, or malformed usage.
    Side effects: none.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JevError(f"response is not JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise JevError("response is not a JSON object")
    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise JevError("response names no model")
    answers = data.get("answers")
    if not isinstance(answers, dict) or sorted(answers) != sorted(expected_ids):
        raise JevError("response does not answer exactly the questions that were asked")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        raise JevError("response has no usage")
    tokens = {}
    for key in ("input_tokens", "output_tokens"):
        count = usage.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise JevError(f"usage.{key} is not a count: {count!r}")
        tokens[key] = count
    cost = usage.get("cost")
    # Rounded to a picodollar so that float noise in a reported price never reaches an exact sum.
    cost_text = None if cost is None else format(
        Decimal(repr(_number(cost, "usage.cost", low=0.0))).quantize(COST_QUANTUM).normalize(), "f")
    return answers, {"model": model, "id": data.get("id"), "provider": data.get("provider"),
                     **tokens, "cost_usd": cost_text}


def parse_choice(text: str, option_ids: Sequence[str]) -> dict[str, Any]:
    """Read the answer to a Choice request strictly.

    Args: text: the response body; option_ids: every option id that was offered, "none" included.
    Returns: the envelope fields plus ``choice`` (an offered id), ``confidence`` and
    ``probabilities`` by option id.
    Raises: JevError unless the answer is a choice among exactly the offered options, with finite
    probabilities in [0, 1] that sum to 1 within SUM_TOLERANCE, a confidence in [0, 1], and a
    ``choice`` that has the highest probability (ties allowed).
    Side effects: none.
    """
    answers, envelope = _envelope_fields(text, [QUESTION_ID])
    answer = answers[QUESTION_ID]
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise JevError("the answer is not a choice")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or sorted(probabilities) != sorted(option_ids):
        raise JevError("the probabilities do not cover exactly the offered options")
    values = {key: _number(p, f"probability of {key}", 0.0, 1.0) for key, p in probabilities.items()}
    drift = abs(math.fsum(values.values()) - 1.0)
    if drift > SUM_TOLERANCE:
        raise JevError(f"the probabilities sum to {1.0 + drift:.3f}, not 1")
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in values:
        raise JevError(f"the choice {choice!r} was not offered")
    if values[choice] < max(values.values()):
        raise JevError("the choice is not the option with the highest probability")
    confidence = _number(answer.get("confidence"), "confidence", 0.0, 1.0)
    return {**envelope, "choice": choice, "confidence": confidence, "probabilities": values,
            "probability_sum_drift": drift}


def parse_noul(text: str, question_ids: Sequence[str]) -> dict[str, Any]:
    """Read the answers to a request of yes/no questions strictly.

    Args: text: the response body; question_ids: every question id that was asked.
    Returns: the envelope fields plus ``nouls`` by question id, each the probability of yes.
    Raises: JevError unless every question has a noul answer with a probability in [0, 1].
    Side effects: none.
    """
    answers, envelope = _envelope_fields(text, question_ids)
    nouls = {}
    for key, answer in answers.items():
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise JevError(f"the answer to {key} is not a noul")
        nouls[key] = _number(answer.get("noul"), f"noul of {key}", 0.0, 1.0)
    return {**envelope, "nouls": nouls}


def choice_scores(parsed: Mapping[str, Any], order: Sequence[int], size: int
                  ) -> tuple[list[float], float, int | None]:
    """Turn a parsed Choice answer into one score per pool candidate.

    Args: parsed: parse_choice output; order: the presentation order that was sent; size: pool
    size.
    Returns: the probability of each candidate in pool order, the probability of "none", and the
    pool index Jev itself chose (None when it chose "none").
    Raises: KeyError if an option is missing.
    Side effects: none.
    """
    scores = [0.0] * size
    for place, index in enumerate(order, start=1):
        scores[index] = parsed["probabilities"][option_id(place)]
    chosen = parsed["choice"]
    jev_index = None if chosen == NONE_KEY else order[int(chosen[1:]) - 1]
    return scores, parsed["probabilities"][NONE_KEY], jev_index


def noul_scores(parsed: Mapping[str, Any], order: Sequence[int], size: int) -> list[float]:
    """Turn a parsed set of yes/no answers into one score per pool candidate.

    Args: parsed: parse_noul output; order: the presentation order that was sent; size: pool size.
    Returns: the probability of yes for each candidate in pool order.
    Raises: KeyError if a question is missing.
    Side effects: none.
    """
    scores = [0.0] * size
    for place, index in enumerate(order, start=1):
        scores[index] = parsed["nouls"][option_id(place)]
    return scores


def scored_record(base: Mapping[str, Any], scores: Sequence[float]) -> dict[str, Any]:
    """Put another scorer's scores where the baseline's were, so the same judge can evaluate them.

    Args: base: a baseline scores record; scores: one score per pool candidate, in pool order.
    Returns: the record with each candidate's ``score`` replaced and its Qwen3 ``document`` removed.
    Raises: ValueError if the number of scores differs from the pool size.
    Side effects: none.
    """
    pool = base["pool"]
    if len(scores) != len(pool):
        raise ValueError(f"{len(scores)} scores for a pool of {len(pool)}")
    return {**{k: v for k, v in base.items() if k not in ("pool", "scoring")},
            "pool": [{**{k: v for k, v in c.items() if k != "document"}, "score": float(s)}
                     for c, s in zip(pool, scores, strict=True)]}
