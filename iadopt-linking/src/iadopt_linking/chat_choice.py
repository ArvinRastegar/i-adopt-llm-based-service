"""A chat model as a listwise entity linker: the prompt, the option labels, and how answers are read.

The question is the one Jev is asked (DECISIONS L-011): which candidate is the entry for the concept
that ``term`` names, with a "none of these" option. A chat model answers in text, so each option is
named by one character that is one token (``1``-``9`` and the capital letters without ``I`` and
``O``; ``0`` is "none of these"), the model is asked to reply with that label alone, and the
probability of every option is read from the log-probabilities the gateway returns for that first
token. That gives every choice a distribution over the options like Jev's, which is what the later
confidence analysis needs: nothing here is a verbalised confidence.

Reasoning is off (the measured control for the model is in ``REASONING_OFF``), the temperature is 0
and one token is generated. The prompt was developed on invented slots only and is frozen
(DECISIONS L-016); a changed prompt is a new experiment in a new results folder.

Answers are read strictly, never repaired: a response that names no model, shows reasoning text,
carries no log-probabilities, puts too little of its mass on the labels, or whose own first token is
not the most probable label is refused. Probabilities are normalised over the option labels found in
the top log-probabilities; labels the gateway did not report get 0, so every reported probability is
a lower bound (``top_mass`` and ``label_mass`` are recorded to say by how much).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from iadopt_lab.canonical import content_hash

from .jev import (
    ARM_CHOICE,
    NONE_KEY,
    ORDER_SEED,
    PASSES,
    JevError,
    PassSpec,
    candidate_key,
    encode,
    option_id,
    presentation_order,
    request_sha256,
)
from .jev_inputs import Baseline, Task
from .population import normalise_term
from .rerank import truncate_description

MODEL = "Qwen3.8-27B"
PROVIDER = "psnc"
PROMPT_VERSION = "psnc-chat-entity-link-v1"
# One character, one token, no look-alikes (no I or O): 33 options, and a pool holds at most 30.
LABELS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
NONE_LABEL = "0"
NONE_DESCRIPTION = "no candidate is the same concept as the term"
TOP_LOGPROBS = 20            # the most the gateway allows
MIN_LABEL_MASS = 0.5         # below this the first token was not an answer
MAX_TOKENS = 1
TEMPERATURE = 0
ASSUMED_CHARS_PER_TOKEN = 1.5
# Measured by `iadopt-lab probe-models` and recorded in the lab's parameters.yml: this switch removes
# the model's reasoning on PSNC. A model that is not listed has no measured control and is refused.
REASONING_OFF: dict[str, dict[str, Any]] = {
    "Qwen3.8-27B": {"chat_template_kwargs": {"enable_thinking": False}}}

SYSTEM_PROMPT = (
    "You link a term from a scientific variable name to the knowledge-base entry that names the "
    "same concept. You are given a `term`, the `variable_definition` of the variable the term is a "
    "component of, and a numbered list of candidate entries. Pick the candidate that is the entry "
    "for the concept that `term` names. Use `variable_definition` only to tell which meaning of "
    "`term` is intended. A candidate matches only if it is the same concept as `term`: a broader, "
    "narrower or merely related concept does not match, and neither does a concept for the whole "
    "variable. Differences in capitalization, plural form or word order do not matter. If no "
    "candidate is the same concept, pick option 0. Reply with the label of exactly one option (a "
    "digit or a capital letter) and nothing else.")
ANSWER_LINE = "Answer with one label."


class ChatAnswerError(JevError):
    """A chat response that cannot be a valid answer to the request it answers."""


def option_label(position: int) -> str:
    """Name the option shown at a 1-based position with its one-token label.

    Args: position: 1-based place in the request.
    Returns: ``1``..``9``, then ``A``, ``B``, ... (``I`` and ``O`` are skipped).
    Raises: ValueError outside 1..len(LABELS).
    Side effects: none.
    """
    if not 1 <= position <= len(LABELS):
        raise ValueError(f"Option position {position} is outside 1..{len(LABELS)}")
    return LABELS[position - 1]


def shown_candidate(candidate: Mapping[str, Any], max_description_chars: int | None) -> str:
    """Write the part of a candidate the model may see: the label and the capped description.

    Args: candidate: a pool entry; max_description_chars: the baseline's description cap.
    Returns: ``"label": description``, or ``"label"`` for a candidate with no description. This is
    the information the Qwen3 reranker's document carries; no vocabulary, identifier or rank.
    Raises: KeyError if the candidate lacks its label or description.
    Side effects: none.
    """
    description = truncate_description(candidate["description"], max_description_chars)
    return f'"{candidate["label"]}"' + (f": {description}" if description else "")


def user_message(term: str, definition: str, pool: Sequence[Mapping[str, Any]],
                 order: Sequence[int], max_description_chars: int | None) -> str:
    """Write the case the model is asked about.

    Args: term: the slot's term as sent to the baseline; definition: the variable's definition;
    pool: candidates in pool order; order: pool indices in presentation order; max_description_chars:
    description cap.
    Returns: the term, the definition, the labelled candidates in presentation order, the "none"
    option last, and the answer line.
    Raises: KeyError for a malformed candidate.
    Side effects: none.
    """
    lines = [f"{option_label(place)}. {shown_candidate(pool[index], max_description_chars)}"
             for place, index in enumerate(order, start=1)]
    lines.append(f"{NONE_LABEL}. none of these: {NONE_DESCRIPTION}")
    return (f"term: {term}\nvariable_definition: {definition}\n\ncandidates:\n" + "\n".join(lines)
            + f"\n\n{ANSWER_LINE}")


def _require_permutation(order: Sequence[int], size: int) -> None:
    """Refuse an order that does not show every candidate exactly once.

    Args: order: pool indices; size: pool size.
    Returns: None.
    Raises: ValueError for an empty pool, an oversized pool or an order that is not a permutation.
    Side effects: none.
    """
    if size == 0:
        raise ValueError("There is nothing to ask about an empty pool")
    if size > len(LABELS):
        raise ValueError(f"A pool of {size} candidates does not fit {len(LABELS)} option labels")
    if sorted(order) != list(range(size)):
        raise ValueError("The presentation order must show every candidate exactly once")


def chat_request(term: str, definition: str, pool: Sequence[Mapping[str, Any]],
                 order: Sequence[int], *, max_description_chars: int | None,
                 model: str = MODEL) -> dict[str, Any]:
    """Build the chat-completions request for one slot in one candidate order.

    Args: term, definition, pool, order, max_description_chars: as user_message; model: the PSNC
    model id.
    Returns: the request body. Insertion order is part of the request and of its hash.
    Raises: ValueError if ``order`` is not a permutation of the pool, the pool is empty or too
    large, or the model has no measured way to turn reasoning off; KeyError for a malformed
    candidate.
    Side effects: none.
    """
    _require_permutation(order, len(pool))
    if model not in REASONING_OFF:
        raise ValueError(f"{model!r} has no measured control for turning reasoning off; probe it "
                         f"first (known: {sorted(REASONING_OFF)})")
    return {"model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_message(
                             term, definition, pool, order, max_description_chars)}],
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "logprobs": True,
            "top_logprobs": TOP_LOGPROBS, "stream": False, **REASONING_OFF[model]}


CHAT_PASSES = tuple(p for p in PASSES if p.arm == ARM_CHOICE)


def prompt_fingerprint(model: str = MODEL) -> str:
    """Identify the whole prompt design in one hash.

    Args: model: the PSNC model id.
    Returns: a hash over the instructions, the labels, the request parameters, the reasoning
    control, the order seed and the passes.
    Raises: KeyError for a model with no reasoning control.
    Side effects: none.
    """
    return content_hash({
        "version": PROMPT_VERSION, "model": model, "system_prompt": SYSTEM_PROMPT,
        "answer_line": ANSWER_LINE, "none_description": NONE_DESCRIPTION, "labels": LABELS,
        "none_label": NONE_LABEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "top_logprobs": TOP_LOGPROBS, "min_label_mass": MIN_LABEL_MASS,
        "reasoning_off": REASONING_OFF[model], "order_seed": ORDER_SEED,
        "passes": [[p.name, p.arm, p.order] for p in CHAT_PASSES]})


def prompt_plan(max_description_chars: int | None, model: str = MODEL) -> dict[str, Any]:
    """Describe the prompt design for the plan.

    Args: max_description_chars: the baseline's description cap; model: the PSNC model id.
    Returns: version, fingerprint, the texts the model is given, the labels, the request
    parameters and the validity thresholds.
    Raises: KeyError for a model with no reasoning control.
    Side effects: none.
    """
    return {"version": PROMPT_VERSION, "fingerprint": prompt_fingerprint(model),
            "system_prompt": SYSTEM_PROMPT, "answer_line": ANSWER_LINE,
            "none_description": NONE_DESCRIPTION, "labels": LABELS, "none_label": NONE_LABEL,
            "request_parameters": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                                   "logprobs": True, "top_logprobs": TOP_LOGPROBS,
                                   "stream": False, **REASONING_OFF[model]},
            "min_label_mass": MIN_LABEL_MASS, "order_seed": ORDER_SEED,
            "max_description_chars": max_description_chars,
            "shown_per_candidate": ["label", "description"]}


def build_tasks(baseline: Baseline, specs: Sequence[PassSpec], *, model: str = MODEL,
                session_id: str | None = None) -> tuple[list[Task], dict[str, int]]:
    """Build every request of a run.

    Args: baseline: the verified baseline; specs: the passes (Choice passes only); model: the PSNC
    model id; session_id: unused (kept so the signature matches the Jev builder).
    Returns: the tasks, pass by pass in population order, and counts: slots and empty pools. A
    slot with no candidates is not asked about: there is nothing to choose.
    Raises: ValueError for a pass that is not a Choice pass, a pool too large for the labels, or a
    model with no reasoning control.
    Side effects: none; deterministic.
    """
    tasks: list[Task] = []
    empty = sum(not baseline.records[slot.slot_id]["pool"] for slot in baseline.slots)
    for spec in specs:
        if spec.arm != ARM_CHOICE:
            raise ValueError(f"pass {spec.name!r} is not a Choice pass; a chat model is asked "
                             "listwise questions only")
        for slot in baseline.slots:
            pool = baseline.records[slot.slot_id]["pool"]
            if not pool:
                continue
            order = presentation_order(pool, slot.slot_id, spec.order)
            body = chat_request(normalise_term(slot.term), slot.definition, pool, order,
                                max_description_chars=baseline.max_description_chars, model=model)
            offered = tuple(option_id(place) for place in range(1, len(pool) + 1)) + (NONE_KEY,)
            shown = tuple(candidate_key(pool[index]) for index in order)
            tasks.append(Task(slot, spec, tuple(order), shown, offered, body,
                              request_sha256(body), len(encode(body))))
    return tasks, {"slots": len(baseline.slots), "empty_pools": empty}


def estimate(tasks: Sequence[Task]) -> dict[str, Any]:
    """Estimate a run's size before any call, erring high.

    Args: tasks: the requests.
    Returns: requests, request characters, the assumed characters per token and the token
    estimate. The service is not billed to the owner, so the estimated cost is 0.
    Raises: None.
    Side effects: none.
    """
    chars = sum(task.chars for task in tasks)
    by_pass = Counter(task.spec.name for task in tasks)
    return {"requests": len(tasks), "requests_by_pass": dict(sorted(by_pass.items())),
            "request_chars": chars, "assumed_chars_per_token": ASSUMED_CHARS_PER_TOKEN,
            "input_tokens_upper_estimate": int(chars / ASSUMED_CHARS_PER_TOKEN) + 1,
            "price_per_input_token_usd": "0", "estimated_cost_usd": "0"}


def label_to_option(count: int) -> dict[str, str]:
    """Map each label to the option id of the position it names.

    Args: count: the number of candidates shown.
    Returns: ``{"1": "c01", ...}`` plus the "none" label.
    Raises: ValueError if count is outside 1..len(LABELS).
    Side effects: none.
    """
    mapping = {option_label(place): option_id(place) for place in range(1, count + 1)}
    mapping[NONE_LABEL] = NONE_KEY
    return mapping


def _logprob(entry: Any, what: str) -> float:
    """Read one log-probability.

    Args: entry: the parsed JSON value; what: its name for messages.
    Returns: the value as a float.
    Raises: ChatAnswerError unless it is a finite number that is not positive (a probability above
    1 is impossible; rounding may put the top token a hair above 0).
    Side effects: none.
    """
    if isinstance(entry, bool) or not isinstance(entry, (int, float)) or not math.isfinite(entry):
        raise ChatAnswerError(f"{what} is not a finite number: {entry!r}")
    if entry > 1e-6:
        raise ChatAnswerError(f"{what} is above zero: {entry!r}")
    return float(min(entry, 0.0))


def _tokens(value: Any, what: str) -> int:
    """Read a token count.

    Args: value: the parsed JSON value; what: its name for messages.
    Returns: the count.
    Raises: ChatAnswerError unless it is a non-negative integer.
    Side effects: none.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChatAnswerError(f"{what} is not a count: {value!r}")
    return value


def parse_chat_choice(text: str, option_ids: Sequence[str]) -> dict[str, Any]:
    """Read the answer to a chat Choice request strictly.

    Args: text: the response body; option_ids: every option id offered, "none" included, as
    ``Task.offered`` lists them (``c01``..``cNN`` then ``none_of_these``).
    Returns: the envelope (model, id, provider, system fingerprint, input and output tokens, no
    cost) plus ``choice`` (the option id the model's own first token names), ``confidence`` (its
    normalised probability), ``probabilities`` by option id (normalised over the labels found),
    ``probability_sum_drift``, ``label_mass`` (the share of the first token's probability that sits
    on option labels), ``top_mass`` (the share on every reported token) and ``labels_reported``.
    Raises: ChatAnswerError unless the response is a single completed choice with no reasoning
    text, token counts, log-probabilities for the first token, at least MIN_LABEL_MASS of its mass
    on option labels, a first token that is a label, and that label the most probable of the labels
    found (ties allowed).
    Side effects: none.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ChatAnswerError(f"response is not JSON ({type(exc).__name__})") from exc
    if not isinstance(data, dict):
        raise ChatAnswerError("response is not a JSON object")
    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise ChatAnswerError("response names no model")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ChatAnswerError("response does not hold exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") not in ("stop", "length"):
        raise ChatAnswerError(f"the generation did not finish normally: "
                              f"{choice.get('finish_reason')!r}")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ChatAnswerError("the message has no text")
    for field in ("reasoning_content", "reasoning"):
        if message.get(field):
            raise ChatAnswerError("the model reasoned although reasoning is switched off")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        raise ChatAnswerError("response has no usage")
    tokens = {"input_tokens": _tokens(usage.get("prompt_tokens"), "usage.prompt_tokens"),
              "output_tokens": _tokens(usage.get("completion_tokens"), "usage.completion_tokens")}
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        raise ChatAnswerError("response carries no log-probabilities for the first token")
    first = content[0]
    generated = first.get("token")
    top = first.get("top_logprobs")
    if not isinstance(generated, str) or not isinstance(top, list) or not top:
        raise ChatAnswerError("the first token has no top log-probabilities")
    mapping = label_to_option(len(option_ids) - 1)
    if sorted(mapping.values()) != sorted(option_ids):
        raise ChatAnswerError("the offered options are not a pool's options plus 'none'")
    on_labels: dict[str, float] = {}
    top_mass = 0.0
    for entry in top:
        if not isinstance(entry, dict) or not isinstance(entry.get("token"), str):
            raise ChatAnswerError("a top log-probability entry has no token")
        p = math.exp(_logprob(entry.get("logprob"), "a top log-probability"))
        top_mass += p
        label = entry["token"].strip()
        if label in mapping:
            on_labels[label] = on_labels.get(label, 0.0) + p
    label_mass = math.fsum(on_labels.values())
    if not on_labels or label_mass < MIN_LABEL_MASS:
        raise ChatAnswerError(f"only {label_mass:.3f} of the first token's probability is on option "
                              f"labels (at least {MIN_LABEL_MASS} is required)")
    own = generated.strip()
    if own not in mapping:
        raise ChatAnswerError(f"the first token {generated!r} is not an option label")
    if own not in on_labels:
        raise ChatAnswerError("the model's own first token is missing from its top log-probabilities")
    probabilities = {option: 0.0 for option in option_ids}
    for label, p in on_labels.items():
        probabilities[mapping[label]] = p / label_mass
    if on_labels[own] < max(on_labels.values()) - 1e-12:
        raise ChatAnswerError("the model's own label is not the most probable option label")
    return {**tokens, "model": model, "id": data.get("id"), "provider": PROVIDER,
            "system_fingerprint": data.get("system_fingerprint"), "cost_usd": None,
            "choice": mapping[own], "confidence": probabilities[mapping[own]],
            "probabilities": probabilities,
            "probability_sum_drift": abs(math.fsum(probabilities.values()) - 1.0),
            "label_mass": label_mass, "top_mass": min(top_mass, 1.0),
            "labels_reported": len(on_labels)}


def reported_cost(text: str | None) -> Decimal | None:
    """Give the cost of a response: none, because the service is not billed to the owner.

    Args: text: a response body, or None (unused).
    Returns: zero.
    Raises: None.
    Side effects: none.
    """
    return Decimal(0)
