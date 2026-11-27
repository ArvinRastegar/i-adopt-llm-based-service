"""A chat model as a listwise entity linker: the prompt, the option labels, and how answers are read.
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
    """Name the option shown at a 1-based position with its one-token label."""
    if not 1 <= position <= len(LABELS):
        raise ValueError(f"Option position {position} is outside 1..{len(LABELS)}")
    return LABELS[position - 1]


def shown_candidate(candidate: Mapping[str, Any], max_description_chars: int | None) -> str:
    """Write the part of a candidate the model may see: the label and the capped description."""
    description = truncate_description(candidate["description"], max_description_chars)
    return f'"{candidate["label"]}"' + (f": {description}" if description else "")


def user_message(term: str, definition: str, pool: Sequence[Mapping[str, Any]],
                 order: Sequence[int], max_description_chars: int | None) -> str:
    """Write the case the model is asked about."""
    lines = [f"{option_label(place)}. {shown_candidate(pool[index], max_description_chars)}"
             for place, index in enumerate(order, start=1)]
    lines.append(f"{NONE_LABEL}. none of these: {NONE_DESCRIPTION}")
    return (f"term: {term}\nvariable_definition: {definition}\n\ncandidates:\n" + "\n".join(lines)
            + f"\n\n{ANSWER_LINE}")


def _require_permutation(order: Sequence[int], size: int) -> None:
    """Refuse an order that does not show every candidate exactly once."""
    if size == 0:
        raise ValueError("There is nothing to ask about an empty pool")
    if size > len(LABELS):
        raise ValueError(f"A pool of {size} candidates does not fit {len(LABELS)} option labels")
    if sorted(order) != list(range(size)):
        raise ValueError("The presentation order must show every candidate exactly once")


def chat_request(term: str, definition: str, pool: Sequence[Mapping[str, Any]],
                 order: Sequence[int], *, max_description_chars: int | None,
                 model: str = MODEL) -> dict[str, Any]:
    """Build the chat-completions request for one slot in one candidate order."""
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
    """Identify the whole prompt design in one hash."""
    return content_hash({
        "version": PROMPT_VERSION, "model": model, "system_prompt": SYSTEM_PROMPT,
        "answer_line": ANSWER_LINE, "none_description": NONE_DESCRIPTION, "labels": LABELS,
        "none_label": NONE_LABEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "top_logprobs": TOP_LOGPROBS, "min_label_mass": MIN_LABEL_MASS,
        "reasoning_off": REASONING_OFF[model], "order_seed": ORDER_SEED,
        "passes": [[p.name, p.arm, p.order] for p in CHAT_PASSES]})


def prompt_plan(max_description_chars: int | None, model: str = MODEL) -> dict[str, Any]:
    """Describe the prompt design for the plan."""
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
    """Build every request of a run."""
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
    """Estimate a run's size before any call, erring high."""
    chars = sum(task.chars for task in tasks)
    by_pass = Counter(task.spec.name for task in tasks)
    return {"requests": len(tasks), "requests_by_pass": dict(sorted(by_pass.items())),
            "request_chars": chars, "assumed_chars_per_token": ASSUMED_CHARS_PER_TOKEN,
            "input_tokens_upper_estimate": int(chars / ASSUMED_CHARS_PER_TOKEN) + 1,
            "price_per_input_token_usd": "0", "estimated_cost_usd": "0"}


def label_to_option(count: int) -> dict[str, str]:
    """Map each label to the option id of the position it names."""
    mapping = {option_label(place): option_id(place) for place in range(1, count + 1)}
    mapping[NONE_LABEL] = NONE_KEY
    return mapping


def _logprob(entry: Any, what: str) -> float:
    """Read one log-probability."""
    if isinstance(entry, bool) or not isinstance(entry, (int, float)) or not math.isfinite(entry):
        raise ChatAnswerError(f"{what} is not a finite number: {entry!r}")
    if entry > 1e-6:
        raise ChatAnswerError(f"{what} is above zero: {entry!r}")
    return float(min(entry, 0.0))


def _tokens(value: Any, what: str) -> int:
    """Read a token count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChatAnswerError(f"{what} is not a count: {value!r}")
    return value


def parse_chat_choice(text: str, option_ids: Sequence[str]) -> dict[str, Any]:
    """Read the answer to a chat Choice request strictly."""
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
    """Give the cost of a response: none, because the service is not billed to the owner."""
    return Decimal(0)
