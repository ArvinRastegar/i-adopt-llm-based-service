"""Run health for a scorer's run: what must hold before its numbers are allowed to mean anything.

Everything here is a function of stored evidence: nothing reads a file or calls a service. A check
has the baseline's three states (passed, failed, not run) and the baseline's rule that a failed
check stops the run before a report is written. A paid response is stored by the worker that
received it whether or not it is a valid answer; the checks judge what is stored.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from typing import Any

from .health import check
from .jev import (
    ARM_CHOICE,
    COST_QUANTUM,
    MODEL,
    PRIMARY_PASS,
    JevError,
    parse_choice,
    parse_noul,
)
from .jev_inputs import Task
from .retrieve import response_hash_matches

CHECK_INPUTS = "inputs identical to the baseline"
CHECK_ANSWERED = "every request answered"
CHECK_REQUESTS = "stored requests regenerate from the plan"
CHECK_HASHES = "stored response hashes verify"
CHECK_PARSED = "every response is a valid answer"
CHECK_MODEL = "one model snapshot served"
CHECK_COST = "spend fully accounted and within the cap"
CHECK_PRIMARY = "primary pass complete"
REQUIRED_JEV_CHECKS = (CHECK_INPUTS, CHECK_ANSWERED, CHECK_REQUESTS, CHECK_HASHES, CHECK_PARSED,
                       CHECK_MODEL, CHECK_COST, CHECK_PRIMARY)


hash_matches = response_hash_matches


def answered_ok(record: Mapping[str, Any] | None) -> bool:
    """Say whether a stored exchange is a finished, successful answer.

    Args: record: the latest stored exchange for a task, or None.
    Returns: True for a 200 with text, no error, no parse error and a matching hash.
    Raises: None.
    Side effects: none.
    """
    return (record is not None and record.get("status") == 200 and record.get("error") is None
            and record.get("parse_error") is None and record.get("response_text") is not None
            and hash_matches(record))


def parse_answer(task: Task, text: str) -> dict[str, Any]:
    """Read a response to a task with the parser for its arm.

    Args: task: the request; text: the response body.
    Returns: parse_choice or parse_noul output.
    Raises: JevError if the response is not a valid answer to that request.
    Side effects: none.
    """
    if task.spec.arm == ARM_CHOICE:
        return parse_choice(text, task.offered)
    return parse_noul(text, task.offered)


def read_answers(tasks: Sequence[Task], recorded: Mapping[tuple[str, str], Mapping[str, Any]],
                 parse: Callable[[Task, str], dict[str, Any]] | None = None
                 ) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[Any]]]:
    """Parse every stored answer and sort out the tasks that have none.

    Args: tasks: the requests of the run; recorded: the latest stored exchange per task key; parse:
    the scorer's answer parser (Jev's by default).
    Returns: parsed answers by task key, and the problems: ``missing`` (never sent), ``failed``
    (sent, no usable response), ``bad_hash`` and ``unparseable`` (a response that is not a valid
    answer, with the reason).
    Raises: None; a parser bug (anything but JevError) propagates.
    Side effects: none.
    """
    parse = parse or parse_answer
    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    problems: dict[str, list[Any]] = {"missing": [], "failed": [], "bad_hash": [], "unparseable": []}
    for task in tasks:
        record = recorded.get(task.key)
        if record is None:
            problems["missing"].append(task.key)
        elif not hash_matches(record):
            problems["bad_hash"].append(task.key)
        elif not answered_ok(record):
            problems["failed"].append(task.key)
        else:
            try:
                parsed[task.key] = parse(task, record["response_text"])
            except JevError as exc:
                problems["unparseable"].append((task.key, str(exc)))
    return parsed, problems


def reported_cost(text: str | None) -> Decimal | None:
    """Read the cost a response reports, whether or not the response is a valid answer.

    A response the validity rule rejects was still paid for, so its cost has to be counted.

    Args: text: a response body, or None.
    Returns: ``usage.cost`` in USD rounded to a picodollar, or None when the text is not JSON or
    carries no usable cost (missing, negative, not finite, or too large to represent).
    Raises: None.
    Side effects: none.
    """
    if text is None:
        return None
    try:
        cost = json.loads(text)["usage"]["cost"]
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            return None
        if not math.isfinite(cost) or cost < 0:
            return None
        return Decimal(repr(float(cost))).quantize(COST_QUANTUM).normalize()
    except (ValueError, KeyError, TypeError, ArithmeticError, RecursionError):
        # Not JSON (or nested too deeply to parse), no usage block, or a number the arithmetic
        # cannot hold (1e30 does not fit a picodollar quantum, an integer of 400 digits does not fit
        # a float, one of 5,000 digits is refused by the parser): no usable cost.
        return None


def history_spend(history: Sequence[Mapping[str, Any]],
                  cost_of: Callable[[str | None], Decimal | None] | None = None
                  ) -> tuple[Decimal, int]:
    """Add up what every stored response cost, valid or rejected.

    Args: history: every stored exchange in file order; cost_of: reads the cost a response text
    reports (Jev's ``usage.cost`` by default).
    Returns: the exact total in USD and how many stored 200 responses report no cost.
    Raises: None.
    Side effects: none.
    """
    cost_of = cost_of or reported_cost
    total, unpriced = Decimal(0), 0
    for record in history:
        if record.get("status") == 200 and record.get("response_text") is not None:
            cost = cost_of(record["response_text"])
            if cost is None:
                unpriced += 1
            else:
                total += cost
    return total, unpriced


def unverified(history: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Find stored exchanges whose response no longer matches its hash, superseded ones included.

    Args: history: every stored exchange in file order.
    Returns: the exchanges that fail the hash check.
    Raises: None.
    Side effects: none.
    """
    return [record for record in history if not hash_matches(record)]


def jev_checks(tasks: Sequence[Task], recorded: Mapping[tuple[str, str], Mapping[str, Any]],
               parsed: Mapping[tuple[str, str], Mapping[str, Any]],
               problems: Mapping[str, Sequence[Any]], *, history: Sequence[Mapping[str, Any]],
               inputs: Mapping[str, int], cap_usd: Decimal | None) -> list[dict[str, Any]]:
    """Run every check on the stored evidence.

    Args: tasks: the requests; recorded: latest stored exchange per task key; parsed and problems:
    read_answers output; history: every stored exchange, including superseded failures; inputs:
    counts from the baseline verification (slots, candidates, files); cap_usd: the spending cap,
    or None for no cap.
    Returns: the checks in REQUIRED_JEV_CHECKS order.
    Raises: None.
    Side effects: none.
    """
    total = len(tasks)
    done = [recorded[t.key] for t in tasks if t.key in parsed]
    retried = sum(r["attempts"] > 1 for r in done)
    unanswered = len(problems["missing"]) + len(problems["failed"])
    models = sorted({a["model"] for a in parsed.values()})
    providers = sorted({str(a["provider"]) for a in parsed.values()})
    spent, no_cost = history_spend(history)
    drifts = [a["probability_sum_drift"] for a in parsed.values() if "probability_sum_drift" in a]
    mismatched = [t.key for t in tasks
                  if t.key in recorded and (recorded[t.key].get("request_sha256") != t.sha256
                                            or recorded[t.key].get("shown") != list(t.shown))]
    primary = [t for t in tasks if t.spec.name == PRIMARY_PASS]
    primary_done = sum(t.key in parsed for t in primary)
    altered = unverified(history)
    baseline_slots = inputs.get("baseline_slots")
    inputs_consistent = baseline_slots is None or (
        inputs["slots"] == baseline_slots == inputs.get("baseline_records"))
    rejected = sum(r.get("parse_error") is not None for r in history)
    return [
        check(CHECK_INPUTS, inputs_consistent,
              f"{inputs['slots']} slots and {inputs['candidates']} candidate documents rebuilt "
              f"and equal to the baseline's; {inputs['baseline_files']} baseline files re-hashed"
              + (f"; the baseline scored {baseline_slots} slots in {inputs['baseline_records']} "
                 "records" if baseline_slots is not None else "")
              if inputs_consistent else
              f"this tree builds {inputs['slots']} slots but the baseline scored {baseline_slots} "
              f"in {inputs.get('baseline_records')} records"),
        check(CHECK_ANSWERED, len(parsed) == total,
              f"{len(parsed)}/{total} requests answered ({retried} needed an HTTP retry; "
              f"{rejected} stored responses were rejected as invalid answers and asked again)"
              if unanswered == 0 else
              f"{unanswered} of {total} requests have no usable response "
              f"({len(problems['missing'])} never sent)"),
        check(CHECK_REQUESTS, not mismatched,
              f"{total - len(mismatched)}/{total} stored requests equal the ones this tree builds"),
        check(CHECK_HASHES, not altered,
              f"{sum(r.get('response_text') is not None for r in history)} stored responses "
              "re-hashed" if not altered else
              f"{len(altered)} stored responses do not match their hash"),
        check(CHECK_PARSED, not problems["unparseable"],
              f"{len(parsed)} responses read strictly; probabilities sum to 1 within "
              f"{max(drifts, default=0.0):.3f} at worst" if not problems["unparseable"] else
              f"{len(problems['unparseable'])} responses are not valid answers, e.g. "
              f"{problems['unparseable'][0][1]}"),
        check(CHECK_MODEL, len(models) == 1 and models[0].startswith(MODEL) and len(providers) == 1,
              f"served by {', '.join(providers) or 'nobody'} as {', '.join(models) or 'nothing'} "
              f"(requested {MODEL})"),
        check(CHECK_COST, no_cost == 0 and (cap_usd is None or spent <= cap_usd),
              f"${format(spent, 'f')} spent over {len(parsed)} answers and {rejected} rejected "
              "responses, every stored response included"
              + (f"; cap ${format(cap_usd, 'f')}" if cap_usd is not None else "")
              + (f"; {no_cost} responses report no cost" if no_cost else "")),
        check(CHECK_PRIMARY, bool(primary) and primary_done == len(primary),
              f"{primary_done}/{len(primary)} requests of {PRIMARY_PASS} answered"),
    ]
