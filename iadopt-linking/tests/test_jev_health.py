"""Run health on small cases: reading a cost, spend over every stored response, checks that can fail."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from iadopt_linking.jev_health import (
    CHECK_ANSWERED,
    CHECK_INPUTS,
    history_spend,
    jev_checks,
    reported_cost,
)


def body(cost: object) -> str:
    return json.dumps({"usage": {"cost": cost, "input_tokens": 1, "output_tokens": 1}})


@pytest.mark.parametrize(("text", "expected"), [
    (body(0.000019992), Decimal("0.000019992")),
    (body(0), Decimal("0")),
    (body(-1), None),
    (body(True), None),
    (body(None), None),
    (body(1e30), None),                                   # does not fit a picodollar quantum
    ('{"usage": {"cost": 1' + "0" * 400 + "}}", None),     # does not fit a float
    ('{"usage": {"cost": 1' + "0" * 5000 + "}}", None),    # the parser refuses so many digits
    ("[" * 100_000 + "]" * 100_000, None),                  # nested too deeply to parse
    ('{"usage": {}}', None),
    ("{}", None),
    ("not json", None),
    (None, None),
])
def test_reading_a_cost_never_raises(text: str | None, expected: Decimal | None) -> None:
    assert reported_cost(text) == expected


def test_spend_counts_every_stored_response_and_flags_the_ones_without_a_cost() -> None:
    history = [{"status": 200, "response_text": body(0.1)},
               {"status": 200, "response_text": body(0.2)},       # perhaps a rejected answer: paid for
               {"status": 429, "response_text": None},            # nothing was billed
               {"status": 200, "response_text": body(-1)}]        # billed, but the cost is unreadable
    total, unpriced = history_spend(history)
    assert total == Decimal("0.3") and unpriced == 1


def named(checks: list[dict], name: str) -> dict:
    return next(c for c in checks if c["name"] == name)


NO_PROBLEMS = {"missing": [], "failed": [], "bad_hash": [], "unparseable": []}


def run_checks(inputs: dict) -> list[dict]:
    return jev_checks([], {}, {}, NO_PROBLEMS, history=[], inputs=inputs, cap_usd=None)


def test_the_inputs_check_can_fail_when_the_population_differs_from_the_baselines() -> None:
    same = {"slots": 4, "candidates": 9, "baseline_files": 7, "baseline_records": 4,
            "baseline_slots": 4}
    assert named(run_checks(same), CHECK_INPUTS)["passed"] is True
    fewer = {**same, "slots": 3}
    failed = named(run_checks(fewer), CHECK_INPUTS)
    assert failed["passed"] is False and "builds 3 slots but the baseline scored 4" in failed["detail"]
    surplus = {**same, "baseline_records": 5}
    assert named(run_checks(surplus), CHECK_INPUTS)["passed"] is False


def test_a_pilot_is_not_held_to_the_baselines_population() -> None:
    pilot = {"slots": 5, "candidates": 9, "baseline_files": 7, "baseline_records": 294,
             "baseline_slots": None}
    assert named(run_checks(pilot), CHECK_INPUTS)["passed"] is True


def test_a_task_without_an_answer_cannot_pass_the_answered_check() -> None:
    class Task:
        key = ("s", "choice-a1")
        sha256 = "x"
        shown = ()

        class spec:
            name = "choice-a1"

    checks = jev_checks([Task()], {}, {}, {**NO_PROBLEMS, "bad_hash": [("s", "choice-a1")]},
                        history=[], inputs={"slots": 1, "candidates": 1, "baseline_files": 7},
                        cap_usd=None)
    assert named(checks, CHECK_ANSWERED)["passed"] is False
