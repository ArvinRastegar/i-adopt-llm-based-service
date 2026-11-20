"""Offline tests for the stage 1 analysis: `decision_value.summarise` and `recompute_decisions`.

Invented scores for the real 102 variables and 25 arms are summarised by the one and checked by
the other; then faults are planted in the stored summary and in the stored calls, and the
recomputation has to name each one. No provider, no similarity model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random

import decision_value as dv
import engine
import pytest
import recompute_decisions as recompute
from engine import ft

DRAWS = 300


def totals_f1(evaluations, ids=None):
    """Micro F1 from the Close totals of some evaluations, standing in for the study's aggregator.

    Args: evaluations: score rows' "evaluation" objects; ids: unused, as the aggregator takes them.
    Returns: {"f1": value}.
    Raises: nothing.
    Side Effects: none.
    """
    tp = fp = fn = 0.0
    for evaluation in evaluations:
        totals = evaluation["close"]["totals"]
        tp += totals["tp"]["numerator"] / totals["tp"]["denominator"]
        fp += totals["fp"]["numerator"] / totals["fp"]["denominator"]
        fn += totals["fn"]["numerator"] / totals["fn"]["denominator"]
    return {"f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


@pytest.fixture(scope="module")
def ctx():
    """Load the stage's context once per module.

    Args: none.
    Returns: the mapping from `decision_value.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files, among them the stored walks of steps 2 to 5.
    """
    if not (engine.OUTPUT / "route-examples-direct-all.jsonl").exists():
        pytest.skip("the stored walks of steps 2 to 5 are not on this machine (gitignored output)")
    return dv.load()


@pytest.fixture(scope="module")
def invented(ctx):
    """Invented calls and scores for every arm, variable and repetition, with the real prompt hashes.

    Args: ctx: from `load`.
    Returns: (calls, scores) keyed like `engine.latest` and `engine.score`; later arms score higher.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    rng = random.Random(5)
    calls, scores = {}, {}
    for position, arm in enumerate(dv.ARMS):
        hashes = {v: ft.sha256(dv.prompt_of(ctx, (arm, 1, v))) for v in ctx["ids"]}
        for repetition in range(1, dv.REPETITIONS + 1):
            for variable_id in ctx["ids"]:
                right = sum(rng.random() < 0.35 + 0.018 * position for _ in range(6))
                answer = f"{arm}|{repetition}|{variable_id}"
                key = (arm, repetition, variable_id)
                calls[key] = {"arm": arm, "repetition": repetition, "variable_id": variable_id, "ok": True,
                              "status": 200, "error": None, "answer": answer, "reasoning_chars": 0,
                              "reasoning_tokens": 0, "finish_reason": "stop", "prompt_tokens": 5000,
                              "completion_tokens": 100, "cost_usd": 0.0003, "host": "Near AI",
                              "served_model": "qwen/qwen3.8-27b", "prompt_sha256": hashes[variable_id]}
                scores[key] = {"key": list(key), "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                               "valid_prediction": rng.random() > 0.02, "prediction": None,
                               "evaluation": {"close": {"totals": {
                                   "tp": {"numerator": 3 * right + (rng.random() < 0.3), "denominator": 3},
                                   "fp": {"numerator": rng.randint(0, 2), "denominator": 1},
                                   "fn": {"numerator": 6 - right, "denominator": 1}}}}}
    return calls, scores


def pooled(scores, arm, ids, repetitions):
    """One arm's micro F1 over some variables and repetitions put together, worked out directly.

    Args: scores: invented scores; arm: an arm; ids: variables; repetitions: repetition numbers.
    Returns: the F1.
    Raises: KeyError when a score is missing.
    Side Effects: none.
    """
    return totals_f1([scores[(arm, r, v)]["evaluation"] for r in repetitions for v in ids])["f1"]


def test_a_complete_run_is_summarised_arm_by_arm_and_pair_by_pair(ctx, invented, monkeypatch):
    """Every arm's mean, every comparison's two arms and difference, and the split by presence."""
    monkeypatch.setattr(dv, "_aggregate", totals_f1)
    calls, scores = invented
    summary = dv.summarise(ctx, calls, scores, draws=DRAWS)
    every = list(range(1, dv.REPETITIONS + 1))
    assert summary["complete"] is True and summary["repetitions_used"] == every
    assert list(summary["arms"]) == list(dv.ARMS) and len(summary["contrasts"]) == 34
    for arm, cell in summary["arms"].items():
        per_run = [pooled(scores, arm, ctx["ids"], [r]) for r in every]
        assert cell["f1_per_repetition"] == pytest.approx(per_run) and cell["repetitions"] == 10
        assert cell["failed_calls"] == 0 and cell["cost_usd"] == pytest.approx(1020 * 0.0003)
        assert cell["invalid"] == sum(not scores[(arm, r, v)]["valid_prediction"] for r in every for v in ctx["ids"])
    for name, first, second in dv.contrasts():
        row = summary["contrasts"][name]
        assert (row["first"], row["second"]) == (first, second)
        assert row["gain"] == pytest.approx(pooled(scores, first, ctx["ids"], every)
                                            - pooled(scores, second, ctx["ids"], every))
        assert row["low"] <= row["gain"] <= row["high"] and 0.0 <= row["p_holm"] <= 1.0
        assert row["welch"]["difference"] == pytest.approx(
            summary["arms"][first]["f1_mean"] - summary["arms"][second]["f1_mean"])
    assert summary["contrasts"]["key vs base"]["gain"] > 0 > summary["contrasts"]["soft-key vs key"]["gain"] - 1
    sizes = {d: (rows["present"]["variables"], rows["absent"]["variables"])
             for d, rows in summary["by_presence"].items()}
    assert sizes == {"matrix": (51, 51), "context": (11, 91), "modifier": (8, 94), "constraint": (86, 16)}
    with_matrix = [v for v in ctx["ids"] if ctx["key"][v]["matrix"]]
    assert summary["by_presence"]["matrix"]["present"]["value"]["gain"] == pytest.approx(
        pooled(scores, "key", with_matrix, every) - pooled(scores, "drop-matrix", with_matrix, every))
    assert set(summary["break_even"]) == set(dv.DECISIONS)


def test_an_unfinished_repetition_enters_no_figure(ctx, invented, monkeypatch):
    """One failed call in one arm's last repetition takes that repetition out for every arm."""
    monkeypatch.setattr(dv, "_aggregate", totals_f1)
    calls, scores = invented
    broken = dict(calls)
    key = ("walk", 10, ctx["ids"][3])
    broken[key] = dict(calls[key], ok=False, status=500, error="HTTP 500", answer=None)
    summary = dv.summarise(ctx, broken, scores, draws=DRAWS)
    nine = list(range(1, 10))
    assert summary["complete"] is False and summary["repetitions_used"] == nine
    assert all(cell["repetitions"] == 9 for cell in summary["arms"].values())
    assert summary["arms"]["walk"]["failed_calls"] == 1
    assert summary["contrasts"]["key vs base"]["gain"] == pytest.approx(
        pooled(scores, "key", ctx["ids"], nine) - pooled(scores, "base", ctx["ids"], nine))
    full = dv.summarise(ctx, calls, scores, draws=DRAWS)
    assert summary["contrasts"]["key vs base"]["gain"] != pytest.approx(full["contrasts"]["key vs base"]["gain"],
                                                                         abs=1e-12)
    missing = {k: row for k, row in calls.items() if k != ("base", 4, ctx["ids"][0])}
    assert dv.summarise(ctx, missing, scores, draws=DRAWS)["repetitions_used"] == [1, 2, 3, 5, 6, 7, 8, 9, 10]


def test_an_arm_that_was_not_run_is_left_out_and_the_summary_is_not_complete(ctx, invented, monkeypatch):
    """Without one arm its comparisons are missing and the summary says it is not complete."""
    monkeypatch.setattr(dv, "_aggregate", totals_f1)
    calls, scores = invented
    fewer = {key: row for key, row in calls.items() if key[0] != "cand-property-3"}
    summary = dv.summarise(ctx, fewer, scores, draws=DRAWS)
    assert summary["complete"] is False and "cand-property-3" not in summary["arms"]
    assert len(summary["arms"]) == 24 and len(summary["contrasts"]) == 32
    never = {key: row for key, row in calls.items() if not (key[0] == "base" and key[2] == ctx["ids"][0])}
    empty = dv.summarise(ctx, never, scores, draws=DRAWS)
    assert empty["complete"] is False and empty["repetitions_used"] == [] and empty["arms"] == {}


@pytest.fixture()
def stored(ctx, invented, tmp_path, monkeypatch):
    """Write the invented run where the recomputation reads, with its summary.

    Args: ctx, invented: as above; tmp_path: a temporary folder; monkeypatch: to point the
        recomputation at it.
    Returns: {"calls", "scores", "summary": the summary object, "write": writes changed calls and
        summary back}.
    Raises: nothing.
    Side Effects: files in the temporary folder.
    """
    monkeypatch.setattr(dv, "_aggregate", totals_f1)
    calls, scores = invented
    summary = dv.summarise(ctx, calls, scores, draws=DRAWS)
    folder = tmp_path / "guided"
    folder.mkdir()
    members = {"examples": [], "practice": [], "held-out": []}
    for variable_id, group in ctx["group_of"].items():
        members[group].append(variable_id)
    for group, file in recompute.GROUP_FILES.items():
        (folder / file).write_text("".join(json.dumps({"variable_id": v}) + "\n" for v in members[group]))
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scores.values()))
    for name, value in (("CALLS", tmp_path / "calls.jsonl"), ("SCORES", tmp_path / "scores.jsonl"),
                        ("SUMMARY", tmp_path / "summary.json"), ("LEDGER", tmp_path / "ledger.jsonl"),
                        ("GROUP_FOLDER", folder), ("DRAWS", DRAWS)):
        monkeypatch.setattr(recompute, name, value)

    def write(changed_calls=None, changed_summary=None):
        """Store calls and summary, changed or as they are.

        Args: changed_calls: calls to store instead; changed_summary: summary to store instead.
        Returns: nothing.
        Raises: nothing.
        Side Effects: writes two files.
        """
        rows = (changed_calls or calls).values()
        (tmp_path / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (tmp_path / "summary.json").write_text(json.dumps(changed_summary or summary))

    write()
    return {"calls": calls, "scores": scores, "summary": summary, "write": write}


def test_the_recomputation_agrees_with_an_untouched_summary(stored, capsys):
    """Worked out a second way, with its own list of comparisons, every figure matches."""
    assert recompute.main() == 0
    out = capsys.readouterr().out
    assert "all figures match" in out and "comparisons checked 34" in out
    assert "prompt digests match the plan for 25 arms" in out
    assert {name: pair for name, pair in recompute.comparisons().items()} == {
        name: (first, second) for name, first, second in dv.contrasts()}


def test_the_recomputation_names_every_fault_planted_in_the_summary(stored, capsys):
    """A wrong figure of any kind, a missing, extra or re-paired comparison: each is reported."""
    summary = copy.deepcopy(stored["summary"])
    contrasts = summary["contrasts"]
    contrasts["key vs base"]["welch"]["p"] *= 1.5
    contrasts["walk vs base"]["p_holm"] = min(1.0, contrasts["walk vs base"]["p_holm"] * 0.5 + 0.01)
    contrasts["soft-key vs key"]["share_positive"] = 0.5 * contrasts["soft-key vs key"]["share_positive"] + 0.2
    contrasts["soft-walk vs walk"]["gain"] += 0.004
    contrasts["soft-walk vs base"]["low"] -= 0.004
    contrasts["soft-key vs base"]["welch"]["difference"] += 0.002
    del contrasts["cand-property-3 vs key"]
    contrasts["made up vs base"] = dict(contrasts["key vs base"], first="walk", second="base")
    contrasts["object alone (only-object vs base)"]["first"] = "only-matrix"
    summary["by_presence"]["matrix"]["present"]["value"]["gain"] += 0.01
    summary["by_presence"]["context"]["absent"]["harm"]["high"] += 0.01
    summary["by_presence"]["modifier"]["present"]["variables"] = 9
    summary["break_even"]["matrix"]["accuracy"] = 0.123          # the invented arms give None or another value
    summary["break_even"]["constraint"]["defined_share"] -= 0.1
    summary["arms"]["key"]["by_group"]["practice"] += 0.01
    summary["arms"]["walk"]["f1_per_repetition"][3] += 0.01
    summary["arms"]["base"]["invalid"] += 1
    summary["arms"]["soft-key"]["cost_usd"] += 0.5
    summary["arms"]["drop-matrix"]["f1_sd"] *= 2
    summary["spent_usd"] += 1.0
    stored["write"](changed_summary=summary)
    assert recompute.main() == 1
    out = capsys.readouterr().out
    for label in ("key vs base runs p", "walk vs base Holm p", "soft-key vs key share_positive",
                  "soft-walk vs walk gain", "soft-walk vs base low", "soft-key vs base runs difference",
                  "missing ['cand-property-3 vs key']", "extra ['made up vs base']",
                  "other arms ['object alone (only-object vs base)']",
                  "by presence matrix present value gain", "by presence context absent harm high",
                  "by presence modifier present variables", "break-even matrix accuracy",
                  "break-even constraint defined", "key practice", "walk repetition 4", "base: invalid",
                  "soft-key cost", "drop-matrix sd", "spent"):
        assert any(line.startswith("MISMATCH") and label in line for line in out.splitlines()), label
    incomplete = copy.deepcopy(stored["summary"])
    incomplete["complete"] = False
    stored["write"](changed_summary=incomplete)
    assert recompute.main() == 1 and "not of a complete run" in capsys.readouterr().out


def test_the_recomputation_names_every_fault_planted_in_the_calls(stored, capsys):
    """Another host, reasoning, a prompt that was not planned, a missing or failed call."""
    calls = stored["calls"]
    ids = sorted({key[2] for key in calls})
    elsewhere = dict(calls)
    elsewhere[("key", 2, ids[0])] = dict(calls[("key", 2, ids[0])], host="Wafer")
    elsewhere[("base", 3, ids[1])] = dict(calls[("base", 3, ids[1])], reasoning_tokens=12)
    elsewhere[("walk", 1, ids[2])] = dict(calls[("walk", 1, ids[2])], prompt_sha256="0" * 64)
    stored["write"](changed_calls=elsewhere)
    assert recompute.main() == 1
    out = capsys.readouterr().out
    for label in ("another host or model", "1 answers carry reasoning", "walk: a variable was sent different prompts"):
        assert any(line.startswith("MISMATCH") and label in line for line in out.splitlines()), label
    other = {key: dict(row, prompt_sha256="1" * 64) if key[0] == "soft-walk" and key[2] == ids[5] else row
             for key, row in calls.items()}
    stored["write"](changed_calls=other)
    assert recompute.main() == 1 and "soft-walk: the prompts sent" in capsys.readouterr().out
    fewer = {key: row for key, row in calls.items() if key != ("only-matrix", 7, ids[4])}
    stored["write"](changed_calls=fewer)
    assert recompute.main() == 1
    assert "not 25 arms x 102 variables x 10 repetitions" in capsys.readouterr().out
    failed = dict(calls)
    failed[("drop-context", 5, ids[6])] = dict(calls[("drop-context", 5, ids[6])], ok=False, answer=None)
    stored["write"](changed_calls=failed)
    assert recompute.main() == 1 and "1 calls failed" in capsys.readouterr().out


def test_the_run_to_run_test_without_a_statistics_library():
    """Welch's test and Holm's adjustment as the recomputation works them out, on known values."""
    out = recompute.welch([0.60, 0.62, 0.61, 0.63, 0.59], [0.50, 0.52, 0.49, 0.51, 0.53])
    assert out["difference"] == pytest.approx(0.10) and out["t"] == pytest.approx(10.0)
    assert out["df"] == pytest.approx(8.0) and out["p"] == pytest.approx(8.4882e-06, rel=1e-4)
    assert recompute.welch([0.5, 0.5], [0.5, 0.5])["p"] == 1.0 and recompute.welch([0.6, 0.6], [0.5, 0.5])["p"] == 0.0
    assert recompute.incomplete_beta(0.5, 0.5, 0.5) == pytest.approx(0.5)
    assert recompute.incomplete_beta(2.0, 3.0, 0.4) == pytest.approx(0.5248)
    assert recompute.incomplete_beta(4.0, 0.5, 1.0) == 1.0 and recompute.incomplete_beta(4.0, 0.5, 0.0) == 0.0
    assert recompute.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}) == pytest.approx(
        {"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.5})
