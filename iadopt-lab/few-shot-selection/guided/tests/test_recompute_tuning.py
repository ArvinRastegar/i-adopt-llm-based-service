"""Offline tests for recompute_tuning.py against rag_tuning.py: two computations of one summary."""

from __future__ import annotations

import copy
import json
import random

import knowledge as kn
import pytest
import rag_tuning as rt
import recompute_tuning as rc
from engine import ft

DRAWS = 300


def overlap(first: str, second: str) -> float:
    """Word overlap of two texts, standing in for the embedding model.

    Args: first, second: texts.
    Returns: shared words over all words.
    Raises: nothing.
    Side Effects: none.
    """
    a, b = set(kn.words(first)), set(kn.words(second))
    return len(a & b) / len(a | b) if a | b else 0.0


def totals_f1(evaluations, ids=None):
    """Micro F1 from Close totals, standing in for the study's aggregator.

    Args: evaluations: score rows' evaluations; ids: unused.
    Returns: {"f1": value}.
    Raises: nothing.
    Side Effects: none.
    """
    tp = sum(e["close"]["totals"]["tp"]["numerator"] / e["close"]["totals"]["tp"]["denominator"] for e in evaluations)
    fp = sum(e["close"]["totals"]["fp"]["numerator"] / e["close"]["totals"]["fp"]["denominator"] for e in evaluations)
    fn = sum(e["close"]["totals"]["fn"]["numerator"] / e["close"]["totals"]["fn"]["denominator"] for e in evaluations)
    return {"f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


@pytest.fixture(scope="module")
def ctx():
    """The stage's context with a word-overlap similarity table.

    Args: none.
    Returns: the mapping from `rag_tuning.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files.
    """
    base = kn.load()
    return rt.load(table=rt.similarity_table(base["ids"], lambda v: base["records"][v]["definition"], overlap))


@pytest.fixture(scope="module")
def invented(ctx):
    """Invented records and scores (with fractional counts) for every arm, variable and repetition.

    Args: ctx: as above.
    Returns: (calls, scores).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    rng = random.Random(13)
    calls, scores = {}, {}
    for repetition in range(1, rt.REPETITIONS + 1):
        for variable_id in ctx["ids"]:
            answers = {arm: json.dumps({"n": rng.random()}) for arm in rt.ARMS}
            repaired = rng.random() < 0.05
            if not repaired:
                answers["rag-25+retry"] = answers["rag-25"]
            for position, arm in enumerate(rt.ARMS):
                key = (arm, repetition, variable_id)
                right = sum(rng.random() < 0.4 + 0.02 * position for _ in range(6))
                calls[key] = {"arm": arm, "repetition": repetition, "variable_id": variable_id, "ok": True,
                              "started_at": f"{repetition}/{variable_id}", "answer": answers[arm],
                              "host": "Near AI", "reasoning_chars": 0, "reasoning_tokens": 0,
                              "calls": rng.randint(0, 1), "prompt_tokens": rng.randint(3000, 9000),
                              "cost_usd": rng.random() / 1000, "unit_cost_usd": rng.random() / 100 if arm == "base" else None,
                              "prompt_sha256": ft.sha256(f"{arm} {variable_id}"),
                              "draft_sha256": ft.sha256(answers["base"]) if arm == "rag-25-labels" else None,
                              "first_sha256": ft.sha256(answers["rag-25"]) if arm == "rag-25+retry" else None,
                              "repaired": arm == "rag-25+retry" and repaired,
                              "repair_valid": (rng.random() < 0.7) if arm == "rag-25+retry" and repaired else None}
                scores[key] = {"key": list(key), "answer_sha256": ft.sha256(answers[arm]),
                               "valid_prediction": rng.random() > 0.02, "prediction": None,
                               "evaluation": {"close": {"totals": {
                                   "tp": {"numerator": 3 * right + rng.randint(0, 2), "denominator": 3},
                                   "fp": {"numerator": rng.randint(0, 2), "denominator": 1},
                                   "fn": {"numerator": 6 - right, "denominator": 1}}}}}
    return calls, scores


@pytest.fixture()
def stored(ctx, invented, tmp_path, monkeypatch):
    """Write the invented run where the recomputation reads, with its summary.

    Args: ctx, invented: as above; tmp_path: a temporary folder; monkeypatch: to point the
        recomputation there.
    Returns: {"calls", "summary", "write"}.
    Raises: nothing.
    Side Effects: files in the temporary folder.
    """
    monkeypatch.setattr(rt, "_aggregate", totals_f1)
    calls, scores = invented
    summary = rt.summarise(ctx, calls, scores, draws=DRAWS)
    folder = tmp_path / "guided"
    folder.mkdir()
    members = {"examples": [], "practice": [], "held-out": []}
    for variable_id, group in ctx["group_of"].items():
        members[group].append(variable_id)
    for group, file in rc.GROUP_FILES.items():
        (folder / file).write_text("".join(json.dumps({"variable_id": v}) + "\n" for v in members[group]))
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scores.values()))
    digests = {arm: ft.sha256("\n".join(ft.sha256(f"{arm} {v}") for v in sorted(ctx["ids"])))[:16]
               for arm in ("base", "rag-25", "rag-85")}
    for name, value in (("CALLS", tmp_path / "calls.jsonl"), ("SCORES", tmp_path / "scores.jsonl"),
                        ("SUMMARY", tmp_path / "summary.json"), ("GROUP_FOLDER", folder), ("DRAWS", DRAWS),
                        ("PLAN_DIGESTS", digests)):
        monkeypatch.setattr(rc, name, value)

    def write(changed_calls=None, changed_summary=None):
        """Store records and summary, changed or as they are.

        Args: changed_calls: records to store instead; changed_summary: summary to store instead.
        Returns: nothing.
        Raises: nothing.
        Side Effects: writes two files.
        """
        (tmp_path / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (changed_calls or calls).values()))
        (tmp_path / "summary.json").write_text(json.dumps(changed_summary or summary))

    write()
    return {"calls": calls, "summary": summary, "write": write}


def test_the_lists_are_the_stages(ctx):
    """The recomputation holds the same arms, settings, seeds and level as the stage, by its own copy."""
    assert rc.ARMS == rt.ARMS and rc.CONFIGS == rt.CONFIGS and (rc.BASELINE, rc.REFERENCE) == (rt.BASELINE, rt.REFERENCE)
    assert (rc.DRAWS, rc.SEED, rc.STRICT, rc.REPETITIONS) == (rt.DRAWS, rt.SEED, rt.STRICT, rt.REPETITIONS)


def test_the_recomputation_agrees_with_an_untouched_summary(stored, capsys):
    """Worked out a second way, every figure matches."""
    assert rc.main() == 0
    out = capsys.readouterr().out
    assert "all figures match" in out and "near relatives found again: 21" in out


def test_the_recomputation_names_every_fault_planted_in_the_summary(stored, capsys):
    """A wrong score, choice, comparison, interval, split, retry tally or cost: each is reported."""
    summary = copy.deepcopy(stored["summary"])
    summary["arms"]["rag-50"]["f1_mean"] += 0.01
    summary["arms"]["rag-85"]["by_group"]["practice"] -= 0.02
    summary["arms"]["base"]["invalid"] += 1
    summary["arms"]["rag-10"]["prompt_tokens_mean"] += 5
    summary["chosen"]["f1_mean"] += 0.003
    summary["chosen"]["by_group"]["held-out"] += 0.01
    summary["comparisons"]["chosen vs base"]["strict_low"] -= 0.01
    summary["comparisons"]["chosen vs rag-25"]["welch"]["p"] *= 3
    summary["against_base"]["rag-25-mmr"]["gain"] += 0.004
    summary["against_reference"]["fixed+rag-10"]["high"] += 0.01
    summary["without_relatives"]["chosen"]["low"] -= 0.01
    summary["retry"]["repaired"] += 1
    summary["spent_usd"] += 0.5
    stored["write"](changed_summary=summary)
    assert rc.main() == 1
    out = capsys.readouterr().out
    for label in ("rag-50 mean", "rag-85 practice", "base invalid", "rag-10 prompt tokens", "chosen mean",
                  "chosen held-out", "chosen vs base strict_low", "chosen vs rag-25 runs p",
                  "against_base rag-25-mmr gain", "against_reference fixed+rag-10 high",
                  "without relatives chosen low", "retry repaired", "spent"):
        assert any(line.startswith("MISMATCH") and label in line for line in out.splitlines()), label
    for change, label in ((lambda s: s["chosen"]["by_fold"].update({"2": "rag-10" if s["chosen"]["by_fold"]["2"] != "rag-10" else "rag-50"}),
                           "chosen by fold"),
                          (lambda s: s.update(complete=False), "not of a complete run"),
                          (lambda s: s["comparisons"].pop("chosen vs rag-25"), "not the two of the plan"),
                          (lambda s: s["against_base"].pop("rag-50"), "descriptive comparisons are not every setting"),
                          (lambda s: s["without_relatives"].pop("rag-85"), "split without relatives is not")):
        other = copy.deepcopy(stored["summary"])
        change(other)
        stored["write"](changed_summary=other)
        assert rc.main() == 1 and label in capsys.readouterr().out, label


def test_the_recomputation_names_every_fault_planted_in_the_records(stored, capsys):
    """Another host, reasoning, a fixed prompt changed, a missing record, a torn unit, a changed answer."""
    calls = stored["calls"]
    ids = sorted({key[2] for key in calls})
    bad = dict(calls)
    bad[("rag-50", 2, ids[0])] = dict(calls[("rag-50", 2, ids[0])], host="Wafer")
    bad[("rag-10", 3, ids[1])] = dict(calls[("rag-10", 3, ids[1])], reasoning_tokens=4)
    bad[("rag-85", 4, ids[2])] = dict(calls[("rag-85", 4, ids[2])], prompt_sha256="0" * 64)
    stored["write"](changed_calls=bad)
    assert rc.main() == 1
    out = capsys.readouterr().out
    for label in ("another host", "1 records carry reasoning", "rag-85: the fixed prompts were not sent once"):
        assert any(line.startswith("MISMATCH") and label in line for line in out.splitlines()), label
    moved = {key: dict(row, prompt_sha256="1" * 64) if key[0] == "base" and key[2] == ids[3] else row
             for key, row in calls.items()}
    stored["write"](changed_calls=moved)
    assert rc.main() == 1 and "base: the prompts sent" in capsys.readouterr().out
    fewer = {key: row for key, row in calls.items() if key != ("rag-25-mmr", 5, ids[4])}
    stored["write"](changed_calls=fewer)
    assert rc.main() == 1 and "not 11 arms x 102 variables x 10 repetitions" in capsys.readouterr().out
    torn = dict(calls)
    torn[("rag-25-labels", 6, ids[5])] = dict(calls[("rag-25-labels", 6, ids[5])], draft_sha256="x")
    stored["write"](changed_calls=torn)
    assert rc.main() == 1
    out = capsys.readouterr().out
    assert "1 units hold records that do not belong together" in out
    swapped = dict(calls)
    swapped[("rag-25", 7, ids[6])] = dict(calls[("rag-25", 7, ids[6])], answer="another answer")
    stored["write"](changed_calls=swapped)
    assert rc.main() == 1 and "not scored" in capsys.readouterr().out


def test_equal_settings_are_chosen_alike_by_both_computations(ctx, invented, tmp_path, monkeypatch, capsys):
    """Where two settings score exactly alike and best, both computations choose the first of them."""
    calls, scores = invented
    tied = {}
    for key, row in scores.items():
        arm = key[0]
        if arm in ("rag-25", "rag-10"):
            source = scores[("rag-25", key[1], key[2])]
            tied[key] = dict(row, evaluation={"close": {"totals": {
                "tp": {"numerator": 30, "denominator": 3}, "fp": source["evaluation"]["close"]["totals"]["fp"],
                "fn": {"numerator": 0, "denominator": 1}}}})
        else:
            tied[key] = row
    monkeypatch.setattr(rt, "_aggregate", totals_f1)
    summary = rt.summarise(ctx, calls, tied, draws=DRAWS)
    assert set(summary["chosen"]["by_fold"].values()) == {"rag-25"}
    folder = tmp_path / "guided"
    folder.mkdir()
    members = {"examples": [], "practice": [], "held-out": []}
    for variable_id, group in ctx["group_of"].items():
        members[group].append(variable_id)
    for group, file in rc.GROUP_FILES.items():
        (folder / file).write_text("".join(json.dumps({"variable_id": v}) + "\n" for v in members[group]))
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tied.values()))
    (tmp_path / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in calls.values()))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    for name, value in (("CALLS", tmp_path / "calls.jsonl"), ("SCORES", tmp_path / "scores.jsonl"),
                        ("SUMMARY", tmp_path / "summary.json"), ("GROUP_FOLDER", folder), ("DRAWS", DRAWS),
                        ("PLAN_DIGESTS", {})):
        monkeypatch.setattr(rc, name, value)
    assert rc.main() == 0
    assert "chosen by fold {'0': 'rag-25'" in capsys.readouterr().out

