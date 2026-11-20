"""Offline tests for recompute_pipelines.py against draft_rag.py: two computations of one summary.

Invented records and scores for the real 102 variables and six arms are summarised by the one and
checked by the other; then faults are planted, and the recomputation has to name each one.
"""

from __future__ import annotations

import copy
import json
import random

import draft_rag as dr
import hints
import knowledge as kn
import pytest
import recompute_pipelines as rp
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
    tp = sum(e["close"]["totals"]["tp"]["numerator"] for e in evaluations)
    fp = sum(e["close"]["totals"]["fp"]["numerator"] for e in evaluations)
    fn = sum(e["close"]["totals"]["fn"]["numerator"] for e in evaluations)
    return {"f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


@pytest.fixture(scope="module")
def ctx():
    """The stage's context with a word-overlap similarity order.

    Args: none.
    Returns: the mapping from `draft_rag.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files.
    """
    base = kn.load()
    return dr.load(order=dr.similarity_order(base["ids"], lambda v: base["records"][v]["definition"], overlap))


@pytest.fixture(scope="module")
def invented(ctx):
    """Invented records and scores for every arm, variable and repetition.

    Args: ctx: as above.
    Returns: (calls, scores).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    rng = random.Random(9)
    tags = {v: hints.expected_form(hints.facts_from_key(ctx["key"][v]), ctx["patterns"], ctx["types"]) for v in ctx["ids"]}
    voted = set(ctx["ids"][::2])
    calls, scores = {}, {}
    for repetition in range(1, dr.REPETITIONS + 1):
        for variable_id in ctx["ids"]:
            started = f"{repetition}/{variable_id}"
            drafts = {"base": json.dumps({"n": rng.random()}), "rag": json.dumps({"n": rng.random()})}
            for position, arm in enumerate(dr.ARMS):
                key = (arm, repetition, variable_id)
                stmt = None
                if arm in dr.GUIDED and (arm != "rag/gated" or variable_id in voted) and rng.random() < 0.9:
                    stmt = {"decisions": ["constraint", "matrix", "property"], "at_least": True,
                            "property": rng.choice([list(ctx["key"][variable_id]["property"]), [104]]),
                            "matrix_shape": rng.choice([tags[variable_id]["matrix"], "empty"]),
                            "matrix": rng.choice([list(ctx["key"][variable_id]["matrix"]), []]),
                            "constraint_types": rng.choice([tags[variable_id]["constraint_types"][:1], ["time"]])}
                fixed = arm in ("base", "rag") or (arm == "rag/gated" and variable_id in voted)
                own = drafts["base" if arm.startswith("draft") else "rag"]
                fell_back = arm in dr.GUIDED and stmt is not None and rng.random() < 0.05
                if arm in ("base", "rag"):
                    answer = drafts[arm]
                elif stmt is None or fell_back:
                    answer = own
                else:
                    answer = json.dumps({"n": rng.random()})
                right = sum(rng.random() < 0.4 + 0.03 * position for _ in range(6))
                calls[key] = {"arm": arm, "repetition": repetition, "variable_id": variable_id, "ok": True,
                              "started_at": started, "answer": answer, "host": "Near AI", "reasoning_chars": 0,
                              "reasoning_tokens": 0, "calls": rng.randint(0, 2), "cost_usd": rng.random() / 1000,
                              "unit_cost_usd": rng.random() / 100 if arm == "base" else None,
                              "prompt_sha256": ft.sha256(f"{arm} {variable_id}") if fixed else None,
                              "draft_sha256": ft.sha256(own) if arm in dr.GUIDED else None,
                              "statement": stmt, "no_guidance": stmt is None,
                              "first_complaints": rng.choice([[], ["x"]]), "repaired": rng.random() < 0.2,
                              "repair_complaints": rng.choice([[], ["x"]]), "fell_back": fell_back,
                              "draft_complaints": rng.choice([[], ["x"]])}
                scores[key] = {"key": list(key), "answer_sha256": ft.sha256(answer),
                               "valid_prediction": rng.random() > 0.03, "prediction": None,
                               "evaluation": {"close": {"totals": {
                                   "tp": {"numerator": right, "denominator": 1},
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
    monkeypatch.setattr(dr, "_aggregate", totals_f1)
    calls, scores = invented
    summary = dr.summarise(ctx, calls, scores, draws=DRAWS)
    folder = tmp_path / "guided"
    folder.mkdir()
    members = {"examples": [], "practice": [], "held-out": []}
    for variable_id, group in ctx["group_of"].items():
        members[group].append(variable_id)
    for group, file in rp.GROUP_FILES.items():
        (folder / file).write_text("".join(json.dumps({"variable_id": v}) + "\n" for v in members[group]))
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scores.values()))
    digests = {}
    for arm in ("base", "rag", "rag/gated"):
        hashes = {v: calls[(arm, 1, v)]["prompt_sha256"] for v in ctx["ids"] if calls[(arm, 1, v)]["prompt_sha256"]}
        digests[arm] = (ft.sha256("\n".join(hashes[v] for v in sorted(hashes)))[:16], len(hashes))
    for name, value in (("CALLS", tmp_path / "calls.jsonl"), ("SCORES", tmp_path / "scores.jsonl"),
                        ("SUMMARY", tmp_path / "summary.json"), ("GROUP_FOLDER", folder), ("DRAWS", DRAWS),
                        ("PLAN_DIGESTS", digests)):
        monkeypatch.setattr(rp, name, value)

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


def test_the_recomputation_reads_the_tags_and_the_relatives_as_the_stage_does(ctx):
    """Its own reading of the tags, of the folds and of the near relatives equals the stage's."""
    tags = json.loads((rp.DATA / "route_key.json").read_text())["variables"]
    patterns = json.loads((rp.DATA / "patterns.json").read_text())["patterns"]
    names = {row["number"]: row["name"].lower() for row in json.loads((rp.DATA / "constraint_types.json").read_text())["types"]}
    for variable_id in ctx["ids"]:
        mine = rp.tagged_form(tags[variable_id], patterns, names)
        theirs = hints.expected_form(hints.facts_from_key(ctx["key"][variable_id]), ctx["patterns"], ctx["types"])
        assert (mine["matrix"], mine["context"], mine["types"]) == (theirs["matrix"], theirs["context"],
                                                                    theirs["constraint_types"])
    by_group = {g: sorted(v for v, group in ctx["group_of"].items() if group == g) for g in ("examples", "practice", "held-out")}
    fold_of = rp.deal_folds(by_group, tags)
    assert fold_of == ctx["fold_of"]
    assert sorted(rp.relatives(rp.definitions(), fold_of)) == ctx["relatives"]
    assert rp.COMPARED == dr.COMPARED and rp.SECONDARY == dr.SECONDARY and rp.GUIDED == dr.GUIDED
    assert (rp.DRAWS, rp.SEED, rp.STRICT) == (dr.DRAWS, dr.SEED, dr.STRICT)


def test_the_recomputation_agrees_with_an_untouched_summary(stored, capsys):
    """Worked out a second way, every figure matches."""
    assert rp.main() == 0
    assert "all figures match" in capsys.readouterr().out


def test_the_recomputation_names_every_fault_planted_in_the_summary(stored, capsys):
    """A wrong score, comparison, interval, contrast, split, mark or cost: each is reported."""
    summary = copy.deepcopy(stored["summary"])
    summary["arms"]["rag"]["f1_mean"] += 0.01
    summary["arms"]["combined"]["by_group"]["held-out"] -= 0.02
    summary["arms"]["draft-first"]["invalid"] += 1
    summary["arms"]["base"]["calls"] += 3
    summary["comparisons"]["rag/gated"]["strict_high"] += 0.01
    summary["comparisons"]["draft-first/gated"]["welch"]["p"] *= 2
    summary["comparisons"]["combined"]["p_holm"] = min(1.0, summary["comparisons"]["combined"]["p_holm"] * 0.5 + 0.01)
    summary["comparisons"]["rag"]["share_positive"] = 0.5 * summary["comparisons"]["rag"]["share_positive"] + 0.2
    summary["secondary"]["combined vs rag"]["low"] -= 0.01
    summary["without_relatives"]["rag"]["gain"] += 0.01
    summary["with_relatives"]["combined"]["variables"] = 20
    summary["patterns"]["draft-first/gated"]["matrix_right"] += 1
    summary["patterns"]["combined"]["fell_back"] += 2
    summary["spent_usd"] += 1.0
    stored["write"](changed_summary=summary)
    assert rp.main() == 1
    out = capsys.readouterr().out
    for label in ("rag mean", "combined held-out", "draft-first invalid", "base calls", "rag/gated strict_high",
                  "draft-first/gated runs p", "combined Holm p", "rag share_positive", "combined vs rag low",
                  "without relatives rag gain", "with relatives combined variables",
                  "patterns draft-first/gated matrix_right", "patterns combined fell_back", "spent"):
        assert any(line.startswith("MISMATCH") and label in line for line in out.splitlines()), label
    for change, label in ((lambda s: s.update(complete=False), "not of a complete run"),
                          (lambda s: s["comparisons"].pop("rag"), "not the five of the plan"),
                          (lambda s: s["comparisons"]["rag"].update(second="draft-first"), "compares rag with draft-first"),
                          (lambda s: s["secondary"].pop("rag/gated vs rag"), "not the three of the plan"),
                          (lambda s: s["patterns"].pop("rag/gated"), "patterns rag/gated: missing")):
        other = copy.deepcopy(stored["summary"])
        change(other)
        stored["write"](changed_summary=other)
        assert rp.main() == 1 and label in capsys.readouterr().out, label


def test_the_recomputation_names_every_fault_planted_in_the_records(stored, capsys):
    """Another host, reasoning, a fixed prompt changed, a missing record, a changed statement or answer."""
    calls = stored["calls"]
    ids = sorted({key[2] for key in calls})
    elsewhere = dict(calls)
    elsewhere[("rag", 3, ids[0])] = dict(calls[("rag", 3, ids[0])], host="Wafer")
    elsewhere[("base", 2, ids[1])] = dict(calls[("base", 2, ids[1])], reasoning_tokens=9)
    elsewhere[("base", 1, ids[2])] = dict(calls[("base", 1, ids[2])], prompt_sha256="0" * 64)
    stored["write"](changed_calls=elsewhere)
    assert rp.main() == 1
    out = capsys.readouterr().out
    for label in ("another host", "1 records carry reasoning", "base: the fixed prompts were not sent once"):
        assert any(line.startswith("MISMATCH") and label in line for line in out.splitlines()), label
    changed = {key: dict(row, prompt_sha256="1" * 64) if key[0] == "rag" and key[2] == ids[4] else row
               for key, row in calls.items()}
    stored["write"](changed_calls=changed)
    assert rp.main() == 1 and "rag: the prompts sent" in capsys.readouterr().out
    fewer = {key: row for key, row in calls.items() if key != ("combined", 6, ids[5])}
    stored["write"](changed_calls=fewer)
    assert rp.main() == 1 and "not 6 arms x 102 variables x 10 repetitions" in capsys.readouterr().out
    moved = dict(calls)
    key = next(k for k in calls if k[0] == "draft-first" and calls[k]["statement"])
    moved[key] = dict(calls[key], statement=dict(calls[key]["statement"], matrix_shape="source_target"),
                      fell_back=not calls[key]["fell_back"])
    stored["write"](changed_calls=moved)
    assert rp.main() == 1
    out = capsys.readouterr().out
    assert "MISMATCH patterns draft-first fell_back" in out
    swapped = dict(calls)
    swapped[("rag", 1, ids[7])] = dict(calls[("rag", 1, ids[7])], answer="another answer")
    stored["write"](changed_calls=swapped)
    assert rp.main() == 1 and "not scored" in capsys.readouterr().out
    torn = dict(calls)
    torn[("combined", 2, ids[8])] = dict(calls[("combined", 2, ids[8])], started_at="another write")
    stored["write"](changed_calls=torn)
    assert rp.main() == 1
    out = capsys.readouterr().out
    assert "1 units hold records that do not belong together" in out and "unit integrity: recomputed 1, stored 0" in out
