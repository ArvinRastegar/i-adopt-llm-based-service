"""Offline tests for recompute_knowledge.py against knowledge.py: two readings of the same answers.

Invented answers for the real 102 variables and 16 cells are summarised by the one and checked by
the other; then faults are planted in the stored summary and in the stored calls, and the
recomputation has to name each one. No provider.
"""

from __future__ import annotations

import copy
import json
import random

import knowledge as kn
import pytest
import recompute_knowledge as recompute
import walk
from engine import ft

DRAWS = 300
RULES = {fold: f"MATRIX\n1. an invented rule of fold {fold}\nCONSTRAINTS\n1. another one" for fold in range(6)}


@pytest.fixture(scope="module")
def ctx():
    """Load the stage's context once per module.

    Args: none.
    Returns: the mapping from `knowledge.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files.
    """
    return kn.load()


@pytest.fixture(scope="module")
def invented(ctx):
    """Invented answers for every cell, variable and repetition, with the real prompt hashes.

    The answers are the experts' tags with mistakes thrown in: a matrix denied or made up, a
    constraint type dropped or added, another property pattern, an answer in a code fence, an
    answer that is no JSON. Cells with more kinds of knowledge make fewer mistakes.

    Args: ctx: from `load`.
    Returns: calls keyed like `engine.latest`; the cells with `I` carry invented rule texts, so
        no stored output is needed.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    rules = RULES
    rng = random.Random(11)
    names = [question["name"] for question in ctx["constraint_questions"]]
    calls = {}
    for cell in kn.cells():
        care = 0.25 - 0.04 * len(kn.factors_of(cell))
        hashes = {v: ft.sha256(kn.prompt_of(ctx, (cell, 1, v), rules)) for v in ctx["ids"]}
        for repetition in range(1, kn.REPETITIONS + 1):
            for variable_id in ctx["ids"]:
                answer = dict(walk.direct_answer(ctx, ctx["key"][variable_id]))
                if rng.random() < care:
                    answer["matrix"] = None if answer["matrix"] else rng.choice([40, 41, 42, 45])
                if rng.random() < care:
                    answer["constraints"] = (answer["constraints"][1:] if rng.random() < 0.5
                                             else [*answer["constraints"], rng.choice(names)])
                if rng.random() < care / 2:
                    answer["property"] = rng.choice(ctx["reachable"])
                if rng.random() < care / 3:
                    answer["context"] = not answer["context"]
                text = json.dumps(answer, indent=2)
                dice = rng.random()
                if dice < 0.03:
                    text = "```json\n" + text + "\n```"
                elif dice < 0.05:
                    text = "I am not sure."
                calls[(cell, repetition, variable_id)] = {
                    "arm": cell, "repetition": repetition, "variable_id": variable_id, "ok": True, "status": 200,
                    "error": None, "answer": text, "reasoning_chars": 0, "reasoning_tokens": 0,
                    "finish_reason": "stop", "prompt_tokens": 9000 + rng.randint(0, 900), "completion_tokens": 50,
                    "cost_usd": 0.0005,
                    "host": "Near AI", "served_model": "qwen/qwen3.8-27b", "prompt_sha256": hashes[variable_id]}
    return calls


@pytest.fixture()
def stored(ctx, invented, tmp_path, monkeypatch):
    """Write the invented run where the recomputation reads, with its summary.

    Args: ctx, invented: as above; tmp_path: a temporary folder; monkeypatch: to point the
        recomputation at it.
    Returns: {"calls", "summary", "rules": the stored rule rows, "write": writes changed calls,
        summary and rule rows back}.
    Raises: nothing.
    Side Effects: files in the temporary folder; the recomputation is given the fingerprints of
        the invented prompts and rule texts in place of the real ones.
    """
    summary = kn.summarise(ctx, invented, draws=DRAWS)
    ids = sorted(ctx["ids"])
    digests = {}
    for cell in kn.cells():
        hashes = [invented[(cell, 1, v)]["prompt_sha256"] for v in ids]
        digests[cell] = ft.sha256("\n".join(hashes))[:16]
    rule_rows = [{"fold": fold, "ok": True, "finish_reason": "stop", "rules": text, "reasoning_chars": 0,
                  "reasoning_tokens": 0, "host": "Near AI", "served_model": "qwen/qwen3.8-27b", "examples": 85,
                  "prompt_sha256": ft.sha256(kn.induction_prompt(ctx, kn.usable_demos(ctx, fold)))}
                 for fold, text in RULES.items()]
    monkeypatch.setattr(recompute, "PLAN_DIGESTS", digests)
    monkeypatch.setattr(recompute, "RULE_TEXT_DIGESTS", {fold: ft.sha256(text)[:16] for fold, text in RULES.items()})
    folder = tmp_path / "guided"
    folder.mkdir()
    members = {"examples": [], "practice": [], "held-out": []}
    for variable_id, group in ctx["group_of"].items():
        members[group].append(variable_id)
    for group, file in recompute.GROUP_FILES.items():
        (folder / file).write_text("".join(json.dumps({"variable_id": v}) + "\n" for v in members[group]))
    for name, value in (("CALLS", tmp_path / "calls.jsonl"), ("SUMMARY", tmp_path / "summary.json"),
                        ("RULES", tmp_path / "rules.jsonl"), ("GROUP_FOLDER", folder), ("DRAWS", DRAWS)):
        monkeypatch.setattr(recompute, name, value)

    def write(changed_calls=None, changed_summary=None, changed_rules=None):
        """Store calls, summary and rule rows, changed or as they are.

        Args: changed_calls: calls to store instead; changed_summary: summary to store instead;
            changed_rules: rule rows to store instead.
        Returns: nothing.
        Raises: nothing.
        Side Effects: writes three files.
        """
        rows = (changed_calls or invented).values()
        (tmp_path / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (tmp_path / "summary.json").write_text(json.dumps(changed_summary or summary))
        (tmp_path / "rules.jsonl").write_text("".join(
            json.dumps(row) + "\n" for row in (rule_rows if changed_rules is None else changed_rules)))

    write()
    return {"calls": invented, "summary": summary, "rules": rule_rows, "write": write}


def test_both_readings_mark_an_answer_alike(ctx):
    """Clean, fenced, odd and unreadable answers get the same marks from the two implementations."""
    book = recompute.Rulebook()
    tags = json.loads((recompute.DATA / "route_key.json").read_text())["variables"]
    names = [question["name"] for question in ctx["constraint_questions"]]
    rng = random.Random(3)
    checked = 0
    for variable_id in ctx["ids"]:
        right = walk.direct_answer(ctx, ctx["key"][variable_id])
        missing = {k: v for k, v in right.items() if k != "constraints"}
        variants = [right, dict(right, matrix=None), dict(right, matrix=45), dict(right, matrix="41"),
                    dict(right, matrix=True), dict(right, matrix=46),
                    dict(right, constraints=[*right["constraints"], rng.choice(names)]),
                    dict(right, constraints=[name.upper() for name in right["constraints"]]),
                    dict(right, constraints=[*right["constraints"], "no such type", " part"]),
                    dict(right, constraints=[*right["constraints"], *right["constraints"]]),
                    dict(right, constraints="none"), dict(right, constraints=[3]), dict(right, context="true"),
                    dict(right, modifier=1), dict(right, property=str(right["property"])),
                    dict(right, property=rng.choice(ctx["reachable"])), dict(right, property=41),
                    dict(right, property=None), dict(right, note="an extra key"), missing, {"matrix": 41},
                    {"property": right["property"]}]
        texts = [json.dumps(v) for v in variants]
        texts += ["```json\n" + texts[0] + "\n```", "", "no idea", "[1, 2]", json.dumps(right, indent=4)]
        for text in texts:
            mine = book.mark(text, tags[variable_id])
            theirs = kn.outcome(ctx, variable_id, text)
            assert {k: mine[k] for k in theirs} == theirs, (variable_id, text[:80])
            checked += 1
    assert checked == 102 * 27
    assert book.mark(None, tags[ctx["ids"][0]]) == {**kn.outcome(ctx, ctx["ids"][0], None)}


def test_the_folds_are_dealt_the_same_way_twice(ctx):
    """The recomputation's own deal of the folds is stage 0's, and so is its list of cells."""
    tags = json.loads((recompute.DATA / "route_key.json").read_text())["variables"]
    by_group = {group: sorted(v for v, g in ctx["group_of"].items() if g == group)
                for group in ("examples", "practice", "held-out")}
    assert recompute.deal_folds(by_group, tags) == ctx["fold_of"]
    assert tuple(recompute.cells()) == kn.cells() and set(recompute.PLAN_DIGESTS) == set(kn.cells())
    assert (recompute.DRAWS, recompute.SEED, recompute.STRICT) == (kn.DRAWS, kn.SEED, kn.STRICT)
    assert recompute.COLUMNS == kn.COLUMNS and recompute.PRIMARY == kn.PRIMARY


def test_the_recomputation_agrees_with_an_untouched_summary(stored, capsys):
    """Marked and summarised a second way, every figure matches."""
    assert recompute.main() == 0
    out = capsys.readouterr().out
    assert "all figures match" in out and "prompt digests match the record for 16 cells" in out
    cells = stored["summary"]["cells"]
    assert cells["G+D+S+I"]["matrix"] > cells["base"]["matrix"] and 0.0 < cells["base"]["parsed"] < 1.0


def test_the_recomputation_names_every_fault_planted_in_the_summary(stored, capsys):
    """A wrong share, interval, effect, choice or cost, a missing cell: each is reported."""
    summary = copy.deepcopy(stored["summary"])
    summary["cells"]["D"]["matrix"] += 0.01
    summary["cells"]["G"]["constraint_precision"] -= 0.02
    summary["cells"]["S"]["by_group"]["held-out"]["constraint"] += 0.05
    summary["cells"]["I"]["intervals"]["matrix"]["strict_low"] -= 0.01
    summary["cells"]["G+D"]["matrix_found_where_there_is_one"] += 0.02
    summary["cells"]["D+S"]["parsed"] = 1.0
    summary["cells"]["G+S"]["cost_usd"] += 0.3
    summary["effects"]["matrix"]["D"]["mean"] += 0.005
    summary["effects"]["constraint"]["S"]["strict_high"] += 0.01
    summary["effects"]["both"]["I"]["low"] -= 0.01
    summary["chosen"]["columns"]["matrix"]["share"] += 0.01
    summary["chosen"]["columns"]["constraint"]["against_base"]["low"] -= 0.02
    summary["chosen"]["by_fold"]["3"] = "base" if summary["chosen"]["by_fold"]["3"] != "base" else "G"
    summary["agreement"] -= 0.05
    summary["spent_usd"] += 1.0
    summary["cells"]["S+I"]["prompt_tokens_mean"] += 40
    summary["variables"] = 101
    stored["write"](changed_summary=summary)
    assert recompute.main() == 1
    out = capsys.readouterr().out
    for label in ("D matrix:", "G constraint_precision", "S held-out constraint", "I interval matrix strict_low",
                  "G+D matrix found where there is one", "D+S parsed", "G+S cost", "effect of D on matrix mean",
                  "effect of S on constraint strict_high", "effect of I on both low", "chosen matrix share",
                  "chosen constraint against base low", "chosen by fold", "agreement", "spent",
                  "S+I prompt tokens", "counts 101 variables"):
        assert any(line.startswith("MISMATCH") and label in line for line in out.splitlines()), label
    for change, label in ((lambda s: s.update(complete=False), "not of a complete run"),
                          (lambda s: s["cells"].pop("G+D+S+I"), "not the 16 of the plan"),
                          (lambda s: s["effects"]["matrix"].pop("G"), "not the four kinds of knowledge"),
                          (lambda s: s["effects"].pop("both"), "summary's effects are not"),
                          (lambda s: s["chosen"]["columns"].pop("both"), "chosen setting is not scored on")):
        other = copy.deepcopy(stored["summary"])
        change(other)
        stored["write"](changed_summary=other)
        assert recompute.main() == 1 and label in capsys.readouterr().out, label


def test_the_recomputation_names_every_fault_planted_in_the_calls(stored, capsys):
    """Another host, reasoning, a prompt that was not fingerprinted, a missing or failed call, a changed answer."""
    calls = stored["calls"]
    ids = sorted({key[2] for key in calls})
    elsewhere = dict(calls)
    elsewhere[("D", 2, ids[0])] = dict(calls[("D", 2, ids[0])], host="Wafer")
    elsewhere[("base", 1, ids[1])] = dict(calls[("base", 1, ids[1])], reasoning_tokens=12)
    elsewhere[("S", 1, ids[2])] = dict(calls[("S", 1, ids[2])], prompt_sha256="0" * 64)
    stored["write"](changed_calls=elsewhere)
    assert recompute.main() == 1
    out = capsys.readouterr().out
    for label in ("another host or model", "1 answers carry reasoning", "S: a variable was sent different prompts"):
        assert any(line.startswith("MISMATCH") and label in line for line in out.splitlines()), label
    other = {key: dict(row, prompt_sha256="1" * 64) if key[0] == "G+I" and key[2] == ids[5] else row
             for key, row in calls.items()}
    stored["write"](changed_calls=other)
    assert recompute.main() == 1 and "G+I: the prompts sent" in capsys.readouterr().out
    fewer = {key: row for key, row in calls.items() if key != ("I", 2, ids[4])}
    stored["write"](changed_calls=fewer)
    assert recompute.main() == 1 and "not 16 cells x 102 variables x 2 repetitions" in capsys.readouterr().out
    failed = dict(calls)
    failed[("D+I", 1, ids[6])] = dict(calls[("D+I", 1, ids[6])], ok=False, answer=None)
    stored["write"](changed_calls=failed)
    assert recompute.main() == 1 and "1 calls failed" in capsys.readouterr().out
    changed = dict(calls)
    for variable_id in ids[:30]:                      # answers swapped for wrong ones after the summary was made
        changed[("G+D", 1, variable_id)] = dict(calls[("G+D", 1, variable_id)], answer='{"property": 0}')
    stored["write"](changed_calls=changed)
    assert recompute.main() == 1 and "MISMATCH G+D matrix" in capsys.readouterr().out


def test_the_recomputation_names_every_fault_planted_in_the_rule_texts(stored, capsys):
    """Another host, a cut text, reasoning, a changed text, another prompt, a missing fold."""
    rows = stored["rules"]
    for change, label in (
            (lambda r: r[1].update(host="Wafer"), "rule text of fold 1: written by Wafer"),
            (lambda r: r[2].update(served_model="another/model"), "rule text of fold 2: written by"),
            (lambda r: r[3].update(rules=r[3]["rules"] + "\n2. one more"), "rule text of fold 3: not the fingerprinted text"),
            (lambda r: r[4].update(prompt_sha256="0" * 64), "rule text of fold 4: not written from the fingerprinted"),
            (lambda r: r[5].update(examples=102), "rule text of fold 5: written from 102 variables"),
            (lambda r: r[0].update(finish_reason="length"), "stored for folds [1, 2, 3, 4, 5]"),
            (lambda r: r[0].update(reasoning_tokens=25), "stored for folds [1, 2, 3, 4, 5]"),
            (lambda r: r.pop(2), "stored for folds [0, 1, 3, 4, 5]")):
        changed = copy.deepcopy(rows)
        change(changed)
        stored["write"](changed_rules=changed)
        assert recompute.main() == 1, label
        out = capsys.readouterr().out
        assert any(line.startswith("MISMATCH") and label in line for line in out.splitlines()), label
    stored["write"]()
    assert recompute.main() == 0 and "rule texts match the record for 6 folds" in capsys.readouterr().out
