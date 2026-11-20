"""Offline tests for knowledge.py: what each cell shows a target, the fold rule, and the marks. No provider."""

from __future__ import annotations

import json
import random
import re

import engine
import hints
import knowledge as kn
import pytest
import scoreboard
import settings
import walk
from engine import ft

RULES = {fold: f"MATRIX\n1. rule of fold {fold}\nCONSTRAINTS\n1. another rule of fold {fold}" for fold in range(6)}


@pytest.fixture(scope="module")
def ctx():
    """Load the stage's context once per module.

    Args: none.
    Returns: the mapping from `knowledge.load`.
    Raises: ValueError when the inputs fail their checks.
    Side Effects: reads local files; none of them is stored output.
    """
    return kn.load()


def examples(text: str) -> list[tuple[str, str | None, str]]:
    """The example variables listed in a block of a prompt, in order.

    One corpus definition runs over two lines, so the block is read example by example (they are
    separated by an empty line) and not line by line.

    Args: text: the part of a prompt that lists variables.
    Returns: (definition, decomposition or None, patterns) of each listed variable that has patterns.
    Raises: ValueError when a listed variable does not start with its definition.
    Side Effects: none.
    """
    out = []
    for chunk in text.split("\n\n"):
        match = re.fullmatch(r"(?s)Definition: (.*?)(?:\nDecomposition: ([^\n]*))?\nPatterns: (\{[^\n]*)", chunk)
        if match:
            out.append((match.group(1), match.group(2), match.group(3)))
        elif chunk.startswith("Definition: ") and not chunk.endswith("\nPatterns:"):
            raise ValueError(f"an example could not be read: {chunk[:80]!r}")
    return out


def shown(prompt: str) -> list[str]:
    """The definitions a prompt shows as examples, in order.

    Args: prompt: a classifier prompt.
    Returns: the example definitions; the target's own, which ends the prompt, is left out.
    Raises: ValueError when an example cannot be read.
    Side Effects: none.
    """
    return [definition for definition, _, _ in examples(prompt.split("\nEXAMPLES\n", 1)[1])]


def test_sixteen_cells_and_3264_calls(ctx):
    """Every combination of four kinds of knowledge, each kind on in eight; two repetitions."""
    names = kn.cells()
    assert len(names) == len(set(names)) == 16 and names[0] == "base" and names[-1] == "G+D+S+I"
    for factor in kn.FACTORS:
        assert sum(factor in kn.factors_of(cell) for cell in names) == 8
    assert kn.factors_of("base") == frozenset() and kn.factors_of("G+S") == {"G", "S"}
    with pytest.raises(KeyError):
        kn.factors_of("S+G")
    jobs = kn.schedule(ctx)
    assert len(jobs) == len(set(jobs)) == 16 * 102 * 2 == 3264
    assert {job[0] for job in jobs} == set(names) and {job[2] for job in jobs} == set(ctx["ids"])
    assert kn.FAMILY == 10 and kn.STRICT == pytest.approx(0.005)


def test_the_baseline_is_the_one_call_prompt_of_steps_4_and_5(ctx):
    """With the same examples, "base" is `settings.direct_prompt_all`, byte for byte."""
    for variable_id in (ctx["ids"][0], ctx["ids"][41], ctx["ids"][77]):
        demos = [ctx["demos"][v] for v in settings.shown_ids(ctx, variable_id)]
        definition = ctx["records"][variable_id]["definition"]
        assert kn.build(ctx, "base", definition, demos) == settings.direct_prompt_all(ctx, variable_id)


def test_a_target_sees_the_other_five_folds_and_nothing_of_its_own(ctx):
    """In all 16 cells the examples are the 85 variables of the other folds; never the target or its fold."""
    definition_of = {v: ctx["records"][v]["definition"] for v in ctx["ids"]}
    assert len(set(definition_of.values())) == 102 and not any("\n\n" in d for d in definition_of.values())
    for variable_id in ctx["ids"]:
        fold = ctx["fold_of"][variable_id]
        allowed = {definition_of[v] for v in ctx["ids"] if ctx["fold_of"][v] != fold}
        own_fold = {definition_of[v] for v in ctx["ids"] if ctx["fold_of"][v] == fold}
        gold = ctx["demos"][variable_id]["gold"]
        for cell in kn.cells():
            prompt = kn.prompt_of(ctx, (cell, 1, variable_id), RULES)
            examples = shown(prompt)
            assert len(examples) == 85 and set(examples) == allowed and not set(examples) & own_fold
            assert prompt.endswith(f"Definition: {definition_of[variable_id]}\nPatterns:")
            assert prompt.count(f"Definition: {definition_of[variable_id]}\n") == 1
            assert prompt.count("\nPatterns: {") == 85
            assert prompt.count("\nDecomposition: ") == (85 if "D" in kn.factors_of(cell) else 0)
            assert f"Decomposition: {gold}" not in prompt
            if "I" in kn.factors_of(cell):
                assert f"rule of fold {fold}" in prompt
                assert not any(f"rule of fold {other}" in prompt for other in range(6) if other != fold)


def test_the_decompositions_shown_are_the_experts_own(ctx):
    """A `D` example carries the gold decomposition of that example and its tags, nothing else."""
    variable_id = ctx["ids"][5]
    prompt = kn.prompt_of(ctx, ("D", 1, variable_id), {})
    triples = examples(prompt.split("\nEXAMPLES\n", 1)[1])
    assert len(triples) == 85 and all(decomposition for _, decomposition, _ in triples)
    by_definition = {ctx["records"][v]["definition"]: v for v in ctx["ids"]}
    for definition, decomposition, patterns in triples:
        example = by_definition[definition]
        assert json.loads(decomposition) == json.loads(ft.answer_text(ctx["records"][example]["gold"]))
        assert json.loads(patterns) == walk.direct_answer(ctx, ctx["key"][example])
    assert "\n" not in ctx["demos"][variable_id]["gold"]
    assert list(json.loads(ctx["demos"][variable_id]["gold"])) == list(ft.TOP_KEYS)


def test_the_nearest_examples_come_last(ctx):
    """`S` keeps the same examples and orders them by rising similarity; without it they are in id order."""
    variable_id = ctx["ids"][9]
    plain = shown(kn.prompt_of(ctx, ("base", 1, variable_id), {}))
    near = shown(kn.prompt_of(ctx, ("S", 1, variable_id), {}))
    assert sorted(plain) == sorted(near) and plain != near
    usable = kn.usable_demos(ctx, ctx["fold_of"][variable_id])
    assert plain == [d["definition"] for d in sorted(usable, key=lambda d: d["id"])]
    demos = [{"id": "c", "definition": "depth of snow on a glacier", "gold": "{}", "patterns": "{}"},
             {"id": "a", "definition": "colour of a tulip", "gold": "{}", "patterns": "{}"},
             {"id": "b", "definition": "colour of a tulip", "gold": "{}", "patterns": "{}"},
             {"id": "d", "definition": "mass of sea ice in the Arctic ocean", "gold": "{}", "patterns": "{}"},
             {"id": "e", "definition": "", "gold": "{}", "patterns": "{}"}]
    ordered = [d["id"] for d in kn.nearest_last("Mass of sea ice", demos)]
    assert ordered[-1] == "d" and ordered.index("a") < ordered.index("b")     # equal ones in id order
    assert [d["id"] for d in kn.nearest_last("colour of a tulip", demos)][-2:] == ["a", "b"]
    assert kn.nearest_last("", demos) == sorted(demos, key=lambda d: d["id"])  # nothing to compare: id order
    assert kn.words("Sea-surface  temperature (SST), 2 m") == ["sea", "surface", "temperature", "sst", "2", "m"]


def test_the_experts_procedure_is_their_tree_in_order(ctx):
    """`G` states the five questions of the matrix tree, then ours, then the guide's sentences, once."""
    assert kn.matrix_steps(ctx) == [
        ("Is it involved in a material flow?", 45), ("Is it involved in a natural process?", 43),
        ("Does it require a reference area?", 44), ("Is it embedded within a different material?", 40),
        ("Is it within a natural container?", 41), (walk.OUR_MATRIX_QUESTION, 42)]
    lines = kn.guide(ctx)
    assert lines[0] == kn.GUIDE_HEADING and lines[2].startswith("1. Is it involved in a material flow? Then")
    assert lines[7] == "6. Is it within an artificial container? Then the matrix is pattern 42 (artificial container)."
    assert lines[8] == kn.GUIDE_NO_MATRIX and tuple(lines[9:]) == kn.GUIDE_SENTENCES
    assert not any("example" in line.lower() or "http" in line for line in lines)
    variable_id = ctx["ids"][3]
    prompt = kn.prompt_of(ctx, ("G+D", 1, variable_id), {})
    assert prompt.count(kn.GUIDE_HEADING) == 1 and prompt.index(kn.GUIDE_HEADING) < prompt.index("\nEXAMPLES\n")
    assert prompt.index("\nCONSTRAINT TYPES\n") < prompt.index(kn.GUIDE_HEADING)
    assert kn.GUIDE_HEADING not in kn.prompt_of(ctx, ("D+S", 1, variable_id), {})
    broken = dict(ctx, matrix_tree=dict(ctx["matrix_tree"], edges=ctx["matrix_tree"]["edges"][:-1]))
    with pytest.raises(ValueError):
        kn.matrix_steps(broken)


def test_the_rules_stand_between_the_lists_and_the_examples_and_are_needed(ctx):
    """An `I` cell carries the rules of the target's fold under their heading; without rules it refuses."""
    variable_id = ctx["ids"][3]
    prompt = kn.prompt_of(ctx, ("G+I", 1, variable_id), RULES)
    assert prompt.count(kn.RULES_HEADING) == 1
    assert prompt.index(kn.GUIDE_HEADING) < prompt.index(kn.RULES_HEADING) < prompt.index("\nEXAMPLES\n")
    with pytest.raises(ValueError):
        kn.prompt_of(ctx, ("I", 1, variable_id), {})
    with pytest.raises(ValueError):
        kn.prompt_of(ctx, ("I", 1, variable_id), {ctx["fold_of"][variable_id]: "  \n"})
    assert kn.RULES_HEADING not in kn.prompt_of(ctx, ("G", 1, variable_id), RULES)


def test_the_rules_are_written_from_the_other_folds_only(ctx):
    """A fold's induction prompt lists the 85 variables of the other folds with decomposition and tags."""
    definition_of = {v: ctx["records"][v]["definition"] for v in ctx["ids"]}
    prompts = set()
    for fold in range(scoreboard.FOLDS):
        prompt = kn.induction_prompt(ctx, kn.usable_demos(ctx, fold))
        prompts.add(prompt)
        listed = [definition for definition, _, _ in examples(prompt.split("\nVARIABLES\n", 1)[1])]
        assert len(listed) == 85 and set(listed) == {definition_of[v] for v in ctx["ids"] if ctx["fold_of"][v] != fold}
        assert prompt.count("\nDecomposition: ") == 85 and prompt.count("\nPatterns: {") == 85
        assert prompt.startswith(kn.INDUCE_INTRO) and prompt.endswith("\nCONVENTIONS")
        assert "\nMATRIX PATTERNS\n40 " in prompt and "\nCONSTRAINT TYPES\n" in prompt
        assert "PROPERTY PATTERNS" not in prompt
    assert len(prompts) == 6


def test_the_invented_decompositions_are_valid_and_fit_their_patterns(ctx):
    """Each invented variable's decomposition passes the schema and has the form its patterns demand."""
    from iadopt_lab.validation import validate_prediction

    items = kn.invented(ctx)
    assert len(items) == 16 == len(kn.INVENTED_GOLD)
    corpus = {record["definition"].lower() for record in ctx["records"].values()}
    for name, item in items.items():
        decomposition = json.loads(item["demo"]["gold"])
        assert validate_prediction(decomposition, ctx["inputs"]["schema"]).valid, name
        form = hints.expected_form(hints.facts_from_key(item["entry"]), ctx["patterns"], ctx["types"])
        assert all(hints.conformance(decomposition, form).values()), name
        assert item["definition"].lower() not in corpus
        assert json.loads(item["demo"]["patterns"]) == walk.direct_answer(ctx, item["entry"])


def test_an_answer_is_marked_against_the_experts_tags(ctx):
    """The tags themselves are right in everything; one changed decision is wrong there only."""
    with_matrix = next(v for v in ctx["ids"] if ctx["key"][v]["matrix"] and ctx["key"][v]["constraint"])
    right = walk.direct_answer(ctx, ctx["key"][with_matrix])
    row = kn.outcome(ctx, with_matrix, json.dumps(right))
    assert all(row[column] for column in kn.COLUMNS) and row["parsed"] and row["matrix_stated"]
    assert (row["fp"], row["fn"]) == (0, 0) and row["tp"] == len(right["constraints"])
    denied = kn.outcome(ctx, with_matrix, json.dumps(dict(right, matrix=None)))
    assert not denied["matrix"] and not denied["both"] and not denied["complete"] and not denied["matrix_stated"]
    assert denied["constraint"] and denied["property"] and denied["context"]
    extra = kn.outcome(ctx, with_matrix, json.dumps(dict(right, constraints=[*right["constraints"], "no such type"])))
    assert not extra["constraint"] and extra["fp"] == 1 and not extra["both"] and not extra["complete"]
    assert extra["matrix"] and extra["property"] and extra["parsed"]   # an invented type spoils the types only
    shouted = kn.outcome(ctx, with_matrix, json.dumps(dict(right, constraints=[n.upper() for n in right["constraints"]])))
    assert shouted["complete"]                                          # a listed name in capitals is that name
    fewer = kn.outcome(ctx, with_matrix, json.dumps(dict(right, constraints=right["constraints"][1:])))
    assert not fewer["constraint"] and fewer["fn"] == 1 and fewer["matrix"]
    without = dict(right)
    del without["matrix"]
    for bad in (None, "", "I cannot say", json.dumps({"property": "x"}), json.dumps(without),
                json.dumps(dict(right, matrix="41")), json.dumps(dict(right, matrix=7)),
                json.dumps(dict(right, matrix=True)), json.dumps(dict(right, context="false")),
                json.dumps(dict(right, modifier=0)), json.dumps(dict(right, constraints="none")),
                json.dumps(dict(right, constraints=[1])), json.dumps([right])):
        row = kn.outcome(ctx, with_matrix, bad)
        assert not row["parsed"] and not any(row[column] for column in kn.COLUMNS), bad
        assert (row["tp"], row["fp"]) == (0, 0) and row["fn"] == len(right["constraints"])
    fenced = kn.outcome(ctx, with_matrix, "```json\n" + json.dumps(right) + "\n```")
    assert fenced["complete"]


def test_a_bare_answer_is_not_right_by_default(ctx):
    """For a variable without matrix or constraints, an answer that names only the property is wrong."""
    plain = next(v for v in ctx["ids"] if not ctx["key"][v]["matrix"] and not ctx["key"][v]["constraint"]
                 and not ctx["key"][v]["context"] and not ctx["key"][v]["statistical_modifier"])
    right = walk.direct_answer(ctx, ctx["key"][plain])
    assert kn.outcome(ctx, plain, json.dumps(right))["matrix"]
    bare = kn.outcome(ctx, plain, json.dumps({"property": right["property"]}))
    assert not bare["parsed"] and not bare["matrix"] and not bare["constraint"] and not bare["complete"]


def answers(ctx, rule):
    """Stored calls in which every cell answers as a rule says.

    Args: ctx: from `load`; rule: (cell, repetition, variable_id, the right answer) -> the answer.
    Returns: calls keyed like `engine.latest`.
    Raises: nothing.
    Side Effects: none.
    """
    calls = {}
    for cell in kn.cells():
        for repetition in range(1, kn.REPETITIONS + 1):
            for variable_id in ctx["ids"]:
                right = walk.direct_answer(ctx, ctx["key"][variable_id])
                calls[(cell, repetition, variable_id)] = {
                    "ok": True, "answer": json.dumps(rule(cell, repetition, variable_id, right)),
                    "cost_usd": 0.0005, "prompt_tokens": 9000}
    return calls


def test_the_summary_reads_what_each_kind_of_knowledge_changes(ctx):
    """With invented answers the cells' shares, the four effects and the chosen setting are as built."""
    def rule(cell, repetition, variable_id, right):
        """Cells with D get the matrix right; the others deny it. Cells with G lose the constraints.

        Args: cell, repetition, variable_id: the call; right: the experts' tags as an answer.
        Returns: the answer.
        Raises: nothing.
        Side Effects: none.
        """
        answer = dict(right)
        if "D" not in kn.factors_of(cell):
            answer["matrix"] = None
        if "G" in kn.factors_of(cell):
            answer["constraints"] = []
        return answer

    calls = answers(ctx, rule)
    summary = kn.summarise(ctx, calls, draws=200)
    assert summary["complete"] is True and summary["repetitions_used"] == [1, 2] and summary["agreement"] == 1.0
    assert summary["spent_usd"] == pytest.approx(3264 * 0.0005)
    # Three variables carry a tagged constraint type that the answer format cannot name, so even
    # the experts' own tags, sent back as an answer, are right in the constraints for 99 of 102.
    sent_back = [kn.outcome(ctx, v, json.dumps(walk.direct_answer(ctx, ctx["key"][v]))) for v in ctx["ids"]]
    reachable = sum(row["constraint"] for row in sent_back)
    assert reachable == 99
    # One more variable's tagged object shape is not the shape its property pattern implies.
    assert sum(row["object"] for row in sent_back) == 101 and sum(row["complete"] for row in sent_back) == 98
    assert all(row["matrix"] and row["property"] and row["context"] and row["modifier"] for row in sent_back)
    assert summary["cells"]["D"]["matrix"] == 1.0 and summary["cells"]["base"]["matrix"] == pytest.approx(0.5)
    assert summary["cells"]["base"]["matrix_found_where_there_is_one"] == 0.0
    assert summary["cells"]["base"]["no_matrix_where_there_is_none"] == 1.0
    assert summary["cells"]["G"]["constraint"] == pytest.approx(16 / 102)   # right only where there are none
    assert summary["cells"]["D"]["constraint"] == pytest.approx(99 / 102)
    # The three tag names the format cannot name count, sent back, as invented types.
    named_right, invented_types = sum(row["tp"] for row in sent_back), sum(row["fp"] for row in sent_back)
    assert invented_types == 3
    assert summary["cells"]["D"]["constraint_precision"] == pytest.approx(named_right / (named_right + 3))
    assert summary["cells"]["D"]["constraint_recall"] < 1.0
    assert summary["cells"]["G+D"]["both"] == pytest.approx(16 / 102)
    assert summary["cells"]["D"]["complete"] == pytest.approx(98 / 102)
    effects = summary["effects"]
    assert effects["matrix"]["D"]["mean"] == pytest.approx(0.5) and effects["matrix"]["G"]["mean"] == 0.0
    assert effects["constraint"]["G"]["mean"] == pytest.approx(-83 / 102)
    assert effects["matrix"]["S"]["mean"] == 0.0 == effects["constraint"]["I"]["mean"]
    row = effects["matrix"]["D"]
    assert row["strict_low"] <= row["low"] <= row["mean"] <= row["high"] <= row["strict_high"]
    assert set(summary["chosen"]["by_fold"].values()) == {"D"}             # the first of the best cells
    picked = summary["chosen"]["columns"]
    assert picked["matrix"]["share"] == 1.0 and picked["matrix"]["against_base"]["mean"] == pytest.approx(0.5)
    assert picked["constraint"]["against_base"]["mean"] == 0.0


def test_a_fold_has_no_say_in_its_own_choice(ctx):
    """A cell that wins only when a fold's own variables are counted is not chosen for that fold."""
    fold_of = ctx["fold_of"]
    outside = [v for v in ctx["ids"] if fold_of[v] != 0]

    def good(cell, variable_id):
        """`S` is right on 50 variables outside fold 0; `I` on 45 of them and on all of fold 0.

        Args: cell: a cell name; variable_id: the variable.
        Returns: whether the cell answers this variable rightly.
        Raises: nothing.
        Side Effects: none.
        """
        if cell == "S":
            return variable_id in outside[:50]
        if cell == "I":
            return variable_id in outside[:45] or fold_of[variable_id] == 0
        return False

    calls = answers(ctx, lambda cell, repetition, variable_id, right: right if good(cell, variable_id)
                    else {"property": 0})
    choice = kn.chosen(ctx, kn.results(ctx, calls), [1, 2])
    assert choice[0] == "S"                    # 50 of 85 beats 45 of 85; with fold 0 counted `I` would win
    assert all(choice[fold] == "I" for fold in range(1, 6))
    summary = kn.summarise(ctx, calls, draws=100)
    assert summary["chosen"]["by_fold"] == {"0": "S", "1": "I", "2": "I", "3": "I", "4": "I", "5": "I"}
    right = sum(good(choice[fold_of[v]], v) for v in ctx["ids"])
    assert right == 45 and summary["chosen"]["columns"]["matrix"]["share"] == pytest.approx(45 / 102)


def test_equal_cells_stay_equal_however_their_right_answers_split(ctx):
    """Two cells with the same number of right answers tie, and the first of them is chosen."""
    outside = [v for v in ctx["ids"] if ctx["fold_of"][v] != 0]
    assert len(outside) == 85

    def rule(cell, repetition, variable_id, right):
        """`D` has 1 matrix and 10 constraint sets right outside fold 0, `S` none and 11; others none.

        Args: cell, repetition, variable_id: the call; right: the experts' tags as an answer.
        Returns: the answer.
        Raises: nothing.
        Side Effects: none.
        """
        wrong_matrix = dict(right, matrix=45 if right["matrix"] != 45 else None)
        wrong_types = dict(right, constraints=["no such type"])
        wrong_both = dict(wrong_matrix, constraints=["no such type"])
        if cell not in ("D", "S") or variable_id not in outside:
            return {"property": 0}
        position = outside.index(variable_id)
        matrix_ok = position < (1 if cell == "D" else 0)
        types_ok = 1 <= position < (11 if cell == "D" else 12)
        return right if matrix_ok and types_ok else wrong_types if matrix_ok else wrong_matrix if types_ok else wrong_both

    marked = kn.results(ctx, answers(ctx, rule))
    for cell, (matrices, types) in (("D", (1, 10)), ("S", (0, 11))):
        assert sum(marked[(cell, 1, v)]["matrix"] for v in outside) == matrices
        assert sum(marked[(cell, 1, v)]["constraint"] for v in outside) == types
    assert 1 / 85 + 10 / 85 < 0 / 85 + 11 / 85                 # as rounded shares the later cell would win
    assert kn.chosen(ctx, marked, [1, 2])[0] == "D"            # as counts they are equal: the first cell wins


def test_the_order_of_the_examples_depends_on_the_definitions_only(ctx):
    """Swapping the decompositions and patterns among the examples leaves their order unchanged."""
    variable_id = ctx["ids"][20]
    usable = kn.usable_demos(ctx, ctx["fold_of"][variable_id])
    swapped = [dict(d, gold=other["gold"], patterns=other["patterns"])
               for d, other in zip(usable, reversed(usable), strict=True)]
    definition = ctx["records"][variable_id]["definition"]
    assert ([d["id"] for d in kn.nearest_last(definition, usable)]
            == [d["id"] for d in kn.nearest_last(definition, swapped)])


def test_the_two_intervals_hold_the_shares_of_the_draws_they_claim():
    """Of the resampled means, about 95% lie inside the plain interval and 99.5% inside the strict one."""
    rng = random.Random(8)
    figures = {f"v{n}": rng.random() for n in range(102)}
    ids = sorted(figures)
    out = kn.resampled(figures, ids, draws=10000, seed=5)
    again = random.Random(5)
    values = [figures[v] for v in ids]
    drawn = [sum(again.choices(values, k=len(values))) / len(values) for _ in range(10000)]
    inside = sum(out["low"] <= d <= out["high"] for d in drawn) / len(drawn)
    strict = sum(out["strict_low"] <= d <= out["strict_high"] for d in drawn) / len(drawn)
    assert 0.9495 <= inside <= 0.9505 and 0.9945 <= strict <= 0.9955
    below, above = sum(d < out["low"] for d in drawn), sum(d > out["high"] for d in drawn)
    assert abs(below - above) <= 2 and abs(sum(d < out["strict_low"] for d in drawn)
                                           - sum(d > out["strict_high"] for d in drawn)) <= 2


def test_a_rule_text_that_was_cut_or_carries_reasoning_is_not_used(tmp_path):
    """Only a text that ended normally, without reasoning, counts as a fold's rules."""
    rows = [{"fold": 0, "ok": True, "finish_reason": "length", "rules": "MATRIX\n1. cut", "reasoning_chars": 0},
            {"fold": 0, "ok": True, "finish_reason": "stop", "rules": "MATRIX\n1. whole", "reasoning_chars": 0},
            {"fold": 0, "ok": True, "finish_reason": "stop", "rules": "MATRIX\n1. later", "reasoning_chars": 0},
            {"fold": 1, "ok": True, "finish_reason": "stop", "rules": "MATRIX\n1. thought", "reasoning_tokens": 30},
            {"fold": 2, "ok": False, "finish_reason": None, "rules": None},
            {"fold": 3, "ok": True, "finish_reason": "stop", "rules": "   "}]
    path = tmp_path / "rules.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert kn.stored_rules(path) == {0: "MATRIX\n1. whole"}
    assert kn.stored_rules(tmp_path / "none.jsonl") == {}


def test_the_command_line_refuses_to_run_without_rules_and_reports_an_unfinished_run(ctx, tmp_path, monkeypatch):
    """`--run` needs the six rule texts; `--score` exits with 1 for an unfinished run and 0 for a finished one."""
    monkeypatch.setattr(kn, "load", lambda: ctx)
    monkeypatch.setattr(kn, "RULES", tmp_path / "rules.jsonl")
    monkeypatch.setattr(kn, "stored_rules", lambda path=tmp_path / "rules.jsonl": {})
    with pytest.raises(SystemExit, match="run --induce first"):
        kn.main(["--run"])
    monkeypatch.setattr(kn, "CALLS", tmp_path / "calls.jsonl")
    monkeypatch.setattr(kn, "SUMMARY", tmp_path / "summary.json")
    monkeypatch.setattr(kn, "DRAWS", 50)
    calls = answers(ctx, lambda cell, repetition, variable_id, right: right)

    def store(kept):
        """Write calls the way the runner stores them.

        Args: kept: {(cell, repetition, variable_id): record}.
        Returns: nothing.
        Raises: nothing.
        Side Effects: writes the calls file.
        """
        (tmp_path / "calls.jsonl").write_text("".join(
            json.dumps(dict(record, arm=key[0], repetition=key[1], variable_id=key[2])) + "\n"
            for key, record in kept.items()))

    store({key: record for key, record in calls.items() if key != ("D", 2, ctx["ids"][0])})
    assert kn.main(["--score"]) == 1
    assert json.loads((tmp_path / "summary.json").read_text())["complete"] is False
    store(calls)
    assert kn.main(["--score"]) == 0
    assert json.loads((tmp_path / "summary.json").read_text())["complete"] is True


def test_the_rules_are_booked_under_the_stage_and_the_dev_check_apart(ctx, monkeypatch):
    """`--induce` spends from the stage's own allowance, `--dev` from a smaller one of its own."""
    import orclient

    seen = []

    class Spend:
        """Stands in for the cap and keeps how it was set up."""

        spent = 0.0

        def __init__(self, purpose=None, allowance=None):
            """Keep the purpose and the allowance.

            Args: purpose: the ledger label; allowance: its total.
            Returns: nothing.
            Raises: nothing.
            Side Effects: appends to `seen`.
            """
            seen.append(("spend", purpose, allowance))

    async def no_work(*args, **kwargs):
        """Do nothing in place of a provider step.

        Args: args, kwargs: ignored.
        Returns: nothing to report.
        Raises: nothing.
        Side Effects: none.
        """
        return {}

    monkeypatch.setattr(kn, "load", lambda: ctx)
    monkeypatch.setattr(orclient, "Spend", Spend)
    monkeypatch.setattr(orclient, "Guard", lambda spend: type("G", (), {"reason": None})())
    monkeypatch.setattr(orclient, "open_transport",
                        lambda spend, purpose, concurrency=0: (seen.append(("transport", purpose)) or (None, None)))
    monkeypatch.setattr(orclient, "close_transport", no_work)
    monkeypatch.setattr(kn, "induce", no_work)
    monkeypatch.setattr(kn, "dev", lambda *args: no_work() and [])
    assert kn.main(["--induce"]) == 0
    assert seen == [("spend", "knowledge", kn.ALLOWANCE_USD), ("transport", "knowledge")] and kn.ALLOWANCE_USD == 5.0
    seen.clear()

    async def no_rows(*args):
        """Return no dev rows.

        Args: args: ignored.
        Returns: an empty list.
        Raises: nothing.
        Side Effects: none.
        """
        return []

    monkeypatch.setattr(kn, "dev", no_rows)
    assert kn.main(["--dev"]) == 0
    assert seen == [("spend", "knowledge-dev", kn.DEV_ALLOWANCE_USD), ("transport", "knowledge-dev")]


def test_an_unfinished_repetition_enters_no_figure(ctx):
    """A failed call takes its repetition out, and the summary says it is not complete."""
    calls = answers(ctx, lambda cell, repetition, variable_id, right: right)
    key = ("G+S", 2, ctx["ids"][7])
    broken = dict(calls)
    broken[key] = {"ok": False, "answer": None, "cost_usd": None}
    summary = kn.summarise(ctx, broken, draws=50)
    assert summary["complete"] is False and summary["repetitions_used"] == [1] and summary["agreement"] is None
    assert summary["cells"]["G+S"]["matrix"] == 1.0
    nothing = kn.summarise(ctx, {k: v for k, v in calls.items() if k[2] != ctx["ids"][0]}, draws=50)
    assert nothing["complete"] is False and nothing["repetitions_used"] == [] and nothing["cells"] == {}


def test_the_resampled_mean_and_its_two_intervals():
    """A per-variable figure's mean with a 95% interval and the wider one for ten comparisons."""
    figures = {f"v{n}": float(n % 2) for n in range(100)}
    out = kn.resampled(figures, list(figures), draws=2000, seed=4)
    assert out["mean"] == pytest.approx(0.5) and 0.38 < out["low"] < 0.45 and 0.55 < out["high"] < 0.62
    assert out["strict_low"] < out["low"] and out["strict_high"] > out["high"]
    assert kn.resampled(figures, list(figures), draws=2000, seed=4) == out
    same = kn.resampled({"a": 0.25, "b": 0.25}, ["a", "b"], draws=100)
    assert same == {"mean": 0.25, "low": 0.25, "high": 0.25, "strict_low": 0.25, "strict_high": 0.25}


async def test_the_rules_are_asked_for_once_per_fold_and_kept(ctx, tmp_path):
    """Induction stores one text per fold, skips folds that have one, and keeps no answer of another host."""
    seen = []

    async def transport(body):
        """Answer with rules, the second time from another host.

        Args: body: the request body.
        Returns: a successful result in the shape of `orclient.read`.
        Raises: nothing.
        Side Effects: appends to `seen`.
        """
        seen.append(body)
        return {"ok": True, "status": 200, "error": None, "answer": f"MATRIX\n1. rule {len(seen)}", "reasoning": "",
                "reasoning_tokens": 0, "finish_reason": "stop", "prompt_tokens": 9000, "completion_tokens": 80,
                "cost_usd": 0.0005, "host": "Wafer" if len(seen) == 2 else "Near AI", "served_model": "x",
                "latency": 0.1, "attempts": 1}

    import orclient

    folds = {fold: kn.usable_demos(ctx, fold) for fold in range(3)}
    path = tmp_path / "rules.jsonl"
    guard = orclient.Guard(orclient.Spend(ledger=tmp_path / "ledger.jsonl"))
    first = await kn.induce(ctx, folds, transport, guard, path)
    assert first == {0: "MATRIX\n1. rule 1"} and "not the pinned host" in guard.reason and len(seen) == 2
    again = orclient.Guard(orclient.Spend(ledger=tmp_path / "ledger.jsonl"))
    second = await kn.induce(ctx, folds, transport, again, path)
    assert sorted(second) == [0, 1, 2] and second[0] == "MATRIX\n1. rule 1" and len(seen) == 4
    assert seen[0]["max_tokens"] == 900 and seen[0]["temperature"] == 0.0
    assert seen[0]["messages"][0]["content"] == kn.induction_prompt(ctx, folds[0])
    rows = list(engine.ft.read_jsonl(path))
    assert [row["fold"] for row in rows] == [0, 1, 1, 2] and [row["ok"] for row in rows] == [True, False, True, True]
