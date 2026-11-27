"""Offline tests for conventions_test.py: the type table, prompts, marking, fresh entries and Part 2 prompts."""

from __future__ import annotations

import json
import re

import conventions_test as ct
import knowledge as kn
import pytest
import rag_tuning as rt
import rulebook


@pytest.fixture(scope="module")
def ctx():
    """Part 1's context without fresh entries (no downloaded file needed).

    Args: none.
    Returns: the mapping from `conventions_test.load`.
    Raises: as the loaders.
    Side Effects: reads local files.
    """
    return ct.load(fresh=False)


def test_overlay_renames_and_adds_and_refuses_a_wrong_base():
    """Two numbers renamed, three added, and a table that does not match the overlay is refused."""
    base = json.loads((rulebook.DATA / "constraint_types.json").read_text(encoding="utf-8"))["types"]
    overlay = json.loads(ct.OVERLAY.read_text(encoding="utf-8"))
    table = {row["number"]: row["name"] for row in ct.overlay_types(base, overlay)}
    assert (table[69], table[203]) == ("thickness", "min temperature")
    assert {table[63], table[86], table[93]} == {"volume", "composition", "purpose"}
    assert len(table) == len({row["number"] for row in base}) + 3
    wrong = [dict(row, name="other") if row["number"] == 69 else row for row in base]
    with pytest.raises(ValueError):
        ct.overlay_types(wrong, overlay)


def test_questions_and_head_use_the_completed_table(ctx):
    """The finder can now name the three added types, and asks for the longer answer."""
    names = {q["name"] for q in ctx["constraint_questions"]}
    assert {"volume", "composition", "purpose", "thickness", "min temperature"} <= names
    assert "layer" not in names
    head = ct.head(ctx)
    assert ct.INSTRUCTION in head and ct.INSTRUCTION_START not in head
    assert "volume:" in head and "thickness:" in head


def test_prompts_of_both_arms(ctx):
    """Only `rules` carries the conventions; examples are other folds only, in the new answer form."""
    target = ctx["ids"][0]
    plain, rules = (ct.prompt1(ctx, (arm, 1, target)) for arm in ct.ARMS)
    assert ct.CONVENTIONS_HEADING in rules and ct.CONVENTIONS_HEADING not in plain
    assert rules.replace(f"{ct.CONVENTIONS_HEADING}\n{ctx['conventions']}\n\n", "") == plain
    assert plain.endswith(f"Definition: {ctx['records'][target]['definition']}\nPatterns:")
    shown = ct.shown(ctx, target)
    assert len(shown) == 85 and all(ctx["fold_of"][d["id"]] != ctx["fold_of"][target] for d in shown)
    first = json.loads(shown[0]["patterns"])
    assert list(first)[:3] == ["object_of_interest", "matrix_entity", "restrictions"]
    with pytest.raises(KeyError):
        ct.build(ctx, "other", "x", shown)


def test_answer_of_reads_the_gold():
    """Entity texts, the matrix as null when empty, and constraint values as restrictions."""
    gold = {"hasStatisticalModifier": "", "hasProperty": "area fraction",
            "hasObjectOfInterest": {"AsymmetricSystem": "cloud / sky", "hasNumerator": "cloud", "hasDenominator": "sky"},
            "hasMatrix": "", "hasContextObject": "",
            "hasConstraint": [{"label": "condition: all cloud types", "on": "cloud"}, {"label": "dry", "on": "sky"}]}
    assert ct.entity_text(gold["hasObjectOfInterest"]) == "cloud / sky" and ct.entity_text("") is None
    ctx = ct.load(fresh=False)
    entry = ctx["key"][ctx["ids"][0]]
    answer = ct.answer_of(ctx, entry, gold)
    assert answer["matrix_entity"] is None and answer["restrictions"] == ["all cloud types", "dry"]


def test_outcome_marks_the_experts_own_answer_right(ctx):
    """The experts' tags as an answer are right; missing new keys only lose the form; junk is unparsed."""
    for v in ctx["ids"][:30]:
        answer = ctx["demos"][v]["patterns"]
        row = ct.outcome(ctx, ctx["key"][v], answer)
        assert row["parsed"] and row["form"] and row["matrix"]
        data = json.loads(answer)
        short = {key: data[key] for key in ("property", "matrix", "context", "modifier", "constraints")}
        row = ct.outcome(ctx, ctx["key"][v], json.dumps(short))
        assert row["parsed"] and not row["form"] and row["matrix"]
    assert not ct.outcome(ctx, ctx["key"][ctx["ids"][0]], "no JSON here")["parsed"]


def test_fresh_rows_read_definitions_and_flag_relatives(ctx, tmp_path):
    """A definition is read from a downloaded file; a missing or broken file leaves the entry out."""
    (tmp_path / "ttl").mkdir()
    near = ctx["records"][ctx["ids"][0]]["definition"]
    (tmp_path / "ttl" / "901.ttl").write_text(
        '@prefix iop: <https://w3id.org/iadopt/ont/> .\n@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n'
        f'<http://example.org/v> a iop:Variable ; rdfs:label "Invented" ; rdfs:comment """{near}""" .\n', encoding="utf-8")
    (tmp_path / "ttl" / "902.ttl").write_text("this is not turtle {", encoding="utf-8")
    for number, text in ((904, near + " zebra quokka"), (905, "xylophone quasar marmalade")):
        (tmp_path / "ttl" / f"{number}.ttl").write_text(
            '@prefix iop: <https://w3id.org/iadopt/ont/> .\n@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n'
            f'<http://example.org/v> a iop:Variable ; rdfs:comment """{text}""" .\n', encoding="utf-8")
    snapshot = {"issues": [{"number": n, "title": f"t{n}", "state": "open", "labels": ["101 property: nominal", "70 constraint: condition"]}
                           for n in (901, 902, 903, 904, 905)]}
    rows = ct.fresh_rows(ctx, snapshot, tmp_path)
    assert list(rows) == ["fresh:901", "fresh:904", "fresh:905"]
    assert rows["fresh:901"]["relative"] and rows["fresh:901"]["definition"] == " ".join(near.split())
    # Near but not the same words (two invented words added) is still a relative; unrelated words are not.
    assert rows["fresh:904"]["relative"] and not rows["fresh:905"]["relative"]
    assert rows["fresh:901"]["entry"]["constraint"] == [70]


def test_difference_and_gate():
    """The comparison's mean and a gate that needs both a reliable gain and the break-even level."""
    ids = [f"v{i}" for i in range(20)]
    row = ct.difference({v: 1.0 for v in ids}, {v: 0.5 for v in ids}, ids, draws=200)
    assert row["mean"] == row["low"] == row["high"] == 0.5
    summary = {"comparisons": {"matrix": {"low": 0.01}, "constraint": {"low": 0.02}},
               "arms": {"corpus": {"rules": {"matrix": 0.80, "constraint": 0.50}}}}
    assert ct.gate(summary) == ["matrix"]
    summary["comparisons"]["matrix"]["low"] = -0.01
    assert ct.gate(summary) == []


def test_conventions_examples_are_invented(ctx):
    """No example sentence of the conventions shares half of its words with a corpus definition."""
    corpus = [set(kn.words(ctx["records"][v]["definition"])) for v in ctx["ids"]]
    examples = re.findall(r"Example: ([^.]+)\.", ctx["conventions"])
    assert len(examples) >= 8
    for sentence in examples:
        mine = set(kn.words(sentence))
        assert all(len(mine & other) / len(mine | other) < 0.5 for other in corpus), sentence


@pytest.fixture(scope="module")
def world():
    """Stage 5's world with a word-overlap similarity table (no stored output needed).

    Args: none.
    Returns: (rag_tuning context, world).
    Raises: as the loaders.
    Side Effects: reads local files.
    """
    def overlap(first: str, second: str) -> float:
        """Shared words over all words.

        Args: first, second: texts.
        Returns: the share.
        Raises: nothing.
        Side Effects: none.
        """
        a, b = set(kn.words(first)), set(kn.words(second))
        return len(a & b) / len(a | b) if a | b else 0.0

    base = kn.load()
    ctx2 = rt.load(table=rt.similarity_table(base["ids"], lambda v: base["records"][v]["definition"], overlap))
    return ctx2, rt.corpus_world(ctx2, None)


def test_part2_prompts(ctx, world):
    """`rag-25` is stage 5's prompt; the guided arms add a statement, or nothing without usable facts."""
    _, w = world
    target = next(v for v in ctx["ids"] if ctx["key"][v]["matrix"])
    base = ct.prompt2(ctx, w, ("rag-25", 1, target), {}, [])
    assert base == rt.prompt_of(w, target, rt.examples(w, "rag-25", target))
    experts = ct.prompt2(ctx, w, ("rag-25+experts", 1, target), {}, [])
    assert experts != base and experts.startswith(base.split(w["definition"](target))[0])
    assert ct.prompt2(ctx, w, ("rag-25+rules", 1, target), {target: None}, ["matrix"]) == base
    facts = ct.hints.facts_from_key(ctx["key"][target])
    stated = ct.prompt2(ctx, w, ("rag-25+rules", 1, target), {target: facts}, ["matrix"])
    assert stated != base and stated != experts
    with pytest.raises(KeyError):
        ct.prompt2(ctx, w, ("other", 1, target), {}, [])
