"""The baseline reranker must keep the legacy pipeline's prompt, character for character."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from iadopt_linking import rerank

LEGACY = (Path(__file__).resolve().parents[2]
          / "misc/benchmarking_example/phaseOneThreeMerged.py")


def legacy_namespace() -> dict:
    """Execute only the prompt constants and the two formatting functions of the legacy script."""
    source = LEGACY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted_names = {"QWEN3_RERANK_PREFIX", "QWEN3_RERANK_SUFFIX", "DEFAULT_RERANK_TASK"}
    keep = [node for node in tree.body
            if (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in wanted_names for t in node.targets))
            or (isinstance(node, ast.FunctionDef) and node.name in {"format_queries", "format_document"})]
    assert len(keep) == 5, "legacy script no longer defines the five prompt pieces"
    namespace: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(LEGACY), "exec"), namespace)
    return namespace


def test_prompt_pieces_equal_the_legacy_script() -> None:
    legacy = legacy_namespace()
    assert rerank.QWEN3_RERANK_PREFIX == legacy["QWEN3_RERANK_PREFIX"]
    assert rerank.QWEN3_RERANK_SUFFIX == legacy["QWEN3_RERANK_SUFFIX"]
    assert rerank.DEFAULT_RERANK_TASK == legacy["DEFAULT_RERANK_TASK"]


@pytest.mark.parametrize("text", ["blood", 'Definition of "x" in context: "y"', "line\nbreak", ""])
def test_formatting_equals_the_legacy_functions(text: str) -> None:
    legacy = legacy_namespace()
    assert rerank.format_query(text) == legacy["format_queries"](text, legacy["DEFAULT_RERANK_TASK"])
    assert rerank.format_document(text) == legacy["format_document"](text)


def test_query_and_document_templates_are_still_the_legacy_ones() -> None:
    source = LEGACY.read_text(encoding="utf-8")
    assert """query = f'Definition of "{term}" in context: "{context}"'""" in source
    assert """f'label: "{s.get("label","")}", description: "{s.get("description","")}"'""" in source
    assert rerank.build_query("blood", "a fluid") == 'Definition of "blood" in context: "a fluid"'
    assert rerank.build_document("blood", "fluid", None) == 'label: "blood", description: "fluid"'


def test_description_cap_cuts_at_a_word_and_leaves_short_text_alone() -> None:
    long_text = "word " * 100
    cut = rerank.truncate_description(long_text, 300)
    assert len(cut) <= 301 and cut.endswith("…") and "  " not in cut
    assert rerank.truncate_description("short  text", 300) == "short text"
    assert rerank.truncate_description(long_text, None) == long_text.strip()
    assert rerank.truncate_description("", 300) == ""


@pytest.mark.parametrize("scores", [[0.5], [0.5, float("nan")], [0.5, 1.5], [-0.1, 0.2]])
def test_impossible_scores_are_refused(scores: list[float]) -> None:
    with pytest.raises(ValueError):
        rerank.validate_scores(scores, 2)


def test_valid_scores_pass_through_as_floats() -> None:
    assert rerank.validate_scores([0, 1, 0.25], 3) == [0.0, 1.0, 0.25]
