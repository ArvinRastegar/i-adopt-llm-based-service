"""The report's voice: one renderer names two scorers, and Jev's published files do not change."""

from __future__ import annotations

from pathlib import Path

import pytest

from iadopt_linking.chat_experiment import voice_for
from iadopt_linking.jev_report import (
    JEV_VOICE,
    _largest_difference,
    confidence_columns,
    outcome_columns,
    signal_labels,
)

JEV_RUN = Path(__file__).resolve().parents[1] / "results" / "jev-20261015"


def header(name: str) -> tuple[str, ...]:
    return tuple((JEV_RUN / name).read_text(encoding="utf-8").splitlines()[0].split(","))


def test_jevs_columns_are_the_ones_its_committed_tables_carry() -> None:
    assert outcome_columns("jev", True) == header("outcomes.csv")
    assert confidence_columns("jev") == header("confidence.csv")


def test_another_scorer_gets_its_own_prefix_and_no_pointwise_columns() -> None:
    columns = outcome_columns("llm", False)
    jev_columns = outcome_columns("jev", True)
    assert "llm_selected_probability" in columns and "llm_confidence" in columns
    assert not any(c.startswith(("jev_", "pointwise_")) for c in columns)
    assert len(columns) == len(jev_columns) - 3 and len(set(columns)) == len(columns)
    assert [c for c in columns if c.startswith("qwen3_")] == [
        c for c in jev_columns if c.startswith("qwen3_")]
    own = confidence_columns("llm")
    assert "llm_confidence" in own and "llm_own_choice" in own and not any("jev" in c for c in own)


def test_signal_labels_follow_the_scorers_key_and_name() -> None:
    jev = signal_labels(JEV_VOICE)
    assert list(jev) == ["jev_selected_probability", "jev_confidence", "jev_margin",
                         "jev_candidate_mass", "jev_pass_agreement", "qwen3_score", "qwen3_margin"]
    assert jev["jev_confidence"] == "Jev: its own `confidence` value"
    chat = signal_labels(voice_for("Qwen3.8-27B"))
    assert list(chat) == [k.replace("jev_", "llm_") for k in jev]
    assert chat["llm_selected_probability"] == "Qwen3.8-27B: probability of the selected candidate"
    assert "own choice" in chat["llm_confidence"]


@pytest.mark.parametrize(("numerator", "denominator", "shown"), [
    (0, 100, "0.00"),            # identical probabilities: nothing to see
    (12, 100, "0.12"),           # what the committed Jev report prints (0.12)
    (11, 10_000, "0.0011"),      # the size a PSNC repeat pass differed by: two decimals print 0.00
    (1, 100_000, "under 0.0001"),
    (5, 1000, "0.01"),
    (1, 2, "0.50"),
])
def test_the_largest_repeat_difference_is_shown_with_enough_digits_to_be_seen(
        numerator: int, denominator: int, shown: str) -> None:
    assert _largest_difference({"numerator": numerator, "denominator": denominator}) == shown


def test_jevs_committed_repeat_sentence_is_the_one_this_code_prints() -> None:
    text = (JEV_RUN / "report.md").read_text(encoding="utf-8")
    assert "the largest difference in a candidate's probability was 0.12 (mean 0.0033)" in text
