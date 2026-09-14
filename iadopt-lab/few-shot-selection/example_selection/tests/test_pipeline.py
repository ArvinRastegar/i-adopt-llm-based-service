"""Acceptance tests for contracts/pipeline.md."""

from __future__ import annotations

import pytest
from pipeline import CALL_CEILING, planned_calls, write_report


def test_planned_calls_counts_every_stage():
    """AT-1 support: the budget gate must see all four stages, not just attribution."""
    total = planned_calls(n_subsets=600, search_reps=5, confirm_reps=15,
                          n_candidates=8, n_eval=38, n_confirm=24)
    assert total >= 600 * 38 + 8 * 5 * 38 + 8 * 15 * 24


def test_budget_gate_costs_the_requested_repetitions():
    """OQ-1: the gate must price the run the caller asked for, not the module defaults.

    Hardcoding DEFAULT_SEARCH_REPS let `run.py --reps 200` clear a gate that had only ever
    costed --reps 5, dispatching 188,400 calls against a 120,000 ceiling undetected.
    """
    from pipeline import _check_budget
    _check_budget(600, 38, 24, search_reps=5, confirm_reps=15)          # within ceiling
    with pytest.raises(RuntimeError):
        _check_budget(600, 38, 24, search_reps=200, confirm_reps=200)   # must now be refused


def test_budget_ceiling_is_enforced_before_dispatch():
    """AT-4: an over-budget plan is refused before any provider call."""
    assert planned_calls(n_subsets=10**6, search_reps=5, confirm_reps=15,
                         n_candidates=8, n_eval=38, n_confirm=24) > CALL_CEILING


@pytest.mark.asyncio
async def test_confirm_refuses_without_attribution(stub_context, split):
    """AT-2: stage D cannot run while stage A is incomplete (INV-4)."""
    from pipeline import stage_confirm
    with pytest.raises(RuntimeError):
        await stage_confirm(stub_context, split, {"x": split["P"][:25]}, reps=1)


def test_report_labels_search_scores_as_biased(tmp_path):
    """AT-7: an E-derived figure must never be presented as the result (INV-3)."""
    path = tmp_path / "report.md"
    write_report(path, {
        "split": {"P": 40, "E": 38, "C": 24},
        "search": {"best": {"mean": 0.5}},
        "confirm": {"top25": {"mean": 0.47, "sd": 0.02, "ci": (0.45, 0.49), "scores": [0.47]},
                    "bottom25": {"mean": 0.40, "sd": 0.02, "ci": (0.38, 0.42), "scores": [0.40]}},
        "falsification": {"passed": True},
    })
    text = path.read_text(encoding="utf-8").lower()
    assert "selection-biased" in text


def test_report_states_falsification_outcome(tmp_path):
    """AT-8: a failed falsification check must be stated, not buried (INV-6)."""
    path = tmp_path / "report.md"
    write_report(path, {
        "split": {"P": 40, "E": 38, "C": 24},
        "search": {"best": {"mean": 0.5}},
        "confirm": {"top25": {"mean": 0.44, "sd": 0.02, "ci": (0.42, 0.46), "scores": [0.44]},
                    "bottom25": {"mean": 0.45, "sd": 0.02, "ci": (0.43, 0.47), "scores": [0.45]}},
        "falsification": {"passed": False},
    })
    text = path.read_text(encoding="utf-8").lower()
    assert "failed to learn" in text or "falsification" in text
