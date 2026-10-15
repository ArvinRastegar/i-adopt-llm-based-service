"""The command line: argument validation, exit codes and the stop-with-a-reason mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iadopt_linking import cli
from iadopt_linking.errors import RunError, UsageError
from iadopt_linking.recorder import StoreError
from iadopt_linking.rerank import ScoreError
from iadopt_linking.retrieve import RetrievalTimeout
from iadopt_linking.score import PoolError


@pytest.mark.parametrize("flag", [
    ["--max-description-chars", "0"], ["--max-description-chars", "-5"],
    ["--max-description-chars", "abc"], ["--max-description-chars", "1.5"],
    ["--max-retrieval-minutes", "0"], ["--max-retrieval-minutes", "nan"],
    ["--max-retrieval-minutes", "inf"], ["--max-retrieval-minutes", "-1"],
])
def test_a_nonsensical_limit_is_a_usage_error_before_anything_runs(flag: list[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.build_parser().parse_args(["run", "--out", "x", *flag])
    assert stopped.value.code == 2


def test_the_device_check_options_cannot_be_combined() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--out", "x", "--skip-device-check", "--recheck-devices"])
    assert parser.parse_args(["run", "--out", "x", "--recheck-devices"]).recheck_devices


def test_valid_limits_are_numbers_of_the_right_kind() -> None:
    args = cli.build_parser().parse_args(
        ["run", "--out", "x", "--max-description-chars", "120", "--max-retrieval-minutes", "7.5"])
    assert args.max_description_chars == 120 and args.max_retrieval_minutes == 7.5


@pytest.mark.parametrize("error", [
    RunError("a check failed"), RetrievalTimeout("too slow"), PoolError("pool"),
    ScoreError("score"), StoreError("store")])
def test_a_deliberate_stop_exits_2_with_its_reason_and_no_traceback(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, error: Exception) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(cli, "run", fail)
    assert cli.main(["run", "--out", "x"]) == 2
    out = capsys.readouterr().out
    assert f"STOPPED: {type(error).__name__}: {error}" in out


def test_a_bug_is_not_disguised_as_a_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise KeyError("oops")

    monkeypatch.setattr(cli, "run", fail)
    with pytest.raises(KeyError):
        cli.main(["run", "--out", "x"])


def test_an_invalid_request_is_a_usage_error(monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise UsageError("No such variable file stem: ['x']")

    monkeypatch.setattr(cli, "run", fail)
    with pytest.raises(SystemExit) as stopped:
        cli.main(["run", "--out", "x"])
    assert stopped.value.code == 2 and "No such variable" in capsys.readouterr().err


def test_the_options_reach_the_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake(out: str, **kwargs: Any) -> None:
        seen.update(out=out, **kwargs)

    monkeypatch.setattr(cli, "run", fake)
    assert cli.main(["run", "--out", str(tmp_path), "--only", "A, B", "--device", "cpu",
                     "--recheck-devices", "--max-description-chars", "50"]) == 0
    assert seen["only"] == ["A", "B"] and seen["device"] == "cpu" and seen["recheck_devices"] is True
    assert seen["check_devices"] is True and seen["max_description_chars"] == 50
    cli.main(["run", "--out", str(tmp_path), "--skip-device-check"])
    assert seen["check_devices"] is False and seen["recheck_devices"] is False
