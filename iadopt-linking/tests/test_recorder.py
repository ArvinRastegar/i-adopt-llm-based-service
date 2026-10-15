"""The append-only store: durable lines, tolerant of one torn last line, strict about the rest."""

from __future__ import annotations

from pathlib import Path

import pytest

from iadopt_linking.recorder import JsonlStore, StoreError, latest


def test_records_round_trip_in_order_with_sorted_keys(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "deep" / "r.jsonl")
    assert store.read() == []
    store.append({"b": 1, "a": "é"})
    store.append({"a": 2})
    assert store.read() == [{"a": "é", "b": 1}, {"a": 2}]
    assert (tmp_path / "deep" / "r.jsonl").read_text(encoding="utf-8").splitlines()[0] == '{"a": "é", "b": 1}'


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('{"a": 1}\n\n  \n{"a": 2}\n', encoding="utf-8")
    assert JsonlStore(path).read() == [{"a": 1}, {"a": 2}]


def test_a_torn_last_line_is_ignored_when_reading_and_removed_when_appending(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('{"a": 1}\n{"a": 2, "b": "tru', encoding="utf-8")
    store = JsonlStore(path)
    assert store.read() == [{"a": 1}]
    store.append({"a": 3})
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n{"a": 3}\n'


def test_a_complete_last_record_without_a_newline_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}', encoding="utf-8")
    store = JsonlStore(path)
    assert store.read() == [{"a": 1}, {"a": 2}]
    store.append({"a": 3})
    assert store.read() == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert path.read_text(encoding="utf-8").count("\n") == 3


def test_a_corrupt_line_before_the_end_is_an_error_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('{"a": 1}\nnot json\n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(StoreError, match=r"r.jsonl:2: not valid JSON"):
        JsonlStore(path).read()


def test_a_line_that_is_json_but_not_an_object_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(StoreError, match="r.jsonl:2: not a JSON object"):
        JsonlStore(path).read()


def test_a_value_json_cannot_hold_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "r.jsonl")
    with pytest.raises(ValueError):
        store.append({"score": float("nan")})
    with pytest.raises(TypeError):
        store.append({"x": object()})
    assert store.read() == []


def test_the_later_record_for_a_key_supersedes_the_earlier_one() -> None:
    records = [{"k": "a", "v": 1}, {"k": "b", "v": 2}, {"k": "a", "v": 3}]
    assert latest(records, lambda r: r["k"]) == {"a": {"k": "a", "v": 3}, "b": {"k": "b", "v": 2}}
