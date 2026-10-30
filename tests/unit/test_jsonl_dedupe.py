"""Тесты дедупликации JSONL (reporting/jsonl_dedupe.py) — без LLM."""
import json
from pathlib import Path

from wtq_translator.reporting.jsonl_dedupe import (
    dedupe_events,
    dedupe_last_wins,
    load_jsonl_rows,
)


def _write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_jsonl_rows_skips_empty_and_broken_lines(tmp_path: Path):
    p = _write(tmp_path / "x.jsonl", [
        '{"a": 1}',
        "",
        "not-json",
        '{"b": 2}',
        "   ",
    ])
    assert load_jsonl_rows(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_rows_missing_file(tmp_path: Path):
    assert load_jsonl_rows(tmp_path / "nope.jsonl") == []


def test_dedupe_last_wins_keeps_last_by_key_in_first_seen_order():
    rows = [
        {"question_id": "q1", "v": 1},
        {"question_id": "q2", "v": 1},
        {"question_id": "q1", "v": 2},
        {"question_id": "q3", "v": 1},
        {"question_id": "q1", "v": 3},
    ]
    out = dedupe_last_wins(rows, lambda r: r.get("question_id"))
    assert out == [
        {"question_id": "q1", "v": 3},
        {"question_id": "q2", "v": 1},
        {"question_id": "q3", "v": 1},
    ]
    # порядок первого вхождения не меняется при обновлении значения
    assert [r["question_id"] for r in out] == ["q1", "q2", "q3"]


def test_dedupe_last_wins_keeps_all_none_key_rows():
    rows = [
        {"question_id": "q1", "v": 1},
        {"v": 10},  # нет question_id
        {"question_id": "q1", "v": 2},
        {"v": 20},
        {"question_id": "q2", "v": 1},
        {"v": 30},
    ]
    out = dedupe_last_wins(rows, lambda r: r.get("question_id"))
    assert out == [
        {"question_id": "q1", "v": 2},
        {"v": 10},
        {"v": 20},
        {"question_id": "q2", "v": 1},
        {"v": 30},
    ]


def test_dedupe_events_keeps_last_attempt_only():
    rows = [
        {"question_id": "q1", "code": "START", "message": "attempt 1"},
        {"question_id": "q1", "code": "FAILED", "message": "w1"},
        {"question_id": "q1", "code": "START", "message": "attempt 2"},
        {"question_id": "q1", "code": "DONE", "message": "ok"},
    ]
    out = dedupe_events(rows)
    assert out == [
        {"question_id": "q1", "code": "START", "message": "attempt 2"},
        {"question_id": "q1", "code": "DONE", "message": "ok"},
    ]


def test_dedupe_events_without_start_last_wins_by_code():
    rows = [
        {"question_id": "q1", "code": "DONE", "message": "old"},
        {"question_id": "q2", "code": "FAILED", "message": "only"},
        {"question_id": "q1", "code": "DONE", "message": "newest"},
    ]
    out = dedupe_events(rows)
    # позиция первого вхождения (q1, DONE) сохранена, значение — последнее
    assert out == [
        {"question_id": "q1", "code": "DONE", "message": "newest"},
        {"question_id": "q2", "code": "FAILED", "message": "only"},
    ]


def test_dedupe_events_passes_rows_without_question_id():
    rows = [
        {"code": "START", "message": "run level"},
        {"question_id": "q1", "code": "START"},
        {"question_id": "q1", "code": "DONE"},
        {"message": "tail"},
    ]
    assert dedupe_events(rows) == rows


def test_dedupe_events_interleaved_questions_restart_drops_early_rows():
    rows = [
        {"question_id": "q1", "code": "START", "m": "a1"},
        {"question_id": "q2", "code": "START", "m": "b1"},
        {"question_id": "q1", "code": "FAILED", "m": "a-fail"},
        {"question_id": "q1", "code": "START", "m": "a2"},
        {"question_id": "q2", "code": "DONE", "m": "b-done"},
        {"question_id": "q1", "code": "DONE", "m": "a-ok"},
    ]
    out = dedupe_events(rows)
    assert out == [
        {"question_id": "q2", "code": "START", "m": "b1"},
        {"question_id": "q1", "code": "START", "m": "a2"},
        {"question_id": "q2", "code": "DONE", "m": "b-done"},
        {"question_id": "q1", "code": "DONE", "m": "a-ok"},
    ]
