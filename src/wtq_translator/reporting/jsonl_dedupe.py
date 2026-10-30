from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

START_CODE = "START"


def load_jsonl_rows(path: str | Path) -> list[dict]:
    """Построчное чтение JSONL-файла в список dict.

    Отсутствующий файл → пустой список; пустые строки и строки с битым JSON
    (возможны legacy-записи старых схем) пропускаются.
    """
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def dedupe_last_wins(rows: list[dict], key_fn: Callable[[dict], Any]) -> list[dict]:
    """Дедупликация по ключу ``key_fn(row)``: остаётся последнее вхождение.

    Порядок результата — порядок первого вхождения каждого ключа (детерминированно).
    Записи с ключом ``None`` (например, без ``question_id``) сохраняются все в своих
    позициях, без дедупликации.
    """
    result: list[dict] = []
    seen: dict[Any, int] = {}
    for row in rows:
        key = key_fn(row)
        if key is None:
            result.append(row)
            continue
        pos = seen.get(key)
        if pos is None:
            seen[key] = len(result)
            result.append(row)
        else:
            result[pos] = row
    return result


def dedupe_events(rows: list[dict]) -> list[dict]:
    """Дедупликация событий по сессии вопроса.

    Для каждого ``question_id``, у которого есть хотя бы одно событие ``START``,
    остаётся только последняя попытка: сам последний ``START`` и все следующие за
    ним события этого вопроса. Для вопроса без ``START`` — last-wins по ключу
    ``(question_id, code)``. События без ``question_id`` не трогаются.
    """
    last_start: dict[Any, int] = {}
    for idx, row in enumerate(rows):
        if row.get("code") == START_CODE and row.get("question_id") is not None:
            last_start[row["question_id"]] = idx

    result: list[dict] = []
    fallback_pos: dict[tuple[Any, Any], int] = {}
    for idx, row in enumerate(rows):
        qid = row.get("question_id")
        if qid is None:
            result.append(row)
            continue
        if qid in last_start:
            if idx >= last_start[qid]:
                result.append(row)
            continue
        key = (qid, row.get("code"))
        pos = fallback_pos.get(key)
        if pos is None:
            fallback_pos[key] = len(result)
            result.append(row)
        else:
            result[pos] = row
    return result
