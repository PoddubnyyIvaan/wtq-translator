from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from collections.abc import Sequence

from ..models import QuestionRecord, TableData

_STANDARD = ("id", "utterance", "context", "targetValue")

# Колонки TSV исключений (алфавитно-зависимые вопросы): исходный вопрос + класс намерения.
_ALPHABET_EXCLUDED_FIELDNAMES = ("id", "utterance", "context", "targetValue", "class", "why")


def _atomic_path(path: Path) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return Path(name), fd


def write_table_tsv(table: TableData, path: str | Path, *, atomic: bool = True) -> None:
    p = Path(path)
    target = p
    fd: int | None = None
    if atomic:
        target, fd = _atomic_path(p)
        os.close(fd)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(table.headers)
            writer.writerows(table.rows)
        if atomic:
            target.replace(p)
    finally:
        if atomic and target.exists() and target != p:
            target.unlink(missing_ok=True)


def _derive_fieldnames(records: Sequence[QuestionRecord], fieldnames: Sequence[str] | None) -> list[str]:
    if fieldnames is not None:
        names = list(fieldnames)
    else:
        names = list(_STANDARD)
        seen = set(names)
        for r in records:
            for key in r.extra_fields:
                if key not in seen:
                    names.append(key)
                    seen.add(key)
    for required in _STANDARD:
        if required not in names:
            names.append(required)
    return names


def write_question_tsv(
    records: Sequence[QuestionRecord],
    path: str | Path,
    *,
    fieldnames: Sequence[str] | None = None,
    atomic: bool = True,
) -> None:
    p = Path(path)
    names = _derive_fieldnames(records, fieldnames)
    target = p
    fd: int | None = None
    if atomic:
        target, fd = _atomic_path(p)
        os.close(fd)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=names, delimiter="\t", lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            for r in records:
                row = dict(r.extra_fields)
                row.update({
                    "id": r.id,
                    "utterance": r.utterance,
                    "context": r.context,
                    "targetValue": r.target_value,
                })
                writer.writerow(row)
        if atomic:
            target.replace(p)
    finally:
        if atomic and target.exists() and target != p:
            target.unlink(missing_ok=True)


def write_alphabet_excluded_tsv(
    rows: Sequence[dict],
    path: str | Path,
    *,
    atomic: bool = True,
) -> None:
    """Записывает TSV исключённых алфавитно-зависимых вопросов.

    Каждая строка — словарь с ключами ``id, utterance, context, targetValue, class, why``
    (``class`` — намерение из таблицы намерений; пусто при ошибке детекции).
    Тот же атомарный паттерн записи, что у ``write_question_tsv``.
    """
    p = Path(path)
    names = list(_ALPHABET_EXCLUDED_FIELDNAMES)
    target = p
    fd: int | None = None
    if atomic:
        target, fd = _atomic_path(p)
        os.close(fd)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=names, delimiter="\t", lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        if atomic:
            target.replace(p)
    finally:
        if atomic and target.exists() and target != p:
            target.unlink(missing_ok=True)
