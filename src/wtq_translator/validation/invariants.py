from __future__ import annotations

import re

from ..models import TableData

_URL_RE = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
_ID_RE = re.compile(r"^[A-Za-z_]+[-_:]?\d{2,}$")
_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)%?$")


def is_immutable_value(value: str) -> bool:
    s = value.strip()
    if s == "":
        return True
    return bool(_URL_RE.match(s) or _ID_RE.match(s) or _NUM_RE.match(s))


def validate_immutable_values(source: TableData, candidate: TableData) -> list[tuple[int | None, int, str, str]]:
    errors: list[tuple[int | None, int, str, str]] = []
    for col, src in enumerate(source.headers):
        if is_immutable_value(src) and candidate.headers[col] != src:
            errors.append((None, col, src, candidate.headers[col]))
    for r, row in enumerate(source.rows):
        for c, src in enumerate(row):
            if is_immutable_value(src) and candidate.rows[r][c] != src:
                errors.append((r, c, src, candidate.rows[r][c]))
    return errors


def validate_untranslated_columns(source: TableData, candidate: TableData, translated_columns: set[int]) -> list[tuple[int | None, int]]:
    changed: list[tuple[int | None, int]] = []
    for c in range(len(source.headers)):
        if c in translated_columns:
            continue
        if source.headers[c] != candidate.headers[c]:
            changed.append((None, c))
        for r in range(len(source.rows)):
            if source.rows[r][c] != candidate.rows[r][c]:
                changed.append((r, c))
    return changed


def validate_nonempty_preservation(
    source: TableData,
    candidate: TableData,
    columns: set[int] | None = None,
) -> list[tuple[int | None, int, str]]:
    """Return source values that disappeared into empty strings.

    Translation may change text, but it must never erase a non-empty header/cell. Shape
    equality alone cannot detect this failure mode.
    """
    selected = set(range(len(source.headers))) if columns is None else set(columns)
    errors: list[tuple[int | None, int, str]] = []
    for c, src in enumerate(source.headers):
        if c in selected and src.strip() and not candidate.headers[c].strip():
            errors.append((None, c, src))
    for r, row in enumerate(source.rows):
        for c, src in enumerate(row):
            if c in selected and src.strip() and not candidate.rows[r][c].strip():
                errors.append((r, c, src))
    return errors
