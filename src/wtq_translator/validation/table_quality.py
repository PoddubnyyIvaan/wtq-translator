from __future__ import annotations

import math
from typing import Any, Literal

from ..config import MentionNormalizationConfig
from ..llm.base import EmbeddingProvider
from ..mentions.normalize import normalize_for_match
from ..models import TableData
from .similarity import cosine_similarity
from .structure import validate_table_shape

CHARS_PER_TOKEN = 4.0
_EXAMPLE_TEXT_LIMIT = 500


def _render_header(headers: list[str], selected: set[int]) -> str:
    """Сериализовать заголовочную строку по выбранным колонкам."""
    values = " | ".join(f"[{headers[c]}]" for c in sorted(selected))
    return f"col : {values}"


def _render_row(row: list[str], row_index: int, selected: set[int]) -> str:
    """Сериализовать одну строку данных по выбранным колонкам."""
    values = " | ".join(f"[{row[c]}]" for c in sorted(selected))
    return f"row {row_index}: {values}"


def _estimate_tokens(text: str) -> int:
    """Оценить число токенов текста через эвристику длина/CHARS_PER_TOKEN."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _empty_result() -> dict[str, Any]:
    """Результат для таблицы без единиц сравнения."""
    return {
        "groups_compared": 0,
        "min_similarity": None,
        "mean_similarity": None,
        "similarity_examples": [],
    }


def _context_units(
    source: TableData,
    back_translated: TableData,
    selected: set[int],
    max_chunk_tokens: int,
) -> list[dict[str, Any]]:
    """Режим context: жадная нарезка подряд идущих строк в чанки по бюджету токенов."""
    src_header = _render_header(source.headers, selected)
    back_header = _render_header(back_translated.headers, selected)
    src_lines = [_render_row(row, r, selected) for r, row in enumerate(source.rows)]
    back_lines = [_render_row(row, r, selected) for r, row in enumerate(back_translated.rows)]

    units: list[dict[str, Any]] = []
    total_rows = len(source.rows)
    if total_rows == 0:
        # Строк нет, но заголовки есть → одна единица только из заголовочной строки.
        units.append({
            "kind": "context",
            "row": None,
            "column": None,
            "start_row": 0,
            "end_row": 0,
            "src": src_header,
            "back": back_header,
        })
        return units

    start = 0
    while start < total_rows:
        end = start
        while end < total_rows:
            candidate_end = end + 1
            block = "\n".join([src_header, *src_lines[start:candidate_end]])
            # Ровно одну строку включаем всегда: бюджет проверяем только для второй и далее.
            if end > start and _estimate_tokens(block) > max_chunk_tokens:
                break
            end = candidate_end
        units.append({
            "kind": "context",
            "row": None,
            "column": None,
            "start_row": start,
            "end_row": end,
            "src": "\n".join([src_header, *src_lines[start:end]]),
            "back": "\n".join([back_header, *back_lines[start:end]]),
        })
        start = end
    return units


def _row_units(
    source: TableData,
    back_translated: TableData,
    selected: set[int],
) -> list[dict[str, Any]]:
    """Режим row: заголовок — отдельная единица, каждая data-строка — отдельная единица."""
    src_header = _render_header(source.headers, selected)
    back_header = _render_header(back_translated.headers, selected)
    units: list[dict[str, Any]] = [{
        "kind": "row",
        "row": None,
        "column": None,
        "start_row": 0,
        "end_row": 0,
        "src": src_header,
        "back": back_header,
    }]
    for r in range(len(source.rows)):
        units.append({
            "kind": "row",
            "row": r,
            "column": None,
            "start_row": r,
            "end_row": r + 1,
            "src": _render_row(source.rows[r], r, selected),
            "back": _render_row(back_translated.rows[r], r, selected),
        })
    return units


def _cell_units(
    source: TableData,
    back_translated: TableData,
    selected: set[int],
) -> list[dict[str, Any]]:
    """Режим cell: единица — одна ячейка (заголовки отдельно от данных)."""
    units: list[dict[str, Any]] = []
    for c in sorted(selected):
        units.append({
            "kind": "cell",
            "row": None,
            "column": c,
            "start_row": None,
            "end_row": None,
            "src": source.headers[c],
            "back": back_translated.headers[c],
        })
    for r in range(len(source.rows)):
        for c in sorted(selected):
            units.append({
                "kind": "cell",
                "row": r,
                "column": c,
                "start_row": None,
                "end_row": None,
                "src": source.rows[r][c],
                "back": back_translated.rows[r][c],
            })
    return units


def _build_examples(
    units: list[dict[str, Any]],
    scores: list[float],
    max_examples: int,
) -> list[dict[str, Any]]:
    """До max_examples худших единиц со score < 1.0, отсортированных по возрастанию."""
    indexed = sorted(
        ((scores[i], i) for i in range(len(units)) if scores[i] < 1.0),
        key=lambda pair: pair[0],
    )
    examples: list[dict[str, Any]] = []
    for score, i in indexed[:max_examples]:
        unit = units[i]
        source_text = unit["src"][:_EXAMPLE_TEXT_LIMIT]
        back_text = unit["back"][:_EXAMPLE_TEXT_LIMIT]
        if unit["kind"] == "cell":
            examples.append({
                "row": unit["row"],
                "column": unit["column"],
                "score": score,
                "source": source_text,
                "back_translated": back_text,
            })
        else:
            examples.append({
                "start_row": unit["start_row"],
                "end_row": unit["end_row"],
                "score": score,
                "source": source_text,
                "back_translated": back_text,
            })
    return examples


async def score_back_translated_table(
    source: TableData,
    back_translated: TableData,
    *,
    embeddings: EmbeddingProvider,
    normalization: MentionNormalizationConfig,
    mode: Literal["context", "row", "cell"] = "context",
    max_chunk_tokens: int = 512,
    columns: set[int] | None = None,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Семантически оценить round-trip таблицы выбранной embedding-моделью.

    Единицы сравнения зависят от mode: context (заголовок + жадные чанки строк),
    row (заголовок и каждая строка отдельно), cell (каждая ячейка отдельно).
    Нормализованно совпавшие единицы дают score=1.0 без вызова модели, остальные
    уходят одним батчем в embeddings.embed, затем считается косинус по парам.
    """
    validate_table_shape(source, back_translated)

    selected = set(range(len(source.headers))) if columns is None else set(columns)
    if not selected:
        return _empty_result()

    if mode == "context":
        units = _context_units(source, back_translated, selected, max_chunk_tokens)
    elif mode == "row":
        units = _row_units(source, back_translated, selected)
    else:
        units = _cell_units(source, back_translated, selected)

    total = len(units)
    if total == 0:
        return _empty_result()

    scores: list[float] = [0.0] * total
    pending: list[int] = []
    texts: list[str] = []
    for i, unit in enumerate(units):
        # Fast-path: нормализованное совпадение не требует вызова модели.
        if normalize_for_match(unit["src"], normalization) == normalize_for_match(unit["back"], normalization):
            scores[i] = 1.0
            continue
        pending.append(i)
        texts.append(unit["src"])
        texts.append(unit["back"])

    if pending:
        vectors = await embeddings.embed(texts)
        for k, i in enumerate(pending):
            scores[i] = cosine_similarity(vectors[2 * k], vectors[2 * k + 1])

    return {
        "groups_compared": total,
        "min_similarity": min(scores),
        "mean_similarity": sum(scores) / total,
        "similarity_examples": _build_examples(units, scores, max_examples),
    }
