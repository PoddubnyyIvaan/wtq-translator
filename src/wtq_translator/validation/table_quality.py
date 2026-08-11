from __future__ import annotations

from typing import Any

from ..config import MentionNormalizationConfig
from ..mentions.normalize import normalize_for_match
from ..models import TableData
from .structure import validate_table_shape


def compare_back_translated_table(
    source: TableData,
    back_translated: TableData,
    *,
    normalization: MentionNormalizationConfig,
    columns: set[int] | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Measure normalized exact recovery after table back translation.

    This is diagnostic, not a hard semantic criterion. Correct back translations can be
    lexically different. The metric is still useful for finding suspicious cells and
    systemic prompt/model failures without another LLM call.
    """
    validate_table_shape(source, back_translated)
    selected = set(range(len(source.headers))) if columns is None else set(columns)
    compared = 0
    matches = 0
    examples: list[dict[str, Any]] = []

    def compare(row: int | None, col: int, src: str, back: str) -> None:
        nonlocal compared, matches
        compared += 1
        if normalize_for_match(src, normalization) == normalize_for_match(back, normalization):
            matches += 1
            return
        if len(examples) < max_examples:
            examples.append({
                "row": row,
                "column": col,
                "source": src,
                "back_translated": back,
            })

    for c in range(len(source.headers)):
        if c in selected:
            compare(None, c, source.headers[c], back_translated.headers[c])
    for r, row in enumerate(source.rows):
        for c in range(len(source.headers)):
            if c in selected:
                compare(r, c, row[c], back_translated.rows[r][c])

    return {
        "compared": compared,
        "normalized_exact_matches": matches,
        "normalized_exact_ratio": (matches / compared) if compared else None,
        "mismatch_examples": examples,
    }
