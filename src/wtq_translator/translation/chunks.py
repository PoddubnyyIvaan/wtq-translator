from __future__ import annotations

from dataclasses import dataclass

from ..models import TableData


@dataclass(frozen=True)
class RowChunk:
    start_row: int
    rows: list[list[str]]


def chunk_rows(table: TableData, rows_per_chunk: int) -> list[RowChunk]:
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    return [RowChunk(i, table.rows[i:i + rows_per_chunk]) for i in range(0, len(table.rows), rows_per_chunk)]
