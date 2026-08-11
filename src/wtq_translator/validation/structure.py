from __future__ import annotations

from ..models import TableData


def validate_table_shape(source: TableData, candidate: TableData) -> None:
    if source.shape != candidate.shape:
        raise ValueError(f"Table shape changed: {source.shape} -> {candidate.shape}")
