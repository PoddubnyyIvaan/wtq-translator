from __future__ import annotations

from ..config import AppConfig
from ..llm.base import LLMProvider
from ..models import TableData
from ..translation.table import translate_table


async def back_translate_table(
    provider: LLMProvider,
    table: TableData,
    *,
    config: AppConfig,
    source_language: str,
    translated_language: str,
    columns: set[int] | None = None,
) -> TableData:
    return await translate_table(
        provider,
        table,
        config=config,
        source_language=translated_language,
        target_language=source_language,
        columns=columns,
        trace_stage_prefix="back_translate_table",
    )
