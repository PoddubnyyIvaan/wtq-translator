from __future__ import annotations

from ..config import AppConfig
from ..llm.base import LLMProvider
from ..models import TableData
from ..mentions.candidates import extract_table_candidates
from ..translation.table import translate_table
from ..translation.table_candidates import reconstruct_table, translate_table_candidates


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


async def back_translate_table_candidates(
    provider: LLMProvider,
    table: TableData,
    *,
    config: AppConfig,
    source_language: str,
    translated_language: str,
    domain: str = "",
    nlp,
    cache=None,
) -> TableData:
    """Обратный перевод таблицы через кандидаты-подпоследовательности (симметрично full-режиму)."""
    candidate_set = extract_table_candidates(
        table, nlp, config.mentions, target_lang=source_language
    )
    await translate_table_candidates(
        provider,
        candidate_set.candidates,
        config=config,
        source_language=translated_language,
        target_language=source_language,
        domain=domain,
        cache=cache,
        headers=table.headers,
        runs=candidate_set.runs,
        trace_stage_prefix="back_translate_table_candidates",
    )
    return reconstruct_table(
        table,
        candidate_set,
        {c.lemma_key: c.translation for c in candidate_set.candidates},
    )
