from __future__ import annotations

from ..config import AppConfig
from ..llm.base import LLMProvider
from ..translation.shell import translate_shell


async def back_translate_shell(
    provider: LLMProvider,
    text: str,
    *,
    config: AppConfig,
    source_language: str,
    translated_language: str,
    tag_glossary: dict[str, str] | None = None,
) -> str:
    return await translate_shell(
        provider,
        text,
        source_language=translated_language,
        target_language=source_language,
        model=config.model,
        seed=config.project.seed + 100_000,
        trace_stage="back_translate_shell",
        tag_glossary=tag_glossary,
    )
