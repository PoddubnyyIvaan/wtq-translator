import json

import pytest

from wtq_translator.config import ModelConfig
from wtq_translator.prompts import SHELL_TRANSLATION_SYSTEM, SHELL_TRANSLATION_TAGGED_SYSTEM
from wtq_translator.translation.shell import translate_shell


class CaptureProvider:
    def __init__(self, translated_text):
        self.translated_text = translated_text
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({"translated_text": self.translated_text}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_tag_glossary_uses_tagged_system_and_payload():
    glossary = {"<TABLE_REF_0000>": "Нью-Йорк"}
    provider = CaptureProvider("Кто играл с <TABLE_REF_0000>Нью-Йорк</TABLE_REF_0000>?")

    result = await translate_shell(
        provider,
        "Who played with <TABLE_REF_0000>New York</TABLE_REF_0000>?",
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=0),
        seed=42,
        tag_glossary=glossary,
    )

    assert result == "Кто играл с <TABLE_REF_0000>Нью-Йорк</TABLE_REF_0000>?"
    call = provider.calls[0]
    assert call["system"] == SHELL_TRANSLATION_TAGGED_SYSTEM
    payload = json.loads(call["user"])
    assert payload["tag_glossary"] == glossary


@pytest.mark.asyncio
async def test_no_tag_glossary_keeps_old_behavior():
    provider = CaptureProvider("Кто играл с <TABLE_REF_0000>?")

    result = await translate_shell(
        provider,
        "Who played with <TABLE_REF_0000>?",
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=0),
        seed=42,
    )

    assert result == "Кто играл с <TABLE_REF_0000>?"
    call = provider.calls[0]
    assert call["system"] == SHELL_TRANSLATION_SYSTEM
    payload = json.loads(call["user"])
    assert "tag_glossary" not in payload


@pytest.mark.asyncio
async def test_lost_tag_raises_after_retries_with_tagged_retry_instruction():
    # Модель теряет закрывающий тег — перевод невалиден, ретраи исчерпываются.
    provider = CaptureProvider("Кто играл с <TABLE_REF_0000>Нью-Йорк?")

    with pytest.raises(RuntimeError, match="Shell translation failed after retries"):
        await translate_shell(
            provider,
            "Who played with <TABLE_REF_0000>New York</TABLE_REF_0000>?",
            source_language="en",
            target_language="ru",
            model=ModelConfig(retries=2),
            seed=42,
            tag_glossary={"<TABLE_REF_0000>": "Нью-Йорк"},
        )

    assert len(provider.calls) == 3  # retries=2 -> 3 попытки
    retry_payload = json.loads(provider.calls[1]["user"])
    instruction = retry_payload["retry_instruction"]
    assert "opening" in instruction and "closing" in instruction
