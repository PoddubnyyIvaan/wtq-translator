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

    async def count_tokens(self, text: str) -> int:
        return len(text.split())


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


@pytest.mark.asyncio
async def test_tag_glossary_payload_has_worked_example_for_en_ru():
    provider = CaptureProvider("кем был капитан <TABLE_REF_0000>команды</TABLE_REF_0000> в 2001 году?")

    await translate_shell(
        provider,
        "who was <TABLE_REF_0000>the team</TABLE_REF_0000>'s captain in 2001?",
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=0),
        seed=42,
        tag_glossary={"<TABLE_REF_0000>": "команда"},
    )

    payload = json.loads(provider.calls[0]["user"])
    example = payload["worked_example"]
    # Теги в примере сохранены, словарное слово «команда» согласовано → «команды»,
    # пара тегов переехала вместе со словом (после «капитан») по правилам РУ-порядка слов.
    assert example["output"] == "кем был капитан <TABLE_REF_0000>команды</TABLE_REF_0000> в 2001 году?"


@pytest.mark.asyncio
async def test_tag_glossary_payload_has_worked_example_for_ru_en():
    provider = CaptureProvider("who was <TABLE_REF_0000>the team</TABLE_REF_0000>'s captain in 2001?")

    await translate_shell(
        provider,
        "кем был капитан <TABLE_REF_0000>команды</TABLE_REF_0000> в 2001 году?",
        source_language="ru",
        target_language="en",
        model=ModelConfig(retries=0),
        seed=42,
        tag_glossary={"<TABLE_REF_0000>": "the team"},
    )

    payload = json.loads(provider.calls[0]["user"])
    example = payload["worked_example"]
    assert example["output"] == "who was <TABLE_REF_0000>the team</TABLE_REF_0000>'s captain in 2001?"


@pytest.mark.asyncio
async def test_tag_glossary_no_worked_example_for_unknown_language_pair():
    provider = CaptureProvider("<TABLE_REF_0000>Berlin</TABLE_REF_0000> liegt in Deutschland?")

    await translate_shell(
        provider,
        "where is <TABLE_REF_0000>Berlin</TABLE_REF_0000>?",
        source_language="en",
        target_language="de",
        model=ModelConfig(retries=0),
        seed=42,
        tag_glossary={"<TABLE_REF_0000>": "Berlin"},
    )

    payload = json.loads(provider.calls[0]["user"])
    assert "worked_example" not in payload
