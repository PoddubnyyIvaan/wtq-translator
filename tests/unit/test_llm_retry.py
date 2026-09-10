import json

import pytest

from wtq_translator.config import ModelConfig
from wtq_translator.translation.shell import translate_shell


class SchemaThenDataProvider:
    def __init__(self):
        self.calls = []

    async def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return json.dumps({
                "type": "object",
                "properties": {"translated_text": {"type": "string"}},
                "required": ["translated_text"],
            })
        return json.dumps({"translated_text": "Кто играл с <TABLE_REF_0000>?"}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_shell_recovers_when_model_returns_json_schema_first():
    provider = SchemaThenDataProvider()
    result = await translate_shell(
        provider,
        "Who played with <TABLE_REF_0000>?",
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=2),
        seed=42,
    )
    assert result == "Кто играл с <TABLE_REF_0000>?"
    assert len(provider.calls) == 2
    retry_payload = json.loads(provider.calls[1]["user"])
    assert "retry_instruction" in retry_payload
