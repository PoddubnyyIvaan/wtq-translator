import json

import pytest

from wtq_translator.llm.logged import LoggedLLMProvider
from wtq_translator.llm.tracing import llm_trace_context


class FakeProvider:
    @property
    def capabilities(self):
        class Caps:
            seed = True
        return Caps()

    async def generate(self, **kwargs):
        return '{"translated_text":"ok"}'


@pytest.mark.asyncio
async def test_logs_every_request_and_response_with_question_context(tmp_path):
    path = tmp_path / "llm_io.jsonl"
    provider = LoggedLLMProvider(FakeProvider(), path, identity={"model": "fake"})

    with llm_trace_context(question_id="nt-0", stage="translate_shell", attempt=0):
        result = await provider.generate(
            system="system prompt",
            user='{"text":"question"}',
            temperature=0.0,
            max_output_tokens=100,
            seed=42,
        )

    assert result == '{"translated_text":"ok"}'
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    event = rows[0]
    assert event["context"]["question_id"] == "nt-0"
    assert event["context"]["stage"] == "translate_shell"
    assert event["request"]["user"] == '{"text":"question"}'
    assert event["response"] == '{"translated_text":"ok"}'
