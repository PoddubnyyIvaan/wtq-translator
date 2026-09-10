import json

import pytest

from wtq_translator.llm.logged import LoggedLLMProvider
from wtq_translator.llm.tracing import llm_trace_context


class FakeProvider:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.token_calls = 0

    @property
    def capabilities(self):
        class Caps:
            seed = True
        return Caps()

    async def generate(self, **kwargs):
        return '{"translated_text":"ok"}'

    async def count_tokens(self, text):
        self.token_calls += 1
        if self.fail:
            raise RuntimeError("tokenizer boom")
        return len(text)


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


@pytest.mark.asyncio
async def test_count_tokens_proxies_and_logs_tokenize_event(tmp_path):
    path = tmp_path / "llm_io.jsonl"
    base = FakeProvider()
    provider = LoggedLLMProvider(base, path, identity={"model": "fake"})

    with llm_trace_context(question_id="nt-1", stage="acap_get", attempt=0):
        count = await provider.count_tokens("тестовый текст")

    assert count == len("тестовый текст")
    assert base.token_calls == 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    event = rows[0]
    assert event["kind"] == "llm_tokenize"
    assert event["status"] == "ok"
    assert event["count"] == count
    assert event["request"]["text"] == "тестовый текст"
    assert event["context"]["question_id"] == "nt-1"
    assert event["context"]["stage"] == "acap_get"


@pytest.mark.asyncio
async def test_count_tokens_logs_error_event(tmp_path):
    path = tmp_path / "llm_io.jsonl"
    base = FakeProvider(fail=True)
    provider = LoggedLLMProvider(base, path, identity={"model": "fake"})

    with pytest.raises(RuntimeError):
        with llm_trace_context(question_id="nt-2", stage="acap_get", attempt=0):
            await provider.count_tokens("boom")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    event = rows[0]
    assert event["kind"] == "llm_tokenize"
    assert event["status"] == "error"
    assert "RuntimeError" in event["error"]
    assert event["request"]["text"] == "boom"
