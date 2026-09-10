import asyncio
import json

import httpx
import pytest

from wtq_translator.llm.logged import LoggedLLMProvider
from wtq_translator.llm.openai_compatible import OpenAICompatibleProvider


def _chat_response(content: str = "ok") -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


class _FakeLLMProvider:
    """Заглушка LLM-провайдера: часть user-запросов бросает исключение."""

    def __init__(self, failing_users: set[str]) -> None:
        self.failing_users = failing_users
        self.calls: list[dict] = []

    @property
    def capabilities(self):
        class Caps:
            seed = True

        return Caps()

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["user"] in self.failing_users:
            raise RuntimeError(f"error for {kwargs['user']}")
        return f"response:{kwargs['user']}"

    async def count_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.mark.asyncio
async def test_openai_provider_allows_at_most_max_concurrent_requests_in_flight():
    """N параллельных generate: одновременно в полёте не больше max_concurrent_requests."""
    max_concurrent_requests = 2
    total_requests = 6
    active = 0
    peak_active = 0
    request_delay = 0.02

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        # Держим запрос «в полёте», чтобы параллельные вызовы реально пересекались.
        await asyncio.sleep(request_delay)
        active -= 1
        return _chat_response()

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = OpenAICompatibleProvider(
        base_url="http://test.local/v1",
        api_key="",
        model="test-model",
        max_concurrent_requests=max_concurrent_requests,
        client=client,
    )
    try:
        results = await asyncio.gather(
            *[
                provider.generate(system="s", user=f"u{i}", temperature=0.0, max_output_tokens=10)
                for i in range(total_requests)
            ]
        )
    finally:
        await client.aclose()

    assert len(results) == total_requests
    assert results == ["ok"] * total_requests
    # Параллельность реально достигается (не 1), но не превышает лимит (не больше N).
    assert peak_active == max_concurrent_requests


@pytest.mark.asyncio
async def test_logged_provider_writes_complete_json_lines_under_concurrency(tmp_path):
    """M параллельных вызовов Logged-провайдера -> файл содержит M целых JSON-строк."""
    path = tmp_path / "llm_io.jsonl"
    failing_users = {f"q-{i}" for i in (1, 5, 9, 13, 17)}
    wrapped = _FakeLLMProvider(failing_users=failing_users)
    provider = LoggedLLMProvider(wrapped, path, identity={"model": "fake"})
    total = 20

    async def call(i: int) -> str:
        # Logged-провайдер ре-райзит ошибку после записи event со status=error.
        try:
            return await provider.generate(
                system="s",
                user=f"q-{i}",
                temperature=0.0,
                max_output_tokens=100,
                seed=0,
            )
        except RuntimeError:
            return "error"

    results = await asyncio.gather(*[call(i) for i in range(total)])

    assert results.count("error") == len(failing_users)
    assert results.count("error") + sum(r.startswith("response:") for r in results) == total

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == total
    events = [json.loads(line) for line in lines]
    assert sum(1 for e in events if e["status"] == "ok") == total - len(failing_users)
    assert sum(1 for e in events if e["status"] == "error") == len(failing_users)
    # Внутри каждой строки полный запрос сохраняется; формат не меняется.
    for event in events:
        assert event["kind"] == "llm_chat"
        assert event["request"]["user"].startswith("q-")
        assert event["identity"] == {"model": "fake"}
