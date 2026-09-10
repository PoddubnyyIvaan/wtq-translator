import json

import httpx
import pytest

from wtq_translator.llm.openai_compatible import OpenAICompatibleProvider


def _provider(client: httpx.AsyncClient, **kwargs) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="https://openai.local/v1",
        api_key="test-key",
        model="gpt-test",
        client=client,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_count_tokens_posts_default_url_with_prompt_and_headers():
    """Запрос идёт на POST {base_url}/tokenize с телом {"model", "prompt"} и
    заготовленным Authorization-заголовком."""
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content)
        captured["headers"] = request.headers
        return httpx.Response(200, json={"count": 7})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = _provider(client)
    try:
        result = await provider.count_tokens("some text here")
    finally:
        await client.aclose()

    assert result == 7
    assert captured["method"] == "POST"
    # Токенизатор живёт в корне сервера: суффикс /v1 из base_url отбрасывается,
    # к корню приклеивается дефолтный tokenize_path "/tokenize".
    assert captured["url"] == "https://openai.local/tokenize"
    assert captured["body"] == {"model": "gpt-test", "prompt": "some text here"}
    assert captured["headers"]["authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_count_tokens_uses_count_field():
    """Приоритет парсинга: поле count возвращается как int."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"count": 12, "tokens": ["a", "b"]})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = _provider(client)
    try:
        result = await provider.count_tokens("hello")
    finally:
        await client.aclose()

    assert result == 12


@pytest.mark.asyncio
async def test_count_tokens_falls_back_to_tokens_length():
    """Без поля count длина считается от списка tokens."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tokens": ["a", "b", "c"]})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = _provider(client)
    try:
        result = await provider.count_tokens("hello")
    finally:
        await client.aclose()

    assert result == 3


@pytest.mark.asyncio
async def test_count_tokens_falls_back_to_input_ids_length():
    """Без count и tokens длина считается из списка input_ids."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"input_ids": [1, 2, 3, 4]})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = _provider(client)
    try:
        result = await provider.count_tokens("hello")
    finally:
        await client.aclose()

    assert result == 4


@pytest.mark.asyncio
async def test_custom_tokenize_path_used_in_url():
    """Кастомный tokenize_path подставляется в URL после base_url."""
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"count": 2})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = _provider(client, tokenize_path="/custom/tokenize")
    try:
        result = await provider.count_tokens("hello")
    finally:
        await client.aclose()

    assert result == 2
    assert captured["url"] == "https://openai.local/custom/tokenize"
