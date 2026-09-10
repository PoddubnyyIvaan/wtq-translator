import asyncio
import json

import httpx
import pytest

from wtq_translator.config import AppConfig, ModelConfig, OllamaConfig, ProjectConfig
from wtq_translator.llm.factory import create_llm_provider
from wtq_translator.llm.ollama import OllamaProvider
from wtq_translator.llm.openai_compatible import OpenAICompatibleProvider


def _ollama_response(content: str = "ok") -> httpx.Response:
    # Ответ нативного Ollama при stream=false — один JSON c message.content.
    return httpx.Response(200, json={"message": {"content": content}})


@pytest.mark.asyncio
async def test_generate_posts_default_payload_to_api_chat():
    """Запрос уходит на /api/chat: stream/think=false по умолчанию,
    options.num_predict == max_output_tokens; возвращается message.content."""
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return _ollama_response("привет")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = OllamaProvider(
        base_url="http://ollama.local:11434",
        api_key="",
        model="qwen",
        ollama=OllamaConfig(),
        client=client,
    )
    try:
        content = await provider.generate(
            system="sys", user="usr", temperature=0.0, max_output_tokens=64
        )
    finally:
        await client.aclose()

    assert content == "привет"
    # base_url берётся как корень сервера (без /v1), путь добавляет провайдер.
    assert captured["url"] == "http://ollama.local:11434/api/chat"
    body = captured["body"]
    assert body["model"] == "qwen"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["num_predict"] == 64
    assert body["options"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_config_options_override_defaults():
    """model.ollama.options перекрывает дефолты num_predict/temperature."""
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ollama_response()

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = OllamaProvider(
        base_url="http://ollama.local:11434",
        api_key="",
        model="qwen",
        ollama=OllamaConfig(options={"num_predict": 999, "temperature": 0.7}),
        client=client,
    )
    try:
        await provider.generate(
            system="s", user="u", temperature=0.0, max_output_tokens=64
        )
    finally:
        await client.aclose()

    options = captured["body"]["options"]
    assert options["num_predict"] == 999
    assert options["temperature"] == 0.7


@pytest.mark.asyncio
async def test_seed_is_passed_to_options_when_set():
    """seed при seed is not None попадает в options.seed; при None — отсутствует."""
    bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ollama_response()

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = OllamaProvider(
        base_url="http://ollama.local:11434",
        api_key="",
        model="qwen",
        client=client,
    )
    try:
        await provider.generate(system="s", user="u", temperature=0.0, max_output_tokens=10)
        await provider.generate(
            system="s", user="u", temperature=0.0, max_output_tokens=10, seed=42
        )
    finally:
        await client.aclose()

    assert "seed" not in bodies[0]["options"]
    assert bodies[1]["options"]["seed"] == 42


@pytest.mark.asyncio
async def test_ollama_provider_allows_at_most_max_concurrent_requests_in_flight():
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
        return _ollama_response()

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = OllamaProvider(
        base_url="http://ollama.local:11434",
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


def _app_config(tmp_path, *, provider: str, max_concurrent_requests: int = 1) -> AppConfig:
    return AppConfig(
        project=ProjectConfig(input_root=tmp_path, output_root=tmp_path),
        model=ModelConfig(
            provider=provider,
            model_name="qwen",
            max_concurrent_requests=max_concurrent_requests,
        ),
    )


def test_factory_returns_ollama_provider_for_ollama(tmp_path):
    """При model.provider: ollama фабрика возвращает OllamaProvider."""
    cfg = _app_config(tmp_path, provider="ollama")
    provider = create_llm_provider(cfg)
    assert isinstance(provider, OllamaProvider)
    assert provider.model == cfg.model.model_name
    assert provider.base_url == cfg.model.base_url.rstrip("/")
    assert provider.ollama is cfg.model.ollama
    assert provider.capabilities.seed is True


def test_factory_returns_openai_compatible_provider_by_default(tmp_path):
    """При model.provider: openai_compatible фабрика по-прежнему возвращает
    OpenAICompatibleProvider (дефолт, поведение не меняется)."""
    cfg = _app_config(tmp_path, provider="openai_compatible", max_concurrent_requests=3)
    provider = create_llm_provider(cfg)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == cfg.model.model_name
