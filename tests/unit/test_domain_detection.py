"""Тесты определения предметной области таблицы (detect_domain).

Стиль: pytest + pytest-asyncio; вместо реального LLM-провайдера используется
фейковый провайдер с async generate(...) и async count_tokens(...),
как принято в tests/unit (см. test_llm_retry.py).
"""
import json

import pytest

from wtq_translator.config import (
    AppConfig,
    DomainDetectionConfig,
    ModelConfig,
    ProjectConfig,
)
from wtq_translator.domain import detect_domain
from wtq_translator.models import TableData


class FakeProvider:
    """Минимальный фейковый LLM-провайдер: реализует async generate/count_tokens."""

    def __init__(self, responses=None, token_count=0):
        # ``responses`` — очередь ответов generate (pop по мере вызова).
        self._responses = list(responses or [])
        self._token_count = token_count
        self.calls = []  # аргументы каждого вызова generate
        self.count_token_calls = 0

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"domain": "finance"}, ensure_ascii=False)

    async def count_tokens(self, text: str) -> int:
        self.count_token_calls += 1
        return self._token_count


def _table():
    return TableData(headers=["Year", "Revenue"], rows=[["2020", "100"], ["2021", "120"]])


def _config(tmp_path, *, enabled=True, retries=2, source_language="en"):
    return AppConfig(
        project=ProjectConfig(
            input_root=tmp_path,
            output_root=tmp_path,
            source_language=source_language,
            seed=7,
        ),
        model=ModelConfig(retries=retries, temperature=0.0, max_output_tokens=256),
        domain_detection=DomainDetectionConfig(enabled=enabled, max_tokens=500),
    )


@pytest.mark.asyncio
async def test_success_returns_domain_and_empty_warning(tmp_path):
    """Провайдер возвращает корректный JSON → (domain, ""), без ретраев."""
    provider = FakeProvider(responses=[json.dumps({"domain": "finance"})])
    config = _config(tmp_path)

    domain, warning = await detect_domain(provider, config, _table())

    assert domain == "finance"
    assert warning == ""
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_non_json_after_retries_returns_warning_without_raising(tmp_path):
    """При постоянном не-JSON ответе после ретраев → ("", warning), не бросает."""
    provider = FakeProvider(responses=["not json at all", "", "still broken"])
    config = _config(tmp_path, retries=2)  # attempts = retries + 1 = 3

    domain, warning = await detect_domain(provider, config, _table())

    assert domain == ""
    assert warning.startswith("Domain detection failed after retries:")
    # Проверяем, что провайдер действительно вызывался все попытки.
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_recovers_on_retry_after_invalid_first_response(tmp_path):
    """Первый ответ не-JSON, повторный — валидный → домен возвращён на ретрае."""
    provider = FakeProvider(responses=["garbage", json.dumps({"domain": "sport"})])
    config = _config(tmp_path, retries=2)

    domain, warning = await detect_domain(provider, config, _table())

    assert domain == "sport"
    assert warning == ""
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_disabled_config_returns_empty_without_calling_provider(tmp_path):
    """domain_detection.enabled False → ("", "") и provider.generate НЕ вызывается."""
    provider = FakeProvider()
    config = _config(tmp_path, enabled=False)

    domain, warning = await detect_domain(provider, config, _table())

    assert domain == ""
    assert warning == ""
    assert provider.calls == []
    assert provider.count_token_calls == 0


@pytest.mark.asyncio
async def test_snippet_is_sent_as_user_payload(tmp_path):
    """Рабочий кейс: линеаризованная таблица уходит в user-поле запроса."""
    provider = FakeProvider(responses=[json.dumps({"domain": "finance"})])
    config = _config(tmp_path)

    await detect_domain(provider, config, _table())

    payload = json.loads(provider.calls[0]["user"])
    assert payload["task"] == "detect_domain"
    assert "snippet" in payload
    assert "col : [Year] | [Revenue]" in payload["snippet"]
