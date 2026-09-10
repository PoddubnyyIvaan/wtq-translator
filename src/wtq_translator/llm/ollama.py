from __future__ import annotations

import asyncio

import httpx

from ..config import OllamaConfig
from .capabilities import ProviderCapabilities


class OllamaProvider:
    """Нативный Ollama-провайдер: POST {base_url}/api/chat.

    Параметр api_key принимается для совместимости сигнатуры с
    OpenAICompatibleProvider, но не используется: Ollama API ключ не требует.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 180,
        max_concurrent_requests: int = 1,
        ollama: OllamaConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # Корень сервера БЕЗ суффикса /v1 — путь /api/chat добавляется ниже.
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.ollama = ollama or OllamaConfig()
        self._capabilities = ProviderCapabilities(seed=True)
        # Единый HTTP-клиент на весь процесс: keep-alive вместо установки соединения
        # на каждый запрос. Параметр client позволяет внедрить готовый клиент
        # (например, httpx.MockTransport) в тестах.
        self._client = client or httpx.AsyncClient(timeout=self.timeout_seconds)
        # Глобальный лимит одновременных запросов к модели: 1 = текущее поведение.
        self._sem = asyncio.Semaphore(max_concurrent_requests)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def aclose(self) -> None:
        """Закрыть общий HTTP-клиент. Вызов необязателен (процесс одноразовый)."""
        await self._client.aclose()

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_output_tokens: int,
        seed: int | None = None,
    ) -> str:
        # Опции по умолчанию: температура и лимит генерации. Конфиг (ollama.options)
        # мержится поверх и перекрывает значения по умолчанию.
        options: dict = {"temperature": temperature, "num_predict": max_output_tokens}
        if seed is not None:
            options["seed"] = seed
        options.update(self.ollama.options)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": self.ollama.stream,  # всегда False (валидировано в конфиге)
            "think": self.ollama.think,
            "options": options,
        }
        async with self._sem:
            r = await self._client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        return data["message"]["content"]

    async def count_tokens(self, text: str) -> int:
        """Посчитать токены через POST {base_url}/api/tokenize."""
        payload = {"model": self.model, "prompt": text}
        async with self._sem:
            r = await self._client.post(f"{self.base_url}/api/tokenize", json=payload)
            r.raise_for_status()
            data = r.json()
        return len(data["tokens"])

