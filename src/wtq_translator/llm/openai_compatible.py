from __future__ import annotations

import asyncio

import httpx

from .capabilities import ProviderCapabilities


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 180,
        supports_seed: bool = True,
        max_concurrent_requests: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._capabilities = ProviderCapabilities(seed=supports_seed)
        # Единый HTTP-клиент на весь процесс: keep-alive вместо установки соединения
        # на каждый запрос. Процесс одноразовый, поэтому клиент можно не закрывать;
        # при необходимости вызывается aclose(). Параметр client позволяет внедрить
        # готовый клиент (например, httpx.MockTransport) в тестах.
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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if seed is not None and self.capabilities.seed:
            payload["seed"] = seed
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # Семафор ограничивает число одновременно выполняющихся HTTP-вызовов.
        async with self._sem:
            r = await self._client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 180,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        # Keep-alive клиент на весь процесс (аналог LLM-провайдера выше).
        self._client = client or httpx.AsyncClient(timeout=self.timeout_seconds)

    async def aclose(self) -> None:
        """Закрыть общий HTTP-клиент. Вызов необязателен (процесс одноразовый)."""
        await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "input": texts}
        r = await self._client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        ordered = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in ordered]
