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
        tokenize_path: str = "/tokenize",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.tokenize_path = tokenize_path
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

    async def count_tokens(self, text: str) -> int:
        """Посчитать токены через токенизатор (тот же клиент/заголовки, что generate)."""
        payload = {"model": self.model, "prompt": text}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # Токенизатор (vLLM/SGLang-стиль) живёт в корне сервера, а не под /v1:
        # суффикс /v1 в base_url нужен только для /v1/chat/completions.
        tokenize_url = (
            f"{self.base_url[:-3]}{self.tokenize_path}"
            if self.base_url.endswith("/v1")
            else f"{self.base_url}{self.tokenize_path}"
        )
        async with self._sem:
            r = await self._client.post(
                tokenize_url, json=payload, headers=headers
            )
            r.raise_for_status()
            data = r.json()
        count = data.get("count")
        if count is not None:
            return int(count)
        if "tokens" in data:
            return len(data["tokens"])
        return len(data.get("input_ids", []))


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
