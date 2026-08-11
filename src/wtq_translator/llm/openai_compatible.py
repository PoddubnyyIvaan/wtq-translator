from __future__ import annotations

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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._capabilities = ProviderCapabilities(seed=supports_seed)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def generate(self, *, system: str, user: str, temperature: float, max_output_tokens: int, seed: int | None = None) -> str:
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
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout_seconds: float = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "input": texts}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            r = await client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        ordered = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in ordered]
