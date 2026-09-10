from __future__ import annotations

from typing import Protocol

from .capabilities import ProviderCapabilities


class LLMProvider(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_output_tokens: int,
        seed: int | None = None,
    ) -> str: ...

    async def count_tokens(self, text: str) -> int: ...


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
