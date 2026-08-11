from __future__ import annotations

from ..config import AppConfig
from .openai_compatible import OpenAICompatibleEmbeddingProvider, OpenAICompatibleProvider


def create_llm_provider(config: AppConfig):
    if config.model.provider != "openai_compatible":
        raise ValueError(f"Unsupported provider: {config.model.provider}")
    return OpenAICompatibleProvider(
        base_url=config.model.base_url,
        api_key=config.model.api_key,
        model=config.model.model_name,
        timeout_seconds=config.model.timeout_seconds,
    )


def create_embedding_provider(config: AppConfig):
    e = config.embeddings
    if not e.enabled:
        return None
    if e.provider != "openai_compatible":
        raise ValueError(f"Unsupported embedding provider: {e.provider}")
    if not e.model_name:
        raise ValueError("embeddings.model_name is required when embeddings are enabled")
    return OpenAICompatibleEmbeddingProvider(
        base_url=e.base_url or config.model.base_url,
        api_key=e.api_key if e.api_key is not None else config.model.api_key,
        model=e.model_name,
        timeout_seconds=e.timeout_seconds,
    )
