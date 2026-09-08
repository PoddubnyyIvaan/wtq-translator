from __future__ import annotations

from pathlib import Path

from .config import AppConfig, load_config
from .llm.factory import create_embedding_provider, create_llm_provider
from .llm.logged import LoggedLLMProvider
from .models import QuestionRecord, TableData, TranslationResult
from .persistence.cache import CachedEmbeddingProvider, CachedLLMProvider, SQLiteCache
from .pipeline import TranslationPipeline


class Translator:
    def __init__(self, config: AppConfig, *, llm_log_path: str | Path | None = None) -> None:
        self.config = config
        llm = create_llm_provider(config)
        embeddings = create_embedding_provider(config)

        cache = None
        if config.cache.enabled:
            cache = SQLiteCache(config.state.sqlite_path)
            llm = CachedLLMProvider(
                llm,
                cache,
                identity={
                    "provider": config.model.provider,
                    "base_url": config.model.base_url,
                    "model": config.model.model_name,
                },
            )
            if embeddings is not None and config.cache.cache_embeddings:
                e = config.embeddings
                embeddings = CachedEmbeddingProvider(
                    embeddings,
                    cache,
                    identity={
                        "provider": e.provider,
                        "base_url": e.base_url or config.model.base_url,
                        "model": e.model_name,
                    },
                )

        # Logging is intentionally outside the cache. Every logical request/response
        # is visible in the log even when its response came from SQLite cache.
        if config.logging.llm_io_enabled:
            path = Path(llm_log_path) if llm_log_path is not None else Path(config.logging.llm_io_path)
            llm = LoggedLLMProvider(
                llm,
                path,
                identity={
                    "provider": config.model.provider,
                    "base_url": config.model.base_url,
                    "model": config.model.model_name,
                },
            )

        self.llm = llm
        self.embeddings = embeddings
        self.pipeline = TranslationPipeline(config, self.llm, self.embeddings, cache=cache)

    async def process(self, question: QuestionRecord, table: TableData) -> TranslationResult:
        return await self.pipeline.process(question, table)


__all__ = ["Translator", "load_config"]
