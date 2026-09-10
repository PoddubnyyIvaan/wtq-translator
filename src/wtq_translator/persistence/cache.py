from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..llm.base import EmbeddingProvider, LLMProvider
from ..llm.capabilities import ProviderCapabilities


def request_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class SQLiteCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self):
        # timeout=30 задаёт busy timeout каждого нового соединения (30 с),
        # чтобы конкурентные записи из воркеров не падали с "database is locked".
        return sqlite3.connect(self.path, timeout=30)

    def _init(self) -> None:
        with self.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    request_hash TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    request_hash TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS candidate_translations (
                    candidate_id TEXT PRIMARY KEY,
                    source_text TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            # WAL — persistent режим журнала файла БД (нужен для конкурентных читателей/писателей),
            # busy_timeout — страховка для init-соединения; остальные соединения получают 30 с через timeout.
            db.execute("PRAGMA journal_mode=WAL").fetchone()
            db.execute("PRAGMA busy_timeout=30000")

    def get_text(self, key: str) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT response FROM llm_cache WHERE request_hash=?", (key,)).fetchone()
        return row[0] if row else None

    def put_text(self, key: str, response: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO llm_cache(request_hash,response,created_at) VALUES(?,?,?)",
                (key, response, now),
            )

    def get_vectors(self, key: str) -> list[list[float]] | None:
        with self.connect() as db:
            row = db.execute("SELECT response_json FROM embedding_cache WHERE request_hash=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_vectors(self, key: str, vectors: list[list[float]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO embedding_cache(request_hash,response_json,created_at) VALUES(?,?,?)",
                (key, json.dumps(vectors, separators=(",", ":")), now),
            )

    def get_candidate_translation(self, candidate_id: str) -> str | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT translation FROM candidate_translations WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
        return row[0] if row else None

    def put_candidate_translation(self, candidate_id: str, source_text: str, target_lang: str, translation: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO candidate_translations(candidate_id,source_text,target_lang,translation,created_at) VALUES(?,?,?,?,?)",
                (candidate_id, source_text, target_lang, translation, now),
            )


class CachedLLMProvider:
    def __init__(self, wrapped: LLMProvider, cache: SQLiteCache, identity: dict) -> None:
        self.wrapped = wrapped
        self.cache = cache
        self.identity = identity

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self.wrapped.capabilities

    async def generate(self, *, system: str, user: str, temperature: float, max_output_tokens: int, seed: int | None = None) -> str:
        payload = {
            "kind": "chat",
            "identity": self.identity,
            "system": system,
            "user": user,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "seed": seed,
        }
        key = request_hash(payload)
        cached = self.cache.get_text(key)
        if cached is not None:
            return cached
        response = await self.wrapped.generate(
            system=system,
            user=user,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=seed,
        )
        self.cache.put_text(key, response)
        return response


class CachedEmbeddingProvider:
    def __init__(self, wrapped: EmbeddingProvider, cache: SQLiteCache, identity: dict) -> None:
        self.wrapped = wrapped
        self.cache = cache
        self.identity = identity

    async def embed(self, texts: list[str]) -> list[list[float]]:
        key = request_hash({"kind": "embeddings", "identity": self.identity, "texts": texts})
        cached = self.cache.get_vectors(key)
        if cached is not None:
            return cached
        vectors = await self.wrapped.embed(texts)
        self.cache.put_vectors(key, vectors)
        return vectors
