import pytest

from wtq_translator.llm.capabilities import ProviderCapabilities
from wtq_translator.persistence.cache import (
    CachedEmbeddingProvider,
    CachedLLMProvider,
    SQLiteCache,
    request_hash,
)

from wtq_translator.api import Translator
from wtq_translator.config import AppConfig, ModelConfig, OllamaConfig, ProjectConfig


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.token_calls = 0
        self._capabilities = ProviderCapabilities(seed=True)

    @property
    def capabilities(self):
        return self._capabilities

    async def generate(self, **kwargs):
        self.calls += 1
        return f"response-{self.calls}"

    async def count_tokens(self, text):
        self.token_calls += 1
        return len(text)


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [[float(len(t)), 1.0] for t in texts]


@pytest.mark.asyncio
async def test_llm_cache_reuses_exact_request(tmp_path):
    base = FakeLLM()
    cached = CachedLLMProvider(base, SQLiteCache(tmp_path / "state.sqlite"), {"model": "fake"})
    kwargs = dict(system="s", user="u", temperature=0.0, max_output_tokens=10, seed=1)
    a = await cached.generate(**kwargs)
    b = await cached.generate(**kwargs)
    assert a == b
    assert base.calls == 1


@pytest.mark.asyncio
async def test_embedding_cache(tmp_path):
    base = FakeEmbeddings()
    cached = CachedEmbeddingProvider(base, SQLiteCache(tmp_path / "state.sqlite"), {"model": "fake-e"})
    a = await cached.embed(["abc", "x"])
    b = await cached.embed(["abc", "x"])
    assert a == b
    assert base.calls == 1


@pytest.mark.asyncio
async def test_count_tokens_proxies_to_wrapped(tmp_path):
    base = FakeLLM()
    cached = CachedLLMProvider(base, SQLiteCache(tmp_path / "state.sqlite"), {"model": "fake"})
    count = await cached.count_tokens("hello world")
    assert count == len("hello world")
    assert base.token_calls == 1


@pytest.mark.asyncio
async def test_count_tokens_memoizes_repeated_text(tmp_path):
    base = FakeLLM()
    cached = CachedLLMProvider(base, SQLiteCache(tmp_path / "state.sqlite"), {"model": "fake"})
    text = "повторяющийся текст"
    first = await cached.count_tokens(text)
    second = await cached.count_tokens(text)
    assert first == len(text)
    assert second == first
    assert base.token_calls == 1


# --- Задача 3c: identity кэша/логов для ollama-параметров (api.Translator) ---


def _app_config(tmp_path, *, name: str, provider: str, think: bool = False, options: dict | None = None) -> AppConfig:
    """AppConfig с кэшем и логированием (дефолты включены) для построения Translator."""
    root = tmp_path / name
    return AppConfig(
        project=ProjectConfig(input_root=root, output_root=root),
        model=ModelConfig(
            provider=provider,
            model_name="qwen",
            ollama=OllamaConfig(think=think, options=options or {}),
        ),
    )


async def _aclose_llm(translator: Translator) -> None:
    """Закрыть самый внутренний LLM-провайдер (общий httpx.AsyncClient)."""
    llm = translator.llm
    while hasattr(llm, "wrapped"):
        llm = llm.wrapped
    if hasattr(llm, "aclose"):
        await llm.aclose()


def _chat_payload(identity: dict) -> dict:
    """Повторяет состав payload из CachedLLMProvider.generate для вычисления request_hash."""
    return {
        "kind": "chat",
        "identity": identity,
        "system": "s",
        "user": "u",
        "temperature": 0.0,
        "max_output_tokens": 10,
        "seed": 1,
    }


@pytest.mark.asyncio
async def test_ollama_params_in_cached_and_logged_identity(tmp_path):
    """При provider: ollama identity Cached- и Logged-обёрток содержат
    model.ollama.model_dump(mode="json") — think/stream/options входят в cache-key и лог."""
    options = {"num_predict": 512}
    cfg = _app_config(tmp_path, name="ollama", provider="ollama", think=True, options=options)
    translator = Translator(cfg)
    try:
        expected = cfg.model.ollama.model_dump(mode="json")
        assert expected["think"] is True
        assert expected["options"] == options
        # translator.llm = Logged(cache) — проверяем identity обеих обёрток.
        for identity in (translator.llm.identity, translator.llm.wrapped.identity):
            assert identity["provider"] == "ollama"
            assert identity["ollama"] == expected
    finally:
        await _aclose_llm(translator)


@pytest.mark.asyncio
async def test_ollama_think_change_changes_identity_and_hash(tmp_path):
    """Смена model.ollama.think (True->False) даёт другую identity и request-hash."""
    on = Translator(_app_config(tmp_path, name="think-on", provider="ollama", think=True))
    off = Translator(_app_config(tmp_path, name="think-off", provider="ollama", think=False))
    try:
        assert on.llm.wrapped.identity != off.llm.wrapped.identity
        assert request_hash(_chat_payload(on.llm.wrapped.identity)) != request_hash(
            _chat_payload(off.llm.wrapped.identity)
        )
    finally:
        await _aclose_llm(on)
        await _aclose_llm(off)


@pytest.mark.asyncio
async def test_ollama_options_change_changes_identity_and_hash(tmp_path):
    """Смена model.ollama.options ({num_predict: 512} vs {}) даёт другую identity и request-hash."""
    with_opts = Translator(
        _app_config(tmp_path, name="opts-set", provider="ollama", options={"num_predict": 512})
    )
    empty_opts = Translator(_app_config(tmp_path, name="opts-empty", provider="ollama", options={}))
    try:
        assert with_opts.llm.wrapped.identity != empty_opts.llm.wrapped.identity
        assert request_hash(_chat_payload(with_opts.llm.wrapped.identity)) != request_hash(
            _chat_payload(empty_opts.llm.wrapped.identity)
        )
    finally:
        await _aclose_llm(with_opts)
        await _aclose_llm(empty_opts)


@pytest.mark.asyncio
async def test_openai_compatible_identity_has_no_ollama_key(tmp_path):
    """При provider: openai_compatible ключа 'ollama' в identity нет (поведение не меняется)."""
    cfg = _app_config(
        tmp_path, name="openai", provider="openai_compatible", options={"num_predict": 512}
    )
    translator = Translator(cfg)
    try:
        for identity in (translator.llm.identity, translator.llm.wrapped.identity):
            assert "ollama" not in identity
            assert set(identity) == {"provider", "base_url", "model"}
    finally:
        await _aclose_llm(translator)
