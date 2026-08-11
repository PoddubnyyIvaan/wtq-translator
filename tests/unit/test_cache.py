import pytest

from wtq_translator.llm.capabilities import ProviderCapabilities
from wtq_translator.persistence.cache import CachedEmbeddingProvider, CachedLLMProvider, SQLiteCache


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self._capabilities = ProviderCapabilities(seed=True)

    @property
    def capabilities(self):
        return self._capabilities

    async def generate(self, **kwargs):
        self.calls += 1
        return f"response-{self.calls}"


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
