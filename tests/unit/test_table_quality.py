import hashlib
import math

import pytest

from wtq_translator.config import MentionNormalizationConfig
from wtq_translator.models import TableData
from wtq_translator.validation.invariants import validate_nonempty_preservation
from wtq_translator.validation.table_quality import score_back_translated_table

NORM = MentionNormalizationConfig()


class FakeEmbeddingProvider:
    def __init__(self, vector_fn):  # vector_fn: (text: str) -> list[float]
        self._vector_fn = vector_fn
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_fn(t) for t in texts]


def _hash_vector(text: str) -> list[float]:
    """Единичный вектор, зависящий от текста; разные тексты почти наверняка непараллельны."""
    h = hashlib.md5(text.encode("utf-8")).digest()
    return [math.sin(h[0]), math.cos(h[0])]


def _zed_vector(text: str) -> list[float]:
    """Ортогональная пара: back-значения (содержат 'zed'/'qux') против source-значений."""
    lowered = text.lower()
    return [0.0, 1.0] if ("zed" in lowered or "qux" in lowered) else [1.0, 0.0]


def test_nonempty_values_cannot_disappear():
    src = TableData(["Name"], [["Alice"], [""]])
    out = TableData([""], [[""], [""]])
    errors = validate_nonempty_preservation(src, out)
    assert (None, 0, "Name") in errors
    assert (0, 0, "Alice") in errors
    assert len(errors) == 2


@pytest.mark.asyncio
async def test_shape_mismatch_raises():
    src = TableData(["A", "B"], [["x", "y"]])
    back_rows = TableData(["A", "B"], [["x", "y"], ["z", "w"]])
    back_cols = TableData(["A"], [["x"]])
    embeddings = FakeEmbeddingProvider(_hash_vector)
    with pytest.raises(ValueError):
        await score_back_translated_table(src, back_rows, embeddings=embeddings, normalization=NORM)
    with pytest.raises(ValueError):
        await score_back_translated_table(src, back_cols, embeddings=embeddings, normalization=NORM)
    assert embeddings.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["context", "row", "cell"])
async def test_identical_normalized_fastpath_all_modes(mode):
    src = TableData(["Name", "City"], [["Alice", "London"]])
    # Нормализованно идентичная back-версия во всех единицах (без внутренних пробелов).
    back = TableData(["name", "city"], [["alice", "london"]])
    embeddings = FakeEmbeddingProvider(lambda t: [1.0, 1.0])
    res = await score_back_translated_table(
        src, back, embeddings=embeddings, normalization=NORM, mode=mode
    )
    assert res["min_similarity"] == 1.0
    assert res["mean_similarity"] == 1.0
    assert embeddings.calls == []


@pytest.mark.asyncio
async def test_context_chunking():
    src = TableData(["Name"], [[v] for v in ["Alice", "Bob", "Carol", "Dave", "Eve"]])
    back = TableData(["Name"], [["Zed"]] * 5)  # значения не совпадают нормализованно
    embeddings = FakeEmbeddingProvider(_zed_vector)
    res = await score_back_translated_table(
        src, back, embeddings=embeddings, normalization=NORM,
        mode="context", max_chunk_tokens=1,
    )
    groups = res["groups_compared"]
    assert groups == 5  # при бюджете 1 каждая строка — отдельный чанк
    assert len(embeddings.calls) == 1
    assert len(embeddings.calls[0]) == 2 * groups  # все группы не прошли fast-path

    segments = [(ex["start_row"], ex["end_row"]) for ex in res["similarity_examples"]]
    assert segments == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


@pytest.mark.asyncio
async def test_row_mode():
    src = TableData(["Name"], [["Alice"], ["Bob"]])
    back = TableData(["name"], [["Zed"], ["Qux"]])  # заголовок fast-path, строки нет
    embeddings = FakeEmbeddingProvider(_zed_vector)
    res = await score_back_translated_table(
        src, back, embeddings=embeddings, normalization=NORM, mode="row"
    )
    assert res["groups_compared"] == 1 + 2
    non_fast = [ex for ex in res["similarity_examples"] if ex["score"] < 1.0]
    assert non_fast
    assert all("start_row" in ex and "end_row" in ex for ex in non_fast)


@pytest.mark.asyncio
async def test_cell_mode():
    src = TableData(["A", "B"], [["a1", "b1"]])
    back = TableData(["x", "y"], [["zed1", "zed2"]])  # все ячейки не совпадают
    embeddings = FakeEmbeddingProvider(_hash_vector)
    res = await score_back_translated_table(
        src, back, embeddings=embeddings, normalization=NORM, mode="cell"
    )
    # columns=None => выбраны все колонки: заголовки + data-ячейки.
    assert res["groups_compared"] == 2 * (1 + 1)
    examples = res["similarity_examples"]
    assert examples
    assert all("row" in ex and "column" in ex for ex in examples)
    assert any(ex["row"] is None and ex["column"] is not None for ex in examples)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["context", "row", "cell"])
async def test_columns_scoping(mode):
    src = TableData(["A", "QUARANTINE"], [["alice", "QUARANTINE1"], ["bob", "QUARANTINE2"]])
    back = TableData(["x", "y"], [["ZedX", "quarantine1"], ["ZedY", "quarantine2"]])
    embeddings = FakeEmbeddingProvider(_zed_vector)
    res = await score_back_translated_table(
        src, back, embeddings=embeddings, normalization=NORM, mode=mode, columns={0}
    )
    for ex in res["similarity_examples"]:
        if mode == "cell":
            assert ex["column"] == 0
        combined = (ex["source"] + ex["back_translated"]).lower()
        assert "quarantine" not in combined


@pytest.mark.asyncio
async def test_aggregation():
    src = TableData(["Name"], [["Alice"]])
    back = TableData(["name"], [["Bob"]])  # заголовок fast-path (1.0), строка — косинус 0.0
    embeddings = FakeEmbeddingProvider(
        lambda t: [1.0, 0.0] if "alice" in t.lower() else [0.0, 1.0]
    )
    res = await score_back_translated_table(
        src, back, embeddings=embeddings, normalization=NORM, mode="row"
    )
    assert res["min_similarity"] == 0.0
    assert res["mean_similarity"] == 0.5
    assert res["similarity_examples"] and res["similarity_examples"][0]["score"] == 0.0


@pytest.mark.asyncio
async def test_max_examples_and_filter():
    src = TableData(["A"], [[f"val{i}"] for i in range(5)])
    back = TableData(["zzz"], [[f"other{i}"] for i in range(5)])  # все единицы не-fast-path
    embeddings = FakeEmbeddingProvider(_hash_vector)
    res = await score_back_translated_table(
        src, back, embeddings=embeddings, normalization=NORM, mode="row", max_examples=2
    )
    assert res["groups_compared"] == 1 + 5
    assert len(res["similarity_examples"]) == 2
    scores = [ex["score"] for ex in res["similarity_examples"]]
    assert all(s < 1.0 for s in scores)
    assert scores == sorted(scores)
