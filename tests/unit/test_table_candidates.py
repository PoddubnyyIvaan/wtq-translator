import asyncio
import json

import pytest

from wtq_translator.config import AppConfig
from wtq_translator.models import CandidateSet, ContentRun, TableCandidate, TableCoordinate, TableData
from wtq_translator.persistence.cache import SQLiteCache
from wtq_translator.translation.table_candidates import (
    reconstruct_table,
    translate_table_candidates,
    validate_untranslated,
)
import wtq_translator.translation.table_candidates as tc


class CandidateProvider:
    """Mock-провайдер: парсит user-JSON и возвращает переводы по входным items."""

    def __init__(self, mapping):
        self.tasks = []
        self.mapping = mapping

    @property
    def capabilities(self):
        class Caps:
            seed = True
        return Caps()

    async def generate(self, **kwargs):
        payload = json.loads(kwargs["user"])
        self.tasks.append(payload)
        assert payload["task"] == "translate_candidates"
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "translation": self.mapping[item["text"]]}
                    for item in payload["items"]
                ]
            },
            ensure_ascii=False,
        )

    async def count_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.fixture(autouse=True)
def _clear_memory_cache():
    tc._MEMORY_CACHE.clear()
    yield
    tc._MEMORY_CACHE.clear()


def _cfg(tmp_path):
    return AppConfig.model_validate({
        "project": {
            "input_root": str(tmp_path),
            "output_root": str(tmp_path / "out"),
            "source_language": "en",
            "target_language": "ru",
        },
        "model": {"retries": 0},
        "cache": {"enabled": False},
    })


def _items_by_id(tasks):
    by_id = {}
    for task in tasks:
        for it in task["items"]:
            by_id[it["id"]] = it
    return by_id


@pytest.mark.asyncio
async def test_tiers_by_length_and_glossary(tmp_path):
    mapping = {"New": "Нью", "York": "Йорк", "New York": "Нью-Йорк"}
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="c1", lemma_key="new", source_text="New", coordinates=[], length=1),
        TableCandidate(id="c2", lemma_key="york", source_text="York", coordinates=[], length=1),
        TableCandidate(id="c3", lemma_key="new york", source_text="New York", coordinates=[], length=2),
    ]

    out = await translate_table_candidates(
        provider,
        candidates,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
    )

    assert out == {"c1": "Нью", "c2": "Йорк", "c3": "Нью-Йорк"}
    assert [c.translation for c in candidates] == ["Нью", "Йорк", "Нью-Йорк"]
    assert len(provider.tasks) == 2

    # Порядок: ярус длины 2 (больший) раньше яруса длины 1 (меньшего).
    assert [item["id"] for item in provider.tasks[0]["items"]] == ["c3"]
    assert [item["id"] for item in provider.tasks[1]["items"]] == ["c1", "c2"]
    # Самый длинный ярус — пустой глоссарий; меньший ярус получает per-item
    # глоссарий из самого длинного вмещающего (New York).
    assert provider.tasks[0]["items"][0]["glossary"] == []
    for item in provider.tasks[1]["items"]:
        assert item["glossary"] == [{"key": "New York", "translation": "Нью-Йорк"}]


@pytest.mark.asyncio
async def test_empty_candidates_no_llm_request(tmp_path):
    provider = CandidateProvider({})
    out = await translate_table_candidates(
        provider, [], config=_cfg(tmp_path), source_language="en", target_language="ru"
    )
    assert out == {}
    assert provider.tasks == []


def test_reconstruct_table_preserves_numbers_and_punctuation():
    table = TableData(
        headers=["City", "Year"],
        rows=[["New York (2010)", "2001"], ["", "3"]],
    )
    candidate_set = CandidateSet(
        candidates=[
            TableCandidate(
                id="c1", lemma_key="new york", source_text="New York",
                coordinates=[TableCoordinate(0, 0)], length=2,
            ),
        ],
        runs={
            TableCoordinate(0, 0): [
                ContentRun(lemma_key="new york", start=0, end=8),
                ContentRun(lemma_key="2010", start=10, end=14),
            ],
        },
    )
    assert table.rows[0][0][0:8] == "New York"
    assert table.rows[0][0][10:14] == "2010"

    out = reconstruct_table(table, candidate_set, {"new york": "Нью-Йорк"})

    assert out.rows[0][0] == "Нью-Йорк (2010)"
    assert out.rows[0][1] == "2001"   # без runs — не меняется
    assert out.rows[1][0] == ""       # пустая ячейка — не меняется
    assert out.rows[1][1] == "3"      # число без runs — не меняется
    assert out.headers == ["City", "Year"]


def test_reconstruct_table_keeps_leading_and_trailing_punctuation():
    table = TableData(["City"], [["«New York»"]])
    candidate_set = CandidateSet(
        candidates=[
            TableCandidate(
                id="c1", lemma_key="new york", source_text="New York",
                coordinates=[TableCoordinate(0, 0)], length=2,
            ),
        ],
        runs={TableCoordinate(0, 0): [ContentRun(lemma_key="new york", start=1, end=9)]},
    )
    assert table.rows[0][0][1:9] == "New York"

    out = reconstruct_table(table, candidate_set, {"new york": "Нью-Йорк"})

    assert out.rows[0][0] == "«Нью-Йорк»"


def test_validate_untranslated_flags_missing_translations():
    coord = TableCoordinate(0, 0)
    candidate_set = CandidateSet(
        candidates=[
            TableCandidate(
                id="c1", lemma_key="new york", source_text="New York",
                coordinates=[coord], length=2,
            ),
        ],
        runs={coord: [ContentRun(lemma_key="new york", start=0, end=8)]},
    )

    assert validate_untranslated(candidate_set, {}) == [f"{coord}: 'new york' not translated"]
    assert validate_untranslated(candidate_set, {"new york": "Нью-Йорк"}) == []


def test_validate_untranslated_ignores_runs_without_candidate():
    coord = TableCoordinate(0, 0)
    candidate_set = CandidateSet(
        candidates=[],
        runs={coord: [ContentRun(lemma_key="2010", start=0, end=4)]},
    )
    assert validate_untranslated(candidate_set, {}) == []


@pytest.mark.asyncio
async def test_in_memory_cache_avoids_repeat_llm_calls(tmp_path):
    mapping = {"New": "Нью"}
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="mem1", lemma_key="new", source_text="New", coordinates=[], length=1),
    ]

    await translate_table_candidates(
        provider, candidates, config=_cfg(tmp_path), source_language="en", target_language="ru"
    )
    assert len(provider.tasks) == 1

    provider2 = CandidateProvider(mapping)
    candidates2 = [
        TableCandidate(id="mem1", lemma_key="new", source_text="New", coordinates=[], length=1),
    ]
    out = await translate_table_candidates(
        provider2, candidates2, config=_cfg(tmp_path), source_language="en", target_language="ru"
    )
    assert out == {"mem1": "Нью"}
    assert provider2.tasks == []


@pytest.mark.asyncio
async def test_sqlite_cache_avoids_repeat_llm_calls(tmp_path):
    cache = SQLiteCache(tmp_path / "cache.sqlite")
    mapping = {"New": "Нью"}
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="sql1", lemma_key="new", source_text="New", coordinates=[], length=1),
    ]

    await translate_table_candidates(
        provider, candidates, config=_cfg(tmp_path),
        source_language="en", target_language="ru", cache=cache,
    )
    assert len(provider.tasks) == 1

    # Чистим in-memory кэш, чтобы проверить именно SQLiteCache.
    tc._MEMORY_CACHE.clear()
    provider2 = CandidateProvider(mapping)
    candidates2 = [
        TableCandidate(id="sql1", lemma_key="new", source_text="New", coordinates=[], length=1),
    ]
    out = await translate_table_candidates(
        provider2, candidates2, config=_cfg(tmp_path),
        source_language="en", target_language="ru", cache=cache,
    )
    assert out == {"sql1": "Нью"}
    assert provider2.tasks == []


# --- Тесты батчинга кандидатов внутри яруса (Задача 6, Шаг 2) ---


class ChunkProvider:
    """Записывающий провайдер: считает параллельные generate и умеет «портить» chunk.

    Если тексты всех items запроса входят в fail_for — возвращается не-JSON мусор,
    что провоцирует retry/ошибку именно этого chunk'а.
    """

    def __init__(self, mapping, fail_for=()):
        self.mapping = mapping
        self.fail_for = set(fail_for)
        self.requests = []
        self.active = 0
        self.max_active = 0

    @property
    def capabilities(self):
        class Caps:
            seed = True
        return Caps()

    async def generate(self, **kwargs):
        payload = json.loads(kwargs["user"])
        self.requests.append(payload)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            # Точка переключения задач, чтобы несколько chunk-запросов реально
            # находились «в полёте» одновременно и замер max_active был честным.
            await asyncio.sleep(0)
            items = payload["items"]
            if any(v["text"] in self.fail_for for v in items):
                return "garbage, not a json object"
            return json.dumps(
                {
                    "translations": [
                        {"id": v["id"], "translation": self.mapping[v["text"]]}
                        for v in items
                    ]
                },
                ensure_ascii=False,
            )
        finally:
            self.active -= 1

    async def count_tokens(self, text: str) -> int:
        return len(text.split())


def _chunk_cfg(tmp_path, *, max_per_request, request_concurrency, retries=0):
    """Конфиг с translation.candidates для тестов батчинга."""
    return AppConfig.model_validate({
        "project": {
            "input_root": str(tmp_path),
            "output_root": str(tmp_path / "out"),
            "source_language": "en",
            "target_language": "ru",
        },
        "model": {"retries": retries},
        "translation": {
            "candidates": {
                "max_candidates_per_request": max_per_request,
                "request_concurrency": request_concurrency,
            }
        },
        "cache": {"enabled": False},
    })


def _three_candidates():
    return [
        TableCandidate(id="ca", lemma_key="a", source_text="A", coordinates=[], length=1),
        TableCandidate(id="cb", lemma_key="b", source_text="B", coordinates=[], length=1),
        TableCandidate(id="cc", lemma_key="c", source_text="C", coordinates=[], length=1),
    ]


@pytest.mark.asyncio
async def test_candidates_chunked_into_two_parallel_requests(tmp_path):
    """(а) 3 кандидата, limit=2, concurrency=2 → 2 непересекающихся chunk'а,
    покрывающих всех кандидатов, в полёте одновременно <= 2 запросов."""
    mapping = {"A": "ТA", "B": "ТB", "C": "ТC"}
    provider = ChunkProvider(mapping)
    candidates = _three_candidates()

    await translate_table_candidates(
        provider,
        candidates,
        config=_chunk_cfg(tmp_path, max_per_request=2, request_concurrency=2),
        source_language="en",
        target_language="ru",
    )

    assert len(provider.requests) == 2  # 2 chunk'а = 2 запроса
    id_sets = [{v["id"] for v in req["items"]} for req in provider.requests]
    assert sorted(len(s) for s in id_sets) == [1, 2]
    assert id_sets[0].isdisjoint(id_sets[1])
    assert set().union(*id_sets) == {"ca", "cb", "cc"}
    assert provider.max_active == 2  # оба chunk'а реально летели параллельно
    assert provider.max_active <= 2


@pytest.mark.asyncio
async def test_chunked_translations_applied_and_returned(tmp_path):
    """(б) переводы всех chunk'ов собраны, записаны в candidates и возвращены."""
    mapping = {"A": "ТA", "B": "ТB", "C": "ТC", "D": "ТD"}
    provider = ChunkProvider(mapping)
    candidates = [
        TableCandidate(id="c1", lemma_key="a", source_text="A", coordinates=[], length=1),
        TableCandidate(id="c2", lemma_key="b", source_text="B", coordinates=[], length=1),
        TableCandidate(id="c3", lemma_key="c", source_text="C", coordinates=[], length=1),
        TableCandidate(id="c4", lemma_key="d", source_text="D", coordinates=[], length=1),
    ]

    out = await translate_table_candidates(
        provider,
        candidates,
        config=_chunk_cfg(tmp_path, max_per_request=2, request_concurrency=2),
        source_language="en",
        target_language="ru",
    )

    assert out == {"c1": "ТA", "c2": "ТB", "c3": "ТC", "c4": "ТD"}
    assert [c.translation for c in candidates] == ["ТA", "ТB", "ТC", "ТD"]
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_chunking_request_concurrency_one_is_sequential(tmp_path):
    """(в) concurrency=1 → запросы идут строго последовательно, покрытие полное."""
    mapping = {"A": "ТA", "B": "ТB", "C": "ТC"}
    provider = ChunkProvider(mapping)
    candidates = _three_candidates()

    out = await translate_table_candidates(
        provider,
        candidates,
        config=_chunk_cfg(tmp_path, max_per_request=2, request_concurrency=1),
        source_language="en",
        target_language="ru",
    )

    assert out == {"ca": "ТA", "cb": "ТB", "cc": "ТC"}
    assert len(provider.requests) == 2
    assert provider.max_active == 1  # последовательно, не больше одного в полёте
    id_sets = [{v["id"] for v in req["items"]} for req in provider.requests]
    assert set().union(*id_sets) == {"ca", "cb", "cc"}


@pytest.mark.asyncio
async def test_invalid_chunk_fails_only_itself_valid_chunk_applied(tmp_path):
    """(г) мусор в первом chunk ретраится только в нём; валидный chunk успешен
    и его переводы применены, ошибка всплывает после исчерпания ретраев."""
    mapping = {"A": "ТA", "B": "ТB", "C": "ТC"}
    # Первый chunk (A, B) всегда возвращает мусор, второй (C) — валидный ответ.
    provider = ChunkProvider(mapping, fail_for={"A", "B"})
    candidates = _three_candidates()

    with pytest.raises(RuntimeError, match="Candidate translation failed after retries"):
        await translate_table_candidates(
            provider,
            candidates,
            config=_chunk_cfg(
                tmp_path, max_per_request=2, request_concurrency=2, retries=2
            ),
            source_language="en",
            target_language="ru",
        )

    # Валидный chunk применён, невалидный — нет.
    by_id = {c.id: c for c in candidates}
    assert by_id["cc"].translation == "ТC"
    assert by_id["ca"].translation == ""
    assert by_id["cb"].translation == ""

    # Невалидный chunk ретраился retries+1 раз (3 попытки), валидный — ровно 1.
    bad = [r for r in provider.requests if "A" in {v["text"] for v in r["items"]}]
    good = [r for r in provider.requests if "A" not in {v["text"] for v in r["items"]}]
    assert len(bad) == 3
    assert len(good) == 1
    assert [v["id"] for v in good[0]["items"]] == ["cc"]


# --- Глоссарий item'а: самые длинные вмещающие последовательности (per-item) ---


@pytest.mark.asyncio
async def test_glossary_only_longest_containing(tmp_path):
    """Ярусы обрабатываются от большего к меньшему; глоссарий каждого item'а —
    переводы самых длинных более длинных последовательностей, в которые значение
    вложено. Несвязанные более короткие кандидаты (Name/Born) не входят, вложенные
    более короткие (York City) отбрасываются в пользу самого длинного вмещающего."""
    mapping = {
        "New": "Нью",
        "York": "Йорк",
        "City": "Сити",
        "Name": "Имя",
        "Born": "Родился",
        "New York": "Нью-Йорк",
        "York City": "Йорк-Сити",
        "New York City": "Нью-Йорк-Сити",
    }
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="nyc-new", lemma_key="new", source_text="New", coordinates=[], length=1),
        TableCandidate(id="nyc-york", lemma_key="york", source_text="York", coordinates=[], length=1),
        TableCandidate(id="nyc-city", lemma_key="city", source_text="City", coordinates=[], length=1),
        TableCandidate(id="nyc-name", lemma_key="name", source_text="Name", coordinates=[], length=1),
        TableCandidate(id="nyc-born", lemma_key="born", source_text="Born", coordinates=[], length=1),
        TableCandidate(id="nyc-new-york", lemma_key="new york", source_text="New York", coordinates=[], length=2),
        TableCandidate(id="nyc-york-city", lemma_key="york city", source_text="York City", coordinates=[], length=2),
        TableCandidate(id="nyc-full", lemma_key="new york city", source_text="New York City", coordinates=[], length=3),
    ]

    await translate_table_candidates(
        provider,
        candidates,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
    )

    assert len(provider.tasks) == 3  # ярус 3, 2, 1 — по одному запросу на ярус
    # Ярус 3 первым, глоссарий пуст (нет более длинных).
    tier3 = provider.tasks[0]
    assert [item["id"] for item in tier3["items"]] == ["nyc-full"]
    assert tier3["items"][0]["glossary"] == []

    # Ярус 2: у каждого значения самый длинный вмещающий = New York City.
    tier2 = _items_by_id([provider.tasks[1]])
    for cid in ("nyc-new-york", "nyc-york-city"):
        assert tier2[cid]["glossary"] == [
            {"key": "New York City", "translation": "Нью-Йорк-Сити"}
        ]

    # Ярус 1: new/york/city → New York City; несвязанные name/born — пусто.
    tier1 = _items_by_id(provider.tasks[2:])
    for cid in ("nyc-new", "nyc-york", "nyc-city"):
        assert tier1[cid]["glossary"] == [
            {"key": "New York City", "translation": "Нью-Йорк-Сити"}
        ]
    for cid in ("nyc-name", "nyc-born"):
        assert tier1[cid]["glossary"] == []


@pytest.mark.asyncio
async def test_glossary_dedups_equal_longest_containing(tmp_path):
    """При нескольких одинаково длинных вмещающих (и дубликатах) включаются все
    максимально длинные с дедупом по (source_text, translation), порядок — по
    порядку более длинных кандидатов."""
    mapping = {
        "A": "А",
        "A B C": "А-Б-В",
        "X A Y": "Х-А-Ы",
    }
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="va", lemma_key="a", source_text="A", coordinates=[], length=1),
        # Два одинаковых вмещающих (дубликат по source_text/translation).
        TableCandidate(id="vabc", lemma_key="a b c", source_text="A B C", coordinates=[], length=3),
        TableCandidate(id="vabc2", lemma_key="a b c", source_text="A B C", coordinates=[], length=3),
        TableCandidate(id="vxay", lemma_key="x a y", source_text="X A Y", coordinates=[], length=3),
    ]

    await translate_table_candidates(
        provider,
        candidates,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
    )

    # `a` вложено в `a b c` (дважды) и `x a y` — все длины 3 максимальны.
    va = _items_by_id(provider.tasks)["va"]
    assert va["glossary"] == [
        {"key": "A B C", "translation": "А-Б-В"},
        {"key": "X A Y", "translation": "Х-А-Ы"},
    ]


@pytest.mark.asyncio
async def test_glossary_per_item_across_values_in_same_request(tmp_path):
    """Два значения одного яруса в одном запросе получают каждый свой глоссарий —
    собственный самый длинный вмещающий, а не общий для запроса."""
    mapping = {
        "City": "Сити",
        "State": "Штат",
        "York City": "Йорк-Сити",
        "York State": "Йорк-Штат",
        "New York City": "Нью-Йорк-Сити",
        "New York State": "Нью-Йорк-Штат",
    }
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="g-city", lemma_key="city", source_text="City", coordinates=[], length=1),
        TableCandidate(id="g-state", lemma_key="state", source_text="State", coordinates=[], length=1),
        TableCandidate(id="g-york-city", lemma_key="york city", source_text="York City", coordinates=[], length=2),
        TableCandidate(id="g-york-state", lemma_key="york state", source_text="York State", coordinates=[], length=2),
        TableCandidate(id="g-nyc", lemma_key="new york city", source_text="New York City", coordinates=[], length=3),
        TableCandidate(id="g-nys", lemma_key="new york state", source_text="New York State", coordinates=[], length=3),
    ]

    await translate_table_candidates(
        provider,
        candidates,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
    )

    tier1 = _items_by_id(provider.tasks[2:])
    # city → New York City; state → New York State; каждый item со своим глоссарием.
    assert tier1["g-city"]["glossary"] == [
        {"key": "New York City", "translation": "Нью-Йорк-Сити"}
    ]
    assert tier1["g-state"]["glossary"] == [
        {"key": "New York State", "translation": "Нью-Йорк-Штат"}
    ]


@pytest.mark.asyncio
async def test_chunk_glossary_isolated_per_value(tmp_path):
    """max_candidates_per_request=1: каждый chunk — отдельный запрос, глоссарий
    item'а считается независимо от других значений (свой самый длинный вмещающий)."""
    mapping = {
        "New": "Нью",
        "Los": "Лос",
        "New York": "Нью-Йорк",
        "Los Angeles": "Лос-Анджелес",
        "New York City": "Нью-Йорк-Сити",
    }
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="n-new", lemma_key="new", source_text="New", coordinates=[], length=1),
        TableCandidate(id="n-los", lemma_key="los", source_text="Los", coordinates=[], length=1),
        TableCandidate(id="n-new-york", lemma_key="new york", source_text="New York", coordinates=[], length=2),
        TableCandidate(id="n-la", lemma_key="los angeles", source_text="Los Angeles", coordinates=[], length=2),
        TableCandidate(id="n-nyc", lemma_key="new york city", source_text="New York City", coordinates=[], length=3),
    ]

    await translate_table_candidates(
        provider,
        candidates,
        config=_chunk_cfg(tmp_path, max_per_request=1, request_concurrency=1),
        source_language="en",
        target_language="ru",
    )

    # Ярус 1 — 2 запроса (по одному значению). Глоссарии независимы.
    single_value_glossaries = {
        task["items"][0]["text"]: task["items"][0]["glossary"]
        for task in provider.tasks
        if len(task["items"]) == 1
    }
    assert single_value_glossaries["New"] == [
        {"key": "New York City", "translation": "Нью-Йорк-Сити"}
    ]
    assert single_value_glossaries["Los"] == [
        {"key": "Los Angeles", "translation": "Лос-Анджелес"}
    ]


# --- column-контекст и пустые глоссарии ---


@pytest.mark.asyncio
async def test_item_column_uses_first_coordinate_header(tmp_path):
    """`column` — полный заголовок колонки первой координаты: для ячейки — её
    колонки, для кандидата-заголовка — сам заголовок, для multi-column — первой
    координаты."""
    mapping = {"City": "Город", "New York": "Нью-Йорк"}
    provider = CandidateProvider(mapping)
    headers = ["Name", "City"]
    candidates = [
        # Кандидат-заголовок: lemma_key == заголовку колонки, координата (None, 1).
        TableCandidate(id="h1", lemma_key="city", source_text="City", coordinates=[TableCoordinate(None, 1)], length=1),
        # Ячейка в колонке 1 (и повторе в той же колонке — multi-column).
        TableCandidate(id="cell", lemma_key="new york", source_text="New York", coordinates=[TableCoordinate(0, 1), TableCoordinate(2, 1)], length=2),
    ]

    await translate_table_candidates(
        provider,
        candidates,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
        headers=headers,
    )

    by_id = _items_by_id(provider.tasks)
    assert by_id["h1"]["column"] == "City"     # кандидат-заголовок → сам заголовок
    assert by_id["cell"]["column"] == "City"   # первая координата col=1


@pytest.mark.asyncio
async def test_item_column_empty_when_no_coordinates(tmp_path):
    """Fallback для пустых coordinates (unit-тесты): column == ""."""
    mapping = {"York": "Йорк"}
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="c", lemma_key="york", source_text="York", coordinates=[], length=1),
    ]

    await translate_table_candidates(
        provider,
        candidates,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
        headers=["City"],
    )

    assert provider.tasks[0]["items"][0]["column"] == ""


@pytest.mark.asyncio
async def test_glossary_empty_when_value_not_nested(tmp_path):
    """Пустой глоссарий, когда значение не вложено ни в одну более длинную
    последовательность."""
    mapping = {"New": "Нью", "York": "Йорк", "Alice": "Алиса", "New York": "Нью-Йорк"}
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="alice", lemma_key="alice", source_text="Alice", coordinates=[], length=1),
        TableCandidate(id="a-new", lemma_key="new", source_text="New", coordinates=[], length=1),
        TableCandidate(id="a-york", lemma_key="york", source_text="York", coordinates=[], length=1),
        TableCandidate(id="new-york", lemma_key="new york", source_text="New York", coordinates=[], length=2),
    ]

    await translate_table_candidates(
        provider,
        candidates,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
    )

    by_id = _items_by_id(provider.tasks)
    # Alice не вложена ни в одну более длинную последовательность.
    assert by_id["alice"]["glossary"] == []
