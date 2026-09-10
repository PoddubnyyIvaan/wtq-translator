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
    """Mock-провайдер: парсит user-JSON и возвращает переводы по входным values."""

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
                    for item in payload["values"]
                ]
            },
            ensure_ascii=False,
        )


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

    # Порядок: ярус длины 1 раньше яруса длины 2.
    assert [item["id"] for item in provider.tasks[0]["values"]] == ["c1", "c2"]
    assert [item["id"] for item in provider.tasks[1]["values"]] == ["c3"]
    # Ярус 2 получает глоссарий переводов яруса 1.
    assert provider.tasks[0]["glossary"] == []
    assert provider.tasks[1]["glossary"] == [
        {"key": "New", "translation": "Нью"},
        {"key": "York", "translation": "Йорк"},
    ]


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

    Если тексты всех values запроса входят в fail_for — возвращается не-JSON мусор,
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
            values = payload["values"]
            if any(v["text"] in self.fail_for for v in values):
                return "garbage, not a json object"
            return json.dumps(
                {
                    "translations": [
                        {"id": v["id"], "translation": self.mapping[v["text"]]}
                        for v in values
                    ]
                },
                ensure_ascii=False,
            )
        finally:
            self.active -= 1


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
    id_sets = [{v["id"] for v in req["values"]} for req in provider.requests]
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
    id_sets = [{v["id"] for v in req["values"]} for req in provider.requests]
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
    bad = [r for r in provider.requests if "A" in {v["text"] for v in r["values"]}]
    good = [r for r in provider.requests if "A" not in {v["text"] for v in r["values"]}]
    assert len(bad) == 3
    assert len(good) == 1
    assert [v["id"] for v in good[0]["values"]] == ["cc"]


# --- Глоссарий релевантен значениям chunk-запроса (per-chunk glossary) ---


@pytest.mark.asyncio
async def test_glossary_excludes_unrelated_shorter_candidates(tmp_path):
    """Ярус 3: глоссарий запроса содержит переводы непересекающихся по токенам
    окон фразы `new york city` — из пересекающихся остаётся самое длинное (при
    равенстве — первое/левое), вложенные удаляются; несвязанные короткие
    кандидаты Name/Born (ярус 1) в глоссарий не попадают."""
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

    out = await translate_table_candidates(
        provider,
        candidates,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
    )

    assert out == {
        "nyc-new": "Нью",
        "nyc-york": "Йорк",
        "nyc-city": "Сити",
        "nyc-name": "Имя",
        "nyc-born": "Родился",
        "nyc-new-york": "Нью-Йорк",
        "nyc-york-city": "Йорк-Сити",
        "nyc-full": "Нью-Йорк-Сити",
    }
    # Ярусы 1, 2 и 3 обрабатываются последовательно — по одному запросу на ярус.
    assert len(provider.tasks) == 3

    # Payload последнего запроса (ярус 3): единственное значение длины 3.
    tier3 = provider.tasks[-1]
    assert [item["id"] for item in tier3["values"]] == ["nyc-full"]
    # Глоссарий — непересекающиеся окна `new york city`: выбраны пара `new york`
    # (длина 2, первое/левое) и не пересекающийся с ней одиночный `city`.
    # `york city` (равной длины, пересекается с `new york`), вложенные `new`/`york`
    # и несвязанные `Name`/`Born` исключены.
    assert tier3["glossary"] == [
        {"key": "City", "translation": "Сити"},
        {"key": "New York", "translation": "Нью-Йорк"},
    ]


@pytest.mark.asyncio
async def test_glossary_keeps_non_overlapping_longest_windows(tmp_path):
    """Ярус 4, фраза `a b c d`: выбираются непересекающиеся по токенам самые
    длинные окна — первая тройка `a b c` и не пересекающийся с ней хвост `d`.
    Вложенные в тройку одиночки и пары `a b`/`b c`, а также пересекающаяся
    с тройкой пара `c d` в глоссарий не попадают."""
    mapping = {
        "A": "ТA",
        "B": "ТB",
        "C": "ТC",
        "D": "ТD",
        "A B": "ТA-B",
        "B C": "ТB-C",
        "C D": "ТC-D",
        "A B C": "ТA-B-C",
        "A B C D": "ТA-B-C-D",
    }
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="w-a", lemma_key="a", source_text="A", coordinates=[], length=1),
        TableCandidate(id="w-b", lemma_key="b", source_text="B", coordinates=[], length=1),
        TableCandidate(id="w-c", lemma_key="c", source_text="C", coordinates=[], length=1),
        TableCandidate(id="w-d", lemma_key="d", source_text="D", coordinates=[], length=1),
        TableCandidate(id="w-a-b", lemma_key="a b", source_text="A B", coordinates=[], length=2),
        TableCandidate(id="w-b-c", lemma_key="b c", source_text="B C", coordinates=[], length=2),
        TableCandidate(id="w-c-d", lemma_key="c d", source_text="C D", coordinates=[], length=2),
        TableCandidate(id="w-a-b-c", lemma_key="a b c", source_text="A B C", coordinates=[], length=3),
        TableCandidate(id="w-full", lemma_key="a b c d", source_text="A B C D", coordinates=[], length=4),
    ]

    await translate_table_candidates(
        provider,
        candidates,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
    )

    # Payload последнего запроса (ярус 4): единственное значение длины 4.
    tier4 = provider.tasks[-1]
    assert [item["id"] for item in tier4["values"]] == ["w-full"]
    # Выбраны `a b c` + `d`; порядок записей — по порядку более коротких кандидатов.
    assert tier4["glossary"] == [
        {"key": "D", "translation": "ТD"},
        {"key": "A B C", "translation": "ТA-B-C"},
    ]
    keys = {item["key"] for item in tier4["glossary"]}
    # Ни одно окно в глоссарии не пересекается с выбранной тройкой по позициям.
    assert keys.isdisjoint({"A", "B", "C", "A B", "B C", "C D"})


@pytest.mark.asyncio
async def test_chunk_glossary_union_across_values(tmp_path):
    """Ярус 3, значения `new york city` и `new york state` в одном запросе:
    непересекающиеся окна выбираются в пределах каждой фразы и объединяются.
    Общее окно `new york` не «перетекает» между фразами, вложенные/пересекающиеся
    окна (`new`, `york`, `york city`, `york state`) исключены."""
    mapping = {
        "New": "Нью",
        "York": "Йорк",
        "City": "Сити",
        "State": "Штат",
        "New York": "Нью-Йорк",
        "York City": "Йорк-Сити",
        "York State": "Йорк-Штат",
        "New York City": "Нью-Йорк-Сити",
        "New York State": "Нью-Йорк-Штат",
    }
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="u-new", lemma_key="new", source_text="New", coordinates=[], length=1),
        TableCandidate(id="u-york", lemma_key="york", source_text="York", coordinates=[], length=1),
        TableCandidate(id="u-city", lemma_key="city", source_text="City", coordinates=[], length=1),
        TableCandidate(id="u-state", lemma_key="state", source_text="State", coordinates=[], length=1),
        TableCandidate(id="u-new-york", lemma_key="new york", source_text="New York", coordinates=[], length=2),
        TableCandidate(id="u-york-city", lemma_key="york city", source_text="York City", coordinates=[], length=2),
        TableCandidate(id="u-york-state", lemma_key="york state", source_text="York State", coordinates=[], length=2),
        TableCandidate(id="u-nyc", lemma_key="new york city", source_text="New York City", coordinates=[], length=3),
        TableCandidate(id="u-nys", lemma_key="new york state", source_text="New York State", coordinates=[], length=3),
    ]

    await translate_table_candidates(
        provider,
        candidates,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
    )

    # Ярус 3 — один chunk с обоими значениями.
    tier3 = provider.tasks[-1]
    assert [item["id"] for item in tier3["values"]] == ["u-nyc", "u-nys"]
    # Объединение per-phrase выборов: `new york` + `city` и `new york` + `state`.
    assert tier3["glossary"] == [
        {"key": "City", "translation": "Сити"},
        {"key": "State", "translation": "Штат"},
        {"key": "New York", "translation": "Нью-Йорк"},
    ]
    keys = {item["key"] for item in tier3["glossary"]}
    assert keys.isdisjoint({"New", "York", "York City", "York State"})


@pytest.mark.asyncio
async def test_chunk_glossary_only_relevant_to_chunk_values(tmp_path):
    """Ярус 2, max_candidates_per_request=1: глоссарий считается per-chunk —
    chunk `New York` получает только переводы New/York, chunk `Los Angeles` —
    только Los/Angeles, а не общий глоссарий всего яруса."""
    mapping = {
        "New": "Нью",
        "York": "Йорк",
        "Los": "Лос",
        "Angeles": "Анджелес",
        "New York": "Нью-Йорк",
        "Los Angeles": "Лос-Анджелес",
    }
    provider = CandidateProvider(mapping)
    candidates = [
        TableCandidate(id="ny-new", lemma_key="new", source_text="New", coordinates=[], length=1),
        TableCandidate(id="ny-york", lemma_key="york", source_text="York", coordinates=[], length=1),
        TableCandidate(id="la-los", lemma_key="los", source_text="Los", coordinates=[], length=1),
        TableCandidate(id="la-angeles", lemma_key="angeles", source_text="Angeles", coordinates=[], length=1),
        TableCandidate(id="ny-full", lemma_key="new york", source_text="New York", coordinates=[], length=2),
        TableCandidate(id="la-full", lemma_key="los angeles", source_text="Los Angeles", coordinates=[], length=2),
    ]

    out = await translate_table_candidates(
        provider,
        candidates,
        config=_chunk_cfg(tmp_path, max_per_request=1, request_concurrency=1),
        source_language="en",
        target_language="ru",
    )

    assert out == {
        "ny-new": "Нью",
        "ny-york": "Йорк",
        "la-los": "Лос",
        "la-angeles": "Анджелес",
        "ny-full": "Нью-Йорк",
        "la-full": "Лос-Анджелес",
    }
    # Ярус 1: 4 кандидата → 4 chunk'а; ярус 2: 2 кандидата → 2 chunk'а.
    assert len(provider.tasks) == 6

    single_value_glossaries = {
        task["values"][0]["text"]: task["glossary"]
        for task in provider.tasks
        if len(task["values"]) == 1
    }
    # У каждого chunk'а яруса 2 свой глоссарий — только подстроки своего значения.
    assert single_value_glossaries["New York"] == [
        {"key": "New", "translation": "Нью"},
        {"key": "York", "translation": "Йорк"},
    ]
    assert single_value_glossaries["Los Angeles"] == [
        {"key": "Los", "translation": "Лос"},
        {"key": "Angeles", "translation": "Анджелес"},
    ]
