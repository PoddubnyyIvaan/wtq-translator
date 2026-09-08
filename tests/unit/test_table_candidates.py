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
