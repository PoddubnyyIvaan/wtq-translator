"""Проверка SQLite-хранилищ на конкурентную работу (WAL + busy_timeout).

Задача 2 плана: и state, и cache должны переживать конкурентные записи
из разных корутин без "database is locked".
"""

import asyncio
import sqlite3

import pytest

from wtq_translator.enums import ProcessingStatus
from wtq_translator.persistence.cache import SQLiteCache
from wtq_translator.persistence.state import StateStore


@pytest.mark.parametrize(
    "make_store",
    [
        pytest.param(lambda p: StateStore(p), id="state"),
        pytest.param(lambda p: SQLiteCache(p), id="cache"),
    ],
)
def test_journal_mode_is_wal(tmp_path, make_store):
    """После инициализации хранилища свежее соединение видит режим WAL."""
    path = tmp_path / "store.sqlite"
    make_store(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.asyncio
async def test_state_store_concurrent_access(tmp_path):
    """Конкурентные записи/чтения StateStore из ~10 корутин не падают."""
    store = StateStore(tmp_path / "state.sqlite")
    run_id = "run-concurrent"
    workers = 10
    iterations = 10

    async def worker(i: int) -> None:
        for j in range(iterations):
            qid = f"q-{i}-{j}"
            store.set_question(run_id, qid, ProcessingStatus.PROCESSING, "full")
            state = store.get_question(run_id, qid)
            assert state is not None and state.question_id == qid
            store.save_quality(run_id, qid, {"worker": i, "iter": j})
            store.status_counts(run_id)
            await asyncio.sleep(0)

    await asyncio.gather(*(worker(i) for i in range(workers)))

    ids = [q.question_id for q in store.list_questions(run_id)]
    assert len(ids) == workers * iterations
    assert set(ids) == {f"q-{i}-{j}" for i in range(workers) for j in range(iterations)}


@pytest.mark.asyncio
async def test_sqlite_cache_concurrent_put_get(tmp_path):
    """Конкурентные put/get SQLiteCache из ~10 корутин не падают."""
    cache = SQLiteCache(tmp_path / "cache.sqlite")
    workers = 10
    iterations = 10

    async def worker(i: int) -> None:
        for j in range(iterations):
            key = f"k-{i}-{j}"
            value = f"v-{i}-{j}"
            cache.put_text(key, value)
            assert cache.get_text(key) == value
            await asyncio.sleep(0)

    await asyncio.gather(*(worker(i) for i in range(workers)))
