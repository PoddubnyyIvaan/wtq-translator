"""Тесты отсева вопросов по алфавитно-зависимому намерению (alphabet_screen).

Фейковый LLM-провайдер возвращает намерение по подстроке вопроса (как в
test_alphabet_detection.py); run_alphabet_screen принимает провайдер напрямую
(параметр llm), поэтому реальный HTTP не используется.
"""
import csv
import json
from pathlib import Path

import pytest

from wtq_translator.alphabet_screen import run_alphabet_screen
from wtq_translator.config import AppConfig
from wtq_translator.dataset import read_question_tsv_document


class FakeProvider:
    """Если вопрос содержит 'alphabet'/'letter' — намерение 1/4, иначе None."""

    def __init__(self, *, fail_on=()):
        self.fail_on = set(fail_on)
        self.calls = []

    async def generate(self, **kwargs):
        payload = json.loads(kwargs["user"])
        self.calls.append(payload)
        qid = payload["question_id"]
        question = payload["question"]
        if qid in self.fail_on:
            return "not json at all"
        if "alphabet" in question:
            return json.dumps({"intent_id": 1, "why": "нужна алфавитная сортировка"}, ensure_ascii=False)
        if "letter" in question:
            return json.dumps({"intent_id": 4, "why": "вопрос про букву"}, ensure_ascii=False)
        return json.dumps({"intent_id": None, "why": None}, ensure_ascii=False)

    async def count_tokens(self, text: str) -> int:
        return len(text.split())


def _config(tmp_path: Path, *, concurrency: int = 2, retries: int = 0, method: str | None = None) -> AppConfig:
    data = {
        "project": {
            "input_root": str(tmp_path / "input"),
            "output_root": str(tmp_path / "output"),
            "source_language": "en",
            "target_language": "ru",
            "seed": 42,
        },
        "translation": {"mode": "full", "answer_language_mode": "table"},
        "model": {"retries": retries, "temperature": 0.0, "max_output_tokens": 256},
        "embeddings": {"enabled": False},
        "back_translation": {"enabled": True},
        "table": {"rows_per_chunk": 20},
        "cache": {"enabled": False},
        "logging": {"llm_io_enabled": False},
    }
    if method is not None:
        data["alphabet_detection"] = {"method": method}
    return AppConfig.model_validate(data)


def _questions(tmp_path: Path) -> Path:
    path = tmp_path / "input" / "train.tsv"
    path.parent.mkdir(parents=True)
    rows = [
        ("id", "utterance", "context", "targetValue", "targetCanon"),
        ("q-1", "Sort by alphabet", "tables/t1.tsv", "A", "A"),
        ("q-2", "Count rows with letter x", "tables/t1.tsv", "2", "2"),
        ("q-3", "How many cities?", "tables/t1.tsv", "3", "3"),
        ("q-4", "What is the most populous city?", "tables/t1.tsv", "A", "A"),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerows(rows)
    return path


@pytest.mark.asyncio
async def test_screen_splits_kept_and_excluded(tmp_path):
    """Вопросы с намерением исключаются (с class/why), без намерения — остаются."""
    questions = _questions(tmp_path)
    cfg = _config(tmp_path)
    provider = FakeProvider()

    stats = await run_alphabet_screen(cfg, questions, llm=provider)

    assert stats["total"] == 4
    assert stats["kept"] == 2
    assert stats["excluded"] == 2
    assert stats["excluded_share"] == pytest.approx(0.5)
    assert stats["detection_errors"] == 0
    assert stats["intent_classes"] == {"алфавитная сортировка": 1, "подсчёт, наличие/отсутствие или сравнение по конкретной букве или множеству букв": 1}
    assert len(provider.calls) == 4

    # kept TSV: только переводимые, с original поле targetCanon сохраняется.
    kept_doc = read_question_tsv_document(Path(stats["kept_tsv"]))
    assert [r.id for r in kept_doc.records] == ["q-3", "q-4"]
    assert kept_doc.records[0].extra_fields["targetCanon"] == "3"

    # excluded TSV: q-1 (алфавитная сортировка), q-2 (буква).
    with open(stats["alphabet_excluded_tsv"], encoding="utf-8", newline="") as f:
        excluded = list(csv.DictReader(f, delimiter="\t"))
    assert [r["id"] for r in excluded] == ["q-1", "q-2"]
    assert excluded[0]["class"] == "алфавитная сортировка"
    assert excluded[1]["why"] == "вопрос про букву"


@pytest.mark.asyncio
async def test_screen_regex_method_skips_llm(tmp_path):
    """method='regex': детерминированный отсев без вызова LLM, why в формате 'regex: …'."""
    questions = _questions(tmp_path)
    cfg = _config(tmp_path, method="regex")
    provider = FakeProvider()

    stats = await run_alphabet_screen(cfg, questions, llm=provider)

    assert stats["total"] == 4
    assert stats["kept"] == 2
    assert stats["excluded"] == 2
    assert stats["detection_errors"] == 0
    assert stats["intent_classes"] == {"алфавитная сортировка": 1, "подсчёт, наличие/отсутствие или сравнение по конкретной букве или множеству букв": 1}
    assert provider.calls == []

    with open(stats["alphabet_excluded_tsv"], encoding="utf-8", newline="") as f:
        excluded = list(csv.DictReader(f, delimiter="\t"))
    assert [r["id"] for r in excluded] == ["q-1", "q-2"]
    assert excluded[0]["class"] == "алфавитная сортировка"
    assert excluded[0]["why"].startswith("regex:")
    assert excluded[1]["why"].startswith("regex:")


@pytest.mark.asyncio
async def test_screen_fail_closed_writes_empty_class(tmp_path):
    """Провал детекции после всех ретраев → вопрос исключён с class='' и why=ошибка."""
    questions = _questions(tmp_path)
    cfg = _config(tmp_path, concurrency=1, retries=2)
    provider = FakeProvider(fail_on={"q-3"})

    stats = await run_alphabet_screen(cfg, questions, llm=provider)

    assert stats["excluded"] == 3
    assert stats["kept"] == 1
    assert stats["detection_errors"] == 1

    with open(stats["alphabet_excluded_tsv"], encoding="utf-8", newline="") as f:
        excluded = list(csv.DictReader(f, delimiter="\t"))
    error_row = next(r for r in excluded if r["id"] == "q-3")
    assert error_row["class"] == ""
    assert "Alphabet detection failed" in error_row["why"]


@pytest.mark.asyncio
async def test_screen_stats_json_persisted(tmp_path):
    """Статистика сохраняется в reports/alphabet-screen-stats.json."""
    questions = _questions(tmp_path)
    cfg = _config(tmp_path)
    stats = await run_alphabet_screen(cfg, questions, llm=FakeProvider())

    saved = json.loads(Path(stats["stats_json"]).read_text(encoding="utf-8"))
    assert saved == stats
    assert saved["total"] == 4