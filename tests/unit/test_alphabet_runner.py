"""Тесты обработки исключённых (алфавитно-зависимых) вопросов в run_dataset_from_tsv.

Паттерн FakeTranslator — из test_dataset_runner.py: воркер, подменяющий реальный
Translator; здесь добавляется вариант, бросающий AlphabetIntentError для заданных id.
"""
import csv
import json
import sqlite3
from pathlib import Path

import pytest

from wtq_translator.alphabet import AlphabetIntentError
from wtq_translator.config import AppConfig
from wtq_translator.dataset import read_question_tsv_document
from wtq_translator.enums import ProcessingStatus, TranslationMode
from wtq_translator.models import MentionMap, QualityReport, TableData, TranslationResult
from wtq_translator.runner import run_dataset_from_tsv


def _config(tmp_path: Path, *, concurrency: int = 1, continue_on_error: bool = True) -> AppConfig:
    return AppConfig.model_validate({
        "project": {
            "input_root": str(tmp_path / "input"),
            "output_root": str(tmp_path / "output"),
            "source_language": "en",
            "target_language": "ru",
            "seed": 42,
        },
        "translation": {"mode": "full", "answer_language_mode": "table"},
        "back_translation": {"enabled": True},
        "cache": {"enabled": False},
        "runner": {"checkpoint_every": 1, "continue_on_error": continue_on_error, "concurrency": concurrency},
    })


def _fixture_multi_dataset(tmp_path: Path, n: int = 6):
    """Датасет из n вопросов q-1..q-n к одной и той же таблице."""
    input_root = tmp_path / "input"
    (input_root / "tables").mkdir(parents=True)
    table_path = input_root / "tables" / "t1.tsv"
    table_path.write_text("Name\tCity\nAlice\tLondon\n", encoding="utf-8")
    questions = input_root / "train.tsv"
    with questions.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "utterance", "context", "targetValue", "targetCanon"])
        for i in range(1, n + 1):
            w.writerow([f"q-{i}", "Where is Alice?", "tables/t1.tsv", "London", "London"])
    return questions


class ExcludingTranslator:
    """Тестовый переводчик: для заданных id бросает AlphabetIntentError, для остальных
    возвращает успешный перевод, как FakeTranslator из test_dataset_runner.py."""

    def __init__(self, config, *, exclude_on_ids=(), label="", why=""):
        self.config = config
        self.exclude_on_ids = set(exclude_on_ids)
        self.label = label
        self.why = why
        self.calls = 0

    async def process(self, question, table):
        self.calls += 1
        if question.id in self.exclude_on_ids:
            raise AlphabetIntentError(question.id, intent_label=self.label, why=self.why)
        out = TableData(["Имя", "Город"], [["Алиса", "Лондон"]])
        report = QualityReport(
            question_id=question.id,
            mode=TranslationMode.FULL,
            table_mentions_found=True,
            source_question=question.utterance,
            translated_question="Где Алиса?",
            back_translated_question=question.utterance,
            status=ProcessingStatus.ACCEPTED,
        )
        return TranslationResult(
            question=question,
            source_table=table,
            output_table=out,
            translated_question="Где Алиса?",
            translated_target_value="Лондон",
            quality=report,
            mention_map=MentionMap([]),
        )


class ProgressSpy:
    """Заглушка протокола прогресса: считает update и запоминает последний set_postfix."""

    def __init__(self) -> None:
        self.updates = 0
        self.postfix: dict | None = None

    def update(self, n: int = 1) -> None:
        self.updates += n

    def set_postfix(self, ordered_dict: dict | None = None, **kwargs) -> None:
        self.postfix = dict(ordered_dict or {})
        self.postfix.update(kwargs)


def _read_excluded(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


@pytest.mark.asyncio
async def test_excluded_record_written_to_tsv_and_state(tmp_path):
    """Один вопрос бросает AlphabetIntentError → в датасете его нет, в TSV исключений
    есть с class/why, статус state = excluded, summary.excluded == 1, failed == 0."""
    questions = _fixture_multi_dataset(tmp_path, n=6)
    cfg = _config(tmp_path)
    summary = await run_dataset_from_tsv(
        cfg, questions, resume=False,
        translator_factory=lambda _: ExcludingTranslator(
            cfg, exclude_on_ids={"q-3"}, label="алфавитная сортировка", why="в алфавитном порядке",
        ),
    )

    assert summary.excluded == 1
    assert summary.failed == 0
    assert summary.processed == 5
    assert summary.statuses[ProcessingStatus.EXCLUDED.value] == 1

    doc = read_question_tsv_document(summary.output_questions_tsv)
    assert [r.id for r in doc.records] == [f"q-{i}" for i in range(1, 7) if i != 3]

    excluded_path = summary.run_dir / "dataset" / "train-alphabet-excluded.tsv"
    assert excluded_path.exists()
    rows = _read_excluded(excluded_path)
    assert len(rows) == 1
    assert rows[0]["id"] == "q-3"
    assert rows[0]["utterance"] == "Where is Alice?"
    assert rows[0]["class"] == "алфавитная сортировка"
    assert rows[0]["why"] == "в алфавитном порядке"

    with sqlite3.connect(cfg.state.sqlite_path) as db:
        row = db.execute(
            "SELECT status, output_record_json FROM questions WHERE run_id=? AND question_id=?",
            (summary.run_id, "q-3"),
        ).fetchone()
    assert row is not None
    assert row[0] == ProcessingStatus.EXCLUDED.value
    output_record = json.loads(row[1])
    assert output_record["extra_fields"] == {"class": "алфавитная сортировка", "why": "в алфавитном порядке"}


@pytest.mark.asyncio
async def test_excluded_share_and_stats_json(tmp_path):
    """excluded_share == 1/total; stats.json содержит excluded_share и excluded_this_invocation."""
    questions = _fixture_multi_dataset(tmp_path, n=2)
    cfg = _config(tmp_path)
    summary = await run_dataset_from_tsv(
        cfg, questions, resume=False,
        translator_factory=lambda _: ExcludingTranslator(
            cfg, exclude_on_ids={"q-1"}, label="алфавитная сортировка", why="почему",
        ),
    )

    assert summary.excluded == 1
    assert summary.excluded_share == pytest.approx(1 / 2)
    assert summary.statuses[ProcessingStatus.EXCLUDED.value] == 1

    stats = json.loads((summary.run_dir / "reports" / "stats.json").read_text(encoding="utf-8"))
    assert stats["excluded_this_invocation"] == 1
    assert stats["excluded_share"] == pytest.approx(0.5)
    assert stats["alphabet_excluded_tsv"] == str(
        (summary.run_dir / "dataset" / "train-alphabet-excluded.tsv").resolve()
    )


@pytest.mark.asyncio
async def test_resume_skips_excluded_and_restores_tsv(tmp_path):
    """Resume: исключённый вопрос не переобрабатывается, TSV исключений восстановлен из state."""
    questions = _fixture_multi_dataset(tmp_path, n=4)
    cfg = _config(tmp_path)
    first = await run_dataset_from_tsv(
        cfg, questions, resume=False,
        translator_factory=lambda _: ExcludingTranslator(
            cfg, exclude_on_ids={"q-2"}, label="сравнение по алфавиту", why="раньше по алфавиту",
        ),
    )
    assert first.excluded == 1

    spy = ProgressSpy()
    second_translator = ExcludingTranslator(cfg)  # не бросает — провалит тест, если вызван
    second = await run_dataset_from_tsv(
        cfg, questions, resume=True,
        translator_factory=lambda _: second_translator,
        progress=spy,
    )
    assert second_translator.calls == 0
    assert second.skipped == 4
    assert second.processed == 0
    assert second.excluded == 0
    assert spy.updates == 4

    rows = _read_excluded(second.run_dir / "dataset" / "train-alphabet-excluded.tsv")
    assert len(rows) == 1
    assert rows[0]["id"] == "q-2"
    assert rows[0]["class"] == "сравнение по алфавиту"
    assert rows[0]["why"] == "раньше по алфавиту"


@pytest.mark.asyncio
async def test_fail_closed_excluded_with_empty_class(tmp_path):
    """Ошибка детекции (fail-closed): class пуст, why содержит сообщение об ошибке."""
    questions = _fixture_multi_dataset(tmp_path, n=2)
    cfg = _config(tmp_path)
    summary = await run_dataset_from_tsv(
        cfg, questions, resume=False,
        translator_factory=lambda _: ExcludingTranslator(
            cfg, exclude_on_ids={"q-1"}, label="", why="Alphabet detection failed after retries: boom",
        ),
    )

    assert summary.excluded == 1
    assert summary.failed == 0
    rows = _read_excluded(summary.run_dir / "dataset" / "train-alphabet-excluded.tsv")
    assert len(rows) == 1
    assert rows[0]["class"] == ""
    assert "Alphabet detection failed" in rows[0]["why"]