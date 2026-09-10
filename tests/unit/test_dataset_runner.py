import asyncio
import csv
import sqlite3
from pathlib import Path

import pytest

from wtq_translator.config import AppConfig
from wtq_translator.dataset import read_question_tsv_document, read_table_tsv
from wtq_translator.enums import ProcessingStatus, TranslationMode
from wtq_translator.models import MentionMap, QualityReport, TableData, TranslationResult
from wtq_translator.runner import make_run_id, run_dataset_from_tsv


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


def _fixture_dataset(tmp_path: Path):
    input_root = tmp_path / "input"
    (input_root / "tables").mkdir(parents=True)
    table_path = input_root / "tables" / "t1.tsv"
    table_path.write_text("Name\tCity\nAlice\tLondon\n", encoding="utf-8")
    questions = input_root / "train.tsv"
    with questions.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "utterance", "context", "targetValue", "targetCanon"])
        w.writerow(["q-1", "Where is Alice?", "tables/t1.tsv", "London", "London"])
    return questions


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


class FakeTranslator:
    """Тестовый переводчик с опциональной задержкой и счётчиками параллелизма.

    Поля для проверки конкурентного пула: calls/called_ids (число и состав вызовов),
    active/max_active (одновременно выполняющиеся process), cancelled_ids (задачи,
    отменённые из-за continue_on_error=False). fail/fail_on_ids эмулируют ошибки.
    """

    def __init__(self, config, *, fail=False, delay=0.0, fail_on_ids=()):
        self.config = config
        self.fail = fail
        self.delay = delay
        self.fail_on_ids = set(fail_on_ids)
        self.calls = 0
        self.called_ids = []
        self.active = 0
        self.max_active = 0
        self.cancelled_ids = []

    async def process(self, question, table):
        self.calls += 1
        self.called_ids.append(question.id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.fail or question.id in self.fail_on_ids:
                raise RuntimeError("synthetic failure")
            if self.delay:
                await asyncio.sleep(self.delay)
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
        except asyncio.CancelledError:
            self.cancelled_ids.append(question.id)
            raise
        finally:
            self.active -= 1


class ProgressSpy:
    """Заглушка протокола прогресса: считает update и запоминает последний set_postfix."""

    def __init__(self) -> None:
        self.updates = 0
        self.postfix: dict | None = None

    def update(self, n: int = 1) -> None:
        self.updates += n

    def set_postfix(self, ordered_dict: dict | None = None, **kwargs) -> None:
        # Сигнатура как у tqdm.set_postfix(ordered_dict=None, **kwargs).
        self.postfix = dict(ordered_dict or {})
        self.postfix.update(kwargs)


@pytest.mark.asyncio
async def test_progress_update_once_per_record_and_reports_failures(tmp_path):
    questions = _fixture_multi_dataset(tmp_path, n=6)
    cfg = _config(tmp_path)
    spy = ProgressSpy()

    # Одна запись (q-3) падает — и успех, и ошибка проходят через обработчик.
    summary = await run_dataset_from_tsv(
        cfg,
        questions,
        resume=False,
        translator_factory=lambda _: FakeTranslator(cfg, fail_on_ids={"q-3"}),
        progress=spy,
    )
    assert summary.processed == 5
    assert summary.failed == 1
    # Ровно один update на каждую запись исходного документа.
    assert spy.updates == 6
    assert spy.postfix == {"failed": 1}


@pytest.mark.asyncio
async def test_progress_updates_skipped_records_on_resume(tmp_path):
    questions = _fixture_multi_dataset(tmp_path, n=6)
    cfg = _config(tmp_path)
    first = await run_dataset_from_tsv(cfg, questions, resume=False, translator_factory=lambda _: FakeTranslator(cfg))
    assert first.processed == 6

    # Полный resume: все записи пропускаются pre-scan'ом, по update на каждую.
    spy = ProgressSpy()
    second = await run_dataset_from_tsv(
        cfg,
        questions,
        resume=True,
        translator_factory=lambda _: FakeTranslator(cfg),
        progress=spy,
    )
    assert second.skipped == 6
    assert second.processed == 0
    assert spy.updates == 6


@pytest.mark.asyncio
async def test_dataset_run_writes_output_and_preserves_extra_columns(tmp_path):
    questions = _fixture_dataset(tmp_path)
    cfg = _config(tmp_path)
    holder = {}

    def factory(config):
        holder["translator"] = FakeTranslator(config)
        return holder["translator"]

    summary = await run_dataset_from_tsv(cfg, questions, translator_factory=factory)
    assert summary.processed == 1
    assert summary.failed == 0

    doc = read_question_tsv_document(summary.output_questions_tsv)
    assert doc.fieldnames == ("id", "utterance", "context", "targetValue", "targetCanon")
    row = doc.records[0]
    assert row.utterance == "Где Алиса?"
    assert row.target_value == "Лондон"
    assert row.extra_fields["targetCanon"] == "London"
    assert row.context == "tables/train/q-1.tsv"

    out_table = read_table_tsv(summary.run_dir / "dataset" / row.context)
    assert out_table.headers == ["Имя", "Город"]
    assert out_table.rows == [["Алиса", "Лондон"]]

    # Same deterministic run resumes and does not invoke process again.
    second = FakeTranslator(cfg, fail=True)
    summary2 = await run_dataset_from_tsv(cfg, questions, translator_factory=lambda _: second, resume=True)
    assert summary2.skipped == 1
    assert second.calls == 0


@pytest.mark.asyncio
async def test_failure_uses_original_table_fallback(tmp_path):
    questions = _fixture_dataset(tmp_path)
    cfg = _config(tmp_path)
    failing = FakeTranslator(cfg, fail=True)
    summary = await run_dataset_from_tsv(cfg, questions, translator_factory=lambda _: failing)
    assert summary.failed == 1
    assert summary.statuses["failed"] == 1

    doc = read_question_tsv_document(summary.output_questions_tsv)
    row = doc.records[0]
    assert row.utterance == "Where is Alice?"
    assert row.target_value == "London"
    fallback = read_table_tsv(summary.run_dir / "dataset" / row.context)
    assert fallback.headers == ["Name", "City"]
    assert fallback.rows == [["Alice", "London"]]


@pytest.mark.asyncio
async def test_resume_reprocesses_corrupted_output_table(tmp_path):
    questions = _fixture_dataset(tmp_path)
    cfg = _config(tmp_path)
    first = FakeTranslator(cfg)
    summary = await run_dataset_from_tsv(cfg, questions, translator_factory=lambda _: first)
    assert first.calls == 1

    # Corrupt the persisted table. Resume must not trust mere path existence.
    doc = read_question_tsv_document(summary.output_questions_tsv)
    table_path = summary.run_dir / "dataset" / doc.records[0].context
    table_path.write_text("", encoding="utf-8")

    second = FakeTranslator(cfg)
    resumed = await run_dataset_from_tsv(cfg, questions, translator_factory=lambda _: second, resume=True)
    assert resumed.processed == 1
    assert second.calls == 1


@pytest.mark.asyncio
async def test_different_run_ids_have_isolated_dataset_artifacts(tmp_path):
    questions = _fixture_dataset(tmp_path)
    cfg = _config(tmp_path)

    a = await run_dataset_from_tsv(cfg, questions, run_id="run-a", resume=False, translator_factory=lambda _: FakeTranslator(cfg))
    b = await run_dataset_from_tsv(cfg, questions, run_id="run-b", resume=False, translator_factory=lambda _: FakeTranslator(cfg))

    assert a.output_questions_tsv != b.output_questions_tsv
    assert a.run_dir != b.run_dir
    assert a.output_questions_tsv.exists()
    assert b.output_questions_tsv.exists()
    assert a.output_questions_tsv.is_relative_to(a.run_dir)
    assert b.output_questions_tsv.is_relative_to(b.run_dir)


@pytest.mark.asyncio
async def test_concurrency_3_processes_all_in_original_order(tmp_path):
    questions = _fixture_multi_dataset(tmp_path, n=6)
    cfg = _config(tmp_path, concurrency=3)
    translator = FakeTranslator(cfg, delay=0.05)

    summary = await run_dataset_from_tsv(cfg, questions, resume=False, translator_factory=lambda _: translator)

    assert translator.calls == 6
    assert 2 <= translator.max_active <= 3
    assert summary.processed == 6
    assert summary.failed == 0
    doc = read_question_tsv_document(summary.output_questions_tsv)
    assert [r.id for r in doc.records] == [f"q-{i}" for i in range(1, 7)]


@pytest.mark.asyncio
async def test_continue_on_error_false_cancels_inflight_tasks(tmp_path):
    questions = _fixture_multi_dataset(tmp_path, n=6)
    cfg = _config(tmp_path, concurrency=3, continue_on_error=False)
    translator = FakeTranslator(cfg, delay=0.1, fail_on_ids={"q-6"})

    with pytest.raises(RuntimeError, match="synthetic failure"):
        await run_dataset_from_tsv(cfg, questions, resume=False, translator_factory=lambda _: translator)

    # Все 6 вопросов были запущены, но после падения q-6 in-flight задачи отменены.
    assert translator.calls == 6
    assert translator.max_active <= 3
    assert "q-6" in translator.called_ids
    assert set(translator.cancelled_ids) <= {"q-4", "q-5"}
    assert translator.cancelled_ids
    assert translator.active == 0

    # Run в state завершён со статусом failed.
    run_id = make_run_id(cfg, questions)
    with sqlite3.connect(cfg.state.sqlite_path) as db:
        row = db.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
    assert row is not None
    assert row[0] == "failed"
