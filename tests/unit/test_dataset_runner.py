import csv
from pathlib import Path

import pytest

from wtq_translator.config import AppConfig
from wtq_translator.dataset import read_question_tsv_document, read_table_tsv
from wtq_translator.enums import ProcessingStatus, TranslationMode
from wtq_translator.models import MentionMap, QualityReport, TableData, TranslationResult
from wtq_translator.runner import run_dataset_from_tsv


def _config(tmp_path: Path) -> AppConfig:
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
        "runner": {"checkpoint_every": 1, "continue_on_error": True},
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


class FakeTranslator:
    def __init__(self, config, *, fail=False):
        self.config = config
        self.fail = fail
        self.calls = 0

    async def process(self, question, table):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic failure")
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
