from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .api import Translator
from .config import AppConfig
from .dataset import (
    read_question_tsv,
    read_question_tsv_document,
    read_table_tsv,
    resolve_context_path,
    write_question_tsv,
    write_table_tsv,
)
from .enums import ProcessingStatus
from .models import QuestionRecord, RunSummary, TranslationResult
from .persistence import StateStore, write_manifest
from .reporting.events import append_event
from .utils import seed_everything


def _safe_question_id(question_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in question_id)


RUN_SCHEMA_VERSION = "2026-08-08-v4"


def _config_hash(config: AppConfig) -> str:
    data = config.model_dump(mode="json")
    data["_run_schema_version"] = RUN_SCHEMA_VERSION
    # Credentials must not affect reproducibility identity and must never be persisted.
    if "model" in data:
        data["model"]["api_key"] = None
    if "embeddings" in data:
        data["embeddings"]["api_key"] = None
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_run_id(config: AppConfig, questions_tsv: str | Path) -> str:
    p = Path(questions_tsv).resolve()
    payload = f"{_config_hash(config)}\n{p.as_posix()}"
    return f"{p.stem}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _output_questions_path(config: AppConfig, questions_tsv: Path, run_id: str) -> Path:
    return _run_dir(config, run_id) / "dataset" / questions_tsv.name


def _table_paths(config: AppConfig, questions_tsv: Path, question_id: str, run_id: str) -> tuple[Path, Path]:
    rel = Path("tables") / questions_tsv.stem / f"{_safe_question_id(question_id)}.tsv"
    return rel, _run_dir(config, run_id) / "dataset" / rel


def _run_dir(config: AppConfig, run_id: str) -> Path:
    return config.project.output_root / "runs" / run_id


def _record_to_state(record: QuestionRecord) -> dict:
    return {
        "id": record.id,
        "utterance": record.utterance,
        "context": record.context,
        "targetValue": record.target_value,
        "extra_fields": dict(record.extra_fields),
    }


def _record_from_state(data: dict) -> QuestionRecord:
    return QuestionRecord(
        id=data["id"],
        utterance=data["utterance"],
        context=data["context"],
        target_value=data["targetValue"],
        extra_fields=dict(data.get("extra_fields") or {}),
    )


def _write_trace(run_dir: Path, split: str, result: TranslationResult, source_context: str, output_context: str) -> Path:
    trace_dir = run_dir / "traces" / split
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{_safe_question_id(result.question.id)}.json"
    trace = {
        "question_id": result.question.id,
        "source_context": source_context,
        "output_context": output_context,
        "source_question": result.question.utterance,
        "translated_question": result.translated_question,
        "source_target_value": result.question.target_value,
        "translated_target_value": result.translated_target_value,
        "mentions": [
            {
                **asdict(m),
                "source_type": m.source_type.value,
                "coordinate": asdict(m.coordinate),
            }
            for m in result.mention_map.mentions
        ],
        "quality": result.quality.to_dict(),
    }
    tmp = trace_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(trace_path)
    return trace_path


def _copy_source_table_as_fallback(source_table_path: Path, output_table_path: Path) -> None:
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_table_path.with_suffix(output_table_path.suffix + ".tmp")
    shutil.copyfile(source_table_path, tmp)
    tmp.replace(output_table_path)


def _state_artifact_is_valid(state) -> bool:
    if state is None or state.output_record is None or not state.output_table_path:
        return False
    path = Path(state.output_table_path)
    if not path.is_file():
        return False
    try:
        read_table_tsv(path)
    except Exception:
        return False
    return True


def _should_skip(state, config: AppConfig, resume: bool) -> bool:
    if not resume or not _state_artifact_is_valid(state):
        return False
    if state.status in {ProcessingStatus.ACCEPTED, ProcessingStatus.ACCEPTED_WITH_WARNINGS}:
        return True
    if state.status == ProcessingStatus.NEEDS_REVIEW:
        return not config.runner.rerun_needs_review
    if state.status == ProcessingStatus.FAILED:
        return not config.runner.retry_failed_on_resume
    return False


async def _process_question(
    config: AppConfig,
    translator: Translator,
    question: QuestionRecord,
) -> TranslationResult:
    """Читает исходную таблицу вопроса и выполняет перевод.

    Вспомогательная функция воркера пула: не выполняет никаких side-эффектов записи
    (state/таблицы/traces/jsonl/события) — результат или ошибка всплывают в главную
    корутину, где обрабатываются единственным обработчиком.
    """
    source_table_path = resolve_context_path(config.project.input_root, question.context)
    source_table = read_table_tsv(source_table_path)
    return await translator.process(question, source_table)


async def run_one_from_tsv(
    config: AppConfig,
    questions_tsv: str | Path,
    question_id: str,
) -> TranslationResult:
    seed_everything(config.project.seed)
    records = read_question_tsv(questions_tsv)
    question = next((r for r in records if r.id == question_id), None)
    if question is None:
        raise KeyError(f"Question not found: {question_id}")

    source_table_path = resolve_context_path(config.project.input_root, question.context)
    table = read_table_tsv(source_table_path)

    qtsv = Path(questions_tsv)
    run_id = make_run_id(config, qtsv)
    run_dir = _run_dir(config, run_id)
    translator = Translator(config, llm_log_path=run_dir / "reports" / "llm_io.jsonl")
    result = await translator.process(question, table)

    table_rel, table_out = _table_paths(config, qtsv, question.id, run_id)
    write_table_tsv(result.output_table, table_out)

    _write_trace(_run_dir(config, run_id), qtsv.stem, result, question.context, table_rel.as_posix())
    return result


async def run_dataset_from_tsv(
    config: AppConfig,
    questions_tsv: str | Path,
    *,
    run_id: str | None = None,
    resume: bool = True,
    translator_factory: Callable[[AppConfig], Translator] = Translator,
    progress: Any | None = None,
) -> RunSummary:
    """Translate one WTQ question TSV end-to-end.

    Each question receives its own output table so different translation modes can
    safely produce different versions from the same source table.
    """
    questions_path = Path(questions_tsv)
    seeded_backends = seed_everything(config.project.seed)
    document = read_question_tsv_document(questions_path)
    run_id = run_id or make_run_id(config, questions_path)
    run_dir = _run_dir(config, run_id)
    output_questions = _output_questions_path(config, questions_path, run_id)
    reports_dir = run_dir / "reports"
    events_path = reports_dir / "events.jsonl"
    failures_path = reports_dir / "failures.jsonl"
    quality_path = reports_dir / "quality.jsonl"
    stats_path = reports_dir / "stats.json"
    manifest_path = run_dir / "manifest.json"

    state = StateStore(config.state.sqlite_path)
    cfg_hash = _config_hash(config)
    state.start_run(
        run_id=run_id,
        questions_tsv=str(questions_path.resolve()),
        output_questions_tsv=str(output_questions.resolve()),
        config_hash=cfg_hash,
    )
    recovered = state.reset_processing(run_id) if resume else 0
    write_manifest(
        manifest_path,
        run_id=run_id,
        config=config,
        extra={
            "questions_tsv": str(questions_path.resolve()),
            "output_questions_tsv": str(output_questions.resolve()),
            "config_hash": cfg_hash,
            "recovered_processing_records": recovered,
            "seeded_local_backends": seeded_backends,
        },
    )

    if translator_factory is Translator:
        translator = Translator(config, llm_log_path=reports_dir / "llm_io.jsonl")
    else:
        translator = translator_factory(config)
    output_by_id: dict[str, QuestionRecord] = {}
    processed = 0
    skipped = 0
    failed = 0

    # Rehydrate already-finished records first. This makes checkpoint output complete
    # even when the current invocation is a resume after interruption.
    if resume:
        for s in state.list_questions(run_id):
            if s.output_record is not None:
                output_by_id[s.question_id] = _record_from_state(s.output_record)

    def checkpoint() -> None:
        ordered = [output_by_id[r.id] for r in document.records if r.id in output_by_id]
        if ordered:
            write_question_tsv(ordered, output_questions, fieldnames=document.fieldnames)

    # Pre-scan: восстановленные из state записи уже в output_by_id (см. выше).
    # Здесь применяем _should_skip и собираем todo = [(ordinal_index, question)] —
    # записи, которые ещё нужно обработать. ordinal_index — позиция в исходном
    # документе, нужна для сообщения START «Processing question N/M».
    todo: list[tuple[int, QuestionRecord]] = []
    for index, question in enumerate(document.records):
        existing = state.get_question(run_id, question.id)
        if _should_skip(existing, config, resume):
            output_by_id[question.id] = _record_from_state(existing.output_record)
            skipped += 1
            # Инвариант: каждая запись document.records даёт ровно один update —
            # либо здесь (пропущена pre-scan'ом), либо в обработчике завершённой задачи.
            if progress is not None:
                progress.update(1)
            continue
        todo.append((index, question))

    # Пул воркеров по схеме «top-up + единственный обработчик»: воркеры только читают
    # таблицу и вызывают translator.process; все записи в БД/jsonl/файлы делает главная
    # корутина (поэтому блокировки не нужны).
    running: set[asyncio.Task[TranslationResult]] = set()
    done: asyncio.Queue[asyncio.Task[TranslationResult]] = asyncio.Queue()
    task_question: dict[asyncio.Task[TranslationResult], QuestionRecord] = {}

    while todo or running:
        # Top-up: заполняем пул, пока есть место и необработанные вопросы. Статус
        # PROCESSING и событие START пишутся в главной корутине до запуска задачи.
        while len(running) < config.runner.concurrency and todo:
            index, question = todo.pop(0)
            state.set_question(run_id, question.id, ProcessingStatus.PROCESSING, config.translation.mode.value)
            append_event(
                events_path,
                run_id=run_id,
                question_id=question.id,
                stage="question",
                level="info",
                code="START",
                message=f"Processing question {index + 1}/{len(document.records)}",
            )
            task = asyncio.create_task(_process_question(config, translator, question))
            running.add(task)
            task_question[task] = question
            task.add_done_callback(lambda t, q=done: q.put_nowait(t))

        # Ждём одну завершённую задачу и обрабатываем её результат в главной корутине.
        task = await done.get()
        running.discard(task)
        question = task_question.pop(task)
        table_rel, table_out = _table_paths(config, questions_path, question.id, run_id)
        # Прогресс: одна завершённая задача == одна обработанная запись.
        # update вызывается до try/except, чтобы учесть и падающую запись даже при
        # continue_on_error=False (перед raise). Восстановленные из state записи здесь
        # не проходят — их учитывает pre-scan, двойных update нет.
        if progress is not None:
            progress.update(1)
        try:
            result = task.result()
            write_table_tsv(result.output_table, table_out)
            output_record = question.with_translation(
                utterance=result.translated_question,
                context=table_rel.as_posix(),
                target_value=result.translated_target_value,
            )
            _write_trace(run_dir, questions_path.stem, result, question.context, table_rel.as_posix())

            state.set_question(
                run_id,
                question.id,
                result.quality.status,
                config.translation.mode.value,
                output_table_path=str(table_out.resolve()),
                output_record=_record_to_state(output_record),
            )
            state.save_quality(run_id, question.id, result.quality.to_dict())
            output_by_id[question.id] = output_record
            processed += 1

            with quality_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result.quality.to_dict(), ensure_ascii=False) + "\n")
            append_event(
                events_path,
                run_id=run_id,
                question_id=question.id,
                stage="question",
                level="info",
                code="DONE",
                message=f"Finished with status {result.quality.status.value}",
            )
        except Exception as exc:
            failed += 1
            # Fallback policy: never leave a partially translated table under the final
            # output path. Copy the complete original table instead and keep the source row.
            try:
                source_table_path = resolve_context_path(config.project.input_root, question.context)
                _copy_source_table_as_fallback(source_table_path, table_out)
                fallback_record = question.with_translation(
                    utterance=question.utterance,
                    context=table_rel.as_posix(),
                    target_value=question.target_value,
                )
                output_by_id[question.id] = fallback_record
                record_state = _record_to_state(fallback_record)
            except Exception as fallback_exc:
                record_state = None
                exc = RuntimeError(f"{exc}; fallback table copy also failed: {fallback_exc}")

            error_text = f"{type(exc).__name__}: {exc}"
            state.set_question(
                run_id,
                question.id,
                ProcessingStatus.FAILED,
                config.translation.mode.value,
                output_table_path=str(table_out.resolve()) if table_out.exists() else None,
                output_record=record_state,
                error=error_text,
            )
            failure = {
                "run_id": run_id,
                "question_id": question.id,
                "error": error_text,
                "traceback": traceback.format_exc(),
            }
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            append_event(
                events_path,
                run_id=run_id,
                question_id=question.id,
                stage="question",
                level="error",
                code="FAILED",
                message=error_text,
            )
            if not config.runner.continue_on_error:
                # continue_on_error=False: отменяем оставшиеся in-flight-задачи,
                # завершаем run со статусом failed и пробрасываем исходную ошибку.
                for pending in running:
                    pending.cancel()
                await asyncio.gather(*running, return_exceptions=True)
                running.clear()
                checkpoint()
                state.finish_run(run_id, "failed")
                raise

        if progress is not None:
            progress.set_postfix({"failed": failed})
        if (processed + failed) % config.runner.checkpoint_every == 0:
            checkpoint()

    checkpoint()
    counts = state.status_counts(run_id)
    total = len(document.records)
    run_status = "completed_with_failures" if counts.get(ProcessingStatus.FAILED.value, 0) else "completed"
    state.finish_run(run_id, run_status)

    stats = {
        "run_id": run_id,
        "total": total,
        "processed_this_invocation": processed,
        "skipped_this_invocation": skipped,
        "failed_this_invocation": failed,
        "statuses": counts,
        "output_questions_tsv": str(output_questions.resolve()),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return RunSummary(
        run_id=run_id,
        total=total,
        processed=processed,
        skipped=skipped,
        failed=failed,
        statuses=counts,
        output_questions_tsv=output_questions,
        run_dir=run_dir,
    )


def get_run_report(config: AppConfig, questions_tsv: str | Path, *, run_id: str | None = None) -> dict:
    qpath = Path(questions_tsv)
    rid = run_id or make_run_id(config, qpath)
    state = StateStore(config.state.sqlite_path)
    states = state.list_questions(rid)
    return {
        "run_id": rid,
        "total_in_state": len(states),
        "statuses": dict(Counter(s.status.value for s in states)),
        "failed": [
            {"question_id": s.question_id, "error": s.error}
            for s in states if s.status == ProcessingStatus.FAILED
        ],
        "needs_review": [s.question_id for s in states if s.status == ProcessingStatus.NEEDS_REVIEW],
        "output_questions_tsv": str(_output_questions_path(config, qpath, rid)),
        "run_dir": str(_run_dir(config, rid)),
    }
