"""Отсев вопросов по алфавитно-зависимому намерению (режим full).

Прогоняет ``detect_alphabet_intent`` по каждому вопросу датасета БЕЗ полного
перевода и разделяет вопросы на:
- переводимые — намерение не определено (``intent_id: null``);
- исключаемые — намерение 1..4 или провал детекции (fail-closed).

Записывает в ``<output_root>/alphabet-screen/<run_id>/``:
- ``dataset/<stem>.tsv``                  — переводимые вопросы (исходный формат);
- ``dataset/<stem>-alphabet-excluded.tsv`` — исключаемые (class + why);
- ``reports/alphabet-screen-stats.json``  — статистика отсева.

Запуск через CLI: ``wtq-translate alphabet-screen <config> <questions_tsv>``,
где config — конфиг run_full (configs/qwen_full.yaml).
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .alphabet import ALPHABET_INTENT_LABELS, detect_alphabet_intent
from .config import AppConfig
from .dataset import (
    read_question_tsv_document,
    write_alphabet_excluded_tsv,
    write_question_tsv,
)
from .llm.base import LLMProvider
from .llm.factory import create_llm_provider
from .llm.logged import LoggedLLMProvider
from .models import QuestionRecord
from .persistence.cache import CachedLLMProvider, SQLiteCache
from .runner import make_run_id
from .utils import seed_everything


def _build_llm(config: AppConfig, llm_log_path: str | Path | None) -> LLMProvider:
    """Строит цепочку LLM-провайдера (кэш + лог), как в api.Translator, без
    embeddings и BERTScore — для отсева нужен только детект-вызов."""
    llm = create_llm_provider(config)
    llm_identity: dict[str, object] = {
        "provider": config.model.provider,
        "base_url": config.model.base_url,
        "model": config.model.model_name,
    }
    if config.model.provider == "ollama":
        llm_identity["ollama"] = config.model.ollama.model_dump(mode="json")
    if config.cache.enabled:
        cache = SQLiteCache(config.state.sqlite_path)
        llm = CachedLLMProvider(llm, cache, identity=dict(llm_identity))
    if config.logging.llm_io_enabled:
        path = Path(llm_log_path) if llm_log_path is not None else Path(config.logging.llm_io_path)
        llm = LoggedLLMProvider(llm, path, identity=dict(llm_identity))
    return llm


async def run_alphabet_screen(
    config: AppConfig,
    questions_tsv: str | Path,
    *,
    llm: LLMProvider | None = None,
    llm_log_path: str | Path | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Отсев вопросов по алфавитно-зависимому намерению.

    Каждый вопрос датасета проходит через ``detect_alphabet_intent``
    (concurrency — ``config.runner.concurrency``). Вопросы без намерения пишутся
    в kept-TSV, вопросы с намерением или ошибкой детекции — в
    ``{stem}-alphabet-excluded.tsv`` (для ошибки ``class=""``). Возвращает
    статистику отсева (также сохраняется в ``reports/alphabet-screen-stats.json``).
    """
    seed_everything(config.project.seed)
    document = read_question_tsv_document(questions_tsv)
    qpath = Path(questions_tsv)
    total = len(document.records)
    run_id = make_run_id(config, qpath)
    root = config.project.output_root / "alphabet-screen" / run_id
    dataset_dir = root / "dataset"
    reports_dir = root / "reports"
    kept_path = dataset_dir / qpath.name
    excluded_path = dataset_dir / f"{qpath.stem}-alphabet-excluded.tsv"
    stats_path = reports_dir / "alphabet-screen-stats.json"

    # В regex-режиме провайдер не нужен (не тратим время на проверку конфига,
    # кэша и логов и не делаем никаких LLM/кэш-вызовов).
    provider = (
        None
        if config.alphabet_detection.method == "regex"
        else (llm if llm is not None else _build_llm(config, llm_log_path))
    )
    sem = asyncio.Semaphore(config.runner.concurrency)

    async def detect_one(index: int, question: QuestionRecord) -> tuple[int, Any]:
        async with sem:
            detection = await detect_alphabet_intent(
                provider, config,
                question_id=question.id,
                question_text=question.utterance,
            )
        return index, detection

    tasks = [asyncio.create_task(detect_one(i, q)) for i, q in enumerate(document.records)]
    outcomes: dict[int, Any] = {}
    for task in asyncio.as_completed(tasks):
        index, detection = await task
        outcomes[index] = detection
        if progress is not None:
            progress.update(1)

    kept_records: list[QuestionRecord] = []
    excluded_rows: list[dict[str, str]] = []
    intent_counts: Counter = Counter()
    detection_errors = 0

    for index, question in enumerate(document.records):
        detection = outcomes[index]
        if detection.error is not None:
            detection_errors += 1
            excluded_rows.append(_excluded_row(question, "", detection.error))
        elif detection.intent_id is not None:
            label = ALPHABET_INTENT_LABELS[detection.intent_id]
            intent_counts[label] += 1
            excluded_rows.append(_excluded_row(question, label, detection.why or ""))
        else:
            kept_records.append(question)

    write_question_tsv(kept_records, kept_path, fieldnames=document.fieldnames)
    write_alphabet_excluded_tsv(excluded_rows, excluded_path)

    excluded = len(excluded_rows)
    kept = len(kept_records)
    stats = {
        "run_id": run_id,
        "total": total,
        "kept": kept,
        "excluded": excluded,
        "excluded_share": excluded / total if total else 0.0,
        "intent_classes": dict(intent_counts),
        "detection_errors": detection_errors,
        "kept_tsv": str(kept_path.resolve()),
        "alphabet_excluded_tsv": str(excluded_path.resolve()),
        "stats_json": str(stats_path.resolve()),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _excluded_row(question: QuestionRecord, label: str, why: str) -> dict[str, str]:
    return {
        "id": question.id,
        "utterance": question.utterance,
        "context": question.context,
        "targetValue": question.target_value,
        "class": label,
        "why": why,
    }