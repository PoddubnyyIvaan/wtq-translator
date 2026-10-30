"""Оркестратор сбора сущностей: главный цикл по таблицам, dedup, чекпоинты."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..dataset import read_question_tsv_document, read_table_tsv
from .ner import EntityExtractor
from .registry import EntitiesRegistry, ProcessedTables

EntityKey = tuple[str, str]  # (cell, col) — ключ dedup-кэша


@dataclass
class EntitiesSummary:
    total_tables: int
    skipped_processed: int
    missing_tables: int
    new_entities: int
    updated_entities: int
    output_tsv: Path
    processed_path: Path


def _normalize_name(value: str) -> str:
    """strip + collapse whitespace (любые пробельные последовательности → один пробел)."""
    return " ".join(value.split())


def _contexts(config: AppConfig, input_tsv: str | Path | None) -> list[str]:
    """Уникальные context-пути входного WTQ-файла в порядке первого упоминания."""
    cfg = config.entities
    input_path = Path(input_tsv) if input_tsv is not None else cfg.input_tsv
    if input_path is None:
        raise ValueError("entities: задайте entities.input_tsv в конфиге или --input-tsv")
    doc = read_question_tsv_document(input_path)
    return list(dict.fromkeys(record.context for record in doc.records))


def remaining_tables(config: AppConfig, input_tsv: str | Path | None = None) -> list[str]:
    """Контексты, ещё не отмеченные в списке обработанных (для тотала прогресс-бара)."""
    processed = ProcessedTables.load(config.entities.processed_path)
    return [ctx for ctx in _contexts(config, input_tsv) if ctx not in processed.seen]


def run_entities(
    config: AppConfig,
    input_tsv: str | Path | None = None,
    progress: Any | None = None,
) -> EntitiesSummary:
    """Пройтись по таблицам входного файла и накопить сущности в реестре.

    Обработанные таблицы помечаются в ``processed_path`` и при дозапуске пропускаются.
    Чекпоинт (реестр + список обработанных) пишется каждые ``checkpoint_every_tables``
    таблиц и в конце. В памяти держатся только реестр и dedup-кэш запуска.
    """
    cfg = config.entities
    contexts = _contexts(config, input_tsv)
    total_tables = len(contexts)

    processed = ProcessedTables.load(cfg.processed_path)
    registry = EntitiesRegistry(sep=cfg.all_types_sep)

    todo = [ctx for ctx in contexts if ctx not in processed.seen]
    skipped_processed = total_tables - len(todo)

    missing_tables = 0
    new_entities = 0
    updated_entities = 0

    if todo:
        extractor: EntityExtractor | None = None
        # Dedup уровня запуска: (cell, col) считается один раз, повторы берутся из кэша.
        dedup: dict[EntityKey, tuple[str | None, frozenset[str]]] = {}
        checkpoint_counter = 0

        def checkpoint() -> None:
            registry.save(cfg.output_tsv)
            processed.flush(cfg.processed_path)

        for ctx in todo:
            table_path = (config.project.input_root / ctx).with_suffix(f".{cfg.table_suffix}")
            if not table_path.exists():
                if cfg.missing_table == "error":
                    raise FileNotFoundError(
                        f"Table referenced by context not found: {ctx}; expected {table_path}"
                    )
                missing_tables += 1
            else:
                table = read_table_tsv(table_path)
                # Отсев ячеек: пустые и ячейки с числом слов > cfg.max_words в NER не идут.
                pairs: list[EntityKey] = [
                    (cell, table.headers[col_idx])
                    for row in table.rows
                    for col_idx, cell in enumerate(row)
                    if cell.strip() and len(cell.split()) <= cfg.max_words
                ]
                todo_pairs = [key for key in pairs if key not in dedup]
                if todo_pairs:
                    if extractor is None:
                        extractor = EntityExtractor(
                            device=cfg.device,
                            batch_size=cfg.batch_size,
                            template=cfg.template,
                            allowed_types=cfg.allowed_types,
                        )
                    for key, result in zip(todo_pairs, extractor.classify_many(todo_pairs)):
                        dedup[key] = result
                for cell, col in pairs:
                    ent_type, cell_types = dedup[(cell, col)]
                    if ent_type is None or ent_type not in cfg.allowed_types:
                        continue
                    name = _normalize_name(cell) if cfg.normalize_name else cell
                    if registry.add(name, ent_type, cell_types):
                        new_entities += 1
                    else:
                        updated_entities += 1
            processed.add(ctx)
            checkpoint_counter += 1
            if checkpoint_counter >= cfg.checkpoint_every_tables:
                checkpoint()
                checkpoint_counter = 0
            if progress is not None:
                progress.update(1)

        checkpoint()

    return EntitiesSummary(
        total_tables=total_tables,
        skipped_processed=skipped_processed,
        missing_tables=missing_tables,
        new_entities=new_entities,
        updated_entities=updated_entities,
        output_tsv=cfg.output_tsv,
        processed_path=cfg.processed_path,
    )