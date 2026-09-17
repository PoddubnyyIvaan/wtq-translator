#!/usr/bin/env python3
"""Конвертация отфильтрованного TSV-датасета wtq-translator в JSONL для eval-скрипта LoRA.

Вход: TSV с колонками  id | utterance | context | targetValue
      (файлы вида <run>/filtered/<name>.tsv, например
       .../filtered/pristine-unseen-tables-*-0.6|back_table_bert_mean_f1-0.75.tsv).
      Пути из колонки context (например "tables/pristine-unseen-tables/nu-124.tsv")
      разрешаются относительно каталога <run>/dataset — там лежат переведённые таблицы.

Выход: JSONL, одна JSON-строка на запись, совместимый с eval-скриптом contrastive_llm_lora
       (формат test.jsonl):
         {"id": "...", "query": "...", "context": "...", "targetValue": "...",
          "table": "<markdown-рендер таблицы>"}
       query берётся из utterance (многострочные вопросы раскодируются: "\\n" -> перенос строки);
       table рендерится как pandas.DataFrame.to_markdown(index=False) — именно в таком виде
       таблицы лежали в оригинальном test.jsonl.

Зависимости: pandas, tabulate.
Установка:   pip install pandas tabulate
             (в этом репозитории: uv add pandas tabulate; venv уже содержит обе)

Запуск (по умолчанию — конкретный отфильтрованный файл):
    python3 scripts/tsv_to_eval_jsonl.py

Явно указать вход/выход:
    python3 scripts/tsv_to_eval_jsonl.py \
        --tsv output/wtq-ru-full/runs/<run>/filtered/<file>.tsv \
        --output /path/to/test_ru.jsonl

Указать базовую директорию для таблиц (к ней добавляется путь из колонки context,
например "tables/.../nu-124.tsv" -> <root>/tables/.../nu-124.tsv):
    python3 scripts/tsv_to_eval_jsonl.py \
        --tsv output/wtq-ru-full/runs/<run>/filtered/<file>.tsv \
        --tables-root /custom/path/to/tables
    (по умолчанию — <run>/dataset, вычисляется автоматически)

Быстрая проверка (первые 5 записей):
    python3 scripts/tsv_to_eval_jsonl.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

DEFAULT_TSV = Path(
    "/home/ivan/work/wtq-translator/output/wtq-ru-full/runs/"
    "pristine-unseen-tables-a22315711e3c/filtered/"
    "pristine-unseen-tables-a22315711e3c-sorted-question_bert_f1-0.6"
    "|back_table_bert_mean_f1-0.75.tsv"
)


def require_module(pkg: str, module: str) -> None:
    """Проверяет наличие библиотеки и завершает работу с понятной ошибкой."""
    try:
        __import__(module)
    except ImportError:
        sys.exit(f"Не установлена библиотека '{pkg}'. Выполните: pip install {pkg}")


def read_rows(tsv: Path) -> list[dict]:
    """Читает TSV вручную (split по '\\t', maxsplit=3).

    Ручной разбор надёжнее pd.read_csv: многострочные вопросы уже экранированы
    как литерал "\\n", поэтому файл состоит ровно из одной физической строки на запись
    и не ломает парсер.
    """
    records: list[dict] = []
    malformed: list[tuple[int, str]] = []
    with tsv.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\r\n")
        if not header.startswith("id\t"):
            print(f"Предупреждение: неожиданный заголовок: {header!r}", file=sys.stderr)
        for lineno, line in enumerate(fh, start=2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 3)
            if len(parts) != 4:
                malformed.append((lineno, line))
                continue
            qid, utterance, context, target_value = parts
            records.append(
                {
                    "id": qid,
                    # В TSV перенос строки внутри вопроса закодирован двумя символами "\\n";
                    # в JSON-датасете он должен стать настоящим переносом строки.
                    "query": utterance.replace("\\n", "\n"),
                    "context": context,
                    "targetValue": target_value,
                }
            )
    return records, malformed


def render_table(path: Path, missing_value: str) -> str:
    """Читает таблицу TSV и возвращает markdown-строку (формат tabulate "pipe")."""
    import pandas as pd

    # Без dtype=str: числовые колонки выравниваются вправо, как в оригинальном
    # рендере test.jsonl (например "|      1 |").
    df = pd.read_csv(path, sep="\t")
    # Пустые ячейки CSV при чтении становятся NaN. pandas.DataFrame.to_markdown()
    # по умолчанию печатает их как "nan"; если в оригинальном test.jsonl пустые
    # клетки были пустыми строками, передайте --missing-value empty.
    missingval = "" if missing_value == "empty" else "nan"
    return df.to_markdown(index=False, missingval=missingval)


def default_tables_root(tsv: Path) -> Path:
    """Каталог, относительно которого разрешается context: <run>/dataset.

    Для <run>/filtered/<file>.tsv это <run>/dataset.
    """
    return Path(tsv).resolve().parents[1] / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Конвертация отфильтрованного TSV wtq-translator в JSONL для eval-скрипта LoRA."
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=DEFAULT_TSV,
        help=f"Входной TSV (по умолчанию: {DEFAULT_TSV})",
    )
    parser.add_argument(
        "--tables-root",
        type=Path,
        default=None,
        help="Каталог для путей из context (по умолчанию: <run>/dataset, выводится автоматически)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Путь до выходного JSONL (по умолчанию: тот же путь, что у TSV, с расширением .jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Обработать только первые N записей (для быстрой проверки)",
    )
    parser.add_argument(
        "--missing-value",
        choices=["nan", "empty"],
        default="nan",
        help="Как рендерить пустые ячейки таблицы: 'nan' (по умолчанию pandas.to_markdown) "
             "или 'empty' (пустая строка)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tsv: Path = args.tsv
    if not tsv.is_file():
        sys.exit(f"Файл не найден: {tsv} (укажите --tsv)")
    tables_root: Path = args.tables_root or default_tables_root(tsv)
    output: Path = args.output or tsv.with_suffix(".jsonl")

    require_module("pandas", "pandas")
    require_module("tabulate", "tabulate")

    rows, malformed = read_rows(tsv)
    if args.limit is not None:
        rows = rows[: args.limit]

    missing: list[str] = []
    written = 0
    with output.open("w", encoding="utf-8") as fh:
        for rec in rows:
            table_path = tables_root / rec["context"]
            if not table_path.is_file():
                missing.append(rec["context"])
                continue
            rec["table"] = render_table(table_path, args.missing_value)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    # Краткий отчёт
    print(f"Входной TSV : {tsv}")
    print(f"Таблицы     : {tables_root}")
    print(f"Выходной    : {output}")
    print(f"Прочитано записей : {len(rows)} (limit={args.limit})")
    print(f"Записано          : {written}")
    if malformed:
        print(f"Пропущено битых строк: {len(malformed)}", file=sys.stderr)
        for lineno, line in malformed[:5]:
            print(f"  строка {lineno}: {line[:120]!r}", file=sys.stderr)
    if missing:
        print(f"Пропущено из-за отсутствующих таблиц: {len(missing)}", file=sys.stderr)
        for ctx in missing[:5]:
            print(f"  нет файла: {tables_root / ctx}", file=sys.stderr)
    if written != len(rows):
        print("ВНИМАНИЕ: не все записи попали в выходной файл (см. выше).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()