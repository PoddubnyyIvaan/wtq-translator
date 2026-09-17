#!/usr/bin/env python3
"""Создаёт TSV той же структуры, что и отфильтрованный TSV (колонки id | utterance | context | targetValue),
в том же каталоге filtered, наполняя его по id исходными примерами из оригинального датасета.

Вход №1 (ids): отфильтрованный TSV, из которого берутся только id в порядке строк
    (например .../filtered/pristine-unseen-tables-*-0.6|back_table_bert_mean_f1-0.75.tsv).
Вход №2 (source): TSV оригинального датасета с колонками id | utterance | context | targetValue
    (например data/wtq/pristine-unseen-tables.tsv).

Выход: TSV с заголовком и строками источника для всех id из Входа №1,
       лежит рядом с Входом №1 в том же каталоге filtered.

Особенности разбора:
    - файлы состоят ровно из одной физической строки на запись (многострочные
      вопросы экранированы как литерал "\\n"), поэтому парсим вручную по '\\t' с maxsplit=3;
    - экранирование "\\n" не раскодируется: выход должен быть идентичен источнику.

Запуск (по умолчанию — конкретные файлы, указанные ниже):
    python3 scripts/extract_original_examples.py

Явно указать файлы:
    python3 scripts/extract_original_examples.py \
        --tsv output/wtq-ru-full/runs/<run>/filtered/<file>.tsv \
        --source data/wtq/pristine-unseen-tables.tsv \
        --output /path/to/<file>-original.tsv

Код возврата 1, если какой-либо id из Входа №1 отсутствует в источнике
(выходной файл в этом случае всё равно создаётся с найденными примерами).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_TSV = Path(
    "/home/ivan/work/wtq-translator/output/wtq-ru-full/runs/"
    "pristine-unseen-tables-a22315711e3c/filtered/"
    "pristine-unseen-tables-a22315711e3c-sorted-question_bert_f1-0.6"
    "|back_table_bert_mean_f1-0.75.tsv"
)
DEFAULT_SOURCE = Path("/home/ivan/work/wtq-translator/data/wtq/pristine-unseen-tables.tsv")


def read_rows(tsv: Path) -> list[dict]:
    """Читает TSV вручную (split по '\\t', maxsplit=3).

    Ждём заголовок id | utterance | context | targetValue. Возвращает список записей
    вида {id, utterance, context, targetValue} в порядке следования строк.
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
                    "utterance": utterance,
                    "context": context,
                    "targetValue": target_value,
                }
            )
    return records, malformed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Наполняет TSV той же структуры исходными примерами по id из отфильтрованного файла."
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=DEFAULT_TSV,
        help=f"Отфильтрованный TSV, откуда берутся id (по умолчанию: {DEFAULT_TSV})",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"TSV оригинального датасета с примерами (по умолчанию: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Путь до выходного TSV (по умолчанию: рядом с --tsv, имя исходного файла с суффиксом -original)",
    )
    args = parser.parse_args()

    if not args.tsv.is_file():
        sys.exit(f"Файл не найден: {args.tsv} (укажите --tsv)")
    if not args.source.is_file():
        sys.exit(f"Файл не найден: {args.source} (укажите --source)")

    output: Path = args.output or args.tsv.with_name(args.tsv.stem + "-original.tsv")

    ids_rows, tsv_malformed = read_rows(args.tsv)
    source_rows, src_malformed = read_rows(args.source)

    # Индекс источника: id -> запись (в источнике id уникальны; проверяем на всякий случай).
    source_by_id: dict[str, dict] = {}
    duplicate_ids: list[str] = []
    for rec in source_rows:
        if rec["id"] in source_by_id:
            duplicate_ids.append(rec["id"])
            continue
        source_by_id[rec["id"]] = rec

    missing_ids: list[str] = []
    written = 0
    with output.open("w", encoding="utf-8") as fh:
        fh.write("id\tutterance\tcontext\ttargetValue\n")
        for rec in ids_rows:
            src = source_by_id.get(rec["id"])
            if src is None:
                missing_ids.append(rec["id"])
                continue
            fh.write(f"{src['id']}\t{src['utterance']}\t{src['context']}\t{src['targetValue']}\n")
            written += 1

    print(f"Входной TSV  : {args.tsv}")
    print(f"Источник     : {args.source}")
    print(f"Выходной     : {output}")
    print(f"ID во входе      : {len(ids_rows)}")
    print(f"Записей в выходе : {written}")

    if tsv_malformed or src_malformed:
        print("Пропущено битых строк:", file=sys.stderr)
        for name, malformed in (("вход", tsv_malformed), ("источник", src_malformed)):
            for lineno, line in malformed[:5]:
                print(f"  {name}, строка {lineno}: {line[:120]!r}", file=sys.stderr)
        print("ВНИМАНИЕ: в разборе найдены строки без 4 колонок (см. выше).", file=sys.stderr)

    if duplicate_ids:
        print(f"ВНИМАНИЕ: в источнике дублируются id: {duplicate_ids[:10]}", file=sys.stderr)

    if missing_ids:
        print(f"В источнике не найдено id ({len(missing_ids)}): {missing_ids[:10]}", file=sys.stderr)
        print("ВНИМАНИЕ: не все примеры попали в выходной файл.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()