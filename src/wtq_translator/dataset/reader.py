from __future__ import annotations

import csv
from pathlib import Path

from ..models import QuestionRecord, QuestionTSVDocument, TableData

_REQUIRED = ("id", "utterance", "context", "targetValue")


def read_question_tsv_document(path: str | Path) -> QuestionTSVDocument:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = tuple(reader.fieldnames or ())
        missing = set(_REQUIRED) - set(fieldnames)
        if missing:
            raise ValueError(f"Missing WTQ columns in {p}: {sorted(missing)}")
        records: list[QuestionRecord] = []
        for row in reader:
            extras = {k: (row.get(k) or "") for k in fieldnames if k not in _REQUIRED}
            records.append(QuestionRecord(
                id=row["id"],
                utterance=row["utterance"],
                context=row["context"],
                target_value=row["targetValue"],
                extra_fields=extras,
            ))
    return QuestionTSVDocument(fieldnames=fieldnames, records=tuple(records))


def read_question_tsv(path: str | Path) -> list[QuestionRecord]:
    return list(read_question_tsv_document(path).records)


def read_table_tsv(path: str | Path) -> TableData:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty table: {p}")
    return TableData(headers=rows[0], rows=rows[1:], source_path=p)
