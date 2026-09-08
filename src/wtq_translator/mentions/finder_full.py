"""Full-матчер: лемматизированные подпоследовательности вопроса ↔ кандидаты таблицы."""
from __future__ import annotations

from typing import Any

from ..config import MentionsConfig
from ..enums import MentionSourceType
from ..models import CandidateSet, MentionSearchResult, TableData, TableMention
from .candidates import extract_question_candidates, extract_table_candidates


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def find_table_mentions_full(
    question: str,
    table: TableData,
    cfg: MentionsConfig,
    nlp: Any,
    target_lang: str,
) -> tuple[MentionSearchResult, CandidateSet]:
    """Сопоставляет вопрос и таблицу по лемма-ключам с жадным отбором longest non-overlap."""
    candidate_set = extract_table_candidates(table, nlp, cfg, target_lang)
    by_lemma = {c.lemma_key: c for c in candidate_set.candidates}
    question_candidates = extract_question_candidates(question, nlp, cfg)

    matched = [qc for qc in question_candidates if qc.lemma_key in by_lemma]

    # Жадный отбор: сначала длинные, при равенстве — по позиции; длинный блокирует вложенные.
    ordered = sorted(matched, key=lambda qc: (-qc.length, qc.start, qc.end))
    selected = []
    for qc in ordered:
        if any(_overlaps(qc.start, qc.end, s.start, s.end) for s in selected):
            continue
        selected.append(qc)
    selected.sort(key=lambda qc: (qc.start, qc.end))

    mentions: list[TableMention] = []
    for i, qc in enumerate(selected):
        candidate = by_lemma[qc.lemma_key]
        first = candidate.coordinates[0]
        source_type = MentionSourceType.HEADER if first.row is None else MentionSourceType.CELL
        mentions.append(TableMention(
            mask=f"<TABLE_REF_{i:04d}>",
            question_text=question[qc.start:qc.end],
            normalized_text=qc.lemma_key.lower(),
            start=qc.start,
            end=qc.end,
            source_type=source_type,
            coordinates=tuple(candidate.coordinates),
            table_source_text=candidate.source_text,
            match_type="lemma",
            candidate_key=candidate.id,
        ))

    return MentionSearchResult(mentions=mentions, ambiguous=[]), candidate_set
