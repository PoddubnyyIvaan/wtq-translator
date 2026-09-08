"""Перевод табличных кандидатов-подпоследовательностей по ярусам длины + реконструкция таблицы."""
from __future__ import annotations

import json

from ..config import AppConfig
from ..llm.base import LLMProvider
from ..llm.tracing import llm_trace_context
from ..models import CandidateSet, TableCandidate, TableCoordinate, TableData
from ..persistence.cache import SQLiteCache
from ..prompts import CANDIDATE_TRANSLATION_SYSTEM
from .table import TableTranslationResponse, _parse_json

# In-memory кэш переводов кандидатов на процесс (ключ — candidate.id).
_MEMORY_CACHE: dict[str, str] = {}


def _validate_candidate_response(
    pending: list[TableCandidate], response: TableTranslationResponse
) -> dict[str, str]:
    """Проверяет, что модель вернула ровно по одному переводу на каждого кандидата и без пустых."""
    expected_ids = [c.id for c in pending]
    returned_ids = [item.id for item in response.translations]

    seen: set[str] = set()
    duplicates: list[str] = []
    for item_id in returned_ids:
        if item_id in seen and item_id not in duplicates:
            duplicates.append(item_id)
        seen.add(item_id)
    if duplicates:
        raise ValueError(f"Model returned duplicate translation IDs: {duplicates[:10]}")

    missing = [i for i in expected_ids if i not in returned_ids]
    unexpected = [i for i in returned_ids if i not in expected_ids]
    if missing or unexpected or len(returned_ids) != len(expected_ids):
        raise ValueError(
            "Translation ID set mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
            f"expected_count={len(expected_ids)}, returned_count={len(returned_ids)}"
        )

    mapping = {item.id: item.translation for item in response.translations}
    empty = [c.id for c in pending if c.source_text.strip() and not mapping[c.id].strip()]
    if empty:
        raise ValueError(f"Model returned empty translations for non-empty values: {empty[:10]}")
    return mapping


async def translate_table_candidates(
    provider: LLMProvider,
    candidates: list[TableCandidate],
    *,
    config: AppConfig,
    source_language: str,
    target_language: str,
    cache: SQLiteCache | None = None,
    trace_stage_prefix: str = "translate_table_candidates",
) -> dict[str, str]:
    """Переводит кандидатов по ярусам возрастающей длины, заполняя candidate.translation.

    Возвращает {candidate.id: translation}. Для согласованности каждый следующий ярус
    получает глоссарий переводов всех более коротких кандидатов.
    """
    if not candidates:
        return {}

    by_length: dict[int, list[TableCandidate]] = {}
    for c in candidates:
        by_length.setdefault(c.length, []).append(c)

    for length in sorted(by_length):
        tier = by_length[length]

        pending: list[TableCandidate] = []
        for c in tier:
            memo = _MEMORY_CACHE.get(c.id)
            if memo is not None:
                c.translation = memo
                continue
            if cache is not None:
                cached = cache.get_candidate_translation(c.id)
                if cached is not None:
                    c.translation = cached
                    _MEMORY_CACHE[c.id] = cached
                    continue
            pending.append(c)

        if not pending:
            continue

        glossary = [
            {"key": c.source_text, "translation": c.translation}
            for c in candidates
            if c.length < length and c.translation
        ]

        payload = {
            "task": "translate_candidates",
            "source_language": source_language,
            "target_language": target_language,
            "values": [{"id": c.id, "text": c.source_text} for c in pending],
            "glossary": glossary,
            "requirements": [
                "Translate every supplied value into the target language.",
                "Use the glossary translations to keep longer subsequences consistent with shorter ones.",
                "Return exactly one translation for every supplied id.",
                "Preserve every id exactly; do not add, remove, rename, or duplicate ids.",
                "Preserve numbers, dates, and punctuation inside the source text exactly.",
                "Translate the text only; do not answer questions, calculate, summarize, or add facts.",
                "Do not return any table coordinates, source text, explanations, or extra keys.",
            ],
        }

        last_error: Exception | None = None
        translations: dict[str, str] | None = None
        for attempt in range(config.model.retries + 1):
            request_payload = dict(payload)
            if last_error is not None:
                request_payload["retry_instruction"] = (
                    "The previous response was invalid. Translate the same values again. Return exactly one JSON "
                    "DATA object containing translations, with exactly one entry for every supplied id and no extra ids."
                )
                request_payload["previous_error"] = str(last_error)[:1400]

            try:
                with llm_trace_context(
                    stage=f"{trace_stage_prefix}.tier{length}",
                    attempt=attempt,
                    unique_values=len(pending),
                ):
                    raw = await provider.generate(
                        system=CANDIDATE_TRANSLATION_SYSTEM,
                        user=json.dumps(request_payload, ensure_ascii=False),
                        temperature=config.model.temperature,
                        max_output_tokens=config.model.max_output_tokens,
                        seed=config.project.seed + 900_000 + attempt,
                    )

                parsed = TableTranslationResponse.model_validate(_parse_json(raw))
                translations = _validate_candidate_response(pending, parsed)
                break
            except Exception as exc:
                last_error = exc

        if translations is None:
            raise RuntimeError(f"Candidate translation failed after retries: {last_error}")

        for c in pending:
            c.translation = translations[c.id]
            _MEMORY_CACHE[c.id] = c.translation
            if cache is not None:
                cache.put_candidate_translation(c.id, c.source_text, target_language, c.translation)

    return {c.id: c.translation for c in candidates}


def reconstruct_table(
    table: TableData, candidate_set: CandidateSet, translations: dict[str, str]
) -> TableData:
    """Реконструирует таблицу из переводов content-run'ов, сохраняя пунктуацию verbatim.

    translations: lemma_key -> текст. Run без перевода (числа/даты) остаётся исходной подстрокой.
    """
    headers = list(table.headers)
    rows = [r.copy() for r in table.rows]

    def rebuild(coord: TableCoordinate, text: str) -> str:
        runs = sorted(candidate_set.runs.get(coord, []), key=lambda r: r.start)
        if not runs:
            return text
        parts: list[str] = []
        pos = 0
        for run in runs:
            parts.append(text[pos:run.start])
            translated = translations.get(run.lemma_key)
            parts.append(translated if translated is not None else text[run.start:run.end])
            pos = run.end
        parts.append(text[pos:])
        return "".join(parts)

    for col in range(len(headers)):
        headers[col] = rebuild(TableCoordinate(None, col), headers[col])
    for r, row in enumerate(rows):
        for c in range(len(row)):
            rows[r][c] = rebuild(TableCoordinate(r, c), rows[r][c])

    return TableData(headers, rows, table.source_path)


def validate_untranslated(candidate_set: CandidateSet, translations: dict[str, str]) -> list[str]:
    """Возвращает описания run'ов, у которых есть кандидат, но нет перевода."""
    candidate_keys = {c.lemma_key for c in candidate_set.candidates}
    errors: list[str] = []
    for coord, runs in candidate_set.runs.items():
        for run in runs:
            if run.lemma_key in candidate_keys and run.lemma_key not in translations:
                errors.append(f"{coord}: {run.lemma_key!r} not translated")
    return errors
