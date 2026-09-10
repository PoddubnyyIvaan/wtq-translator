"""Перевод табличных кандидатов-подпоследовательностей по ярусам длины + реконструкция таблицы."""
from __future__ import annotations

import asyncio
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

# Требования, общие для всех запросов перевода кандидатов внутри яруса.
_CANDIDATE_REQUIREMENTS = [
    "Translate every supplied value into the target language.",
    "Use the glossary translations to keep longer subsequences consistent with shorter ones.",
    "Return exactly one translation for every supplied id.",
    "Preserve every id exactly; do not add, remove, rename, or duplicate ids.",
    "Preserve numbers, dates, and punctuation inside the source text exactly.",
    "Translate the text only; do not answer questions, calculate, summarize, or add facts.",
    "Do not return any table coordinates, source text, explanations, or extra keys.",
]


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


async def _translate_chunk(
    provider: LLMProvider,
    chunk: list[TableCandidate],
    *,
    glossary: list[dict[str, str]],
    length: int,
    chunk_idx: int,
    config: AppConfig,
    source_language: str,
    target_language: str,
    trace_stage_prefix: str,
) -> dict[str, str]:
    """Переводит один chunk кандидатов яруса со своим retry-циклом.

    Возвращает {candidate.id: translation} для кандидатов chunk. Глоссарий для всех
    chunk'ов яруса одинаковый (строится от всех более коротких кандидатов). При
    исчерпании ретраев бросает RuntimeError — изоляция ошибок теперь пер-chunk,
    неудача одного chunk не отменяет запросы остальных.
    """
    payload = {
        "task": "translate_candidates",
        "source_language": source_language,
        "target_language": target_language,
        "values": [{"id": c.id, "text": c.source_text} for c in chunk],
        "glossary": glossary,
        "requirements": _CANDIDATE_REQUIREMENTS,
    }

    last_error: Exception | None = None
    translations: dict[str, str] | None = None
    # Смещение seed зависит от chunk_idx и попытки: у chunk'ов непересекающиеся
    # диапазоны seed'ов, при chunk_idx=0 формула совпадает с прежним поведением.
    seed_base = config.project.seed + 900_000 + chunk_idx * (config.model.retries + 1)
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
                chunk=chunk_idx,
                unique_values=len(chunk),
            ):
                raw = await provider.generate(
                    system=CANDIDATE_TRANSLATION_SYSTEM,
                    user=json.dumps(request_payload, ensure_ascii=False),
                    temperature=config.model.temperature,
                    max_output_tokens=config.model.max_output_tokens,
                    seed=seed_base + attempt,
                )

            parsed = TableTranslationResponse.model_validate(_parse_json(raw))
            translations = _validate_candidate_response(chunk, parsed)
            break
        except Exception as exc:
            last_error = exc

    if translations is None:
        raise RuntimeError(f"Candidate translation failed after retries: {last_error}")
    return translations


def _relevant_glossary(
    values: list[TableCandidate],
    shorter: list[TableCandidate],
) -> list[dict[str, str]]:
    """Возвращает записи глоссария для кандидатов values.

    Для каждой фразы из values выбираются непересекающиеся по токен-позициям окна
    (lemma_key окна — непрерывное подокно токенов фразы, присутствующее в shorter):
    из пересекающихся окон остаётся самое длинное, при равенстве длины — самое левое;
    вложенные в выбранные окна удаляются. Выбранные lemma_key объединяются по всем
    values, порядок результата сохраняется по порядку `shorter`.
    """
    if not values or not shorter:
        return []
    by_lemma = {c.lemma_key: c for c in shorter}
    chosen: set[str] = set()
    for value in values:
        tokens = value.lemma_key.split(" ")
        n = len(tokens)
        # Окна фразы, присутствующие среди более коротких кандидатов.
        windows: list[tuple[int, int, str]] = []  # (start, end, lemma_key)
        for i in range(n):
            window = tokens[i]
            for j in range(i, n):
                if j > i:
                    window = window + " " + tokens[j]
                if window in by_lemma:
                    windows.append((i, j + 1, window))
        if not windows:
            continue
        # Самые длинные первыми; при равенстве длины — левое окно (первое во фразе).
        windows.sort(key=lambda w: (-(w[1] - w[0]), w[0]))
        kept: list[tuple[int, int]] = []  # отсортированы по start, без пересечений
        for start, end, key in windows:
            # Бинарный поиск позиции вставки по start среди выбранных окон.
            lo, hi = 0, len(kept)
            while lo < hi:
                mid = (lo + hi) // 2
                if kept[mid][0] < start:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > 0 and kept[lo - 1][1] > start:
                continue  # пересечение с левым соседом
            if lo < len(kept) and kept[lo][0] < end:
                continue  # пересечение с правым соседом
            kept.insert(lo, (start, end))
            chosen.add(key)
    return [
        {"key": c.source_text, "translation": c.translation}
        for c in shorter
        if c.lemma_key in chosen
    ]


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
    получает глоссарий переводов всех более коротких кандидатов. Внутри яруса pending
    может быть разбит на chunk'и (translation.candidates.max_candidates_per_request),
    которые запрашиваются параллельно с ограничением
    translation.candidates.request_concurrency. Дефолты (None/1) дают один
    последовательный запрос на ярус — как раньше.
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

        shorter = [c for c in candidates if c.length < length and c.translation]

        limit = config.translation.candidates.max_candidates_per_request
        if limit is None or len(pending) <= limit:
            chunks = [pending]
        else:
            # Упорядоченные непересекающиеся подсписки размера limit, полное покрытие.
            chunks = [pending[i : i + limit] for i in range(0, len(pending), limit)]

        sem = asyncio.Semaphore(config.translation.candidates.request_concurrency)

        async def _run_chunk(
            chunk_idx: int, chunk: list[TableCandidate]
        ) -> dict[str, str]:
            async with sem:
                return await _translate_chunk(
                    provider,
                    chunk,
                    glossary=_relevant_glossary(chunk, shorter),
                    length=length,
                    chunk_idx=chunk_idx,
                    config=config,
                    source_language=source_language,
                    target_language=target_language,
                    trace_stage_prefix=trace_stage_prefix,
                )

        # return_exceptions=True: результат успешного chunk — dict[id, перевод],
        # упавшего — исключение; ошибка одного chunk не отменяет остальные задачи.
        results = await asyncio.gather(
            *(_run_chunk(idx, chunk) for idx, chunk in enumerate(chunks)),
            return_exceptions=True,
        )

        first_error: BaseException | None = None
        for chunk_idx, chunk in enumerate(chunks):
            result = results[chunk_idx]
            if isinstance(result, BaseException):
                if first_error is None:
                    first_error = result
                continue
            # Применяем и кэшируем переводы успешных chunk'ов, даже если другой
            # chunk упал: уже полученные переводы не теряются при ретрае на resume.
            for c in chunk:
                translation = result[c.id]
                c.translation = translation
                _MEMORY_CACHE[c.id] = translation
                if cache is not None:
                    cache.put_candidate_translation(c.id, c.source_text, target_language, translation)

        if first_error is not None:
            # Пробрасываем ошибку chunk'а как раньше (после исчерпания ретраев).
            if isinstance(first_error, RuntimeError):
                raise first_error
            raise RuntimeError(f"Candidate translation failed after retries: {first_error}")

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
