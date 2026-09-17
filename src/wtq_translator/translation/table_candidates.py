"""Перевод табличных кандидатов-подпоследовательностей по ярусам убывающей длины
(глоссарий: самые длинные вмещающие последовательности) + реконструкция таблицы."""
from __future__ import annotations

import asyncio
import json

from ..config import AppConfig
from ..llm.base import LLMProvider
from ..llm.tracing import llm_trace_context
from ..models import CandidateSet, TableCandidate, TableCoordinate, TableData
from ..persistence.cache import SQLiteCache
from ..prompts import candidate_system_prompt
from .table import TableTranslationResponse, _parse_json

# In-memory кэш переводов кандидатов на процесс (ключ — candidate.id).
_MEMORY_CACHE: dict[str, str] = {}

# Требования, общие для всех запросов перевода кандидатов внутри яруса.
_CANDIDATE_REQUIREMENTS = [
    "Translate every supplied value into the target language.",
    "Use each item's glossary to keep shorter subsequences consistent with the longer ones they are part of.",
    "The `column` field is context only and must not be translated or returned.",
    "Return exactly one translation for every supplied id.",
    "Preserve every id exactly; do not add, remove, rename, or duplicate ids.",
    "Preserve numbers, dates, and punctuation inside the source text exactly.",
    "Translate the text only; do not answer questions, calculate, summarize, or add facts.",
    "Do not return any table coordinates, source text, explanations, or extra keys.",
    "For items where `is_column` is false, translate consistent with the meaning of the `column` field; when `column_translation` is provided, prefer its terms.",
]

# Требование форсинга транслитерации; добавляется к _CANDIDATE_REQUIREMENTS только когда
# включен translation.candidates.transliteration_enabled.
_TRANSLITERATION_REQUIREMENT = (
    "When an item provides a `transliterations` list, use exactly the supplied `translation` "
    "for the exact `source` word inside that item; do not translate, paraphrase, or omit that word."
)


_FORCED_NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}


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
    items: list[dict[str, object]],
    length: int,
    chunk_idx: int,
    config: AppConfig,
    source_language: str,
    target_language: str,
    trace_stage_prefix: str,
    domain: str,
    requirements: list[str],
) -> dict[str, str]:
    """Переводит один chunk кандидатов яруса со своим retry-циклом.

    Возвращает {candidate.id: translation} для кандидатов chunk. items — готовые
    payload-записи `{"id","text","column","glossary"}` для этого chunk'а (glossary
    каждого item'а — его собственные самые длинные вмещающие последовательности).
    requirements — требования уровня промпта (общие + при включенном форсинге
    транслитерации). При исчерпании ретраев бросает RuntimeError — изоляция ошибок
    теперь пер-chunk, неудача одного chunk не отменяет запросы остальных.
    """
    payload = {
        "task": "translate_candidates",
        "source_language": source_language,
        "target_language": target_language,
        "items": items,
        "requirements": requirements,
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
                    system=candidate_system_prompt(domain),
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


def _is_contiguous_window(window: list[str], full: list[str]) -> bool:
    """True, если window — непрерывное подокно full (порядок токенов идёт подряд)."""
    n, m = len(window), len(full)
    if n > m:
        return False
    return any(full[i : i + n] == window for i in range(m - n + 1))


# Кириллические целевые языки: только для них осмысленна транслитерация латиницы→кириллицы.
_CYRILLIC_TARGETS = {"ru", "uk", "bg", "mk", "be"}


def _translit_token(token: str, target_lang: str) -> str | None:
    """Транслитерирует одиночный латинский токен в кириллицу для целевого языка.

    Возвращает None, если: токен без латинских букв, целевой язык не кириллический,
    библиотека не установлена или произошла ошибка транслитерации (fallback None —
    токен не попадает в forcing).
    """
    if not any(ch.isascii() and ch.isalpha() for ch in token):
        return None
    if target_lang.lower() not in _CYRILLIC_TARGETS:
        return None
    try:
        from transliterate import translit
    except ImportError:
        return None
    try:
        # RU-пакет: reversed=False даёт латиница→кириллица ("Berlin" -> "Берлин").
        return translit(token, "ru", reversed=False).casefold()
    except Exception:
        return None


def _forced_transliterations(c: TableCandidate, target_lang: str) -> list[dict[str, str]]:
    """Forcing-записи для NER-токенов кандидата: источник и транслит, оба в нижнем регистре.

    Дедуп по (source, translation) в порядке токенов кандидата.
    """
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token, ner in zip(c.tokens, c.ner):
        if ner not in _FORCED_NER_LABELS:
            continue
        translit = _translit_token(token, target_lang)
        if translit is None:
            continue
        key = (token.casefold(), translit)
        if key in seen:
            continue
        seen.add(key)
        result.append({"source": token.casefold(), "translation": translit})
    return result


def _nested_longest_glossary(
    value: TableCandidate, longer: list[TableCandidate]
) -> list[dict[str, str]]:
    """Возвращает глоссарий для value: переводы самых длинных уже переведённых
    более длинных последовательностей, в которые value вложено как непрерывное
    подокно токенов lemma_key. При нескольких одинаково длинных — включаются все,
    с дедупом по (source_text, translation) и сохранением порядка `longer`.

    Пустой глоссарий возвращается, когда value не вложено ни в одну более длинную
    переведённую последовательность (в т.ч. для самого длинного яруса).
    """
    if not longer:
        return []
    tokens = value.lemma_key.split(" ")
    matches: list[TableCandidate] = []
    for c in longer:
        if c.length <= value.length or not c.translation:
            continue
        if _is_contiguous_window(tokens, c.lemma_key.split(" ")):
            matches.append(c)
    if not matches:
        return []
    max_len = max(c.length for c in matches)
    longest = [c for c in matches if c.length == max_len]

    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for c in longest:
        key = (c.source_text.casefold(), c.translation.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append({"key": c.source_text.casefold(), "translation": c.translation.casefold()})
    return result


async def translate_table_candidates(
    provider: LLMProvider,
    candidates: list[TableCandidate],
    *,
    config: AppConfig,
    source_language: str,
    target_language: str,
    cache: SQLiteCache | None = None,
    trace_stage_prefix: str = "translate_table_candidates",
    domain: str = "",
    headers: list[str] | None = None,
    runs: dict[TableCoordinate, list[ContentRun]] | None = None,
) -> dict[str, str]:
    """Переводит кандидатов по ярусам убывающей длины, заполняя candidate.translation.

    Возвращает {candidate.id: translation}. Ярусы обрабатываются от большего к меньшему:
    для согласованности каждый следующий (меньший) ярус получает per-item глоссарий —
    переводы самых длинных уже переведённых более длинных последовательностей, в которые
    переводимая подпоследовательность вложена. `column` каждого item — полный заголовок
    колонки первой координаты кандидата (из `headers`), только как контекст. Заголовочные
    кандидаты (`is_column: true`) получают только контекст; нестолбцовые — `is_column:
    false`, `column_translation` (перевод заголовка своей колонки, если уже переведён) и
    требование согласованности. NER-токены (PERSON/ORG/GPE/LOC/PRODUCT) через `tokens`/`ner`
    кандидата транслитерируются в `transliterations`. `runs` (per-координата content-run'ы)
    нужен для получения lemma_key заголовков колонок.

    Внутри яруса pending может быть разбит на chunk'и
    (translation.candidates.max_candidates_per_request), которые запрашиваются параллельно
    с ограничением translation.candidates.request_concurrency. Дефолты (None/1) дают один
    последовательный запрос на ярус — как раньше.
    """
    if not candidates:
        return {}

    transliteration_enabled = config.translation.candidates.transliteration_enabled
    requirements = (
        [*_CANDIDATE_REQUIREMENTS, _TRANSLITERATION_REQUIREMENT]
        if transliteration_enabled
        else _CANDIDATE_REQUIREMENTS
    )

    by_length: dict[int, list[TableCandidate]] = {}
    for c in candidates:
        by_length.setdefault(c.length, []).append(c)

    for length in sorted(by_length, reverse=True):
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

        # Контекст для глоссария: все уже переведённые более длинные кандидаты.
        longer = [c for c in candidates if c.length > length and c.translation]

        # lemma_key -> translation уже переведённых кандидатов на момент сборки яруса
        # (для column_translation нестолбцовых кандидатов).
        translated_lemmas = {c.lemma_key: c.translation for c in candidates if c.translation}

        # Заголовок каждой колонки по её lemma_key (из content-runs заголовков).
        col_to_header_lemma: dict[int, str] = {}
        if runs is not None and headers:
            for col in range(len(headers)):
                cell_runs = runs.get(TableCoordinate(None, col))
                if cell_runs:
                    col_to_header_lemma[col] = cell_runs[0].lemma_key

        # Готовые payload-записи, выровненные по порядку `pending`.
        items: list[dict[str, object]] = []
        for c in pending:
            column = ""
            if headers and c.coordinates and 0 <= c.coordinates[0].column < len(headers):
                column = headers[c.coordinates[0].column]
            is_column = bool(c.coordinates) and c.coordinates[0].row is None

            item: dict[str, object] = {
                "id": c.id,
                "text": c.source_text,
                "column": column.casefold() if column else "",
                "is_column": is_column,
                "glossary": _nested_longest_glossary(c, longer),
            }

            transliterations = (
                _forced_transliterations(c, target_language) if transliteration_enabled else []
            )
            if transliterations:
                item["transliterations"] = transliterations

            if not is_column and c.coordinates:
                header_lemma = col_to_header_lemma.get(c.coordinates[0].column)
                if header_lemma is not None:
                    header_translation = translated_lemmas.get(header_lemma)
                    if header_translation:
                        item["column_translation"] = header_translation.casefold()

            items.append(item)

        limit = config.translation.candidates.max_candidates_per_request
        if limit is None or len(pending) <= limit:
            chunks = [pending]
        else:
            # Упорядоченные непересекающиеся подсписки размера limit, полное покрытие.
            chunks = [pending[i : i + limit] for i in range(0, len(pending), limit)]

        chunk_items: list[list[dict[str, object]]] = []
        pos = 0
        for chunk in chunks:
            chunk_items.append(items[pos : pos + len(chunk)])
            pos += len(chunk)

        sem = asyncio.Semaphore(config.translation.candidates.request_concurrency)

        async def _run_chunk(
            chunk_idx: int, chunk: list[TableCandidate]
        ) -> dict[str, str]:
            async with sem:
                return await _translate_chunk(
                    provider,
                    chunk,
                    items=chunk_items[chunk_idx],
                    length=length,
                    chunk_idx=chunk_idx,
                    config=config,
                    source_language=source_language,
                    target_language=target_language,
                    trace_stage_prefix=trace_stage_prefix,
                    domain=domain,
                    requirements=requirements,
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
