from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, Field

from ..config import ModelConfig
from ..llm.base import LLMProvider
from ..llm.tracing import llm_trace_context
from ..mentions.masking import closing_mask, strip_tags
from ..models import TableMention
from ..prompts import (
    SHELL_TRANSLATION_SYSTEM,
    SHELL_TRANSLATION_TAGGED_SYSTEM,
    question_translation_system_prompt,
)
from ..validation.masks import extract_close_tags, extract_open_tags, validate_masks, validate_tags


class ShellResponse(BaseModel):
    translated_text: str


class QuestionMentionUsage(BaseModel):
    id: str
    surface: str
    lemma: str


class QuestionTranslationResponse(BaseModel):
    question: str
    question_tagged: str
    mentions_used: list[QuestionMentionUsage] = Field(default_factory=list)
    missing_mentions: list[str] = Field(default_factory=list)


class ShellValidationError(ValueError):
    """Ошибка валидации структурированного перевода вопроса.

    Проверки НЕ прерываются на первой ошибке: все замечания (Tags changed,
    Surface check и т.д.) собираются в `errors`, а сырой ответ модели
    сохраняется в `raw`. Это позволяет ретраю передать модели её же последний
    ответ и список конкретных ошибок, чтобы исправлять точечно, а не генерить
    перевод заново с нуля.
    """

    def __init__(self, errors: Sequence[str], raw: str):
        self.errors = list(errors)
        self.raw = raw
        super().__init__("; ".join(self.errors))


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as second_error:
                preview = cleaned[:1200].replace("\n", "\\n")
                raise ValueError(f"Invalid shell JSON: {second_error}. Raw preview: {preview}") from second_error
        preview = cleaned[:1200].replace("\n", "\\n")
        raise ValueError(f"Invalid shell JSON: {first_error}. Raw preview: {preview}") from first_error


def _tagged_worked_example(source_language: str, target_language: str) -> dict[str, str] | None:
    """Returns a direction-matched worked example for the tagged shell prompt.

    The example demonstrates the two rules that models most often violate:
    1) a tag pair is NOT dropped or unwrapped - it stays glued around its content;
    2) the pair may travel together with its content to follow the target-language
    word order, and the content is the tag_glossary word inflected to agree with
    the sentence (en: "<the team>'s captain" -> ru: "капитан <команды>").
    """
    if source_language == "en" and target_language == "ru":
        return {
            "input": "who was <TABLE_REF_0000>the team</TABLE_REF_0000>'s captain in 2001?",
            "glossary": {"<TABLE_REF_0000>": "команда"},
            "output": "кем был капитан <TABLE_REF_0000>команды</TABLE_REF_0000> в 2001 году?",
        }
    if source_language == "ru" and target_language == "en":
        return {
            "input": "кем был капитан <TABLE_REF_0000>команды</TABLE_REF_0000> в 2001 году?",
            "glossary": {"<TABLE_REF_0000>": "the team"},
            "output": "who was <TABLE_REF_0000>the team</TABLE_REF_0000>'s captain in 2001?",
        }
    return None


# --- Structured question-translation protocol («перевод вопроса по шаблону») ---


def _normalize_response_keys(raw: dict, target_language: str) -> dict:
    """Приводит поля ответа `question_<lang>`, `surface_<lang>` и т.п. к общим ключам."""
    surface_key = f"surface_{target_language}"
    lemma_key = f"lemma_{target_language}"
    return {
        "question": raw.get(f"question_{target_language}", ""),
        "question_tagged": raw.get(f"question_{target_language}_tagged", ""),
        "mentions_used": [
            {
                "id": item.get("id", ""),
                "surface": item.get(surface_key, ""),
                "lemma": item.get(lemma_key, ""),
            }
            for item in raw.get("mentions_used", [])
        ],
        "missing_mentions": raw.get("missing_mentions", []),
    }


def _parse_question_translation(
    raw: str, target_language: str, mentions: Sequence[TableMention]
) -> QuestionTranslationResponse:
    """Парсит ответ структурированного перевода вопроса и проверяет его целостность.

    Все проверки выполняются до конца и не прерываются на первой ошибке:
    покрытие id, пустые поля и сохранение пар тегов. Все замечания собираются и, если они
    есть, выбрасываются как ShellValidationError со списком `errors` и сырым
    ответом `raw` — так ретрай может точечно исправлять прошлую генерацию.

    Согласованность plain- и tagged-вариантов НЕ проверяется: литературный
    plain-перевод и тегированный перевод независимы, итоговый вопрос
    строится из tagged (strip_tags).
    """
    parsed = QuestionTranslationResponse.model_validate(
        _normalize_response_keys(_extract_json(raw), target_language)
    )

    expected_ids = {m.candidate_key for m in mentions}

    # A2: при пустом списке mentions никаких тегов не ожидается. Вместо падения на
    # «unknown ids» вычищаем надуманные теги и usage: если mentions пуст, любой тег
    # в ответе — галлюцинация по определению.
    if not expected_ids:
        parsed.question_tagged = strip_tags(parsed.question_tagged)
        parsed.mentions_used = []
        parsed.missing_mentions = []

    used_ids = {u.id for u in parsed.mentions_used}
    missing_ids = set(parsed.missing_mentions)

    errors: list[str] = []

    unknown = (used_ids | missing_ids) - expected_ids
    if unknown:
        errors.append(f"Unknown mention ids in response: {sorted(unknown)[:10]}")
    overlap = used_ids & missing_ids
    if overlap:
        errors.append(f"Ids in both mentions_used and missing_mentions: {sorted(overlap)[:10]}")
    dropped = expected_ids - used_ids - missing_ids
    if dropped:
        errors.append(f"Mentions neither used nor marked missing: {sorted(dropped)[:10]}")
    if not parsed.question_tagged.strip():
        errors.append("Empty question_tagged in response")

    # Пары тегов должны быть сохранены символ-в-символ.
    expected_counts: Counter[str] = Counter()
    for m in mentions:
        expected_counts[m.mask] += 1
        expected_counts[closing_mask(m.mask)] += 1
    actual_counts = Counter(extract_open_tags(parsed.question_tagged))
    actual_counts.update(extract_close_tags(parsed.question_tagged))
    if expected_counts != actual_counts:
        errors.append(
            "Tags changed: "
            f"missing={dict(expected_counts - actual_counts)}, added={dict(actual_counts - expected_counts)}"
        )

    if errors:
        raise ShellValidationError(errors, raw)
    return parsed


def build_question_translation_payload(
    masked_question: str,
    mentions: Sequence[TableMention],
    *,
    source_language: str,
    target_language: str,
    id_to_canonical: Mapping[str, str],
) -> dict:
    """Собирает user-JSON для перевода вопроса: тегированный вопрос и табличные
    упоминания (id, tag, исходный текст, канонический перевод, лемма, inflect_allowed).

    inflect_allowed=False для имён колонок и их частей (любая координата упоминания —
    заголовок), True для остальных упоминаний.
    """
    items: list[dict] = []
    for m in mentions:
        canonical = id_to_canonical.get(m.candidate_key, "")
        is_header = any(c.row is None for c in m.coordinates)
        items.append({
            "id": m.candidate_key,
            "tag": m.mask,
            "current_text": m.question_text,
            "translation": canonical,
            "inflect_allowed": not is_header,
        })
    return {
        "task": "translate_question",
        "source_language": source_language,
        "target_language": target_language,
        "instruction": (
            f"Translate the tagged_question from {source_language} "
            f"into {target_language}. Keep every mention from the `mentions` list in the "
            f"translation, wrapped in its tag."
        ),
        "tagged_question": masked_question,
        "mentions": items,
    }


def _structured_retry_instruction(target_language: str) -> str:
    return (
        "The previous response was invalid. Return exactly the JSON object documented in the "
        f"system prompt: question_{target_language}_tagged with every mention from the original "
        "`mentions` list wrapped in its own tag pair, the mentions_used array, and missing_mentions. "
        "Preserve every tag and id of the original mentions exactly; never add a tag for an id "
        "that is NOT in the `mentions` list."
    )


def _structured_fix_retry_instruction(target_language: str) -> str:
    return (
        "The previous response failed validation. Do NOT translate the question from scratch: "
        "take your previous `previous_response` answer and fix ONLY the `errors` listed below, "
        "then return the corrected JSON object with the exact same shape documented in the "
        f"system prompt (question_{target_language}_tagged, mentions_used, missing_mentions). "
        "Keep the tag pair (same id) of every mention that is in the original `mentions` list, "
        "and NEVER invent, rename, re-add, or keep a tag whose id is NOT in that list: if an error "
        "says an id is unknown, DELETE that tag pair and its mentions_used entry from the answer. "
        "Change only the parts that the errors point to; keep the rest of the translation unchanged."
    )


async def translate_shell(
    provider: LLMProvider,
    text: str,
    *,
    source_language: str,
    target_language: str,
    model: ModelConfig,
    seed: int,
    trace_stage: str = "translate_shell",
    tag_glossary: dict[str, str] | None = None,
    mentions: Sequence[TableMention] | None = None,
    id_to_canonical: Mapping[str, str] | None = None,
    domain: str = "",
    on_missing: Callable[[list[str]], None] | None = None,
) -> str:
    """Переводит «оболочку» вопроса. Три режима:

    * mentions is not None — структурированный перевод по новому шаблону
      (QUESTION_TRANSLATION_SYSTEM, заполненный языками и доменом): возвращает
      тегированный перевод, валидирует теги/покрытие упоминаний; при непустом
      `missing_mentions` вызывает `on_missing(ids)`.
    * tag_glossary is not None — legacy tagged-режим (глоссарий словарных переводов).
    * иначе — legacy mask-режим (неизменяемые <TABLE_REF_XXXX> плейсхолдеры).
    Поведение legacy-путей не меняется.
    """
    if mentions is not None:
        system = question_translation_system_prompt(source_language, target_language, domain)
        base_payload = build_question_translation_payload(
            text, mentions,
            source_language=source_language,
            target_language=target_language,
            id_to_canonical=id_to_canonical or {},
        )
    else:
        tagged = tag_glossary is not None
        system = SHELL_TRANSLATION_TAGGED_SYSTEM if tagged else SHELL_TRANSLATION_SYSTEM
        base_payload = {
            "source_language": source_language,
            "target_language": target_language,
            "text": text,
        }
        if tagged:
            base_payload["tag_glossary"] = tag_glossary
            example = _tagged_worked_example(source_language, target_language)
            if example is not None:
                base_payload["worked_example"] = example

    last_error: Exception | None = None
    for attempt in range(model.retries + 1):
        payload = dict(base_payload)
        if last_error is not None:
            if mentions is not None:
                if isinstance(last_error, ShellValidationError):
                    payload["retry_instruction"] = _structured_fix_retry_instruction(target_language)
                    payload["previous_response"] = last_error.raw
                    payload["errors"] = list(last_error.errors)
                else:
                    payload["retry_instruction"] = _structured_retry_instruction(target_language)
            elif tag_glossary is not None:
                payload["retry_instruction"] = (
                    "The previous response was invalid. Return the translated data object itself, not a schema, "
                    "and preserve every opening and closing TABLE_REF tag exactly. "
                    "Do not drop or unwrap tags: a tag pair moves together with its content to fit the "
                    "target-language word order, and the content inside the pair must be the matching "
                    "tag_glossary word inflected to agree with the sentence."
                )
            else:
                payload["retry_instruction"] = (
                    "The previous response was invalid. Return the translated data object itself, not a schema, "
                    "and preserve all TABLE_REF placeholders exactly."
                )
            payload["previous_error"] = str(last_error)[:1000]
        try:
            with llm_trace_context(stage=trace_stage, attempt=attempt):
                raw = await provider.generate(
                    system=system,
                    user=json.dumps(payload, ensure_ascii=False),
                    temperature=model.temperature,
                    max_output_tokens=model.max_output_tokens,
                    seed=seed + attempt,
                )
            if mentions is not None:
                parsed = _parse_question_translation(raw, target_language, mentions)
                if parsed.missing_mentions and on_missing is not None:
                    on_missing(sorted(parsed.missing_mentions))
                return parsed.question_tagged
            parsed = ShellResponse.model_validate(_extract_json(raw))
            mv = validate_tags(text, parsed.translated_text) if tag_glossary is not None else validate_masks(text, parsed.translated_text)
            if not mv.ok:
                raise ValueError(f"LLM changed placeholders: missing={mv.missing}, added={mv.added}")
            return parsed.translated_text
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Shell translation failed after retries: {last_error}")
