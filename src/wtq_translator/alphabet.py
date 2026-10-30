"""Детекция алфавитно-зависимых вопросов (режим full).

Вопросы, чья семантика зависит от алфавита языка (алфавитная сортировка, положение
по алфавиту, сравнение по алфавиту, конкретные буквы), не переводятся: модель не
может передать алфавитный порядок в целевом языке. Такие вопросы исключаются из
переведённого датасета и сохраняются в отдельный TSV.

Два метода классификации (``alphabet_detection.method``):
- ``llm`` (по умолчанию) — промпт-классификация намерения через LLM-провайдера;
- ``regex`` — детерминированные регулярки по словоформам (детальный отсев):

  | Корень        | Покрываемые формы                                   | Regex                   | intent |
  |---------------|-----------------------------------------------------|-------------------------|--------|
  | alphabet      | alphabet, alphabetic(al)(ly), alphabetize(d/sed)…   | ``\\balphabet``           | 1      |
  | lexicograph   | lexicographic(al)(ly)…                              | ``\\blexicograph``        | 1      |
  | letter        | letter(s), letter's…                                | ``\\bletter(?:s)?\\b``    | 4      |
  | character     | character(s)…                                       | ``\\bcharacter(?:s)?\\b`` | 4      |

  ``\\b``-границы исключают ``newsletter``, ``letterhead``, ``characteristic``
  и т.п.; ``re.IGNORECASE`` ловит ``ALPHABETICALLY``. Остаточные ложные
  срабатывания: ``letter`` = «письмо», ``character`` = «персонаж» — приняты
  (у regex-метода recall выше, precision ниже, чем у LLM; для отсева норма).
"""
from __future__ import annotations

import json
import re

from .config import AppConfig
from .llm.base import LLMProvider
from .llm.tracing import llm_trace_context
from .models import AlphabetDetection
from .prompts import ALPHABET_DETECTION_SYSTEM
from .translation.table import _parse_json


class AlphabetIntentError(Exception):
    """Вопрос исключён из перевода: он зависит от алфавита или детекция не удалась.

    ``intent_label`` — название намерения (класс в TSV исключений); при ошибке
    детекции (fail-closed) — пустая строка. ``why`` — обоснование модели либо текст
    ошибки детекции.
    """

    def __init__(self, question_id: str, *, intent_label: str, why: str) -> None:
        self.question_id = question_id
        self.intent_label = intent_label
        self.why = why
        super().__init__(f"question {question_id}: alphabet-dependent ({intent_label}): {why}")


# id намерения -> название намерения (класс записи в TSV исключений).
ALPHABET_INTENT_LABELS: dict[int, str] = {
    1: "алфавитная сортировка",
    2: "положение или группирование с учётом алфавитного порядка",
    3: "сравнение нескольких значений по алфавиту",
    4: "подсчёт, наличие/отсутствие или сравнение по конкретной букве или множеству букв",
}

_VALID_INTENT_IDS = frozenset({1, 2, 3, 4})

# Корни алфавитной сортировки (intent 1): префиксное сопоставление без правой
# границы — покрывает alphabet, alphabetic(al)(ly), alphabetize(d), lexicographic(al)(ly).
_ALPHABET_SORT_RE = re.compile(r"\b(?:alphabet|lexicograph)", re.IGNORECASE)
# Конкретные буквы (intent 4): правая \b исключает characteristic, letterhead и т.п.
_LETTER_CHAR_RE = re.compile(r"\b(?:letter(?:s)?)\b", re.IGNORECASE)


def detect_alphabet_intent_rules(question_text: str) -> AlphabetDetection:
    """Детерминированная regex-классификация алфавитно-зависимых вопросов.

    Ищет корни ``alphabet*|lexicograph*`` (intent 1) и ``letter*|character*``
    (intent 4). Приоритет у intent 1: если оба корня совпали — класс 1.
    ``why`` — одна строка ``regex: keyword(s) 'word1, word2'`` с полными
    найденными словоформами (порядок: alphabet/lexicograph, затем letter/character).
    Ошибок не бывает: при отсутствии совпадений — ``(None, None)``, ``error``
    всегда ``None``.
    """
    matches = list(_ALPHABET_SORT_RE.finditer(question_text))
    alphabet_hit = bool(matches)
    matches.extend(_LETTER_CHAR_RE.finditer(question_text))

    keywords: list[str] = []
    for match in matches:
        word = _expand_alpha_span(question_text, match.span())
        if word not in keywords:
            keywords.append(word)
    if not keywords:
        return AlphabetDetection(None, None)

    why = f"regex: keyword(s) '{', '.join(keywords)}'"
    return AlphabetDetection(1 if alphabet_hit else 4, why)


def _expand_alpha_span(text: str, span: tuple[int, int]) -> str:
    """Расширяет span regex-совпадения до полной словоформы (буквы в обе стороны)."""
    start, end = span
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    while end < len(text) and text[end].isalpha():
        end += 1
    return text[start:end]


async def detect_alphabet_intent(
    provider: LLMProvider,
    config: AppConfig,
    *,
    question_id: str,
    question_text: str,
) -> AlphabetDetection:
    """Определяет, зависит ли семантика вопроса от алфавита языка.

    Возвращает ``AlphabetDetection``. При выключенной конфигурации — ``(None, None)``
    без вызова провайдера. При ``alphabet_detection.method == "regex"`` —
    детерминированная классификация без вызова провайдера (см.
    ``detect_alphabet_intent_rules``). При ``intent_id: null`` вопрос допускается —
    ``(None, None)``. При исчерпании ретраев или некорректном ответе — ``(None, None, error)``
    (fail-closed), исключение не бросается; решение об исключении принимает вызывающий
    код.
    """
    if not config.alphabet_detection.enabled:
        return AlphabetDetection(None, None)

    if config.alphabet_detection.method == "regex":
        return detect_alphabet_intent_rules(question_text)

    payload = {
        "task": "detect_alphabet_intent",
        "question_id": question_id,
        "question": question_text,
    }

    last_error: Exception | None = None
    for attempt in range(config.model.retries + 1):
        request_payload = dict(payload)
        if last_error is not None:
            request_payload["retry_instruction"] = (
                "Предыдущий ответ невалиден. Определи намерение вопроса ещё раз. "
                "Верни ровно один JSON-объект вида {\"intent_id\": null|1|2|3|4, "
                "\"why\": null|\"...\"}."
            )
            request_payload["previous_error"] = str(last_error)[:1400]

        try:
            with llm_trace_context(stage="detect_alphabet_intent", attempt=attempt):
                raw = await provider.generate(
                    system=ALPHABET_DETECTION_SYSTEM,
                    user=json.dumps(request_payload, ensure_ascii=False),
                    temperature=config.model.temperature,
                    max_output_tokens=config.model.max_output_tokens,
                    seed=config.project.seed + 1_300_000 + attempt,
                )

            parsed = _parse_json(raw)
            intent_id = parsed.get("intent_id")
            if intent_id is None:
                return AlphabetDetection(None, None)
            if not isinstance(intent_id, int) or isinstance(intent_id, bool):
                raise ValueError(f"intent_id must be an int in {sorted(_VALID_INTENT_IDS)} or null, got: {intent_id!r}")
            if intent_id not in _VALID_INTENT_IDS:
                raise ValueError(f"intent_id must be one of {sorted(_VALID_INTENT_IDS)}, got: {intent_id!r}")
            why = parsed.get("why")
            if not isinstance(why, str) or not why.strip():
                raise ValueError("why must be a non-empty string when an intent is defined")
            return AlphabetDetection(intent_id, why.strip())
        except Exception as exc:
            last_error = exc

    return AlphabetDetection(
        None,
        None,
        error=f"Alphabet detection failed after retries: {last_error}",
    )