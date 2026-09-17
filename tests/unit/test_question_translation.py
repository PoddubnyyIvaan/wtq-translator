import json

import pytest

from wtq_translator.config import ModelConfig
from wtq_translator.enums import MentionSourceType
from wtq_translator.models import TableCoordinate, TableMention
from wtq_translator.prompts import question_translation_system_prompt
from wtq_translator.translation.shell import build_question_translation_payload, translate_shell


def _mention(mask: str, key: str, text: str, *, header: bool, column: int, row: int = 0) -> TableMention:
    coord = TableCoordinate(None, column) if header else TableCoordinate(row, column)
    return TableMention(
        mask=mask,
        question_text=text,
        normalized_text=text.casefold(),
        start=0,
        end=len(text),
        source_type=MentionSourceType.HEADER if header else MentionSourceType.CELL,
        coordinates=(coord,),
        table_source_text=text,
        match_type="lemma",
        candidate_key=key,
    )


class CaptureProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[min(len(self.calls), len(self.responses)) - 1]

    async def count_tokens(self, text: str) -> int:
        return len(text.split())


MASKED = (
    "what was <TABLE_REF_0000>Number of goals</TABLE_REF_0000> of "
    "<TABLE_REF_0001>New York</TABLE_REF_0001> in 2001?"
)
MENTIONS = [
    _mention("<TABLE_REF_0000>", "hdr_0000", "Number of goals", header=True, column=0),
    _mention("<TABLE_REF_0001>", "cell_0001", "New York", header=False, column=1),
]
CANONICAL = {"hdr_0000": "Количество голов", "cell_0001": "Нью-Йорк"}


def _valid_response(missing=None):
    used = [
        {"id": "hdr_0000", "surface_ru": "количество голов", "lemma_ru": "количество голов"},
        {"id": "cell_0001", "surface_ru": "Нью-Йорка", "lemma_ru": "Нью-Йорк"},
    ]
    if missing:
        used = [u for u in used if u["id"] not in missing]
    return {
        "question_ru": "Каким было количество голов у Нью-Йорка в 2001 году?",
        "question_ru_tagged": (
            "Каким было <TABLE_REF_0000>количество голов</TABLE_REF_0000> "
            "у <TABLE_REF_0001>Нью-Йорка</TABLE_REF_0001> в 2001 году?"
        ),
        "mentions_used": used,
        "missing_mentions": missing or [],
    }


def test_build_payload_inflect_allowed():
    payload = build_question_translation_payload(
        MASKED, MENTIONS, source_language="en", target_language="ru", id_to_canonical=CANONICAL
    )
    assert payload["task"] == "translate_question"
    assert payload["tagged_question"] == MASKED
    by_id = {item["id"]: item for item in payload["mentions"]}
    assert by_id["hdr_0000"]["inflect_allowed"] is False
    assert by_id["cell_0001"]["inflect_allowed"] is True
    assert by_id["hdr_0000"]["translation"] == "Количество голов"
    assert by_id["cell_0001"]["translation"] == "Нью-Йорк"
    assert by_id["hdr_0000"]["tag"] == "<TABLE_REF_0000>"
    assert "instruction" in payload


@pytest.mark.asyncio
async def test_structured_uses_filled_system_and_returns_tagged():
    response = json.dumps(_valid_response(), ensure_ascii=False)
    provider = CaptureProvider([response])

    result = await translate_shell(
        provider,
        MASKED,
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=0),
        seed=42,
        mentions=MENTIONS,
        id_to_canonical=CANONICAL,
        domain="football",
    )

    assert result == (
        "Каким было <TABLE_REF_0000>количество голов</TABLE_REF_0000> "
        "у <TABLE_REF_0001>Нью-Йорка</TABLE_REF_0001> в 2001 году?"
    )
    call = provider.calls[0]
    assert call["system"] == question_translation_system_prompt("en", "ru", "football")
    payload = json.loads(call["user"])
    assert payload["mentions"][0]["inflect_allowed"] is False


@pytest.mark.asyncio
async def test_structured_missing_mentions_reported_and_returned():
    missing = ["hdr_0000"]
    provider = CaptureProvider([json.dumps(_valid_response(missing=missing), ensure_ascii=False)])
    reported = []

    result = await translate_shell(
        provider,
        MASKED,
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=0),
        seed=42,
        mentions=MENTIONS,
        id_to_canonical=CANONICAL,
        on_missing=lambda ids: reported.extend(ids),
    )

    assert reported == missing
    assert "<TABLE_REF_0000>количество голов</TABLE_REF_0000>" in result


@pytest.mark.asyncio
async def test_structured_dropped_mention_raises_after_retries():
    # hdr_0000 отсутствует и в mentions_used, и в missing_mentions.
    response = {
        "question_ru": "Каким было количество голов у Нью-Йорка в 2001 году?",
        "question_ru_tagged": (
            "Каким было <TABLE_REF_0000>количество голов</TABLE_REF_0000> "
            "у <TABLE_REF_0001>Нью-Йорка</TABLE_REF_0001> в 2001 году?"
        ),
        "mentions_used": [
            {"id": "cell_0001", "surface_ru": "Нью-Йорка", "lemma_ru": "Нью-Йорк"},
        ],
        "missing_mentions": [],
    }
    provider = CaptureProvider([json.dumps(response, ensure_ascii=False)])

    with pytest.raises(RuntimeError, match="Shell translation failed after retries"):
        await translate_shell(
            provider,
            MASKED,
            source_language="en",
            target_language="ru",
            model=ModelConfig(retries=2),
            seed=42,
            mentions=MENTIONS,
            id_to_canonical=CANONICAL,
        )
    assert len(provider.calls) == 3  # retries=2 -> 3 попытки
    retry_payload = json.loads(provider.calls[1]["user"])
    assert "mentions_used" in retry_payload["retry_instruction"]


@pytest.mark.asyncio
async def test_structured_surface_mismatch_raises():
    response = _valid_response()
    response["mentions_used"][0]["surface_ru"] = "голов количество"
    provider = CaptureProvider([json.dumps(response, ensure_ascii=False)])

    with pytest.raises(RuntimeError, match="Shell translation failed after retries"):
        await translate_shell(
            provider,
            MASKED,
            source_language="en",
            target_language="ru",
            model=ModelConfig(retries=0),
            seed=42,
            mentions=MENTIONS,
            id_to_canonical=CANONICAL,
        )


@pytest.mark.asyncio
async def test_structured_plain_tagged_mismatch_is_allowed():
    # Проверки «plain == strip_tags(tagged)» больше нет: литературный plain-перевод
    # и тегированный перевод независимы, итоговый вопрос строится из tagged.
    response = _valid_response()
    response["question_ru"] = "Каким было число голов у Нью-Йорка в 2001 году?"
    provider = CaptureProvider([json.dumps(response, ensure_ascii=False)])

    result = await translate_shell(
        provider,
        MASKED,
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=0),
        seed=42,
        mentions=MENTIONS,
        id_to_canonical=CANONICAL,
    )

    assert result == (
        "Каким было <TABLE_REF_0000>количество голов</TABLE_REF_0000> "
        "у <TABLE_REF_0001>Нью-Йорка</TABLE_REF_0001> в 2001 году?"
    )
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_structured_surface_mismatch_allowed_for_inflectable_mention():
    # cell_0001 — inflect_allowed=True: форма в теге может свободно меняться,
    # поверхность носит справочный характер и не проверяется.
    response = _valid_response()
    response["mentions_used"][1]["surface_ru"] = "Нью-Йорк"  # в теге «Нью-Йорка»
    provider = CaptureProvider([json.dumps(response, ensure_ascii=False)])

    result = await translate_shell(
        provider,
        MASKED,
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=0),
        seed=42,
        mentions=MENTIONS,
        id_to_canonical=CANONICAL,
    )

    assert "<TABLE_REF_0001>Нью-Йорка</TABLE_REF_0001>" in result


@pytest.mark.asyncio
async def test_structured_retry_carries_previous_response_and_all_errors():
    # Нарушаем сразу две проверки: Tags changed (убрали пару cell-упоминания)
    # и Surface mismatch (у header-упоминания hdr_0000, inflect_allowed=False).
    bad = _valid_response()
    bad["mentions_used"][0]["surface_ru"] = "голов количество"
    bad["question_ru_tagged"] = (
        "Каким было <TABLE_REF_0000>количество голов</TABLE_REF_0000> "
        "у Нью-Йорка в 2001 году?"
    )
    bad_raw = json.dumps(bad, ensure_ascii=False)
    provider = CaptureProvider([bad_raw, bad_raw])  # вторая попытка тоже неудачная

    with pytest.raises(RuntimeError, match="Shell translation failed after retries"):
        await translate_shell(
            provider,
            MASKED,
            source_language="en",
            target_language="ru",
            model=ModelConfig(retries=1),
            seed=42,
            mentions=MENTIONS,
            id_to_canonical=CANONICAL,
        )

    assert len(provider.calls) == 2
    retry_payload = json.loads(provider.calls[1]["user"])
    # Ретрай должен исправлять прошлый ответ, а не генерить с нуля.
    assert retry_payload["previous_response"] == bad_raw
    assert "from scratch" in retry_payload["retry_instruction"]
    # Обе ошибки собраны и переданы в ретрай.
    joined = "\n".join(retry_payload["errors"])
    assert "Tags changed" in joined
    assert "Surface mismatch" in joined


@pytest.mark.asyncio
async def test_structured_retry_fixes_previous_response():
    # На ретрае модель «чинит» прошлый ответ: возвращается корректный tagged.
    bad = _valid_response()
    bad["mentions_used"][0]["surface_ru"] = "голов количество"  # hdr_0000
    bad_raw = json.dumps(bad, ensure_ascii=False)
    good_raw = json.dumps(_valid_response(), ensure_ascii=False)
    provider = CaptureProvider([bad_raw, good_raw])

    result = await translate_shell(
        provider,
        MASKED,
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=1),
        seed=42,
        mentions=MENTIONS,
        id_to_canonical=CANONICAL,
    )

    assert result == (
        "Каким было <TABLE_REF_0000>количество голов</TABLE_REF_0000> "
        "у <TABLE_REF_0001>Нью-Йорка</TABLE_REF_0001> в 2001 году?"
    )
    retry_payload = json.loads(provider.calls[1]["user"])
    assert retry_payload["previous_response"] == bad_raw


def test_system_prompt_fill():
    prompt = question_translation_system_prompt("en", "ru", "football")
    assert "Source language: en" in prompt
    assert "Target language: ru" in prompt
    assert "Domain: football" in prompt
    assert '"question_ru_tagged"' in prompt
    assert '"surface_ru"' in prompt
    assert "inflect_allowed" in prompt
    assert "not specified" in question_translation_system_prompt("ru", "en", "")


def test_system_prompt_ru_contains_declension_worked_example():
    # A: worked-example склонения добавляется только для русского как целевого.
    ru_prompt = question_translation_system_prompt("en", "ru", "football")
    assert "кем был <TABLE_REF_0000>соперником</TABLE_REF_0000>" in ru_prompt
    assert "INSTRUMENTAL" in ru_prompt
    en_prompt = question_translation_system_prompt("ru", "en", "football")
    assert "соперником" not in en_prompt


@pytest.mark.asyncio
async def test_empty_mentions_hallucinated_tag_is_sanitized():
    # A2: mentions=[] и модель придумала тег — тег вычищается, фейла нет.
    response = {
        "question_ru": "в каком городе Пётр добился своего последнего первого места?",
        "question_ru_tagged": (
            "в каком городе <TABLE_REF_0000>Пётр</TABLE_REF_0000> "
            "добился своего последнего первого места?"
        ),
        "mentions_used": [
            {"id": "TABLE_REF_0000", "surface_ru": "Пётр", "lemma_ru": "Пётр"},
        ],
        "missing_mentions": [],
    }
    provider = CaptureProvider([json.dumps(response, ensure_ascii=False)])

    result = await translate_shell(
        provider,
        "in what city did piotr's last 1st place finish occur?",
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=2),
        seed=42,
        mentions=[],
        id_to_canonical={},
    )

    assert result == "в каком городе Пётр добился своего последнего первого места?"
    assert len(provider.calls) == 1  # санитизация предотвращает ретраи


@pytest.mark.asyncio
async def test_empty_mentions_plain_response_accepted():
    response = {
        "question_ru": "какое было первое место для азиатских игр?",
        "question_ru_tagged": "какое было первое место для азиатских игр?",
        "mentions_used": [],
        "missing_mentions": [],
    }
    provider = CaptureProvider([json.dumps(response, ensure_ascii=False)])

    result = await translate_shell(
        provider,
        "what was the first venue for the asian games?",
        source_language="en",
        target_language="ru",
        model=ModelConfig(retries=0),
        seed=42,
        mentions=[],
        id_to_canonical={},
    )

    assert result == "какое было первое место для азиатских игр?"


@pytest.mark.asyncio
async def test_fix_retry_instruction_is_language_aware():
    # A3: retry-инструкция параметризована целевым языком (не зашит question_ru_tagged)
    # и запрещает сохранять теги с неизвестными id.
    mention = _mention("<TABLE_REF_0000>", "cell_0001", "Пётр", header=False, column=0)
    bad = {
        "question_en": "in which city did Peter achieve a first place?",
        "question_en_tagged": (
            "in which city did <TABLE_REF_0000>Peter</TABLE_REF_0000> achieve "
            "<TABLE_REF_0001>a first place</TABLE_REF_0001>?"
        ),
        "mentions_used": [
            {"id": "cell_0001", "surface_en": "Peter", "lemma_en": "Peter"},
            {"id": "TABLE_REF_0001", "surface_en": "a first place", "lemma_en": "first place"},
        ],
        "missing_mentions": [],
    }
    bad_raw = json.dumps(bad, ensure_ascii=False)
    provider = CaptureProvider([bad_raw, bad_raw])

    with pytest.raises(RuntimeError, match="Shell translation failed after retries"):
        await translate_shell(
            provider,
            "в каком городе <TABLE_REF_0000>Пётр</TABLE_REF_0000> занял первое место?",
            source_language="ru",
            target_language="en",
            model=ModelConfig(retries=1),
            seed=42,
            mentions=[mention],
            id_to_canonical={"cell_0001": "Peter"},
        )

    retry_payload = json.loads(provider.calls[1]["user"])
    instruction = retry_payload["retry_instruction"]
    assert "question_en_tagged" in instruction
    assert "question_ru_tagged" not in instruction
    assert "NOT in that list" in instruction
