"""Тесты детекции алфавитно-зависимых вопросов (detect_alphabet_intent).

Стиль: pytest + pytest-asyncio; вместо реального LLM-провайдера используется
фейковый провайдер с async generate(...) и async count_tokens(...),
как принято в tests/unit (см. test_domain_detection.py).
"""
import json

import pytest

from wtq_translator.alphabet import (
    ALPHABET_INTENT_LABELS,
    AlphabetIntentError,
    detect_alphabet_intent,
    detect_alphabet_intent_rules,
)
from wtq_translator.config import (
    AlphabetDetectionConfig,
    AppConfig,
    ModelConfig,
    ProjectConfig,
)
from wtq_translator.models import QuestionRecord, TableData
from wtq_translator.pipeline import TranslationPipeline


class FakeProvider:
    """Минимальный фейковый LLM-провайдер: async generate/count_tokens.

    ``responses`` — очередь ответов generate (pop по мере вызова); финальный ответ
    по умолчанию — отсутствие намерения.
    """

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []  # аргументы каждого вызова generate

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"intent_id": None, "why": None}, ensure_ascii=False)

    async def count_tokens(self, text: str) -> int:
        return len(text.split())


def _config(tmp_path, *, enabled=True, retries=2, method="llm"):
    return AppConfig(
        project=ProjectConfig(
            input_root=tmp_path,
            output_root=tmp_path,
            source_language="en",
            seed=7,
        ),
        model=ModelConfig(retries=retries, temperature=0.0, max_output_tokens=256),
        alphabet_detection=AlphabetDetectionConfig(enabled=enabled, method=method),
    )


@pytest.mark.asyncio
async def test_intent_detected_and_payload_task(tmp_path):
    """Провайдер возвращает намерение → AlphabetDetection(2, why); payload корректен."""
    provider = FakeProvider(responses=[json.dumps({"intent_id": 2, "why": "топ по алфавиту"}, ensure_ascii=False)])
    config = _config(tmp_path, retries=2)

    detection = await detect_alphabet_intent(
        provider, config, question_id="q-1", question_text="What is the first city alphabetically?"
    )

    assert detection.intent_id == 2
    assert detection.why == "топ по алфавиту"
    assert detection.error is None
    assert len(provider.calls) == 1
    payload = json.loads(provider.calls[0]["user"])
    assert payload["task"] == "detect_alphabet_intent"
    assert payload["question_id"] == "q-1"
    assert payload["question"] == "What is the first city alphabetically?"
    # Системный промпт содержит таблицу намерений.
    assert "1 | алфавитная сортировка" in provider.calls[0]["system"]


@pytest.mark.asyncio
async def test_null_intent_returns_none_and_calls_once(tmp_path):
    """intent_id: null → вопрос допускается, провайдер вызван ровно 1 раз."""
    provider = FakeProvider(responses=[json.dumps({"intent_id": None, "why": None}, ensure_ascii=False)])
    config = _config(tmp_path)

    detection = await detect_alphabet_intent(provider, config, question_id="q-1", question_text="How many cities?")

    assert detection.intent_id is None
    assert detection.why is None
    assert detection.error is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        json.dumps({"intent_id": 5, "why": "вне диапазона"}, ensure_ascii=False),
        json.dumps({"intent_id": "2", "why": "строка вместо числа"}, ensure_ascii=False),
        json.dumps({"intent_id": True, "why": "bool не int"}, ensure_ascii=False),
        json.dumps({"intent_id": 1, "why": "  "}, ensure_ascii=False),
    ],
)
async def test_invalid_responses_retry_then_error(tmp_path, response):
    """Невалидный intent_id или пустой why → ретраи, затем error заполнен (fail-closed)."""
    provider = FakeProvider(responses=[response] * 3)
    config = _config(tmp_path, retries=2)  # attempts = retries + 1 = 3

    detection = await detect_alphabet_intent(provider, config, question_id="q-1", question_text="Which is earlier?")

    assert detection.intent_id is None
    assert detection.why is None
    assert detection.error is not None
    assert detection.error.startswith("Alphabet detection failed after retries:")
    assert len(provider.calls) == 3
    # На ретраях payload дополнен инструкцией.
    second_payload = json.loads(provider.calls[1]["user"])
    assert "retry_instruction" in second_payload
    assert "previous_error" in second_payload


@pytest.mark.asyncio
async def test_invalid_json_after_all_attempts_sets_error_without_raising(tmp_path):
    """Постоянный не-JSON ответ → error заполнен, исключение не бросается."""
    provider = FakeProvider(responses=["not json at all", '"just a string"', ""])
    config = _config(tmp_path, retries=2)

    detection = await detect_alphabet_intent(provider, config, question_id="q-1", question_text="What is the order?")

    assert detection.intent_id is None
    assert detection.error is not None
    assert detection.error.startswith("Alphabet detection failed after retries:")
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_disabled_config_skips_provider(tmp_path):
    """alphabet_detection.enabled False → (None, None, None), провайдер НЕ вызывается."""
    provider = FakeProvider()
    config = _config(tmp_path, enabled=False)

    detection = await detect_alphabet_intent(provider, config, question_id="q-1", question_text="Sort by name")

    assert detection.intent_id is None
    assert detection.why is None
    assert detection.error is None
    assert provider.calls == []


@pytest.mark.parametrize(
    "question, intent_id",
    [
        ("Sort by alphabet", 1),
        ("alphabetical order", 1),
        ("list alphabetically", 1),
        ("alphabetized rows", 1),
        ("alphabetic order", 1),
        ("lexicographic order", 1),
        ("lexicographically smallest", 1),
        ("what does the letter x mean", 4),
        ("letters", 4),
        ("first letter 'a'", 4),
        ("character at position 2", 4),
        ("count characters", 4),
        ("ALPHABETICALLY", 1),
    ],
)
def test_regex_rules_map_forms_to_intent(question, intent_id):
    """Словоформы корней alphabet/lexicograph/letter/character → intent 1/4, why в regex-формате."""
    detection = detect_alphabet_intent_rules(question)

    assert detection.intent_id == intent_id
    assert detection.error is None
    assert detection.why.startswith("regex: keyword(s) '")
    assert detection.why.endswith("'")


@pytest.mark.parametrize(
    "question",
    [
        "characteristic property",
        "characterize the data",
        "newsletter",
        "letterhead",
        "lettering",
        "practice",
        "What is the most populous city?",
    ],
)
def test_regex_rules_no_keyword_returns_none(question):
    """Ложные друзья и вопросы без ключевых слов → (None, None), error None."""
    detection = detect_alphabet_intent_rules(question)

    assert detection.intent_id is None
    assert detection.why is None
    assert detection.error is None


def test_regex_rules_alphabet_priority_and_why_both_keywords():
    """Оба корня совпали → класс 1 (приоритет alphabet/lexicograph), в why оба слова."""
    detection = detect_alphabet_intent_rules("list the letters alphabetically")

    assert detection.intent_id == 1
    assert detection.error is None
    assert "alphabetically" in detection.why
    assert "letters" in detection.why


@pytest.mark.asyncio
async def test_regex_method_is_deterministic_and_skips_provider(tmp_path):
    """method='regex' → детерминированный результат, провайдер не вызывается."""
    provider = FakeProvider()
    config = _config(tmp_path, method="regex")

    detection = await detect_alphabet_intent(
        provider, config, question_id="q-1", question_text="Sort by alphabet"
    )

    assert detection.intent_id == 1
    assert detection.error is None
    assert provider.calls == []


@pytest.mark.asyncio
async def test_regex_method_null_for_non_alphabet_question(tmp_path):
    """method='regex', вопрос без ключевых слов → (None, None), провайдер не вызывается."""
    provider = FakeProvider()
    config = _config(tmp_path, method="regex")

    detection = await detect_alphabet_intent(
        provider, config, question_id="q-3", question_text="How many cities?"
    )

    assert detection.intent_id is None
    assert detection.why is None
    assert detection.error is None
    assert provider.calls == []


@pytest.mark.asyncio
async def test_regex_method_disabled_returns_none_without_calls(tmp_path):
    """enabled=False + method='regex' → (None, None), провайдер не вызывается."""
    provider = FakeProvider()
    config = _config(tmp_path, enabled=False, method="regex")

    detection = await detect_alphabet_intent(
        provider, config, question_id="q-1", question_text="Sort by alphabet"
    )

    assert detection.intent_id is None
    assert detection.why is None
    assert detection.error is None
    assert provider.calls == []


def _stub_nlp():
    class _Morph:
        def __init__(self, poss=None):
            self._poss = poss

        def get(self, key):
            if key == "Poss" and self._poss is not None:
                return [self._poss]
            return []

    class _Tok:
        def __init__(self, text, idx, *, lemma=None, pos="", num=False, date=False, punct=False, poss=None):
            self.text = text
            self.idx = idx
            self.lemma_ = text.lower() if lemma is None else lemma
            self.pos_ = pos
            self.like_num = num
            self.ent_type_ = "DATE" if date else ""
            self.is_punct = punct
            self.is_space = False
            self.morph = _Morph(poss)

    def make_nlp(mapping):
        def nlp(text):
            return list(mapping[text])
        return nlp

    return make_nlp({
        "Where is Alice?": [
            _Tok("Where", 0, lemma="where", pos="ADV"),
            _Tok("is", 6, lemma="be", pos="AUX"),
            _Tok("Alice", 9, lemma="alice", pos="PROPN"),
        ],
        "Name": [_Tok("Name", 0, lemma="name", pos="NOUN")],
        "City": [_Tok("City", 0, lemma="city", pos="NOUN")],
        "Alice": [_Tok("Alice", 0, lemma="alice", pos="PROPN")],
        "London": [_Tok("London", 0, lemma="london", pos="PROPN")],
    })


def _pipeline_config(tmp_path):
    return AppConfig.model_validate({
        "project": {
            "input_root": str(tmp_path),
            "output_root": str(tmp_path / "out"),
            "source_language": "en",
            "target_language": "ru",
            "seed": 42,
        },
        "translation": {"mode": "full", "answer_language_mode": "table"},
        "model": {"retries": 0},
        "embeddings": {"enabled": False},
        "back_translation": {"enabled": True},
        "table": {"rows_per_chunk": 20},
        "cache": {"enabled": False},
        "alphabet_detection": {"enabled": True},
    })


@pytest.mark.asyncio
async def test_pipeline_raises_alphabet_intent_error_before_table_translation(tmp_path, monkeypatch):
    """Full-пайплайн: сначала домен, затем детект намерения 1 → AlphabetIntentError.

    Табличные переводы (translate_candidates) при этом не вызывались.
    """
    stub_nlp = _stub_nlp()
    monkeypatch.setattr("wtq_translator.pipeline.load_spacy_model", lambda name: stub_nlp)

    cfg = _pipeline_config(tmp_path)
    provider = FakeProvider(responses=[
        json.dumps({"domain": "geography"}, ensure_ascii=False),
        json.dumps({"intent_id": 1, "why": "вопрос требует алфавитного порядка"}, ensure_ascii=False),
    ])
    pipeline = TranslationPipeline(cfg, provider)

    q = QuestionRecord("q1", "Where is Alice?", "table.tsv", "London")
    table = TableData(["Name", "City"], [["Alice", "London"]])

    with pytest.raises(AlphabetIntentError) as excinfo:
        await pipeline.process(q, table)

    exc = excinfo.value
    assert exc.question_id == "q1"
    assert exc.intent_label == ALPHABET_INTENT_LABELS[1]
    assert exc.why == "вопрос требует алфавитного порядка"
    assert ALPHABET_INTENT_LABELS[1] == "алфавитная сортировка"
    # Ровно два вызова: detect_domain + detect_alphabet_intent; никаких переводов.
    assert len(provider.calls) == 2
    tasks = [json.loads(c["user"]).get("task") for c in provider.calls]
    assert tasks == ["detect_domain", "detect_alphabet_intent"]
    assert not any("translate" in t for t in tasks)