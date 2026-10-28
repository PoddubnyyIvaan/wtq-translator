"""Интеграционный тест нового full-режима (candidate-based пайплайн с теговой маской)."""
import json
import re

import pytest

from wtq_translator.config import AppConfig
from wtq_translator.enums import ProcessingStatus
from wtq_translator.models import QuestionRecord, TableData
from wtq_translator.pipeline import TranslationPipeline


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


def _make_nlp(mapping):
    """Stub-nlp: `__call__(text)` возвращает список токенов для заданного текста."""
    def nlp(text):
        return list(mapping[text])
    return nlp


def _stub_nlp():
    return _make_nlp({
        "Where is Alice?": [
            _Tok("Where", 0, lemma="where", pos="ADV"),
            _Tok("is", 6, lemma="be", pos="AUX"),
            _Tok("Alice", 9, lemma="alice", pos="PROPN"),
        ],
        "Name": [_Tok("Name", 0, lemma="name", pos="NOUN")],
        "City": [_Tok("City", 0, lemma="city", pos="NOUN")],
        "Alice": [_Tok("Alice", 0, lemma="alice", pos="PROPN")],
        "London": [_Tok("London", 0, lemma="london", pos="PROPN")],
        # Кириллица для обратного перевода таблицы.
        "Имя": [_Tok("Имя", 0, lemma="имя", pos="NOUN")],
        "Город": [_Tok("Город", 0, lemma="город", pos="NOUN")],
        "Алиса": [_Tok("Алиса", 0, lemma="алиса", pos="PROPN")],
        "Лондон": [_Tok("Лондон", 0, lemma="лондон", pos="PROPN")],
    })


class RuleBasedProvider:
    """Детерминированный провайдер: кандидаты по словарю, shell по scripted-маппингу."""

    def __init__(self):
        self.payloads = []
        self.systems = []

    @property
    def capabilities(self):
        class Caps:
            seed = True
        return Caps()

    async def count_tokens(self, text: str) -> int:
        # Детерминированный подсчёт токенов (по словам).
        return len(text.split())

    async def generate(self, *, system, user, temperature, max_output_tokens, seed=None):
        payload = json.loads(user)
        self.payloads.append(payload)
        self.systems.append(system)
        src = payload.get("source_language")
        dst = payload.get("target_language")
        task = payload.get("task")

        en_ru = {"Name": "Имя", "City": "Город", "Alice": "Алиса", "London": "Лондон"}
        ru_en = {v: k for k, v in en_ru.items()}

        if task == "detect_domain":
            return json.dumps({"domain": "geography"}, ensure_ascii=False)

        if task == "detect_alphabet_intent":
            # Новый обязательный этап full-режима: намерения нет — вопрос переводится.
            return json.dumps({"intent_id": None, "why": None}, ensure_ascii=False)

        if task == "translate_candidates":
            mapping = en_ru if (src, dst) == ("en", "ru") else ru_en
            return json.dumps(
                {
                    "translations": [
                        {"id": item["id"], "translation": mapping.get(item["text"], item["text"])}
                        for item in payload["items"]
                    ]
                },
                ensure_ascii=False,
            )

        if task == "translate_question":
            # Структурированный протокол: scripted-маппинг предложения + поверхности
            # из канонических переводов упоминаний; plain = tagged без тегов.
            tagged_text = payload["tagged_question"]
            tagged = {
                ("en", "ru", "Where is <TABLE_REF_0000>Alice</TABLE_REF_0000>?"):
                    "Где находится <TABLE_REF_0000>Алиса</TABLE_REF_0000>?",
                ("ru", "en", "Где находится <TABLE_REF_0000>Алиса</TABLE_REF_0000>?"):
                    "Where is <TABLE_REF_0000>Alice</TABLE_REF_0000>?",
            }.get((src, dst, tagged_text), tagged_text)
            plain = re.sub(r"</?TABLE_REF_\d{4}>", "", tagged)
            used = [
                {"id": item["id"], f"surface_{dst}": item["translation"], f"lemma_{dst}": item["translation"]}
                for item in payload["mentions"]
            ]
            return json.dumps(
                {
                    f"question_{dst}": plain,
                    f"question_{dst}_tagged": tagged,
                    "mentions_used": used,
                    "missing_mentions": [],
                },
                ensure_ascii=False,
            )

        if "text" in payload:
            text = payload["text"]
            shell_mapping = {
                ("en", "ru", "Where is <TABLE_REF_0000>Alice</TABLE_REF_0000>?"):
                    "Где находится <TABLE_REF_0000>Алиса</TABLE_REF_0000>?",
                ("ru", "en", "Где находится <TABLE_REF_0000>Алиса</TABLE_REF_0000>?"):
                    "Where is <TABLE_REF_0000>Alice</TABLE_REF_0000>?",
            }
            if (src, dst, text) in shell_mapping:
                return json.dumps({"translated_text": shell_mapping[(src, dst, text)]}, ensure_ascii=False)
            return json.dumps({"translated_text": text}, ensure_ascii=False)

        raise AssertionError(payload)


class IdenticalEmbeddingProvider:
    async def embed(self, texts):
        # Scripted-обратный перевод точен, поэтому этого достаточно для проверки связки.
        return [[1.0, 0.0] for _ in texts]


def _config(tmp_path, mode):
    return AppConfig.model_validate({
        "project": {
            "input_root": str(tmp_path),
            "output_root": str(tmp_path / "out"),
            "source_language": "en",
            "target_language": "ru",
            "seed": 42,
        },
        "translation": {"mode": mode, "answer_language_mode": "table"},
        "model": {"retries": 0},
        "embeddings": {"enabled": False},
        "back_translation": {"enabled": True},
        "table": {"rows_per_chunk": 20},
        "cache": {"enabled": False},
    })


def _input():
    q = QuestionRecord("q1", "Where is Alice?", "table.tsv", "London")
    table = TableData(["Name", "City"], [["Alice", "London"]])
    return q, table


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end(tmp_path, monkeypatch):
    stub_nlp = _stub_nlp()
    monkeypatch.setattr("wtq_translator.pipeline.load_spacy_model", lambda name: stub_nlp)

    cfg = _config(tmp_path, "full")
    q, table = _input()
    provider = RuleBasedProvider()
    result = await TranslationPipeline(cfg, provider, IdenticalEmbeddingProvider()).process(q, table)

    assert result.translated_question == "Где находится Алиса?"
    assert result.translated_target_value == "Лондон"
    assert result.output_table.headers == ["Имя", "Город"]
    assert result.output_table.rows == [["Алиса", "Лондон"]]
    assert result.quality.back_translated_question == "Where is Alice?"
    assert result.quality.question_cosine_similarity == 1.0
    assert result.quality.back_table_min_similarity == 1.0
    assert result.quality.back_table_mean_similarity == 1.0
    assert result.quality.back_table_groups_compared >= 1
    assert result.quality.status == ProcessingStatus.ACCEPTED
    assert result.quality.original_masks == ["<TABLE_REF_0000>"]

    paired = [
        (system, p)
        for system, p in zip(provider.systems, provider.payloads)
        if p.get("task") == "translate_candidates"
    ]
    candidate_payloads = [p for _, p in paired]
    candidate_systems = [system for system, _ in paired]
    assert len(candidate_payloads) >= 2  # прямой + обратный перевод кандидатов таблицы
    # Домен указывается один раз в системном промпте кандидатных запросов...
    assert any("geography" in system for system in candidate_systems)
    # ...и больше не передаётся в user-JSON.
    assert all("domain" not in p for p in candidate_payloads)
    assert result.quality.domain == "geography"
