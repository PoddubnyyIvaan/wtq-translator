import json

import pytest

from wtq_translator.backtranslation.table import back_translate_table_candidates
from wtq_translator.config import AppConfig
from wtq_translator.models import TableData


class _StubMorph:
    def get(self, key):
        return []


class StubToken:
    def __init__(self, text):
        self.text = text
        self.idx = 0
        self.lemma_ = text.casefold()
        self.pos_ = "PROPN"
        self.like_num = False
        self.ent_type_ = ""
        self.is_punct = False
        self.is_space = False
        self.morph = _StubMorph()


class StubNlp:
    def __call__(self, text):
        return [StubToken(text)]


class MockProvider:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    @property
    def capabilities(self):
        class Caps:
            seed = True
        return Caps()

    async def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def generate(self, **kwargs):
        payload = json.loads(kwargs["user"])
        self.calls.append(payload)
        assert payload["task"] == "translate_candidates"
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "translation": self.mapping[item["text"]]}
                    for item in payload["items"]
                ]
            },
            ensure_ascii=False,
        )


def _cfg(tmp_path):
    return AppConfig.model_validate({
        "project": {
            "input_root": str(tmp_path),
            "output_root": str(tmp_path / "out"),
            "source_language": "en",
            "target_language": "ru",
        },
        "model": {"retries": 0},
        "cache": {"enabled": False},
    })


@pytest.mark.asyncio
async def test_back_translate_table_candidates_reconstructs(tmp_path):
    provider = MockProvider({"Name": "Имя", "Alice": "Алиса"})
    table = TableData(["Name"], [["Alice"]])

    out = await back_translate_table_candidates(
        provider,
        table,
        config=_cfg(tmp_path),
        source_language="en",
        translated_language="ru",
        nlp=StubNlp(),
    )

    assert isinstance(out, TableData)
    assert out.headers == ["Имя"]
    assert out.rows == [["Алиса"]]


@pytest.mark.asyncio
async def test_back_translate_table_candidates_swaps_languages(tmp_path):
    # Другое содержимое таблицы, чтобы не попасть в in-memory кэш кандидатов.
    provider = MockProvider({"City": "Город", "Berlin": "Берлин"})
    table = TableData(["City"], [["Berlin"]])

    await back_translate_table_candidates(
        provider,
        table,
        config=_cfg(tmp_path),
        source_language="en",
        translated_language="ru",
        nlp=StubNlp(),
    )

    assert len(provider.calls) == 1
    payload = provider.calls[0]
    assert payload["task"] == "translate_candidates"
    assert payload["source_language"] == "ru"  # translated_language
    assert payload["target_language"] == "en"  # source_language
