import json

import pytest

from wtq_translator.config import AppConfig
from wtq_translator.models import TableData
from wtq_translator.translation.table import translate_table


class CountingProvider:
    def __init__(self):
        self.tasks = []

    @property
    def capabilities(self):
        class Caps:
            seed = True
        return Caps()

    async def generate(self, **kwargs):
        payload = json.loads(kwargs["user"])
        self.tasks.append(payload)
        assert payload["task"] == "translate_table_values"
        mapping = {
            "Value": "Значение",
            "same": "одинаково",
        }
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "translation": mapping[item["text"]]}
                    for item in payload["values"]
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
async def test_unique_values_are_one_llm_request_and_duplicates_are_propagated(tmp_path):
    provider = CountingProvider()
    table = TableData(["Value"], [["same"], ["same"], ["same"], ["2001"], [""]])
    out = await translate_table(
        provider,
        table,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
    )

    assert out.headers == ["Значение"]
    assert out.rows == [["одинаково"], ["одинаково"], ["одинаково"], ["2001"], [""]]
    assert len(provider.tasks) == 1

    payload = provider.tasks[0]
    assert payload["values"] == [
        {"id": "u0000", "text": "Value"},
        {"id": "u0001", "text": "same"},
    ]
    # The LLM is not given table structure, coordinates, or values that do not need translation.
    assert "headers" not in payload
    assert "rows" not in payload
    assert "columns_to_translate" not in payload
    assert "protected_cells" not in payload
    assert "2001" not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_selected_columns_only_are_sent_to_model(tmp_path):
    class PartialProvider:
        def __init__(self):
            self.tasks = []

        @property
        def capabilities(self):
            class Caps:
                seed = True
            return Caps()

        async def generate(self, **kwargs):
            payload = json.loads(kwargs["user"])
            self.tasks.append(payload)
            mapping = {"Name": "Имя", "Alice": "Алиса"}
            return json.dumps(
                {
                    "translations": [
                        {"id": item["id"], "translation": mapping[item["text"]]}
                        for item in payload["values"]
                    ]
                },
                ensure_ascii=False,
            )

    provider = PartialProvider()
    table = TableData(["Name", "City"], [["Alice", "London"]])
    out = await translate_table(
        provider,
        table,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
        columns={0},
    )

    assert out.headers == ["Имя", "City"]
    assert out.rows == [["Алиса", "London"]]
    assert len(provider.tasks) == 1
    assert [v["text"] for v in provider.tasks[0]["values"]] == ["Name", "Alice"]
    assert "City" not in json.dumps(provider.tasks[0], ensure_ascii=False)
    assert "London" not in json.dumps(provider.tasks[0], ensure_ascii=False)


@pytest.mark.asyncio
async def test_no_translatable_values_means_no_llm_request(tmp_path):
    provider = CountingProvider()
    table = TableData(["2020"], [["1"], ["2"], [""]])
    out = await translate_table(
        provider,
        table,
        config=_cfg(tmp_path),
        source_language="en",
        target_language="ru",
    )

    assert out.headers == table.headers
    assert out.rows == table.rows
    assert provider.tasks == []


@pytest.mark.asyncio
async def test_missing_or_extra_ids_are_rejected(tmp_path):
    class BadProvider:
        @property
        def capabilities(self):
            class Caps:
                seed = True
            return Caps()

        async def generate(self, **kwargs):
            return json.dumps({"translations": [{"id": "wrong", "translation": "x"}]})

    table = TableData(["Value"], [["same"]])
    with pytest.raises(RuntimeError, match="Translation ID set mismatch"):
        await translate_table(
            BadProvider(),
            table,
            config=_cfg(tmp_path),
            source_language="en",
            target_language="ru",
        )
