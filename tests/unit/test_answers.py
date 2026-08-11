import pytest

from wtq_translator.config import AppConfig
from wtq_translator.models import TableData
from wtq_translator.translation.answers import build_target_value


class NeverCallProvider:
    @property
    def capabilities(self):
        raise AssertionError("not needed")

    async def generate(self, **kwargs):
        raise AssertionError("LLM should not be called")


@pytest.mark.asyncio
async def test_table_language_answer_uses_output_coordinate_in_mixed_table(tmp_path):
    cfg = AppConfig.model_validate({
        "project": {"input_root": str(tmp_path), "output_root": str(tmp_path / "out"), "source_language": "en", "target_language": "ru"},
        "back_translation": {"enabled": True},
    })
    source = TableData(["Name", "City"], [["Alice", "London"]])
    mixed = TableData(["Имя", "City"], [["Алиса", "London"]])
    value, warnings = await build_target_value(
        NeverCallProvider(),
        source_target_value="Alice",
        source_table=source,
        output_table=mixed,
        config=cfg,
        desired_language="en",
        source_language="en",
        table_language_is_desired=True,
    )
    assert value == "Алиса"
    assert warnings == []
