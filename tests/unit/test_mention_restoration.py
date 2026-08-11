import pytest

from wtq_translator.config import MentionsConfig
from wtq_translator.models import MentionMap, TableData
from wtq_translator.mentions.finder import find_table_mentions
from wtq_translator.mentions.masking import mask_question, restore_from_table


def test_repeated_same_column_translation_must_be_consistent():
    source = TableData(["Team"], [["X"], ["X"]])
    search = find_table_mentions("Was X selected?", source, MentionsConfig())
    masked = mask_question("Was X selected?", search.mentions)
    mention_map = MentionMap(search.mentions)

    translated = TableData(["Команда"], [["Икс"], ["Икс"]])
    assert restore_from_table(masked, mention_map, translated) == "Was Икс selected?"

    inconsistent = TableData(["Команда"], [["Икс"], ["X"]])
    with pytest.raises(ValueError, match="inconsistent values"):
        restore_from_table(masked, mention_map, inconsistent)
