from wtq_translator.config import MentionsConfig
from wtq_translator.models import TableData
from wtq_translator.mentions.finder import find_table_mentions
from wtq_translator.mentions.masking import mask_question, restore_original
from wtq_translator.models import MentionMap


def test_longest_match_first():
    table = TableData(["Place"], [["York"], ["New York"], ["New York City"]])
    result = find_table_mentions("Was New York City first?", table, MentionsConfig())
    assert [m.question_text for m in result.mentions] == ["New York City"]


def test_mask_roundtrip():
    table = TableData(["Player"], [["Ardo Kreek"]])
    q = "Who played with Ardo Kreek?"
    result = find_table_mentions(q, table, MentionsConfig())
    masked = mask_question(q, result.mentions)
    assert "<TABLE_REF_0000>" in masked
    assert restore_original(masked, MentionMap(result.mentions)) == q


def test_ambiguous_same_text_different_cells():
    table = TableData(["A", "B"], [["X", "X"]])
    result = find_table_mentions("Was X selected?", table, MentionsConfig())
    assert result.ambiguous
    assert not result.mentions


def test_repeated_value_same_column_is_one_resolvable_mention():
    table = TableData(["Team"], [["X"], ["X"]])
    result = find_table_mentions("Was X selected?", table, MentionsConfig())
    assert not result.ambiguous
    assert len(result.mentions) == 1
    mention = result.mentions[0]
    assert [(c.row, c.column) for c in mention.coordinates] == [(0, 0), (1, 0)]


def test_original_roundtrip_preserves_question_case_not_table_case():
    table = TableData(["City"], [["London"]])
    q = "where is london?"
    result = find_table_mentions(q, table, MentionsConfig())
    masked = mask_question(q, result.mentions)
    assert restore_original(masked, MentionMap(result.mentions)) == q


def test_ambiguous_longest_span_blocks_nested_shorter_match():
    table = TableData(
        ["A", "B", "C"],
        [["New York City", "New York City", "York"]],
    )
    result = find_table_mentions("Was New York City selected?", table, MentionsConfig())
    assert [a.question_text for a in result.ambiguous] == ["New York City"]
    assert result.mentions == []
