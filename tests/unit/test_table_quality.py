from wtq_translator.config import MentionNormalizationConfig
from wtq_translator.models import TableData
from wtq_translator.validation.invariants import validate_nonempty_preservation
from wtq_translator.validation.table_quality import compare_back_translated_table


def test_nonempty_values_cannot_disappear():
    src = TableData(["Name"], [["Alice"], [""]])
    out = TableData([""], [[""], [""]])
    errors = validate_nonempty_preservation(src, out)
    assert (None, 0, "Name") in errors
    assert (0, 0, "Alice") in errors
    assert len(errors) == 2


def test_back_table_normalized_exact_metrics_can_be_scoped_to_columns():
    src = TableData(["Name", "City"], [["Alice", "London"]])
    back = TableData(["name", "Город"], [["alice", "Лондон"]])
    metrics = compare_back_translated_table(
        src,
        back,
        normalization=MentionNormalizationConfig(),
        columns={0},
    )
    assert metrics["compared"] == 2
    assert metrics["normalized_exact_matches"] == 2
    assert metrics["normalized_exact_ratio"] == 1.0
    assert metrics["mismatch_examples"] == []
