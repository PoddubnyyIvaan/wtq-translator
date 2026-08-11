from wtq_translator.models import TableData
from wtq_translator.validation.invariants import validate_untranslated_columns


def test_untranslated_columns():
    src = TableData(["A", "B"], [["x", "y"]])
    out = TableData(["AA", "B"], [["xx", "y"]])
    assert validate_untranslated_columns(src, out, {0}) == []
