from wtq_translator.mentions.masking import extract_tagged_spans


def test_extract_tagged_spans_returns_mask_to_content():
    text = (
        "Где находится <TABLE_REF_0000>Алиса</TABLE_REF_0000> "
        "в <TABLE_REF_0001>Лондоне</TABLE_REF_0001>?"
    )
    assert extract_tagged_spans(text) == {
        "<TABLE_REF_0000>": "Алиса",
        "<TABLE_REF_0001>": "Лондоне",
    }


def test_extract_tagged_spans_empty_input():
    assert extract_tagged_spans("") == {}
    assert extract_tagged_spans("no tags here") == {}


def test_extract_tagged_spans_preserves_order():
    text = "<TABLE_REF_0001>second</TABLE_REF_0001> <TABLE_REF_0000>first</TABLE_REF_0000>"
    assert list(extract_tagged_spans(text)) == ["<TABLE_REF_0001>", "<TABLE_REF_0000>"]


def test_extract_tagged_spans_content_with_spaces_and_punctuation():
    text = "в <TABLE_REF_0000>Нью-Йорке, США</TABLE_REF_0000> в 2001 году"
    assert extract_tagged_spans(text) == {"<TABLE_REF_0000>": "Нью-Йорке, США"}


def test_extract_tagged_spans_inner_non_mention_tags_are_content():
    # Внутренние теги, не являющиеся <TABLE_REF_XXXX>, считаются содержимым упоминания.
    text = "x <TABLE_REF_0000>a <b>c</b></TABLE_REF_0000> y"
    assert extract_tagged_spans(text) == {"<TABLE_REF_0000>": "a <b>c</b>"}


def test_extract_tagged_spans_multiline_content():
    text = "<TABLE_REF_0000>line1\nline2</TABLE_REF_0000>"
    assert extract_tagged_spans(text) == {"<TABLE_REF_0000>": "line1\nline2"}