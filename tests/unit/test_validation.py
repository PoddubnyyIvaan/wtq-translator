from wtq_translator.validation.masks import validate_masks
from wtq_translator.validation.similarity import cosine_similarity


def test_mask_validation_detects_duplicate():
    expected = "A <TABLE_REF_0000> B"
    actual = "A <TABLE_REF_0000> <TABLE_REF_0000> B"
    result = validate_masks(expected, actual)
    assert not result.ok
    assert result.added["<TABLE_REF_0000>"] == 1


def test_cosine():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
