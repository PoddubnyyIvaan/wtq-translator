from __future__ import annotations

from collections import Counter
import re

from ..models import MaskValidation

MASK_RE = re.compile(r"<TABLE_REF_\d{4}>")


def extract_masks(text: str) -> list[str]:
    return MASK_RE.findall(text)


def validate_masks(expected_text: str, actual_text: str) -> MaskValidation:
    expected = Counter(extract_masks(expected_text))
    actual = Counter(extract_masks(actual_text))
    missing = expected - actual
    added = actual - expected
    return MaskValidation(dict(expected), dict(actual), dict(missing), dict(added))
