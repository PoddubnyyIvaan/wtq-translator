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


OPEN_TAG_RE = re.compile(r"<TABLE_REF_\d{4}>")
CLOSE_TAG_RE = re.compile(r"</TABLE_REF_\d{4}>")


def extract_open_tags(text: str) -> list[str]:
    return OPEN_TAG_RE.findall(text)


def extract_close_tags(text: str) -> list[str]:
    return CLOSE_TAG_RE.findall(text)


def validate_tags(expected: str, actual: str) -> MaskValidation:
    expected_counts = Counter(extract_open_tags(expected)) + Counter(extract_close_tags(expected))
    actual_counts = Counter(extract_open_tags(actual)) + Counter(extract_close_tags(actual))
    missing = expected_counts - actual_counts
    added = actual_counts - expected_counts
    return MaskValidation(dict(expected_counts), dict(actual_counts), dict(missing), dict(added))
