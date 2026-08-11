from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..config import MentionNormalizationConfig

_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT = " \t\r\n.,;:!?\"'`()[]{}<>«»“”‘’"


def normalize_for_match(text: str, cfg: MentionNormalizationConfig | None = None) -> str:
    cfg = cfg or MentionNormalizationConfig()
    value = unicodedata.normalize(cfg.unicode, text)
    if cfg.casefold:
        value = value.casefold()
    if cfg.collapse_whitespace:
        value = _WS_RE.sub(" ", value)
    value = value.strip()
    if cfg.strip_outer_punctuation:
        value = value.strip(_EDGE_PUNCT)
    return value


@dataclass(frozen=True)
class NormalizedText:
    text: str
    source_start: tuple[int, ...]
    source_end: tuple[int, ...]


def normalize_with_mapping(text: str, cfg: MentionNormalizationConfig | None = None) -> NormalizedText:
    """Normalize while preserving a best-effort map back to source spans.

    NFKC can expand characters; each normalized char maps to the source character that
    produced it. Whitespace runs map to the full source whitespace run.
    """
    cfg = cfg or MentionNormalizationConfig()
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []

    for i, ch in enumerate(text):
        piece = unicodedata.normalize(cfg.unicode, ch)
        if cfg.casefold:
            piece = piece.casefold()
        for out in piece:
            chars.append(out)
            starts.append(i)
            ends.append(i + 1)

    if cfg.collapse_whitespace:
        out_chars: list[str] = []
        out_starts: list[int] = []
        out_ends: list[int] = []
        i = 0
        while i < len(chars):
            if chars[i].isspace():
                j = i + 1
                while j < len(chars) and chars[j].isspace():
                    j += 1
                out_chars.append(" ")
                out_starts.append(starts[i])
                out_ends.append(ends[j - 1])
                i = j
            else:
                out_chars.append(chars[i])
                out_starts.append(starts[i])
                out_ends.append(ends[i])
                i += 1
        chars, starts, ends = out_chars, out_starts, out_ends

    # strip whitespace only. Outer punctuation stripping is candidate-side policy,
    # because removing punctuation from the question would make span mapping unsafe.
    left = 0
    right = len(chars)
    while left < right and chars[left].isspace():
        left += 1
    while right > left and chars[right - 1].isspace():
        right -= 1
    return NormalizedText("".join(chars[left:right]), tuple(starts[left:right]), tuple(ends[left:right]))
