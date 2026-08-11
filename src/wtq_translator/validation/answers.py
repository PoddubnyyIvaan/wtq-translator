from __future__ import annotations

from ..models import AnswerMatch


def answer_matches_are_ambiguous(matches: list[AnswerMatch]) -> bool:
    return any(len(m.coordinates) > 1 for m in matches)
