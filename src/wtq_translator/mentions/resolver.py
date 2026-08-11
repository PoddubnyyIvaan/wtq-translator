from __future__ import annotations

from ..models import MentionMap, MentionSearchResult


def build_mention_map(search: MentionSearchResult) -> MentionMap:
    return MentionMap(search.mentions)
