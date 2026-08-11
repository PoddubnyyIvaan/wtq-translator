from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

from ..config import MentionsConfig
from ..enums import MentionSourceType
from ..models import (
    AmbiguousMention,
    MentionSearchResult,
    TableCoordinate,
    TableData,
    TableMention,
)
from .normalize import normalize_for_match, normalize_with_mapping


@dataclass(frozen=True)
class _Candidate:
    normalized: str
    source_text: str
    source_type: MentionSourceType
    coordinate: TableCoordinate


@dataclass(frozen=True)
class _Hit:
    start: int
    end: int
    normalized: str
    source_text: str
    source_type: MentionSourceType
    coordinate: TableCoordinate

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class _SpanGroup:
    start: int
    end: int
    normalized: str
    hits: tuple[_Hit, ...]

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def coordinates(self) -> tuple[TableCoordinate, ...]:
        return tuple(sorted({h.coordinate for h in self.hits}, key=_coordinate_sort_key))

    @property
    def is_resolvable(self) -> bool:
        coords = self.coordinates
        if len(coords) == 1:
            return True
        # Repeated occurrences of the same normalized cell value inside ONE column are
        # still an unambiguous link at the level required by the project. We retain all
        # rows and later require their translated values to be identical before the
        # value can be inserted into the question.
        return (
            all(h.source_type == MentionSourceType.CELL for h in self.hits)
            and len({c.column for c in coords}) == 1
        )


def _coordinate_sort_key(c: TableCoordinate) -> tuple[int, int]:
    # Header precedes rows of the same column for deterministic serialization.
    return (c.column, -1 if c.row is None else c.row)


def _is_numeric(text: str) -> bool:
    value = text.strip().replace(",", "")
    try:
        float(value)
        return True
    except ValueError:
        return False


def _build_candidates(table: TableData, cfg: MentionsConfig) -> list[_Candidate]:
    result: list[_Candidate] = []
    if cfg.include_headers:
        for col, value in enumerate(table.headers):
            norm = normalize_for_match(value, cfg.normalization)
            if norm and (cfg.include_numeric_mentions or not _is_numeric(norm)):
                result.append(_Candidate(norm, value, MentionSourceType.HEADER, TableCoordinate(None, col)))

    for row_idx, row in enumerate(table.rows):
        for col_idx, value in enumerate(row):
            norm = normalize_for_match(value, cfg.normalization)
            if not norm:
                continue
            if not cfg.include_numeric_mentions and _is_numeric(norm):
                continue
            result.append(_Candidate(norm, value, MentionSourceType.CELL, TableCoordinate(row_idx, col_idx)))
    return result


def _collect_hits(question: str, table: TableData, cfg: MentionsConfig) -> list[_Hit]:
    normalized_q = normalize_with_mapping(question, cfg.normalization)
    candidates = _build_candidates(table, cfg)
    hits: list[_Hit] = []

    for c in candidates:
        if not c.normalized:
            continue
        for m in re.finditer(re.escape(c.normalized), normalized_q.text):
            ns, ne = m.span()
            # Prevent accidental substring matches such as header "A" inside "Was".
            if c.normalized[0].isalnum() and ns > 0 and normalized_q.text[ns - 1].isalnum():
                continue
            if c.normalized[-1].isalnum() and ne < len(normalized_q.text) and normalized_q.text[ne].isalnum():
                continue
            if ns >= len(normalized_q.source_start) or ne <= 0:
                continue
            src_start = normalized_q.source_start[ns]
            src_end = normalized_q.source_end[ne - 1]
            hits.append(_Hit(src_start, src_end, c.normalized, c.source_text, c.source_type, c.coordinate))
    return hits


def _group_hits(hits: list[_Hit]) -> list[_SpanGroup]:
    grouped: dict[tuple[int, int, str], list[_Hit]] = defaultdict(list)
    for hit in hits:
        grouped[(hit.start, hit.end, hit.normalized)].append(hit)
    return [
        _SpanGroup(start, end, normalized, tuple(group))
        for (start, end, normalized), group in grouped.items()
    ]


def _select_maximal_non_overlapping(groups: list[_SpanGroup]) -> list[_SpanGroup]:
    """Select longest non-overlapping spans BEFORE resolving ambiguity.

    Ambiguous long matches intentionally block shorter nested matches. Otherwise a
    phrase such as `New York City` could be ambiguous while the matcher silently masks
    only nested `York`, violating the maximal-fragment rule.
    """
    ordered = sorted(groups, key=lambda g: (-g.length, g.start, g.end, g.normalized))
    selected: list[_SpanGroup] = []
    for group in ordered:
        overlaps = any(not (group.end <= x.start or group.start >= x.end) for x in selected)
        if not overlaps:
            selected.append(group)
    selected.sort(key=lambda g: (g.start, g.end))
    return selected


def find_table_mentions(question: str, table: TableData, cfg: MentionsConfig | None = None) -> MentionSearchResult:
    cfg = cfg or MentionsConfig()
    groups = _select_maximal_non_overlapping(_group_hits(_collect_hits(question, table, cfg)))

    mentions: list[TableMention] = []
    ambiguous: list[AmbiguousMention] = []

    for group in groups:
        coords = group.coordinates
        source_texts = tuple(sorted({h.source_text for h in group.hits}))
        if not group.is_resolvable:
            ambiguous.append(AmbiguousMention(
                question_text=question[group.start:group.end],
                normalized_text=group.normalized,
                start=group.start,
                end=group.end,
                coordinates=coords,
                candidates=source_texts,
            ))
            continue

        source_types = {h.source_type for h in group.hits}
        # A resolvable repeated group can only be all cells in one column. Single-hit
        # groups keep their original type.
        source_type = next(iter(source_types)) if len(source_types) == 1 else group.hits[0].source_type
        question_text = question[group.start:group.end]
        exact_source = next((s for s in source_texts if s == question_text), None)
        table_source_text = exact_source or source_texts[0]

        mentions.append(TableMention(
            mask=f"<TABLE_REF_{len(mentions):04d}>",
            question_text=question_text,
            normalized_text=group.normalized,
            start=group.start,
            end=group.end,
            source_type=source_type,
            coordinates=coords,
            table_source_text=table_source_text,
        ))

    return MentionSearchResult(mentions=mentions, ambiguous=ambiguous)
