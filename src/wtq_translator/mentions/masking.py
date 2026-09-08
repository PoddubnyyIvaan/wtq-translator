from __future__ import annotations

import re

from collections.abc import Mapping, Sequence

from ..models import MentionMap, TableMention


def mask_question(question: str, mentions: Sequence[TableMention]) -> str:
    value = question
    for m in sorted(mentions, key=lambda x: x.start, reverse=True):
        if question[m.start:m.end] != m.question_text:
            raise ValueError(f"Mention span mismatch for {m.mask}")
        value = value[:m.start] + m.mask + value[m.end:]
    return value


def restore_masks(masked_question: str, replacements: Mapping[str, str]) -> str:
    value = masked_question
    for mask, replacement in replacements.items():
        value = value.replace(mask, replacement)
    return value


def restore_original(masked_question: str, mention_map: MentionMap) -> str:
    """Restore the exact fragments that originally appeared in the question.

    This deliberately uses `question_text`, not the source table cell. Exact-normalized
    matching may accept harmless case/punctuation differences, and back-translation
    quality must be compared with the actual original question.
    """
    return restore_masks(masked_question, {m.mask: m.question_text for m in mention_map.mentions})


def restore_from_table(masked_question: str, mention_map: MentionMap, table) -> str:
    """Insert table values linked to each mask.

    If the same source mention occurred repeatedly in one table column, the mention may
    have several equivalent coordinates. All corresponding values in `table` must be
    identical. This prevents silently inserting an arbitrary translation when a model
    translated repeated source values inconsistently.
    """
    return restore_masks(masked_question, {m.mask: mention_map.table_value(m.mask, table) for m in mention_map.mentions})


def mask_question_tagged(question: str, mentions: Sequence[TableMention]) -> str:
    """Wrap each mention fragment in an opening/closing tag without inner spaces."""
    value = question
    for m in sorted(mentions, key=lambda x: x.start, reverse=True):
        if question[m.start:m.end] != m.question_text:
            raise ValueError(f"Mention span mismatch for {m.mask}")
        value = value[:m.start] + m.mask + m.question_text + closing_mask(m.mask) + value[m.end:]
    return value


def strip_tags(text: str) -> str:
    """Remove all opening and closing <TABLE_REF_XXXX> tags."""
    return re.sub(r"</?TABLE_REF_\d{4}>", "", text)


def closing_mask(mask: str) -> str:
    """Convert '<TABLE_REF_0000>' into '</TABLE_REF_0000>'."""
    return mask[:1] + "/" + mask[1:]
