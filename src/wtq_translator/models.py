from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .enums import MentionSourceType, ProcessingStatus, TranslationMode


@dataclass(frozen=True)
class QuestionRecord:
    id: str
    utterance: str
    context: str
    target_value: str
    # Unknown/extra WTQ columns are preserved across dataset-wide runs.
    extra_fields: dict[str, str] = field(default_factory=dict, compare=False)

    def with_translation(self, *, utterance: str, context: str, target_value: str) -> "QuestionRecord":
        return QuestionRecord(
            id=self.id,
            utterance=utterance,
            context=context,
            target_value=target_value,
            extra_fields=dict(self.extra_fields),
        )


@dataclass(frozen=True)
class QuestionTSVDocument:
    fieldnames: tuple[str, ...]
    records: tuple[QuestionRecord, ...]


@dataclass(frozen=True)
class AlphabetDetection:
    """Результат детекции алфавитно-зависимого намерения вопроса.

    ``intent_id`` — определённое намерение (1..4) либо None, если намерения нет или
    детекция отключена. ``why`` — обоснование модели. ``error`` — непусто, если
    детекция не удалась после всех ретраев (fail-closed).
    """

    intent_id: int | None
    why: str | None
    error: str | None = None


@dataclass
class TableData:
    headers: list[str]
    rows: list[list[str]]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        width = len(self.headers)
        bad = [i for i, row in enumerate(self.rows) if len(row) != width]
        if bad:
            raise ValueError(f"Rows have wrong width: {bad[:5]}; expected {width}")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.headers)

    def value_at(self, coordinate: "TableCoordinate") -> str:
        if coordinate.column < 0 or coordinate.column >= len(self.headers):
            raise IndexError(f"Column out of range: {coordinate.column}")
        if coordinate.row is None:
            return self.headers[coordinate.column]
        if coordinate.row < 0 or coordinate.row >= len(self.rows):
            raise IndexError(f"Row out of range: {coordinate.row}")
        return self.rows[coordinate.row][coordinate.column]

    def copy(self) -> "TableData":
        return TableData(self.headers.copy(), [r.copy() for r in self.rows], self.source_path)


@dataclass(frozen=True, order=True)
class TableCoordinate:
    row: int | None
    column: int


@dataclass(frozen=True)
class TableMention:
    mask: str
    question_text: str
    normalized_text: str
    start: int
    end: int
    source_type: MentionSourceType
    coordinates: tuple[TableCoordinate, ...]
    table_source_text: str
    match_type: str = "exact_normalized"
    candidate_key: str = ""

    def __post_init__(self) -> None:
        if not self.coordinates:
            raise ValueError("TableMention must contain at least one table coordinate")

    @property
    def coordinate(self) -> TableCoordinate:
        """Primary coordinate kept for code that only needs the linked column.

        Repeated occurrences of the same exact value in one column are represented by
        multiple coordinates. They are not ambiguous at the column level, but all of
        their translated values must remain consistent before reinsertion.
        """
        return self.coordinates[0]


@dataclass(frozen=True)
class QuestionCandidate:
    lemma_key: str
    source_text: str
    start: int
    end: int
    length: int


@dataclass
class TableCandidate:
    id: str
    lemma_key: str
    source_text: str
    coordinates: list["TableCoordinate"]
    length: int
    translation: str = ""
    tokens: list[str] = field(default_factory=list)
    pos: list[str] = field(default_factory=list)
    ner: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContentRun:
    """Максимальный непрерывный content-run внутри одной ячейки/заголовка, с span'ом в исходной строке."""
    lemma_key: str
    start: int
    end: int


@dataclass
class CandidateSet:
    """Дедуплицированный набор табличных кандидатов + per-координата максимальные content-run'ы для реконструкции."""
    candidates: list["TableCandidate"] = field(default_factory=list)
    runs: dict["TableCoordinate", list["ContentRun"]] = field(default_factory=dict)


@dataclass(frozen=True)
class AmbiguousMention:
    question_text: str
    normalized_text: str
    start: int
    end: int
    coordinates: tuple[TableCoordinate, ...]
    candidates: tuple[str, ...]


@dataclass
class MentionSearchResult:
    mentions: list[TableMention] = field(default_factory=list)
    ambiguous: list[AmbiguousMention] = field(default_factory=list)


@dataclass
class MentionMap:
    mentions: list[TableMention]

    def by_mask(self) -> dict[str, TableMention]:
        return {m.mask: m for m in self.mentions}

    def masks(self) -> list[str]:
        return [m.mask for m in self.mentions]

    def original_value(self, mask: str) -> str:
        # The question fragment is the only value that can reconstruct the original
        # utterance byte-for-byte after masking. The table source value may differ in
        # case or surrounding punctuation while still matching after normalization.
        return self.by_mask()[mask].question_text

    def table_value(self, mask: str, table: TableData) -> str:
        mention = self.by_mask()[mask]
        values = [table.value_at(c) for c in mention.coordinates]
        unique = list(dict.fromkeys(values))
        if len(unique) != 1:
            raise ValueError(
                f"Table mention {mask} has inconsistent values at equivalent coordinates: {unique!r}"
            )
        return unique[0]


@dataclass(frozen=True)
class AnswerMatch:
    answer_index: int
    answer_text: str
    coordinates: tuple[TableCoordinate, ...]


@dataclass
class MaskValidation:
    expected: dict[str, int]
    actual: dict[str, int]
    missing: dict[str, int]
    added: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.added


@dataclass
class CellTranslationTrace:
    coordinate: TableCoordinate
    source: str
    translated: str
    back_translated: str | None = None
    similarity: float | None = None


@dataclass
class QualityReport:
    question_id: str
    mode: TranslationMode
    table_mentions_found: bool
    source_question: str
    translated_question: str = ""
    back_translated_question: str | None = None
    question_cosine_similarity: float | None = None
    question_bert_precision: float | None = None
    question_bert_recall: float | None = None
    question_bert_f1: float | None = None
    back_table_groups_compared: int = 0
    back_table_min_similarity: float | None = None
    back_table_mean_similarity: float | None = None
    back_table_bert_groups_compared: int = 0
    back_table_bert_min_f1: float | None = None
    back_table_bert_mean_f1: float | None = None
    back_table_bert_min_precision: float | None = None
    back_table_bert_mean_precision: float | None = None
    back_table_bert_min_recall: float | None = None
    back_table_bert_mean_recall: float | None = None
    back_table_similarity_examples: list[dict[str, Any]] = field(default_factory=list)
    original_masks: list[str] = field(default_factory=list)
    forward_masks: list[str] = field(default_factory=list)
    backward_masks: list[str] = field(default_factory=list)
    missing_forward_masks: list[str] = field(default_factory=list)
    added_forward_masks: list[str] = field(default_factory=list)
    missing_backward_masks: list[str] = field(default_factory=list)
    added_backward_masks: list[str] = field(default_factory=list)
    ambiguous_mentions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.PENDING
    domain: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        value["status"] = self.status.value
        return value


@dataclass
class TranslationResult:
    question: QuestionRecord
    source_table: TableData
    output_table: TableData
    translated_question: str
    translated_target_value: str
    quality: QualityReport
    mention_map: MentionMap


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    total: int
    processed: int
    skipped: int
    failed: int
    excluded: int = 0
    excluded_share: float = 0.0
    statuses: dict[str, int] = field(default_factory=dict)
    output_questions_tsv: Path = field(default_factory=Path)
    run_dir: Path = field(default_factory=Path)
