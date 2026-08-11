from enum import Enum


class TranslationMode(str, Enum):
    FULL = "full"
    SHELL_ONLY = "shell_only"
    MENTIONS_ONLY = "mentions_only"


class AnswerLanguageMode(str, Enum):
    TABLE = "table"
    QUESTION = "question"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    ACCEPTED = "accepted"
    ACCEPTED_WITH_WARNINGS = "accepted_with_warnings"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class MentionSourceType(str, Enum):
    CELL = "cell"
    HEADER = "header"
