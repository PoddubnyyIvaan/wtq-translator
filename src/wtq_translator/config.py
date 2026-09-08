from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .enums import AnswerLanguageMode, TranslationMode


class ProjectConfig(BaseModel):
    input_root: Path
    output_root: Path
    source_language: str = "auto"
    target_language: str = "ru"
    seed: int = 42


class TranslationConfig(BaseModel):
    mode: TranslationMode = TranslationMode.FULL
    answer_language_mode: AnswerLanguageMode = AnswerLanguageMode.TABLE


class MentionNormalizationConfig(BaseModel):
    unicode: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFKC"
    casefold: bool = True
    collapse_whitespace: bool = True
    strip_outer_punctuation: bool = True


class MentionsConfig(BaseModel):
    normalization: MentionNormalizationConfig = Field(default_factory=MentionNormalizationConfig)
    longest_match_first: bool = True
    include_headers: bool = True
    include_numeric_mentions: bool = True
    allow_lemma_match: bool = False
    allow_fuzzy_match: bool = False
    allow_semantic_match: bool = False
    on_ambiguity: Literal["needs_review", "failed"] = "needs_review"
    spacy_model: str = "en_core_web_trf"
    stop_pos: list[str] = Field(default_factory=lambda: ["DET", "ADP", "PART"])
    exclude_numbers: bool = True
    exclude_dates: bool = True

    @model_validator(mode="after")
    def deterministic_mvp(self) -> "MentionsConfig":
        if self.allow_lemma_match or self.allow_fuzzy_match or self.allow_semantic_match:
            raise ValueError("MVP supports only exact matching after basic normalization")
        return self


class ModelConfig(BaseModel):
    provider: str = "openai_compatible"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model_name: str = "qwen2.5:3b"
    temperature: float = 0.0
    max_output_tokens: int = 4096
    timeout_seconds: float = 180.0
    retries: int = 3


class EmbeddingConfig(BaseModel):
    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    timeout_seconds: float = 180.0


class TableConfig(BaseModel):
    rows_per_chunk: int = 20


class BackTranslationConfig(BaseModel):
    enabled: bool = True


class ValidationConfig(BaseModel):
    question_cosine_threshold: float | None = None
    strict_masks: bool = True
    strict_table_shape: bool = True
    strict_immutable_values: bool = True


class StateConfig(BaseModel):
    sqlite_path: Path | None = None


class CacheConfig(BaseModel):
    enabled: bool = True
    cache_embeddings: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    llm_io_enabled: bool = True
    llm_io_path: Path | None = None


class RunnerConfig(BaseModel):
    checkpoint_every: int = 25
    continue_on_error: bool = True
    retry_failed_on_resume: bool = False
    rerun_needs_review: bool = False


class AppConfig(BaseModel):
    project: ProjectConfig
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    mentions: MentionsConfig = Field(default_factory=MentionsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    table: TableConfig = Field(default_factory=TableConfig)
    back_translation: BackTranslationConfig = Field(default_factory=BackTranslationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)

    @model_validator(mode="after")
    def resolve_and_validate(self) -> "AppConfig":
        if self.state.sqlite_path is None:
            self.state.sqlite_path = self.project.output_root / "state.sqlite"
        if self.logging.llm_io_path is None:
            self.logging.llm_io_path = self.project.output_root / "logs" / "llm_io.jsonl"
        if not self.back_translation.enabled:
            raise ValueError("Back translation is mandatory in the current project version")
        if self.table.rows_per_chunk <= 0:
            raise ValueError("table.rows_per_chunk must be > 0")
        if self.runner.checkpoint_every <= 0:
            raise ValueError("runner.checkpoint_every must be > 0")
        return self


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
