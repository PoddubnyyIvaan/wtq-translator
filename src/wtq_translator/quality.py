from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from .enums import ProcessingStatus

# Metric fields on QualityReport that may be used as filter thresholds. Any other
# key in the filter config is treated as a typo and rejected on load.
KNOWN_METRICS: tuple[str, ...] = (
    "question_cosine_similarity",
    "question_bert_precision",
    "question_bert_recall",
    "question_bert_f1",
    "back_table_min_similarity",
    "back_table_mean_similarity",
    "back_table_bert_min_f1",
    "back_table_bert_mean_f1",
    "back_table_bert_min_precision",
    "back_table_bert_mean_precision",
    "back_table_bert_min_recall",
    "back_table_bert_mean_recall",
)


class FilterConfig(BaseModel):
    """Schema for the standalone `wtq-translate filter` config YAML.

    `status_include` lists accepted `ProcessingStatus` values; omitted/empty means
    "accept all statuses". `metrics` maps a quality-metric field name to a lower
    bound (inclusive `>=`); `null` means "no lower bound" — the metric is still
    included in the reported avg/min description set but imposes no filtering.
    """

    status_include: list[ProcessingStatus] | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_metrics(self) -> "FilterConfig":
        unknown = [m for m in self.metrics if m not in KNOWN_METRICS]
        if unknown:
            raise ValueError(
                f"Unknown filter metric(s): {sorted(unknown)}; "
                f"known metrics: {sorted(KNOWN_METRICS)}"
            )
        for name, threshold in self.metrics.items():
            if threshold is not None and not (0.0 <= threshold <= 1.0):
                raise ValueError(f"filter metric {name} threshold must be in [0, 1]")
        return self

    @property
    def effective_status_include(self) -> tuple[ProcessingStatus, ...]:
        """Statuses accepted by this config; all statuses when not specified."""
        return tuple(self.status_include or list(ProcessingStatus))

    @property
    def threshold_metrics(self) -> tuple[str, ...]:
        """Metric keys that impose a lower bound, in YAML declaration order."""
        return tuple(m for m, t in self.metrics.items() if t is not None)

    @property
    def description_metrics(self) -> tuple[str, ...]:
        """All configured metrics (incl. unbounded/null), in YAML declaration order."""
        return tuple(self.metrics.keys())


def load_filter_config(path: str | Path) -> FilterConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return FilterConfig.model_validate(raw)
