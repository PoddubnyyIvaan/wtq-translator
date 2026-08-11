from __future__ import annotations

from collections import Counter

from ..models import QualityReport


def summarize(reports: list[QualityReport]) -> dict:
    statuses = Counter(r.status.value for r in reports)
    return {"total": len(reports), "statuses": dict(statuses)}
