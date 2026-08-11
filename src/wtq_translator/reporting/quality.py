from __future__ import annotations

import json
from pathlib import Path

from ..models import QualityReport


def append_quality(path: str | Path, report: QualityReport) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")
