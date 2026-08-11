from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_event(path: str | Path, *, run_id: str, question_id: str, stage: str, level: str, code: str, message: str, extra: dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "question_id": question_id,
        "stage": stage,
        "level": level,
        "code": code,
        "message": message,
        "extra": extra or {},
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
