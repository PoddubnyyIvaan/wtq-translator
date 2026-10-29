from __future__ import annotations

import json
from pathlib import Path

from ..enums import ProcessingStatus
from ..quality import KNOWN_METRICS
from .jsonl_dedupe import dedupe_last_wins, load_jsonl_rows


def aggregate_metrics(rows: list[dict]) -> dict:
    """Чистая агрегация строк quality.jsonl.

    Для каждой метрики из ``KNOWN_METRICS`` считает ``avg``/``min``/``metric_counts``
    по не-None значениям (нет значений → метрика равна None/0); плюс счётчики
    warnings (``records`` — число строк с непустым ``warnings``, ``total`` — сумма
    длин всех ``warnings``) и ``quality_rows``.
    """
    values = {m: [] for m in KNOWN_METRICS}
    warning_records = 0
    warning_total = 0
    for row in rows:
        warnings = row.get("warnings")
        if warnings:
            warning_records += 1
            warning_total += len(warnings)
        for m in KNOWN_METRICS:
            v = row.get(m)
            if v is not None:
                values[m].append(v)
    return {
        "quality_rows": len(rows),
        "warnings": {"records": warning_records, "total": warning_total},
        "avg": {m: (sum(vs) / len(vs) if vs else None) for m, vs in values.items()},
        "min": {m: (min(vs) if vs else None) for m, vs in values.items()},
        "metric_counts": {m: len(vs) for m, vs in values.items()},
    }


def load_quality_rows(run_dir: Path) -> list[dict]:
    """Построчное чтение ``reports/quality.jsonl`` с дедупликацией.

    Возвращает по одной строке на ``question_id`` (last wins по порядку файла);
    устойчив к пустым строкам, битому JSON и отсутствию файла.
    """
    path = Path(run_dir) / "reports" / "quality.jsonl"
    if not path.exists():
        return []
    return dedupe_last_wins(load_jsonl_rows(path), key_fn=lambda r: r.get("question_id"))


def run_stats(
    run_dir: Path,
    *,
    status_counts: dict[str, int],
    total: int,
    extra: dict | None = None,
    save: bool = True,
) -> dict:
    """Сводная статистика рана.

    Статусы/доли считаются только из ``status_counts`` (state), метрики и warnings —
    только из ``reports/quality.jsonl``. При ``save=True`` итог пишется в
    ``run_dir/reports/stats.json`` (indent=2, ensure_ascii=False).
    """
    run_dir = Path(run_dir)
    agg = aggregate_metrics(load_quality_rows(run_dir))
    warn_records = agg["warnings"]["records"]
    report = {
        "run_id": run_dir.name,
        "total": total,
        **(extra or {}),
        "statuses": status_counts,
        "status_shares": {s: (n / total if total else 0.0) for s, n in status_counts.items()},
        "excluded_share": (status_counts.get(ProcessingStatus.EXCLUDED.value, 0) / total) if total else 0.0,
        "quality_rows": agg["quality_rows"],
        "warnings": {**agg["warnings"], "records_share": (warn_records / total) if total else 0.0},
        "avg": agg["avg"],
        "min": agg["min"],
        "metric_counts": agg["metric_counts"],
    }
    if save:
        out = run_dir / "reports" / "stats.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report