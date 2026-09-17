from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import read_question_tsv_document, write_question_tsv
from .enums import ProcessingStatus
from .quality import FilterConfig

_STANDARD_FIELDNAMES = ("id", "utterance", "context", "targetValue")


def _fmt(value: float) -> str:
    """Shortest-lossless string form used in filenames and drop reasons."""
    return str(value)


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _locate_dataset(run_dir: Path) -> Path:
    """Resolve the run's translated dataset TSV.

    Prefers the path recorded in `manifest.json` (its basename under
    `<run_dir>/dataset/`), then falls back to a single top-level dataset TSV.
    """
    try:
        manifest = _load_manifest(run_dir)
    except FileNotFoundError:
        manifest = {}
    rel = manifest.get("output_questions_tsv")
    if rel:
        candidate = run_dir / "dataset" / Path(rel).name
        if candidate.is_file():
            return candidate
    dataset_dir = run_dir / "dataset"
    candidates = sorted(p for p in dataset_dir.glob("*.tsv") if p.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Ambiguous dataset TSV under {dataset_dir}: "
            f"{[c.name for c in candidates]}"
        )
    raise FileNotFoundError(f"No dataset TSV found under {dataset_dir}")


def _load_quality(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load reports/quality.jsonl into {question_id: raw_row}."""
    path = run_dir / "reports" / "quality.jsonl"
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get("question_id")
            if qid is not None:
                rows[qid] = data
    return rows


def _desc_rank(values: list[float]) -> list[int]:
    """Competition rank (higher value -> lower number) over `values`.

    Ties share the same rank = 1 + count of strictly-higher values.
    """
    n = len(values)
    return [
        1 + sum(1 for j in range(n) if values[j] > values[i])
        for i in range(n)
    ]


def _run_filter(run_dir: Path, config: FilterConfig) -> dict[str, Any]:
    manifest = _load_manifest(run_dir)
    run_id = manifest.get("run_id", run_dir.name)
    dataset_path = _locate_dataset(run_dir)
    quality_path = run_dir / "reports" / "quality.jsonl"

    document = read_question_tsv_document(dataset_path)
    records = list(document.records)
    quality = _load_quality(run_dir)

    total = len(records)
    desc_metrics = list(config.description_metrics)

    status_counts: Counter = Counter()
    filtered: list[dict[str, Any]] = []
    filtered_out_status = 0
    filtered_out_threshold = 0

    kept_values: list[list[float | None]] = []
    kept_records: list[Any] = []

    for rec in records:
        qdata = quality.get(rec.id)
        if qdata is None:
            filtered_out_threshold += 1
            filtered.append({
                "question_id": rec.id,
                "reason": "no quality row",
                "metric": None,
                "status": None,
                "value": None,
            })
            continue

        status = ProcessingStatus(qdata.get("status", ProcessingStatus.PENDING.value))
        status_counts[status.value] += 1
        if status not in config.effective_status_include:
            filtered_out_status += 1
            continue

        values = [qdata.get(m) for m in desc_metrics]
        drop: tuple[str, float | None, float] | None = None
        for i, metric in enumerate(desc_metrics):
            threshold = config.metrics[metric]
            if threshold is None:
                continue
            value = values[i]
            if value is None or value < threshold:
                drop = (metric, value, threshold)
                break
        if drop is not None:
            metric, value, threshold = drop
            filtered_out_threshold += 1
            reason = (
                f"{metric}: n/a < {_fmt(threshold)}"
                if value is None
                else f"{metric}: {_fmt(value)} < {_fmt(threshold)}"
            )
            filtered.append({
                "question_id": rec.id,
                "reason": reason,
                "metric": metric,
                "status": status.value,
                "value": value,
            })
            continue

        kept_records.append(rec)
        kept_values.append(values)

    kept = len(kept_records)

    # Order kept rows by ascending minimum competition rank across threshold
    # metrics; tie-break deterministically by (question_id, context).
    ordered: list[Any] = []
    if kept:
        n = kept
        min_rank: list[int] = []
        for metric in config.threshold_metrics:
            index = desc_metrics.index(metric)
            column = [row[index] for row in kept_values]  # type: ignore[index]
            ranks = _desc_rank(column)
            if not min_rank:
                min_rank = ranks
            else:
                min_rank = [min(a, b) for a, b in zip(min_rank, ranks)]
        if not config.threshold_metrics:
            min_rank = [0] * n
        order = sorted(
            range(n),
            key=lambda i: (min_rank[i], kept_records[i].id, kept_records[i].context),
        )
        ordered = [kept_records[i] for i in order]

    # avg / min over the kept set for every configured (description) metric.
    avg: dict[str, float | None] = {}
    minimum: dict[str, float | None] = {}
    for index, metric in enumerate(desc_metrics):
        column = [row[index] for row in kept_values if row[index] is not None]
        if column:
            avg[metric] = sum(column) / len(column)
            minimum[metric] = min(column)
        else:
            avg[metric] = None
            minimum[metric] = None

    filtered_dir = run_dir / "filtered"
    params = "|".join(
        f"{m}-{_fmt(config.metrics[m])}" for m in config.threshold_metrics  # type: ignore[misc]
    )
    base = f"{run_dir.name}-sorted{'-' + params if params else ''}"
    tsv_path = filtered_dir / f"{base}.tsv"
    stats_path = filtered_dir / f"{base}-stats.json"

    write_question_tsv(ordered, tsv_path, fieldnames=_STANDARD_FIELDNAMES)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "run_id": run_id,
        "source_tsv": str(dataset_path.resolve()),
        "source_quality_jsonl": str(quality_path.resolve()),
        "status_include": [s.value for s in config.effective_status_include],
        "metrics": {
            m: {"min": config.metrics[m]}
            for m in config.threshold_metrics
        },
        "total": total,
        "status_counts": dict(status_counts),
        "filtered_out_status": filtered_out_status,
        "filtered_out_threshold": filtered_out_threshold,
        "kept": kept,
        "filtered": filtered,
        "avg": avg,
        "min": minimum,
    }
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_filter(run_dir: str | Path, config: FilterConfig) -> dict[str, Any]:
    """Filter a run's translated dataset by status + metric thresholds.

    Writes `<run_dir>/filtered/<run-folder>-sorted-<params>.tsv` and the matching
    `-stats.json`, and returns the stats report (also persisted to disk).
    """
    return _run_filter(Path(run_dir), config)
