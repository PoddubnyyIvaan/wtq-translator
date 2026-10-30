"""Тесты агрегации статистики рана (reporting/stats.py) — без LLM."""
import json
from pathlib import Path

import pytest

from wtq_translator.quality import KNOWN_METRICS
from wtq_translator.reporting.stats import aggregate_metrics, load_quality_rows, run_stats


def _write_quality(run_dir: Path, rows: list[dict]) -> None:
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "quality.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_rows() -> list[dict]:
    """Часть метрик None, часть строк с warnings (вторая и третья)."""
    return [
        {
            "question_id": "q1", "status": "accepted", "warnings": [],
            "question_cosine_similarity": 0.8, "question_bert_f1": 0.5,
        },
        {
            "question_id": "q2", "status": "accepted_with_warnings", "warnings": ["w1", "w2"],
            "question_cosine_similarity": 0.6, "question_bert_f1": None,
        },
        {
            "question_id": "q3", "status": "needs_review", "warnings": ["w3"],
            "question_cosine_similarity": None, "question_bert_f1": 0.7,
        },
        {
            "question_id": "q4", "status": "accepted", "warnings": [],
            "question_cosine_similarity": 0.9, "question_bert_f1": None,
        },
    ]


def test_aggregate_metrics_avg_min_counts_ignore_none(tmp_path: Path):
    run_dir = tmp_path / "run-abc"
    _write_quality(run_dir, _sample_rows())
    agg = aggregate_metrics(load_quality_rows(run_dir))

    assert agg["quality_rows"] == 4
    # question_cosine_similarity: [0.8, 0.6, 0.9]
    assert agg["avg"]["question_cosine_similarity"] == pytest.approx((0.8 + 0.6 + 0.9) / 3)
    assert agg["min"]["question_cosine_similarity"] == 0.6
    assert agg["metric_counts"]["question_cosine_similarity"] == 3
    # question_bert_f1: [0.5, 0.7]
    assert agg["avg"]["question_bert_f1"] == 0.6
    assert agg["min"]["question_bert_f1"] == 0.5
    assert agg["metric_counts"]["question_bert_f1"] == 2
    # метрика без значений → None/0
    assert agg["avg"]["back_table_bert_mean_f1"] is None
    assert agg["min"]["back_table_bert_mean_f1"] is None
    assert agg["metric_counts"]["back_table_bert_mean_f1"] == 0
    # warnings: строки с warning'ами q2, q3; сумма длин = 3
    assert agg["warnings"] == {"records": 2, "total": 3}
    # задействованы все KNOWN_METRICS
    assert set(agg["avg"]) == set(KNOWN_METRICS)
    assert set(agg["min"]) == set(KNOWN_METRICS)
    assert set(agg["metric_counts"]) == set(KNOWN_METRICS)


def test_run_stats_shares_and_persisted_stats_json(tmp_path: Path):
    run_dir = tmp_path / "run-abc"
    _write_quality(run_dir, _sample_rows())
    status_counts = {
        "accepted": 2, "accepted_with_warnings": 1,
        "needs_review": 1, "failed": 1, "excluded": 1,
    }
    total = sum(status_counts.values())

    report = run_stats(
        run_dir, status_counts=status_counts, total=total,
        extra={"processed_this_invocation": 4},
    )

    assert report["run_id"] == "run-abc"
    assert report["total"] == 6
    # backward-compat ключи
    assert report["statuses"] == status_counts
    assert report["excluded_share"] == 1 / 6
    assert report["processed_this_invocation"] == 4
    # статусные и доли warnings
    assert report["status_shares"]["excluded"] == 1 / 6
    assert report["status_shares"]["failed"] == 1 / 6
    assert report["status_shares"]["accepted"] == 2 / 6
    assert report["warnings"]["records"] == 2
    assert report["warnings"]["records_share"] == 2 / 6
    assert report["warnings"]["total"] == 3
    assert report["quality_rows"] == 4
    assert report["avg"]["question_bert_f1"] == 0.6
    assert report["min"]["question_bert_f1"] == 0.5
    assert report["metric_counts"]["question_cosine_similarity"] == 3

    # записанный reports/stats.json идентичен отчёту
    persisted = json.loads((run_dir / "reports" / "stats.json").read_text(encoding="utf-8"))
    assert persisted == report


def test_empty_quality_rows(tmp_path: Path):
    run_dir = tmp_path / "run-empty"
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    # пустой файл и строка-пустышка
    (reports / "quality.jsonl").write_text("\n\n", encoding="utf-8")

    agg = aggregate_metrics(load_quality_rows(run_dir))
    assert agg["quality_rows"] == 0
    assert all(v is None for v in agg["avg"].values())
    assert all(v is None for v in agg["min"].values())
    assert all(v == 0 for v in agg["metric_counts"].values())
    assert agg["warnings"] == {"records": 0, "total": 0}

    report = run_stats(run_dir, status_counts={"accepted": 1}, total=1)
    assert report["avg"]["question_bert_f1"] is None
    assert report["warnings"] == {"records": 0, "total": 0, "records_share": 0.0}


def test_run_stats_total_zero_no_division(tmp_path: Path):
    run_dir = tmp_path / "run-zero"
    (run_dir / "reports").mkdir(parents=True)
    report = run_stats(run_dir, status_counts={}, total=0)
    assert report["status_shares"] == {}
    assert report["excluded_share"] == 0.0
    assert report["warnings"]["records_share"] == 0.0
    assert report["total"] == 0


def test_load_quality_rows_deduplicates_by_question_id(tmp_path: Path):
    # Две (и более) строки одного question_id — как при rerun_needs_review/re-resume:
    # в статистику идёт последняя строка, порядок первого вхождения сохраняется.
    run_dir = tmp_path / "run-dup"
    _write_quality(run_dir, [
        {
            "question_id": "q1", "status": "accepted_with_warnings", "warnings": ["w-a", "w-b"],
            "question_cosine_similarity": 0.5, "question_bert_f1": 0.4,
        },
        {
            "question_id": "q2", "status": "accepted", "warnings": [],
            "question_cosine_similarity": 0.7, "question_bert_f1": 0.6,
        },
        {
            "question_id": "q1", "status": "accepted", "warnings": ["w-last"],
            "question_cosine_similarity": 0.95, "question_bert_f1": 0.85,
        },
    ])

    rows = load_quality_rows(run_dir)
    assert [r["question_id"] for r in rows] == ["q1", "q2"]

    agg = aggregate_metrics(rows)
    assert agg["quality_rows"] == 2
    # метрики и warnings берутся из последней строки q1 + строки q2
    assert agg["avg"]["question_cosine_similarity"] == pytest.approx((0.95 + 0.7) / 2)
    assert agg["min"]["question_cosine_similarity"] == 0.7
    assert agg["metric_counts"]["question_cosine_similarity"] == 2
    assert agg["avg"]["question_bert_f1"] == pytest.approx((0.85 + 0.6) / 2)
    assert agg["warnings"] == {"records": 1, "total": 1}

    report = run_stats(run_dir, status_counts={"accepted": 1, "accepted_with_warnings": 1}, total=2)
    assert report["quality_rows"] == 2
    assert report["warnings"]["records"] == 1
    assert report["warnings"]["total"] == 1