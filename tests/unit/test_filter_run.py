from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from wtq_translator.quality import FilterConfig, load_filter_config
from wtq_translator.quality_stats import run_filter

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "filter_run"


def _run_dir(tmp_path: Path) -> Path:
    dest = tmp_path / "run"
    shutil.copytree(FIXTURE, dest)
    return dest


def _read_tsv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def test_filter_status_and_thresholds(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    config = FilterConfig.model_validate({
        "status_include": ["accepted"],
        "metrics": {
            "question_cosine_similarity": None,
            "question_bert_f1": 0.85,
            "back_table_bert_mean_f1": 0.9,
            "back_table_mean_similarity": None,
        },
    })
    report = run_filter(run_dir, config)

    # Kept rows, ascending by min rank across threshold metrics: q8(f1 .93/bf1 .96),
    # q1, q2.
    kept_tsv = sorted((run_dir / "filtered").glob("*.tsv"))
    assert len(kept_tsv) == 1
    rows = _read_tsv(kept_tsv[0])
    assert [r["id"] for r in rows] == ["q8", "q1", "q2"]
    assert list(rows[0].keys()) == ["id", "utterance", "context", "targetValue"]

    assert report["total"] == 8
    assert report["kept"] == 3
    assert report["filtered_out_status"] == 1  # q5 needs_review
    assert report["filtered_out_threshold"] == 4  # q3, q4, q6, q7
    assert report["status_include"] == ["accepted"]
    assert report["status_counts"] == {"accepted": 6, "needs_review": 1}

    # threshold drops by reason, first failing metric preferred
    reasons = {f["question_id"]: f for f in report["filtered"]}
    assert reasons["q3"]["reason"] == "question_bert_f1: 0.6 < 0.85"
    assert reasons["q4"]["reason"] == "back_table_bert_mean_f1: 0.88 < 0.9"
    assert reasons["q6"]["reason"] == "question_bert_f1: 0.8 < 0.85"
    assert reasons["q7"]["reason"] == "no quality row"
    assert reasons["q3"]["metric"] == "question_bert_f1"
    assert reasons["q7"]["status"] is None

    # metrics report echoes only bounded thresholds
    assert report["metrics"] == {
        "question_bert_f1": {"min": 0.85},
        "back_table_bert_mean_f1": {"min": 0.9},
    }

    # avg/min cover all four configured (description) metrics
    assert report["avg"]["question_bert_f1"] == pytest.approx((0.93 + 0.90 + 0.88) / 3)
    assert report["avg"]["back_table_bert_mean_f1"] == pytest.approx((0.96 + 0.95 + 0.92) / 3)
    assert report["avg"]["question_cosine_similarity"] == pytest.approx((0.68 + 0.50 + 0.60) / 3)
    assert report["avg"]["back_table_mean_similarity"] == pytest.approx((0.88 + 0.70 + 0.75) / 3)
    assert report["min"]["question_bert_f1"] == pytest.approx(0.88)
    assert report["min"]["back_table_bert_mean_f1"] == pytest.approx(0.92)

    # stats file matches returned report and exists next to the tsv
    stats_path = kept_tsv[0].with_suffix("").as_posix() + "-stats.json"
    saved = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    assert saved == report
    assert Path(stats_path).name == "run-sorted-question_bert_f1-0.85|back_table_bert_mean_f1-0.9-stats.json"


def test_all_statuses_config_keeps_every_passing_row(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    config = FilterConfig.model_validate({
        "metrics": {"question_bert_f1": 0.85},
    })
    report = run_filter(run_dir, config)
    # q5 (needs_review) now kept since all statuses accepted
    assert report["filtered_out_status"] == 0
    assert report["kept"] == 5
    assert report["status_include"] == [
        "pending", "processing", "accepted", "accepted_with_warnings",
        "needs_review", "failed", "excluded",
    ]


def test_null_metrics_filter_nothing_at_config_level(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    config = FilterConfig.model_validate({
        "metrics": {"question_bert_f1": None, "back_table_mean_similarity": None},
    })
    # No bounded metrics -> no threshold drops; only status/status handled.
    report = run_filter(run_dir, config)
    assert report["filtered_out_threshold"] == 1  # only q7 missing quality row
    assert report["kept"] == 7
    assert "question_bert_f1" in report["avg"]


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown filter metric"):
        FilterConfig.model_validate({"metrics": {"question_bert_f1x": 0.9}})


def test_load_filter_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "filter.yaml"
    cfg_path.write_text(
        "status_include:\n  - accepted\nmetrics:\n  question_bert_f1: 0.85\n",
        encoding="utf-8",
    )
    cfg = load_filter_config(cfg_path)
    assert cfg.threshold_metrics == ("question_bert_f1",)
    assert cfg.effective_status_include[0].value == "accepted"


def test_unknown_metric_in_yaml_is_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("metrics:\n  bogus_field: 0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown filter metric"):
        load_filter_config(cfg_path)
