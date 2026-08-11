import csv
from pathlib import Path

from wtq_translator.audit import audit_dataset_mentions
from wtq_translator.config import AppConfig


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate({
        "project": {
            "input_root": str(tmp_path / "input"),
            "output_root": str(tmp_path / "output"),
            "source_language": "en",
            "target_language": "ru",
        },
        "back_translation": {"enabled": True},
        "cache": {"enabled": False},
    })


def test_audit_reports_mention_coverage_and_ambiguity(tmp_path):
    root = tmp_path / "input"
    (root / "tables").mkdir(parents=True)
    (root / "tables" / "one.tsv").write_text("Name\tCity\nAlice\tLondon\n", encoding="utf-8")
    (root / "tables" / "amb.tsv").write_text("A\tB\nX\tX\n", encoding="utf-8")

    qpath = root / "train.tsv"
    with qpath.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "utterance", "context", "targetValue"])
        w.writerow(["q1", "Where is Alice?", "tables/one.tsv", "London"])
        w.writerow(["q2", "What happened?", "tables/one.tsv", "London"])
        w.writerow(["q3", "Was X selected?", "tables/amb.tsv", "X"])

    out = tmp_path / "audit.json"
    report = audit_dataset_mentions(_cfg(tmp_path), qpath, output_path=out)
    assert report["total_questions"] == 3
    assert report["questions_with_mentions"] == 1
    assert report["questions_without_mentions"] == 2
    assert report["questions_with_ambiguity"] == 1
    assert report["mention_coverage"] == 1 / 3
    assert out.exists()
