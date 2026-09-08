import csv
from pathlib import Path

from wtq_translator.audit import audit_dataset_mentions
from wtq_translator.config import AppConfig


def _cfg(tmp_path: Path, mode: str = "mentions_only") -> AppConfig:
    return AppConfig.model_validate({
        "project": {
            "input_root": str(tmp_path / "input"),
            "output_root": str(tmp_path / "output"),
            "source_language": "en",
            "target_language": "ru",
        },
        "translation": {"mode": mode},
        "back_translation": {"enabled": True},
        "cache": {"enabled": False},
    })


class _Morph:
    def __init__(self, poss=None):
        self._poss = poss

    def get(self, key):
        if key == "Poss" and self._poss is not None:
            return [self._poss]
        return []


class _Tok:
    def __init__(self, text, idx, *, lemma=None, pos="", num=False, date=False, punct=False):
        self.text = text
        self.idx = idx
        self.lemma_ = text.lower() if lemma is None else lemma
        self.pos_ = pos
        self.like_num = num
        self.ent_type_ = "DATE" if date else ""
        self.is_punct = punct
        self.is_space = False
        self.morph = _Morph()


def _make_nlp(mapping):
    def nlp(text):
        return list(mapping[text])
    return nlp


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


def test_audit_full_mode_uses_lemma_matcher(tmp_path, monkeypatch):
    root = tmp_path / "input"
    (root / "tables").mkdir(parents=True)
    (root / "tables" / "one.tsv").write_text("Name\tCity\nAlice\tLondon\n", encoding="utf-8")

    qpath = root / "train.tsv"
    with qpath.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "utterance", "context", "targetValue"])
        w.writerow(["q1", "Where is Alice?", "tables/one.tsv", "London"])
        w.writerow(["q2", "Show totals", "tables/one.tsv", "London"])

    nlp = _make_nlp({
        "Where is Alice?": [
            _Tok("Where", 0, lemma="where", pos="ADV"),
            _Tok("is", 6, lemma="be", pos="AUX"),
            _Tok("Alice", 9, lemma="alice", pos="PROPN"),
        ],
        "Name": [_Tok("Name", 0, lemma="name", pos="NOUN")],
        "City": [_Tok("City", 0, lemma="city", pos="NOUN")],
        "Alice": [_Tok("Alice", 0, lemma="alice", pos="PROPN")],
        "London": [_Tok("London", 0, lemma="london", pos="PROPN")],
        "Show totals": [
            _Tok("Show", 0, lemma="show", pos="VERB"),
            _Tok("totals", 5, lemma="total", pos="NOUN"),
        ],
    })
    monkeypatch.setattr("wtq_translator.audit.load_spacy_model", lambda name: nlp)

    report = audit_dataset_mentions(_cfg(tmp_path, "full"), qpath)
    assert report["mode"] == "full"
    assert report["matcher"] == "lemma"
    assert report["questions_with_mentions"] == 1
    assert report["questions_without_mentions"] == 1
