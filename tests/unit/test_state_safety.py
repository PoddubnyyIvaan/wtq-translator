from pathlib import Path

import pytest

from wtq_translator.persistence.state import StateStore


def test_run_id_cannot_be_reused_with_different_config(tmp_path: Path):
    store = StateStore(tmp_path / "state.sqlite")
    q = tmp_path / "train.tsv"
    q.write_text("id\tutterance\tcontext\ttargetValue\n", encoding="utf-8")
    store.start_run(
        run_id="run-1",
        questions_tsv=str(q.resolve()),
        output_questions_tsv=str((tmp_path / "out1.tsv").resolve()),
        config_hash="aaa",
    )
    with pytest.raises(ValueError, match="different configuration"):
        store.start_run(
            run_id="run-1",
            questions_tsv=str(q.resolve()),
            output_questions_tsv=str((tmp_path / "out2.tsv").resolve()),
            config_hash="bbb",
        )
