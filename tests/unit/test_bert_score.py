import pytest

from wtq_translator.config import BERTScoreConfig, MentionNormalizationConfig
from wtq_translator.models import TableData
from wtq_translator.validation.bert_score import score_bertscore_pair, score_bertscore_table

NORM = MentionNormalizationConfig()


def _cfg(**overrides) -> BERTScoreConfig:
    base = dict(enabled=True)
    base.update(overrides)
    return BERTScoreConfig(**base)


class FakeScorer:
    """Имитирует выдачу _score_batch, не затрагивая вызов модели / torch."""

    def __init__(self, fn):
        self.fn = fn
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(self, cands, refs, cfg):
        self.calls.append((list(cands), list(refs)))
        return self.fn(cands, refs)


@pytest.fixture(autouse=True)
def _patch_batch(monkeypatch):
    import wtq_translator.validation.bert_score as mod

    fake = FakeScorer(
        lambda cands, refs: (
            [0.9] * len(cands),
            [0.8] * len(cands),
            [0.85] * len(cands),
        )
    )
    monkeypatch.setattr(mod, "_score_batch", fake)
    return fake


def test_config_validation_batch_size():
    with pytest.raises(ValueError):
        BERTScoreConfig(batch_size=0)


def test_config_validation_thresholds():
    with pytest.raises(ValueError):
        BERTScoreConfig(question_threshold=1.5)
    with pytest.raises(ValueError):
        BERTScoreConfig(table_threshold=-0.1)


@pytest.mark.asyncio
async def test_score_pair_returns_three_floats(_patch_batch):
    res = await score_bertscore_pair("what is x", "o chyom x", _cfg())
    assert set(res) == {"f1", "precision", "recall"}
    assert res == {"f1": 0.85, "precision": 0.9, "recall": 0.8}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["context", "row", "cell"])
async def test_identical_normalized_fastpath_all_modes(mode, _patch_batch):
    src = TableData(["Name", "City"], [["Alice", "London"]])
    back = TableData(["name", "city"], [["alice", "london"]])
    res = await score_bertscore_table(
        src, back, cfg=_cfg(), normalization=NORM, mode=mode
    )
    assert res["min_f1"] == 1.0
    assert res["mean_f1"] == 1.0
    assert res["min_precision"] == 1.0
    assert res["min_recall"] == 1.0
    # Fast-path: _score_batch не вызван.
    assert _patch_batch.calls == []


@pytest.mark.asyncio
async def test_aggregation_mixed(_patch_batch):
    # header fast-path (1.0,1.0,1.0), row выдаёт (0.0,1.0,0.0)
    src = TableData(["Name"], [["Alice"]])
    back = TableData(["name"], [["Bob"]])
    _patch_batch.fn = lambda cands, refs: ([0.0], [1.0], [0.0])
    res = await score_bertscore_table(
        src, back, cfg=_cfg(), normalization=NORM, mode="row"
    )
    assert res["groups_compared"] == 2
    assert res["min_f1"] == 0.0
    assert res["mean_f1"] == 0.5
    assert res["min_precision"] == 0.0
    assert res["mean_precision"] == 0.5
    # header fast-path даёт recall=1.0, row-пара (P=0.0,R=1.0,F1=0.0) => recall всегда 1.0
    assert res["min_recall"] == 1.0
    assert res["mean_recall"] == 1.0
    assert len(_patch_batch.calls) == 1


@pytest.mark.asyncio
async def test_columns_scoping(_patch_batch):
    src = TableData(["A", "QUARANTINE"], [["alice", "QUARANTINE1"], ["bob", "QUARANTINE2"]])
    back = TableData(["x", "y"], [["ZedX", "quarantine1"], ["ZedY", "quarantine2"]])
    _patch_batch.fn = lambda cands, refs: ([0.5] * len(cands), [0.5] * len(cands), [0.5] * len(cands))
    res = await score_bertscore_table(
        src, back, cfg=_cfg(), normalization=NORM, mode="cell", columns={0}
    )
    assert res["groups_compared"] == 3  # только колонка 0: header + 2 data-ячейки


@pytest.mark.asyncio
async def test_empty_result(_patch_batch):
    src = TableData(["A"], [["x"]])
    back = TableData(["y"], [["z"]])
    res = await score_bertscore_table(
        src, back, cfg=_cfg(), normalization=NORM, mode="row", columns=set()
    )
    assert res["groups_compared"] == 0
    assert res["min_f1"] is None
    assert _patch_batch.calls == []


@pytest.mark.asyncio
async def test_shape_mismatch_raises(_patch_batch):
    src = TableData(["A", "B"], [["x", "y"]])
    back = TableData(["A"], [["x"]])
    with pytest.raises(ValueError):
        await score_bertscore_table(src, back, cfg=_cfg(), normalization=NORM, mode="row")
    assert _patch_batch.calls == []
