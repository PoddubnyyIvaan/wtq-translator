from __future__ import annotations

import asyncio
import threading
from typing import Any, Literal

from ..config import BERTScoreConfig, MentionNormalizationConfig
from ..mentions.normalize import normalize_for_match
from ..models import TableData
from .structure import validate_table_shape
from .table_quality import build_comparison_units

# Синглтон BERTScorer на процесс. Именно один объект: каждый BERTScorer грузит
# веса модели в CUDA (AutoModel.from_pretrained + .to(device)), а PyTorch CUDA
# caching allocator не возвращает память драйверу — лишние копии навсегда
# остаются зарезервированными на видеокарте.
_GET_SCORER_LOCK = threading.Lock()
_SCORER_CACHE: dict[tuple[str, str, str, bool], Any] = {}


def _get_scorer(
    model_type: str,
    lang: str,
    device: str,
    rescale_with_baseline: bool,
) -> Any:
    """Единственный BERTScorer на процесс (ключ — параметры конфигурации).

    Двойная проверка под threading.Lock: в отличие от functools.lru_cache (который
    потокобезопасен только для книги кэша и не сериализует первое вычисление),
    здесь конкурентные потоки asyncio.to_thread не строят одновременно несколько
    копий модели на GPU.
    """
    key = (model_type, lang, device, rescale_with_baseline)
    scorer = _SCORER_CACHE.get(key)
    if scorer is not None:
        return scorer
    with _GET_SCORER_LOCK:
        scorer = _SCORER_CACHE.get(key)
        if scorer is not None:
            return scorer
        from bert_score import BERTScorer

        scorer = BERTScorer(
            model_type=model_type,
            lang=lang,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
        )
        _SCORER_CACHE[key] = scorer
        return scorer


def prewarm_bertscore(cfg: BERTScoreConfig) -> Any:
    """Подогреть BERTScore-синглтон до старта воркеров (один раз на процесс).

    Вызов из Translator.__init__: к моменту запуска пула задач кэш уже заполнен,
    гонка первого использования между потоками asyncio.to_thread отсутствует полностью.
    """
    return _get_scorer(
        cfg.model_type,
        cfg.lang,
        cfg.device,
        cfg.rescale_with_baseline,
    )


def _floats(values: Any) -> list[float]:
    """Конвертировать тензор/список в плоский список float."""
    out = values.tolist() if hasattr(values, "tolist") else list(values)
    return [float(v) for v in out]


def _score_batch(
    cands: list[str],
    refs: list[str],
    cfg: BERTScoreConfig,
) -> tuple[list[float], list[float], list[float]]:
    """Синхронный вызов BERTScore: возвращает (precision, recall, f1) по элементам."""
    scorer = _get_scorer(
        cfg.model_type,
        cfg.lang,
        cfg.device,
        cfg.rescale_with_baseline,
    )
    precision, recall, f1 = scorer.score(cands, refs, batch_size=cfg.batch_size)
    return _floats(precision), _floats(recall), _floats(f1)


async def score_bertscore_pair(
    src: str,
    back: str,
    cfg: BERTScoreConfig,
) -> dict[str, float]:
    """Оценить одну пару (src, back) BERTScore-ом через to_thread."""
    precision, recall, f1 = await asyncio.to_thread(_score_batch, [src], [back], cfg)
    return {
        "f1": f1[0],
        "precision": precision[0],
        "recall": recall[0],
    }


async def score_bertscore_pairs(
    cands: list[str],
    refs: list[str],
    cfg: BERTScoreConfig,
) -> dict[str, list[float]]:
    """Батч BERTScore для списков пар (cands, refs); возвращает списки f1/precision/recall."""
    if not cands:
        return {"f1": [], "precision": [], "recall": []}
    precision, recall, f1 = await asyncio.to_thread(_score_batch, cands, refs, cfg)
    return {"f1": f1, "precision": precision, "recall": recall}


def _aggr(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None}
    return {"min": min(values), "mean": sum(values) / len(values)}


async def score_bertscore_table(
    source: TableData,
    back: TableData,
    *,
    cfg: BERTScoreConfig,
    normalization: MentionNormalizationConfig,
    mode: Literal["context", "row", "cell"] = "context",
    max_chunk_tokens: int = 512,
    columns: set[int] | None = None,
) -> dict[str, Any]:
    """Оценить round-trip таблицы BERTScore-ом с per-unit агрегацией.

    Единицы сравнения те же, что и для косинусного скоринга (build_comparison_units).
    Нормализованно совпавшие единицы дают полный балл (1.0, 1.0, 1.0) без вызова модели.
    """
    validate_table_shape(source, back)

    selected = set(range(len(source.headers))) if columns is None else set(columns)
    if not selected:
        return {
            "groups_compared": 0,
            "min_f1": None,
            "mean_f1": None,
            "min_precision": None,
            "mean_precision": None,
            "min_recall": None,
            "mean_recall": None,
        }

    units = build_comparison_units(source, back, selected, mode, max_chunk_tokens)
    total = len(units)
    if total == 0:
        return {
            "groups_compared": 0,
            "min_f1": None,
            "mean_f1": None,
            "min_precision": None,
            "mean_precision": None,
            "min_recall": None,
            "mean_recall": None,
        }

    f1: list[float] = []
    precision: list[float] = []
    recall: list[float] = []
    cands: list[str] = []
    refs: list[str] = []
    pending_index: list[int] = []

    for i, unit in enumerate(units):
        if normalize_for_match(unit["src"], normalization) == normalize_for_match(unit["back"], normalization):
            f1.append(1.0)
            precision.append(1.0)
            recall.append(1.0)
            continue
        pending_index.append(i)
        cands.append(unit["src"])
        refs.append(unit["back"])

    if pending_index:
        p, r, f = await asyncio.to_thread(_score_batch, cands, refs, cfg)
        # Расширить массивы до полной длины единиц, сохранив fast-path 1.0.
        extended_f1: list[float] = [1.0] * total
        extended_precision: list[float] = [1.0] * total
        extended_recall: list[float] = [1.0] * total
        for k, i in enumerate(pending_index):
            extended_f1[i] = f[k]
            extended_precision[i] = p[k]
            extended_recall[i] = r[k]
        f1, precision, recall = extended_f1, extended_precision, extended_recall

    ag_f1 = _aggr(f1)
    ag_p = _aggr(precision)
    ag_r = _aggr(recall)

    return {
        "groups_compared": total,
        "min_f1": ag_f1["min"],
        "mean_f1": ag_f1["mean"],
        "min_precision": ag_p["min"],
        "mean_precision": ag_p["mean"],
        "min_recall": ag_r["min"],
        "mean_recall": ag_r["mean"],
    }
