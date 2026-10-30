"""NER-классификатор ячеек таблиц через spaCy (en_core_web_trf, GPU).

Запрос к модели — ``<cell> is <col>``; тип ячейки определяется по NER-спанам,
чей символьный диапазон целиком лежит в пределах ячейки. Спаны колонки
(``col``) и служебного ``is`` отбрасываются. Выравнивание по char-offset
устойчиво к пере-токенизации (``1,000``, ``1950s``, дефисы).
"""
from __future__ import annotations

from typing import Any, Sequence

# Тип-аннотации без верхнеуровневого импорта spacy (модуль импортируется и без spaCy).
_DEFAULT_MODEL = "en_core_web_trf"
_DEFAULT_ALLOWED_TYPES = ("PERSON", "ORG", "GPE", "LOC", "PRODUCT")


class EntityExtractor:
    """Батч-классификация ячеек: ``(type | None, all_types)`` на запрос.

    ``type`` — первый тип из ``allowed_types`` (в порядке списка), найденный среди
    NER-типов спанов ячейки; ``None`` если совпадений нет. ``all_types`` — все
    NER-типы внутри ячейки (включая не входящие в ``allowed_types``, напр. DATE).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        device: str = "auto",
        batch_size: int = 64,
        template: str = "{cell} is {col}",
        allowed_types: Sequence[str] = _DEFAULT_ALLOWED_TYPES,
    ) -> None:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError("spaCy не установлен. Установите: `pip install 'spacy>=3.7'`.") from exc
        if device == "gpu":
            try:
                spacy.require_gpu()
            except ValueError as exc:
                raise ValueError(
                    f"{exc}. Для GPU установите CuPy под вашу CUDA "
                    f"(например `pip install cupy-cuda12x`) или задайте `entities.device: auto`."
                ) from exc
        elif device == "auto":
            # Полагаемся на доступность CUDA: GPU если CuPy + устройство есть, иначе CPU.
            try:
                spacy.require_gpu()
            except (ImportError, ValueError):
                pass
        try:
            nlp = spacy.load(model)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"spaCy-модель '{model}' не найдена. "
                f"Скачайте: `python -m spacy download {model}`."
            ) from exc
        # Остаются transformer + ner; tagger/parser/attribute_ruler/lemmatizer не нужны.
        nlp.select_pipes(disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
        self._nlp = nlp
        self._template = template
        self._allowed_types = tuple(allowed_types)
        self._batch_size = batch_size

    # ------------------------------------------------------------------ helpers

    def _render(self, cell: str, col: str) -> str:
        return self._template.format(cell=cell, col=col)

    def _classify_doc(self, doc: Any, cell: str, col: str) -> tuple[str | None, frozenset[str], list[tuple[str, str, str]]]:
        """Разбор одного doc: спаны ячейки (по char-offset), суммарные типы, тип по allowed."""
        cell_len = len(cell)
        col_start = len(doc.text) - len(col)  # колонка — всегда хвост текста запроса
        cell_ents: list[tuple[str, str, str]] = []
        cell_types: set[str] = set()
        for ent in doc.ents:
            if ent.end_char <= cell_len:
                cell_ents.append((ent.text, ent.label_, f"{ent.start_char}:{ent.end_char}"))
                cell_types.add(ent.label_)
        ent_type = next((t for t in self._allowed_types if t in cell_types), None)
        return ent_type, frozenset(cell_types), cell_ents

    # ------------------------------------------------------------- public API

    def classify(self, cell: str, col: str) -> tuple[str | None, frozenset[str], list[tuple[str, str, str]]]:
        """Одиночный запрос (для тестов); боевой путь — ``classify_many``."""
        return self._classify_doc(self._nlp(self._render(cell, col)), cell, col)

    def classify_many(self, items: list[tuple[str, str]]) -> list[tuple[str | None, frozenset[str]]]:
        """Батч запросов ``(cell, col)`` с length bucketing (паддинг к длине батча)."""
        texts = [self._render(cell, col) for cell, col in items]
        # Стабильная сортировка по длине: короткие запросы паддятся к более коротким батчам.
        order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), i))
        results: list[tuple[str | None, frozenset[str]]] = [None] * len(items)  # type: ignore[list-item]
        for pos, doc in zip(order, self._nlp.pipe([texts[i] for i in order], batch_size=self._batch_size)):
            cell, col = items[pos]
            ent_type, cell_types, _ = self._classify_doc(doc, cell, col)
            results[pos] = (ent_type, cell_types)
        return results