"""Кандидаты-подпоследовательности для full-матчера (лемматизация через spaCy)."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..config import MentionsConfig
from ..models import (
    CandidateSet,
    ContentRun,
    QuestionCandidate,
    TableCandidate,
    TableCoordinate,
    TableData,
)
from ..validation.invariants import is_immutable_value

_NON_ALNUM_RE = re.compile(r"[\W_]")  # удалить не-алфавитно-цифровое (Unicode-корректно, кириллица сохраняется)

# Ленивый кэш загруженных spaCy-моделей на процесс (синглтон по имени модели).
_MODEL_CACHE: dict[str, Any] = {}

# In-memory кэш кандидатов таблиц на процесс.
_TABLE_CACHE: dict[str, CandidateSet] = {}


def load_spacy_model(model_name: str) -> Any:
    """Лениво загружает spaCy-модель: кэш по имени, понятные ошибки установки/загрузки."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        import spacy
        from spacy.language import Language
        from spacy.matcher import Matcher
        # нет верхнеуровневого импорта: модуль импортируется и без spaCy

    except ImportError as exc:
        raise ImportError("spaCy не установлен. Установите: `pip install 'spacy>=3.7'`.") from exc
    try:
        nlp = spacy.load(model_name)
        matcher = Matcher(nlp.vocab)
        # IS_ALPHA означает, что токен состоит только из букв (не цифры, не знак)
        pattern = [{"IS_ALPHA": True}, {"TEXT": "-"}, {"IS_ALPHA": True}]
        matcher.add("HYPHENATED_WORD", [pattern])

        @Language.component("merge_hyphens")
        def merge_hyphens(doc):
            # Находим все совпадения
            matches = matcher(doc)

            # Собираем спаны (фрагменты текста), которые нужно склеить
            spans = [doc[start:end] for _, start, end in matches]

            # ИСПРАВЛЕННАЯ СТРОКА: используем doc.retokenize()
            with doc.retokenize() as retokenizer:
                for span in spans:
                    # Склеиваем токены. spaCy сам корректно определит атрибуты (POS, LEMMA)
                    # на основе головного слова (root) спана.
                    retokenizer.merge(span)

            return doc

        if "merge_hyphens" not in nlp.pipe_names:
            nlp.add_pipe("merge_hyphens", first=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"spaCy-модель '{model_name}' не найдена. "
            f"Скачайте: `python -m spacy download {model_name}`."
        ) from exc
    _MODEL_CACHE[model_name] = nlp
    return nlp


def _classify(tok: Any) -> str:
    """Классификация токена: punct / number / date / word (порядок проверок важен)."""
    if tok.is_punct or tok.is_space:
        return "punct"
    if tok.like_num or tok.pos_ == "NUM":
        return "number"
    if tok.ent_type_ == "DATE":
        return "date"
    return "word"


def _token_lemma_key(tok: Any, kind: str) -> str:
    """Лемма-ключ токена: word — нормализованная лемма, number — поверхностная форма,
    date — поверхностная форма в нижнем регистре (не-числа приводятся к lower case)."""
    if kind == "number":
        return tok.text
    if kind == "word":
        return _NON_ALNUM_RE.sub("", tok.lemma_.casefold())
    return tok.text.casefold()


def _lemma_key(tokens: list[Any]) -> str:
    return " ".join(_token_lemma_key(t, _classify(t)) for t in tokens)


def _content_runs(doc: Any) -> list[list[Any]]:
    """Максимальные непрерывные run'ы контентных токенов (всё, кроме punct)."""
    runs: list[list[Any]] = []
    current: list[Any] = []
    for tok in doc:
        if _classify(tok) == "punct":
            if current:
                runs.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        runs.append(current)
    return runs


def _is_question_stop(tok: Any, cfg: MentionsConfig) -> bool:
    return tok.pos_ in cfg.stop_pos or tok.morph.get("Poss") == ["Yes"]


def _subsequences(
    run: list[Any],
    text: str,
    cfg: MentionsConfig,
    *,
    drop_stop: bool,
) -> list[QuestionCandidate]:
    """Все непрерывные подпоследовательности одного content-run'а с фильтрацией одиночных."""
    result: list[QuestionCandidate] = []
    n = len(run)
    for i in range(n):
        for j in range(i, n):
            length = j - i + 1
            if length == 1:
                tok = run[i]
                kind = _classify(tok)
                if drop_stop and _is_question_stop(tok, cfg):
                    continue
                if cfg.exclude_numbers and kind == "number":
                    continue
                if cfg.exclude_dates and kind == "date":
                    continue
            start = run[i].idx
            end = run[j].idx + len(run[j].text)
            result.append(QuestionCandidate(
                lemma_key=_lemma_key(run[i:j + 1]),
                source_text=text[start:end],
                start=start,
                end=end,
                length=length,
            ))
    return result


def extract_question_candidates(question: str, nlp: Any, cfg: MentionsConfig) -> list[QuestionCandidate]:
    """Все подпоследовательности вопроса; стоп-слова/числа/даты отбрасываются только из одиночных."""
    doc = nlp(question)
    result: list[QuestionCandidate] = []
    for run in _content_runs(doc):
        result.extend(_subsequences(run, question, cfg, drop_stop=True))
    return result


def _candidate_id(lemma_key: str, source_text: str, target_lang: str) -> str:
    return hashlib.sha256(f"{lemma_key}\x00{source_text}\x00{target_lang}".encode()).hexdigest()[:16]


def _table_cache_key(table: TableData, target_lang: str) -> str:
    payload = json.dumps([table.headers, table.rows], ensure_ascii=False) + target_lang
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_table_candidates(
    table: TableData,
    nlp: Any,
    cfg: MentionsConfig,
    target_lang: str,
) -> CandidateSet:
    """Кандидаты таблицы: дедуп по лемма-ключу с объединением координат + per-координата content-run'ы."""
    key = _table_cache_key(table, target_lang)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    result = CandidateSet()
    by_lemma: dict[str, TableCandidate] = {}
    order: list[str] = []

    cells: list[tuple[TableCoordinate, str]] = [
        (TableCoordinate(None, col), text) for col, text in enumerate(table.headers)
    ] + [
        (TableCoordinate(row_idx, col_idx), text)
        for row_idx, row in enumerate(table.rows)
        for col_idx, text in enumerate(row)
    ]

    for coord, text in cells:
        # Immutable-значения (чистые числа/URL/ID/пусто) — без кандидатов вовсе.
        if is_immutable_value(text):
            continue
        doc = nlp(text)
        # Ячейка обрабатывается, только если в ней есть word-токен либо разрешённая дата
        # (даты порождают кандидатов лишь при exclude_dates=False).
        if not any(
            _classify(t) == "word" or (_classify(t) == "date" and not cfg.exclude_dates)
            for t in doc
        ):
            continue
        runs = _content_runs(doc)
        result.runs[coord] = [
            ContentRun(
                lemma_key=_lemma_key(run),
                start=run[0].idx,
                end=run[-1].idx + len(run[-1].text),
            )
            for run in runs
        ]
        for run in runs:
            for qc in _subsequences(run, text, cfg, drop_stop=False):
                if qc.lemma_key not in by_lemma:
                    by_lemma[qc.lemma_key] = TableCandidate(
                        id=_candidate_id(qc.lemma_key, qc.source_text, target_lang),
                        lemma_key=qc.lemma_key,
                        source_text=qc.source_text,
                        coordinates=[coord],
                        length=qc.length,
                    )
                    order.append(qc.lemma_key)
                else:
                    by_lemma[qc.lemma_key].coordinates.append(coord)

    result.candidates = [by_lemma[k] for k in order]
    _TABLE_CACHE[key] = result
    return result
