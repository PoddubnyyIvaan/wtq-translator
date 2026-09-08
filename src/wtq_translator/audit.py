from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .config import AppConfig
from .dataset import read_question_tsv_document, read_table_tsv, resolve_context_path
from .enums import TranslationMode
from .mentions import find_table_mentions
from .mentions.candidates import load_spacy_model
from .mentions.finder_full import find_table_mentions_full
from .models import TableData


def _mention_to_dict(m) -> dict[str, Any]:
    return {
        "mask": m.mask,
        "question_text": m.question_text,
        "normalized_text": m.normalized_text,
        "start": m.start,
        "end": m.end,
        "source_type": m.source_type.value,
        "coordinates": [{"row": c.row, "column": c.column} for c in m.coordinates],
        "table_source_text": m.table_source_text,
        "match_type": m.match_type,
    }


def _ambiguity_to_dict(a) -> dict[str, Any]:
    return {
        "question_text": a.question_text,
        "normalized_text": a.normalized_text,
        "start": a.start,
        "end": a.end,
        "coordinates": [{"row": c.row, "column": c.column} for c in a.coordinates],
        "candidates": list(a.candidates),
    }


def audit_dataset_mentions(
    config: AppConfig,
    questions_tsv: str | Path,
    *,
    max_examples: int = 20,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit deterministic table-mention coverage without any LLM calls.

    This should be run before an expensive translation job. It measures how much of
    the dataset can be linked to tables. The matcher is selected by
    ``config.translation.mode``: ``full`` uses the lemma matcher (requires spaCy),
    while ``shell_only``/``mentions_only`` use the exact-normalized matcher.
    """
    qpath = Path(questions_tsv)
    document = read_question_tsv_document(qpath)

    use_full_matcher = config.translation.mode == TranslationMode.FULL
    nlp = load_spacy_model(config.mentions.spacy_model) if use_full_matcher else None

    # Many WTQ questions share a source table. Cache parsed tables by resolved path to
    # avoid re-reading the same TSV hundreds of times during an audit.
    table_cache: dict[Path, TableData] = {}

    with_mentions = 0
    without_mentions = 0
    with_ambiguity = 0
    with_repeated_same_column = 0
    total_mentions = 0
    total_ambiguous_spans = 0
    source_types: Counter[str] = Counter()
    mention_count_hist: Counter[int] = Counter()
    selected_column_count_hist: Counter[int] = Counter()
    table_error_count = 0

    no_mention_examples: list[dict[str, Any]] = []
    ambiguity_examples: list[dict[str, Any]] = []
    repeated_examples: list[dict[str, Any]] = []
    table_errors: list[dict[str, Any]] = []

    for q in document.records:
        try:
            table_path = resolve_context_path(config.project.input_root, q.context).resolve()
            table = table_cache.get(table_path)
            if table is None:
                table = read_table_tsv(table_path)
                table_cache[table_path] = table
            if use_full_matcher:
                search, _ = find_table_mentions_full(
                    q.utterance, table, config.mentions, nlp, config.project.target_language
                )
            else:
                search = find_table_mentions(q.utterance, table, config.mentions)
        except Exception as exc:
            table_error_count += 1
            if len(table_errors) < max_examples:
                table_errors.append({
                    "question_id": q.id,
                    "context": q.context,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            continue

        n = len(search.mentions)
        total_mentions += n
        mention_count_hist[n] += 1
        columns = {m.coordinate.column for m in search.mentions}
        selected_column_count_hist[len(columns)] += 1
        source_types.update(m.source_type.value for m in search.mentions)

        if n:
            with_mentions += 1
        else:
            without_mentions += 1
            if len(no_mention_examples) < max_examples:
                no_mention_examples.append({
                    "question_id": q.id,
                    "question": q.utterance,
                    "context": q.context,
                })

        if search.ambiguous:
            with_ambiguity += 1
            total_ambiguous_spans += len(search.ambiguous)
            if len(ambiguity_examples) < max_examples:
                ambiguity_examples.append({
                    "question_id": q.id,
                    "question": q.utterance,
                    "context": q.context,
                    "ambiguous": [_ambiguity_to_dict(a) for a in search.ambiguous],
                })

        repeated = [m for m in search.mentions if len(m.coordinates) > 1]
        if repeated:
            with_repeated_same_column += 1
            if len(repeated_examples) < max_examples:
                repeated_examples.append({
                    "question_id": q.id,
                    "question": q.utterance,
                    "context": q.context,
                    "mentions": [_mention_to_dict(m) for m in repeated],
                })

    successfully_audited = len(document.records) - table_error_count
    summary: dict[str, Any] = {
        "questions_tsv": str(qpath.resolve()),
        "mode": config.translation.mode.value,
        "matcher": "lemma" if use_full_matcher else "exact_normalized",
        "total_questions": len(document.records),
        "successfully_audited": successfully_audited,
        "table_errors": table_error_count,
        "questions_with_mentions": with_mentions,
        "questions_without_mentions": without_mentions,
        "questions_with_ambiguity": with_ambiguity,
        "questions_with_repeated_same_column_mentions": with_repeated_same_column,
        "total_mentions": total_mentions,
        "total_ambiguous_spans": total_ambiguous_spans,
        "mention_coverage": (with_mentions / successfully_audited) if successfully_audited else 0.0,
        "ambiguity_rate": (with_ambiguity / successfully_audited) if successfully_audited else 0.0,
        "mention_source_types": dict(sorted(source_types.items())),
        "mention_count_histogram": {str(k): v for k, v in sorted(mention_count_hist.items())},
        "selected_column_count_histogram": {str(k): v for k, v in sorted(selected_column_count_hist.items())},
        "examples": {
            "no_mentions": no_mention_examples,
            "ambiguous": ambiguity_examples,
            "repeated_same_column": repeated_examples,
            "table_errors": table_errors,
        },
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)

    return summary
