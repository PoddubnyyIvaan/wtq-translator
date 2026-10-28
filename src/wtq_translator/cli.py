from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from .alphabet import AlphabetIntentError
from .alphabet_screen import run_alphabet_screen
from .audit import audit_dataset_mentions
from .config import load_config
from .dataset import read_question_tsv, read_table_tsv, resolve_context_path
from .entities import remaining_tables, run_entities
from .mentions import find_table_mentions
from .persistence import StateStore
from .quality import load_filter_config
from .quality_stats import run_filter
from .reporting.stats import run_stats
from .runner import get_run_report, make_run_id, run_dataset_from_tsv, run_one_from_tsv


def _make_progress_bar(
    total: int,
    *,
    disabled: bool,
    desc: str = "questions",
    unit: str = "q",
) -> Any:
    """Создаёт прогресс-бар tqdm для run/resume/entities.

    Ленивый импорт: окружение без установленного tqdm (например, тесты) не падает,
    пока прогресс-бар реально не нужен. Вывод бара идёт в stderr — stdout
    остаётся чистым для финального JSON.
    """
    from tqdm import tqdm

    return tqdm(desc=desc, unit=unit, total=total, disable=disabled)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wtq-translate")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-config")
    v.add_argument("config")

    i = sub.add_parser("inspect")
    i.add_argument("config")
    i.add_argument("questions_tsv")
    i.add_argument("question_id")

    r = sub.add_parser("run-one")
    r.add_argument("config")
    r.add_argument("questions_tsv")
    r.add_argument("question_id")

    run = sub.add_parser("run")
    run.add_argument("config")
    run.add_argument("questions_tsv")
    run.add_argument("--run-id", default=None)
    run.add_argument("--fresh", action="store_true", help="Do not resume existing state; use a new run id unless --run-id is set")
    run.add_argument("--no-progress", action="store_true", help="Do not show the tqdm progress bar")

    resume = sub.add_parser("resume")
    resume.add_argument("config")
    resume.add_argument("questions_tsv")
    resume.add_argument("--run-id", default=None)
    resume.add_argument("--no-progress", action="store_true", help="Do not show the tqdm progress bar")

    rep = sub.add_parser("report")
    rep.add_argument("config")
    rep.add_argument("questions_tsv")
    rep.add_argument("--run-id", default=None)

    st = sub.add_parser("stats", help="Aggregate run statistics from quality.jsonl and state")
    st.add_argument("config")
    st.add_argument("questions_tsv")
    st.add_argument("--run-id", default=None)

    audit = sub.add_parser("audit", help="Audit exact table-mention coverage without LLM calls")
    audit.add_argument("config")
    audit.add_argument("questions_tsv")
    audit.add_argument("--output", default=None, help="Optional JSON path for the audit report")
    audit.add_argument("--max-examples", type=int, default=20)

    flt = sub.add_parser("filter", help="Filter a run's translated dataset by status + metric thresholds")
    flt.add_argument("config")
    flt.add_argument("run_dir")

    scr = sub.add_parser(
        "alphabet-screen",
        help="Screen (pre-filter) a question TSV by alphabet-dependent intent, without translating",
    )
    scr.add_argument("config")
    scr.add_argument("questions_tsv")
    scr.add_argument("--no-progress", action="store_true", help="Do not show the tqdm progress bar")

    e = sub.add_parser("entities", help="Collect named entities from table cells via spaCy NER")
    e.add_argument("config")
    e.add_argument("--input-tsv", default=None, help="Overrides entities.input_tsv from the config")
    e.add_argument("--no-progress", action="store_true", help="Do not show the tqdm progress bar")

    return p


def main() -> None:
    args = _parser().parse_args()

    # The `filter` subcommand uses its own standalone config schema, not AppConfig.
    if args.command == "filter":
        filter_config = load_filter_config(args.config)
        report = run_filter(args.run_dir, filter_config)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    config = load_config(args.config)

    if args.command == "validate-config":
        print(config.model_dump_json(indent=2))
        return

    if args.command == "inspect":
        records = read_question_tsv(args.questions_tsv)
        q = next((x for x in records if x.id == args.question_id), None)
        if q is None:
            raise SystemExit(f"Question not found: {args.question_id}")
        table_path = resolve_context_path(config.project.input_root, q.context)
        table = read_table_tsv(table_path)
        search = find_table_mentions(q.utterance, table, config.mentions)
        print(json.dumps({
            "question": q.utterance,
            "table": str(table_path),
            "mentions": [
                {
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
                for m in search.mentions
            ],
            "ambiguous": [
                {
                    "question_text": a.question_text,
                    "normalized_text": a.normalized_text,
                    "start": a.start,
                    "end": a.end,
                    "coordinates": [{"row": c.row, "column": c.column} for c in a.coordinates],
                    "candidates": list(a.candidates),
                }
                for a in search.ambiguous
            ],
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "run-one":
        try:
            result = asyncio.run(run_one_from_tsv(config, args.questions_tsv, args.question_id))
        except AlphabetIntentError as exc:
            print(json.dumps({
                "question_id": exc.question_id,
                "status": "excluded",
                "class": exc.intent_label,
                "why": exc.why,
            }, ensure_ascii=False, indent=2))
            return
        print(json.dumps({
            "question_id": result.question.id,
            "status": result.quality.status.value,
            "translated_question": result.translated_question,
            "translated_target_value": result.translated_target_value,
            "cosine_similarity": result.quality.question_cosine_similarity,
            "warnings": result.quality.warnings,
        }, ensure_ascii=False, indent=2))
        return

    if args.command in {"run", "resume"}:
        resume_mode = args.command == "resume" or not getattr(args, "fresh", False)
        run_id = args.run_id
        if args.command == "run" and getattr(args, "fresh", False) and run_id is None:
            run_id = f"{make_run_id(config, args.questions_tsv)}-{uuid.uuid4().hex[:8]}"
        # Прогресс-бар: total = число вопросов, disable = --no-progress или не tty.
        # Бар пишет в stderr и закрывается в finally ДО печати JSON в stdout.
        bar = _make_progress_bar(
            total=len(read_question_tsv(args.questions_tsv)),
            disabled=args.no_progress or not sys.stderr.isatty(),
        )
        try:
            summary = asyncio.run(run_dataset_from_tsv(
                config,
                args.questions_tsv,
                run_id=run_id,
                resume=resume_mode,
                progress=bar,
            ))
        finally:
            bar.close()
        print(json.dumps({
            "run_id": summary.run_id,
            "total": summary.total,
            "processed": summary.processed,
            "skipped": summary.skipped,
            "failed": summary.failed,
            "excluded": summary.excluded,
            "excluded_share": summary.excluded_share,
            "statuses": summary.statuses,
            "output_questions_tsv": str(summary.output_questions_tsv),
            "run_dir": str(summary.run_dir),
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "report":
        print(json.dumps(get_run_report(config, args.questions_tsv, run_id=args.run_id), ensure_ascii=False, indent=2))
        return

    if args.command == "stats":
        run_id = args.run_id or make_run_id(config, args.questions_tsv)
        state = StateStore(config.state.sqlite_path)
        status_counts = state.status_counts(run_id)
        total = sum(status_counts.values())
        if total == 0:
            raise SystemExit(f"no state for run: {run_id}")
        run_dir = config.project.output_root / "runs" / run_id
        extra = {}
        output_questions = run_dir / "dataset" / Path(args.questions_tsv).name
        alphabet_excluded = run_dir / "dataset" / f"{Path(args.questions_tsv).stem}-alphabet-excluded.tsv"
        if output_questions.exists():
            extra["output_questions_tsv"] = str(output_questions.resolve())
        if alphabet_excluded.exists():
            extra["alphabet_excluded_tsv"] = str(alphabet_excluded.resolve())
        print(json.dumps(
            run_stats(run_dir, status_counts=status_counts, total=total, extra=extra),
            ensure_ascii=False, indent=2,
        ))
        return

    if args.command == "audit":
        if args.max_examples < 0:
            raise SystemExit("--max-examples must be >= 0")
        report = audit_dataset_mentions(
            config,
            args.questions_tsv,
            max_examples=args.max_examples,
            output_path=args.output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "alphabet-screen":
        bar = _make_progress_bar(
            total=len(read_question_tsv(args.questions_tsv)),
            disabled=args.no_progress or not sys.stderr.isatty(),
        )
        try:
            stats = asyncio.run(run_alphabet_screen(config, args.questions_tsv, progress=bar))
        finally:
            bar.close()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.command == "entities":
        # Прогресс-бар: total = число оставшихся (необработанных) таблиц.
        bar = _make_progress_bar(
            total=len(remaining_tables(config, args.input_tsv)),
            disabled=args.no_progress or not sys.stderr.isatty(),
            desc="tables",
            unit="tables",
        )
        try:
            summary = run_entities(config, args.input_tsv, progress=bar)
        finally:
            bar.close()
        print(json.dumps({
            "total_tables": summary.total_tables,
            "skipped_processed": summary.skipped_processed,
            "missing_tables": summary.missing_tables,
            "new_entities": summary.new_entities,
            "updated_entities": summary.updated_entities,
            "output_tsv": str(summary.output_tsv),
            "processed_path": str(summary.processed_path),
        }, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
