from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..enums import ProcessingStatus


@dataclass(frozen=True)
class QuestionState:
    run_id: str
    question_id: str
    status: ProcessingStatus
    mode: str
    output_table_path: str | None
    output_record: dict[str, Any] | None
    error: str | None
    updated_at: str


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self):
        # timeout=30 задаёт busy timeout каждого нового соединения (30 с),
        # чтобы конкурентные записи не падали с "database is locked".
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _columns(self, db, table: str) -> set[str]:
        return {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}

    def _init(self) -> None:
        with self.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    questions_tsv TEXT NOT NULL,
                    output_questions_tsv TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    run_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    output_table_path TEXT,
                    output_record_json TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, question_id)
                )
            """)
            # Migration from v0.2 state DB.
            cols = self._columns(db, "questions")
            if "output_record_json" not in cols:
                db.execute("ALTER TABLE questions ADD COLUMN output_record_json TEXT")
            db.execute("""
                CREATE TABLE IF NOT EXISTS quality (
                    run_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, question_id)
                )
            """)
            # WAL — persistent режим журнала файла БД (нужен для конкурентных читателей/писателей),
            # busy_timeout — страховка для init-соединения; остальные соединения получают 30 с через timeout.
            db.execute("PRAGMA journal_mode=WAL").fetchone()
            db.execute("PRAGMA busy_timeout=30000")

    def start_run(self, *, run_id: str, questions_tsv: str, output_questions_tsv: str, config_hash: str) -> None:
        """Create or resume a run without allowing identity drift.

        Reusing a run id with another config or source TSV would make old accepted
        question states unsafe to skip, so this is treated as a hard error.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            existing = db.execute(
                "SELECT questions_tsv, config_hash FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                if existing["config_hash"] != config_hash:
                    raise ValueError(
                        f"Run id {run_id!r} already exists with a different configuration"
                    )
                if Path(existing["questions_tsv"]).resolve() != Path(questions_tsv).resolve():
                    raise ValueError(
                        f"Run id {run_id!r} already exists for a different questions TSV"
                    )
                db.execute(
                    "UPDATE runs SET output_questions_tsv=?, status='processing', updated_at=? WHERE run_id=?",
                    (output_questions_tsv, now, run_id),
                )
                return

            db.execute(
                """
                INSERT INTO runs(run_id,questions_tsv,output_questions_tsv,config_hash,status,started_at,updated_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                (run_id, questions_tsv, output_questions_tsv, config_hash, "processing", now, now),
            )

    def finish_run(self, run_id: str, status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?, updated_at=? WHERE run_id=?", (status, now, run_id))

    def reset_processing(self, run_id: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            cur = db.execute(
                "UPDATE questions SET status=?, error=?, updated_at=? WHERE run_id=? AND status=?",
                (ProcessingStatus.PENDING.value, "Recovered stale processing state after interruption", now, run_id, ProcessingStatus.PROCESSING.value),
            )
            return cur.rowcount

    def set_question(
        self,
        run_id: str,
        question_id: str,
        status: ProcessingStatus,
        mode: str,
        *,
        output_table_path: str | None = None,
        output_record: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        record_json = json.dumps(output_record, ensure_ascii=False) if output_record is not None else None
        with self.connect() as db:
            db.execute("""
                INSERT INTO questions(run_id, question_id, status, mode, output_table_path, output_record_json, error, updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id, question_id) DO UPDATE SET
                    status=excluded.status, mode=excluded.mode,
                    output_table_path=COALESCE(excluded.output_table_path, questions.output_table_path),
                    output_record_json=COALESCE(excluded.output_record_json, questions.output_record_json),
                    error=excluded.error, updated_at=excluded.updated_at
            """, (run_id, question_id, status.value, mode, output_table_path, record_json, error, now))

    def get_status(self, run_id: str, question_id: str) -> ProcessingStatus | None:
        state = self.get_question(run_id, question_id)
        return state.status if state else None

    def get_question(self, run_id: str, question_id: str) -> QuestionState | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM questions WHERE run_id=? AND question_id=?", (run_id, question_id)).fetchone()
        if not row:
            return None
        return QuestionState(
            run_id=row["run_id"],
            question_id=row["question_id"],
            status=ProcessingStatus(row["status"]),
            mode=row["mode"],
            output_table_path=row["output_table_path"],
            output_record=json.loads(row["output_record_json"]) if row["output_record_json"] else None,
            error=row["error"],
            updated_at=row["updated_at"],
        )

    def list_questions(self, run_id: str) -> list[QuestionState]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM questions WHERE run_id=? ORDER BY updated_at, question_id", (run_id,)).fetchall()
        return [
            QuestionState(
                run_id=r["run_id"], question_id=r["question_id"], status=ProcessingStatus(r["status"]),
                mode=r["mode"], output_table_path=r["output_table_path"],
                output_record=json.loads(r["output_record_json"]) if r["output_record_json"] else None,
                error=r["error"], updated_at=r["updated_at"],
            ) for r in rows
        ]

    def save_quality(self, run_id: str, question_id: str, report: dict) -> None:
        with self.connect() as db:
            db.execute("""
                INSERT INTO quality(run_id, question_id, report_json) VALUES(?,?,?)
                ON CONFLICT(run_id, question_id) DO UPDATE SET report_json=excluded.report_json
            """, (run_id, question_id, json.dumps(report, ensure_ascii=False)))

    def load_quality(self, run_id: str, question_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT report_json FROM quality WHERE run_id=? AND question_id=?", (run_id, question_id)).fetchone()
        return json.loads(row[0]) if row else None

    def status_counts(self, run_id: str) -> dict[str, int]:
        with self.connect() as db:
            rows = db.execute("SELECT status, COUNT(*) AS n FROM questions WHERE run_id=? GROUP BY status", (run_id,)).fetchall()
        return {r["status"]: r["n"] for r in rows}
