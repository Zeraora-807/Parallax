"""Transactional event store used by Ganymede for audit and crash recovery."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


class SQLiteEventStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    generation_id TEXT,
                    policy_version TEXT NOT NULL,
                    config_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    call_id TEXT,
                    attempt_id TEXT,
                    reason_code TEXT,
                    policy_version TEXT,
                    data_json TEXT NOT NULL,
                    UNIQUE(run_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS events_run_sequence
                    ON events(run_id, sequence);
                CREATE TRIGGER IF NOT EXISTS events_no_update
                BEFORE UPDATE ON events BEGIN
                    SELECT RAISE(ABORT, 'events are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS events_no_delete
                BEFORE DELETE ON events BEGIN
                    SELECT RAISE(ABORT, 'events are append-only');
                END;
                """
            )

    def create_run(
        self,
        run_id: str,
        *,
        generation_id: str | None,
        policy_version: str,
        config: Mapping[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO runs
                   (run_id, created_at, status, generation_id, policy_version, config_json)
                   VALUES (?, ?, 'running', ?, ?, ?)""",
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    generation_id,
                    policy_version,
                    json.dumps(dict(config), ensure_ascii=False, default=str),
                ),
            )

    def finish_run(self, run_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status=?, finished_at=? WHERE run_id=?",
                (status, datetime.now(timezone.utc).isoformat(), run_id),
            )

    def record(
        self,
        run_id: str,
        event: str,
        status: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(data or {})
        trace_id = str(payload.pop("trace_id", run_id))
        call_id = _optional_string(payload.get("call_id"))
        attempt_id = _optional_string(payload.get("attempt_id"))
        reason_code = _optional_string(payload.get("reason_code"))
        policy_version = _optional_string(payload.get("policy_version"))
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
            connection.execute(
                """INSERT INTO events
                   (run_id, sequence, timestamp, event_type, status, trace_id,
                    call_id, attempt_id, reason_code, policy_version, data_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    sequence,
                    datetime.now(timezone.utc).isoformat(),
                    event,
                    status,
                    trace_id,
                    call_id,
                    attempt_id,
                    reason_code,
                    policy_version,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT event_id, sequence, timestamp, event_type, status, trace_id,
                          call_id, attempt_id, reason_code, policy_version, data_json
                   FROM events WHERE run_id=? ORDER BY sequence""",
                (run_id,),
            ).fetchall()
        return [
            {
                "event_id": row[0],
                "run_id": run_id,
                "sequence": row[1],
                "timestamp": row[2],
                "event": row[3],
                "status": row[4],
                "trace_id": row[5],
                "call_id": row[6],
                "attempt_id": row[7],
                "reason_code": row[8],
                "policy_version": row[9],
                "data": json.loads(row[10]),
            }
            for row in rows
        ]

    def run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT run_id, created_at, finished_at, status, generation_id,
                          policy_version, config_json
                   FROM runs WHERE run_id=?""",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "created_at": row[1],
            "finished_at": row[2],
            "status": row[3],
            "generation_id": row[4],
            "policy_version": row[5],
            "config": json.loads(row[6]),
        }

    def recent_runs(self, limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("recent run limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT run_id, created_at, finished_at, status, generation_id,
                          policy_version
                   FROM runs ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": row[0],
                "created_at": row[1],
                "finished_at": row[2],
                "status": row[3],
                "generation_id": row[4],
                "policy_version": row[5],
            }
            for row in rows
        ]

    def export_jsonl(self, run_id: str, path: Path) -> None:
        run = self.run(run_id)
        if run is None:
            raise ValueError(f"unknown run: {run_id}")
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"trace export destination already exists: {path}")
        created = False
        try:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                created = True
                stream.write(
                    json.dumps(
                        {"record_type": "run", "data": run},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
                for event in self.events(run_id):
                    stream.write(
                        json.dumps(
                            {"record_type": "event", "data": event},
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
        except Exception:
            if created:
                path.unlink(missing_ok=True)
            raise

    def unfinished_runs(self) -> Iterable[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM runs WHERE status='running' ORDER BY created_at"
            ).fetchall()
        return [row[0] for row in rows]


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
