"""Persistent conversations and settings for the local WebUI."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class WebStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    reference_ids_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversations_updated
                    ON conversations(updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_conversation
                    ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS attachments (
                    attachment_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                    name TEXT NOT NULL, path TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS attachments_conversation
                    ON attachments(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS web_settings (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_conversation(self, title: str, reference_ids: list[str]) -> str:
        conversation_id = uuid.uuid4().hex
        now = self._now()
        with self._lock, self._connect() as db:
            existing = (
                {
                    row[0]
                    for row in db.execute(
                        f"SELECT conversation_id FROM conversations WHERE conversation_id IN ({','.join('?' for _ in reference_ids)})",
                        reference_ids,
                    ).fetchall()
                }
                if reference_ids
                else set()
            )
            references = [item for item in dict.fromkeys(reference_ids) if item in existing]
            db.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    title.strip()[:120] or "未命名对话",
                    json.dumps(references),
                    now,
                    now,
                ),
            )
        return conversation_id

    def list_conversations(self) -> list[dict[str, object]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 200"
            ).fetchall()
        return [self._conversation(row) for row in rows]

    def conversation(self, conversation_id: str) -> dict[str, object] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
        return self._conversation(row) if row else None

    @staticmethod
    def _conversation(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": row["conversation_id"],
            "title": row["title"],
            "reference_ids": json.loads(row["reference_ids_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def add_message(
        self, conversation_id: str, role: str, content: str, status: str = "done"
    ) -> str:
        message_id = uuid.uuid4().hex
        now = self._now()
        with self._lock, self._connect() as db:
            if not db.execute(
                "SELECT 1 FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone():
                raise ValueError("unknown conversation")
            db.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, status, now),
            )
            db.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                (now, conversation_id),
            )
        return message_id

    def update_message(self, message_id: str, content: str, status: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE messages SET content=?, status=? WHERE message_id=?",
                (content, status, message_id),
            )

    def begin_turn(self, conversation_id: str, content: str) -> str:
        """Publish the user message and pending response in one transaction."""
        response_id = uuid.uuid4().hex
        now = self._now()
        with self._lock, self._connect() as db:
            if not db.execute(
                "SELECT 1 FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone():
                raise ValueError("对话不存在")
            db.executemany(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (uuid.uuid4().hex, conversation_id, "user", content, "done", now),
                    (response_id, conversation_id, "assistant", "", "running", now),
                ],
            )
            db.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                (now, conversation_id),
            )
        return response_id

    def recover_interrupted(self) -> int:
        with self._lock, self._connect() as db:
            cursor = db.execute(
                "UPDATE messages SET status='error', content=? WHERE status='running'",
                ("服务已重启，上次分析已中断。可以重新发送任务。",),
            )
        return cursor.rowcount

    def messages(self, conversation_id: str, limit: int = 200) -> list[dict[str, str]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY rowid DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [
            {
                "id": row["message_id"],
                "role": row["role"],
                "content": row["content"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]

    def add_attachment(self, conversation_id: str, name: str, path: Path) -> str:
        attachment_id = uuid.uuid4().hex
        with self._lock, self._connect() as db:
            if not db.execute(
                "SELECT 1 FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone():
                raise ValueError("unknown conversation")
            db.execute(
                "INSERT INTO attachments VALUES (?, ?, ?, ?, ?)",
                (attachment_id, conversation_id, name, str(path), self._now()),
            )
        return attachment_id

    def attachments(self, conversation_id: str) -> list[dict[str, str]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM attachments WHERE conversation_id=? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        return [
            {"id": row["attachment_id"], "name": row["name"], "path": row["path"]} for row in rows
        ]

    def settings(self) -> dict[str, str]:
        with self._connect() as db:
            return {row[0]: row[1] for row in db.execute("SELECT key, value FROM web_settings")}

    def save_settings(self, values: dict[str, str]) -> None:
        allowed = {
            "theme",
            "base_url",
            "model",
            "skills",
            "agents",
            "multi_agent",
            "max_agents",
        }
        with self._lock, self._connect() as db:
            for key, value in values.items():
                if key in allowed:
                    db.execute(
                        "INSERT OR REPLACE INTO web_settings VALUES (?, ?)", (key, value[:4000])
                    )
