"""Bounded persistent caches for exact model replies and deterministic tool executions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from parallax.core import CommandSpec, ExecutionResult, ModelReply, ToolCall
from parallax.core import ArtifactRef, JoinedArtifactRef, JoinedWorkspacePath, WorkspacePath


CACHE_FORMAT_VERSION = "ganymede-cache-v1"


class SQLiteCacheStore:
    def __init__(
        self,
        path: Path,
        *,
        max_entries: int = 4096,
        max_table_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if max_entries < 1:
            raise ValueError("cache max_entries must be positive")
        if max_table_bytes < 1:
            raise ValueError("cache max_table_bytes must be positive")
        self.path = path
        self.max_entries = max_entries
        self.max_table_bytes = max_table_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_access REAL NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS tool_cache (
                    cache_key TEXT PRIMARY KEY,
                    stdout BLOB NOT NULL,
                    stderr BLOB NOT NULL,
                    stdout_text TEXT NOT NULL,
                    stderr_text TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    last_access REAL NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS model_cache_access ON model_cache(last_access);
                CREATE INDEX IF NOT EXISTS tool_cache_access ON tool_cache(last_access);
                """
            )
            columns = {
                str(row[1]) for row in db.execute("PRAGMA table_info(tool_cache)").fetchall()
            }
            if "duration_ms" not in columns:
                db.execute(
                    "ALTER TABLE tool_cache ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0"
                )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def model_key(
        namespace: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> str:
        return _digest(
            {
                "kind": "model",
                "namespace": namespace,
                "messages": list(messages),
                "tools": list(tools),
            }
        )

    @staticmethod
    def tool_key(namespace: str, tool_name: str, command: CommandSpec) -> str:
        return _digest(
            {
                "kind": "tool",
                "namespace": namespace,
                "tool": tool_name,
                "program": command.program,
                "arguments": [_command_argument(value) for value in command.arguments],
                "image": command.image,
            }
        )

    def get_model(self, cache_key: str) -> ModelReply | None:
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT response_json FROM model_cache WHERE cache_key=?", (cache_key,)
            ).fetchone()
            if row is None:
                return None
            try:
                reply = _decode_model_reply(row[0])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                db.execute("DELETE FROM model_cache WHERE cache_key=?", (cache_key,))
                return None
            db.execute(
                "UPDATE model_cache SET last_access=?, hits=hits+1 WHERE cache_key=?",
                (time.time(), cache_key),
            )
        return reply

    def put_model(self, cache_key: str, reply: ModelReply) -> None:
        payload = json.dumps(
            {
                "message": dict(reply.message),
                "finish_reason": reply.finish_reason,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
                    for call in reply.tool_calls
                ],
                "usage": dict(reply.usage),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        now = time.time()
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO model_cache "
                "(cache_key, response_json, created_at, last_access, hits) "
                "VALUES (?, ?, ?, ?, 0)",
                (cache_key, payload, now, now),
            )
            self._prune(db, "model_cache")

    def get_tool(self, cache_key: str) -> ExecutionResult | None:
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT stdout, stderr, stdout_text, stderr_text, duration_ms "
                "FROM tool_cache "
                "WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            try:
                result = _decode_tool_result(row)
            except (TypeError, ValueError):
                db.execute("DELETE FROM tool_cache WHERE cache_key=?", (cache_key,))
                return None
            db.execute(
                "UPDATE tool_cache SET last_access=?, hits=hits+1 WHERE cache_key=?",
                (time.time(), cache_key),
            )
        return result

    def put_tool(self, cache_key: str, result: ExecutionResult) -> None:
        if result.exit_code != 0 or result.truncated:
            return
        stdout = result.stdout_bytes or result.stdout.encode("utf-8")
        stderr = result.stderr_bytes or result.stderr.encode("utf-8")
        now = time.time()
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO tool_cache "
                "(cache_key, stdout, stderr, stdout_text, stderr_text, duration_ms, "
                "created_at, last_access, hits) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    cache_key,
                    stdout,
                    stderr,
                    result.stdout,
                    result.stderr,
                    result.duration_ms,
                    now,
                    now,
                ),
            )
            self._prune(db, "tool_cache")

    def stats(self) -> dict[str, int]:
        with self._lock, self._connect() as db:
            model_rows = db.execute("SELECT response_json, hits FROM model_cache").fetchall()
            tool = db.execute("SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM tool_cache").fetchone()
        model_hits = sum(int(row[1]) for row in model_rows)
        saved_tokens = 0
        for raw, hits in model_rows:
            try:
                payload = json.loads(raw)
                total_tokens = payload.get("usage", {}).get("total_tokens", 0)
            except (AttributeError, json.JSONDecodeError, TypeError):
                continue
            if type(total_tokens) is int and total_tokens >= 0:
                saved_tokens += total_tokens * int(hits)
        return {
            "model_entries": len(model_rows),
            "model_hits": model_hits,
            "model_saved_tokens": saved_tokens,
            "tool_entries": int(tool[0]),
            "tool_hits": int(tool[1]),
        }

    def _prune(self, db: sqlite3.Connection, table: str) -> None:
        size_expressions = {
            "model_cache": "length(CAST(response_json AS BLOB))",
            "tool_cache": (
                "length(stdout) + length(stderr) + "
                "length(CAST(stdout_text AS BLOB)) + length(CAST(stderr_text AS BLOB))"
            ),
        }
        size_expression = size_expressions[table]
        db.execute(
            f"DELETE FROM {table} WHERE cache_key IN "  # noqa: S608 - fixed internal table names
            f"(SELECT cache_key FROM {table} "
            "ORDER BY last_access DESC, created_at DESC, cache_key DESC "
            "LIMIT -1 OFFSET ?)",
            (self.max_entries,),
        )
        rows = db.execute(
            f"SELECT cache_key, {size_expression} FROM {table} "  # noqa: S608
            "ORDER BY last_access ASC, created_at ASC, cache_key ASC"
        ).fetchall()
        total_bytes = sum(int(row[1]) for row in rows)
        expired: list[tuple[str]] = []
        for cache_key, size in rows:
            if total_bytes <= self.max_table_bytes:
                break
            expired.append((str(cache_key),))
            total_bytes -= int(size)
        if expired:
            db.executemany(
                f"DELETE FROM {table} WHERE cache_key=?",  # noqa: S608
                expired,
            )


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _decode_model_reply(raw: object) -> ModelReply:
    if not isinstance(raw, str):
        raise TypeError("cached model response is not text")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
        raise TypeError("cached model response has an invalid message")
    raw_calls = payload.get("tool_calls", [])
    usage = payload.get("usage", {})
    if not isinstance(raw_calls, list) or not isinstance(usage, dict):
        raise TypeError("cached model response has invalid metadata")
    total_tokens = usage.get("total_tokens", 0)
    if type(total_tokens) is not int or total_tokens < 0:
        raise ValueError("cached model response has invalid token usage")
    tool_calls: list[ToolCall] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            raise TypeError("cached tool call is not an object")
        call_id = item["id"]
        name = item["name"]
        arguments = item["arguments"]
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise TypeError("cached tool call has invalid identity")
        if not isinstance(arguments, dict):
            raise TypeError("cached tool call has invalid arguments")
        tool_calls.append(ToolCall(call_id, name, arguments))
    return ModelReply(
        message=payload["message"],
        finish_reason=payload.get("finish_reason"),
        tool_calls=tuple(tool_calls),
        usage={**usage, "cached_total_tokens": total_tokens, "total_tokens": 0},
    )


def _decode_tool_result(row: tuple[object, ...]) -> ExecutionResult:
    stdout_bytes, stderr_bytes, stdout, stderr, duration_ms = row
    if not isinstance(stdout_bytes, (bytes, bytearray, memoryview)) or not isinstance(
        stderr_bytes, (bytes, bytearray, memoryview)
    ):
        raise TypeError("cached tool output is not binary data")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise TypeError("cached tool preview is not text")
    if not isinstance(duration_ms, int) or duration_ms < 0:
        raise ValueError("cached tool duration is invalid")
    return ExecutionResult(
        0,
        stdout,
        stderr,
        0,
        stdout_bytes=bytes(stdout_bytes),
        stderr_bytes=bytes(stderr_bytes),
    )


def _command_argument(value: object) -> object:
    if isinstance(value, ArtifactRef):
        return {"artifact": value.digest}
    if isinstance(value, JoinedArtifactRef):
        return {"prefix": value.prefix, "artifact": value.artifact.digest}
    if isinstance(value, WorkspacePath):
        return {"workspace": value.relative.as_posix()}
    if isinstance(value, JoinedWorkspacePath):
        return {"prefix": value.prefix, "workspace": value.path.relative.as_posix()}
    return value
