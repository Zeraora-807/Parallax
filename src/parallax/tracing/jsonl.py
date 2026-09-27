"""Small append-only JSONL tracing implementation."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol


class TraceRecorder(Protocol):
    def record(
        self,
        run_id: str,
        event: str,
        status: str,
        data: Mapping[str, Any] | None = None,
    ) -> None: ...


class NullTracer:
    def record(
        self,
        run_id: str,
        event: str,
        status: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        return None


class JsonlTracer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequences: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(
        self,
        run_id: str,
        event: str,
        status: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._sequences[run_id] += 1
            entry = {
                "run_id": run_id,
                "sequence": self._sequences[run_id],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "status": status,
                "data": dict(data or {}),
            }
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
