"""Exact-request model cache that preserves the ChatModel protocol."""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, Mapping, Sequence

from parallax.core import ModelReply
from parallax.harness.openai_compatible import ChatModel
from parallax.runtime.cache import SQLiteCacheStore


class CachedChatModel:
    def __init__(self, model: ChatModel, cache: SQLiteCacheStore, namespace: str) -> None:
        self.model = model
        self.cache = cache
        self.namespace = namespace
        self._thread_state = threading.local()

    @property
    def last_cache_hit(self) -> bool:
        return getattr(self._thread_state, "last_cache_hit", False)

    @last_cache_hit.setter
    def last_cache_hit(self, value: bool) -> None:
        self._thread_state.last_cache_hit = value

    @property
    def last_cache_error(self) -> str | None:
        return getattr(self._thread_state, "last_cache_error", None)

    @last_cache_error.setter
    def last_cache_error(self, value: str | None) -> None:
        self._thread_state.last_cache_error = value

    @property
    def last_attempt_count(self) -> int:
        return getattr(self._thread_state, "last_attempt_count", 0)

    @last_attempt_count.setter
    def last_attempt_count(self, value: int) -> None:
        self._thread_state.last_attempt_count = value

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float | None = None,
    ) -> ModelReply:
        self.last_cache_error = None
        cache_key = self.cache.model_key(self.namespace, messages, tools)
        try:
            cached = self.cache.get_model(cache_key)
        except (OSError, sqlite3.Error) as error:
            self.last_cache_error = type(error).__name__
            cached = None
        if cached is not None:
            self.last_cache_hit = True
            self.last_attempt_count = 0
            return cached
        self.last_cache_hit = False
        reply = self.model.complete(messages, tools, timeout_seconds=timeout_seconds)
        self.last_attempt_count = getattr(self.model, "last_attempt_count", 1)
        try:
            self.cache.put_model(cache_key, reply)
        except (OSError, sqlite3.Error) as error:
            self.last_cache_error = type(error).__name__
        return reply

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
