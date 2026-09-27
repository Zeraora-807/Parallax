"""Minimal OpenAI-compatible chat completions client."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from typing import Any, Mapping, Protocol, Sequence

import httpx

from parallax.core import ModelReply, ToolCall
from parallax.errors import ModelProtocolError


def _tool_aliases(tools: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    names = [tool["function"]["name"] for tool in tools]
    if len(set(names)) != len(names):
        raise ModelProtocolError("model tools must have unique names")
    aliases = {name: name for name in names if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)}
    used = set(aliases.values())
    for name in sorted(set(names) - aliases.keys()):
        alias = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "tool"
        if alias in used:
            digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
            alias = f"{alias[:47]}_{digest}"
        if alias in used:
            raise ModelProtocolError("model tool aliases collide")
        aliases[name] = alias
        used.add(alias)
    return aliases


def _map_message_names(message: Mapping[str, Any], names: Mapping[str, str]) -> dict[str, Any]:
    mapped = dict(message)
    if message.get("tool_calls"):
        calls = []
        for call in message["tool_calls"]:
            function = dict(call["function"])
            function["name"] = names.get(function["name"], function["name"])
            calls.append({**call, "function": function})
        mapped["tool_calls"] = calls
    if message.get("role") == "tool" and isinstance(message.get("name"), str):
        mapped["name"] = names.get(message["name"], message["name"])
    return mapped


class ChatModel(Protocol):
    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float | None = None,
    ) -> ModelReply: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        max_response_bytes: int = 10_000_000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self._thread_state = threading.local()
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self._api_key = api_key
        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

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
        aliases = _tool_aliases(tools)
        original_names = {alias: name for name, alias in aliases.items()}
        request = {
            "model": self.model,
            "messages": [_map_message_names(message, aliases) for message in messages],
        }
        if tools:
            request["tools"] = [
                {
                    **tool,
                    "function": {**tool["function"], "name": aliases[tool["function"]["name"]]},
                }
                for tool in tools
            ]
            request["tool_choice"] = "auto"
        deadline = time.monotonic() + min(
            self.timeout_seconds,
            timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
        )
        payload: object | None = None
        for attempt in range(self.max_retries + 1):
            self.last_attempt_count = attempt + 1
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise httpx.TimeoutException("model request exhausted its deadline")
                with self._client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=request,
                    timeout=remaining,
                ) as response:
                    content = bytearray()
                    for chunk in response.iter_bytes():
                        if len(content) + len(chunk) > self.max_response_bytes:
                            raise ModelProtocolError(
                                f"model response exceeds {self.max_response_bytes} bytes"
                            )
                        content.extend(chunk)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as error:
                        raise httpx.HTTPStatusError(
                            self._http_error_message(response.status_code, content),
                            request=error.request,
                            response=error.response,
                        ) from error
                try:
                    payload = json.loads(content)
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise ModelProtocolError("model response is not valid JSON") from error
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                status = (
                    error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                )
                if attempt >= self.max_retries or status not in {None, 429, 502, 503, 504}:
                    raise
                retry_after = (
                    error.response.headers.get("retry-after")
                    if isinstance(error, httpx.HTTPStatusError)
                    else None
                )
                delay = (
                    min(5.0, float(retry_after))
                    if retry_after and retry_after.isdigit()
                    else 0.5 * 2**attempt
                )
                time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        if payload is None:
            raise ModelProtocolError("model request produced no response")
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise ModelProtocolError("response has no choices[0].message") from error

        parsed_calls = []
        if not isinstance(message, dict):
            raise ModelProtocolError("response message must be an object")
        for raw_call in message.get("tool_calls") or []:
            try:
                function = raw_call["function"]
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise TypeError("arguments must decode to an object")
                if not isinstance(raw_call["id"], str) or not isinstance(function["name"], str):
                    raise TypeError("tool id and name must be strings")
                parsed_calls.append(
                    ToolCall(
                        id=raw_call["id"],
                        name=original_names.get(function["name"], function["name"]),
                        arguments=arguments,
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ModelProtocolError("model returned a malformed tool call") from error
        return ModelReply(
            message=_map_message_names(message, original_names),
            finish_reason=choice.get("finish_reason"),
            tool_calls=tuple(parsed_calls),
            usage={
                key: value
                for key, value in (payload.get("usage") or {}).items()
                if isinstance(value, int)
            },
        )

    def _http_error_message(self, status: int, content: bytes | bytearray) -> str:
        try:
            payload = json.loads(content)
        except (ValueError, UnicodeError):
            payload = None
        detail = content.decode("utf-8", errors="replace")
        if isinstance(payload, dict):
            error = payload.get("error", payload)
            if isinstance(error, dict):
                detail = str(error.get("message") or "no error details")
                code = error.get("code")
                if code:
                    detail = f"{code}: {detail}"
            elif isinstance(error, str):
                detail = error
            else:
                detail = "no error details"
        if self._api_key:
            detail = detail.replace(self._api_key, "[REDACTED]")
        detail = detail.strip() or "no error details"
        if len(detail) > 2048:
            detail = detail[:2048] + "…"
        return f"Model API returned HTTP {status}: {detail}"

    def close(self) -> None:
        self._client.close()
