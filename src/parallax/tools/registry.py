"""Explicit allowlist of tool definitions visible to the model."""

from __future__ import annotations

from collections.abc import Iterable

from parallax.errors import ToolNotFoundError
from parallax.tools.base import ToolDefinition


class ToolRegistry:
    def __init__(self, tools: Iterable[ToolDefinition] = ()) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool registration: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotFoundError(f"unknown tool: {name}") from error

    def openai_schemas(self) -> list[dict[str, object]]:
        return [self._tools[name].openai_schema() for name in sorted(self._tools)]

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def copy(self) -> "ToolRegistry":
        return ToolRegistry(self.definitions())

    def filtered(self, capabilities: Iterable[str]) -> "ToolRegistry":
        allowed = frozenset(capabilities)
        return ToolRegistry(
            tool for tool in self.definitions() if tool.required_capability in allowed
        )
