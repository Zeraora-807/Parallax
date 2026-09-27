"""Trusted tool definitions compile validated values into commands."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping

from jsonschema import Draft202012Validator

from parallax.core import CommandSpec, ExecutionResult
from parallax.errors import ToolValidationError

if TYPE_CHECKING:
    from parallax.runtime.engine import RunContext


class ToolDefinition(ABC):
    name: str
    description: str
    parameters: Mapping[str, Any]
    required_capability: str
    path_fields: tuple[str, ...] = ()
    artifact_fields: tuple[str, ...] = ()

    def validate(self, arguments: Mapping[str, Any]) -> None:
        validator = Draft202012Validator(self.parameters)
        errors = sorted(validator.iter_errors(arguments), key=lambda item: list(item.path))
        if errors:
            messages = []
            for error in errors:
                location = ".".join(str(part) for part in error.path) or "$"
                messages.append(f"{location}: {error.message}")
            raise ToolValidationError("; ".join(messages))

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }

    @abstractmethod
    def prepare(self, arguments: Mapping[str, Any]) -> CommandSpec:
        """Compile already validated and path-resolved arguments."""


class InternalToolDefinition(ToolDefinition):
    """Trusted kernel service exposed as a tool; never loaded from third-party plugins."""

    def prepare(self, arguments: Mapping[str, Any]) -> CommandSpec:
        raise TypeError("internal tools do not produce command specs")

    @abstractmethod
    def invoke(self, arguments: Mapping[str, Any], context: "RunContext") -> ExecutionResult: ...
