"""Model-visible self-evolution tools, registered only when explicitly enabled."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from parallax.core import ExecutionResult
from parallax.evolution.manager import EvolutionManager
from parallax.runtime.engine import RunContext
from parallax.tools.base import InternalToolDefinition


@dataclass(frozen=True)
class ProposeEvolutionTool(InternalToolDefinition):
    manager: EvolutionManager
    name: str = "evolution.propose_change"
    description: str = "Create an immutable candidate generation with proposed source-file changes."
    required_capability: str = "evolution.propose"
    parameters: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            {
                "type": "object",
                "properties": {
                    "rationale": {"type": "string", "minLength": 1, "maxLength": 4000},
                    "changes": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "minLength": 1, "maxLength": 300},
                                "content": {"type": "string", "maxLength": 500_000},
                            },
                            "required": ["path", "content"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["rationale", "changes"],
                "additionalProperties": False,
            },
        )

    def invoke(self, arguments: Mapping[str, Any], context: RunContext) -> ExecutionResult:
        changes = {item["path"]: item["content"] for item in arguments["changes"]}
        generation = self.manager.propose(changes, arguments["rationale"])
        return _json_result(
            {
                "generation_id": generation.id,
                "status": generation.status,
                "next_action": "validate in an isolated candidate validator",
            }
        )


@dataclass(frozen=True)
class ValidateEvolutionTool(InternalToolDefinition):
    manager: EvolutionManager
    name: str = "evolution.validate_candidate"
    description: str = (
        "Run an immutable candidate generation through the isolated validation suite."
    )
    required_capability: str = "evolution.validate"
    parameters: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _generation_schema())

    def invoke(self, arguments: Mapping[str, Any], context: RunContext) -> ExecutionResult:
        generation, report = self.manager.validate(arguments["generation_id"])
        return _json_result(
            {
                "generation_id": generation.id,
                "status": generation.status,
                "validation_exit_code": report.result.exit_code,
            },
            exit_code=report.result.exit_code,
        )


@dataclass(frozen=True)
class ActivateEvolutionTool(InternalToolDefinition):
    manager: EvolutionManager
    name: str = "evolution.activate_candidate"
    description: str = "Atomically activate a validated candidate generation."
    required_capability: str = "evolution.activate"
    parameters: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _generation_schema())

    def invoke(self, arguments: Mapping[str, Any], context: RunContext) -> ExecutionResult:
        generation = self.manager.activate(arguments["generation_id"])
        return _json_result({"generation_id": generation.id, "status": "active"})


def evolution_tools(manager: EvolutionManager) -> list[InternalToolDefinition]:
    tools: list[InternalToolDefinition] = [ProposeEvolutionTool(manager)]
    if manager.mode in {"trial", "auto"}:
        tools.append(ValidateEvolutionTool(manager))
    if manager.mode == "auto":
        tools.append(ActivateEvolutionTool(manager))
    return tools


def _generation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"generation_id": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
        "required": ["generation_id"],
        "additionalProperties": False,
    }


def _json_result(value: object, exit_code: int = 0) -> ExecutionResult:
    return ExecutionResult(
        exit_code=exit_code,
        stdout=json.dumps(value, ensure_ascii=False),
        stderr="",
        duration_ms=0,
    )
