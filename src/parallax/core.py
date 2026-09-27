"""Shared immutable values crossing Parallax component boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypeAlias


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class WorkspacePath:
    """A path already canonicalized and proven to be inside the workspace."""

    host: Path
    relative: Path


@dataclass(frozen=True)
class ArtifactRef:
    """An immutable, content-addressed file authorized for one or more runs."""

    id: str
    digest: str
    path: Path
    size: int
    media_type: str = "application/octet-stream"
    source_name: str = "artifact"


@dataclass(frozen=True)
class JoinedWorkspacePath:
    """A path argument joined to a fixed flag, e.g. ``--file=/workspace/a``."""

    prefix: str
    path: WorkspacePath


@dataclass(frozen=True)
class JoinedArtifactRef:
    prefix: str
    artifact: ArtifactRef


CommandArgument: TypeAlias = (
    str | WorkspacePath | JoinedWorkspacePath | ArtifactRef | JoinedArtifactRef
)


@dataclass(frozen=True)
class CommandSpec:
    program: str
    arguments: tuple[CommandArgument, ...]
    timeout_seconds: float = 20.0
    image: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    truncated: bool = False
    termination_reason: str | None = None
    stdout_bytes: bytes | None = None
    stderr_bytes: bytes | None = None


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    call_id: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int | None = None
    truncated: bool = False
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    execution_state: str = "completed"
    recovery_hint: str | None = None
    output_artifacts: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        value = {
            "ok": self.ok,
            "tool": self.tool,
            "call_id": self.call_id,
        }
        if self.exit_code is not None:
            value["exit_code"] = self.exit_code
            value["stdout"] = self.stdout
            value["stderr"] = self.stderr
            value["duration_ms"] = self.duration_ms
            value["truncated"] = self.truncated
            if self.output_artifacts:
                value["output_artifacts"] = dict(self.output_artifacts)
        if self.error_code is not None:
            value["error"] = {
                "code": self.error_code,
                "message": self.error_message,
                "retryable": self.retryable,
                "execution_state": self.execution_state,
            }
            if self.recovery_hint:
                value["error"]["recovery_hint"] = self.recovery_hint
        return value


@dataclass(frozen=True)
class ModelReply:
    message: Mapping[str, Any]
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    usage: Mapping[str, int] = field(default_factory=dict)
