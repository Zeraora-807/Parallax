"""The deterministic reference monitor for every tool invocation."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field, replace
from typing import Mapping

from parallax.core import ToolCall, ToolResult
from parallax.errors import InternalRuntimeError, ParallaxError
from parallax.executor.base import Executor
from parallax.runtime.artifacts import ArtifactStore
from parallax.runtime.budget import BudgetLease, BudgetLedger, RunBudget
from parallax.runtime.cache import SQLiteCacheStore
from parallax.runtime.policy import PermissionSet
from parallax.runtime.workspace import Workspace
from parallax.tools.base import InternalToolDefinition
from parallax.tools.registry import ToolRegistry
from parallax.tracing.jsonl import NullTracer, TraceRecorder


POLICY_VERSION = "ganymede-v1"


@dataclass(frozen=True)
class RunContext:
    run_id: str
    workspace: Workspace
    permissions: PermissionSet
    artifacts: ArtifactStore | None = None
    budget: BudgetLedger | BudgetLease = field(default_factory=lambda: BudgetLedger(RunBudget()))
    policy_version: str = POLICY_VERSION
    generation_id: str | None = None
    agent_id: str = "root"
    task_id: str = "root"
    parent_task_id: str | None = None
    depth: int = 0
    role: str = "coordinator"


class Runtime:
    def __init__(
        self,
        registry: ToolRegistry,
        executor: Executor,
        tracer: TraceRecorder | None = None,
        cache: SQLiteCacheStore | None = None,
        cache_namespace: str = POLICY_VERSION,
    ) -> None:
        self.registry = registry
        self.executor = executor
        self.tracer = tracer or NullTracer()
        self.cache = cache
        self.cache_namespace = cache_namespace

    def execute_tool(self, call: ToolCall, context: RunContext) -> ToolResult:
        execution_started = False
        try:
            self.tracer.record(
                context.run_id,
                "tool.received",
                "started",
                {
                    **_context_data(context),
                    "call_id": call.id,
                    "tool": call.name,
                    "arguments": _audit_arguments(dict(call.arguments)),
                },
            )
            tool = self.registry.get(call.name)
            tool.validate(call.arguments)
            context.permissions.require(
                tool.required_capability,
                self.executor.required_capability,
            )
            resolved_arguments: Mapping[str, object] = dict(call.arguments)
            if tool.path_fields:
                resolved_arguments = context.workspace.resolve_fields(
                    resolved_arguments, tool.path_fields
                )
            if tool.artifact_fields:
                if context.artifacts is None:
                    raise InternalRuntimeError("run has no artifact store")
                resolved_arguments = context.artifacts.resolve_fields(
                    context.run_id,
                    dict(resolved_arguments),
                    tool.artifact_fields,
                    agent_id=context.agent_id,
                )
            command = None
            cache_key = None
            cache_hit = False
            execution = None
            if not isinstance(tool, InternalToolDefinition):
                command = tool.prepare(resolved_arguments)
                if self.cache is not None and tool.artifact_fields and not tool.path_fields:
                    cache_key = self.cache.tool_key(self.cache_namespace, call.name, command)
            timeout = context.budget.reserve_tool_call()
            if cache_key is not None and self.cache is not None:
                try:
                    execution = self.cache.get_tool(cache_key)
                except (OSError, sqlite3.Error) as error:
                    self.tracer.record(
                        context.run_id,
                        "tool.cache_error",
                        "degraded",
                        {
                            **_context_data(context),
                            "call_id": call.id,
                            "tool": call.name,
                            "phase": "read",
                            "reason_code": type(error).__name__,
                        },
                    )
                cache_hit = execution is not None
            if execution is None and command is not None:
                command = replace(command, timeout_seconds=min(command.timeout_seconds, timeout))
            self.tracer.record(
                context.run_id,
                "tool.authorized",
                "ok",
                {
                    **_context_data(context),
                    "call_id": call.id,
                    "tool": call.name,
                    "program": command.program if command is not None else f"internal:{tool.name}",
                    "input_refs": [resolved_arguments[name].id for name in tool.artifact_fields],
                    "workspace_paths": [
                        str(resolved_arguments[name].relative) for name in tool.path_fields
                    ],
                    "cache_hit": cache_hit,
                },
            )
            execution_started = True
            if execution is None:
                execution = (
                    tool.invoke(resolved_arguments, context)
                    if isinstance(tool, InternalToolDefinition)
                    else self.executor.execute(command, context.workspace)
                )
                if cache_key is not None and self.cache is not None:
                    try:
                        self.cache.put_tool(cache_key, execution)
                    except (OSError, sqlite3.Error) as error:
                        self.tracer.record(
                            context.run_id,
                            "tool.cache_error",
                            "degraded",
                            {
                                **_context_data(context),
                                "call_id": call.id,
                                "tool": call.name,
                                "phase": "write",
                                "reason_code": type(error).__name__,
                            },
                        )
            else:
                self.tracer.record(
                    context.run_id,
                    "tool.cache_hit",
                    "ok",
                    {
                        **_context_data(context),
                        "call_id": call.id,
                        "tool": call.name,
                        "cache_key": cache_key,
                    },
                )
            stdout_bytes = execution.stdout_bytes or execution.stdout.encode("utf-8")
            stderr_bytes = execution.stderr_bytes or execution.stderr.encode("utf-8")
            context.budget.record_output_bytes(len(stdout_bytes) + len(stderr_bytes))
            output_artifacts: dict[str, str] = {}
            stdout = execution.stdout
            stderr = execution.stderr
            if context.artifacts is not None:
                stdout_ref = context.artifacts.store_bytes(
                    stdout_bytes,
                    f"{call.id}.stdout",
                    context.run_id,
                    agent_id=context.agent_id,
                )
                stderr_ref = context.artifacts.store_bytes(
                    stderr_bytes,
                    f"{call.id}.stderr",
                    context.run_id,
                    agent_id=context.agent_id,
                )
                output_artifacts = {"stdout": stdout_ref.id, "stderr": stderr_ref.id}
                preview_limit = context.budget.budget.max_observation_bytes
                stdout = _truncate_text(execution.stdout, preview_limit)
                stderr = _truncate_text(execution.stderr, preview_limit)
            result = ToolResult(
                ok=execution.exit_code == 0,
                tool=call.name,
                call_id=call.id,
                exit_code=execution.exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=execution.duration_ms,
                truncated=(
                    execution.truncated or stdout != execution.stdout or stderr != execution.stderr
                ),
                output_artifacts=output_artifacts,
            )
            self.tracer.record(
                context.run_id,
                "tool.completed",
                "ok" if result.ok else "error",
                {
                    **_context_data(context),
                    "call_id": call.id,
                    "tool": call.name,
                    "exit_code": execution.exit_code,
                    "duration_ms": execution.duration_ms,
                    "output_refs": output_artifacts,
                    "budget": context.budget.snapshot(),
                    "cache_hit": cache_hit,
                },
            )
            return result
        except ParallaxError as error:
            result = ToolResult(
                ok=False,
                tool=call.name,
                call_id=call.id,
                error_code=error.code,
                error_message=str(error),
                retryable=error.retryable,
                execution_state=(
                    "completed"
                    if execution_started and error.execution_state == "not_started"
                    else error.execution_state
                ),
                recovery_hint=error.recovery_hint,
            )
            self.tracer.record(
                context.run_id,
                "tool.failed" if execution_started else "tool.rejected",
                "error",
                {
                    **_context_data(context),
                    **result.as_dict(),
                    "reason_code": error.code,
                },
            )
            return result
        except Exception as error:  # trusted component bug or unexpected OS failure
            wrapped = InternalRuntimeError(f"trusted runtime failed ({type(error).__name__})")
            result = ToolResult(
                ok=False,
                tool=call.name,
                call_id=call.id,
                error_code=wrapped.code,
                error_message=str(wrapped),
                execution_state="completed" if execution_started else wrapped.execution_state,
            )
            self.tracer.record(
                context.run_id,
                "tool.failed",
                "error",
                {
                    **_context_data(context),
                    **result.as_dict(),
                    "exception_type": type(error).__name__,
                    "reason_code": wrapped.code,
                },
            )
            return result


def _truncate_text(value: str, max_bytes: int, marker: str = "\n[observation truncated]") -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    marker_bytes = marker.encode("utf-8")
    prefix_bytes = max(0, max_bytes - len(marker_bytes))
    prefix = encoded[:prefix_bytes].decode("utf-8", errors="ignore")
    suffix = marker_bytes[: max_bytes - len(prefix.encode("utf-8"))].decode(
        "utf-8", errors="ignore"
    )
    return prefix + suffix


def _context_data(context: RunContext) -> dict[str, object]:
    return {
        "agent_id": context.agent_id,
        "task_id": context.task_id,
        "parent_task_id": context.parent_task_id,
        "agent_role": context.role,
        "policy_version": context.policy_version,
    }


def _audit_arguments(value: object, key: str = "") -> object:
    if isinstance(value, dict):
        return {
            str(item_key): _audit_arguments(item, str(item_key)) for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_audit_arguments(item, key) for item in value[:32]]
    if isinstance(value, str):
        lowered = key.casefold()
        if any(
            marker in lowered
            for marker in (
                "content",
                "basis",
                "objective",
                "secret",
                "token",
                "password",
                "api_key",
            )
        ):
            encoded = value.encode("utf-8")
            return {
                "redacted": True,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        if len(value) > 512:
            return value[:512] + "…"
    return value
