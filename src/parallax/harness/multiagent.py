"""Ganymede bounded delegation for specialist analysis agents."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, Mapping, Protocol

from parallax.core import ExecutionResult
from parallax.errors import BudgetExceededError, PermissionDeniedError
from parallax.harness.agent import AgentLoop
from parallax.harness.openai_compatible import ChatModel
from parallax.harness.policy import AgentProfile
from parallax.runtime.budget import RunBudget
from parallax.runtime.engine import RunContext, Runtime, _truncate_text
from parallax.runtime.policy import PermissionSet
from parallax.tools.base import InternalToolDefinition


_MAX_DELEGATION_DEPTH = 1


class DelegationScheduler(Protocol):
    name: str

    def run(self, worker: Callable[[], str]) -> str: ...


class ResultAggregator(Protocol):
    name: str

    def payload(
        self,
        *,
        role: str,
        answer: str,
        result_artifact_id: str,
        evidence_artifact_ids: list[str],
    ) -> Mapping[str, object]: ...


class SerialDelegationScheduler:
    name = "serial"

    def run(self, worker: Callable[[], str]) -> str:
        return worker()


class ParallelDelegationScheduler(SerialDelegationScheduler):
    """Enable bounded parallel dispatch when a coordinator returns multiple delegates."""

    name = "parallel"


class CoordinatorEvidenceAggregator:
    name = "coordinator-evidence"

    def payload(
        self,
        *,
        role: str,
        answer: str,
        result_artifact_id: str,
        evidence_artifact_ids: list[str],
    ) -> Mapping[str, object]:
        return {"result": answer}


_SCHEDULERS: dict[str, DelegationScheduler] = {
    "serial": SerialDelegationScheduler(),
    "parallel": ParallelDelegationScheduler(),
}
_AGGREGATORS: dict[str, ResultAggregator] = {
    "coordinator-evidence": CoordinatorEvidenceAggregator(),
}
SUPPORTED_ROUTING = frozenset({"coordinator"})
SUPPORTED_SCHEDULERS = frozenset(_SCHEDULERS)
SUPPORTED_AGGREGATIONS = frozenset(_AGGREGATORS)


class DelegationController:
    """Run-local admission control for bounded serial or parallel delegation."""

    def __init__(self, max_agents: int = 4) -> None:
        if not 2 <= max_agents <= 8:
            raise ValueError("multi-agent mode requires between two and eight agents")
        self.max_agents = max_agents
        self._created = 1  # The coordinator already exists.
        self._lock = threading.Lock()

    def allocate(self) -> tuple[str, str]:
        with self._lock:
            if self._created >= self.max_agents:
                raise BudgetExceededError("run agent-count budget exhausted")
            self._created += 1
        return f"agent-{uuid.uuid4().hex}", f"task-{uuid.uuid4().hex}"


class AgentDelegateTool(InternalToolDefinition):
    """Kernel-owned tool that executes one scoped specialist synchronously."""

    name = "agent.delegate"
    required_capability = "agent.delegate"
    path_fields = ()
    artifact_fields = ()

    def __init__(
        self,
        *,
        model: ChatModel,
        worker_runtime: Runtime,
        controller: DelegationController,
        profiles: tuple[AgentProfile, ...],
        routing: str,
        scheduler: str,
        aggregation: str,
        skills: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.worker_runtime = worker_runtime
        self.controller = controller
        self.profiles = {profile.role: profile for profile in profiles}
        if not self.profiles:
            raise ValueError("at least one agent profile is required")
        if routing not in SUPPORTED_ROUTING:
            raise ValueError(f"unsupported task router: {routing}")
        self.routing = routing
        try:
            self.scheduler = _SCHEDULERS[scheduler]
        except KeyError as error:
            raise ValueError(f"unsupported delegation scheduler: {scheduler}") from error
        try:
            self.aggregator = _AGGREGATORS[aggregation]
        except KeyError as error:
            raise ValueError(f"unsupported result aggregator: {aggregation}") from error
        self.skills = dict(skills or {})
        role_descriptions = "; ".join(
            f"{role}: {profile.description}" for role, profile in sorted(self.profiles.items())
        )
        max_model_turns = max(profile.max_model_turns for profile in self.profiles.values())
        max_tool_calls = max(profile.max_tool_calls for profile in self.profiles.values())
        max_tokens = max(profile.max_tokens for profile in self.profiles.values())
        self.description = (
            f"Delegate one bounded, {self.scheduler.name} analysis task to a specialist. "
            f"Tasks are routed by {self.routing}. "
            f"Available roles — {role_descriptions}. "
            "Pass only artifacts and exact capabilities required."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "role": {"type": "string", "enum": sorted(self.profiles)},
                "objective": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 8000,
                    "pattern": r"\S",
                },
                "artifact_ids": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
                    "maxItems": 16,
                    "uniqueItems": True,
                    "default": [],
                },
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                    "maxItems": 16,
                    "uniqueItems": True,
                    "default": [],
                },
                "budget": {
                    "type": "object",
                    "properties": {
                        "model_turns": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": max_model_turns,
                        },
                        "tool_calls": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": max_tool_calls,
                        },
                        "tokens": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": max_tokens,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["role", "objective"],
            "additionalProperties": False,
        }

    def invoke(self, arguments: Mapping[str, Any], context: RunContext) -> ExecutionResult:
        started = time.monotonic()
        if context.depth >= _MAX_DELEGATION_DEPTH:
            raise PermissionDeniedError("maximum delegation depth reached")
        if context.artifacts is None:
            raise PermissionDeniedError("delegation requires an artifact store")

        profile = self.profiles[str(arguments["role"])]
        objective = str(arguments["objective"]).strip()
        raw_artifacts = arguments.get("artifact_ids", [])
        artifact_ids = (
            [str(value) for value in raw_artifacts] if isinstance(raw_artifacts, list) else []
        )
        requested = self._requested_capabilities(arguments, profile, context)
        budget = self._child_budget(arguments, profile, context)
        child_permissions = set(requested)
        executor_capability = self.worker_runtime.executor.required_capability
        context.permissions.require(executor_capability)
        child_permissions.add(executor_capability)
        child_registry = self.worker_runtime.registry.filtered(child_permissions)
        for artifact_id in artifact_ids:
            context.artifacts.resolve(
                context.run_id,
                artifact_id,
                "tool.read",
                agent_id=context.agent_id,
            )

        agent_id, task_id = self.controller.allocate()
        objective_bytes = objective.encode("utf-8")
        event_context = {
            "agent_id": agent_id,
            "task_id": task_id,
            "parent_agent_id": context.agent_id,
            "parent_task_id": context.task_id,
            "agent_role": profile.role,
            "policy_version": context.policy_version,
        }

        self.worker_runtime.tracer.record(
            context.run_id,
            "agent.created",
            "ok",
            {
                **event_context,
                "depth": context.depth + 1,
            },
        )
        self.worker_runtime.tracer.record(
            context.run_id,
            "task.delegated",
            "queued",
            {
                **event_context,
                "objective_bytes": len(objective_bytes),
                "objective_sha256": hashlib.sha256(objective_bytes).hexdigest(),
                "requested_capabilities": sorted(requested),
                "capabilities": sorted(child_permissions),
                "artifact_ids": artifact_ids,
            },
        )

        for artifact_id in artifact_ids:
            context.artifacts.delegate(
                context.run_id,
                context.agent_id,
                agent_id,
                artifact_id,
                {"tool.read"},
            )
            self.worker_runtime.tracer.record(
                context.run_id,
                "artifact.delegated",
                "ok",
                {
                    **event_context,
                    "artifact_id": artifact_id,
                    "actions": ["tool.read"],
                },
            )

        child_runtime = Runtime(
            registry=child_registry,
            executor=self.worker_runtime.executor,
            tracer=self.worker_runtime.tracer,
            cache=self.worker_runtime.cache,
            cache_namespace=self.worker_runtime.cache_namespace,
        )
        child_context = RunContext(
            run_id=context.run_id,
            workspace=context.workspace,
            permissions=PermissionSet.from_iterable(child_permissions),
            artifacts=context.artifacts,
            budget=context.budget.lease(budget),
            policy_version=context.policy_version,
            generation_id=context.generation_id,
            agent_id=agent_id,
            task_id=task_id,
            parent_task_id=context.task_id,
            depth=context.depth + 1,
            role=profile.role,
        )
        self.worker_runtime.tracer.record(
            context.run_id,
            "budget.leased",
            "ok",
            {
                **event_context,
                "budget": budget.__dict__,
            },
        )
        self.worker_runtime.tracer.record(
            context.run_id,
            "task.started",
            "running",
            event_context,
        )

        try:
            answer = self.scheduler.run(
                lambda: AgentLoop(
                    self.model,
                    child_runtime,
                    max_steps=budget.max_model_turns,
                    system_prompt=profile.system_prompt,
                    skills=self.skills,
                ).run(objective, child_context)
            )
            answer = _truncate_text(answer, 64 * 1024, "\n[result truncated]")
            summary_ref = context.artifacts.store_bytes(
                answer.encode("utf-8"),
                f"{task_id}.result.txt",
                context.run_id,
                actions={"tool.read", "model.send"},
                agent_id=agent_id,
            )
            produced = [
                ref
                for ref in context.artifacts.list_for_run(context.run_id, agent_id=agent_id)
                if ref.id not in artifact_ids and (ref.size > 0 or ref.id == summary_ref.id)
            ]
            for ref in produced:
                context.artifacts.delegate(
                    context.run_id,
                    agent_id,
                    context.agent_id,
                    ref.id,
                    {"tool.read", "model.send"},
                )
                self.worker_runtime.tracer.record(
                    context.run_id,
                    "artifact.promoted",
                    "ok",
                    {
                        **event_context,
                        "target_agent_id": context.agent_id,
                        "artifact_id": ref.id,
                        "actions": ["model.send", "tool.read"],
                    },
                )
            evidence_ids = sorted({ref.id for ref in produced} | {summary_ref.id})
            self.worker_runtime.tracer.record(
                context.run_id,
                "task.completed",
                "ok",
                {
                    **event_context,
                    "result_artifact_id": summary_ref.id,
                    "evidence_artifact_ids": evidence_ids,
                    "budget": child_context.budget.snapshot(),
                },
            )
        except Exception as error:
            self.worker_runtime.tracer.record(
                context.run_id,
                "task.failed",
                "error",
                {
                    **event_context,
                    "reason_code": type(error).__name__,
                    "budget": child_context.budget.snapshot(),
                },
            )
            raise

        payload = dict(
            self.aggregator.payload(
                role=profile.role,
                answer=answer,
                result_artifact_id=summary_ref.id,
                evidence_artifact_ids=evidence_ids,
            )
        )
        payload.update(
            {
                "role": profile.role,
                "status": "completed",
                "trust": "untrusted",
                "result_artifact_id": summary_ref.id,
                "evidence_artifact_ids": evidence_ids,
            }
        )
        stdout = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return ExecutionResult(
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_ms=round((time.monotonic() - started) * 1000),
        )

    def _requested_capabilities(
        self,
        arguments: Mapping[str, Any],
        profile: AgentProfile,
        context: RunContext,
    ) -> frozenset[str]:
        raw = arguments.get("capabilities", [])
        requested = (
            frozenset(str(value) for value in raw)
            if isinstance(raw, list) and raw
            else profile.default_capabilities
        )
        forbidden = {
            capability
            for capability in requested
            if capability == "agent.delegate"
            or capability == "memory.write"
            or capability.startswith("evolution.")
            or capability == "artifact.export"
            or capability.endswith(".export")
        }
        if forbidden:
            raise PermissionDeniedError(
                "capabilities cannot be delegated: " + ", ".join(sorted(forbidden))
            )
        missing = requested - context.permissions.capabilities
        if missing:
            raise PermissionDeniedError(
                "parent agent lacks delegated capabilities: " + ", ".join(sorted(missing))
            )
        if profile.allowed_capabilities is not None:
            outside_profile = requested - profile.allowed_capabilities
            if outside_profile:
                raise PermissionDeniedError(
                    f"role {profile.role!r} cannot receive capabilities: "
                    + ", ".join(sorted(outside_profile))
                )
        return requested

    @staticmethod
    def _child_budget(
        arguments: Mapping[str, Any], profile: AgentProfile, context: RunContext
    ) -> RunBudget:
        raw = arguments.get("budget", {})
        requested = raw if isinstance(raw, dict) else {}
        remaining = context.budget.remaining_wall_seconds()
        if remaining <= 0:
            raise BudgetExceededError("run wall-clock budget exhausted")
        parent = context.budget.budget
        return RunBudget(
            max_wall_seconds=min(profile.max_wall_seconds, remaining),
            max_tool_calls=min(
                int(requested.get("tool_calls", profile.max_tool_calls)), profile.max_tool_calls
            ),
            max_model_turns=min(
                int(requested.get("model_turns", profile.max_model_turns)),
                profile.max_model_turns,
            ),
            max_total_tokens=min(
                int(requested.get("tokens", profile.max_tokens)), profile.max_tokens
            ),
            max_tool_seconds=min(parent.max_tool_seconds, remaining),
            max_output_bytes=min(parent.max_output_bytes, 256_000),
            max_observation_bytes=parent.max_observation_bytes,
            max_artifact_bytes=parent.max_artifact_bytes,
            max_total_artifact_bytes=parent.max_total_artifact_bytes,
        )
