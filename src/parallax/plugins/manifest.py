"""Declarative Ganymede plugin manifests; discovery never executes plugin code."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from parallax.errors import PluginValidationError


PLUGIN_ID = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
TOOL_NAME = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+$")
VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
PINNED_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
PROGRAM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
AGENT_ROLE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
STRATEGY_NAME = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
MAX_MANIFEST_BYTES = 1_000_000
MAX_AGENT_PROFILES = 32
MAX_AGENT_DESCRIPTION_CHARS = 512
MAX_AGENT_CAPABILITIES = 32


@dataclass(frozen=True)
class ToolManifest:
    name: str
    description: str
    program: str
    capability: str
    schema_file: str
    artifact_fields: tuple[str, ...]
    argument_fields: tuple[str, ...] = ()
    fixed_arguments: tuple[str, ...] = ()
    joined_artifact_prefix: str | None = None


@dataclass(frozen=True)
class AgentManifest:
    role: str
    description: str
    system_prompt_file: str
    default_capabilities: tuple[str, ...]
    allowed_capabilities: tuple[str, ...] | None
    max_model_turns: int
    max_tool_calls: int
    max_tokens: int
    max_wall_seconds: float


@dataclass(frozen=True)
class MultiAgentManifest:
    coordinator_prompt_file: str | None
    routing: str | None
    scheduler: str | None
    aggregation: str | None


@dataclass(frozen=True)
class PluginManifest:
    id: str
    version: str
    runtime_api: str
    description: str
    permissions: tuple[str, ...]
    image: str | None
    tools: tuple[ToolManifest, ...]
    skill_files: tuple[str, ...]
    agents: tuple[AgentManifest, ...]
    multiagent: MultiAgentManifest | None
    root: Path

    @classmethod
    def load(cls, path: Path, *, allow_unpinned_images: bool = False) -> "PluginManifest":
        path = path.resolve(strict=True)
        root = path.parent
        try:
            if path.stat().st_size > MAX_MANIFEST_BYTES:
                raise PluginValidationError("plugin manifest exceeds the size limit")
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
            plugin = payload["plugin"]
        except PluginValidationError:
            raise
        except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
            raise PluginValidationError(f"invalid plugin manifest {path}: {error}") from error
        if not isinstance(payload, dict) or not isinstance(plugin, dict):
            raise PluginValidationError("plugin manifest tables must be objects")

        plugin_id = _string(plugin, "id")
        version = _string(plugin, "version")
        runtime_api = _string(plugin, "runtime_api")
        description = _string(plugin, "description")
        if not PLUGIN_ID.fullmatch(plugin_id):
            raise PluginValidationError(f"invalid plugin id: {plugin_id}")
        if not VERSION.fullmatch(version):
            raise PluginValidationError(f"invalid plugin version: {version}")
        if runtime_api != "1":
            raise PluginValidationError(f"unsupported runtime API: {runtime_api}")

        permissions = tuple(_string_list(plugin.get("permissions", []), "plugin.permissions"))
        if any(not TOOL_NAME.fullmatch(capability) for capability in permissions):
            raise PluginValidationError("plugin permissions must be exact dotted capabilities")
        execution = payload.get("execution", {})
        if not isinstance(execution, dict):
            raise PluginValidationError("execution must be a table")
        image = execution.get("image")
        if image is not None and not isinstance(image, str):
            raise PluginValidationError("execution.image must be a string")
        if image and not allow_unpinned_images and not PINNED_IMAGE.fullmatch(image):
            raise PluginValidationError("plugin execution image must be pinned by sha256 digest")

        raw_tools = payload.get("tools", [])
        if not isinstance(raw_tools, list) or any(not isinstance(item, dict) for item in raw_tools):
            raise PluginValidationError("tools must be an array of tables")
        tools = tuple(_load_tool(item) for item in raw_tools)
        skills = payload.get("skills", {})
        if not isinstance(skills, dict):
            raise PluginValidationError("skills must be a table")
        skill_files = tuple(_string_list(skills.get("files", []), "skills.files"))
        raw_agents = payload.get("agents", [])
        if not isinstance(raw_agents, list) or any(
            not isinstance(item, dict) for item in raw_agents
        ):
            raise PluginValidationError("agents must be an array of tables")
        if len(raw_agents) > MAX_AGENT_PROFILES:
            raise PluginValidationError(
                f"plugin cannot declare more than {MAX_AGENT_PROFILES} agent profiles"
            )
        agents = tuple(parse_agent_manifest(item) for item in raw_agents)
        roles = [agent.role for agent in agents]
        if len(roles) != len(set(roles)):
            raise PluginValidationError("plugin agent roles must be unique")
        multiagent = _load_multiagent(payload.get("multiagent"))
        members = [*(tool.schema_file for tool in tools), *skill_files]
        members.extend(agent.system_prompt_file for agent in agents)
        if multiagent and multiagent.coordinator_prompt_file:
            members.append(multiagent.coordinator_prompt_file)
        for relative in members:
            _resolve_member(root, relative)

        requested = set(permissions)
        for tool in tools:
            if tool.capability not in requested:
                raise PluginValidationError(
                    f"tool {tool.name} requires undeclared permission {tool.capability}"
                )
        for agent in agents:
            agent_capabilities = set(agent.default_capabilities)
            if agent.allowed_capabilities is not None:
                agent_capabilities.update(agent.allowed_capabilities)
            undeclared = agent_capabilities - requested
            if undeclared:
                raise PluginValidationError(
                    f"agent {agent.role!r} uses undeclared permissions: "
                    + ", ".join(sorted(undeclared))
                )
        return cls(
            id=plugin_id,
            version=version,
            runtime_api=runtime_api,
            description=description,
            permissions=permissions,
            image=image,
            tools=tools,
            skill_files=skill_files,
            agents=agents,
            multiagent=multiagent,
            root=root,
        )

    def member(self, relative: str) -> Path:
        return _resolve_member(self.root, relative)


def _load_tool(payload: Mapping[str, Any]) -> ToolManifest:
    name = _string(payload, "name")
    if not TOOL_NAME.fullmatch(name):
        raise PluginValidationError(f"invalid tool name: {name}")
    program = _string(payload, "program")
    if not PROGRAM.fullmatch(program):
        raise PluginValidationError(f"tool program must be a bare executable name: {program}")
    fields = tuple(_string_list(payload.get("artifact_fields", []), f"{name}.artifact_fields"))
    if not fields:
        raise PluginValidationError(f"tool {name} must declare at least one artifact field")
    argument_fields = tuple(
        _string_list(payload.get("argument_fields", []), f"{name}.argument_fields")
    )
    if len(set(fields)) != len(fields) or len(set(argument_fields)) != len(argument_fields):
        raise PluginValidationError(f"tool {name} has duplicate argument fields")
    if set(fields) & set(argument_fields):
        raise PluginValidationError(f"tool {name} reuses an artifact field as a scalar argument")
    joined = payload.get("joined_artifact_prefix")
    if joined is not None and (not isinstance(joined, str) or len(fields) != 1):
        raise PluginValidationError(
            f"tool {name} joined_artifact_prefix requires exactly one artifact field"
        )
    return ToolManifest(
        name=name,
        description=_string(payload, "description"),
        program=program,
        capability=_string(payload, "capability"),
        schema_file=_string(payload, "schema_file"),
        artifact_fields=fields,
        argument_fields=argument_fields,
        fixed_arguments=tuple(
            _string_list(payload.get("fixed_arguments", []), f"{name}.fixed_arguments")
        ),
        joined_artifact_prefix=joined,
    )


def parse_agent_manifest(payload: Mapping[str, Any]) -> AgentManifest:
    role = _string(payload, "role")
    if not AGENT_ROLE.fullmatch(role):
        raise PluginValidationError(f"invalid agent role: {role}")
    defaults = tuple(
        _string_list(payload.get("default_capabilities", []), f"agent {role}.default_capabilities")
    )
    if len(defaults) > MAX_AGENT_CAPABILITIES or len(defaults) != len(set(defaults)):
        raise PluginValidationError(
            f"agent {role!r} default capabilities must be unique and bounded"
        )
    if any(not TOOL_NAME.fullmatch(capability) for capability in defaults):
        raise PluginValidationError(f"agent {role!r} capabilities must be exact and dotted")
    capability_policy = payload.get("capability_policy", "declared")
    if capability_policy not in {"declared", "parent"}:
        raise PluginValidationError(
            f"agent {role!r} capability_policy must be 'declared' or 'parent'"
        )
    if capability_policy == "parent":
        if "allowed_capabilities" in payload:
            raise PluginValidationError(
                f"agent {role!r} cannot combine parent policy with allowed_capabilities"
            )
        allowed = None
    else:
        allowed = tuple(
            _string_list(
                payload.get("allowed_capabilities", list(defaults)),
                f"agent {role}.allowed_capabilities",
            )
        )
        if len(allowed) > MAX_AGENT_CAPABILITIES or len(allowed) != len(set(allowed)):
            raise PluginValidationError(
                f"agent {role!r} allowed capabilities must be unique and bounded"
            )
        if any(not TOOL_NAME.fullmatch(capability) for capability in allowed):
            raise PluginValidationError(f"agent {role!r} capabilities must be exact and dotted")
        if not set(defaults).issubset(allowed):
            raise PluginValidationError(f"agent {role!r} default capabilities must be allowed")
    description = _string(payload, "description")
    if len(description) > MAX_AGENT_DESCRIPTION_CHARS:
        raise PluginValidationError(
            f"agent {role!r} description exceeds {MAX_AGENT_DESCRIPTION_CHARS} characters"
        )
    return AgentManifest(
        role=role,
        description=description,
        system_prompt_file=_string(payload, "system_prompt_file"),
        default_capabilities=defaults,
        allowed_capabilities=allowed,
        max_model_turns=_positive_int(payload, "max_model_turns", 6),
        max_tool_calls=_positive_int(payload, "max_tool_calls", 8),
        max_tokens=_positive_int(payload, "max_tokens", 30_000),
        max_wall_seconds=_positive_number(payload, "max_wall_seconds", 300.0),
    )


def _load_multiagent(value: object) -> MultiAgentManifest | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PluginValidationError("multiagent must be a table")
    prompt_file = value.get("coordinator_prompt_file")
    if prompt_file is not None and (not isinstance(prompt_file, str) or not prompt_file):
        raise PluginValidationError("multiagent.coordinator_prompt_file must be a string")
    strategies: dict[str, str | None] = {}
    for field in ("routing", "scheduler", "aggregation"):
        strategy = value.get(field)
        if strategy is not None and (
            not isinstance(strategy, str) or not STRATEGY_NAME.fullmatch(strategy)
        ):
            raise PluginValidationError(f"multiagent.{field} is invalid")
        strategies[field] = strategy
    return MultiAgentManifest(
        coordinator_prompt_file=prompt_file,
        routing=strategies["routing"],
        scheduler=strategies["scheduler"],
        aggregation=strategies["aggregation"],
    )


def _resolve_member(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise PluginValidationError(f"plugin member must be relative: {relative}")
    try:
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise PluginValidationError(f"plugin member escapes package: {relative}") from error
    if not resolved.is_file():
        raise PluginValidationError(f"plugin member is not a file: {relative}")
    return resolved


def _string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise PluginValidationError(f"{key} must be a non-empty string")
    return value


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise PluginValidationError(f"{name} must be a list of non-empty strings")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PluginValidationError(f"{key} must be a positive integer")
    return value


def _positive_number(mapping: Mapping[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise PluginValidationError(f"{key} must be a positive number")
    return float(value)
