"""Plugin discovery, validation, and contribution loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from parallax.core import ArtifactRef, CommandSpec, JoinedArtifactRef
from parallax.errors import PluginValidationError
from parallax.harness.policy import AgentProfile, MultiAgentContribution
from parallax.plugins.manifest import PluginManifest, ToolManifest
from parallax.tools.base import ToolDefinition


@dataclass(frozen=True)
class PluginContribution:
    manifest: PluginManifest
    digest: str
    tools: tuple[ToolDefinition, ...]
    skills: Mapping[str, str]
    multiagent: MultiAgentContribution


@dataclass(frozen=True)
class ManifestCommandTool(ToolDefinition):
    name: str
    description: str
    program: str
    parameters: Mapping[str, Any]
    required_capability: str
    artifact_fields: tuple[str, ...]
    argument_fields: tuple[str, ...] = ()
    fixed_arguments: tuple[str, ...] = ()
    image: str | None = None
    joined_artifact_prefix: str | None = None

    def prepare(self, arguments: Mapping[str, Any]) -> CommandSpec:
        argv: list[object] = list(self.fixed_arguments)
        for field_name in self.argument_fields:
            value = arguments[field_name]
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                raise TypeError(f"plugin argument {field_name!r} must be a scalar")
            argv.append(str(value))
        for field_name in self.artifact_fields:
            artifact = arguments[field_name]
            if not isinstance(artifact, ArtifactRef):
                raise TypeError("runtime invariant violated: artifact was not resolved")
            if self.joined_artifact_prefix:
                argv.append(JoinedArtifactRef(self.joined_artifact_prefix, artifact))
            else:
                argv.append(artifact)
        return CommandSpec(
            program=self.program,
            arguments=tuple(argv),  # type: ignore[arg-type]
            image=self.image,
        )


class PluginManager:
    MAX_SCHEMA_BYTES = 1_000_000
    MAX_SKILL_BYTES = 256_000
    MAX_TOTAL_SKILL_BYTES = 1_000_000
    MAX_AGENT_PROMPT_BYTES = 64_000
    MAX_TOTAL_AGENT_PROMPT_BYTES = 512_000
    MAX_COORDINATOR_PROMPT_BYTES = 64_000

    def __init__(self, roots: list[Path], *, allow_unpinned_images: bool = False) -> None:
        self.roots = [root.resolve() for root in roots]
        self.allow_unpinned_images = allow_unpinned_images

    def discover(self) -> dict[str, PluginContribution]:
        contributions: dict[str, PluginContribution] = {}
        for root in self.roots:
            if not root.exists():
                continue
            for manifest_path in sorted(root.glob("*/plugin.toml")):
                contribution = self.load(manifest_path)
                plugin_id = contribution.manifest.id
                if plugin_id in contributions:
                    raise PluginValidationError(f"duplicate plugin id: {plugin_id}")
                contributions[plugin_id] = contribution
        return contributions

    def load(self, manifest_path: Path) -> PluginContribution:
        manifest = PluginManifest.load(
            manifest_path, allow_unpinned_images=self.allow_unpinned_images
        )
        tools = tuple(self._tool(manifest, item) for item in manifest.tools)
        skills: dict[str, str] = {}
        total_skill_bytes = 0
        for relative in manifest.skill_files:
            content = _read_limited(manifest.member(relative), self.MAX_SKILL_BYTES, "skill")
            total_skill_bytes += len(content.encode("utf-8"))
            if total_skill_bytes > self.MAX_TOTAL_SKILL_BYTES:
                raise PluginValidationError("plugin skills exceed the total size limit")
            name = Path(relative).stem
            if name in skills:
                raise PluginValidationError(f"duplicate skill name: {name}")
            skills[name] = content
        profiles_list: list[AgentProfile] = []
        total_agent_prompt_bytes = 0
        for agent in manifest.agents:
            system_prompt = _read_limited(
                manifest.member(agent.system_prompt_file),
                self.MAX_AGENT_PROMPT_BYTES,
                "agent prompt",
            )
            total_agent_prompt_bytes += len(system_prompt.encode("utf-8"))
            if total_agent_prompt_bytes > self.MAX_TOTAL_AGENT_PROMPT_BYTES:
                raise PluginValidationError("plugin agent prompts exceed the total size limit")
            profiles_list.append(
                AgentProfile(
                    role=agent.role,
                    description=agent.description,
                    system_prompt=system_prompt,
                    default_capabilities=frozenset(agent.default_capabilities),
                    allowed_capabilities=(
                        frozenset(agent.allowed_capabilities)
                        if agent.allowed_capabilities is not None
                        else None
                    ),
                    max_model_turns=agent.max_model_turns,
                    max_tool_calls=agent.max_tool_calls,
                    max_tokens=agent.max_tokens,
                    max_wall_seconds=agent.max_wall_seconds,
                    source_plugin=manifest.id,
                )
            )
        profiles = tuple(profiles_list)
        orchestration = manifest.multiagent
        coordinator_prompt = (
            _read_limited(
                manifest.member(orchestration.coordinator_prompt_file),
                self.MAX_COORDINATOR_PROMPT_BYTES,
                "coordinator prompt",
            )
            if orchestration and orchestration.coordinator_prompt_file
            else ""
        )
        multiagent = MultiAgentContribution(
            plugin_id=manifest.id,
            profiles=profiles,
            coordinator_prompt=coordinator_prompt,
            routing=orchestration.routing if orchestration else None,
            scheduler=orchestration.scheduler if orchestration else None,
            aggregation=orchestration.aggregation if orchestration else None,
        )
        digest = self._digest(manifest)
        return PluginContribution(
            manifest=manifest,
            digest=digest,
            tools=tools,
            skills=skills,
            multiagent=multiagent,
        )

    @staticmethod
    def _tool(manifest: PluginManifest, item: ToolManifest) -> ToolDefinition:
        try:
            schema = json.loads(
                _read_limited(
                    manifest.member(item.schema_file),
                    PluginManager.MAX_SCHEMA_BYTES,
                    "schema",
                )
            )
            Draft202012Validator.check_schema(schema)
        except (OSError, UnicodeError, json.JSONDecodeError, SchemaError) as error:
            raise PluginValidationError(f"invalid schema for tool {item.name}: {error}") from error
        if not isinstance(schema, dict):
            raise PluginValidationError(f"schema for tool {item.name} must be an object")
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            raise PluginValidationError(
                f"schema for tool {item.name} must be a closed object schema"
            )
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for field in (*item.argument_fields, *item.artifact_fields):
            if field not in properties or field not in required:
                raise PluginValidationError(
                    f"argument field {field!r} must be required by schema for {item.name}"
                )
            field_schema = properties[field]
            if not isinstance(field_schema, dict):
                raise PluginValidationError(f"schema field {field!r} must be an object")
            if field in item.artifact_fields and field_schema.get("type") != "string":
                raise PluginValidationError(f"artifact field {field!r} must have string type")
            if field in item.argument_fields and field_schema.get("type") not in {
                "string",
                "integer",
                "number",
            }:
                raise PluginValidationError(
                    f"scalar argument field {field!r} has an unsupported type"
                )
        return ManifestCommandTool(
            name=item.name,
            description=item.description,
            program=item.program,
            parameters=schema,
            required_capability=item.capability,
            artifact_fields=item.artifact_fields,
            argument_fields=item.argument_fields,
            fixed_arguments=item.fixed_arguments,
            joined_artifact_prefix=item.joined_artifact_prefix,
            image=manifest.image,
        )

    @staticmethod
    def _digest(manifest: PluginManifest) -> str:
        digest = hashlib.sha256()
        paths = [manifest.root / "plugin.toml"]
        paths.extend(manifest.member(tool.schema_file) for tool in manifest.tools)
        paths.extend(manifest.member(skill) for skill in manifest.skill_files)
        paths.extend(manifest.member(agent.system_prompt_file) for agent in manifest.agents)
        if manifest.multiagent and manifest.multiagent.coordinator_prompt_file:
            paths.append(manifest.member(manifest.multiagent.coordinator_prompt_file))
        for path in sorted(paths):
            digest.update(path.relative_to(manifest.root).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()


def load_builtin_multiagent_contribution() -> PluginContribution:
    manifest = Path(__file__).with_name("builtin") / "ganymede-default" / "plugin.toml"
    return PluginManager([]).load(manifest)


def _read_limited(path: Path, limit: int, kind: str) -> str:
    try:
        if path.stat().st_size > limit:
            raise PluginValidationError(
                f"plugin {kind} exceeds the {limit}-byte limit: {path.name}"
            )
        return path.read_text(encoding="utf-8")
    except PluginValidationError:
        raise
    except (OSError, UnicodeError) as error:
        raise PluginValidationError(f"plugin {kind} is not valid UTF-8: {path.name}") from error
