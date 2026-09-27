"""Bounded standalone Agent definitions authored by local users."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from parallax.errors import PluginValidationError
from parallax.harness.policy import AgentProfile, MultiAgentContribution
from parallax.plugins.manifest import parse_agent_manifest


MAX_USER_AGENT_BYTES = 128 * 1024
MAX_USER_AGENT_PROMPT_BYTES = 64 * 1024
_AGENT_FIELDS = {
    "role",
    "description",
    "system_prompt",
    "default_capabilities",
    "allowed_capabilities",
    "capability_policy",
    "max_model_turns",
    "max_tool_calls",
    "max_tokens",
    "max_wall_seconds",
}


@dataclass(frozen=True)
class UserAgentFile:
    path: Path
    digest: str
    contribution: MultiAgentContribution


def load_user_agent(path: Path) -> UserAgentFile:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or not resolved.name.endswith(".agent.toml"):
        raise PluginValidationError(f"user Agent must be an .agent.toml file: {path}")
    raw = resolved.read_bytes()
    if len(raw) > MAX_USER_AGENT_BYTES:
        raise PluginValidationError("user Agent file exceeds 128 KiB")
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise PluginValidationError(f"invalid user Agent file {path}: {error}") from error
    if set(payload) - {"agent", "coordinator"}:
        raise PluginValidationError("user Agent file contains unsupported top-level tables")
    agent = payload.get("agent")
    if not isinstance(agent, dict):
        raise PluginValidationError("user Agent file requires an [agent] table")
    unknown = set(agent) - _AGENT_FIELDS
    if unknown:
        raise PluginValidationError(
            "user Agent contains unsupported fields: " + ", ".join(sorted(unknown))
        )
    system_prompt = agent.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise PluginValidationError("agent.system_prompt must be a non-empty string")
    if len(system_prompt.encode("utf-8")) > MAX_USER_AGENT_PROMPT_BYTES:
        raise PluginValidationError("agent.system_prompt exceeds 64 KiB")
    manifest_payload = dict(agent)
    manifest_payload.pop("system_prompt")
    manifest_payload["system_prompt_file"] = "<inline>"
    parsed = parse_agent_manifest(manifest_payload)

    coordinator = payload.get("coordinator", {})
    if not isinstance(coordinator, dict) or set(coordinator) - {"prompt"}:
        raise PluginValidationError("coordinator must contain only an optional prompt")
    coordinator_prompt = coordinator.get("prompt", "")
    if not isinstance(coordinator_prompt, str):
        raise PluginValidationError("coordinator.prompt must be a string")
    if len(coordinator_prompt.encode("utf-8")) > MAX_USER_AGENT_PROMPT_BYTES:
        raise PluginValidationError("coordinator.prompt exceeds 64 KiB")

    digest = hashlib.sha256(raw).hexdigest()
    source = f"user-agent-{parsed.role}-{digest[:12]}"
    profile = AgentProfile(
        role=parsed.role,
        description=parsed.description,
        system_prompt=system_prompt.strip(),
        default_capabilities=frozenset(parsed.default_capabilities),
        allowed_capabilities=(
            frozenset(parsed.allowed_capabilities)
            if parsed.allowed_capabilities is not None
            else None
        ),
        max_model_turns=parsed.max_model_turns,
        max_tool_calls=parsed.max_tool_calls,
        max_tokens=parsed.max_tokens,
        max_wall_seconds=parsed.max_wall_seconds,
        source_plugin=source,
    )
    return UserAgentFile(
        path=resolved,
        digest=digest,
        contribution=MultiAgentContribution(
            plugin_id=source,
            profiles=(profile,),
            coordinator_prompt=coordinator_prompt.strip(),
        ),
    )
