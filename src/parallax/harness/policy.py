"""Composable declarative policy for Ganymede multi-agent behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MAX_COMPOSED_AGENT_PROFILES = 64
MAX_COMPOSED_COORDINATOR_PROMPT_BYTES = 256_000


@dataclass(frozen=True)
class AgentProfile:
    role: str
    description: str
    system_prompt: str
    default_capabilities: frozenset[str]
    allowed_capabilities: frozenset[str] | None
    max_model_turns: int
    max_tool_calls: int
    max_tokens: int
    max_wall_seconds: float
    source_plugin: str


@dataclass(frozen=True)
class MultiAgentContribution:
    plugin_id: str
    profiles: tuple[AgentProfile, ...] = ()
    coordinator_prompt: str = ""
    routing: str | None = None
    scheduler: str | None = None
    aggregation: str | None = None


@dataclass(frozen=True)
class MultiAgentPolicy:
    profiles: tuple[AgentProfile, ...]
    coordinator_prompt: str
    routing: str
    scheduler: str
    aggregation: str
    source_plugins: tuple[str, ...]


def compose_multiagent_policy(
    contributions: Iterable[MultiAgentContribution],
) -> MultiAgentPolicy:
    """Compose enabled contributions in order, rejecting ambiguous role ownership."""

    profiles: dict[str, AgentProfile] = {}
    prompt_fragments: list[str] = []
    sources: list[str] = []
    routing = "coordinator"
    scheduler = "serial"
    aggregation = "coordinator-evidence"
    for contribution in contributions:
        if not (
            contribution.profiles
            or contribution.coordinator_prompt
            or contribution.routing
            or contribution.scheduler
            or contribution.aggregation
        ):
            continue
        sources.append(contribution.plugin_id)
        if contribution.coordinator_prompt:
            prompt_fragments.append(contribution.coordinator_prompt.strip())
        routing = contribution.routing or routing
        scheduler = contribution.scheduler or scheduler
        aggregation = contribution.aggregation or aggregation
        for profile in contribution.profiles:
            if len(profiles) >= MAX_COMPOSED_AGENT_PROFILES:
                raise ValueError(
                    f"multi-agent policy cannot exceed {MAX_COMPOSED_AGENT_PROFILES} profiles"
                )
            if profile.role in profiles:
                owner = profiles[profile.role].source_plugin
                raise ValueError(
                    f"duplicate agent role {profile.role!r} from plugins "
                    f"{owner!r} and {profile.source_plugin!r}"
                )
            profiles[profile.role] = profile
    if not profiles:
        raise ValueError("multi-agent mode requires at least one enabled agent profile")
    coordinator_prompt = "\n\n".join(prompt_fragments)
    if len(coordinator_prompt.encode("utf-8")) > MAX_COMPOSED_COORDINATOR_PROMPT_BYTES:
        raise ValueError("composed coordinator prompt exceeds the size limit")
    return MultiAgentPolicy(
        profiles=tuple(profiles[role] for role in sorted(profiles)),
        coordinator_prompt=coordinator_prompt,
        routing=routing,
        scheduler=scheduler,
        aggregation=aggregation,
        source_plugins=tuple(dict.fromkeys(sources)),
    )
