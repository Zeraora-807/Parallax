"""Command line entry point for Parallax Ganymede."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from parallax import __codename__, __version__
from parallax.evolution import EvolutionManager, evolution_tools
from parallax.executor import DockerCandidateValidator, DockerExecutor, LocalExecutor
from parallax.generations import BackupManager, GenerationManager, Supervisor
from parallax.harness.agent import DEFAULT_SYSTEM_PROMPT, AgentLoop
from parallax.harness.cached_model import CachedChatModel
from parallax.harness.multiagent import (
    AgentDelegateTool,
    DelegationController,
    SUPPORTED_AGGREGATIONS,
    SUPPORTED_ROUTING,
    SUPPORTED_SCHEDULERS,
)
from parallax.harness.policy import MultiAgentPolicy, compose_multiagent_policy
from parallax.harness.user_agents import UserAgentFile, load_user_agent
from parallax.harness.openai_compatible import OpenAICompatibleClient
from parallax.plugins import (
    PluginContribution,
    PluginManager,
    Preset,
    load_builtin_multiagent_contribution,
)
from parallax.runtime import (
    ArtifactStore,
    BudgetLedger,
    PermissionSet,
    RunBudget,
    RunContext,
    Runtime,
    Workspace,
)
from parallax.runtime.engine import POLICY_VERSION
from parallax.runtime.cache import CACHE_FORMAT_VERSION, SQLiteCacheStore
from parallax.runtime.run_lock import RunLock
from parallax.tools import ToolRegistry, builtin_binary_tools
from parallax.tracing import SQLiteEventStore
from parallax.ui import AfterglowArgumentParser, AfterglowUI


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _agent_count(value: str) -> int:
    parsed = int(value)
    if not 2 <= parsed <= 8:
        raise argparse.ArgumentTypeError("value must be between 2 and 8")
    return parsed


def _safe_endpoint(value: str) -> dict[str, object]:
    parsed = urlsplit(value)
    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path,
    }


def _model_cache_namespace(
    base_url: str,
    model: str,
    api_key: str | None,
    max_response_bytes: int,
) -> str:
    return json.dumps(
        {
            "format": CACHE_FORMAT_VERSION,
            "policy": POLICY_VERSION,
            # The complete endpoint and credential affect provider behavior. Hash them so
            # cache isolation does not persist secrets embedded in either value.
            "endpoint_sha256": hashlib.sha256(base_url.rstrip("/").encode("utf-8")).hexdigest(),
            "credential_sha256": hashlib.sha256((api_key or "").encode("utf-8")).hexdigest(),
            "model": model,
            "max_response_bytes": max_response_bytes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_skill_files(paths: list[Path]) -> dict[str, str]:
    skills: dict[str, str] = {}
    total_bytes = 0
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file() or resolved.suffix.casefold() != ".md":
            raise ValueError(f"skill must be a Markdown file: {path}")
        name = resolved.stem
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError(f"invalid skill name: {name}")
        if name in skills:
            raise ValueError(f"duplicate skill name: {name}")
        raw = resolved.read_bytes()
        if len(raw) > 64 * 1024:
            raise ValueError(f"skill exceeds 64 KiB: {name}")
        total_bytes += len(raw)
        if total_bytes > 256 * 1024:
            raise ValueError("skills exceed 256 KiB in total")
        try:
            skills[name] = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"skill is not valid UTF-8: {name}") from error
    return skills


def _load_bounded_text(path: Path, max_bytes: int, label: str) -> str:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file")
    with resolved.open("rb") as source:
        raw = source.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error


def build_parser() -> argparse.ArgumentParser:
    parser = AfterglowArgumentParser(
        prog="parallax",
        description="Artifact-scoped agent runtime for adversarial analysis.",
        epilog="Environment: NO_COLOR disables ANSI; FORCE_COLOR=1 enables it.",
    )
    parser.add_argument(
        "--version", action="version", version=f"Parallax {__codename__} {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    status = subparsers.add_parser("status", help="show runtime state and recent runs")
    status.add_argument("--state-root", type=Path, required=True)
    status.add_argument("--json", action="store_true", help="emit stable machine-readable JSON")

    boot = subparsers.add_parser("boot", help="launch the active immutable generation")
    boot.add_argument("--state-root", type=Path, required=True)
    boot.add_argument("--rollback-on-failure", action="store_true")
    boot.add_argument("arguments", nargs=argparse.REMAINDER)

    run = subparsers.add_parser("run", help="run an artifact-scoped analysis")
    run.add_argument("prompt")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--input", action="append", default=[], help="workspace-relative input")
    run.add_argument("--executor", choices=("docker", "local"), default="docker")
    run.add_argument("--docker-image", default="parallax-tools:latest")
    run.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--model", required=True)
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--max-steps", type=_positive_int, default=12)
    run.add_argument("--max-seconds", type=_positive_float, default=900)
    run.add_argument("--max-tool-calls", type=_positive_int, default=24)
    run.add_argument("--max-tool-seconds", type=_positive_float, default=60)
    run.add_argument("--max-tokens", type=_positive_int, default=200_000)
    run.add_argument("--max-output-bytes", type=_positive_int, default=1_000_000)
    run.add_argument("--max-observation-bytes", type=_positive_int, default=32_000)
    run.add_argument("--max-artifact-bytes", type=_positive_int, default=512 * 1024 * 1024)
    run.add_argument(
        "--max-total-artifact-bytes", type=_positive_int, default=2 * 1024 * 1024 * 1024
    )
    run.add_argument("--max-model-response-bytes", type=_positive_int, default=10_000_000)
    run.add_argument("--no-cache", action="store_true", help="disable persistent exact caches")
    run.add_argument(
        "--multi-agent",
        action="store_true",
        help="enable bounded coordinator-to-specialist delegation",
    )
    run.add_argument(
        "--max-agents",
        type=_agent_count,
        default=4,
        help="maximum agents in a multi-agent run, including the coordinator (2-8)",
    )
    run.add_argument(
        "--agent-file",
        action="append",
        type=Path,
        default=[],
        help="standalone .agent.toml profile to add (repeatable; requires --multi-agent)",
    )
    run.add_argument("--docker-memory", default="512m")
    run.add_argument("--docker-cpus", type=_positive_float, default=1.0)
    run.add_argument("--docker-pids", type=_positive_int, default=64)
    run.add_argument("--allow", action="append", default=[], help="additional exact capability")
    run.add_argument("--plugin-root", action="append", type=Path, default=[])
    run.add_argument("--plugin", action="append", default=[], help="enabled plugin id")
    run.add_argument("--preset", type=Path)
    run.add_argument(
        "--skill-file",
        action="append",
        type=Path,
        default=[],
        help="declarative Markdown skill file (repeatable)",
    )
    run.add_argument("--memory-database", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--memory-source-text", help=argparse.SUPPRESS)
    run.add_argument("--memory-source-file", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--prompt-file", type=Path, help=argparse.SUPPRESS)
    run.add_argument(
        "--evolution-mode",
        choices=("off", "candidate", "trial", "auto"),
        default="off",
    )
    run.add_argument(
        "--evolution-source-root",
        type=Path,
        help="source tree copied into candidates (required when evolution is enabled)",
    )
    run.add_argument("--validator-image", default="parallax-ganymede-validator:latest")

    plugin = subparsers.add_parser("plugin", help="inspect declarative plugins")
    plugin_sub = plugin.add_subparsers(dest="plugin_command", required=True)
    plugin_list = plugin_sub.add_parser("list")
    plugin_list.add_argument("--root", action="append", type=Path, required=True)
    plugin_validate = plugin_sub.add_parser("validate")
    plugin_validate.add_argument("manifest", type=Path)
    plugin_validate.add_argument("--allow-unpinned-image", action="store_true")

    generation = subparsers.add_parser("generation", help="manage immutable generations")
    generation_sub = generation.add_subparsers(dest="generation_command", required=True)
    for name in ("list", "rollback"):
        command = generation_sub.add_parser(name)
        command.add_argument("--state-root", type=Path, required=True)
    generation_create = generation_sub.add_parser("create")
    generation_create.add_argument("--state-root", type=Path, required=True)
    generation_create.add_argument("--source-root", type=Path, required=True)
    generation_validate = generation_sub.add_parser("validate")
    generation_validate.add_argument("generation_id")
    generation_validate.add_argument("--state-root", type=Path, required=True)
    generation_validate.add_argument(
        "--validator-image", default="parallax-ganymede-validator:latest"
    )
    generation_activate = generation_sub.add_parser("activate")
    generation_activate.add_argument("generation_id")
    generation_activate.add_argument("--state-root", type=Path, required=True)

    backup = subparsers.add_parser("backup", help="create or verify a state backup")
    backup_sub = backup.add_subparsers(dest="backup_command", required=True)
    backup_create = backup_sub.add_parser("create")
    backup_create.add_argument("--state-root", type=Path, required=True)
    backup_create.add_argument("--output", type=Path, required=True)
    backup_verify = backup_sub.add_parser("verify")
    backup_verify.add_argument("archive", type=Path)
    backup_restore = backup_sub.add_parser("restore")
    backup_restore.add_argument("archive", type=Path)
    backup_restore.add_argument("--destination", type=Path, required=True)

    artifact = subparsers.add_parser("artifact", help="inspect or export stored artifacts")
    artifact_sub = artifact.add_subparsers(dest="artifact_command", required=True)
    for name in ("show", "export"):
        command = artifact_sub.add_parser(name)
        command.add_argument("artifact_id")
        command.add_argument("--state-root", type=Path, required=True)
        if name == "export":
            command.add_argument("--output", type=Path, required=True)

    logs = subparsers.add_parser("logs", help="inspect or export structured run events")
    logs_sub = logs.add_subparsers(dest="logs_command", required=True)
    for name in ("show", "export"):
        command = logs_sub.add_parser(name)
        command.add_argument("run_id")
        command.add_argument("--database", type=Path, required=True)
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
    web = subparsers.add_parser("web", help="start the local WebUI")
    web.add_argument("--data-root", type=Path, default=Path(".parallax-web"))
    web.add_argument("--skill-root", type=Path, default=Path("skills"))
    web.add_argument("--agent-root", type=Path, default=Path("agents"))
    web.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    ui = AfterglowUI(sys.stderr)
    setattr(arguments, "ui", ui)
    if arguments.command is None:
        parser.print_help()
        return 0
    try:
        handlers = {
            "status": _status,
            "boot": _boot,
            "run": _run,
            "plugin": _plugin,
            "generation": _generation,
            "backup": _backup,
            "artifact": _artifact,
            "logs": _logs,
            "web": _web,
        }
        return handlers[arguments.command](arguments)
    except KeyboardInterrupt:
        ui.error("interrupted by operator")
        return 130
    except Exception as error:
        ui.error(error)
        return 1


def _status(arguments: argparse.Namespace) -> int:
    snapshot = _status_snapshot(arguments.state_root)
    if arguments.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        AfterglowUI(sys.stdout).dashboard(snapshot)
    return 0


def _web(arguments: argparse.Namespace) -> int:
    if not 0 <= arguments.port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    from parallax.web import serve

    serve(
        arguments.data_root,
        arguments.skill_root,
        arguments.host,
        arguments.port,
        open_browser=not arguments.no_browser,
        agent_root=arguments.agent_root,
    )
    return 0


def _status_snapshot(state_root: Path) -> dict[str, object]:
    state_root = state_root.resolve(strict=True)
    if not state_root.is_dir():
        raise ValueError(f"state root is not a directory: {state_root}")
    active: str | None = None
    active_path = state_root / "profiles" / "active.json"
    if active_path.exists():
        active = str(json.loads(active_path.read_text(encoding="utf-8"))["generation_id"])
    generation_root = state_root / "generations"
    generation_count = (
        sum(1 for item in generation_root.iterdir() if item.is_dir())
        if generation_root.is_dir()
        else 0
    )
    object_root = state_root / "artifacts" / "objects"
    artifact_files = list(object_root.glob("*/*")) if object_root.is_dir() else []
    artifact_files = [
        item
        for item in artifact_files
        if item.is_file()
        and len(item.name) == 64
        and all(character in "0123456789abcdef" for character in item.name)
    ]
    event_database = state_root / "events.sqlite3"
    recent_runs = (
        SQLiteEventStore(event_database).recent_runs(limit=5) if event_database.is_file() else []
    )
    cache_database = state_root / "cache.sqlite3"
    cache_stats: dict[str, int] | None = None
    if cache_database.is_file():
        try:
            cache_stats = SQLiteCacheStore(cache_database).stats()
        except (OSError, sqlite3.DatabaseError):
            cache_stats = None
    return {
        "version": __version__,
        "state_root": str(state_root),
        "active_generation": active,
        "generation_count": generation_count,
        "artifact_count": len(artifact_files),
        "artifact_bytes": sum(item.stat().st_size for item in artifact_files),
        "cache": cache_stats,
        "recent_runs": recent_runs,
    }


def _boot(arguments: argparse.Namespace) -> int:
    child_arguments = list(arguments.arguments)
    if child_arguments[:1] == ["--"]:
        child_arguments = child_arguments[1:]
    if not child_arguments:
        raise ValueError("boot requires a Parallax command after '--'")
    events = SQLiteEventStore(arguments.state_root / "events.sqlite3")
    supervisor = Supervisor(GenerationManager(arguments.state_root), events)
    return supervisor.launch(
        child_arguments,
        rollback_on_failure=arguments.rollback_on_failure,
    )


def _run(arguments: argparse.Namespace) -> int:
    workspace = Workspace(arguments.workspace)
    run_id = uuid.uuid4().hex
    with RunLock(workspace.root / ".parallax" / "run-locks" / f"{run_id}.lock"):
        return _run_locked(arguments, run_id)


def _run_locked(arguments: argparse.Namespace, run_id: str) -> int:
    ui: AfterglowUI = arguments.ui
    if any("*" in capability for capability in arguments.allow):
        raise ValueError("CLI capabilities must be exact; wildcards are not accepted")
    if arguments.multi_agent and not 2 <= arguments.max_agents <= 8:
        raise ValueError("multi-agent mode requires --max-agents between 2 and 8")
    if arguments.agent_file and not arguments.multi_agent:
        raise ValueError("--agent-file requires --multi-agent")
    prompt = (
        _load_bounded_text(arguments.prompt_file, 1024 * 1024, "prompt file")
        if arguments.prompt_file
        else arguments.prompt
    )
    memory_source = (
        _load_bounded_text(arguments.memory_source_file, 80_000, "memory source file")
        if arguments.memory_source_file
        else arguments.memory_source_text or prompt
    )
    workspace = Workspace(arguments.workspace)
    state_root = workspace.root / ".parallax"
    event_database = state_root / "events.sqlite3"
    events = SQLiteEventStore(event_database)
    generations = GenerationManager(state_root)
    Supervisor(generations, events).recover_unfinished_runs()

    budget_config = RunBudget(
        max_wall_seconds=arguments.max_seconds,
        max_tool_calls=arguments.max_tool_calls,
        max_model_turns=arguments.max_steps,
        max_total_tokens=arguments.max_tokens,
        max_tool_seconds=arguments.max_tool_seconds,
        max_output_bytes=arguments.max_output_bytes,
        max_observation_bytes=arguments.max_observation_bytes,
        max_artifact_bytes=arguments.max_artifact_bytes,
        max_total_artifact_bytes=arguments.max_total_artifact_bytes,
    )
    enabled_plugins = list(arguments.plugin)
    capabilities = ["binary.inspect", *arguments.allow]
    if arguments.multi_agent:
        capabilities.append("agent.delegate")
    selected_skills: dict[str, str] = {}
    preset = Preset.load(arguments.preset) if arguments.preset else None
    if preset:
        enabled_plugins.extend(preset.plugins)
        capabilities.extend(preset.capabilities)

    contributions = PluginManager(arguments.plugin_root).discover()
    registry = ToolRegistry(builtin_binary_tools())
    plugin_lock: dict[str, dict[str, str]] = {}
    enabled_contributions: list[PluginContribution] = []
    for plugin_id in dict.fromkeys(enabled_plugins):
        if plugin_id not in contributions:
            raise ValueError(f"enabled plugin was not found: {plugin_id}")
        contribution = contributions[plugin_id]
        enabled_contributions.append(contribution)
        plugin_lock[plugin_id] = {
            "version": contribution.manifest.version,
            "digest": contribution.digest,
        }
        for tool in contribution.tools:
            registry.register(tool)
        selected_skills.update(contribution.skills)
    if preset:
        selected_skills = {
            name: content for name, content in selected_skills.items() if name in preset.skills
        }
    standalone_skills = _load_skill_files(arguments.skill_file)
    duplicate_skills = set(selected_skills) & set(standalone_skills)
    if duplicate_skills:
        raise ValueError(f"duplicate skill name: {sorted(duplicate_skills)[0]}")
    selected_skills.update(standalone_skills)
    if arguments.memory_database:
        from parallax.memory import MemoryStore, RememberTool

        registry.register(
            RememberTool(
                MemoryStore(arguments.memory_database),
                memory_source,
            )
        )
        capabilities.append("memory.write")

    multiagent_policy: MultiAgentPolicy | None = None
    builtin_multiagent = None
    user_agents: list[UserAgentFile] = []
    if arguments.multi_agent:
        builtin_multiagent = load_builtin_multiagent_contribution()
        user_agents = [load_user_agent(path) for path in arguments.agent_file]
        multiagent_policy = compose_multiagent_policy(
            [builtin_multiagent.multiagent]
            + [contribution.multiagent for contribution in enabled_contributions]
            + [agent.contribution for agent in user_agents]
        )
        if multiagent_policy.routing not in SUPPORTED_ROUTING:
            raise ValueError(
                f"unsupported multi-agent routing strategy: {multiagent_policy.routing}"
            )
        if multiagent_policy.scheduler not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"unsupported multi-agent scheduler: {multiagent_policy.scheduler}")
        if multiagent_policy.aggregation not in SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"unsupported multi-agent aggregation strategy: {multiagent_policy.aggregation}"
            )

    executor = (
        DockerExecutor(
            image=arguments.docker_image,
            max_output_bytes=arguments.max_output_bytes,
            memory=arguments.docker_memory,
            cpus=str(arguments.docker_cpus),
            pids_limit=arguments.docker_pids,
        )
        if arguments.executor == "docker"
        else LocalExecutor(max_output_bytes=arguments.max_output_bytes)
    )
    capabilities.append(executor.required_capability)
    cache_error: str | None = None
    cache = None
    if not arguments.no_cache:
        try:
            cache = SQLiteCacheStore(state_root / "cache.sqlite3")
        except (OSError, sqlite3.Error) as error:
            cache_error = type(error).__name__
            ui.status("cache", f"disabled after {cache_error}", state="warning")
    cache_namespace = json.dumps(
        {
            "format": CACHE_FORMAT_VERSION,
            "policy": POLICY_VERSION,
            "executor": arguments.executor,
            "image": arguments.docker_image if arguments.executor == "docker" else "local",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    if arguments.evolution_mode != "off":
        if arguments.evolution_source_root is None:
            raise ValueError("--evolution-source-root is required when evolution is enabled")
        evolution = EvolutionManager(
            source_root=arguments.evolution_source_root,
            generations=generations,
            mode=arguments.evolution_mode,
            validator=DockerCandidateValidator(image=arguments.validator_image),
        )
        for tool in evolution_tools(evolution):
            registry.register(tool)
        capabilities.append("evolution.propose")
        if arguments.evolution_mode in {"trial", "auto"}:
            capabilities.append("evolution.validate")
        if arguments.evolution_mode == "auto":
            capabilities.append("evolution.activate")

    permissions = PermissionSet.from_iterable(capabilities)
    generation_id = generations.active_id()
    events.create_run(
        run_id,
        generation_id=generation_id,
        policy_version=POLICY_VERSION,
        config={
            "executor": arguments.executor,
            "executor_image": arguments.docker_image if arguments.executor == "docker" else None,
            "model": arguments.model,
            "model_endpoint": _safe_endpoint(arguments.base_url),
            "plugins": plugin_lock,
            "preset": preset.name if preset else None,
            "skills": sorted(selected_skills),
            "capabilities": sorted(permissions.capabilities),
            "evolution_mode": arguments.evolution_mode,
            "multi_agent": arguments.multi_agent,
            "max_agents": arguments.max_agents if arguments.multi_agent else 1,
            "multiagent_policy": (
                {
                    "profiles": [profile.role for profile in multiagent_policy.profiles],
                    "routing": multiagent_policy.routing,
                    "scheduler": multiagent_policy.scheduler,
                    "aggregation": multiagent_policy.aggregation,
                    "plugins": list(multiagent_policy.source_plugins),
                    "builtin_digest": builtin_multiagent.digest,
                }
                if multiagent_policy is not None and builtin_multiagent is not None
                else None
            ),
            "user_agents": [
                {
                    "role": agent.contribution.profiles[0].role,
                    "source_name": agent.path.name,
                    "digest": agent.digest,
                }
                for agent in user_agents
            ],
            "requested_inputs": list(arguments.input),
            "budget": budget_config.__dict__,
            "cache": {
                "enabled": cache is not None,
                "format": CACHE_FORMAT_VERSION,
                "error": cache_error,
            },
        },
    )
    model: OpenAICompatibleClient | CachedChatModel | None = None
    try:
        artifacts = ArtifactStore(
            state_root / "artifacts",
            max_artifact_bytes=budget_config.max_artifact_bytes,
            max_total_artifact_bytes=budget_config.max_total_artifact_bytes,
        )
        input_refs = []
        for path in arguments.input:
            ref = artifacts.import_workspace_file(workspace, path, run_id)
            input_refs.append(ref)
            events.record(
                run_id,
                "artifact.imported",
                "ok",
                {
                    "artifact_id": ref.id,
                    "source_name": ref.source_name,
                    "size": ref.size,
                    "policy_version": POLICY_VERSION,
                },
            )

        ui.run_started(
            run_id,
            workspace=workspace.root,
            executor=arguments.executor,
            model=arguments.model,
            inputs=len(input_refs),
        )
        api_key = os.environ.get(arguments.api_key_env)
        raw_model = OpenAICompatibleClient(
            base_url=arguments.base_url,
            model=arguments.model,
            api_key=api_key,
            max_response_bytes=arguments.max_model_response_bytes,
        )
        model = (
            CachedChatModel(
                raw_model,
                cache,
                _model_cache_namespace(
                    arguments.base_url,
                    arguments.model,
                    api_key,
                    arguments.max_model_response_bytes,
                ),
            )
            if cache is not None
            else raw_model
        )
        system_prompt = DEFAULT_SYSTEM_PROMPT
        if arguments.multi_agent:
            if multiagent_policy is None:
                raise RuntimeError("multi-agent policy was not composed")
            worker_runtime = Runtime(
                registry=registry.copy(),
                executor=executor,
                tracer=events,
                cache=cache,
                cache_namespace=cache_namespace,
            )
            registry.register(
                AgentDelegateTool(
                    model=model,
                    worker_runtime=worker_runtime,
                    controller=DelegationController(max_agents=arguments.max_agents),
                    profiles=multiagent_policy.profiles,
                    routing=multiagent_policy.routing,
                    scheduler=multiagent_policy.scheduler,
                    aggregation=multiagent_policy.aggregation,
                    skills=selected_skills,
                )
            )
            system_prompt += "\n\n" + multiagent_policy.coordinator_prompt
        runtime = Runtime(
            registry=registry,
            executor=executor,
            tracer=events,
            cache=cache,
            cache_namespace=cache_namespace,
        )
        context = RunContext(
            run_id=run_id,
            workspace=workspace,
            permissions=permissions,
            artifacts=artifacts,
            budget=BudgetLedger(budget_config),
            policy_version=POLICY_VERSION,
            generation_id=generation_id,
        )
        answer = AgentLoop(
            model=model,
            runtime=runtime,
            max_steps=arguments.max_steps,
            system_prompt=system_prompt,
            skills=selected_skills,
            parallel_delegations=(
                multiagent_policy is not None and multiagent_policy.scheduler == "parallel"
            ),
        ).run(prompt, context)
    except Exception as error:
        events.record(
            run_id,
            "run.failed",
            "error",
            {"reason_code": type(error).__name__, "policy_version": POLICY_VERSION},
        )
        events.finish_run(run_id, "failed")
        ui.error(error)
        ui.run_finished(run_id, event_database, ok=False)
        return 1
    finally:
        if model is not None:
            model.close()

    events.finish_run(run_id, "completed")
    print(ui.safe_model_text(answer, sys.stdout))
    ui.run_finished(run_id, event_database, ok=True)
    return 0


def _plugin(arguments: argparse.Namespace) -> int:
    if arguments.plugin_command == "validate":
        contribution = PluginManager(
            [arguments.manifest.parent],
            allow_unpinned_images=arguments.allow_unpinned_image,
        ).load(arguments.manifest)
        print(json.dumps(_plugin_summary(contribution), ensure_ascii=False, indent=2))
        return 0
    contributions = PluginManager(arguments.root).discover()
    print(
        json.dumps(
            [_plugin_summary(item) for item in contributions.values()],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _plugin_summary(contribution: PluginContribution) -> dict[str, object]:
    manifest = contribution.manifest
    return {
        "id": manifest.id,
        "version": manifest.version,
        "digest": contribution.digest,
        "permissions_requested": list(manifest.permissions),
        "tools": [tool.name for tool in contribution.tools],
        "skills": list(contribution.skills),
        "agents": [profile.role for profile in contribution.multiagent.profiles],
        "multiagent": {
            "routing": contribution.multiagent.routing,
            "scheduler": contribution.multiagent.scheduler,
            "aggregation": contribution.multiagent.aggregation,
            "coordinator_prompt": bool(contribution.multiagent.coordinator_prompt),
        },
    }


def _generation(arguments: argparse.Namespace) -> int:
    manager = GenerationManager(arguments.state_root)
    if arguments.generation_command == "create":
        generation = manager.create_candidate(arguments.source_root)
    elif arguments.generation_command == "validate":
        validator = DockerCandidateValidator(image=arguments.validator_image)
        generation = manager.get(arguments.generation_id)
        report = validator.validate(generation.path / "source")
        evidence = {
            "exit_code": report.result.exit_code,
            "stdout": report.result.stdout[-16_000:],
            "stderr": report.result.stderr[-16_000:],
        }
        generation = (
            manager.mark_validated(generation.id, evidence)
            if report.passed
            else manager.mark_rejected(generation.id, evidence)
        )
    elif arguments.generation_command == "activate":
        events = SQLiteEventStore(arguments.state_root / "events.sqlite3")
        generation = Supervisor(manager, events).activate(arguments.generation_id)
    elif arguments.generation_command == "rollback":
        events = SQLiteEventStore(arguments.state_root / "events.sqlite3")
        generation = Supervisor(manager, events).rollback("operator_request")
    else:
        print(
            json.dumps(
                [
                    {"id": item.id, "status": item.status, "active": item.id == manager.active_id()}
                    for item in manager.list()
                ],
                indent=2,
            )
        )
        return 0
    print(json.dumps({"id": generation.id, "status": generation.status}, indent=2))
    return 0


def _backup(arguments: argparse.Namespace) -> int:
    if arguments.backup_command == "verify":
        print(json.dumps(BackupManager.verify(arguments.archive), ensure_ascii=False, indent=2))
        return 0
    if arguments.backup_command == "restore":
        print(BackupManager.restore_to(arguments.archive, arguments.destination))
        return 0
    database = arguments.state_root / "events.sqlite3"
    path = BackupManager(arguments.state_root, database).create(arguments.output)
    print(path)
    return 0


def _logs(arguments: argparse.Namespace) -> int:
    events = SQLiteEventStore(arguments.database)
    if arguments.logs_command == "export":
        events.export_jsonl(arguments.run_id, arguments.output)
        print(arguments.output)
        return 0
    print(
        json.dumps(
            {"run": events.run(arguments.run_id), "events": events.events(arguments.run_id)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _artifact(arguments: argparse.Namespace) -> int:
    store = ArtifactStore(arguments.state_root / "artifacts")
    ref = store.inspect_existing(arguments.artifact_id)
    if arguments.artifact_command == "export":
        print(store.export_existing(ref.id, arguments.output))
        return 0
    print(
        json.dumps(
            {"artifact_id": ref.id, "sha256": ref.digest, "size": ref.size},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
