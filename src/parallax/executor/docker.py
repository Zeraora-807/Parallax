"""Docker executor with a narrow, read-only workspace mount."""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from dataclasses import replace

from parallax.core import CommandSpec, ExecutionResult
from parallax.errors import ToolUnavailableError
from parallax.executor.base import (
    Executor,
    command_artifacts,
    render_container_argument,
    run_bounded_process,
)
from parallax.runtime.workspace import Workspace


class DockerExecutor(Executor):
    required_capability = "executor.docker"

    def __init__(
        self,
        image: str = "parallax-tools:latest",
        max_output_bytes: int = 1_000_000,
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 64,
    ) -> None:
        self.image = image
        self.max_output_bytes = max_output_bytes
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit

    def resolve_cache_command(self, command: CommandSpec) -> CommandSpec | None:
        docker = shutil.which("docker")
        if docker is None:
            return None
        try:
            inspected = subprocess.run(  # noqa: S603 - fixed Docker image inspection
                [docker, "image", "inspect", "--format={{.Id}}", "--", command.image or self.image],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        image_id = inspected.stdout.strip()
        if inspected.returncode or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            return None
        # Execute this exact ID as well as keying the cache by it. A concurrent
        # tag update must not change the environment after cache authorization.
        return replace(command, image=image_id)

    def execute(self, command: CommandSpec, workspace: Workspace) -> ExecutionResult:
        docker = shutil.which("docker")
        if docker is None:
            raise ToolUnavailableError("docker is not installed")

        container_name = f"parallax-{uuid.uuid4().hex}"
        mounts: list[str] = []
        for artifact in command_artifacts(command):
            mounts.extend(
                [
                    "--mount",
                    f"type=bind,src={artifact.path},dst=/inputs/{artifact.digest},readonly",
                ]
            )
        image = command.image or self.image
        argv = [
            docker,
            "run",
            "--rm",
            f"--name={container_name}",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges=true",
            f"--pids-limit={self.pids_limit}",
            f"--memory={self.memory}",
            f"--cpus={self.cpus}",
            "--user=65534:65534",
            "--workdir=/work",
            *mounts,
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--tmpfs",
            "/work:rw,noexec,nosuid,nodev,size=64m",
            "--",
            image,
            command.program,
            *(render_container_argument(arg) for arg in command.arguments),
        ]

        def cleanup() -> None:
            subprocess.run(  # noqa: S603 - fixed Docker cleanup command
                [docker, "kill", container_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )

        return run_bounded_process(
            argv,
            cwd=None,
            environment=None,
            timeout_seconds=command.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
            cleanup=cleanup,
        )
