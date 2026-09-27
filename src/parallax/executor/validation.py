"""Validate self-generated candidates inside a dedicated, resource-limited container."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from parallax.core import ExecutionResult
from parallax.errors import ToolUnavailableError
from parallax.executor.base import run_bounded_process


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    result: ExecutionResult


class DockerCandidateValidator:
    def __init__(
        self,
        image: str = "parallax-ganymede-validator:latest",
        timeout_seconds: float = 300,
        max_output_bytes: int = 2_000_000,
    ) -> None:
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def validate(self, candidate_source: Path) -> ValidationReport:
        docker = shutil.which("docker")
        if docker is None:
            raise ToolUnavailableError("docker is not installed")
        candidate_source = candidate_source.resolve(strict=True)
        name = f"parallax-validation-{uuid.uuid4().hex}"
        argv = [
            docker,
            "run",
            "--rm",
            f"--name={name}",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges=true",
            "--pids-limit=128",
            "--memory=1g",
            "--cpus=2",
            "--user=65534:65534",
            "--mount",
            f"type=bind,src={candidate_source},dst=/candidate,readonly",
            "--tmpfs",
            "/work:rw,nosuid,nodev,size=512m",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=256m",
            "--",
            self.image,
        ]

        def cleanup() -> None:
            subprocess.run(  # noqa: S603
                [docker, "kill", name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )

        result = run_bounded_process(
            argv,
            cwd=None,
            environment=None,
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
            cleanup=cleanup,
        )
        return ValidationReport(passed=result.exit_code == 0, result=result)
