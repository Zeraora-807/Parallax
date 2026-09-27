"""Trusted-development executor. This is deliberately not described as a sandbox."""

from __future__ import annotations

import os
import shutil

from parallax.core import CommandSpec, ExecutionResult
from parallax.errors import ToolUnavailableError
from parallax.executor.base import Executor, render_local_argument, run_bounded_process
from parallax.runtime.workspace import Workspace


class LocalExecutor(Executor):
    required_capability = "executor.local"

    def __init__(self, max_output_bytes: int = 1_000_000) -> None:
        self.max_output_bytes = max_output_bytes

    def execute(self, command: CommandSpec, workspace: Workspace) -> ExecutionResult:
        if command.image is not None:
            raise ToolUnavailableError("a tool with a container image requires Docker executor")
        executable = shutil.which(command.program)
        if executable is None:
            raise ToolUnavailableError(f"program is not installed: {command.program}")

        argv = [executable, *(render_local_argument(arg) for arg in command.arguments)]
        environment = self._environment(workspace)
        return run_bounded_process(
            argv,
            cwd=str(workspace.root),
            environment=environment,
            timeout_seconds=command.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
            start_new_session=os.name != "nt",
        )

    @staticmethod
    def _environment(workspace: Workspace) -> dict[str, str]:
        keep = ("PATH", "PATHEXT", "SystemRoot", "WINDIR", "COMSPEC", "LANG", "LC_ALL")
        environment = {key: os.environ[key] for key in keep if key in os.environ}
        environment.update(
            {
                "PARALLAX_WORKSPACE": str(workspace.root),
                "PWNLIB_NOTERM": "1",
                "PWNLIB_UPDATE_NEVER": "1",
            }
        )
        return environment
