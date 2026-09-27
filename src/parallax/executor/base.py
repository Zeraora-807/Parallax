"""Executor boundary: the only package allowed to create host processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import PurePosixPath
from typing import Callable

from parallax.core import (
    CommandArgument,
    CommandSpec,
    ExecutionResult,
    ArtifactRef,
    JoinedArtifactRef,
    JoinedWorkspacePath,
    WorkspacePath,
)
from parallax.runtime.workspace import Workspace
from parallax.errors import ExecutionTimeoutError, OutputLimitExceededError


class Executor(ABC):
    required_capability: str

    @abstractmethod
    def execute(self, command: CommandSpec, workspace: Workspace) -> ExecutionResult: ...


def render_local_argument(argument: CommandArgument) -> str:
    if isinstance(argument, ArtifactRef):
        return str(argument.path)
    if isinstance(argument, JoinedArtifactRef):
        return argument.prefix + str(argument.artifact.path)
    if isinstance(argument, WorkspacePath):
        return str(argument.host)
    if isinstance(argument, JoinedWorkspacePath):
        return argument.prefix + str(argument.path.host)
    return argument


def render_container_argument(argument: CommandArgument) -> str:
    if isinstance(argument, ArtifactRef):
        return f"/inputs/{argument.digest}"
    if isinstance(argument, JoinedArtifactRef):
        return argument.prefix + f"/inputs/{argument.artifact.digest}"
    if isinstance(argument, WorkspacePath):
        return _container_path(argument)
    if isinstance(argument, JoinedWorkspacePath):
        return argument.prefix + _container_path(argument.path)
    return argument


def _container_path(path: WorkspacePath) -> str:
    relative = PurePosixPath(*path.relative.parts)
    return str(PurePosixPath("/workspace") / relative)


def command_artifacts(command: CommandSpec) -> list[ArtifactRef]:
    found: dict[str, ArtifactRef] = {}
    for argument in command.arguments:
        artifact = argument.artifact if isinstance(argument, JoinedArtifactRef) else argument
        if isinstance(artifact, ArtifactRef):
            found[artifact.id] = artifact
    return list(found.values())


def run_bounded_process(
    argv: list[str],
    *,
    cwd: str | None,
    environment: dict[str, str] | None,
    timeout_seconds: float,
    max_output_bytes: int,
    start_new_session: bool = False,
    cleanup: Callable[[], None] | None = None,
) -> ExecutionResult:
    """Stream both output pipes and terminate before memory or disk can grow without bound."""

    process = subprocess.Popen(  # noqa: S603 - callers supply trusted, shell-free argv
        argv,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=start_new_session,
    )
    started = time.monotonic()
    exceeded = threading.Event()
    lock = threading.Lock()
    remaining = max_output_bytes
    captured = {"stdout": bytearray(), "stderr": bytearray()}

    def drain(name: str, stream: object) -> None:
        nonlocal remaining
        reader = stream
        while True:
            chunk = reader.read(64 * 1024)  # type: ignore[attr-defined]
            if not chunk:
                return
            with lock:
                take = min(len(chunk), remaining)
                if take:
                    captured[name].extend(chunk[:take])
                    remaining -= take
                if take < len(chunk):
                    exceeded.set()
                    return

    threads = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()

    deadline = started + timeout_seconds
    reason: str | None = None
    while process.poll() is None:
        if exceeded.is_set():
            reason = "output_limit"
            break
        if time.monotonic() >= deadline:
            reason = "timeout"
            break
        time.sleep(0.02)

    if reason is not None:
        if cleanup is not None:
            # Cleanup is best-effort and must not hide the primary budget failure.
            with suppress(OSError, subprocess.SubprocessError):
                cleanup()
        if process.poll() is None:
            if start_new_session and os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=1)
    if reason is None and exceeded.is_set():
        reason = "output_limit"

    stdout_bytes = bytes(captured["stdout"])
    stderr_bytes = bytes(captured["stderr"])
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if reason == "timeout":
        raise ExecutionTimeoutError(f"tool exceeded {timeout_seconds:g}s timeout")
    if reason == "output_limit":
        raise OutputLimitExceededError(f"tool exceeded {max_output_bytes} output bytes")
    return ExecutionResult(
        exit_code=process.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=round((time.monotonic() - started) * 1000),
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
    )
