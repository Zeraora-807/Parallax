"""Isolated process boundary used by the local WebUI coordinator."""

from __future__ import annotations

import os
from pathlib import Path
from parallax.executor.base import run_bounded_process


def run_agent_process(
    command: list[str], workspace: Path, environment: dict[str, str]
) -> tuple[int, str, str]:
    completed = run_bounded_process(
        command,
        cwd=str(workspace),
        environment={**os.environ, **environment, "NO_COLOR": "1", "PYTHONUTF8": "1"},
        timeout_seconds=960,
        max_output_bytes=4 * 1024 * 1024,
        start_new_session=os.name != "nt",
    )
    return completed.exit_code, completed.stdout, completed.stderr
