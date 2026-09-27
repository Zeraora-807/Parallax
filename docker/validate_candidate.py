"""Runs inside the isolated Ganymede validation image, never in the control process."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def main() -> int:
    source = Path("/candidate")
    work = Path("/work/source")
    shutil.copytree(source, work)
    commands = [
        [sys.executable, "-m", "compileall", "-q", "src", "tests"],
        ["ruff", "check", "src", "tests"],
        ["ruff", "format", "--check", "src", "tests"],
    ]
    for command in commands:
        completed = subprocess.run(command, cwd=work, check=False)
        if completed.returncode:
            return completed.returncode
    sys.path.insert(0, str(work / "src"))
    os.environ["PARALLAX_CANDIDATE_SOURCE"] = str(work / "src")
    os.chdir(work)
    baseline = pytest.main(["-q", "-p", "no:cacheprovider", "/opt/parallax/tests"])
    if baseline:
        return int(baseline)
    return int(pytest.main(["-q", "-p", "no:cacheprovider", str(work / "tests")]))


if __name__ == "__main__":
    raise SystemExit(main())
