"""Small trusted supervisor for integrity checks, activation, rollback, and crash marking."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

from parallax.errors import EvolutionError
from parallax.generations.manager import Generation, GenerationManager
from parallax.tracing.sqlite import SQLiteEventStore


class Supervisor:
    def __init__(self, generations: GenerationManager, events: SQLiteEventStore) -> None:
        self.generations = generations
        self.events = events

    def activate(self, generation_id: str) -> Generation:
        generation = self.generations.get(generation_id)
        self.verify_integrity(generation)
        activated = self.generations.activate(generation_id)
        self.events.record(
            "supervisor",
            "generation.activated",
            "ok",
            {"generation_id": activated.id},
        )
        return activated

    def rollback(self, reason: str) -> Generation:
        target = self.generations.rollback_target()
        self.verify_integrity(target)
        generation = self.generations.rollback()
        self.events.record(
            "supervisor",
            "generation.rolled_back",
            "ok",
            {"generation_id": generation.id, "reason_code": reason},
        )
        return generation

    def recover_unfinished_runs(self) -> list[str]:
        recovered = list(self.events.unfinished_runs())
        for run_id in recovered:
            self.events.record(
                run_id,
                "run.recovered",
                "interrupted",
                {"reason_code": "supervisor_restart"},
            )
            self.events.finish_run(run_id, "interrupted")
        return recovered

    def launch(self, arguments: list[str], *, rollback_on_failure: bool = False) -> int:
        """Launch the active immutable generation; the supervisor stays outside it."""

        active_id = self.generations.active_id()
        if active_id is None:
            raise EvolutionError("there is no active generation")
        generation = self.generations.get(active_id)
        self.verify_integrity(generation)
        if rollback_on_failure and not self._healthcheck_generation(generation):
            self.events.record(
                "supervisor",
                "generation.health_failed",
                "error",
                {"generation_id": generation.id},
            )
            generation = self.rollback("active_generation_healthcheck_failed")
            if not self._healthcheck_generation(generation):
                raise EvolutionError("rollback generation failed its startup health check")
        return self._launch_generation(generation, arguments)

    @staticmethod
    def _generation_environment(generation: Generation) -> dict[str, str]:
        environment = dict(os.environ)
        source_path = str(generation.path / "source" / "src")
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = source_path + (os.pathsep + existing if existing else "")
        environment["PARALLAX_ACTIVE_GENERATION"] = generation.id
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return environment

    @classmethod
    def _healthcheck_generation(cls, generation: Generation) -> bool:
        try:
            completed = subprocess.run(  # noqa: S603 - validated generation import probe
                [sys.executable, "-c", "import parallax; import parallax.cli"],
                env=cls._generation_environment(generation),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            return completed.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _launch_generation(generation: Generation, arguments: list[str]) -> int:
        completed = subprocess.run(  # noqa: S603 - validated generation and literal module
            [sys.executable, "-m", "parallax.cli", *arguments],
            env=Supervisor._generation_environment(generation),
            check=False,
        )
        return completed.returncode

    @staticmethod
    def verify_integrity(generation: Generation) -> None:
        source = generation.path / "source"
        manifest = generation.manifest
        try:
            files = manifest["files"]
            identity = {
                "runtime_version": manifest["runtime_version"],
                "parent_id": manifest["parent_id"],
                "files": files,
                "plugins": manifest["plugins"],
                "config": manifest["config"],
            }
            identity_digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            expected = {entry["path"]: entry for entry in files}
        except (KeyError, TypeError) as error:
            raise EvolutionError("generation manifest is malformed") from error
        if manifest.get("generation_id") != generation.id or identity_digest != generation.id:
            raise EvolutionError("generation manifest identity failed")
        if len(expected) != len(files):
            raise EvolutionError("generation manifest contains duplicate files")
        for path in source.rglob("*"):
            if path.is_symlink():
                raise EvolutionError(
                    f"generation contains a symlink: {path.relative_to(source).as_posix()}"
                )
        actual = {
            path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()
        }
        expected_paths = set(expected)
        if missing := sorted(expected_paths - actual):
            raise EvolutionError(f"generation file is missing: {missing[0]}")
        if unexpected := sorted(actual - expected_paths):
            raise EvolutionError(f"generation contains an untracked file: {unexpected[0]}")
        for relative, entry in expected.items():
            path = source / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.stat().st_size != entry["size"] or digest != entry["sha256"]:
                raise EvolutionError(f"generation file integrity failed: {relative}")
