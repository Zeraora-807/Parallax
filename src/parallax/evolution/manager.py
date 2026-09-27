"""Experimental candidate generation, validation, and guarded activation."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Mapping

from parallax.errors import EvolutionError
from parallax.executor.validation import DockerCandidateValidator, ValidationReport
from parallax.generations.manager import Generation, GenerationManager


class EvolutionManager:
    MODES = {"candidate", "trial", "auto"}
    AUTO_ALLOWED_PREFIXES = (
        "examples/plugins/",
        "examples/presets/",
        "src/parallax/skills/",
        "docs/",
    )
    AUTO_ALLOWED_FILES = {"README.md"}

    def __init__(
        self,
        source_root: Path,
        generations: GenerationManager,
        *,
        mode: str = "candidate",
        validator: DockerCandidateValidator | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"invalid evolution mode: {mode}")
        self.source_root = source_root.resolve(strict=True)
        self.generations = generations
        self.mode = mode
        self.validator = validator

    def propose(self, changes: Mapping[str, str], rationale: str) -> Generation:
        if not changes:
            raise EvolutionError("an evolution proposal must change at least one file")
        if len(changes) > 16:
            raise EvolutionError("an evolution proposal may change at most 16 files")
        if sum(len(content.encode("utf-8")) for content in changes.values()) > 2_000_000:
            raise EvolutionError("evolution proposal exceeds the 2 MB source limit")
        if self.mode == "auto":
            disallowed = []
            for path in changes:
                normalized = PurePosixPath(path.replace("\\", "/"))
                normalized_text = normalized.as_posix()
                if (
                    normalized.is_absolute()
                    or ".." in normalized.parts
                    or (
                        normalized_text not in self.AUTO_ALLOWED_FILES
                        and not normalized_text.startswith(self.AUTO_ALLOWED_PREFIXES)
                    )
                ):
                    disallowed.append(path)
            if disallowed:
                raise EvolutionError(
                    "auto mode may only change declarative plugins, presets, skills, and docs: "
                    + ", ".join(sorted(disallowed))
                )
        return self.generations.create_candidate(
            self.source_root,
            changes=changes,
            config={"evolution_rationale": rationale, "evolution_mode": self.mode},
            parent_id=self.generations.active_id(),
        )

    def validate(self, generation_id: str) -> tuple[Generation, ValidationReport]:
        if self.mode == "candidate":
            raise EvolutionError("candidate mode does not permit autonomous validation")
        if self.validator is None:
            raise EvolutionError("no isolated candidate validator is configured")
        generation = self.generations.get(generation_id)
        report = self.validator.validate(generation.path / "source")
        evidence = {
            "exit_code": report.result.exit_code,
            "duration_ms": report.result.duration_ms,
            "stdout": report.result.stdout[-16_000:],
            "stderr": report.result.stderr[-16_000:],
        }
        if report.passed:
            generation = self.generations.mark_validated(generation_id, evidence)
        else:
            generation = self.generations.mark_rejected(generation_id, evidence)
        return generation, report

    def activate(self, generation_id: str) -> Generation:
        if self.mode != "auto":
            raise EvolutionError("autonomous activation requires evolution mode 'auto'")
        # Status files are mutable control-plane metadata. Re-hash the immutable
        # generation immediately before switching the active profile.
        from parallax.generations.supervisor import Supervisor

        Supervisor.verify_integrity(self.generations.get(generation_id))
        return self.generations.activate(generation_id)
