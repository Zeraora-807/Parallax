"""Nix-inspired immutable generations and atomic profile switching."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from parallax import __version__
from parallax.errors import EvolutionError


IGNORED_NAMES = {
    ".git",
    ".mypy_cache",
    ".nox",
    ".parallax",
    ".parallax-web",
    ".pip-tmp",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "source-release",
    "wheels",
    "workspace",
    "start.local.json",
    "start.local.json.tmp",
    "pip-install.log",
}


@dataclass(frozen=True)
class Generation:
    id: str
    path: Path
    manifest: Mapping[str, Any]
    status: str


class GenerationManager:
    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root.resolve()
        self.generations = self.state_root / "generations"
        self.profiles = self.state_root / "profiles"
        self.statuses = self.state_root / "generation-status"
        for directory in (self.generations, self.profiles, self.statuses):
            directory.mkdir(parents=True, exist_ok=True)

    def create_candidate(
        self,
        source_root: Path,
        *,
        changes: Mapping[str, str] | None = None,
        plugin_lock: Mapping[str, Mapping[str, str]] | None = None,
        config: Mapping[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> Generation:
        source_root = source_root.resolve(strict=True)
        if not source_root.is_dir():
            raise EvolutionError("generation source must be a directory")
        if source_root.is_relative_to(self.state_root):
            raise EvolutionError("generation source cannot be inside the state root")
        if self.state_root.is_relative_to(source_root):
            state_relative = self.state_root.relative_to(source_root)
            if not any(_is_ignored_name(part) for part in state_relative.parts):
                raise EvolutionError(
                    "state root inside generation source must be under an ignored directory"
                )
        temporary = Path(tempfile.mkdtemp(prefix="ganymede-generation-", dir=self.generations))
        try:
            snapshot = temporary / "source"
            self._copy_source(source_root, snapshot)
            for relative, content in (changes or {}).items():
                target = self._candidate_member(snapshot, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8", newline="\n")

            files = self._file_inventory(snapshot)
            identity = {
                "runtime_version": __version__,
                "parent_id": parent_id,
                "files": files,
                "plugins": dict(plugin_lock or {}),
                "config": dict(config or {}),
            }
            generation_id = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            manifest = {
                "generation_id": generation_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **identity,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            destination = self.generations / generation_id
            created = not destination.exists()
            if not created:
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, destination)
            if created:
                self._write_status(generation_id, "candidate", {})
            return self.get(generation_id)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def mark_validated(self, generation_id: str, evidence: Mapping[str, Any]) -> Generation:
        self.get(generation_id)
        self._write_status(generation_id, "validated", evidence)
        return self.get(generation_id)

    def mark_rejected(self, generation_id: str, evidence: Mapping[str, Any]) -> Generation:
        self.get(generation_id)
        self._write_status(generation_id, "rejected", evidence)
        return self.get(generation_id)

    def activate(self, generation_id: str) -> Generation:
        generation = self.get(generation_id)
        if generation.status != "validated":
            raise EvolutionError("only validated generations can be activated")
        active = self.active_id()
        history = self._history()
        if active and (not history or history[-1] != active):
            history.append(active)
        if not history or history[-1] != generation_id:
            history.append(generation_id)
        self._atomic_json(self.profiles / "history.json", history)
        self._atomic_json(self.profiles / "active.json", {"generation_id": generation_id})
        return generation

    def rollback(self) -> Generation:
        target_generation = self.rollback_target()
        target = target_generation.id
        active = self.active_id()
        history = self._history()
        candidates = [item for item in history if item != active]
        history = candidates
        self._atomic_json(self.profiles / "history.json", history)
        self._atomic_json(self.profiles / "active.json", {"generation_id": target})
        return target_generation

    def rollback_target(self) -> Generation:
        active = self.active_id()
        candidates = [item for item in self._history() if item != active]
        if not candidates:
            raise EvolutionError("there is no previous generation to roll back to")
        return self.get(candidates[-1])

    def active_id(self) -> str | None:
        path = self.profiles / "active.json"
        if not path.exists():
            return None
        return str(json.loads(path.read_text(encoding="utf-8"))["generation_id"])

    def get(self, generation_id: str) -> Generation:
        if not _is_digest(generation_id):
            raise EvolutionError("invalid generation id")
        path = self.generations / generation_id
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise EvolutionError(f"unknown generation: {generation_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status_path = self.statuses / f"{generation_id}.json"
        status = "candidate"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))["status"]
        return Generation(id=generation_id, path=path, manifest=manifest, status=status)

    def list(self) -> list[Generation]:
        return [self.get(path.name) for path in sorted(self.generations.iterdir()) if path.is_dir()]

    def _write_status(self, generation_id: str, status: str, evidence: Mapping[str, Any]) -> None:
        self._atomic_json(
            self.statuses / f"{generation_id}.json",
            {
                "generation_id": generation_id,
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "evidence": dict(evidence),
            },
        )

    @staticmethod
    def _copy_source(source: Path, destination: Path) -> None:
        def ignore(_: str, names: list[str]) -> set[str]:
            return {name for name in names if _is_ignored_name(name)}

        for path in source.rglob("*"):
            relative_parts = path.relative_to(source).parts
            if path.is_symlink() and not any(_is_ignored_name(part) for part in relative_parts):
                raise EvolutionError(f"generation source contains a symlink: {path}")
        shutil.copytree(source, destination, ignore=ignore)

    @staticmethod
    def _candidate_member(root: Path, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or not relative:
            raise EvolutionError(f"candidate path must be relative: {relative}")
        if ".." in candidate.parts:
            raise EvolutionError(f"candidate path escapes source root: {relative}")
        target = (root / candidate).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as error:
            raise EvolutionError(f"candidate path escapes source root: {relative}") from error
        if any(_is_ignored_name(part) for part in candidate.parts):
            raise EvolutionError(f"candidate path is reserved: {relative}")
        return target

    @staticmethod
    def _file_inventory(root: Path) -> list[dict[str, object]]:
        entries = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            content = path.read_bytes()
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                }
            )
        return entries

    def _history(self) -> list[str]:
        path = self.profiles / "history.json"
        if not path.exists():
            return []
        return list(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_ignored_name(name: str) -> bool:
    return name in IGNORED_NAMES or name.endswith((".pyc", ".egg-info"))
