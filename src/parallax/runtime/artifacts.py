"""Immutable artifacts replace model-visible host paths."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import stat
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO, Iterable

from parallax.core import ArtifactRef
from parallax.errors import ArtifactAccessError, WorkspaceViolationError
from parallax.runtime.workspace import Workspace


class ArtifactStore:
    def __init__(
        self,
        root: Path,
        max_artifact_bytes: int = 512 * 1024 * 1024,
        max_total_artifact_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        self.root = root.resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            # Object files are container-readable; keep their host directory private.
            self.objects.chmod(0o700)
        self.max_artifact_bytes = max_artifact_bytes
        self.max_total_artifact_bytes = max_total_artifact_bytes
        self._grants: dict[str, dict[str, dict[str, set[str]]]] = {}
        self._refs: dict[str, ArtifactRef] = {}
        self._run_bytes: dict[str, int] = {}
        self._lock = threading.Lock()

    def import_workspace_file(
        self, workspace: Workspace, raw_path: str, run_id: str, *, agent_id: str = "root"
    ) -> ArtifactRef:
        resolved = workspace.resolve_input(raw_path)
        descriptor = os.open(resolved.host, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise WorkspaceViolationError("input changed before it could be imported")
            if metadata.st_size > self.max_artifact_bytes:
                raise WorkspaceViolationError("input exceeds the artifact size limit")
            self._reserve_run_bytes(run_id, metadata.st_size)
            try:
                with os.fdopen(descriptor, "rb", closefd=False) as source:
                    ref = self._store_stream(source, resolved.host.name)
                final_metadata = os.fstat(descriptor)
                if (
                    ref.size != metadata.st_size
                    or final_metadata.st_size != metadata.st_size
                    or final_metadata.st_mtime_ns != metadata.st_mtime_ns
                ):
                    raise WorkspaceViolationError("input changed while it was being imported")
            except Exception:
                self._release_run_bytes(run_id, metadata.st_size)
                raise
        finally:
            os.close(descriptor)
        self.grant(
            run_id,
            ref.id,
            {"tool.read", "model.send", "export"},
            agent_id=agent_id,
        )
        return ref

    def store_bytes(
        self,
        content: bytes,
        source_name: str,
        run_id: str,
        actions: set[str] | None = None,
        *,
        agent_id: str = "root",
    ) -> ArtifactRef:
        if len(content) > self.max_artifact_bytes:
            raise ArtifactAccessError("generated artifact exceeds the size limit")
        self._reserve_run_bytes(run_id, len(content))
        try:
            with tempfile.TemporaryFile() as stream:
                stream.write(content)
                stream.seek(0)
                ref = self._store_stream(stream, source_name)
        except Exception:
            self._release_run_bytes(run_id, len(content))
            raise
        granted_actions = actions if actions is not None else {"tool.read", "model.send", "export"}
        self.grant(run_id, ref.id, granted_actions, agent_id=agent_id)
        return ref

    def _reserve_run_bytes(self, run_id: str, size: int) -> None:
        with self._lock:
            used = self._run_bytes.get(run_id, 0)
            if used + size > self.max_total_artifact_bytes:
                raise ArtifactAccessError("run artifact budget exhausted")
            self._run_bytes[run_id] = used + size

    def _release_run_bytes(self, run_id: str, size: int) -> None:
        with self._lock:
            self._run_bytes[run_id] = max(0, self._run_bytes.get(run_id, 0) - size)

    def _store_stream(self, source: BinaryIO, source_name: str) -> ArtifactRef:
        digest = hashlib.sha256()
        size = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.objects, delete=False) as target:
                temporary_path = Path(target.name)
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_artifact_bytes:
                        raise ArtifactAccessError("artifact exceeds the size limit")
                    digest.update(chunk)
                    target.write(chunk)
            hex_digest = digest.hexdigest()
            artifact_id = f"sha256:{hex_digest}"
            destination = self.objects / hex_digest[:2] / hex_digest
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                temporary_path.unlink(missing_ok=True)
                if (
                    destination.is_symlink()
                    or destination.stat().st_size != size
                    or _hash_path(destination) != hex_digest
                ):
                    raise ArtifactAccessError(f"existing artifact object is corrupt: {artifact_id}")
            else:
                os.replace(temporary_path, destination)
            if os.name != "nt":
                destination.chmod(0o644)
            media_type = mimetypes.guess_type(source_name)[0] or "application/octet-stream"
            ref = ArtifactRef(
                id=artifact_id,
                digest=hex_digest,
                path=destination,
                size=size,
                media_type=media_type,
                source_name=Path(source_name).name,
            )
            with self._lock:
                self._refs[artifact_id] = ref
            return ref
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def grant(
        self,
        run_id: str,
        artifact_id: str,
        actions: set[str],
        *,
        agent_id: str = "root",
    ) -> None:
        with self._lock:
            self._grants.setdefault(run_id, {}).setdefault(agent_id, {}).setdefault(
                artifact_id, set()
            ).update(actions)

    def delegate(
        self,
        run_id: str,
        source_agent_id: str,
        target_agent_id: str,
        artifact_id: str,
        actions: set[str],
    ) -> None:
        """Grant a strict subset of one agent's rights to another agent."""

        with self._lock:
            source_actions = (
                self._grants.get(run_id, {}).get(source_agent_id, {}).get(artifact_id, set())
            )
            missing = actions - source_actions
            if missing:
                raise ArtifactAccessError(
                    f"agent {source_agent_id!r} cannot delegate unauthorized actions: "
                    + ", ".join(sorted(missing))
                )
            if artifact_id not in self._refs:
                raise ArtifactAccessError(f"unknown artifact: {artifact_id}")
            self._grants.setdefault(run_id, {}).setdefault(target_agent_id, {}).setdefault(
                artifact_id, set()
            ).update(actions)

    def resolve(
        self,
        run_id: str,
        artifact_id: str,
        action: str = "tool.read",
        *,
        agent_id: str = "root",
    ) -> ArtifactRef:
        with self._lock:
            actions = self._grants.get(run_id, {}).get(agent_id, {}).get(artifact_id, set())
            if action not in actions:
                raise ArtifactAccessError(
                    f"artifact {artifact_id!r} is not authorized for {action!r} "
                    f"in run {run_id!r} for agent {agent_id!r}"
                )
            try:
                return self._refs[artifact_id]
            except KeyError as error:
                raise ArtifactAccessError(f"unknown artifact: {artifact_id}") from error

    def list_for_run(
        self, run_id: str, action: str = "tool.read", *, agent_id: str = "root"
    ) -> list[ArtifactRef]:
        with self._lock:
            return [
                self._refs[artifact_id]
                for artifact_id, actions in sorted(
                    self._grants.get(run_id, {}).get(agent_id, {}).items()
                )
                if artifact_id in self._refs and action in actions
            ]

    def resolve_fields(
        self,
        run_id: str,
        arguments: dict[str, object],
        fields: Iterable[str],
        *,
        agent_id: str = "root",
    ) -> dict[str, object]:
        resolved = dict(arguments)
        for field in fields:
            artifact_id = resolved.get(field)
            if not isinstance(artifact_id, str):
                raise ArtifactAccessError(f"artifact field {field!r} must be a string")
            resolved[field] = self.resolve(run_id, artifact_id, "tool.read", agent_id=agent_id)
        return resolved

    def export(
        self,
        run_id: str,
        artifact_id: str,
        destination: Path,
        *,
        agent_id: str = "root",
    ) -> Path:
        ref = self.resolve(run_id, artifact_id, "export", agent_id=agent_id)
        return self._copy_for_export(ref, destination)

    def inspect_existing(self, artifact_id: str) -> ArtifactRef:
        prefix = "sha256:"
        if not artifact_id.startswith(prefix):
            raise ArtifactAccessError("artifact id must use the sha256 scheme")
        digest = artifact_id[len(prefix) :]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ArtifactAccessError("artifact id has an invalid sha256 digest")
        path = self.objects / digest[:2] / digest
        if not path.is_file():
            raise ArtifactAccessError(f"unknown artifact: {artifact_id}")
        if _hash_path(path) != digest:
            raise ArtifactAccessError(f"artifact integrity failed: {artifact_id}")
        return ArtifactRef(
            id=artifact_id,
            digest=digest,
            path=path,
            size=path.stat().st_size,
            source_name=digest,
        )

    def export_existing(self, artifact_id: str, destination: Path) -> Path:
        """Operator-plane export; possession of state-root access is the authority."""

        return self._copy_for_export(self.inspect_existing(artifact_id), destination)

    @staticmethod
    def _copy_for_export(ref: ArtifactRef, destination: Path) -> Path:
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ArtifactAccessError(f"export destination already exists: {destination}")
        created = False
        try:
            with ref.path.open("rb") as source:
                with destination.open("xb") as target:
                    created = True
                    shutil.copyfileobj(source, target)
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            raise
        return destination


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
