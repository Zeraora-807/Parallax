"""Portable, verifiable backups for Ganymede generations and event state."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from parallax.errors import EvolutionError


class BackupManager:
    MAX_FILES = 100_000
    MAX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024 * 1024

    def __init__(self, state_root: Path, event_database: Path) -> None:
        self.state_root = state_root.resolve(strict=True)
        self.event_database = event_database.resolve(strict=True)

    def create(self, destination: Path) -> Path:
        destination = destination.resolve()
        if destination.is_relative_to(self.state_root):
            raise EvolutionError("backup destination must be outside the state root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise EvolutionError(f"backup destination already exists: {destination}")
        with tempfile.TemporaryDirectory(prefix="parallax-backup-") as temporary_name:
            temporary = Path(temporary_name)
            snapshot = temporary / "state"
            snapshot.mkdir()
            for directory_name in ("generations", "profiles", "generation-status", "artifacts"):
                source = self.state_root / directory_name
                if source.exists():
                    for path in source.rglob("*"):
                        if path.is_symlink():
                            raise EvolutionError(f"state backup contains a symlink: {path}")
                    shutil.copytree(source, snapshot / directory_name)
            database_copy = snapshot / "events.sqlite3"
            with closing(sqlite3.connect(self.event_database)) as source_db:
                with closing(sqlite3.connect(database_copy)) as destination_db:
                    source_db.backup(destination_db)
            inventory = []
            for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
                content = path.read_bytes()
                inventory.append(
                    {
                        "path": path.relative_to(snapshot).as_posix(),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size": len(content),
                    }
                )
            manifest = {
                "format": "parallax-europa-backup-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": inventory,
            }
            (snapshot / "backup-manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            created = False
            try:
                with destination.open("xb"):
                    created = True
                with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
                        archive.write(path, path.relative_to(snapshot).as_posix())
            except Exception:
                if created:
                    destination.unlink(missing_ok=True)
                raise
        return destination

    @staticmethod
    def verify(archive_path: Path) -> dict[str, object]:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            if len(members) > BackupManager.MAX_FILES:
                raise EvolutionError("backup contains too many members")
            if sum(member.file_size for member in members) > BackupManager.MAX_UNCOMPRESSED_BYTES:
                raise EvolutionError("backup exceeds the uncompressed size limit")
            for member in members:
                BackupManager._validate_member(member)
            try:
                manifest = json.loads(archive.read("backup-manifest.json"))
            except (KeyError, UnicodeError, json.JSONDecodeError) as error:
                raise EvolutionError("backup manifest is missing or invalid") from error
            if manifest.get("format") != "parallax-europa-backup-v1":
                raise EvolutionError("unsupported backup format")
            entries = manifest.get("files")
            if not isinstance(entries, list):
                raise EvolutionError("backup manifest files must be a list")
            expected = {"backup-manifest.json"}
            for entry in entries:
                if not isinstance(entry, dict):
                    raise EvolutionError("backup manifest has an invalid file entry")
                try:
                    path = str(entry["path"])
                    digest = str(entry["sha256"])
                    size = int(entry["size"])
                    content = archive.read(path)
                except (KeyError, TypeError, ValueError) as error:
                    raise EvolutionError("backup manifest has an invalid file entry") from error
                if path in expected:
                    raise EvolutionError(f"duplicate backup member: {path}")
                expected.add(path)
                if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
                    raise EvolutionError(f"backup integrity failed: {path}")
            actual = {member.filename for member in members if not member.is_dir()}
            if actual != expected:
                raise EvolutionError("backup contains untracked or missing members")
        return manifest

    @classmethod
    def restore_to(cls, archive_path: Path, destination: Path) -> Path:
        cls.verify(archive_path)
        destination = destination.resolve()
        if destination.exists() and any(destination.iterdir()):
            raise EvolutionError("backup restore destination must be empty")
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                cls._validate_member(member)
                target = (destination / member.filename).resolve()
                try:
                    target.relative_to(destination)
                except ValueError as error:
                    raise EvolutionError(
                        f"backup member escapes destination: {member.filename}"
                    ) from error
            archive.extractall(destination)
        return destination

    @staticmethod
    def _validate_member(member: zipfile.ZipInfo) -> None:
        path = PurePosixPath(member.filename.replace("\\", "/"))
        unix_mode = member.external_attr >> 16
        if (
            not member.filename
            or path.is_absolute()
            or ".." in path.parts
            or stat.S_ISLNK(unix_mode)
        ):
            raise EvolutionError(f"unsafe backup member: {member.filename}")
