"""Canonical workspace path validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from parallax.core import WorkspacePath
from parallax.errors import WorkspaceViolationError


class Workspace:
    _reserved_top_levels = frozenset({".parallax"})

    def __init__(self, root: Path) -> None:
        if not root.exists() or not root.is_dir():
            raise WorkspaceViolationError(f"workspace is not a directory: {root}")
        self.root = root.resolve(strict=True)

    def resolve_input(self, raw_path: str) -> WorkspacePath:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            raise WorkspaceViolationError("absolute paths are not accepted")
        try:
            resolved = (self.root / candidate).resolve(strict=True)
            relative = resolved.relative_to(self.root)
        except FileNotFoundError as error:
            raise WorkspaceViolationError(f"input path does not exist: {raw_path}") from error
        except (OSError, ValueError) as error:
            raise WorkspaceViolationError(f"path escapes workspace: {raw_path}") from error
        if not resolved.is_file():
            raise WorkspaceViolationError(f"input path is not a regular file: {raw_path}")
        if relative.parts and relative.parts[0].casefold() in self._reserved_top_levels:
            raise WorkspaceViolationError(f"path is reserved for runtime metadata: {raw_path}")
        return WorkspacePath(host=resolved, relative=relative)

    def resolve_fields(
        self,
        arguments: Mapping[str, Any],
        path_fields: tuple[str, ...],
    ) -> dict[str, Any]:
        resolved = dict(arguments)
        for field in path_fields:
            value = resolved.get(field)
            if not isinstance(value, str):
                raise WorkspaceViolationError(f"path field {field!r} must be a string")
            resolved[field] = self.resolve_input(value)
        return resolved
