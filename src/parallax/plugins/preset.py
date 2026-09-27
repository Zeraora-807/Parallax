"""A run preset freezes a named composition of plugins, skills, and permissions."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from parallax.errors import PluginValidationError


@dataclass(frozen=True)
class Preset:
    name: str
    plugins: tuple[str, ...]
    skills: tuple[str, ...]
    capabilities: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "Preset":
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))["preset"]
            if not isinstance(payload, dict):
                raise TypeError("preset must be a table")
            name = payload["name"]
            plugins = _preset_list(payload.get("plugins", []), "plugins")
            skills = _preset_list(payload.get("skills", []), "skills")
            capabilities = _preset_list(payload.get("capabilities", []), "capabilities")
        except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
            raise PluginValidationError(f"invalid preset {path}: {error}") from error
        values = [name, *plugins, *skills, *capabilities]
        if any(not isinstance(item, str) or not item for item in values):
            raise PluginValidationError("preset values must be non-empty strings")
        if any("*" in capability for capability in capabilities):
            raise PluginValidationError(
                "preset capabilities must be exact; wildcards are forbidden"
            )
        return cls(
            name=name,
            plugins=tuple(plugins),
            skills=tuple(skills),
            capabilities=tuple(capabilities),
        )


def _preset_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"preset {name} must be an array")
    return value
