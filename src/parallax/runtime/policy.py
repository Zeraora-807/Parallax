"""Deny-by-default capability checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from parallax.errors import PermissionDeniedError


@dataclass(frozen=True)
class PermissionSet:
    capabilities: frozenset[str]

    @classmethod
    def from_iterable(cls, values: Iterable[str]) -> "PermissionSet":
        return cls(frozenset(values))

    def allows(self, capability: str) -> bool:
        return capability in self.capabilities

    def require(self, *capabilities: str) -> None:
        missing = sorted(value for value in capabilities if not self.allows(value))
        if missing:
            raise PermissionDeniedError("missing required capabilities: " + ", ".join(missing))
