"""Read-only binary inspection tools with fixed programs and flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from parallax.core import ArtifactRef, CommandSpec, JoinedArtifactRef
from parallax.tools.base import ToolDefinition


ARTIFACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"artifact_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}},
    "required": ["artifact_id"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class FixedBinaryTool(ToolDefinition):
    name: str
    description: str
    program: str
    fixed_arguments: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=lambda: ARTIFACT_SCHEMA)
    required_capability: str = "binary.inspect"
    artifact_fields: tuple[str, ...] = ("artifact_id",)
    joined_path_prefix: str | None = None

    def prepare(self, arguments: Mapping[str, Any]) -> CommandSpec:
        artifact = arguments["artifact_id"]
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("runtime invariant violated: artifact was not resolved")
        artifact_argument = (
            JoinedArtifactRef(self.joined_path_prefix, artifact)
            if self.joined_path_prefix is not None
            else artifact
        )
        return CommandSpec(
            program=self.program,
            arguments=(*self.fixed_arguments, artifact_argument),
        )


class StringsTool(FixedBinaryTool):
    """Extract strings using an explicitly selected character width and byte order."""

    parameters = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "min_length": {"type": "integer", "minimum": 1, "maximum": 1024},
            "encoding": {"type": "string", "enum": ["s", "S", "b", "l", "B", "L"]},
        },
        "required": ["artifact_id"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        super().__init__(
            name="binary.strings",
            description=(
                "Extract printable strings from a binary file. Select single-byte, UTF-16/UCS-2, "
                "or UTF-32 byte order with encoding."
            ),
            program="strings",
            parameters=self.parameters,
        )

    def prepare(self, arguments: Mapping[str, Any]) -> CommandSpec:
        artifact = arguments["artifact_id"]
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("runtime invariant violated: artifact was not resolved")
        minimum = arguments.get("min_length", 4)
        encoding = str(arguments.get("encoding", "s"))
        return CommandSpec(
            program=self.program,
            arguments=("-a", "-n", str(minimum), "-e", encoding, "--", artifact),
        )


class DisassembleTool(FixedBinaryTool):
    parameters = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "section": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
                "pattern": r"^[A-Za-z0-9_.-]+$",
            },
        },
        "required": ["artifact_id"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        super().__init__(
            name="binary.disassemble",
            description=(
                "Disassemble executable sections without running the artifact; optionally limit "
                "the output to one validated section name."
            ),
            program="objdump",
            parameters=self.parameters,
        )

    def prepare(self, arguments: Mapping[str, Any]) -> CommandSpec:
        artifact = arguments["artifact_id"]
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("runtime invariant violated: artifact was not resolved")
        section = arguments.get("section")
        section_argument = (f"--section={section}",) if isinstance(section, str) else ()
        return CommandSpec(
            program=self.program,
            arguments=(
                "--disassemble",
                "--demangle",
                "--no-show-raw-insn",
                "--wide",
                *section_argument,
                "--",
                artifact,
            ),
        )


class HexdumpTool(FixedBinaryTool):
    parameters = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "offset": {
                "type": "integer",
                "minimum": 0,
                "maximum": 4 * 1024 * 1024 * 1024,
                "default": 0,
            },
            "length": {
                "type": "integer",
                "minimum": 1,
                "maximum": 64 * 1024,
                "default": 512,
            },
        },
        "required": ["artifact_id"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        super().__init__(
            name="binary.hexdump",
            description="Display a bounded hexadecimal and ASCII window from an artifact.",
            program="od",
            parameters=self.parameters,
        )

    def prepare(self, arguments: Mapping[str, Any]) -> CommandSpec:
        artifact = arguments["artifact_id"]
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("runtime invariant violated: artifact was not resolved")
        offset = int(arguments.get("offset", 0))
        length = int(arguments.get("length", 512))
        return CommandSpec(
            program=self.program,
            arguments=(
                "--address-radix=x",
                "--format=x1z",
                "--width=16",
                f"--skip-bytes={offset}",
                f"--read-bytes={length}",
                "--",
                artifact,
            ),
        )


class DwarfTool(FixedBinaryTool):
    _VIEWS = {
        "info": "info",
        "decoded-lines": "decodedline",
        "frames": "frames",
        "locations": "loc",
        "ranges": "ranges",
    }
    parameters = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "view": {"type": "string", "enum": sorted(_VIEWS)},
        },
        "required": ["artifact_id", "view"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        super().__init__(
            name="binary.dwarf",
            description=(
                "Inspect one DWARF debug-data view: info, decoded-lines, frames, locations, "
                "or ranges."
            ),
            program="readelf",
            parameters=self.parameters,
        )

    def prepare(self, arguments: Mapping[str, Any]) -> CommandSpec:
        artifact = arguments["artifact_id"]
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("runtime invariant violated: artifact was not resolved")
        view = self._VIEWS[str(arguments["view"])]
        return CommandSpec(
            program=self.program,
            arguments=(f"--debug-dump={view}", "--wide", "--", artifact),
        )


class ArchiveListTool(FixedBinaryTool):
    _FORMATS = {
        "ar": ("ar", ("t",)),
        "tar": ("tar", ("-tf",)),
        "zip": ("unzip", ("-Z1",)),
    }
    parameters = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "format": {"type": "string", "enum": sorted(_FORMATS), "default": "ar"},
        },
        "required": ["artifact_id"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        super().__init__(
            name="binary.archive-list",
            description="List members of an ar, tar, or ZIP archive without extracting them.",
            program="ar",
            parameters=self.parameters,
        )

    def prepare(self, arguments: Mapping[str, Any]) -> CommandSpec:
        artifact = arguments["artifact_id"]
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("runtime invariant violated: artifact was not resolved")
        program, flags = self._FORMATS[str(arguments.get("format", "ar"))]
        return CommandSpec(program=program, arguments=(*flags, artifact))


class PcapSummaryTool(FixedBinaryTool):
    def __init__(self) -> None:
        super().__init__(
            name="binary.pcap-summary",
            description=(
                "Summarize protocols in a PCAP/PCAPNG capture without name resolution or "
                "network access."
            ),
            program="tshark",
        )

    def prepare(self, arguments: Mapping[str, Any]) -> CommandSpec:
        artifact = arguments["artifact_id"]
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("runtime invariant violated: artifact was not resolved")
        return CommandSpec(
            program=self.program,
            arguments=("-n", "-r", artifact, "-q", "-z", "io,phs"),
        )


def builtin_binary_tools() -> list[ToolDefinition]:
    return [
        FixedBinaryTool(
            name="binary.file",
            description="Identify the format and basic properties of a binary file.",
            program="file",
            fixed_arguments=("--brief", "--"),
        ),
        StringsTool(),
        HexdumpTool(),
        DisassembleTool(),
        DwarfTool(),
        FixedBinaryTool(
            name="binary.headers",
            description=(
                "Display generic object-file headers using objdump, including formats supported "
                "outside ELF."
            ),
            program="objdump",
            fixed_arguments=("--all-headers", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.private-headers",
            description=(
                "Display format-specific private headers, including PE data directories and "
                "import metadata when available."
            ),
            program="objdump",
            fixed_arguments=("--private-headers", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.readelf",
            description="Display all available ELF headers, sections, symbols, and dynamic metadata.",
            program="readelf",
            fixed_arguments=("--all", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.sections",
            description="List ELF section headers with offsets, sizes, flags, and types.",
            program="readelf",
            fixed_arguments=("--sections", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.segments",
            description="List ELF program headers and section-to-segment mappings.",
            program="readelf",
            fixed_arguments=("--program-headers", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.symbols",
            description="List ELF static and dynamic symbol tables without executing the file.",
            program="readelf",
            fixed_arguments=("--symbols", "--dyn-syms", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.relocations",
            description="List ELF relocation entries and their referenced symbols.",
            program="readelf",
            fixed_arguments=("--relocs", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.dynamic",
            description="Display the ELF dynamic table, including declared shared-library needs.",
            program="readelf",
            fixed_arguments=("--dynamic", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.notes",
            description="Display ELF notes such as build IDs and ABI metadata.",
            program="readelf",
            fixed_arguments=("--notes", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.version-info",
            description="Display ELF symbol version definitions, requirements, and indexes.",
            program="readelf",
            fixed_arguments=("--version-info", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.unwind",
            description="Display supported ELF unwind information without executing code.",
            program="readelf",
            fixed_arguments=("--unwind", "--wide", "--"),
        ),
        FixedBinaryTool(
            name="binary.size",
            description="Report section sizes and address layout in System V format.",
            program="size",
            fixed_arguments=("--format=sysv", "--"),
        ),
        ArchiveListTool(),
        PcapSummaryTool(),
        FixedBinaryTool(
            name="binary.checksec",
            description="Report common executable hardening features such as NX, PIE, RELRO, and canaries.",
            program="checksec",
            joined_path_prefix="--file=",
        ),
    ]
