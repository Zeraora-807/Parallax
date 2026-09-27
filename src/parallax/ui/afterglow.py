"""TTY-aware Afterglow presentation; machine-readable stdout remains unstyled."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from parallax import __codename__, __version__


_OSC_SEQUENCE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)
_CSI_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ESC_SEQUENCE = re.compile(r"\x1b(?:[ -/]*[@-~])?")
_INLINE_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_MULTILINE_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


@dataclass(frozen=True)
class Palette:
    foreground: str = "d6d6d6"
    dim: str = "636363"
    red: str = "bc5653"
    green: str = "909d63"
    yellow: str = "ebc17a"
    blue: str = "7eaac7"
    magenta: str = "aa6292"
    cyan: str = "86d3ce"
    white: str = "f7f7f7"


class AfterglowUI:
    def __init__(self, stream: TextIO | None = None, *, color: bool | None = None) -> None:
        self.stream = stream or sys.stderr
        self.palette = Palette()
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.color = _supports_color(self.stream) if color is None else color

    def paint(self, value: object, color: str, *, bold: bool = False) -> str:
        text = _safe_terminal_text(value)
        if not self.color:
            return text
        red, green, blue = (int(color[index : index + 2], 16) for index in (0, 2, 4))
        weight = "1;" if bold else ""
        return f"\x1b[{weight}38;2;{red};{green};{blue}m{text}\x1b[0m"

    def banner(self, version: str) -> str:
        title = self.paint("PARALLAX", self.palette.cyan, bold=True)
        codename = self.paint(__codename__.upper(), self.palette.magenta, bold=True)
        version_text = self.paint(f"v{version}", self.palette.dim)
        rule = self.paint("-" * 58, self.palette.dim)
        return (
            f"{title} {self.paint('/', self.palette.dim)} {codename}  {version_text}\n"
            f"{self.paint('ADVERSARIAL ANALYSIS RUNTIME', self.palette.yellow)}\n{rule}"
        )

    def render_help(self, value: str, version: str) -> str:
        lines = value.rstrip().splitlines()
        rendered = [self.banner(version), ""]
        for line in lines:
            if line.startswith("usage:"):
                rendered.append(
                    self.paint("usage:", self.palette.yellow, bold=True) + line[len("usage:") :]
                )
            elif line and not line.startswith(" ") and line.endswith(":"):
                rendered.append(self.paint(line[:-1].upper(), self.palette.magenta, bold=True))
            else:
                match = re.match(r"^(\s{2,})(\S.*?)(\s{2,})(\S.*)$", line)
                if match:
                    rendered.append(
                        match.group(1)
                        + self.paint(match.group(2), self.palette.cyan)
                        + match.group(3)
                        + self.paint(match.group(4), self.palette.foreground)
                    )
                else:
                    rendered.append(line)
        return "\n".join(rendered) + "\n"

    def error(self, message: object) -> None:
        label = self.paint("ERROR", self.palette.red, bold=True)
        print(f"{label}  {_safe_terminal_text(message)}", file=self.stream)

    def prompt(self, label: str, default: object | None = None) -> str:
        name = self.paint(label.upper(), self.palette.cyan, bold=True)
        fallback = (
            "" if default in {None, ""} else f" {self.paint(f'[{default}]', self.palette.dim)}"
        )
        return f"{name}{fallback}  "

    def status(self, label: str, value: object, *, state: str = "normal") -> None:
        colors = {
            "normal": self.palette.foreground,
            "success": self.palette.green,
            "warning": self.palette.yellow,
        }
        name = self.paint(f"{label.upper():>10}", self.palette.dim)
        print(f"{name}  {self.paint(value, colors[state])}", file=self.stream)

    @staticmethod
    def safe_model_text(value: object, stream: TextIO) -> str:
        """Make untrusted model text safe for the terminal and its active encoding."""

        text = str(value)
        if not bool(getattr(stream, "isatty", lambda: False)()):
            return text
        safe_text = _safe_terminal_text(text, multiline=True)
        encoding = getattr(stream, "encoding", None) or "utf-8"
        return safe_text.encode(encoding, errors="backslashreplace").decode(encoding)

    def run_started(
        self,
        run_id: str,
        *,
        workspace: Path,
        executor: str,
        model: str,
        inputs: int,
    ) -> None:
        if not self.interactive:
            return
        label = self.paint("RUN", self.palette.yellow, bold=True)
        identifier = self.paint(run_id[:12], self.palette.cyan)
        print(f"\n{label}  {identifier}", file=self.stream)
        self._field("workspace", workspace)
        self._field("executor", executor)
        self._field("model", model)
        self._field("artifacts", inputs)
        print(self.paint("-" * 58, self.palette.dim), file=self.stream)

    def run_finished(self, run_id: str, database: Path, *, ok: bool) -> None:
        if not self.interactive:
            safe_run_id = _safe_terminal_text(run_id)
            safe_database = _safe_terminal_text(database)
            print(f"\nrun: {safe_run_id}\ndatabase: {safe_database}", file=self.stream)
            return
        color = self.palette.green if ok else self.palette.red
        state = "COMPLETE" if ok else "FAILED"
        print(f"{self.paint(state, color, bold=True)}  {run_id}", file=self.stream)
        self._field("trace", database)

    def dashboard(self, snapshot: dict[str, object]) -> None:
        print(self.banner(str(snapshot["version"])), file=self.stream)
        state = self.paint("STATE", self.palette.yellow, bold=True)
        print(f"\n{state}", file=self.stream)
        self._field("root", snapshot["state_root"])
        active = snapshot.get("active_generation") or "none"
        self._field("active", active)
        self._field("generations", snapshot["generation_count"])
        self._field("artifacts", snapshot["artifact_count"])
        self._field("artifact bytes", _human_bytes(int(snapshot["artifact_bytes"])))
        cache = snapshot.get("cache")
        if isinstance(cache, dict):
            self._field(
                "model cache",
                f"{cache.get('model_entries', 0)} entries / "
                f"{cache.get('model_hits', 0)} hits / "
                f"{cache.get('model_saved_tokens', 0)} tokens saved",
            )
            self._field(
                "tool cache",
                f"{cache.get('tool_entries', 0)} entries / {cache.get('tool_hits', 0)} hits",
            )
        raw_runs = snapshot.get("recent_runs", [])
        runs = raw_runs if isinstance(raw_runs, list) else []
        print(f"\n{self.paint('RECENT RUNS', self.palette.magenta, bold=True)}", file=self.stream)
        if not runs:
            print(self.paint("  no recorded runs", self.palette.dim), file=self.stream)
            return
        for item in runs:
            if not isinstance(item, dict):
                continue
            status = str(item["status"])
            color = self.palette.green if status == "completed" else self.palette.red
            identifier = self.paint(str(item["run_id"])[:12], self.palette.cyan)
            created_at = self.paint(item["created_at"], self.palette.foreground)
            print(
                f"  {identifier}  {self.paint(status, color)}  {created_at}",
                file=self.stream,
            )

    def _field(self, name: str, value: object) -> None:
        key = self.paint(f"{name:>14}", self.palette.dim)
        print(f"{key}  {self.paint(value, self.palette.foreground)}", file=self.stream)


def _supports_color(stream: TextIO) -> bool:
    if "NO_COLOR" in os.environ or os.environ.get("TERM", "").casefold() == "dumb":
        return False
    if os.environ.get("FORCE_COLOR", "").casefold() not in {"", "0", "false", "no"}:
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _safe_terminal_text(value: object, *, multiline: bool = False) -> str:
    text = str(value)
    text = _OSC_SEQUENCE.sub("", text)
    text = _CSI_SEQUENCE.sub("", text)
    text = _ESC_SEQUENCE.sub("", text)
    if multiline:
        return _MULTILINE_CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    return _INLINE_CONTROL.sub(" ", text)


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"


class AfterglowArgumentParser(argparse.ArgumentParser):
    """Argparse behavior with styled help and errors, but unchanged parsing semantics."""

    def format_help(self) -> str:
        plain = super().format_help()
        return AfterglowUI(sys.stdout).render_help(plain, __version__)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        AfterglowUI(sys.stderr).error(message)
        self.exit(2)
