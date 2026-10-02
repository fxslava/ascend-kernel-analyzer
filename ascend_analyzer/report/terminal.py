"""Human-readable terminal report.

Layout: a header naming the chip profile, one block per diagnostic with a
source frame and a remediation, the memory footprint chart per domain, a
pipeline synchronisation summary, and a verdict line.  Colour is used only to
rank severity and is switched off automatically when the output is not a
terminal, when ``NO_COLOR`` is set, or when the stream cannot encode it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, IO, List, Optional, Sequence

from ..diagnostics import Diagnostic, Severity
from ..hardware import HardwareModel
from ..ir import AnalysisUnit, KernelIR
from .memory_map import Glyphs, build_memory_maps, render_ascii_map

__all__ = ["TerminalReporter", "Palette"]


_RESET = "\033[0m"


@dataclass(frozen=True)
class Palette:
    """ANSI styles, or empty strings when colour is disabled."""

    enabled: bool = True

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}{_RESET}" if self.enabled else text

    def fatal(self, text: str) -> str:
        return self._wrap("1;31", text)

    def warning(self, text: str) -> str:
        return self._wrap("1;33", text)

    def info(self, text: str) -> str:
        return self._wrap("1;36", text)

    def ok(self, text: str) -> str:
        return self._wrap("1;32", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def code(self, text: str) -> str:
        return self._wrap("35", text)

    def for_severity(self, severity: Severity) -> str:
        return {
            Severity.FATAL: self.fatal(severity.value),
            Severity.WARNING: self.warning(severity.value),
            Severity.INFO: self.info(severity.value),
        }[severity]

    def segment(self, text: str, style: str) -> str:
        if style == "collision":
            return self._wrap("1;31", text)
        return self._wrap("36", text)


def _supports_color(stream: IO[str], force: Optional[bool]) -> bool:
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if sys.platform == "win32" and not os.environ.get("WT_SESSION"):
        # Modern Windows terminals handle VT sequences; older consoles do not.
        return bool(os.environ.get("ANSICON") or os.environ.get("TERM"))
    return True


class TerminalReporter:
    """Writes the analysis result as formatted text."""

    def __init__(
        self,
        stream: Optional[IO[str]] = None,
        *,
        color: Optional[bool] = None,
        ascii_only: bool = False,
        width: int = 78,
        show_memory_map: bool = True,
        show_sync: bool = True,
        show_perf: bool = True,
        max_diagnostics: int = 0,
    ) -> None:
        self.stream = stream or sys.stdout
        self.palette = Palette(enabled=_supports_color(self.stream, color))
        self.glyphs = Glyphs.best_for(
            getattr(self.stream, "encoding", None), force_ascii=ascii_only
        )
        self.width = width
        self.show_memory_map = show_memory_map
        self.show_sync = show_sync
        self.show_perf = show_perf
        self.max_diagnostics = max_diagnostics

    # -- helpers ------------------------------------------------------------

    def _write(self, text: str = "") -> None:
        self.stream.write(text + "\n")

    def _rule(self, char: Optional[str] = None) -> None:
        self._write(self.palette.dim((char or self.glyphs.h) * self.width))

    def _heading(self, text: str) -> None:
        self._write()
        self._write(self.palette.bold(text))
        self._rule()

    # -- top level ----------------------------------------------------------

    def report(
        self,
        unit: AnalysisUnit,
        hardware: HardwareModel,
        diagnostics: Sequence[Diagnostic],
        artifacts: Optional[Dict[str, object]] = None,
        solver_name: str = "interval",
    ) -> None:
        artifacts = artifacts or {}

        self._write()
        self._write(self.palette.bold("Ascend Static Kernel Analyzer"))
        self._rule(self.glyphs.h)
        self._write(f"  source      {unit.path}")
        self._write(f"  target      {hardware.chip.display_name}")
        self._write(
            f"  solver      {solver_name}    "
            f"kernels: {', '.join(k.name for k in unit.kernels) or '(none)'}"
        )
        if hardware.chip.provisional:
            self._write(
                self.palette.warning("  note        ")
                + self.palette.dim(
                    f"chip profile is provisional. {hardware.chip.notes}"
                )
            )
        if unit.had_parse_errors:
            self._write(
                self.palette.warning("  note        ")
                + self.palette.dim(
                    f"{len(unit.parse_error_locs)} region(s) failed to parse; "
                    "coverage in those regions is reduced"
                )
            )

        self._report_diagnostics(unit, diagnostics)

        if self.show_memory_map:
            for kernel in unit.kernels:
                self._report_memory_map(kernel, hardware)

        if self.show_sync:
            for kernel in unit.kernels:
                self._report_sync(kernel, artifacts)

        if self.show_perf:
            for kernel in unit.kernels:
                self._report_perf(kernel, artifacts)

        self._report_verdict(diagnostics)

    # -- diagnostics --------------------------------------------------------

    def _report_diagnostics(
        self, unit: AnalysisUnit, diagnostics: Sequence[Diagnostic]
    ) -> None:
        self._heading("Findings")
        if not diagnostics:
            self._write(
                "  "
                + self.palette.ok("No findings.")
                + " Layout and synchronisation verified."
            )
            return

        shown = (
            diagnostics[: self.max_diagnostics]
            if self.max_diagnostics
            else diagnostics
        )
        for index, diag in enumerate(shown, start=1):
            self._write_diagnostic(index, diag, unit)
        hidden = len(diagnostics) - len(shown)
        if hidden > 0:
            self._write(self.palette.dim(f"  ... {hidden} more finding(s) suppressed "
                                         "by --max-findings"))

    def _write_diagnostic(
        self, index: int, diag: Diagnostic, unit: AnalysisUnit
    ) -> None:
        self._write()
        header = (
            f"  {index:>3}. {self.palette.for_severity(diag.severity)}  "
            f"{self.palette.code(diag.code.value)}  {self.palette.bold(diag.title)}"
        )
        self._write(header)
        self._write(
            f"       {self.palette.dim(diag.loc.short)}"
            f"   domain: {diag.hardware_domain}"
        )

        for line in diag.message.splitlines():
            self._write(f"       {line}")

        frame = self._source_frame(unit, diag)
        for line in frame:
            self._write(line)

        if diag.remediation:
            self._write(f"       {self.palette.ok('fix:')}")
            for line in _wrap_text(diag.remediation, self.width - 13):
                self._write(f"         {line}")

        for label, loc in diag.related:
            snippet = unit.line_text(loc.line).strip()
            self._write(
                f"       {self.palette.dim('see also')} {loc.short}  {label}"
                + (f"  {self.palette.dim(snippet[:60])}" if snippet else "")
            )

    def _source_frame(self, unit: AnalysisUnit, diag: Diagnostic) -> List[str]:
        """A one-line source excerpt with a caret under the construct."""
        line_no = diag.loc.line
        if line_no <= 0:
            return []
        text = unit.line_text(line_no)
        if not text.strip():
            return []
        indent_removed = len(text) - len(text.lstrip())
        body = text.strip()[: self.width - 16]
        gutter = f"{line_no:>6} | "
        out = [f"       {self.palette.dim(gutter)}{body}"]

        column = max(0, (diag.loc.column - 1) - indent_removed)
        if column >= len(body):
            return out
        # Clang-style underline: a caret at the start of the construct and a
        # tilde run across the rest of it, never past the end of the line.
        span = 1
        if diag.loc.end_line == line_no and diag.loc.end_column:
            span = max(1, diag.loc.end_column - diag.loc.column)
        span = min(span, len(body) - column, 48)
        underline = " " * column + "^" + "~" * (span - 1)
        out.append(
            f"       {' ' * len(gutter)}" + self.palette.warning(underline)
        )
        return out

    # -- memory map ---------------------------------------------------------

    def _report_memory_map(self, kernel: KernelIR, hardware: HardwareModel) -> None:
        maps = build_memory_maps(kernel, hardware)
        self._heading(f"SRAM footprint - {kernel.name}")
        if not maps:
            self._write("  No on-core SRAM tensors were resolved for this kernel.")
            return
        for domain_map in maps:
            for line in render_ascii_map(
                domain_map,
                width=min(52, self.width - 34),
                glyphs=self.glyphs,
                colorize=self.palette.segment,
            ):
                self._write("  " + line)
            self._write()

    # -- synchronisation ----------------------------------------------------

    def _report_sync(self, kernel: KernelIR, artifacts: Dict[str, object]) -> None:
        graph = artifacts.get(f"sync_graph::{kernel.name}")
        if not isinstance(graph, dict):
            return
        nodes = graph.get("nodes") or []
        if not nodes:
            return

        self._heading(f"Pipeline synchronisation - {kernel.name}")
        pipes = graph.get("pipes") or []
        pairs = graph.get("sync_pairs") or []
        acyclic = graph.get("acyclic")
        cycles = graph.get("cycles") or []

        self._write(f"  pipelines in use     {', '.join(pipes) or '(none)'}")
        self._write(f"  flag operations      {len(nodes)}")
        self._write(f"  matched set/wait     {len(pairs)}")
        carried = sum(1 for p in pairs if p.get("loop_carried"))
        if carried:
            self._write(f"  loop-carried pairs   {carried}")

        if acyclic:
            self._write(
                "  dependency graph     "
                + self.palette.ok("acyclic")
                + self.palette.dim("  (topological order exists: no circular wait)")
            )
        else:
            self._write(
                "  dependency graph     "
                + self.palette.fatal(f"cyclic - {len(cycles)} circular wait(s)")
            )

        self._write()
        self._write(
            "  " + self.palette.dim("pipeline timeline (program order per pipe)")
        )
        for pipe_name in pipes:
            marks = [
                _timeline_glyph(op)
                for op in kernel.ops
                if op.pipe.value == pipe_name
            ]
            self._write(f"    {pipe_name:<10} {''.join(marks)}")
        self._write(
            "    "
            + self.palette.dim(
                "legend: S=SetFlag  W=WaitFlag  B=PipeBarrier  "
                "C=DataCopy  V=compute"
            )
        )

    # -- performance ---------------------------------------------------------

    def _report_perf(self, kernel: KernelIR, artifacts: Dict[str, object]) -> None:
        profile = artifacts.get(f"perf_profile::{kernel.name}")
        if not isinstance(profile, dict) or "makespan_cycles" not in profile:
            return
        self._heading(f"Pipeline performance - {kernel.name}")
        makespan = profile["makespan_cycles"]
        overlap = profile.get("overlap_ratio") or 0.0
        bottleneck = profile.get("bottleneck") or "NONE"
        handoff = profile.get("handoff_cycles", 30)
        self._write(
            f"  estimated makespan    {makespan} cycles   "
            + self.palette.dim("(analytical model)")
        )
        self._write(
            f"  concurrency           {overlap:.0%}   "
            + self.palette.dim(
                f"overlap across {profile.get('active_pipes', 0)} pipelines"
            )
        )
        self._write(
            f"  bottleneck            {bottleneck}   "
            + self.palette.dim(f"per-flag hand-off {handoff} cycles")
        )
        drain = profile.get("drain_cycles") or 0
        if drain:
            self._write(
                f"  serialized tail       {drain} cycles   "
                + self.palette.dim(f"{drain / max(makespan, 1):.0%} of the makespan")
            )

        self._write()
        self._write("  " + self.palette.dim("utilization per pipeline"))
        for pipe in profile.get("pipes") or []:
            self._write(_utilization_bar(pipe, self.glyphs, self.palette))
        self._write(
            "    "
            + self.palette.dim(
                "stall = cycles the queue sat at a WaitFlag with nothing else issued"
            )
        )

    # -- verdict ------------------------------------------------------------

    def _report_verdict(self, diagnostics: Sequence[Diagnostic]) -> None:
        fatal = sum(1 for d in diagnostics if d.severity is Severity.FATAL)
        warning = sum(1 for d in diagnostics if d.severity is Severity.WARNING)
        info = sum(1 for d in diagnostics if d.severity is Severity.INFO)

        self._write()
        self._rule()
        parts = [
            self.palette.fatal(f"{fatal} fatal") if fatal else self.palette.ok("0 fatal"),
            self.palette.warning(f"{warning} warning") if warning else "0 warning",
            self.palette.info(f"{info} info") if info else "0 info",
        ]
        self._write("  " + "   ".join(parts))
        if fatal:
            self._write(
                "  "
                + self.palette.fatal("REJECTED")
                + "  this kernel will not run correctly as written."
            )
        elif warning:
            self._write(
                "  "
                + self.palette.warning("ACCEPTED WITH WARNINGS")
                + "  no correctness violation found."
            )
        else:
            self._write(
                "  " + self.palette.ok("ACCEPTED") + "  layout and synchronisation verified."
            )
        self._write()


def _utilization_bar(pipe: dict, glyphs: Glyphs, palette: Palette) -> str:
    """One ASCII gauge per pipeline: filled = busy, empty = idle of makespan."""
    width = 26
    utilization = min(1.0, max(0.0, float(pipe.get("utilization") or 0.0)))
    filled = round(utilization * width)
    bar = glyphs.fill * filled + glyphs.empty * (width - filled)
    stall = pipe.get("stall_cycles") or 0
    stall_note = (
        palette.warning(f"  stall {stall:>5}") if stall else f"  stall {0:>5}"
    )
    return (
        f"    {str(pipe.get('pipe')):<9} [{bar}] "
        f"{utilization:>4.0%}  busy {pipe.get('busy_cycles', 0):>5}"
        + stall_note
    )


def _timeline_glyph(op) -> str:
    """One character per operation, so the handshake pattern is scannable."""
    kind = op.kind
    if kind == "SetFlag":
        return "S"
    if kind == "WaitFlag":
        return "W"
    if kind == "PipeBarrier":
        return "B"
    if kind == "ApiCall":
        return "C" if getattr(op, "name", "").startswith("DataCopy") else "V"
    return "."


def _wrap_text(text: str, width: int) -> List[str]:
    """Wrap text, honouring explicit newlines and leading indentation."""
    out: List[str] = []
    for raw in text.splitlines():
        indent = len(raw) - len(raw.lstrip())
        prefix = " " * indent
        words = raw.split()
        if not words:
            out.append("")
            continue
        line = prefix
        for word in words:
            candidate = f"{line}{word} " if line.strip() else f"{prefix}{word} "
            if len(candidate) > width and line.strip():
                out.append(line.rstrip())
                line = f"{prefix}{word} "
            else:
                line = candidate
        if line.strip():
            out.append(line.rstrip())
    return out
