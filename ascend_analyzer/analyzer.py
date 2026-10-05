"""The analyzer facade: parse, check, collect, decide.

This is the single entry point library consumers should use.  It owns the
pipeline order (parse -> memory checks -> synchronisation checks -> filter ->
verdict) and nothing else; all the real work lives in the parser and checkers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .checkers.base import CheckerContext
from .checkers.deadlock import DeadlockChecker
from .checkers.hazard import HazardChecker
from .checkers.memory import MemoryChecker
from .checkers.perf_model import PerfModelChecker
from .diagnostics import Diagnostic, DiagnosticCollector, Severity
from .hardware import HardwareModel, PhysicalDomain
from .ir import AnalysisUnit
from .parsing import VisitorOptions, parse_source
from .solver import make_solver

__all__ = ["AnalyzerOptions", "AnalysisResult", "KernelAnalyzer", "TOOL_VERSION"]

TOOL_VERSION = "0.1.0"

#: Checker names recognised by ``--disable``.
AVAILABLE_CHECKERS: Tuple[str, ...] = (
    MemoryChecker.name,
    DeadlockChecker.name,
    HazardChecker.name,
    PerfModelChecker.name,
)


@dataclass
class AnalyzerOptions:
    """Everything that changes what the analyzer concludes."""

    chip: str = "ascend910b"
    chip_profile: Optional[Path] = None
    solver: str = "auto"
    solver_timeout_ms: int = 5000
    #: Promote analyzability gaps (unresolved offsets, unknown domains) to FATAL.
    strict: bool = False
    #: Diagnostic codes to drop entirely.
    suppress: Tuple[str, ...] = ()
    #: Checker names to skip.
    disable: Tuple[str, ...] = ()
    #: Analyze every function, not only ``__global__``/``__aicore__`` entries.
    all_functions: bool = False
    #: Per-domain capacity overrides, in bytes.
    capacity_overrides: Dict[PhysicalDomain, int] = field(default_factory=dict)
    max_ops: int = 20000
    #: Treat warnings as failures when computing the exit code.
    warnings_as_errors: bool = False
    #: Honour in-source ``@ascend-ignore`` annotations.
    honour_inline_ignores: bool = True
    #: Concrete tiling-struct field values, as ``{field: value}``.  The host
    #: tiling function runs on the CPU, so a kernel's ``tilingData->rowFactor``
    #: is a runtime load that no amount of source analysis can fold.  Supplying
    #: one real configuration (from a test harness, a profile, or by hand) binds
    #: those loads and lets the layout - and with it the alignment, aliasing and
    #: bank-conflict checks - resolve for that configuration.
    tiling_values: Dict[str, int] = field(default_factory=dict)
    #: Infer values for unresolved tiling-struct fields from the role each
    #: plays at its call sites (see
    #: :meth:`~ascend_analyzer.parsing.ast_visitor.ASTVisitor._infer_tiling_roles`).
    infer_tiling_roles: bool = False
    #: Which parsing frontend to use: ``"tree-sitter"`` (the original
    #: regex-free text walker) or ``"mlir"`` (BiSheng Clang AST extraction,
    #: lowered through the ``ascend`` MLIR dialect; falls back to tree-sitter
    #: when no toolchain is reachable).
    frontend: str = "tree-sitter"


@dataclass
class AnalysisResult:
    """The outcome of analyzing one translation unit."""

    unit: AnalysisUnit
    hardware: HardwareModel
    diagnostics: List[Diagnostic]
    artifacts: Dict[str, object]
    solver_name: str
    #: Diagnostics dropped by ``@ascend-ignore`` or ``--suppress``.
    suppressed: List[Diagnostic] = field(default_factory=list)

    @property
    def fatal_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity is Severity.FATAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity is Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity is Severity.INFO)

    @property
    def verdict(self) -> str:
        if self.fatal_count:
            return "rejected"
        if self.warning_count:
            return "accepted_with_warnings"
        return "accepted"

    def exit_code(self, warnings_as_errors: bool = False) -> int:
        """``0`` clean, ``1`` fatal findings, ``2`` warnings under ``-Werror``."""
        if self.fatal_count:
            return 1
        if warnings_as_errors and self.warning_count:
            return 2
        return 0

    def codes(self) -> List[str]:
        return [d.code.value for d in self.diagnostics]

    def has_code(self, code: str) -> bool:
        return any(d.code.value == code for d in self.diagnostics)


class KernelAnalyzer:
    """Runs the full analysis pipeline over Ascend C kernel sources."""

    def __init__(self, options: Optional[AnalyzerOptions] = None) -> None:
        self.options = options or AnalyzerOptions()
        self.hardware = self._build_hardware(self.options)
        self._solver = make_solver(
            self.options.solver, timeout_ms=self.options.solver_timeout_ms
        )

    # -- construction -------------------------------------------------------

    @staticmethod
    def _build_hardware(options: AnalyzerOptions) -> HardwareModel:
        hardware = (
            HardwareModel.from_profile_file(options.chip_profile)
            if options.chip_profile is not None
            else HardwareModel.for_chip(options.chip)
        )
        if options.capacity_overrides:
            hardware = hardware.with_capacity_overrides(options.capacity_overrides)
        return hardware

    @property
    def solver_name(self) -> str:
        return self._solver.name

    # -- analysis -----------------------------------------------------------

    def analyze_source(self, path: str, source: str) -> AnalysisResult:
        """Analyze kernel source text."""
        collector = DiagnosticCollector(suppress=self.options.suppress)
        visitor_options = VisitorOptions(
            analyze_all_functions=self.options.all_functions,
            max_ops=self.options.max_ops,
            tiling_values=dict(self.options.tiling_values),
            infer_tiling_roles=self.options.infer_tiling_roles,
        )
        if self.options.frontend == "mlir":
            from .analyzer_mlir import parse_source_mlir

            unit = parse_source_mlir(path, source, self.hardware, collector,
                                     visitor_options=visitor_options)
        else:
            unit = parse_source(path, source, self.hardware, collector,
                                visitor_options)

        context = CheckerContext(
            unit=unit,
            hardware=self.hardware,
            diagnostics=collector,
            strict=self.options.strict,
        )

        disabled = {name.strip().lower() for name in self.options.disable}
        if MemoryChecker.name not in disabled:
            MemoryChecker(context, solver=self._solver).run_all()
        if DeadlockChecker.name not in disabled:
            DeadlockChecker(context).run_all()
        if HazardChecker.name not in disabled:
            # Cross-pipeline hazard sufficiency: needs the flags the deadlock
            # checker validates, but adds no ordering assumptions of its own.
            HazardChecker(context).run_all()
        if PerfModelChecker.name not in disabled:
            # Runs last: the overlap profiler schedules the DAG the deadlock
            # checker proved acyclic and publishes perf_profile artifacts.
            PerfModelChecker(context).run_all()

        kept, dropped = self._partition(unit, collector.items)
        return AnalysisResult(
            unit=unit,
            hardware=self.hardware,
            diagnostics=kept,
            artifacts=context.artifacts,
            solver_name=self.solver_name,
            suppressed=dropped,
        )

    def analyze_file(self, path: Path | str) -> AnalysisResult:
        """Read and analyze a kernel source file."""
        target = Path(path)
        source = target.read_text(encoding="utf-8", errors="replace")
        return self.analyze_source(str(target), source)

    def analyze_paths(self, paths: Sequence[Path | str]) -> List[AnalysisResult]:
        """Analyze several files, expanding directories by common extensions."""
        results: List[AnalysisResult] = []
        for path in paths:
            target = Path(path)
            if target.is_dir():
                for child in sorted(_iter_sources(target)):
                    results.append(self.analyze_file(child))
            else:
                results.append(self.analyze_file(target))
        return results

    # -- filtering ----------------------------------------------------------

    def _partition(
        self, unit: AnalysisUnit, diagnostics: Sequence[Diagnostic]
    ) -> Tuple[List[Diagnostic], List[Diagnostic]]:
        if not self.options.honour_inline_ignores or not unit.suppressions:
            return list(diagnostics), []
        kept: List[Diagnostic] = []
        dropped: List[Diagnostic] = []
        for diag in diagnostics:
            if unit.is_suppressed(diag.code.value, diag.loc.line):
                dropped.append(diag)
            else:
                kept.append(diag)
        return kept, dropped


#: Extensions scanned when a directory is passed to :meth:`analyze_paths`.
SOURCE_EXTENSIONS: Tuple[str, ...] = (".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cce")


def _iter_sources(root: Path):
    for extension in SOURCE_EXTENSIONS:
        yield from root.rglob(f"*{extension}")
