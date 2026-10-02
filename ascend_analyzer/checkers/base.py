"""Common scaffolding for checkers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict

from ..diagnostics import DiagnosticCollector
from ..hardware import HardwareModel
from ..ir import AnalysisUnit, KernelIR

__all__ = ["CheckerContext", "Checker"]


@dataclass
class CheckerContext:
    """Everything a checker needs, plus a place to publish structured results."""

    unit: AnalysisUnit
    hardware: HardwareModel
    diagnostics: DiagnosticCollector
    #: Treat analyzability gaps (symbolic offsets, unknown domains) as errors.
    strict: bool = False
    #: Per-checker structured output, merged into the JSON report.
    artifacts: Dict[str, object] = field(default_factory=dict)

    def publish(self, key: str, value: object) -> None:
        self.artifacts[key] = value


class Checker(abc.ABC):
    """Base class for an analysis pass over one kernel."""

    #: Short stable identifier used in report sections and ``--disable``.
    name: str = "checker"

    def __init__(self, context: CheckerContext) -> None:
        self.ctx = context
        self.hw = context.hardware
        self.diags = context.diagnostics

    @abc.abstractmethod
    def check(self, kernel: KernelIR) -> None:
        """Analyze ``kernel``, emitting diagnostics and publishing artifacts."""

    def run_all(self) -> None:
        for kernel in self.ctx.unit.kernels:
            self.check(kernel)
