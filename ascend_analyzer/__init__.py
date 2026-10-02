"""Static kernel analyzer for Huawei Ascend C / DaVinci NPU kernels.

Targets the *static tensor programming* model: raw ``LocalTensor`` addressing
with no ``TPipe``/``TQue`` buffer manager, where every byte offset and every
pipeline handshake is written by hand.

Typical use::

    from ascend_analyzer import KernelAnalyzer, AnalyzerOptions

    analyzer = KernelAnalyzer(AnalyzerOptions(chip="ascend910b"))
    result = analyzer.analyze_file("vec_add.cpp")
    if result.fatal_count:
        for diag in result.diagnostics:
            print(diag.code.value, diag.loc, diag.message)
"""

from __future__ import annotations

from .analyzer import (
    TOOL_VERSION,
    AnalysisResult,
    AnalyzerOptions,
    KernelAnalyzer,
)
from .diagnostics import Code, Diagnostic, DiagnosticCollector, Severity, SourceLoc
from .hardware import (
    CHIP_PROFILES,
    ChipSpec,
    HardEventRoute,
    HardwareModel,
    PhysicalDomain,
    Pipe,
    TPosition,
)
from .ir import AnalysisUnit, KernelIR, TensorDecl

__version__ = TOOL_VERSION

__all__ = [
    "__version__",
    "TOOL_VERSION",
    "KernelAnalyzer",
    "AnalyzerOptions",
    "AnalysisResult",
    "Code",
    "Diagnostic",
    "DiagnosticCollector",
    "Severity",
    "SourceLoc",
    "HardwareModel",
    "ChipSpec",
    "CHIP_PROFILES",
    "PhysicalDomain",
    "Pipe",
    "TPosition",
    "HardEventRoute",
    "AnalysisUnit",
    "KernelIR",
    "TensorDecl",
]
