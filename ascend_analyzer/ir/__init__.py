"""Analyzer IR package: the flat kernel trace and the ``ascend`` MLIR dialect.

:mod:`.kernel_ir` holds the operation-trace IR the checkers consume;
:mod:`.mlir_ascend` holds the structured dialect the BiSheng frontend lowers
into.  Everything from :mod:`.kernel_ir` is re-exported here so the
historical ``from .ir import KernelIR`` import path keeps working.
"""

from .kernel_ir import (
    AnalysisUnit,
    ApiCallOp,
    ArgRef,
    BarrierOp,
    CoreView,
    FlagKind,
    FlagOp,
    KernelIR,
    LoopInfo,
    Operation,
    Scope,
    ScopeKind,
    TensorDecl,
)
def __getattr__(name):
    if name == "AscendModule":
        from .mlir_ascend import AscendModule
        return AscendModule
    raise AttributeError(name)

__all__ = [
    "FlagKind",
    "ScopeKind",
    "Scope",
    "LoopInfo",
    "TensorDecl",
    "ArgRef",
    "Operation",
    "FlagOp",
    "BarrierOp",
    "ApiCallOp",
    "KernelIR",
    "AnalysisUnit",
    "CoreView",
    "AscendModule",
]
