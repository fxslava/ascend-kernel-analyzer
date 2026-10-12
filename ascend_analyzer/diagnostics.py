"""Diagnostic model: severities, stable error codes, and the collector.

Every finding the analyzer emits is a :class:`Diagnostic` carrying a stable
``AKAnnnn`` code, a severity, the hardware domain or pipeline it concerns, a
source location, a human-readable message and - crucially - a concrete
*remediation* string.  A static checker that tells you something is wrong
without telling you what to type instead is only half a tool.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Severity",
    "Code",
    "SourceLoc",
    "Diagnostic",
    "DiagnosticCollector",
]


class Severity(enum.Enum):
    """How badly a finding will hurt.

    ``FATAL``
        The kernel is wrong: it will hang, corrupt memory, or fail to load.
    ``WARNING``
        The kernel is likely correct but will under-perform, or relies on
        something the analyzer cannot prove safe.
    ``INFO``
        Informational; surfaced for context, never a build breaker.
    """

    FATAL = "FATAL"
    WARNING = "WARNING"
    INFO = "INFO"

    @property
    def rank(self) -> int:
        return {"FATAL": 0, "WARNING": 1, "INFO": 2}[self.value]


class Code(str, enum.Enum):
    """Stable diagnostic identifiers.

    The numeric blocks are: ``1xxx`` memory layout, ``2xxx`` synchronisation
    and launch validation, ``3xxx`` performance and analyzability, ``4xxx``
    matmul constraints, and ``9xxx``
    analyzer infrastructure.
    """

    # -- 1xxx: memory layout ------------------------------------------------
    SRAM_OVERFLOW = "AKA1001"
    BASE_MISALIGNED = "AKA1002"
    BUFFER_COLLISION = "AKA1003"
    DOMAIN_MISMATCH = "AKA1004"
    SIZE_MISALIGNED = "AKA1005"
    BROADCAST_TMP_MISSING = "AKA1006"
    BROADCAST_COPY_UNCHECKED = "AKA1007"
    BROADCAST_COEF_STRIDE = "AKA1008"
    INVALID_EXTENT = "AKA1012"
    STRIDE_MISALIGNED = "AKA1013"
    NEGATIVE_OFFSET = "AKA1014"
    UNKNOWN_DOMAIN = "AKA1009"
    # 351x SIMD/SIMT DataCache headroom below the 32 KiB hardware minimum.
    INSUFFICIENT_DATACACHE = "AKA1010"
    UNSUPPORTED_INTRINSIC = "AKA1011"

    # -- 2xxx: synchronisation / pipeline ----------------------------------
    UNPAIRED_FLAG = "AKA2001"
    UNMATCHED_SET_FLAG = "AKA2001"  # compatibility names
    UNMATCHED_WAIT_FLAG = "AKA2001"
    RESERVED_EVENT_ID = "AKA2003"
    DEADLOCK_CYCLE = "AKA2002"
    PIPE_ROUTE_MISMATCH = "AKA2004"
    UNPRIMED_LOOP_WAIT = "AKA2005"
    LOOP_FLAG_IMBALANCE = "AKA2006"
    FLAG_DOUBLE_SET = "AKA2007"
    EVENT_ID_OUT_OF_RANGE = "AKA2008"
    SELF_ROUTE_SYNC = "AKA2009"
    MISSING_SYNC = "AKA2010"
    DEPRECATED_VF_DISPATCH = "AKA2011"
    INVALID_LAUNCH_BOUNDS = "AKA2012"

    # -- 3xxx: performance / analyzability ---------------------------------
    GLOBAL_BARRIER = "AKA3001"
    SYMBOLIC_OFFSET = "AKA3002"
    RAGGED_TAIL_OVERLAP = "AKA3004"
    VF_SWITCHING = "AKA3005"
    SYMBOLIC_EVENT_ID = "AKA3010"
    UB_FRAGMENTATION = "AKA3011"
    REDUNDANT_BARRIER = "AKA3012"
    UB_BANK_CONFLICT = "AKA3003"
    SIMT_DATACACHE_CONFLICT = "AKA3007"
    REGISTER_SPILL = "AKA3008"
    VECTOR_ALU_CONTENTION = "AKA3009"

    PERF_SYNC_STALL = "AKA3013"
    PERF_CUBE_UNDERUTIL = "AKA3014"

    # -- 4xxx: advanced matmul ----------------------------------------------
    MISSING_SET_TAIL = "AKA4001"
    ASYNC_WORKSPACE_MISSING = "AKA4002"
    MISSING_WAIT_ITERATE_ALL = "AKA4003"
    QUANT_CONFIG_MISMATCH = "AKA4004"
    TSCM_RESIDENCY = "AKA4005"
    UNSUPPORTED_SPARSE_MATMUL = "AKA4006"
    INVALID_MATMUL_TYPE = "AKA4011"

    # -- 9xxx: analyzer infrastructure -------------------------------------
    PARSE_ERROR = "AKA9001"
    UNRESOLVED_TENSOR = "AKA9002"
    NO_KERNEL_FOUND = "AKA9003"
    ANALYSIS_LIMIT = "AKA9004"


#: One-line human titles for each code, used in report headers.
CODE_TITLES: Dict[str, str] = {
    Code.BROADCAST_TMP_MISSING.value: "Missing broadcast temporary buffer",
    Code.BROADCAST_COPY_UNCHECKED.value: "Unchecked broadcast DataCopy",
    Code.BROADCAST_COEF_STRIDE.value: "Broadcast coefficient stride alignment",
    Code.PIPE_ROUTE_MISMATCH.value: "Inter-pipe route mismatch",
    Code.RAGGED_TAIL_OVERLAP.value: "Ragged tail overlap non-idempotency",
    Code.VF_SWITCHING.value: "Excessive VF switching",
    Code.MISSING_SET_TAIL.value: "Missing SetTail",
    Code.ASYNC_WORKSPACE_MISSING.value: "Unregistered asynchronous workspace",
    Code.MISSING_WAIT_ITERATE_ALL.value: "Missing WaitIterateAll",
    Code.QUANT_CONFIG_MISMATCH.value: "Quantization configuration mismatch",
    Code.SRAM_OVERFLOW.value: "SRAM capacity overflow",
    Code.BASE_MISALIGNED.value: "Base address alignment violation",
    Code.BUFFER_COLLISION.value: "Buffer aliasing / address collision",
    Code.DOMAIN_MISMATCH.value: "Memory domain mismatch",
    Code.SIZE_MISALIGNED.value: "Allocation size alignment violation",
    Code.INVALID_EXTENT.value: "Invalid tensor extent",
    Code.STRIDE_MISALIGNED.value: "DMA stride alignment violation",
    Code.NEGATIVE_OFFSET.value: "Negative buffer offset",
    Code.UNKNOWN_DOMAIN.value: "Undetermined memory domain",
    Code.INSUFFICIENT_DATACACHE.value: "Insufficient UB DataCache headroom (351x SIMT)",
    Code.UNSUPPORTED_INTRINSIC.value: "Intrinsic unsupported on target architecture",
    Code.DEPRECATED_VF_DISPATCH.value: "Deprecated SIMT VF dispatch",
    Code.INVALID_LAUNCH_BOUNDS.value: "Invalid SIMT VF signature or launch bounds",
    Code.UNPAIRED_FLAG.value: "Unpaired SetFlag / WaitFlag",
    Code.RESERVED_EVENT_ID.value: "Reserved or unavailable EVENT_ID used",
    Code.DEADLOCK_CYCLE.value: "Pipeline deadlock (circular wait)",
    Code.UNPRIMED_LOOP_WAIT.value: "Unprimed loop-carried WaitFlag",
    Code.LOOP_FLAG_IMBALANCE.value: "SetFlag/WaitFlag imbalance in loop body",
    Code.FLAG_DOUBLE_SET.value: "Flag set twice without an intervening wait",
    Code.EVENT_ID_OUT_OF_RANGE.value: "EVENT_ID out of hardware range",
    Code.SELF_ROUTE_SYNC.value: "Redundant same-pipeline synchronisation",
    Code.MISSING_SYNC.value: "Cross-pipeline hazard without synchronisation",
    Code.GLOBAL_BARRIER.value: "PipeBarrier(PIPE_ALL) antipattern",
    Code.SYMBOLIC_OFFSET.value: "Offset or size not statically resolvable",
    Code.SYMBOLIC_EVENT_ID.value: "EVENT_ID not statically resolvable",
    Code.UB_FRAGMENTATION.value: "Fragmented SRAM layout",
    Code.REDUNDANT_BARRIER.value: "Redundant consecutive barrier",
    Code.UB_BANK_CONFLICT.value: "UB bank conflict on Vector ALU",
    Code.SIMT_DATACACHE_CONFLICT.value: "SIMT DataCache access conflict",
    Code.REGISTER_SPILL.value: "Vector register spill pressure",
    Code.VECTOR_ALU_CONTENTION.value: "SIMT and SIMD Vector ALU contention",
    Code.PERF_SYNC_STALL.value: "Exposed sync bubbles / pipeline stalls",
    Code.PERF_CUBE_UNDERUTIL.value: "Cube compute underutilization",
    Code.TSCM_RESIDENCY.value: "Invalid TSCM full residency",
    Code.UNSUPPORTED_SPARSE_MATMUL.value: "Unsupported sparse matmul mode",
    Code.INVALID_MATMUL_TYPE.value: "Invalid matmul type or GEMV configuration",
    Code.PARSE_ERROR.value: "Source could not be parsed cleanly",
    Code.UNRESOLVED_TENSOR.value: "Tensor argument could not be resolved",
    Code.NO_KERNEL_FOUND.value: "No kernel entry point found",
    Code.ANALYSIS_LIMIT.value: "Analysis limit reached",
}


@dataclass(frozen=True)
class SourceLoc:
    """A 1-based source position, optionally spanning a range."""

    file: str
    line: int
    column: int = 1
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    #: The source text of the construct, trimmed to a single line.
    snippet: str = ""

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"

    @property
    def short(self) -> str:
        """``file:line`` without the column, for compact listings."""
        return f"{self.file}:{self.line}"

    def to_json(self) -> Dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "snippet": self.snippet,
        }

    @classmethod
    def unknown(cls, file: str = "<unknown>") -> "SourceLoc":
        return cls(file=file, line=0, column=0)


@dataclass
class Diagnostic:
    """A single analyzer finding."""

    code: Code
    severity: Severity
    message: str
    loc: SourceLoc
    #: Hardware domain (``UB``, ``L1``, ...) or pipeline (``PIPE_V``) involved.
    hardware_domain: str = "-"
    #: Concrete instructions for fixing the problem.
    remediation: str = ""
    #: Secondary locations that help explain the finding.
    related: Tuple[Tuple[str, SourceLoc], ...] = ()
    #: Structured, code-specific payload (byte ranges, cycle paths, ...).
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return CODE_TITLES.get(self.code.value, self.code.value)

    def sort_key(self) -> Tuple[int, int, int, str]:
        return (self.severity.rank, self.loc.line, self.loc.column, self.code.value)

    def to_json(self) -> Dict[str, object]:
        return {
            "code": self.code.value,
            "title": self.title,
            "severity": self.severity.value,
            "hardware_domain": self.hardware_domain,
            "message": self.message,
            "remediation": self.remediation,
            "location": self.loc.to_json(),
            "related": [
                {"label": label, "location": loc.to_json()}
                for label, loc in self.related
            ],
            "details": self.details,
        }


class DiagnosticCollector:
    """Accumulates diagnostics, with de-duplication and severity filtering."""

    def __init__(self, suppress: Iterable[str] = ()) -> None:
        self._items: List[Diagnostic] = []
        self._seen: set[Tuple[str, int, int, str]] = set()
        self._suppress = {s.strip().upper() for s in suppress if s.strip()}

    def emit(self, diag: Diagnostic) -> Optional[Diagnostic]:
        """Record ``diag`` unless suppressed or an exact duplicate.

        Returns the stored diagnostic, or ``None`` if it was dropped.
        """
        if diag.code.value in self._suppress:
            return None
        key = (diag.code.value, diag.loc.line, diag.loc.column, diag.message)
        if key in self._seen:
            return None
        self._seen.add(key)
        self._items.append(diag)
        return diag

    def add(
        self,
        code: Code,
        severity: Severity,
        message: str,
        loc: SourceLoc,
        *,
        hardware_domain: str = "-",
        remediation: str = "",
        related: Sequence[Tuple[str, SourceLoc]] = (),
        **details: object,
    ) -> Optional[Diagnostic]:
        """Convenience constructor + :meth:`emit`."""
        return self.emit(
            Diagnostic(
                code=code,
                severity=severity,
                message=message,
                loc=loc,
                hardware_domain=hardware_domain,
                remediation=remediation,
                related=tuple(related),
                details=dict(details),
            )
        )

    # -- access -------------------------------------------------------------

    @property
    def items(self) -> List[Diagnostic]:
        return sorted(self._items, key=lambda d: d.sort_key())

    def of_severity(self, severity: Severity) -> List[Diagnostic]:
        return [d for d in self.items if d.severity is severity]

    def count(self, severity: Severity) -> int:
        return sum(1 for d in self._items if d.severity is severity)

    @property
    def fatal_count(self) -> int:
        return self.count(Severity.FATAL)

    @property
    def warning_count(self) -> int:
        return self.count(Severity.WARNING)

    @property
    def info_count(self) -> int:
        return self.count(Severity.INFO)

    def has_fatal(self) -> bool:
        return self.fatal_count > 0

    def summary(self) -> Dict[str, int]:
        return {
            "fatal": self.fatal_count,
            "warning": self.warning_count,
            "info": self.info_count,
            "total": len(self._items),
        }

    def extend(self, other: "DiagnosticCollector") -> None:
        for diag in other.items:
            self.emit(diag)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self.items)
