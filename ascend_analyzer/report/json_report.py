"""Structured JSON report, for CI gates and editor integrations.

The schema is versioned by :data:`SCHEMA_VERSION`.  Consumers should key off
``diagnostics[].code`` (stable) rather than message text (not stable).
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence

from ..diagnostics import Diagnostic, Severity
from ..hardware import HardwareModel
from ..ir import AnalysisUnit, KernelIR
from .memory_map import build_memory_maps

__all__ = ["SCHEMA_VERSION", "build_json_report", "dump_json_report"]

SCHEMA_VERSION = "1.0"


def build_json_report(
    unit: AnalysisUnit,
    hardware: HardwareModel,
    diagnostics: Sequence[Diagnostic],
    artifacts: Optional[Dict[str, object]] = None,
    *,
    solver_name: str = "interval",
    tool_version: str = "0.1.0",
) -> Dict[str, object]:
    """Assemble the full machine-readable report."""
    artifacts = artifacts or {}

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {
            "name": "ascend-kernel-analyzer",
            "version": tool_version,
            "solver": solver_name,
            "python": platform.python_version(),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "source": {
            "path": unit.path,
            "line_count": len(unit.lines),
            "had_parse_errors": unit.had_parse_errors,
            "parse_error_locations": [
                loc.to_json() for loc in unit.parse_error_locs
            ],
            "constants": unit.constants,
        },
        "target": hardware.describe(),
        "summary": _summary(diagnostics),
        "verdict": _verdict(diagnostics),
        "diagnostics": [d.to_json() for d in diagnostics],
        "kernels": [
            _kernel_json(kernel, hardware, artifacts) for kernel in unit.kernels
        ],
    }


def _summary(diagnostics: Sequence[Diagnostic]) -> Dict[str, object]:
    by_code: Dict[str, int] = {}
    for diag in diagnostics:
        by_code[diag.code.value] = by_code.get(diag.code.value, 0) + 1
    return {
        "fatal": sum(1 for d in diagnostics if d.severity is Severity.FATAL),
        "warning": sum(1 for d in diagnostics if d.severity is Severity.WARNING),
        "info": sum(1 for d in diagnostics if d.severity is Severity.INFO),
        "total": len(diagnostics),
        "by_code": dict(sorted(by_code.items())),
    }


def _verdict(diagnostics: Sequence[Diagnostic]) -> str:
    if any(d.severity is Severity.FATAL for d in diagnostics):
        return "rejected"
    if any(d.severity is Severity.WARNING for d in diagnostics):
        return "accepted_with_warnings"
    return "accepted"


def _kernel_json(
    kernel: KernelIR, hardware: HardwareModel, artifacts: Dict[str, object]
) -> Dict[str, object]:
    payload = kernel.to_json()
    payload["memory_map"] = [m.to_json() for m in build_memory_maps(kernel, hardware)]
    payload["trace"] = [op.to_json() for op in kernel.ops]

    sync = artifacts.get(f"sync_graph::{kernel.name}")
    if isinstance(sync, dict):
        payload["synchronisation"] = sync
    usage = artifacts.get(f"memory_usage::{kernel.name}")
    if usage is not None:
        payload["domain_usage"] = usage
    suppressed = artifacts.get(f"suppressed_cycles::{kernel.name}")
    if suppressed is not None:
        payload["suppressed_cycles"] = suppressed
    return payload


def dump_json_report(report: Dict[str, object], indent: int = 2) -> str:
    """Serialise a report, keeping non-serialisable values visible as strings."""
    return json.dumps(report, indent=indent, default=_fallback, sort_keys=False)


def _fallback(value: object) -> object:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)
