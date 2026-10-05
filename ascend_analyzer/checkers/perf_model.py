"""KernelIR performance adapter to the graph backend.

Completion fences and pipe order determine a dependency schedule. Throughput
inputs are estimates; a nonzero synchronization hand-off requires explicit
calibration. Compute/DMA overlap is measured as an interval intersection.
Continuous streaming requires explicit buffer-lifetime edges in backend.flow;
this adapter does not manufacture them from tensor capacity or source order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..backend.graph import overlap_metrics
from ..backend.trace import schedule_trace
from ..apis import lookup_api, ArgRole
from ..diagnostics import Code, Severity
from ..hardware import REAL_PIPES, PhysicalDomain, Pipe
from ..ir import ApiCallOp, FlagKind, FlagOp, KernelIR, Operation
from .base import Checker

__all__ = ["PerfModelChecker", "PipeProfile"]

#: One token of a shape argument: a name, or an integer literal.  A
#: dimension reaches here as a name before macro substitution and as a
#: literal after it, and both spellings have to resolve.
_SHAPE_TOKEN_RE = re.compile(r"\b([A-Za-z_]\w*)\b|\b(0[xX][0-9a-fA-F]+|\d+)")

_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")


@dataclass
class PipeProfile:
    """Schedule statistics for one hardware pipeline."""

    pipe: str
    op_count: int
    busy_cycles: int
    stall_cycles: int
    utilization: float
    #: Schedule length the idle time is measured against.
    makespan_cycles: int = 0

    @property
    def idle_cycles(self) -> int:
        return max(0, self.makespan_cycles - self.busy_cycles)

    def to_json(self) -> Dict[str, object]:
        return {
            "pipe": self.pipe,
            "op_count": self.op_count,
            "busy_cycles": self.busy_cycles,
            "idle_cycles": self.idle_cycles,
            "stall_cycles": self.stall_cycles,
            "utilization": round(self.utilization, 4),
        }


class PerfModelChecker(Checker):
    """Computes the analytical schedule and reports overlap deficiencies."""

    name = "perf"

    #: Advisories are suppressed below this modeled kernel length: a short
    #: kernel's fill/drain bubbles are inherent, not a tuning target.
    min_makespan_cycles: int = 500
    #: Hand-off bubbles must exceed this fraction of all busy time to be the
    #: primary bottleneck.
    stall_ratio_threshold: float = 0.30
    #: A serialized tail longer than this fraction of the makespan is a drain.
    drain_ratio_threshold: float = 0.25
    #: Cube busy fraction below this counts as underutilization.
    cube_util_threshold: float = 0.50

    _MEMORY_PIPES = frozenset({Pipe.MTE1, Pipe.MTE2, Pipe.MTE3, Pipe.FIX})
    _CUBE_APIS = frozenset({"Mmad", "MmadWithSparse", "mad_mx"})

    def check(self, kernel: KernelIR) -> None:
        graph = self.ctx.artifacts.get(f"sync_graph::{kernel.name}")
        if not isinstance(graph, dict):
            return  # no synchronisation analysis to build on
        if not kernel.ops:
            return
        if graph.get("acyclic") is False:
            # Deadlocked kernels already carry a fatal finding; a schedule of
            # a cyclic dependency graph would be fiction.
            self.ctx.publish(
                f"perf_profile::{kernel.name}",
                {"skipped": "dependency graph is cyclic - fix the deadlock first"},
            )
            return

        schedule = self._schedule(kernel, graph)
        if schedule is None:
            return
        start, end, cycles, stall_by_pipe, stall_by_route = schedule

        makespan = max(end.values(), default=0)
        pipes = self._pipe_profiles(kernel, cycles, stall_by_pipe, makespan)
        active = [p for p in pipes if p.op_count]
        total_busy = sum(p.busy_cycles for p in active)
        total_stall = sum(p.stall_cycles for p in active)
        overlap = (
            total_busy / (makespan * len(active)) if makespan and active else 0.0
        )
        work = [op for op in kernel.ops if isinstance(op, ApiCallOp)]
        timing = overlap_metrics(
            [(start[op.index], end[op.index]) for op in work if op.pipe in (Pipe.M, Pipe.V)],
            [(start[op.index], end[op.index]) for op in work if op.pipe in self._MEMORY_PIPES],
        )
        critical = max(
            active, key=lambda p: p.busy_cycles, default=None
        )
        drain = self._drain_cycles(kernel, end, makespan)
        bottleneck = self._classify(
            kernel, active, total_busy, total_stall, drain, makespan, critical
        )

        self.ctx.publish(
            f"perf_profile::{kernel.name}",
            {
                "makespan_cycles": makespan,
                "total_busy_cycles": total_busy,
                "total_stall_cycles": total_stall,
                "stall_by_route": dict(
                    sorted(stall_by_route.items(), key=lambda kv: -kv[1])
                ),
                "drain_cycles": drain,
                "concurrency_efficiency": round(overlap, 4),
                "overlap_ratio": round(timing["dma_hidden_ratio"], 4),
                **timing,
                "model": "dependency-constrained service; uncalibrated throughput estimates",
                "queue_coverage": "no streaming queues inferred from tensor allocation; use backend.flow with explicit buffer lifetimes",
                "horizon": "frontend trace (peeled loops are representative, not whole-kernel latency)",
                "active_pipes": len(active),
                "critical_pipe": critical.pipe if critical else None,
                "bottleneck": bottleneck,
                "pipes": [p.to_json() for p in pipes],
                "handoff_cycles": self.hw.chip.sync_handoff_cycles,
                "operation_intervals": [
                    {"index": op.index, "pipe": op.pipe.value,
                     "start": start[op.index], "end": end[op.index], "kind": op.kind}
                    for op in kernel.ops
                ],
            },
        )

        if makespan < self.min_makespan_cycles:
            return
        self._report_sync_stalls(
            kernel, active, total_busy, total_stall, stall_by_route, start, makespan
        )
        self._report_cube_underutil(kernel, active, makespan)

    # -- scheduling ---------------------------------------------------------

    def _schedule(self, kernel: KernelIR, graph: dict):
        """ASAP-schedule the trace; returns ``(start, end, cycles, stalls)``.

        The backend constructs and topologically sorts dependencies; numeric
        trace indices need not be a legal issue order.
        """
        cycles = {op.index: self._service_cycles(kernel, op) for op in kernel.ops}
        return schedule_trace(kernel, graph, cycles, self.hw)

    # -- service times ------------------------------------------------------

    def _service_cycles(self, kernel: KernelIR, op: Operation) -> int:
        if isinstance(op, ApiCallOp):
            if op.name in self._CUBE_APIS:
                return self._cube_cycles(kernel, op)
            bytes_moved = self._transfer_bytes(kernel, op)
            bandwidth = self.hw.pipe_bytes_per_cycle(op.pipe)
            if bandwidth is None:
                return self.hw.chip.issue_cycles
            return max(1, -(-bytes_moved // bandwidth))
        return self.hw.chip.issue_cycles

    def _cube_cycles(self, kernel: KernelIR, op: ApiCallOp) -> int:
        shape = self._cube_shape(kernel, op)
        if shape is not None:
            return self.hw.chip.cube_contraction_cycles(*shape)
        operands = sum(
            kernel.tensors[arg.tensor].size_value or 0
            for arg in op.tensor_args()
            if arg.tensor in kernel.tensors
        )
        if operands <= 0:
            return self.hw.chip.issue_cycles
        throughput = self.hw.chip.cube_read_bytes_per_cycle
        return max(1, -(-operands // throughput))

    def _cube_shape(self, kernel: KernelIR, op: ApiCallOp) -> Optional[Tuple[int, int, int]]:
        """Recover ``(M, K, N)`` from a shape argument.

        Cube kernels spell the tile shape as ``mmad_t::shape_t((uint16_t)M,
        (uint16_t)K, (uint16_t)N)``.  Each dimension may arrive either as an
        identifier that resolves against the kernel's folded
        ``constexpr``/``#define`` environment, or as an integer literal -
        which is what it already is once the token preprocessor has
        substituted the macro.  The first three of either kind, in source
        order, are taken as ``(m, k, n)``.

        A type name such as ``uint16_t`` is neither a known constant nor a
        literal, so the cast around each dimension is skipped; the ``16``
        inside that name is never seen on its own because the identifier
        matches as one token.
        """
        constants = kernel.constants
        for arg in op.args:
            if arg.tensor:
                continue
            found: List[int] = []
            for match in _SHAPE_TOKEN_RE.finditer(arg.text):
                name, literal = match.group(1), match.group(2)
                if literal is not None:
                    try:
                        found.append(int(literal, 0))
                    except ValueError:
                        continue
                elif name in constants:
                    found.append(constants[name])
                if len(found) >= 3:
                    return (found[0], found[1], found[2])
        return None

    def _transfer_bytes(self, kernel: KernelIR, op: ApiCallOp) -> int:
        """Volume of one DMA / vector operation, in bytes.

        The on-core (SRAM) participant bounds the moved tile: a ``GlobalTensor``
        spans the whole buffer, but a single ``DataCopy`` only moves the tile
        that lands in UB/L1/L0.  GM-only calls fall back to the count
        parameter, then to one hardware block.
        """
        spec = lookup_api(op.name)
        if spec is not None:
            for arg in op.args:
                param = spec.param_at(arg.index)
                if param is not None and param.role is ArgRole.COUNT and arg.value is not None:
                    operand = next((kernel.tensors.get(a.tensor) for a in op.args
                                    if a.tensor and kernel.tensors.get(a.tensor) is not None), None)
                    if operand is not None and operand.elem_size:
                        return max(0, arg.value * operand.elem_size)
        on_core: List[int] = []
        global_only: List[int] = []
        for name in (*op.writes, *op.reads):
            tensor = kernel.tensors.get(name)
            if tensor is None or tensor.size_value is None:
                continue
            if tensor.domain is PhysicalDomain.GM:
                global_only.append(tensor.size_value)
            else:
                on_core.append(tensor.size_value)
        if on_core:
            return max(on_core)
        if global_only:
            return min(global_only)
        for arg in op.args:
            if arg.tensor is None or arg.value is None:
                continue
            dst = kernel.tensors.get(arg.tensor)
            if dst is not None and dst.elem_size:
                return max(self.hw.chip.block_bytes, arg.value * dst.elem_size)
        return self.hw.chip.block_bytes

    # -- metrics --------------------------------------------------------------

    def _pipe_profiles(
        self,
        kernel: KernelIR,
        cycles: Dict[int, int],
        stall_by_pipe: Dict[str, int],
        makespan: int,
    ) -> List[PipeProfile]:
        profiles: Dict[str, PipeProfile] = {}
        for op in kernel.ops:
            entry = profiles.get(op.pipe.value)
            if entry is None:
                entry = PipeProfile(
                    pipe=op.pipe.value,
                    op_count=0,
                    busy_cycles=0,
                    stall_cycles=0,
                    utilization=0.0,
                )
                profiles[op.pipe.value] = entry
            entry.op_count += 1
            entry.busy_cycles += cycles[op.index]
        for entry in profiles.values():
            entry.stall_cycles = stall_by_pipe.get(entry.pipe, 0)
            entry.utilization = (
                entry.busy_cycles / makespan if makespan else 0.0
            )
            entry.makespan_cycles = makespan
        order = {pipe.value: i for i, pipe in enumerate(REAL_PIPES)}
        return sorted(profiles.values(), key=lambda p: order.get(p.pipe, 99))

    @staticmethod
    def _drain_cycles(kernel: KernelIR, end: Dict[int, int], makespan: int) -> int:
        """Serialized tail: time after the last bulk transfer or compute op."""
        work_ends = [end[i.index] for i in kernel.ops if isinstance(i, ApiCallOp)]
        if not work_ends:
            return 0
        return max(0, makespan - max(work_ends))

    def _classify(
        self,
        kernel: KernelIR,
        active: Sequence[PipeProfile],
        total_busy: int,
        total_stall: int,
        drain: int,
        makespan: int,
        critical: Optional[PipeProfile],
    ) -> str:
        if not active or makespan <= 0:
            return "NONE"
        # A serialized tail is the more specific diagnosis: when a quarter of
        # the makespan runs after the last bulk op, fix the drain first.
        if drain >= self.drain_ratio_threshold * makespan:
            return "DRAIN_BOUND"
        if total_busy and total_stall >= self.stall_ratio_threshold * total_busy:
            return "SYNC_BOUND"
        if critical is None:
            return "NONE"
        pipe = Pipe.parse(critical.pipe)
        if pipe in self._MEMORY_PIPES:
            return "MEMORY_BOUND"
        return "COMPUTE_BOUND"

    # -- advisories ------------------------------------------------------------

    def _report_sync_stalls(
        self,
        kernel: KernelIR,
        active: Sequence[PipeProfile],
        total_busy: int,
        total_stall: int,
        stall_by_route: Dict[str, int],
        start: Dict[int, int],
        makespan: int,
    ) -> None:
        if not total_busy or total_stall < self.stall_ratio_threshold * total_busy:
            return
        worst_pipe = max(active, key=lambda p: p.stall_cycles)
        worst_wait = self._worst_wait(kernel, start)
        ratio = total_stall / total_busy
        ranked = sorted(stall_by_route.items(), key=lambda kv: -kv[1])[:2]
        route_note = (
            " Largest hand-off exposures: "
            + ", ".join(f"{route} {cycles} cycles" for route, cycles in ranked)
            + "."
            if ranked
            else ""
        )
        loc = worst_wait.loc if worst_wait is not None else kernel.loc
        self.diags.add(
            Code.PERF_SYNC_STALL,
            Severity.WARNING,
            f"cross-queue hand-offs exposed {total_stall} stall cycles against "
            f"{total_busy} cycles of issued work ({ratio:.0%}) on a "
            f"{makespan}-cycle critical path; {worst_pipe.pipe} alone idled "
            f"{worst_pipe.stall_cycles} cycles at flags with its own queue "
            f"empty.{route_note}",
            loc,
            hardware_domain=worst_pipe.pipe,
            remediation=(
                "Overlap the stages these flags connect: issue the next tile's "
                "load before waiting on the current tile's result (double "
                "buffer with two event ids), batch several small tiles into "
                "one larger transfer to amortise measured hand-off costs. "
                "Keep all synchronization required by data dependencies."
            ),
            makespan_cycles=makespan,
            stall_cycles=total_stall,
            stall_ratio=round(ratio, 4),
            worst_pipe=worst_pipe.pipe,
            worst_pipe_stall=worst_pipe.stall_cycles,
            stall_by_route=dict(
                sorted(stall_by_route.items(), key=lambda kv: -kv[1])
            ),
        )

    def _report_cube_underutil(
        self, kernel: KernelIR, active: Sequence[PipeProfile], makespan: int
    ) -> None:
        cube = next((p for p in active if p.pipe == Pipe.M.value), None)
        if cube is None or len(active) < 2:
            return
        if cube.utilization >= self.cube_util_threshold:
            return
        first_cube = next(
            (op for op in kernel.ops if isinstance(op, ApiCallOp)
             and op.pipe is Pipe.M),
            None,
        )
        loc = first_cube.loc if first_cube is not None else kernel.loc
        self.diags.add(
            Code.PERF_CUBE_UNDERUTIL,
            Severity.WARNING,
            f"the cube unit is busy only {cube.busy_cycles} of {makespan} "
            f"modeled cycles ({cube.utilization:.0%}) while {len(active) - 1} "
            "other pipeline(s) run - the contraction is not the critical "
            "path, the feeding chain is",
            loc,
            hardware_domain=Pipe.M.value,
            remediation=(
                "Deepen the software pipeline: prefetch and fractal-load the "
                "next tile while the cube works on the current one (one more "
                "buffer slot per stage usually suffices), or enlarge K so one "
                "contraction covers more of the move work feeding it."
            ),
            cube_busy_cycles=cube.busy_cycles,
            makespan_cycles=makespan,
            cube_utilization=round(cube.utilization, 4),
        )

    @staticmethod
    def _worst_wait(kernel: KernelIR, start: Dict[int, int]) -> Optional[FlagOp]:
        """The WaitFlag whose issue was delayed the longest by synchronisation."""
        worst: Optional[FlagOp] = None
        worst_gap = -1
        for op in kernel.ops:
            if not (isinstance(op, FlagOp) and op.flag_kind is FlagKind.WAIT):
                continue
            gap = start.get(op.index, 0)
            if gap > worst_gap:
                worst, worst_gap = op, gap
        return worst
