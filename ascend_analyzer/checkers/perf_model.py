"""Analytical pipeline performance and overlap profiler.

Model
-----
Once the deadlock checker has proven the kernel's marked graph acyclic, its
topological order is a legal issue order - which makes an *as-soon-as-possible*
schedule over the dependency DAG a faithful first-order performance model:

* every traced operation gets a **service time in cycles** from the chip's
  latency model: DMA engines are credited ``bytes / bandwidth`` (MTE2 move-in,
  MTE1 fractal loads, Fixpipe/MTE3 drains), the vector unit retires one
  ``vector_bytes`` register footprint per cycle, and Cube contractions cost
  ``ceil(M/16) * ceil(K/16) * ceil(N/16)`` cycles when the tile shape is
  recoverable from the call site (falling back to an operand-read throughput);
* **program-order edges** (0 cost) chain each pipeline's in-order instruction
  queue, including barriers as full fences;
* **synchronisation edges** - the forward ``SetFlag`` -> ``WaitFlag`` pairs the
  deadlock checker matched - carry the cross-queue hand-off penalty
  (``sync_handoff_cycles``, ~30 cycles).  Loop-carried edges hold a token and
  are already satisfied at issue, so they add nothing.

The ASAP schedule then yields the **critical-path makespan**, per-pipe
**busy / idle / stall** timestamps (a stall is time a pipeline sat at a
``WaitFlag`` with its own queue empty - an exposed bubble), the **concurrency
efficiency** (overlap ratio) and a bottleneck classification:
``DRAIN_BOUND`` (a serialized tail nothing overlaps), ``SYNC_BOUND``
(hand-off bubbles dominate), ``MEMORY_BOUND`` (a move engine drives the
critical path) or ``COMPUTE_BOUND`` (the cube/vector unit does).

Findings are advisories in the 4xxx block: ``AKA4001`` for exposed sync
bubbles, ``AKA4002`` for cube underutilization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..diagnostics import Code, Severity
from ..hardware import REAL_PIPES, PhysicalDomain, Pipe
from ..ir import ApiCallOp, BarrierOp, FlagKind, FlagOp, KernelIR, Operation
from .base import Checker

__all__ = ["PerfModelChecker", "PipeProfile"]

#: Issue cost of a flag, barrier or scalar instruction, in cycles.
_ISSUE_CYCLES = 1

#: Operand transfer whose volume cannot be recovered: assume one 32 B block.
_DEFAULT_TRANSFER_BYTES = 32

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
                "overlap_ratio": round(overlap, 4),
                "active_pipes": len(active),
                "critical_pipe": critical.pipe if critical else None,
                "bottleneck": bottleneck,
                "pipes": [p.to_json() for p in pipes],
                "handoff_cycles": self.hw.chip.sync_handoff_cycles,
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

        Every dependency edge runs forward in trace index (program order per
        pipeline, forward sync pairs), so index order is a topological order
        and one linear pass settles every timestamp.
        """
        cycles = {op.index: self._service_cycles(kernel, op) for op in kernel.ops}

        preds: Dict[int, List[Tuple[int, int, str, str]]] = {}
        for edge in graph.get("edges") or []:
            if edge.get("kind") != "sync":
                continue  # program/back edges are rebuilt below, exactly
            if edge.get("tokens", 0) != 0:
                continue  # loop-carried: satisfied by the previous iteration
            source, target = edge.get("from"), edge.get("to")
            if not (isinstance(source, int) and isinstance(target, int)):
                continue
            if target <= source:
                continue
            preds.setdefault(target, []).append(
                (
                    source,
                    self.hw.chip.sync_handoff_cycles,
                    "sync",
                    str(edge.get("route", "?")),
                )
            )

        start: Dict[int, int] = {}
        end: Dict[int, int] = {}
        prev_end_by_pipe: Dict[Pipe, int] = {}
        fence_by_pipe: Dict[Pipe, int] = {}
        stall_by_pipe: Dict[str, int] = {}
        stall_by_route: Dict[str, int] = {}

        def effective_prev(pipe: Pipe) -> int:
            return max(prev_end_by_pipe.get(pipe, 0), fence_by_pipe.get(pipe, 0))

        for op in kernel.ops:  # trace order == topological order
            ready = 0
            binding: Optional[Tuple[int, str]] = None
            for source, weight, kind, route in preds.get(op.index, ()):
                arrival = end.get(source, 0) + weight
                if arrival > ready:
                    ready = arrival
                    binding = (source, route)

            if isinstance(op, BarrierOp):
                # A barrier fences every queue it names, so its issue waits
                # for them all and they in turn wait for it.
                fenced = list(REAL_PIPES) if op.target is Pipe.ALL else [op.target]
                ready = max(ready, max((effective_prev(p) for p in fenced), default=0))
            else:
                prev = effective_prev(op.pipe)
                if prev > ready:
                    # The pipe's own in-order queue is the binding constraint:
                    # not a synchronisation bubble.
                    ready = prev
                    binding = None
                elif (
                    isinstance(op, FlagOp)
                    and op.flag_kind is FlagKind.WAIT
                    and binding is not None
                ):
                    gap = ready - prev
                    if gap > 0:
                        # A cross-queue hand-off held this pipe back while its
                        # own queue was empty: an exposed bubble.
                        stall_by_pipe[op.pipe.value] = (
                            stall_by_pipe.get(op.pipe.value, 0) + gap
                        )
                        if binding[1] != "?":
                            stall_by_route[binding[1]] = (
                                stall_by_route.get(binding[1], 0) + gap
                            )

            start[op.index] = ready
            end[op.index] = ready + cycles[op.index]
            if isinstance(op, BarrierOp):
                for pipe in fenced:
                    fence_by_pipe[pipe] = end[op.index]
            else:
                prev_end_by_pipe[op.pipe] = end[op.index]

        return start, end, cycles, stall_by_pipe, stall_by_route

    # -- service times ------------------------------------------------------

    def _service_cycles(self, kernel: KernelIR, op: Operation) -> int:
        if isinstance(op, ApiCallOp):
            if op.name in self._CUBE_APIS:
                return self._cube_cycles(kernel, op)
            bytes_moved = self._transfer_bytes(kernel, op)
            bandwidth = self.hw.pipe_bytes_per_cycle(op.pipe)
            if bandwidth is None:
                return _ISSUE_CYCLES
            return max(1, -(-bytes_moved // bandwidth))
        return _ISSUE_CYCLES

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
            return _ISSUE_CYCLES
        throughput = self.hw.chip.cube_read_bytes_per_cycle
        return max(1, -(-operands // throughput))

    def _cube_shape(self, kernel: KernelIR, op: ApiCallOp) -> Optional[Tuple[int, int, int]]:
        """Recover ``(M, K, N)`` from a shape argument.

        Cube kernels spell the tile shape as ``mmad_t::shape_t((uint16_t)M,
        (uint16_t)K, (uint16_t)N)``: the identifiers resolve against the
        kernel's folded ``constexpr``/``#define`` environment.  Any argument
        naming three known constants in order is accepted as ``(m, k, n)``.
        """
        constants = kernel.constants
        for arg in op.args:
            if arg.tensor:
                continue
            found = [
                constants[name]
                for name in _IDENTIFIER_RE.findall(arg.text)
                if name in constants
            ]
            if len(found) >= 3:
                return (found[0], found[1], found[2])
        return None

    @staticmethod
    def _transfer_bytes(kernel: KernelIR, op: ApiCallOp) -> int:
        """Volume of one DMA / vector operation, in bytes.

        The on-core (SRAM) participant bounds the moved tile: a ``GlobalTensor``
        spans the whole buffer, but a single ``DataCopy`` only moves the tile
        that lands in UB/L1/L0.  GM-only calls fall back to the count
        parameter, then to one hardware block.
        """
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
                return max(_DEFAULT_TRANSFER_BYTES, arg.value * dst.elem_size)
        return _DEFAULT_TRANSFER_BYTES

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
                "one larger transfer so each 30-cycle hand-off amortises, or "
                "drop handshakes between stages that already have a real "
                "data dependency."
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
