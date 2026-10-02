"""Pipeline synchronisation verification: pairing, priming and deadlock.

Model
-----
A DaVinci AI Core is a set of pipelines (``PIPE_MTE2``, ``PIPE_V``, ...), each
consuming its own instruction queue strictly in order and otherwise running
free.  ``SetFlag<HardEvent::SRC_DST>(id)`` raises a counting flag on pipeline
``SRC``; ``WaitFlag<HardEvent::SRC_DST>(id)`` blocks pipeline ``DST`` until the
flag is available, then consumes one token.  A ``(route, event_id)`` pair is
therefore a counting semaphore, and the kernel is a **marked graph**: nodes are
synchronisation operations, edges are happens-before constraints, and each edge
carries the number of tokens initially available on it.

Three edge kinds are built:

* *program order* - consecutive operations on the same pipeline, 0 tokens;
* *back edges* - the last operation on a pipeline to its first, 1 token, closing
  the loop body so loop-carried dependencies are visible;
* *synchronisation* - each ``SetFlag`` to the ``WaitFlag`` that consumes it,
  with 0 tokens when the set precedes the wait in program order, and the
  prologue priming count when the wait precedes the set (a loop-carried
  dependency satisfied by the previous iteration).

**A marked graph deadlocks exactly when some directed cycle holds no tokens.**
Since every 0-token edge advances the trace index *except* an unprimed
loop-carried synchronisation edge, the search reduces to cycle detection over
the 0-token subgraph - linear, exact, and it hands back the full circular-wait
path for the report.

What this catches
-----------------
* ``SetFlag``/``WaitFlag`` that are not one-to-one (``AKA2001``/``AKA2002``),
  including per-loop-body imbalance (``AKA2006``) that drifts the semaphore.
* Reserved ``EVENT_ID`` 6 and 7, which the runtime owns (``AKA2003``).
* Loop-carried waits with no prologue priming: iteration 0 blocks for ever
  (``AKA2005``) - the single most common double-buffering hang.
* Circular waits across pipelines (``AKA2004``).
* ``PipeBarrier(PIPE_ALL)`` as a performance antipattern (``AKA3001``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from ..diagnostics import Code, Severity, SourceLoc
from ..hardware import REAL_PIPES, HardEventRoute, Pipe
from ..ir import BarrierOp, FlagKind, FlagOp, KernelIR, Operation
from .base import Checker

__all__ = ["DeadlockChecker", "Channel", "SyncEdge"]


#: A synchronisation channel: one ``HardEvent`` route plus one event id.
Channel = Tuple[str, object]

#: Ceiling on how many circular waits are reported as diagnostics.
_MAX_REPORTED_CYCLES = 6
#: Ceiling on elementary-cycle enumeration, which is exponential in the worst
#: case; kernels this dense are pathological, and six reports is already plenty.
_MAX_ENUMERATED_CYCLES = 256


@dataclass(frozen=True)
class SyncEdge:
    """A matched ``SetFlag`` -> ``WaitFlag`` pair."""

    setter: FlagOp
    waiter: FlagOp
    channel: Channel
    #: Initial tokens on this edge (prologue primes for a loop-carried edge).
    tokens: int
    #: ``True`` when the wait precedes the set in program order.
    loop_carried: bool

    def to_json(self) -> Dict[str, object]:
        return {
            "channel": {"route": self.channel[0], "event_id": self.channel[1]},
            "set_index": self.setter.index,
            "set_line": self.setter.loc.line,
            "wait_index": self.waiter.index,
            "wait_line": self.waiter.loc.line,
            "tokens": self.tokens,
            "loop_carried": self.loop_carried,
        }


@dataclass
class _ChannelStats:
    """Set/wait tallies for one channel, split by region."""

    sets: List[FlagOp] = field(default_factory=list)
    waits: List[FlagOp] = field(default_factory=list)
    sets_by_loop: Dict[Optional[int], List[FlagOp]] = field(
        default_factory=lambda: defaultdict(list)
    )
    waits_by_loop: Dict[Optional[int], List[FlagOp]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @property
    def route_name(self) -> str:
        source = self.sets or self.waits
        route = source[0].route if source else None
        return route.name if route else "?"


class DeadlockChecker(Checker):
    """Verifies flag pairing, priming and the absence of circular waits."""

    name = "deadlock"

    def check(self, kernel: KernelIR) -> None:
        flags = kernel.flag_ops()
        barriers = kernel.barriers()

        self._check_event_ids(flags)
        self._check_self_routes(flags)
        self._check_barriers(kernel, barriers)

        if not flags:
            self.ctx.publish(f"sync_graph::{kernel.name}", {"nodes": [], "edges": []})
            return

        stats = self._tally(flags)
        self._check_orphans(kernel, stats)
        self._check_loop_balance(kernel, stats)
        self._check_outer_balance(kernel, stats)
        self._check_double_sets(stats)

        edges = self._match_channels(kernel, stats)
        graph = self._build_graph(kernel, flags, barriers, edges)

        # Find the token-free cycles first, so the priming diagnostic can say
        # which circular wait each missing prime closes. Reporting a missing
        # prime *and* every cycle it creates would be the same root cause twice.
        cycles = self._find_token_free_cycles(graph)
        unprimed = self._check_priming(kernel, edges, cycles)
        self._report_cycles(kernel, graph, cycles, unprimed)
        self._publish(kernel, graph, edges, cycles)

    # -- event id hygiene ---------------------------------------------------

    def _check_event_ids(self, flags: Sequence[FlagOp]) -> None:
        reserved = sorted(self.hw.chip.reserved_event_ids)
        usable = [
            i for i in range(self.hw.chip.max_event_id + 1) if i not in set(reserved)
        ]
        for op in flags:
            if op.event_id is None:
                self.diags.add(
                    Code.SYMBOLIC_EVENT_ID,
                    Severity.WARNING,
                    f"{op.label} uses an event id the analyzer cannot resolve "
                    f"({op.event_id_text or 'unknown'}); its pairing is unchecked",
                    op.loc,
                    hardware_domain=op.pipe.value,
                    remediation=(
                        "Use a literal EVENT_IDn, or unroll the ping/pong selection "
                        "so each buffer slot has a fixed event id. A runtime-selected "
                        "id defeats static pairing analysis."
                    ),
                    operation=op.label,
                )
                continue
            if self.hw.is_event_id_reserved(op.event_id):
                self.diags.add(
                    Code.RESERVED_EVENT_ID,
                    Severity.FATAL,
                    f"{op.label} uses EVENT_ID{op.event_id}, which is reserved by "
                    "the runtime; the framework may consume or raise this flag "
                    "behind your back",
                    op.loc,
                    hardware_domain=op.pipe.value,
                    remediation=(
                        f"Reserved ids on {self.hw.chip.display_name}: "
                        f"{', '.join(f'EVENT_ID{i}' for i in reserved)}. "
                        f"Use one of EVENT_ID{usable[0]}..EVENT_ID{usable[-1]} instead."
                    ),
                    event_id=op.event_id,
                    reserved_ids=reserved,
                )
            elif not self.hw.is_event_id_in_range(op.event_id):
                self.diags.add(
                    Code.EVENT_ID_OUT_OF_RANGE,
                    Severity.FATAL,
                    f"{op.label} uses event id {op.event_id}, outside the "
                    f"0..{self.hw.chip.max_event_id} range the hardware provides",
                    op.loc,
                    hardware_domain=op.pipe.value,
                    remediation=f"Use EVENT_ID0..EVENT_ID{self.hw.chip.max_event_id}, "
                    f"excluding the reserved {reserved}.",
                    event_id=op.event_id,
                )

    def _check_self_routes(self, flags: Sequence[FlagOp]) -> None:
        for op in flags:
            if op.route is not None and op.route.is_self_route:
                self.diags.add(
                    Code.SELF_ROUTE_SYNC,
                    Severity.WARNING,
                    f"{op.label} synchronises {op.route.src.value} with itself; "
                    "a pipeline is already ordered against its own instructions",
                    op.loc,
                    hardware_domain=op.pipe.value,
                    remediation="Remove this flag pair, or correct the route if a "
                    "different pipeline was intended.",
                    route=op.route.name,
                )

    # -- barriers -----------------------------------------------------------

    def _check_barriers(self, kernel: KernelIR, barriers: Sequence[BarrierOp]) -> None:
        previous: Optional[BarrierOp] = None
        for barrier in barriers:
            if barrier.is_global:
                self.diags.add(
                    Code.GLOBAL_BARRIER,
                    Severity.WARNING,
                    "PipeBarrier(PIPE_ALL) drains every pipeline, serialising MTE2, "
                    "Vector, MTE3 and Cube at this point and discarding the overlap "
                    "that double buffering exists to create",
                    barrier.loc,
                    hardware_domain="PIPE_ALL",
                    remediation=self._barrier_remediation(kernel, barrier),
                    operation=barrier.label,
                )
            if (
                previous is not None
                and previous.target is barrier.target
                and barrier.index == previous.index + 1
            ):
                self.diags.add(
                    Code.REDUNDANT_BARRIER,
                    Severity.INFO,
                    f"consecutive {barrier.label} has no effect; the preceding "
                    "barrier already drained these pipelines",
                    barrier.loc,
                    hardware_domain=barrier.target.value,
                    remediation="Delete the duplicate barrier.",
                )
            previous = barrier

    def _barrier_remediation(self, kernel: KernelIR, barrier: BarrierOp) -> str:
        """Name the specific pipelines a localised HardEvent should connect."""
        before = self._last_pipe_before(kernel, barrier.index)
        after = self._first_pipe_after(kernel, barrier.index)
        if before is not None and after is not None and before is not after:
            route = HardEventRoute.from_pipes(before, after)
            return (
                f"Replace it with the localised pair that expresses the real "
                f"dependency here:\n"
                f"    AscendC::SetFlag<AscendC::HardEvent::{route.name}>(EVENT_ID0);"
                f"   // on {before.value}, after the producing instruction\n"
                f"    AscendC::WaitFlag<AscendC::HardEvent::{route.name}>(EVENT_ID0);"
                f"   // on {after.value}, before the consuming instruction\n"
                f"That stalls only {after.value} on {before.value}, leaving the other "
                "pipelines free to run ahead."
            )
        return (
            "Replace it with a HardEvent pair naming the two pipelines that actually "
            "share data, for example SetFlag<HardEvent::MTE2_V>(EVENT_ID0) on the "
            "producer and WaitFlag<HardEvent::MTE2_V>(EVENT_ID0) on the consumer."
        )

    @staticmethod
    def _last_pipe_before(kernel: KernelIR, index: int) -> Optional[Pipe]:
        for op in reversed(kernel.ops[:index]):
            if op.pipe.is_real:
                return op.pipe
        return None

    @staticmethod
    def _first_pipe_after(kernel: KernelIR, index: int) -> Optional[Pipe]:
        for op in kernel.ops[index + 1 :]:
            if op.pipe.is_real:
                return op.pipe
        return None

    # -- channel tallies ----------------------------------------------------

    def _tally(self, flags: Sequence[FlagOp]) -> Dict[Channel, _ChannelStats]:
        stats: Dict[Channel, _ChannelStats] = defaultdict(_ChannelStats)
        for op in flags:
            if op.route is None or op.event_id is None:
                continue  # already reported as unresolvable
            entry = stats[op.channel]
            if op.flag_kind is FlagKind.SET:
                entry.sets.append(op)
                entry.sets_by_loop[op.loop_id].append(op)
            else:
                entry.waits.append(op)
                entry.waits_by_loop[op.loop_id].append(op)
        return stats

    def _check_orphans(
        self, kernel: KernelIR, stats: Dict[Channel, _ChannelStats]
    ) -> None:
        """Channels where one side is missing entirely."""
        for channel, entry in stats.items():
            route, event_id = channel
            if entry.sets and not entry.waits:
                for op in entry.sets:
                    self.diags.add(
                        Code.UNMATCHED_SET_FLAG,
                        self._soften(Severity.FATAL, op),
                        f"SetFlag<{route}>(EVENT_ID{event_id}) is never consumed: "
                        f"no WaitFlag<{route}>(EVENT_ID{event_id}) exists in "
                        f"{kernel.name!r}",
                        op.loc,
                        hardware_domain=op.pipe.value,
                        remediation=(
                            f"Add AscendC::WaitFlag<AscendC::HardEvent::{route}>"
                            f"(EVENT_ID{event_id}); on {op.route.dst.value if op.route else '?'} "
                            "before the instruction that consumes the data, or delete "
                            "this SetFlag. An unconsumed flag leaks a hardware event "
                            "slot and will eventually stall the pipeline."
                        ),
                        route=route,
                        event_id=event_id,
                    )
            elif entry.waits and not entry.sets:
                for op in entry.waits:
                    self.diags.add(
                        Code.UNMATCHED_WAIT_FLAG,
                        self._soften(Severity.FATAL, op),
                        f"WaitFlag<{route}>(EVENT_ID{event_id}) blocks "
                        f"{op.pipe.value} on a flag that is never set - this is an "
                        "unconditional hang",
                        op.loc,
                        hardware_domain=op.pipe.value,
                        remediation=(
                            f"Add AscendC::SetFlag<AscendC::HardEvent::{route}>"
                            f"(EVENT_ID{event_id}); on "
                            f"{op.route.src.value if op.route else '?'} after the "
                            "instruction that produces the data, or delete this "
                            "WaitFlag."
                        ),
                        route=route,
                        event_id=event_id,
                    )

    def _check_loop_balance(
        self, kernel: KernelIR, stats: Dict[Channel, _ChannelStats]
    ) -> None:
        """Within one loop body, sets and waits must balance per channel."""
        for channel, entry in stats.items():
            route, event_id = channel
            loop_ids = set(entry.sets_by_loop) | set(entry.waits_by_loop)
            for loop_id in loop_ids:
                if loop_id is None:
                    continue
                loop = kernel.loops.get(loop_id)
                if loop is None:
                    continue
                sets = entry.sets_by_loop.get(loop_id, [])
                waits = entry.waits_by_loop.get(loop_id, [])
                if len(sets) == len(waits):
                    continue
                anchor = (sets or waits)[0]
                delta = len(sets) - len(waits)
                direction = (
                    f"{delta} more SetFlag than WaitFlag"
                    if delta > 0
                    else f"{-delta} more WaitFlag than SetFlag"
                )
                consequence = (
                    "the flag counter grows by one every iteration until the event "
                    "slot overflows"
                    if delta > 0
                    else "the flag counter is drained faster than it is filled, so a "
                    "later iteration blocks for ever"
                )
                self.diags.add(
                    Code.LOOP_FLAG_IMBALANCE,
                    self._soften(Severity.FATAL, anchor),
                    f"loop body at line {loop.loc.line} has {direction} for "
                    f"{route}/EVENT_ID{event_id} ({len(sets)} set, {len(waits)} wait); "
                    f"{consequence}",
                    anchor.loc,
                    hardware_domain=anchor.pipe.value,
                    remediation=(
                        f"Every iteration must set and wait {route}/EVENT_ID{event_id} "
                        "the same number of times. Prime the pipeline before the loop "
                        "and drain it after, rather than leaving the body unbalanced."
                    ),
                    related=[("loop header", loop.loc)],
                    route=route,
                    event_id=event_id,
                    sets=len(sets),
                    waits=len(waits),
                )

    def _check_outer_balance(
        self, kernel: KernelIR, stats: Dict[Channel, _ChannelStats]
    ) -> None:
        """Outside loops, prologue primes must equal epilogue drains."""
        for channel, entry in stats.items():
            route, event_id = channel
            if not entry.sets or not entry.waits:
                continue  # already reported by the orphan check
            outer_sets = entry.sets_by_loop.get(None, [])
            outer_waits = entry.waits_by_loop.get(None, [])
            delta = len(outer_sets) - len(outer_waits)
            if delta == 0:
                continue
            has_loops = bool(kernel.loops)
            anchor = (outer_sets or outer_waits)[0]
            if delta > 0:
                message = (
                    f"{len(outer_sets)} SetFlag<{route}>(EVENT_ID{event_id}) outside "
                    f"any loop but only {len(outer_waits)} matching WaitFlag"
                )
                detail = (
                    "the extra priming flags are never drained, leaking hardware "
                    "event slots"
                    if has_loops
                    else "the extra flags are never consumed"
                )
                code, remediation = (
                    Code.UNMATCHED_SET_FLAG,
                    (
                        f"Add {delta} draining AscendC::WaitFlag<AscendC::HardEvent::"
                        f"{route}>(EVENT_ID{event_id}); after the loop, matching the "
                        "priming SetFlag calls before it."
                    ),
                )
            else:
                message = (
                    f"{len(outer_waits)} WaitFlag<{route}>(EVENT_ID{event_id}) outside "
                    f"any loop but only {len(outer_sets)} matching SetFlag"
                )
                detail = "the surplus waits block on flags nothing will raise"
                code, remediation = (
                    Code.UNMATCHED_WAIT_FLAG,
                    (
                        f"Add {-delta} priming AscendC::SetFlag<AscendC::HardEvent::"
                        f"{route}>(EVENT_ID{event_id}); before the loop, or remove the "
                        "surplus WaitFlag calls after it."
                    ),
                )
            self.diags.add(
                code,
                self._soften(Severity.FATAL, anchor),
                f"{message}; {detail}",
                anchor.loc,
                hardware_domain=anchor.pipe.value,
                remediation=remediation,
                route=route,
                event_id=event_id,
                outer_sets=len(outer_sets),
                outer_waits=len(outer_waits),
            )

    def _check_double_sets(self, stats: Dict[Channel, _ChannelStats]) -> None:
        """Two sets on one channel with no wait between them overflow the flag."""
        for channel, entry in stats.items():
            route, event_id = channel
            timeline = sorted(entry.sets + entry.waits, key=lambda op: op.index)
            pending: Optional[FlagOp] = None
            for op in timeline:
                if op.flag_kind is FlagKind.SET:
                    if pending is not None:
                        self.diags.add(
                            Code.FLAG_DOUBLE_SET,
                            Severity.WARNING,
                            f"SetFlag<{route}>(EVENT_ID{event_id}) is raised again "
                            f"before the pending flag from line {pending.loc.line} is "
                            "consumed; the event counter can saturate",
                            op.loc,
                            hardware_domain=op.pipe.value,
                            remediation=(
                                "Insert the matching WaitFlag between the two "
                                "SetFlag calls, or give the second one a distinct "
                                "EVENT_ID so the two dependencies are tracked "
                                "separately."
                            ),
                            related=[("earlier unconsumed SetFlag", pending.loc)],
                            route=route,
                            event_id=event_id,
                        )
                    pending = op
                else:
                    pending = None

    # -- matching and priming ----------------------------------------------

    def _match_channels(
        self, kernel: KernelIR, stats: Dict[Channel, _ChannelStats]
    ) -> List[SyncEdge]:
        """FIFO-match sets to waits, per channel and per region.

        A counting semaphore is consumed in order, so the k-th wait in a region
        takes the k-th set.  Loop bodies are matched independently of the
        surrounding straight-line code: inside the body, a wait that precedes
        its set is satisfied by the previous iteration, and the tokens on that
        edge are the flags the prologue primed.

        Kernels with *peeled* loops are matched globally instead: their head
        and tail iterations are straight-line code that flows flags into and
        out of the steady-state representative cycle, so splitting the channel
        ledger by region would pair a head-primed set with the wrong wait and
        report a primed handshake as unprimed.  Global FIFO matching is the
        hardware's actual semaphore semantics.
        """
        if any(loop.peeled for loop in kernel.loops.values()):
            return self._match_channels_globally(kernel, stats)
        edges: List[SyncEdge] = []
        for channel, entry in stats.items():
            regions = set(entry.sets_by_loop) | set(entry.waits_by_loop)
            for loop_id in regions:
                sets = sorted(entry.sets_by_loop.get(loop_id, []), key=lambda o: o.index)
                waits = sorted(
                    entry.waits_by_loop.get(loop_id, []), key=lambda o: o.index
                )
                primes = self._prime_count(kernel, entry, loop_id)
                for setter, waiter in zip(sets, waits):
                    loop_carried = waiter.index < setter.index
                    edges.append(
                        SyncEdge(
                            setter=setter,
                            waiter=waiter,
                            channel=channel,
                            tokens=primes if loop_carried else 0,
                            loop_carried=loop_carried,
                        )
                    )
        return edges

    def _match_channels_globally(
        self, kernel: KernelIR, stats: Dict[Channel, _ChannelStats]
    ) -> List[SyncEdge]:
        """FIFO-match whole channels across regions (peeled kernels)."""
        edges: List[SyncEdge] = []
        for channel, entry in stats.items():
            sets = sorted(entry.sets, key=lambda o: o.index)
            waits = sorted(entry.waits, key=lambda o: o.index)
            for setter, waiter in zip(sets, waits):
                loop_carried = waiter.index < setter.index
                tokens = 0
                if loop_carried:
                    loop = (
                        kernel.loops.get(waiter.loop_id)
                        if waiter.loop_id is not None
                        else None
                    )
                    if loop is not None:
                        tokens = sum(
                            1
                            for op in entry.sets
                            if op.index < loop.start_index
                            and not loop.contains(op.index)
                        )
                edges.append(
                    SyncEdge(
                        setter=setter,
                        waiter=waiter,
                        channel=channel,
                        tokens=tokens,
                        loop_carried=loop_carried,
                    )
                )
        return edges

    def _prime_count(
        self, kernel: KernelIR, entry: _ChannelStats, loop_id: Optional[int]
    ) -> int:
        """How many flags are raised on this channel before ``loop_id`` starts."""
        if loop_id is None:
            return 0
        loop = kernel.loops.get(loop_id)
        if loop is None:
            return 0
        return sum(
            1
            for op in entry.sets
            if op.index < loop.start_index and not loop.contains(op.index)
        )

    def _check_priming(
        self,
        kernel: KernelIR,
        edges: Sequence[SyncEdge],
        cycles: Sequence[List[int]],
    ) -> Set[Tuple[int, int]]:
        """Report loop-carried waits with nothing priming them.

        Returns the ``(set_index, wait_index)`` edges reported, so cycle
        reporting can skip the cycles these already explain.
        """
        reported: Set[Tuple[int, int]] = set()
        for edge in edges:
            if not edge.loop_carried or edge.tokens > 0:
                continue
            route, event_id = edge.channel
            waiter = edge.waiter
            loop = kernel.loops.get(waiter.loop_id) if waiter.loop_id is not None else None
            loop_line = loop.loc.line if loop else waiter.loc.line
            closes = self._cycle_containing(cycles, edge.setter.index, waiter.index)
            cycle_note = (
                " This wait also closes a circular wait across "
                + ", ".join(
                    sorted({
                        op.pipe.value
                        for op in kernel.ops
                        if op.index in closes and op.pipe.is_real
                    })
                )
                + "."
                if closes
                else ""
            )
            reported.add((edge.setter.index, waiter.index))
            self.diags.add(
                Code.UNPRIMED_LOOP_WAIT,
                self._soften(Severity.FATAL, waiter),
                f"WaitFlag<{route}>(EVENT_ID{event_id}) at the top of the loop body "
                f"(line {loop_line}) consumes a flag that is only raised later in the "
                f"same body (line {edge.setter.loc.line}), so it can only be satisfied "
                "by the previous iteration - but nothing primes it before the loop. "
                f"{waiter.pipe.value} blocks on the first iteration and the kernel "
                "hangs." + cycle_note,
                waiter.loc,
                hardware_domain=waiter.pipe.value,
                remediation=(
                    f"Prime the flag once before the loop:\n"
                    f"    AscendC::SetFlag<AscendC::HardEvent::{route}>"
                    f"(EVENT_ID{event_id});\n"
                    f"and drain it once after the loop:\n"
                    f"    AscendC::WaitFlag<AscendC::HardEvent::{route}>"
                    f"(EVENT_ID{event_id});\n"
                    "so every iteration, including the first, finds a token waiting."
                ),
                related=[
                    ("flag is raised here, after the wait", edge.setter.loc),
                    ("loop header", loop.loc if loop else waiter.loc),
                ],
                route=route,
                event_id=event_id,
                set_line=edge.setter.loc.line,
                wait_line=waiter.loc.line,
                closes_cycle=[int(n) for n in closes],
            )
        return reported

    @staticmethod
    def _cycle_containing(
        cycles: Sequence[List[int]], source: int, target: int
    ) -> List[int]:
        """The first token-free cycle that traverses the edge ``source->target``."""
        for cycle in cycles:
            for position, node in enumerate(cycle):
                nxt = cycle[(position + 1) % len(cycle)]
                if node == source and nxt == target:
                    return cycle
        return []

    # -- graph construction and cycle detection -----------------------------

    def _build_graph(
        self,
        kernel: KernelIR,
        flags: Sequence[FlagOp],
        barriers: Sequence[BarrierOp],
        edges: Sequence[SyncEdge],
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        nodes: List[Operation] = sorted(
            list(flags) + list(barriers), key=lambda op: op.index
        )
        for op in nodes:
            graph.add_node(
                op.index,
                label=op.label,
                pipe=op.pipe.value,
                line=op.loc.line,
                kind=op.kind,
                loop_id=op.loop_id,
            )

        # Program order within each pipeline. A global barrier participates in
        # every pipeline's order, which is what makes it a fence.
        for pipe in REAL_PIPES:
            chain = [
                op
                for op in nodes
                if op.pipe is pipe
                or (isinstance(op, BarrierOp) and op.target in (Pipe.ALL, pipe))
            ]
            for earlier, later in zip(chain, chain[1:]):
                graph.add_edge(earlier.index, later.index, tokens=0, kind="program")
            # Close the loop body so loop-carried dependencies are visible.
            if len(chain) > 1 and any(op.loop_id is not None for op in chain):
                graph.add_edge(chain[-1].index, chain[0].index, tokens=1, kind="back")

        for edge in edges:
            graph.add_edge(
                edge.setter.index,
                edge.waiter.index,
                tokens=edge.tokens,
                kind="sync",
                route=edge.channel[0],
                event_id=edge.channel[1],
                loop_carried=edge.loop_carried,
            )
        return graph

    @staticmethod
    def _token_free_subgraph(graph: nx.DiGraph) -> nx.DiGraph:
        """The subgraph of edges that carry no initial token.

        A marked graph deadlocks exactly when a directed cycle holds no
        tokens, so acyclicity of this subgraph *is* deadlock freedom.
        """
        blocked = nx.DiGraph()
        blocked.add_nodes_from(graph.nodes(data=True))
        for source, target, data in graph.edges(data=True):
            if data.get("tokens", 0) == 0:
                blocked.add_edge(source, target, **data)
        return blocked

    def _find_token_free_cycles(self, graph: nx.DiGraph) -> List[List[int]]:
        blocked = self._token_free_subgraph(graph)
        if nx.is_directed_acyclic_graph(blocked):
            return []
        cycles: List[List[int]] = []
        for cycle in nx.simple_cycles(blocked):
            cycles.append(list(cycle))
            if len(cycles) >= _MAX_ENUMERATED_CYCLES:
                break
        # Shortest cycles first: they are the tightest explanation of the hang.
        cycles.sort(key=lambda c: (len(c), min(c)))
        return cycles

    def _report_cycles(
        self,
        kernel: KernelIR,
        graph: nx.DiGraph,
        cycles: Sequence[List[int]],
        already_explained: Set[Tuple[int, int]],
    ) -> None:
        """Report circular waits that the priming check has not already covered."""
        blocked = self._token_free_subgraph(graph)
        reported = 0
        suppressed = 0
        for cycle in cycles:
            if self._uses_edge(cycle, already_explained):
                suppressed += 1
                continue
            if reported >= _MAX_REPORTED_CYCLES:
                self.diags.add(
                    Code.ANALYSIS_LIMIT,
                    Severity.INFO,
                    f"more than {_MAX_REPORTED_CYCLES} distinct circular waits were "
                    "found; only the shortest are listed",
                    kernel.loc,
                    hardware_domain="sync",
                    remediation="Fix the reported cycles and re-run.",
                )
                break
            self._report_cycle(kernel, blocked, cycle)
            reported += 1
        if suppressed:
            self.ctx.publish(f"suppressed_cycles::{kernel.name}", suppressed)

    @staticmethod
    def _uses_edge(cycle: Sequence[int], edges: Set[Tuple[int, int]]) -> bool:
        for position, node in enumerate(cycle):
            nxt = cycle[(position + 1) % len(cycle)]
            if (node, nxt) in edges:
                return True
        return False

    def _report_cycle(
        self, kernel: KernelIR, graph: nx.DiGraph, cycle: Sequence[int]
    ) -> None:
        ordered = list(cycle)
        anchor_index = min(ordered)
        anchor = next((op for op in kernel.ops if op.index == anchor_index), None)
        loc = anchor.loc if anchor is not None else kernel.loc

        steps: List[str] = []
        related: List[Tuple[str, SourceLoc]] = []
        for position, node in enumerate(ordered):
            nxt = ordered[(position + 1) % len(ordered)]
            data = graph.get_edge_data(node, nxt) or {}
            info = graph.nodes[node]
            arrow = (
                f"--{data.get('route')}-->"
                if data.get("kind") == "sync"
                else "--program order-->"
            )
            steps.append(
                f"  line {info['line']:>4}  [{info['pipe']:<9}] {info['label']} {arrow}"
            )
            op = next((o for o in kernel.ops if o.index == node), None)
            if op is not None and len(related) < 8:
                related.append((f"{op.pipe.value}: {op.label}", op.loc))

        pipes = sorted({graph.nodes[n]["pipe"] for n in ordered})
        self.diags.add(
            Code.DEADLOCK_CYCLE,
            Severity.FATAL,
            "circular wait across "
            + ", ".join(pipes)
            + " - no pipeline in this cycle can make progress:\n"
            + "\n".join(steps)
            + f"\n  (back to line {graph.nodes[ordered[0]]['line']})",
            loc,
            hardware_domain=", ".join(pipes),
            remediation=(
                "Break the cycle by reordering so that every flag is raised before "
                "the pipeline that waits on it reaches its WaitFlag, or by priming "
                "one of the loop-carried flags before the loop so the first "
                "iteration has a token to consume."
            ),
            related=related,
            cycle=[int(n) for n in ordered],
            cycle_lines=[graph.nodes[n]["line"] for n in ordered],
            pipes=pipes,
        )

    # -- artifacts ----------------------------------------------------------

    def _publish(
        self,
        kernel: KernelIR,
        graph: nx.DiGraph,
        edges: Sequence[SyncEdge],
        cycles: Sequence[List[int]],
    ) -> None:
        blocked = self._token_free_subgraph(graph)
        acyclic = nx.is_directed_acyclic_graph(blocked)
        # On an acyclic graph the topological order is both a proof of
        # deadlock freedom and a legal issue order for the pipelines.
        schedule = list(nx.topological_sort(blocked)) if acyclic else None

        self.ctx.publish(
            f"sync_graph::{kernel.name}",
            {
                "nodes": [
                    {"index": node, **dict(data)}
                    for node, data in sorted(graph.nodes(data=True))
                ],
                "edges": [
                    {"from": source, "to": target, **dict(data)}
                    for source, target, data in graph.edges(data=True)
                ],
                "sync_pairs": [edge.to_json() for edge in edges],
                "acyclic": acyclic,
                "topological_order": schedule,
                "cycles": [[int(n) for n in cycle] for cycle in cycles],
                "pipes": [pipe.value for pipe in kernel.active_pipes()],
                "peeled_loops": [
                    {
                        "id": loop.id,
                        "steady_first": loop.steady_first,
                        "steady_reps": loop.steady_reps,
                        "peeled_head": loop.peeled_head,
                        "peeled_tail": loop.peeled_tail,
                    }
                    for loop in kernel.loops.values()
                    if loop.peeled
                ],
            },
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _soften(severity: Severity, op: Operation) -> Severity:
        """Downgrade to a warning when the operation sits on a conditional path.

        The trace walks both arms of an ``if``, so a pairing imbalance that only
        exists because of that over-approximation should not be fatal.
        """
        if severity is Severity.FATAL and op.conditional:
            return Severity.WARNING
        return severity
