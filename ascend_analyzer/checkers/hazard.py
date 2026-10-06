"""Happens-before sufficiency for cross-pipeline tensor traffic (AKA2010).

Model
-----
Each DaVinci pipeline runs its own instruction queue strictly in order, and
two different pipelines are *not* ordered against each other unless the
program orders them.  The two ordering primitives are:

* ``SetFlag<HardEvent::A_B>(id)`` raised on pipe ``A`` paired with
  ``WaitFlag<HardEvent::A_B>(id)`` blocking pipe ``B`` - a counting
  semaphore carrying A-happens-before-B;
* ``PipeBarrier(PIPE_ALL)`` - every pipeline drains before anything after it.

So a tensor written on pipe ``A`` and later read on pipe ``B != A`` is only
safe when the program put one of those primitives between the write and the
read.  This checker tests exactly that, for every ``(tensor, A, B)`` triple:

* **forward** (write precedes read in trace order): a set on an ``A -> B``
  channel *after* the write and a wait on that channel *before* the read -
  the classic ``DataCopy; SetFlag<MTE2_V>; WaitFlag<MTE2_V>; Add`` prologue -
  or a global barrier between them;
* **loop-carried** (read precedes write, both inside one loop body): the same
  evidence predicate, which then matches the ping-pong ``WaitFlag ... compute
  ... DataCopy ... SetFlag`` body shape - the wait guards this iteration's
  read against the previous iteration's write.

It also owns two sharper members of the same family:

* **vacuous handshakes** (below) - a ``SetFlag``/``WaitFlag`` pair that is
  written like the drain idiom but orders nothing, because the producing
  pipeline has no outstanding asynchronous work at the set (the fake
  ``WaitLoad()`` that runs before the loads it claims to guard), or because
  the flag names a producing pipeline other than the one with the work (a
  ``SetFlag<HardEvent::MTE2_V>`` that follows vector instructions, not DMA);
* **loop-invariant GM flushes** (AKA2012) - a store to global memory inside
  a loop whose destination ignores the loop's induction variable, so every
  iteration destroys the previous one's partial sums of the inner reduction.

What is deliberately *not* flagged
----------------------------------
* same-pipe pairs - program order already serialises them;
* tensors whose declaration chain roots in a ``TQue`` accessor
  (``AllocTensor``/``DeQue``): the queue machinery raises and consumes its
  event flags inside the CANN runtime, where this analyzer cannot see them;
* pairs split across ``AIC``/``AIV`` core views of a mix kernel - separate
  binaries with per-core UB, coordinated through GM by the runtime;
* conditional operations, whose execution this trace cannot decide.

The verdict is a performance/correctness warning rather than a rejection: a
missing handshake can still win by timing on every run - which is precisely
why it is worth naming.

The check is specified as "AKA2007" in the task list, but that code was
already assigned to FLAG_DOUBLE_SET when the analyzer shipped; it carries
AKA2010 (MISSING_SYNC), which the code table has reserved for exactly this
finding.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..diagnostics import Code, Severity, SourceLoc
from ..hardware import REAL_PIPES, Pipe, PhysicalDomain
from ..backend.parity import (
    GmStoreProbe,
    LoopProbe,
    SyncPairProbe,
    invariant_flush_loops,
    vacuous_sync_pairs,
)
from ..backend.trace import dependency_graph
from ..backend.safety import Access, unordered_hazards
import networkx as nx
from ..ir import ApiCallOp, CoreView, FlagKind, KernelIR, TensorDecl
from .base import Checker

__all__ = ["HazardChecker"]


#: A synchronisation channel: ``(source pipe, destination pipe, event id)``.
ChannelKey = Tuple[Pipe, Pipe, Optional[int]]

#: Cap on diagnostics per kernel: the artifact carries the full census, the
#: report needs enough to act on.  Hazards cluster - one missing handshake
#: fans out over every tensor that crosses it - so a small cap loses little.
_MAX_REPORTS = 8


class HazardChecker(Checker):
    """Flags cross-pipeline write->read pairs with no happens-before edge."""

    name = "hazard"

    def check(self, kernel: KernelIR) -> None:
        sync = self.ctx.artifacts.get(f"sync_graph::{kernel.name}", {})
        self._dependencies = dependency_graph(kernel, sync, self.hw)
        sets: Dict[ChannelKey, List[int]] = defaultdict(list)
        waits: Dict[ChannelKey, List[int]] = defaultdict(list)
        for op in kernel.flag_ops():
            if op.route is None or op.event_id is None or op.conditional:
                continue  # symbolic channels are the deadlock checker's to name
            key: ChannelKey = (op.route.src, op.route.dst, op.event_id)
            (sets if op.flag_kind is FlagKind.SET else waits)[key].append(op.index)
        for indices in list(sets.values()) + list(waits.values()):
            indices.sort()

        barriers = sorted(
            op.index
            for op in kernel.barriers()
            if op.is_global and not op.conditional
        )

        writers: Dict[str, List[ApiCallOp]] = defaultdict(list)
        readers: Dict[str, List[ApiCallOp]] = defaultdict(list)
        for op in kernel.api_calls():
            if op.pipe not in REAL_PIPES or op.conditional:
                continue
            for name in op.writes:
                writers[name].append(op)
            for name in op.reads:
                readers[name].append(op)

        checked = hazards = 0
        reported = 0
        seen: Set[Tuple[str, Pipe, Pipe]] = set()
        for tensor in sorted(set(writers) & set(readers)):
            if self._queue_mediated(kernel, tensor):
                continue
            for write in writers[tensor]:
                for read in readers[tensor]:
                    if write.pipe is read.pipe:
                        continue
                    if not self._same_view(write, read):
                        continue
                    key = (tensor, write.pipe, read.pipe)
                    if key in seen:
                        continue
                    seen.add(key)
                    checked += 1
                    if self._synchronized(write, read, sets, waits, barriers):
                        continue
                    hazards += 1
                    if reported < _MAX_REPORTS:
                        reported += 1
                        self._report(tensor, write, read)

        # Published whether or not anything was found, for the same reason as
        # the AKA3006 coverage artifact: silence must not read as "checked".
        self.ctx.publish(
            f"hazard_coverage::{kernel.name}",
            {
                "pairs_checked": checked,
                "hazards": hazards,
                "reported": reported,
                "queue_mediated_tensors": sum(
                    1
                    for name in set(writers) & set(readers)
                    if self._queue_mediated(kernel, name)
                ),
            },
        )
        self._check_interval_hazards(kernel)
        self._check_vacuous_handshakes(kernel)
        self._check_loop_invariant_gm_flush(kernel)

    def _check_interval_hazards(self, kernel):
        if kernel.loops:
            self.ctx.publish(f"interval_hazard_coverage::{kernel.name}",
                             {"hazards": [], "skipped": len(kernel.ops),
                              "scope": "requires iteration-specific extents; use expanded backend graph"})
            return
        accesses, operations, skipped = [], {}, 0
        for op in kernel.api_calls():
            if op.conditional or op.core_view is CoreView.NONE:
                skipped += 1
                continue
            operations[op.index] = op
            for mode, names in (("read", op.reads), ("write", op.writes)):
                for name in names:
                    tensor = kernel.tensors.get(name)
                    if tensor is None or self._queue_mediated(kernel, name):
                        skipped += 1
                        continue
                    # Loop-carried access extents require an expanded graph;
                    # keep the existing loop-aware RAW check for this adapter.
                    if op.loop_id is not None:
                        skipped += 1
                        continue
                    offset, size = tensor.offset_value, tensor.size_value
                    upper = offset+size if offset is not None and size is not None else None
                    resource = tensor.domain.value
                    if tensor.domain is PhysicalDomain.GM:
                        # The trace does not retain GM pointer provenance:
                        # unrelated symbols cannot be diagnosed as definite alias.
                        root, visited = tensor, set()
                        while root.view_source and root.name not in visited:
                            visited.add(root.name)
                            parent = kernel.tensors.get(root.view_source)
                            if parent is None:
                                break
                            root = parent
                        resource += ":"+root.name
                    if tensor.domain is PhysicalDomain.UNKNOWN:
                        resource += ":"+name
                    if tensor.domain is not PhysicalDomain.GM:
                        resource += ":"+op.core_view.value
                    accesses.append(Access(op.index, resource, mode, offset, upper))
        candidates = unordered_hazards(self._dependencies, accesses)
        found = []
        seen = set()
        for kind, a, b in candidates:
            first, second = operations[a.node], operations[b.node]
            if first.pipe is second.pipe and first.core_view.overlaps(second.core_view):
                continue
            if kind == "RAW" and set(first.writes) & set(second.reads) and self._same_view(first, second):
                continue  # reported by the loop-aware RAW pass above
            key = (kind, a.node, b.node, a.resource)
            if key in seen:
                continue
            seen.add(key)
            found.append({"kind": kind, "first": a.node, "second": b.node, "resource": a.resource})
            if len(found) <= _MAX_REPORTS:
                self.diags.add(Code.MEMORY_ORDERING, Severity.WARNING,
                               f"{kind} on {a.resource}: {first.name} and {second.name} lack happens-before",
                               second.loc, hardware_domain=second.pipe.value, hazard_kind=kind,
                               remediation="Order the shared buffer accesses with a supported synchronization mechanism.")
        self.ctx.publish(f"interval_hazard_coverage::{kernel.name}",
                         {"hazards": found, "skipped": skipped,
                          "scope": "straight-line extents; unrelated GM base provenance unchecked; local core views distinct"})

    # -- vacuous handshakes (AKA2010) ----------------------------------------

    def _check_vacuous_handshakes(self, kernel: KernelIR) -> None:
        """Flag SetFlag/WaitFlag pairs that order nothing (AKA2010).

        The written drain idiom is ``<produce on A>; SetFlag<A_B>(id);
        WaitFlag<A_B>(id); <consume on B>``: the set is enqueued on the
        producing pipeline's queue and fires when that pipeline drains
        through it, so the wait really does order B's continuation after A's
        work.  The idiom turns *pseudo* when A has no outstanding
        asynchronous work at the set - the flag fires the instant it issues,
        and the pair costs two instructions while ordering nothing (the fake
        ``WaitLoad()`` that sits before the loads it claims to guard) - or
        when the outstanding work lives on a pipe the route does not name.

        The pairing, adjacency and reachability all come from the sync graph
        the deadlock checker already published, so the two passes never
        disagree about which set belongs to which wait.
        """
        sync = self.ctx.artifacts.get(f"sync_graph::{kernel.name}")
        if not isinstance(sync, dict):
            return
        pairs: List[SyncPairProbe] = []
        by_index = {op.index: op for op in kernel.ops}
        for entry in sync.get("sync_pairs", []):
            setter = by_index.get(entry.get("set_index"))
            waiter = by_index.get(entry.get("wait_index"))
            if setter is None or waiter is None:
                continue
            if setter.conditional or waiter.conditional:
                continue
            if not setter.core_view.overlaps(waiter.core_view):
                continue
            route = setter.route
            if route is None:
                continue
            channel = entry.get("channel") or {}
            pairs.append(
                SyncPairProbe(
                    set_index=setter.index,
                    wait_index=waiter.index,
                    src_pipe=route.src.value,
                    dst_pipe=route.dst.value,
                    route=route.name,
                    event_id=channel.get("event_id", entry.get("event_id")),
                )
            )
        if not pairs:
            return
        pipe_ops: Dict[str, List[int]] = defaultdict(list)
        work: Dict[str, List[int]] = defaultdict(list)
        for op in kernel.ops:
            pipe_ops[op.pipe.value].append(op.index)
            if isinstance(op, ApiCallOp):
                work[op.pipe.value].append(op.index)
        loop_spans = [
            (loop.start_index, loop.end_index) for loop in kernel.loops.values()
        ]
        for verdict in vacuous_sync_pairs(
            self._dependencies, pairs, pipe_ops, work, loop_spans
        ):
            self._report_vacuous(kernel, verdict, by_index)

    def _report_vacuous(self, kernel: KernelIR, verdict, by_index) -> None:
        pair = verdict.pair
        setter = by_index[pair.set_index]
        waiter = by_index[pair.wait_index]
        if verdict.kind == "no_producer_work":
            detail = (
                f"SetFlag<{pair.route}>({self._id_text(pair.event_id)}) is raised on "
                f"{pair.src_pipe} with no outstanding asynchronous work behind "
                f"it, so the flag fires the moment it issues and the paired "
                f"WaitFlag on {pair.dst_pipe} releases immediately"
            )
            if verdict.producer_hint:
                detail += (
                    f"; the nearest preceding asynchronous work runs on "
                    f"{', '.join(verdict.producer_hint)}, not on "
                    f"{pair.src_pipe} - the route names the wrong producer"
                )
            fix = (
                f"Issue the producing instructions before the "
                f"SetFlag<HardEvent::{pair.route}>({self._id_text(pair.event_id)}); "
                f"the set must sit at the tail of {pair.src_pipe}'s work, and "
                "the wait after it orders the consumer against exactly that work"
            )
        else:
            detail = (
                f"SetFlag<{pair.route}>({self._id_text(pair.event_id)}) at line "
                f"{setter.loc.line} orders nothing the program does not "
                "already order: every asynchronous operation on "
                f"{pair.src_pipe} before it is reachable from the WaitFlag "
                "through other synchronisation"
            )
            fix = (
                "Delete the pair, or move the set to a point where it fences "
                "work that is genuinely outstanding"
            )
        self.diags.add(
            Code.MISSING_SYNC,
            Severity.WARNING,
            detail
            + ". The handshake is pseudo-synchronisation: the code reads as a "
            "drain, the hardware executes a no-op, and any cross-pipeline "
            "data movement it was meant to guard stays unordered",
            waiter.loc,
            hardware_domain=waiter.pipe.value,
            remediation=fix,
            related=[
                ("flag raised here", setter.loc),
                ("flag waited here", waiter.loc),
            ],
            route=pair.route,
            event_id=pair.event_id,
            vacuity=verdict.kind,
            producer_hint=list(verdict.producer_hint),
        )

    @staticmethod
    def _id_text(event_id) -> str:
        return f"EVENT_ID{event_id}" if event_id is not None else "its event id"

    # -- loop-invariant GM flush (AKA2012) ------------------------------------

    #: Ops that overwrite a global-memory destination in one transaction.
    _GM_STORES = frozenset({"DataCopy", "DataCopyPad", "Fixpipe"})
    _ATOMIC_ON = frozenset({"SetAtomicAdd", "SetAtomicMax", "SetAtomicMin"})
    _ATOMIC_OFF = frozenset({"SetAtomicNone"})

    def _check_loop_invariant_gm_flush(self, kernel: KernelIR) -> None:
        """Stores to a GM address that their own loop never advances (AKA2012).

        In a ``row0``/``col0`` reduction nest, a store whose destination
        mentions only the outer induction variable while it sits inside the
        inner loop rewrites the same GM bytes once per inner iteration.  The
        accumulated partial sums of the inner dimension are destroyed as
        they are produced (an early flush); without atomics only the last
        inner iteration's contribution survives.
        """
        stores: List[GmStoreProbe] = []
        atomic_active = False
        for op in kernel.api_calls():
            if op.name in self._ATOMIC_ON:
                atomic_active = True
            elif op.name in self._ATOMIC_OFF:
                atomic_active = False
            if op.name not in self._GM_STORES or not op.args:
                continue
            dst = op.args[0].tensor
            if not dst:
                continue
            decl = kernel.tensors.get(dst)
            if decl is None or decl.domain is not PhysicalDomain.GM:
                continue
            stores.append(
                GmStoreProbe(
                    index=op.index,
                    line=op.loc.line,
                    dst_text=op.args[0].text,
                    store_name=op.name,
                    loop_id=op.loop_id,
                    conditional=op.conditional,
                    atomic=atomic_active,
                )
            )
        if not stores:
            return
        accumulating = self._accumulating_loops(kernel)
        loops = {
            loop.id: LoopProbe(
                id=loop.id,
                var=loop.induction_var,
                parent=loop.parent,
                trip_count=loop.trip_count,
                line=loop.loc.line,
                accumulates=loop.id in accumulating,
            )
            for loop in kernel.loops.values()
        }
        for verdict in invariant_flush_loops(stores, loops):
            self._report_flush(kernel, verdict)

    def _accumulating_loops(self, kernel: KernelIR) -> Set[int]:
        """Loops whose body accumulates into one buffer (``Add(x, x, ...)``)."""
        accumulating: Set[int] = set()
        for op in kernel.api_calls():
            if op.name != "Add" or len(op.args) < 2:
                continue
            if op.args[0].tensor and op.args[0].tensor == op.args[1].tensor:
                if op.loop_id is not None:
                    accumulating.add(op.loop_id)
        return accumulating

    def _report_flush(self, kernel: KernelIR, verdict) -> None:
        store = verdict.store
        loop = verdict.loop
        severity = Severity.FATAL if verdict.reduction else Severity.WARNING
        where = (
            f"loop at line {loop.line} ('{loop.var}')"
            if loop.var
            else f"loop at line {loop.line}"
        )
        if verdict.reduction:
            consequence = (
                "the loop accumulates a reduction into this buffer, so every "
                "inner iteration destroys the partial sums of the previous "
                "one - an early flush that leaves only the last chunk's "
                "contribution in global memory"
            )
            fix = (
                f"Move the store out of the {loop.var} loop, past its closing "
                "brace, so the full inner reduction completes first; or index "
                f"the destination with {loop.var} when a per-iteration "
                "destination is intended"
            )
        else:
            consequence = (
                "each iteration overwrites the previous one's bytes without "
                "accumulating, so only the final iteration's data survives"
            )
            fix = (
                f"Advance the destination with {loop.var} (e.g. out[{loop.var}]) "
                "or hoist the store out of the loop"
            )
        self.diags.add(
            Code.MEMORY_ORDERING,
            severity,
            f"{store.store_name} at line {store.line} writes GM destination "
            f"'{store.dst_text}' from inside the {where}, but the address does "
            f"not depend on {loop.var}"
            + (f" (it depends on {', '.join(verdict.mentioned)} only)"
               if verdict.mentioned else "")
            + f"; {consequence}",
            SourceLoc(file=self.ctx.unit.path, line=store.line),
            hardware_domain="GM",
            remediation=fix,
            hazard_kind="WAW_LOOP_INVARIANT",
            store=store.store_name,
            store_line=store.line,
            loop_line=loop.line,
            induction_var=loop.var,
            reduction=verdict.reduction,
        )

    # -- evidence -----------------------------------------------------------

    def _synchronized(
        self,
        write: ApiCallOp,
        read: ApiCallOp,
        sets: Dict[ChannelKey, List[int]],
        waits: Dict[ChannelKey, List[int]],
        barriers: Sequence[int],
    ) -> bool:
        """``True`` when a flag pair or global barrier orders write -> read.

        The evidence is kept *weak on purpose*: any set on an ``A -> B``
        channel after the write, together with any wait on that channel
        before the read, counts.  Tighter pairing belongs to the deadlock
        checker, which models the semaphores exactly; here the question is
        only whether the program attempted the handshake at all.
        """
        if write.index < read.index:
            return nx.has_path(self._dependencies, write.index, read.index)
        for event_id in set(sets) & set(waits):
            src, dst, _ = event_id
            if src is not write.pipe or dst is not read.pipe:
                continue
            if sets[event_id][-1] > write.index and waits[event_id][0] < read.index:
                return True
        # A global barrier between the two ops drains every pipeline, which
        # orders them whatever the trace direction (the backward case is a
        # loop body, where the barrier also closes the previous iteration).
        low, high = sorted((write.index, read.index))
        for index in barriers:
            if low < index < high:
                return True
        return False

    @staticmethod
    def _same_view(write: ApiCallOp, read: ApiCallOp) -> bool:
        """``False`` for pairs split across the AIC/AIV views of a mix kernel."""
        return write.core_view.overlaps(read.core_view)

    def _queue_mediated(self, kernel: KernelIR, tensor: str) -> bool:
        """``True`` when the tensor's declaration chain roots in a ``TQue``.

        ``que.AllocTensor<T>()`` / ``que.DeQue<T>()`` storage is handed
        between pipelines by the CANN runtime's own event flags.  Those are
        invisible here, so the pair is out of static reach, not unsynchronised.
        """
        decl: Optional[TensorDecl] = kernel.tensors.get(tensor)
        visited = set()
        while decl is not None and decl.name not in visited:
            visited.add(decl.name)
            if decl.origin.startswith("TQue::"):
                return True
            decl = kernel.tensors.get(decl.view_source or "")
        return False

    def _report(
        self, tensor: str, write: ApiCallOp, read: ApiCallOp
    ) -> None:
        self.diags.add(
            Code.MISSING_SYNC,
            Severity.WARNING,
            f"'{tensor}' is written by {write.name} on {write.pipe.value} "
            f"and read by {read.name} on {read.pipe.value} with no "
            "SetFlag/WaitFlag pair or global barrier ordering them; the "
            "pipelines run concurrently, so the read races the write",
            read.loc,
            hardware_domain=read.pipe.value,
            remediation=(
                "Raise SetFlag<HardEvent::"
                f"{write.pipe.short}_{read.pipe.short}>(EVENT_IDn) after the "
                f"{write.name} and WaitFlag on the same channel before the "
                f"{read.name}, or issue PipeBarrier(PIPE_ALL) between them."
            ),
            tensor=tensor,
            write_op=write.name,
            read_op=read.name,
            write_pipe=write.pipe.value,
            read_pipe=read.pipe.value,
            write_line=write.loc.line,
            read_line=read.loc.line,
        )
