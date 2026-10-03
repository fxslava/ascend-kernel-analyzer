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

from ..diagnostics import Code, Severity
from ..hardware import REAL_PIPES, Pipe
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

    # -- evidence -----------------------------------------------------------

    @staticmethod
    def _synchronized(
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
        if write.core_view is CoreView.BOTH or read.core_view is CoreView.BOTH:
            return True
        return write.core_view is read.core_view

    def _queue_mediated(self, kernel: KernelIR, tensor: str) -> bool:
        """``True`` when the tensor's declaration chain roots in a ``TQue``.

        ``que.AllocTensor<T>()`` / ``que.DeQue<T>()`` storage is handed
        between pipelines by the CANN runtime's own event flags.  Those are
        invisible here, so the pair is out of static reach, not unsynchronised.
        """
        decl: Optional[TensorDecl] = kernel.tensors.get(tensor)
        hop = 0
        while decl is not None and hop < 16:  # a view chain cannot be deeper
            if decl.origin.startswith("TQue::"):
                return True
            decl = kernel.tensors.get(decl.view_source or "")
            hop += 1
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
