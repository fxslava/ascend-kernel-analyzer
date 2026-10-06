"""Numerical-parity decision procedures shared by the checker adapters.

Three hard-to-debug failure families are decided here over plain data, so the
backend tests can drive them from synthetic structures without any C++ or
MLIR frontend:

* **vacuous handshakes** (AKA2010) - a ``SetFlag``/``WaitFlag`` pair adjacent
  in the waiting pipeline that orders nothing new, because the producing
  pipeline has no outstanding asynchronous work the rest of the program does
  not already order.  The classic shape is a ``WaitLoad()`` helper whose
  flag pair sits *before* the loads it claims to guard: the wait releases
  immediately, the reads race the copies, and the kernel still runs - it just
  computes on stale bytes.
* **split-window bleed** (AKA1015) - a de-interleave whose window has fallen
  below the pairing floor of a flattened ``[rows, stride]`` layout, so the
  split pairs (and the merge tree above it sums) elements from independent
  rows of the batch.  A balanced ``[8, 32]`` merge tree may descend to a
  window of ``2 * rows``; below that, row ``2i`` is paired with row ``2i+1``.
* **loop-invariant GM flush** (AKA2012) - a store to global memory inside a
  loop whose destination does not depend on the loop's induction variable and
  nothing accumulates atomically, so every iteration destroys the previous
  one's partial result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import re

__all__ = [
    "SyncPairProbe",
    "VacuousVerdict",
    "vacuous_sync_pairs",
    "SplitWindow",
    "MergeTreeTracker",
    "replay_halving_chain",
    "GmStoreProbe",
    "FlushVerdict",
    "invariant_flush_loops",
]


# ---------------------------------------------------------------------------
# Vacuous handshakes (AKA2010)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncPairProbe:
    """One FIFO-matched ``SetFlag`` -> ``WaitFlag`` pair, as pure data."""

    set_index: int
    wait_index: int
    #: Value of ``HardEventRoute.src`` (the flag's producing pipeline).
    src_pipe: str
    #: Value of ``HardEventRoute.dst`` (the waiting pipeline).
    dst_pipe: str
    route: str = "?"
    event_id: object = None


@dataclass(frozen=True)
class VacuousVerdict:
    """Why a handshake orders nothing, for the diagnostic to explain."""

    pair: SyncPairProbe
    #: ``"no_producer_work"`` - the producing pipeline never issued async work
    #: before the set, so the flag fires the moment it executes;
    #: ``"nothing_new"`` - work exists but is already ordered elsewhere.
    kind: str
    #: Pipelines (other than ``src_pipe``) with the nearest preceding async
    #: work: evidence the route names the wrong producer.
    producer_hint: Tuple[str, ...] = ()
    #: How many async ops on the producing pipe precede the set.
    outstanding_before: int = 0


def vacuous_sync_pairs(
    graph: nx.DiGraph,
    pairs: Sequence[SyncPairProbe],
    pipe_ops: Dict[str, Sequence[int]],
    work: Dict[str, Sequence[int]],
    loop_spans: Sequence[Tuple[int, int]] = (),
) -> List[VacuousVerdict]:
    """Handshake pairs that provide no ordering the program lacks.

    ``graph`` is the happens-before graph over trace indices (program order
    per pipeline plus matched flag edges); ``pipe_ops``/``work`` map pipeline
    names to sorted trace indices of, respectively, every operation and the
    asynchronous (data-moving/computing) operations on that pipeline;
    ``loop_spans`` are the ``(start, end)`` trace extents of loop bodies.

    A pair is examined only when it is *basic-block local*: the wait is the
    next operation on its pipeline after the set, so the two form one written
    handshake rather than a loop-carried or priming arrangement.  Pairs that
    close a ring - a set before a loop whose body opens with the wait (a
    prologue prime), or a set inside a loop whose drain waits after it - are
    excluded by the loop extents, unrolled or not.  For the remaining pairs
    the set orders ``[everything on the producing pipeline before the set]``
    before ``[everything on the waiting pipeline after the wait]``; the pair
    is vacuous when that ordering adds nothing - either because there is no
    such work at all, or because it is already reachable through the rest of
    the graph once this pair's own edge is removed.
    """
    verdicts: List[VacuousVerdict] = []
    for pair in pairs:
        if pair.wait_index <= pair.set_index:
            continue  # loop-carried; the priming checks own that shape
        if _closes_ring(pair, loop_spans):
            continue
        dst_stream = pipe_ops.get(pair.dst_pipe, ())
        if any(pair.set_index < i < pair.wait_index for i in dst_stream):
            continue  # another op on the waiting pipe sits inside the pair
        src_work = [i for i in work.get(pair.src_pipe, ()) if i < pair.set_index]
        if src_work and _all_ordered(graph, src_work, pair):
            verdicts.append(
                VacuousVerdict(
                    pair=pair,
                    kind="nothing_new",
                    outstanding_before=len(src_work),
                    producer_hint=_producer_hint(
                        pair, work, exclude=pair.src_pipe
                    ),
                )
            )
            continue
        if not src_work:
            verdicts.append(
                VacuousVerdict(
                    pair=pair,
                    kind="no_producer_work",
                    producer_hint=_producer_hint(
                        pair, work, exclude=pair.src_pipe
                    ),
                )
            )
    return verdicts


def _closes_ring(pair: SyncPairProbe, spans: Sequence[Tuple[int, int]]) -> bool:
    """``True`` for prologue-prime / epilogue-drain pairs of a loop ring."""
    for start, end in spans:
        if pair.set_index < start <= pair.wait_index <= end:
            return True  # primes the loop's first iteration
        if start <= pair.set_index <= end < pair.wait_index:
            return True  # drains the loop's last iteration
    return False


def _all_ordered(
    graph: nx.DiGraph, sources: Iterable[int], pair: SyncPairProbe
) -> bool:
    """``True`` when every source reaches the wait without this pair's edge."""
    edge_data = None
    if graph.has_edge(pair.set_index, pair.wait_index):
        edge_data = graph[pair.set_index][pair.wait_index]
        graph.remove_edge(pair.set_index, pair.wait_index)
    try:
        return all(nx.has_path(graph, s, pair.wait_index) for s in sources)
    finally:
        if edge_data is not None:
            graph.add_edge(pair.set_index, pair.wait_index, **edge_data)


def _producer_hint(
    pair: SyncPairProbe, work: Dict[str, Sequence[int]], *, exclude: str
) -> Tuple[str, ...]:
    """Pipelines with async work between the last barrier-free point and the
    set - the routes the flag *should* have named when it named none."""
    nearest = -1
    for pipe, indices in work.items():
        if pipe == exclude:
            continue
        for i in indices:
            if i < pair.set_index:
                nearest = max(nearest, i)
    if nearest < 0:
        return ()
    return tuple(
        sorted(
            {
                pipe
                for pipe, indices in work.items()
                if pipe != exclude and any(i == nearest for i in indices)
            }
        )
    )


# ---------------------------------------------------------------------------
# Split-window bleed (AKA1015)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitWindow:
    """One de-interleave event over a flattened multi-row buffer."""

    #: Source buffer the split reads (trace name).
    tensor: str
    #: Elements moved to *each* destination half; the full window is twice
    #: this, because a unit-strand de-interleave takes the even and the odd
    #: elements of the window.
    half_count: int
    #: Trace index of the split's first half (for ordering and related info).
    index: int
    line: int
    #: ``"deinterleave"`` (direct intrinsic) or ``"gather_pair"`` (the
    #: two-Gather idiom parts without the instruction use instead).
    kind: str


@dataclass(frozen=True)
class SplitViolation:
    window: SplitWindow
    rows: int
    #: Elements the split window actually covers.
    length: int
    #: ``"cross_row"`` - window below ``2 * rows``, pairs span two rows;
    #: ``"ragged"`` - window not a whole multiple of the row count.
    kind: str


class MergeTreeTracker:
    """Tracks the live logical extent of buffers reduced by split trees.

    A balanced merge halves a buffer's live extent in place: splitting
    ``[rows, stride]`` and adding the halves back yields ``[rows, stride/2]``
    occupying ``rows * stride / 2`` leading elements.  Feeding the tracker
    the halving steps lets a later split on the same buffer be judged against
    the *current* extent rather than the allocation.
    """

    def __init__(self, rows: Optional[int]) -> None:
        self.rows = rows
        self.extent: Dict[str, int] = {}

    def apply_halving(self, tensor: str, halved_extent: int) -> None:
        self.extent[tensor] = halved_extent

    def check_window(self, window: SplitWindow) -> Optional[SplitViolation]:
        rows = self.rows
        if rows is None or rows < 2:
            return None  # no trustworthy row anchor; silence over noise
        length = 2 * window.half_count
        if length % rows != 0:
            return SplitViolation(
                window, rows, length,
                "ragged" if length >= 2 * rows else "cross_row",
            )
        if length // rows < 2:
            return SplitViolation(window, rows, length, "cross_row")
        self.extent.setdefault(window.tensor, length)
        return None


def replay_halving_chain(
    initial: int, bound: int, *, divisor: int = 2, strict: bool = True
) -> Tuple[List[int], int]:
    """Levels a ``while (len > bound) ... len /= divisor`` loop runs through.

    Returns ``(body_levels, exit_value)``: the window lengths each loop-body
    execution sees, and the length the code after the loop observes.  With
    ``strict=False`` the comparison is ``>=`` instead of ``>``.
    """
    if initial <= 0 or divisor < 2:
        return [], initial
    body: List[int] = []
    value = initial
    while value > bound if strict else value >= bound:
        body.append(value)
        next_value = value // divisor
        if next_value == value:
            break
        value = next_value
    return body, value


# ---------------------------------------------------------------------------
# Loop-invariant GM flush (AKA2012)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GmStoreProbe:
    """One store to global memory, as pure data."""

    index: int
    line: int
    #: Destination argument text; scanned for induction-variable mentions.
    dst_text: str
    store_name: str
    #: Innermost loop id containing the store (``None`` at kernel level).
    loop_id: Optional[int]
    conditional: bool = False
    #: Whether an ``SetAtomicAdd``/``Max``/``Min`` was active at the store.
    atomic: bool = False


@dataclass(frozen=True)
class LoopProbe:
    """One loop nest level, as pure data."""

    id: object
    var: Optional[str]
    parent: Optional[object] = None
    trip_count: Optional[int] = None
    line: int = 0
    #: The loop's body accumulates into one buffer (``Add(x, x, ...)``),
    #: making a repeated store an early flush of a partial reduction.
    accumulates: bool = False


@dataclass(frozen=True)
class FlushVerdict:
    store: GmStoreProbe
    loop: LoopProbe
    #: Induction variables of the full nest that the destination mentions.
    mentioned: Tuple[str, ...]
    #: ``True`` when the loop body accumulates a reduction the store truncates.
    reduction: bool


def invariant_flush_loops(
    stores: Sequence[GmStoreProbe], loops: Dict[object, LoopProbe]
) -> List[FlushVerdict]:
    """Stores whose GM destination ignores their loop's induction variable.

    A destination invariant across the iterations of the loop that contains
    it (while that loop runs more than once and nothing accumulates
    atomically) is overwritten every iteration; only the last iteration's
    data survives.  Inside a reduction nest that is the early-flush bug: the
    partial sums of the inner dimension are destroyed as they are produced.
    """
    verdicts: List[FlushVerdict] = []
    for store in stores:
        if store.conditional or store.atomic:
            continue
        loop = loops.get(store.loop_id) if store.loop_id is not None else None
        if loop is None or not loop.var:
            continue
        if loop.trip_count is not None and loop.trip_count < 2:
            continue
        if re.search(rf"\b{re.escape(loop.var)}\b", store.dst_text):
            continue  # the address advances with the loop it lives in
        mentioned: List[str] = []
        node: Optional[LoopProbe] = loop
        while node is not None:
            if node.var and re.search(rf"\b{re.escape(node.var)}\b", store.dst_text):
                mentioned.append(node.var)
            node = loops.get(node.parent) if node.parent is not None else None
        verdicts.append(
            FlushVerdict(
                store=store,
                loop=loop,
                mentioned=tuple(mentioned),
                reduction=loop.accumulates,
            )
        )
    return verdicts
