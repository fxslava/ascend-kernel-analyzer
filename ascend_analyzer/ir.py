"""The analyzer's intermediate representation.

:class:`KernelIR` is the single hand-off point between the parser and the
checkers.  It is a flat, ordered *operation trace* over a kernel body plus a
tensor symbol table, with enough structure retained (lexical scopes, loop
nesting) to reason about liveness and loop-carried synchronisation without
keeping the C++ syntax tree alive.

Ordering convention: every operation carries a monotonically increasing
``index`` that reflects source program order within the kernel.  Projecting
the trace onto a single :class:`~ascend_analyzer.hardware.Pipe` yields that
pipeline's in-order instruction stream, which is exactly the abstraction the
deadlock checker needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .diagnostics import SourceLoc
from .hardware import HardEventRoute, PhysicalDomain, Pipe, TPosition
from .symbolic import Expr, render, to_int

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
]


class FlagKind(Enum):
    """Whether a flag operation raises or consumes an event."""

    SET = "SetFlag"
    WAIT = "WaitFlag"


class ScopeKind(Enum):
    """What kind of lexical region a :class:`Scope` represents."""

    KERNEL = "kernel"
    BLOCK = "block"
    LOOP = "loop"
    BRANCH = "branch"
    #: The body of a callee walked into its caller's trace.
    INLINE = "inline"
    #: A region guarded to one physical core (AIC or AIV).
    CORE = "core"


class CoreView(Enum):
    """Which physical core of a dual-core DaVinci part runs a region.

    A "mix" kernel is one source file compiled twice: once for the Cube core
    (AIC) and once for the Vector core (AIV), selected by
    ``__DAV_C220_CUBE__`` / ``__DAV_C220_VEC__`` at compile time and by
    ``ASCEND_IS_AIC`` / ``ASCEND_IS_AIV`` at run time.  The two cores hold
    **separate event-id spaces**, so a ``SetFlag`` compiled into one of them
    can never be consumed by a ``WaitFlag`` compiled into the other.  Treating
    the merged source as one event space reports every such half as an orphan.
    """

    #: Unguarded: compiled into both binaries.
    BOTH = "both"
    #: Cube core only.
    AIC = "aic"
    #: Vector core only.
    AIV = "aiv"
    #: Guarded into both cores at once, so it is compiled into neither.
    NONE = "none"

    def intersect(self, other: "CoreView") -> "CoreView":
        """The cores on which both views are resident."""
        if self is CoreView.BOTH:
            return other
        if other is CoreView.BOTH:
            return self
        return self if self is other else CoreView.NONE

    def overlaps(self, other: "CoreView") -> bool:
        """``True`` when some core runs both views, so they can interact."""
        return self.intersect(other) is not CoreView.NONE

    @property
    def complement(self) -> "CoreView":
        """The opposite arm of a core guard."""
        return {
            CoreView.AIC: CoreView.AIV,
            CoreView.AIV: CoreView.AIC,
            CoreView.BOTH: CoreView.NONE,
            CoreView.NONE: CoreView.BOTH,
        }[self]


@dataclass
class Scope:
    """A lexical region, used to bound tensor liveness."""

    id: int
    kind: ScopeKind
    parent: Optional[int]
    loc: SourceLoc
    #: Trace index of the first and last operation inside the scope.
    start_index: int = 0
    end_index: int = 0

    def ancestry(self, scopes: Dict[int, "Scope"]) -> List[int]:
        """This scope's id followed by every enclosing scope id."""
        chain, cur = [], self
        while True:
            chain.append(cur.id)
            if cur.parent is None:
                return chain
            parent = scopes.get(cur.parent)
            if parent is None:
                return chain
            cur = parent


@dataclass
class LoopInfo:
    """A loop nest level discovered in the kernel body."""

    id: int
    scope_id: int
    loc: SourceLoc
    #: Enclosing loop id, or ``None`` for an outermost loop.
    parent: Optional[int] = None
    #: Induction variable name, when recognisable.
    induction_var: Optional[str] = None
    #: Statically known trip count, when derivable from the loop header.
    trip_count: Optional[int] = None
    #: First induction-variable value, when the header folded to a constant.
    start: Optional[int] = None
    #: Induction-variable stride, when statically known.
    step: Optional[int] = None
    #: ``True`` when the visitor replayed the body once per iteration with the
    #: induction variable bound to a concrete constant.  Operations from such
    #: a loop carry ``loop_id=None``: the trace *is* the full execution, so
    #: straight-line reasoning is exact and no back edges are needed.
    unrolled: bool = False
    #: ``True`` when the visitor executed this loop as three peeled phases:
    #: a straight-line head, a cyclic steady-state representative cycle and a
    #: straight-line tail (see ``ast_visitor``).  Head/tail operations carry
    #: ``loop_id=None``; the steady-state representatives carry this loop's
    #: id, so the marked graph models them as a cycle with one-token back
    #: edges while the bulk iterations they stand for stay abstract.
    peeled: bool = False
    #: Induction-variable value of the first steady-state representative.
    steady_first: Optional[int] = None
    #: How many steady-state representative iterations were emitted (the
    #: loop's modular period, e.g. 2 for ``p = t & 1`` ping-pong parity).
    steady_reps: int = 0
    #: How many peeled head / tail iterations were emitted.
    peeled_head: int = 0
    peeled_tail: int = 0
    #: Trace indices spanned by the loop body (inclusive).
    start_index: int = 0
    end_index: int = 0
    #: Source text of the loop header, for diagnostics.
    header: str = ""

    @property
    def is_bounded(self) -> bool:
        return self.trip_count is not None

    def contains(self, index: int) -> bool:
        return self.start_index <= index <= self.end_index


@dataclass
class TensorDecl:
    """A tensor (or raw buffer pointer) bound to a byte range in one domain.

    ``byte_offset`` and ``byte_size`` are symbolic expressions; the checkers
    fold them where possible and fall back to the solver when they cannot.
    """

    name: str
    loc: SourceLoc
    position: Optional[TPosition]
    domain: PhysicalDomain
    dtype: Optional[str] = None
    elem_size: Optional[int] = None
    byte_offset: Optional[Expr] = None
    elem_count: Optional[Expr] = None
    byte_size: Optional[Expr] = None
    scope_id: int = 0
    #: How the parser learned about this tensor (for report provenance).
    origin: str = "declaration"
    #: Tensors sharing a non-empty reuse group may legally overlap.
    reuse_group: Optional[str] = None
    #: Name of the ``TBuf`` this tensor was obtained from via ``.Get()``, so
    #: aggregate budgeting can count the buffer once instead of once per view.
    source_buffer: Optional[str] = None
    #: Trace indices of first and last observed use; ``None`` when never used.
    first_use: Optional[int] = None
    last_use: Optional[int] = None
    #: Locations at which the tensor is read or written, for related-info.
    use_locs: List[SourceLoc] = field(default_factory=list)
    #: ``True`` when the declaration never received an address binding.
    unbound: bool = False

    # -- derived views ------------------------------------------------------

    @property
    def offset_value(self) -> Optional[int]:
        return to_int(self.byte_offset)

    @property
    def size_value(self) -> Optional[int]:
        return to_int(self.byte_size)

    @property
    def end_value(self) -> Optional[int]:
        off, size = self.offset_value, self.size_value
        return None if off is None or size is None else off + size

    @property
    def is_fully_static(self) -> bool:
        return self.offset_value is not None and self.size_value is not None

    @property
    def is_sram(self) -> bool:
        return self.domain.is_on_core_sram

    def live_range(self, scopes: Dict[int, Scope]) -> Tuple[int, int]:
        """Trace-index interval over which this tensor must hold its data.

        Uses observed first/last use when available; otherwise falls back to
        the extent of the declaring lexical scope, which is the conservative
        assumption for a declared-but-unused buffer.
        """
        if self.first_use is not None and self.last_use is not None:
            return (self.first_use, self.last_use)
        scope = scopes.get(self.scope_id)
        if scope is not None:
            return (scope.start_index, scope.end_index)
        return (0, 0)

    def describe_range(self) -> str:
        off, size = self.offset_value, self.size_value
        if off is None or size is None:
            return f"[{render(self.byte_offset)} .. +{render(self.byte_size)})"
        return f"[0x{off:X} .. 0x{off + size:X})  ({size} B)"

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "domain": self.domain.value,
            "position": self.position.value if self.position else None,
            "dtype": self.dtype,
            "elem_size": self.elem_size,
            "byte_offset": self.offset_value,
            "byte_offset_expr": render(self.byte_offset),
            "byte_size": self.size_value,
            "byte_size_expr": render(self.byte_size),
            "elem_count": to_int(self.elem_count),
            "origin": self.origin,
            "reuse_group": self.reuse_group,
            "first_use": self.first_use,
            "last_use": self.last_use,
            "location": self.loc.to_json(),
        }


@dataclass(frozen=True)
class ArgRef:
    """One positional argument of a recognised API call."""

    index: int
    text: str
    #: Name of the tensor this argument resolves to, when it is one.
    tensor: Optional[str] = None
    #: Folded integer value, when the argument is a constant.
    value: Optional[int] = None
    expr: Optional[Expr] = None


@dataclass
class Operation:
    """Base class for everything in the kernel trace."""

    index: int
    loc: SourceLoc
    pipe: Pipe
    scope_id: int
    #: Id of the innermost enclosing loop, or ``None`` at kernel level.
    loop_id: Optional[int] = None
    #: ``True`` when the operation sits inside an ``if``/``else`` arm, so the
    #: trace over-approximates it as unconditionally executed.  Pairing
    #: diagnostics are softened for such operations.
    conditional: bool = False
    #: Which physical core this operation is compiled into.  A mix kernel is
    #: one source file but two binaries, and the AIC and AIV cores have
    #: separate event spaces, so a flag raised in one view can never be seen
    #: in the other.
    core_view: CoreView = CoreView.BOTH

    @property
    def kind(self) -> str:  # pragma: no cover - overridden
        return "op"

    @property
    def label(self) -> str:  # pragma: no cover - overridden
        return self.kind

    def to_json(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "kind": self.kind,
            "label": self.label,
            "pipe": self.pipe.value,
            "loop_id": self.loop_id,
            "conditional": self.conditional,
            "core_view": self.core_view.value,
            "location": self.loc.to_json(),
        }


@dataclass
class FlagOp(Operation):
    """A ``SetFlag`` or ``WaitFlag`` on one ``HardEvent`` route."""

    flag_kind: FlagKind = FlagKind.SET
    route: Optional[HardEventRoute] = None
    event_id: Optional[int] = None
    #: Source text of the event-id argument (useful when it is symbolic).
    event_id_text: str = ""
    #: ``True`` for the ISASI ``set_flag(PIPE_A, PIPE_B, id)`` spelling.
    isasi_form: bool = False

    @property
    def kind(self) -> str:
        return self.flag_kind.value

    @property
    def label(self) -> str:
        route = self.route.name if self.route else "?"
        eid = self.event_id_text or (
            f"EVENT_ID{self.event_id}" if self.event_id is not None else "?"
        )
        return f"{self.flag_kind.value}<{route}>({eid})"

    @property
    def channel(self) -> Tuple[str, object]:
        """The ``(route, event_id)`` pair this operation synchronises on."""
        return (self.route.name if self.route else "?",
                self.event_id if self.event_id is not None else self.event_id_text)

    def to_json(self) -> Dict[str, object]:
        base = super().to_json()
        base.update(
            {
                "flag_kind": self.flag_kind.value,
                "route": self.route.name if self.route else None,
                "src_pipe": self.route.src.value if self.route else None,
                "dst_pipe": self.route.dst.value if self.route else None,
                "event_id": self.event_id,
                "event_id_text": self.event_id_text,
                "isasi_form": self.isasi_form,
            }
        )
        return base


@dataclass
class BarrierOp(Operation):
    """A ``PipeBarrier`` / ``pipe_barrier`` fence."""

    target: Pipe = Pipe.ALL

    @property
    def kind(self) -> str:
        return "PipeBarrier"

    @property
    def label(self) -> str:
        return f"PipeBarrier({self.target.value})"

    @property
    def is_global(self) -> bool:
        return self.target is Pipe.ALL

    def to_json(self) -> Dict[str, object]:
        base = super().to_json()
        base["target"] = self.target.value
        base["global"] = self.is_global
        return base


@dataclass
class ApiCallOp(Operation):
    """A recognised DMA or compute intrinsic such as ``DataCopy`` or ``Add``."""

    name: str = ""
    args: Tuple[ArgRef, ...] = ()
    #: Tensor names this call writes / reads, after argument resolution.
    writes: Tuple[str, ...] = ()
    reads: Tuple[str, ...] = ()
    #: Raw source text of the whole call.
    text: str = ""

    @property
    def kind(self) -> str:
        return "ApiCall"

    @property
    def label(self) -> str:
        return f"{self.name}({', '.join(a.text for a in self.args)})"

    def tensor_args(self) -> Iterator[ArgRef]:
        return (a for a in self.args if a.tensor)

    def to_json(self) -> Dict[str, object]:
        base = super().to_json()
        base.update(
            {
                "name": self.name,
                "args": [
                    {"index": a.index, "text": a.text, "tensor": a.tensor, "value": a.value}
                    for a in self.args
                ],
                "writes": list(self.writes),
                "reads": list(self.reads),
            }
        )
        return base


@dataclass
class KernelIR:
    """Everything the checkers need to know about one kernel function."""

    name: str
    loc: SourceLoc
    #: Ordered operation trace in source program order.
    ops: List[Operation] = field(default_factory=list)
    #: Tensor symbol table, keyed by source identifier.
    tensors: Dict[str, TensorDecl] = field(default_factory=dict)
    scopes: Dict[int, Scope] = field(default_factory=dict)
    loops: Dict[int, LoopInfo] = field(default_factory=dict)
    #: Folded ``constexpr`` environment, for report context.
    constants: Dict[str, int] = field(default_factory=dict)
    #: ``TPipe::InitBuffer`` / ``LocalMemAllocator`` byte sizes, folded to
    #: integers, by buffer name.  The memory checker sums these with tensor
    #: declarations when budgeting architectures that partition the Unified
    #: Buffer (351x SIMD/SIMT DataCache).
    buffer_sizes: Dict[str, int] = field(default_factory=dict)
    #: ``True`` when the function carried ``__global__``/``__aicore__``.
    is_kernel_entry: bool = True
    #: Callees whose bodies were walked into this trace, in call order.
    inlined: List[str] = field(default_factory=list)

    # -- trace views --------------------------------------------------------

    def flag_ops(self) -> List[FlagOp]:
        return [op for op in self.ops if isinstance(op, FlagOp)]

    def barriers(self) -> List[BarrierOp]:
        return [op for op in self.ops if isinstance(op, BarrierOp)]

    def api_calls(self) -> List[ApiCallOp]:
        return [op for op in self.ops if isinstance(op, ApiCallOp)]

    def ops_on(self, pipe: Pipe) -> List[Operation]:
        """The in-order instruction stream of a single pipeline."""
        return [op for op in self.ops if op.pipe is pipe]

    def active_pipes(self) -> List[Pipe]:
        seen: List[Pipe] = []
        for op in self.ops:
            if op.pipe.is_real and op.pipe not in seen:
                seen.append(op.pipe)
        return seen

    def tensors_in(self, domain: PhysicalDomain) -> List[TensorDecl]:
        return [t for t in self.tensors.values() if t.domain is domain]

    def innermost_loop_of(self, index: int) -> Optional[LoopInfo]:
        """The deepest loop whose body contains trace position ``index``."""
        best: Optional[LoopInfo] = None
        for loop in self.loops.values():
            if loop.contains(index):
                if best is None or loop.start_index > best.start_index:
                    best = loop
        return best

    def loop_depth(self, loop_id: Optional[int]) -> int:
        depth, cur = 0, loop_id
        while cur is not None:
            loop = self.loops.get(cur)
            if loop is None:
                break
            depth += 1
            cur = loop.parent
        return depth

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "location": self.loc.to_json(),
            "is_kernel_entry": self.is_kernel_entry,
            "op_count": len(self.ops),
            # Which callees this trace covers: a finding may point into one of
            # them rather than into the entry's own body.
            "inlined": list(self.inlined),
            "tensors": [t.to_json() for t in self.tensors.values()],
            "loops": [
                {
                    "id": loop.id,
                    "induction_var": loop.induction_var,
                    "trip_count": loop.trip_count,
                    "peeled": loop.peeled,
                    "steady_reps": loop.steady_reps,
                    "header": loop.header,
                    "location": loop.loc.to_json(),
                }
                for loop in self.loops.values()
            ],
            "constants": self.constants,
            "buffer_sizes": self.buffer_sizes,
        }


@dataclass
class AnalysisUnit:
    """One parsed translation unit: its source plus every kernel found in it."""

    path: str
    source: str
    kernels: List[KernelIR] = field(default_factory=list)
    #: Global ``constexpr`` constants visible to all kernels in the file.
    constants: Dict[str, int] = field(default_factory=dict)
    #: ``True`` when tree-sitter reported syntax errors.
    had_parse_errors: bool = False
    parse_error_locs: List[SourceLoc] = field(default_factory=list)
    #: Diagnostic codes silenced per line by ``@ascend-ignore`` annotations.
    suppressions: Dict[int, List[str]] = field(default_factory=dict)
    #: Tiling fields bound by role inference rather than by a manifest, as
    #: ``{access: value}``.  Non-empty means some layout in this unit rests on
    #: an inference; the report says so, and so does the JSON.
    inferred_tiling_bindings: Dict[str, int] = field(default_factory=dict)

    def is_suppressed(self, code: str, line: int) -> bool:
        """``True`` when an ``@ascend-ignore`` covers ``code`` at ``line``.

        An annotation applies to its own line and the line below it, so it can
        sit either on or above the construct it excuses. A bare
        ``@ascend-ignore`` with no codes silences everything on that line.
        """
        for candidate in (line, line - 1):
            codes = self.suppressions.get(candidate)
            if codes is None:
                continue
            if not codes or code.upper() in codes:
                return True
        return False

    @property
    def lines(self) -> Sequence[str]:
        return self.source.splitlines()

    def line_text(self, line: int) -> str:
        """1-based line lookup, returning ``""`` when out of range."""
        lines = self.lines
        return lines[line - 1] if 1 <= line <= len(lines) else ""
