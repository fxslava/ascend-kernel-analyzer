"""The ``ascend`` MLIR-style dialect: canonical SSA IR for Ascend C kernels.

This module defines the in-memory dialect the BiSheng frontend lowers into
(:mod:`ascend_analyzer.parsing.mlir_bridge`) and the verification engine
lowers out of (:mod:`ascend_analyzer.analyzer_mlir`).  It is a faithful
structural model of an MLIR dialect rather than a binding to the C++ MLIR
libraries: types, operations, regions and SSA values are plain dataclasses,
and :meth:`AscendModule.dump` prints canonical MLIR assembly so kernels can
be inspected during debugging with ``module.dump()``.

Design notes
------------
* Every SSA value has an identity (``%name`` plus an optional defining op);
  ops reference operands by :class:`SsaValue` so the graph is traversable in
  both directions without string matching.
* Memory is modelled by :class:`AscendMemRefType` with an explicit physical
  ``space``, byte offset and byte size; offsets are integers whenever the
  layout is concrete (the whole point of the MLIR path: deterministic,
  solver-free layout audits).
* Heterogeneous kernels wrap per-core op sequences in
  :class:`CoreRegionOp` so two classes defining ``Init()``/``Process()``
  never collide: their operations belong strictly to their own region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "MemorySpace",
    "CoreType",
    "SsaValue",
    "QueueType",
    "AscendMemRefType",
    "AscendOp",
    "AllocBufferOp",
    "GetTensorOp",
    "AllocTensorOp",
    "EnQueOp",
    "DeQueOp",
    "FreeTensorOp",
    "MteCopyOp",
    "SetFlagOp",
    "WaitFlagOp",
    "BarrierOp",
    "MmadOp",
    "VectorOp",
    "CoreRegionOp",
    "KernelOp",
    "AscendModule",
]


class MemorySpace(Enum):
    """Physical memory spaces of the DaVinci memory hierarchy."""

    UB = "UB"
    L1 = "L1"
    L0A = "L0A"
    L0B = "L0B"
    L0C = "L0C"
    BT = "BT"
    FB = "FB"
    GM = "GM"

    def mangled(self) -> str:
        return f"#ascend.space<{self.value}>"


class CoreType(Enum):
    """Which physical core executes a region of a heterogeneous kernel."""

    AIC = "AIC"
    AIV = "AIV"


@dataclass
class SsaValue:
    """An SSA value: printable name plus optional defining information."""

    name: str
    #: Human-readable type of the value (dialect type string or C++ type).
    type_str: str = ""
    #: Tensor name this value carries, when it is a tensor handle.
    tensor: Optional[str] = None
    #: Source line the value was declared at (0 = unknown).
    line: int = 0

    def __str__(self) -> str:  # pragma: no cover - debug aid
        return f"%{self.name}"


@dataclass(frozen=True)
class QueueType:
    """``!ascend.queue<position, depth>`` - a TQue's static shape."""

    position: str
    depth: int

    def __str__(self) -> str:
        return f"!ascend.queue<{self.position}, {self.depth}>"


@dataclass(frozen=True)
class AscendMemRefType:
    """``!ascend.memref<shape, dtype, space, offset>`` - a tensor's layout."""

    dtype: str
    space: MemorySpace
    shape: Tuple[int, ...] = ()
    byte_offset: int = 0
    byte_size: int = 0

    def __str__(self) -> str:
        shape = "x".join(str(d) for d in self.shape) or "?"
        return (f"!ascend.memref<{shape}x{self.dtype}, "
                f"{self.space.mangled()}, offset={self.byte_offset}, size={self.byte_size}>")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@dataclass
class AscendOp:
    """Base class of dialect operations."""

    #: Result values produced by this op (may be empty for side-effecting ops).
    results: List[SsaValue] = field(default_factory=list)
    #: Source line this op came from, for diagnostics (1-based; 0 = unknown).
    line: int = 0
    #: Source text of the originating expression, for reports.
    text: str = ""
    #: Nesting depth of enclosing loops (0 = top level).
    loop_depth: int = 0
    #: Id of the innermost enclosing loop, when tracked.
    loop_id: Optional[int] = None
    #: ``True`` when the op sits inside an ``if`` arm.
    conditional: bool = False

    @property
    def op_name(self) -> str:  # pragma: no cover - overridden
        return "ascend.op"

    def operands(self) -> Sequence[SsaValue]:
        return ()

    def print(self) -> str:  # pragma: no cover - overridden
        return f"{self.op_name}"

    # -- convenience ---------------------------------------------------------
    def walk(self) -> Iterable["AscendOp"]:
        """This op plus every nested op (regions)."""
        yield self
        for child in self.nested_ops():
            yield from child.walk()

    def nested_ops(self) -> List["AscendOp"]:
        return getattr(self, "ops", []) or []


@dataclass
class AllocBufferOp(AscendOp):
    """``ascend.alloc_buffer`` - TPipe::InitBuffer physical allocation."""

    buffer: str = ""
    space: MemorySpace = MemorySpace.UB
    byte_size: int = 0
    #: Queue depth for TQue allocations, 1 for plain TBufs.
    depth: int = 1
    #: Allocation order within the pipe (fixes the bump-pointer layout).
    order: int = 0

    @property
    def op_name(self) -> str:
        return "ascend.alloc_buffer"

    def print(self) -> str:
        return (f"  %{self.buffer} = ascend.alloc_buffer space={self.space.value} "
                f"bytes={self.byte_size} depth={self.depth} order={self.order}")


@dataclass
class GetTensorOp(AscendOp):
    """``ascend.get_tensor`` - TBuf::Get<T>() view over an allocated buffer."""

    buffer: str = ""
    dtype: str = ""
    space: MemorySpace = MemorySpace.UB

    @property
    def op_name(self) -> str:
        return "ascend.get_tensor"

    def print(self) -> str:
        name = self.results[0].name if self.results else "_"
        return (f"  %{name} = ascend.get_tensor %{self.buffer} : "
                f"!ascend.memref<?x{self.dtype}, {self.space.mangled()}>")


@dataclass
class AllocTensorOp(AscendOp):
    """``ascend.alloc_tensor`` - TQue::AllocTensor<T>() queue-slot allocation."""

    queue: str = ""
    dtype: str = ""

    @property
    def op_name(self) -> str:
        return "ascend.alloc_tensor"

    def print(self) -> str:
        name = self.results[0].name if self.results else "_"
        return f"  %{name} = ascend.alloc_tensor %{self.queue} : !ascend.queue"


@dataclass
class EnQueOp(AscendOp):
    """``ascend.enque`` - TQue::EnQue(tensor)."""

    queue: str = ""

    @property
    def op_name(self) -> str:
        return "ascend.enque"

    def operands(self) -> Sequence[SsaValue]:
        return self.results

    def print(self) -> str:
        tensor = self.results[0].name if self.results else "_"
        return f"  ascend.enque %{tensor}, %{self.queue}"


@dataclass
class DeQueOp(AscendOp):
    """``ascend.deque`` - TQue::DeQue<T>()."""

    queue: str = ""
    dtype: str = ""

    @property
    def op_name(self) -> str:
        return "ascend.deque"

    def print(self) -> str:
        name = self.results[0].name if self.results else "_"
        return f"  %{name} = ascend.deque %{self.queue} : !ascend.queue"


@dataclass
class FreeTensorOp(AscendOp):
    """``ascend.free_tensor`` - TQue::FreeTensor(tensor): slot release."""

    queue: str = ""

    @property
    def op_name(self) -> str:
        return "ascend.free_tensor"

    def print(self) -> str:
        tensor = self.results[0].name if self.results else "_"
        return f"  ascend.free_tensor %{tensor}, %{self.queue}"


@dataclass
class MteCopyOp(AscendOp):
    """``ascend.mte_copy`` - a DataCopy (MTE2/MTE3 transfer) of N bytes."""

    src: SsaValue = field(default_factory=lambda: SsaValue("?"))
    dst: SsaValue = field(default_factory=lambda: SsaValue("?"))
    length_bytes: int = 0
    #: Route mnemonic, e.g. ``"MTE2"`` (GM->UB) or ``"MTE3"`` (UB->GM).
    pipe_route: str = ""
    #: Api-level name as written (DataCopy).
    api_name: str = "DataCopy"

    @property
    def op_name(self) -> str:
        return "ascend.mte_copy"

    def operands(self) -> Sequence[SsaValue]:
        return (self.src, self.dst)

    def print(self) -> str:
        return (f"  ascend.mte_copy %{self.src.name} -> %{self.dst.name} : "
                f"{self.length_bytes} bytes via {self.pipe_route or '?'}")


@dataclass
class SetFlagOp(AscendOp):
    """``ascend.set_flag`` - raise a HardEvent cross-pipe event."""

    pipe_route: str = ""
    event_id: int = 0

    @property
    def op_name(self) -> str:
        return "ascend.set_flag"

    def print(self) -> str:
        return f"  ascend.set_flag route={self.pipe_route} event={self.event_id}"


@dataclass
class WaitFlagOp(AscendOp):
    """``ascend.wait_flag`` - consume a HardEvent cross-pipe event."""

    pipe_route: str = ""
    event_id: int = 0

    @property
    def op_name(self) -> str:
        return "ascend.wait_flag"

    def print(self) -> str:
        return f"  ascend.wait_flag route={self.pipe_route} event={self.event_id}"


@dataclass
class BarrierOp(AscendOp):
    """``ascend.barrier`` - PipeBarrier / pipe_barrier fence."""

    target: str = "ALL"

    @property
    def op_name(self) -> str:
        return "ascend.barrier"

    def print(self) -> str:
        return f"  ascend.barrier target={self.target}"


@dataclass
class MmadOp(AscendOp):
    """``ascend.mmad`` - cube multiply-accumulate (Mmad / mad_mx)."""

    op_a: SsaValue = field(default_factory=lambda: SsaValue("?"))
    op_b: SsaValue = field(default_factory=lambda: SsaValue("?"))
    op_c: SsaValue = field(default_factory=lambda: SsaValue("?"))
    api_name: str = "Mmad"
    m: int = 0
    n: int = 0
    k: int = 0

    @property
    def op_name(self) -> str:
        return "ascend.mmad"

    def operands(self) -> Sequence[SsaValue]:
        return (self.op_a, self.op_b, self.op_c)

    def print(self) -> str:
        shape = f" m={self.m} n={self.n} k={self.k}" if self.m or self.n or self.k else ""
        return (f"  ascend.mmad %{self.op_a.name}, %{self.op_b.name} -> %{self.op_c.name}"
                f"{shape}")


@dataclass
class VectorOp(AscendOp):
    """``ascend.vector`` - a vector-pipe compute intrinsic (Add, Mul, ...)."""

    opcode: str = ""
    dst: SsaValue = field(default_factory=lambda: SsaValue("?"))
    srcs: Tuple[SsaValue, ...] = ()
    elem_count: int = 0

    @property
    def op_name(self) -> str:
        return "ascend.vector"

    def operands(self) -> Sequence[SsaValue]:
        return (self.dst,) + tuple(self.srcs)

    def print(self) -> str:
        srcs = ", ".join(v.name for v in self.srcs)
        return f"  ascend.vector {self.opcode} {srcs} -> %{self.dst.name} x{self.elem_count}"


@dataclass
class CoreRegionOp(AscendOp):
    """``ascend.core_region`` - ops compiled onto exactly one core type.

    Heterogeneous (MIX) kernels guard stage code with the runtime core-type
    sentinels; the lowering keeps each arm's operations strictly inside the
    matching region so two classes defining ``Init()``/``Process()`` never
    collide in the symbol table.
    """

    core_type: CoreType = CoreType.AIC
    #: The guarded statement sequence.
    ops: List[AscendOp] = field(default_factory=list)
    #: How the region was selected (condition text).
    condition: str = ""

    @property
    def op_name(self) -> str:
        return "ascend.core_region"

    def print(self) -> str:
        lines = [f"  ascend.core_region<{self.core_type.value}> {{  // {self.condition}"]
        for op in self.ops:
            lines.append(op.print())
        lines.append("  }")
        return "\n".join(lines)


@dataclass
class KernelOp(AscendOp):
    """``ascend.kernel`` - one kernel entry with its full op sequence."""

    name: str = ""
    #: Top-level ops and CoreRegionOps in program order.
    ops: List[AscendOp] = field(default_factory=list)
    #: GM parameters as SSA values.
    params: List[SsaValue] = field(default_factory=list)
    is_kernel_entry: bool = True
    #: Tensor registry the analyzer lowering reads: name -> metadata dict
    #: (dtype / origin / line / type text).
    tensors: Dict[str, dict] = field(default_factory=dict)
    #: Loop records from the lowering walk: id / induction / trip_count ...
    loops: List[dict] = field(default_factory=list)

    @property
    def op_name(self) -> str:
        return "ascend.kernel"

    def print(self) -> str:
        lines = [f"  ascend.kernel @\"{self.name}\"({', '.join(p.name for p in self.params)}) {{"]
        for op in self.ops:
            lines.append(op.print())
        lines.append("  }")
        return "\n".join(lines)


@dataclass
class AscendModule:
    """``ascend.module`` - one translation unit lowered into the dialect."""

    name: str = ""
    kernels: List[KernelOp] = field(default_factory=list)
    #: Folded ``constexpr`` constants visible in the module.
    constants: Dict[str, int] = field(default_factory=dict)
    #: HardEvent route table recovered from the parsed enum, {value: name}.
    hard_event_routes: Dict[int, str] = field(default_factory=dict)
    #: Header mode used during extraction ("stub" or "real").
    header_mode: str = "stub"

    def walk(self) -> Iterable[AscendOp]:
        for kernel in self.kernels:
            yield kernel
            for op in kernel.ops:
                yield from op.walk()

    def dump(self) -> str:
        """Canonical MLIR text for inspection and debugging."""
        lines = [f'ascend.module @"{self.name or "kernel"}" {{']
        for name, value in sorted(self.constants.items()):
            lines.append(f"  ascend.const @{name} = {value}")
        for kernel in self.kernels:
            lines.append(kernel.print())
        lines.append("}")
        return "\n".join(lines)
