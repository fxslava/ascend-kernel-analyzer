"""MLIR verification engine: lowers ``ascend`` dialect modules to the
operation-trace IR and drives the existing rule checkers.

This is the consumer half of the MLIR frontend.  :func:`parse_source_mlir`
runs the full lowering path

    BiSheng extraction -> AST-to-MLIR bridge -> ``ascend`` module
        -> AnalysisUnit (KernelIR per kernel)

and returns the same :class:`~ascend_analyzer.ir.AnalysisUnit` contract the
tree-sitter frontend produces, so every checker (``AKA1001``-``AKA4002``),
report and exit code works unchanged on either frontend.

Because the dialect carries *concrete* layout - :class:`AllocBufferOp` order
fixes a per-space bump layout and queue slots cycle modulo the queue depth -
the lowered ``TensorDecl`` byte ranges are fully static integers.  The memory
audits (overflow, collision, alignment, bank conflicts) therefore evaluate
deterministically from ``AscendMemRefType`` data with no solver involvement,
and the synchronisation audits read literal ``HardEvent`` routes and event
ids straight off the ``SetFlagOp``/``WaitFlagOp`` pairs.

Frontend resilience: when no BiSheng/Clang toolchain can be reached, or a
source uses constructs the extractor cannot typecheck, the function falls
back to the tree-sitter frontend (with an INFO diagnostic) instead of
failing the run - ``--frontend=mlir`` upgrades the analysis where possible
but never breaks it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .diagnostics import DiagnosticCollector, Severity, SourceLoc, Code
from .hardware import (
    HardEventRoute,
    HardwareModel,
    PhysicalDomain,
    Pipe,
    TPosition,
    TPOSITION_TO_DOMAIN,
)
from .ir import AnalysisUnit, ArgRef, KernelIR, TensorDecl
from .ir.kernel_ir import LoopInfo, ApiCallOp, BarrierOp as TraceBarrier, CoreView, FlagKind, FlagOp, Operation, Scope, ScopeKind
from .ir.mlir_ascend import (
    AllocBufferOp,
    AllocTensorOp,
    AscendModule,
    BarrierOp,
    CoreRegionOp,
    DeQueOp,
    EnQueOp,
    FreeTensorOp,
    GetTensorOp,
    KernelOp,
    MemorySpace,
    MmadOp,
    MteCopyOp,
    SetFlagOp,
    VectorOp,
    WaitFlagOp,
)
from .parsing.ast_visitor import VisitorOptions
from .parsing.bisheng_extractor import BishengError, extract_ast
from .parsing.mlir_bridge import BridgeOptions, lower_module
from .symbolic import Const

__all__ = ["parse_source_mlir", "MlirFrontendOptions", "MLIR_FRONTEND_INFO"]

#: Status line surfaced through artifacts/diagnostics when the MLIR path ran.
MLIR_FRONTEND_INFO = "mlir-frontend"

_SPACE_TO_DOMAIN: Dict[MemorySpace, PhysicalDomain] = {
    MemorySpace.UB: PhysicalDomain.UB,
    MemorySpace.L1: PhysicalDomain.L1,
    MemorySpace.L0A: PhysicalDomain.L0A,
    MemorySpace.L0B: PhysicalDomain.L0B,
    MemorySpace.L0C: PhysicalDomain.L0C,
    MemorySpace.BT: PhysicalDomain.BT,
    MemorySpace.FB: PhysicalDomain.FB,
    MemorySpace.GM: PhysicalDomain.GM,
}

_ROUTE_TO_PIPE: Dict[str, Pipe] = {
    "MTE2": Pipe.MTE2, "MTE3": Pipe.MTE3, "MTE1": Pipe.MTE1, "FIX": Pipe.FIX,
}

#: Positions a queue/buffer name may carry; anything else maps to UB.
_POSITION_TO_TPOS: Dict[str, TPosition] = {
    tp.value: tp for tp in TPosition
}


class MlirFrontendOptions:
    """Options for the MLIR frontend, mirroring the relevant visitor knobs."""

    def __init__(self,
                 all_functions: bool = False,
                 max_ops: int = 20000,
                 header_mode: str = "stub",
                 use_cache: bool = True,
                 emit_queue_ops: bool = True,
                 strict_frontend: bool = False,
                 tiling_values: Optional[Dict[str, int]] = None,
                 dump_path: Optional[Path] = None) -> None:
        self.dump_path = dump_path
        self.strict_frontend = strict_frontend
        self.tiling_values = dict(tiling_values or {})
        self.all_functions = all_functions
        self.max_ops = max_ops
        self.header_mode = header_mode
        self.use_cache = use_cache
        self.emit_queue_ops = emit_queue_ops


def parse_source_mlir(path: str, source: str, hardware: HardwareModel,
                      diagnostics: DiagnosticCollector,
                      options: Optional[MlirFrontendOptions] = None,
                      visitor_options: Optional[VisitorOptions] = None,
                      toolchain_cache: Optional[Dict] = None) -> AnalysisUnit:
    """Analyze one translation unit through the MLIR frontend.

    Falls back to the tree-sitter frontend (with an INFO diagnostic) when the
    BiSheng toolchain is unavailable or extraction fails, so callers get a
    usable unit either way.
    """
    opts = options or MlirFrontendOptions()
    try:
        result = extract_ast(path, source, chip=hardware.chip.name,
                             header_mode=opts.header_mode,
                             use_cache=opts.use_cache and not opts.strict_frontend)
        module = lower_module(
            result.ast, path, source,
            options=BridgeOptions(queue_ops=opts.emit_queue_ops,
                                  max_ops=opts.max_ops, tiling_values=opts.tiling_values,
                                  strict_layout=opts.strict_frontend),
            all_functions=opts.all_functions)
    except BishengError as exc:
        if opts.strict_frontend:
            raise
        _fallback_info(diagnostics, path, f"BiSheng extraction unavailable ({exc}); "
                                         f"falling back to the tree-sitter frontend")
        return _tree_sitter_fallback(path, source, hardware, diagnostics,
                                     visitor_options)
    if result.errors:
        if opts.strict_frontend:
            raise BishengError(f"MLIR extraction failed for {path}:\n{result.compiler_log or chr(10).join(result.errors)}")
        _fallback_info(diagnostics, path,
                       f"source did not typecheck against the Ascend C headers "
                       f"({len(result.errors)} error(s), first: {result.errors[0][:120]}); "
                       f"falling back to the tree-sitter frontend")
        return _tree_sitter_fallback(path, source, hardware, diagnostics,
                                     visitor_options)
    if opts.dump_path is not None:
        opts.dump_path.write_text(module.dump() + chr(10), encoding="utf-8")
    unit = lower_to_unit(module, path, source, hardware, diagnostics, opts)
    unit.frontend_metadata = {
        "frontend": "mlir", "origin": result.origin, "header_mode": result.header_mode,
        "toolchain": result.toolchain.describe() if result.toolchain else result.origin,
        "tiling_values": dict(opts.tiling_values), "fallback": False,
    }
    return unit


def _fallback_info(diagnostics: DiagnosticCollector, path: str, message: str) -> None:
    diagnostics.add(code=Code.PARSE_ERROR, severity=Severity.INFO, message=message,
                    loc=SourceLoc.unknown(path))


def _tree_sitter_fallback(path: str, source: str, hardware: HardwareModel,
                          diagnostics: DiagnosticCollector,
                          visitor_options: Optional[VisitorOptions]) -> AnalysisUnit:
    from .parsing import parse_source as parse_tree_sitter

    return parse_tree_sitter(path, source, hardware, diagnostics,
                             visitor_options or VisitorOptions())


# ---------------------------------------------------------------------------
# Module -> AnalysisUnit lowering
# ---------------------------------------------------------------------------


class _LayoutSynthesizer:
    """Concrete per-space bump layout from ``alloc_buffer`` order.

    Mirrors the TPipe semantics the tree-sitter frontend synthesises: buffers
    allocate in ``InitBuffer`` call order inside each physical space, each
    32-byte aligned; queue slots cycle modulo the queue depth.
    """

    ALIGN = 32

    def __init__(self) -> None:
        self.bases: Dict[str, int] = {}
        self.sizes: Dict[str, int] = {}
        self.depths: Dict[str, int] = {}
        self.spaces: Dict[str, MemorySpace] = {}
        self.cursors: Dict[MemorySpace, int] = {}
        #: Queue slot counters for round-robin slot assignment.
        self.slots: Dict[str, int] = {}

    def alloc(self, op: AllocBufferOp) -> None:
        cursor = self.cursors.get(op.space, 0)
        base = _align_up(cursor, self.ALIGN)
        self.bases[op.buffer] = base
        self.sizes[op.buffer] = op.byte_size * max(op.depth, 1)
        self.depths[op.buffer] = max(op.depth, 1)
        self.spaces[op.buffer] = op.space
        self.cursors[op.space] = base + max(op.byte_size, 0) * max(op.depth, 1)
        self.slots.setdefault(op.buffer, 0)

    def slot_offset(self, buffer: str) -> Optional[int]:
        """Byte offset of the next queue slot in ``buffer``."""
        if buffer not in self.bases:
            return None
        depth = self.depths[buffer]
        total = self.sizes[buffer]
        slot = self.slots.get(buffer, 0) % depth if depth else 0
        self.slots[buffer] = self.slots.get(buffer, 0) + 1
        block = total // depth if depth else total
        return self.bases[buffer] + slot * block

    def peek_offset(self, buffer: str) -> Optional[int]:
        """Next slot offset without consuming it."""
        if buffer not in self.bases:
            return None
        depth = self.depths[buffer]
        total = self.sizes[buffer]
        slot = self.slots.get(buffer, 0) % depth if depth else 0
        block = total // depth if depth else total
        return self.bases[buffer] + slot * block

    def buffer_offset(self, buffer: str) -> Optional[int]:
        return self.bases.get(buffer)


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


_DTYPE_SIZES: Dict[str, int] = {
    "bool": 1, "char": 1, "int8_t": 1, "uint8_t": 1, "unsigned char": 1,
    "int4b_t": 1, "uint4b_t": 1,
    "int16_t": 2, "uint16_t": 2, "short": 2, "half": 2, "bfloat16_t": 2,
    "unsigned short": 2,
    "int32_t": 4, "uint32_t": 4, "int": 4, "unsigned int": 4, "float": 4,
    "int64_t": 8, "uint64_t": 8, "long": 8, "unsigned long": 8, "double": 8,
}


def lower_to_unit(module: AscendModule, path: str, source: str,
                  hardware: HardwareModel,
                  diagnostics: DiagnosticCollector,
                  options: Optional[MlirFrontendOptions] = None) -> AnalysisUnit:
    """Lower an ``ascend`` module into the checker-facing AnalysisUnit."""
    opts = options or MlirFrontendOptions()
    lines = source.splitlines()
    unit = AnalysisUnit(path=path, source=source)

    for kernel_op in module.kernels:
        kernel = _lower_kernel(kernel_op, path, lines, hardware, module)
        if kernel is not None:
            unit.kernels.append(kernel)

    unit.constants.update(module.constants)
    if not unit.kernels:
        unit.had_parse_errors = False
    return unit


def _loc(path: str, lines: List[str], line: int) -> SourceLoc:
    snippet = lines[line - 1].strip()[:160] if 1 <= line <= len(lines) else ""
    return SourceLoc(file=path, line=max(line, 1), snippet=snippet)


def _lower_kernel(kernel_op: KernelOp, path: str, lines: List[str],
                  hardware: HardwareModel, module: AscendModule) -> Optional[KernelIR]:
    kernel = KernelIR(name=kernel_op.name,
                      loc=_loc(path, lines, kernel_op.line or 1))
    kernel.constants.update(module.constants)
    layout = _LayoutSynthesizer()

    # Pass 1: physical layout from allocation order (regions contribute in
    # program order; both cores' buffers share the per-space budget).
    for op in kernel_op.walk():
        if isinstance(op, AllocBufferOp):
            layout.alloc(op)
            kernel.buffer_sizes[op.buffer] = op.byte_size

    # Pass 2: tensor table.
    _lower_tensors(kernel, kernel_op, layout, path, lines)

    # Pass 3: operation trace.
    state = _TraceState(kernel=kernel, path=path, lines=lines, layout=layout)
    for op in kernel_op.ops:
        _lower_trace_op(state, op, CoreView.BOTH, 0, False)
    for info in kernel_op.loops:
        ranges = [state.op_indices[id(op)] for op in info.get("ops", []) if id(op) in state.op_indices]
        if not ranges:
            continue
        kernel.loops[info["id"]] = LoopInfo(
            id=info["id"], scope_id=0, loc=_loc(path, lines, info["line"]),
            induction_var=info.get("induction"), trip_count=info.get("trip_count"),
            start=info.get("start"), step=info.get("step"),
            unrolled=info.get("unrolled", False),
            start_index=min(a for a, b in ranges), end_index=max(b for a, b in ranges),
            header=info.get("header", ""))
    return kernel


class _TraceState:
    """Mutable state for one kernel's trace lowering."""

    def __init__(self, kernel: KernelIR, path: str, lines: List[str],
                 layout: _LayoutSynthesizer) -> None:
        self.kernel = kernel
        self.path = path
        self.lines = lines
        self.layout = layout
        self.index = 0
        self.op_indices: Dict[int, Tuple[int, int]] = {}
        kernel.scopes[0] = Scope(id=0, kind=ScopeKind.KERNEL, parent=None,
                                 loc=kernel.loc)

    def next_args(self, line: int, loop_id: Optional[int], conditional: bool,
                  core_view: CoreView) -> Dict:
        self.index += 1
        return {
            "index": self.index,
            "loc": _loc(self.path, self.lines, line or 1),
            "scope_id": 0,
            "loop_id": loop_id,
            "conditional": conditional,
            "core_view": core_view,
        }


def _lower_tensors(kernel: KernelIR, kernel_op: KernelOp,
                   layout: _LayoutSynthesizer, path: str, lines: List[str]) -> None:
    """Materialise TensorDecls from params and the tensor registry.

    Buffer families (TQue/TBuf) are *not* declared as tensors themselves -
    their bytes are budgeted through ``kernel.buffer_sizes`` and attributed to
    the views, exactly like the tree-sitter frontend's TPipe layout
    synthesis, so a view and its buffer never count as two live tensors.
    """
    # Kernel pointer parameters are GM by Ascend C convention.
    for param in kernel_op.params:
        qt = param.type_str
        if "*" not in qt or "Tensor" in qt:
            continue
        dtype = qt.replace("*", "").strip().rsplit("::", 1)[-1]
        kernel.tensors[param.name] = TensorDecl(
            name=param.name, loc=_loc(path, lines, param.line or 1),
            position=TPosition.GM, domain=PhysicalDomain.GM,
            dtype=dtype, elem_size=_DTYPE_SIZES.get(dtype, 1),
            origin="kernel parameter")

    for name, info in kernel_op.tensors.items():
        if name in kernel.tensors and kernel.tensors[name].domain is PhysicalDomain.GM:
            # A local re-binding of a GM pointer keeps its parameter decl.
            continue
        dtype = info.get("dtype")
        origin = info.get("origin", "declaration")
        type_text = info.get("type", "")
        line = info.get("line") or 1
        decl = TensorDecl(name=name, loc=_loc(path, lines, line),
                          position=None, domain=PhysicalDomain.UB,
                          dtype=dtype, elem_size=_DTYPE_SIZES.get(dtype or "", 1),
                          origin=f"mlir:{origin}")

        if "GlobalTensor" in type_text or ("*" in type_text and "Tensor" not in type_text):
            # Raw pointer locals re-binding GM addresses.
            decl.domain = PhysicalDomain.GM
            decl.position = TPosition.GM
            kernel.tensors[name] = decl
            continue

        method = origin.split(":", 1)[1] if origin.startswith("queue:") else None
        if method in ("AllocTensor", "DeQue"):
            queue = _queue_of_tensor(kernel_op, name)
            if queue:
                offset = layout.slot_offset(queue)
                if offset is not None:
                    decl.byte_offset = Const(offset)
                    space = _space_of_buffer(kernel_op, queue)
                    decl.domain = _SPACE_TO_DOMAIN.get(space, PhysicalDomain.UB)
                    decl.position = _position_of_buffer(kernel_op, queue)
                    block = (layout.sizes.get(queue, 0) //
                             max(layout.depths.get(queue, 1), 1))
                    decl.byte_size = Const(block)
                    decl.source_buffer = queue
        else:
            # TBuf::Get views and subscripted views: base + element offset.
            base_name, element_offset = _split_view_offset(name)
            buffer = _buffer_of_view(kernel_op, base_name, name)
            if buffer is None and method == "Get":
                buffer = _queue_of_get(kernel_op, name)
            if buffer:
                offset = layout.buffer_offset(buffer)
                if offset is not None:
                    elem = decl.elem_size or 1
                    decl.byte_offset = Const(offset + element_offset * elem)
                    decl.byte_size = Const(info["view_count"] * elem if info.get("view_count") is not None else layout.sizes.get(buffer, 0))
                    decl.source_buffer = buffer
                    decl.origin = "mlir:view"
                    space = _space_of_buffer(kernel_op, buffer)
                    decl.domain = _SPACE_TO_DOMAIN.get(space, PhysicalDomain.UB)
                    decl.position = _position_of_buffer(kernel_op, buffer)
        kernel.tensors[name] = decl

    # Resolve typed aliases after all defining buffer views are materialised.
    for _ in range(len(kernel_op.tensors)):
        changed = False
        for name, info in kernel_op.tensors.items():
            alias = info.get("alias")
            if not alias:
                continue
            base_name, element_offset = _split_view_offset(alias)
            base = kernel.tensors.get(base_name)
            decl = kernel.tensors.get(name)
            if base is None or decl is None or base.offset_value is None:
                continue
            offset = base.offset_value + element_offset * (base.elem_size or 1)
            if decl.offset_value != offset:
                decl.byte_offset = Const(offset)
                decl.byte_size = Const(max(0, (base.size_value or 0) - element_offset * (base.elem_size or 1)))
                decl.source_buffer = base.source_buffer
                decl.origin = "mlir:alias"
                decl.domain, decl.position = base.domain, base.position
                changed = True
        if not changed:
            break

    # Views that appear only as operand names (inline ``buf.Get<T>()[i]``
    # expressions) still need their own disjoint byte ranges.
    for value_name in _operand_view_names(kernel_op):
        if value_name in kernel.tensors:
            continue
        base_name, element_offset = _split_view_offset(value_name)
        buffer = _buffer_of_view(kernel_op, base_name, value_name)
        dtype = None
        if buffer is None:
            continue
        offset = layout.buffer_offset(buffer)
        if offset is None:
            continue
        for op in kernel_op.walk():
            if isinstance(op, GetTensorOp) and op.results and \
                    op.results[0].name in (value_name, base_name):
                dtype = op.dtype
                break
        elem = _DTYPE_SIZES.get(dtype or "", 1) if dtype else 1
        view_decl = TensorDecl(
            name=value_name, loc=kernel.loc, position=None,
            domain=_SPACE_TO_DOMAIN.get(_space_of_buffer(kernel_op, buffer),
                                        PhysicalDomain.UB),
            dtype=dtype, elem_size=elem or 1,
            byte_offset=Const(offset + element_offset * (elem or 1)),
            byte_size=Const(layout.sizes.get(buffer, 0)),
            source_buffer=buffer, origin="mlir:view")
        view_decl.position = _position_of_buffer(kernel_op, buffer)
        kernel.tensors[value_name] = view_decl


def _operand_view_names(kernel_op: KernelOp) -> List[str]:
    """Distinct operand names that reference buffer views."""
    names: List[str] = []

    def add(value) -> None:
        name = getattr(value, "name", None)
        if name and ".view" in name and name not in names:
            names.append(name)

    for op in kernel_op.walk():
        for value in getattr(op, "results", []) or []:
            add(value)
        operands = getattr(op, "operands", None)
        if operands:
            for value in operands():
                add(value)
        for attr in ("src", "dst", "op_a", "op_b", "op_c"):
            add(getattr(op, attr, None))
        for value in getattr(op, "srcs", ()) or ():
            add(value)
    return names


def _split_view_offset(name: str) -> Tuple[str, int]:
    """Split an SSA view name ``base+<elements>`` into (base, offset)."""
    if "+" in name:
        base, _, suffix = name.rpartition("+")
        try:
            return base, int(suffix)
        except ValueError:
            return name, 0
    return name, 0


def _queue_of_get(kernel_op: KernelOp, name: str) -> Optional[str]:
    for op in kernel_op.walk():
        if isinstance(op, GetTensorOp) and op.results and \
                op.results[0].name in (name, _split_view_offset(name)[0]):
            return op.buffer
    return None


def _space_of_buffer(kernel_op: KernelOp, buffer: str) -> MemorySpace:
    for op in kernel_op.walk():
        if isinstance(op, AllocBufferOp) and op.buffer == buffer:
            return op.space
    return MemorySpace.UB


def _position_of_buffer(kernel_op: KernelOp, buffer: str) -> Optional[TPosition]:
    for op in kernel_op.walk():
        if isinstance(op, AllocBufferOp) and op.buffer == buffer:
            for tpos in TPosition:
                if tpos.value in buffer:
                    return tpos
    return None


def _queue_of_tensor(kernel_op: KernelOp, tensor: str) -> Optional[str]:
    """The queue a tensor came from, by matching queue-op results."""
    for op in kernel_op.walk():
        if isinstance(op, (AllocTensorOp, DeQueOp)) and op.results:
            if op.results[0].tensor == tensor or op.results[0].name == tensor:
                return op.queue
    return None


def _buffer_of_view(kernel_op: KernelOp, name: str,
                    full_name: Optional[str] = None) -> Optional[str]:
    """The TBuf behind a ``...Get``/``...view`` access path.

    ``name`` is the offset-stripped base; ``full_name`` the original value
    name as it appears in ops (used for GetTensorOp result matching).
    """
    if name.endswith(".view") or name.endswith(".Get"):
        return name.rsplit(".", 1)[0]
    for op in kernel_op.walk():
        if isinstance(op, GetTensorOp) and op.results:
            for candidate in (full_name, name):
                if candidate and op.results[0].name == candidate:
                    return op.buffer
    return None


# -- trace lowering -----------------------------------------------------------


def _lower_trace_op(state: _TraceState, op, core: CoreView, loop_depth: int,
                    conditional: bool) -> None:
    start = state.index + 1
    _lower_trace_operation(state, op, core, loop_depth, conditional)
    if state.index >= start:
        state.op_indices[id(op)] = (start, state.index)


def _lower_trace_operation(state: _TraceState, op, core: CoreView, loop_depth: int,
                           conditional: bool) -> None:
    if isinstance(op, CoreRegionOp):
        view = CoreView.AIC if op.core_type.value == "AIC" else CoreView.AIV
        for nested in op.ops:
            _lower_trace_op(state, nested, view, loop_depth,
                            conditional or op.conditional)
        return

    if isinstance(op, (SetFlagOp, WaitFlagOp)):
        route = HardEventRoute.parse(op.pipe_route) if op.pipe_route else None
        flag = FlagOp(
            flag_kind=FlagKind.SET if isinstance(op, SetFlagOp) else FlagKind.WAIT,
            route=route,
            event_id=op.event_id if op.event_id is not None else None,
            event_id_text=str(op.event_id) if op.event_id is not None else op.text[:60],
            pipe=(route.src if isinstance(op, SetFlagOp) else route.dst)
            if route else Pipe.S,
            **state.next_args(op.line, op.loop_id, conditional, core),
        )
        _append(state, flag)
        return

    if isinstance(op, BarrierOp):
        target = _PIPE_BY_NAME.get(op.target, Pipe.ALL)
        _append(state, TraceBarrier(
            target=target,
            pipe=target if target is not Pipe.ALL else Pipe.ALL,
            **state.next_args(op.line, op.loop_id, conditional, core)))
        return

    if isinstance(op, MteCopyOp):
        pipe = _ROUTE_TO_PIPE.get(op.pipe_route, Pipe.MTE1)
        args = [
            ArgRef(index=0, text=op.dst.name, tensor=_resolve(state, op.dst.name)),
            ArgRef(index=1, text=op.src.name, tensor=_resolve(state, op.src.name)),
            ArgRef(index=2, text=str(op.length_bytes), value=op.length_bytes),
        ]
        writes = [t for t in (_resolve(state, op.dst.name),) if t]
        reads = [t for t in (_resolve(state, op.src.name),) if t]
        _append(state, ApiCallOp(
            name=op.api_name, args=tuple(args),
            writes=tuple(writes), reads=tuple(reads), text=op.text or op.api_name,
            pipe=pipe,
            **state.next_args(op.line, op.loop_id, conditional, core)))
        _touch_tensors(state, writes + reads)
        return

    if isinstance(op, MmadOp):
        args = [ArgRef(index=i, text=v.name, tensor=_resolve(state, v.name))
                for i, v in enumerate((op.op_c, op.op_a, op.op_b))]
        _append(state, ApiCallOp(
            name=op.api_name, args=tuple(args),
            writes=(_resolve(state, op.op_c.name) or op.op_c.name,),
            reads=tuple(t for t in (
                _resolve(state, op.op_a.name), _resolve(state, op.op_b.name)) if t),
            text=op.text or op.api_name, pipe=Pipe.M,
            **state.next_args(op.line, op.loop_id, conditional, core)))
        return

    if isinstance(op, VectorOp):
        args = [ArgRef(index=0, text=op.dst.name, tensor=_resolve(state, op.dst.name))]
        for i, v in enumerate(op.srcs):
            args.append(ArgRef(index=i + 1, text=v.name,
                               tensor=_resolve(state, v.name)))
        args.append(ArgRef(index=len(args), text=str(op.elem_count),
                           value=op.elem_count))
        if op.api_args:
            args = [ArgRef(index=i, text=v.name, tensor=_resolve(state, v.name) if v.tensor else None,
                           value=int(v.name) if v.name.lstrip("-").isdigit() else None)
                    for i, v in enumerate(op.api_args)]
        _append(state, ApiCallOp(
            name=op.opcode, args=tuple(args),
            writes=tuple(_resolve(state, v.name) or v.name for v in (op.dst,) + op.extra_dsts),
            reads=tuple(t for t in
                        (_resolve(state, v.name) for v in op.srcs) if t),
            text=op.text or op.opcode, pipe=Pipe.V,
            **state.next_args(op.line, op.loop_id, conditional, core)))
        return

    # Queue bookkeeping and allocations shape the tensor table only.
    if isinstance(op, (AllocBufferOp, GetTensorOp, AllocTensorOp, DeQueOp,
                       EnQueOp, FreeTensorOp)):
        return


def _append(state: _TraceState, operation: Operation) -> None:
    state.kernel.ops.append(operation)
    if isinstance(operation, ApiCallOp):
        _touch_tensors(state, list(operation.writes) + list(operation.reads))
    if state.kernel.scopes:
        scope = state.kernel.scopes[0]
        scope.end_index = operation.index


def _resolve(state: _TraceState, name: str) -> Optional[str]:
    """Map an SSA value name onto a registered tensor, if any."""
    if name in state.kernel.tensors:
        return name
    # Subscripted views resolve onto their own decl, then the base view.
    base, offset = _split_view_offset(name)
    if offset:
        plain = f"{base}+{offset}"
        if plain in state.kernel.tensors:
            return plain
    if base != name and base in state.kernel.tensors:
        return base
    return name


def _touch_tensors(state: _TraceState, names: List[str]) -> None:
    for name in names:
        decl = state.kernel.tensors.get(name)
        if decl is None:
            continue
        idx = len(state.kernel.ops)
        if decl.first_use is None:
            decl.first_use = idx
        decl.last_use = idx


_PIPE_BY_NAME: Dict[str, Pipe] = {
    "S": Pipe.S, "V": Pipe.V, "M": Pipe.M, "MTE1": Pipe.MTE1,
    "MTE2": Pipe.MTE2, "MTE3": Pipe.MTE3, "FIX": Pipe.FIX, "ALL": Pipe.ALL,
}
