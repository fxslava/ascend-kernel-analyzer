"""The AST visitor: C++ syntax tree -> :class:`~ascend_analyzer.ir.KernelIR`.

The visitor extracts exactly four things from each kernel body, in source
program order:

* **tensor bindings** - a name, a data type, a memory domain and a byte range,
  recovered from ``LocalTensor`` declarations, ``SetAddr``/``SetSize`` calls,
  raw address-space pointers, or an explicit ``@ascend-layout`` annotation;
* **synchronisation** - ``SetFlag``/``WaitFlag`` in both the templated Ascend C
  spelling and the low-level ISASI ``set_flag(PIPE_A, PIPE_B, id)`` form;
* **barriers** - ``PipeBarrier<PIPE_X>()`` / ``pipe_barrier(PIPE_X)``; and
* **DMA and compute intrinsics** - resolved against the signature table in
  :mod:`ascend_analyzer.apis` so each lands on the right hardware pipeline.

Recognised tensor-binding forms are documented in ``README.md``.  Anything the
visitor cannot resolve is recorded as unresolved rather than guessed at, and
the checkers downgrade their conclusions accordingly - a static analyzer that
invents facts is worse than one that admits ignorance.
"""

from __future__ import annotations

import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import gcd as _gcd
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from tree_sitter import Language, Node, Parser

from ..apis import ApiSpec, ArgRole, data_copy_pipe, lookup_api
from ..diagnostics import Code, DiagnosticCollector, Severity, SourceLoc
from ..hardware import (
    DOMAIN_TO_DEFAULT_TPOSITION,
    HardEventRoute,
    HardwareModel,
    PhysicalDomain,
    Pipe,
    TPosition,
)
from ..ir import (
    CoreView,
    AnalysisUnit,
    ApiCallOp,
    ArgRef,
    BarrierOp,
    FlagKind,
    FlagOp,
    KernelIR,
    LoopInfo,
    Operation,
    Scope,
    ScopeKind,
    TensorDecl,
)
from ..symbolic import (
    BinOp,
    Const,
    Expr,
    Var,
    dtype_size,
    free_vars,
    mul,
    simplify,
    substitute,
    to_int,
)
from .expr_eval import (
    ConstEnv,
    ExpressionEvaluator,
    canonical_accessor,
    collect_define_value,
)
from .preprocess import PreparedSource, prepare_translation_unit

__all__ = ["ASTVisitor", "parse_source", "parse_file", "VisitorOptions"]


# ---------------------------------------------------------------------------
# Recognition patterns
# ---------------------------------------------------------------------------

_TENSOR_TYPE_RE = re.compile(
    r"\b(?P<kind>LocalTensor|GlobalTensor)\s*<\s*(?P<dtype>[A-Za-z_][\w:]*)\s*>"
)
#: Bound on nested single-return helper folding.
_MAX_HELPER_DEPTH = 8
#: Bound on call-graph inlining depth.
_MAX_INLINE_DEPTH = 6

_TBUF_TYPE_RE = re.compile(r"\bT(?:Buf|Que|QueBind)\b.*?TPosition::(?P<pos>[A-Z0-9_]+)")
#: The ping/pong depth of ``TQue<TPosition::VECIN, 2>`` - the trailing integer
#: template argument.  ``TBuf`` has no depth argument and reserves one block.
_QUEUE_DEPTH_RE = re.compile(
    r"\bT(?:Que|QueBind)\b[^>]*?TPosition::[A-Z0-9_]+\s*,\s*(?P<depth>\d+)"
)


#: ``hidden_``, ``this->hidden_`` and ``obj.hidden_`` all name one member; a
#: subscript or call target is not a scalar member and is rejected.
_MEMBER_TARGET_RE = re.compile(r"^(?:this\s*->\s*|[A-Za-z_]\w*\s*\.\s*)?([A-Za-z_]\w*)$")


def _strip_template_args(name: str) -> str:
    """``Service<A::B>::f`` -> ``Service::f``; drops every ``<...>`` group."""
    depth = 0
    out: List[str] = []
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip()


def _enclosing_class_name(visitor: "ASTVisitor", func: Node) -> Optional[str]:
    """The class a function belongs to, or ``None`` for a free function.

    Mirrored method names are the norm in these kernels - a Vector service and
    a Cube service both define ``AllocEventID`` - so the owning class is what
    makes a call site resolvable.  Taken from the qualified declarator when the
    definition is written out of class, otherwise from the enclosing class or
    struct body.
    """
    declarator = func.child_by_field_name("declarator")
    if declarator is not None:
        qualified = _strip_template_args(visitor.text(declarator))
        head = qualified.split("(", 1)[0]
        if "::" in head:
            return head.rsplit("::", 2)[-2].strip() or None
    node = func.parent
    for _ in range(8):
        if node is None:
            break
        if node.type in {"class_specifier", "struct_specifier"}:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                return _strip_template_args(visitor.text(name_node)) or None
            return None
        node = node.parent
    return None


def _bare_function_name(name: str) -> str:
    """``Service<T>::AllocEventID`` -> ``AllocEventID``.

    A member definition carries its class qualification and template
    arguments; the call site does not.  Matching the two needs the bare name.
    Template arguments are dropped before splitting so a ``::`` inside them
    (``Service<A::B>::f``) cannot be mistaken for the class separator.
    """
    depth = 0
    out: List[str] = []
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out).rsplit("::", 1)[-1].strip()


def _member_target_name(text: str) -> Optional[str]:
    """The bare member name an assignment writes, or ``None`` if not a member."""
    match = _MEMBER_TARGET_RE.match(text.strip())
    return match.group(1) if match else None


#: Compile-time core selectors and the core each one guards.
_CORE_GUARD_MACROS = {
    "__DAV_C220_CUBE__": CoreView.AIC,
    "__DAV_C220_VEC__": CoreView.AIV,
}
#: Run-time core predicates, as the macro expander rewrites them.
_CORE_PREDICATES = {
    "__ascend_core_is_aic": CoreView.AIC,
    "__ascend_core_is_aiv": CoreView.AIV,
}
_CORE_PREDICATE_RE = re.compile(
    r"^\s*\(*\s*(?P<not>!\s*)?\(*\s*(?P<name>__ascend_core_is_ai[cv])\s*\)*\s*$"
)


def _core_view_of_ifdef(visitor: "ASTVisitor", node: Node) -> CoreView:
    """The core a ``#ifdef``/``#if`` selects, or ``BOTH`` if it is unrelated."""
    name_node = node.child_by_field_name("name") or node.child_by_field_name(
        "condition"
    )
    if name_node is None:
        return CoreView.BOTH
    text = visitor.text(name_node)
    negated = "!" in text or "ndef" in visitor.text(node)[:12]
    for macro, view in _CORE_GUARD_MACROS.items():
        if macro in text:
            return view.complement if negated else view
    return CoreView.BOTH


def _core_view_of_condition(
    visitor: "ASTVisitor", condition: Optional[Node]
) -> CoreView:
    """The core an ``if`` condition selects, or ``BOTH`` if it is unrelated."""
    if condition is None:
        return CoreView.BOTH
    match = _CORE_PREDICATE_RE.match(visitor.text(condition))
    if match is None:
        return CoreView.BOTH
    view = _CORE_PREDICATES[match.group("name")]
    return view.complement if match.group("not") else view


#: A cast whose target type names a tiling struct.  This is what identifies a
#: tiling pointer; the name of the variable it lands in is never consulted.
_TILING_CAST_RE = re.compile(
    r"(?:reinterpret_cast|static_cast|const_cast)\s*<[^>]*Tiling\w*\s*\*"
    r"|\(\s*(?:\w+\s+|/\*\w+\*/\s+)*\w*Tiling\w*\s*\*\s*\)",
    re.IGNORECASE,
)

#: The core index, which marks a field as a core-grid bound rather than an
#: extent.
_CORE_INDEX_RE = re.compile(r"\bGet(?:Block|SubBlock)(?:Idx|Num)\s*\(")

#: APIs whose trailing parameter block carries the DMA strides and gaps.
_DMA_APIS = frozenset({"DataCopy", "DataCopyPad"})
#: ``DataCopy(dst, src, count, params)`` - the parameter block is argument 3.
_DMA_PARAMS_INDEX = 3
#: An aggregate whose type names a DMA parameter block.
_DMA_PARAMS_RE = re.compile(r"\bDataCopy\w*Params\b|\bNd2NzParams\b")

#: Architecture-minimal valid extents, in the units ``InitBuffer`` takes.
#: The Cube unit addresses L1/L0 in 16-wide fractals; the Vector unit works in
#: 64-element tiles over UB.
_CUBE_MIN_FRACTAL = 16
_VECTOR_MIN_TILE = 64
#: Domains the Cube unit addresses.
_CUBE_DOMAINS = frozenset(
    {PhysicalDomain.L1, PhysicalDomain.L0A, PhysicalDomain.L0B, PhysicalDomain.L0C}
)

#: Receiver spellings a kernel uses to reach its tiling struct.
_TILING_RECEIVERS = (
    "tilingData", "tiling_data", "tiling", "tilingDataPtr", "tilingPtr",
)

#: Accessors that hand out storage from a TPipe-managed buffer, and the
#: tensor origin each one records.
_PIPE_ACCESSORS = {
    "Get": "TBuf::Get",
    "AllocTensor": "TQue::AllocTensor",
    "DeQue": "TQue::DeQue",
}


def _queue_depth(type_text: str) -> int:
    """Blocks reserved by a queue declaration; 1 when it is a plain ``TBuf``."""
    match = _QUEUE_DEPTH_RE.search(type_text)
    if match is None:
        return 1
    depth = int(match.group("depth"))
    return depth if depth > 0 else 1


@dataclass(frozen=True)
class PipeBuffer:
    """One ``TBuf``/``TQue`` member and the extent its ``InitBuffer`` reserves.

    ``block_bytes`` is the per-block length passed to ``InitBuffer``; a queue of
    ``depth`` blocks reserves ``depth * block_bytes`` contiguous bytes, which is
    what the bump allocator consumes.
    """

    name: str
    position: TPosition
    depth: int = 1
    block_bytes: Optional[Expr] = None
    #: Position in ``InitBuffer`` call order, which is what fixes the layout.
    #: ``None`` for a buffer that is never sized.
    order: Optional[int] = None

    @property
    def total_bytes(self) -> Optional[Expr]:
        if self.block_bytes is None:
            return None
        return mul(self.block_bytes, Const(self.depth)) if self.depth > 1 else self.block_bytes
_LEADING_IDENT_RE = re.compile(r"^\s*\(*\s*(?:[A-Za-z_]\w*::)*(?P<name>[A-Za-z_]\w*)")

#: Scalar type words that may appear as the operand of a C-style cast.
_BUILTIN_TYPE_WORDS = frozenset({
    "void", "bool", "char", "int", "float", "double", "half", "long", "short",
    "signed", "unsigned", "size_t", "uintptr_t", "intptr_t", "ptrdiff_t",
    "auto", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "bfloat16", "int4b_t", "uint4b_t",
})
_TYPE_NAME_RE = re.compile(r"^(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*$")


def _looks_like_type_operand(text: str) -> bool:
    """``True`` when *text* is a bare type name with optional ``*``/``&``."""
    body = text.strip()
    stars = 0
    while body.endswith("*") or body.endswith("&"):
        body = body[:-1].rstrip()
        stars += 1
    if not _TYPE_NAME_RE.match(body):
        return False
    name = body.rsplit("::", 1)[-1]
    return name.endswith("_t") or name in _BUILTIN_TYPE_WORDS


def _strip_leading_casts(text: str) -> str:
    """Remove leading C-style casts and comments from an argument's text.

    Cube kernels pass raw addresses as ``(/*ca*/ fp4x2_e2m1_t *)(uintptr_t)
    l0a[...].GetPhyAddr()``; the tensor underneath all that casting is what
    the analyzer must resolve.  Parenthesised *expressions* are left alone -
    only groups whose content is a type name are treated as casts.
    """
    cleaned = re.sub(r"/\*.*?\*/", " ", text)
    changed = True
    while changed and cleaned.lstrip().startswith("("):
        changed = False
        depth = 0
        end = -1
        for index, ch in enumerate(cleaned):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end < 0:
            break
        inner = cleaned[cleaned.index("(") + 1 : end]
        if not _looks_like_type_operand(inner):
            break
        cleaned = cleaned[end + 1 :]
        changed = True
    return cleaned

#: Methods that bind a tensor's base byte offset.
_SET_OFFSET_METHODS = frozenset({"SetAddr", "SetBufferAddr", "SetAddrByByte"})
#: Methods that bind a tensor's length in *elements*.
_SET_COUNT_METHODS = frozenset({"SetSize", "SetShapeSize"})
#: Methods that bind a tensor's length in *bytes*.
_SET_BYTES_METHODS = frozenset({"SetBufferLen", "SetByteSize", "SetBufferSize"})
#: Methods that bind the logical tensor position.
_SET_POSITION_METHODS = frozenset({"SetTPosition", "SetPosition"})
#: Methods that constitute a data access (and therefore extend liveness).
_USE_METHODS = frozenset(
    {"GetValue", "SetValue", "GetPhyAddr", "GetSize", "GetLength", "Get", "operator[]"}
)
#: Factory calls that yield a fully specified local tensor.
_TENSOR_FACTORIES = frozenset({"GetLocalTensor", "CreateLocalTensor", "MakeLocalTensor"})
#: Buffer accessor methods returning a sub-tensor at a byte offset.
#: Byte-first buffer accessors: the single argument is a byte offset.
#: (``GetWithOffset`` is NOT here: its CANN signature is
#: ``(elementCount, byteOffset)`` and gets a dedicated branch.)
_BUFFER_GET_BYTE_METHODS = frozenset({"GetBufferByByte", "GetBufferAddr"})

_SET_FLAG_NAMES = frozenset({"SetFlag", "set_flag", "SetFlagImpl"})
_WAIT_FLAG_NAMES = frozenset({"WaitFlag", "wait_flag", "WaitFlagImpl"})
_BARRIER_NAMES = frozenset({"PipeBarrier", "pipe_barrier"})

_EVENT_ID_RE = re.compile(r"^EVENT_ID(?P<n>\d+)$")

_LOOP_NODE_TYPES = frozenset({"for_statement", "while_statement", "do_statement",
                              "for_range_loop"})


@dataclass
class VisitorOptions:
    """Knobs controlling how permissive the visitor is."""

    #: Analyze every function, not just those marked ``__global__``/``__aicore__``.
    analyze_all_functions: bool = False
    #: Assume this trip count for loops whose bound is not statically known.
    assumed_trip_count: int = 2
    #: Maximum operations to record per kernel, as a runaway guard.
    max_ops: int = 20000
    #: Loops with a statically known trip count at or below this limit are
    #: replayed once per iteration with the induction variable bound to each
    #: concrete value.  That is what lets ``p = t & 1`` and the
    #: ``p ? EVENT_ID1 : EVENT_ID0`` ping-pong selection fold to constants the
    #: pairing analysis can reason about.
    unroll_trip_limit: int = 8
    #: Concrete tiling-struct field values to bind, as ``{field: value}``.
    tiling_values: Dict[str, int] = field(default_factory=dict)
    #: Infer values for unresolved tiling-struct fields from the *role* each
    #: one plays at its call sites (see
    #: :meth:`ASTVisitor._infer_tiling_roles`).  Off by default: the values
    #: are inferred, and an inference must never be the reason a kernel is
    #: rejected.
    infer_tiling_roles: bool = False
    #: How many iterations adjacent to each loop boundary the three-phase
    #: peeling traversal may explicitly replay ("peeled head" / "peeled tail").
    peel_window: int = 4
    #: Largest modular period (``t & 1`` -> 2, ``t % 4`` -> 4, ...) for which
    #: a steady-state representative cycle is emitted.  A loop whose parity
    #: exceeds this stays symbolic rather than being mis-abstracted.
    max_steady_period: int = 4


@dataclass
class _PeeledPlan:
    """A three-phase (head / steady / tail) traversal schedule for one loop.

    *Phase A* - ``head`` - and *Phase C* - ``tail`` - replay iterations with
    the induction variable bound to concrete constants, as straight-line code.
    *Phase B* - ``steady`` - stands in for every elided bulk iteration with a
    minimal representative cycle of the loop's modular period; its bindings
    are concrete constants when the trip count is known and point-bounded
    variables (symbolic induction offsets) in the symbolic-trip-count fallback.
    """

    head: List[int]
    steady: List[object]
    tail: List[int]


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------


class ASTVisitor:
    """Builds an :class:`AnalysisUnit` from one prepared translation unit."""

    def __init__(
        self,
        prepared: PreparedSource,
        hardware: HardwareModel,
        diagnostics: DiagnosticCollector,
        options: Optional[VisitorOptions] = None,
    ) -> None:
        self.prepared = prepared
        self.hw = hardware
        self.diags = diagnostics
        self.opts = options or VisitorOptions()
        self._source_bytes = prepared.encoded
        self._lines = prepared.original.splitlines()
        self._parser = _make_parser()
        self._eval = ExpressionEvaluator(self._source_bytes)
        self._global_env = ConstEnv()
        #: Tiling fields bound by role inference, as ``{access text: value}``.
        self._inferred_tiling: Dict[str, int] = {}
        #: Small scalar helper functions (``event_t ev(int p)``) whose body is
        #: a single ``return <expr>;``, for constexpr-style event-id folding.
        #: Maps name -> (parameter names, returned expression node).
        self._helpers: Dict[str, Tuple[Tuple[str, ...], Node]] = {}
        self._helper_depth = 0
        self._eval.helper_resolver = self.fold_helper_call
        #: Every named function body in the unit, keyed by (owning class or
        #: ``None``, bare name) -> (parameter names, body node, definition).
        self._functions: Dict[
            Tuple[Optional[str], str], Tuple[Tuple[str, ...], Node, Node]
        ] = {}
        #: Keys whose class declares that name more than once (an overload set).
        self._ambiguous_functions: Set[Tuple[Optional[str], str]] = set()
        #: Bare name -> every key defining it, for unqualified resolution.
        self._definitions_by_name: Dict[str, List[Tuple[Optional[str], str]]] = {}
        #: Definition node -> its owning class, for resolving calls made inside it.
        self._owner_of: Dict[Node, Optional[str]] = {}
        #: Bare names that appear as a callee anywhere in the unit.
        self._called_functions: Set[str] = set()
        #: Every ``TBuf``/``TQue`` in the unit, by member name, with the byte
        #: size its ``InitBuffer`` reserves.  Unit-wide because the declaration,
        #: the sizing call and the ``Get<T>()`` live in different scopes.
        self._pipe_buffers: Dict[str, "PipeBuffer"] = {}

    # -- entry point --------------------------------------------------------

    def run(self) -> AnalysisUnit:
        tree = self._parser.parse(self._source_bytes)
        root = tree.root_node

        unit = AnalysisUnit(path=self.prepared.path, source=self.prepared.original)
        self._report_frontend_errors(unit)
        self._report_parse_errors(root, unit)
        self._seed_builtin_constants()
        self._seed_tiling_values()
        self._collect_global_constants(root)
        self._collect_helper_functions(root)
        self._collect_scalar_members(root)
        self._collect_pipe_buffers(root)
        if self.opts.infer_tiling_roles:
            # Between the two buffer phases: inference needs each
            # buffer's TPosition, and the sizing phase needs the values
            # inference supplies.
            self._infer_tiling_roles(root)
        self._size_pipe_buffers(root)
        self._collect_call_graph(root)
        unit.constants = dict(self._global_env.flatten())
        unit.suppressions = {
            ann.line: [
                token.strip().upper()
                for token in ann.body.replace(",", " ").split()
                if token.strip()
            ]
            for ann in self.prepared.annotations_of("ignore")
        }

        functions = self._find_functions(root)
        covered = self._inlined_elsewhere(functions)
        for func in functions:
            is_entry = self.prepared.has_kernel_attribute_before(func.start_byte)
            if not is_entry and not self.opts.analyze_all_functions:
                continue
            if func in covered and not self.opts.analyze_all_functions:
                # Reached by inlining it into the entry that calls it, where
                # its flags pair with their partners.  Walking it again on its
                # own would re-report that one handshake as two orphans.
                continue
            walker = _KernelWalker(self, func, is_entry)
            unit.kernels.append(walker.build())

        if not unit.kernels:
            hint = (
                "No function carried __global__ or __aicore__. Re-run with "
                "--all-functions to analyze every function in the file."
            )
            self.diags.add(
                Code.NO_KERNEL_FOUND,
                Severity.WARNING,
                f"no kernel entry point found in {self.prepared.path}",
                SourceLoc(file=self.prepared.path, line=1),
                remediation=hint,
            )
        # Whatever the heuristic had to invent is recorded on the unit, so a
        # reader can see exactly which fields a layout conclusion rests on.
        unit.inferred_tiling_bindings = dict(self._inferred_tiling)
        return unit

    # -- shared helpers used by the per-kernel walker -----------------------

    def text(self, node: Node) -> str:
        return self._source_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def loc(self, node: Node) -> SourceLoc:
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        # The parse basis may be macro-expanded text; every reported location
        # must land in the original file the user is looking at.
        line = self.prepared.origin_line(start_row + 1)
        end_line = self.prepared.origin_line(end_row + 1)
        return SourceLoc(
            file=self.prepared.path,
            line=line,
            column=start_col + 1,
            end_line=end_line,
            end_column=end_col + 1,
            snippet=self.line_text(line).strip()[:160],
        )

    def line_text(self, line: int) -> str:
        return self._lines[line - 1] if 1 <= line <= len(self._lines) else ""

    # -- file-level passes --------------------------------------------------

    def _report_frontend_errors(self, unit: AnalysisUnit) -> None:
        """Report a preprocessor fault rather than quietly analyzing less.

        When the token preprocessor cannot finish, the analysis continues on
        unexpanded source - a partial result beats no result - but every
        conclusion then rests on a translation unit the compiler would not
        recognise, so the reader has to be told.
        """
        for message in getattr(self.prepared, "frontend_errors", ()):
            self.diags.add(
                Code.PARSE_ERROR,
                Severity.WARNING,
                f"preprocessing was incomplete: {message}",
                SourceLoc(file=self.prepared.path, line=1),
                remediation=(
                    "Findings for this file may be incomplete. Check that its "
                    "includes are reachable and UTF-8 encoded."
                ),
            )

    def _report_parse_errors(self, root: Node, unit: AnalysisUnit) -> None:
        if not root.has_error:
            return
        unit.had_parse_errors = True
        reported = 0
        for node in _walk(root):
            if node.type != "ERROR" and not node.is_missing:
                continue
            loc = self.loc(node)
            unit.parse_error_locs.append(loc)
            reported += 1
            if reported <= 10:
                self.diags.add(
                    Code.PARSE_ERROR,
                    Severity.WARNING,
                    f"could not parse {self.text(node)[:60]!r}; "
                    "constructs in this region may be missed",
                    loc,
                    remediation="Check for unsupported syntax, or pre-expand the "
                    "macro covering this region before analysis.",
                )

    def _seed_builtin_constants(self) -> None:
        """Define the constants every kernel can rely on.

        ``EVENT_IDn`` are compile-time enum values on every DaVinci part, and
        without them the ``p ? EVENT_ID1 : EVENT_ID0`` ping-pong selection
        can never fold.  The object-like ``#define`` bodies collected by the
        macro expander are folded here in source order, so chains such as
        ``NV_A_BYTES = (NV_M * NV_K / 2)`` resolve from the header.
        """
        for event_id in range(self.hw.chip.max_event_id + 1):
            self._global_env.define(f"EVENT_ID{event_id}", event_id)
        for name, body in self.prepared.macro_object_defs:
            value = collect_define_value(self._eval, self._parser, body,
                                         self._global_env)
            if value is not None:
                self._global_env.define(name, value)

    def _collect_helper_functions(self, root: Node) -> None:
        """Index single-``return`` helper functions for constant folding.

        ``static __aicore__ inline event_t ev(int p) { return p ? EV1 : EV0; }``
        is a common spelling of the ping-pong event selector; the event-id
        resolver evaluates such bodies with concrete arguments instead of
        giving up on the call (which used to defeat pairing analysis).
        """
        for func in self._find_functions(root):
            body = func.child_by_field_name("body")
            if body is None:
                continue
            statements = [c for c in body.named_children if c.type != "comment"]
            if len(statements) != 1 or statements[0].type != "return_statement":
                continue
            # ``return <expr>;`` exposes the expression as a direct named
            # child (no field name in this grammar).
            value = next(
                (c for c in statements[0].named_children if c.type != "comment"),
                None,
            )
            if value is None:
                continue
            declarator = func.child_by_field_name("declarator")
            name = _KernelWalker._function_name(func)
            params = _parameter_names(declarator) if declarator is not None else []
            if not name or name in self._helpers:
                continue
            self._helpers[name] = (tuple(params), value)

    def fold_helper_call(self, node: Node, env: ConstEnv) -> Optional[int]:
        """Evaluate a call to a single-``return`` helper to a constant.

        Covers both the ``ev(p) { return p ? EV1 : EV0; }`` ping-pong selector
        and the arithmetic helpers kernels size their buffers with, such as
        ``AlignUpBytes(bytes)``.  Every argument must fold, the body is then
        evaluated with the parameters bound, and the result must be constant.
        Recursion is bounded by ``_helper_depth``: a helper that calls itself
        would otherwise recurse until the stack gives out.
        """
        if node.type != "call_expression":
            return None
        func = node.child_by_field_name("function")
        name = _callee_base_name(self, func)
        if name is None or name not in self._helpers:
            return None
        if self._helper_depth >= _MAX_HELPER_DEPTH:
            return None
        params, body = self._helpers[name]
        values: List[int] = []
        self._helper_depth += 1
        try:
            for arg in _argument_nodes(node):
                folded = self._eval.fold(arg, env)
                if folded is None:
                    return None
                values.append(folded)
            if len(values) != len(params):
                return None
            local = self._global_env.child()
            for param, value in zip(params, values):
                local.define(param, value)
            return self._eval.fold(body, local)
        finally:
            self._helper_depth -= 1

    def _seed_tiling_values(self) -> None:
        """Bind supplied tiling fields under every spelling kernels use.

        The field is reached as ``tilingData->rowFactor``, ``tiling->rowFactor``
        or ``tilingData.rowFactor`` depending on the operator, and the evaluator
        looks names up by their source text, so each receiver spelling is
        defined.  The bare field name is defined too, which covers the common
        ``GET_TILING_DATA`` style where the struct is unpacked into locals.
        """
        values = self.opts.tiling_values
        if not values:
            return
        for field_name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                continue  # floats are not integral layout inputs
            for receiver in _TILING_RECEIVERS:
                self._global_env.define(f"{receiver}->{field_name}", value)
                self._global_env.define(f"{receiver}.{field_name}", value)
            self._global_env.define(field_name, value)

    # -- tiling role inference ----------------------------------------------

    def _tiling_pointers(self, root: Node) -> Set[str]:
        """Variables that demonstrably hold a pointer to a tiling struct.

        Identified by the *cast that produces them*, not by their name::

            __gm__ MlaTilingData *tilingData =
                reinterpret_cast<__gm__ MlaTilingData *>(tiling);

        A declaration or assignment whose initialiser casts to a type matching
        :data:`_TILING_CAST_RE` binds its target.  The conventional
        spellings in :data:`_TILING_RECEIVERS` are accepted too, because the
        ``GET_TILING_DATA`` macro produces them with no visible cast.

        Restricting inference to this set is what keeps it away from locals,
        loop indices and macros: nothing outside it is ever given a value.
        """
        found: Set[str] = set(_TILING_RECEIVERS)
        for node in _walk(root):
            if node.type == "declaration":
                for declarator in node.children_by_field_name("declarator"):
                    if declarator.type != "init_declarator":
                        continue
                    name_node = _declarator_identifier(declarator)
                    value = declarator.child_by_field_name("value")
                    if name_node is None or value is None:
                        continue
                    if _TILING_CAST_RE.search(self.text(value)):
                        found.add(self.text(name_node))
            elif node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                if left is None or right is None:
                    continue
                if left.type == "identifier" and _TILING_CAST_RE.search(
                    self.text(right)
                ):
                    found.add(self.text(left))
        return found

    def _enclosing_call(self, node: Node) -> Optional[Tuple[str, int, Node]]:
        """``(callee, argument index, call node)`` for the call containing *node*.

        The index is the position of the top-level argument the node sits
        inside, so a member access buried in a nested aggregate initialiser is
        still attributed to the argument it belongs to.
        """
        parent = node.parent
        while parent is not None:
            if parent.type == "call_expression":
                for index, arg in enumerate(_argument_nodes(parent)):
                    # Compare against the access itself: walking up leaves the
                    # argument_list as the call's child, and its own start is
                    # the '(', which precedes every argument.
                    if arg.start_byte <= node.start_byte < arg.end_byte:
                        func = parent.child_by_field_name("function")
                        callee = (
                            _callee_base_name(self, func) if func is not None else ""
                        )
                        return callee, index, parent
                return None
            parent = parent.parent
        return None

    def _is_core_grid_use(self, node: Node) -> bool:
        """``True`` when the access is compared against the core index.

        ``if (tilingData->coreNum <= GetBlockIdx()) return;`` partitions work
        across cores.  Such a field counts cores, so giving it a tile extent
        would not resolve a layout - it would invent a wrong one.
        """
        parent = node.parent
        depth = 0
        while parent is not None and depth < 4:
            if parent.type in {"binary_expression", "conditional_expression"}:
                if _CORE_INDEX_RE.search(self.text(parent)):
                    return True
            parent = parent.parent
            depth += 1
        return False

    def _buffer_role_value(self, call: Node) -> int:
        """The minimal valid extent for the buffer an ``InitBuffer`` sizes.

        A Cube-resident buffer is addressed in 16-wide fractals, a
        vector-resident one in 64-element tiles.  The value is deliberately
        *minimal*: it only has to make the bump allocator in
        :meth:`_synthesize_tpipe_layout` yield concrete, distinct offsets so
        the offset-dependent checks can run at all.  A minimal extent cannot
        manufacture an SRAM overflow, which a guessed large one could.
        """
        args = _argument_nodes(call)
        if not args:
            return _VECTOR_MIN_TILE
        name = _leading_identifier(self.text(args[0]))
        buffer = self._pipe_buffers.get(name or "")
        if buffer is None:
            return _VECTOR_MIN_TILE
        domain = self.hw.domain_of(buffer.position)
        return _CUBE_MIN_FRACTAL if domain in _CUBE_DOMAINS else _VECTOR_MIN_TILE

    def _tiling_aliases(
        self, root: Node, pointers: Set[str]
    ) -> Dict[str, Set[str]]:
        """``{name: tiling accesses that flow into it}`` for single assignments.

        A name written more than once anywhere in the unit is excluded: it may
        hold either value where it is used, so attributing a role through it
        would be guesswork.
        """
        writes: Dict[str, List[Node]] = defaultdict(list)
        for node in _walk(root):
            if node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                operator = node.child_by_field_name("operator")
                if left is None or right is None:
                    continue
                if operator is not None and self.text(operator) != "=":
                    continue  # compound assignment depends on the prior value
                writes[_member_target_name(self.text(left)) or ""].append(right)
            elif node.type == "init_declarator":
                name_node = _declarator_identifier(node)
                value = node.child_by_field_name("value")
                if name_node is not None and value is not None:
                    writes[self.text(name_node)].append(value)

        aliases: Dict[str, Set[str]] = {}
        for name, values in writes.items():
            if not name or len(values) != 1:
                continue
            accesses = self._direct_tiling_accesses(values[0], pointers)
            if accesses:
                aliases[name] = accesses
        return aliases

    def _direct_tiling_accesses(self, node: Node, pointers: Set[str]) -> Set[str]:
        """Tiling-pointer member accesses appearing literally inside *node*."""
        found: Set[str] = set()
        for child in _walk(node):
            if child.type != "field_expression":
                continue
            receiver = child.child_by_field_name("argument")
            field = child.child_by_field_name("field")
            if receiver is None or field is None:
                continue
            if _leading_identifier(self.text(receiver)) in pointers:
                found.add(canonical_accessor(self.text(child)))
        return found

    def _feeding_accesses(
        self,
        node: Node,
        pointers: Set[str],
        aliases: Dict[str, Set[str]],
        depth: int = 3,
    ) -> Set[str]:
        """Tiling fields that flow into *node*, directly or through aliases.

        ``depth`` caps the hops so a chain of assignments cannot loop; three is
        more than the unpack-then-size pattern needs.
        """
        found = self._direct_tiling_accesses(node, pointers)
        if depth <= 0:
            return found
        seen: Set[str] = set()
        for child in _walk(node):
            if child.type == "identifier":
                name = self.text(child)
            elif child.type == "field_expression":
                name = _member_target_name(self.text(child)) or ""
            else:
                continue
            if not name or name in seen:
                continue
            seen.add(name)
            for access in aliases.get(name, ()):  # already canonical
                found.add(access)
        return found

    def _infer_tiling_roles(self, root: Node) -> None:
        """Bind unresolved tiling fields from the role they play at their uses.

        Every field reached through a verified tiling pointer is classified by
        the *context of its use*, never by its spelling:

        * the extent argument of ``InitBuffer``/``InitQueue`` - a buffer
          extent, bound to the architecture-minimal valid dimension.  This is
          the one that matters: an unresolved extent blocks the ``TPipe`` bump
          allocation for its whole domain, and every tensor the allocator
          hands out afterwards loses its offset, which silences the
          bank-conflict check (AKA3006) along with the rest.
        * the parameter block of ``DataCopy``/``DataCopyPad``, or any
          ``*Params`` aggregate - a DMA stride or gap, bound to the 32-byte
          DaVinci block.
        * a comparison against ``GetBlockIdx()`` - a core-grid bound, left
          symbolic on purpose.

        The walk runs from the call site *backwards*, through names assigned
        exactly once, because the extent is usually a member unpacked in a
        different method than the one that sizes the buffer.

        A field used in two roles that disagree is left symbolic: no single
        value is right for both, and picking one would invent a layout.
        """
        pointers = self._tiling_pointers(root)
        aliases = self._tiling_aliases(root, pointers)
        roles: Dict[str, Set[Tuple[str, Optional[int]]]] = defaultdict(set)

        # The core-grid veto first: a field compared against the core index is
        # a core count, whatever else it is used for.
        for node in _walk(root):
            if node.type != "field_expression":
                continue
            receiver = node.child_by_field_name("argument")
            if receiver is None:
                continue
            if _leading_identifier(self.text(receiver)) not in pointers:
                continue
            if self._is_core_grid_use(node):
                roles[canonical_accessor(self.text(node))].add(("core_grid", None))

        block_bytes = self.hw.chip.block_bytes
        for node in _walk(root):
            if node.type != "call_expression":
                continue
            func = node.child_by_field_name("function")
            callee = _callee_base_name(self, func) if func is not None else ""
            args = _argument_nodes(node)
            if not args:
                continue
            if callee in {"InitBuffer", "InitQueue"} and len(args) >= 2:
                value = self._buffer_role_value(node)
                # The extent is the last argument for both the TBuf form and
                # the TQue form; the depth argument is never a tiling extent.
                for access in self._feeding_accesses(args[-1], pointers, aliases):
                    roles[access].add(("buffer", value))
            elif callee in _DMA_APIS and len(args) > _DMA_PARAMS_INDEX:
                for access in self._feeding_accesses(
                    args[_DMA_PARAMS_INDEX], pointers, aliases
                ):
                    roles[access].add(("dma_stride", block_bytes))
            elif _DMA_PARAMS_RE.search(self.text(node)):
                for access in self._feeding_accesses(node, pointers, aliases):
                    roles[access].add(("dma_stride", block_bytes))

        for access, observed in sorted(roles.items()):
            kinds = {kind for kind, _ in observed}
            values = {value for _, value in observed if value is not None}
            if "core_grid" in kinds or len(values) != 1:
                continue  # ambiguous, or deliberately left symbolic
            # An exact binding - from a manifest, or from the source itself -
            # always wins; inference only ever fills a gap.
            if self._global_env.lookup(access) is not None:
                continue
            # A manifest binds a field under the conventional receiver
            # spellings and under its bare name, but a kernel may reach it
            # through any pointer it likes (``t->tileBytes``).  Re-binding the
            # supplied value under this spelling is what stops an inference
            # from shadowing a measured number.
            field_name = access.rsplit("->", 1)[-1].rsplit(".", 1)[-1]
            supplied = self._global_env.lookup(field_name)
            if supplied is not None:
                self._global_env.define(access, supplied)
                continue
            value = values.pop()
            self._global_env.define(access, value)
            self._inferred_tiling[access] = value

    def _local_env_before(self, call: Node) -> ConstEnv:
        """Globals plus the foldable locals declared before *call* in its scope.

        A buffer's extent is usually computed into a local of the method that
        sizes it, so folding the ``InitBuffer`` argument needs those locals.
        Only declarations lexically before the call are bound, and only when
        they fold, so nothing is read out of order.
        """
        env = self._global_env.child()
        function = call
        while function is not None and function.type not in {
            "function_definition", "lambda_expression"
        }:
            function = function.parent
        if function is None:
            return env
        for node in _walk(function):
            if node.start_byte >= call.start_byte:
                break
            if node.type != "declaration":
                continue
            for declarator in node.children_by_field_name("declarator"):
                if declarator.type != "init_declarator":
                    continue
                name_node = _declarator_identifier(declarator)
                value = declarator.child_by_field_name("value")
                if name_node is None or value is None:
                    continue
                folded = self._eval.fold(value, env)
                if folded is not None:
                    env.define(self.text(name_node), folded)
        return env

    def _collect_scalar_members(self, root: Node) -> None:
        """Fold scalar members that the unit assigns exactly once.

        A kernel class unpacks its tiling struct in one method and sizes its
        buffers in another: ``hidden_ = tilingData.hiddenSize;`` in ``Init()``,
        ``InitBuffer(xBitsBuf_, AlignUpBytes(hidden_ * 2))`` in
        ``InitBuffers()``.  Without carrying the member across that boundary a
        supplied tiling configuration binds the struct field and still leaves
        every dependent extent symbolic.

        Only members assigned *once* in the whole unit are bound, and only when
        the right-hand side already folds.  A member written twice may hold
        either value at the point of use, so pinning one would invent a layout.
        """
        assignments: Dict[str, List[Node]] = defaultdict(list)
        for node in _walk(root):
            if node.type != "assignment_expression":
                continue
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is None or right is None:
                continue
            operator = node.child_by_field_name("operator")
            if operator is not None and self.text(operator) != "=":
                continue  # compound assignment depends on the prior value
            name = _member_target_name(self.text(left))
            if name is None:
                continue
            assignments[name].append(right)

        for name, writes in assignments.items():
            if len(writes) != 1:
                continue
            if self._global_env.lookup(name) is not None:
                continue  # a real constant of that name already won
            folded = self._eval.fold(writes[0], self._global_env)
            if folded is not None:
                self._global_env.define(name, folded)

    def _collect_pipe_buffers(self, root: Node) -> None:
        """Index every ``TBuf``/``TQue`` in the unit and its ``InitBuffer`` size.

        Production kernels encapsulate the buffer manager: the ``TBuf``/``TQue``
        members are declared in a class body, ``pipe_->InitBuffer(...)`` runs in
        ``Init()``, and ``buf.Get<T>()`` runs in ``Process()`` - three different
        scopes.  A per-function walker therefore never sees a buffer and its
        size together, which is why every queued tensor used to come out with no
        domain (AKA1009) and no offset, disabling the whole memory and
        bank-conflict analysis.  Collecting the declarations and the sizes once
        per translation unit lets each function's walker resolve them.
        """
        ambiguous: Set[str] = set()
        for node in _walk(root):
            if node.type not in {"declaration", "field_declaration"}:
                continue
            type_node = node.child_by_field_name("type")
            type_text = self.text(type_node) if type_node is not None else ""
            decl_text = self.text(node)
            # AST template-argument parse first (named depth constants and
            # multi-line declarations survive it); textual regex fallback.
            info = _buffer_template_info(self, type_node)
            if info is not None:
                position, depth = info
                if position is None:
                    continue
                if depth is None:
                    depth = 1  # InitBuffer(que, num, len) overrides this
            else:
                match = _TBUF_TYPE_RE.search(type_text) or _TBUF_TYPE_RE.search(decl_text)
                if match is None:
                    continue
                position = TPosition.parse(match.group("pos"))
                if position is None:
                    continue
                depth = _queue_depth(type_text or decl_text)
            for declarator in node.children_by_field_name("declarator"):
                name_node = _declarator_identifier(declarator)
                if name_node is None:
                    continue
                name = self.text(name_node)
                prior = self._pipe_buffers.get(name)
                if prior is not None and (
                    prior.position is not position or prior.depth != depth
                ):
                    # Two classes in one unit using the same member name for
                    # different buffers; binding either way could invent a
                    # layout, so neither is trusted.
                    ambiguous.add(name)
                    continue
                self._pipe_buffers[name] = PipeBuffer(name, position, depth)

        for name in ambiguous:
            self._pipe_buffers.pop(name, None)

    def _size_pipe_buffers(self, root: Node) -> None:
        """Evaluate every ``InitBuffer``/``InitQueue`` extent.

        Phase two of the index.  It runs *after* tiling role inference,
        because an extent is routinely ``InitBuffer(buf, t->tileBytes)``:
        until that field has a value the call cannot be folded, the buffer
        never gets an allocation order, and the bump allocator in
        :meth:`_synthesize_tpipe_layout` blocks its whole domain.
        """
        conflicting: Set[str] = set()
        next_order = 0
        for node in _walk(root):
            if node.type != "call_expression":
                continue
            func = node.child_by_field_name("function")
            if func is None:
                continue
            if _callee_base_name(self, func) not in {"InitBuffer", "InitQueue"}:
                continue
            args = _argument_nodes(node)
            if len(args) < 2:
                continue
            name = _leading_identifier(self.text(args[0]))
            buf = self._pipe_buffers.get(name or "")
            if buf is None:
                continue
            # The extent is routinely a local of the sizing method
            # (``const int64_t xFloatBytes = AlignUpBytes(...)``), so the call
            # is evaluated in that method's scope, not the bare global one.
            env = self._local_env_before(node)
            # InitBuffer(buf, len) for a TBuf; InitBuffer(que, num, len) for a
            # TQue - the block length is always the last argument.
            size = self._eval.evaluate(args[-1], env)
            if size is None:
                continue
            depth = buf.depth
            if len(args) >= 3:
                num = to_int(self._eval.evaluate(args[1], env))
                if num is not None and num > 0:
                    depth = num
            if buf.order is not None:
                # Sized twice in one unit.  Same extent: harmless repetition.
                # Different extent: two kernels reusing one member name, so no
                # unit-wide size is correct and only the function-local
                # InitBuffer may size it.
                if to_int(size) != to_int(buf.block_bytes) or depth != buf.depth:
                    conflicting.add(name)
                continue
            self._pipe_buffers[name] = PipeBuffer(
                buf.name, buf.position, depth, size, order=next_order
            )
            next_order += 1

        for name in conflicting:
            entry = self._pipe_buffers.get(name)
            if entry is not None:
                # Keep the TPosition (consistent across the declarations) but
                # drop the size, so the layout comes from the owning function.
                self._pipe_buffers[name] = PipeBuffer(entry.name, entry.position)

    def _collect_global_constants(self, root: Node) -> None:
        """Fold file-scope ``constexpr``, ``#define`` and ``enum`` constants."""
        for node in self._walk_skipping_bodies(root):
            if node.type == "preproc_def":
                self._collect_define(node, self._global_env)
            elif node.type == "declaration":
                self._collect_constexpr(node, self._global_env)
            elif node.type == "enum_specifier":
                self._collect_enum(node, self._global_env)

    def _walk_skipping_bodies(self, root: Node) -> Iterator[Node]:
        """Pre-order walk that does not descend into function bodies."""
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            if node.type == "function_definition":
                # Still visit the signature, just not the body.
                body = node.child_by_field_name("body")
                stack.extend(
                    reversed([c for c in node.children if c is not body])
                )
                continue
            stack.extend(reversed(node.children))

    def _collect_define(self, node: Node, env: ConstEnv) -> None:
        name_node = node.child_by_field_name("name")
        value_node = node.child_by_field_name("value")
        if name_node is None or value_node is None:
            return
        value = collect_define_value(
            self._eval, self._parser, self.text(value_node), env
        )
        if value is not None:
            env.define(self.text(name_node), value)

    def _collect_constexpr(self, node: Node, env: ConstEnv) -> None:
        """Record ``constexpr``/``const`` integer scalars into ``env``."""
        decl_text = self.text(node)
        if "constexpr" not in decl_text and "const " not in decl_text:
            return
        type_node = node.child_by_field_name("type")
        type_text = self.text(type_node) if type_node is not None else ""
        if _TENSOR_TYPE_RE.search(type_text) or "*" in decl_text.split("=")[0]:
            return
        for declarator in node.children_by_field_name("declarator"):
            if declarator.type != "init_declarator":
                continue
            name_node = declarator.child_by_field_name("declarator")
            value_node = declarator.child_by_field_name("value")
            if name_node is None or value_node is None:
                continue
            if name_node.type != "identifier":
                continue
            folded = self._eval.fold(value_node, env)
            if folded is not None:
                env.define(self.text(name_node), folded)

    def _collect_enum(self, node: Node, env: ConstEnv) -> None:
        body = node.child_by_field_name("body")
        if body is None:
            return
        next_value = 0
        for enumerator in body.named_children:
            if enumerator.type != "enumerator":
                continue
            name_node = enumerator.child_by_field_name("name")
            value_node = enumerator.child_by_field_name("value")
            if name_node is None:
                continue
            if value_node is not None:
                folded = self._eval.fold(value_node, env)
                if folded is not None:
                    next_value = folded
            env.define(self.text(name_node), next_value)
            next_value += 1

    def _find_functions(self, root: Node) -> List[Node]:
        return [n for n in _walk(root) if n.type == "function_definition"]

    def resolve_callee(
        self, name: str, caller_owner: Optional[str]
    ) -> Optional[Tuple[Tuple[str, ...], Node, Node]]:
        """Find the body an unqualified call names, or ``None`` if unclear.

        A method calling ``AllocEventID()`` means *its own* class's method, so
        the caller's class is tried first.  Failing that, a name defined
        exactly once in the unit resolves to that definition.  A name defined
        by several classes with no class context is left alone: inlining the
        wrong sibling would trace code that never runs here.
        """
        for key in ((caller_owner, name), (None, name)):
            if key in self._ambiguous_functions:
                return None
            entry = self._functions.get(key)
            if entry is not None:
                return entry
        keys = self._definitions_by_name.get(name) or []
        if len(keys) == 1 and keys[0] not in self._ambiguous_functions:
            return self._functions.get(keys[0])
        return None

    def _inlined_elsewhere(self, functions: Sequence[Node]) -> Set[Node]:
        """Functions covered by inlining, so not analyzed in their own right.

        A function qualifies when it is called inside this unit, is not itself
        a ``__global__`` launch entry, and resolves unambiguously by name.  The
        set is only applied when the unit has at least one launch entry to
        inline it *into*: a header or fixture whose ``__aicore__`` helpers are
        the only things present must still be analyzed directly, or the file
        would report nothing at all.
        """
        launches = [
            f for f in functions
            if self.prepared.has_launch_attribute_before(f.start_byte)
        ]
        if not launches:
            return set()
        covered: Set[Node] = set()
        for func in functions:
            if func in launches:
                continue
            name = _bare_function_name(_KernelWalker._function_name(func))
            if not name or name not in self._called_functions:
                continue
            owner = self._owner_of.get(func)
            if (owner, name) in self._ambiguous_functions:
                continue
            # Only suppress a definition some call site actually resolves to;
            # otherwise it would be dropped from the report without ever being
            # inlined anywhere.
            if self.resolve_callee(name, owner) is not self._functions.get((owner, name)):
                continue
            covered.add(func)
        return covered

    def _collect_call_graph(self, root: Node) -> None:
        """Index the unit's functions and record which of them get called.

        Kernels are written as a ``__global__`` entry that delegates to
        ``__aicore__ inline`` helpers and member methods: the entry calls
        ``AllocEventID()`` to prime its flags, then ``Process()`` to consume
        them.  Analyzing each of those bodies on its own sees a ``SetFlag``
        with no ``WaitFlag`` in one function and the mirror image in another,
        and reports both as fatal - one real handshake, counted twice as two
        orphans.  Indexing the definitions lets the walker inline a callee into
        its caller's trace, where the pair is visible.
        """
        for func in self._find_functions(root):
            body = func.child_by_field_name("body")
            if body is None:
                continue
            # Indexed by (owning class, bare name): a method is defined as
            # ``Service<T>::AllocEventID`` but called as ``AllocEventID``, and
            # a sibling class usually defines the same method name.
            name = _bare_function_name(_KernelWalker._function_name(func))
            if not name or name == "<anonymous>":
                continue
            declarator = func.child_by_field_name("declarator")
            params = tuple(_parameter_names(declarator)) if declarator else ()
            owner = _enclosing_class_name(self, func)
            entry = (params, body, func)
            key = (owner, name)
            # The same class declaring one name twice is an overload set that
            # cannot be told apart by name, so neither body is inlined.
            if key in self._functions:
                self._ambiguous_functions.add(key)
                continue
            self._functions[key] = entry
            self._owner_of[func] = owner
            self._definitions_by_name.setdefault(name, []).append(key)

        for node in _walk(root):
            if node.type != "call_expression":
                continue
            base = _callee_base_name(self, node.child_by_field_name("function"))
            if base is not None:
                self._called_functions.add(base)

    def _owner_class_names(self) -> FrozenSet[str]:
        """Classes that own at least one indexed method in this unit.

        Used to recognise class-typed locals (``CubeStage cube;``) so member
        calls through a known receiver resolve to that class's method body
        rather than dying on a same-named sibling method.
        """
        try:
            return self._owner_class_names_cache
        except AttributeError:
            cache = frozenset(
                owner for (owner, _name) in self._functions if owner is not None
            )
            self._owner_class_names_cache = cache
            return cache


# ---------------------------------------------------------------------------
# Per-kernel walker
# ---------------------------------------------------------------------------


class _KernelWalker:
    """Builds one :class:`KernelIR` by walking a single function body."""

    def __init__(self, visitor: ASTVisitor, func: Node, is_entry: bool) -> None:
        self.v = visitor
        self.func = func
        self.is_entry = is_entry
        self.ir = KernelIR(
            name=self._function_name(func),
            loc=visitor.loc(func),
            is_kernel_entry=is_entry,
        )
        self._index = 0
        self._next_scope_id = 0
        self._next_loop_id = 0
        self._scope_stack: List[Scope] = []
        self._loop_stack: List[LoopInfo] = []
        self._env_stack: List[ConstEnv] = [visitor._global_env.child()]
        self._conditional_depth = 0
        self._buffer_positions: Dict[str, TPosition] = {}
        #: Byte sizes recorded by ``TPipe::InitBuffer``.
        self._buffer_sizes: Dict[str, Expr] = {}
        #: LocalTensor name -> the TBuf it was obtained from via ``.Get()``.
        self._tensor_buffers: Dict[str, str] = {}
        #: Local variable name -> the stage CLASS it was declared as
        #: (``CubeStage cube;``).  Member calls through that receiver resolve
        #: against that class's methods first, which is what disambiguates
        #: ``cube.Init(...)`` from ``vec.Init(...)`` in a MIX_AIC_1_1 entry
        #: that instantiates two stage classes with colliding method names.
        self._var_classes: Dict[str, str] = {}
        #: Per-block length of each queue, which is what one ``AllocTensor``
        #: hands out (``_buffer_sizes`` holds the whole reserved extent).
        self._buffer_blocks: Dict[str, Expr] = {}
        #: Callees currently being inlined, innermost last; the recursion guard.
        self._inline_stack: List[str] = []
        #: Core whose binary the region being walked belongs to.
        self._core_view_current: CoreView = CoreView.BOTH
        #: Class owning the function being walked, so an unqualified call
        #: resolves to this class's method rather than a sibling's.
        self._owner_class: Optional[str] = visitor._owner_of.get(func)
        if self._owner_class is None:
            self._owner_class = _enclosing_class_name(visitor, func)
        # Seed from the unit-wide registry so a ``Get<T>()`` in ``Process()``
        # resolves against a buffer declared in the class body and sized in
        # ``Init()``.  A same-function InitBuffer still overwrites these.
        for buf in visitor._pipe_buffers.values():
            self._buffer_positions[buf.name] = buf.position
            total = buf.total_bytes
            if total is not None:
                self._buffer_sizes[buf.name] = total
                self._buffer_blocks[buf.name] = buf.block_bytes
                folded = to_int(total)
                if folded is not None:
                    self.ir.buffer_sizes[buf.name] = folded
        self._truncated = False

    # -- public -------------------------------------------------------------

    def build(self) -> KernelIR:
        body = self.func.child_by_field_name("body")
        self._register_parameters()
        if body is not None:
            self._visit_block(body, ScopeKind.KERNEL)
        self._apply_annotations()
        self._finalize_tensors()
        self.ir.constants = dict(self._env.flatten())
        return self.ir

    # -- state accessors ----------------------------------------------------

    @property
    def _env(self) -> ConstEnv:
        return self._env_stack[-1]

    @property
    def _scope_id(self) -> int:
        return self._scope_stack[-1].id if self._scope_stack else 0

    @property
    def _loop_id(self) -> Optional[int]:
        return self._loop_stack[-1].id if self._loop_stack else None

    def _text(self, node: Node) -> str:
        return self.v.text(node)

    def _loc(self, node: Node) -> SourceLoc:
        return self.v.loc(node)

    # -- scope and op bookkeeping ------------------------------------------

    def _push_scope(self, node: Node, kind: ScopeKind) -> Scope:
        scope = Scope(
            id=self._next_scope_id,
            kind=kind,
            parent=self._scope_stack[-1].id if self._scope_stack else None,
            loc=self._loc(node),
            start_index=self._index,
            end_index=self._index,
        )
        self._next_scope_id += 1
        self._scope_stack.append(scope)
        self.ir.scopes[scope.id] = scope
        self._env_stack.append(self._env.child())
        return scope

    def _pop_scope(self) -> None:
        scope = self._scope_stack.pop()
        scope.end_index = max(self._index - 1, scope.start_index)
        self._env_stack.pop()

    def _emit(self, op: Operation) -> Optional[Operation]:
        if self._index >= self.v.opts.max_ops:
            if not self._truncated:
                self._truncated = True
                self.v.diags.add(
                    Code.ANALYSIS_LIMIT,
                    Severity.WARNING,
                    f"kernel {self.ir.name!r} exceeded {self.v.opts.max_ops} "
                    "tracked operations; the trace was truncated",
                    op.loc,
                    remediation="Raise --max-ops, or analyze a smaller kernel.",
                )
            return None
        self.ir.ops.append(op)
        self._index += 1
        return op

    def _next_op_args(self, node: Node) -> Dict[str, object]:
        """Common constructor arguments for an operation at ``node``."""
        return {
            "index": self._index,
            "loc": self._loc(node),
            "scope_id": self._scope_id,
            "loop_id": self._loop_id,
            "conditional": self._conditional_depth > 0,
            "core_view": self._core_view_current,
        }

    # -- parameters ---------------------------------------------------------

    def _register_parameters(self) -> None:
        """Register ``__gm__`` pointer parameters as global-memory tensors."""
        declarator = self.func.child_by_field_name("declarator")
        if declarator is None:
            return
        params = declarator.child_by_field_name("parameters")
        if params is None:
            return
        for param in params.named_children:
            if param.type != "parameter_declaration":
                continue
            name_node = _declarator_identifier(param.child_by_field_name("declarator"))
            if name_node is None:
                continue
            name = self._text(name_node)
            type_node = param.child_by_field_name("type")
            type_text = self._text(type_node) if type_node is not None else ""
            domain = self.v.prepared.address_space_for_range(
                param.start_byte, param.end_byte
            )
            is_gm_addr = "GM_ADDR" in type_text or domain is PhysicalDomain.GM
            if not is_gm_addr:
                continue
            self.ir.tensors[name] = TensorDecl(
                name=name,
                loc=self._loc(param),
                position=TPosition.GM,
                domain=PhysicalDomain.GM,
                dtype=_template_dtype(type_text),
                scope_id=0,
                origin="kernel parameter",
            )

    # -- statement dispatch -------------------------------------------------

    def _visit_block(self, node: Node, kind: ScopeKind) -> None:
        self._push_scope(node, kind)
        for child in node.named_children:
            if child.type == "comment":
                continue
            self._visit_statement(child)
        self._pop_scope()

    def _visit_statement(self, node: Node) -> None:
        kind = node.type

        if kind == "compound_statement":
            self._visit_block(node, ScopeKind.BLOCK)
            return
        if kind == "declaration":
            self._handle_declaration(node)
            return
        if kind == "expression_statement":
            self._scan_for_ops(node)
            return
        if kind in _LOOP_NODE_TYPES:
            self._handle_loop(node)
            return
        if kind == "if_statement":
            self._handle_if(node)
            return
        if kind == "switch_statement":
            self._handle_switch(node)
            return
        if kind in {"comment", "preproc_include"}:
            return
        if kind == "preproc_def":
            self.v._collect_define(node, self._env)
            return
        if kind in {"preproc_if", "preproc_ifdef", "preproc_else", "preproc_elif"}:
            self._handle_preproc_conditional(node)
            return

        # Anything else (return, labelled statements, try blocks, ...) is
        # scanned for operations and then walked structurally.
        self._scan_for_ops(node)

    def _handle_preproc_conditional(self, node: Node) -> None:
        """Walk both arms of a ``#if``, tracking which core each selects.

        ``#ifdef __DAV_C220_CUBE__`` / ``#ifdef __DAV_C220_VEC__`` is how a mix
        kernel splits itself between the two cores.  Both arms are still walked
        - the trace covers the whole file - but the operations in each are
        tagged with the core they are compiled into, so the event spaces stay
        separate instead of being merged into one.
        """
        selected = _core_view_of_ifdef(self.v, node)
        # Children of a preproc conditional are flat: the guarded statements
        # first, then the `preproc_else`/`preproc_elif` node holding the rest.
        # The else arm is the opposite core, but only when the guard selected a
        # core at all: for any other `#ifdef` both arms stay resident on both,
        # since the complement of "both cores" is "neither" and would silently
        # drop the else arm of every unrelated conditional.
        otherwise = selected.complement if selected is not CoreView.BOTH else CoreView.BOTH
        self._conditional_depth += 1
        try:
            for child in node.named_children:
                if child.type in {"preproc_else", "preproc_elif"}:
                    with self._core_view(otherwise):
                        self._visit_statement(child)
                    continue
                with self._core_view(selected):
                    self._visit_statement(child)
        finally:
            self._conditional_depth -= 1

    @contextmanager
    def _core_view(self, view: CoreView):
        """Narrow the active core view for a region, then restore it.

        Views intersect, so a core guard nested inside the opposite guard marks
        its region ``NONE``: it is compiled into neither binary, and nothing in
        it can pair with anything.
        """
        if view is CoreView.BOTH:
            yield
            return
        previous = self._core_view_current
        self._core_view_current = previous.intersect(view)
        try:
            yield
        finally:
            self._core_view_current = previous

    def _handle_loop(self, node: Node) -> None:
        # ``do { ... } while (0)`` is the stage-macro idiom, not a loop: the
        # body executes exactly once.  Treating it as a loop would fragment
        # every synchronisation channel across bogus per-stage regions.
        if node.type == "do_statement" and self._is_zero_trip_do(node):
            condition = node.child_by_field_name("condition")
            if condition is not None:
                self._scan_for_ops(condition)
            body = node.child_by_field_name("body") or _last_statement_child(node)
            if body is not None:
                self._visit_statement(body)
            return

        loop = LoopInfo(
            id=self._next_loop_id,
            scope_id=self._scope_id,
            loc=self._loc(node),
            parent=self._loop_id,
            start_index=self._index,
            header=self._loop_header_text(node),
        )
        self._next_loop_id += 1
        self.ir.loops[loop.id] = loop

        body = node.child_by_field_name("body") or _last_statement_child(node)
        self._push_scope(node, ScopeKind.LOOP)
        self._parse_loop_header(node, loop)

        unrolled = self._maybe_unroll(loop, body)
        plan = None if unrolled else self._plan_peeling(loop, body)
        if unrolled:
            # Replay the body once per iteration with the induction variable
            # bound to each concrete value.  The emitted operations carry no
            # loop id: an unrolled body is straight-line code and the trace
            # already captures every iteration exactly.
            step = loop.step or 1
            start = loop.start if loop.start is not None else 0
            for iteration in range(loop.trip_count or 0):
                self._env_stack.append(self._env.child())
                self._env.define(loop.induction_var, start + iteration * step)
                self._visit_loop_body(body)
                self._env_stack.pop()
        elif plan is not None:
            # Three-phase traversal: a peeled straight-line head, a cyclic
            # steady-state representative cycle (operations carry this loop's
            # id, so the marked graph closes them with one-token back edges),
            # and a peeled straight-line tail.
            loop.peeled = True
            loop.steady_first = (
                plan.steady[0] if isinstance(plan.steady[0], int) else None
            )
            loop.steady_reps = len(plan.steady)
            loop.peeled_head = len(plan.head)
            loop.peeled_tail = len(plan.tail)
            self._execute_peeled(loop, body, plan)
        else:
            self._loop_stack.append(loop)
            self._visit_loop_body(body)
            self._loop_stack.pop()

        loop.end_index = max(self._index - 1, loop.start_index)
        self._pop_scope()

    def _is_zero_trip_do(self, node: Node) -> bool:
        """``True`` for ``do { ... } while (0)`` and constant-false tails."""
        condition = node.child_by_field_name("condition")
        if condition is None:
            return False
        folded = self.v._eval.fold(condition, self._env)
        return folded is not None and folded == 0

    def _visit_loop_body(self, body: Optional[Node]) -> None:
        if body is None:
            return
        if body.type == "compound_statement":
            for child in body.named_children:
                if child.type != "comment":
                    self._visit_statement(child)
        else:
            self._visit_statement(body)

    def _maybe_unroll(self, loop: LoopInfo, body: Optional[Node]) -> bool:
        """Decide whether to replay this loop with concrete induction values.

        Only loops whose header folded completely (start, positive step and a
        trip count at or below :attr:`VisitorOptions.unroll_trip_limit`) are
        unrolled, and only when the body actually references the induction
        variable - otherwise every replay would emit an identical trace and
        the loop-level pairing checks would lose their per-iteration meaning
        for nothing.
        """
        if not loop.induction_var or loop.trip_count is None:
            return False
        if not (1 <= loop.trip_count <= self.v.opts.unroll_trip_limit):
            return False
        if loop.step is None or loop.step <= 0 or loop.start is None:
            return False
        if body is None:
            return False
        if not re.search(rf"\b{re.escape(loop.induction_var)}\b", self._text(body)):
            return False
        loop.unrolled = True
        return True

    # -- three-phase loop peeling -------------------------------------------

    def _plan_peeling(self, loop: LoopInfo, body: Optional[Node]) -> Optional[_PeeledPlan]:
        """Plan a head / steady-state / tail traversal for a long loop.

        Full unrolling explodes past ``unroll_trip_limit`` iterations, but a
        single symbolic body pass is blind to the transient pipeline states at
        the loop boundaries: ``if (t >= 1)`` prologue guards, ``if (t + 2 < T)``
        epilogue guards and ``p = t & 1`` ping-pong parity all refuse to fold,
        so the checkers see phantom operations from dead branches and symbolic
        event ids instead of real pairings.

        The plan replaces the bulk iterations with a minimal representative
        cycle whose length is the loop's modular period (two iterations for a
        ``t & 1`` double buffer), chosen where every prologue condition has
        settled true and every epilogue condition has settled false, and
        replays only the boundary iterations explicitly.  A loop with no
        induction-dependent conditions and no parity has nothing to peel, and
        keeps the exact single-pass treatment.
        """
        var = loop.induction_var
        if not var or body is None or loop.start is None:
            return None
        step = loop.step or 1
        if step <= 0 or loop.trip_count == 0:
            return None
        start = loop.start
        if not re.search(rf"\b{re.escape(var)}\b", self._text(body)):
            return None

        switches, period = self._critical_points(loop, body)
        if period > self.v.opts.max_steady_period:
            return None
        if not switches and period <= 1:
            return None

        window = self.v.opts.peel_window

        def snap(value: int) -> int:
            """First induction value on the loop's grid at or after *value*."""
            return start + (-(-(value - start) // step)) * step

        switches = {snap(v) for v in switches}

        if loop.trip_count is None:
            # Symbolic trip count (spec: fallback).  Phase A runs with the
            # conservative concrete bounds that fold without knowing T; the
            # steady state is projected at symbolic induction offsets, so the
            # trace keeps loop-carried edges without inventing facts about how
            # many iterations actually execute.
            head_end = max(
                (v for v in switches if start <= v <= start + window * step),
                default=start,
            )
            head = list(range(start, head_end, step))
            steady = [
                Var(var, lower=value, upper=value)
                for value in range(head_end, head_end + period * step, step)
            ]
            return _PeeledPlan(head=head, steady=steady, tail=[])

        end = start + (loop.trip_count - 1) * step
        reachable = {v for v in switches if start <= v <= end}
        early = {v for v in reachable if (v - start) // step <= window}
        late = {
            v for v in reachable if (end - v) // step <= window and v not in early
        }
        if reachable - early - late:
            # A condition flips mid-loop: the prologue/steady/epilogue model
            # does not apply, so keep the exact symbolic treatment.
            return None

        steady_start = max(early) if early else start
        tail_start = min(late) if late else end + step
        if steady_start + (period - 1) * step >= tail_start:
            return None  # no room for a whole representative cycle
        head = list(range(start, steady_start, step))
        tail = list(range(tail_start, end + step, step))
        if len(head) > window or len(tail) > window:
            return None
        steady = list(range(steady_start, steady_start + period * step, step))
        return _PeeledPlan(head=head, steady=steady, tail=tail)

    def _critical_points(self, loop: LoopInfo, body: Node) -> Tuple[set, int]:
        """Boundary switch points and modular period of a loop body.

        *Switch points* are induction values at which some guard condition
        changes truth value (``t >= C`` -> ``C``; ``t + k < T`` -> ``T - k``).
        The *period* is the least common multiple of the ``t % m`` / ``t & m``
        modular expressions in the body - 2 for standard ping-pong parity.
        """
        var = loop.induction_var or ""
        switches: set = set()
        for node in _walk(body):
            if node.type not in {"if_statement", "conditional_expression"}:
                continue
            condition = node.child_by_field_name("condition")
            if condition is None:
                continue
            found = self._switch_values_of(condition, var)
            if found:
                switches.update(found)
        return switches, self._steady_period(body, var)

    def _switch_values_of(self, condition: Node, var: str) -> Optional[set]:
        """Induction values where a linear comparison over *var* flips.

        Returns ``None`` for conditions that are constant, reference further
        unknown variables (an unresolved trip count), or are not linear in the
        induction variable - those are either already folded by ``_handle_if``
        or belong to the period scan, not to boundary extraction.
        """
        node = condition
        while node.type in {"parenthesized_expression", "condition_clause"}:
            inner = next((c for c in node.named_children if c.type != "comment"), None)
            if inner is None or inner.type in {"(", ")"}:
                return None
            node = inner
        if node.type != "binary_expression":
            return None
        op_node = node.child_by_field_name("operator")
        op = self._text(op_node) if op_node is not None else ""
        if op not in {"<", "<=", ">", ">=", "==", "!="}:
            return None
        left = self.v._eval.evaluate(node.child_by_field_name("left"), self._env)
        right = self.v._eval.evaluate(node.child_by_field_name("right"), self._env)
        if left is None or right is None:
            return None
        diff = simplify(BinOp("-", left, right))
        variables = free_vars(diff)
        if var not in variables or set(variables) - {var}:
            return None

        def probe(k: int) -> Optional[int]:
            return to_int(substitute(diff, var, k))

        f0, f1, f2 = probe(0), probe(1), probe(2)
        if f0 is None or f1 is None or f2 is None:
            return None
        slope, intercept = f1 - f0, f0
        if slope == 0 or f2 - f0 != 2 * slope:
            return None  # not linear in t (e.g. a parity mask)
        if slope < 0:
            # Normalise to a positive slope by negating the comparison.
            slope, intercept = -slope, -intercept
            op = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}[op]

        # The condition reads slope*t + intercept <op> 0 with slope > 0.
        ceil_r = -(-(-intercept) // slope)   # ceil((-intercept) / slope)
        floor_r = (-intercept) // slope      # floor((-intercept) / slope)
        if op == ">=":
            return {ceil_r}                  # false -> true at ceil_r
        if op == ">":
            return {floor_r + 1}
        if op == "<":
            return {ceil_r}                  # true -> false at ceil_r
        if op == "<=":
            return {floor_r + 1}
        # Point conditions flip on at v and flip back off at v + 1.
        if (-intercept) % slope:
            return None
        point = (-intercept) // slope
        return {point, point + 1}

    def _steady_period(self, body: Node, var: str) -> int:
        """LCM of the modular periods (``t % m``, ``t & m``) in the body."""
        period = 1
        for node in _walk(body):
            if node.type != "binary_expression":
                continue
            op_node = node.child_by_field_name("operator")
            op = self._text(op_node) if op_node is not None else ""
            if op not in {"&", "%"}:
                continue
            for var_side, const_side in (
                (node.child_by_field_name("left"), node.child_by_field_name("right")),
                (node.child_by_field_name("right"), node.child_by_field_name("left")),
            ):
                if var_side is None or const_side is None:
                    continue
                if var_side.type != "identifier" or self._text(var_side) != var:
                    continue
                modulus = self.v._eval.fold(const_side, self._env)
                if modulus is None or modulus < 1:
                    continue
                if op == "%" and modulus > 1:
                    period = period * modulus // _gcd(period, modulus)
                elif op == "&" and modulus & (modulus + 1) == 0:
                    # A 2^k - 1 mask: t & (2^k - 1) has period 2^k.
                    period = period * (modulus + 1) // _gcd(period, modulus + 1)
        return period

    def _execute_peeled(self, loop: LoopInfo, body: Node, plan: _PeeledPlan) -> None:
        """Run a planned head / steady / tail traversal of a loop body."""
        var = loop.induction_var
        for value in plan.head:
            self._run_loop_iteration(body, {var: value})
        self._loop_stack.append(loop)
        for value in plan.steady:
            self._run_loop_iteration(body, {var: value})
        self._loop_stack.pop()
        for value in plan.tail:
            self._run_loop_iteration(body, {var: value})

    def _run_loop_iteration(self, body: Node, bindings: Dict[str, object]) -> None:
        """Visit one loop-body instance with the given induction binding.

        A plain integer binding folds every induction-dependent expression in
        the body; a :class:`~ascend_analyzer.symbolic.Var` binding (symbolic
        trip-count fallback) keeps expressions symbolic but bounded to the
        representative's exact value.
        """
        self._env_stack.append(self._env.child())
        for name, value in bindings.items():
            if isinstance(value, Var):
                self._env.define_bounded(name, value.lower, value.upper)
            else:
                self._env.define(name, value)
        self._visit_loop_body(body)
        self._env_stack.pop()

    def _loop_header_text(self, node: Node) -> str:
        body = node.child_by_field_name("body")
        end = body.start_byte if body is not None else node.end_byte
        return " ".join(
            self.v._source_bytes[node.start_byte : end]
            .decode("utf-8", "replace")
            .split()
        )[:120]

    def _parse_loop_header(self, node: Node, loop: LoopInfo) -> None:
        """Recover the induction variable and trip count where possible."""
        init = node.child_by_field_name("initializer")
        condition = node.child_by_field_name("condition")
        update = node.child_by_field_name("update")

        if init is None or condition is None:
            init, condition, update = _scan_for_header_parts(node, init, condition, update)

        start = 0
        start_known = False
        name: Optional[str] = None
        if init is not None:
            name, start_expr = self._parse_loop_init(init)
            folded = to_int(start_expr) if start_expr is not None else None
            if folded is not None:
                start = folded
                start_known = True
        loop.induction_var = name
        if start_known:
            loop.start = start

        limit: Optional[int] = None
        inclusive = False
        if condition is not None and condition.type == "binary_expression":
            op_node = condition.child_by_field_name("operator")
            op = self._text(op_node) if op_node is not None else ""
            right = condition.child_by_field_name("right")
            left = condition.child_by_field_name("left")
            if op in {"<", "<=", "!="} and right is not None:
                limit = self.v._eval.fold(right, self._env)
                inclusive = op == "<="
                if name is None and left is not None and left.type == "identifier":
                    name = self._text(left)
                    loop.induction_var = name

        parsed_step: Optional[int] = None
        if update is not None:
            parsed_step = self._parse_loop_step(update)
        loop.step = parsed_step
        step = parsed_step if parsed_step is not None else 1

        if limit is not None and step > 0:
            span = (limit - start + 1) if inclusive else (limit - start)
            loop.trip_count = max(0, -(-span // step))  # ceil division

        if name:
            upper = (limit - 1) if (limit is not None and not inclusive) else limit
            self._env.define_bounded(name, lower=start, upper=upper)

    def _parse_loop_init(self, init: Node) -> Tuple[Optional[str], Optional[Expr]]:
        if init.type == "declaration":
            for declarator in init.children_by_field_name("declarator"):
                if declarator.type != "init_declarator":
                    continue
                name_node = _declarator_identifier(
                    declarator.child_by_field_name("declarator")
                )
                value_node = declarator.child_by_field_name("value")
                if name_node is not None:
                    return (
                        self._text(name_node),
                        self.v._eval.evaluate(value_node, self._env),
                    )
        if init.type == "assignment_expression":
            left = init.child_by_field_name("left")
            right = init.child_by_field_name("right")
            if left is not None and left.type == "identifier":
                return self._text(left), self.v._eval.evaluate(right, self._env)
        return None, None

    def _parse_loop_step(self, update: Node) -> Optional[int]:
        text = self._text(update)
        if "++" in text:
            return 1
        if "--" in text:
            return None  # counting down; trip count left unknown
        if update.type == "assignment_expression":
            op_node = update.child_by_field_name("operator")
            op = self._text(op_node) if op_node is not None else ""
            right = update.child_by_field_name("right")
            if op == "+=" and right is not None:
                return self.v._eval.fold(right, self._env)
            if op == "=" and right is not None and right.type == "binary_expression":
                inner_op = right.child_by_field_name("operator")
                if inner_op is not None and self._text(inner_op) == "+":
                    for side in ("left", "right"):
                        value = self.v._eval.fold(
                            right.child_by_field_name(side), self._env
                        )
                        if value is not None:
                            return value
        return None

    def _handle_if(self, node: Node) -> None:
        condition = node.child_by_field_name("condition")
        if condition is not None:
            self._scan_for_ops(condition)
        folded = self.v._eval.fold(condition, self._env) if condition is not None else None
        if folded is not None:
            # A compile-time-constant condition selects its arm for every
            # execution; walking only the taken branch keeps the trace exact
            # (this is what prunes the ``if (t + 2 < NV_TILES)`` epilogue
            # guards of an unrolled pipeline loop).
            branch_name = "consequence" if folded != 0 else "alternative"
            branch = node.child_by_field_name(branch_name)
            if branch is not None:
                if branch.type == "compound_statement":
                    self._visit_block(branch, ScopeKind.BRANCH)
                else:
                    self._visit_statement(branch)
            return
        # ``if ASCEND_IS_AIV { ... } else { ... }`` is the run-time spelling of
        # the core split: the two arms run on different physical cores, never
        # on one, so each arm's operations belong to that core's event space.
        guard = _core_view_of_condition(self.v, condition)
        otherwise = guard.complement if guard is not CoreView.BOTH else CoreView.BOTH
        self._conditional_depth += 1
        for field_name, view in (
            ("consequence", guard),
            ("alternative", otherwise),
        ):
            branch = node.child_by_field_name(field_name)
            if branch is None:
                continue
            with self._core_view(view):
                if branch.type == "compound_statement":
                    self._visit_block(branch, ScopeKind.BRANCH)
                else:
                    self._visit_statement(branch)
        self._conditional_depth -= 1

    def _handle_switch(self, node: Node) -> None:
        body = node.child_by_field_name("body")
        if body is None:
            return
        self._conditional_depth += 1
        self._visit_block(body, ScopeKind.BRANCH)
        self._conditional_depth -= 1

    # -- declarations -------------------------------------------------------

    def _handle_declaration(self, node: Node) -> None:
        type_node = node.child_by_field_name("type")
        type_text = self._text(type_node) if type_node is not None else ""
        decl_text = self._text(node)

        # Buffer objects carrying an explicit TPosition template argument.
        # Prefer the AST template-argument parse (handles named depth
        # constants and multi-line declarations); fall back to the legacy
        # textual match for malformed-but-regexable spellings.
        buf_info = _buffer_template_info(self.v, type_node)
        if buf_info is not None:
            if buf_info[0] is not None:
                for declarator in node.children_by_field_name("declarator"):
                    name_node = _declarator_identifier(declarator)
                    if name_node is not None:
                        self._buffer_positions[self._text(name_node)] = buf_info[0]
            return
        tbuf = _TBUF_TYPE_RE.search(type_text) or _TBUF_TYPE_RE.search(decl_text)
        if tbuf is not None:
            position = TPosition.parse(tbuf.group("pos"))
            for declarator in node.children_by_field_name("declarator"):
                name_node = _declarator_identifier(declarator)
                if name_node is not None and position is not None:
                    self._buffer_positions[self._text(name_node)] = position
            return

        # Class-typed locals (``CubeStage cube;`` / ``StageA a;``): remember
        # the static type so ``cube.Process()`` resolves to CubeStage's method
        # even when a sibling class defines the same method name.  Without
        # this, a MIX entry owning two stage classes cannot resolve either
        # call and the whole kernel walk collapses to zero operations.
        bare_type = _bare_type_name(type_text)
        if bare_type and bare_type in self.v._owner_class_names():
            for declarator in node.children_by_field_name("declarator"):
                name_node = _declarator_identifier(declarator)
                if name_node is not None:
                    self._var_classes[self._text(name_node)] = bare_type
            return

        # Foldable scalar initialisers (``int p = t & 1;`` inside an unrolled
        # iteration) feed the folding environment; non-constant inits simply
        # do not fold and are skipped.
        self._collect_scalar_inits(node)

        tensor_match = _TENSOR_TYPE_RE.search(type_text)
        if tensor_match is not None:
            self._declare_tensors(node, tensor_match)
            self._scan_for_ops(node)
            return

        # ``auto x = que.AllocTensor<T>();`` / ``auto y = buf.Get<T>();`` --
        # the initializer's template argument names the element type, so the
        # tensor is declared exactly as an explicitly typed LocalTensor
        # declaration would have been.
        if _bare_type_name(type_text) == "auto":
            if self._declare_auto_tensors(node):
                return

        # ``constexpr``/``const`` scalars feed the folding environment.
        self.v._collect_constexpr(node, self._env)

        # Raw address-space pointers: ``__ubuf__ half* p = (__ubuf__ half*)(off);``
        domain = self.v.prepared.address_space_for_range(node.start_byte, node.end_byte)
        if domain is not None and "*" in decl_text:
            self._declare_raw_pointers(node, domain, type_text)
            return

        self._scan_for_ops(node)

    def _collect_scalar_inits(self, node: Node) -> None:
        """Define plain local scalars whose initialiser folds to a constant.

        Inside an unrolled loop iteration ``int p = t & 1;`` folds against the
        concrete ``t`` of that iteration, which is what lets the
        ``p ? EVENT_ID1 : EVENT_ID0`` event selection resolve per iteration.
        """
        for declarator in node.children_by_field_name("declarator"):
            if declarator.type != "init_declarator":
                continue
            name_node = _declarator_identifier(
                declarator.child_by_field_name("declarator")
            )
            value_node = declarator.child_by_field_name("value")
            if name_node is None or name_node.type != "identifier" or value_node is None:
                continue
            folded = self.v._eval.fold(value_node, self._env)
            if folded is not None:
                self._env.define(self._text(name_node), folded)

    def _declare_tensors(self, node: Node, match: re.Match) -> None:
        kind = match.group("kind")
        dtype = match.group("dtype").rsplit("::", 1)[-1]
        elem = dtype_size(dtype)
        is_global = kind == "GlobalTensor"

        for declarator in node.children_by_field_name("declarator"):
            name_node = _declarator_identifier(declarator)
            if name_node is None:
                continue
            name = self._text(name_node)
            decl = TensorDecl(
                name=name,
                loc=self._loc(declarator if declarator.type != "identifier" else node),
                position=TPosition.GM if is_global else None,
                domain=PhysicalDomain.GM if is_global else PhysicalDomain.UNKNOWN,
                dtype=dtype,
                elem_size=elem,
                scope_id=self._scope_id,
                origin=f"{kind} declaration",
                unbound=True,
            )
            value_node = (
                declarator.child_by_field_name("value")
                if declarator.type == "init_declarator"
                else None
            )
            if value_node is not None:
                self._bind_from_initializer(decl, value_node)
            self.ir.tensors[name] = decl

    def _declare_auto_tensors(self, node: Node) -> bool:
        """Declare ``auto x = <tensor accessor call>();`` locals as tensors.

        Covers ``que.AllocTensor<T>()``, ``que.DeQue<T>()``, ``buf.Get<T>()``
        and ``buf.GetWithOffset<T>(count, offset)``: the template argument
        carries the element type, so the binding below sees a TensorDecl
        with a real ``elem_size`` and the layout checks cover it like any
        explicitly typed declaration.  Returns ``True`` when at least one
        declarator was recognized (the caller then skips generic scanning).
        """
        made = False
        for declarator in node.children_by_field_name("declarator"):
            if declarator.type != "init_declarator":
                continue
            name_node = _declarator_identifier(declarator.child_by_field_name("declarator"))
            value_node = declarator.child_by_field_name("value")
            if name_node is None or name_node.type != "identifier" or value_node is None:
                continue
            dtype = _auto_tensor_dtype(self.v, value_node)
            if dtype is None:
                continue
            name = self._text(name_node)
            decl = TensorDecl(
                name=name,
                loc=self._loc(declarator),
                position=None,
                domain=PhysicalDomain.UNKNOWN,
                dtype=dtype,
                elem_size=dtype_size(dtype),
                scope_id=self._scope_id,
                origin="auto tensor declaration",
                unbound=True,
            )
            self._bind_from_initializer(decl, value_node)
            self.ir.tensors[name] = decl
            made = True
        return made

    def _bind_from_initializer(self, decl: TensorDecl, value: Node) -> None:
        """Resolve ``LocalTensor<T> t = <expr>;`` into a concrete binding."""
        if value.type == "call_expression":
            func = value.child_by_field_name("function")
            args = _argument_nodes(value)
            base = _callee_base_name(self.v, func)

            # ``buf.GetWithOffset<T>(elementCount, byteOffset)`` -- the CANN
            # parameter order: the element count comes FIRST and supplies the
            # tensor's extent (count * sizeof(T)); the byte offset into the
            # buffer comes SECOND and is pool-relative.
            if base == "GetWithOffset" and func is not None:
                receiver = _field_receiver_name(self.v, func)
                if receiver and receiver in self._buffer_positions:
                    self._assign_position(decl, self._buffer_positions[receiver])
                    self._tensor_buffers[decl.name] = receiver
                    decl.source_buffer = receiver
                if len(args) >= 2:
                    count = self.v._eval.evaluate(args[0], self._env)
                    if count is not None:
                        decl.elem_count = count
                    decl.byte_offset = self.v._eval.evaluate(args[1], self._env)
                    decl.unbound = decl.byte_offset is None
                elif args:
                    # Degenerate single-argument spelling: byte offset only.
                    decl.byte_offset = self.v._eval.evaluate(args[0], self._env)
                    decl.unbound = decl.byte_offset is None
                decl.origin = "buffer accessor"
                return

            # ``buf.GetBufferByByte<T>(byteOffset)``
            if base in _BUFFER_GET_BYTE_METHODS and func is not None:
                receiver = _field_receiver_name(self.v, func)
                if receiver and receiver in self._buffer_positions:
                    self._assign_position(decl, self._buffer_positions[receiver])
                    self._tensor_buffers[decl.name] = receiver
                    decl.source_buffer = receiver
                if args:
                    decl.byte_offset = self.v._eval.evaluate(args[0], self._env)
                    decl.unbound = decl.byte_offset is None
                decl.origin = "buffer accessor"
                return

            # ``GetLocalTensor<T>(TPosition::VECIN, byteOffset, elemCount)``
            if base in _TENSOR_FACTORIES:
                if args:
                    position = TPosition.parse(self._text(args[0]))
                    if position is not None:
                        self._assign_position(decl, position)
                if len(args) >= 2:
                    decl.byte_offset = self.v._eval.evaluate(args[1], self._env)
                if len(args) >= 3:
                    decl.elem_count = self.v._eval.evaluate(args[2], self._env)
                decl.unbound = decl.byte_offset is None
                decl.origin = "tensor factory"
                return

            # ``buf.Get<T>()`` - the canonical TPipe allocation accessor.  The
            # tensor inherits the parent buffer's TPosition domain, so
            # ``TBuf<TPosition::A1> b; b.Get<int8_t>()`` binds to L1 without
            # any manual SetTPosition call or annotation.
            # ``buf.Get<T>()`` is the TBuf accessor; ``que.AllocTensor<T>()`` and
            # ``que.DeQue<T>()`` are the TQue equivalents.  All three hand out
            # storage inside the receiver's reserved extent, so the tensor
            # inherits its TPosition domain and its place in the layout.
            if base in _PIPE_ACCESSORS and func is not None:
                receiver = _field_receiver_name(self.v, func)
                if receiver and receiver in self._buffer_positions:
                    self._assign_position(decl, self._buffer_positions[receiver])
                    self._tensor_buffers[decl.name] = receiver
                    decl.source_buffer = receiver
                    decl.origin = _PIPE_ACCESSORS[base]
                return

            # ``other.ReinterpretCast<T>()`` - inherits the source binding.
            if base == "ReinterpretCast" and func is not None:
                receiver = _field_receiver_name(self.v, func)
                source = self.ir.tensors.get(receiver or "")
                if source is not None:
                    self._alias_from(decl, source, element_offset=None)
                    decl.origin = "reinterpret cast"
                return

        if value.type == "identifier":
            source = self.ir.tensors.get(self._text(value))
            if source is not None:
                self._alias_from(decl, source, element_offset=None)
                decl.origin = "alias"
            return

        if value.type == "subscript_expression":
            argument = value.child_by_field_name("argument")
            index = _subscript_index(value)
            base_name = (
                _leading_identifier(self._text(argument))
                if argument is not None
                else _leading_identifier(self._text(value))
            )
            source = self.ir.tensors.get(base_name or "")
            if source is not None:
                self._alias_from(
                    decl,
                    source,
                    element_offset=self.v._eval.evaluate(index, self._env),
                )
                decl.origin = "sub-tensor view"
                return
            # ``buf.Get<T>()[i]`` - a view carved straight out of an accessor
            # call.  The buffer, not a tensor, is what has a base, so the view
            # registers against it and the element shift is applied once the
            # TPipe layout gives that buffer its byte range.
            receiver = self._accessor_receiver(argument)
            if receiver is not None and receiver in self._buffer_positions:
                self._assign_position(decl, self._buffer_positions[receiver])
                self._tensor_buffers[decl.name] = receiver
                decl.source_buffer = receiver
                decl.view_element_offset = self.v._eval.evaluate(
                    index, self._env
                )
                decl.unbound = True
                decl.origin = "buffer view"
            return

        # Anything else that folds to an integer is treated as a byte offset.
        folded = self.v._eval.evaluate(value, self._env)
        if folded is not None and to_int(folded) is not None:
            decl.byte_offset = folded
            decl.unbound = False
            decl.origin = "literal offset"

    def _alias_from(
        self, decl: TensorDecl, source: TensorDecl, element_offset: Optional[Expr]
    ) -> None:
        decl.position = decl.position or source.position
        if decl.domain is PhysicalDomain.UNKNOWN:
            decl.domain = source.domain
        decl.elem_size = decl.elem_size or source.elem_size
        # Record the provenance whatever happens below: the source's own base
        # is usually not known yet (TPipe offsets are synthesised after the
        # walk), so the view's offset is completed by
        # :meth:`_propagate_view_offsets` once the source has one.
        decl.view_source = source.name
        decl.view_element_offset = element_offset
        base = source.byte_offset
        if element_offset is not None and decl.elem_size:
            shift = mul(element_offset, Const(decl.elem_size))
            base = simplify(BinOp("+", base, shift)) if base is not None else None
        decl.byte_offset = base
        decl.unbound = base is None
        if decl.elem_count is None:
            decl.elem_count = source.elem_count
        decl.reuse_group = decl.reuse_group or source.reuse_group

    def _accessor_receiver(self, node: Optional[Node]) -> Optional[str]:
        """The buffer name when *node* is ``buf.Get<T>()`` / ``que.AllocTensor<T>()``."""
        if node is None or node.type != "call_expression":
            return None
        func = node.child_by_field_name("function")
        if _callee_base_name(self.v, func) not in _PIPE_ACCESSORS:
            return None
        return _field_receiver_name(self.v, func)

    def _declare_raw_pointers(
        self, node: Node, domain: PhysicalDomain, type_text: str
    ) -> None:
        dtype = type_text.replace("*", "").strip().rsplit("::", 1)[-1] or None
        elem = dtype_size(dtype)
        for declarator in node.children_by_field_name("declarator"):
            name_node = _declarator_identifier(declarator)
            if name_node is None:
                continue
            name = self._text(name_node)
            value_node = (
                declarator.child_by_field_name("value")
                if declarator.type == "init_declarator"
                else None
            )
            offset = self.v._eval.evaluate(value_node, self._env) if value_node else None
            self.ir.tensors[name] = TensorDecl(
                name=name,
                loc=self._loc(node),
                position=DOMAIN_TO_DEFAULT_TPOSITION.get(domain),
                domain=domain,
                dtype=dtype,
                elem_size=elem,
                byte_offset=offset,
                scope_id=self._scope_id,
                origin="address-space pointer",
                unbound=offset is None,
            )

    def _assign_position(self, decl: TensorDecl, position: TPosition) -> None:
        decl.position = position
        decl.domain = self.v.hw.domain_of(position)

    # -- expression scanning ------------------------------------------------

    def _scan_for_ops(self, node: Node) -> None:
        """Emit operations for every recognised call in ``node``, in order."""
        for call in _call_expressions(node):
            self._handle_call(call)

    def _handle_call(self, node: Node) -> None:
        func = node.child_by_field_name("function")
        if func is None:
            return
        base = _callee_base_name(self.v, func)
        if base is None:
            return

        if base in _SET_FLAG_NAMES:
            self._handle_flag(node, func, FlagKind.SET, isasi=base == "set_flag")
            return
        if base in _WAIT_FLAG_NAMES:
            self._handle_flag(node, func, FlagKind.WAIT, isasi=base == "wait_flag")
            return
        if base in _BARRIER_NAMES:
            self._handle_barrier(node, func)
            return

        # TPipe::InitBuffer records the byte size backing a TBuf, which the
        # layout synthesiser turns into concrete offsets and extents.
        if base == "InitBuffer":
            self._handle_init_buffer(node)
            return

        spec = lookup_api(base)
        if spec is not None:
            self._handle_api_call(node, base, spec)
            return

        receiver = _field_receiver_name(self.v, func)
        if receiver is not None and receiver in self.ir.tensors:
            self._handle_tensor_method(node, base, self.ir.tensors[receiver])
            return

        self._inline_callee(node, base)

    def _inline_callee(self, node: Node, base: str) -> bool:
        """Walk an intra-TU callee's body into this trace, in call position.

        This is what makes a flag raised in a setup helper and consumed in the
        processing method one paired handshake instead of two orphans.  The
        callee's parameters are bound to its arguments where they fold, so an
        event id passed in still resolves.

        Not inlined: anything outside this unit, an overloaded or re-declared
        name, a call already on the inlining stack (recursion), and anything
        past ``_MAX_INLINE_DEPTH``.
        """
        if base in self._inline_stack:
            return False
        # Receiver-typed resolution: ``cube.Process()`` where ``cube`` was
        # declared as ``CubeStage cube;`` resolves against CubeStage's
        # methods first.  This is what keeps a MIX_AIC_1_1 entry that owns
        # two stage classes (both typically defining Init/Process) from
        # collapsing into an unresolvable name collision.
        func = node.child_by_field_name("function")
        receiver = _field_receiver_name(self.v, func)
        receiver_owner = self._var_classes.get(receiver) if receiver else None
        entry = self.v.resolve_callee(base, receiver_owner or self._owner_class)
        if entry is None and receiver_owner is not None:
            entry = self.v.resolve_callee(base, self._owner_class)
        if entry is None:
            return False
        params, body, definition = entry
        if definition is self.func or len(self._inline_stack) >= _MAX_INLINE_DEPTH:
            return False
        if self._truncated:
            return False

        env = self._env.child()
        for param, arg in zip(params, _argument_nodes(node)):
            folded = self.v._eval.fold(arg, self._env)
            if folded is not None:
                env.define(param, folded)

        self._inline_stack.append(base)
        self._env_stack.append(env)
        try:
            self._visit_block(body, ScopeKind.INLINE)
        finally:
            self._env_stack.pop()
            self._inline_stack.pop()
        self.ir.inlined.append(base)
        return True

    def _handle_init_buffer(self, node: Node) -> None:
        """Record ``pipe.InitBuffer(buf, bytes)`` sizes."""
        args = _argument_nodes(node)
        if not args:
            return
        name = _leading_identifier(self._text(args[0]))
        if not name or name not in self._buffer_positions:
            return
        # InitBuffer(buf, len) sizes a TBuf; InitBuffer(que, num, len) sizes a
        # queue of num blocks.  The block length is the last argument either
        # way, and the reserved extent is num blocks of it.
        block = self.v._eval.evaluate(args[-1], self._env)
        if block is None:
            return
        depth = 1
        if len(args) >= 3:
            num = to_int(self.v._eval.evaluate(args[1], self._env))
            if num is not None and num > 0:
                depth = num
        total = mul(block, Const(depth)) if depth > 1 else block
        self._buffer_sizes[name] = total
        self._buffer_blocks[name] = block
        folded = to_int(total)
        if folded is not None:
            self.ir.buffer_sizes[name] = folded

    # -- synchronisation ----------------------------------------------------

    def _handle_flag(
        self, node: Node, func: Node, kind: FlagKind, *, isasi: bool
    ) -> None:
        args = _argument_nodes(node)
        route: Optional[HardEventRoute] = None
        event_node: Optional[Node] = None

        if isasi:
            # set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)
            if len(args) >= 2:
                src = Pipe.parse(self._text(args[0]))
                dst = Pipe.parse(self._text(args[1]))
                if src is not None and dst is not None and src.is_real and dst.is_real:
                    route = HardEventRoute.from_pipes(src, dst)
            if len(args) >= 3:
                event_node = args[2]
        else:
            template_args = _template_arguments(self.v, func)
            if template_args:
                route = HardEventRoute.parse(template_args[0])
            if args:
                event_node = args[0]

        event_id, event_text = self._resolve_event_id(event_node)

        if route is None:
            self.v.diags.add(
                Code.PARSE_ERROR,
                Severity.WARNING,
                f"could not determine the HardEvent route of {self._text(node)[:70]!r}",
                self._loc(node),
                hardware_domain="sync",
                remediation="Spell the route explicitly, e.g. "
                "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0).",
            )
            return

        pipe = route.src if kind is FlagKind.SET else route.dst
        self._emit(
            FlagOp(
                pipe=pipe,
                flag_kind=kind,
                route=route,
                event_id=event_id,
                event_id_text=event_text,
                isasi_form=isasi,
                **self._next_op_args(node),
            )
        )

    def _resolve_event_id(self, node: Optional[Node]) -> Tuple[Optional[int], str]:
        if node is None:
            return None, ""
        text = self._text(node).strip()
        bare = text.rsplit("::", 1)[-1]
        match = _EVENT_ID_RE.match(bare)
        if match is not None:
            return int(match.group("n")), text
        folded = self.v._eval.fold(node, self._env)
        if folded is not None:
            return folded, text
        folded = self._fold_helper_call(node)
        if folded is not None:
            return folded, text
        return None, text

    def _fold_helper_call(self, node: Node) -> Optional[int]:
        """Evaluate ``ev(p)`` against a single-``return`` helper function.

        Covers the ``static __aicore__ inline event_t ev(int p)``
        { return p ? EV1 : EV0; }`` spelling of the ping-pong selector: the
        arguments must fold to constants, the body is then evaluated with the
        parameters bound to them, and the result must itself be constant.
        """
        return self.v.fold_helper_call(node, self._env)

    def _handle_barrier(self, node: Node, func: Node) -> None:
        target = Pipe.ALL
        template_args = _template_arguments(self.v, func)
        if template_args:
            parsed = Pipe.parse(template_args[0])
            if parsed is not None:
                target = parsed
        else:
            args = _argument_nodes(node)
            if args:
                parsed = Pipe.parse(self._text(args[0]))
                if parsed is not None:
                    target = parsed
        self._emit(
            BarrierOp(pipe=target, target=target, **self._next_op_args(node))
        )

    # -- tensor methods -----------------------------------------------------

    def _handle_tensor_method(self, node: Node, method: str, decl: TensorDecl) -> None:
        args = _argument_nodes(node)
        first = args[0] if args else None

        if method in _SET_OFFSET_METHODS and first is not None:
            decl.byte_offset = self.v._eval.evaluate(first, self._env)
            decl.unbound = decl.byte_offset is None
            decl.origin = f"{decl.origin} + {method}"
            return
        if method in _SET_COUNT_METHODS and first is not None:
            decl.elem_count = self.v._eval.evaluate(first, self._env)
            return
        if method in _SET_BYTES_METHODS and first is not None:
            decl.byte_size = self.v._eval.evaluate(first, self._env)
            return
        if method in _SET_POSITION_METHODS and first is not None:
            position = TPosition.parse(self._text(first))
            if position is not None:
                self._assign_position(decl, position)
            return
        if method == "SetGlobalBuffer":
            decl.position = TPosition.GM
            decl.domain = PhysicalDomain.GM
            decl.unbound = False
            if len(args) >= 2:
                decl.elem_count = self.v._eval.evaluate(args[1], self._env)
            return
        if method in _USE_METHODS:
            self._record_use(decl, self._loc(node))

    def _record_use(self, decl: TensorDecl, loc: SourceLoc) -> None:
        if decl.first_use is None:
            decl.first_use = self._index
        decl.last_use = self._index
        if len(decl.use_locs) < 32:
            decl.use_locs.append(loc)

    # -- intrinsics ---------------------------------------------------------

    def _handle_api_call(self, node: Node, name: str, spec: ApiSpec) -> None:
        arg_nodes = _argument_nodes(node)
        args: List[ArgRef] = []
        writes: List[str] = []
        reads: List[str] = []

        for position, arg_node in enumerate(arg_nodes):
            text = self._text(arg_node)
            tensor_name = self._resolve_tensor_arg(arg_node, text)
            expr = self.v._eval.evaluate(arg_node, self._env)
            args.append(
                ArgRef(
                    index=position,
                    text=" ".join(text.split())[:60],
                    tensor=tensor_name,
                    value=to_int(expr),
                    expr=expr,
                )
            )
            if tensor_name is None:
                continue
            param = spec.param_at(position)
            if param is not None and param.role is ArgRole.DST:
                writes.append(tensor_name)
            else:
                reads.append(tensor_name)

        pipe = self._resolve_pipe(spec, args)
        op = ApiCallOp(
            pipe=pipe,
            name=name,
            args=tuple(args),
            writes=tuple(writes),
            reads=tuple(reads),
            text=" ".join(self._text(node).split())[:200],
            **self._next_op_args(node),
        )
        emitted = self._emit(op)
        if emitted is None:
            return
        self._infer_transfer_volume(name, spec, args)
        for arg in args:
            if arg.tensor and arg.tensor in self.ir.tensors:
                self._record_use(self.ir.tensors[arg.tensor], op.loc)

    def _infer_transfer_volume(
        self, name: str, spec: ApiSpec, args: Sequence[ArgRef]
    ) -> None:
        """Derive a loader's transaction volume from its repeat parameter.

        Raw CCE loaders hand the analyzer ``(__ca__ void *)`` operands, so the
        usual ``elem_count * sizeof(T)`` extent is unavailable.  When the
        signature carries a volume model and the repeat argument folds, the
        referenced tensors get ``repeat * unit_bytes`` as their byte size
        instead of remaining unverifiable (AKA3002).
        """
        volume = spec.volume
        if volume is None or not (0 <= volume.repeat_index < len(args)):
            return
        repeat = args[volume.repeat_index].value
        if repeat is None:
            return
        total = repeat * volume.unit_bytes
        for arg in args:
            if not arg.tensor:
                continue
            decl = self.ir.tensors.get(arg.tensor)
            if decl is None or decl.byte_size is not None:
                continue
            # A TPipe InitBuffer allocates the buffer's whole extent; the
            # repeat-derived volume would only under-approximate it.
            buffer_name = self._tensor_buffers.get(arg.tensor)
            if buffer_name is not None and buffer_name in self._buffer_sizes:
                continue
            decl.byte_size = Const(total)
            if decl.elem_size:
                decl.elem_count = Const(total // decl.elem_size)
            decl.origin = f"{decl.origin} + {name} volume ({volume.basis})"

    def _resolve_pipe(self, spec: ApiSpec, args: Sequence[ArgRef]) -> Pipe:
        if spec.pipe is not None:
            return spec.pipe
        # DataCopy and friends: the pipeline follows the (dst, src) domains.
        dst = self._domain_of_arg(args, 0)
        src = self._domain_of_arg(args, 1)
        return data_copy_pipe(dst, src) or Pipe.MTE2

    def _domain_of_arg(self, args: Sequence[ArgRef], index: int) -> PhysicalDomain:
        if index >= len(args):
            return PhysicalDomain.UNKNOWN
        name = args[index].tensor
        if not name:
            return PhysicalDomain.UNKNOWN
        decl = self.ir.tensors.get(name)
        return decl.domain if decl is not None else PhysicalDomain.UNKNOWN

    def _resolve_tensor_arg(self, node: Node, text: str) -> Optional[str]:
        """Map an argument expression back to a declared tensor name."""
        candidate = _leading_identifier(_strip_leading_casts(text))
        if candidate and candidate in self.ir.tensors:
            return candidate
        if node.type == "parenthesized_expression":
            inner = next(
                (c for c in node.named_children if c.type != "comment"), None
            )
            if inner is not None:
                return self._resolve_tensor_arg(inner, self._text(inner))
        if node.type == "cast_expression":
            value = node.child_by_field_name("value")
            if value is not None:
                return self._resolve_tensor_arg(value, self._text(value))
        return None

    # -- post-processing ----------------------------------------------------

    def _apply_annotations(self) -> None:
        """Apply ``@ascend-layout`` / ``@ascend-reuse-group`` annotations."""
        for ann in self.v.prepared.annotations:
            if ann.kind == "layout":
                self._apply_layout_annotation(ann)
            elif ann.kind in {"reuse-group", "reuse"}:
                self._apply_reuse_annotation(ann)

    def _apply_layout_annotation(self, ann) -> None:
        name = ann.get("name")
        if not name:
            return
        loc = SourceLoc(
            file=self.v.prepared.path,
            line=ann.line,
            column=1,
            snippet=self.v.line_text(ann.line).strip()[:160],
        )
        decl = self.ir.tensors.get(name)
        if decl is None:
            decl = TensorDecl(
                name=name,
                loc=loc,
                position=None,
                domain=PhysicalDomain.UNKNOWN,
                scope_id=0,
                origin="@ascend-layout annotation",
            )
            self.ir.tensors[name] = decl

        position = TPosition.parse(ann.get("pos") or ann.get("position") or "")
        if position is not None:
            self._assign_position(decl, position)
        dtype = ann.get("dtype")
        if dtype:
            decl.dtype = dtype
            decl.elem_size = dtype_size(dtype) or decl.elem_size
        offset = ann.get_int("offset")
        if offset is not None:
            decl.byte_offset = Const(offset)
            decl.unbound = False
        count = ann.get_int("count")
        if count is not None:
            decl.elem_count = Const(count)
        nbytes = ann.get_int("bytes")
        if nbytes is not None:
            decl.byte_size = Const(nbytes)
        group = ann.get("group")
        if group:
            decl.reuse_group = group
        decl.origin = f"{decl.origin} + annotation"

    def _apply_reuse_annotation(self, ann) -> None:
        group = ann.get("group")
        if not group:
            return
        names: List[str] = []
        single = ann.get("name")
        if single:
            names.append(single)
        many = ann.get("names")
        if many:
            names.extend(part for part in many.split(",") if part)
        for name in names:
            decl = self.ir.tensors.get(name.strip())
            if decl is not None:
                decl.reuse_group = group

    def _finalize_tensors(self) -> None:
        """Derive byte sizes, synthesise TPipe allocations, close scopes."""
        self._synthesize_tpipe_layout()
        self._propagate_view_offsets()
        for decl in self.ir.tensors.values():
            if decl.byte_size is None and decl.elem_count is not None:
                if decl.elem_size:
                    decl.byte_size = mul(decl.elem_count, Const(decl.elem_size))
            if decl.elem_count is None and decl.byte_size is not None and decl.elem_size:
                size = to_int(decl.byte_size)
                if size is not None and decl.elem_size:
                    decl.elem_count = Const(size // decl.elem_size)
        for scope in self.ir.scopes.values():
            scope.end_index = max(scope.end_index, scope.start_index)

    def _allocation_order(self) -> List[str]:
        """Buffer names in the order ``TPipe`` carves UB for them.

        The allocator is a bump pointer driven by ``InitBuffer`` call order,
        which need not match the order the members were declared in.  Buffers
        that were never sized sort last; they only mark their domain blocked.
        """
        names = list(self._buffer_positions)
        registry = self.v._pipe_buffers

        def key(name: str) -> Tuple[int, int]:
            entry = registry.get(name)
            if entry is None or entry.order is None:
                return (1, names.index(name))
            return (0, entry.order)

        return sorted(names, key=key)

    def _synthesize_tpipe_layout(self) -> None:
        """Give ``TBuf<TPosition>`` allocations concrete byte ranges.

        ``TPipe`` hands out offsets at runtime, but the *sequence* of
        ``InitBuffer`` calls in program order fixes a bump allocation per
        physical domain.  Replaying it (each buffer aligned to its domain's
        base alignment) yields concrete, disjoint layouts for every tensor
        obtained via ``buf.Get<T>()`` - enough to verify capacity, alignment
        and aliasing exactly, which an unknown runtime offset never could.
        Buffers without a visible ``InitBuffer`` size keep symbolic extents.
        """
        if not self._buffer_positions:
            return
        cursors: Dict[PhysicalDomain, int] = {}
        layouts: Dict[str, Tuple[int, int]] = {}
        #: Domains whose bump allocation has hit a buffer of unknown size.
        #: Everything the allocator hands out after that sits at an offset we
        #: cannot know, so no later buffer in that domain gets a concrete base -
        #: inventing one would fabricate overlaps and bank deltas.
        blocked: Set[PhysicalDomain] = set()
        for name in self._allocation_order():
            position = self._buffer_positions.get(name)
            if position is None:
                continue
            domain = self.v.hw.domain_of(position)
            if not domain.is_on_core_sram or domain in blocked:
                continue
            size = self._buffer_sizes.get(name)
            size_value = to_int(size) if size is not None else None
            if size_value is None or size_value <= 0:
                blocked.add(domain)
                continue
            align = self.v.hw.base_alignment(domain)
            base = -(-cursors.get(domain, 0) // align) * align  # ceil to align
            layouts[name] = (base, size_value)
            cursors[domain] = base + size_value

        for tensor_name, buffer_name in self._tensor_buffers.items():
            decl = self.ir.tensors.get(tensor_name)
            layout = layouts.get(buffer_name)
            if decl is None or layout is None:
                continue
            base, _ = layout
            if decl.byte_offset is None:
                decl.byte_offset = Const(base)
                decl.unbound = False
            elif decl.source_buffer == buffer_name:
                # Accessor views (GetWithOffset / GetBufferByByte) carry a
                # POOL-RELATIVE offset argument; place it inside the buffer.
                rel = to_int(decl.byte_offset)
                if rel is not None:
                    decl.byte_offset = Const(base + rel)
                    decl.unbound = False
            # One accessor call hands out one block, not the queue's whole
            # reserved extent, so a depth-2 queue sizes its tensors by block.
            # A view that already named its element count (GetWithOffset's
            # first argument) keeps that exact extent.
            block = to_int(self._buffer_blocks.get(buffer_name))
            if block is not None:
                if decl.byte_size is None and decl.elem_count is None:
                    decl.byte_size = Const(block)
                if decl.elem_count is None and decl.elem_size:
                    decl.elem_count = Const(block // decl.elem_size)
            # Every tensor drawn from one buffer shares that storage by design:
            # successive Get<T>() calls return the same region, and a queue's
            # ping/pong blocks are deliberate recycling.  Grouping them keeps
            # the aliasing check from reporting the buffer manager's own reuse.
            decl.reuse_group = decl.reuse_group or f"tpipe:{buffer_name}"
            decl.origin = f"{decl.origin} + TPipe layout"

    def _propagate_view_offsets(self) -> None:
        """Complete view offsets now that their bases exist.

        A view is declared as ``LocalTensor t = arena[i]``, but at that moment
        the arena itself has no base - TPipe offsets are only synthesised
        after the walk - so the declaration records the provenance instead
        (``view_source`` plus an element offset).  This pass replays it:

        * a view with a ``view_source`` lands at
          ``offset(source) + element_offset * sizeof(T)``, iterated to a
          fixpoint so a view of a view resolves too;
        * a view carved straight out of an accessor call
          (``buf.Get<T>()[i]``) has no source tensor; its buffer base was
          just synthesised, so the element shift is applied to that.
        """
        # Buffer-rooted views first: their base came from the layout pass.
        for decl in self.ir.tensors.values():
            offset = decl.view_element_offset
            if decl.view_source or offset is None:
                continue
            if decl.byte_offset is None or not decl.elem_size:
                continue
            decl.byte_offset = simplify(
                BinOp("+", decl.byte_offset, mul(offset, Const(decl.elem_size)))
            )
            decl.unbound = False

        views = [d for d in self.ir.tensors.values() if d.view_source]
        for _ in range(len(views) or 1):  # fixpoint; each pass binds one level
            changed = False
            for decl in views:
                if decl.byte_offset is not None:
                    continue  # an explicit binding (annotation, SetAddr) wins
                source = self.ir.tensors.get(decl.view_source or "")
                if source is None or source.byte_offset is None:
                    continue
                base = source.byte_offset
                if decl.view_element_offset is not None and decl.elem_size:
                    base = simplify(
                        BinOp(
                            "+",
                            base,
                            mul(decl.view_element_offset, Const(decl.elem_size)),
                        )
                    )
                decl.byte_offset = base
                decl.unbound = False
                decl.origin = f"{decl.origin} + view offset"
                changed = True
            if not changed:
                break

    @staticmethod
    def _function_name(func: Node) -> str:
        declarator = func.child_by_field_name("declarator")
        while declarator is not None:
            if declarator.type == "function_declarator":
                inner = declarator.child_by_field_name("declarator")
                if inner is not None:
                    return inner.text.decode("utf-8", "replace")
                break
            nxt = declarator.child_by_field_name("declarator")
            if nxt is None or nxt is declarator:
                break
            declarator = nxt
        return "<anonymous>"


# ---------------------------------------------------------------------------
# Node utilities
# ---------------------------------------------------------------------------


def _walk(node: Node) -> Iterator[Node]:
    """Pre-order traversal over every node in document order."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _call_expressions(node: Node) -> List[Node]:
    """Every ``call_expression`` under ``node``, in source order."""
    found = [n for n in _walk(node) if n.type == "call_expression"]
    found.sort(key=lambda n: n.start_byte)
    return found


def _argument_nodes(call: Node) -> List[Node]:
    args = call.child_by_field_name("arguments")
    if args is None:
        return []
    return [c for c in args.named_children if c.type != "comment"]


def _subscript_index(node: Node) -> Optional[Node]:
    """The index expression of ``base[index]``.

    tree-sitter-cpp has no ``index`` field on a ``subscript_expression``: the
    bracketed operand arrives as a ``subscript_argument_list`` child.  Asking
    for the field by name yields ``None``, which is why sub-tensor views were
    all recorded without an offset until now.  Both spellings are tried, so
    the helper survives a grammar that names the field.
    """
    index = node.child_by_field_name("index")
    if index is not None:
        return index
    for child in node.children:
        if child.type != "subscript_argument_list":
            continue
        for inner in child.named_children:
            if inner.type != "comment":
                return inner
    return None


def _callee_base_name(visitor: ASTVisitor, func: Optional[Node]) -> Optional[str]:
    """The bare callee name: ``AscendC::SetFlag<...>`` -> ``SetFlag``."""
    if func is None:
        return None
    node = func
    # Unwrap qualified_identifier / template_function / field_expression.
    for _ in range(6):
        if node.type == "qualified_identifier":
            inner = node.child_by_field_name("name")
            if inner is None:
                break
            node = inner
            continue
        if node.type in {"template_function", "template_method"}:
            inner = node.child_by_field_name("name")
            if inner is None:
                break
            node = inner
            continue
        if node.type == "field_expression":
            inner = node.child_by_field_name("field")
            if inner is None:
                break
            node = inner
            continue
        break
    text = visitor.text(node).strip()
    text = text.split("<", 1)[0].rsplit("::", 1)[-1].strip()
    return text or None


def _field_receiver_name(visitor: ASTVisitor, func: Optional[Node]) -> Optional[str]:
    """For ``obj.Method(...)`` return ``obj``."""
    if func is None or func.type != "field_expression":
        return None
    argument = func.child_by_field_name("argument")
    if argument is None:
        return None
    return _leading_identifier(visitor.text(argument))


def _bare_type_name(type_text: str) -> str:
    """The unqualified, unadorned class name of a declaration's type text.

    ``"AscendC::CubeStage"``/``"const StageA &"``/``"StageB *"`` all reduce to
    their bare class identifier so it can be matched against the unit's
    method-owning classes.
    """
    if not type_text:
        return ""
    for strip in ("*", "&"):
        type_text = type_text.replace(strip, " ")
    tokens = type_text.split()
    if not tokens:
        return ""
    return tokens[-1].rsplit("::", 1)[-1].strip()


#: Tensor accessors whose template argument names the element type, so an
#: ``auto x = que.AllocTensor<half>();`` declaration still yields a fully
#: typed tensor binding.
_AUTO_INFER_ACCESSORS = frozenset({"AllocTensor", "DeQue", "Get", "GetWithOffset"})


def _auto_tensor_dtype(visitor: ASTVisitor, value: Node) -> Optional[str]:
    """Element type named by ``que.AllocTensor<T>()`` & friends, else ``None``."""
    if value.type != "call_expression":
        return None
    func = value.child_by_field_name("function")
    if func is None:
        return None
    base = _callee_base_name(visitor, func)
    if base not in _AUTO_INFER_ACCESSORS:
        return None
    args = _template_arguments(visitor, func)
    if not args:
        return None
    dtype = args[0].rsplit("::", 1)[-1].strip()
    return dtype if dtype_size(dtype) else None


#: Type templates whose arguments fix a buffer's TPosition and ping-pong depth.
_BUFFER_TEMPLATE_NAMES = frozenset({"TBuf", "TQue", "TQueBind"})


def _buffer_template_info(
    visitor: ASTVisitor, type_node: Optional[Node]
) -> Optional[Tuple[Optional[TPosition], Optional[int]]]:
    """Parse a ``TQue<TPosition::X, depth>`` type node via its AST.

    Walks the ``template_argument_list`` directly instead of regexing the
    declaration text, so a named depth constant (``TQue<TPosition::VECIN,
    AIV_QUE_DEPTH>``) resolves through the unit's constant environment and a
    multi-line declaration cannot defeat the match.  Both ``TPosition::X``
    and the legacy ``QuePosition::X`` spellings are accepted (the position is
    taken from the argument's last ``::`` segment).

    Returns ``(position, depth)`` -- position ``None`` when the first
    argument does not name a position, depth ``None`` when the second
    argument is absent or does not fold (callers fall back to the InitBuffer
    ``num`` argument, which overrides it anyway) -- or ``None`` when the type
    node carries no buffer template at all.
    """
    if type_node is None:
        return None
    for node in _walk(type_node):
        if node.type != "template_type":
            continue
        name = node.child_by_field_name("name")
        if name is None:
            continue
        if visitor.text(name).rsplit("::", 1)[-1].strip() not in _BUFFER_TEMPLATE_NAMES:
            continue
        arguments = node.child_by_field_name("arguments")
        if arguments is None:
            continue
        args = [c for c in arguments.named_children if c.type != "comment"]
        if not args:
            return None
        pos_text = visitor.text(args[0]).strip()
        position = TPosition.parse(pos_text)
        if position is None:
            position = TPosition.parse(pos_text.rsplit("::", 1)[-1])
        depth: Optional[int] = None
        if len(args) >= 2:
            folded = visitor._eval.fold(args[1], visitor._global_env)
            if folded is None:
                folded = visitor._eval.evaluate(args[1], visitor._global_env)
            if folded is not None:
                try:
                    depth = int(folded)
                except (TypeError, ValueError):
                    depth = None
        return (position, depth)
    return None


def _template_arguments(visitor: ASTVisitor, func: Node) -> List[str]:
    """Textual template arguments of a templated callee."""
    for node in _walk(func):
        if node.type == "template_argument_list":
            return [
                visitor.text(child).strip()
                for child in node.named_children
                if child.type != "comment"
            ]
    return []


def _parameter_names(declarator: Node) -> List[str]:
    """Parameter names of a function declarator, in declaration order."""
    names: List[str] = []
    parameters = declarator.child_by_field_name("parameters")
    if parameters is None:
        return names
    for param in parameters.named_children:
        if param.type != "parameter_declaration":
            continue
        name_node = _declarator_identifier(param.child_by_field_name("declarator"))
        if name_node is not None:
            names.append(name_node.text.decode("utf-8", "replace"))
    return names


def _declarator_identifier(declarator: Optional[Node]) -> Optional[Node]:
    """Drill through pointer/array/init declarators to the bare identifier."""
    node = declarator
    seen = 0
    while node is not None and seen < 10:
        seen += 1
        if node.type == "identifier":
            return node
        if node.type in {
            "init_declarator",
            "pointer_declarator",
            "array_declarator",
            "reference_declarator",
            "parenthesized_declarator",
        }:
            inner = node.child_by_field_name("declarator")
            if inner is None:
                inner = next(
                    (c for c in node.named_children if c.type != "comment"), None
                )
            node = inner
            continue
        if node.type == "field_identifier":
            return node
        break
    return None


def _leading_identifier(text: str) -> Optional[str]:
    match = _LEADING_IDENT_RE.match(text)
    return match.group("name") if match else None


def _template_dtype(type_text: str) -> Optional[str]:
    match = re.search(r"<\s*([A-Za-z_][\w:]*)\s*>", type_text)
    return match.group(1).rsplit("::", 1)[-1] if match else None


def _last_statement_child(node: Node) -> Optional[Node]:
    for child in reversed(node.named_children):
        if child.type.endswith("_statement") or child.type == "compound_statement":
            return child
    return None


def _scan_for_header_parts(
    node: Node,
    init: Optional[Node],
    condition: Optional[Node],
    update: Optional[Node],
) -> Tuple[Optional[Node], Optional[Node], Optional[Node]]:
    """Positional fallback for ``for`` headers when field names are absent."""
    body = node.child_by_field_name("body")
    for child in node.named_children:
        if child is body or child.type == "comment":
            continue
        if init is None and child.type in {"declaration", "assignment_expression"}:
            init = child
            continue
        if condition is None and child.type == "binary_expression":
            condition = child
            continue
        if update is None and child.type in {
            "update_expression",
            "assignment_expression",
        }:
            update = child
    return init, condition, update


# ---------------------------------------------------------------------------
# Parser construction and façade helpers
# ---------------------------------------------------------------------------

_PARSER: Optional[Parser] = None


def _make_parser() -> Parser:
    global _PARSER
    if _PARSER is None:
        import tree_sitter_cpp

        _PARSER = Parser(Language(tree_sitter_cpp.language()))
    return _PARSER


def parse_source(
    path: str,
    source: str,
    hardware: HardwareModel,
    diagnostics: DiagnosticCollector,
    options: Optional[VisitorOptions] = None,
) -> AnalysisUnit:
    """Parse ``source`` into an :class:`AnalysisUnit`.

    The source first goes through the macro expander (local includes, stage
    macros, launch syntax) and then the qualifier rewrite, so kernels written
    as ``#define`` pipeline stages parse as cleanly as their expanded form.
    """
    prepared = prepare_translation_unit(path, source)
    return ASTVisitor(prepared, hardware, diagnostics, options).run()


def parse_file(
    path: str,
    hardware: HardwareModel,
    diagnostics: DiagnosticCollector,
    options: Optional[VisitorOptions] = None,
) -> AnalysisUnit:
    """Read and parse a kernel source file."""
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    return parse_source(path, text, hardware, diagnostics, options)
