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
from dataclasses import dataclass
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
    TPOSITION_TO_DOMAIN,
)
from ..ir import (
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
from .expr_eval import ConstEnv, ExpressionEvaluator, collect_define_value
from .preprocess import PreparedSource, prepare_source, prepare_translation_unit

__all__ = ["ASTVisitor", "parse_source", "parse_file", "VisitorOptions"]


# ---------------------------------------------------------------------------
# Recognition patterns
# ---------------------------------------------------------------------------

_TENSOR_TYPE_RE = re.compile(
    r"\b(?P<kind>LocalTensor|GlobalTensor)\s*<\s*(?P<dtype>[A-Za-z_][\w:]*)\s*>"
)
_TBUF_TYPE_RE = re.compile(r"\bT(?:Buf|Que|QueBind)\b.*?TPosition::(?P<pos>[A-Z0-9_]+)")
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
_BUFFER_GET_BYTE_METHODS = frozenset({"GetBufferByByte", "GetWithOffset", "GetBufferAddr"})

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
        #: Small scalar helper functions (``event_t ev(int p)``) whose body is
        #: a single ``return <expr>;``, for constexpr-style event-id folding.
        #: Maps name -> (parameter names, returned expression node).
        self._helpers: Dict[str, Tuple[Tuple[str, ...], Node]] = {}

    # -- entry point --------------------------------------------------------

    def run(self) -> AnalysisUnit:
        tree = self._parser.parse(self._source_bytes)
        root = tree.root_node

        unit = AnalysisUnit(path=self.prepared.path, source=self.prepared.original)
        self._report_parse_errors(root, unit)
        self._seed_builtin_constants()
        self._collect_global_constants(root)
        self._collect_helper_functions(root)
        unit.constants = dict(self._global_env.flatten())
        unit.suppressions = {
            ann.line: [
                token.strip().upper()
                for token in ann.body.replace(",", " ").split()
                if token.strip()
            ]
            for ann in self.prepared.annotations_of("ignore")
        }

        for func in self._find_functions(root):
            is_entry = self.prepared.has_kernel_attribute_before(func.start_byte)
            if not is_entry and not self.opts.analyze_all_functions:
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
            # Both arms of a conditional-compilation block are walked; mark
            # them conditional so pairing diagnostics are softened.
            self._conditional_depth += 1
            for child in node.named_children:
                self._visit_statement(child)
            self._conditional_depth -= 1
            return

        # Anything else (return, labelled statements, try blocks, ...) is
        # scanned for operations and then walked structurally.
        self._scan_for_ops(node)

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
        self._conditional_depth += 1
        for field_name in ("consequence", "alternative"):
            branch = node.child_by_field_name(field_name)
            if branch is None:
                continue
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
        tbuf = _TBUF_TYPE_RE.search(type_text) or _TBUF_TYPE_RE.search(decl_text)
        if tbuf is not None:
            position = TPosition.parse(tbuf.group("pos"))
            for declarator in node.children_by_field_name("declarator"):
                name_node = _declarator_identifier(declarator)
                if name_node is not None and position is not None:
                    self._buffer_positions[self._text(name_node)] = position
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

    def _bind_from_initializer(self, decl: TensorDecl, value: Node) -> None:
        """Resolve ``LocalTensor<T> t = <expr>;`` into a concrete binding."""
        if value.type == "call_expression":
            func = value.child_by_field_name("function")
            args = _argument_nodes(value)
            base = _callee_base_name(self.v, func)

            # ``buf.GetBufferByByte<T>(byteOffset)``
            if base in _BUFFER_GET_BYTE_METHODS and func is not None:
                receiver = _field_receiver_name(self.v, func)
                if receiver and receiver in self._buffer_positions:
                    self._assign_position(decl, self._buffer_positions[receiver])
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
            if base == "Get" and func is not None:
                receiver = _field_receiver_name(self.v, func)
                if receiver and receiver in self._buffer_positions:
                    self._assign_position(decl, self._buffer_positions[receiver])
                    self._tensor_buffers[decl.name] = receiver
                    decl.source_buffer = receiver
                    decl.origin = "TBuf::Get"
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
            base_name = _leading_identifier(self._text(value))
            source = self.ir.tensors.get(base_name or "")
            index = value.child_by_field_name("index")
            if source is not None:
                self._alias_from(
                    decl, source, element_offset=self.v._eval.evaluate(index, self._env)
                )
                decl.origin = "sub-tensor view"
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
        base = source.byte_offset
        if element_offset is not None and decl.elem_size:
            shift = mul(element_offset, Const(decl.elem_size))
            base = simplify(BinOp("+", base, shift)) if base is not None else shift
        decl.byte_offset = base
        decl.unbound = base is None
        if decl.elem_count is None:
            decl.elem_count = source.elem_count
        decl.reuse_group = decl.reuse_group or source.reuse_group

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

    def _handle_init_buffer(self, node: Node) -> None:
        """Record ``pipe.InitBuffer(buf, bytes)`` sizes."""
        args = _argument_nodes(node)
        if not args:
            return
        name = _leading_identifier(self._text(args[0]))
        if not name or name not in self._buffer_positions:
            return
        size_node = args[1] if len(args) >= 2 else None
        if size_node is None:
            return
        size = self.v._eval.evaluate(size_node, self._env)
        if size is not None:
            self._buffer_sizes[name] = size
            folded = to_int(size)
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
        if node.type != "call_expression":
            return None
        func = node.child_by_field_name("function")
        name = _callee_base_name(self.v, func)
        if name is None or name not in self.v._helpers:
            return None
        params, body = self.v._helpers[name]
        arg_nodes = _argument_nodes(node)
        values: List[int] = []
        for arg in arg_nodes:
            folded = self.v._eval.fold(arg, self._env)
            if folded is None:
                return None
            values.append(folded)
        if len(values) != len(params):
            return None
        env = self.v._global_env.child()
        for param, value in zip(params, values):
            env.define(param, value)
        return self.v._eval.fold(body, env)

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
        for name, position in self._buffer_positions.items():
            domain = self.v.hw.domain_of(position)
            if not domain.is_on_core_sram:
                continue
            size = self._buffer_sizes.get(name)
            size_value = to_int(size) if size is not None else None
            align = self.v.hw.base_alignment(domain)
            base = -(-cursors.get(domain, 0) // align) * align  # ceil to align
            layouts[name] = (base, size_value) if size_value is not None else (base, None)
            if size_value is not None:
                cursors[domain] = base + size_value

        for tensor_name, buffer_name in self._tensor_buffers.items():
            decl = self.ir.tensors.get(tensor_name)
            layout = layouts.get(buffer_name)
            if decl is None or layout is None:
                continue
            base, size = layout
            if decl.byte_offset is None:
                decl.byte_offset = Const(base)
                decl.unbound = False
            if size is not None:
                if decl.byte_size is None:
                    decl.byte_size = Const(size)
                if decl.elem_count is None and decl.elem_size:
                    decl.elem_count = Const(size // decl.elem_size)
            decl.origin = f"{decl.origin} + TPipe layout"

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
