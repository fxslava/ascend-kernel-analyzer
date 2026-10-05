"""AST-to-MLIR lowering bridge: Clang AST JSON -> ``ascend`` dialect module.

Consumes the BiSheng extraction result (:mod:`.bisheng_extractor`) and emits
a structured :class:`~ascend_analyzer.ir.mlir_ascend.AscendModule`.  The
bridge relies on the C++ parser for everything semantic - template
monomorphisation, ``auto`` deduction, overload resolution, constant folding -
and only *reads* the typed AST:

* template types give ``TQue`` positions and depths as evaluated integer and
  enum arguments (read structurally, never re-parsed by hand);
* ``VarDecl`` sugar/desugared types give tensor dtypes, including for
  ``auto`` variables;
* instantiated ``SetFlag``/``WaitFlag`` carry their ``HardEvent`` argument as
  an evaluated enum value, joined back to a route name through the enum
  layout the AST itself provides;
* the runtime core sentinels (``ASCEND_IS_AIC``/``ASCEND_IS_AIV``) wrap
  guarded statements into :class:`CoreRegionOp`s, so heterogeneous kernels
  keep per-core operation sequences that never collide.

No regular expressions are used anywhere in this module; template argument
lists are read from a small structural tokenizer over type strings when the
JSON only offers the rendered form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..ir.mlir_ascend import (
    AllocBufferOp,
    AllocTensorOp,
    AscendModule,
    BarrierOp,
    CoreRegionOp,
    CoreType,
    DeQueOp,
    EnQueOp,
    FreeTensorOp,
    GetTensorOp,
    KernelOp,
    MemorySpace,
    MmadOp,
    MteCopyOp,
    SetFlagOp,
    SsaValue,
    VectorOp,
    WaitFlagOp,
)
from .bisheng_extractor import is_main_file, node_line

__all__ = ["BridgeOptions", "lower_module", "split_template_args"]


#: Vector intrinsics recognised as compute ops.
VECTOR_OPS: Tuple[str, ...] = (
    "Add", "Sub", "Mul", "And", "Or", "Xor", "Not",
    "ShiftLeft", "ShiftRight", "Cast", "Max", "Min",
    "Exp", "Reciprocal", "Duplicate", "Gather", "Scatter", "Select",
)

#: Queue bookkeeping methods (modelled as queue ops).
_ALLOC_METHODS = frozenset({"AllocTensor"})
_FREE_METHODS = frozenset({"FreeTensor"})

#: Statement kinds the walker dispatches on.
_STMT_KINDS = frozenset({
    "CompoundStmt", "DeclStmt", "IfStmt", "ForStmt", "WhileStmt", "DoStmt",
    "CXXForRangeStmt", "CallExpr", "CXXMemberCallExpr", "BinaryOperator",
    "CompoundAssignOperator", "ReturnStmt", "CXXConstructExpr",
})

#: Cast wrappers to see through when reading expressions.
_CAST_KINDS = frozenset({
    "ImplicitCastExpr", "CStyleCastExpr", "CXXStaticCastExpr",
    "CXXReinterpretCastExpr", "CXXConstCastExpr", "ConstantExpr", "FullExpr",
    "MaterializeTemporaryExpr", "CXXBindTemporaryExpr", "ParenExpr",
})

_SENTINEL_AIC = "__ascend_core_is_aic"
_SENTINEL_AIV = "__ascend_core_is_aiv"

#: Loop trip counts up to this bound replay the body exactly.
UNROLL_LIMIT = 8

_POSITION_SPACES: Dict[str, MemorySpace] = {
    "VECIN": MemorySpace.UB, "VECOUT": MemorySpace.UB, "VECCALC": MemorySpace.UB,
    "A1": MemorySpace.L1, "B1": MemorySpace.L1, "C1": MemorySpace.L1,
    "A2": MemorySpace.L0A, "B2": MemorySpace.L0B, "CO1": MemorySpace.L0C,
    "CO2": MemorySpace.L0C, "LCM": MemorySpace.L1, "TSCM": MemorySpace.L1,
    "GM": MemorySpace.GM,
}

_DTYPE_SIZES: Dict[str, int] = {
    "bool": 1, "char": 1, "int8_t": 1, "uint8_t": 1, "unsigned char": 1,
    "signed char": 1, "int4b_t": 1, "uint4b_t": 1,
    "int16_t": 2, "uint16_t": 2, "short": 2, "half": 2, "bfloat16_t": 2,
    "unsigned short": 2,
    "int32_t": 4, "uint32_t": 4, "int": 4, "unsigned int": 4, "float": 4,
    "int64_t": 8, "uint64_t": 8, "long": 8, "unsigned long": 8, "double": 8,
}

_DTYPE_ALIASES: Dict[str, str] = {
    "_Float16": "half", "short": "int16_t", "unsigned short": "uint16_t",
    "unsigned char": "uint8_t", "unsigned int": "uint32_t", "unsigned long": "uint64_t",
}


@dataclass
class BridgeOptions:
    """Tuning knobs for the lowering."""

    #: Emit dialect ops for queue bookkeeping calls (AllocTensor/EnQue/...).
    queue_ops: bool = True
    #: Class method / function inlining depth.
    inline_depth: int = 6
    #: Cap on ops per kernel, mirroring ``VisitorOptions.max_ops``.
    max_ops: int = 20000


# ---------------------------------------------------------------------------
# Structural helpers (no regex anywhere)
# ---------------------------------------------------------------------------


def split_template_args(text: str) -> List[str]:
    """Split a rendered template-argument list at its top level.

    ``AscendC::TQue<AscendC::TPosition::VECIN, 2>`` becomes
    ``["AscendC::TPosition::VECIN", "2"]``.  Angle brackets nest; commas
    inside a nested argument are not separators.  An empty list is returned
    when ``text`` carries no argument list.
    """
    start = text.find("<")
    if start < 0:
        return []
    args: List[str] = []
    current: List[str] = []
    depth = 0
    for ch in text[start + 1:]:
        if ch == "<":
            depth += 1
        elif ch == ">":
            if depth == 0:
                if current:
                    args.append("".join(current).strip())
                return args
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    return args


def _last_scope_name(qualified: str) -> str:
    return qualified.rsplit("::", 1)[-1].strip()


def _template_head(type_text: str) -> str:
    """``AscendC::TQue<...>`` -> ``TQue`` (head name, no arguments)."""
    return _last_scope_name(type_text.split("<", 1)[0])


def _normalise_dtype(type_text: str) -> str:
    head = _last_scope_name(type_text.split("<", 1)[0].strip())
    return _DTYPE_ALIASES.get(head, head)


def _dtype_size(dtype: str) -> Optional[int]:
    return _DTYPE_SIZES.get(dtype)


def _children(node: dict, *kinds: str) -> List[dict]:
    if not kinds:
        return list(node.get("inner", []) or [])
    return [c for c in node.get("inner", []) or [] if c.get("kind") in kinds]


def _first_child(node: Optional[dict], *kinds: str) -> Optional[dict]:
    if node is None:
        return None
    for child in node.get("inner", []) or []:
        if not kinds or child.get("kind") in kinds:
            return child
    return None


def _referenced_name(node: dict) -> Optional[str]:
    ref = node.get("referencedDecl") or node.get("foundReferencedDecl") or {}
    name = ref.get("name")
    if isinstance(name, str):
        return name
    return node.get("name")


def _skip_casts(node: dict) -> dict:
    while node.get("kind") in _CAST_KINDS:
        inner = _first_child(node)
        if inner is None:
            return node
        node = inner
    return node


def _stmts(node: Optional[dict]) -> List[dict]:
    if node is None:
        return []
    if node.get("kind") == "CompoundStmt":
        return list(node.get("inner", []) or [])
    return [node]


def _is_strict(opcode: Optional[str]) -> bool:
    return opcode in ("<", ">")


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------


class _Folder:
    """Integer expression folder over clang AST JSON nodes."""

    def __init__(self, ctx: "AstContext", bound: Optional[Dict[str, int]] = None) -> None:
        self.ctx = ctx
        self.bound: Dict[str, int] = dict(bound or {})

    def child(self, extra: Dict[str, int]) -> "_Folder":
        merged = dict(self.bound)
        merged.update(extra)
        return _Folder(self.ctx, merged)

    def fold(self, node: Optional[dict]) -> Optional[int]:
        if node is None:
            return None
        kind = node.get("kind")
        if kind == "IntegerLiteral":
            value = node.get("value")
            return int(value) if value is not None else None
        if kind == "CharacterLiteral":
            return int(node.get("value", 0))
        if kind == "CXXBoolLiteralExpr":
            return 1 if node.get("value") else 0
        if kind == "UnaryExprOrTypeTraitExpr":
            name = node.get("name") or ""
            if name not in ("sizeof", ""):
                return None
            arg = node.get("argType") or {}
            if isinstance(arg, dict):
                qt = arg.get("qualType") or arg.get("desugaredQualType") or ""
                if qt:
                    return _dtype_size(_last_scope_name(qt))
            return None
        if kind == "DeclRefExpr":
            name = _referenced_name(node) or ""
            if name in self.bound:
                return self.bound[name]
            if name in self.ctx.enum_values:
                return self.ctx.enum_values[name]
            return self.ctx.constants.get(name)
        if kind in ("BinaryOperator", "CompoundAssignOperator"):
            return self._fold_binary(node)
        if kind == "UnaryOperator":
            return self._fold_unary(node)
        if kind == "ConditionalOperator":
            kids = _children(node)
            cond = self.fold(kids[0]) if kids else None
            if cond is not None and len(kids) >= 3:
                return self.fold(kids[1]) if cond else self.fold(kids[2])
            return None
        if kind in ("CXXFunctionalCastExpr", "CXXConstructExpr",
                    "CXXTemporaryObjectExpr", "InitListExpr"):
            if kind == "InitListExpr":
                return None
            first = _first_child(node)
            if first is None:
                return None
            return self.fold(first)
        if kind in ("CallExpr", "CXXMemberCallExpr"):
            return self._fold_call(node)
        if kind in _CAST_KINDS:
            inner = _first_child(node)
            return self.fold(inner)
        return None

    def _fold_call(self, node: dict) -> Optional[int]:
        callee = _first_child(node)
        fn: Optional[dict] = None
        if callee is not None:
            ref = _skip_casts(callee).get("referencedDecl") or {}
            decl_id = ref.get("id")
            fn = self.ctx.decls_by_id.get(decl_id or "")
            if fn is None:
                fn = self.ctx.functions.get(ref.get("name") or "")
        if fn is None:
            return None
        args = [_skip_casts(a) for a in _children(node)[1:]]
        return self._fold_constexpr_function(fn, args)

    def _fold_constexpr_function(self, fn: dict, args: List[dict]) -> Optional[int]:
        body = _first_child(fn, "CompoundStmt")
        ret = _first_child(body, "ReturnStmt") if body is not None else None
        if ret is None:
            return None
        expr = _first_child(ret)
        if expr is None:
            return None
        parms = [p.get("name", "") for p in _children(fn, "ParmVarDecl")]
        if not parms:
            return None
        bound: Dict[str, int] = {}
        for name, arg in zip(parms, args):
            if not name:
                continue
            value = self.fold(arg)
            if value is None:
                return None
            bound[name] = value
        if len(bound) != len([p for p in parms if p]):
            return None
        return self.child(bound).fold(_skip_casts(expr))

    def _fold_binary(self, node: dict) -> Optional[int]:
        kids = _children(node)
        if len(kids) < 2:
            return None
        lhs, rhs = self.fold(kids[0]), self.fold(kids[1])
        if lhs is None or rhs is None:
            return None
        op = node.get("opcode", "")
        try:
            table: Dict[str, int] = {
                "+": lhs + rhs, "-": lhs - rhs, "*": lhs * rhs,
                "/": lhs // rhs if rhs else 0, "%": lhs % rhs if rhs else 0,
                "<<": lhs << rhs, ">>": lhs >> rhs,
                "&": lhs & rhs, "|": lhs | rhs, "^": lhs ^ rhs,
            }
            if op in table:
                return table[op] if (rhs or op not in ("/", "%")) else None
            if op == "<":
                return int(lhs < rhs)
            if op == "<=":
                return int(lhs <= rhs)
            if op == ">":
                return int(lhs > rhs)
            if op == ">=":
                return int(lhs >= rhs)
            if op == "==":
                return int(lhs == rhs)
            if op == "!=":
                return int(lhs != rhs)
            if op == "&&":
                return int(bool(lhs) and bool(rhs))
            if op == "||":
                return int(bool(lhs) or bool(rhs))
        except (ValueError, OverflowError):
            return None
        return None

    def _fold_unary(self, node: dict) -> Optional[int]:
        kids = _children(node)
        if not kids:
            return None
        value = self.fold(kids[0])
        if value is None:
            return None
        op = node.get("opcode", "")
        if op == "-":
            return -value
        if op == "+":
            return value
        if op == "!":
            return int(not value)
        if op == "~":
            return ~value
        return None


# ---------------------------------------------------------------------------
# TU indexing
# ---------------------------------------------------------------------------


@dataclass
class AstContext:
    """TU-level indexes shared by every kernel walk."""

    decls_by_id: Dict[str, dict] = field(default_factory=dict)
    enums: Dict[str, Dict[int, str]] = field(default_factory=dict)
    enum_values: Dict[str, int] = field(default_factory=dict)
    constants: Dict[str, int] = field(default_factory=dict)
    records: Dict[str, dict] = field(default_factory=dict)
    methods: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    functions: Dict[str, dict] = field(default_factory=dict)
    hard_event: Dict[int, str] = field(default_factory=dict)
    path: str = ""
    lines: List[str] = field(default_factory=list)


def _is_const_var(node: dict) -> bool:
    if node.get("isConstexpr"):
        return True
    qt = node.get("type", {}).get("qualType", "")
    return qt.strip().startswith("const ")


def _index_unit(ast: dict, path: str, lines: List[str]) -> AstContext:
    ctx = AstContext(path=path, lines=lines)

    def visit(node: dict, record: Optional[str] = None) -> None:
        kind = node.get("kind")
        if kind in ("FunctionDecl", "CXXMethodDecl"):
            name = node.get("name") or ""
            ctx.decls_by_id[node.get("id", "")] = node
            if kind == "CXXMethodDecl" and record:
                ctx.methods[(record, name)] = node
            elif name:
                existing = ctx.functions.get(name)
                if existing is None or _first_child(existing, "CompoundStmt") is None:
                    ctx.functions[name] = node
            return
        if kind == "EnumDecl":
            table: Dict[int, str] = {}
            next_value = 0
            for const in _children(node, "EnumConstantDecl"):
                cname = const.get("name") or ""
                value = const.get("initVal")
                current = int(value) if isinstance(value, int) else next_value
                if cname:
                    table[current] = cname
                    ctx.enum_values[cname] = current
                next_value = current + 1
            name = node.get("name")
            if name:
                ctx.enums[name] = table
                if name == "HardEvent":
                    ctx.hard_event.update(table)
        elif kind == "VarDecl":
            name = node.get("name") or ""
            if name and _is_const_var(node):
                value = _Folder(ctx).fold(_first_child(node))
                if value is not None:
                    ctx.constants[name] = value
        elif kind in ("CXXRecordDecl", "ClassTemplateSpecializationDecl"):
            name = node.get("name") or ""
            if name and not node.get("isImplicit"):
                ctx.records.setdefault(name, node)
                for child in node.get("inner", []) or []:
                    if child.get("kind") in ("CXXMethodDecl", "CXXRecordDecl",
                                             "ClassTemplateSpecializationDecl"):
                        visit(child, name)
                return
        for child in node.get("inner", []) or []:
            visit(child, record)

    visit(ast)
    return ctx


def _has_used_attr(node: dict) -> bool:
    return any(child.get("kind") == "UsedAttr" for child in node.get("inner", []) or [])


def _iter_kernel_entries(ast: dict, all_functions: bool) -> List[dict]:
    """Kernel entry FunctionDecls: ``__global__``-marked (UsedAttr) or all."""
    entries: List[dict] = []
    seen: set = set()

    def visit(node: dict) -> None:
        kind = node.get("kind")
        if kind == "FunctionDecl":
            name = node.get("name") or ""
            marked = _has_used_attr(node) and not name.endswith("Internal")
            has_body = _first_child(node, "CompoundStmt") is not None
            if (marked or all_functions) and has_body and name not in seen:
                seen.add(name)
                entries.append(node)
            return
        for child in node.get("inner", []) or []:
            visit(child)

    visit(ast)
    return entries


# ---------------------------------------------------------------------------
# Per-kernel lowering walker
# ---------------------------------------------------------------------------


@dataclass
class _ArgBind:
    """One inlined-call parameter binding: source expression plus name."""

    expr: Optional[dict] = None
    name: Optional[str] = None
    #: Folded integer value, when the argument constant-folds.
    value: Optional[int] = None


class _KernelLowering:
    """Walks one kernel entry and emits dialect ops."""

    def __init__(self, ctx: AstContext, options: BridgeOptions) -> None:
        self.ctx = ctx
        self.options = options
        self._op_count = 0
        self._loop_seq = 0
        #: Access path -> type text for TQue/TBuf/TPipe objects in scope.
        self.buffer_types: Dict[str, str] = {}
        #: Local variable -> type text (for record/base resolution).
        self.var_types: Dict[str, str] = {}
        #: Parameter bindings of the inlined call stack.
        self.binds: List[Dict[str, _ArgBind]] = []
        #: Records of the inlined method stack (``this`` resolution).
        self.this_stack: List[Optional[str]] = []
        #: Loop induction bindings of the innermost unrolled iteration.
        self.loop_env: Dict[str, int] = {}
        #: Folded values of scalar locals declared during the walk.
        self.scalar_env: Dict[str, int] = {}

    # -- bookkeeping ----------------------------------------------------------

    def _line(self, node: dict) -> int:
        return node_line(node)

    def _snippet(self, node: dict) -> str:
        line = self._line(node)
        if 1 <= line <= len(self.ctx.lines):
            return self.ctx.lines[line - 1].strip()[:160]
        return ""

    def _emit(self, kernel: KernelOp, op) -> bool:
        if self._op_count >= self.options.max_ops:
            return False
        self._op_count += 1
        kernel.ops.append(op)
        return True

    def _folder(self) -> _Folder:
        env = dict(self.scalar_env)
        env.update(self.loop_env)
        for frame in self.binds:
            for name, bind in frame.items():
                if isinstance(bind.value, int):
                    env[name] = bind.value
        return _Folder(self.ctx, env)

    # -- naming ---------------------------------------------------------------

    def _expr_name(self, node: Optional[dict]) -> Optional[str]:
        """The access path a tensor expression refers to, if any."""
        if node is None:
            return None
        node = _skip_casts(node)
        kind = node.get("kind")
        if kind == "DeclRefExpr":
            return self._resolve_ref(_referenced_name(node))
        if kind == "MemberExpr":
            base = _first_child(node)
            base_name = self._expr_name(base)
            member = node.get("name") or ""
            if base_name:
                return f"{base_name}.{member}"
            record = self._this_record()
            if record:
                return f"{record}.{member}"
            return member or None
        if kind == "CXXThisExpr":
            record = self._this_record()
            return record
        if kind == "ArraySubscriptExpr":
            base = _first_child(node)
            if base is not None and base.get("kind") == "CXXMemberCallExpr":
                method = self._call_name(base)
                member_base = self._member_base(base)
                base_path = self._expr_name(member_base)
                if method == "Get" and base_path:
                    return f"{base_path}.view"
            return self._expr_name(base)
        if kind in ("BinaryOperator", "CompoundAssignOperator"):
            # Pointer arithmetic: name the base object being offset.
            for child in _children(node):
                found = self._expr_name(child)
                if found:
                    return found
            return None
        return None

    def _resolve_ref(self, name: Optional[str]) -> Optional[str]:
        """Resolve a parameter reference through the inline-bind stack."""
        if name is None:
            return None
        for frame in reversed(self.binds):
            bind = frame.get(name)
            if bind is not None:
                return bind.name or self._expr_name(bind.expr)
        return name

    def _this_record(self) -> Optional[str]:
        return self.this_stack[-1] if self.this_stack else None

    def _type_text(self, node: Optional[dict]) -> str:
        if node is None:
            return ""
        t = node.get("type", {})
        return t.get("qualType") or t.get("desugaredQualType", "")

    def _position_of(self, type_text: str) -> Optional[str]:
        args = split_template_args(type_text)
        if not args:
            return None
        return _last_scope_name(args[0]) or None

    def _depth_of(self, type_text: str) -> int:
        args = split_template_args(type_text)
        if len(args) < 2:
            return 1
        try:
            return int(args[1].strip())
        except ValueError:
            return 1

    def _dtype_of_tensor_type(self, type_text: str) -> Optional[str]:
        args = split_template_args(type_text)
        if not args:
            return None
        return _normalise_dtype(args[0])

    # -- tensor registry ------------------------------------------------------

    def register_tensor(self, kernel: KernelOp, name: str, var_or_parm: dict,
                        dtype: Optional[str], origin: str, line: int,
                        type_text: str = "") -> None:
        entry = kernel.tensors.setdefault(name, {
            "name": name, "dtype": dtype, "origin": origin, "line": line,
            "type": type_text or self._type_text(var_or_parm),
        })
        if dtype and not entry.get("dtype"):
            entry["dtype"] = dtype

    # -- statements -----------------------------------------------------------

    def walk_body(self, kernel: KernelOp, body: Optional[dict],
                  core: Optional[CoreType], depth: int,
                  loop_depth: int, conditional: bool) -> None:
        for stmt in _stmts(body):
            self.walk_stmt(kernel, stmt, core, depth, loop_depth, conditional)

    def walk_stmt(self, kernel: KernelOp, stmt: Optional[dict],
                  core: Optional[CoreType], depth: int,
                  loop_depth: int, conditional: bool) -> None:
        if stmt is None or depth > self.options.inline_depth:
            return
        kind = stmt.get("kind")
        if kind == "CompoundStmt":
            self.walk_body(kernel, stmt, core, depth, loop_depth, conditional)
        elif kind == "DeclStmt":
            for var in _children(stmt, "VarDecl"):
                self._walk_var_decl(kernel, var, core, depth, loop_depth, conditional)
        elif kind == "IfStmt":
            self._walk_if(kernel, stmt, core, depth, loop_depth, conditional)
        elif kind in ("ForStmt", "WhileStmt", "DoStmt", "CXXForRangeStmt"):
            self._walk_loop(kernel, stmt, core, depth, loop_depth, conditional)
        elif kind in ("CallExpr", "CXXMemberCallExpr"):
            self._walk_call(kernel, stmt, core, depth, loop_depth, conditional)
        elif kind in ("BinaryOperator", "CompoundAssignOperator", "ReturnStmt",
                      "CXXConstructExpr", "InitListExpr", "ExpressionStmt"):
            pass  # scalar bookkeeping only
        else:
            for child in stmt.get("inner", []) or []:
                if child.get("kind") in _STMT_KINDS:
                    self.walk_stmt(kernel, child, core, depth, loop_depth, conditional)

    def _walk_var_decl(self, kernel: KernelOp, var: dict,
                       core: Optional[CoreType], depth: int,
                       loop_depth: int, conditional: bool) -> None:
        name = var.get("name") or ""
        if not name:
            return
        sugar = self._type_text(var)
        desugar = var.get("type", {}).get("desugaredQualType", "") or sugar
        head = _template_head(sugar)
        init = _first_child(var)
        init = _skip_casts(init) if init is not None else None
        self.var_types[name] = sugar
        # Scalar locals fold into the walk-time environment so later
        # expressions (event selectors, sizes) see this iteration's value.
        if init is not None:
            folded = self._folder().fold(init)
            if folded is not None:
                self.scalar_env[name] = folded

        if head in ("TQue", "TBuf", "TPipe"):
            self.buffer_types[name] = sugar
            return

        tensor_head = _template_head(desugar)
        if head in ("LocalTensor", "GlobalTensor") or \
                (sugar == "auto" and tensor_head in ("LocalTensor", "GlobalTensor")):
            effective = desugar if head not in ("LocalTensor", "GlobalTensor") else sugar
            dtype = self._dtype_of_tensor_type(effective)
            origin = "declaration"
            if init is not None and init.get("kind") in ("CXXMemberCallExpr", "CallExpr"):
                method = self._call_name(init)
                origin = f"queue:{method}" if method else "declaration"
            self.register_tensor(kernel, name, var, dtype, origin,
                                 self._line(var), type_text=effective)
            if self.options.queue_ops and init is not None and \
                    init.get("kind") in ("CXXMemberCallExpr", "CallExpr"):
                self._walk_call(kernel, init, core, depth, loop_depth,
                                conditional, result_name=name)
            return

        if init is not None and init.get("kind") in ("CXXMemberCallExpr", "CallExpr"):
            if self.options.queue_ops:
                self._walk_call(kernel, init, core, depth, loop_depth,
                                conditional, result_name=name)

    # -- control flow ---------------------------------------------------------

    def _walk_if(self, kernel: KernelOp, stmt: dict, core: Optional[CoreType],
                 depth: int, loop_depth: int, conditional: bool) -> None:
        kids = _children(stmt)
        cond = next((k for k in kids if k.get("role") == "cond"), None)
        then = next((k for k in kids if k.get("role") == "then"), None)
        els = next((k for k in kids if k.get("role") == "else"), None)
        if cond is None:
            # clang 15 emits cond/then/else without roles, in order.
            stmt_kids = [k for k in kids if k.get("kind") in _STMT_KINDS or
                         k.get("kind") in ("ImplicitCastExpr", "BinaryOperator",
                                           "DeclRefExpr", "IntegerLiteral",
                                           "ParenExpr", "CXXBoolLiteralExpr",
                                           "UnaryOperator", "DeclStmt", "CallExpr")]
            if stmt_kids:
                cond = stmt_kids[0]
                then = stmt_kids[1] if len(stmt_kids) > 1 else then
                els = stmt_kids[2] if len(stmt_kids) > 2 else els

        sentinel = self._sentinel_of(cond)
        if sentinel is not None and core is None:
            arm_core = CoreType.AIC if sentinel == _SENTINEL_AIC else CoreType.AIV
            region = CoreRegionOp(core_type=arm_core, line=self._line(stmt),
                                  text=self._snippet(stmt), loop_depth=loop_depth,
                                  conditional=True)
            holder = KernelOp(name=f"{arm_core.value}-region")
            self.walk_body(holder, then, arm_core, depth + 1, loop_depth, True)
            region.ops = holder.ops
            region.condition = "ASCEND_IS_" + arm_core.value
            if self._emit(kernel, region):
                # Regions are part of this kernel: their loops and tensor
                # registry entries belong to it too.
                kernel.loops.extend(holder.loops)
                for tname, tinfo in holder.tensors.items():
                    kernel.tensors.setdefault(tname, tinfo)
            if els is not None:
                if els.get("kind") == "IfStmt" and self._inner_if_sentinel(els) is not None:
                    self.walk_stmt(kernel, els, core, depth, loop_depth, conditional)
                else:
                    other = CoreType.AIV if arm_core is CoreType.AIC else CoreType.AIC
                    region_else = CoreRegionOp(core_type=other, line=self._line(els),
                                               text=self._snippet(els),
                                               loop_depth=loop_depth, conditional=True)
                    holder = KernelOp(name=f"{other.value}-region")
                    self.walk_body(holder, els, other, depth + 1, loop_depth, True)
                    region_else.ops = holder.ops
                    region_else.condition = "ASCEND_IS_" + other.value
                    if self._emit(kernel, region_else):
                        kernel.loops.extend(holder.loops)
                        for tname, tinfo in holder.tensors.items():
                            kernel.tensors.setdefault(tname, tinfo)
            return

        # Static condition: during an exact unroll the guard folds, and only
        # the live arm is walked (mirroring the tree-sitter replay).
        value = self._folder().fold(cond) if cond is not None else None
        if value is not None:
            if value:
                self.walk_stmt(kernel, then, core, depth, loop_depth, True)
            elif els is not None:
                self.walk_stmt(kernel, els, core, depth, loop_depth, True)
            return

        if then is not None:
            self.walk_stmt(kernel, then, core, depth, loop_depth, True)
        if els is not None:
            self.walk_stmt(kernel, els, core, depth, loop_depth, True)

    def _inner_if_sentinel(self, els: Optional[dict]) -> Optional[str]:
        if els is None or els.get("kind") != "IfStmt":
            return None
        kids = _children(els)
        cond = kids[0] if kids else None
        return self._sentinel_of(cond)

    def _sentinel_of(self, cond: Optional[dict]) -> Optional[str]:
        if cond is None:
            return None
        node = _skip_casts(cond)
        if node.get("kind") == "UnaryOperator" and node.get("opcode") == "!":
            return self._sentinel_of(_first_child(node))
        if node.get("kind") == "DeclRefExpr":
            name = _referenced_name(node) or ""
            if name in (_SENTINEL_AIC, _SENTINEL_AIV):
                return name
        return None

    def _walk_loop(self, kernel: KernelOp, stmt: dict, core: Optional[CoreType],
                   depth: int, loop_depth: int, conditional: bool) -> None:
        kids = [k for k in _children(stmt) if k.get("kind")]
        # Classify ForStmt children by kind: clang 15 emits positional
        # placeholders (kindless nodes) that shift fixed indices.
        cond = next((k for k in kids
                     if k.get("kind") in ("BinaryOperator", "ImplicitCastExpr",
                                          "DeclRefExpr", "IntegerLiteral",
                                          "CXXBoolLiteralExpr", "ParenExpr",
                                          "UnaryOperator", "CallExpr")
                     and k.get("type", {}).get("qualType") == "bool"), None)
        if cond is None and stmt.get("kind") == "ForStmt":
            cond = next((k for k in kids if k.get("kind") == "BinaryOperator"
                         and k.get("type", {}).get("qualType", "").startswith("bool")), None)
        body: Optional[dict] = None
        for k in reversed(kids):
            if k.get("kind") in _STMT_KINDS and k is not cond:
                body = k
                break
        inc = next((k for k in kids
                    if k.get("kind") in ("UnaryOperator", "CompoundAssignOperator")
                    and k is not body), None)
        init: Optional[dict] = None
        for k in kids:
            if k is cond or k is body or k is inc:
                break
            if k.get("kind") in ("DeclStmt", "BinaryOperator", "CompoundAssignOperator"):
                init = k
                break

        folder = self._folder()
        induction: Optional[str] = None
        start: Optional[int] = None
        step: Optional[int] = None
        trip: Optional[int] = None

        if init is not None and init.get("kind") == "DeclStmt":
            var = _first_child(init, "VarDecl")
            if var is not None:
                induction = var.get("name")
                start = folder.fold(_first_child(var))
        elif init is not None:
            init_kids = _children(init)
            induction = self._expr_name(init_kids[0]) if init_kids else None
            start = folder.fold(init_kids[1]) if len(init_kids) > 1 else None
        cond_node = _skip_casts(cond) if cond is not None else None
        if cond_node is not None and cond_node.get("kind") == "BinaryOperator" and \
                cond_node.get("opcode") in ("<", "<=", ">", ">="):
            cond_kids = _children(cond_node)
            bound = folder.fold(cond_kids[1]) if len(cond_kids) > 1 else None
            base = folder.fold(cond_kids[0]) if cond_kids else None
            step = self._loop_step(inc, induction, folder)
            if bound is not None and step:
                lo = start if start is not None else base
                if lo is not None:
                    span = bound - lo
                    if cond_node.get("opcode") in ("<", ">"):
                        trip = (span + step - 1) // step if span > 0 else 0
                    else:
                        trip = span // step + 1 if span >= 0 else 0

        self._loop_seq += 1
        loop_id = self._loop_seq
        kernel.loops.append({
            "id": loop_id, "induction": induction, "start": start, "step": step,
            "trip_count": trip, "loop_depth": loop_depth,
            "line": self._line(stmt), "header": self._snippet(stmt),
        })

        if trip is not None and 0 < trip <= UNROLL_LIMIT:
            base_val = start if start is not None else 0
            step_val = step or 1
            saved_env = dict(self.loop_env)
            saved_scalars = dict(self.scalar_env)
            for i in range(trip):
                self.scalar_env = dict(saved_scalars)
                if induction:
                    self.loop_env = dict(saved_env)
                    self.loop_env[induction] = base_val + i * step_val
                self.walk_body(kernel, body, core, depth, loop_depth + 1, conditional)
            self.loop_env = saved_env
            self.scalar_env = saved_scalars
            return

        mark = len(kernel.ops)
        self.walk_body(kernel, body, core, depth, loop_depth + 1, conditional)
        for op in kernel.ops[mark:]:
            if op.loop_id is None:
                op.loop_id = loop_id

    def _loop_step(self, inc: Optional[dict], induction: Optional[str],
                   folder: _Folder) -> Optional[int]:
        if inc is None:
            return None
        node = _skip_casts(inc)
        if node.get("kind") == "UnaryOperator":
            op = node.get("opcode")
            if op == "++":
                return 1
            if op == "--":
                return -1
        if node.get("kind") in ("BinaryOperator", "CompoundAssignOperator"):
            kids = _children(node)
            lhs = self._expr_name(kids[0]) if kids else None
            if lhs != induction or len(kids) < 2:
                return None
            value = folder.fold(kids[1])
            if value is None:
                return None
            opcode = node.get("opcode", "")
            if opcode in ("+=", "="):
                return value
            if opcode == "-=":
                return -value
        return None

    # -- calls ----------------------------------------------------------------

    def _call_name(self, call: dict) -> Optional[str]:
        kind = call.get("kind")
        if kind == "CXXMemberCallExpr":
            member = next((c for c in call.get("inner", []) or []
                           if c.get("kind") in ("MemberExpr",
                                                "CXXDependentScopeMemberExpr")), None)
            return member.get("name") if member is not None else None
        first = _first_child(call)
        if first is None:
            return None
        skipped = _skip_casts(first)
        if skipped.get("kind") in ("DeclRefExpr", "MemberExpr"):
            return _referenced_name(skipped)
        return None

    def _member_base(self, call: dict) -> Optional[dict]:
        member = next((c for c in call.get("inner", []) or []
                       if c.get("kind") in ("MemberExpr",
                                            "CXXDependentScopeMemberExpr")), None)
        return _first_child(member) if member is not None else None

    def _callee_decl(self, call: dict) -> Optional[dict]:
        first = _first_child(call)
        if first is None:
            return None
        ref = _skip_casts(first).get("referencedDecl") or {}
        return self.ctx.decls_by_id.get(ref.get("id") or "")

    def _args(self, call: dict) -> List[dict]:
        kids = [k for k in call.get("inner", []) or [] if k.get("kind") != "Attr"]
        return kids[1:] if kids else []

    def _template_int_argument(self, decl: Optional[dict]) -> Optional[int]:
        if decl is None:
            return None
        for child in decl.get("inner", []) or []:
            if child.get("kind") == "TemplateArgument" and \
                    isinstance(child.get("value"), int):
                return int(child["value"])
        return None

    def _base_record(self, base: Optional[dict]) -> Optional[str]:
        """Record name of a member-call base expression, when it is a class."""
        if base is None:
            return None
        node = _skip_casts(base)
        if node.get("kind") == "CXXThisExpr":
            return self._this_record()
        qt = self._type_text(node)
        head = _template_head(qt)
        if head in ("LocalTensor", "GlobalTensor", "TQue", "TBuf", "TPipe"):
            return None
        if head and head in self.ctx.records:
            return head
        return None

    def _lookup_buffer_type(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if path in self.buffer_types:
            return self.buffer_types[path]
        if "." in path:
            owner, member = path.rsplit(".", 1)
            record = self._record_of_var(owner)
            if record and record in self.ctx.records:
                for field_node in _children(self.ctx.records[record], "FieldDecl"):
                    if field_node.get("name") == member:
                        return self._type_text(field_node)
        return None

    def _record_of_var(self, var_name: str) -> Optional[str]:
        qt = self.var_types.get(var_name)
        if qt:
            head = _template_head(qt)
            if head in self.ctx.records:
                return head
        return var_name if var_name in self.ctx.records else None

    def _walk_call(self, kernel: KernelOp, call: dict, core: Optional[CoreType],
                   depth: int, loop_depth: int, conditional: bool,
                   result_name: Optional[str] = None) -> None:
        name = self._call_name(call)
        if not name:
            return
        line = self._line(call)
        snippet = self._snippet(call)
        args = self._args(call)

        if call.get("kind") == "CXXMemberCallExpr":
            base = self._member_base(call)
            base_path = self._expr_name(base)
            self._walk_member_call(kernel, call, name, base, base_path, args,
                                   core, depth, loop_depth, conditional,
                                   result_name, line, snippet)
            return

        # Free functions.
        if name in ("SetFlag", "WaitFlag"):
            route_value = self._template_int_argument(self._callee_decl(call))
            route = self.ctx.hard_event.get(route_value) if route_value is not None else None
            event = self._folder().fold(args[0]) if args else None
            op_class = SetFlagOp if name == "SetFlag" else WaitFlagOp
            self._emit(kernel, op_class(
                pipe_route=route or "", event_id=event if event is not None else 0,
                line=line, text=snippet, loop_depth=loop_depth,
                conditional=conditional))
            return

        if name in ("set_flag", "wait_flag") and len(args) >= 3:
            src = self._pipe_arg(args[0])
            dst = self._pipe_arg(args[1])
            event = self._folder().fold(args[2])
            op_class = SetFlagOp if name == "set_flag" else WaitFlagOp
            self._emit(kernel, op_class(
                pipe_route=f"{src}_{dst}" if src and dst else "",
                event_id=event if event is not None else 0,
                line=line, text=snippet, loop_depth=loop_depth,
                conditional=conditional))
            return

        if name in ("PipeBarrier", "pipe_barrier"):
            target = self._pipe_arg(args[0]) if args else "ALL"
            self._emit(kernel, BarrierOp(target=target or "ALL", line=line,
                                         text=snippet, loop_depth=loop_depth,
                                         conditional=conditional))
            return

        if name == "DataCopy":
            self._walk_data_copy(kernel, args, line, snippet, loop_depth, conditional)
            return

        if name == "Mmad":
            dst = self._value_of(args[0]) if len(args) > 0 else SsaValue("?")
            a = self._value_of(args[1]) if len(args) > 1 else SsaValue("?")
            b = self._value_of(args[2]) if len(args) > 2 else SsaValue("?")
            m = n = k = 0
            if len(args) > 3:
                ctor = self._find_construct(args[3])
                if ctor is not None:
                    ints = [self._folder().fold(a_) for a_ in _children(ctor)]
                    ints = [v for v in ints if v is not None]
                    if len(ints) >= 3:
                        m, n, k = ints[0], ints[1], ints[2]
            self._emit(kernel, MmadOp(op_a=a, op_b=b, op_c=dst, api_name=name,
                                      m=m, n=n, k=k, line=line, text=snippet,
                                      loop_depth=loop_depth, conditional=conditional))
            return

        if name == "mad_mx":
            dst = self._value_of(args[0]) if args else SsaValue("?")
            a = self._value_of(args[3]) if len(args) > 3 else SsaValue("?")
            b = self._value_of(args[5]) if len(args) > 5 else SsaValue("?")
            self._emit(kernel, MmadOp(op_a=a, op_b=b, op_c=dst, api_name=name,
                                      line=line, text=snippet, loop_depth=loop_depth,
                                      conditional=conditional))
            return

        if name in ("load_cbuf_to_ca_s4", "load_cbuf_to_ca", "load_cbuf_to_cb",
                    "load_cbuf_to_cb_s4", "load_cbuf_to_cb_transpose_s4"):
            dst = self._value_of(args[0]) if args else SsaValue("?")
            src = self._value_of(args[1]) if len(args) > 1 else SsaValue("?")
            self._emit(kernel, MteCopyOp(src=src, dst=dst, pipe_route="MTE1",
                                         api_name=name, line=line, text=snippet,
                                         loop_depth=loop_depth,
                                         conditional=conditional))
            return

        if name == "Fixpipe":
            dst = self._value_of(args[0]) if args else SsaValue("?")
            src = self._value_of(args[1]) if len(args) > 1 else SsaValue("?")
            self._emit(kernel, MteCopyOp(src=src, dst=dst, pipe_route="FIX",
                                         api_name=name, line=line, text=snippet,
                                         loop_depth=loop_depth,
                                         conditional=conditional))
            return

        if name in VECTOR_OPS:
            dst = self._value_of(args[0]) if args else SsaValue("?")
            if name == "Cast":
                srcs: Tuple[SsaValue, ...] = (
                    self._value_of(args[1]),) if len(args) > 1 else ()
            else:
                srcs = tuple(self._value_of(a) for a in args[1:3])
            count = self._folder().fold(args[-1]) if args else 0
            self._emit(kernel, VectorOp(opcode=name, dst=dst, srcs=srcs,
                                        elem_count=count or 0, line=line,
                                        text=snippet, loop_depth=loop_depth,
                                        conditional=conditional))
            return

        # User free function with a body: inline.
        fn = self.ctx.functions.get(name)
        if fn is not None and depth < self.options.inline_depth:
            body = _first_child(fn, "CompoundStmt")
            if body is not None:
                self._inline_call(kernel, fn, body, args, None, core, depth + 1,
                                  loop_depth, conditional)

    def _walk_member_call(self, kernel: KernelOp, call: dict, method: str,
                          base: Optional[dict], base_path: Optional[str],
                          args: List[dict], core: Optional[CoreType], depth: int,
                          loop_depth: int, conditional: bool,
                          result_name: Optional[str], line: int,
                          snippet: str) -> None:
        if method == "InitBuffer" and base_path:
            target = self._expr_name(args[0]) if args else None
            length = self._folder().fold(args[-1]) if args else None
            depth_n = 1
            if len(args) >= 3:
                folded = self._folder().fold(args[1])
                if folded is not None:
                    depth_n = folded
            buffer_type = self._lookup_buffer_type(target)
            position = self._position_of(buffer_type) if buffer_type else None
            space = _POSITION_SPACES.get(position or "", MemorySpace.UB)
            order = sum(1 for op in kernel.ops if isinstance(op, AllocBufferOp))
            self._emit(kernel, AllocBufferOp(
                buffer=target or "", space=space, byte_size=length or 0,
                depth=depth_n, order=order, line=line, text=snippet,
                loop_depth=loop_depth, conditional=conditional))
            return

        if method == "Get" and base_path:
            dtype = self._dtype_of_tensor_type(self._type_text(call)) or "?"
            buffer_type = self._lookup_buffer_type(base_path)
            position = self._position_of(buffer_type) if buffer_type else None
            space = _POSITION_SPACES.get(position or "", MemorySpace.UB)
            value = SsaValue(result_name or f"{base_path}_view",
                             type_str=f"LocalTensor<{dtype}>",
                             tensor=result_name or f"{base_path}_view")
            self._emit(kernel, GetTensorOp(results=[value], buffer=base_path,
                                           dtype=dtype, space=space, line=line,
                                           text=snippet, loop_depth=loop_depth,
                                           conditional=conditional))
            return

        if method in ("AllocTensor", "DeQue") and base_path:
            if not self.options.queue_ops:
                return
            dtype = self._dtype_of_tensor_type(self._type_text(call)) or ""
            value = SsaValue(result_name or f"{base_path}.{method.lower()}",
                             type_str=f"LocalTensor<{dtype}>" if dtype else "",
                             tensor=result_name)
            op_class = AllocTensorOp if method == "AllocTensor" else DeQueOp
            self._emit(kernel, op_class(results=[value], queue=base_path,
                                        dtype=dtype, line=line, text=snippet,
                                        loop_depth=loop_depth,
                                        conditional=conditional))
            return

        if method in ("EnQue", "FreeTensor") and base_path:
            if not self.options.queue_ops:
                return
            tensor_name = self._expr_name(args[0]) if args else None
            handle = SsaValue(tensor_name or "?", tensor=tensor_name)
            op_class = EnQueOp if method == "EnQue" else FreeTensorOp
            self._emit(kernel, op_class(results=[handle], queue=base_path,
                                        line=line, text=snippet,
                                        loop_depth=loop_depth,
                                        conditional=conditional))
            return

        # Method on a user class: inline the body when available.
        record = self._base_record(base)
        if record:
            method_decl = self.ctx.methods.get((record, method))
            if method_decl is not None and depth < self.options.inline_depth:
                body = _first_child(method_decl, "CompoundStmt")
                if body is not None:
                    self._inline_call(kernel, method_decl, body, args, record,
                                      core, depth + 1, loop_depth, conditional)
                    return

    def _inline_call(self, kernel: KernelOp, fn: dict, body: dict,
                     args: List[dict], record: Optional[str],
                     core: Optional[CoreType], depth: int,
                     loop_depth: int, conditional: bool) -> None:
        parms = _children(fn, "ParmVarDecl")
        frame: Dict[str, _ArgBind] = {}
        for parm, arg in zip(parms, [_skip_casts(a) for a in args]):
            pname = parm.get("name") or ""
            if not pname:
                continue
            skipped = _skip_casts(arg)
            frame[pname] = _ArgBind(
                expr=skipped,
                name=self._expr_name(skipped),
                value=self._folder().fold(skipped),
            )
        self.binds.append(frame)
        self.this_stack.append(record)
        saved_env = dict(self.loop_env)
        saved_scalars = dict(self.scalar_env)
        self.loop_env = {}
        self.scalar_env = dict(saved_scalars)
        self.walk_body(kernel, body, core, depth, loop_depth, conditional)
        self.loop_env = saved_env
        self.scalar_env = saved_scalars
        self.this_stack.pop()
        self.binds.pop()

    def _walk_data_copy(self, kernel: KernelOp, args: List[dict], line: int,
                        snippet: str, loop_depth: int, conditional: bool) -> None:
        dst_arg = args[0] if args else None
        src_arg = args[1] if len(args) > 1 else None
        count = self._folder().fold(args[2]) if len(args) > 2 else None
        dst = self._value_of(dst_arg)
        src = self._value_of(src_arg)
        dtype = self._copy_dtype(dst_arg, src_arg)
        elem_size = _dtype_size(dtype or "") if dtype else None
        length = (count or 0) * (elem_size or 1)
        route = self._copy_route(dst_arg, src_arg)
        self._emit(kernel, MteCopyOp(src=src, dst=dst, length_bytes=length,
                                     pipe_route=route, api_name="DataCopy",
                                     line=line, text=snippet,
                                     loop_depth=loop_depth,
                                     conditional=conditional))

    def _copy_dtype(self, dst_arg: Optional[dict],
                    src_arg: Optional[dict]) -> Optional[str]:
        for arg in (dst_arg, src_arg):
            if arg is None:
                continue
            text = self._type_text(arg)
            if "Tensor" in text:
                dtype = self._dtype_of_tensor_type(text)
                if dtype:
                    return dtype
            skipped = _skip_casts(arg)
            qt = skipped.get("type", {}).get("qualType", "")
            if "*" in qt:
                return _normalise_dtype(qt.replace("*", "").strip())
        return None

    def _copy_route(self, dst_arg: Optional[dict], src_arg: Optional[dict]) -> str:
        src_gm = self._is_gm_expr(src_arg)
        dst_gm = self._is_gm_expr(dst_arg)
        if src_gm and not dst_gm:
            return "MTE2"
        if dst_gm and not src_gm:
            return "MTE3"
        return "MTE1"

    def _is_gm_expr(self, arg: Optional[dict]) -> bool:
        """GM-ness of a copy operand: kernel pointer parameters and their
        arithmetic are GM by Ascend C convention; GlobalTensor is GM too."""
        if arg is None:
            return False
        node = _skip_casts(arg)
        kind = node.get("kind")
        if kind == "DeclRefExpr":
            qt = node.get("type", {}).get("qualType", "")
            if "GlobalTensor" in qt:
                return True
            if "*" in qt and "Tensor" not in qt:
                return True
            return False
        if kind == "CXXMemberCallExpr":
            return "GlobalTensor" in self._type_text(node)
        if kind == "ArraySubscriptExpr":
            inner = _first_child(node)
            if inner is not None and inner.get("kind") == "CXXMemberCallExpr":
                return "GlobalTensor" in self._type_text(inner)
            return self._is_gm_expr(inner)
        if kind == "BinaryOperator" and node.get("opcode") == "+":
            return any(self._is_gm_expr(k) for k in _children(node))
        return False

    def _value_of(self, arg: Optional[dict]) -> SsaValue:
        if arg is None:
            return SsaValue("?")
        node = _skip_casts(arg)
        if node.get("kind") == "CXXMemberCallExpr":
            method = self._call_name(node)
            base = self._member_base(node)
            base_path = self._expr_name(base)
            if method and base_path:
                suffix = "view" if method == "Get" else method
                return SsaValue(f"{base_path}.{suffix}",
                                tensor=f"{base_path}.{suffix}")
        if node.get("kind") == "ArraySubscriptExpr":
            base = SsaValue("?")
            element_offset: Optional[int] = None
            kids = _children(node)
            if kids:
                base = self._value_of(kids[0])
            if len(kids) > 1:
                element_offset = self._folder().fold(_skip_casts(kids[1]))
            if element_offset:
                return SsaValue(f"{base.name}+{element_offset}",
                                tensor=base.tensor, type_str=base.type_str,
                                line=base.line)
            return base
        name = self._expr_name(node)
        if name:
            return SsaValue(name, tensor=name)
        value = self._folder().fold(node)
        if value is not None:
            return SsaValue(str(value))
        return SsaValue("?")

    def _pipe_arg(self, arg: Optional[dict]) -> Optional[str]:
        if arg is None:
            return None
        node = _skip_casts(arg)
        if node.get("kind") == "DeclRefExpr":
            name = _referenced_name(node) or ""
            if name.startswith("PIPE_"):
                return name[len("PIPE_"):]
        return None

    def _find_construct(self, node: dict) -> Optional[dict]:
        node = _skip_casts(node)
        if node.get("kind") in ("CXXConstructExpr", "CXXTemporaryObjectExpr",
                                "CXXFunctionalCastExpr"):
            return node
        for child in node.get("inner", []) or []:
            found = self._find_construct(child)
            if found is not None:
                return found
        return None


# ---------------------------------------------------------------------------
# Public lowering entry point
# ---------------------------------------------------------------------------


def lower_module(ast: dict, path: str, source: str,
                 options: Optional[BridgeOptions] = None,
                 all_functions: bool = False) -> AscendModule:
    """Lower one extraction AST into an :class:`AscendModule`."""
    opts = options or BridgeOptions()
    lines = source.splitlines()
    ctx = _index_unit(ast, path, lines)

    module = AscendModule(name=path.replace("\\", "/").rsplit("/", 1)[-1],
                          hard_event_routes=dict(ctx.hard_event))
    module.constants.update(ctx.constants)

    for entry in _iter_kernel_entries(ast, all_functions):
        kernel = KernelOp(name=entry.get("name") or "kernel",
                          line=node_line(entry))
        walker = _KernelLowering(ctx, opts)
        _collect_entry_context(walker, ast, entry, path)

        for parm in _children(entry, "ParmVarDecl"):
            name = parm.get("name") or ""
            if not name:
                continue
            qt = walker._type_text(parm)
            kernel.params.append(SsaValue(name, type_str=qt, tensor=name,
                                          line=node_line(parm)))
            walker.var_types[name] = qt
            if "*" in qt and "Tensor" not in qt:
                walker.register_tensor(
                    kernel, name, parm,
                    _normalise_dtype(qt.replace("*", "").strip()),
                    "kernel parameter", node_line(parm), type_text=qt)

        walker.walk_body(kernel, _first_child(entry, "CompoundStmt"),
                         core=None, depth=1, loop_depth=0, conditional=False)
        module.kernels.append(kernel)
    return module


def _collect_entry_context(walker: _KernelLowering, ast: dict, entry: dict,
                           path: str) -> None:
    """Index the buffer/record context visible to one kernel entry.

    Only main-file declarations are indexed: buffer objects and classes in
    the kernel's own translation unit.  Template instantiations living in
    headers are already covered by the TU-level context.
    """

    def visit(node: dict) -> None:
        kind = node.get("kind")
        if kind in ("FunctionDecl", "CXXMethodDecl"):
            # Other function bodies are reached through inlining, not scopes.
            return
        if kind == "VarDecl":
            name = node.get("name") or ""
            if name and is_main_file(node, path):
                qt = walker._type_text(node)
                walker.var_types[name] = qt
                if _template_head(qt) in ("TQue", "TBuf", "TPipe"):
                    walker.buffer_types[name] = qt
            return
        if kind in ("CXXRecordDecl", "ClassTemplateSpecializationDecl"):
            name = node.get("name") or ""
            if name and is_main_file(node, path):
                for field_node in _children(node, "FieldDecl"):
                    fname = field_node.get("name") or ""
                    if fname:
                        walker.buffer_types[f"{name}.{fname}"] = \
                            walker._type_text(field_node)
                for child in node.get("inner", []) or []:
                    visit(child)
            return
        for child in node.get("inner", []) or []:
            visit(child)

    visit(ast)
