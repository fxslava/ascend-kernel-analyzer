"""Lower tree-sitter expression nodes into the symbolic integer IR.

Address arithmetic in static-tensor-programming kernels is overwhelmingly
``constexpr`` folding::

    constexpr uint32_t TILE_ELEMS = 256;
    constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(half);
    constexpr uint32_t UB_PING    = 0;
    constexpr uint32_t UB_PONG    = UB_PING + TILE_BYTES;

so the analyzer evaluates these properly instead of pattern-matching
literals.  Anything that genuinely is not a compile-time constant - a loop
induction variable, a host-chosen tiling factor - becomes a
:class:`~ascend_analyzer.symbolic.Var`, carrying inferred bounds where the
parser knows them, and the solver takes it from there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from tree_sitter import Node

from ..symbolic import BinOp, Const, Expr, UnOp, Var, dtype_size, simplify

__all__ = ["ConstEnv", "ExpressionEvaluator", "canonical_accessor"]


_INT_SUFFIX_RE = re.compile(r"[uUlLzZ]+$")
#: Whitespace around a member accessor, so ``a -> b`` canonicalises to ``a->b``.
_ACCESSOR_WS_RE = re.compile(r"\s*(->|\.)\s*")
#: Integral target types of a value-preserving cast.
_INT_TYPE_WORDS = (
    r"(?:u?int(?:8|16|32|64)_t|size_t|ssize_t|ptrdiff_t|"
    r"(?:(?:unsigned|signed)\s+)?(?:char|short|int|long(?:\s+long)?)|"
    r"unsigned|signed)"
)
#: ``static_cast<int64_t>(x)`` or the functional ``int64_t(x)``.  Deliberately
#: excludes ``reinterpret_cast`` (reinterprets an address - must stay opaque)
#: and floating-point targets (truncation is not value-preserving).
_INT_CAST_RE = re.compile(
    rf"^(?:(?:static_cast|const_cast)\s*<\s*(?:const\s+)?{_INT_TYPE_WORDS}\s*>"
    rf"|{_INT_TYPE_WORDS})$"
)

_BIN_OPS = frozenset({"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"})
_UN_OPS = frozenset({"-", "+", "~"})
#: Comparison and logical operators fold only when both sides are constant;
#: they never become symbolic expressions (the solver does not model them).
_CMP_OPS = {
    "<": lambda a, b: int(a < b),
    "<=": lambda a, b: int(a <= b),
    ">": lambda a, b: int(a > b),
    ">=": lambda a, b: int(a >= b),
    "==": lambda a, b: int(a == b),
    "!=": lambda a, b: int(a != b),
}


def _strip_type_text(text: str) -> str:
    """Normalise a type operand: ``"( half )"`` -> ``"half"``."""
    cleaned = text.strip()
    while cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _parse_int_literal(text: str) -> Optional[int]:
    """Parse a C integer literal, tolerating digit separators and suffixes."""
    raw = text.strip().replace("'", "")
    raw = _INT_SUFFIX_RE.sub("", raw)
    if not raw:
        return None
    try:
        # int(x, 0) handles 0x / 0b / 0o and plain decimal, but rejects legacy
        # octal like 0755, so retry that explicitly.
        return int(raw, 0)
    except ValueError:
        pass
    if re.fullmatch(r"0[0-7]+", raw):
        try:
            return int(raw, 8)
        except ValueError:
            return None
    return None


@dataclass
class ConstEnv:
    """A chained environment of compile-time integer constants.

    Lookups walk outwards from the innermost scope.  Loop induction variables
    are registered separately as *bounded free variables* rather than
    constants, so expressions like ``i * TILE_BYTES`` stay symbolic but remain
    usefully bounded for the solver.
    """

    values: Dict[str, int] = field(default_factory=dict)
    bounded_vars: Dict[str, Var] = field(default_factory=dict)
    parent: Optional["ConstEnv"] = None

    def child(self) -> "ConstEnv":
        return ConstEnv(parent=self)

    def define(self, name: str, value: int) -> None:
        self.values[name] = int(value)

    def define_bounded(
        self, name: str, lower: Optional[int], upper: Optional[int]
    ) -> None:
        self.bounded_vars[name] = Var(name=name, lower=lower, upper=upper)

    def lookup(self, name: str) -> Optional[int]:
        env: Optional[ConstEnv] = self
        while env is not None:
            if name in env.values:
                return env.values[name]
            env = env.parent
        return None

    def lookup_var(self, name: str) -> Optional[Var]:
        env: Optional[ConstEnv] = self
        while env is not None:
            if name in env.bounded_vars:
                return env.bounded_vars[name]
            env = env.parent
        return None

    def flatten(self) -> Dict[str, int]:
        """All visible constants, innermost definitions winning."""
        out: Dict[str, int] = {}
        chain = []
        env: Optional[ConstEnv] = self
        while env is not None:
            chain.append(env)
            env = env.parent
        for env in reversed(chain):
            out.update(env.values)
        return out


def canonical_accessor(text: str) -> str:
    """Canonicalise a member access: ``a -> b`` becomes ``a->b``.

    The evaluator looks names up by their source text, so anything that binds
    a member access has to spell it the same way.  Sharing this function is
    what keeps a binding and a lookup from disagreeing over whitespace.
    """
    return _ACCESSOR_WS_RE.sub(lambda m: m.group(1), text).strip()


class ExpressionEvaluator:
    """Lowers tree-sitter expression nodes to :data:`~ascend_analyzer.symbolic.Expr`."""

    def __init__(self, source: bytes) -> None:
        self._source = source
        #: Resolver for calls to single-``return`` helper functions, installed
        #: by the visitor once it has indexed them.  Takes the call node and the
        #: environment, returns a constant or ``None``.  Without it a helper
        #: such as ``AlignUpBytes(hidden_ * 2)`` stays opaque, which leaves
        #: every extent it computes unresolvable.
        self.helper_resolver: Optional[
            "Callable[[Node, ConstEnv], Optional[int]]"
        ] = None

    # -- helpers ------------------------------------------------------------

    def text(self, node: Node) -> str:
        return self._source[node.start_byte : node.end_byte].decode("utf-8", "replace")

    # -- public API ---------------------------------------------------------

    def evaluate(self, node: Optional[Node], env: ConstEnv) -> Optional[Expr]:
        """Lower ``node`` to an expression, or ``None`` if it is not integral."""
        if node is None:
            return None
        expr = self._lower(node, env)
        return None if expr is None else simplify(expr)

    def fold(self, node: Optional[Node], env: ConstEnv) -> Optional[int]:
        """Lower and demand a constant; ``None`` when not statically known."""
        expr = self.evaluate(node, env)
        return expr.value if isinstance(expr, Const) else None

    # -- lowering -----------------------------------------------------------

    def _lower(self, node: Node, env: ConstEnv) -> Optional[Expr]:
        kind = node.type

        if kind == "number_literal":
            value = _parse_int_literal(self.text(node))
            return None if value is None else Const(value)

        if kind == "char_literal":
            body = self.text(node).strip("'")
            return Const(ord(body[0])) if len(body) == 1 else None

        if kind in {"true", "false"}:
            return Const(1 if kind == "true" else 0)

        if kind == "identifier":
            return self._lower_name(self.text(node), env)

        if kind in {"qualified_identifier", "field_expression", "scoped_identifier"}:
            return self._lower_name(self.text(node), env)

        if kind == "parenthesized_expression":
            inner = self._first_named_child(node)
            return self._lower(inner, env) if inner is not None else None

        if kind == "binary_expression":
            return self._lower_binary(node, env)

        if kind == "unary_expression":
            return self._lower_unary(node, env)

        if kind == "cast_expression":
            value = node.child_by_field_name("value")
            return self._lower(value, env) if value is not None else None

        if kind == "sizeof_expression":
            return self._lower_sizeof(node, env)

        if kind == "conditional_expression":
            return self._lower_conditional(node, env)

        if kind == "call_expression":
            return self._lower_call(node, env)

        if kind in {"subscript_expression", "pointer_expression"}:
            # Not an integer address expression in its own right; keep it as an
            # opaque symbol so downstream checks know it is unresolved.
            return Var(name=self.text(node))

        if kind == "ERROR":
            return None

        # A single named child that is itself an expression (common with
        # grammar wrappers) is transparent.
        only = self._first_named_child(node)
        if only is not None and only is not node:
            return self._lower(only, env)
        return None

    def _lower_name(self, raw: str, env: ConstEnv) -> Optional[Expr]:
        name = raw.strip()
        # ``tilingData -> rowFactor`` and ``tilingData->rowFactor`` are the same
        # member; normalise the accessor so a single binding matches both.
        canonical = _ACCESSOR_WS_RE.sub(r"\1", name)
        candidates = [name, name.rsplit("::", 1)[-1]]
        if canonical != name:
            candidates.insert(1, canonical)
        for candidate in candidates:
            value = env.lookup(candidate)
            if value is not None:
                return Const(value)
            bounded = env.lookup_var(candidate)
            if bounded is not None:
                return bounded
        return Var(name=name)

    def _lower_binary(self, node: Node, env: ConstEnv) -> Optional[Expr]:
        op_node = node.child_by_field_name("operator")
        op = self.text(op_node) if op_node is not None else self._infer_operator(node)
        left = self._lower_field(node, "left", env)
        right = self._lower_field(node, "right", env)
        if left is None or right is None:
            return None
        left = simplify(left)
        right = simplify(right)
        fold = _CMP_OPS.get(op)
        if fold is not None:
            # ``t + 2 < NV_TILES`` folds once ``t`` is concrete (an unrolled
            # loop iteration), which is what lets ``_handle_if`` prune the
            # dead arm of pipeline epilogue guards.
            if isinstance(left, Const) and isinstance(right, Const):
                return Const(fold(left.value, right.value))
            return None
        if op in {"&&", "||"}:
            if isinstance(left, Const) and isinstance(right, Const):
                if op == "&&":
                    return Const(int(left.value != 0 and right.value != 0))
                return Const(int(left.value != 0 or right.value != 0))
            if isinstance(left, Const):
                if op == "&&" and left.value == 0:
                    return Const(0)
                if op == "||" and left.value != 0:
                    return Const(1)
            return None
        if op not in _BIN_OPS:
            return None
        return BinOp(op=op, left=left, right=right)

    def _lower_unary(self, node: Node, env: ConstEnv) -> Optional[Expr]:
        op_node = node.child_by_field_name("operator")
        op = self.text(op_node) if op_node is not None else self._infer_operator(node)
        operand = self._lower_field(node, "argument", env)
        if op == "!":
            operand = simplify(operand) if operand is not None else None
            if isinstance(operand, Const):
                return Const(int(operand.value == 0))
            return None
        if op not in _UN_OPS:
            return None
        return None if operand is None else UnOp(op=op, operand=operand)

    def _lower_sizeof(self, node: Node, env: ConstEnv) -> Optional[Expr]:
        """Fold ``sizeof(T)`` for the scalar types Ascend C kernels use.

        tree-sitter exposes the operand either as a ``type`` field (for
        ``sizeof(int)``) or as a ``value`` field holding a parenthesized
        expression (for ``sizeof(half)``, where ``half`` is a typedef the
        grammar cannot distinguish from a variable), so both are tried.
        """
        for field_name in ("type", "value"):
            operand = node.child_by_field_name(field_name)
            if operand is None:
                continue
            size = dtype_size(_strip_type_text(self.text(operand)))
            if size is not None:
                return Const(size)
        return Var(name=self.text(node))

    def _lower_conditional(self, node: Node, env: ConstEnv) -> Optional[Expr]:
        cond = self._lower_field(node, "condition", env)
        folded = simplify(cond) if cond is not None else None
        if isinstance(folded, Const):
            branch = "consequence" if folded.value != 0 else "alternative"
            return self._lower_field(node, branch, env)
        return Var(name=self.text(node))

    def _lower_call(self, node: Node, env: ConstEnv) -> Optional[Expr]:
        func = node.child_by_field_name("function")
        fname = self.text(func).rsplit("::", 1)[-1] if func is not None else ""
        args = node.child_by_field_name("arguments")
        arg_nodes = (
            [c for c in args.named_children if c.type != "comment"] if args else []
        )

        if fname == "sizeof" and arg_nodes:
            size = dtype_size(_strip_type_text(self.text(arg_nodes[0])))
            return Const(size) if size is not None else Var(name=self.text(node))

        # ``static_cast<int64_t>(x)`` is a call_expression in this grammar, not
        # a cast_expression, so without this it stays opaque - and kernels spell
        # nearly every width conversion that way, which left the extents built
        # from them unresolvable.  A named-cast to an integral type is
        # value-preserving for layout arithmetic, so the operand passes through.
        if _INT_CAST_RE.match(fname) and len(arg_nodes) == 1:
            return self._lower(arg_nodes[0], env)

        # Common host-side helpers that are pure integer arithmetic.
        if fname in {"AlignUp", "ALIGN_UP", "CeilAlign"} and len(arg_nodes) == 2:
            value = self._lower(arg_nodes[0], env)
            align = self._lower(arg_nodes[1], env)
            if value is not None and align is not None:
                # ((value + align - 1) / align) * align
                numerator = BinOp("-", BinOp("+", value, align), Const(1))
                return BinOp("*", BinOp("/", numerator, align), align)
        if fname in {"AlignDown", "ALIGN_DOWN", "FloorAlign"} and len(arg_nodes) == 2:
            value = self._lower(arg_nodes[0], env)
            align = self._lower(arg_nodes[1], env)
            if value is not None and align is not None:
                return BinOp("*", BinOp("/", value, align), align)
        if fname in {"CeilDiv", "CEIL_DIV", "DivCeil"} and len(arg_nodes) == 2:
            value = self._lower(arg_nodes[0], env)
            divisor = self._lower(arg_nodes[1], env)
            if value is not None and divisor is not None:
                return BinOp("/", BinOp("-", BinOp("+", value, divisor), Const(1)), divisor)

        # A project-local single-``return`` helper, evaluated with its arguments
        # bound.  Tried last so the built-in spellings above keep their
        # symbolic (non-constant) form where the arguments do not fold.
        if self.helper_resolver is not None:
            folded = self.helper_resolver(node, env)
            if folded is not None:
                return Const(folded)

        return Var(name=self.text(node))

    # -- node utilities -----------------------------------------------------

    def _lower_field(self, node: Node, field_name: str, env: ConstEnv) -> Optional[Expr]:
        child = node.child_by_field_name(field_name)
        return self._lower(child, env) if child is not None else None

    def _infer_operator(self, node: Node) -> str:
        """Fall back to the first anonymous child when the field is missing."""
        for child in node.children:
            if not child.is_named:
                return self.text(child)
        return ""

    @staticmethod
    def _first_named_child(node: Node) -> Optional[Node]:
        for child in node.named_children:
            if child.type != "comment":
                return child
        return None


def collect_define_value(
    evaluator: ExpressionEvaluator, parser, value_text: str, env: ConstEnv
) -> Optional[int]:
    """Fold an object-like ``#define`` body by re-parsing it as an expression.

    ``preproc_def`` hands back the macro body as raw text, so the cheapest way
    to evaluate it with the full expression machinery is to wrap it in a
    throwaway declaration and parse that.
    """
    body = value_text.strip()
    if not body:
        return None
    snippet = f"int __aka_probe = ({body});".encode("utf-8")
    tree = parser.parse(snippet)
    probe = ExpressionEvaluator(snippet)
    for node in _walk(tree.root_node):
        if node.type == "init_declarator":
            value = node.child_by_field_name("value")
            if value is not None:
                return probe.fold(value, env)
    return None


def _walk(node: Node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))
