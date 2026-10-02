"""A tiny integer expression IR with constant folding and Z3 lowering.

Kernel offsets in the static tensor programming model are almost always
compile-time arithmetic over ``constexpr`` scalars::

    constexpr uint32_t TILE_BYTES = 256 * sizeof(half);
    constexpr uint32_t UB_PONG    = UB_PING + TILE_BYTES;

so the analyzer needs real constant folding rather than regex scraping.  What
it *cannot* fold - a loop induction variable, a kernel argument, a tiling
value chosen by the host - is kept symbolically so the Z3 backend can still
reason about it and, when a check fails, hand back a concrete counterexample.

The IR is deliberately minimal: literals, free variables, and the integer
operators that appear in address arithmetic.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Mapping, Optional, Union

__all__ = [
    "Expr",
    "Const",
    "Var",
    "BinOp",
    "UnOp",
    "const",
    "var",
    "add",
    "mul",
    "simplify",
    "free_vars",
    "is_decidable",
    "is_z3_lowerable",
    "to_int",
    "Interval",
    "interval_of",
    "intervals_definitely_disjoint",
    "render",
    "DTYPE_SIZES",
    "dtype_size",
]


# ---------------------------------------------------------------------------
# Data type widths
# ---------------------------------------------------------------------------

#: Byte widths of the scalar types that appear in Ascend C kernels.
DTYPE_SIZES: Mapping[str, int] = {
    "bool": 1,
    "int8_t": 1,
    "uint8_t": 1,
    "char": 1,
    "signed char": 1,
    "unsigned char": 1,
    "int4b_t": 1,  # packed; one nibble, rounded up for sizing purposes
    "half": 2,
    "float16": 2,
    "fp16": 2,
    "bfloat16_t": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int16_t": 2,
    "uint16_t": 2,
    "short": 2,
    "unsigned short": 2,
    "float": 4,
    "float32": 4,
    "fp32": 4,
    "int32_t": 4,
    "uint32_t": 4,
    "int": 4,
    "unsigned": 4,
    "unsigned int": 4,
    "double": 8,
    "int64_t": 8,
    "uint64_t": 8,
    "long long": 8,
    "unsigned long long": 8,
}


def dtype_size(dtype: Optional[str]) -> Optional[int]:
    """Byte width of ``dtype``, or ``None`` when unknown."""
    if not dtype:
        return None
    key = dtype.strip().replace("AscendC::", "")
    key = key.removeprefix("const ").strip()
    return DTYPE_SIZES.get(key) or DTYPE_SIZES.get(key.lower())


# ---------------------------------------------------------------------------
# Expression IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Const:
    """An integer literal."""

    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Var:
    """A free integer variable the parser could not fold to a constant.

    ``lower``/``upper`` carry any range the parser could infer (for example a
    loop induction variable bounded by its trip count), which lets Z3 produce
    realistic counterexamples instead of absurd ones.
    """

    name: str
    lower: Optional[int] = None
    upper: Optional[int] = None

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class BinOp:
    """A binary integer operation."""

    op: str
    left: "Expr"
    right: "Expr"

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class UnOp:
    """A unary integer operation (``-`` or ``~``)."""

    op: str
    operand: "Expr"

    def __str__(self) -> str:
        return f"{self.op}{self.operand}"


Expr = Union[Const, Var, BinOp, UnOp]


_BIN_FOLD: Mapping[str, Callable[[int, int], int]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": lambda a, b: a // b,       # C integer division on non-negative operands
    "%": operator.mod,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
}

_UN_FOLD: Mapping[str, Callable[[int], int]] = {
    "-": operator.neg,
    "+": lambda a: a,
    "~": operator.invert,
}

#: Operators the Z3 backend can lower losslessly over mathematical integers.
_Z3_SAFE_OPS: FrozenSet[str] = frozenset({"+", "-", "*", "/", "%", "<<"})


# ---------------------------------------------------------------------------
# Constructors and folding
# ---------------------------------------------------------------------------


def const(value: int) -> Const:
    return Const(int(value))


def var(name: str, lower: Optional[int] = None, upper: Optional[int] = None) -> Var:
    return Var(name=name, lower=lower, upper=upper)


def simplify(expr: Expr) -> Expr:
    """Fold constants bottom-up and apply cheap algebraic identities."""
    if isinstance(expr, (Const, Var)):
        return expr

    if isinstance(expr, UnOp):
        inner = simplify(expr.operand)
        if isinstance(inner, Const):
            return Const(_UN_FOLD[expr.op](inner.value))
        if expr.op == "+":
            return inner
        return UnOp(expr.op, inner)

    left, right = simplify(expr.left), simplify(expr.right)
    op = expr.op

    if isinstance(left, Const) and isinstance(right, Const):
        if op in {"/", "%"} and right.value == 0:
            return BinOp(op, left, right)  # leave division by zero unfolded
        try:
            return Const(_BIN_FOLD[op](left.value, right.value))
        except (ValueError, OverflowError):  # pragma: no cover - defensive
            return BinOp(op, left, right)

    # Identities that routinely show up in generated address arithmetic.
    if isinstance(right, Const):
        if op in {"+", "-", "|", "^", ">>", "<<"} and right.value == 0:
            return left
        if op in {"*", "/"} and right.value == 1:
            return left
        if op == "*" and right.value == 0:
            return Const(0)
        if op == "&" and right.value == 0:
            return Const(0)
    if isinstance(left, Const):
        if op in {"+", "|", "^"} and left.value == 0:
            return right
        if op == "*" and left.value == 1:
            return right
        if op in {"*", "&"} and left.value == 0:
            return Const(0)
        if op in {"/", "%", "<<", ">>"} and left.value == 0:
            return Const(0)

    return BinOp(op, left, right)


def add(left: Expr, right: Expr) -> Expr:
    return simplify(BinOp("+", left, right))


def mul(left: Expr, right: Expr) -> Expr:
    return simplify(BinOp("*", left, right))


def to_int(expr: Optional[Expr]) -> Optional[int]:
    """Return the folded integer value of ``expr``, or ``None`` if symbolic."""
    if expr is None:
        return None
    folded = simplify(expr)
    return folded.value if isinstance(folded, Const) else None


def free_vars(expr: Optional[Expr]) -> Dict[str, Var]:
    """Collect the free variables of ``expr``, keyed by name."""
    out: Dict[str, Var] = {}

    def walk(node: Optional[Expr]) -> None:
        if node is None or isinstance(node, Const):
            return
        if isinstance(node, Var):
            existing = out.get(node.name)
            if existing is None:
                out[node.name] = node
            else:
                # Keep the tightest known bounds if the same name recurs.
                out[node.name] = Var(
                    node.name,
                    lower=_tighter(existing.lower, node.lower, max),
                    upper=_tighter(existing.upper, node.upper, min),
                )
            return
        if isinstance(node, UnOp):
            walk(node.operand)
            return
        walk(node.left)
        walk(node.right)

    walk(expr)
    return out


def _tighter(
    a: Optional[int], b: Optional[int], pick: Callable[[int, int], int]
) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return pick(a, b)


def is_decidable(expr: Optional[Expr]) -> bool:
    """``True`` when a solver can reach a *useful* verdict about ``expr``.

    A folded constant is trivially decidable.  A symbolic expression is only
    decidable when every free variable carries both bounds - typically a loop
    induction variable the parser could bound from the loop header.

    An *unbounded* free variable means the analyzer failed to read a value that
    is a compile-time constant in the real kernel (an unparsed macro, a
    template parameter, a host-supplied tiling field).  Letting the solver
    range freely over it manufactures guaranteed-looking violations from pure
    ignorance - "this could overflow if the offset were 196096" - which is
    worse than silence.  Such expressions are reported as unresolved
    (``AKA3002``) and excluded from the layout proofs instead.
    """
    if expr is None:
        return False
    if to_int(expr) is not None:
        return True
    variables = free_vars(expr)
    if not variables:
        return True
    return all(v.lower is not None and v.upper is not None for v in variables.values())


def is_z3_lowerable(expr: Optional[Expr]) -> bool:
    """``True`` when every operator in ``expr`` has an exact Z3 integer form.

    Bitwise ``&``/``|``/``^``/``~`` and ``>>`` on symbolic operands have no
    clean mathematical-integer encoding, so the Z3 backend declines them and
    the caller falls back to treating the value as opaque.
    """
    if expr is None:
        return False
    if isinstance(expr, (Const, Var)):
        return True
    if isinstance(expr, UnOp):
        return expr.op == "-" and is_z3_lowerable(expr.operand)
    if expr.op not in _Z3_SAFE_OPS:
        return False
    return is_z3_lowerable(expr.left) and is_z3_lowerable(expr.right)


def render(expr: Optional[Expr], *, max_len: int = 48) -> str:
    """Render ``expr`` for display, preferring the folded constant."""
    if expr is None:
        return "?"
    folded = simplify(expr)
    if isinstance(folded, Const):
        return str(folded.value)
    text = str(folded)
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Interval abstraction (used by the non-Z3 backend and for quick pruning)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A closed integer interval, with ``None`` meaning unbounded."""

    lo: Optional[int]
    hi: Optional[int]

    @property
    def is_point(self) -> bool:
        return self.lo is not None and self.lo == self.hi

    @property
    def is_unbounded(self) -> bool:
        return self.lo is None or self.hi is None


_UNBOUNDED = Interval(None, None)


def interval_of(expr: Optional[Expr]) -> Interval:
    """Conservatively bound ``expr``.

    Returns an exact point interval for folded constants, a bounded interval
    when every free variable carries bounds, and an unbounded interval
    otherwise.  Used to prune obviously-disjoint tensor pairs before invoking
    the solver, and as the sole engine in ``--solver interval`` mode.
    """
    if expr is None:
        return _UNBOUNDED
    node = simplify(expr)

    if isinstance(node, Const):
        return Interval(node.value, node.value)

    if isinstance(node, Var):
        return Interval(node.lower, node.upper)

    if isinstance(node, UnOp):
        inner = interval_of(node.operand)
        if node.op == "-":
            return Interval(
                None if inner.hi is None else -inner.hi,
                None if inner.lo is None else -inner.lo,
            )
        return _UNBOUNDED

    left, right = interval_of(node.left), interval_of(node.right)

    if node.op == "+":
        return Interval(_oadd(left.lo, right.lo), _oadd(left.hi, right.hi))
    if node.op == "-":
        return Interval(_osub(left.lo, right.hi), _osub(left.hi, right.lo))
    if node.op in {"*", "<<"}:
        if node.op == "<<":
            # x << k  ==  x * 2**k, only sound for a constant shift.
            if not right.is_point or right.lo is None or right.lo < 0:
                return _UNBOUNDED
            right = Interval(1 << right.lo, 1 << right.lo)
        corners = [
            _omul(a, b)
            for a in (left.lo, left.hi)
            for b in (right.lo, right.hi)
        ]
        if any(c is None for c in corners):
            return _UNBOUNDED
        return Interval(min(corners), max(corners))  # type: ignore[arg-type]
    if node.op == "/":
        if right.is_point and right.lo not in (None, 0) and left.lo is not None and left.hi is not None:
            divisor = right.lo
            assert divisor is not None
            if divisor > 0 and left.lo >= 0:
                return Interval(left.lo // divisor, left.hi // divisor)
        return _UNBOUNDED
    if node.op == "%":
        if right.is_point and right.lo is not None and right.lo > 0:
            return Interval(0, right.lo - 1)
        return _UNBOUNDED

    return _UNBOUNDED


def _oadd(a: Optional[int], b: Optional[int]) -> Optional[int]:
    return None if a is None or b is None else a + b


def _osub(a: Optional[int], b: Optional[int]) -> Optional[int]:
    return None if a is None or b is None else a - b


def _omul(a: Optional[int], b: Optional[int]) -> Optional[int]:
    return None if a is None or b is None else a * b


def intervals_definitely_disjoint(
    a_start: Optional[Expr],
    a_len: Optional[Expr],
    b_start: Optional[Expr],
    b_len: Optional[Expr],
) -> Optional[bool]:
    """Decide disjointness of ``[a_start, a_start+a_len)`` vs the ``b`` range.

    Returns ``True`` (provably disjoint), ``False`` (provably overlapping), or
    ``None`` (undecided with interval arithmetic alone - ask the solver).
    """
    ai, al = interval_of(a_start), interval_of(a_len)
    bi, bl = interval_of(b_start), interval_of(b_len)

    a_end_hi = _oadd(ai.hi, al.hi)
    b_end_hi = _oadd(bi.hi, bl.hi)

    # Provably disjoint: one range's greatest end is at or below the other's
    # least start.
    if a_end_hi is not None and bi.lo is not None and a_end_hi <= bi.lo:
        return True
    if b_end_hi is not None and ai.lo is not None and b_end_hi <= ai.lo:
        return True

    # Both ranges are exact: the checks above would have proved disjointness
    # if it held, so reaching here means they definitely overlap.
    if ai.is_point and bi.is_point and al.is_point and bl.is_point:
        return False

    return None
