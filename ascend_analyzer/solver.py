"""Memory-interval decision procedures: a Z3 backend and an interval backend.

The memory checker asks three kinds of question about byte ranges that may
contain free variables (loop induction variables, host-chosen tiling factors):

* *bounds*  - can ``[offset, offset + size)`` ever leave the domain?
* *alignment* - can ``offset`` ever fail to be a multiple of N?
* *disjointness* - can two ranges ever overlap?

Each is posed as a satisfiability question about the **violation**.  ``UNSAT``
is a proof the kernel is safe for every value the free variables can take;
``SAT`` hands back a concrete counterexample - "with ``t = 7`` this tile ends
at byte 200704, which is 8 KiB past the end of UB" - which is far more useful
to the engineer than a bare warning.

Two interchangeable backends implement the same protocol:

``Z3Backend``
    Complete over linear integer arithmetic, and the only one that can produce
    counterexamples for symbolic offsets.  Used by default when ``z3-solver``
    is installed.
``IntervalBackend``
    Dependency-free interval arithmetic.  Exact for fully constant layouts -
    which is the common case - and deliberately *abstains* rather than
    guessing when a query needs real reasoning.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Protocol, Tuple

from .symbolic import (
    BinOp,
    Const,
    Expr,
    UnOp,
    Var,
    free_vars,
    interval_of,
    intervals_definitely_disjoint,
    is_z3_lowerable,
    simplify,
    to_int,
)

__all__ = [
    "Verdict",
    "Finding",
    "MemorySolver",
    "IntervalBackend",
    "Z3Backend",
    "make_solver",
    "z3_available",
]


class Verdict(Enum):
    """The outcome of a query about a potential violation."""

    SAFE = "safe"            # proved the violation cannot happen
    VIOLATED = "violated"    # the violation is reachable (counterexample held)
    UNKNOWN = "unknown"      # the backend could not decide

    @property
    def is_violation(self) -> bool:
        return self is Verdict.VIOLATED


@dataclass
class Finding:
    """A solver answer, with a counterexample when one exists."""

    verdict: Verdict
    #: Variable assignment that realises the violation, when available.
    counterexample: Dict[str, int] = field(default_factory=dict)
    #: Human-readable explanation of the witness.
    witness: str = ""

    @classmethod
    def safe(cls) -> "Finding":
        return cls(verdict=Verdict.SAFE)

    @classmethod
    def unknown(cls, reason: str = "") -> "Finding":
        return cls(verdict=Verdict.UNKNOWN, witness=reason)

    @classmethod
    def violated(
        cls, counterexample: Optional[Mapping[str, int]] = None, witness: str = ""
    ) -> "Finding":
        return cls(
            verdict=Verdict.VIOLATED,
            counterexample=dict(counterexample or {}),
            witness=witness,
        )

    def describe_counterexample(self) -> str:
        if not self.counterexample:
            return self.witness
        assignment = ", ".join(
            f"{name} = {value}" for name, value in sorted(self.counterexample.items())
        )
        return f"{self.witness} (with {assignment})" if self.witness else f"with {assignment}"


class MemorySolver(Protocol):
    """The interface both backends implement."""

    name: str

    def check_bounds(
        self, offset: Optional[Expr], size: Optional[Expr], capacity: int
    ) -> Finding:
        """Can ``[offset, offset+size)`` escape ``[0, capacity)``?"""

    def check_alignment(self, value: Optional[Expr], alignment: int) -> Finding:
        """Can ``value`` fail to be a multiple of ``alignment``?"""

    def check_disjoint(
        self,
        a_offset: Optional[Expr],
        a_size: Optional[Expr],
        b_offset: Optional[Expr],
        b_size: Optional[Expr],
    ) -> Finding:
        """Can the two byte ranges overlap?"""


# ---------------------------------------------------------------------------
# Interval backend
# ---------------------------------------------------------------------------


class IntervalBackend:
    """Interval-arithmetic backend; exact for constants, abstains otherwise."""

    name = "interval"

    def check_bounds(
        self, offset: Optional[Expr], size: Optional[Expr], capacity: int
    ) -> Finding:
        off, length = to_int(offset), to_int(size)
        if off is not None and length is not None:
            if off < 0:
                return Finding.violated(witness=f"base offset {off} is negative")
            end = off + length
            if end > capacity:
                return Finding.violated(
                    witness=f"range ends at byte {end}, {end - capacity} B past the "
                    f"{capacity} B capacity"
                )
            return Finding.safe()

        # Symbolic: use conservative bounds where they exist.
        oi, si = interval_of(offset), interval_of(size)
        if oi.lo is not None and oi.lo < 0:
            return Finding.violated(witness=f"base offset can reach {oi.lo}")
        if oi.hi is not None and si.hi is not None:
            if oi.hi + si.hi > capacity:
                return Finding.violated(
                    witness=f"range can end at byte {oi.hi + si.hi}, past the "
                    f"{capacity} B capacity"
                )
            return Finding.safe()
        return Finding.unknown("offset or size is not statically bounded")

    def check_alignment(self, value: Optional[Expr], alignment: int) -> Finding:
        if alignment <= 1:
            return Finding.safe()
        folded = to_int(value)
        if folded is None:
            return Finding.unknown("value is not a compile-time constant")
        remainder = folded % alignment
        if remainder:
            return Finding.violated(
                witness=f"{folded} is {remainder} B past the previous "
                f"{alignment} B boundary"
            )
        return Finding.safe()

    def check_disjoint(
        self,
        a_offset: Optional[Expr],
        a_size: Optional[Expr],
        b_offset: Optional[Expr],
        b_size: Optional[Expr],
    ) -> Finding:
        decision = intervals_definitely_disjoint(a_offset, a_size, b_offset, b_size)
        if decision is True:
            return Finding.safe()
        if decision is False:
            a_lo, a_len = to_int(a_offset), to_int(a_size)
            b_lo, b_len = to_int(b_offset), to_int(b_size)
            witness = ""
            if None not in (a_lo, a_len, b_lo, b_len):
                lo = max(a_lo, b_lo)  # type: ignore[arg-type]
                hi = min(a_lo + a_len, b_lo + b_len)  # type: ignore[operator]
                witness = f"ranges share bytes [0x{lo:X}, 0x{hi:X}) ({hi - lo} B)"
            return Finding.violated(witness=witness)
        return Finding.unknown("ranges are not statically comparable")


# ---------------------------------------------------------------------------
# Z3 backend
# ---------------------------------------------------------------------------


def z3_available() -> bool:
    """``True`` when ``z3-solver`` is installed and importable."""
    return importlib.util.find_spec("z3") is not None


class Z3Backend:
    """Z3-backed backend over mathematical integers.

    Each query asserts the *negation* of the safety property together with the
    inferred ranges of every free variable, then asks Z3 for a model.  Because
    the encoding is linear integer arithmetic, ``UNSAT`` is a genuine proof.
    """

    name = "z3"

    def __init__(self, timeout_ms: int = 5000) -> None:
        import z3

        self._z3 = z3
        self._timeout_ms = timeout_ms
        self._fallback = IntervalBackend()

    # -- lowering -----------------------------------------------------------

    def _lower(self, expr: Expr, variables: Dict[str, object]):
        z3 = self._z3
        if isinstance(expr, Const):
            return z3.IntVal(expr.value)
        if isinstance(expr, Var):
            if expr.name not in variables:
                variables[expr.name] = z3.Int(_sanitize(expr.name))
            return variables[expr.name]
        if isinstance(expr, UnOp):
            inner = self._lower(expr.operand, variables)
            return -inner
        left = self._lower(expr.left, variables)
        right = self._lower(expr.right, variables)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        if expr.op == "%":
            return left % right
        if expr.op == "<<":
            # Only reachable with a constant shift; symbolic shifts are
            # rejected by is_z3_lowerable before we get here.
            shift = to_int(expr.right)
            return left * z3.IntVal(1 << shift) if shift is not None else left
        raise ValueError(f"operator {expr.op!r} is not lowerable to Z3")

    def _bound_constraints(self, exprs: Tuple[Optional[Expr], ...], variables):
        """Range assumptions for every free variable appearing in ``exprs``."""
        z3 = self._z3
        constraints = []
        for expr in exprs:
            for name, v in free_vars(expr).items():
                if name not in variables:
                    variables[name] = z3.Int(_sanitize(name))
                symbol = variables[name]
                # Byte offsets and sizes are non-negative by construction, and
                # so is every induction variable we infer.
                constraints.append(symbol >= (v.lower if v.lower is not None else 0))
                if v.upper is not None:
                    constraints.append(symbol <= v.upper)
        return constraints

    def _solve(
        self, violation, assumptions, variables: Dict[str, object]
    ) -> Finding:
        z3 = self._z3
        solver = z3.Solver()
        solver.set("timeout", self._timeout_ms)
        for assumption in assumptions:
            solver.add(assumption)
        solver.add(violation)
        result = solver.check()
        if result == z3.unsat:
            return Finding.safe()
        if result == z3.sat:
            model = solver.model()
            counterexample: Dict[str, int] = {}
            for name, symbol in variables.items():
                value = model.eval(symbol, model_completion=True)
                try:
                    counterexample[name] = value.as_long()
                except AttributeError:  # pragma: no cover - non-integer model
                    continue
            return Finding.violated(counterexample=counterexample)
        return Finding.unknown("solver returned unknown (timeout or incompleteness)")

    # -- queries ------------------------------------------------------------

    def check_bounds(
        self, offset: Optional[Expr], size: Optional[Expr], capacity: int
    ) -> Finding:
        if offset is None or size is None:
            return self._fallback.check_bounds(offset, size, capacity)
        if not (is_z3_lowerable(offset) and is_z3_lowerable(size)):
            return self._fallback.check_bounds(offset, size, capacity)

        variables: Dict[str, object] = {}
        assumptions = self._bound_constraints((offset, size), variables)
        off = self._lower(simplify(offset), variables)
        length = self._lower(simplify(size), variables)
        z3 = self._z3
        violation = z3.Or(off < 0, off + length > capacity)
        finding = self._solve(violation, assumptions, variables)
        if finding.verdict is Verdict.VIOLATED:
            finding.witness = _bounds_witness(finding, offset, size, capacity)
        return finding

    def check_alignment(self, value: Optional[Expr], alignment: int) -> Finding:
        if alignment <= 1:
            return Finding.safe()
        if value is None:
            return Finding.unknown("value is unknown")
        folded = to_int(value)
        if folded is not None:
            return self._fallback.check_alignment(value, alignment)
        if not is_z3_lowerable(value):
            return Finding.unknown("expression contains non-arithmetic operators")

        variables: Dict[str, object] = {}
        assumptions = self._bound_constraints((value,), variables)
        lowered = self._lower(simplify(value), variables)
        violation = lowered % alignment != 0
        finding = self._solve(violation, assumptions, variables)
        if finding.verdict is Verdict.VIOLATED and not finding.witness:
            finding.witness = f"is not always a multiple of {alignment} B"
        return finding

    def check_disjoint(
        self,
        a_offset: Optional[Expr],
        a_size: Optional[Expr],
        b_offset: Optional[Expr],
        b_size: Optional[Expr],
    ) -> Finding:
        exprs = (a_offset, a_size, b_offset, b_size)
        if any(e is None for e in exprs):
            return self._fallback.check_disjoint(*exprs)
        if not all(is_z3_lowerable(e) for e in exprs):
            return self._fallback.check_disjoint(*exprs)

        variables: Dict[str, object] = {}
        assumptions = self._bound_constraints(exprs, variables)
        a_lo = self._lower(simplify(a_offset), variables)       # type: ignore[arg-type]
        a_hi = a_lo + self._lower(simplify(a_size), variables)  # type: ignore[arg-type]
        b_lo = self._lower(simplify(b_offset), variables)       # type: ignore[arg-type]
        b_hi = b_lo + self._lower(simplify(b_size), variables)  # type: ignore[arg-type]
        z3 = self._z3
        # Half-open ranges overlap iff each starts before the other ends.
        overlap = z3.And(a_lo < b_hi, b_lo < a_hi)
        finding = self._solve(overlap, assumptions, variables)
        if finding.verdict is Verdict.VIOLATED and not finding.witness:
            fallback = self._fallback.check_disjoint(*exprs)
            finding.witness = fallback.witness or "ranges can overlap"
        return finding


def _bounds_witness(
    finding: Finding,
    offset: Optional[Expr],
    size: Optional[Expr],
    capacity: int,
) -> str:
    """Describe a bounds violation by evaluating the model concretely."""
    substituted = _substitute(offset, finding.counterexample)
    length = _substitute(size, finding.counterexample)
    off_value, len_value = to_int(substituted), to_int(length)
    if off_value is None or len_value is None:
        return "range can leave the domain"
    end = off_value + len_value
    if off_value < 0:
        return f"base offset can reach {off_value}"
    return (
        f"range can end at byte {end}, {end - capacity} B past the "
        f"{capacity} B capacity"
    )


def _substitute(expr: Optional[Expr], values: Mapping[str, int]) -> Optional[Expr]:
    """Replace free variables in ``expr`` with concrete values."""
    if expr is None:
        return None
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Var):
        value = values.get(expr.name)
        return Const(value) if value is not None else expr
    if isinstance(expr, UnOp):
        inner = _substitute(expr.operand, values)
        return simplify(UnOp(expr.op, inner)) if inner is not None else None
    left = _substitute(expr.left, values)
    right = _substitute(expr.right, values)
    if left is None or right is None:
        return None
    return simplify(BinOp(expr.op, left, right))


def _sanitize(name: str) -> str:
    """Make an arbitrary source expression usable as a Z3 symbol name."""
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name) or "v"


def make_solver(preference: str = "auto", timeout_ms: int = 5000) -> MemorySolver:
    """Construct a solver backend.

    ``preference`` is ``"auto"`` (Z3 when importable, else intervals),
    ``"z3"`` (hard requirement) or ``"interval"``.
    """
    choice = preference.strip().lower()
    if choice == "interval":
        return IntervalBackend()
    if choice == "z3":
        return Z3Backend(timeout_ms=timeout_ms)
    if z3_available():
        return Z3Backend(timeout_ms=timeout_ms)
    return IntervalBackend()
