"""Tests for the symbolic integer IR, folding, intervals and decidability."""

from __future__ import annotations

import pytest

from ascend_analyzer.symbolic import (
    BinOp,
    Const,
    Interval,
    UnOp,
    Var,
    dtype_size,
    free_vars,
    interval_of,
    intervals_definitely_disjoint,
    is_decidable,
    is_z3_lowerable,
    render,
    simplify,
    to_int,
)


class TestDtypeSizes:
    @pytest.mark.parametrize(
        "dtype, size",
        [
            ("half", 2),
            ("float16", 2),
            ("bfloat16_t", 2),
            ("float", 4),
            ("int32_t", 4),
            ("uint8_t", 1),
            ("int8_t", 1),
            ("int64_t", 8),
            ("double", 8),
            ("AscendC::half", 2),
        ],
    )
    def test_known_widths(self, dtype, size):
        assert dtype_size(dtype) == size

    @pytest.mark.parametrize("dtype", [None, "", "MyStruct", "LocalTensor"])
    def test_unknown_widths_return_none(self, dtype):
        assert dtype_size(dtype) is None


class TestFolding:
    def test_folds_nested_arithmetic(self):
        expr = BinOp("+", BinOp("*", Const(250), Const(2)), Const(12))
        assert to_int(expr) == 512

    @pytest.mark.parametrize(
        "op, left, right, expected",
        [
            ("+", 7, 5, 12),
            ("-", 7, 5, 2),
            ("*", 7, 5, 35),
            ("/", 7, 2, 3),
            ("%", 7, 5, 2),
            ("<<", 1, 10, 1024),
            (">>", 1024, 10, 1),
            ("&", 0xF0, 0x3C, 0x30),
            ("|", 0xF0, 0x0C, 0xFC),
            ("^", 0xFF, 0x0F, 0xF0),
        ],
    )
    def test_every_operator(self, op, left, right, expected):
        assert to_int(BinOp(op, Const(left), Const(right))) == expected

    def test_unary_negation(self):
        assert to_int(UnOp("-", Const(32))) == -32

    def test_division_by_zero_is_left_unfolded(self):
        expr = simplify(BinOp("/", Const(8), Const(0)))
        assert to_int(expr) is None  # not a crash, just not a constant

    @pytest.mark.parametrize(
        "expr, expected_name",
        [
            (BinOp("+", Var("t"), Const(0)), "t"),
            (BinOp("*", Var("t"), Const(1)), "t"),
            (BinOp("-", Var("t"), Const(0)), "t"),
        ],
    )
    def test_identity_simplification(self, expr, expected_name):
        folded = simplify(expr)
        assert isinstance(folded, Var)
        assert folded.name == expected_name

    def test_multiplication_by_zero_collapses(self):
        assert to_int(simplify(BinOp("*", Var("t"), Const(0)))) == 0

    def test_symbolic_expression_has_no_constant_value(self):
        assert to_int(BinOp("*", Var("t"), Const(512))) is None


class TestFreeVars:
    def test_collects_names(self):
        expr = BinOp("+", BinOp("*", Var("t", 0, 7), Const(512)), Var("base"))
        names = free_vars(expr)
        assert set(names) == {"t", "base"}

    def test_keeps_the_tightest_bounds_when_a_name_recurs(self):
        expr = BinOp("+", Var("t", 0, 31), Var("t", 4, 7))
        bounds = free_vars(expr)["t"]
        assert bounds.lower == 4
        assert bounds.upper == 7

    def test_constants_have_no_free_vars(self):
        assert free_vars(Const(5)) == {}


class TestDecidability:
    def test_constants_are_decidable(self):
        assert is_decidable(Const(512))

    def test_bounded_loop_variable_is_decidable(self):
        # t * 8192 with t in [0, 31] is exactly what the solver can settle.
        assert is_decidable(BinOp("*", Var("t", 0, 31), Const(8192)))

    def test_unbounded_variable_is_not_decidable(self):
        # An unresolved constant must not be treated as a free-ranging value;
        # doing so manufactures violations out of ignorance.
        assert not is_decidable(Var("UB_Y_PONG"))
        assert not is_decidable(BinOp("+", Var("base"), Const(512)))

    def test_half_bounded_variable_is_not_decidable(self):
        assert not is_decidable(Var("t", 0, None))
        assert not is_decidable(Var("t", None, 31))

    def test_none_is_not_decidable(self):
        assert not is_decidable(None)


class TestZ3Lowerability:
    def test_plain_arithmetic_is_lowerable(self):
        assert is_z3_lowerable(BinOp("*", Var("t", 0, 3), Const(512)))

    @pytest.mark.parametrize("op", ["&", "|", "^", ">>"])
    def test_bitwise_operators_on_symbols_are_not_lowerable(self, op):
        # These have no clean mathematical-integer encoding, so the backend
        # declines instead of silently approximating.
        assert not is_z3_lowerable(BinOp(op, Var("t", 0, 3), Const(7)))

    def test_shift_left_is_lowerable(self):
        assert is_z3_lowerable(BinOp("<<", Var("t", 0, 3), Const(5)))


class TestIntervals:
    def test_constant_is_a_point(self):
        assert interval_of(Const(512)) == Interval(512, 512)

    def test_bounded_variable_scales(self):
        got = interval_of(BinOp("*", Var("t", 0, 31), Const(8192)))
        assert got == Interval(0, 253952)

    def test_unbounded_variable_is_unbounded(self):
        assert interval_of(Var("base")).is_unbounded

    def test_addition_widens(self):
        got = interval_of(BinOp("+", Var("t", 2, 4), Const(10)))
        assert got == Interval(12, 14)

    def test_modulo_by_constant_is_bounded(self):
        got = interval_of(BinOp("%", Var("t"), Const(32)))
        assert got == Interval(0, 31)

    def test_shift_left_by_constant(self):
        got = interval_of(BinOp("<<", Var("t", 1, 2), Const(4)))
        assert got == Interval(16, 32)


class TestDisjointness:
    def test_adjacent_ranges_are_disjoint(self):
        # [0,512) and [512,1024) touch but do not overlap.
        assert intervals_definitely_disjoint(
            Const(0), Const(512), Const(512), Const(512)
        ) is True

    def test_overlapping_constant_ranges_are_detected(self):
        # [512,1012) and [1000,1500) share 12 bytes.
        assert intervals_definitely_disjoint(
            Const(512), Const(500), Const(1000), Const(500)
        ) is False

    def test_identical_ranges_overlap(self):
        assert intervals_definitely_disjoint(
            Const(0), Const(512), Const(0), Const(512)
        ) is False

    def test_reversed_argument_order_gives_the_same_answer(self):
        forward = intervals_definitely_disjoint(
            Const(0), Const(512), Const(512), Const(512)
        )
        backward = intervals_definitely_disjoint(
            Const(512), Const(512), Const(0), Const(512)
        )
        assert forward == backward is True

    def test_undecidable_pairs_return_none(self):
        assert (
            intervals_definitely_disjoint(Var("a"), Const(512), Var("b"), Const(512))
            is None
        )


class TestRender:
    def test_prefers_the_folded_constant(self):
        assert render(BinOp("*", Const(256), Const(2))) == "512"

    def test_renders_symbolic_expressions_readably(self):
        assert render(BinOp("*", Var("t"), Const(8192))) == "t * 8192"

    def test_none_renders_as_a_placeholder(self):
        assert render(None) == "?"

    def test_long_expressions_are_truncated(self):
        expr = Var("x" * 100)
        assert len(render(expr, max_len=20)) == 20
