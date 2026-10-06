"""Tests for the vector-ISA parity checks (AKA1015, AKA1020).

Both failure families are silent: the kernel compiles and returns numbers
that are wrong.  The split-window check guards flattened ``[rows, stride]``
reductions merged by de-interleaving - below a ``2 * rows`` window the
merge sums independent batch rows.  The repeat-encoding check guards the
repeat-stride instruction form - a repeat counter past the one-byte field,
or strides that disagree with the operand lane ratio, skip or duplicate
elements at full speed.
"""

from __future__ import annotations

from conftest import analyze_body, codes_of, only

from ascend_analyzer.diagnostics import Severity

nl = chr(10)

#: Concrete UB tensors; offsets spaced 2 KiB apart.


def tensor(name: str, dtype: str, offset: int) -> str:
    return (
        f"AscendC::LocalTensor<{dtype}> {name};" + nl
        + f"{name}.SetTPosition(AscendC::TPosition::VECCALC);" + nl
        + f"{name}.SetAddr({offset});" + nl
        + f"{name}.SetSize(1024);" + nl
    )


FLOATS = (
    tensor("src", "float", 0)
    + tensor("a", "float", 2048)
    + tensor("b", "float", 4096)
    + tensor("off", "float", 6144)
    + tensor("dstBf", "bfloat16_t", 8192)
    + tensor("srcF32", "float", 10240)
)

#: The row anchor: folding the final two columns per row, so its count (8)
#: is the number of independent rows the tree reduces.
ANCHOR = "AscendC::Add(a, a, b, 8);" + nl


def deinterleave(count: int) -> str:
    return f"AscendC::DeInterleave(a, b, src, {count});" + nl


def gather_pair(count: int) -> str:
    """The two-Gather idiom parts without a native de-interleave use."""
    return (
        f"AscendC::Gather(a, src, off, 0, {count});" + nl
        + f"AscendC::Adds(off, off, 4, {count});" + nl
        + f"AscendC::Gather(b, src, off, 0, {count});" + nl
    )


def repeat_cast(per_repeat, repeats, strides) -> str:
    return (
        f"AscendC::Cast(dstBf, srcF32, AscendC::RoundMode::CAST_RINT, "
        f"{per_repeat}, {repeats}, {{{strides}}});" + nl
    )


class TestSplitWindowBleed:
    def test_window_below_two_rows_is_fatal(self):
        result = analyze_body(FLOATS + deinterleave(8) + ANCHOR)
        diag = only(result, "AKA1015")
        assert diag.severity is Severity.FATAL
        assert diag.details["rows"] == 8
        assert diag.details["window_length"] == 8
        assert diag.details["violation"] == "cross_row"

    def test_window_at_the_floor_is_clean(self):
        result = analyze_body(FLOATS + deinterleave(16) + ANCHOR)
        assert "AKA1015" not in codes_of(result)

    def test_full_tree_descent_is_clean(self):
        body = (
            FLOATS
            + deinterleave(256)
            + "AscendC::Add(src, a, b, 128);" + nl
            + deinterleave(128)
            + "AscendC::Add(src, a, b, 64);" + nl
            + deinterleave(32)
            + "AscendC::Add(src, a, b, 16);" + nl
            + deinterleave(16)
            + ANCHOR
        )
        assert "AKA1015" not in codes_of(analyze_body(body))

    def test_ragged_window_is_fatal(self):
        result = analyze_body(FLOATS + deinterleave(20) + ANCHOR)
        diag = only(result, "AKA1015")
        assert diag.details["violation"] == "ragged"
        assert diag.details["window_length"] == 20

    def test_without_a_row_anchor_nothing_is_claimed(self):
        """No `Add(x, x, half, rows)` anchor, no verdict - precision first."""
        result = analyze_body(FLOATS + deinterleave(8))
        assert "AKA1015" not in codes_of(result)

    def test_gather_pair_idiom_below_the_floor_is_fatal(self):
        result = analyze_body(FLOATS + gather_pair(4) + ANCHOR)
        diag = only(result, "AKA1015")
        assert diag.details["violation"] == "cross_row"
        assert diag.details["window_length"] == 8

    def test_gather_pair_idiom_at_the_floor_is_clean(self):
        result = analyze_body(FLOATS + gather_pair(8) + ANCHOR)
        assert "AKA1015" not in codes_of(result)

    def test_two_gathers_without_the_offset_bump_are_not_a_split(self):
        body = (
            FLOATS
            + "AscendC::Gather(a, src, off, 0, 4);" + nl
            + "AscendC::Gather(b, src, off, 0, 4);" + nl
            + ANCHOR
        )
        assert "AKA1015" not in codes_of(analyze_body(body))


class TestHalvingChainReplay:
    """The symbolic while-loop sees one body instance; the chain is replayed."""

    @staticmethod
    def _tree(bound) -> str:
        return (
            "int32_t mergeLen = 256;" + nl
            + f"while (mergeLen > {bound}) {{" + nl
            + "    AscendC::DeInterleave(a, b, src, mergeLen);" + nl
            + "    AscendC::Add(src, a, b, mergeLen / 2);" + nl
            + "    mergeLen /= 2;" + nl
            + "}" + nl
            + "AscendC::DeInterleave(a, b, src, mergeLen);" + nl
            + ANCHOR
        )

    def test_over_descended_tree_tail_is_fatal(self):
        result = analyze_body(FLOATS + self._tree("8"))
        diag = only(result, "AKA1015")
        assert diag.details["violation"] == "cross_row"
        assert diag.details["window_length"] == 8
        assert diag.details["level"] == "tail"

    def test_correctly_bounded_tree_is_clean(self):
        result = analyze_body(FLOATS + self._tree("16"))
        assert "AKA1015" not in codes_of(result)

    def test_the_finding_points_at_the_loop_header(self):
        result = analyze_body(FLOATS + self._tree("8"))
        diag = only(result, "AKA1015")
        assert "while (mergeLen > 8)" in diag.details["loop_header"]


class TestRepeatEncoding:
    def test_repeat_counter_past_the_byte_field_is_fatal(self):
        result = analyze_body(FLOATS + repeat_cast(64, 512, "1, 1, 4, 8"))
        diag = only(result, "AKA1020")
        assert diag.severity is Severity.FATAL
        assert diag.details["repeats"] == 512
        assert diag.details["max_repeats"] == 255

    def test_swapped_lane_ratio_strides_are_fatal(self):
        result = analyze_body(FLOATS + repeat_cast(64, 4, "1, 1, 8, 4"))
        diag = only(result, "AKA1020")
        assert diag.details["strides"] == [1, 1, 8, 4]
        assert diag.details["per_repeat"] == 64

    def test_contiguous_strides_matching_the_geometry_are_clean(self):
        result = analyze_body(FLOATS + repeat_cast(64, 4, "1, 1, 4, 8"))
        assert "AKA1020" not in codes_of(result)

    def test_single_repeat_never_advances_the_strides(self):
        result = analyze_body(FLOATS + repeat_cast(64, 1, "1, 1, 8, 4"))
        assert "AKA1020" not in codes_of(result)

    def test_sub_block_repeat_footprint_is_fatal(self):
        # 8 bf16 elements are 16 B: half a DaVinci block per repeat.
        result = analyze_body(FLOATS + repeat_cast(8, 4, "1, 1, 1, 1"))
        diag = only(result, "AKA1020")
        assert "not a whole 32-byte block" in diag.message

    def test_masked_form_without_an_element_count_is_skipped(self):
        body = (
            FLOATS
            + "uint32_t maskCount;" + nl
            + "AscendC::Cast(dstBf, srcF32, AscendC::RoundMode::CAST_RINT, maskCount, 4, {1, 1, 8, 4});"
            + nl
        )
        assert "AKA1020" not in codes_of(analyze_body(body))

    def test_coverage_artifact_is_always_published(self):
        result = analyze_body(FLOATS + deinterleave(16) + ANCHOR)
        artifact = result.artifacts.get("parity_coverage::test_kernel")
        assert isinstance(artifact, dict)
        assert artifact["rows_anchor"] == 8
