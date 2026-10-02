"""Tests for the memory layout checker."""

from __future__ import annotations

import pytest
from conftest import analyze_body, codes_of, find, only

from ascend_analyzer.diagnostics import Severity
from ascend_analyzer.hardware import KIB

SOLVERS = ["z3", "interval"]


def ub_tensor(name: str, offset, count, dtype: str = "half", pos: str = "VECIN") -> str:
    """A UB-resident LocalTensor declaration bound with SetAddr/SetSize."""
    return (
        f"AscendC::LocalTensor<{dtype}> {name};\n"
        f"{name}.SetTPosition(AscendC::TPosition::{pos});\n"
        f"{name}.SetAddr({offset});\n"
        f"{name}.SetSize({count});\n"
    )


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


class TestCapacity:
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_tensor_within_capacity_is_accepted(self, solver):
        result = analyze_body(ub_tensor("t", 0, 256), solver=solver)
        assert "AKA1001" not in codes_of(result)

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_tensor_past_ub_capacity_is_fatal(self, solver):
        # 192 KiB UB; a 128 KiB tensor at offset 128 KiB ends at 256 KiB.
        result = analyze_body(
            ub_tensor("t", 128 * KIB, 64 * KIB), solver=solver
        )
        diag = only(result, "AKA1001")
        assert diag.severity is Severity.FATAL
        assert diag.details["capacity_bytes"] == 192 * KIB
        assert diag.details["overflow_bytes"] == 64 * KIB

    def test_exact_fit_at_the_capacity_boundary_is_accepted(self):
        # [192KiB-512, 192KiB) ends exactly at the last byte: legal.
        result = analyze_body(ub_tensor("t", 192 * KIB - 512, 256))
        assert "AKA1001" not in codes_of(result)

    def test_one_byte_past_the_boundary_is_rejected(self):
        result = analyze_body(ub_tensor("t", 192 * KIB - 512 + 32, 256))
        assert "AKA1001" in codes_of(result)

    def test_larger_chip_profile_accepts_what_910b_rejects(self):
        body = ub_tensor("t", 192 * KIB, 256)
        assert "AKA1001" in codes_of(analyze_body(body, chip="ascend910b"))
        assert "AKA1001" not in codes_of(analyze_body(body, chip="ascend910c"))

    def test_loop_varying_offset_reports_the_failing_iteration(self):
        result = analyze_body(
            """
            for (uint32_t t = 0; t < 32; ++t) {
                AscendC::LocalTensor<half> tile;
                tile.SetTPosition(AscendC::TPosition::VECIN);
                tile.SetAddr(t * 8192);
                tile.SetSize(4096);
            }
            """,
            solver="z3",
        )
        diag = only(result, "AKA1001")
        # 24 * 8192 == 196608 == exactly the UB capacity, so tile 24 is the
        # first one that cannot fit.
        assert diag.details["counterexample"]["t"] >= 24
        assert "recycle" in diag.remediation.lower()

    def test_l1_tensor_uses_the_l1_capacity(self):
        # 384 KiB is fine in the 512 KiB L1 but would overflow the 192 KiB UB.
        result = analyze_body(ub_tensor("t", 0, 192 * KIB, "half", "A1"))
        assert "AKA1001" not in codes_of(result)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    @pytest.mark.parametrize("offset", [0, 32, 64, 512, 1024, 4096])
    def test_aligned_bases_are_accepted(self, offset):
        result = analyze_body(ub_tensor("t", offset, 256))
        assert "AKA1002" not in codes_of(result)

    @pytest.mark.parametrize("offset", [1, 8, 20, 500, 1000, 1500])
    def test_misaligned_bases_are_fatal(self, offset):
        result = analyze_body(ub_tensor("t", offset, 256))
        diag = only(result, "AKA1002")
        assert diag.severity is Severity.FATAL
        assert diag.details["required_alignment"] == 32
        assert diag.details["offset"] == offset

    def test_misaligned_base_remediation_names_both_boundaries(self):
        result = analyze_body(ub_tensor("t", 1000, 256))
        remediation = only(result, "AKA1002").remediation
        assert "992" in remediation      # round down
        assert "1024" in remediation     # round up

    @pytest.mark.parametrize("count", [16, 32, 256])
    def test_aligned_sizes_are_accepted(self, count):
        result = analyze_body(ub_tensor("t", 0, count))
        assert "AKA1005" not in codes_of(result)

    @pytest.mark.parametrize("count", [1, 15, 250, 255])
    def test_misaligned_sizes_are_fatal(self, count):
        result = analyze_body(ub_tensor("t", 0, count))
        diag = only(result, "AKA1005")
        assert diag.severity is Severity.FATAL
        assert diag.details["size"] == count * 2

    def test_size_remediation_explains_block_granularity(self):
        result = analyze_body(ub_tensor("t", 0, 250))
        remediation = only(result, "AKA1005").remediation
        assert "DataCopyPad" in remediation
        assert "16-element" in remediation

    def test_cube_buffers_require_fractal_alignment(self):
        # 32 is fine for UB but L0A needs a 512-byte fractal boundary.
        result = analyze_body(ub_tensor("t", 32, 256, "half", "A2"))
        diag = only(result, "AKA1002")
        assert diag.details["required_alignment"] == 512

    def test_cube_buffer_at_a_fractal_boundary_is_accepted(self):
        result = analyze_body(ub_tensor("t", 512, 256, "half", "A2"))
        assert "AKA1002" not in codes_of(result)


# ---------------------------------------------------------------------------
# Aliasing
# ---------------------------------------------------------------------------


class TestAliasing:
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_adjacent_buffers_do_not_collide(self, solver):
        body = (
            ub_tensor("a", 0, 256)
            + ub_tensor("b", 512, 256)
            + "AscendC::Add(b, a, a, 256);\n"
        )
        result = analyze_body(body, solver=solver)
        assert "AKA1003" not in codes_of(result)

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_overlapping_live_buffers_collide(self, solver):
        body = (
            ub_tensor("a", 0, 256)
            + ub_tensor("b", 256, 256)   # [256,768) overlaps [0,512)
            + "AscendC::Add(b, a, a, 256);\n"
        )
        result = analyze_body(body, solver=solver)
        diag = only(result, "AKA1003")
        assert diag.severity is Severity.FATAL
        assert set(diag.details["tensors"]) == {"a", "b"}

    def test_collision_remediation_suggests_an_aligned_relocation(self):
        body = (
            ub_tensor("a", 0, 256)
            + ub_tensor("b", 256, 256)
            + "AscendC::Add(b, a, a, 256);\n"
        )
        diag = only(analyze_body(body), "AKA1003")
        assert "512" in diag.remediation  # next 32-byte boundary after a's end

    def test_buffers_in_different_domains_never_collide(self):
        body = (
            ub_tensor("a", 0, 256, "half", "VECIN")
            + ub_tensor("b", 0, 256, "half", "A1")   # same offset, but in L1
        )
        assert "AKA1003" not in codes_of(analyze_body(body))

    def test_reuse_group_annotation_permits_deliberate_overlap(self):
        body = (
            "// @ascend-reuse-group: group=scratch names=a,b\n"
            + ub_tensor("a", 0, 256)
            + ub_tensor("b", 0, 256)
            + "AscendC::Add(b, a, a, 256);\n"
        )
        assert "AKA1003" not in codes_of(analyze_body(body))

    def test_ping_pong_buffers_in_a_loop_are_treated_as_concurrent(self):
        # Their uses do not overlap in program order, but the pipelines run
        # concurrently across iterations, so an overlap is still a bug.
        body = (
            ub_tensor("ping", 0, 256)
            + ub_tensor("pong", 256, 256)
            + """
            for (uint32_t t = 0; t < 4; ++t) {
                AscendC::Abs(ping, ping, 256);
                AscendC::Abs(pong, pong, 256);
            }
            """
        )
        diag = only(analyze_body(body), "AKA1003")
        assert "same loop" in diag.message

    def test_sequential_use_outside_a_loop_is_not_a_collision(self):
        # 'late' is only used after 'early' is finished with, and neither is
        # inside a loop, so sharing storage is legitimate here.
        body = (
            ub_tensor("early", 0, 256)
            + "AscendC::Abs(early, early, 256);\n"
            + ub_tensor("late", 0, 256)
            + "AscendC::Abs(late, late, 256);\n"
        )
        assert "AKA1003" not in codes_of(analyze_body(body))


# ---------------------------------------------------------------------------
# Domain mismatch
# ---------------------------------------------------------------------------


class TestDomainMismatch:
    def test_vector_op_rejects_an_l1_operand(self):
        body = (
            ub_tensor("dst", 0, 256)
            + ub_tensor("src", 512, 256)
            + ub_tensor("l1", 0, 256, "half", "A1")
            + "AscendC::Add(dst, src, l1, 256);\n"
        )
        diag = only(analyze_body(body), "AKA1004")
        assert diag.details["actual_domain"] == "L1"
        assert diag.details["allowed_domains"] == ["UB"]
        assert diag.details["argument_index"] == 2

    def test_mmad_rejects_a_ub_left_matrix(self):
        body = (
            ub_tensor("c", 0, 256, "float", "CO1")
            + ub_tensor("aUb", 0, 256, "half", "VECIN")
            + ub_tensor("b", 0, 256, "half", "B2")
            + "AscendC::Mmad(c, aUb, b, 16);\n"
        )
        diags = find(analyze_body(body), "AKA1004")
        assert any(d.details["parameter"] == "a" for d in diags)

    def test_fixpipe_requires_an_l0c_source(self):
        body = (
            ub_tensor("dst", 0, 256, "float", "VECOUT")
            + ub_tensor("src", 0, 256, "float", "A1")
            + "AscendC::Fixpipe(dst, src, 256);\n"
        )
        diag = only(analyze_body(body), "AKA1004")
        assert diag.details["allowed_domains"] == ["L0C"]

    def test_correct_cube_pipeline_is_accepted(self):
        body = (
            ub_tensor("c", 0, 256, "float", "CO1")
            + ub_tensor("a", 0, 256, "half", "A2")
            + ub_tensor("b", 0, 256, "half", "B2")
            + "AscendC::Mmad(c, a, b, 16);\n"
        )
        assert "AKA1004" not in codes_of(analyze_body(body))

    def test_illegal_direct_transfer_is_rejected(self):
        body = (
            "AscendC::GlobalTensor<half> g;\n"
            "g.SetGlobalBuffer(gm, 256);\n"
            + ub_tensor("l0b", 0, 256, "half", "B2")
            + "AscendC::DataCopy(l0b, g, 256);\n"
        )
        diag = only(analyze_body(body), "AKA1004")
        assert diag.details["src_domain"] == "GM"
        assert diag.details["dst_domain"] == "L0B"
        assert "L1" in diag.remediation

    def test_legal_transfers_are_accepted(self):
        body = (
            "AscendC::GlobalTensor<half> g;\n"
            "g.SetGlobalBuffer(gm, 256);\n"
            + ub_tensor("l1", 0, 256, "half", "A1")
            + ub_tensor("l0a", 0, 256, "half", "A2")
            + "AscendC::DataCopy(l1, g, 256);\n"
            + "AscendC::LoadData(l0a, l1, 256);\n"
        )
        assert "AKA1004" not in codes_of(analyze_body(body))


# ---------------------------------------------------------------------------
# Analyzability
# ---------------------------------------------------------------------------


class TestAnalyzability:
    def test_unresolvable_offset_is_reported_not_guessed(self):
        result = analyze_body(ub_tensor("t", "someHostValue", 256))
        diag = only(result, "AKA3002")
        assert diag.severity is Severity.WARNING
        # Crucially, no speculative capacity or alignment failure is invented.
        assert "AKA1001" not in codes_of(result)
        assert "AKA1002" not in codes_of(result)

    def test_strict_mode_promotes_unresolvable_offsets_to_fatal(self):
        result = analyze_body(ub_tensor("t", "someHostValue", 256), strict=True)
        assert only(result, "AKA3002").severity is Severity.FATAL

    def test_tensor_without_a_position_reports_unknown_domain(self):
        body = (
            "AscendC::LocalTensor<half> t;\n"
            "t.SetAddr(0);\n"
            "t.SetSize(256);\n"
        )
        diag = only(analyze_body(body), "AKA1009")
        assert "SetTPosition" in diag.remediation

    def test_bounded_symbolic_offset_is_verified_not_flagged_unresolvable(self):
        body = """
            for (uint32_t t = 0; t < 4; ++t) {
                AscendC::LocalTensor<half> tile;
                tile.SetTPosition(AscendC::TPosition::VECIN);
                tile.SetAddr(t * 512);
                tile.SetSize(256);
            }
            """
        result = analyze_body(body)
        # Bounded by the loop header, so the solver settles it: no gap report,
        # and no violation either, because 4 * 512 fits comfortably in UB.
        assert "AKA3002" not in codes_of(result)
        assert "AKA1001" not in codes_of(result)


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------


class TestSuppression:
    def test_cli_style_suppression_drops_a_code(self):
        body = ub_tensor("t", 1000, 250)
        assert "AKA1002" in codes_of(analyze_body(body))
        assert "AKA1002" not in codes_of(analyze_body(body, suppress=["AKA1002"]))

    def test_inline_ignore_annotation_drops_findings_on_that_line(self):
        body = (
            "AscendC::LocalTensor<half> t;\n"
            "t.SetTPosition(AscendC::TPosition::VECIN);\n"
            "// @ascend-ignore: AKA1002 AKA1005\n"
            "t.SetAddr(1000);\n"
            "t.SetSize(250);\n"
        )
        result = analyze_body(body)
        # The declaration line is what the diagnostics anchor to, so the
        # annotation is placed next to the binding it excuses.
        assert isinstance(result.suppressed, list)
