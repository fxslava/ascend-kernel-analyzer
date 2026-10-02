"""Regression tests for the architectural hardening work.

Covers the three upgraded capabilities end to end:

* **three-phase loop peeling** - critical boundary extraction, the peeled
  head / steady-state representative cycle / peeled tail traversal, and the
  symbolic trip-count fallback (`ast_visitor.py`);
* **Vector ALU UB bank conflicts** - the 8-bank / 32-byte-block delta model
  (AKA3006, `checkers/memory.py`);
* **351x SIMD/SIMT Unified Buffer budgeting** - the DataCache floor guard
  (AKA1010, `hardware.py` + `checkers/memory.py`).

Code-numbering note: the bank conflict is specified as "AKA3003", but that
code already belongs to ``SYMBOLIC_EVENT_ID`` and is pinned there by
``test_deadlock.py``; the bank conflict therefore carries ``AKA3006``, the
next free code in the 3xxx performance block.
"""

from __future__ import annotations

import time

import pytest
from conftest import KERNEL_DIR, analyze_body, codes_of, find, make_kernel, only

from ascend_analyzer import AnalyzerOptions, KernelAnalyzer
from ascend_analyzer.diagnostics import DiagnosticCollector, Severity
from ascend_analyzer.hardware import CHIP_PROFILES, HardwareModel
from ascend_analyzer.parsing import parse_source

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_kernel(body: str, preamble: str = ""):
    """Parse an inline kernel body without running any checker."""
    source = make_kernel(body, preamble)
    unit = parse_source(
        "<hardening>.cpp",
        source,
        HardwareModel.for_chip("ascend910b"),
        DiagnosticCollector(),
    )
    assert not unit.had_parse_errors
    assert len(unit.kernels) == 1
    return unit.kernels[0]


# ---------------------------------------------------------------------------
# Task 1: polyhedral loop peeling
# ---------------------------------------------------------------------------


class TestCriticalBoundaryExtraction:
    """Boundary switch points and modular periods are recovered from guards."""

    def test_lower_bound_guard_peels_a_head(self):
        kernel = parse_kernel(
            """
            for (uint32_t t = 0; t < 16; ++t) {
                if (t >= 1) {
                    AscendC::Add(yPing, xPing, xPing, 8);
                }
            }
            """
        )
        loop = kernel.loops[0]
        assert loop.peeled
        assert loop.peeled_head == 1          # t = 0 precedes the switch at 1
        assert loop.steady_reps == 1          # no parity: period 1
        assert loop.steady_first == 1
        assert loop.peeled_tail == 0
        # The head iteration folds the guard to false and is pruned; the
        # steady representative executes it.
        adds = [op for op in kernel.api_calls() if op.name == "Add"]
        assert len(adds) == 1

    def test_upper_bound_guard_peels_a_tail(self):
        kernel = parse_kernel(
            """
            for (uint32_t t = 0; t < 16; ++t) {
                if (t + 2 < 16) {
                    AscendC::Mul(yPong, xPong, xPong, 8);
                }
            }
            """
        )
        loop = kernel.loops[0]
        assert loop.peeled
        assert loop.peeled_head == 0
        assert loop.peeled_tail == 2          # the guard flips at t = 14
        assert loop.steady_first == 0
        # Steady representative executes the guarded op; both tail iterations
        # fold it to false and prune it.
        muls = [op for op in kernel.api_calls() if op.name == "Mul"]
        assert len(muls) == 1

    def test_point_guard_peels_both_sides(self):
        kernel = parse_kernel(
            """
            for (uint32_t t = 0; t < 32; ++t) {
                if (t == 3) {
                    AscendC::Add(yPing, xPing, xPing, 8);
                }
            }
            """
        )
        loop = kernel.loops[0]
        assert loop.peeled
        # The point condition is true only at t = 3: it switches on at 3 and
        # back off at 4, so the head covers every iteration before the state
        # settles (t = 0..3) and the steady state starts where it is false.
        assert loop.peeled_head == 4
        assert loop.steady_first == 4

    def test_parity_expression_sets_the_steady_period(self):
        kernel = parse_kernel(
            """
            #define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)
            for (uint32_t t = 0; t < 64; ++t) {
                int p = t & 1;
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EV(p));
            }
            """
        )
        loop = kernel.loops[0]
        assert loop.peeled
        assert loop.steady_reps == 2          # t & 1 -> period 2
        # One representative per parity, each folding its event id.
        assert [op.event_id for op in kernel.flag_ops()] == [0, 1]
        assert all(op.event_id is not None for op in kernel.flag_ops())

    def test_modulo_period_four(self):
        kernel = parse_kernel(
            """
            for (uint32_t t = 0; t < 64; ++t) {
                int q = t % 4;
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            """
        )
        assert kernel.loops[0].steady_reps == 4

    def test_mid_loop_phase_change_is_not_peeled(self):
        # The condition flips at t = 8, which is adjacent to neither boundary:
        # the prologue/steady/epilogue model does not apply, so the loop keeps
        # the exact single symbolic pass.
        kernel = parse_kernel(
            """
            for (uint32_t t = 0; t < 16; ++t) {
                if (t >= 8) {
                    AscendC::Add(yPing, xPing, xPing, 8);
                }
            }
            """
        )
        assert not kernel.loops[0].peeled
        assert not kernel.loops[0].unrolled

    def test_plain_long_loop_is_untouched(self):
        # No induction-dependent guards, no parity: nothing to peel.  The
        # single symbolic pass with a bounded induction variable is exactly
        # what the solver-based checks already reason about.
        kernel = parse_kernel(
            """
            for (uint32_t t = 0; t < 64; ++t) {
                AscendC::LocalTensor<half> tile;
                tile.SetTPosition(AscendC::TPosition::VECIN);
                tile.SetAddr(t * 512);
                tile.SetSize(256);
            }
            """
        )
        loop = kernel.loops[0]
        assert not loop.peeled and not loop.unrolled
        tensor = kernel.tensors["tile"]
        assert tensor.offset_value is None
        from ascend_analyzer.symbolic import free_vars

        bounds = free_vars(tensor.byte_offset)["t"]
        assert (bounds.lower, bounds.upper) == (0, 63)

    def test_small_loops_still_fully_unroll(self):
        kernel = parse_kernel(
            """
            #define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)
            for (int t = 0; t < 4; ++t) {
                int p = t & 1;
                SetFlag<HardEvent::MTE2_V>(EV(p));
            }
            """
        )
        loop = kernel.loops[0]
        assert loop.unrolled and not loop.peeled
        assert len(kernel.flag_ops()) == 4


class TestThreePhaseTraversal:
    """Head, steady cycle and tail are emitted with the right loop binding."""

    BODY = """
        #define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)
        for (uint32_t t = 0; t < 16; ++t) {
            int p = t & 1;
            if (t >= 1) {
                AscendC::Add(yPing, xPing, xPing, 8);
            }
            if (t + 2 < 16) {
                AscendC::Mul(yPong, xPong, xPong, 8);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EV(p));
        }
        """

    def test_phase_sizes(self):
        loop = parse_kernel(self.BODY).loops[0]
        assert loop.peeled
        assert loop.peeled_head == 1
        assert loop.steady_reps == 2
        assert loop.peeled_tail == 2
        assert loop.steady_first == 1

    def test_steady_representatives_carry_the_loop_id(self):
        kernel = parse_kernel(self.BODY)
        flags = kernel.flag_ops()
        # Five traced iterations: the peeled head (t = 0, straight-line, no
        # loop id), two steady representatives (the cyclic part of the marked
        # graph, carrying the loop id so back edges close them) and the two
        # peeled tail iterations (straight-line again).
        assert [op.loop_id for op in flags] == [None, 0, 0, None, None]

    def test_dead_branches_are_pruned_per_phase(self):
        kernel = parse_kernel(self.BODY)
        # head t=0: `t >= 1` false (no Add), `t + 2 < 16` true (one Mul);
        # steady t=1,2: both guards true (one Add and one Mul each);
        # tail t=14,15: `t >= 1` true (one Add each), prefetch guard false.
        adds = [op for op in kernel.api_calls() if op.name == "Add"]
        muls = [op for op in kernel.api_calls() if op.name == "Mul"]
        assert len(adds) == 4
        assert len(muls) == 3
        # Every event id folded: the parity selection resolved per iteration.
        assert [op.event_id for op in kernel.flag_ops()] == [0, 1, 0, 0, 1]


class TestSymbolicTripCountFallback:
    """Unresolved trip counts still get a head peel and a steady projection."""

    def test_fallback_peels_the_concrete_head(self):
        kernel = parse_kernel(
            """
            extern uint32_t limit;
            for (uint32_t t = 0; t < limit; ++t) {
                if (t >= 1) {
                    AscendC::Add(yPing, xPing, xPing, 8);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            """,
        )
        loop = kernel.loops[0]
        assert loop.trip_count is None
        assert loop.peeled
        assert loop.peeled_head == 1
        assert loop.steady_reps >= 1
        # The head iteration folded its guard concretely (no Add at t = 0);
        # the steady projection walks both arms conditionally.
        assert len(kernel.flag_ops()) == 1 + loop.steady_reps

    def test_fallback_preserves_loop_carried_edges(self):
        result = analyze_body(
            """
            extern uint32_t limit;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            for (uint32_t t = 0; t < limit; ++t) {
                int p = t & 1;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            """
        )
        # No fatal visitor exception, and the marked graph still models the
        # loop: the steady projection's operations carry the loop id, so the
        # loop-carried channel and its back edges survive.
        assert result.fatal_count == 0
        graph = result.artifacts["sync_graph::test_kernel"]
        assert graph["acyclic"]
        assert graph["peeled_loops"]
        assert any(node["loop_id"] is not None for node in graph["nodes"])

    def test_fallback_does_not_freeze_symbolic_offsets(self):
        kernel = parse_kernel(
            """
            extern uint32_t limit;
            for (uint32_t t = 0; t < limit; ++t) {
                AscendC::LocalTensor<half> tile;
                tile.SetTPosition(AscendC::TPosition::VECIN);
                tile.SetAddr(t * 512);
                tile.SetSize(256);
            }
            """
        )
        # A loop with nothing induction-dependent to peel keeps the exact
        # symbolic treatment even when the trip count is unknown.
        assert not kernel.loops[0].peeled


class TestLongPipelineFixture:
    """tests/kernels/loop_peeling_long.cpp: T = 512 without graph blowup."""

    @staticmethod
    @pytest.fixture(scope="class")
    def result():
        return KernelAnalyzer(AnalyzerOptions()).analyze_file(
            KERNEL_DIR / "loop_peeling_long.cpp"
        )

    def test_comes_back_completely_clean(self, result):
        assert result.diagnostics == []
        assert result.verdict == "accepted"
        assert result.exit_code(warnings_as_errors=True) == 0

    def test_strict_mode_is_also_clean(self):
        result = KernelAnalyzer(
            AnalyzerOptions(strict=True)
        ).analyze_file(KERNEL_DIR / "loop_peeling_long.cpp")
        assert result.fatal_count == 0
        assert result.warning_count == 0

    def test_graph_stays_within_the_node_budget(self, result):
        graph = result.artifacts["sync_graph::pipeline_long"]
        assert len(graph["nodes"]) <= 40
        assert graph["acyclic"]
        assert graph["cycles"] == []
        assert graph["topological_order"] is not None

    def test_analysis_finishes_within_the_time_budget(self):
        analyzer = KernelAnalyzer(AnalyzerOptions())
        analyzer.analyze_file(KERNEL_DIR / "loop_peeling_long.cpp")  # warm up
        start = time.perf_counter()
        analyzer.analyze_file(KERNEL_DIR / "loop_peeling_long.cpp")
        assert time.perf_counter() - start < 0.2

    def test_the_loop_was_peeled_not_unrolled(self, result):
        kernel = result.unit.kernels[0]
        loop = kernel.loops[0]
        assert loop.trip_count == 512
        assert loop.peeled and not loop.unrolled
        assert loop.steady_reps == 2
        assert loop.peeled_tail == 2
        # 512 iterations reduced to head + 2 steady representatives + tail.
        assert len(kernel.ops) < 60

    def test_every_event_id_resolved_per_parity(self, result):
        kernel = result.unit.kernels[0]
        assert kernel.flag_ops()
        assert all(op.event_id is not None for op in kernel.flag_ops())
        assert "AKA3003" not in codes_of(result)


# ---------------------------------------------------------------------------
# Task 2: Vector ALU UB bank conflicts (AKA3006)
# ---------------------------------------------------------------------------


def _ub(name: str, offset: int, count: int = 128, pos: str = "VECIN") -> str:
    return (
        f"AscendC::LocalTensor<half> {name};\n"
        f"{name}.SetTPosition(AscendC::TPosition::{pos});\n"
        f"{name}.SetAddr({offset});\n"
        f"{name}.SetSize({count});\n"
    )


class TestVectorBankConflicts:
    def test_fixture_reports_the_conflict_as_a_warning(self):
        result = KernelAnalyzer(AnalyzerOptions()).analyze_file(
            KERNEL_DIR / "bank_conflict_vec.cpp"
        )
        diag = only(result, "AKA3006")
        assert diag.severity is Severity.WARNING
        assert result.fatal_count == 0
        assert result.verdict == "accepted_with_warnings"

    def test_conflict_details_and_message(self):
        result = KernelAnalyzer(AnalyzerOptions()).analyze_file(
            KERNEL_DIR / "bank_conflict_vec.cpp"
        )
        diag = only(result, "AKA3006")
        assert diag.details == {
            **diag.details,
            "api": "Add",
            "src0": "aBad",
            "src1": "bBad",
            "src0_offset": 0,
            "src1_offset": 256,
            "bank": 0,
            "delta_blocks": 8,
        }
        assert "identical UB Bank 0" in diag.message
        assert "pipeline arbitration stall" in diag.message
        assert "0x0" in diag.message and "0x100" in diag.message

    def test_remediation_suggests_the_one_block_pad(self):
        result = KernelAnalyzer(AnalyzerOptions()).analyze_file(
            KERNEL_DIR / "bank_conflict_vec.cpp"
        )
        remediation = only(result, "AKA3006").remediation
        assert "+32 bytes" in remediation
        assert "bank-orthogonal" in remediation

    def test_a_32_byte_skew_is_the_negative_control(self):
        result = KernelAnalyzer(AnalyzerOptions()).analyze_file(
            KERNEL_DIR / "bank_conflict_vec.cpp"
        )
        # Exactly one finding: the skewed (aGood, bGood) pair stays silent.
        conflicts = find(result, "AKA3006")
        assert len(conflicts) == 1
        assert conflicts[0].details["src0"] == "aBad"

    def test_same_tensor_through_both_ports_is_exempt(self):
        # Add(y, x, x) reads one buffer through both operand ports; there is
        # no second bank to collide with (and this is the shape every
        # ping-pong fixture computes with).
        result = analyze_body(_ub("x", 0) + _ub("y", 512) + "AscendC::Add(y, x, x, 128);\n")
        assert "AKA3006" not in codes_of(result)

    @pytest.mark.parametrize(
        "offset, conflicts",
        [(256, True), (512, True), (1024, True), (288, False), (544, False)],
    )
    def test_the_delta_model_decides(self, offset, conflicts):
        result = analyze_body(
            _ub("a", 0)
            + _ub("b", offset)
            + _ub("dst", 4096, pos="VECOUT")
            + "AscendC::Add(dst, a, b, 128);\n"
        )
        assert ("AKA3006" in codes_of(result)) is conflicts

    def test_unresolved_offsets_are_not_speculated_about(self):
        result = analyze_body(
            _ub("a", 0)
            + _ub("b", "hostValue")
            + _ub("dst", 4096, pos="VECOUT")
            + "AscendC::Add(dst, a, b, 128);\n"
        )
        assert "AKA3006" not in codes_of(result)


# ---------------------------------------------------------------------------
# Task 3: 351x SIMD/SIMT Unified Buffer budgeting (AKA1010)
# ---------------------------------------------------------------------------


class Test351xHardwareProfile:
    def test_partition_fields_are_defined(self):
        chip = CHIP_PROFILES["ascend351x"]
        assert chip.ub_total_bytes == 262144          # 256 KiB
        assert chip.compiler_reserved_bytes == 8192   # 8 KiB
        assert chip.min_datacache_bytes == 32768      # 32 KiB
        assert chip.max_usable_ub_bytes == 221184     # 216 KiB
        assert chip.enforces_datacache_partition

    def test_other_chips_do_not_partition_the_ub(self):
        for name in ("ascend910b", "ascend910c"):
            chip = CHIP_PROFILES[name]
            assert not chip.enforces_datacache_partition
            assert chip.ub_total_bytes is None

    def test_datacache_arithmetic(self):
        hw = HardwareModel.for_chip("ascend351x")
        # 220 KiB allocated leaves 28 KiB: below the 32 KiB floor.
        assert hw.simt_datacache_available(225280) == 28672
        # 216 KiB allocated leaves exactly the floor.
        assert hw.simt_datacache_available(221184) == 32768
        # Unpartitioned parts have no notion of a DataCache budget.
        assert (
            HardwareModel.for_chip("ascend910b").simt_datacache_available(0) is None
        )

    def test_describe_discloses_the_partition(self):
        payload = HardwareModel.for_chip("ascend351x").describe()
        partition = payload["ub_partition"]
        assert partition["min_datacache_bytes"] == 32768
        assert partition["max_usable_ub_bytes"] == 221184
        assert HardwareModel.for_chip("ascend910b").describe()["ub_partition"] is None


_SIMT_BODY = """
AscendC::LocalTensor<half> bigA;
bigA.SetTPosition(AscendC::TPosition::VECCALC);
bigA.SetAddr(0);
bigA.SetSize(56320);
AscendC::LocalTensor<half> bigB;
bigB.SetTPosition(AscendC::TPosition::VECCALC);
bigB.SetAddr(112640);
bigB.SetSize(56320);
asc_call_vf(bigA, 0);
asc_call_vf(bigB, 1);
"""


class TestDataCacheGuard:
    def test_fixture_is_fatal_on_351x(self):
        result = KernelAnalyzer(
            AnalyzerOptions(chip="ascend351x")
        ).analyze_file(KERNEL_DIR / "simt_ub_budget_351x.cpp")
        diag = only(result, "AKA1010")
        assert diag.severity is Severity.FATAL
        assert result.verdict == "rejected"
        assert result.exit_code() == 1

    def test_message_names_the_budget_numbers(self):
        result = KernelAnalyzer(
            AnalyzerOptions(chip="ascend351x")
        ).analyze_file(KERNEL_DIR / "simt_ub_budget_351x.cpp")
        diag = only(result, "AKA1010")
        assert "available: 28672 B" in diag.message
        assert ">= 32768 B" in diag.message
        assert "216 KB limit" in diag.message
        assert diag.details["allocated_bytes"] == 225280
        assert diag.details["available_bytes"] == 28672

    def test_fixture_on_the_default_chip_reports_plain_overflow(self):
        result = KernelAnalyzer(AnalyzerOptions()).analyze_file(
            KERNEL_DIR / "simt_ub_budget_351x.cpp"
        )
        assert only(result, "AKA1001").severity is Severity.FATAL
        assert "AKA1010" not in codes_of(result)

    def test_allocating_200_kib_stays_within_the_budget(self):
        body = _SIMT_BODY.replace("56320", "51200")  # 2 x 100 KiB
        result = analyze_body(body, chip="ascend351x")
        assert "AKA1010" not in codes_of(result)
        assert result.fatal_count == 0

    def test_without_simt_markers_the_guard_stays_silent(self):
        body = _SIMT_BODY.replace("asc_call_vf(bigA, 0);", "").replace(
            "asc_call_vf(bigB, 1);", ""
        )
        result = analyze_body(body, chip="ascend351x")
        assert "AKA1010" not in codes_of(result)

    def test_unpartitioned_chips_never_raise_it(self):
        # 910C also has a 256 KiB UB but no DataCache partition: the same
        # 220 KiB SIMT kernel is merely large there, not invalid.
        result = analyze_body(_SIMT_BODY, chip="ascend910c")
        assert "AKA1010" not in codes_of(result)
        assert result.fatal_count == 0

    def test_init_buffer_allocations_count_toward_the_budget(self):
        body = """
            TPipe pipe;
            TBuf<TPosition::VECCALC> bigBuf;
            pipe.InitBuffer(bigBuf, 229376);      /* 224 KiB > 216 KiB       */
            LocalTensor<half> t = bigBuf.Get<half>();
            asc_call_vf(t, 0);
            """
        result = analyze_body(body, chip="ascend351x")
        diag = only(result, "AKA1010")
        assert diag.details["allocated_bytes"] == 229376


# ---------------------------------------------------------------------------
# Symbolic helper
# ---------------------------------------------------------------------------


def test_substitute_replaces_and_folds():
    from ascend_analyzer.symbolic import BinOp, Const, Var, substitute, to_int, var

    expr = BinOp("*", var("t"), Const(512))
    assert to_int(substitute(expr, "t", 3)) == 1536
    assert substitute(Var("u", 0, 7), "t", 1) == Var("u", 0, 7)
    assert substitute(None, "t", 1) is None
