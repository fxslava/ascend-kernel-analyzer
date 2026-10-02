"""Tests for the analytical pipeline performance & overlap profiler.

Covers the three layers of the 4xxx block: the hardware latency model
(bandwidths, cube contraction cycles, hand-off penalty), the ASAP schedule
over the acyclic dependency DAG (makespan, busy/idle/stall, overlap ratio,
bottleneck classification), and the AKA4001 / AKA4002 advisories - including
the guarantee that correct, well-pipelined kernels stay silent.
"""

from __future__ import annotations

import io
import json

import pytest
from conftest import KERNEL_DIR, analyze_body, codes_of, find

from ascend_analyzer import AnalyzerOptions, KernelAnalyzer
from ascend_analyzer.checkers.perf_model import PerfModelChecker
from ascend_analyzer.diagnostics import Severity
from ascend_analyzer.hardware import CHIP_PROFILES, HardwareModel, Pipe
from ascend_analyzer.report.json_report import build_json_report
from ascend_analyzer.report.terminal import TerminalReporter

FIXTURE = KERNEL_DIR / "nvfp4_pipelined_dequant.cpp"


def profile_of(result, kernel_name: str) -> dict:
    profile = result.artifacts.get(f"perf_profile::{kernel_name}")
    assert isinstance(profile, dict) and "makespan_cycles" in profile, profile
    return profile


def analyze_fixture(**kwargs) -> object:
    return KernelAnalyzer(AnalyzerOptions(**kwargs)).analyze_file(FIXTURE)


# ---------------------------------------------------------------------------
# Task 1: the hardware latency model
# ---------------------------------------------------------------------------


class TestLatencyModel:
    def test_transfer_bandwidths_are_defined(self):
        hw = HardwareModel.for_chip("ascend910b")
        chip = hw.chip
        assert chip.mte2_bytes_per_cycle == 64      # GM -> L1 move-in
        assert chip.mte1_bytes_per_cycle == 64      # L1 -> L0A/L0B
        assert chip.fixpipe_bytes_per_cycle == 64   # L0C -> GM drain
        assert hw.pipe_bytes_per_cycle(Pipe.MTE2) == 64
        assert hw.pipe_bytes_per_cycle(Pipe.MTE1) == 64
        assert hw.pipe_bytes_per_cycle(Pipe.FIX) == 64
        # MTE3 stores share the GM write path with fixpipe; the vector unit
        # retires one register footprint per cycle.
        assert hw.pipe_bytes_per_cycle(Pipe.MTE3) == chip.fixpipe_bytes_per_cycle
        assert hw.pipe_bytes_per_cycle(Pipe.V) == chip.vector_bytes

    def test_pipes_without_a_bulk_model_are_nominal(self):
        hw = HardwareModel.for_chip("ascend910b")
        assert hw.pipe_bytes_per_cycle(Pipe.S) is None
        assert hw.pipe_bytes_per_cycle(Pipe.M) is None

    def test_sync_handoff_penalty(self):
        assert HardwareModel.for_chip("ascend910b").chip.sync_handoff_cycles == 30

    @pytest.mark.parametrize(
        "m,k,n,expected",
        [
            (16, 16, 16, 1),     # exactly one fractal
            (32, 16, 16, 2),     # two M fractals
            (16, 64, 16, 4),     # four K steps
            (16, 16, 32, 2),     # two N fractals
            (32, 32, 32, 8),
            # Ceil-rounded partial fractals: 3 x 2 x 2.
            (33, 17, 17, 12),
        ],
    )
    def test_cube_contraction_cycles(self, m, k, n, expected):
        chip = CHIP_PROFILES["ascend910b"]
        assert chip.cube_contraction_cycles(m, k, n) == expected

    def test_profile_file_overrides_the_model(self, tmp_path):
        profile = tmp_path / "fast.json"
        profile.write_text(
            json.dumps(
                {
                    "name": "fast910b",
                    "base": "ascend910b",
                    "mte2_bytes_per_cycle": 128,
                    "sync_handoff_cycles": 12,
                }
            ),
            encoding="utf-8",
        )
        hw = HardwareModel.from_profile_file(profile)
        assert hw.pipe_bytes_per_cycle(Pipe.MTE2) == 128
        assert hw.chip.sync_handoff_cycles == 12
        assert hw.chip.mte1_bytes_per_cycle == 64  # inherited from the base

    def test_describe_discloses_the_model(self):
        payload = HardwareModel.for_chip("ascend910b").describe()
        model = payload["perf_model"]
        assert model["sync_handoff_cycles"] == 30
        assert model["cube_macs_per_cycle"] == 4096


# ---------------------------------------------------------------------------
# Task 2: service times and the ASAP schedule
# ---------------------------------------------------------------------------


COPY_AND_ADD = """
    AscendC::GlobalTensor<half> g;
    g.SetGlobalBuffer(gm, 256);
    AscendC::LocalTensor<half> u;
    u.SetTPosition(AscendC::TPosition::VECIN);
    u.SetAddr(0);
    u.SetSize(128);
    AscendC::DataCopy(u, g, 128);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Add(u, u, u, 128);
"""


class TestSchedule:
    def test_service_times_follow_the_bandwidth_model(self):
        result = analyze_body(COPY_AND_ADD)
        pipes = {p["pipe"]: p for p in profile_of(result, "test_kernel")["pipes"]}
        # 256 B at 64 B/cycle = 4 cycles for the copy; flags cost 1 each.
        assert pipes["PIPE_MTE2"]["busy_cycles"] == 4 + 1
        # 256 B at 256 B/cycle = 1 cycle for the vector op, plus the wait.
        assert pipes["PIPE_V"]["busy_cycles"] == 1 + 1

    def test_makespan_includes_the_hand_off_penalty(self):
        result = analyze_body(COPY_AND_ADD)
        # MTE2: copy [0,4), set [4,5).  The wait on V cannot issue before
        # 5 + 30 = 35, so V runs [35,36) and [36,37): makespan 37.
        assert profile_of(result, "test_kernel")["makespan_cycles"] == 37

    def test_the_wait_is_an_exposed_stall_on_its_pipe(self):
        result = analyze_body(COPY_AND_ADD)
        pipes = {p["pipe"]: p for p in profile_of(result, "test_kernel")["pipes"]}
        assert pipes["PIPE_V"]["stall_cycles"] == 35
        assert pipes["PIPE_MTE2"]["stall_cycles"] == 0

    def test_cube_shape_is_recovered_from_the_call_site(self):
        result = analyze_body(
            """
            #define MM 16
            #define KK 128
            #define NN 16
            AscendC::LocalTensor<float> c;
            c.SetTPosition(AscendC::TPosition::CO1);
            c.SetAddr(0);
            c.SetSize(64);
            AscendC::LocalTensor<half> a;
            a.SetTPosition(AscendC::TPosition::A2);
            a.SetAddr(0);
            a.SetSize(2048);
            AscendC::LocalTensor<half> b;
            b.SetTPosition(AscendC::TPosition::B2);
            b.SetAddr(0);
            b.SetSize(2048);
            mad_mx(c, 0, a, 0, b, 0,
                   mmad_t::shape_t((uint16_t)MM, (uint16_t)KK, (uint16_t)NN), ctl);
            """,
        )
        pipes = {p["pipe"]: p for p in profile_of(result, "test_kernel")["pipes"]}
        # shape (16, 128, 16) -> 1 x 8 x 1 fractals = 8 cycles.
        assert pipes["PIPE_M"]["busy_cycles"] == 8

    def test_cube_without_a_shape_falls_back_to_operand_bytes(self):
        result = analyze_body(
            """
            AscendC::LocalTensor<float> c;
            c.SetTPosition(AscendC::TPosition::CO1);
            c.SetAddr(0);
            c.SetSize(64);
            AscendC::LocalTensor<half> a;
            a.SetTPosition(AscendC::TPosition::A2);
            a.SetAddr(0);
            a.SetSize(2048);
            AscendC::LocalTensor<half> b;
            b.SetTPosition(AscendC::TPosition::B2);
            b.SetAddr(0);
            b.SetSize(2048);
            AscendC::Mmad(c, a, b, 16);
            """
        )
        pipes = {p["pipe"]: p for p in profile_of(result, "test_kernel")["pipes"]}
        # SetSize counts elements: 2 x 4096 B (half) + 256 B (float dst)
        # = 8448 B at the 1024 B/cycle cube read throughput = 9 cycles.
        assert pipes["PIPE_M"]["busy_cycles"] == 9

    def test_transfer_volume_uses_the_tile_not_the_gm_buffer(self):
        result = analyze_body(
            """
            AscendC::GlobalTensor<half> g;
            g.SetGlobalBuffer(gm, 65536);
            AscendC::LocalTensor<half> u;
            u.SetTPosition(AscendC::TPosition::VECIN);
            u.SetAddr(0);
            u.SetSize(128);
            AscendC::DataCopy(u, g, 128);
            """
        )
        pipes = {p["pipe"]: p for p in profile_of(result, "test_kernel")["pipes"]}
        # 256 B tile (not the 128 KiB GM tensor) -> 4 cycles, not 2048.
        assert pipes["PIPE_MTE2"]["busy_cycles"] == 4


# ---------------------------------------------------------------------------
# Task 2: bottleneck classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_memory_bound_when_a_move_engine_drives_the_critical_path(self):
        # No hand-offs at all: the move engine's busy time is the makespan.
        result = analyze_body(
            """
            AscendC::GlobalTensor<half> g;
            g.SetGlobalBuffer(gm, 65536);
            AscendC::LocalTensor<half> u;
            u.SetTPosition(AscendC::TPosition::VECIN);
            u.SetAddr(0);
            u.SetSize(16384);
            AscendC::DataCopy(u, g, 16384);
            AscendC::Add(u, u, u, 16384);
            """
        )
        assert profile_of(result, "test_kernel")["bottleneck"] == "MEMORY_BOUND"

    def test_compute_bound_when_the_vector_unit_drives_the_makespan(self):
        result = analyze_body(
            """
            AscendC::LocalTensor<half> u;
            u.SetTPosition(AscendC::TPosition::VECCALC);
            u.SetAddr(0);
            u.SetSize(16384);
            AscendC::Add(u, u, u, 16384);
            AscendC::Abs(u, u, 16384);
            """
        )
        assert profile_of(result, "test_kernel")["bottleneck"] == "COMPUTE_BOUND"

    def test_drain_bound_when_only_the_tail_serializes(self):
        # Bulk work up front, then a chain of flag hand-offs nothing overlaps:
        # nearly the whole makespan runs after the last transfer/compute op.
        result = analyze_body(
            """
            AscendC::GlobalTensor<half> g;
            g.SetGlobalBuffer(gm, 256);
            AscendC::LocalTensor<half> u;
            u.SetTPosition(AscendC::TPosition::VECIN);
            u.SetAddr(0);
            u.SetSize(128);
            AscendC::DataCopy(u, g, 128);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            """
        )
        profile = profile_of(result, "test_kernel")
        assert profile["bottleneck"] == "DRAIN_BOUND"
        assert profile["drain_cycles"] > profile["makespan_cycles"] // 2

    def test_deadlocked_kernels_are_not_scheduled(self):
        result = KernelAnalyzer(AnalyzerOptions()).analyze_file(
            KERNEL_DIR / "pingpong_broken.cpp"
        )
        profile = result.artifacts["perf_profile::vec_add_pingpong_broken"]
        assert "skipped" in profile
        assert "AKA4001" not in codes_of(result)


# ---------------------------------------------------------------------------
# Task 2 + 4: the nvfp4 fixture
# ---------------------------------------------------------------------------


class TestNvfp4PipelineA:
    """Pipeline A: double-buffered, but small tiles expose the hand-offs."""

    @staticmethod
    @pytest.fixture(scope="class")
    def result():
        return analyze_fixture()

    def test_reports_exposed_sync_stalls(self, result):
        stalls = find(result, "AKA4001")
        assert len(stalls) == 2  # one per kernel
        assert all(d.severity is Severity.WARNING for d in stalls)
        assert result.fatal_count == 0

    def test_the_m_fix_hand_off_is_highlighted(self, result):
        profile = profile_of(result, "nvfp4_dequant_overlap")
        assert profile["stall_by_route"]["M_FIX"] >= 400
        assert any(
            "M_FIX" in d.message and "M_FIX 647 cycles" in d.message
            for d in find(result, "AKA4001")
        )

    def test_small_tiles_underutilise_the_cube(self, result):
        diags = find(result, "AKA4002")
        assert len(diags) == 2
        assert all(d.details["cube_utilization"] < 0.5 for d in diags)
        assert all(d.severity is Severity.WARNING for d in diags)

    def test_shape_recovery_gives_exact_contraction_cycles(self, result):
        profile = profile_of(result, "nvfp4_dequant_overlap")
        cube = next(p for p in profile["pipes"] if p["pipe"] == "PIPE_M")
        # 8 tiles of (16, 128, 16): 8 x (1 x 8 x 1) = 64 mad cycles, plus
        # the interleaved flag issue (a wait and two sets per tile).
        assert cube["busy_cycles"] == 8 * 8 + 8 * 3


class TestNvfp4PipelineB:
    """Pipeline B: the same stages serialized behind five handshakes."""

    @staticmethod
    @pytest.fixture(scope="class")
    def result():
        return analyze_fixture()

    def test_reports_serialized_execution(self, result):
        profile = profile_of(result, "nvfp4_dequant_serial")
        assert profile["bottleneck"] == "SYNC_BOUND"
        assert "AKA4001" in codes_of(result)

    def test_overlap_is_worse_than_the_pipelined_variant(self, result):
        a = profile_of(result, "nvfp4_dequant_overlap")
        b = profile_of(result, "nvfp4_dequant_serial")
        assert b["overlap_ratio"] < 0.25 < a["overlap_ratio"]
        assert b["overlap_ratio"] * 2 < a["overlap_ratio"] * 2

    def test_stall_cycles_dwarf_issued_work(self, result):
        profile = profile_of(result, "nvfp4_dequant_serial")
        assert profile["total_stall_cycles"] > 3 * profile["total_busy_cycles"]


class TestAdvisoryDiscipline:
    """Correct, well-pipelined kernels must not be nagged."""

    @pytest.mark.parametrize(
        "fixture",
        ["pingpong_clean.cpp", "cube_tbuf_pipeline.cpp", "loop_peeling_long.cpp"],
    )
    def test_clean_fixtures_stay_silent(self, fixture):
        result = KernelAnalyzer(AnalyzerOptions()).analyze_file(
            KERNEL_DIR / fixture
        )
        assert not [c for c in codes_of(result) if c.startswith("AKA4")]

    def test_the_gate_is_the_makespan_not_the_ratio(self):
        # A tiny kernel can have a terrible stall ratio (see COPY_AND_ADD)
        # without warranting advice: short kernels are inherently fill/drain.
        result = analyze_body(COPY_AND_ADD)
        assert "AKA4001" not in codes_of(result)

    def test_advisory_thresholds_are_class_attributes(self):
        # Tuning surface for the model, pinned so changes are deliberate.
        assert PerfModelChecker.min_makespan_cycles == 500
        assert PerfModelChecker.stall_ratio_threshold == 0.30
        assert PerfModelChecker.cube_util_threshold == 0.50


# ---------------------------------------------------------------------------
# Task 3: reporting
# ---------------------------------------------------------------------------


def render(result, **kwargs) -> str:
    stream = io.StringIO()
    reporter = TerminalReporter(stream=stream, color=False, ascii_only=True, **kwargs)
    reporter.report(
        result.unit, result.hardware, result.diagnostics,
        result.artifacts, result.solver_name,
    )
    return stream.getvalue()


class TestReporting:
    @staticmethod
    @pytest.fixture(scope="class")
    def result():
        return analyze_fixture()

    def test_terminal_shows_the_performance_summary(self, result):
        text = render(result)
        assert "Pipeline performance - nvfp4_dequant_overlap" in text
        assert "estimated makespan" in text
        assert "bottleneck            SYNC_BOUND" in text
        assert "concurrency" in text

    def test_terminal_renders_utilization_bars(self, result):
        text = render(result)
        assert "utilization per pipeline" in text
        gauge_lines = [
            line for line in text.splitlines()
            if line.strip().startswith("PIPE_") and "[" in line and "]" in line
        ]
        assert {"PIPE_MTE2", "PIPE_M", "PIPE_FIX"} <= {
            line.split("[")[0].strip() for line in gauge_lines
        }
        # The bars are drawn from the ASCII glyph set.
        for line in gauge_lines:
            assert set(line.split("[")[1].split("]")[0]) <= {"#", "."}

    def test_perf_section_can_be_omitted(self, result):
        assert "Pipeline performance" not in render(result, show_perf=False)
        assert "Pipeline performance" in render(result)

    def test_json_report_carries_the_profile(self, result):
        report = build_json_report(
            result.unit, result.hardware, result.diagnostics,
            result.artifacts, solver_name=result.solver_name,
        )
        kernel = next(
            k for k in report["kernels"] if k["name"] == "nvfp4_dequant_serial"
        )
        assert kernel["performance"]["bottleneck"] == "SYNC_BOUND"
        assert kernel["performance"]["makespan_cycles"] > 0

    def test_cli_lists_the_4xxx_block(self, capsys):
        from ascend_analyzer.cli import main

        assert main(["--list-codes"]) == 0
        out = capsys.readouterr().out
        assert "AKA4001  Exposed sync bubbles / pipeline stalls" in out
        assert "AKA4002  Cube compute underutilization" in out

    def test_perf_checker_can_be_disabled(self):
        result = KernelAnalyzer(
            AnalyzerOptions(disable=("perf",))
        ).analyze_file(FIXTURE)
        assert not [c for c in codes_of(result) if c.startswith("AKA4")]
