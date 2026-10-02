"""End-to-end checks against the kernel fixtures.

Each fixture declares what it expects in its own header comment, so these
tests and ``harness.py`` share one source of truth.  The explicit assertions
below pin down the behaviour that matters most: the broken ping-pong kernel
must report both headline defect classes, and the clean counterpart must stay
completely silent.
"""

from __future__ import annotations

import pytest
from conftest import KERNEL_DIR, codes_of, find
from harness import Expectation, check_fixture

from ascend_analyzer import AnalyzerOptions, KernelAnalyzer

FIXTURES = sorted(KERNEL_DIR.glob("*.cpp"))
assert FIXTURES, "no fixtures found"


@pytest.fixture(scope="module")
def default_analyzer() -> KernelAnalyzer:
    return KernelAnalyzer(AnalyzerOptions())


# ---------------------------------------------------------------------------
# Declaration-driven checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_matches_its_declared_expectations(path, default_analyzer):
    outcome = check_fixture(path, default_analyzer)
    assert not outcome.missing_codes, (
        f"{path.name}: declared codes not reported: {outcome.missing_codes}; "
        f"actually reported {codes_of(outcome.result)}"
    )
    assert outcome.fatal_mismatch is None, f"{path.name}: {outcome.fatal_mismatch}"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_every_fixture_parses_cleanly(path, default_analyzer):
    result = default_analyzer.analyze_file(path)
    assert not result.unit.had_parse_errors, (
        f"{path.name} produced parse errors at lines "
        f"{[loc.line for loc in result.unit.parse_error_locs]}"
    )
    assert result.unit.kernels, f"{path.name}: no kernel entry point found"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_every_finding_carries_a_remediation(path, default_analyzer):
    result = default_analyzer.analyze_file(path)
    for diag in result.diagnostics:
        assert diag.remediation.strip(), (
            f"{path.name}:{diag.loc.line} {diag.code.value} has no remediation"
        )
        assert diag.loc.line > 0


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
@pytest.mark.parametrize("solver", ["z3", "interval"])
def test_both_solver_backends_agree_on_the_verdict(path, solver):
    analyzer = KernelAnalyzer(AnalyzerOptions(solver=solver))
    assert analyzer.analyze_file(path).verdict == (
        KernelAnalyzer(AnalyzerOptions(solver="z3")).analyze_file(path).verdict
    )


# ---------------------------------------------------------------------------
# The headline case
# ---------------------------------------------------------------------------


class TestBrokenPingPong:
    """The kernel the specification asks for: a deadlock plus a bad offset."""

    @staticmethod
    @pytest.fixture(scope="class")
    def result():
        return KernelAnalyzer(AnalyzerOptions()).analyze_file(
            KERNEL_DIR / "pingpong_broken.cpp"
        )

    def test_is_rejected(self, result):
        assert result.verdict == "rejected"
        assert result.exit_code() == 1

    def test_reports_the_intentional_deadlock(self, result):
        unprimed = find(result, "AKA2005")
        # Both buffer slots, on both the V_MTE2 and MTE3_V channels.
        assert len(unprimed) == 4
        routes = {d.details["route"] for d in unprimed}
        assert routes == {"V_MTE2", "MTE3_V"}
        for diag in unprimed:
            assert "hangs" in diag.message
            assert "Prime the flag once before the loop" in diag.remediation

    def test_deadlock_is_found_as_a_graph_cycle(self, result):
        graph = result.artifacts["sync_graph::vec_add_pingpong_broken"]
        assert not graph["acyclic"]
        assert graph["cycles"]
        assert graph["topological_order"] is None

    def test_reports_the_intentional_unaligned_offset(self, result):
        misaligned = find(result, "AKA1002")
        offsets = {d.details["offset"] for d in misaligned}
        # yPing starts at 1000 and yPong at 1500; neither is a 32-byte multiple.
        assert offsets == {1000, 1500}
        for diag in misaligned:
            assert diag.details["required_alignment"] == 32

    def test_reports_the_ragged_tile_length(self, result):
        sizes = find(result, "AKA1005")
        # All four 250-element tiles are 500 bytes: not a whole 32-byte block.
        assert len(sizes) == 4
        assert {d.details["size"] for d in sizes} == {500}

    def test_reports_the_ping_pong_collision(self, result):
        collisions = find(result, "AKA1003")
        assert len(collisions) == 1
        assert set(collisions[0].details["tensors"]) == {"xPong", "yPing"}

    def test_reports_the_reserved_event_id(self, result):
        reserved = find(result, "AKA2003")
        assert len(reserved) == 2
        assert {d.details["event_id"] for d in reserved} == {6}

    def test_reports_the_global_barrier_as_a_warning(self, result):
        barriers = find(result, "AKA3001")
        assert len(barriers) == 1
        assert barriers[0].severity.value == "WARNING"

    def test_memory_map_shows_the_collision(self, result):
        kernel = result.unit.kernels[0]
        from ascend_analyzer.report.memory_map import build_memory_maps

        ub = build_memory_maps(kernel, result.hardware)[0]
        colliding = {s.name for s in ub.segments if s.collides}
        assert colliding == {"xPong", "yPing"}


class TestCleanPingPong:
    """The negative control: a correct kernel must produce nothing at all."""

    @staticmethod
    @pytest.fixture(scope="class")
    def result():
        return KernelAnalyzer(AnalyzerOptions()).analyze_file(
            KERNEL_DIR / "pingpong_clean.cpp"
        )

    def test_is_accepted_with_no_findings(self, result):
        assert result.diagnostics == []
        assert result.verdict == "accepted"
        assert result.exit_code(warnings_as_errors=True) == 0

    def test_synchronisation_graph_is_acyclic(self, result):
        graph = result.artifacts["sync_graph::vec_add_pingpong_clean"]
        assert graph["acyclic"]
        assert graph["topological_order"] is not None
        assert graph["cycles"] == []

    def test_loop_carried_handshakes_are_primed(self, result):
        # The loop body references its induction variable and the trip count
        # is small, so the visitor replays it per iteration: the trace is the
        # full straight-line execution and every handshake is matched
        # forward, the prologue primes feeding the first iteration's waits.
        graph = result.artifacts["sync_graph::vec_add_pingpong_clean"]
        pairs = graph["sync_pairs"]
        assert pairs
        carried = [p for p in pairs if p["loop_carried"]]
        assert carried == []
        assert all(p["tokens"] == 0 for p in pairs)
        assert graph["acyclic"]

    def test_ub_layout_is_contiguous_and_aligned(self, result):
        from ascend_analyzer.report.memory_map import build_memory_maps

        ub = build_memory_maps(result.unit.kernels[0], result.hardware)[0]
        assert ub.gaps() == []
        assert ub.collision_count == 0
        assert [s.start for s in ub.segments] == [0, 512, 1024, 1536]
        assert all(s.start % 32 == 0 and s.size % 32 == 0 for s in ub.segments)

    def test_still_clean_under_strict_mode(self):
        result = KernelAnalyzer(
            AnalyzerOptions(strict=True)
        ).analyze_file(KERNEL_DIR / "pingpong_clean.cpp")
        assert result.fatal_count == 0

    def test_still_clean_on_the_other_chip_profiles(self):
        for chip in ("ascend910c", "ascend351x"):
            result = KernelAnalyzer(
                AnalyzerOptions(chip=chip)
            ).analyze_file(KERNEL_DIR / "pingpong_clean.cpp")
            assert result.fatal_count == 0, chip


class TestBrokenVersusClean:
    """The two kernels differ only in the seeded defects."""

    def test_clean_fixes_everything_broken_reports(self):
        analyzer = KernelAnalyzer(AnalyzerOptions())
        broken = analyzer.analyze_file(KERNEL_DIR / "pingpong_broken.cpp")
        clean = analyzer.analyze_file(KERNEL_DIR / "pingpong_clean.cpp")
        assert set(broken.codes())
        assert set(clean.codes()) == set()

    def test_both_kernels_have_the_same_shape(self):
        analyzer = KernelAnalyzer(AnalyzerOptions())
        broken = analyzer.analyze_file(KERNEL_DIR / "pingpong_broken.cpp")
        clean = analyzer.analyze_file(KERNEL_DIR / "pingpong_clean.cpp")
        # Four UB tiles and one loop in both; only the numbers differ.
        for result in (broken, clean):
            kernel = result.unit.kernels[0]
            assert len(kernel.loops) == 1
            assert sum(1 for t in kernel.tensors.values() if t.is_sram) == 4


# ---------------------------------------------------------------------------
# Expectation parsing
# ---------------------------------------------------------------------------


class TestExpectationParsing:
    def test_reads_codes_and_fatal_count(self):
        expectation = Expectation.from_source(KERNEL_DIR / "pingpong_broken.cpp")
        assert "AKA2005" in expectation.codes
        assert "AKA1002" in expectation.codes
        assert expectation.fatal_count == 13

    def test_clean_fixture_declares_no_codes_and_no_fatals(self):
        expectation = Expectation.from_source(KERNEL_DIR / "pingpong_clean.cpp")
        assert expectation.codes == set()
        assert expectation.fatal_count == 0

    def test_does_not_confuse_expect_with_expect_fatal(self):
        expectation = Expectation.from_source(KERNEL_DIR / "ub_overflow.cpp")
        assert expectation.codes == {"AKA1001"}
        assert expectation.fatal_count == 1
