"""Tests for the cross-pipeline hazard checker (AKA2010).

Each DaVinci pipeline runs its own queue in order; two pipelines are unordered
unless the program orders them with a `SetFlag`/`WaitFlag` pair on the right
channel or a global `PipeBarrier`. A tensor written on one pipe and read on
another with neither between them is a race that wins by timing.

The exclusions matter as much as the detections, because each one is a place
where a warning would be wrong rather than merely noisy: same-pipe pairs are
already ordered by program order, `TQue` storage is handed over by runtime
flags this analyzer cannot see, and the two core views of a mix kernel are
separate binaries with their own UB.
"""

from __future__ import annotations

import pytest
from conftest import analyze, analyze_body, codes_of, find

from ascend_analyzer.diagnostics import Severity

nl = chr(10)


def ub(name: str, offset: int, size: int = 256, position: str = "VECIN") -> str:
    """A UB-resident `LocalTensor` at a concrete address."""
    return (
        f"AscendC::LocalTensor<half> {name};" + nl
        + f"{name}.SetTPosition(AscendC::TPosition::{position});" + nl
        + f"{name}.SetAddr({offset});" + nl
        + f"{name}.SetSize({size});" + nl
    )


#: Offsets chosen 17 blocks apart so the bank-conflict check (AKA3006) stays
#: quiet and these tests only ever see hazard findings.
DECLS = ub("a", 0) + ub("b", 544) + ub("c", 1088)
GLOBAL = (
    "AscendC::GlobalTensor<half> g;" + nl
    + "g.SetGlobalBuffer(gm, 4096);" + nl
)

MOVE_IN = "AscendC::DataCopy(a, g, 128);" + nl      # MTE2 writes 'a'
COMPUTE = "AscendC::Add(c, a, b, 128);" + nl        # V reads 'a', writes 'c'
MOVE_OUT = "AscendC::DataCopy(g, c, 128);" + nl     # MTE3 reads 'c'
BARRIER = "AscendC::PipeBarrier<PIPE_ALL>();" + nl


def set_flag(route: str, eid: int = 0) -> str:
    return f"AscendC::SetFlag<AscendC::HardEvent::{route}>(EVENT_ID{eid});" + nl


def wait_flag(route: str, eid: int = 0) -> str:
    return f"AscendC::WaitFlag<AscendC::HardEvent::{route}>(EVENT_ID{eid});" + nl


def coverage(result, kernel_name: str = "test_kernel") -> dict:
    artifact = result.artifacts.get(f"hazard_coverage::{kernel_name}")
    assert isinstance(artifact, dict), result.artifacts
    return artifact


def hazards(result) -> list:
    return find(result, "AKA2010")


class TestDetection:
    def test_unguarded_move_in_then_compute_is_flagged(self):
        """`DataCopy` on MTE2 then `Add` on V with nothing between them."""
        result = analyze_body(DECLS + GLOBAL + MOVE_IN + COMPUTE)
        assert len(hazards(result)) == 1
        assert coverage(result)["hazards"] == 1

    def test_unguarded_compute_then_move_out_is_flagged(self):
        """The reverse direction, V writing and MTE3 reading."""
        result = analyze_body(DECLS + GLOBAL + COMPUTE + MOVE_OUT)
        assert len(hazards(result)) == 1

    def test_the_finding_is_a_warning(self):
        """A race can still pass every run; that is why it is worth naming."""
        diag = hazards(analyze_body(DECLS + GLOBAL + MOVE_IN + COMPUTE))[0]
        assert diag.severity is Severity.WARNING

    def test_the_finding_names_both_sides(self):
        diag = hazards(analyze_body(DECLS + GLOBAL + MOVE_IN + COMPUTE))[0]
        assert diag.details["tensor"] == "a"
        assert diag.details["write_pipe"] == "PIPE_MTE2"
        assert diag.details["read_pipe"] == "PIPE_V"
        assert diag.details["write_op"] == "DataCopy"
        assert diag.details["read_op"] == "Add"

    def test_the_remediation_names_the_channel_to_use(self):
        diag = hazards(analyze_body(DECLS + GLOBAL + MOVE_IN + COMPUTE))[0]
        assert "MTE2_V" in diag.remediation
        assert "PipeBarrier" in diag.remediation

    def test_the_finding_points_at_the_read(self):
        """The read is where the stale data is consumed."""
        diag = hazards(analyze_body(DECLS + GLOBAL + MOVE_IN + COMPUTE))[0]
        assert diag.details["read_line"] == diag.loc.line
        assert diag.details["write_line"] < diag.details["read_line"]


class TestSufficientSynchronisation:
    def test_a_matching_flag_pair_clears_the_hazard(self):
        body = (
            DECLS + GLOBAL + MOVE_IN
            + set_flag("MTE2_V") + wait_flag("MTE2_V")
            + COMPUTE
        )
        result = analyze_body(body)
        assert hazards(result) == []
        # The pair was still examined - silence must not read as "not checked".
        assert coverage(result)["pairs_checked"] == 1

    def test_a_global_barrier_clears_the_hazard(self):
        result = analyze_body(DECLS + GLOBAL + MOVE_IN + BARRIER + COMPUTE)
        assert hazards(result) == []
        assert coverage(result)["pairs_checked"] == 1

    def test_the_other_direction_is_cleared_too(self):
        body = (
            DECLS + GLOBAL + COMPUTE
            + set_flag("V_MTE3") + wait_flag("V_MTE3")
            + MOVE_OUT
        )
        assert hazards(analyze_body(body)) == []

    @pytest.mark.parametrize("eid", [0, 1, 2, 3, 4, 5])
    def test_any_usable_event_id_counts(self, eid):
        body = (
            DECLS + GLOBAL + MOVE_IN
            + set_flag("MTE2_V", eid) + wait_flag("MTE2_V", eid)
            + COMPUTE
        )
        assert hazards(analyze_body(body)) == []


class TestInsufficientSynchronisation:
    def test_a_flag_on_the_wrong_channel_does_not_count(self):
        """`V_MTE2` orders V before MTE2, not MTE2 before V."""
        body = (
            DECLS + GLOBAL + MOVE_IN
            + set_flag("V_MTE2") + wait_flag("V_MTE2")
            + COMPUTE
        )
        assert len(hazards(analyze_body(body))) == 1

    def test_a_pair_entirely_before_the_write_does_not_count(self):
        """The set has to follow the write it is meant to publish."""
        body = (
            DECLS + GLOBAL
            + set_flag("MTE2_V") + wait_flag("MTE2_V")
            + MOVE_IN + COMPUTE
        )
        assert len(hazards(analyze_body(body))) == 1

    def test_a_set_without_a_wait_does_not_count(self):
        body = DECLS + GLOBAL + MOVE_IN + set_flag("MTE2_V") + COMPUTE
        assert len(hazards(analyze_body(body))) == 1

    def test_a_wait_without_a_set_does_not_count(self):
        body = DECLS + GLOBAL + MOVE_IN + wait_flag("MTE2_V") + COMPUTE
        assert len(hazards(analyze_body(body))) == 1

    def test_a_barrier_outside_the_span_does_not_count(self):
        body = DECLS + GLOBAL + BARRIER + MOVE_IN + COMPUTE
        assert len(hazards(analyze_body(body))) == 1


class TestExclusions:
    def test_same_pipe_pairs_are_not_examined(self):
        """Program order already serialises one pipeline's own queue."""
        body = (
            DECLS + GLOBAL
            + "AscendC::Add(c, a, b, 128);" + nl
            + "AscendC::Mul(b, c, a, 128);" + nl
        )
        result = analyze_body(body)
        assert hazards(result) == []
        assert coverage(result)["pairs_checked"] == 0

    def test_queue_managed_storage_is_out_of_static_reach(self):
        """`AllocTensor` storage is handed over by runtime flags we cannot see."""
        source = (
            '#include "kernel_operator.h"' + nl
            + "using namespace AscendC;" + nl
            + 'extern "C" __global__ __aicore__ void test_kernel(__gm__ half* gm) {' + nl
            + "  GlobalTensor<half> g; g.SetGlobalBuffer(gm, 4096);" + nl
            + "  TPipe pipe;" + nl
            + "  TQue<TPosition::VECIN, 1> q;" + nl
            + "  TQue<TPosition::VECOUT, 1> o;" + nl
            + "  pipe.InitBuffer(q, 1, 512);" + nl
            + "  pipe.InitBuffer(o, 1, 512);" + nl
            + "  LocalTensor<half> a = q.AllocTensor<half>();" + nl
            + "  LocalTensor<half> c = o.AllocTensor<half>();" + nl
            + "  DataCopy(a, g, 128);" + nl
            + "  Add(c, a, a, 128);" + nl
            + "}" + nl
        )
        result = analyze(source)
        assert hazards(result) == []
        artifact = coverage(result)
        assert artifact["pairs_checked"] == 0
        assert artifact["queue_mediated_tensors"] >= 1

    def test_a_loop_carried_ping_pong_is_not_a_hazard(self):
        """`WaitFlag ... read ... write ... SetFlag` guards this iteration's
        read against the previous iteration's write."""
        body = DECLS + GLOBAL + (
            "for (int t = 0; t < 4; ++t) {" + nl
            + wait_flag("MTE2_V")
            + COMPUTE
            + MOVE_IN
            + set_flag("MTE2_V")
            + "}" + nl
        )
        result = analyze_body(body)
        assert hazards(result) == []
        assert coverage(result)["pairs_checked"] == 1


class TestCoverageArtifact:
    def test_the_artifact_is_published_even_when_clean(self):
        """A check that silently did not run must not look like a clean one."""
        body = (
            DECLS + GLOBAL + MOVE_IN
            + set_flag("MTE2_V") + wait_flag("MTE2_V")
            + COMPUTE
        )
        artifact = coverage(analyze_body(body))
        assert artifact == {
            "pairs_checked": 1,
            "hazards": 0,
            "reported": 0,
            "queue_mediated_tensors": 0,
        }

    def test_the_artifact_is_published_with_nothing_to_check(self):
        result = analyze_body(DECLS + GLOBAL + "AscendC::Add(c, a, b, 128);" + nl)
        assert coverage(result)["pairs_checked"] == 0

    def test_reported_never_exceeds_hazards(self):
        result = analyze_body(DECLS + GLOBAL + MOVE_IN + COMPUTE + MOVE_OUT)
        artifact = coverage(result)
        assert artifact["reported"] <= artifact["hazards"]
        assert len(hazards(result)) == artifact["reported"]


class TestCheckerPlumbing:
    def test_the_checker_can_be_disabled(self):
        from conftest import make_kernel

        from ascend_analyzer.analyzer import AnalyzerOptions, KernelAnalyzer

        source = make_kernel(DECLS + GLOBAL + MOVE_IN + COMPUTE)
        analyzer = KernelAnalyzer(AnalyzerOptions(disable=("hazard",)))
        result = analyzer.analyze_source("<test>.cpp", source)
        assert "AKA2010" not in codes_of(result)
        assert not any(
            k.startswith("hazard_coverage::") for k in result.artifacts
        )

    def test_the_checker_is_listed(self):
        from ascend_analyzer.analyzer import AVAILABLE_CHECKERS

        assert "hazard" in AVAILABLE_CHECKERS

    def test_the_code_has_a_title(self):
        from ascend_analyzer.diagnostics import CODE_TITLES

        assert "AKA2010" in CODE_TITLES

    def test_a_hazard_does_not_reject_the_kernel(self):
        """AKA2010 is a warning, so it must not change the exit verdict."""
        result = analyze_body(DECLS + GLOBAL + MOVE_IN + COMPUTE)
        assert result.fatal_count == 0
        assert result.verdict == "accepted_with_warnings"
        assert "AKA2010" in codes_of(result)
