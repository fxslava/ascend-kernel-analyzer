"""Tests for the hazard checker's parity extensions (AKA2010 vacuity, AKA2012
loop-invariant GM flush).

A ``SetFlag``/``WaitFlag`` pair written like the drain idiom but raised with
nothing outstanding on the producing pipeline is pseudo-synchronisation: the
wait releases immediately and whatever cross-pipeline traffic it was meant to
guard stays unordered.  A store to global memory whose destination ignores
its own loop's induction variable overwrites the previous iteration's bytes
- inside a reduction nest, an early flush of the partial sums.
"""

from __future__ import annotations

from conftest import analyze_body, find

from ascend_analyzer.diagnostics import Severity

nl = chr(10)


def ub(name: str, offset: int) -> str:
    return (
        f"AscendC::LocalTensor<half> {name};" + nl
        + f"{name}.SetTPosition(AscendC::TPosition::VECCALC);" + nl
        + f"{name}.SetAddr({offset});" + nl
        + f"{name}.SetSize(256);" + nl
    )


DECLS = ub("a", 0) + ub("b", 1024) + ub("acc", 2048)
GLOBAL = (
    "AscendC::GlobalTensor<half> g;" + nl
    + "g.SetGlobalBuffer(gm, 4096);" + nl
)

MOVE_IN = "AscendC::DataCopy(a, g, 128);" + nl
COMPUTE = "AscendC::Add(acc, a, b, 128);" + nl
MOVE_OUT = "AscendC::DataCopy(g, acc, 128);" + nl


def set_flag(route: str, eid: int = 0) -> str:
    return f"AscendC::SetFlag<AscendC::HardEvent::{route}>(EVENT_ID{eid});" + nl


def wait_flag(route: str, eid: int = 0) -> str:
    return f"AscendC::WaitFlag<AscendC::HardEvent::{route}>(EVENT_ID{eid});" + nl


def vacuous(result):
    return [d for d in find(result, "AKA2010") if d.details.get("vacuity")]


def flushes(result):
    return [
        d
        for d in find(result, "AKA2012")
        if d.details.get("hazard_kind") == "WAW_LOOP_INVARIANT"
    ]


class TestVacuousHandshake:
    def test_a_pair_with_no_producer_work_is_pseudo(self):
        """The fake WaitLoad(): set/wait with nothing outstanding behind it."""
        result = analyze_body(DECLS + GLOBAL + set_flag("MTE2_V") + wait_flag("MTE2_V"))
        diags = vacuous(result)
        assert len(diags) == 1
        assert diags[0].details["vacuity"] == "no_producer_work"
        assert diags[0].severity is Severity.WARNING

    def test_the_canonical_drain_idiom_is_not_flagged(self):
        result = analyze_body(
            DECLS + GLOBAL + MOVE_IN + set_flag("MTE2_V") + wait_flag("MTE2_V") + COMPUTE
        )
        assert vacuous(result) == []

    def test_produce_then_wrong_route_names_the_right_producer(self):
        """Vector work fenced by an MTE2_V route: the route names the wrong
        producer, and the finding says which pipe the work is on."""
        result = analyze_body(
            DECLS + GLOBAL + COMPUTE + set_flag("MTE2_V") + wait_flag("MTE2_V") + MOVE_OUT
        )
        diags = vacuous(result)
        assert len(diags) == 1
        assert diags[0].details["vacuity"] == "no_producer_work"
        assert diags[0].details["producer_hint"] == ["PIPE_V"]
        assert diags[0].details["route"] == "MTE2_V"

    def test_the_finding_points_at_the_wait_and_names_the_pair(self):
        result = analyze_body(DECLS + GLOBAL + set_flag("MTE2_V") + wait_flag("MTE2_V"))
        diag = vacuous(result)[0]
        assert "pseudo-synchronisation" in diag.message
        assert "MTE2" in diag.message
        labels = [label for label, _ in diag.related]
        assert "flag raised here" in labels
        assert "flag waited here" in labels

    def test_a_prologue_prime_is_not_pseudo(self):
        """Setting the flag before the loop it opens is loop priming."""
        body = (
            DECLS + GLOBAL
            + set_flag("V_MTE2")
            + "for (uint32_t t = 0; t < 4; ++t) {" + nl
            + "    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);" + nl
            + "    AscendC::DataCopy(a, g, 128);" + nl
            + "    AscendC::Add(acc, a, b, 128);" + nl
            + "    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);" + nl
            + "}" + nl
        )
        assert vacuous(analyze_body(body)) == []

    def test_a_redundant_re_drain_is_reported_as_nothing_new(self):
        """A global barrier already ordered everything; the pair adds none."""
        result = analyze_body(
            DECLS + GLOBAL
            + MOVE_IN
            + "AscendC::PipeBarrier<PIPE_ALL>();" + nl
            + set_flag("MTE2_V") + wait_flag("MTE2_V")
            + COMPUTE
        )
        diags = vacuous(result)
        assert len(diags) == 1
        assert diags[0].details["vacuity"] == "nothing_new"


class TestLoopInvariantGmFlush:
    def test_inner_loop_store_to_an_outer_indexed_row_is_fatal(self):
        body = (
            DECLS + GLOBAL
            + "for (int row0 = 0; row0 < 64; row0 += 8) {" + nl
            + "    AscendC::Duplicate(acc, 0, 8);" + nl
            + "    for (int col0 = 0; col0 < 512; col0 += 64) {" + nl
            + "        AscendC::Add(acc, acc, a, 64);" + nl
            + "        AscendC::DataCopyPad(g[row0], acc, 64);" + nl
            + "    }" + nl
            + "}" + nl
        )
        result = analyze_body(body)
        diags = flushes(result)
        assert len(diags) == 1
        assert diags[0].severity is Severity.FATAL
        assert diags[0].details["induction_var"] == "col0"
        assert diags[0].details["reduction"] is True
        assert "early flush" in diags[0].message

    def test_destination_advancing_with_its_loop_is_clean(self):
        body = (
            DECLS + GLOBAL
            + "for (int col0 = 0; col0 < 512; col0 += 64) {" + nl
            + "    AscendC::Add(acc, acc, a, 64);" + nl
            + "    AscendC::DataCopyPad(g[col0], acc, 64);" + nl
            + "}" + nl
        )
        assert flushes(analyze_body(body)) == []

    def test_plain_overwrite_loop_reports_a_warning(self):
        body = (
            DECLS + GLOBAL
            + "for (int c = 0; c < 8; ++c) {" + nl
            + "    AscendC::DataCopy(g, acc, 64);" + nl
            + "}" + nl
        )
        diags = flushes(analyze_body(body))
        assert len(diags) == 1
        assert diags[0].severity is Severity.WARNING
        assert diags[0].details["reduction"] is False

    def test_atomic_accumulation_is_clean(self):
        body = (
            DECLS + GLOBAL
            + "AscendC::SetAtomicAdd<half>();" + nl
            + "for (int c = 0; c < 8; ++c) {" + nl
            + "    AscendC::DataCopy(g, acc, 64);" + nl
            + "}" + nl
            + "AscendC::SetAtomicNone();" + nl
        )
        assert flushes(analyze_body(body)) == []

    def test_single_trip_loop_is_clean(self):
        body = (
            DECLS + GLOBAL
            + "for (int c = 0; c < 1; ++c) {" + nl
            + "    AscendC::DataCopy(g, acc, 64);" + nl
            + "}" + nl
        )
        assert flushes(analyze_body(body)) == []

    def test_kernel_level_store_is_clean(self):
        assert flushes(analyze_body(DECLS + GLOBAL + MOVE_OUT)) == []

    def test_the_remediation_names_the_induction_variable(self):
        body = (
            DECLS + GLOBAL
            + "for (int c = 0; c < 8; ++c) {" + nl
            + "    AscendC::DataCopy(g, acc, 64);" + nl
            + "}" + nl
        )
        diag = flushes(analyze_body(body))[0]
        assert "c" in diag.remediation
