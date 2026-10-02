"""Tests for the pipeline synchronisation checker."""

from __future__ import annotations

import pytest
from conftest import analyze_body, codes_of, find, only

from ascend_analyzer.diagnostics import Severity


def sync_graph(result, kernel_name: str = "test_kernel") -> dict:
    graph = result.artifacts.get(f"sync_graph::{kernel_name}")
    assert isinstance(graph, dict)
    return graph


SET = "AscendC::SetFlag<AscendC::HardEvent::{route}>(EVENT_ID{eid});"
WAIT = "AscendC::WaitFlag<AscendC::HardEvent::{route}>(EVENT_ID{eid});"


def set_flag(route: str, eid: int = 0) -> str:
    return SET.format(route=route, eid=eid) + "\n"


def wait_flag(route: str, eid: int = 0) -> str:
    return WAIT.format(route=route, eid=eid) + "\n"


# ---------------------------------------------------------------------------
# Event id hygiene
# ---------------------------------------------------------------------------


class TestEventIds:
    @pytest.mark.parametrize("eid", [6, 7])
    def test_reserved_event_ids_are_fatal(self, eid):
        result = analyze_body(set_flag("MTE2_V", eid) + wait_flag("MTE2_V", eid))
        diags = find(result, "AKA2003")
        assert len(diags) == 2  # both the set and the wait are flagged
        assert all(d.severity is Severity.FATAL for d in diags)
        assert diags[0].details["event_id"] == eid
        assert "EVENT_ID0..EVENT_ID5" in diags[0].remediation

    @pytest.mark.parametrize("eid", [0, 1, 2, 3, 4, 5])
    def test_usable_event_ids_are_accepted(self, eid):
        result = analyze_body(set_flag("MTE2_V", eid) + wait_flag("MTE2_V", eid))
        assert "AKA2003" not in codes_of(result)

    def test_out_of_range_event_id_is_fatal(self):
        body = (
            "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(BIG);\n"
            "AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(BIG);\n"
        )
        result = analyze_body(body, preamble="constexpr int32_t BIG = 11;\n")
        assert "AKA2008" in codes_of(result)

    def test_reserved_ids_follow_the_chip_profile(self):
        body = set_flag("MTE2_V", 6) + wait_flag("MTE2_V", 6)
        # The default profile reserves 6 and 7 on every shipped chip, so this
        # is fatal everywhere; the point is that the message cites the profile.
        diag = find(analyze_body(body, chip="ascend910c"), "AKA2003")[0]
        assert "910C" in diag.remediation

    def test_unresolvable_event_id_warns_instead_of_guessing(self):
        body = """
            for (uint32_t i = 0; i < 4; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pick(i));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pick(i));
            }
            """
        result = analyze_body(body)
        assert "AKA3003" in codes_of(result)
        assert all(d.severity is Severity.WARNING for d in find(result, "AKA3003"))


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


class TestPairing:
    def test_matched_pair_is_accepted(self):
        result = analyze_body(set_flag("MTE2_V") + wait_flag("MTE2_V"))
        assert result.fatal_count == 0

    def test_set_without_wait_is_fatal(self):
        diag = only(analyze_body(set_flag("MTE2_V")), "AKA2001")
        assert diag.severity is Severity.FATAL
        assert diag.details["route"] == "MTE2_V"
        assert "WaitFlag" in diag.remediation

    def test_wait_without_set_is_fatal(self):
        diag = only(analyze_body(wait_flag("MTE2_V")), "AKA2002")
        assert diag.severity is Severity.FATAL
        assert "hang" in diag.message

    def test_event_ids_are_separate_channels(self):
        # Setting id 0 does not satisfy a wait on id 1.
        result = analyze_body(set_flag("MTE2_V", 0) + wait_flag("MTE2_V", 1))
        assert "AKA2001" in codes_of(result)
        assert "AKA2002" in codes_of(result)

    def test_routes_are_separate_channels(self):
        result = analyze_body(set_flag("MTE2_V") + wait_flag("V_MTE3"))
        assert "AKA2001" in codes_of(result)
        assert "AKA2002" in codes_of(result)

    def test_double_set_without_an_intervening_wait_warns(self):
        body = set_flag("MTE2_V") + set_flag("MTE2_V") + wait_flag("MTE2_V")
        diag = only(analyze_body(body), "AKA2007")
        assert diag.severity is Severity.WARNING

    def test_self_route_is_flagged_as_redundant(self):
        body = set_flag("V_V") + wait_flag("V_V")
        diags = find(analyze_body(body), "AKA2009")
        assert diags and all(d.severity is Severity.WARNING for d in diags)

    def test_imbalance_inside_a_conditional_is_softened_to_a_warning(self):
        # The trace walks both arms of an 'if', so an imbalance that exists
        # only because of that over-approximation must not be fatal.
        body = """
            if (block_idx == 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            """
        result = analyze_body(body)
        diag = only(result, "AKA2001")
        assert diag.severity is Severity.WARNING


# ---------------------------------------------------------------------------
# Loop balance and priming
# ---------------------------------------------------------------------------


class TestLoopBalance:
    def test_balanced_primed_loop_is_accepted(self):
        body = (
            set_flag("V_MTE2")
            + """
            for (uint32_t t = 0; t < 4; ++t) {
            """
            + wait_flag("V_MTE2")
            + set_flag("MTE2_V")
            + wait_flag("MTE2_V")
            + set_flag("V_MTE2")
            + "}\n"
            + wait_flag("V_MTE2")
        )
        result = analyze_body(body)
        assert result.fatal_count == 0
        assert sync_graph(result)["acyclic"]

    def test_unprimed_loop_carried_wait_is_fatal(self):
        body = """
            for (uint32_t t = 0; t < 4; ++t) {
            """ + wait_flag("V_MTE2") + set_flag("MTE2_V") \
             + wait_flag("MTE2_V") + set_flag("V_MTE2") + "}\n"
        result = analyze_body(body)
        diag = only(result, "AKA2005")
        assert diag.severity is Severity.FATAL
        assert "nothing primes it before the loop" in diag.message
        assert "SetFlag" in diag.remediation
        # The set is later in the body than the wait that consumes it.
        assert diag.details["set_line"] > diag.details["wait_line"]

    def test_extra_set_in_the_loop_body_is_fatal(self):
        body = (
            "for (uint32_t t = 0; t < 4; ++t) {\n"
            + set_flag("MTE2_V")
            + set_flag("MTE2_V")
            + wait_flag("MTE2_V")
            + "}\n"
        )
        diag = only(analyze_body(body), "AKA2006")
        assert diag.severity is Severity.FATAL
        assert diag.details["sets"] == 2
        assert diag.details["waits"] == 1
        assert "overflow" in diag.message

    def test_extra_wait_in_the_loop_body_is_fatal(self):
        body = (
            "for (uint32_t t = 0; t < 4; ++t) {\n"
            + set_flag("MTE2_V")
            + wait_flag("MTE2_V")
            + wait_flag("MTE2_V")
            + "}\n"
        )
        diag = only(analyze_body(body), "AKA2006")
        assert diag.details["sets"] == 1
        assert diag.details["waits"] == 2
        assert "blocks for ever" in diag.message

    def test_prologue_primes_must_be_drained(self):
        # Primed twice, drained once: one hardware event slot is left occupied.
        body = (
            set_flag("V_MTE2")
            + set_flag("V_MTE2")
            + "for (uint32_t t = 0; t < 4; ++t) {\n"
            + wait_flag("V_MTE2")
            + set_flag("V_MTE2")
            + "}\n"
            + wait_flag("V_MTE2")
        )
        diag = only(analyze_body(body), "AKA2001")
        assert diag.details["outer_sets"] == 2
        assert diag.details["outer_waits"] == 1


# ---------------------------------------------------------------------------
# Deadlock
# ---------------------------------------------------------------------------


class TestDeadlock:
    def test_straight_line_handshake_is_acyclic(self):
        body = (
            set_flag("MTE2_V")
            + wait_flag("MTE2_V")
            + set_flag("V_MTE3")
            + wait_flag("V_MTE3")
        )
        result = analyze_body(body)
        graph = sync_graph(result)
        assert graph["acyclic"]
        assert graph["topological_order"] is not None
        assert "AKA2004" not in codes_of(result)

    def test_unprimed_cycle_is_detected_in_the_graph(self):
        body = """
            for (uint32_t t = 0; t < 4; ++t) {
            """ + wait_flag("V_MTE2") + set_flag("MTE2_V") \
             + wait_flag("MTE2_V") + set_flag("V_MTE2") + "}\n"
        graph = sync_graph(analyze_body(body))
        assert not graph["acyclic"]
        assert graph["cycles"]

    def test_a_missing_prime_is_reported_once_not_as_every_cycle(self):
        # The precise root cause (AKA2005) is reported; the cycles it creates
        # are suppressed so one mistake does not become a wall of findings.
        body = """
            for (uint32_t t = 0; t < 4; ++t) {
            """ + wait_flag("V_MTE2") + set_flag("MTE2_V") \
             + wait_flag("MTE2_V") + set_flag("V_MTE2") + "}\n"
        result = analyze_body(body)
        assert len(find(result, "AKA2005")) == 1
        assert "AKA2004" not in codes_of(result)
        assert result.artifacts.get("suppressed_cycles::test_kernel", 0) >= 1

    def test_priming_removes_the_cycle(self):
        body = (
            set_flag("V_MTE2")
            + "for (uint32_t t = 0; t < 4; ++t) {\n"
            + wait_flag("V_MTE2")
            + set_flag("MTE2_V")
            + wait_flag("MTE2_V")
            + set_flag("V_MTE2")
            + "}\n"
            + wait_flag("V_MTE2")
        )
        result = analyze_body(body)
        assert sync_graph(result)["acyclic"]
        assert "AKA2005" not in codes_of(result)

    def test_loop_carried_edges_carry_their_token_count(self):
        body = (
            set_flag("V_MTE2")
            + "for (uint32_t t = 0; t < 4; ++t) {\n"
            + wait_flag("V_MTE2")
            + set_flag("MTE2_V")
            + wait_flag("MTE2_V")
            + set_flag("V_MTE2")
            + "}\n"
            + wait_flag("V_MTE2")
        )
        pairs = sync_graph(analyze_body(body))["sync_pairs"]
        carried = [p for p in pairs if p["loop_carried"]]
        assert carried, "the V_MTE2 handshake is loop-carried"
        # A primed loop-carried edge holds a token, which is what breaks the cycle.
        assert all(p["tokens"] >= 1 for p in carried)


# ---------------------------------------------------------------------------
# Barriers
# ---------------------------------------------------------------------------


class TestBarriers:
    def test_global_barrier_warns_with_a_localised_alternative(self):
        body = (
            "AscendC::GlobalTensor<half> g;\n"
            "g.SetGlobalBuffer(gm, 256);\n"
            "AscendC::LocalTensor<half> u;\n"
            "u.SetTPosition(AscendC::TPosition::VECIN);\n"
            "u.SetAddr(0);\n"
            "u.SetSize(256);\n"
            "AscendC::DataCopy(u, g, 256);\n"
            "AscendC::PipeBarrier<PIPE_ALL>();\n"
            "AscendC::Abs(u, u, 256);\n"
        )
        diag = only(analyze_body(body), "AKA3001")
        assert diag.severity is Severity.WARNING
        # The advice should name the two pipelines actually involved here.
        assert "MTE2_V" in diag.remediation

    def test_single_pipe_barrier_is_not_flagged(self):
        result = analyze_body("AscendC::PipeBarrier<PIPE_V>();")
        assert "AKA3001" not in codes_of(result)

    def test_isasi_global_barrier_is_flagged(self):
        assert "AKA3001" in codes_of(analyze_body("pipe_barrier(PIPE_ALL);"))

    def test_consecutive_identical_barriers_report_the_duplicate(self):
        body = "AscendC::PipeBarrier<PIPE_V>();\nAscendC::PipeBarrier<PIPE_V>();\n"
        diags = find(analyze_body(body), "AKA3005")
        assert diags and diags[0].severity is Severity.INFO

    def test_barrier_participates_in_pipeline_ordering(self):
        body = (
            set_flag("MTE2_V")
            + "AscendC::PipeBarrier<PIPE_ALL>();\n"
            + wait_flag("MTE2_V")
        )
        graph = sync_graph(analyze_body(body))
        kinds = {node["kind"] for node in graph["nodes"]}
        assert "PipeBarrier" in kinds


def test_kernel_without_synchronisation_publishes_an_empty_graph():
    result = analyze_body("return;")
    graph = sync_graph(result)
    assert graph["nodes"] == []


# ---------------------------------------------------------------------------
# AIC/AIV core-split isolation
# ---------------------------------------------------------------------------


class TestCoreSplit:
    """A mix kernel is one source file but two binaries, one per core.

    The AIC and AIV cores hold separate event-id spaces, so flag operations
    guarded into different cores can neither pair with each other nor be
    counted against each other.
    """

    def test_ifdef_guards_tag_each_arm_with_its_core(self):
        result = analyze_body(
            "#ifdef __DAV_C220_CUBE__\n"
            + set_flag("M_MTE1", 0)
            + "#endif\n"
            "#ifdef __DAV_C220_VEC__\n"
            + set_flag("MTE3_V", 0)
            + "#endif\n"
        )
        views = {
            op.loc.line: op.core_view.value
            for op in result.unit.kernels[0].ops
        }
        assert sorted(views.values()) == ["aic", "aiv"]

    def test_runtime_predicate_guards_both_arms(self):
        # `if ASCEND_IS_AIC {` carries no parentheses of its own - the CANN
        # macro supplies them - so this also pins that the predicate parses.
        result = analyze_body(
            "if ASCEND_IS_AIC {\n"
            + set_flag("FIX_M", 1)
            + "} else {\n"
            + set_flag("V_MTE2", 1)
            + "}\n"
        )
        assert not find(result, "AKA9001")
        views = sorted(op.core_view.value for op in result.unit.kernels[0].ops)
        assert views == ["aic", "aiv"]

    def test_an_unrelated_ifdef_leaves_both_arms_on_both_cores(self):
        # Regression: the complement of "both cores" is "neither", so applying
        # it to a non-core guard would drop the else arm from every analysis.
        result = analyze_body(
            "#ifdef SOME_OTHER_FEATURE\n"
            + set_flag("MTE2_V", 0)
            + "#else\n"
            + set_flag("MTE2_V", 1)
            + "#endif\n"
        )
        views = sorted(op.core_view.value for op in result.unit.kernels[0].ops)
        assert views == ["both", "both"]

    def test_a_cross_core_handoff_is_not_a_local_orphan(self):
        result = analyze_body(
            "if ASCEND_IS_AIC {\n"
            + set_flag("MTE3_MTE2", 2)
            + "}\n"
            "if ASCEND_IS_AIV {\n"
            + wait_flag("MTE3_MTE2", 2)
            + "}\n"
        )
        assert not find(result, "AKA2001")
        assert not find(result, "AKA2002")

    def test_loop_balance_is_counted_per_core(self):
        # One set on the Cube core and one wait on the Vector core: merged
        # counting sees 1 set + 1 wait and calls the body balanced, hiding that
        # each core's body is off by one.
        result = analyze_body(
            "for (int i = 0; i < 4; ++i) {\n"
            "if ASCEND_IS_AIC {\n"
            + set_flag("V_MTE2", 0)
            + "}\n"
            "if ASCEND_IS_AIV {\n"
            + wait_flag("V_MTE2", 0)
            + "}\n"
            "}\n"
        )
        assert find(result, "AKA2006")

    def test_a_per_core_balanced_loop_is_silent(self):
        result = analyze_body(
            "for (int i = 0; i < 4; ++i) {\n"
            "if ASCEND_IS_AIC {\n"
            + set_flag("M_MTE1", 0)
            + wait_flag("M_MTE1", 0)
            + "}\n"
            "if ASCEND_IS_AIV {\n"
            + set_flag("V_MTE2", 1)
            + wait_flag("V_MTE2", 1)
            + "}\n"
            "}\n"
        )
        assert not find(result, "AKA2006")
        assert not find(result, "AKA2001")
        assert not find(result, "AKA2002")
