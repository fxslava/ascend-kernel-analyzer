"""Synthetic KernelIR adapter coverage, without source extraction."""
from dataclasses import replace
import pytest
import networkx as nx
from ascend_analyzer.backend.trace import dependency_graph, schedule_trace
from ascend_analyzer.backend.graph import overlap_metrics
from ascend_analyzer.checkers.base import CheckerContext
from ascend_analyzer.checkers.deadlock import DeadlockChecker
from ascend_analyzer.checkers.hazard import HazardChecker
from ascend_analyzer.checkers.perf_model import PerfModelChecker
from ascend_analyzer.diagnostics import DiagnosticCollector, SourceLoc
from ascend_analyzer.hardware import HardwareModel, HardEventRoute, Pipe, PhysicalDomain
from ascend_analyzer.ir import AnalysisUnit, ApiCallOp, ArgRef, BarrierOp, CoreView, KernelIR, TensorDecl
from ascend_analyzer.symbolic import Const

LOC = SourceLoc("<synthetic>", 1, 1)


def operation(index, pipe, **kwargs):
    return ApiCallOp(index=index, loc=LOC, pipe=pipe, scope_id=0, **kwargs)


def context(kernel, hardware=None):
    return CheckerContext(AnalysisUnit(path="<synthetic>", source="", kernels=[kernel]),
                          hardware or HardwareModel.for_chip(), DiagnosticCollector())


def test_pipe_all_fences_dma_and_compute():
    kernel = KernelIR("k", LOC, ops=[operation(0, Pipe.MTE2), operation(1, Pipe.V)])
    hw = HardwareModel.for_chip()
    start, end, *_ = schedule_trace(kernel, {}, {0: 10, 1: 10}, hw)
    assert overlap_metrics([(start[1], end[1])], [(start[0], end[0])])["dma_hidden_ratio"] == 1
    kernel.ops.insert(1, BarrierOp(index=9, loc=LOC, pipe=Pipe.ALL, scope_id=0, target=Pipe.ALL))
    start, end, *_ = schedule_trace(kernel, {}, {0: 10, 9: 1, 1: 10}, hw)
    assert start[1] == 11
    assert overlap_metrics([(start[1], end[1])], [(start[0], end[0])])["dma_hidden_ratio"] == 0


def test_distinct_core_streams_have_no_implicit_program_order():
    kernel = KernelIR("k", LOC, ops=[operation(0, Pipe.MTE2, core_view=CoreView.AIC),
                                     operation(1, Pipe.MTE2, core_view=CoreView.AIV)])
    assert not nx.has_path(dependency_graph(kernel, {}), 0, 1)


@pytest.mark.parametrize("delay", [0, 12])
def test_explicit_handoff_delay_changes_only_real_dependency(delay):
    hw = HardwareModel.for_chip()
    hw = HardwareModel(replace(hw.chip, sync_handoff_cycles=delay))
    kernel = KernelIR("k", LOC, ops=[operation(8, Pipe.MTE2), operation(2, Pipe.V)])
    sync = {"edges": [{"from": 8, "to": 2, "tokens": 0, "kind": "sync", "route": "MTE2_V"}]}
    start, _, *_ = schedule_trace(kernel, sync, {8: 5, 2: 5}, hw)
    assert start[2] == 5+delay


def test_operation_count_precedes_backing_allocation():
    tensor = TensorDecl("u", LOC, None, PhysicalDomain.UB, elem_size=2, byte_size=Const(4096))
    op = operation(0, Pipe.MTE2, name="DataCopy", args=(ArgRef(0, "u", tensor="u"),
                   ArgRef(1, "gm", tensor="gm"), ArgRef(2, "32", value=32)), writes=("u",))
    kernel = KernelIR("k", LOC, ops=[op], tensors={"u": tensor})
    assert PerfModelChecker(context(kernel))._transfer_bytes(kernel, op) == 64


def test_unrelated_flag_generations_cannot_prove_forward_raw():
    from ascend_analyzer.ir import FlagOp, FlagKind
    route = HardEventRoute.parse("MTE2_V")
    def flag(index, kind):
        return FlagOp(index=index, loc=LOC, scope_id=0, pipe=route.src if kind is FlagKind.SET else route.dst,
                      route=route, event_id=0, flag_kind=kind)
    kernel = KernelIR("k", LOC, ops=[flag(0, FlagKind.SET), flag(1, FlagKind.WAIT),
                                     operation(2, Pipe.MTE2, name="load", writes=("u",)),
                                     operation(3, Pipe.V, name="compute", reads=("u",)),
                                     flag(4, FlagKind.SET), flag(5, FlagKind.WAIT)])
    ctx = context(kernel)
    ctx.artifacts["sync_graph::k"] = {"edges": [
        {"from": 0, "to": 1, "kind": "sync", "tokens": 0},
        {"from": 4, "to": 5, "kind": "sync", "tokens": 0}]}
    HazardChecker(ctx).check(kernel)
    assert ctx.artifacts["hazard_coverage::k"]["hazards"] == 1


@pytest.mark.parametrize("modes,kind", [(("read", "write"), "WAR"), (("write", "write"), "WAW")])
def test_interval_hazard_adapter_emits_war_and_waw(modes, kind):
    tensor = TensorDecl("u", LOC, None, PhysicalDomain.UB, byte_offset=Const(0), byte_size=Const(32))
    ops = [operation(i, pipe, name="op", **{"reads" if mode == "read" else "writes": ("u",)})
           for i, (pipe, mode) in enumerate(zip((Pipe.V, Pipe.MTE2), modes))]
    kernel = KernelIR("k", LOC, ops=ops, tensors={"u": tensor})
    ctx = context(kernel)
    HazardChecker(ctx).check(kernel)
    assert ctx.artifacts["interval_hazard_coverage::k"]["hazards"][0]["kind"] == kind


def test_explicit_route_allowlist_is_enforced():
    from ascend_analyzer.ir import FlagOp
    hw = HardwareModel.for_chip()
    hw = HardwareModel(replace(hw.chip, supported_routes=frozenset({"M_FIX"})))
    flag = FlagOp(index=0, loc=LOC, pipe=Pipe.MTE2, scope_id=0,
                  route=HardEventRoute.parse("MTE2_V"), event_id=0)
    kernel = KernelIR("k", LOC, ops=[flag])
    ctx = context(kernel, hw)
    DeadlockChecker(ctx).check(kernel)
    assert "AKA2011" in {d.code.value for d in ctx.diagnostics}


def test_conditional_barrier_is_not_unconditional_happens_before():
    kernel = KernelIR("k", LOC, ops=[operation(0, Pipe.MTE2),
        BarrierOp(index=1, loc=LOC, pipe=Pipe.ALL, scope_id=0, target=Pipe.ALL, conditional=True),
        operation(2, Pipe.V)])
    assert not nx.has_path(dependency_graph(kernel, {}), 0, 2)
