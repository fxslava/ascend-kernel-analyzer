from dataclasses import replace
import pytest
from ascend_analyzer.backend.flow import Stage, Buffer, solve_flow
from ascend_analyzer.hardware import HardwareModel, resolve_chip


def hardware(rates=None, buffers=None):
    return HardwareModel(replace(resolve_chip("910b"),
                         flow_rates=rates or {"MTE2": 10, "V": 5, "MTE3": 2},
                         flow_buffers=buffers or {"in": 10, "out": 10}))


def test_full_egress_propagates_pressure_to_ingress():
    result = solve_flow([Stage("load", "MTE2", 100), Stage("compute", "V", 100),
                         Stage("store", "MTE3", 100)],
                        [Buffer("a", "load", "compute", "in"),
                         Buffer("b", "compute", "store", "out")], hardware())
    assert not result.deadlocked
    assert result.makespan == pytest.approx(50)
    assert result.peak_occupancy == {"a": 10, "b": 10}
    assert result.occupancy == {"a": 0, "b": 0}
    assert any(s["rates"]["load"] == s["rates"]["compute"] == s["rates"]["store"] == 2
               for s in result.segments)


def test_empty_queue_throttles_faster_compute():
    result = solve_flow([Stage("load", "MTE2", 20), Stage("compute", "V", 20)],
                        [Buffer("a", "load", "compute", "in")],
                        hardware({"MTE2": 2, "V": 10}))
    assert result.makespan == 10 and result.peak_occupancy["a"] == 0


def test_delayed_consumer_causes_a_finite_full_queue_stall():
    result = solve_flow([Stage("load", "MTE2", 30), Stage("compute", "V", 30, release=5)],
                        [Buffer("a", "load", "compute", "in")], hardware())
    assert result.makespan == pytest.approx(11)
    assert any(s["rates"]["load"] == 0 and s["start"] == 1 for s in result.segments)


def test_same_engine_head_of_line_cannot_be_bypassed():
    result = solve_flow([Stage("first", "MTE2", 10, release=5), Stage("second", "MTE2", 10)], [], hardware())
    assert result.completed == {"first": 6, "second": 7}


def test_completion_fence_serializes_but_streaming_does_not():
    stages = [Stage("load", "MTE2", 100), Stage("compute", "V", 100, after=("load",))]
    result = solve_flow(stages, [], hardware())
    assert result.makespan == 30


def test_consumer_starvation_is_reported_without_fake_completion():
    result = solve_flow([Stage("load", "MTE2", 10), Stage("compute", "V", 20)],
                        [Buffer("a", "load", "compute", "in")], hardware())
    assert result.deadlocked and "compute" in result.blocked
    assert "compute" not in result.completed


@pytest.mark.parametrize("initial,deadlock", [(0, True), (1, False)])
def test_streaming_ring_needs_initial_material(initial, deadlock):
    result = solve_flow([Stage("a", "MTE2", 10), Stage("b", "V", 10)],
                        [Buffer("ab", "a", "b", "in", initial), Buffer("ba", "b", "a", "out")], hardware())
    assert result.deadlocked is deadlock


def test_completion_cycle_and_fifo_deadlock():
    result = solve_flow([Stage("a", "MTE2", 10, after=("b",)), Stage("b", "MTE2", 10)], [], hardware())
    assert result.deadlocked
    assert result.blocked["a"] == "completion dependency"
    assert result.blocked["b"] == "head-of-line blocked"


def test_unknown_profile_data_is_rejected():
    with pytest.raises(ValueError, match="uncalibrated"):
        solve_flow([Stage("a", "V", 10)], [], HardwareModel.for_chip())


@pytest.mark.parametrize("capacity", [0, -1, float("nan")])
def test_bad_capacity_rejected(capacity):
    with pytest.raises(ValueError):
        hardware(buffers={"in": capacity})


def test_zero_work_obeys_fifo():
    result = solve_flow([Stage("a", "V", 10), Stage("b", "V", 0)], [], hardware())
    assert result.completed["b"] == 2


def test_two_queues_cannot_claim_one_dedicated_storage_window():
    with pytest.raises(ValueError, match="dedicated storage"):
        solve_flow([Stage("a", "MTE2", 10), Stage("b", "V", 10), Stage("c", "MTE3", 10)],
                   [Buffer("ab", "a", "b", "in"), Buffer("bc", "b", "c", "in")], hardware())


def test_fractional_rates_and_conservation():
    stages = [Stage("a", "MTE2", 3), Stage("b", "V", 3)]
    result = solve_flow(stages, [Buffer("ab", "a", "b", "in")],
                        hardware({"MTE2": 1.5, "V": .75}, {"in": .4}))
    assert result.makespan == pytest.approx(4)
    for stage in stages:
        moved = sum((s["end"]-s["start"])*s["rates"][stage.name] for s in result.segments)
        assert moved == pytest.approx(stage.volume)
