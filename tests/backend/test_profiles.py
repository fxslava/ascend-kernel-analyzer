import json
from pathlib import Path
import subprocess
import sys
import pytest
from ascend_analyzer.hardware import HardwareModel, HardEventRoute
from ascend_analyzer.backend.graph import schedule, expand_iterations
import networkx as nx


def test_graph_runner_reports_finite_flow_and_actual_overlap():
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "ascend_analyzer.backend", str(root/"examples/backend/pipeline.json"),
               "--profile", str(root/"examples/backend/synthetic-profile.json"), "--iterations", "4"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)
    assert payload["flow"]["makespan"] == 50
    assert payload["flow"]["dma_hidden_ratio"] == .9  # compute ends at 45; drain ends at 50
    assert payload["flow"]["peak_occupancy"] == {"ingress": 10, "egress": 10}


def test_formal_profile_overrides_are_independent(tmp_path):
    profile = tmp_path/"profile.json"
    profile.write_text(json.dumps({"base": "ascend910b", "flow_rates": {"AIC": 3.5},
                                  "flow_buffers": {"L0C-out": 7.5}, "queue_depths": {"AIC": 2},
                                  "supported_routes": ["M_FIX"], "ub_bank_count": 4, "block_bytes": 64,
                                  "cube_macs_per_cycle": 2048, "core_topology": "unified"}))
    hw = HardwareModel.from_profile_file(profile)
    assert hw.flow_rate("AIC") == 3.5 and hw.flow_capacity("L0C-out") == 7.5
    assert hw.queue_depth("AIC") == 2 and hw.queue_depth("MTE2") is None
    assert hw.supports_route(HardEventRoute.parse("M_FIX")) is True
    assert hw.supports_route(HardEventRoute.parse("FIX_MTE3")) is False
    assert hw.ub_bank(256) == 0 and hw.ub_bank(64) == 1
    assert hw.chip.cube_contraction_cycles(16, 16, 16) == 2
    assert HardwareModel.for_chip().chip.block_bytes == 32


def test_310p_is_distinct_and_provisional():
    hw = HardwareModel.for_chip("310p")
    assert hw.chip.provisional and hw.chip.core_topology == "unified"
    assert hw.queue_depth("MTE2") is None
    assert hw.supports_route(HardEventRoute.parse("MTE2_V")) is None


def test_parallel_place_delays_take_maximum():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, delay=10, tokens=0)
    graph.add_edge(0, 1, delay=2, tokens=0)
    assert schedule(graph, {0: 1, 1: 1})[1][1] == 12
    expanded = expand_iterations(graph, 1)
    assert schedule(expanded, {(0, 0): 1, (1, 0): 1})[1][(1, 0)] == 12


def test_profile_controls_transfer_paths_and_logical_positions(tmp_path):
    from ascend_analyzer.hardware import PhysicalDomain, TPosition, Pipe
    from ascend_analyzer.apis import data_copy_pipe
    profile = tmp_path/"routes.json"
    profile.write_text(json.dumps({"pipes": ["S", "V"], "transfer_pipes": {"UB,UB": "V"},
                                  "position_domains": {"CO2": "GM"}}))
    hw = HardwareModel.from_profile_file(profile)
    assert hw.active_pipes() == (Pipe.S, Pipe.V)
    assert data_copy_pipe(PhysicalDomain.UB, PhysicalDomain.UB, hw) is Pipe.V
    assert hw.transfer_pipe(PhysicalDomain.UB, PhysicalDomain.GM) is None
    assert hw.domain_of(TPosition.CO2) is PhysicalDomain.GM
