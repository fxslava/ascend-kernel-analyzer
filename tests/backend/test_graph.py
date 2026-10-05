"""Synthetic marked graphs; no AST fixtures or compiler invocation."""
import subprocess
import sys
import networkx as nx
import pytest
from ascend_analyzer.backend.graph import (
    token_free_cycles, token_free_subgraph, expand_iterations, schedule, overlap_metrics,
)
from ascend_analyzer.backend.analyzer import analyze_graph


def test_backend_import_does_not_load_frontend():
    code = '''
import sys
from ascend_analyzer.backend.analyzer import analyze_graph
from ascend_analyzer.hardware import HardwareModel
assert not any(n.startswith(("tree_sitter", "ascend_analyzer.parsing", "ascend_analyzer.ir.mlir", "ascend_analyzer.analyzer")) for n in sys.modules)
'''
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("tokens,deadlock", [(0, True), (1, False), (2, False)])
def test_ring_marking(tokens, deadlock):
    graph = nx.MultiDiGraph()
    graph.add_edge("wait", "set", tokens=0)
    graph.add_edge("set", "wait", tokens=tokens)
    assert bool(token_free_cycles(graph)) is deadlock


def test_parallel_token_place_cannot_hide_an_empty_place():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, tokens=0)
    graph.add_edge(1, 0, tokens=0)
    graph.add_edge(1, 0, tokens=1)
    assert token_free_cycles(graph)


def test_backward_numbered_dependency_is_scheduled():
    graph = nx.DiGraph([(100, 4), (4, 0)])
    start, end = schedule(graph, {100: 2, 4: 3, 0: 1})
    assert start == {100: 0, 4: 2, 0: 5}
    assert end[0] == 6


@pytest.mark.parametrize("tokens,expected", [(1, 40), (2, 20)])
def test_pingpong_reuse_tokens_control_overlap(tokens, expected):
    graph = nx.MultiDiGraph()
    graph.add_edge("load", "compute", tokens=0)
    graph.add_edge("compute", "load", tokens=tokens)
    expanded = expand_iterations(graph, 4)
    _, end = schedule(expanded, {n: 5 for n in expanded})
    assert max(end.values()) == expected


def test_unprimed_iteration_graph_refuses_schedule():
    graph = nx.DiGraph([(0, 1), (1, 0)])
    result = analyze_graph(graph, {0: 1, 1: 1}, iterations=3)
    assert result["deadlocked"] and "start" not in result


def test_overlap_unions_do_not_double_count_aic_and_aiv():
    metrics = overlap_metrics([(0, 10), (3, 8)], [(5, 15), (7, 12)])
    assert metrics["compute_dma_overlap_cycles"] == 5
    assert metrics["dma_hidden_ratio"] == .5


def test_serialized_pipeline_has_zero_overlap():
    graph = nx.DiGraph([(0, 1)])
    result = analyze_graph(graph, {0: 5, 1: 10}, compute_nodes=[1], dma_nodes=[0])
    assert result["dma_hidden_ratio"] == 0


@pytest.mark.parametrize("marking", [-1, 0.5])
def test_invalid_markings_rejected(marking):
    graph = nx.DiGraph()
    graph.add_edge(0, 1, tokens=marking)
    with pytest.raises(ValueError):
        token_free_subgraph(graph)
