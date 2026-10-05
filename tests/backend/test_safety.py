import networkx as nx
import pytest
from ascend_analyzer.backend.safety import Access, unordered_hazards


@pytest.mark.parametrize("modes,kind", [(("write", "read"), "RAW"), (("read", "write"), "WAR"), (("write", "write"), "WAW")])
def test_shared_gm_aic_aiv_hazards(modes, kind):
    graph = nx.DiGraph()
    graph.add_nodes_from(["aic", "aiv"])
    accesses = [Access("aic", "shared:GM", modes[0], 0, 64), Access("aiv", "shared:GM", modes[1], 32, 96)]
    assert unordered_hazards(graph, accesses)[0][0] == kind
    graph.add_edge("aic", "aiv")
    assert not unordered_hazards(graph, accesses)


def test_transitive_barrier_sufficiency():
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    assert not unordered_hazards(graph, [Access(0, "UB", "write", 0, 32), Access(3, "UB", "read", 0, 32)])


@pytest.mark.parametrize("resource,lower,upper", [("UB:aiv", 0, 32), ("UB:aic", 32, 64), ("UB:aic", 0, 0)])
def test_core_local_or_disjoint_memory_does_not_alias(resource, lower, upper):
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    assert not unordered_hazards(graph, [Access(0, "UB:aic", "write", 0, 32), Access(1, resource, "read", lower, upper)])


def test_tainted_unknown_address_may_alias():
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    assert unordered_hazards(graph, [Access(0, "GM", "write", None, None), Access(1, "GM", "read", 1024, 2048)])
