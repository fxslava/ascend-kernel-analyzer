"""Marked graph algorithms. Parallel places retain independent markings."""
from itertools import islice
from math import isfinite
import networkx as nx


def token_free_subgraph(graph):
    blocked = nx.DiGraph()
    blocked.add_nodes_from(graph.nodes(data=True))
    for source, target, data in graph.edges(data=True):
        tokens = data.get("tokens", 0)
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("markings must be nonnegative integers")
        if tokens == 0:
            blocked.add_edge(source, target, **data)
    return blocked


def token_free_cycles(graph, limit=256):
    if limit <= 0:
        raise ValueError("cycle limit must be positive")
    blocked = token_free_subgraph(graph)
    if nx.is_directed_acyclic_graph(blocked):
        return []
    return sorted((list(c) for c in islice(nx.simple_cycles(blocked), limit)),
                  key=lambda c: (len(c), repr(c)))


def expand_iterations(graph, count):
    """Edge marking k imposes u(i-k) -> v(i), including loop recurrences.

    Missing negative iterations are initial tokens. This is a finite horizon,
    not a claim that a peeled trace represents the complete kernel.
    """
    if count < 1:
        raise ValueError("iteration count must be positive")
    token_free_subgraph(graph)  # validate all markings
    expanded = nx.DiGraph()
    for i in range(count):
        for node, data in graph.nodes(data=True):
            expanded.add_node((node, i), **data)
        for u, v, data in graph.edges(data=True):
            k = data.get("tokens", 0)
            if i >= k:
                source, target = (u, i-k), (v, i)
                delay = data.get("delay", 0)
                if expanded.has_edge(source, target):
                    delay = max(delay, expanded[source][target].get("delay", 0))
                expanded.add_edge(source, target, **{**data, "delay": delay})
    return expanded


def schedule(graph, durations):
    """ASAP dependency schedule; never assumes numeric node/source order."""
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("cannot schedule a cyclic dependency graph")
    start, end = {}, {}
    for node in nx.topological_sort(graph):
        duration = durations[node]
        if not isfinite(duration) or duration < 0:
            raise ValueError("negative service duration")
        arrivals = []
        for p in graph.predecessors(node):
            places = graph[p][node].values() if graph.is_multigraph() else [graph[p][node]]
            for place in places:
                delay = place.get("delay", 0)
                if not isfinite(delay) or delay < 0:
                    raise ValueError("invalid dependency delay")
                arrivals.append(end[p]+delay)
        start[node] = max(arrivals, default=0)
        end[node] = start[node] + duration
    return start, end


def interval_union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def overlap_metrics(compute, dma):
    """Lengths of unions/intersection, independent of pipeline count."""
    compute, dma = interval_union(compute), interval_union(dma)
    intersection = sum(max(0, min(b, d)-max(a, c))
                       for a, b in compute for c, d in dma)
    compute_time = sum(b-a for a, b in compute)
    dma_time = sum(b-a for a, b in dma)
    return {"compute_cycles": compute_time, "dma_cycles": dma_time,
            "compute_dma_overlap_cycles": intersection,
            "dma_hidden_ratio": intersection/dma_time if dma_time else 0.0,
            "compute_overlap_ratio": intersection/compute_time if compute_time else 0.0}
