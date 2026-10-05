"""KernelIR adapter: completion dependencies, pipe order and barrier fences."""
import networkx as nx
from ..hardware import Pipe, REAL_PIPES
from ..ir import BarrierOp, FlagOp, FlagKind, CoreView
from .graph import schedule


def dependency_graph(kernel, synchronization, hardware=None):
    graph = nx.DiGraph()
    graph.add_nodes_from(op.index for op in kernel.ops)
    previous = {}
    real_pipes = hardware.active_pipes() if hardware is not None else REAL_PIPES
    for op in kernel.ops:
        if op.core_view is CoreView.NONE or isinstance(op, BarrierOp) and op.conditional:
            continue
        views = (CoreView.AIC, CoreView.AIV) if op.core_view is CoreView.BOTH else (op.core_view,)
        pipes = (list(real_pipes) if op.target is Pipe.ALL else [op.target]) if isinstance(op, BarrierOp) else [op.pipe]
        for pipe in pipes:
            for view in views:
                key = (pipe, view)
                if key in previous:
                    graph.add_edge(previous[key], op.index, delay=0, kind="program")
                previous[key] = op.index
    for edge in synchronization.get("edges", []):
        if edge.get("kind") != "sync" or edge.get("tokens", 0) != 0:
            continue
        u, v = edge["from"], edge["to"]
        if u not in graph or v not in graph:
            raise ValueError("synchronization references absent operation")
        graph.add_edge(u, v, kind="sync", route=edge.get("route", "?"), delay=0)
    return graph


def schedule_trace(kernel, synchronization, durations, hardware):
    graph = dependency_graph(kernel, synchronization, hardware)
    for _, _, data in graph.edges(data=True):
        if data.get("kind") == "sync":
            data["delay"] = hardware.chip.sync_handoff_cycles
    if not nx.is_directed_acyclic_graph(graph):
        return None
    start, end = schedule(graph, durations)
    stalls, routes = {}, {}
    for op in kernel.ops:
        if not isinstance(op, FlagOp) or op.flag_kind is not FlagKind.WAIT:
            continue
        program_ready = max((end[p] for p in graph.predecessors(op.index)
                             if graph[p][op.index].get("kind") == "program"), default=0)
        gap = start[op.index]-program_ready
        if gap > 0:
            stalls[op.pipe.value] = stalls.get(op.pipe.value, 0)+gap
            route = op.route.name if op.route else "?"
            routes[route] = routes.get(route, 0)+gap
    return start, end, durations, stalls, routes
