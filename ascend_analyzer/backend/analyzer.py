"""Pure graph entry point; no parser, compiler, source locations or emitters."""
from dataclasses import asdict
from .graph import token_free_cycles, expand_iterations, schedule, overlap_metrics
from .safety import unordered_hazards
from .flow import solve_flow


def analyze_graph(graph, durations, *, iterations=1, accesses=(),
                  compute_nodes=(), dma_nodes=(), flow_stages=None,
                  flow_buffers=(), hardware=None):
    cycles = token_free_cycles(graph)
    result = {"token_free_cycles": cycles, "deadlocked": bool(cycles)}
    if not cycles:
        expanded = expand_iterations(graph, iterations)
        start, end = schedule(expanded, {(n, i): durations[n] for n in graph for i in range(iterations)})
        result["start"], result["end"] = start, end
        result["makespan_cycles"] = max(end.values(), default=0)
        result.update(overlap_metrics(
            [(start[(n, i)], end[(n, i)]) for n in compute_nodes for i in range(iterations)],
            [(start[(n, i)], end[(n, i)]) for n in dma_nodes for i in range(iterations)]))
        # Marked loop edges do not establish same-iteration ordering.
        from .graph import token_free_subgraph
        result["hazards"] = unordered_hazards(token_free_subgraph(graph), accesses)
    if flow_stages is not None:
        if hardware is None:
            raise ValueError("flow analysis requires hardware calibration")
        flow = solve_flow(flow_stages, list(flow_buffers), hardware)
        result["flow"] = asdict(flow)
        result["flow"]["deadlocked"] = flow.deadlocked
        roles = {s.name: hardware.flow_role(s.engine) for s in flow_stages}
        result["flow"].update(overlap_metrics(
            [(seg["start"], seg["end"]) for seg in flow.segments
             if any(rate > 0 and roles[n] == "compute" for n, rate in seg["rates"].items())],
            [(seg["start"], seg["end"]) for seg in flow.segments
             if any(rate > 0 and roles[n] == "dma" for n, rate in seg["rates"].items())]))
    return result
