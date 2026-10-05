"""Run synthetic graphs and fluid networks without importing a frontend."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import networkx as nx
from .analyzer import analyze_graph
from .flow import Stage, Buffer
from .safety import Access
from ..hardware import HardwareModel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args(argv)
    raw = json.loads(args.graph.read_text(encoding="utf-8"))
    graph, durations = nx.MultiDiGraph(), {}
    for node in raw.get("nodes", []):
        if node["id"] in graph:
            raise ValueError("duplicate graph node")
        graph.add_node(node["id"])
        durations[node["id"]] = node["duration"]
    for edge in raw.get("edges", []):
        if edge["from"] not in graph or edge["to"] not in graph:
            raise ValueError("edge references absent node")
        graph.add_edge(edge["from"], edge["to"], tokens=edge.get("tokens", 0), delay=edge.get("delay", 0))
    flow = raw.get("flow")
    hardware = HardwareModel.from_profile_file(args.profile) if args.profile else None
    result = analyze_graph(graph, durations, iterations=args.iterations,
                           accesses=[Access(**a) for a in raw.get("accesses", [])],
                           compute_nodes=raw.get("compute_nodes", []), dma_nodes=raw.get("dma_nodes", []),
                           flow_stages=[Stage(**s) for s in flow.get("stages", [])] if flow is not None else None,
                           flow_buffers=[Buffer(**b) for b in flow.get("buffers", [])] if flow is not None else (),
                           hardware=hardware)
    for key in ("start", "end"):
        if key in result:
            result[key] = [{"node": n, "iteration": i, "cycles": t} for (n, i), t in result[key].items()]
    if "hazards" in result:
        result["hazards"] = [{"kind": kind, "first": asdict(a), "second": asdict(b)} for kind, a, b in result["hazards"]]
    print(json.dumps(result, indent=2))
    return 2 if result["deadlocked"] or result.get("flow", {}).get("deadlocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
