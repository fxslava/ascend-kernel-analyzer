"""Interval hazards on explicit physical resources, including shared GM.

Unknown bounds are may-alias, never proven disjoint. Local resources must
include their core instance in the resource key; GM keys are shared.
"""
from dataclasses import dataclass
import networkx as nx


@dataclass(frozen=True)
class Access:
    node: object
    resource: str
    mode: str
    lower: int | None
    upper: int | None

    def __post_init__(self):
        if self.mode not in {"read", "write"}:
            raise ValueError("invalid access mode")
        if self.lower is not None and self.upper is not None and self.upper < self.lower:
            raise ValueError("reversed interval")


def unordered_hazards(graph, accesses):
    """Return RAW/WAR/WAW candidates lacking happens-before in either direction.

    The access list defines intended source order, never execution order.
    Graph edges must express real synchronization, not estimated timestamps.
    """
    hazards = []
    for i, a in enumerate(accesses):
        for b in accesses[i+1:]:
            if a.node == b.node or a.resource != b.resource or a.mode == b.mode == "read":
                continue
            if a.lower == a.upper and a.lower is not None or b.lower == b.upper and b.lower is not None:
                continue
            if all(x is not None for x in (a.lower, a.upper, b.lower, b.upper)) and (a.upper <= b.lower or b.upper <= a.lower):
                continue
            if a.node not in graph or b.node not in graph:
                raise ValueError("access node missing from dependency graph")
            if nx.has_path(graph, a.node, b.node) or nx.has_path(graph, b.node, a.node):
                continue
            kind = "WAW" if a.mode == b.mode else "RAW" if a.mode == "write" else "WAR"
            hazards.append((kind, a, b))
    return hazards
