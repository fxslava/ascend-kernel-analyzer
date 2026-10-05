"""Event-driven, piecewise-linear finite-volume flow network.

Volumes use a common byte-equivalent unit. A stage consumes and produces one
unit per unit of progress; callers must normalize compute work/tensor expansion.
No instruction slots or time ticks are used. Task dependencies are completion
fences; queue edges allow streaming only where the caller explicitly permits it.
"""
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol
import networkx as nx


class FlowHardware(Protocol):
    def flow_rate(self, engine: str) -> float: ...
    def flow_capacity(self, buffer: str) -> float: ...


@dataclass(frozen=True)
class Stage:
    name: str
    engine: str
    volume: float
    release: float = 0.0
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Buffer:
    name: str
    producer: str
    consumer: str
    storage: str
    initial: float = 0.0


@dataclass
class FlowResult:
    makespan: float
    completed: dict[str, float]
    occupancy: dict[str, float]
    peak_occupancy: dict[str, float]
    segments: list[dict]
    blocked: dict[str, str] = field(default_factory=dict)

    @property
    def deadlocked(self):
        return bool(self.blocked)


def solve_flow(stages: list[Stage], buffers: list[Buffer], hardware: FlowHardware,
               *, max_events=100000, tolerance=1e-9) -> FlowResult:
    """FIFO per engine with starvation and downstream pressure propagation.

    At empty queues r_consumer <= r_producer; at full queues
    r_producer <= r_consumer. Monotone rate reduction reaches the greatest
    feasible rates under these boundary conditions. The next event is a
    release, completion, empty boundary or full boundary.
    """
    if tolerance <= 0 or max_events < 1:
        raise ValueError("invalid solver limits")
    nodes = {s.name: s for s in stages}
    if len(nodes) != len(stages) or len({b.name for b in buffers}) != len(buffers):
        raise ValueError("duplicate stage or buffer")
    if len({b.storage for b in buffers}) != len(buffers):
        raise ValueError("flow buffers require distinct dedicated storage capacities")
    remaining, limits, capacities, occupancy = {}, {}, {}, {}
    for s in stages:
        rate = hardware.flow_rate(s.engine)
        if not all(isfinite(x) for x in (s.volume, s.release, rate)) or min(s.volume, s.release) < 0 or rate <= 0:
            raise ValueError("finite positive rates and nonnegative work/releases required")
        if any(d not in nodes for d in s.after):
            raise ValueError("unknown dependency")
        remaining[s.name], limits[s.name] = s.volume, rate
    for b in buffers:
        cap = hardware.flow_capacity(b.storage)
        if b.producer not in nodes or b.consumer not in nodes or b.producer == b.consumer:
            raise ValueError("invalid queue endpoints")
        if not isfinite(cap) or cap <= 0 or not isfinite(b.initial) or not 0 <= b.initial <= cap:
            raise ValueError("invalid queue capacity/initial occupancy")
        capacities[b.name], occupancy[b.name] = cap, b.initial
    peak = dict(occupancy)
    done, segments, now = {}, [], 0.0
    for _ in range(max_events):
        # Zero-work tasks still respect release and completion dependencies.
        changed = True
        while changed:
            changed = False
            zero_heads = {}
            for s in stages:
                if s.name not in done:
                    zero_heads.setdefault(s.engine, s.name)
            for s in stages:
                if s.name not in done and zero_heads.get(s.engine) == s.name and remaining[s.name] <= tolerance and s.release <= now and all(d in done for d in s.after):
                    done[s.name] = now
                    changed = True
        if len(done) == len(stages):
            return FlowResult(now, done, occupancy, peak, segments)
        heads = {}
        for s in stages:
            if s.name not in done:
                heads.setdefault(s.engine, s)
        rates = {s.name: 0.0 for s in stages}
        for s in heads.values():
            if s.release <= now and all(d in done for d in s.after):
                rates[s.name] = limits[s.name]
        # An empty streaming ring cannot invent circulating material. Positive
        # equal rates satisfy dq/dt=0 algebraically but violate causality.
        empty = nx.DiGraph()
        empty.add_edges_from((b.producer, b.consumer) for b in buffers
                             if occupancy[b.name] <= tolerance)
        for component in nx.strongly_connected_components(empty):
            if len(component) > 1:
                for n in component:
                    rates[n] = 0.0
        # All constraints are min-equalities; at most |V| propagation rounds
        # are needed, but stop at equality to handle arbitrary graph ordering.
        while True:
            previous = dict(rates)
            for b in buffers:
                p, c = b.producer, b.consumer
                if occupancy[b.name] <= tolerance:
                    rates[c] = min(rates[c], rates[p])
                if occupancy[b.name] >= capacities[b.name]-tolerance:
                    rates[p] = min(rates[p], rates[c])
            if rates == previous:
                break
        events = [s.release-now for s in stages if s.name not in done and s.release > now]
        events += [remaining[n]/r for n, r in rates.items() if r > 0 and remaining[n] > tolerance]
        for b in buffers:
            slope = rates[b.producer]-rates[b.consumer]
            if slope > 0:
                events.append((capacities[b.name]-occupancy[b.name])/slope)
            elif slope < 0:
                events.append(-occupancy[b.name]/slope)
        events = [dt for dt in events if dt > tolerance]
        if not events:
            reasons = {}
            for s in stages:
                if s.name in done:
                    continue
                reason = "head-of-line blocked"
                if heads[s.engine] is s:
                    reason = "completion dependency" if any(d not in done for d in s.after) else "starvation/backpressure"
                reasons[s.name] = reason
            return FlowResult(now, done, occupancy, peak, segments, reasons)
        dt = min(events)
        reasons = {}
        for s in heads.values():
            if s.name in done or s.release > now or any(d not in done for d in s.after):
                continue
            if rates[s.name] < limits[s.name]:
                reasons[s.name] = "starvation/backpressure"
        segments.append({"start": now, "end": now+dt, "rates": dict(rates),
                         "occupancy": dict(occupancy), "limited": reasons})
        for n, r in rates.items():
            remaining[n] = max(0.0, remaining[n]-r*dt)
        for b in buffers:
            occupancy[b.name] += (rates[b.producer]-rates[b.consumer])*dt
            if not -tolerance <= occupancy[b.name] <= capacities[b.name]+tolerance:
                raise ArithmeticError("queue conservation failure")
            occupancy[b.name] = min(capacities[b.name], max(0.0, occupancy[b.name]))
            peak[b.name] = max(peak[b.name], occupancy[b.name])
        now += dt
    raise RuntimeError("flow event limit exceeded")
