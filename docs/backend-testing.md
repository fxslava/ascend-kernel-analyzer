# Backend graph regression harness

Run the isolated graph tier:

```powershell
python -m pytest tests/backend
```

Run the complete compatibility/integration suite:

```powershell
python -m pytest
```

Validation on 2026-10-05: **52 isolated backend cases passed; 844 total tests passed**.
Pyflakes on changed backend/processor/checker modules and `git diff --check` also passed.

The graph tier constructs NetworkX graphs, `Stage`/`Buffer` networks, interval
`Access` records and processor overrides. It does not parse C++ or invoke a
compiler. Fresh-process tests verify frontend import isolation and exercise
the JSON graph runner independently.

| Area | Regression evidence |
|---|---|
| Zero-token cycles | Empty ring, primed ring, multiple initial tokens, parallel empty and marked places |
| Finite recurrence | One/two-token ping-pong reuse and iteration expansion; cyclic graphs refuse timestamps |
| Scheduling | Reverse node numbering; maximum parallel delay; serial DMA/compute overlap zero |
| Overlap | Union/intersection avoids double counting AIC/AIV or concurrent DMA intervals |
| Backpressure | Slow egress throttles compute and ingress; peak occupancy remains bounded |
| Starvation | Faster compute follows ingress; missing producer material leaves consumer blocked |
| FIFO/HOL | Delayed head prevents bypass; completion dependency on a later head is blocked |
| Causality | Unprimed fluid ring cannot invent circulating bytes; primed ring progresses |
| Numeric behavior | Fractional rates, mass conservation, invalid capacities, zero-work FIFO semantics |
| Safety | RAW/WAR/WAW on shared GM, transitive synchronization, disjoint intervals, local-core separation, tainted bounds |
| Processor inputs | Explicit calibrated rates/buffers, unknown depth/route status, allowlists, bank granule, MAC throughput |
| Serialization | JSON runner exposes segments, queues, completions and finite-horizon schedules |

Existing source performance expectations were updated because the old fixture
latency included invented 30-cycle handoffs and arena-sized transfers. The
corrected short trace no longer crosses the advisory gate. This is a semantic
correction with separately tested graph timing, not a disabled checker.

`analyze_graph` accepts explicit node durations and optional normalized flow
stages. The completion graph and streaming graph are independently reported;
tests do not silently treat them as equivalent. Graph callers must include
engine recurrence edges and real happens-before constraints. Source fixture
coverage remains valuable for validating that the frontend constructs them.
