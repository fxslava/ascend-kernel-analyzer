# Hydrodynamic queue and overlap specification

## Two explicit models

The completion-dependency graph models instruction order, matched flags,
barriers and loop recurrences. The continuous-flow network models data that
can stream through finite buffers. A buffer edge does **not** imply a completed
tile is safe for computation. Use a completion dependency (`Stage.after`) for
whole-tile availability. Callers choose streaming edges only for workloads that
actually support it. Combining a completion fence with insufficient buffer
capacity can deadlock; that is a real model result, not permission to ignore
the fence.

## Continuous finite-volume equations

For each queue edge e=(producer, consumer):

```
dq_e/dt = r_producer - r_consumer
0 <= q_e <= C_e
0 <= r_stage <= calibrated_engine_rate
```

`q` and `C` are byte-equivalent volume, not slot counts. Processing stages must
be normalized to the same units as incoming/outgoing edges; a matmul rate is
not directly a GM byte rate. `HardwareModel.flow_rate` and `flow_capacity`
require explicit profile entries and reject missing calibration.
Each buffer names a dedicated storage window. Reusing one window for two
independent queues is rejected; partition it explicitly or express reuse in
the completion graph rather than claiming its capacity twice.

At an empty queue the consumer cannot outrun its producer. At a full queue
the producer cannot outrun its consumer. Rates are reduced monotonically
until these inequalities hold across the network. This propagates pressure
through MTE3 -> Vector/Cube -> MTE2, including multiple queues and Fixpipe
stages when supplied. Initially empty cyclic streaming dependencies are
blocked: algebraically equal rates cannot create material out of nothing.

Each engine serves the first unfinished stage in input order, so a delayed
or dependency-blocked head prevents later work from bypassing it. Engines with
distinct keys are independent; callers must identify physical instances
explicitly. Release times and completion dependencies gate stage eligibility.
Zero-work stages respect the same FIFO and dependency constraints.

After rates stabilize, the solver advances directly to the next release,
work completion, queue-empty boundary, or queue-full boundary. It integrates
occupancy and remaining volume over that segment. There are no fixed ticks
and no discrete instruction-slot queue. Numeric tolerance and an event limit
bound floating-point drift and pathological inputs; limit exhaustion raises
an error instead of returning a partial schedule as successful.

Outputs include completion times, segment rates, pressure-limited segments,
final and peak occupancy, and blocked stages. Conservation and boundary checks
guard each integration step. Initial buffer inventory is retained in the mass
balance. A producer finishing before the consumer has enough material yields
starvation, not fabricated completion.

## Marked recurrence semantics

Every parallel place retains its marking. A zero-token cycle is a deadlock
witness for this marked-graph abstraction; conditional branches and external
runtime behavior are outside that theorem's scope.

For a place u -> v with k initial tokens, finite iteration expansion adds:

```
u(i-k) -> v(i), whenever i >= k
```

Negative iteration predecessors are satisfied by the initial marking. This
supports primed ping-pong reuse without simply ignoring all loop edges in
every iteration. Per-engine iteration recurrence must also be present in
synthetic graphs; buffer tokens alone do not describe engine throughput.
Scheduling uses topological order, with maximum predecessor completion plus
edge delay, regardless of node numbering. Parallel delays use the maximum
constraint. Cyclic dependencies are not scheduled.

## Overlap definition and anti-patterns

Let C be the union of actual modeled compute busy intervals and D the union of
DMA busy intervals. Compute-only and DMA-only intervals exclude flag issue.

```
overlap_cycles       = length(C intersect D)
dma_hidden_ratio     = length(C intersect D) / length(D)
compute_overlap_ratio = length(C intersect D) / length(C)
```

Empty denominators yield zero. Unioning intervals prevents AIC/AIV overlap or
simultaneous DMA engines from counting a cycle twice. `overlap_ratio` in the
source performance artifact now aliases DMA hiding; the old average pipe
utilization is `concurrency_efficiency`.
The JSON report schema is now 2.0 to identify this metric semantics change;
hazard and bank coverage artifacts are also included in each kernel report.

An immediate wait is harmful when it holds a consumer's ordered stream while
independent work could execute. The analyzer reports observed modeled waits,
not a synthetic ideal-pipeline schedule. `PIPE_ALL` inserts full completion
fences; the existing barrier diagnostic names the serialization. Finite reuse
tokens/iteration-specific accesses expose single-buffer reuse and ping-pong
collisions. The backend does not recognize arbitrary runtime ping-pong indexing
that the frontend cannot resolve.

The default extra flag handoff delay is zero. Profiles can specify a measured
positive delay. Flag/scalar issue remains a nominal one-cycle trace estimate;
bulk throughput values are disclosed estimates, not measured silicon latency.
Known API transfer counts take precedence over backing allocation size.

## Runnable example

```powershell
python -m ascend_analyzer.backend examples/backend/pipeline.json --profile examples/backend/synthetic-profile.json --iterations 4
```

The two outputs deliberately have different semantics: the marked graph models
finite completion/reuse constraints; the fluid fixture explicitly permits
streaming. Its egress rate 2, compute rate 5 and ingress rate 10 produce a
50-cycle drain, 10-unit peak queues, and 90% DMA hiding. These are synthetic
regression values, not Ascend bandwidth measurements.
