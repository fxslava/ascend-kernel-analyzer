# Architectural audit — 2026-10-05

## Findings and changes

Baseline: 792 tests passed before modification. The implementation is a Python
package with tree-sitter/pcpp and optional BiSheng/MLIR extraction, a flat
`KernelIR`, checker passes, NetworkX synchronization graphs, and optional Z3
interval reasoning. There is no CMake backend to reorganize. Existing untracked
`NPUs/` and `artifacts/` inputs were inspected but not changed.

| Finding | Consequence | Implemented response |
|---|---|---|
| Package root imported the analyzer and parser eagerly | Hardware/graph consumers loaded the frontend | Lazy analyzer and MLIR exports; import isolation regression |
| Performance schedule walked numeric trace order, discarding backward sync edges | Legal non-source-topological dependencies could be ignored | Explicit dependency DAG and NetworkX topological scheduling |
| `DiGraph` merged parallel marked places | A marked edge could overwrite an empty edge and hide a circular wait | Synchronization builder uses `MultiDiGraph`; zero-token projection preserves every blocking place |
| Fixed 30-cycle handoff on every flag | Fabricated stalls dominated short kernels | Default zero extra delay; positive delay is explicit profile calibration |
| Average busy time divided by pipe count was called overlap | Scalar/flag activity altered reported DMA hiding | Interval union/intersection; mean utilization is separately named |
| DMA volume selected tensor allocation before explicit count | Larger arenas made small transfers look expensive | API count metadata takes precedence over allocation fallback |
| RAW evidence used any late set and any early wait | Different generations of an event could incorrectly establish ordering | Forward RAW proof uses paths through matched synchronization edges |
| Hazards only compared tensor names/write-read pairs | Aliased ranges, WAR/WAW and shared GM were missed | Pure interval access graph pass; straight-line adapter emits AKA2012 |
| Processor metadata mixed taxonomy, registry, and profile loading in one public module | Hard to identify ownership of hardware assumptions | Implementation moved to `processors/profiles.py`; public compatibility exports retained |
| No finite-volume pipeline model | Could not inspect buffer pressure or upstream throttling | Explicit continuous-flow graph solver and JSON runner |
| Almost all pipeline regressions started from C++ strings | Graph algorithm defects were coupled to frontend behavior | Separate `tests/backend/` tier and synthetic JSON examples |

## Existing domains and test coverage

The AST visitor is roughly 3,395 lines; MLIR bridge roughly 1,613; the deadlock
and memory passes roughly 950 and 870. Moving these wholesale would obscure
the semantic changes and break established imports. They remain frontend and
checker adapters, while reusable graph algorithms have moved out.

Existing `test_parser`, `test_frontend`, `test_cce_syntax`, `test_tiling_roles`,
and `test_mlir_frontend` primarily validate extraction. Existing `test_deadlock`,
`test_hazard`, `test_memory`, `test_perf`, `test_symbolic`, and `test_hardening`
mostly validate the complete source-to-diagnostic path. Hardware profile tests
exercise metadata directly; symbolic/solver tests have some isolated arithmetic
coverage. CLI, reports, baselines, fixtures and the C++ tiling smoke test cover
other boundaries. File names alone did not imply graph isolation.

The new tier exercises graph cycles, parallel markings, iteration expansion,
FIFO causality, continuous buffer boundaries, interval hazards, calibration,
and report serialization. Shared pytest helpers now import the frontend lazily.
A fresh-process regression proves importing graph/hardware APIs does not load
tree-sitter, source parsing, MLIR IR or the analyzer facade.

## Remaining limits

* Source traces do not expose complete TQue internals or operation-specific
  buffer lifetimes. Fluid queues are explicit graph inputs, not inferred from
  SRAM capacity. The source checker advertises that coverage limit.
* Peeled loops represent head/steady/tail regions. The source performance
  artifact is trace latency, not a claimed whole-kernel execution time.
* The interval adapter excludes loop-containing kernels from additional
  WAR/WAW findings because mutable tensor declarations do not retain each
  iteration's physical extent. The pure backend accepts iteration-specific
  accesses. Existing loop-aware RAW analysis remains in place.
* Symbolic addresses in the pure interval pass are may-alias. Existing Z3 and
  interval checks remain responsible for source address bounds and alignment.
* The source adapter recognizes named GM aliases but cannot establish the base
  pointer identity of unrelated GM symbols. Pure graphs must use shared GM
  resource keys explicitly.
* AIC/AIV guards need per-instance local resources. Distinct guarded pipe
  streams no longer share program order; unguarded BOTH operations still
  represent a merged trace rather than a fully expanded multicore execution.
* Hardware throughput and logical banks are modeling assumptions until
  calibrated. Exact silicon capacities/routes/depths are not certified here.
* Flow stages are normalized one-input-unit/one-output-unit transformations.
  Arbitrary tensor expansion and shared memory-port contention require caller
  normalization and explicit resource modeling, respectively.

These limits are exposed rather than converted into claims of verification.
See [hardware audit](hardware-profile-audit.md), [module rationale](module-refactoring.md),
[flow specification](hydrodynamic-overlap.md), and [regression harness](backend-testing.md).
