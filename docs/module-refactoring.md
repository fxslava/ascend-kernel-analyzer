# Module refactoring rationale

The existing setuptools Python package and public checker API determine the
boundaries. No replacement directory hierarchy or build system is imposed.

```text
ascend_analyzer/
  parsing/                  source/AST and BiSheng/MLIR extraction (retained)
  frontend/                 existing preprocessing support (retained)
  ir/                       source trace and lazy MLIR exports (retained)
  processors/profiles.py    processor types, registry, validation, query facade
  hardware.py               compatibility exports
  backend/
    graph.py                marked places, cycles, finite iteration expansion,
                            scheduling, interval overlap
    flow.py                 explicit finite-volume networks and pressure solver
    safety.py               physical interval access/happens-before checks
    trace.py                KernelIR-to-dependency-graph adapter
    analyzer.py             pure graph entry point, combined results
    __main__.py             JSON graph/flow runner
  checkers/                 safety, synchronization, performance diagnostics
  symbolic.py, solver.py    existing expression/taint and optional SMT boundary
  diagnostics.py            stable diagnostic catalog (retained)
  report/                   presentation (retained)
tests/
  backend/                  independent synthetic graphs and flow regressions
  test_*.py                 existing extraction/integration/report suites
  kernels/, data/           existing source and MLIR fixtures
examples/backend/           runnable synthetic flow profile and graph
```

`hardware.py` moved to `processors/profiles.py`; a compatibility module preserves
`from ascend_analyzer.hardware import ...`. No chip values were silently promoted
to validated facts during migration. New optional metadata includes flow rates,
byte-equivalent capacities, engine roles, queue depths, route allowlists,
topology, bank ports, and Cube granules.

Deadlock zero-token projection/cycle enumeration moved into `backend.graph`.
The checker keeps pairing, loop interpretation and source-located diagnostics.
The performance checker delegates graph construction/scheduling to
`backend.trace`; API-specific byte and Cube shape recovery remain in the adapter.
Interval hazards are reusable independently of source names or locations.

Dependency direction is frontend -> trace IR -> checker adapters -> backend
algorithms. Processor metadata and diagnostics do not depend on frontends.
The graph entry point does not depend on `KernelIR`; the optional trace adapter
does. Source interval/SMT passes remain checkers because they require source
declarations and existing reporting semantics.

No wholesale renaming of `parsing/` into `frontend/` was performed: the two
already have different responsibilities and moving 6,000+ lines offers little
solver isolation. Existing integration test paths remain stable; the new backend
tier can be run independently.
