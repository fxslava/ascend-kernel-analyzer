# Ascend Static Kernel Analyzer

A static analyzer for Huawei Ascend C kernels written in the **static tensor
programming model** — raw `LocalTensor` addressing with no `TPipe`/`TQue`
buffer manager, where every byte offset and every pipeline handshake is written
by hand.

That programming model trades safety for control. There is no allocator to keep
your tiles from overlapping, and no scheduler to insert the flags your
pipelines need. The two failure modes that follow are both silent at compile
time: a misaligned or overlapping UB offset corrupts data, and a missing
`SetFlag` hangs the core with no diagnostic at all. This tool finds both before
you reach hardware.

```
$ ascend-analyze vec_add.cpp --chip ascend910b

   8. FATAL  AKA2005  Unprimed loop-carried WaitFlag
      vec_add.cpp:82   domain: PIPE_MTE2
      WaitFlag<V_MTE2>(EVENT_ID0) at the top of the loop body (line 80) consumes a
      flag that is only raised later in the same body (line 89), so it can only be
      satisfied by the previous iteration - but nothing primes it before the loop.
      PIPE_MTE2 blocks on the first iteration and the kernel hangs. This wait also
      closes a circular wait across PIPE_MTE2, PIPE_V.
          82 | AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      fix:
        Prime the flag once before the loop:
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        and drain it once after the loop:
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        so every iteration, including the first, finds a token waiting.
```

---

## What it detects

### Memory layout violations

| Code | Finding | Why it is fatal |
|---|---|---|
| `AKA1001` | SRAM capacity overflow | The tile leaves UB/L1/L0x. With a loop-varying offset the solver names the iteration that first escapes. |
| `AKA1002` | Base address misaligned | DaVinci addresses SRAM in 32-byte blocks; a misaligned base makes the DMA engine touch the wrong bytes. |
| `AKA1005` | Allocation size misaligned | The tail block is transferred *whole*, so a ragged length silently clobbers whatever follows it. |
| `AKA1003` | Buffer aliasing / collision | Two tensors live at the same time in one domain must be disjoint. This is where ping/pong and in/out bugs surface. |
| `AKA1004` | Memory domain mismatch | `LocalTensor<T>` hides the address space, so handing a UB tensor to a Cube API that reads L0A type-checks in C++ and fails on hardware. |
| `AKA1010` | Insufficient UB DataCache headroom (351x SIMT) | On 351x the 256 KiB UB is partitioned: `DataCache = 256 KiB − allocations − 8 KiB (compiler reserved)`, and the SIMT runtime requires `DataCache ≥ 32 KiB`. More than 216 KiB of tensor allocation corrupts memory at run time. |
| `AKA1007`/`AKA1008`/`AKA1009` | Stride alignment, negative offset, undetermined domain | |

### Pipeline deadlocks and synchronisation

| Code | Finding | Why it is fatal |
|---|---|---|
| `AKA2005` | Unprimed loop-carried `WaitFlag` | Iteration 0 waits on a flag only the *previous* iteration raises. The single most common double-buffering hang. |
| `AKA2004` | Circular wait | A token-free cycle in the pipeline dependency graph. The report prints the full cycle path. |
| `AKA2001`/`AKA2002` | `SetFlag` with no `WaitFlag`, or the reverse | An unconsumed flag leaks an event slot; an unraised one blocks for ever. |
| `AKA2006` | `SetFlag`/`WaitFlag` imbalance in a loop body | The semaphore drifts one count per iteration until it overflows or underflows. |
| `AKA2003` | Reserved `EVENT_ID` 6 or 7 | The runtime owns these and may consume or raise them behind your back. |
| `AKA2007`/`AKA2008`/`AKA2009` | Double set, out-of-range id, same-pipeline route | |
| `AKA3001` | `PipeBarrier(PIPE_ALL)` antipattern | Warning. Drains every pipeline, discarding the overlap double buffering exists to create. The fix names the two pipelines that actually share data. |
| `AKA3006` | UB bank conflict on Vector ALU | Warning. UB is an interleaved 8-bank structure in 32-byte blocks; a dual-operand instruction (`Add`, `Mul`, `Sub`, `Max`, `Min`, ...) whose two sources sit `8k` blocks apart reads both from the same bank and stalls the read ports. Pad one source by +32 bytes (one DaVinci block) for bank-orthogonal bases. Note: this check was specified as "AKA3003", but that code already belongs to `SYMBOLIC_EVENT_ID` and is pinned there by the shipped tests, so the bank conflict carries the next free code in the 3xxx performance block. |

### Performance model and overlap profiling

| Code | Finding | What it means |
|---|---|---|
| `AKA4001` | Exposed sync bubbles / pipeline stalls | Warning. Cross-queue `SetFlag`/`WaitFlag` hand-offs each cost ~30 cycles; when they exceed a third of the kernel's issued work, the pipelines are stalling more than computing. The message names the two worst hand-off routes. |
| `AKA4002` | Cube compute underutilization | Warning. The cube unit is busy under half the modeled makespan while the move engines feed it - the contraction is not the critical path, the feeding chain is. |

`ascend-analyze --list-codes` prints the full table.

---

## Install

```bash
pip install -e ".[dev]"
```

Or just the runtime dependencies:

```bash
pip install tree-sitter tree-sitter-cpp networkx z3-solver
```

`z3-solver` is **optional**. Without it the analyzer falls back to interval
arithmetic, which is exact for fully constant layouts — the common case — but
cannot produce counterexamples for loop-varying offsets. `--solver interval`
forces that backend; `--solver z3` requires the solver.

---

## Use

```bash
# Terminal report with the SRAM footprint chart
ascend-analyze kernel.cpp --chip ascend910b

# CI gate: exit 1 on any FATAL, 2 on warnings with -W
ascend-analyze kernels/ --quiet -W

# Machine-readable output
ascend-analyze kernel.cpp --format json -o findings.json

# Standalone HTML report with a proportional memory map
ascend-analyze kernel.cpp --html report.html

# Record today's findings, then gate on regressions only
ascend-analyze csrc/ --write-baseline baseline.json --baseline-root csrc
ascend-analyze csrc/ --baseline baseline.json --quiet

# Resolve tiling-dependent layouts from a mined manifest, or by role
ascend-analyze kernel.cpp --tiling-data tilings.json:default
ascend-analyze kernel.cpp --infer-tiling-roles
```

Exit codes: `0` clean, `1` fatal findings, `2` warnings under
`--warnings-as-errors`, `3` usage or I/O error. A baselined run computes its
verdict — and therefore its exit code — on the regressions only, so an adopted
backlog does not keep the gate red.

### Baselines

`--write-baseline` records every current finding; `--baseline` suppresses those
and reports only what is new. A finding is identified by its file, its code and
its message, deliberately **not** by its line: inserting a comment above a
kernel shifts every line below it, and treating that as a hundred new findings
would make the baseline useless after the first unrelated edit. A code that
fires *more* often than the baseline recorded reports the surplus, so a second
instance of a known problem is not hidden by the first.

Paths are normalised, and `--baseline-root` records them relative to a
directory, so one baseline matches a tree spelled `D:/Projects/x`,
`/mnt/d/Projects/x` or `/d/Projects/x`.

### Tiling-dependent layouts

A kernel that sizes its buffers from a host tiling struct resolves no offsets
at all without that struct, which silently disables every offset-dependent
check. Two options close the gap:

* `--tiling-data PATH[:KEY]` binds concrete field values from a JSON manifest —
  the shapes a test suite already pins.
* `--infer-tiling-roles` infers values from the **role** each field plays at
  its call sites, for the fields no manifest supplies.

Role inference reads context, never names. A field that reaches the extent
argument of `InitBuffer`/`InitQueue` is a buffer extent and takes the
architecture-minimal valid dimension (16 for a Cube fractal, 64 for a vector
tile); one that reaches the parameter block of `DataCopy`/`DataCopyPad` is a
stride and takes the 32-byte DaVinci block; one compared against
`GetBlockIdx()` is a core count and is left symbolic. The walk runs from the
call site backwards through names assigned exactly once, because production
kernels unpack the struct in one method and size their buffers in another.

Three restrictions keep an inference from doing harm: only fields reached
through a verified tiling pointer are touched, a supplied or source-derived
value always wins, and a field used in two disagreeing roles is left symbolic.
The values are inferred, so the flag is off by default — an inference should
never be the reason a kernel is rejected. Whatever was inferred is reported on
the unit, so every conclusion resting on one is visible.

As a library:

```python
from ascend_analyzer import KernelAnalyzer, AnalyzerOptions

analyzer = KernelAnalyzer(AnalyzerOptions(chip="ascend910b"))
result = analyzer.analyze_file("vec_add.cpp")

for diag in result.diagnostics:
    print(f"{diag.loc} {diag.code.value} {diag.message}")
    print(f"  fix: {diag.remediation}")
```

---

## How it works

```
source.cpp
    │
    ├─ macro expand ─── inline #include "local.h", blank #define directives,
    │                  expand function-like macro invocations (stage macros,
    │                  ping/pong event selectors), fold f<<<cfg>>>(args)
    │                  launches - tracking which original line each output
    │                  line came from
    │
    ├─ preprocess ──── rewrite __ubuf__ → /*ubuf*/  (equal length, so every
    │                  line and column still matches the parse basis)
    │
    ├─ tree-sitter-cpp ──── C++ syntax tree
    │
    ├─ ASTVisitor ──── KernelIR: an ordered operation trace + a tensor table
    │                  with symbolic byte offsets, plus loop and scope
    │                  nesting; small statically-bounded loops whose body
    │                  uses the induction variable are replayed once per
    │                  iteration with concrete values
    │
    ├─ MemoryChecker ──── Z3 / interval engine over byte ranges
    │
    └─ DeadlockChecker ── NetworkX marked graph over pipeline dependencies
```

### Parsing: macro expansion before tree-sitter

Cube kernels are written as *stage macros* — `#define A_MAD(t, p) do { ... }
while (0)` blocks of ten-plus lines. `tree-sitter-cpp` cannot digest a
multi-line function-like macro defined inside a function body: its
`preproc_function_def` node terminates early and the remaining body lines leak
into the enclosing function as stray statements, landing every interesting
construct in `ERROR` nodes.

The macro expander runs *before* tree-sitter and handles exactly what kernels
need: local `#include "header.h"` files are inlined (CANN and system headers
stay untouched), `#define`/`#undef` lines are blanked, and function-like
invocations are expanded with argument substitution and rescanning — so
`A_MAD(t, p)` becomes the stage's statements at the call site and
`EV(p)` becomes `((p) ? EVENT_ID1 : EVENT_ID0)`, which folds once `p` is
concrete. Every output line remembers the original line it came from
(invocation sites for expansions, the `#include` line for headers), so
diagnostics still point into the file you are editing.

### Parsing: the equal-length comment trick

DaVinci address-space qualifiers (`__ubuf__`, `__cbuf__`, `__gm__`) and kernel
attributes (`__global__`, `__aicore__`) are compiler extensions that
`tree-sitter-cpp` does not know, and they derail its grammar into `ERROR` nodes
exactly where the interesting declarations are.

Rather than tolerate a broken tree, each qualifier is rewritten into a block
comment of **identical byte length** — `__ubuf__` becomes `/*ubuf*/`. The
grammar then parses the file cleanly while every byte offset, line and column
still matches the original source, so diagnostics point at the real file with
no mapping table. The qualifiers are not lost: their byte spans are recorded,
so the visitor can recover each declaration's address space.

The rewrite is comment- and string-aware. Rewriting `__ubuf__` *inside* a block
comment would inject a `*/` that closes the comment early and spill prose into
the token stream — and kernel files routinely mention these qualifiers in their
header comments. Operands of `#ifdef`/`#ifndef`/`#undef` are left alone too:
`#ifndef __force_inline__` is a guard testing whether a macro is defined, and
rewriting that name would leave the directive without an identifier.

### Parsing: CCE declaration decoration

Three further constructs used to turn a whole translation unit into a single
`ERROR` node. Each is blanked to spaces of the same byte length, so line and
column coordinates are untouched:

* the **CCE location qualifier**, `__forceinline__ [host, aicore] void f()`.
  `tree-sitter-cpp` reads the `[` as a lambda capture and never recovers. It is
  told apart from an array subscript (`buf[host]`), a C++ attribute
  (`[[nodiscard]]`) and a lambda by three tests: a qualifier is never glued to
  the preceding token, never doubled, and is always followed by something that
  starts a type. A list of two or more names skips the first test, since a comma
  inside a subscript is not valid C++.
* an object-like macro that expands to **nothing but decoration**, such as
  `#define HOST_DEVICE __forceinline__ [host, aicore]`. The expander blanks the
  `#define` but leaves every use standing, and a bare identifier ahead of a
  constructor derails the rest of the file. Such a macro is detected from its
  body, not from a hard-coded name list, and blanked at each use.
* `#ifdef __cplusplus` guarding `extern "C" {`, where the brace opens inside one
  preprocessor block and closes inside another. `tree-sitter-cpp` requires each
  block to be brace-balanced. Resolving the guard is exact rather than
  approximate: this analyzer always parses as C++, so `__cplusplus` *is*
  defined, the directive lines are inert, and blanking them leaves
  `extern "C" { ... }` as ordinary balanced code.

Two more constructs come from headers that are **not in the tree at all** —
Catlass and the CANN tiling-key DSL are external dependencies, so their
`#define`s can never be found and the body-based test above cannot classify
them. Both rules are therefore *positional*, keyed on where the identifier
sits rather than on what it is called:

* a bare ALL-CAPS identifier **alone on its own line** at a declaration
  boundary, or **directly in front of** `struct`/`class`/`template`/a type
  keyword. No such line is valid C++ on its own, so either it is decoration —
  and blanking it fixes the parse — or it expands to a whole declaration, which
  was already unparseable. `CATLASS_DEVICE` decorates 42 files this way.
* a **multi-line** ALL-CAPS macro invocation used as a statement, which is how
  `ASCENDC_TPL_SEL(...)` writes a tiling-key table. Its argument list is not
  valid C++ even as an expression. Only the multi-line form is blanked: a
  one-line `FOO(a, b);` parses as an ordinary declaration, and a `)` followed
  by `{` is a definition whose body is left intact, so `TORCH_LIBRARY` blocks
  survive.

Macro expansion handles four more cases that cascaded the same way:

* a **variadic** `#define M(...)`, whose `__VA_ARGS__` was never substituted;
* **stringification** — `GetOpApiFuncAddr(#aclCreateTensor)` left a `#` reading
  as a directive in mid-statement, which alone accounted for thousands of error
  lines in `torch_binding.cpp`;
* an **operator token passed as an argument** — `UNARY_OP(+)` came out as the
  ill-formed `operator (+)`, because arguments are parenthesised to protect
  precedence;
* a **single-token argument**, which was parenthesised for the same reason and
  so produced `(SCFABlockCube)<ARGS>` in a type position.

The last two share one rule: parentheses protect *precedence within an
expression*. An argument that is not an expression, or that is a single token,
has no precedence to protect, so it goes in bare.

### Memory verification: satisfiability, not guesswork

Each layout question is posed as a satisfiability question about the
**violation**:

* bounds — can `[offset, offset + size)` ever leave the domain?
* alignment — can `offset` ever fail to be a multiple of 32?
* disjointness — can two ranges ever overlap?

`UNSAT` is a proof the kernel is safe for every value the free variables can
take. `SAT` hands back a concrete counterexample:

```
tensor 'tile' does not fit in UB: [t * 8192 .. +8192) against a 196608 B
(192 KiB) capacity; range can end at byte 204800, 8192 B past the 196608 B
capacity (with t = 24)
```

Offsets are constant-folded first, including `sizeof(T)` and chains of
dependent `constexpr` declarations, so `UB_PONG = UB_PING + TILE_ELEMS *
sizeof(half)` resolves to a number. Loop induction variables become *bounded*
free variables, which is what lets the solver find `t = 24` above.

**On not inventing findings.** When an offset contains a free variable the
analyzer could not bound — an unparsed macro, a host-supplied tiling field — it
reports the gap (`AKA3002`) and *excludes* that tensor from the layout proofs.
Letting the solver range freely over an unknown manufactures
guaranteed-looking violations out of pure ignorance, which is worse than
silence.

### Liveness: why program order is not enough

Two tensors conflict when their byte ranges overlap *and* both hold live data
at the same time. Lexical trace order is not sufficient to decide the second
part: the whole point of a software-pipelined kernel is that MTE2 is filling the
next tile while the vector unit computes the current one and MTE3 drains the
previous one. Tensors whose uses never overlap in program order are routinely
live simultaneously on hardware.

So any two tensors referenced from the same loop body are treated as
concurrent. That is what makes the ping/pong collision in
`tests/kernels/pingpong_broken.cpp` visible at all. Deliberate buffer recycling
is *declared* rather than inferred — see `@ascend-reuse-group` below.

### Deadlock detection: a marked graph

A DaVinci AI Core is a set of pipelines, each consuming its own instruction
queue strictly in order and otherwise running free.
`SetFlag<HardEvent::SRC_DST>(id)` raises a counting flag on pipeline `SRC`;
`WaitFlag<HardEvent::SRC_DST>(id)` blocks pipeline `DST` until it can consume
one. A `(route, event_id)` pair is therefore a counting semaphore, and the
kernel is a **marked graph** with three edge kinds:

* **program order** — consecutive operations on one pipeline, 0 tokens;
* **back edges** — the last operation on a pipeline to its first, 1 token,
  closing the loop body so loop-carried dependencies become visible;
* **synchronisation** — each `SetFlag` to the `WaitFlag` that consumes it
  (FIFO-matched per channel), with 0 tokens when the set precedes the wait, and
  the prologue priming count when the wait precedes the set.

A marked graph deadlocks exactly when some directed cycle holds no tokens.
Every 0-token edge advances the trace index *except* an unprimed loop-carried
synchronisation edge, so the search reduces to cycle detection over the 0-token
subgraph — linear, exact, and it yields the full circular-wait path for the
report. When the graph is acyclic, its topological order is both the proof of
deadlock freedom and a legal issue order; it is published in the JSON report.

A missing prime is reported once, as the precise root cause (`AKA2005`), rather
than as every cycle it creates. One mistake should not produce a wall of
findings.

### Long loops: three-phase peeling and the steady state

Full unrolling is exact but explodes past small trip counts, while a single
symbolic body pass is blind to the transient states at the loop boundaries:
`if (t >= 1)` prologue guards, `if (t + 2 < T)` epilogue guards and the
`p = t & 1` ping-pong parity all refuse to fold, so the checkers see phantom
operations from dead branches and symbolic event ids instead of real pairings.

Loops with a known trip count above the unroll limit are therefore executed as
three phases, after extracting the loop's **critical boundary points**:

* linear guard roots — `t >= C` switches at `C`, `t + k < T` switches at
  `T − k`, point guards (`t == C`) at `C` and `C + 1`;
* modular periods — `t & 1`/`t % 2` give period 2 (standard double
  buffering), and the representative cycle length is the `lcm` of all of them.

**Phase A (peeled head)** replays the iterations before the last early switch
(`t ∈ [0, max C)`, typically `{0, 1}`) with concrete induction values, so dead
branches prune statically. **Phase B (steady state)** does *not* unroll the
bulk: it emits a minimal representative cycle of the modular period (two
iterations for ping-pong parity, chosen where prologue conditions have settled
true and epilogue conditions false) whose operations carry the loop id — they
form the cyclic subgraph of the marked graph, closed by one-token back edges.
**Phase C (peeled tail)** replays the terminal iterations where the epilogue
guards flip (`t ∈ [T − k, T)`), where the loop-draining `WaitFlag`s consume
the leftover tokens.

A guard that flips mid-loop fits none of the phases, so such loops keep the
exact symbolic treatment rather than being mis-abstracted. When the trip count
is unresolved, Phase A still runs with the bounds that fold without knowing
`T`, and the steady state is projected at symbolic induction offsets — the
loop-carried sync edges survive without inventing a trip count.

The result: `tests/kernels/loop_peeling_long.cpp` (a four-channel pipelined
vector kernel, `T = 512`) analyses clean in well under the 200 ms budget with
a 32-node synchronisation graph instead of ~4 000 traced operations.

### 351x SIMD/SIMT Unified Buffer budgeting

The Ascend 351x runs isomorphic SIMD/SIMT execution, and its Unified Buffer is
strictly partitioned:

```
DataCache = 256 KiB − StaticMem − DynamicMem − 8 KiB (compiler reserved)
```

Whenever the SIMT path is present — `__simt_vf__`, `__simt_callee__` or
`asc_call_vf` appears in the translation unit — the runtime requires
`DataCache ≥ 32 KiB`. The analyzer sums static tensor allocations and dynamic
`TPipe::InitBuffer` extents (each storage counted once) against the profile's
`max_usable_ub_bytes` (216 KiB) and raises `AKA1010` when the floor would be
breached. Other profiles do not partition the UB and are unaffected.

### The analytical performance model

Once the deadlock checker has proven the marked graph acyclic, its
topological order is a legal issue order - so an as-soon-as-possible schedule
over the dependency DAG is a faithful first-order performance model. The
`perf` checker builds one from the chip's latency model:

| Quantity | Model |
|---|---|
| MTE2 / MTE1 / Fixpipe(+MTE3) | `bytes / 64 B-per-cycle` sustained transfer bandwidth |
| Vector unit | one `vector_bytes` (256 B) register footprint per cycle |
| Cube contraction | `ceil(M/16) · ceil(K/16) · ceil(N/16)` cycles - one 16×16×16 fractal per cycle, with the shape recovered from `mmad_t::shape_t((uint16_t)M, ...)` call sites |
| `SetFlag` → `WaitFlag` hand-off | ~30 cycles of cross-queue latency per matched pair |

The schedule yields the **critical-path makespan**, per-pipe busy / idle /
**stall** timestamps (a stall is time a queue sat at a `WaitFlag` with nothing
else issued), the **overlap ratio** (`Σ busy / (makespan × pipes)`) and a
bottleneck classification: `DRAIN_BOUND` (a serialized tail nothing overlaps),
`SYNC_BOUND` (hand-off bubbles dominate), `MEMORY_BOUND` (a move engine drives
the critical path) or `COMPUTE_BOUND` (the cube/vector unit does). The
terminal report prints the summary with ASCII utilization bars per pipeline,
and the JSON report carries the full profile (`--disable perf` skips it;
`--no-perf-summary` hides the section).

Advisories (`AKA4001`, `AKA4002`) are gated on a modeled makespan of at least
500 cycles so short kernels' inherent fill/drain bubbles are not nagged - the
negative-control fixtures stay silent. `tests/kernels/
nvfp4_pipelined_dequant.cpp` demonstrates both ends: its double-buffered
Pipeline A still exposes the ~30-cycle `M_FIX` hand-off per small tile, while
its Pipeline B serializes every stage behind five handshakes and lands at a
fraction of Pipeline A's overlap.

---

## Recognised source forms

The analyzer reads the following tensor-binding idioms. Anything outside this
set is reported as unresolved rather than guessed at.

```cpp
// 1. LocalTensor with explicit position and byte offset
AscendC::LocalTensor<half> xUb;
xUb.SetTPosition(AscendC::TPosition::VECIN);
xUb.SetAddr(UB_X_PING);      // byte offset
xUb.SetSize(TILE_ELEMS);     // element count  (SetBufferLen sets bytes)

// 2. Raw DaVinci address-space pointer
__ubuf__ half* p = (__ubuf__ half*)(UB_OFFSET);

// 3. Buffer accessor carrying a TPosition
AscendC::TBuf<AscendC::TPosition::VECIN> ubBuf;
AscendC::LocalTensor<half> t = ubBuf.GetBufferByByte<half>(256);

// 4. Explicit factory
AscendC::LocalTensor<half> t =
    AscendC::GetLocalTensor<half>(AscendC::TPosition::VECIN, 128, 64);

// 5. Global memory
AscendC::GlobalTensor<half> g;
g.SetGlobalBuffer(xGm, TILE_ELEMS * TILE_COUNT);

// 6. TPipe buffers: the position and size propagate from the TBuf
//    declaration and the InitBuffer call, so no annotation is needed.
AscendC::TPipe pipe;
AscendC::TBuf<AscendC::TPosition::A2> l0aBuf;
pipe.InitBuffer(l0aBuf, 2 * TILE_A_BYTES);
AscendC::LocalTensor<int8_t> l0a = l0aBuf.Get<int8_t>();
```

Low-level Cube intrinsics are recognised with their Clang CCE address-space
casts — the casts are stripped when resolving which tensor an argument
touches:

```cpp
load_cbuf_to_ca_s4(
    (__ca__ fp4x2_e2m1_t *)(uintptr_t)l0a[ping * 512].GetPhyAddr(),
    (__cbuf__ fp4x2_e2m1_t *)(uintptr_t)l1a[ping * 512].GetPhyAddr(),
    ...);
mad_mx((__cc__ float *)(uintptr_t)l0c.GetPhyAddr(), (uint64_t)0,
       (__ca__ float4_e2m1x2_t *)(uintptr_t)l0a.GetPhyAddr(), (uint64_t)0,
       (__cb__ float4_e1m2x2_t *)(uintptr_t)l0b.GetPhyAddr(), (uint64_t)0,
       mmad_t::shape_t(M, K, N), ctl);
```

Event ids may be any expression that folds: a literal, a macro-expanded
ternary over the loop's ping/pong selector, or a single-`return` helper
function such as `static __aicore__ inline event_t ev(int p)
{ return p ? EV1 : EV0; }` with constant arguments.

Synchronisation is recognised in both the Ascend C and low-level ISASI
spellings:

```cpp
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
AscendC::PipeBarrier<PIPE_ALL>();

set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
pipe_barrier(PIPE_ALL);
```

### Annotations

Comment annotations are the documented escape hatch for what the parser cannot
infer, and for declaring intent it must not infer.

```cpp
// Declare a layout the parser cannot see (a raw pointer carries no length):
// @ascend-layout: name=xPingRaw pos=VECIN offset=0 count=256 dtype=half

// Declare that an overlap is deliberate buffer recycling, not a bug:
// @ascend-reuse-group: group=scratch names=stage1,stage2

// Silence specific codes on this line or the next:
// @ascend-ignore: AKA1005
```

---

## Target hardware

```
$ ascend-analyze --list-chips
Built-in chip profiles:

  ascend910b   Ascend 910B (Atlas A2 training series)
               aliases: 910b, ascend910b2, ascend910b3, atlas-a2, a2
               UB     196608 B ( 192.0 KiB)  base align 32 B
               L1     524288 B ( 512.0 KiB)  base align 32 B
               L0A     65536 B (  64.0 KiB)  base align 512 B
               L0B     65536 B (  64.0 KiB)  base align 512 B
               L0C    131072 B ( 128.0 KiB)  base align 1024 B
               BT       1024 B (   1.0 KiB)  base align 64 B
               FB       2048 B (   2.0 KiB)  base align 128 B
               reserved event ids: [6, 7]  max: 7
               Baseline profile used by the analyzer regression suite.
```

Huawei does not publish a single authoritative capacity table for every SKU, so
profiles carry a `provisional` flag and a `notes` string, and the report
discloses when a provisional profile was used. **Treat the shipped numbers as
sensible defaults, not as gospel**, and override them for your part:

```bash
ascend-analyze kernel.cpp --ub-bytes 262144 --l1-bytes 524288
ascend-analyze kernel.cpp --chip-profile my910b.json
```

```json
{
  "name": "my910b", "base": "ascend910b",
  "reserved_event_ids": [6, 7], "max_event_id": 7,
  "domains": { "UB": { "capacity_bytes": 262144, "base_alignment": 32 } }
}
```

The `ascend910b` profile is the one the regression suite pins; `ascend910c` and
`ascend351x` are marked provisional.

---

## Tests

```bash
python tests/harness.py            # the runnable harness, exits non-zero on mismatch
python tests/harness.py --verbose  # plus the full report per fixture
python -m pytest                   # 529 tests
```

The harness runs every fixture and checks the findings against expectations
each fixture declares in its own header comment, so the fixtures stay
self-documenting:

```
  fixture                 verdict                 check      fatal  warn  info   codes
  ------------------------------------------------------------------------------------
  domain_mismatch.cpp     rejected                ok             4     0     0   AKA1004
  isasi_raw.cpp           accepted_with_warnings  ok             0     1     0   AKA3001
  pingpong_broken.cpp     rejected                ok            13     1     0   AKA1002, AKA1003, ...
  pingpong_clean.cpp      accepted                ok             0     0     0   -
  ub_overflow.cpp         rejected                ok             1     0     0   AKA1001

PASSED  all 5 fixtures match their declarations
```

| Fixture | What it exercises |
|---|---|
| `pingpong_broken.cpp` | The headline case: a double-buffered vector add with an intentional deadlock (unprimed loop-carried handshakes) and intentional unaligned offsets (a 250-element tile is not a whole 32-byte block). |
| `pingpong_clean.cpp` | The negative control — the same kernel, repaired. Must report **nothing**. A checker that cannot stay quiet on correct code is worthless however many real bugs it finds. |
| `ub_overflow.cpp` | A loop-varying offset that overflows UB at iteration 24; exercises the solver's counterexample path. |
| `domain_mismatch.cpp` | Cube "phantom type" errors: UB tensors handed to L0A-reading APIs, an illegal GM→L0B transfer. |
| `isasi_raw.cpp` | Raw `__ubuf__` pointers, ISASI `set_flag`/`wait_flag`, and `@ascend-layout` annotations. |
| `bank_conflict_vec.cpp` | A dual-operand `Add` whose sources sit 8 blocks (256 B) apart — same UB bank, `AKA3006` — next to a control pair skewed by one 32-byte block. |
| `loop_peeling_long.cpp` | A four-channel pipelined kernel with `T = 512` and ping-pong parity: three-phase peeling keeps the sync graph at 32 nodes and the analysis far under 200 ms, with zero findings. |
| `simt_ub_budget_351x.cpp` | 220 KiB of UB tensors plus `asc_call_vf` calls: fatal `AKA1010` DataCache starvation on `--chip ascend351x` (plain `AKA1001` overflow on the default 910B profile). |
| `nvfp4_pipelined_dequant.cpp` | The performance-profiler fixture: Pipeline A is double-buffered but its small NVFP4 tiles cannot amortise the `M_FIX` hand-off (`AKA4001`, `AKA4002`), while Pipeline B serializes the identical stages behind five handshakes at half the overlap ratio. |

---

## Known limits

* **Interprocedural analysis** — only the kernel body is walked. A handshake
  split across a helper function is not tracked.
* **Conditionals** — an `if` whose condition folds to a constant keeps only
  the taken arm (this is what prunes the epilogue guards of an unrolled
  pipeline loop); a genuinely runtime condition walks both arms as if
  executed, and pairing diagnostics on such a path are *softened to warnings*
  rather than reported as fatal.
* **Loop bounds** — trip counts are recovered from simple `for` headers
  (`i < N`, `i <= N`, `i += k`). Loops with a known trip count of at most 8
  whose body references the induction variable are replayed once per
  iteration with concrete values (so `p = t & 1` and `EV(p)` fold); larger
  loops with induction-dependent guards or parity are executed as three
  peeled phases (head / steady-state representative cycle / tail — see
  above), and the rest stay symbolic with the trip count unknown.
* **Macro expander** — conditional compilation is not evaluated, with one
  exception: an `#ifdef __cplusplus` guard is resolved, because this analyzer
  always parses as C++. Other `#if` arms are left for tree-sitter.
  Token pasting, stringification and variadic macros are supported.
* **SFINAE template parameters** — `tree-sitter-cpp` cannot parse
  `template <class T, typename std::enable_if<...>::type* = nullptr>`, even in
  its simplest one-line form. Headers in the `tla/` style therefore keep a
  small residue of `ERROR` nodes, one per constrained declaration. The failure
  no longer *cascades* — the rest of the file parses — and a type constraint
  carries nothing the analyzer models, so no finding depends on it.
* **TPipe layout synthesis** — buffers are bump-allocated per domain in
  `InitBuffer` program order. A buffer without a visible `InitBuffer` size
  keeps a symbolic extent, **and blocks its whole domain**: everything the
  allocator hands out after it sits at an offset that cannot be known, so no
  later buffer in that domain gets a concrete base. That is what
  `--infer-tiling-roles` exists to unblock when the missing size is a tiling
  field.
* **Flag depth** — the hardware event counter is modelled as an unbounded
  semaphore; `AKA2007` warns about double-sets but no exact saturation depth is
  enforced.
* **One translation unit at a time** — `#include` of headers *next to the
  source* is inlined for constant folding; CANN headers such as
  `kernel_operator.h` are not read.

## Layout

```
ascend_analyzer/
  hardware.py          chip profiles, pipes, HardEvent routes, TPosition mapping
  apis.py              intrinsic signatures: issuing pipe + legal operand domains
  symbolic.py          integer expression IR, folding, intervals, decidability
  solver.py            Z3 and interval backends behind one protocol
  diagnostics.py       severities, stable codes, the collector
  ir.py                KernelIR: operation trace, tensor table, scopes, loops
  analyzer.py          the pipeline facade
  cli.py               command line interface
  parsing/
    preprocess.py      equal-length qualifier rewrite, annotation extraction
    expr_eval.py       tree-sitter nodes → symbolic IR
    ast_visitor.py     syntax tree → KernelIR
  checkers/
    memory.py          capacity, alignment, aliasing, domain mismatch
    deadlock.py        pairing, priming, marked-graph cycle detection
  report/
    terminal.py        coloured terminal report with source frames
    memory_map.py      ASCII and HTML memory footprint charts
    json_report.py     versioned structured output
    html_report.py     standalone HTML report
```

## License

Apache-2.0.
