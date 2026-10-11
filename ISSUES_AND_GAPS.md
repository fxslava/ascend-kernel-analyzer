# ascend_analyzer — issues and gaps

Found while auditing the five DeepSeek-V4-Flash Stage-1 kernels for Ascend
950PR / arch35 (`--chip ascend950pr`, CANN 9.2.0-beta.2, `dav-3510`).

Everything below is a gap in the analyzer, not in the kernels. Each one is
either a check that silently does not run, or a false finding on correct code.

## 1. (FATAL class, silent) The CANN 9.2 static-mode `LocalTensor` constructor is not modeled

`LocalTensor<T>(TPosition pos, uint32_t addr, uint32_t tileSize)` is the
canonical constructor of the static tensor programming model the analyzer
targets, and it is not recognised. A kernel built entirely from it resolves
**zero** on-core tensors, and the report says so only in passing:

```
SRAM footprint - ker02_mhc_post_residual
  No on-core SRAM tensors were resolved for this kernel.
```

while still concluding `ACCEPTED  layout and synchronisation verified.` with
`0 fatal 0 warning 0 info`.

**Why this matters:** UB capacity (AKA1001), base alignment (AKA1002),
aliasing (AKA1003), allocation-size alignment (AKA1005) and bank conflicts
(AKA3003) all depend on resolved layouts. On a kernel using this constructor
they are all skipped, so the audit passes while verifying nothing about the
layout. This is the most dangerous gap here because it fails open.

*Repro:* any kernel whose tensors come from that 3-argument constructor.

*Workaround:* emit a `// @ascend-layout:` line per tensor. The Stage-1 kernels
generate theirs from `scripts/gen_arch35_ub_layout.py`, which is also what
keeps them from drifting out of sync with the offsets the kernel uses. After
annotating, all five kernels resolved their full layout (7 to 26 tensors each)
and the checks ran for real.

*Suggested fix:* in `ascend_analyzer/parsing/ast_visitor.py`, bind
`LocalTensor<T>(TPosition, addr, tileSize)` the way a `TBuf`/`TQue`
allocation is bound: position from argument 0, offset from argument 1, count
from argument 2, element size from the template argument. Also consider making
"no on-core tensors resolved" a diagnostic rather than a note, so an audit that
verified nothing cannot report `ACCEPTED`.

## 2. (false positive) Intrinsics missing from the table make correct barriers look redundant

`ascend_analyzer/apis.py` has no entry for these CANN 9.2 intrinsics, all of
which are PIPE_V operations:

| Intrinsic | Used by |
|---|---|
| `Gather` (level-0, `(dst, src, srcOffset, srcBaseOffset, count)`) | table decode, lane deinterleave |
| `CreateVecIndex` | index-vector construction |
| `Sort<T, isFullSort>` / `Extract` | deterministic top-k |
| `asc_vf_call` | SIMT VF dispatch |

Because the analyzer does not see an operation between two barriers, it emits
`AKA3005 Redundant consecutive barrier` and advises *"Delete the duplicate
barrier."* Following that advice removes a real RAW edge.

```
    3. INFO  AKA3005  Redundant consecutive barrier
       ker04_fp8_indexer_topk.cpp:352   domain: PIPE_V
       consecutive PipeBarrier(PIPE_V) has no effect ...
       fix: Delete the duplicate barrier.
```
Line 352 sits between `Sort<float, true>(...)` and `Extract(...)`.

*Note:* `GatherMask`, `Brcb`, `DataCopyGather` and `DataCopyScatter` **are** in
the table, so this looks like an omission rather than a design decision.

*Suggested fix:* add the four above to the intrinsic table. `Gather`,
`CreateVecIndex`, `Sort` and `Extract` are ordinary PIPE_V ops;
`asc_vf_call` is a PIPE_V-draining dispatch whose `__simt_vf__` callee's
`__ubuf__` arguments are its reads and writes.

## 3. (gap) SIMT raw `__ubuf__` pointers cannot be verified

A `__simt_vf__` function takes `__ubuf__ T*`, which a kernel obtains from an
existing tensor's `GetPhyAddr()`. The analyzer reports:

```
    1. WARNING  AKA3002  Offset or size not statically resolvable
       tensor 'x_ub' has an unresolvable base offset and byte size
       (offset=reinterpret_cast</*ubuf*/ float*>(xf32.GetPhy..., size=?)
```

*Workaround used:* declare the alias with `@ascend-layout` at its source
tensor's offset and put the pair in an `@ascend-reuse-group`, so the
deliberate overlap is not then re-reported as AKA1003.

*Suggested fix:* propagate the layout through
`reinterpret_cast<__ubuf__ T*>(tensor.GetPhyAddr())` instead of treating the
result as a fresh unresolvable tensor; `GetPhyAddr()` on a bound tensor has a
known offset.

## 4. (documentation) Rule catalog disagrees with the implementation

The `ascend-kernel-analyzer-rules` skill documents `AKA3005` as *"Excessive VF
Switching Frequency"* and `AKA3003` as *"UB Bank Conflict"*. The tool emits
`AKA3005` for *"Redundant consecutive barrier"* and `AKA3003` for both *"UB
bank conflict"* and *"EVENT_ID not statically resolvable"*. The skill also
gives the bank formula as `mod 16`; the `ascend950pr` profile is 8 banks on a
32-byte stride, so a collision is a 256-byte offset multiple, not 512.

## 5. (works as intended, worth noting) The 216 KiB arch35 ceiling needs `--ub-bytes`

The `ascend950pr` profile reports the physical 256 KiB (`capacity 262144 B`)
and its own note says at most 216 KiB is usable after the 32 KiB SIMT
DataCache floor and the 8 KiB compiler reserve. A layout between 216 and
256 KiB therefore passes by default. Auditing against the real ceiling needs
`--ub-bytes 221184`.

*Suggested fix:* give the profile a separate usable-tensor-budget field and
check against it, keeping the physical capacity for reporting.

---

### What did work well

* The deadlock/Marked-Graph checker caught a genuine cross-pipeline race in
  KER-03: state written by `DataCopy` on PIPE_MTE3 and read back by
  `DataCopy` on PIPE_MTE2 with only an `MTE3_V` pair between them. The
  diagnostic named both pipes and the fix (`MTE3_MTE2`) was correct.
* `AKA3003 EVENT_ID not statically resolvable` correctly rejected a
  runtime-selected ping/pong event id and suggested unrolling, which is the
  right shape for a 2-stage software pipeline.
* Loop-carried token balance and the acyclicity proof were accurate on all
  five kernels.
