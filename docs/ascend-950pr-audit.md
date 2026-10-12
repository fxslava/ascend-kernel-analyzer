# Ascend 950PR analyzer audit

Specification baseline: the local CANN 9.1.0 audit in
`C:/ascend-kernel-skills/`, reviewed 2026-10-12. The target aliases
`Ascend950pr`, `dav-c310`, `dav-3510`, `arch35`, and `arch_3510` resolve to
the same profile. Other chip profiles keep their allocation policies.

## Implemented checks

| Area | Predicate and diagnostic |
| --- | --- |
| Logical allocations | Only VECIN, VECOUT, GM, TSCM are accepted on 950PR; UB and other logical placements emit FATAL AKA1004. Raw physical `__ca__`, `__cb__`, `__cc__` pointers remain valid Cube operands. |
| UB partition | `min(262144 - static_bytes - dynamic_bytes - 8192, 131072)`; less than 32768 cache bytes or more than 221184 allocation bytes emits FATAL AKA1001. Reports include both allocation components and the equation. |
| Allocation accounting | TPipe/InitBuffer and static UB arrays are static; LocalMemAllocator and extern UB arrays are dynamic. L1/TSCM is excluded. LocalMemAllocator element counts are folded in the AST at the allocation site and multiplied by dtype width. Extern views share a dynamic base. A known host launch reservation includes, rather than adds to, dynamic allocations. |
| Allocator exclusivity | Referencing an extern UB array while using LocalMemAllocator in the same analyzed kernel emits FATAL AKA1003 for the shared base. |
| VF dispatch | Deprecated `asc_call_vf` emits WARNING AKA2011, with an `autofix` suggestion preserving template and arguments when available. Invalid `asc_vf_call` template/dim3 syntax emits FATAL AKA2012. |
| Launch bounds | Constant N must be within 1..2048, divisible by 32, and cover the dim3 product. The AST folds dispatch dimensions in their lexical environment. Invalid bounds or block dimensions emit FATAL AKA2012. |
| Register pressure | WARNING AKA3008 above 256 threads (64 registers), above 512 (32), and above 1024 (16). These are spill-risk advisories, not measured register usage. |
| Sparse matmul | `SparseMatmulType` and `SetSparse(true)` emit FATAL AKA4006, with MxMatmul/MXMODE remediation. |
| Matrix signatures | `asc_mmad`, `asc_mmad_sync`, `asc_mmad_mx`, and `asc_mmad_mx_sync` enter the API/AST model with ten positional arguments and L0C/L0A/L0B operand domains. Domain mismatches emit AKA1004; invalid arity or type combinations emit AKA4011. |
| Low-bit overloads | Ordinary MMAD permits FP8 pairs and matching hifloat8; MX permits FP4/FP4 and FP8/FP8 pairs. Mixed FP4/FP8, hifloat8 MX, and legacy INT4 are rejected. FP4x2 occupies one byte for two packed values. Accumulator types are checked too. |
| GEMV | Argument 3 is left_height and argument 7 is disable_gemv (zero-based). For height 1, a known true disable flag emits FATAL AKA4011 with ND/L0A remediation. |
| TSCM | Unique backing allocations must total at most 524288 bytes (FATAL AKA4005). Partial copy counts and staging inside iteration loops violate Full Load staging. Unused TBuf allocations count; multiple views of a buffer count once. |

## Diagnostic registry migration

The condensed analyzer skill and its `rules/aka-registry.md` reserve IDs that
the older analyzer had assigned to unrelated findings. The enum, report titles,
README, fixtures, and test expectations now agree with those reservations.
Existing suppressions and saved baselines using the old IDs need migration.

| Finding | Previous ID | Current ID |
| --- | --- | --- |
| Invalid extent / stride alignment / negative offset | AKA1006/1007/1008 | AKA1012/1013/1014 |
| Unpaired WaitFlag | AKA2002 | AKA2001 (shared with unpaired SetFlag) |
| Deadlock cycle | AKA2004 | AKA2002 |
| Symbolic event ID | AKA3003 | AKA3010 |
| UB fragmentation | AKA3004 | AKA3011 |
| Redundant barrier | AKA3005 | AKA3012 |
| UB bank conflict | AKA3006 | AKA3003 |
| Performance stall / underutilization | AKA4001/4002 | AKA3013/3014 |

Skill-owned broadcast, route-mismatch, ragged-tail, VF-switching, and advanced
matmul IDs are reserved with matching titles; this change does not implement
every rule listed in those broader skills. AKA4005, AKA4006, and AKA4011 are the
950PR extensions requested by this task. The external skill repository is not
modified.

## Evidence and limits

Primary local references:

- `ascend-950pr-api-signatures-quickref/references/api-signatures.md`: the
  target-specific ordinary and MX MMAD overloads.
- `ascend-950pr-aiv-simt/references/execution-and-memory.md`: VF launch,
  UB partition, and register budget table.
- `ascend-950pr-cube-matmul/SKILL.md` and its references: unsupported sparse
  operations and complete TSCM residency.
- `ascend-kernel-analyzer-rules/SKILL.md` and `rules/aka-registry.md`: code
  ownership. The requested multiple-of-32 launch-bound policy is enforced
  explicitly; the source's block-size recommendation alone is weaker.

Unresolved allocation extents, launch dimensions/bounds, and matrix configuration
are reported as analysis boundaries (AKA3002), not proven safe or automatically
fatal. Loop-dependent bump allocations require a concrete reservation to establish
the complete budget. Host-launch and sparse/legacy-dispatch discovery retain source
scanners: arbitrary C++ indirection and macro-generated forms are not a complete
compiler-level proof. Full-load checking covers recognized transfer operations;
it does not prove the entire high-level matmul operand shape or synchronization
contract. No CANN compiler or physical NPU execution is claimed.

Validation commands:

Result: **875 tests passed**, including **88** cases in
`tests/test_ascend_950pr.py`; Pyflakes and `git diff --check` passed.

```text
python -m pytest
python -m pyflakes ascend_analyzer tests/test_ascend_950pr.py
git diff --check
```
