---
name: ascend-pipeline-and-kernel-audit
description: End-to-end design, simulation, and static verification workflow for Huawei Ascend C kernels. Connects pre-code spatial-temporal modeling and autotiling via PipeSim with post-codegen static deadlock, hazard, and bank conflict auditing via ascend_analyzer. Use when designing custom Ascend NPU operators (310P, 910B, 910C, 351x), exploring tiling spaces, diagnosing bottlenecks (MTE1/MTE2/Fixpipe/Cube), synthesizing kernels, or statically verifying Ascend C code against AKA diagnostics.
---

For Ascend 950PR / arch35 KER-01..KER-05, apply the sibling ascend-kernel-verification-protocol skill before running any verification command. Its static audit and device compilation boundary overrides broader simulation examples here.

# Ascend Pipeline & Kernel Audit (PipeSim + ascend_analyzer)

Unified engineering standard for designing, simulating, synthesizing, and statically auditing custom kernels on Huawei Ascend NPU architectures (DaVinci v1/v2/v3; 310P, 910B, 910C, 351x).

This skill combines two complementary tools located in local workspaces:

1. **Pre-code Oracle (`PipeSim`)**: Analytical 3-tier feasibility filter, hierarchical roofline, and Max-Plus discrete pipeline scheduler for finding optimal tiling before writing C++.
2. **Post-code Static Auditor (`ascend_analyzer`)**: Static AST and Happens-Before analyzer detecting hardware deadlocks, UB bank conflicts, address alignment violations, and missing barriers.

### Tool Locations

| Tool | Windows path | WSL / Linux path |
| --- | --- | --- |
| PipeSim | `D:\Projects\PipeSim` | `/mnt/d/Projects/PipeSim` |
| ascend_analyzer | `D:\Projects\ascend-kernel-analyzer` | `/mnt/d/Projects/ascend-kernel-analyzer` |

Use the WSL paths when running under Linux/WSL; if neither resolves, ask for the current checkout location rather than guessing.

---

## The Closed-Loop Workflow

    [Target Algorithm / Shape (M, N, K, Dtype)]
                       │
                       ▼
      Step 1: Pre-Code Exploration & Autotiling (PipeSim)
      - Evaluate candidate tiling configs in < 0.1 ms
      - Check SRAM feasibility (L1 <= 512K, L0A/B <= 64K, L0C <= 128K FP32, UB <= 256K)
      - Run Hierarchical Roofline (GM, L1, L0C ceilings) & Max-Plus T_iter analysis
      - Pinpoint limiting stage: MTE1_BOUND, MTE2_STARVATION, FIXPIPE_BACKPRESSURE, CUBE_BOUND
                       │
                       ▼ [Winning TileConfig & Execution Metadata]
      Step 2: Kernel Synthesis (C++ Ascend C)
      - Queue layout: TQue<TPosition::A1, 2>, B1, A2, B2, CO1
      - Byte-exact buffer sizing derived from PipeSim sram_footprints
      - Ragged edge logic: runtime SetTail(tailM, tailN) when M/N % 16 != 0
      - Explicit Event ID handshake: SetFlag/WaitFlag<HardEvent::...>(EVENT_ID0)
                       │
                       ▼ [Generated kernel.cpp + kernel_tiling.h]
      Step 3: Static AST & Barrier Audit (ascend_analyzer)
      - Run audit with --chip <target> --warnings-as-errors
      - Catch AKA1xxx (Alignments), AKA2xxx (Deadlocks & Leaks), AKA3xxx (UB Banks), AKA4xxx
                       │
                [Audit Verdict]
                ├── REJECTED ──► Apply Self-Healing Patch ──► Re-Audit
                └── ACCEPTED ──► Production Ready

---

## Tool 1: PipeSim Execution

Run from the PipeSim root (or with its `src` directory in `PYTHONPATH`).

### Quick Invocations

```bash
# 1. Probe a specific tiling configuration
python -m ascend_pipesim.cli probe \
  --arch ascend910b \
  --gemm 2048,2048,4096 \
  --tile 128,128,64 \
  --subtile 64,64,32 \
  --buffers double \
  --stationary m \
  --pretty
```

```bash
# 2. Autotile search for optimal tile/subtile configurations
python -m ascend_pipesim.cli autotile \
  --arch ascend910b \
  --gemm 4096,4096,8192 \
  --dtype fp16 \
  --topk 3 \
  --pretty
```

```bash
# 3. Synthesize Ascend C kernel and tiling header to disk
python -m ascend_pipesim.cli codegen \
  --arch ascend910b \
  --gemm 2048,2048,4096 \
  --tile 128,128,64 \
  --subtile 64,64,32 \
  --buffers double \
  --outdir .scratch/generated_kernel/
```

### Key Output Fields

* `is_valid` (bool): True if all SRAM footprints and alignment constraints pass.
* `cube_utilization_pct`: Theoretical utilization of the Cube engine.
* `limiting_stage`: Bottleneck identification (`MTE1_BOUND`, `MTE2_STARVATION`, `FIXPIPE_BACKPRESSURE`, `CUBE_COMPUTE_BOUND`).
* `hierarchical_roofline`: Operational intensity ($I_{\text{GM}}, I_{\text{L1}}, I_{\text{L0C}}$) and attainable ceilings.
* `sram_footprints`: Exact byte occupancy for L1, L0A, L0B, L0C, UB.
* `settail_required` (bool): True if ragged dimension handling is needed.
* `code_generation_hints`: Actionable diagnostics for the kernel writer.

---

## Tool 2: ascend_analyzer Audit

Run with the ascend-kernel-analyzer root in `PYTHONPATH`.

### Quick Invocations

```bash
# Audit a single kernel file
python -m ascend_analyzer path/to/kernel.cpp \
  --chip ascend910b \
  --warnings-as-errors
```

```bash
# Audit with external tiling constants injected
python -m ascend_analyzer path/to/kernel.cpp \
  --chip ascend910b \
  --tiling-data path/to/tiling_manifest.json \
  --warnings-as-errors
```

### Diagnostic Codes & Self-Healing Guide

| Diagnostic Code | Issue | Self-Healing Action |
| --- | --- | --- |
| **AKA1002** `UNALIGNED_ADDRESS` | Address in UB/L1 not aligned to 32 bytes | Enforce `AlignUp(bytes, 32)` on buffer offsets and ensure contiguous row strides are multiples of 32 bytes. |
| **AKA1009** `UNKNOWN_DOMAIN` | Tensor memory domain not recognized | Ensure queues are declared with canonical CANN syntax: `TQue<TPosition::A1, 2>` (use `TPosition`, not `QuePosition`). |
| **AKA2001** `FLAG_LEAK` | `SetFlag` without matching `WaitFlag` | Add matching downstream `WaitFlag<Route>(event_id)` in the consumer pipeline. |
| **AKA2002** `DANGLING_WAIT` | `WaitFlag` without preceding `SetFlag` | Remove unseeded initial wait or trigger an upfront `SetFlag` before loop entry. |
| **AKA2006** `LOOP_TOKEN_IMBALANCE` | Token generation/consumption mismatch in loop | Balance event counts per iteration for ping-pong double buffers. |
| **AKA2010** `MISSING_SYNC` | RAW/WAR/WAW hazard between pipeline stages | Insert required hardware barrier (e.g., `SetFlag<HardEvent::MTE2_V>(EVENT_ID0)` followed by `WaitFlag`). For queue-managed buffers, verify `EnQue`/`DeQue` sequence. |
| **AKA3003** `EVENT_PARAM_MISSING` | `SetFlag`/`WaitFlag` missing event ID | Pass an explicit event ID; on Ascend 950PR / arch35 use only `(EVENT_ID0)` through `(EVENT_ID3)`. |
| **AKA3006** `BANK_CONFLICT` | Two UB operands collide on 8 banks ($\Delta \equiv 0 \pmod{256\text{ B}}$) | Shift second buffer allocation base offset by $+32$ bytes (1 block) to resolve port conflict. |
| **AKA4001** `PIPE_SERIALIZATION` | Overlap destroyed by global barriers | Replace coarse `PipeBarrier(PIPE_ALL)` with targeted pairwise HardEvent flags. |

---

## Mandatory Code Conventions for Ascend C Generation

When writing or synthesizing Ascend C kernels to satisfy both PipeSim budgets and ascend_analyzer:

1. **Queue Position Enum:** Always write `TQue<TPosition::A1, 2>`, never `QuePosition::A1`.
2. **Event IDs:** Always pass explicit event IDs: `SetFlag<HardEvent::M_FIX>(EVENT_ID0);`.
3. **SetTail Invariant:** When `flow.total_m % tile.tile_m != 0` or $N$ is ragged, compute `tailM = min(total_m - idx * tile_m, tile_m)` and call `matmulObj.SetTail(tailM, tailN)` immediately before `Iterate()`.
4. **Buffer Offsets in UB:** When allocating multi-operand work buffers in Unified Buffer, avoid powers-of-two strides of 256/512/1024 bytes without a 32-byte skew offset.
