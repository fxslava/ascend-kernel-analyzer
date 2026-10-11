---
name: ascend-lowlevel-kernel-dev
description: Guidelines and patterns for developing high-performance, low-level Ascend C kernels targeting Huawei Ascend NPUs (310P, 910B, 910C, 351x, 950PR, 950DT). Use when implementing custom operators, optimizing memory layout (UB/L1), designing static pipelines, managing ISASI HardEvents or BufferID ownership, or architecting static kernel analyzers and SIMD/SIMT hybrid kernels.
---

# Low-Level Ascend C Kernel Engineering

Engineering standards for custom, latency-sensitive compute kernels on Huawei Ascend NPU architectures (DaVinci v1/v2/v3, AIC/AIV, 351x).

## Core Architectural Axioms

1. **Avoid High-Level TPipe/TQue Overheads:**
   - Dynamic allocations and queues incur non-trivial init latency (~hundreds of ns) and runtime queue management.
   - For custom operators (fused attention, layernorm, activations, quant/dequant), employ static memory layouts using compile-time byte offsets with `LocalTensor`.

2. **Microarchitecture-Specific Targeting:**
   - Always target concrete chip architectures (`Ascend 310P`, `Ascend 910B`, `Ascend 910C`, `351x`, `950PR`, `950DT`).
   - Call `AscendC::InitSocState()` at kernel entry in static mode to ensure global configuration registers (FPU masks, saturation, precision modes like HF32) are initialized.

3. **No Indiscriminate Pipeline Flushes:**
   - Never use coarse global stops like `PipeBarrier(PIPE_ALL)`.
   - Restrict intra-pipe barriers (`PipeBarrier<PIPE_V>()`) strictly to actual algorithmic hazards (e.g., horizontal reductions in Softmax before subsequent elementwise scaling).

---

## Memory Layout and Bank Interleaving

- **SRAM Alignment:**
  - Enforce 32-byte alignment for all tensors and DMA transactions.
- **Unified Buffer (UB) Management:**
  - Allocate static offsets explicitly.
  - Interleave ping-pong buffers and multi-operand addresses across UB memory banks (32-byte stride) to prevent hardware bank conflicts during concurrent vector loads and stores.
- **Double / Multi-Buffering Pattern:**
  - Maintain explicit ping-pong `LocalTensor` instances:
    ```cpp
    LocalTensor<half> xPing(TPosition::VECCALC, addrPing, TILE_LEN);
    LocalTensor<half> xPong(TPosition::VECCALC, addrPong, TILE_LEN);
    ```

---

## ISASI HardEvent Synchronization

Synchronize only interdependent functional pipelines via targeted hardware events.

### Hardware Event Routing

- `MTE2_V`: Vector pipeline awaits DMA-in from Global Memory to UB.
- `V_MTE3`: DMA-out from UB to Global Memory awaits Vector computation completion.
- `V_MTE2`: DMA-in awaits Vector read completion before overwriting ping/pong buffer (WAR hazard prevention).
- `MTE3_MTE2`: Multi-buffering feedback; prevents MTE2 from corrupting buffers before MTE3 completes writing to HBM.
- `V_S` / `S_V`: Handshake between Vector Unit and Scalar Unit during reduction/loop boundary checks.

### Rules of Engagement

- **Event ID Restrictions:** On Ascend 950PR / arch35, use only `EVENT_ID0` through `EVENT_ID3`; `EVENT_ID4` through `EVENT_ID7` are unavailable. Check the installed target for other chips.
- **Strict Event Pairing:** Every `SetFlag<Route>(id)` must have a deterministic matching `WaitFlag<Route>(id)` in the corresponding downstream unit. Avoid counter drift across loop iterations.

---

## SIMD and SIMT Hybrid Architecture (351x / Vector Functions)

On supporting hardware (351x architecture):

- **Vector Function (VF) Granularity:** Execution alternates sequentially between SIMD and SIMT modes at VF granularity on the AIV core.
- **SIMT VF Role:**
  - Utilize for irregular control flow, dynamic branching, Paged KV-cache index traversal, and direct byte unpacking/dequantization.
  - Access Global Memory directly via hardware Data Cache without staging through explicit DMA descriptors.
- **SIMD VF Role:**
  - Execute dense, compute-heavy, elementwise vector arithmetic on wide 256-byte ALU registers.
- **Cube Offload:**
  - Unpack directly from SIMT into UB/L1, feeding Cube engines directly to eliminate redundant copy pipelines.

---

## Static Kernel Analyzer Guidelines

When auditing or statically analyzing Ascend C kernels:

1. **Memory Budgeting:** Verify $\sum \text{allocated buffers} \le \text{SRAM capacity}$ for the selected SoC. Ascend 950 has 256 KiB L0C and 512 KiB UB per Cube-plus-two-Vector subsystem (256 KiB per Vector core); older chips differ.
2. **Deadlock & Barrier Checker:**
   - Detect unpaired `SetFlag` / `WaitFlag` pairs.
   - Flag any presence of `PIPE_ALL` or unwarranted `PipeBarrier` calls as performance bottlenecks.
3. **Roofline & Tile Analysis:**
   - Compute Operational Intensity ($FLOPs / Bytes_{DMA}$).
   - Verify that computation latency on a tile amortizes DMA transfer setup and scalar instruction dispatch.

## Ascend 950 DaVinci v3 additions

- Use the `ascend-950-architecture` skill for 950-specific precision, NDDMA, CV fusion, and BufferID rules. Select 950PR 32/64 or 950DT 36/72 by default; inspect the actual 28/56 or 32/64 bin before choosing block dimensions.
- Size FP32/INT32 accumulator tiles against 262144 B L0C. The 524288 B UB figure is aggregate across two Vector cores; do not assume that any one Vector core can allocate it all.
- For v3, track paired `get_buf(pipe, #id)` / `rel_buf(pipe, #id)` ownership as well as SetFlag/WaitFlag. Treat the exact intrinsic spelling and legal IDs as CANN-version-specific; older hard-event ID restrictions must not be copied blindly to BufferID.
- Prefer NDDMA for supported up-to-5D GM-to-UB layout conversion and the direct Cube L1-to-Vector UB pathway for fused attention/epilogues.
