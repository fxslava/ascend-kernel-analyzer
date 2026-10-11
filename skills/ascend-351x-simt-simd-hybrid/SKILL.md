---
name: ascend-351x-simt-simd-hybrid
description: Programming guidelines and optimization patterns for DaVinci v3 / 351x isomorphic SIMD and SIMT architectures. Use when implementing hybrid kernels, unpacking 4-bit nibbles or quantized lattices, designing online softmax, and managing shared Vector ALU resources and Unified Buffer DataCache.
---

For Ascend 950PR / arch35 KER-01..KER-05, apply the sibling ascend-kernel-verification-protocol skill before running any verification command. Its static audit and device compilation boundary overrides broader simulation examples here.

# Ascend 351x SIMD/SIMT Hybrid Kernel Engineering

Guidelines for implementing isomorphic SIMD/SIMT compute kernels on Huawei Ascend 351x (AIV cores, DaVinci v3).

## Hardware Execution Model

1. **Vector Function (VF) Scheduling Unit:**
   - Kernels define device entrypoint: `__global__ __aicore__ void KernelEntry(...)`.
   - Host launches kernel with explicit dynamic UB size: `<<<blocks, nullptr, stream, ubuf_dynamic_size>>>`.
   - Specialized functions:
     - SIMT VF: `__simt_vf__ LAUNCH_BOUND(max_threads) inline void MySimtVF(...)`
     - SIMD VF: `__simd_vf__ inline void MySimdVF(...)`
     - Callee functions: Marked with `__simt_callee__` or `__simd_callee__`.
   - Invocation via dispatcher: `asc_call_vf<FunctionName>(args...)`.

2. **Mutual Exclusion & Pipeline Drain:**
   - Single AIV core executes strictly in-order: either SIMT or SIMD at any given moment.
   - Vector ALU is physically shared. Mode transition flushes and drains active vector pipelines (~30-80 cycles).
   - Execution context and UB Data Cache tags are retained across VF switches.
   - **Optimization Rule:** Never alternate VFs on sub-tile granularity (<512 elements). Batch data transformations inside SIMT, then compute math in SIMD.

3. **Unified Buffer (UB) 256 KB Partitioning:**
   - `[Static Memory]` + `[Dynamic Memory]` + `[Reserved 8 KB]` + `[Data Cache >= 32 KB]` = 256 KB.
   - Max allowable dynamic memory: $216\text{ KB} - \text{StaticMemory}$.
   - Allocation of Data Cache below 32 KB triggers compiler and hardware check faults.

---

## High-Performance SIMT Patterns

### 1. Direct-to-Register Nibble / Lattice Unpacking (UB Bypass)

- **Problem in SIMD:** Unpacking 4-bit weights/KV-cache requires DMA to UB, vector bit shifts, bitwise masks, and type conversions, consuming up to 50% UB capacity with transient raw data.
- **SIMT Solution:**
  - Threads read packed bytes directly from GM via Data Cache:

    ```cpp
    __simt_vf__ LAUNCH_BOUND(1024) inline void UnpackNibbles_SIMT(
        __gm__ uint8_t* packed_gm,
        __ubuf__ half* unpacked_ub,
        uint32_t total_elements)
    {
        uint32_t idx = asc_get_block_idx() * blockDim.x + threadIdx.x;
        if (idx >= total_elements) return;

        uint8_t byte = packed_gm[idx >> 1];
        uint8_t nibble = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
        unpacked_ub[threadIdx.x] = static_cast<half>(nibble);
    }
    ```

  - Unpacks on-the-fly straight into registers, writing clean FP16/BF16 tiles directly into UB ready for Cube/SIMD consumption.

### 2. Tiled Online Softmax with Running Max/Sum

- **Problem in SIMD:** Calculating Softmax across long context lengths requires repeated reductions and intermediate rescaling across tiles, incurring massive intra-pipe barrier overhead (`PipeBarrier<PIPE_V>()`).
- **SIMT Solution:**
  - Execute row statistics update inside SIMT VF using register shuffle/warp tree reduction:
    $$m_{\text{new}} = \max(m_{\text{old}}, \text{row\_max}), \quad l_{\text{new}} = l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum e^{x_i - m_{\text{new}}}$$
  - Compute elementwise exponentiation and scaling back in SIMD VF.

### 3. Irregular Routing & Top-K Sampling

- Use SIMT VF for dynamic MoE token gating, Bitonic sorting in thread registers, and Paged Attention block-table pointer chasing before issuing dense contiguous GEMM to Cube.
