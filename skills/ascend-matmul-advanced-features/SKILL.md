---
name: ascend-matmul-advanced-features
description: Architecture rules, Tiling configurations, and kernel programming patterns for advanced Matmul features on Huawei Ascend NPUs (910B, 910C, 950PR, 950DT, Cube/AIC, MIX mode). Use when implementing multi-core aligned/unaligned split-M/N/K Matmul, MIX-mode asynchronous Iterate/IterateAll with workspace, hardware Fixpipe-based on-the-fly quantization/dequantization (Per-Tensor and Per-Channel), or handling ragged tail matrices with SetTail.
---

# Ascend Advanced Matmul Engineering

Engineering standards for high-performance Matrix Multiplication (`AscendC::Matmul`) exploiting advanced hardware features across AIC (Cube) and AIV (Vector) on Huawei Ascend NPUs (Ascend 910B, 910C, Atlas A2/A3).

---

## 1. Multi-Core Matrix Partitioning & Tail Tiling

### A. Partitioning Paradigms

1. **Split-M & Split-N (No K-Split):**
   - Left matrix $A$ split along $M$ ($\text{singleCoreM}$); right matrix $B$ split along $N$ ($\text{singleCoreN}$).
   - Single core computes $(\text{singleCoreM} \times K) \times (K \times \text{singleCoreN}) \to \text{singleCoreM} \times \text{singleCoreN}$.
   - Independent outputs; no inter-core reduction or global accumulation needed.
2. **Split-K (3D Slicing):**
   - Outer loop partitions $K$ into slices of size $\text{singleCoreK}$.
   - Cores calculate partial accumulation tiles: $C_{\text{block}} = \sum A_i \times B_i$.
   - Requires global workspace accumulation or inter-core barrier aggregation before writing final output.

### B. Core Count Configuration Invariants

- `MultiCoreMatmulTiling::SetDim(coreNum)`: Sets the count of *available* Cube cores to calculate tiling parameters on the Host.
- `context->SetBlockDim(actualUsedCores)`: Sets the exact number of blocks launched on the Device. In pure Cube mode, $\text{actualUsedCores} \le \text{GetCoreNumAic()}$.

### C. Unaligned Multi-Core Tiling (Ragged Tails)

When dimensions are not evenly divisible by core slice lengths:

- **Host (Tiling):** Retains base uniform `singleCoreM`, `singleCoreN`, `singleCoreK`.
- **Device (Kernel):** Computes residual bounds:
  $$\text{tailM} = \min(\text{M} - \text{mIdx} \times \text{singleCoreM}, \text{singleCoreM})$$
  $$\text{tailN} = \min(\text{N} - \text{nIdx} \times \text{singleCoreN}, \text{singleCoreN})$$
- **Mandatory Invocation:** If $\text{tailM} < \text{singleCoreM}$ or $\text{tailN} < \text{singleCoreN}$, call `matmulObj.SetTail(tailM, tailN)` **strictly before** calling `Iterate()` or `IterateAll()`.

```cpp
int tailM = tiling.M - mCoreIndex * tiling.singleCoreM;
tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
int tailN = tiling.N - nCoreIndex * tiling.singleCoreN;
tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;

if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN) {
    matmulObj.SetTail(tailM, tailN);
}
```

---

## 2. MIX-Mode Asynchronous Matmul (AIC/AIV Overlap)

In mixed kernels (AIC Cube + AIV Vector), Cube Matmul operations can run asynchronously while Vector ALUs perform element-wise logic, normalization, or memory staging.

### A. Asynchronous `Iterate` & `GetTensorC`

* **Execution Model:** Non-blocking iteration with template parameter `Iterate<false>()`.
* **Workspace Requirement:** Requires dedicated scratchpad memory registered via `SetWorkspace(workspace, size)` prior to `Iterate`. The size must be at least $\text{singleCoreM} \times \text{singleCoreN} \times \text{sizeof(dtype)}$.
* **Output Formats:** NZ format for `TPosition::VECIN`; ND or NZ for `TPosition::GM`.

```cpp
mm.SetWorkspace(workspaceGm, singleCoreM * singleCoreN * sizeof(float));

// Launch non-blocking Cube iteration
mm.template Iterate<false>();

// Concurrently execute Vector operations on AIV
// ... vector compute / prefetching ...

// Drain slices asynchronously into target buffer
auto mIter = Ceil(singleCoreM, baseM);
auto nIter = Ceil(singleCoreN, baseN);
for (int i = 0; i < mIter * nIter; ++i) {
    mm.GetTensorC<false>(gm_c);
}
```

### B. Asynchronous `IterateAll` & Pipeline Barrier

* Compute full matrix block asynchronously: `mm.template IterateAll<false>(workspaceGm, 0, false, true);`.
* Synchronize before consuming output: call `mm.WaitIterateAll()` to block until Cube calculation finishes and memory flush to GM completes.

---

## 3. On-The-Fly Fixpipe Quantization & Dequantization

Ascend 910B and 910C support hardware-accelerated quantization/dequantization in the **Fixpipe unit** during data drain from $CO1$ (L0C accumulator) to Global Memory ($GM$), eliminating separate Vector ALU passes.

### A. Supported Formats & Transitions

| Input Matrix A | Input Matrix B | Output Matrix C | Fixpipe Transformation | Target Hardware |
| --- | --- | --- | --- | --- |
| `half` / `bfloat16_t` | `half` / `bfloat16_t` | `int8_t` | Quantization | Ascend 910B / 910C |
| `int8_t` | `int8_t` | `half` | Dequantization | Ascend 910B / 910C |
| `int4b_t` | `int4b_t` | `half` | Dequantization | Ascend 910B / 910C |
| `int8_t` | `int8_t` | `int8_t` | Re-quantization | Ascend 910B / 910C |

### B. Quantization Granularity & Modes

| Mode | Tiling Configuration | Kernel Parameter API | Parameter Shape |
| --- | --- | --- | --- |
| **Per-Tensor** | `tiling.SetDequantType(DequantType::SCALAR)` | `mm.SetQuantScalar(scaleUint64)` | $[1]$ |
| **Per-Channel** | `tiling.SetDequantType(DequantType::TENSOR)` | `mm.SetQuantVector(gmQuantTensor)` | $[N]$ |

### C. Parameter Passing Invariants

* `SetQuantScalar` takes a 64-bit unsigned integer containing the raw bitcast of the floating-point scaling factor:

```cpp
float scale = 0.1f;
uint64_t quantVal = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&scale));
mm.SetQuantScalar(quantVal);
```

* Both `SetQuantScalar` and `SetQuantVector` **must be called strictly before** `Iterate` or `IterateAll`.
* If INF/NAN propagation is required for `int8`/`int4` $\to$ `half`, Fixpipe cannot be used; output must be staged as `int32_t` into `TPosition::VECIN` and converted via Vector API `AscendDequant`.

---

## 4. Verification & Static Analyzer Invariants

1. **AKA4001 (FATAL): Missing SetTail in Unaligned Matmul:**
   * Detect non-divisible $M, N, K$ dimensions in multi-core dispatch where `Iterate`/`IterateAll` is invoked without a preceding `SetTail` call on boundary cores.

2. **AKA4002 (FATAL): Unregistered Asynchronous Workspace:**
   * Flag invocations of `Iterate<false>()` or `IterateAll<false>()` where `SetWorkspace` was not called beforehand with verified size $\ge \text{singleCoreM} \times \text{singleCoreN}$.

3. **AKA4003 (FATAL): Missing WaitIterateAll Barrier:**
   * Detect direct Vector reads or DMA transfers from the output GM destination without an intermediate `WaitIterateAll()` call following asynchronous `IterateAll<false>()`.

4. **AKA4004 (FATAL): Quantization Mode Configuration Mismatch:**
   * Detect discrepancies between Tiling (`DequantType::SCALAR` vs `DequantType::TENSOR`) and Kernel (`SetQuantScalar` vs `SetQuantVector`).

## 5. Ascend 950 DaVinci v3 matmul

- 950PR default has 32 Cube / 64 Vector; 950DT default has 36 Cube / 72 Vector. Both preserve **one Cube to two Vector cores** per AI subsystem. Use the actual enabled bin for `SetDim` and block dispatch.
- Budget padded M × N × 4 bytes × L0C depth against **256 KiB L0C**. L1 is 512 KiB, L0A/L0B are 64 KiB each, and the **512 KiB UB belongs to the two-Vector subsystem** (256 KiB per Vector), not to each Vector independently.
- For MXFP4, MXFP8, HiF8 and FP8, use the appropriate packed K fractal, scale/metadata storage, and FP32 accumulation. The analytical Cube ratios are 4x FP16 for MXFP4 and 2x for the 8-bit types; validate numeric accuracy and installed CANN support before enabling a path.
- Use the direct Cube L1-to-Vector UB CV path for fused online Softmax/FlashAttention and inline L0C-to-UB conversion for the epilogue when supported. Use NDDMA for supported up-to-5D GM-to-UB reorder rather than hand-coded scalar address loops.
- BufferID `get_buf(pipe, #id)` / `rel_buf(pipe, #id)` expresses buffer ownership on v3. Pair acquisition and release around each live tile, including asynchronous `Iterate` and `GetTensorC`; check the target SDK for callable signatures. See `ascend-950-architecture` for full guidance.
