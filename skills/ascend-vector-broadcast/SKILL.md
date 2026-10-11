---
name: ascend-vector-broadcast
description: Architecture rules, Tiling constraints, and kernel implementation patterns for Vector Broadcast operations on Huawei Ascend NPUs (DaVinci/AIV). Use when implementing elementwise operators with unequal input shapes, managing DataCopyPad alignment, computing Broadcast buffer budgets in UB, or auditing shape expansion hazards.
---

# Ascend Vector Broadcast & Dimension Expansion

Technical guidelines and hardware patterns for implementing element-wise operations with broadcasted inputs (e.g., $(M, N) \oplus (M, 1)$ or $(M, N) \oplus (1, N)$) on Huawei Ascend AI Cores (Vector/AIV).

## Core Architectural Constraints

1. **Strict Operator Shape Homogeneity:**
   - Standard element-wise vector instructions (`Add`, `Sub`, `Mul`) in Ascend C require all operands to have strictly identical physical layouts and lengths in the Unified Buffer (UB).
   - In-place broadcast inside compute instructions is not supported; expansion requires an intermediate expansion step via `AscendC::BroadCast` or MTE layout manipulation.

2. **UB Memory Budget & Intermediate Tensor Expansion:**
   - Broadcast expansion produces a full-sized tensor from a compressed tensor.
   - For an expansion coefficient $\text{coef}$, the shorter input occupies $L / \text{coef}$ elements, but its broadcast output requires a full tile $L$.
   - **UB Budgeting:** The kernel must allocate an additional scratchpad buffer (`tmpBuf`) in UB of size $L \times \text{sizeof(dataType)}$ to store the expanded tensor before issuing ALU instructions.

3. **32-Byte Sub-Tile Alignment & DataCopyPad:**
   - Because the shorter tensor's dimension is scaled down by $\text{coef}$, addresses and byte lengths ($L / \text{coef}$) frequently become non-multiples of the 32-byte DaVinci hardware block.
   - Standard `DataCopy` triggers alignment exceptions or data corruption.
   - Kernels must strictly employ `DataCopyPad` with `DataCopyExtParams` and `DataCopyPadExtParams` to safely load and store unaligned slices from Global Memory.

---

## Tiling Strategy (Host-Side)

### 1. Partitioning by Shorter Axis Length

To maintain proportional work across multi-core dispatch and ping-pong pipelines, tiling must partition the **shorter input length** (`shorterAxisLen`), scaling the resulting blocks by `coef` for the primary tensor:

$$\text{coef} = \frac{\text{totalLength}}{\text{shorterAxisLen}}$$

- **Even core split:** $\text{blockLength}_{\text{short}} = \text{shorterAxisLen} / \text{BLOCK\_DIM}$
- **Full tensor length:** $\text{blockLength} = \text{blockLength}_{\text{short}} \times \text{coef}$

### 2. Multi-Constraint UB Alignment

Tile sizes must be aligned simultaneously to the 32-byte hardware block (`alignNum`), the broadcast coefficient (`coef`), and the double-buffering factor (`BUFFER_NUM = 2`):

$$\text{alignStride} = \text{coef} \times \text{BUFFER\_NUM}$$
$$\text{ubBlockAligned} = \left\lfloor \frac{\text{UB\_BLOCK\_NUM} \times \text{alignNum}}{\text{alignStride}} \right\rfloor \times \text{alignStride}$$

This guarantees that:
- Every sub-tile length divided by `coef` represents an exact integer count of elements.
- Ping-pong alternating slices map cleanly into physical UB memory boundaries.

---

## Kernel Implementation Patterns (Device)

### 1. Buffer Initialization & Dynamic Sizing

```cpp
// Full tile queues for primary input and output
pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dataType));
pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dataType));

// Compact queue for broadcasted input (proportional to tileLength / coef)
pipe.InitBuffer(inQueueY, BUFFER_NUM, (this->tileLength / this->coef) * sizeof(dataType));

// Mandatory scratchpad buffer for holding post-broadcast expanded data
pipe.InitBuffer(broadcastTmpBuf, this->tileLength * sizeof(dataType));
```

### 2. Robust Memory Movement via DataCopyPad

Always supply explicit byte lengths via `DataCopyExtParams` to prevent 32-byte boundary overflows:

```cpp
AscendC::DataCopyExtParams copyXParams = {1, (uint32_t)(this->tileLength * sizeof(dataType)), 0, 0, 0};
AscendC::DataCopyExtParams copyYParams = {1, (uint32_t)(this->tileLength * sizeof(dataType) / this->coef), 0, 0, 0};
AscendC::DataCopyPadExtParams<dataType> padParams = {false, 0, 0, 0};

// Ingest shorter input with scaled indexing
AscendC::DataCopyPad<dataType>(
    yLocal,
    yGm[progress * this->tileLength / this->coef],
    copyYParams,
    padParams
);
```

### 3. Compute Phase & Shape Synthesis

```cpp
AscendC::LocalTensor<dataType> xLocal = inQueueX.DeQue<dataType>();
AscendC::LocalTensor<dataType> yLocal = inQueueY.DeQue<dataType>();
AscendC::LocalTensor<dataType> zLocal = outQueueZ.AllocTensor<dataType>();
AscendC::LocalTensor<dataType> tmpLocal = broadcastTmpBuf.Get<dataType>();

// Define 2D expansion parameters: {M, 1} -> {M, coef}
uint32_t srcShape[] = {this->tileLength / this->coef, 1};
uint32_t dstShape[] = {this->tileLength / this->coef, this->coef};

// Step 1: Expand dimension along axis 1
AscendC::BroadCast<dataType, 2, 1>(tmpLocal, yLocal, dstShape, srcShape);

// Step 2: Symmetrical vector ALU execution
AscendC::Add(zLocal, xLocal, tmpLocal, this->tileLength);

outQueueZ.EnQue(zLocal);
inQueueX.FreeTensor(xLocal);
inQueueY.FreeTensor(yLocal);
```

---

## Static Analyzer Verification Rules (AKA Extensions)

When validating broadcast kernels:

1. **AKA1006: Missing Broadcast Temporary Buffer:**
   * Detect element-wise operations on disparate logical shapes without an intermediate buffer allocation in UB.

2. **AKA1007: Unchecked DataCopy on Scaled Dimension:**
   * Flag raw `DataCopy` calls on addresses indexed with `/ coef` that are not provably aligned to 32 bytes; recommend `DataCopyPad`.

3. **AKA1008: Coefficient Stride Alignment Violation:**
   * Check that `tileLength` is an exact multiple of `coef`. A remainder causes partial-element memory corruption during `BroadCast`.
