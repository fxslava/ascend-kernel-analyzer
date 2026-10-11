---
name: ascend-kernel-analyzer-rules
description: Formal verification specifications, diagnostic rule catalog (AKAxxxx), and static check algorithms for Huawei Ascend C kernels. Use when extending ascend-kernel-analyzer, auditing DaVinci SRAM layouts, detecting pipeline deadlocks via Marked Graphs, checking bank conflicts, or validating tail invariants.
---

# Ascend Kernel Analyzer Verification Rules

Technical rules, invariants, and diagnostic definitions for static analysis of Ascend C kernels targeting DaVinci architecture (Ascend 910B, 910C, 351x).

## Diagnostic Code Catalog (AKAxxxx)

### Memory & SRAM Layout (AKA1000 - AKA1999)

- **AKA1001 (FATAL): SRAM capacity overflow.**
  - Condition: $\sum \text{live\_tensors} > \text{Capacity}_{\text{domain}}$ or symbolic upper bound exceeds physical limit ($\text{offset} + \text{size} > \text{Capacity}$).
  - Hardware Limits: UB (192 KB / 248 KB depending on chip), L1 (512 KB), L0A (64 KB), L0B (64 KB), L0C (128 KB / 256 KB).
- **AKA1002 (FATAL): Base address alignment violation.**
  - Condition: $\text{BaseOffset} \pmod{32} \neq 0$.
  - All SRAM base addresses must align to 32-byte DaVinci block boundaries.
- **AKA1003 (FATAL): Buffer aliasing / address collision.**
  - Condition: $\text{Interval}(T_A) \cap \text{Interval}(T_B) \neq \emptyset$ while both tensors are concurrently live within the loop pipeline.
  - Exclusion: Permitted only when explicitly annotated with `// @ascend-reuse-group: group=<id> names=A,B`.
- **AKA1004 (FATAL): Memory domain mismatch.**
  - Condition: Instruction operand located in incompatible physical memory.
  - Examples: `DataCopy` GM -> L0B (illegal direct route on 910B), `Mmad` op1 not in L0A, `Add` operands outside UB.
- **AKA1005 (FATAL): Allocation size alignment violation.**
  - Condition: $\text{TensorByteSize} \pmod{32} \neq 0$. Unpadded lengths spill into adjacent memory during block DMA/ALU instructions.

### Synchronization & Deadlock Analysis (AKA2000 - AKA2999)

- **AKA2001 (FATAL): Unpaired synchronization flag.**
  - Condition: `SetFlag<Route>(id)` without corresponding downstream `WaitFlag<Route>(id)` or vice versa.
- **AKA2002 (FATAL): Circular wait / pipeline deadlock.**
  - Condition: Zero-token cycle detected in the Marked Graph representation of the inter-pipeline dependency graph.
- **AKA2003 (FATAL): Reserved EVENT_ID used.**
  - Condition: `EVENT_ID4` through `EVENT_ID7` on Ascend 950PR / arch35; only IDs 0 through 3 are available. Use the target profile for other chips.
- **AKA2004 (FATAL): Inter-pipe route mismatch.**
  - Condition: Producing and consuming pipelines do not match hardware event route (e.g., `MTE2_V` invoked on `PIPE_MTE3`).
- **AKA2005 (FATAL): Unprimed loop-carried WaitFlag.**
  - Condition: Loop-carried dependency has no prologue `SetFlag` before loop entry, stalling the first iteration.

### Performance & Microarchitectural Hazards (AKA3000 - AKA3999)

- **AKA3001 (WARNING): PipeBarrier(PIPE_ALL) antipattern.**
  - Condition: Full pipeline flush serializing MTE2, Vector, MTE3, and Cube, discarding double-buffering overlap.
- **AKA3002 (INFO): Decidability boundary / unresolved constant.**
  - Condition: Non-constant symbolic offset cannot be bounded by Z3; excluded from proofs to prevent false positives.
- **AKA3003 (WARNING): UB Bank Conflict.**
  - Condition: Dual vector operands $A$ and $B$ satisfy $\left\vert{}\frac{\text{Offset}_A - \text{Offset}_B}{32}\right\vert{} \pmod{N_{\text{banks}}} = 0$.
- **AKA3004 (WARNING): Ragged Tail Overlap Non-Idempotency.**
  - Condition: Tail offset shift $(\text{progress}-2)\cdot L + L_{\text{tail}}$ used in non-idempotent operations (reductions/in-place stores).
- **AKA3005 (WARNING): Excessive VF Switching Frequency.**
  - Condition: `asc_call_vf` mode alternation invoked inside fine-grained tile loops ($< 512$ elements per switch).

---

## Formal Analysis Algorithms

### 1. Marked Graph Pipeline Deadlock Detector

1. Construct directed graph $G = (V, E)$, where $V$ represents sync operations (`SetFlag`, `WaitFlag`) and compute dispatches.
2. Assign initial tokens:
   - Program-order edges: $w(e) = 0$.
   - Loop back-edges: $w(e) = 1$.
   - Sync edges (`Set` $\to$ `Wait`): $w(e) = \text{ProloguePrimingCount}$.
3. Deadlock invariant: A marked graph deadlocks if and only if there exists a directed cycle $C$ such that $\sum_{e \in C} w(e) = 0$.
4. Root cause identification: Flag unprimed loop-carried waits directly rather than reporting all cascading downstream cycles.

### 2. UB Bank Conflict Verification

For vector instructions taking two or more input tensors (`Add`, `Mul`, etc.):
- Retrieve base byte offsets $O_1, O_2$.
- Compute bank index: $\text{Bank}(O) = \lfloor O / 32 \rfloor \pmod{16}$ (for 16-bank / 4KB-stride architectures).
- Emit AKA3003 if $\text{Bank}(O_1) == \text{Bank}(O_2)$ and instruction latency is degraded.

### 3. Tail Overlap Verification

- Detect tail index calculations: $O_{\text{tail}} = (\text{progress} - 2) \cdot \text{tileLength} + \text{lastTileLength}$.
- Check consumer operation:
  - If pure elementwise assignment (`DataCopy`, `Muls`): mark ACCEPTED.
  - If reduction or accumulation: emit AKA3004 (overlapping memory re-accumulation hazard).
