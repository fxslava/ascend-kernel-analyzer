---
name: ascend-950-architecture
description: Ascend 950PR and 950DT DaVinci v3 hardware facts and kernel optimization rules. Use when sizing Ascend 950 Cube/Vector tiles, implementing MXFP4/MXFP8/HiF8 pipelines, NDDMA or CV fusion, or auditing BufferID synchronization.
---

# Ascend 950PR / 950DT DaVinci v3 kernel development

Use `D:/Projects/PipeSim/ASCEND_950_ARCHITECTURE.md` and the 2026 `昇腾950 NPU架构白皮书.pdf` as the local architecture reference. Treat actual CANN intrinsic signatures and measured throughput as release-specific; check installed headers and benchmarks before emitting production code.

## Choose the bin first

- 950PR default: 32 Cube / 64 Vector, 128 GB package memory, 1.6 TB/s HBM-class bandwidth; reduced 28/56, 112 GB, 1.4 TB/s, 112 MB L2 bin also exists.
- 950DT default: 36 Cube / 72 Vector, 144 GB, 4.0 TB/s; 32/64 and 28/56 core bins and a 96 GB memory variant are listed without an exact pairing.
- Every AI subsystem has one Cube plus two Vector cores. Full design is 36/72. Use the enabled core count from the actual device for launch tiling.
- The package has two AI Dies and two IO Dies in a UMA address space. Allow die-local L2 affinity and cross-die transfer in performance estimates.

## Capacity and layout rules

- L1: 512 KiB per AI subsystem. L0A/L0B: 64 KiB each, with Zz/Nz fractal layouts. L0C: **256 KiB** of FP32/INT32 accumulator, i.e. 4 bytes per output element. Check padded M × N × 4 × accumulator queue depth against 262144 bytes.
- UB: **512 KiB aggregate** per AI subsystem across two Vector cores, notionally UB0 and UB1 at 256 KiB each. Do not grant 512 KiB to each Vector core. Eight banks use 32 B interleave in the local kernel model; arrange ping-pong offsets to avoid bank conflicts.
- L2 is a coherent 128 MB UMA cache in full bins, with 512 B lines / four 128 B sectors. Use allocate/non-allocate L2 hints for reuse and CMO Prefetch, Writeback, Invalidate, Flush where the runtime exposes them.
- Align and pad every packed MXFP4 transfer in **bytes**, not elements. Its storage is 0.5 B/element before scale metadata; FP8/MXFP8/HiF8 are 1 B/element.

## Precision pipeline

1. Keep FP16/BF16 as a correctness baseline. The analytical model is 4096 MAC/cycle; FP8/MXFP8/HiF8 are 2x and MXFP4 is 4x at matched frequency. These are relative ratios, not measured instruction rates.
2. For MXFP4/MXFP8, budget scale blocks, packing/unpacking, and any format conversion alongside operands. Stage packed A/B with the correct widened K fractal and avoid L0A/L0B overflow.
3. For HiF8, preserve its tapered exponent behavior (38 exponent positions over −22..+15). It has no separate 8-bit MX scale. Verify saturation, NaN/Inf, and accumulated error against the application's precision criteria.
4. Keep accumulation in FP32/INT32 L0C. Use inline L0C-to-UB quantization/layout conversion when supported; avoid an extra GM materialization before a Vector epilogue.

## NDDMA, CV fusion, and synchronization

- Use NDDMA for supported up-to-5D GM-to-UB permutations, NCHW/NHWC rearrangement, ND-to-NZ-like loads, and regular irregular strides. It coalesces fine reads into 128 B accesses. Validate the exact CANN descriptor and target alignment; 128 B coalescing does not imply 128 B/cycle throughput.
- For FlashAttention and online Softmax, stream Cube data over the direct Cube L1 ↔ Vector UB CV path, normalize/update statistics on Vector, then feed the next Cube tile. Preserve producer/consumer ownership until reads complete. Use on-path precision/layout conversion when the selected kernel API supports it.
- Prefer BufferID ownership notation `get_buf(pipe, #id)` / `rel_buf(pipe, #id)` on DaVinci v3 where the installed CANN release exposes it. Acquire before writing or consuming a buffer, release after the last dependent operation; pair every acquisition on all control paths. These spellings describe whitepaper semantics, not guaranteed C++ signatures. If unsupported, use correctly paired targeted SetFlag/WaitFlag events.
- Avoid global pipeline barriers. Benchmark MTE/NDDMA/CV rates rather than assuming the PipeSim profile's 128 B/cycle modeling width.
