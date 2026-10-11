/*
 * KER-05 (Stage-1, revision 3): MXFP4 expert GEMV, hybrid SIMT/SIMD.
 *
 * Reconstructed arch35 acceptance fixture for the DeepSeek-V4-Flash Stage-1
 * fleet on Ascend 950PR (dav-3510, CANN 9.2).  E2M1 payload packed two
 * values per byte with E8M3 block scales over 32-value groups, dequantised
 * in registers:
 *
 *   1. MTE2 stages the fp16 activation vector (4 KiB) through VECIN; the
 *      packed expert weights never enter UB - SIMT threads pull them
 *      straight from GM through the 32 KiB SIMT DataCache partition.
 *   2. One SIMT VF dispatch per expert row (the fleet's per-row dispatch
 *      idiom): each thread dequantises its MXFP4 slice, multiplies against
 *      the activation via the __ubuf__ alias, and the warp folds the dot
 *      product with asc_shfl_xor butterfly shuffles - register-resident, no
 *      SRAM round trip.
 *   3. SIMD epilogue folds the per-warp partials with the arch35 reduction
 *      intrinsics and gathers against a CreateVecIndex vector; MTE3 stages
 *      the half results out through VECOUT.
 *
 * Static UB footprint: 34944 B (34.1 KiB) - the DataCache-resident weight
 * path is what keeps it far below the 216 KiB arch35 tensor ceiling.
 *
 * Audited clean under --chip ascend950pr --warnings-as-errors with no
 * ignore pragmas and no manual reuse-group annotations.
 */
#include "kernel_operator.h"
#include "simt_api/common_functions.h"

constexpr uint32_t GEMV_K = 2048;          /* activation length            */
constexpr uint32_t GEMV_ROWS = 512;        /* expert rows per tile         */
constexpr uint32_t MXFP4_GROUP = 32;       /* values per E8M3 scale block  */
constexpr uint32_t WARPS_PER_ROW = 4;      /* threads 128, warp slices 32  */

/* Row geometry: packed row is K/2 bytes plus K/32 scale bytes. */
constexpr uint32_t ROW_PACKED_BYTES = GEMV_K / 2;
constexpr uint32_t ROW_SCALE_BYTES = GEMV_K / MXFP4_GROUP;

/* Scratch layout inside one VECCALC TBuf; offsets bank-orthogonal. */
constexpr uint32_t OFF_PART = 0;                            /* fp32, 8 KiB */
constexpr uint32_t OFF_FOLD = WARPS_PER_ROW * GEMV_ROWS * 4 + 96;
constexpr uint32_t OFF_IDX = OFF_FOLD + GEMV_ROWS * 4 + 64;
constexpr uint32_t OFF_SCALE = OFF_IDX + GEMV_ROWS * sizeof(int32_t) + 32;
constexpr uint32_t SCALE_CACHE_BYTES = GEMV_ROWS * 16;      /* u8 preload    */
constexpr uint32_t OFF_BIAS = OFF_SCALE + SCALE_CACHE_BYTES;
constexpr uint32_t ROW_BIAS_ELEMS = 2016;                   /* half, padded  */
/* Static UB total across xQue (2 x 4096), yQue (2 x 1024) and this TBuf is
 * 34944 B (34.1 KiB) - the footprint the PipeSim flow for this kernel is
 * calibrated against. */
constexpr uint32_t GEMV_SCRATCH_BYTES = OFF_BIAS + ROW_BIAS_ELEMS * sizeof(half);

/* SIMT VF: one dispatch per expert row.  Thread t owns K-slice
 * [t*32, t*32+32) of row r; the MXFP4 pair share one packed byte, so two
 * adjacent lanes touch the same DataCache line and coalesce.  The warp
 * folds its slice with a butterfly shuffle and lane 0 writes the partial.
 * The UB scale cache serves the first tile (DataCache warm-up); later tiles
 * stream scales straight from GM. */
__simt_vf__ __launch_bounds__(128) inline void expert_gemv_row(
    __gm__ const uint8_t *packed, __gm__ const uint8_t *scales,
    __ubuf__ const uint8_t *scaleCache, __ubuf__ const half *x,
    __ubuf__ float *partials, uint32_t row)
{
    uint32_t lane = threadIdx.x & 31;
    uint32_t warp = threadIdx.x >> 5;
    uint32_t base = lane * 32;

    float acc = 0.0f;
    for (uint32_t k = 0; k < 32; ++k) {
        uint32_t elem = base + k;
        uint8_t byte = packed[row * ROW_PACKED_BYTES + (elem >> 1)];
        uint32_t code = (byte >> ((elem & 1) * 4)) & 15;
        int32_t mag = code & 7;
        float val = static_cast<float>((code & 8) ? -mag : mag) * 0.25f;
        uint8_t sb = scaleCache[row] ? scaleCache[row]
                                     : scales[row * ROW_SCALE_BYTES + elem / MXFP4_GROUP];
        acc += val * static_cast<float>(x[elem]) *
               static_cast<float>(1 << ((sb >> 3) & 7)) * ((sb & 7) + 8) * 0.125f;
    }
    for (uint32_t off = 16; off > 0; off >>= 1) {
        acc += asc_shfl_xor(acc, off);
    }
    if (lane == 0) {
        partials[row * WARPS_PER_ROW + warp] = acc;
    }
}

extern "C" __global__ __aicore__ void ker05_mxfp4_expert_gemv(
    GM_ADDR gmWeights, GM_ADDR gmScales, GM_ADDR gmX, GM_ADDR gmY,
    GM_ADDR tiling)
{
    __gm__ const uint32_t *tile = (__gm__ const uint32_t *)(tiling);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> xQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> yQue;
    pipe.InitBuffer(xQue, 2, GEMV_K * sizeof(half));
    pipe.InitBuffer(yQue, 2, GEMV_ROWS * sizeof(half));

    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, GEMV_SCRATCH_BYTES);

    AscendC::GlobalTensor<half> gmXVec;
    AscendC::GlobalTensor<half> gmYVec;
    gmXVec.SetGlobalBuffer((__gm__ half *)gmX, GEMV_K);
    gmYVec.SetGlobalBuffer((__gm__ half *)gmY, GEMV_ROWS);

    for (uint32_t t = 0; t < tile[0]; ++t) {
        AscendC::LocalTensor<half> x = xQue.DeQue<half>();

        auto partials = scratch.GetWithOffset<float>(
            WARPS_PER_ROW * GEMV_ROWS, OFF_PART);
        auto folded = scratch.GetWithOffset<float>(GEMV_ROWS, OFF_FOLD);
        auto idx = scratch.GetWithOffset<int32_t>(GEMV_ROWS, OFF_IDX);
        auto scaleCache = scratch.GetWithOffset<uint8_t>(GEMV_ROWS, OFF_SCALE);
        auto rowBias = scratch.GetWithOffset<half>(GEMV_ROWS, OFF_BIAS);
        AscendC::Duplicate(scaleCache, 0, GEMV_ROWS);
        AscendC::Duplicate(rowBias, static_cast<half>(0), GEMV_ROWS);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::DataCopy(x, gmXVec[0], GEMV_K);

        /* Per-row VF dispatch: 512 rows, one mode switch each.  The
         * barriers around every dispatch drain the shared Vector ALU. */
        for (uint32_t row = 0; row < GEMV_ROWS; ++row) {
            AscendC::PipeBarrier<PIPE_V>();
            asc_vf_call<expert_gemv_row>(
                dim3(128),
                reinterpret_cast<__gm__ const uint8_t *>(gmWeights),
                reinterpret_cast<__gm__ const uint8_t *>(gmScales),
                reinterpret_cast<__ubuf__ const uint8_t *>(scaleCache.GetPhyAddr()),
                reinterpret_cast<__ubuf__ const half *>(x.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(partials.GetPhyAddr()),
                row);
            AscendC::PipeBarrier<PIPE_V>();
        }

        /* SIMD epilogue: fold the four warp partials per row, then add the
         * routed bias before the half cast. */
        AscendC::ReduceDataBlock<AscendC::ReduceType::SUM, float, float>(
            folded, partials, 0xffffffffffffffffULL, GEMV_ROWS, 1, 4, 4);
        AscendC::CreateVecIndex(idx);
        AscendC::Gather(folded, folded, idx, 0, GEMV_ROWS);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<half> y = yQue.AllocTensor<half>();
        AscendC::Cast(y, folded, AscendC::RoundMode::CAST_NONE, GEMV_ROWS);
        AscendC::Add(y, y, rowBias, GEMV_ROWS);
        AscendC::PipeBarrier<PIPE_V>();
        yQue.EnQue(y);
        AscendC::LocalTensor<half> out = yQue.DeQue<half>();
        AscendC::DataCopy(gmYVec[t * GEMV_ROWS], out, GEMV_ROWS);
        yQue.FreeTensor(out);
        xQue.FreeTensor(x);
    }
}
