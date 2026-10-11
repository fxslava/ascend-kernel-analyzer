/*
 * KER-01 (Stage-1, revision 3): MHC pre-norm + Sinkhorn normalisation.
 *
 * Reconstructed arch35 acceptance fixture for the DeepSeek-V4-Flash Stage-1
 * fleet on Ascend 950PR (dav-3510, CANN 9.2).  One 64x128 half tile per
 * iteration:
 *
 *   1. MTE2 stage-in through a double-buffered VECIN queue.
 *   2. SIMD pre-norm on PIPE_V: square, ReduceRepeat row energy, reciprocal
 *      root via the arch35 reduction intrinsics.
 *   3. Sinkhorn iterations: Exp on PIPE_V, then one SIMT VF dispatch per
 *      iteration applies the per-row scale; the dispatch is bracketed by the
 *      mandatory PipeBarrier<PIPE_V> pair that drains the physically shared
 *      Vector ALU around warp mode.
 *   4. Gather builds the output table against a CreateVecIndex base vector.
 *   5. MTE3 stage-out through VECOUT.
 *
 * Audited clean under --chip ascend950pr --warnings-as-errors with no
 * ignore pragmas and no manual reuse-group annotations: the __ubuf__ alias
 * handed to the VF function is tracked from its source tensor's layout.
 */
#include "kernel_operator.h"
#include "simt_api/common_functions.h"

constexpr uint32_t MHC_ROWS = 64;
constexpr uint32_t MHC_COLS = 128;
constexpr uint32_t MHC_ELEMS = MHC_ROWS * MHC_COLS;  /* 8192 half = 16 KiB */
constexpr uint32_t SINKHORN_ITERS = 4;
constexpr uint32_t ROW_TABLE_ELEMS = 8192;

/* Scratch layout inside one VECCALC TBuf (byte offsets, bank-orthogonal:
 * no dual-source pair below differs by a multiple of 256 B). */
constexpr uint32_t OFF_SQ = 0;                        /* fp32 squared, 32 KiB */
constexpr uint32_t OFF_ROW = 32 * 1024 + 96;          /* fp32 row sums       */
constexpr uint32_t OFF_INV = OFF_ROW + 4 * MHC_ROWS + 64; /* half row scale  */
constexpr uint32_t SCRATCH_BYTES = OFF_INV + 2 * MHC_ROWS + 32;

/* SIMT VF: divide every element by its row denominator (fused softmax scale
 * plus the Sinkhorn epsilon floor).  Reads the row table straight from UB
 * through the __ubuf__ alias the kernel hands it; no GM traffic. */
__simt_vf__ __launch_bounds__(512) inline void sinkhorn_row_apply(
    __ubuf__ half *rows, __ubuf__ const half *inv, uint32_t iter)
{
    for (uint32_t i = threadIdx.x; i < MHC_ELEMS; i += blockDim.x) {
        uint32_t r = i / MHC_COLS;
        float scale = static_cast<float>(inv[r]) + 1.0f / static_cast<float>(1 << (10 - iter));
        rows[i] = static_cast<half>(static_cast<float>(rows[i]) / scale);
    }
}

extern "C" __global__ __aicore__ void ker01_mhc_pre_norm_sinkhorn(
    GM_ADDR gmIn, GM_ADDR gmOut, GM_ADDR tiling)
{
    __gm__ const uint32_t *tile = (__gm__ const uint32_t *)(tiling);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQue;
    pipe.InitBuffer(inQue, 2, MHC_ELEMS * sizeof(half));
    pipe.InitBuffer(outQue, 2, MHC_ELEMS * sizeof(half));

    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, SCRATCH_BYTES);

    AscendC::GlobalTensor<half> gmX;
    AscendC::GlobalTensor<half> gmY;
    gmX.SetGlobalBuffer((__gm__ half *)gmIn, MHC_ELEMS);
    gmY.SetGlobalBuffer((__gm__ half *)gmOut, MHC_ELEMS);

    for (uint32_t t = 0; t < tile[0]; ++t) {
        AscendC::LocalTensor<half> x = inQue.DeQue<half>();

        auto sq = scratch.GetWithOffset<float>(MHC_ELEMS, OFF_SQ);
        auto rowSum = scratch.GetWithOffset<float>(MHC_ROWS, OFF_ROW);
        auto rowInv = scratch.GetWithOffset<half>(MHC_ROWS, OFF_INV);

        /* -- pre-norm: row energy and reciprocal root -------------------- */
        AscendC::Mul(sq, sq, sq, MHC_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float, float>(
            rowSum, sq, 0xffffffffffffffffULL, MHC_ROWS, 8, 1, 128, true);
        AscendC::Sqrt(rowSum, rowSum, MHC_ROWS);
        AscendC::Reciprocal(rowInv, rowSum, MHC_ROWS, 1, 1, 1, 1);

        /* -- Sinkhorn: exp, then per-row scaling inside the VF ------------- */
        for (uint32_t iter = 0; iter < SINKHORN_ITERS; ++iter) {
            AscendC::Exp(sq, sq, MHC_ELEMS, 1, 1, 1, 1);
            AscendC::PipeBarrier<PIPE_V>();
            asc_vf_call<sinkhorn_row_apply>(
                dim3(512),
                reinterpret_cast<__ubuf__ half *>(x.GetPhyAddr()),
                reinterpret_cast<__ubuf__ const half *>(rowInv.GetPhyAddr()),
                iter);
            AscendC::PipeBarrier<PIPE_V>();
        }

        /* -- gather the normalised table against a linear index vector ---- */
        auto idx = scratch.GetWithOffset<int32_t>(MHC_ELEMS, OFF_SQ);
        AscendC::CreateVecIndex(idx);
        AscendC::Gather(x, x, idx, 0, MHC_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();

        outQue.EnQue(x);
        AscendC::LocalTensor<half> y = outQue.DeQue<half>();
        AscendC::DataCopy(gmY[t * MHC_ELEMS], y, MHC_ELEMS);
        outQue.FreeTensor(y);
        inQue.FreeTensor(x);
    }
}
