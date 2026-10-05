/*
 * naive_aiv_fp4_to_fp16.cpp -- naive AIV dequantisation: FP4 (e2m1) -> FP16.
 *
 * The plain vector shape every dequant kernel starts from: one VECIN queue
 * streams packed FP4 tiles in, a VECCALC pair splits the nibbles, a VECOUT
 * queue streams FP16 tiles out.  No flags anywhere: the queues carry the
 * pipeline handshakes, so the whole kernel is straight-line per tile.
 *
 * Written against the CANN 8.5 Ascend C API surface (TPipe / TQue / TBuf,
 * AllocTensor/EnQue/DeQue, And/ShiftRight/Mul/Cast) so the BiSheng extraction
 * frontend can typecheck it against the real headers, and the tree-sitter
 * frontend can still walk it unaided.
 *
 * The analyzer should report zero findings on this file.
 *
 * @ascend-chip: ascend910b
 * @ascend-expect:
 * @ascend-expect-fatal: 0
 */
#include "kernel_operator.h"

namespace {
constexpr uint32_t TILE_ELEMS = 512;                  /* fp4 values per tile  */
constexpr uint32_t TILE_BYTES = TILE_ELEMS / 2;       /* 256 B packed fp4     */
constexpr uint32_t OUT_BYTES = TILE_ELEMS * sizeof(half); /* 1 KiB fp16       */
constexpr uint32_t SCRATCH_ELEMS = TILE_ELEMS;        /* int16 intermediates  */
constexpr uint32_t TILE_COUNT = 8;

/* One extra 32-byte block on the lo scratch keeps the hi scratch in a
 * different UB bank: dual-operand vector reads (the Or below) need lo and
 * hi one bank apart, and the bump layout places buffers back to back. */
constexpr uint32_t BANK_PAD = 32;

/* Linear-lut stand-in for the real e2m1 decode: value = code * 0.25. */
constexpr int16_t NIBBLE_MASK = 0x000F;
constexpr int16_t SHIFT_BY = 4;
constexpr half FP4_STEP = 0.25;
}  // namespace

extern "C" __global__ __aicore__ void naive_aiv_fp4_to_fp16(
    __gm__ uint8_t* src, __gm__ half* dst)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> loScratch;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hiScratch;
    AscendC::TBuf<AscendC::TPosition::VECCALC> codesScratch;
    pipe.InitBuffer(inQue, 2, TILE_BYTES);
    pipe.InitBuffer(outQue, 2, OUT_BYTES);
    pipe.InitBuffer(loScratch, SCRATCH_ELEMS * sizeof(int16_t) + BANK_PAD);
    pipe.InitBuffer(hiScratch, SCRATCH_ELEMS * sizeof(int16_t));
    pipe.InitBuffer(codesScratch, SCRATCH_ELEMS * sizeof(int16_t));

    AscendC::LocalTensor<int16_t> lo = loScratch.Get<int16_t>();
    AscendC::LocalTensor<int16_t> hi = hiScratch.Get<int16_t>();
    AscendC::LocalTensor<int16_t> codes = codesScratch.Get<int16_t>();

    for (uint32_t t = 0; t < TILE_COUNT; ++t) {
        auto packed = inQue.AllocTensor<uint8_t>();
        AscendC::DataCopy(packed, src + t * TILE_BYTES, TILE_BYTES);
        inQue.EnQue(packed);

        AscendC::LocalTensor<uint8_t> in = inQue.DeQue<uint8_t>();
        AscendC::Cast(codes, in, AscendC::RoundMode::CAST_NONE, TILE_ELEMS);

        /* even values: low nibble of each byte; odd values: high nibble */
        AscendC::And(lo, codes, NIBBLE_MASK, TILE_ELEMS);
        AscendC::ShiftRight(hi, codes, SHIFT_BY, TILE_ELEMS);
        AscendC::And(hi, hi, NIBBLE_MASK, TILE_ELEMS);
        AscendC::Or(codes, lo, hi, TILE_ELEMS);

        AscendC::LocalTensor<half> out = outQue.AllocTensor<half>();
        AscendC::Cast(out, codes, AscendC::RoundMode::CAST_NONE, TILE_ELEMS);
        AscendC::Mul(out, out, FP4_STEP, TILE_ELEMS);
        outQue.EnQue(out);

        auto res = outQue.DeQue<half>();
        AscendC::DataCopy(dst + t * TILE_ELEMS, res, TILE_ELEMS);
        inQue.FreeTensor(in);
        outQue.FreeTensor(res);
    }
}
