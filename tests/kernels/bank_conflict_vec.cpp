/*
 * Vector ALU UB bank-conflict fixture (WARNING AKA3006).
 *
 * The Unified Buffer is an interleaved 8-bank structure addressed in 32-byte
 * quantization blocks; a dual-operand vector instruction that reads both
 * sources from the same bank stalls the read ports for extra pipeline beats.
 *
 *   * the CONFLICT case reads aBad (0x0) and bBad (0x100): 256 bytes apart,
 *     which is 8 blocks = 0 mod 8 -> both bases sit in UB Bank 0;
 *   * the CONTROL case reads aGood (0x400) and bGood (0x420): the 32-byte
 *     (one-block) skew puts them 9 blocks apart = 1 mod 8 -> distinct banks.
 *
 * Code note: this check is specified as "AKA3003", but that code was already
 * assigned to SYMBOLIC_EVENT_ID when the analyzer shipped, so the bank
 * conflict carries AKA3006, the next free code in the 3xxx performance
 * block.  The finding is a performance warning; the kernel is still
 * functionally correct and the verdict is accepted_with_warnings.
 *
 * @ascend-expect: AKA3006
 * @ascend-expect-fatal: 0
 */
#include "kernel_operator.h"

constexpr uint32_t TILE_ELEMS = 128;                  /* 256 B of half values */

/* ---- UB layout ------------------------------------------------------------
 * aBad     [   0,  256)  -> bank 0      aGood    [1024, 1280) -> bank 0
 * bBad     [ 256,  512)  -> bank 0      bGood    [1312, 1568) -> bank 1
 * dstBad   [ 512,  768)                 dstGood  [1568, 1824)
 * -------------------------------------------------------------------------- */
constexpr uint32_t OFF_A_BAD = 0;
constexpr uint32_t OFF_B_BAD = 256;                   /* 8 blocks -> bank 0   */
constexpr uint32_t OFF_DST_BAD = 512;
constexpr uint32_t OFF_A_GOOD = 1024;                 /* bank 0               */
constexpr uint32_t OFF_B_GOOD = 1312;                 /* +9 blocks -> bank 1  */
constexpr uint32_t OFF_DST_GOOD = 1568;

extern "C" __global__ __aicore__ void bank_conflict_vec(__gm__ half* gm)
{
    AscendC::GlobalTensor<half> g;
    g.SetGlobalBuffer(gm, 8 * TILE_ELEMS);

    AscendC::LocalTensor<half> aBad;
    aBad.SetTPosition(AscendC::TPosition::VECIN);
    aBad.SetAddr(OFF_A_BAD);
    aBad.SetSize(TILE_ELEMS);

    AscendC::LocalTensor<half> bBad;
    bBad.SetTPosition(AscendC::TPosition::VECIN);
    bBad.SetAddr(OFF_B_BAD);
    bBad.SetSize(TILE_ELEMS);

    AscendC::LocalTensor<half> dstBad;
    dstBad.SetTPosition(AscendC::TPosition::VECOUT);
    dstBad.SetAddr(OFF_DST_BAD);
    dstBad.SetSize(TILE_ELEMS);

    AscendC::LocalTensor<half> aGood;
    aGood.SetTPosition(AscendC::TPosition::VECIN);
    aGood.SetAddr(OFF_A_GOOD);
    aGood.SetSize(TILE_ELEMS);

    AscendC::LocalTensor<half> bGood;
    bGood.SetTPosition(AscendC::TPosition::VECIN);
    bGood.SetAddr(OFF_B_GOOD);
    bGood.SetSize(TILE_ELEMS);

    AscendC::LocalTensor<half> dstGood;
    dstGood.SetTPosition(AscendC::TPosition::VECOUT);
    dstGood.SetAddr(OFF_DST_GOOD);
    dstGood.SetSize(TILE_ELEMS);

    AscendC::DataCopy(aBad, g[0], TILE_ELEMS);
    AscendC::DataCopy(bBad, g[2 * TILE_ELEMS], TILE_ELEMS);
    AscendC::DataCopy(aGood, g[4 * TILE_ELEMS], TILE_ELEMS);
    AscendC::DataCopy(bGood, g[6 * TILE_ELEMS], TILE_ELEMS);

    /* CONFLICT: both sources in UB Bank 0 (delta 8 blocks = 0 mod 8). */
    AscendC::Add(dstBad, aBad, bBad, TILE_ELEMS);

    /* CONTROL: the 32-byte skew separates the banks (delta 9 = 1 mod 8). */
    AscendC::Add(dstGood, aGood, bGood, TILE_ELEMS);

    AscendC::DataCopy(g[0], dstBad, TILE_ELEMS);
    AscendC::DataCopy(g[2 * TILE_ELEMS], dstGood, TILE_ELEMS);
}
