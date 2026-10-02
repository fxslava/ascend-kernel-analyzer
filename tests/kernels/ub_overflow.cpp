/*
 * Tiled reduction whose UB tile offsets grow with the loop counter.
 *
 * This is the case interval arithmetic alone cannot settle and the solver can:
 * the base offset is not a constant, it is `t * TILE_BYTES` with `t` bounded by
 * the loop header. Asking whether the range can ever leave UB is a satisfiability
 * question, and the answer comes back with the iteration number that first
 * escapes - far more actionable than "this might overflow".
 *
 * INTENTIONALLY BROKEN: 32 tiles of 8 KiB is 256 KiB, but the 910B Unified
 * Buffer holds 192 KiB. The kernel is fine for the first 24 tiles and corrupts
 * memory from tile 24 onwards, which is precisely the kind of bug that survives
 * small-input testing and fails in production.
 *
 * @ascend-expect: AKA1001
 * @ascend-expect-fatal: 1
 */
#include "kernel_operator.h"

constexpr uint32_t TILE_BYTES = 8 * 1024;                  /* 8 KiB per tile  */
constexpr uint32_t TILE_ELEMS = TILE_BYTES / sizeof(half); /* 4096 half       */
constexpr uint32_t TILE_COUNT = 32;                        /* 32 * 8 KiB      */

extern "C" __global__ __aicore__ void tiled_reduce_overflow(
    __gm__ half* xGm, __gm__ half* yGm)
{
    AscendC::GlobalTensor<half> xGlobal;
    AscendC::GlobalTensor<half> yGlobal;
    xGlobal.SetGlobalBuffer(xGm, TILE_ELEMS * TILE_COUNT);
    yGlobal.SetGlobalBuffer(yGm, TILE_ELEMS * TILE_COUNT);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

    for (uint32_t t = 0; t < TILE_COUNT; ++t) {
        /* Every tile gets its own UB slot, which is the bug: the slots are
         * never recycled, so the footprint grows without bound. */
        AscendC::LocalTensor<half> tile;
        tile.SetTPosition(AscendC::TPosition::VECIN);
        tile.SetAddr(t * TILE_BYTES);
        tile.SetSize(TILE_ELEMS);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopy(tile, xGlobal[t * TILE_ELEMS], TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Abs(tile, tile, TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(yGlobal[t * TILE_ELEMS], tile, TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
}
