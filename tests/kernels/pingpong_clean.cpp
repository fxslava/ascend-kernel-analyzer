/*
 * The corrected counterpart of pingpong_broken.cpp.
 *
 * Same algorithm - double-buffered vector add over eight tiles - with every
 * seeded defect repaired:
 *
 *   * TILE_ELEMS is a multiple of 16 half values, so every byte length is a
 *     whole number of 32-byte DaVinci blocks;
 *   * the four UB tiles are contiguous, 32-byte aligned and disjoint;
 *   * the loop-carried V_MTE2 and MTE3_V handshakes are primed before the
 *     loop and drained after it, so iteration 0 finds a token waiting;
 *   * no reserved event ids, and no global barrier.
 *
 * The analyzer should report zero FATAL findings for this file. It is the
 * negative control: a checker that cannot stay quiet on correct code is
 * worthless, however many real bugs it finds.
 *
 * @ascend-expect:
 * @ascend-expect-fatal: 0
 */
#include "kernel_operator.h"

constexpr uint32_t TILE_ELEMS = 256;
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(half);   /* 512 B */
constexpr uint32_t TILE_COUNT = 8;

/* ---- UB layout: four contiguous 512 B tiles, all 32 B aligned ----------- */
constexpr uint32_t UB_X_PING = 0;                            /* [   0,  512) */
constexpr uint32_t UB_X_PONG = UB_X_PING + TILE_BYTES;       /* [ 512, 1024) */
constexpr uint32_t UB_Y_PING = UB_X_PONG + TILE_BYTES;       /* [1024, 1536) */
constexpr uint32_t UB_Y_PONG = UB_Y_PING + TILE_BYTES;       /* [1536, 2048) */

extern "C" __global__ __aicore__ void vec_add_pingpong_clean(
    __gm__ half* xGm, __gm__ half* yGm)
{
    AscendC::GlobalTensor<half> xGlobal;
    AscendC::GlobalTensor<half> yGlobal;
    xGlobal.SetGlobalBuffer(xGm, TILE_ELEMS * TILE_COUNT);
    yGlobal.SetGlobalBuffer(yGm, TILE_ELEMS * TILE_COUNT);

    AscendC::LocalTensor<half> xPing;
    xPing.SetTPosition(AscendC::TPosition::VECIN);
    xPing.SetAddr(UB_X_PING);
    xPing.SetSize(TILE_ELEMS);

    AscendC::LocalTensor<half> xPong;
    xPong.SetTPosition(AscendC::TPosition::VECIN);
    xPong.SetAddr(UB_X_PONG);
    xPong.SetSize(TILE_ELEMS);

    AscendC::LocalTensor<half> yPing;
    yPing.SetTPosition(AscendC::TPosition::VECOUT);
    yPing.SetAddr(UB_Y_PING);
    yPing.SetSize(TILE_ELEMS);

    AscendC::LocalTensor<half> yPong;
    yPong.SetTPosition(AscendC::TPosition::VECOUT);
    yPong.SetAddr(UB_Y_PONG);
    yPong.SetSize(TILE_ELEMS);

    /* Prologue: both x tiles start free for MTE2 to fill, and both y tiles
     * start free for the vector unit to overwrite. */
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

    for (uint32_t t = 0; t < TILE_COUNT / 2; ++t) {
        /* ------------------------- ping slot ------------------------- */
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopy(xPing, xGlobal[(2 * t) * TILE_ELEMS], TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::Add(yPing, xPing, xPing, TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(yGlobal[(2 * t) * TILE_ELEMS], yPing, TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

        /* ------------------------- pong slot ------------------------- */
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::DataCopy(xPong, xGlobal[(2 * t + 1) * TILE_ELEMS], TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::Add(yPong, xPong, xPong, TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopy(yGlobal[(2 * t + 1) * TILE_ELEMS], yPong, TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

    /* Epilogue: drain the flags the prologue primed, so the kernel leaves no
     * event slot occupied. */
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
}
