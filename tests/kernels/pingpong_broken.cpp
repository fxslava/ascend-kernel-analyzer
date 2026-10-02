/*
 * Double-buffered (ping/pong) vector add, static tensor programming model.
 *
 *     y[i] = x[i] + x[i]      over TILE_COUNT tiles of TILE_ELEMS half values
 *
 * Raw LocalTensor addressing: no TPipe, no TQue, no automatic buffer manager.
 * Every byte offset and every pipeline handshake is the author's problem,
 * which is exactly why a static analyzer earns its keep here.
 *
 * ---------------------------------------------------------------------------
 * THIS KERNEL IS INTENTIONALLY BROKEN. It is the regression fixture for
 * ascend-kernel-analyzer and carries four seeded defects:
 *
 *   [1] TILE_ELEMS is not a multiple of the 16-element (32-byte) DaVinci
 *       block, so every derived byte length is misaligned.   -> AKA1005
 *   [2] UB_Y_PING sits too low and overlaps the X pong tile. -> AKA1003
 *   [3] UB_Y_PONG inherits the ragged 500-byte stride and
 *       lands off a 32-byte boundary.                        -> AKA1002
 *   [4] The loop-carried V_MTE2 / MTE3_V handshakes are never
 *       primed before the loop, so iteration 0 waits on flags
 *       nothing has raised: a hard hang.           -> AKA2005 / AKA2004
 *
 * Two further smells are deliberate: EVENT_ID6 is reserved by the runtime
 * (-> AKA2003) and the trailing PipeBarrier(PIPE_ALL) throws away the
 * pipeline overlap that double buffering exists to create (-> AKA3001).
 * ---------------------------------------------------------------------------
 *
 * @ascend-expect: AKA1002 AKA1003 AKA1005 AKA2003 AKA2005 AKA3001
 * @ascend-expect-fatal: 13
 */
#include "kernel_operator.h"

/* [1] 250 is not a multiple of 16 half values, so TILE_BYTES is not a
 *     multiple of the 32-byte block the DMA engine actually transfers. */
constexpr uint32_t TILE_ELEMS = 250;
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(half);   /* 500 B */
constexpr uint32_t TILE_COUNT = 8;

/* ---- UB layout ---------------------------------------------------------- */
constexpr uint32_t UB_X_PING = 0;                            /* [   0,  500) */
constexpr uint32_t UB_X_PONG = 512;                          /* [ 512, 1012) */
/* [2] should start at 1024; 1000 overlaps the X pong tile. */
constexpr uint32_t UB_Y_PING = 1000;                         /* [1000, 1500) */
/* [3] 1000 + 500 = 1500, which is not 32-byte aligned. */
constexpr uint32_t UB_Y_PONG = UB_Y_PING + TILE_BYTES;       /* [1500, 2000) */

extern "C" __global__ __aicore__ void vec_add_pingpong_broken(
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

    /* [4] MISSING PROLOGUE. A correct kernel primes both slots here:
     *
     *     AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
     *     AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
     *     AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
     *     AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
     */

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

    /* EVENT_ID6 is reserved by the runtime: the framework may consume or
     * raise this flag behind the kernel's back. */
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

    /* A full fence where a localised MTE3 -> MTE2 handshake would do. */
    AscendC::PipeBarrier<PIPE_ALL>();
}
