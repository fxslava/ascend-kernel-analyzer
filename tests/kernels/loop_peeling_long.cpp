/*
 * A software-pipelined vector kernel with a long trip count (T = 512): the
 * shape the three-phase loop-peeling traversal exists for.
 *
 * The loop body is a four-channel double-buffered pipeline:
 *
 *   MTE2_V  - tile t has been prefetched into x slot p
 *   MTE3_V  - y slot p's previous store has been issued
 *   V_MTE3  - y slot p holds tile t's result
 *   V_MTE2  - x slot p has been consumed and may be refilled
 *
 * and it carries exactly the constructs that defeat a single symbolic body
 * pass at large trip counts:
 *
 *   * ping/pong parity, `p = t & 1`, selecting the event id and the slot;
 *   * an epilogue guard, `if (t + 2 < TILE_COUNT)`, that stops prefetching
 *     two iterations before the end so no tile beyond the last is fetched.
 *
 * Full unrolling at T = 512 would emit ~4 000 traced operations; the peeled
 * traversal emits the steady-state representative cycle (two iterations, one
 * per parity) plus the two tail iterations where the guard flips, keeping the
 * synchronisation graph at 34 nodes.  The kernel is correct: it must come
 * back with zero findings, an acyclic marked graph, and analysis time far
 * below the 200 ms budget.
 *
 * @ascend-expect:
 * @ascend-expect-fatal: 0
 */
#include "kernel_operator.h"

constexpr uint32_t TILE_ELEMS = 128;                  /* 256 B of half values */
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(half);
constexpr uint32_t TILE_COUNT = 512;                  /* the pipeline length  */

/* ---- UB layout: x slots [0, 512), y slots [512, 1024) -------------------- */
constexpr uint32_t UB_X_BASE = 0;
constexpr uint32_t UB_Y_BASE = UB_X_BASE + 2 * TILE_BYTES;

#define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)

extern "C" __global__ __aicore__ void pipeline_long(
    __gm__ half* xGm, __gm__ half* yGm)
{
    AscendC::GlobalTensor<half> xGlobal;
    AscendC::GlobalTensor<half> yGlobal;
    xGlobal.SetGlobalBuffer(xGm, TILE_ELEMS * TILE_COUNT);
    yGlobal.SetGlobalBuffer(yGm, TILE_ELEMS * TILE_COUNT);

    AscendC::LocalTensor<half> xSlots;                /* ping [0,256) pong [256,512) */
    xSlots.SetTPosition(AscendC::TPosition::VECIN);
    xSlots.SetAddr(UB_X_BASE);
    xSlots.SetSize(2 * TILE_ELEMS);

    AscendC::LocalTensor<half> ySlots;                /* ping [512,768) pong [768,1024) */
    ySlots.SetTPosition(AscendC::TPosition::VECOUT);
    ySlots.SetAddr(UB_Y_BASE);
    ySlots.SetSize(2 * TILE_ELEMS);

    /* Prologue: prefetch tiles 0 and 1 into the x slots, and declare both y
     * slots drained (nothing has been stored into them yet). */
    AscendC::DataCopy(xSlots[0], xGlobal[0], TILE_ELEMS);
    AscendC::DataCopy(xSlots[TILE_ELEMS], xGlobal[TILE_ELEMS], TILE_ELEMS);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

    for (uint32_t t = 0; t < TILE_COUNT; ++t) {
        const uint32_t p = t & 1;                     /* ping/pong parity     */

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EV(p));   /* tile t arrived    */
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EV(p));   /* y slot p drained  */
        AscendC::Add(ySlots[p * TILE_ELEMS], xSlots[p * TILE_ELEMS],
                     xSlots[p * TILE_ELEMS], TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EV(p));    /* y slot p computed */
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EV(p));
        AscendC::DataCopy(yGlobal[t * TILE_ELEMS], ySlots[p * TILE_ELEMS],
                          TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EV(p));    /* store issued      */

        if (t + 2 < TILE_COUNT) {
            /* Refill x slot p with tile t+2.  The slot-consumed flag is only
             * raised when a refill actually follows (the last two iterations
             * never prefetch, so raising it there would leak the token); the
             * parity lines up because p == (t + 2) & 1. */
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EV(p));  /* x slot consumed */
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EV(p));  /* free to refill  */
            AscendC::DataCopy(xSlots[p * TILE_ELEMS],
                              xGlobal[(t + 2) * TILE_ELEMS], TILE_ELEMS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EV(p));
        }
    }

    /* Epilogue: drain the two MTE3_V tokens per channel the last iterations
     * left pending, so the kernel exits with no event slot occupied. */
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
}
