/*
 * The same double-buffered add written at the lowest level available: raw
 * DaVinci address-space pointers and the ISASI intrinsic spelling of the
 * pipeline handshake.
 *
 *   __ubuf__ half* p = (__ubuf__ half*)(OFFSET);
 *   set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
 *   wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
 *   pipe_barrier(PIPE_ALL);
 *
 * There is no LocalTensor here at all, so the analyzer recovers the memory
 * domain from the address-space qualifier and the byte offset from the cast.
 * A raw pointer carries no length, so the tile sizes are declared with the
 * documented '@ascend-layout' annotation - the escape hatch for layouts the
 * parser cannot infer on its own.
 *
 * This file is CORRECT apart from one seeded defect: the trailing
 * pipe_barrier(PIPE_ALL) is a global fence where a localised handshake would
 * do, so it should report exactly one WARNING and no FATAL findings.
 *
 * @ascend-expect: AKA3001
 * @ascend-expect-fatal: 0
 */
#include "kernel_operator.h"

#define TILE_BYTES 512
#define TILE_ELEMS 256

constexpr uint32_t UB_X_PING = 0;
constexpr uint32_t UB_X_PONG = UB_X_PING + TILE_BYTES;
constexpr uint32_t UB_Y_PING = UB_X_PONG + TILE_BYTES;
constexpr uint32_t UB_Y_PONG = UB_Y_PING + TILE_BYTES;

extern "C" __global__ __aicore__ void raw_isasi_add(
    __gm__ half* xGm, __gm__ half* yGm)
{
    // @ascend-layout: name=xPingRaw pos=VECIN  offset=0    count=256 dtype=half
    __ubuf__ half* xPingRaw = (__ubuf__ half*)(UB_X_PING);
    // @ascend-layout: name=xPongRaw pos=VECIN  offset=512  count=256 dtype=half
    __ubuf__ half* xPongRaw = (__ubuf__ half*)(UB_X_PONG);
    // @ascend-layout: name=yPingRaw pos=VECOUT offset=1024 count=256 dtype=half
    __ubuf__ half* yPingRaw = (__ubuf__ half*)(UB_Y_PING);
    // @ascend-layout: name=yPongRaw pos=VECOUT offset=1536 count=256 dtype=half
    __ubuf__ half* yPongRaw = (__ubuf__ half*)(UB_Y_PONG);

    /* Prime both slots: the x tiles are free for MTE2, the y tiles are free
     * for the vector unit to overwrite. */
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    for (uint32_t t = 0; t < 4; ++t) {
        /* ---- ping ---- */
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        AscendC::DataCopy(xPingRaw, xGm, TILE_ELEMS);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        AscendC::Add(yPingRaw, xPingRaw, xPingRaw, TILE_ELEMS);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        AscendC::DataCopy(yGm, yPingRaw, TILE_ELEMS);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        /* ---- pong ---- */
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        AscendC::DataCopy(xPongRaw, xGm, TILE_ELEMS);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        AscendC::Add(yPongRaw, xPongRaw, xPongRaw, TILE_ELEMS);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        AscendC::DataCopy(yGm, yPongRaw, TILE_ELEMS);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }

    /* Drain the primed flags. */
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    /* Seeded antipattern: a full fence at the end of the kernel. */
    pipe_barrier(PIPE_ALL);
}
