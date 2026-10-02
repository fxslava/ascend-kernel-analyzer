/*
 * cube_tbuf_pipeline.cpp -- regression fixture for the Cube-only kernel shape.
 *
 * Exercises, in one file, the frontend features that low-level Cube kernels
 * need and that the plain vector-model fixtures do not cover:
 *
 *   * stage macros: multi-line function-like #define bodies with do/while(0)
 *     blocks, expanded at their invocation sites before parsing;
 *   * a local header included next to the kernel, providing the tile
 *     geometry macros the layout folder depends on;
 *   * TBuf<TPosition> + TPipe::InitBuffer allocations, whose domains and
 *     sizes must propagate to the LocalTensor obtained via .Get<T>() -
 *     no SetTPosition calls and no @ascend-layout annotations anywhere;
 *   * Clang CCE address-space qualifiers (__ca__/__cb__/__cc__/__cbuf__)
 *     inside C-style casts around raw .GetPhyAddr() pointers;
 *   * the ping/pong event selector EV(p) = p ? EVENT_ID1 : EVENT_ID0,
 *     which must fold to concrete ids per unrolled loop iteration;
 *   * a skewed double-buffered loop with prologue primes and an epilogue
 *     that drains every flag it raised.
 *
 * The analyzer should report zero findings on this file, strict mode
 * included: every SetFlag has exactly one statically paired WaitFlag and the
 * whole SRAM layout is concrete.
 *
 * The Cube stage issues mad_mx, the MX microscale multiply-accumulate,
 * which is a DaVinci v3 instruction: 910B has no MX operand format. The
 * fixture therefore declares a v3 target, so the instruction-set gate
 * (AKA1011) judges it against a part that can actually issue it.
 *
 * @ascend-chip: ascend351x
 * @ascend-expect:
 * @ascend-expect-fatal: 0
 */
#include "kernel_operator.h"
#include "cube_tbuf_geometry.h"

using namespace AscendC;

#define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)

extern "C" __global__ __aicore__ void cube_tbuf_pipeline(GM_ADDR aq, GM_ADDR bq,
                                                         GM_ADDR dst)
{
    GlobalTensor<int8_t> gA;  gA.SetGlobalBuffer((__gm__ int8_t *)aq, TILE_A_BYTES * TILES);
    GlobalTensor<int8_t> gB;  gB.SetGlobalBuffer((__gm__ int8_t *)bq, TILE_B_BYTES * TILES);
    GlobalTensor<float>  gD;  gD.SetGlobalBuffer((__gm__ float *)dst, TILE_ELEMS * TILES);

    TPipe pipe;
    TBuf<TPosition::A1> bL1a;
    TBuf<TPosition::B1> bL1b;
    TBuf<TPosition::A2> bL0a;
    TBuf<TPosition::B2> bL0b;
    TBuf<TPosition::CO1> bL0c;
    pipe.InitBuffer(bL1a, 2 * TILE_A_BYTES);
    pipe.InitBuffer(bL1b, 2 * TILE_B_BYTES);
    pipe.InitBuffer(bL0a, 2 * TILE_A_BYTES);
    pipe.InitBuffer(bL0b, 2 * TILE_B_BYTES);
    pipe.InitBuffer(bL0c, 2 * TILE_C_BYTES);

    LocalTensor<int8_t> l1a = bL1a.Get<int8_t>();
    LocalTensor<int8_t> l1b = bL1b.Get<int8_t>();
    LocalTensor<int8_t> l0a = bL0a.Get<int8_t>();
    LocalTensor<int8_t> l0b = bL0b.Get<int8_t>();
    LocalTensor<float>  l0c = bL0c.Get<float>();

#define CT_MTE2(t, p)                                                              \
    do {                                                                           \
        DataCopy(l1a[(p) * TILE_A_BYTES], gA[(t) * TILE_A_BYTES], TILE_A_BYTES);   \
        DataCopy(l1b[(p) * TILE_B_BYTES], gB[(t) * TILE_B_BYTES], TILE_B_BYTES);   \
        SetFlag<HardEvent::MTE2_MTE1>(EV(p));                                      \
    } while (0)

#define CT_MTE1(p)                                                                 \
    do {                                                                           \
        WaitFlag<HardEvent::MTE2_MTE1>(EV(p));                                     \
        load_cbuf_to_ca_s4(                                                        \
            (__ca__ fp4x2_e2m1_t *)(uintptr_t)l0a[(p) * TILE_A_BYTES].GetPhyAddr(),\
            (__cbuf__ fp4x2_e2m1_t *)(uintptr_t)l1a[(p) * TILE_A_BYTES].GetPhyAddr(),\
            (uint16_t)0, (uint16_t)0, (uint8_t)1, (uint8_t)1,                      \
            (int16_t)1, (uint16_t)1, false);                                       \
        load_cbuf_to_cb_s4(                                                        \
            (__cb__ fp4x2_e1m2_t *)(uintptr_t)l0b[(p) * TILE_B_BYTES].GetPhyAddr(),\
            (__cbuf__ fp4x2_e1m2_t *)(uintptr_t)l1b[(p) * TILE_B_BYTES].GetPhyAddr(),\
            (uint16_t)0, (uint16_t)0, (uint8_t)1, (uint8_t)1,                      \
            (int16_t)1, (uint16_t)1, false);                                       \
        SetFlag<HardEvent::MTE1_M>(EV(p));                                         \
        SetFlag<HardEvent::MTE1_MTE2>(EV(p));                                      \
    } while (0)

#define CT_MAD(t, p)                                                               \
    do {                                                                           \
        WaitFlag<HardEvent::MTE1_M>(EV(p));                                        \
        mad_mx((__cc__ float *)(uintptr_t)l0c[(p) * TILE_ELEMS].GetPhyAddr(),      \
               (uint64_t)0,                                                        \
               (__ca__ float4_e2m1x2_t *)(uintptr_t)l0a[(p) * TILE_A_BYTES].GetPhyAddr(),\
               (uint64_t)0,                                                        \
               (__cb__ float4_e1m2x2_t *)(uintptr_t)l0b[(p) * TILE_B_BYTES].GetPhyAddr(),\
               (uint64_t)0,                                                        \
               mmad_t::shape_t((uint16_t)M, (uint16_t)K, (uint16_t)N), ctl);       \
        SetFlag<HardEvent::M_FIX>(EV(p));                                          \
        WaitFlag<HardEvent::M_FIX>(EV(p));                                         \
        Fixpipe(gD[(t) * TILE_ELEMS], l0c[(p) * TILE_ELEMS],                       \
                FixpipeParamsV220((uint16_t)N, (uint16_t)M, 0,                     \
                                  (uint32_t)N, false));                            \
        SetFlag<HardEvent::M_MTE1>(EV(p));                                         \
    } while (0)

    mmad_t::control_t ctl;
    ctl.unit_flag_ctrl = 0; ctl.gemv_ctrl = false;
    ctl.BTbuf_ctrl = false; ctl.zero_Cmatrix_ctrl = true;

    CT_MTE2(0, 0);
    CT_MTE2(1, 1);
    CT_MTE1(0);

    for (int t = 0; t < TILES; ++t) {
        int p = t & 1;
        if (t + 2 < TILES) {
            WaitFlag<HardEvent::MTE1_MTE2>(EV(p));
            CT_MTE2(t + 2, p);
        }
        if (t + 1 < TILES) {
            if (t >= 1) WaitFlag<HardEvent::M_MTE1>(EV((t + 1) & 1));
            CT_MTE1((t + 1) & 1);
        }
        CT_MAD(t, p);
    }

    /* Epilogue: drain the reuse flags whose consumer would only exist for a
     * next tile that never comes, so every SetFlag raised is paired. */
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
#undef CT_MTE2
#undef CT_MTE1
#undef CT_MAD
}
