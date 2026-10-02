/*
 * nvfp4_pipelined_dequant.cpp -- fixture for the analytical pipeline
 * performance & overlap profiler (WARNING AKA4001 / AKA4002).
 *
 * Two kernels do the same NVFP4 dequant-and-GEMM work per tile -
 * MTE2 moves packed fp4 A/B tiles GM -> L1, MTE1 fractal-loads them
 * L1 -> L0A/L0B, the cube contracts via mad_mx, and Fixpipe drains L0C
 * back to GM - but with opposite overlap structure:
 *
 *   Pipeline A (nvfp4_dequant_overlap) keeps the textbook double-buffered
 *   skew of cube_tbuf_pipeline.cpp, but with deliberately small tiles
 *   (16x128x16, 1 KiB per operand).  The cube contraction itself is only 8
 *   cycles, so nothing amortises the fixed ~30-cycle M_FIX hand-off or the
 *   per-tile fill/drain bubbles: the profiler must flag the exposed sync
 *   stalls (AKA4001) with the M_FIX route carrying a large share.
 *
 *   Pipeline B (nvfp4_dequant_serial) runs the identical stages one tile at
 *   a time behind five SetFlag/WaitFlag handshakes with single buffers: load,
 *   wait, fractal-load, wait, contract, wait, drain, wait, repeat.  Nothing
 *   overlaps anything; the profiler must report serialized execution - a
 *   SYNC_BOUND bottleneck with the overlap ratio far below Pipeline A.
 *
 * Both kernels are functionally correct (no memory or pairing findings);
 * only performance advisories from the 4xxx block are expected.
 *
 * @ascend-expect: AKA4001 AKA4002
 * @ascend-expect-fatal: 0
 */
#include "kernel_operator.h"

using namespace AscendC;

/* ---- tile geometry: small tiles so hand-offs dominate the work ---------- */
#define M            16
#define N            16
#define K            128
#define TILES_A      8
#define TILES_B      4

#define TILE_ELEMS   (M * N)
#define TILE_A_BYTES (M * K / 2)   /* 16x128 packed FP4 = 1024 B */
#define TILE_B_BYTES (K * N / 2)   /* 128x16 packed FP4 = 1024 B */
#define TILE_C_BYTES (TILE_ELEMS * 4)

#define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)

/* ========================================================================
 * Pipeline A: double-buffered but small-tile - exposed M_FIX hand-off and
 * per-tile bubbles.
 * ======================================================================== */
extern "C" __global__ __aicore__ void nvfp4_dequant_overlap(GM_ADDR aq, GM_ADDR bq,
                                                            GM_ADDR dst)
{
    GlobalTensor<int8_t> gA;  gA.SetGlobalBuffer((__gm__ int8_t *)aq, TILE_A_BYTES * TILES_A);
    GlobalTensor<int8_t> gB;  gB.SetGlobalBuffer((__gm__ int8_t *)bq, TILE_B_BYTES * TILES_A);
    GlobalTensor<float>  gD;  gD.SetGlobalBuffer((__gm__ float *)dst, TILE_ELEMS * TILES_A);

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

#define NV_MTE2(t, p)                                                              \
    do {                                                                           \
        DataCopy(l1a[(p) * TILE_A_BYTES], gA[(t) * TILE_A_BYTES], TILE_A_BYTES);   \
        DataCopy(l1b[(p) * TILE_B_BYTES], gB[(t) * TILE_B_BYTES], TILE_B_BYTES);   \
        SetFlag<HardEvent::MTE2_MTE1>(EV(p));                                       \
    } while (0)

#define NV_MTE1(p)                                                                 \
    do {                                                                           \
        WaitFlag<HardEvent::MTE2_MTE1>(EV(p));                                      \
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
        SetFlag<HardEvent::MTE1_M>(EV(p));                                          \
        SetFlag<HardEvent::MTE1_MTE2>(EV(p));                                       \
    } while (0)

#define NV_MAD(t, p)                                                               \
    do {                                                                           \
        WaitFlag<HardEvent::MTE1_M>(EV(p));                                         \
        mad_mx((__cc__ float *)(uintptr_t)l0c[(p) * TILE_ELEMS].GetPhyAddr(),       \
               (uint64_t)0,                                                         \
               (__ca__ float4_e2m1x2_t *)(uintptr_t)l0a[(p) * TILE_A_BYTES].GetPhyAddr(),\
               (uint64_t)0,                                                         \
               (__cb__ float4_e1m2x2_t *)(uintptr_t)l0b[(p) * TILE_B_BYTES].GetPhyAddr(),\
               (uint64_t)0,                                                         \
               mmad_t::shape_t((uint16_t)M, (uint16_t)K, (uint16_t)N), ctl);        \
        SetFlag<HardEvent::M_FIX>(EV(p));                                           \
        WaitFlag<HardEvent::M_FIX>(EV(p));                                          \
        Fixpipe(gD[(t) * TILE_ELEMS], l0c[(p) * TILE_ELEMS],                        \
                FixpipeParamsV220((uint16_t)N, (uint16_t)M, 0,                      \
                                  (uint32_t)N, false));                              \
        SetFlag<HardEvent::M_MTE1>(EV(p));                                          \
    } while (0)

    mmad_t::control_t ctl;
    ctl.unit_flag_ctrl = 0; ctl.gemv_ctrl = false;
    ctl.BTbuf_ctrl = false; ctl.zero_Cmatrix_ctrl = true;

    NV_MTE2(0, 0);
    NV_MTE2(1, 1);
    NV_MTE1(0);

    for (int t = 0; t < TILES_A; ++t) {
        int p = t & 1;
        if (t + 2 < TILES_A) {
            WaitFlag<HardEvent::MTE1_MTE2>(EV(p));
            NV_MTE2(t + 2, p);
        }
        if (t + 1 < TILES_A) {
            if (t >= 1) WaitFlag<HardEvent::M_MTE1>(EV((t + 1) & 1));
            NV_MTE1((t + 1) & 1);
        }
        NV_MAD(t, p);
    }

    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
#undef NV_MTE2
#undef NV_MTE1
#undef NV_MAD
}

/* ========================================================================
 * Pipeline B: the same stages, one tile at a time, every transition behind
 * a SetFlag/WaitFlag handshake - serialized execution.
 * ======================================================================== */
extern "C" __global__ __aicore__ void nvfp4_dequant_serial(GM_ADDR aq, GM_ADDR bq,
                                                           GM_ADDR dst)
{
    GlobalTensor<int8_t> gA;  gA.SetGlobalBuffer((__gm__ int8_t *)aq, TILE_A_BYTES * TILES_B);
    GlobalTensor<int8_t> gB;  gB.SetGlobalBuffer((__gm__ int8_t *)bq, TILE_B_BYTES * TILES_B);
    GlobalTensor<float>  gD;  gD.SetGlobalBuffer((__gm__ float *)dst, TILE_ELEMS * TILES_B);

    TPipe pipe;
    TBuf<TPosition::A1> bL1a;
    TBuf<TPosition::B1> bL1b;
    TBuf<TPosition::A2> bL0a;
    TBuf<TPosition::B2> bL0b;
    TBuf<TPosition::CO1> bL0c;
    pipe.InitBuffer(bL1a, TILE_A_BYTES);
    pipe.InitBuffer(bL1b, TILE_B_BYTES);
    pipe.InitBuffer(bL0a, TILE_A_BYTES);
    pipe.InitBuffer(bL0b, TILE_B_BYTES);
    pipe.InitBuffer(bL0c, TILE_C_BYTES);

    LocalTensor<int8_t> l1a = bL1a.Get<int8_t>();
    LocalTensor<int8_t> l1b = bL1b.Get<int8_t>();
    LocalTensor<int8_t> l0a = bL0a.Get<int8_t>();
    LocalTensor<int8_t> l0b = bL0b.Get<int8_t>();
    LocalTensor<float>  l0c = bL0c.Get<float>();

    mmad_t::control_t ctl;
    ctl.unit_flag_ctrl = 0; ctl.gemv_ctrl = false;
    ctl.BTbuf_ctrl = false; ctl.zero_Cmatrix_ctrl = true;

    /* The single L1/L0 slots start free and the accumulator starts drained. */
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);

#define NV_STEP(t)                                                                  \
    do {                                                                            \
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);      /* L1 slot free          */ \
        DataCopy(l1a[0], gA[(t) * TILE_A_BYTES], TILE_A_BYTES);                     \
        DataCopy(l1b[0], gB[(t) * TILE_B_BYTES], TILE_B_BYTES);                     \
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);       /* L1 filled             */ \
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);                                  \
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);           /* L0 slot free        */ \
        load_cbuf_to_ca_s4(                                                         \
            (__ca__ fp4x2_e2m1_t *)(uintptr_t)l0a[0].GetPhyAddr(),                  \
            (__cbuf__ fp4x2_e2m1_t *)(uintptr_t)l1a[0].GetPhyAddr(),                \
            (uint16_t)0, (uint16_t)0, (uint8_t)1, (uint8_t)1,                       \
            (int16_t)1, (uint16_t)1, false);                                        \
        load_cbuf_to_cb_s4(                                                         \
            (__cb__ fp4x2_e1m2_t *)(uintptr_t)l0b[0].GetPhyAddr(),                  \
            (__cbuf__ fp4x2_e1m2_t *)(uintptr_t)l1b[0].GetPhyAddr(),                \
            (uint16_t)0, (uint16_t)0, (uint8_t)1, (uint8_t)1,                       \
            (int16_t)1, (uint16_t)1, false);                                        \
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);           /* L0 filled             */ \
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);        /* L1 consumed           */ \
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);            /* L0C drained           */ \
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);            /* L0 filled           */ \
        mad_mx((__cc__ float *)(uintptr_t)l0c[0].GetPhyAddr(),                      \
               (uint64_t)0,                                                         \
               (__ca__ float4_e2m1x2_t *)(uintptr_t)l0a[0].GetPhyAddr(),            \
               (uint64_t)0,                                                         \
               (__cb__ float4_e1m2x2_t *)(uintptr_t)l0b[0].GetPhyAddr(),            \
               (uint64_t)0,                                                         \
               mmad_t::shape_t((uint16_t)M, (uint16_t)K, (uint16_t)N), ctl);        \
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);            /* L0C filled             */ \
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);            /* L0 consumed            */ \
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);                                      \
        Fixpipe(gD[(t) * TILE_ELEMS], l0c[0],                                       \
                FixpipeParamsV220((uint16_t)N, (uint16_t)M, 0,                      \
                                  (uint32_t)N, false));                              \
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);             /* L0C drained            */ \
    } while (0)

    NV_STEP(0);
    NV_STEP(1);
    NV_STEP(2);
    NV_STEP(3);

    /* Drain the three flags whose consumer would only exist for a fifth tile. */
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
#undef NV_STEP
}
