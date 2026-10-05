/*
 * cube_trick_fp4_to_fp16.cpp -- heterogeneous MIX (AIC + AIV) FP4 pipeline.
 *
 * The "trick" in FP4 dequant-GEMM kernels: the vector core expands the packed
 * e2m1 nibbles and their shared scale exponents into FP16 operands before the
 * cube core consumes them, so the MMAD pipeline never sees a sub-byte format.
 * This file models that split as two stage classes in one translation unit:
 *
 *   * CubeStage  -- the AIC side: MTE2 loads A/B into L1, MTE1 stages them
 *     into L0A/L0B, the M pipe runs Mmad into L0C and Fixpipe drains L0C to
 *     GM.  Ping/pong double buffering with prologue primes and an epilogue
 *     that drains every reuse flag it raised.
 *   * VectorStage -- the AIV side: a queue-driven FP4 -> FP16 expansion of
 *     the B operand, straight-line per tile, no raw flags.
 *
 * The kernel entry selects the stage at run time via the ASCEND_IS_AIC /
 * ASCEND_IS_AIV sentinels, so both regions live in the same binary pair and
 * the two stages must never collide in the symbol table: their TBufs, queues
 * and flag pairs belong to their respective CoreRegionOps.
 *
 * Written against the CANN 8.5 API surface (Mmad with MmadParams, Fixpipe
 * with FixpipeParamsV220) so the BiSheng extraction frontend typechecks it
 * against the real headers.
 *
 * @ascend-chip: ascend910b
 * @ascend-expect:
 * @ascend-expect-fatal: 0
 */
#include "kernel_operator.h"

namespace {
constexpr uint16_t GM_M = 32;
constexpr uint16_t GM_N = 32;
constexpr uint16_t GM_K = 16;

constexpr uint32_t A_ELEMS = static_cast<uint32_t>(GM_M) * GM_K;   /* 512 */
constexpr uint32_t B_ELEMS = static_cast<uint32_t>(GM_K) * GM_N;   /* 512 */
constexpr uint32_t C_ELEMS = static_cast<uint32_t>(GM_M) * GM_N;   /* 1024 */
constexpr uint32_t A_BYTES = A_ELEMS * sizeof(half);               /* 1 KiB */
constexpr uint32_t B_BYTES = B_ELEMS * sizeof(half);               /* 1 KiB */
constexpr uint32_t B_PACKED_BYTES = B_ELEMS / 2;                   /* 256 B */
constexpr uint32_t C_BYTES = C_ELEMS * sizeof(float);              /* 4 KiB */
constexpr uint32_t TILES = 4;

/* Ping/pong event selector: one event id per buffer slot. */
constexpr uint32_t EV(uint32_t p)
{
    return p ? EVENT_ID1 : EVENT_ID0;
}
}  // namespace

class CubeStage {
public:
    __aicore__ void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(l1A, 2 * A_BYTES);
        pipe.InitBuffer(l1B, 2 * B_BYTES);
        pipe.InitBuffer(l0A, 2 * A_BYTES);
        pipe.InitBuffer(l0B, 2 * B_BYTES);
        pipe.InitBuffer(l0C, 2 * C_BYTES);
    }

    __aicore__ void CopyIn(uint32_t t, uint32_t p, __gm__ half* gA, __gm__ half* gB)
    {
        AscendC::DataCopy(l1A.Get<half>()[p * A_ELEMS], gA + t * A_ELEMS, A_ELEMS);
        AscendC::DataCopy(l1B.Get<half>()[p * B_ELEMS], gB + t * B_ELEMS, B_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EV(p));
    }

    __aicore__ void StageToL0(uint32_t p)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EV(p));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EV(p));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EV(p));
    }

    __aicore__ void Compute(uint32_t t, uint32_t p, __gm__ float* gD)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EV(p));
        AscendC::Mmad(l0C.Get<float>()[p * C_ELEMS], l0A.Get<half>()[p * A_ELEMS],
                      l0B.Get<half>()[p * B_ELEMS],
                      AscendC::MmadParams(GM_M, GM_N, GM_K, false, 0, false, false, false));
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EV(p));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EV(p));
        AscendC::Fixpipe(gD + t * C_ELEMS, l0C.Get<float>()[p * C_ELEMS],
                         AscendC::FixpipeParamsV220(GM_N, GM_M, GM_N, GM_N * GM_M, false));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EV(p));
    }

private:
    AscendC::TBuf<AscendC::TPosition::A1> l1A;
    AscendC::TBuf<AscendC::TPosition::B1> l1B;
    AscendC::TBuf<AscendC::TPosition::A2> l0A;
    AscendC::TBuf<AscendC::TPosition::B2> l0B;
    AscendC::TBuf<AscendC::TPosition::CO1> l0C;
};

class VectorStage {
public:
    __aicore__ void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(inQue, 2, B_PACKED_BYTES);
        pipe.InitBuffer(outQue, 2, B_BYTES);
        pipe.InitBuffer(scratch, B_ELEMS * sizeof(int16_t));
    }

    __aicore__ void Process(uint32_t t, __gm__ uint8_t* packedB, __gm__ half* gB)
    {
        AscendC::LocalTensor<uint8_t> packed = inQue.AllocTensor<uint8_t>();
        AscendC::DataCopy(packed, packedB + t * B_PACKED_BYTES, B_PACKED_BYTES);
        inQue.EnQue(packed);

        AscendC::LocalTensor<uint8_t> in = inQue.DeQue<uint8_t>();
        AscendC::LocalTensor<int16_t> codes = scratch.Get<int16_t>();
        AscendC::Cast(codes, in, AscendC::RoundMode::CAST_NONE, B_ELEMS);
        AscendC::LocalTensor<half> out = outQue.AllocTensor<half>();
        AscendC::Cast(out, codes, AscendC::RoundMode::CAST_NONE, B_ELEMS);
        AscendC::Mul(out, out, static_cast<half>(0.5), B_ELEMS);
        outQue.EnQue(out);

        AscendC::LocalTensor<half> res = outQue.DeQue<half>();
        AscendC::DataCopy(gB + t * B_ELEMS, res, B_ELEMS);
        inQue.FreeTensor(in);
        outQue.FreeTensor(res);
    }

private:
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch;
};

class KernelMix {
public:
    __aicore__ void Init(AscendC::TPipe& pipe)
    {
        if (ASCEND_IS_AIC) {
            cube.Init(pipe);
        } else if (ASCEND_IS_AIV) {
            vec.Init(pipe);
        }
    }

    __aicore__ void Process(AscendC::TPipe& pipe, __gm__ half* gA, __gm__ half* gB,
                            __gm__ uint8_t* packedB, __gm__ float* gD)
    {
        if (ASCEND_IS_AIC) {
            cube.CopyIn(0, 0, gA, gB);
            cube.CopyIn(1, 1, gA, gB);
            cube.StageToL0(0);
            for (uint32_t t = 0; t < TILES; ++t) {
                uint32_t p = t & 1;
                if (t + 2 < TILES) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EV(p));
                    cube.CopyIn(t + 2, p, gA, gB);
                }
                if (t + 1 < TILES) {
                    if (t >= 1) {
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EV((t + 1) & 1));
                    }
                    cube.StageToL0((t + 1) & 1);
                }
                cube.Compute(t, p, gD);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        } else if (ASCEND_IS_AIV) {
            for (uint32_t t = 0; t < TILES; ++t) {
                vec.Process(t, packedB, gB);
            }
        }
    }

private:
    CubeStage cube;
    VectorStage vec;
};

extern "C" __global__ __aicore__ void cube_trick_fp4_to_fp16(
    GM_ADDR a, GM_ADDR b, GM_ADDR packedB, GM_ADDR dst)
{
    AscendC::TPipe pipe;
    __gm__ half* gA = (__gm__ half*)a;
    __gm__ half* gB = (__gm__ half*)b;
    __gm__ uint8_t* gPacked = (__gm__ uint8_t*)packedB;
    __gm__ float* gD = (__gm__ float*)dst;

    KernelMix mix;
    mix.Init(pipe);
    mix.Process(pipe, gA, gB, gPacked, gD);
}
