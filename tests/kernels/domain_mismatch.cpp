/*
 * Cube (matmul) kernel with "phantom type" memory-domain errors.
 *
 * Ascend C hides the address space inside LocalTensor<T>: a tensor in UB and a
 * tensor in L0A have the *same* C++ type, so the compiler accepts handing one
 * to an API that can only read the other. The result is not a type error, it is
 * a silently wrong kernel. These are exactly the mistakes a domain-aware static
 * check exists to catch.
 *
 * INTENTIONALLY BROKEN. Seeded defects:
 *   [1] Mmad reads its left matrix from UB; the Cube unit can only read L0A.
 *   [2] A vector Add is handed an L1-resident operand; the Vector unit can
 *       only address UB.
 *   [3] DataCopy goes straight from global memory to L0A. There is no such
 *       hardware path - it has to be staged through L1.
 *   [4] Fixpipe is handed an L1 source; it drains the L0C accumulator only.
 *
 * @ascend-expect: AKA1004
 * @ascend-expect-fatal: 4
 */
#include "kernel_operator.h"

constexpr uint32_t M = 16;
constexpr uint32_t K = 16;
constexpr uint32_t N = 16;

/* A 16x16 half fractal is 512 B; a 16x16 float accumulator fractal is 1024 B. */
constexpr uint32_t FRACTAL_HALF = 512;
constexpr uint32_t FRACTAL_FP32 = 1024;

extern "C" __global__ __aicore__ void mmad_wrong_domains(
    __gm__ half* aGm, __gm__ half* bGm, __gm__ float* cGm)
{
    AscendC::GlobalTensor<half> aGlobal;
    AscendC::GlobalTensor<half> bGlobal;
    AscendC::GlobalTensor<float> cGlobal;
    aGlobal.SetGlobalBuffer(aGm, M * K);
    bGlobal.SetGlobalBuffer(bGm, K * N);
    cGlobal.SetGlobalBuffer(cGm, M * N);

    /* L1 staging buffers. */
    AscendC::LocalTensor<half> aL1;
    aL1.SetTPosition(AscendC::TPosition::A1);
    aL1.SetAddr(0);
    aL1.SetSize(M * K);

    AscendC::LocalTensor<half> bL1;
    bL1.SetTPosition(AscendC::TPosition::B1);
    bL1.SetAddr(FRACTAL_HALF);
    bL1.SetSize(K * N);

    /* Cube input buffers. */
    AscendC::LocalTensor<half> aL0A;
    aL0A.SetTPosition(AscendC::TPosition::A2);
    aL0A.SetAddr(0);
    aL0A.SetSize(M * K);

    AscendC::LocalTensor<half> bL0B;
    bL0B.SetTPosition(AscendC::TPosition::B2);
    bL0B.SetAddr(0);
    bL0B.SetSize(K * N);

    /* Cube accumulator. */
    AscendC::LocalTensor<float> cL0C;
    cL0C.SetTPosition(AscendC::TPosition::CO1);
    cL0C.SetAddr(0);
    cL0C.SetSize(M * N);

    /* UB working buffers. */
    AscendC::LocalTensor<half> aUb;
    aUb.SetTPosition(AscendC::TPosition::VECIN);
    aUb.SetAddr(0);
    aUb.SetSize(M * K);

    AscendC::LocalTensor<float> cUb;
    cUb.SetTPosition(AscendC::TPosition::VECOUT);
    cUb.SetAddr(FRACTAL_FP32);
    cUb.SetSize(M * N);

    AscendC::LocalTensor<float> sumUb;
    sumUb.SetTPosition(AscendC::TPosition::VECOUT);
    sumUb.SetAddr(FRACTAL_FP32 * 2);
    sumUb.SetSize(M * N);

    /* Stage A through L1 as the hardware requires. */
    AscendC::DataCopy(aL1, aGlobal, M * K);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::LoadData(aL0A, aL1, M * K);

    /* [3] No hardware path from global memory straight into L0B. */
    AscendC::DataCopy(bL0B, bGlobal, K * N);

    /* [1] Mmad's left operand must come from L0A, not UB. */
    AscendC::Mmad(cL0C, aUb, bL0B, M);

    /* [4] Fixpipe drains the L0C accumulator; bL1 is in L1. */
    AscendC::Fixpipe(cUb, bL1, M * N);

    /* [2] The vector unit addresses UB only; aL1 lives in L1. */
    AscendC::SetFlag<AscendC::HardEvent::FIX_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_V>(EVENT_ID1);
    AscendC::Add(sumUb, cUb, aL1, M * N);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
    AscendC::DataCopy(cGlobal, sumUb, M * N);
}
