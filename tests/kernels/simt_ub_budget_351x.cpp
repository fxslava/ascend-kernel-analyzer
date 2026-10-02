/*
 * 351x SIMD/SIMT Unified Buffer budget fixture (FATAL AKA1010).
 *
 * On Ascend 351x the Unified Buffer is strictly partitioned:
 *
 *   DataCache = 256 KiB - StaticMem - DynamicMem - 8 KiB (compiler reserved)
 *
 * and whenever the SIMT vector path is in use (``__simt_vf__``,
 * ``__simt_callee__`` or ``asc_call_vf`` appears) the runtime requires
 * DataCache >= 32 KiB, i.e. at most 216 KiB of tensor allocation.
 *
 * This kernel allocates 220 KiB (2 x 110 KiB) of UB tensors and calls the
 * SIMT VF entry point, leaving only 28 KiB of DataCache - below the 32 KiB
 * hardware floor, which corrupts memory at run time.
 *
 * Analysed against --chip ascend351x this reports exactly one FATAL AKA1010.
 * The shared regression suite runs the default 910b profile, where the same
 * 220 KiB layout simply overflows the smaller 192 KiB UB (AKA1001) instead.
 *
 * @ascend-expect: AKA1001
 * @ascend-expect-fatal: 1
 */
#include "kernel_operator.h"

constexpr uint32_t BIG_ELEMS = 56320;                 /* 110 KiB of half data */
constexpr uint32_t BIG_BYTES = BIG_ELEMS * sizeof(half);
constexpr uint32_t OFF_B = BIG_BYTES;                 /* second 110 KiB tile  */

extern "C" __global__ __aicore__ void simt_ub_budget_351x(__gm__ half* gm)
{
    AscendC::GlobalTensor<half> g;
    g.SetGlobalBuffer(gm, 2 * BIG_ELEMS);

    AscendC::LocalTensor<half> bigA;
    bigA.SetTPosition(AscendC::TPosition::VECCALC);
    bigA.SetAddr(0);
    bigA.SetSize(BIG_ELEMS);

    AscendC::LocalTensor<half> bigB;
    bigB.SetTPosition(AscendC::TPosition::VECCALC);
    bigB.SetAddr(OFF_B);
    bigB.SetSize(BIG_ELEMS);

    AscendC::DataCopy(bigA, g[0], BIG_ELEMS);
    AscendC::DataCopy(bigB, g[BIG_ELEMS], BIG_ELEMS);

    /* SIMT vector-function calls make the DataCache partition mandatory. */
    asc_call_vf(bigA, 0);
    asc_call_vf(bigB, 1);

    AscendC::DataCopy(g[0], bigA, BIG_ELEMS);
    AscendC::DataCopy(g[BIG_ELEMS], bigB, BIG_ELEMS);
}
