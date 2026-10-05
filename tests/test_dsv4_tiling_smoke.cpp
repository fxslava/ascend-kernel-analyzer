#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void dsv4_tiling_smoke(GM_ADDR src, GM_ADDR dst) {
    TCubeTiling cube{};
    TPipe pipe;
    TQue<TPosition::VECIN, 2> input;
    GlobalTensor<half> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(src), 128);
    pipe.InitBuffer(input, 2, 256);
    if (ASCEND_IS_AIV) {
        auto tensor = input.AllocTensor<half>();
        DataCopy(tensor, gm, 128);
        input.EnQue(tensor);
        auto ready = input.DeQue<half>();
        input.FreeTensor(ready);
    }
}
