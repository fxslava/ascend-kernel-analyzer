// Stub Ascend C facade for BiSheng/Clang AST extraction (-fsyntax-only
// -ast-dump=json).  No code is generated from this configuration; the stub
// exists so the real C++ parser can typecheck kernel sources whose API
// surface spans several CANN versions, while template monomorphisation,
// `auto` deduction and overload resolution all remain the real compiler's
// job.  Member bodies are deliberately undefined: only declarations are
// needed to parse call sites, and undefined bodies keep extraction honest
// (nothing from this header ever executes).
//
// The API surface mirrors what the analyzer models in apis.py plus the
// constructs the fixture corpus exercises.  When a real CANN include tree is
// available and compatible, `bisheng_extractor` prefers it (header
// mode "real"); this stub is the always-available baseline (mode "stub").
#pragma once

#include <cstdint>
#include <cstddef>

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __global__
#define __global__ __attribute__((used))
#endif

typedef _Float16 half;
typedef unsigned short bfloat16_t;

// Sub-byte operand format handles used in low-level Cube casts.
struct fp4x2_e2m1_t { unsigned char bits; };
struct fp4x2_e1m2_t { unsigned char bits; };
struct float4_e2m1x2_t { unsigned char bits[1]; };
struct float4_e1m2x2_t { unsigned char bits[1]; };

// Kernel entry parameters are declared GM_ADDR in host code.
#define GM_ADDR __gm__ void*

namespace AscendC {

enum class TPosition {
    GM = 0, A1, B1, C1, A2, B2, CO1, CO2,
    VECIN, VECOUT, VECCALC, LCM, SHM, TSCM, SPM, C2, C2PIPE2GM, MAX,
};

enum class HardEvent {
    MTE1_M = 0, MTE1_V, MTE1_MTE2, MTE1_MTE3, MTE1_FIX,
    M_MTE1, M_MTE2, M_MTE3, M_FIX,
    MTE2_M, MTE2_V, MTE2_MTE1, MTE2_MTE3, MTE2_FIX,
    MTE3_M, MTE3_V, MTE3_MTE1, MTE3_MTE2, MTE3_FIX,
    V_M, V_MTE1, V_MTE2, V_MTE3, V_FIX,
    FIX_M, FIX_MTE1, FIX_MTE2, FIX_MTE3, FIX_V,
    MAX,
};

enum class RoundMode {
    CAST_NONE = 0, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_ODD,
};

enum class MemoryT { L0A = 0, L0B, L0C, BIAS, UB, L1 };

template <typename T>
class GlobalTensor {
public:
    GlobalTensor() = default;
    __aicore__ inline void SetGlobalBuffer(__gm__ T* address, uint32_t len);
    __aicore__ inline void SetGlobalBuffer(__gm__ T* address);
    __aicore__ inline GlobalTensor<T> operator[](uint32_t index) const;
    __aicore__ inline __gm__ T* GetPhyAddr() const;
};

template <typename T>
class LocalTensor {
public:
    __aicore__ inline void SetTPosition(TPosition pos);
    __aicore__ inline void SetAddr(uint32_t address);
    __aicore__ inline void SetSize(uint32_t size);
    __aicore__ inline TPosition GetTPosition() const;
    __aicore__ inline uint32_t GetPhyAddr() const;
    __aicore__ inline LocalTensor<T> operator[](uint32_t index) const;
    __aicore__ inline void SetValue(uint32_t index, T value) const;
    __aicore__ inline T GetValue(uint32_t index) const;
    template <typename U>
    __aicore__ inline LocalTensor<U> ReinterpretCast() const;
};

template <TPosition tPosition = TPosition::LCM>
class TBuf {
public:
    template <typename T>
    __aicore__ inline LocalTensor<T> Get();
    template <typename T>
    __aicore__ inline LocalTensor<T> Get(uint32_t len);
};

template <TPosition tPosition = TPosition::VECIN, uint32_t depth = 1>
class TQue {
public:
    template <typename T>
    __aicore__ inline LocalTensor<T> AllocTensor();
    template <typename T>
    __aicore__ inline void AllocTensor(LocalTensor<T>& tensor);
    template <typename T>
    __aicore__ inline void FreeTensor(LocalTensor<T>& tensor);
    template <typename T>
    __aicore__ inline void EnQue(const LocalTensor<T>& tensor);
    template <typename T>
    __aicore__ inline LocalTensor<T> DeQue();
    __aicore__ inline int32_t GetTensorCountInQue();
};

class TPipe {
public:
    template <class T>
    __aicore__ inline void InitBuffer(T& que, uint8_t num, uint32_t len);
    template <TPosition pos>
    __aicore__ inline void InitBuffer(TBuf<pos>& buf, uint32_t len);
};

// -- synchronisation ---------------------------------------------------------

template <HardEvent event>
__aicore__ inline void SetFlag(uint8_t eventId);
template <HardEvent event>
__aicore__ inline void WaitFlag(uint8_t eventId);

__aicore__ inline void PipeBarrier(pipe_t pipe);

// -- data movement -----------------------------------------------------------

template <typename T, typename U>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<U>& src,
                                uint32_t count);
template <typename T, typename U>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
                                uint32_t count);
template <typename T, typename U>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src,
                                uint32_t count);
template <typename T, typename U>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, __gm__ U* src, uint32_t count);
template <typename T, typename U>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dst, __gm__ U* src, uint32_t count);
template <typename T, typename U>
__aicore__ inline void DataCopy(__gm__ T* dst, const LocalTensor<U>& src, uint32_t count);

// -- vector compute ----------------------------------------------------------

template <typename T>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint32_t count);
template <typename T, typename U>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const U scalarValue, uint32_t count);
template <typename T>
__aicore__ inline void Sub(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint32_t count);
template <typename T, typename U>
__aicore__ inline void Sub(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const U scalarValue, uint32_t count);
template <typename T>
__aicore__ inline void Mul(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint32_t count);
template <typename T, typename U>
__aicore__ inline void Mul(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const U scalarValue, uint32_t count);
template <typename T>
__aicore__ inline void And(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint32_t count);
template <typename T, typename U>
__aicore__ inline void And(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const U scalarValue, uint32_t count);
template <typename T>
__aicore__ inline void Or(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                          const LocalTensor<T>& src1, uint32_t count);
template <typename T, typename U>
__aicore__ inline void Or(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                          const U scalarValue, uint32_t count);
template <typename T, typename U>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                 const U scalarValue, uint32_t count);
template <typename T, typename U>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                  const U scalarValue, uint32_t count);
template <typename T, typename U>
__aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src,
                            const RoundMode& roundMode, uint32_t count);

// -- cube compute ------------------------------------------------------------

struct MmadParams {
    __aicore__ MmadParams() {}
    __aicore__ MmadParams(const uint16_t mIn, const uint16_t nIn, const uint16_t kIn,
                          const bool isBiasIn, const int32_t fmOffsetIn,
                          const bool enSsparseIn, const bool enWinogradAIn,
                          const bool enWinogradBIn);
};

struct FixpipeParamsV220 {
    __aicore__ FixpipeParamsV220() {}
    __aicore__ FixpipeParamsV220(const uint16_t nSizeIn, const uint16_t mSizeIn,
                                 const uint16_t srcStrideIn, const uint32_t dstStrideIn,
                                 const bool reluEnIn);
};

struct mmad_t {
    struct control_t {
        unsigned unit_flag_ctrl;
        bool gemv_ctrl;
        bool BTbuf_ctrl;
        bool zero_Cmatrix_ctrl;
    };
    struct shape_t {
        __aicore__ shape_t(uint16_t m, uint16_t k, uint16_t n);
    };
};

template <typename T, typename U, typename S>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
                            const LocalTensor<S>& filter, const MmadParams& mmadParams);

template <typename T, typename U>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
                               const FixpipeParamsV220& intriParams);
template <typename T, typename U>
__aicore__ inline void Fixpipe(__gm__ T* dst, const LocalTensor<U>& src,
                               const FixpipeParamsV220& intriParams);
template <typename T, typename U>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
                               const FixpipeParamsV220& intriParams);

// Low-level Cube staging intrinsics (shape-level casts in real kernels).
__aicore__ inline void load_cbuf_to_ca_s4(__ca__ fp4x2_e2m1_t* dst, __cbuf__ fp4x2_e2m1_t* src,
                                          uint16_t dstGap, uint16_t srcGap, uint8_t dstRepeatSize,
                                          uint8_t srcRepeatSize, int16_t repeatTimes,
                                          uint16_t dstRepeatStride, bool transpose);
__aicore__ inline void load_cbuf_to_cb_s4(__cb__ fp4x2_e1m2_t* dst, __cbuf__ fp4x2_e1m2_t* src,
                                          uint16_t dstGap, uint16_t srcGap, uint8_t dstRepeatSize,
                                          uint8_t srcRepeatSize, int16_t repeatTimes,
                                          uint16_t dstRepeatStride, bool transpose);
__aicore__ inline void mad_mx(__cc__ float* dstC, uint64_t dstGap,
                              __ca__ float4_e2m1x2_t* srcA, uint64_t srcAGap,
                              __cb__ float4_e1m2x2_t* srcB, uint64_t srcBGap,
                              const mmad_t::shape_t& shape, const mmad_t::control_t& control);

}  // namespace AscendC
