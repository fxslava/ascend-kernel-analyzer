// Host-mode shims that let a stock Clang -cc1 parse Ascend C translation
// units for AST extraction (-fsyntax-only -ast-dump=json).  No code is ever
// generated from this configuration, so the mappings only need to preserve
// the *structure* the extractor reads back:
//
//   * __global__   -> `used` attribute: a real attribute the AST JSON keeps,
//                     which marks kernel entry points after expansion.
//   * __gm__/__ubuf__ & friends -> address_space attributes so raw pointer
//                     declarations still parse and the space stays visible in
//                     the desugared type.
//   * __ascend_core_is_aic/__ascend_core_is_aiv -> extern sentinels so both
//     arms of `if (ASCEND_IS_AIC)` survive into the AST as runtime branches
//     (heterogeneous MIX kernels need both regions).
#pragma once

#include <cstddef>
using std::size_t;

#if defined(__clang__) && defined(__has_attribute)
#define ASCEND_PREFIX_HAS_ATTR 1
#else
#define ASCEND_PREFIX_HAS_ATTR 0
#endif

#define __global__ __attribute__((used))
#define __aicore__
#define __host_aicore__
#define __device_aicore__

// Address-space qualifiers for on/off-core SRAM.  __gm__ expands to nothing:
// an address_space attribute here breaks const_cast patterns the CANN headers
// use, and GM-ness is structurally recoverable anyway (kernel entry pointer
// parameters are GM by Ascend C convention).
#define __gm__
#define __ubuf__ __attribute__((address_space(2)))
#define __cbuf__ __attribute__((address_space(3)))
#define __ca__ __attribute__((address_space(4)))
#define __cb__ __attribute__((address_space(5)))
#define __cc__ __attribute__((address_space(6)))
#define __bt__ __attribute__((address_space(7)))
#define __fbuf__ __attribute__((address_space(8)))
#define __local_mem__ __attribute__((address_space(9)))

// Hardware pipe enum.  In aicore mode the compiler provides `pipe_t` and the
// PIPE_* constants as builtins; the AST only needs their names (values are
// never inspected), so a plain enum restores parsability in host mode.
typedef enum pipe_t {
    PIPE_S = 0,
    PIPE_V = 1,
    PIPE_M = 2,
    PIPE_MTE1 = 3,
    PIPE_MTE2 = 4,
    PIPE_MTE3 = 5,
    PIPE_FIX = 6,
    PIPE_ALL = 7,
    PIPE_NUM = 8
} pipe_t;

// Scalar half type: a language builtin in aicore mode.  _Float16 parses
// everywhere (including return-by-value on x86) and carries the right width.
typedef _Float16 half;
typedef unsigned short bfloat16_t;

// Hardware handles the compiler provides natively on device.
typedef unsigned int event_t;
typedef int mem_t;
typedef int mem_dsb_t;
typedef int atomic_op_t;
typedef int atomic_type_t;
typedef int addr_cal_mode_t;
enum class pad_t { PAD_NONE = 0 };

// Fixpipe quantisation mode: a compiler-provided scoped enum on device.  The
// enumerator list mirrors every QuantMode_t:: name the CANN headers reference;
// values are irrelevant to AST extraction.
enum class QuantMode_t {
    N = 0,
    NoQuant,
    DEQF16, F322BF16, F322F16, QF322B8, QF322BF16, QF322F16,
    QF322F32, QF322FP8, QF322HIF8, QS322BF16, REQ8,
    VDEQF16, VQF322B8, VQF322BF16, VQF322F16, VQF322F32,
    VQF322FP8, VQF322HIF8, VQS322BF16, VREQ8,
    QF322B8_PRE, QF322BF16_PRE, QF322F16_PRE, QF322F32_PRE,
    QF322FP8_PRE, QF322HIF8_PRE, QS322BF16_PRE,
    VQF322B8_PRE, VQF322BF16_PRE, VQF322F16_PRE, VQF322F32_PRE,
    VQF322FP8_PRE, VQF322HIF8_PRE, VQS322BF16_PRE,
};

// SIMD calling-convention / pipe-annotation keywords (function-like: they
// carry a pipe letter argument, e.g. __inout_pipe__(S)).
#define __simd_callee__
#define __inout_pipe__(...)
#define __in_pipe__(...)
#define __out_pipe__(...)
#define __sync_alias__
#define __check_sync_alias__

// Reduction ordering handle (scoped enum on device).
enum class Order_t { ONLY_INDEX = 0, ONLY_VALUE, INDEX_VALUE, VALUE_INDEX };

// Cache-maintenance handle enums (scoped like QuantMode_t on device).
enum class dcci_dst_t { CACHELINE_OUT = 0, CACHELINE_ALL };
enum class cache_line_t { ENTIRE_DATA_CACHE = 0, SINGLE_CACHE_LINE };

// Raw cross-pipe event handoff builtins (templated on the pipe pair).
template <pipe_t Src, pipe_t Dst> void SetFlagInternal(event_t evt);
template <pipe_t Src, pipe_t Dst> void WaitFlagInternal(event_t evt);

template <class T> T sbitset0(T bits, int bit);
template <class T> T sbitset1(T bits, int bit);

extern "C" {
// Scalar builtins referenced unguarded by the CANN headers.
unsigned long long get_ctrl(void);
void set_ctrl(unsigned long long ctrl);
void set_vector_mask(unsigned long long mask0, unsigned long long mask1);
void set_mask_norm(void);
void set_mask_count(void);
void set_atomic_none(void);
void pipe_barrier(pipe_t pipe);
unsigned long get_rsvd_cnt(...);
unsigned long get_imm(...);
unsigned long get_block_idx(...);
unsigned long get_subblockdim(...);
unsigned long get_arch_ver(void);
unsigned long get_pc(void);
unsigned long get_block_num(void);
unsigned long get_coreid(void);
unsigned long get_subblockid(void);
unsigned long get_icache_prl_st(void);
unsigned long get_max_min_cnt(void);
unsigned long get_st_atomic_cfg(void);
unsigned long get_vms4_sr(void);
void __ib_set_stub(...);
void __ib_wait_stub(...);
void __sync_all_stub(...);
void set_flag(pipe_t src, pipe_t dst, event_t evt);
void wait_flag(pipe_t src, pipe_t dst, event_t evt);
void hset_flag(pipe_t src, pipe_t dst, event_t evt, mem_t mem, bool isVirtual);
void hwait_flag(pipe_t src, pipe_t dst, event_t evt, mem_t mem, bool isVirtual);
void dsb(mem_dsb_t barrier);
void dcci(...);
void set_l1_3d_size(...);
void set_padding(...);
void set_aipp_spr_0(...);
void set_aipp_spr_1(...);
void set_aipp_spr_2(...);
void set_aipp_spr_3(...);
void set_aipp_spr_4(...);
void set_aipp_spr_5(...);
void set_aipp_spr_6(...);
void set_aipp_spr_7(...);
void set_aipp_spr_8(...);
void set_aipp_spr_9(...);
void set_aipp_spr_10(...);
void set_aipp_spr_11(...);
void set_aipp_spr_12(...);
void set_aipp_spr_13(...);
void set_aipp_spr_14(...);
void set_aipp_spr_15(...);
void set_aipp_spr_16(...);
void set_aipp_spr_17(...);
void set_aipp_spr_18(...);
void set_aipp_spr_19(...);
void set_aipp_spr_20(...);
void set_aipp_spr_21(...);
void set_aipp_spr_22(...);
void set_aipp_spr_23(...);
void set_aipp_spr_24(...);
void set_fpc(...);
void copy_ubuf_to_gm(...);
void copy_gm_to_ubuf(...);
void copy_cbuf_to_gm(...);
void copy_gm_to_cbuf(...);
void ffts_cross_core_sync(...);
void wait_flag_dev(...);
void vreducev2(...);
void vgather(...);
void st_dev(...);
void ld_dev(...);
void vmuls(...);
void vmax(...);
long get_acc_val(...);
void vld_va_reg(...);
void scatter_vnchwconv_b32(...);
void scatter_vnchwconv_b16(...);
void scatter_vnchwconv_b8(...);
void set_va_reg_sb(...);
void vcopy(...);

// Vector-argument register aliases the compiler provides on device.
enum {
    EVENT_ID0 = 0, EVENT_ID1 = 1, EVENT_ID2 = 2, EVENT_ID3 = 3,
    EVENT_ID4 = 4, EVENT_ID5 = 5, EVENT_ID6 = 6, EVENT_ID7 = 7,
};
extern const unsigned long long VA0;
extern const unsigned long long VA1;
extern const unsigned long long VA2;
extern const unsigned long long VA3;
extern const unsigned long long H128;
extern const unsigned long long L128;

// Remaining scalar / vector / matrix intrinsics the headers call unguarded.
// Declared variadic: extraction only needs name resolution, not signatures.
namespace bisheng {
namespace cce {
void metrics_prof_start();
void metrics_prof_stop();
} // namespace cce
} // namespace bisheng
void trap(void);
unsigned long clz(...);
unsigned long sflbits(...);
unsigned long bcnt0(...);
unsigned long bcnt1(...);
unsigned long inc(...);
unsigned long dec(...);
void dc_preload(...);
void preload(...);
void mad_sp(...);
void vector_dup(...);
void vtranspose(...);
void vadd(...);
void vand(...);
void vor(...);
void vbrcb(...);
void vmulconv_f162s8(...);
void vmulconv_f162u8(...);
void vcmpv_eq(...);
void vcmpvs_eq(...);
void vconv_deq(...);
float conv_f322f16o(...);
float conv_f322s32a(...);
float conv_f322s32c(...);
float conv_f322s32f(...);
float conv_f322s32r(...);
void copy_cbuf_to_bt(...);
void copy_cbuf_to_fbuf(...);
void copy_cbuf_to_gm(...);
void copy_gm_to_cbuf(...);
void copy_gm_to_cbuf_multi_nd2nz_b8(...);
void copy_gm_to_cbuf_multi_nd2nz_b16(...);
void copy_gm_to_cbuf_multi_nd2nz_b32s(...);
void copy_matrix_cc_to_gm(...);
void copy_ubuf_to_gm(...);
void copy_ubuf_to_ubuf(...);
void copy_gm_to_ubuf(...);
void create_cbuf_matrix(...);
void img2colv2_cbuf_to_ca(...);
void img2colv2_cbuf_to_ca_s4(...);
void img2colv2_cbuf_to_cb(...);
void load_cbuf_to_cb_sp(...);
void set_atomic_add(...);
void set_atomic_bf16(...);
void set_atomic_f16(...);
void set_atomic_f32(...);
void set_atomic_max(...);
void set_atomic_min(...);
void set_atomic_s16(...);
void set_atomic_s32(...);
void set_atomic_s8(...);
void set_cmpmask(...);
void set_data_exp_0(...);
void set_data_exp_1(...);
void set_data_exp_2(...);
void set_data_exp_3(...);
void set_deqscale(...);
void set_ffts_base_addr(...);
void set_fmatrix(...);
void set_fmatrix_b(...);
void set_l3d_rpt(...);
void set_mov_pad_val(...);
void set_nd_para(...);
void set_quant_pre(...);
void set_st_atomic_cfg(...);
unsigned long sff0(...);
unsigned long sff1(...);
void load_cbuf_to_ca(...);
void load_cbuf_to_ca_s4(...);
void load_cbuf_to_cb(...);
void load_cbuf_to_cb_s4(...);
void load_cbuf_to_cb_transpose_s4(...);
void vconv_f162f32(...);
void vconv_f162s16a(...);
void vconv_f162s16c(...);
void vconv_f162s16f(...);
void vconv_f162s16r(...);
void vconv_f162s16z(...);
void vconv_f162s32a(...);
void vconv_f162s32c(...);
void vconv_f162s32f(...);
void vconv_f162s32r(...);
void vconv_f162s32z(...);
void vconv_f162s4(...);
void vconv_f162s4a(...);
void vconv_f162s4c(...);
void vconv_f162s4f(...);
void vconv_f162s4r(...);
void vconv_f162s4z(...);
void vconv_f162s8(...);
void vconv_f162s8a(...);
void vconv_f162s8c(...);
void vconv_f162s8f(...);
void vconv_f162s8r(...);
void vconv_f162s8z(...);
void vconv_f162u8(...);
void vconv_f162u8a(...);
void vconv_f162u8c(...);
void vconv_f162u8f(...);
void vconv_f162u8r(...);
void vconv_f162u8z(...);
void vconv_bf162f32(...);
void vconv_bf162s32a(...);
void vconv_bf162s32c(...);
void vconv_bf162s32f(...);
void vconv_bf162s32r(...);
void vconv_bf162s32z(...);
void vconv_f322bf16a(...);
void vconv_f322bf16c(...);
void vconv_f322bf16f(...);
void vconv_f322bf16r(...);
void vconv_f322bf16z(...);
void vconv_f322f16(...);
void vconv_f322f16a(...);
void vconv_f322f16c(...);
void vconv_f322f16f(...);
void vconv_f322f16o(...);
void vconv_f322f16r(...);
void vconv_f322f16z(...);
void vconv_f322f32a(...);
void vconv_f322f32c(...);
void vconv_f322f32f(...);
void vconv_f322f32r(...);
void vconv_f322f32z(...);
void vconv_f322s16a(...);
void vconv_f322s16c(...);
void vconv_f322s16f(...);
void vconv_f322s16r(...);
void vconv_f322s16z(...);
void vconv_f322s32a(...);
void vconv_f322s32c(...);
void vconv_f322s32f(...);
void vconv_f322s32r(...);
void vconv_f322s32z(...);
void vconv_f322s64a(...);
void vconv_f322s64c(...);
void vconv_f322s64f(...);
void vconv_f322s64r(...);
void vconv_f322s64z(...);
void vconv_s162f16(...);
void vconv_s162f16a(...);
void vconv_s162f16c(...);
void vconv_s162f16f(...);
void vconv_s162f16r(...);
void vconv_s162f16z(...);
void vconv_s162f32(...);
void vconv_s322f32(...);
void vconv_s322f32a(...);
void vconv_s322f32c(...);
void vconv_s322f32f(...);
void vconv_s322f32r(...);
void vconv_s322f32z(...);
void vconv_s322s16(...);
void vconv_s322s64(...);
void vconv_s42f16(...);
void vconv_s642f32a(...);
void vconv_s642f32c(...);
void vconv_s642f32f(...);
void vconv_s642f32r(...);
void vconv_s642f32z(...);
void vconv_s642s32(...);
void vconv_s82f16(...);
void vconv_u82f16(...);

// Runtime core-type sentinels.  ASCEND_IS_AIC / ASCEND_IS_AIV expand to these
// (see bisheng_extractor.py), keeping both MIX branches in the AST.
extern const bool __ascend_core_is_aic;
extern const bool __ascend_core_is_aiv;
}
