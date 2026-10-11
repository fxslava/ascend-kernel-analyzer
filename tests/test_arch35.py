"""Ascend 950PR arch35 and CANN 9.2 regression cases."""

from pathlib import Path

import pytest

from ascend_analyzer import AnalyzerOptions, KernelAnalyzer
from ascend_analyzer.apis import lookup_api
from ascend_analyzer.diagnostics import Severity
from ascend_analyzer.hardware import HardwareModel, Pipe
from ascend_analyzer.hardware import PhysicalDomain
from ascend_analyzer.ir import ScopeKind

from conftest import analyze, analyze_body, codes_of, find


@pytest.mark.parametrize("chip", ["ascend950pr", "dav-3510"])
@pytest.mark.parametrize("event_id", range(4, 8))
def test_arch35_profile_and_unavailable_event_ids_are_fatal(chip, event_id):
    hw = HardwareModel.for_chip("arch35")
    assert hw.chip.name == "ascend950pr"
    assert hw.chip.max_event_id == 3
    assert hw.chip.ub_total_bytes == 256 * 1024
    assert hw.chip.max_usable_ub_bytes == 216 * 1024
    result = analyze_body(
        f"AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID{event_id});\n"
        f"AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID{event_id});",
        chip=chip,
    )
    bad = find(result, "AKA2003")
    assert len(bad) == 2
    assert all(d.severity is Severity.FATAL for d in bad)
    assert "unsupported on arch35" in bad[0].message


def test_arch35_tensor_allocation_ceiling_applies_without_simt_marker():
    result = analyze_body(
        """
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
        pipe.InitBuffer(ub, 224 * 1024);
        """,
        chip="ascend950pr",
    )
    assert "AKA1010" in codes_of(result)


@pytest.mark.parametrize("marker", ["asc_call_vf", "__simt_vf__"])
@pytest.mark.parametrize("allocation_kib, rejected", [(216, False), (217, True)])
def test_simt_datacache_reserves_32_kib_at_ub_boundary(marker, allocation_kib, rejected):
    call = "vf(local);" if marker == "__simt_vf__" else "asc_call_vf(local, 0);"
    helper = (
        "__simt_vf__ void vf(AscendC::LocalTensor<half> t) { asc_ballot(1); }"
        if marker == "__simt_vf__"
        else ""
    )
    source = f"""
    {helper}
    extern "C" __global__ __aicore__ void test_kernel() {{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
    pipe.InitBuffer(ub, {allocation_kib} * 1024);
    auto local = ub.GetWithOffset<half>(64, 0);
    {call}
    }}
    """
    result = analyze(source, chip="ascend950pr")
    bad = find(result, "AKA1010")
    assert bool(bad) is rejected
    if rejected:
        assert bad[0].severity is Severity.FATAL
        assert bad[0].details["required_bytes"] == 32 * 1024


def test_uncalibrated_arch35_timing_does_not_emit_perf_warnings():
    path = Path(__file__).parent / "kernels" / "nvfp4_pipelined_dequant.cpp"
    result = KernelAnalyzer(AnalyzerOptions(chip="ascend950pr")).analyze_file(path)
    assert "AKA4001" not in codes_of(result)
    assert any(key.startswith("perf_profile::") for key in result.artifacts)


def test_cann92_reduction_between_barriers_is_not_redundant():
    result = analyze_body(
        """
        AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
        AscendC::LocalTensor<float> a = ub.GetWithOffset<float>(64, 0);
        AscendC::LocalTensor<float> b = ub.GetWithOffset<float>(8, 256);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceDataBlock<AscendC::ReduceType::SUM, float, float>(
            b, a, 64, 1, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        """,
        chip="ascend950pr",
    )
    assert "AKA3005" not in codes_of(result)
    ops = result.unit.kernels[0].ops
    assert any(
        getattr(op, "name", "") == "ReduceDataBlock" and op.pipe is Pipe.V for op in ops
    )


def test_reduction_followed_by_pipe_v_barrier_has_zero_warnings():
    result = analyze_body(
        """
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
        pipe.InitBuffer(ub, 512);
        auto src = ub.GetWithOffset<float>(64, 0);
        auto dst = ub.GetWithOffset<float>(8, 256);
        AscendC::ReduceDataBlock<AscendC::ReduceType::SUM, float, float>(
            dst, src, 64, 1, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        """,
        chip="ascend950pr",
    )
    assert not [d for d in result.diagnostics if d.severity is Severity.WARNING]
    assert "AKA3005" not in codes_of(result)


def test_direct_get_with_offset_operand_has_ub_domain():
    result = analyze_body(
        """
        AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
        AscendC::ReduceDataBlock<AscendC::ReduceType::SUM, float, float>(
            ub.GetWithOffset<float>(8, 256),
            ub.GetWithOffset<float>(64, 0), 64, 1, 1, 1, 8);
        """,
        chip="ascend950pr",
    )
    kernel = result.unit.kernels[0]
    reduction = next(
        op for op in kernel.ops if getattr(op, "name", "") == "ReduceDataBlock"
    )
    assert all(
        kernel.tensors[a.tensor].domain.value == "UB" for a in reduction.args[:2]
    )


def test_member_and_parameter_global_tensor_stores_use_mte3():
    source = """
    #include "kernel_operator.h"
    class Writer {
      AscendC::GlobalTensor<half> output_;
    public:
      __aicore__ inline void Store(AscendC::LocalTensor<half> local) {
        AscendC::DataCopyPad(output_[0], local, 128);
      }
    };
    __aicore__ inline void StoreArg(AscendC::GlobalTensor<half> output,
                                   AscendC::LocalTensor<half> local) {
      AscendC::DataCopyPad(output[0], local, 128);
    }
    extern "C" __global__ __aicore__ void test_kernel(__gm__ half* gm) {
      AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
      AscendC::LocalTensor<half> local = ub.GetWithOffset<half>(128, 0);
      AscendC::GlobalTensor<half> output;
      output.SetGlobalBuffer(gm, 128);
      Writer writer;
      writer.Store(local);
      StoreArg(output, local);
    }
    """
    result = analyze(source, chip="ascend950pr")
    stores = [
        op
        for op in result.unit.kernels[0].ops
        if getattr(op, "name", "") == "DataCopyPad"
    ]
    assert len(stores) == 2
    assert all(op.pipe is Pipe.MTE3 for op in stores)
    assert "AKA2010" not in codes_of(result)


def test_member_global_tensor_initialized_from_gm_addr_stays_on_mte3():
    source = """
    class Writer {
      AscendC::GlobalTensor<half> output_;
    public:
      __aicore__ inline void Init(GM_ADDR gm) {
        output_.SetGlobalBuffer((__gm__ half*)gm, 128);
      }
      __aicore__ inline void Store(AscendC::LocalTensor<half> local) {
        AscendC::DataCopyPad(output_[0], local, 128);
      }
    };
    extern "C" __global__ __aicore__ void test_kernel(GM_ADDR gm) {
      AscendC::TPipe pipe;
      AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
      pipe.InitBuffer(ub, 256);
      auto local = ub.GetWithOffset<half>(128, 0);
      Writer writer;
      writer.Init(gm);
      writer.Store(local);
    }
    """
    result = analyze(source, chip="ascend950pr")
    stores = [op for op in result.unit.kernels[0].ops if getattr(op, "name", "") == "DataCopyPad"]
    assert len(stores) == 1 and stores[0].pipe is Pipe.MTE3
    assert "AKA2010" not in codes_of(result)


def test_gm_addr_passed_to_store_helper_stays_on_mte3():
    source = """
    __aicore__ inline void Store(GM_ADDR dst, AscendC::LocalTensor<half> src) {
      GM_ADDR target = dst;
      AscendC::DataCopyPad(target, src, 128);
    }
    extern "C" __global__ __aicore__ void test_kernel(GM_ADDR gm) {
      AscendC::TPipe pipe;
      AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
      pipe.InitBuffer(ub, 256);
      auto local = ub.GetWithOffset<half>(128, 0);
      Store(gm, local);
    }
    """
    result = analyze(source, chip="ascend950pr")
    stores = [op for op in result.unit.kernels[0].ops if getattr(op, "name", "") == "DataCopyPad"]
    assert len(stores) == 1 and stores[0].pipe is Pipe.MTE3
    assert "AKA2010" not in codes_of(result)


def test_vec_scope_register_microapi_has_no_false_memory_or_barrier_warning():
    result = analyze_body(
        """
        __VEC_SCOPE__ {
          AscendC::RegTensor<float> a;
          AscendC::RegTensor<float> b;
          AscendC::RegTensor<float> out;
          Reg::Add(out, a, b, 0xffffffffffffffffULL);
          AscendC::ReduceDataBlock<AscendC::ReduceType::SUM, float, float>(
              out, a, 1);
          AscendC::PipeBarrier<PIPE_V>();
        }
        """,
        chip="ascend950pr",
    )
    kernel = result.unit.kernels[0]
    assert any(s.kind is ScopeKind.VEC_SCOPE for s in kernel.scopes.values())
    assert all(
        kernel.tensors[n].domain is PhysicalDomain.REG for n in ("a", "b", "out")
    )
    assert {"AKA1009", "AKA3005"}.isdisjoint(codes_of(result))
    assert not [d for d in result.diagnostics if d.severity is Severity.WARNING]


def test_160_byte_get_with_offset_pitch_reaches_bank_conflict_verdict():
    views = "\n".join(
        f"auto a{i} = ub.GetWithOffset<half>(64, {i * 160});" for i in range(9)
    )
    result = analyze_body(
        "AscendC::TBuf<AscendC::TPosition::VECCALC> ub;\n"
        + views
        + "\nAscendC::Add(a4, a0, a8, 64);",
        chip="ascend950pr",
    )
    conflict = find(result, "AKA3006")
    assert conflict and conflict[0].details["delta_blocks"] == 40


def test_get_with_offset_arguments_keep_bank_offsets_in_hybrid_simt_helper():
    source = """
    __simt_vf__ void vector_kernel(AscendC::LocalTensor<float> lhs,
                                   AscendC::LocalTensor<float> rhs) {
      asc_ballot(1);
      AscendC::Add(rhs, lhs, rhs, 64);
    }
    extern "C" __global__ __aicore__ void test_kernel() {
      AscendC::TPipe pipe;
      AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
      pipe.InitBuffer(ub, 512);
      vector_kernel(ub.GetWithOffset<float>(64, 0),
                    ub.GetWithOffset<float>(64, 256));
    }
    """
    result = analyze(source, chip="ascend950pr")
    conflict = find(result, "AKA3006")
    assert len(conflict) == 1
    assert conflict[0].details["delta_blocks"] == 8
    assert result.artifacts["bank_conflict_coverage::test_kernel"]["evaluated"] == 1


def test_static_simt_cross_line_access_is_reported():
    source = """
    __simt_vf__ void load() {
      __gm__ double* src = (__gm__ double*)(4);
      double x = src[15];
    }
    extern "C" __global__ __aicore__ void test_kernel() {
      load();
    }
    """
    result = analyze(source, chip="ascend950pr")
    assert "AKA3007" in codes_of(result)


def test_register_tile_pressure_reports_spill_without_ub_domain_warning():
    declarations = "\n".join(f"AscendC::RegTile<float, 16, 16> r{i};" for i in range(9))
    use = "Reg::Add(" + ", ".join(f"r{i}" for i in range(9)) + ");"
    result = analyze_body(
        "__VEC_SCOPE__ {\n" + declarations + "\n" + use + "\n}",
        chip="ascend950pr",
    )
    assert "AKA3008" in codes_of(result)
    assert "AKA1009" not in codes_of(result)
    usage = result.artifacts["register_usage::test_kernel"]
    assert usage["peak_slots"] == 36


def test_simt_warp_and_simd_issue_contend_inside_one_loop():
    result = analyze_body(
        """
        AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
        auto a = ub.GetWithOffset<float>(64, 0);
        auto b = ub.GetWithOffset<float>(64, 256);
        for (int i = 0; i < 2; ++i) {
          asc_ballot(1);
          AscendC::Add(b, a, a, 64);
        }
        """,
        chip="ascend950pr",
    )
    assert "AKA3009" in codes_of(result)


def test_cann92_signatures_have_vector_engine_and_argument_roles():
    for name in (
        "ReduceDataBlock",
        "ReduceRepeat",
        "ReducePairElem",
        "Brcb",
        "Brcc",
        "DataCopyGather",
        "DataCopyScatter",
        "Reg::Add",
        "Reg::Mul",
        "Reg::Div",
        "Reg::Axpy",
        "Reg::Exp",
        "asc_loadalign",
        "asc_storealign",
    ):
        spec = lookup_api(name)
        assert spec is not None and spec.pipe is Pipe.V
        assert spec.engine == "SIMD_VECTOR" and spec.latency_cycles > 0
    assert lookup_api("ReduceDataBlock").param_at(6).name == "srcRepStride"
    assert lookup_api("ReduceRepeat").param_at(7).name == "order"
    assert lookup_api("Brcb").param_at(3).name == "repeatParams"
    for name in ("ReduceDataBlock", "ReduceRepeat", "ReducePairElem", "Brcb"):
        spec = lookup_api(name)
        assert spec.params[0].allowed == frozenset({PhysicalDomain.UB})
        assert spec.params[1].allowed == frozenset({PhysicalDomain.UB})
    assert lookup_api("Reg::ReduceDataBlock").params[0].allowed == frozenset(
        {PhysicalDomain.REG}
    )
    for name in ("asc_ballot", "asc_any", "asc_all", "asc_shfl", "asc_shfl_xor"):
        assert lookup_api(name).engine == "SIMT_WARP"


def test_cann92_gather_vecindex_sort_extract_are_pipe_v():
    for name in ("Gather", "CreateVecIndex", "Sort", "Extract"):
        spec = lookup_api(name)
        assert spec is not None and spec.pipe is Pipe.V
        assert spec.engine == "SIMD_VECTOR" and spec.latency_cycles > 0
    gather = lookup_api("Gather")
    assert gather.params[0].allowed == frozenset({PhysicalDomain.UB})
    assert gather.params[1].allowed == frozenset({PhysicalDomain.UB})
    assert lookup_api("asc_vf_call").engine == "SIMT_WARP"


def test_cann92_topk_stages_between_barriers_are_not_redundant():
    result = analyze_body(
        """
        AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
        auto keys = ub.GetWithOffset<float>(64, 0);
        auto order = ub.GetWithOffset<float>(64, 256);
        auto top = ub.GetWithOffset<float>(8, 512);
        AscendC::CreateVecIndex(order);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sort<float, true>(keys, order, 4, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Extract<float>(top, keys, 2, 8);
        AscendC::Gather(top, keys, order, 0, 8);
        AscendC::PipeBarrier<PIPE_V>();
        """,
        chip="ascend950pr",
    )
    assert "AKA3005" not in codes_of(result)
    names = {getattr(op, "name", "") for op in result.unit.kernels[0].ops}
    assert {"CreateVecIndex", "Sort", "Extract", "Gather"} <= names


def _vf_dispatch_source(marker: str = "__simt_vf__") -> str:
    return f"""
    #include "kernel_operator.h"
    {marker} __launch_bounds__(1024) inline void scale_rows(
        __ubuf__ float *out, __gm__ const float *in, uint32_t base)
    {{
        float bias = in[8];
        for (uint32_t i = threadIdx.x; i < 64; i += blockDim.x) {{
            out[i] = in[base] * 2.0f + bias + i;
        }}
    }}
    extern "C" __global__ __aicore__ void test_kernel(
        GM_ADDR gmIn, GM_ADDR gmOut, GM_ADDR tiling)
    {{
        __gm__ const uint32_t *tile = (__gm__ const uint32_t *)(tiling);
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECCALC, 2> q;
        pipe.InitBuffer(q, 2, 256);
        AscendC::LocalTensor<float> stage = q.AllocTensor<float>();
        AscendC::PipeBarrier<PIPE_V>();
        asc_vf_call<scale_rows>(
            dim3(1024),
            reinterpret_cast<__ubuf__ float *>(stage.GetPhyAddr()),
            reinterpret_cast<__gm__ const float *>(gmIn),
            tile[0]);
        AscendC::PipeBarrier<PIPE_V>();
        q.EnQue(stage);
    }}
    """


def test_vf_dispatch_barriers_are_not_redundant():
    result = analyze(_vf_dispatch_source(), chip="ascend950pr")
    assert "AKA3005" not in codes_of(result)
    assert not [d for d in result.diagnostics if d.severity is Severity.WARNING]
    dispatch = [
        op
        for op in result.unit.kernels[0].ops
        if getattr(op, "name", "") == "asc_vf_call"
    ]
    assert len(dispatch) == 1 and dispatch[0].pipe is Pipe.V
    assert "stage" in dispatch[0].writes
    assert "gmIn" in dispatch[0].reads


def test_vf_dispatch_inlines_callee_and_records_gm_access():
    source = _vf_dispatch_source()
    result = analyze(source, chip="ascend950pr")
    kernel = result.unit.kernels[0]
    # The callee body was walked inside the dispatch trace.
    assert "scale_rows" in kernel.inlined
    gm_access = [
        op for op in kernel.ops if getattr(op, "name", "") == "SIMT_GM_ACCESS"
    ]
    assert gm_access and "gmIn" in gm_access[0].reads


def test_simd_vf_helper_is_traced_when_called():
    source = """
    #include "kernel_operator.h"
    __simd_vf__ inline void pack(AscendC::LocalTensor<half> dst,
                                 AscendC::LocalTensor<half> src) {
      AscendC::Add(dst, src, src, 64);
    }
    extern "C" __global__ __aicore__ void test_kernel() {
      AscendC::TPipe pipe;
      AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
      pipe.InitBuffer(ub, 512);
      auto a = ub.GetWithOffset<half>(64, 0);
      auto b = ub.GetWithOffset<half>(64, 256);
      pack(b, a);
    }
    """
    result = analyze(source, chip="ascend950pr")
    kernel = result.unit.kernels[0]
    assert "pack" in kernel.inlined
    assert any(getattr(op, "name", "") == "Add" for op in kernel.ops)
    assert "AKA1009" not in codes_of(result)


def test_getphyaddr_ubuf_pointer_alias_keeps_source_layout():
    source = """
    #include "kernel_operator.h"
    __simt_vf__ inline void touch(__ubuf__ float *p) {
      p[0] = p[0] * 2.0f;
    }
    extern "C" __global__ __aicore__ void test_kernel(__gm__ half* gm) {
      AscendC::TPipe pipe;
      AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
      pipe.InitBuffer(ub, 512);
      AscendC::LocalTensor<float> xf32 = ub.GetWithOffset<float>(64, 0);
      __ubuf__ float* raw = reinterpret_cast<__ubuf__ float*>(xf32.GetPhyAddr());
      AscendC::Add(xf32, xf32, xf32, 64);
      AscendC::PipeBarrier<PIPE_V>();
      touch(raw);
    }
    """
    result = analyze(source, chip="ascend950pr")
    assert {"AKA1009", "AKA3002", "AKA1003"}.isdisjoint(codes_of(result))
    kernel = result.unit.kernels[0]
    raw = kernel.tensors["raw"]
    assert raw.domain is PhysicalDomain.UB
    assert "alias" in raw.origin
    assert raw.byte_offset is not None
    # The __ubuf__ alias parameter of the SIMT helper binds back to the
    # caller's tensor, so the deliberate overlap is not re-reported.
    assert "touch" in kernel.inlined


# ---------------------------------------------------------------------------
# Stage-1 fleet acceptance fixtures (KER-01 / KER-05, revision 3)
# ---------------------------------------------------------------------------

_STAGE1_KERNELS = (
    "ker01_mhc_pre_norm_sinkhorn_3.cpp",
    "ker05_mxfp4_expert_gemv_3.cpp",
)


@pytest.mark.parametrize("source", _STAGE1_KERNELS)
def test_stage1_fleet_kernels_audit_clean_without_pragmas(source):
    """The acceptance gate for the arch35 SIMT/SIMD fleet.

    Both fixtures exercise ``asc_vf_call<Func>(dim3, ...)`` dispatch with the
    mandatory ``PipeBarrier<PIPE_V>`` pair, ``__simt_vf__``/``__launch_bounds__``
    helpers and ``reinterpret_cast<__ubuf__ T*>(t.GetPhyAddr())`` aliases, and
    carry neither ``@ascend-ignore`` nor ``@ascend-reuse-group`` annotations.
    They must audit with zero fatal findings and zero warnings.
    """
    path = Path(__file__).parent / "kernels" / source
    text = path.read_text(encoding="utf-8")
    assert "@ascend-ignore" not in text
    assert "@ascend-reuse-group" not in text
    result = KernelAnalyzer(AnalyzerOptions(chip="ascend950pr")).analyze_file(path)
    assert "AKA3005" not in codes_of(result)
    assert not [
        d for d in result.diagnostics if d.severity is Severity.FATAL
    ]
    assert not [
        d for d in result.diagnostics if d.severity is Severity.WARNING
    ]
    # The dispatch itself must be visible to the barrier checker.
    ops = result.unit.kernels[0].ops
    assert any(getattr(op, "name", "") == "asc_vf_call" for op in ops)


def test_stage1_ker05_static_ub_footprint_is_34_1_kib():
    """KER-05's static UB reservation (xQue + yQue + scratch) is 34944 B."""
    path = Path(__file__).parent / "kernels" / "ker05_mxfp4_expert_gemv_3.cpp"
    result = KernelAnalyzer(AnalyzerOptions(chip="ascend950pr")).analyze_file(path)
    buffer_bytes = sum(result.unit.kernels[0].buffer_sizes.values())
    assert buffer_bytes == 34944  # 34.125 KiB, rendered as 34.1 KiB
