"""CANN 9.1.0 / Ascend 950PR (arch_3510) rule fixtures."""

from __future__ import annotations

import pytest

from ascend_analyzer.diagnostics import Severity
from ascend_analyzer.hardware import HardwareModel
from conftest import analyze, codes_of, find, make_kernel


def analyze_950(body: str, *, preamble: str = ""):
    return analyze(make_kernel(body, preamble), chip="Ascend950pr")


@pytest.mark.parametrize("position", ["VECIN", "VECOUT"])
def test_functional_vector_positions_are_accepted(position):
    result = analyze_950(
        f"""
        AscendC::LocalTensor<half> tile;
        tile.SetTPosition(AscendC::TPosition::{position});
        tile.SetAddr(0);
        tile.SetSize(256);
        """
    )
    assert "AKA1004" not in codes_of(result)


def test_flat_ub_position_is_rejected_on_950pr():
    result = analyze_950(
        """
        AscendC::LocalTensor<half> tile;
        tile.SetTPosition(AscendC::TPosition::UB);
        tile.SetAddr(0);
        tile.SetSize(256);
        """
    )
    diagnostic = next(item for item in result.diagnostics if item.code.value == "AKA1004")
    assert diagnostic.severity is Severity.FATAL
    assert diagnostic.details["valid_positions"] == ("VECIN", "VECOUT", "GM", "TSCM")


def test_datacache_exact_equation_and_cap():
    hardware = HardwareModel.for_chip("dav-c310")
    assert hardware.chip.name == "ascend950pr"
    assert hardware.simt_datacache_available(0) == 128 * 1024
    assert hardware.simt_datacache_available(221184) == 32768


def test_tscm_buffer_is_not_charged_to_the_ub_datacache_partition():
    result = analyze_950(
        """
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::TSCM> l1;
        pipe.InitBuffer(l1, 225280);
        """
    )
    assert "AKA1001" not in codes_of(result)
    assert result.unit.kernels[0].buffer_positions["l1"].value == "TSCM"


@pytest.mark.parametrize("allocation", [225280, 221216])
def test_datacache_underflow_and_ub_ceiling_are_fatal_aka1001(allocation):
    result = analyze_950(
        f"""
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECIN> ub;
        pipe.InitBuffer(ub, {allocation});
        """
    )
    diagnostic = next(item for item in result.diagnostics if item.code.value == "AKA1001")
    assert diagnostic.severity is Severity.FATAL
    assert diagnostic.details["static_plus_dynamic_bytes"] == allocation
    assert diagnostic.details["capacity_equation"] == (
        "min(262144 - static_bytes - dynamic_bytes - 8192, 131072)"
    )


def test_extern_ub_and_local_allocator_cannot_overlap():
    result = analyze_950(
        """
        LocalMemAllocator allocator;
        auto dynamic = allocator.Alloc<half>(1024);
        auto external = dynamic_array;
        """,
        preamble="extern __ubuf__ half dynamic_array[1024];",
    )
    findings = [
        item for item in result.diagnostics
        if "extern __ubuf__ arrays" in item.message
    ]
    assert len(findings) == 1
    assert findings[0].severity is Severity.FATAL


@pytest.mark.parametrize(
    "body",
    [
        "using SparseMatmulType = int;",
        "tiling.SetSparse(true);",
    ],
)
def test_structured_sparse_matmul_is_rejected(body):
    result = analyze_950(body)
    finding = next(item for item in result.diagnostics if item.code.value == "AKA4006")
    assert finding.severity is Severity.FATAL
    assert "MXMODE" in finding.remediation


def test_vf_call_accepts_documented_template_dim3_signature():
    result = analyze_950(
        "asc_vf_call<scale>(dim3{32, 1, 1}, data);",
        preamble="__simt_vf__ void scale(int* data) {}",
    )
    assert "AKA2012" not in codes_of(result)
    assert "AKA2011" not in codes_of(result)


def test_deprecated_vf_dispatch_has_migration_suggestion():
    result = analyze_950("asc_call_vf(data, 0);")
    finding = next(item for item in result.diagnostics if item.code.value == "AKA2011")
    assert finding.severity is Severity.WARNING
    assert finding.details["autofix"] == "asc_vf_call<funcPtr>(dim3{...}, args...)"


def test_vf_call_rejects_non_dim3_first_argument():
    result = analyze_950("asc_vf_call<scale>(32, data);")
    assert "AKA2012" in codes_of(result)


def test_vf_call_accepts_a_dim3_variable():
    result = analyze_950(
        """
        dim3 block{32, 1, 1};
        asc_vf_call<scale>(block, data);
        """,
        preamble="__simt_vf__ void scale(int* data) {}",
    )
    assert "AKA2012" not in codes_of(result)


@pytest.mark.parametrize("bound, valid", [(32, True), (33, False), (2048, True), (2080, False)])
def test_launch_bound_range_and_warp_multiple(bound, valid):
    result = analyze_950(
        f"""
        __simt_vf__ __launch_bounds__({bound}) void scale() {{}}
        asc_vf_call<scale>(dim3{{32, 1, 1}});
        """
    )
    assert ("AKA2012" not in codes_of(result)) is valid


def test_launch_bound_must_cover_dispatched_block_and_warns_on_register_budget():
    result = analyze_950(
        """
        __simt_vf__ __launch_bounds__(256) void scale() {}
        asc_vf_call<scale>(dim3{512, 1, 1});
        """
    )
    invalid_bounds = [
        item for item in result.diagnostics
        if item.code.value == "AKA2012" and "smaller than" in item.message
    ]
    assert len(invalid_bounds) == 1
    assert invalid_bounds[0].details["threads"] == 512
    spill = next(item for item in result.diagnostics if item.code.value == "AKA3008")
    assert spill.details["register_budget"] == 64


def matrix_body(api="asc_mmad", a="fp8_e4m3fn_t", b="fp8_e5m2_t", *, height="16", disabled="false", domain="__ca__"):
    return f"""
        __cc__ float* c = (__cc__ float*)0;
        {domain} {a}* lhs = ({domain} {a}*)0;
        __cb__ {b}* rhs = (__cb__ {b}*)0;
        {api}(c, lhs, rhs, {height}, 32, 16, 0, {disabled}, false, true);
    """


@pytest.mark.parametrize("api,a,b", [
    ("asc_mmad_mx", "fp4x2_e1m2_t", "fp4x2_e2m1_t"),
    ("asc_mmad_mx", "fp8_e4m3fn_t", "fp8_e5m2_t"),
    ("asc_mmad_sync", "fp8_e4m3fn_t", "fp8_e5m2_t"),
    ("asc_mmad", "hifloat8_t", "hifloat8_t"),
])
def test_audited_low_bit_matmul_types_are_accepted(api, a, b):
    result = analyze_950(matrix_body(api, a, b))
    assert not {"AKA1004", "AKA4011"}.intersection(codes_of(result))
    call = next(op for op in result.unit.kernels[0].api_calls() if op.name == api)
    assert call.reads == ("lhs", "rhs")
    assert call.writes == ("c",)
    assert call.args[3].value == 16


def test_fp4_requires_mx_matmul_and_gemv_must_not_be_disabled():
    result = analyze_950(matrix_body(a="fp4x2_e1m2_t", b="fp4x2_e2m1_t", height="1", disabled="true"))
    findings = find(result, "AKA4011")
    assert len(findings) == 2
    assert all(item.severity is Severity.FATAL for item in findings)
    assert any(item.details.get("disable_gemv") is True for item in findings)


def test_tscm_total_residency_is_limited_to_512_kib():
    result = analyze_950(
        """
        AscendC::LocalTensor<half> tile;
        tile.SetTPosition(AscendC::TPosition::TSCM);
        tile.SetAddr(0);
        tile.SetSize(300 * 1024);
        """
    )
    finding = next(item for item in result.diagnostics if item.code.value == "AKA4005")
    assert finding.details["capacity_bytes"] == 512 * 1024
    assert finding.details["allocated_bytes"] > finding.details["capacity_bytes"]


def test_tscm_partial_tile_staging_inside_loop_is_rejected():
    result = analyze_950(
        """
        AscendC::LocalTensor<half> tile;
        tile.SetTPosition(AscendC::TPosition::TSCM);
        tile.SetAddr(0);
        tile.SetSize(256);
        for (int i = 0; i < 2; ++i) {
            AscendC::DataCopy(tile, gm, 256);
        }
        """
    )
    assert any(
        item.code.value == "AKA4005"
        and "partial tiles" in item.message
        for item in result.diagnostics
    )


def test_matmul_signature_uses_declared_low_bit_operand_types():
    result = analyze_950(
        """
        fp4x2_e1m2_t* lhs;
        fp4x2_e2m1_t* rhs;
        asc_mmad(lhs, rhs);
        """
    )
    assert "AKA4011" in codes_of(result)


@pytest.mark.parametrize("target", ["Ascend950pr", "dav-c310", "arch_3510"])
def test_target_aliases_enforce_placement(target):
    from ascend_analyzer.hardware import TPosition
    hw = HardwareModel.for_chip(target)
    assert not hw.valid_allocation_position(TPosition.UB)
    assert hw.valid_allocation_position(TPosition.TSCM)
    assert "AKA1004" in codes_of(analyze(make_kernel("TBuf<TPosition::UB> b;"), chip=target))


def test_old_architecture_keeps_ub_allocation():
    assert "AKA1004" not in codes_of(analyze(make_kernel("TBuf<TPosition::UB> b;")))


@pytest.mark.parametrize("size,overflow", [(221184, False), (221216, True), (262144, True)])
def test_static_raw_ub_arrays_are_budgeted(size, overflow):
    result = analyze_950(f"__ubuf__ char storage[{size}];")
    assert ("AKA1001" in codes_of(result)) is overflow
    assert result.artifacts["ub_partition::test_kernel"]["static_bytes"] == size


def test_static_tpipe_and_dynamic_allocator_are_added_in_bytes():
    result = analyze_950("""
        constexpr int N = 16 * 1024;
        TPipe pipe;
        TBuf<TPosition::VECIN> b;
        pipe.InitBuffer(b, 200 * 1024);
        AscendC::LocalMemAllocator<> alloc;
        auto tile = alloc.Alloc<half>(N);
    """)
    diag = find(result, "AKA1001")[0]
    assert diag.details["static_bytes"] == 200 * 1024
    assert diag.details["dynamic_bytes"] == 32 * 1024
    assert diag.details["available_bytes"] == 16 * 1024


def test_host_dynamic_reservation_not_double_counted():
    src = make_kernel("LocalMemAllocator<> alloc; auto tile = alloc.Alloc<half>(1024);")
    src += "\nvoid host() { test_kernel<<<1, 221184, stream>>>(gm); }"
    result = analyze(src, chip="arch_3510")
    assert "AKA1001" not in codes_of(result)
    assert result.artifacts["ub_partition::test_kernel"]["dynamic_bytes"] == 221184


def test_unused_external_array_and_other_alloc_method_do_not_conflict():
    result = analyze_950("OtherAllocator a; a.Alloc<half>(200000); auto x = external;",
                         preamble="extern __ubuf__ half external[32];")
    assert "AKA1003" not in codes_of(result)
    assert "AKA1001" not in codes_of(result)


def test_second_external_array_is_checked_for_allocator_collision():
    result = analyze_950("LocalMemAllocator<> a; a.Alloc<half>(32); auto x = used;",
                         preamble="extern __ubuf__ half unused[]; extern __ubuf__ half used[];")
    assert "AKA1003" in codes_of(result)


@pytest.mark.parametrize("api", ["asc_mmad", "asc_mmad_sync", "asc_mmad_mx", "asc_mmad_mx_sync"])
@pytest.mark.parametrize("disabled", ["false", "true"])
def test_positional_gemv_contract(api, disabled):
    result = analyze_950(matrix_body(api, height="1", disabled=disabled))
    assert ("AKA4011" in codes_of(result)) is (disabled == "true")


def test_gemv_constant_expressions_and_unrelated_identifier_names():
    body = "constexpr int M = 2 - 1; constexpr bool flag = true;"
    result = analyze_950(body + matrix_body(height="M", disabled="flag"))
    assert any(d.details.get("disable_gemv") for d in find(result, "AKA4011"))


@pytest.mark.parametrize("api,a,b", [
    ("asc_mmad", "hifloat8_t", "fp8_e5m2_t"),
    ("asc_mmad_mx", "fp4x2_e1m2_t", "fp8_e5m2_t"),
    ("asc_mmad_mx", "hifloat8_t", "hifloat8_t"),
    ("asc_mmad_sync", "int4b_t", "int4b_t"),
])
def test_invalid_operand_pairs(api, a, b):
    assert "AKA4011" in codes_of(analyze_950(matrix_body(api, a, b)))


def test_matrix_operand_domain_is_checked_by_signature():
    assert "AKA1004" in codes_of(analyze_950(matrix_body(domain="__ubuf__")))


def test_tscm_buffer_without_tensor_does_not_crash():
    result = analyze_950("TPipe p; TBuf<TPosition::TSCM> b; p.InitBuffer(b, 524320);")
    assert find(result, "AKA4005")[0].details["allocated_bytes"] == 524320


def test_tscm_exact_capacity_is_accepted():
    result = analyze_950("TPipe p; TBuf<TPosition::TSCM> b; p.InitBuffer(b, 524288);")
    assert "AKA4005" not in codes_of(result)
    assert "AKA1001" not in codes_of(result)


@pytest.mark.parametrize("threads,budget", [(256, None), (288, 64), (512, 64), (544, 32), (1024, 32), (1056, 16)])
def test_simt_register_budget_thresholds(threads, budget):
    result = analyze_950(f"asc_vf_call<f>(dim3{{{threads}}});")
    warnings = find(result, "AKA3008")
    assert [d.details["register_budget"] for d in warnings] == ([] if budget is None else [budget])


@pytest.mark.parametrize("call", ["asc_vf_call(dim3{32});", "asc_vf_call<f,g>(dim3{32});", "asc_vf_call<f>(dim3{0});", "asc_vf_call<f>(dim3{4096});"])
def test_invalid_dispatch_signatures(call):
    assert "AKA2012" in codes_of(analyze_950(call))


def test_launch_bound_nested_expression_and_address_of_function():
    result = analyze_950("asc_vf_call<&f>(dim3{32, 2, 1});",
        preamble="__simt_vf__ __launch_bounds__((32 * 1)) void f() {}")
    assert any("smaller than" in d.message for d in find(result, "AKA2012"))


def test_diagnostic_keywords_inside_comments_and_strings_are_ignored():
    result = analyze_950('const char* text = "SparseMatmulType asc_call_vf(x) TPosition::UB"; // tiling.SetSparse(true);')
    assert not {"AKA4006", "AKA2011", "AKA1004"}.intersection(codes_of(result))


def test_datatype_widths_preserve_packed_fp4():
    from ascend_analyzer.symbolic import dtype_size
    for name in ("fp4x2_e1m2_t", "fp4x2_e2m1_t", "fp8_e4m3fn_t", "fp8_e5m2_t", "hifloat8_t"):
        assert dtype_size(name) == 1


def test_catalog_matches_skill_owned_ids():
    from ascend_analyzer.diagnostics import Code
    assert Code.UNMATCHED_SET_FLAG.value == Code.UNMATCHED_WAIT_FLAG.value == "AKA2001"
    assert Code.DEADLOCK_CYCLE.value == "AKA2002"
    assert Code.PIPE_ROUTE_MISMATCH.value == "AKA2004"
    assert Code.UB_BANK_CONFLICT.value == "AKA3003"
    assert Code.RAGGED_TAIL_OVERLAP.value == "AKA3004"
    assert Code.VF_SWITCHING.value == "AKA3005"
    assert Code.MISSING_SET_TAIL.value == "AKA4001"
    assert Code.ASYNC_WORKSPACE_MISSING.value == "AKA4002"


@pytest.mark.parametrize("bound", [0, -32, 1, 31, 2049])
def test_invalid_launch_bound_edges(bound):
    result = analyze_950("asc_vf_call<f>(dim3{32});", preamble=f"__simt_vf__ __launch_bounds__({bound}) void f() {{}}")
    assert "AKA2012" in codes_of(result)


def test_deprecated_dispatch_fix_preserves_template_and_arguments():
    result = analyze_950("asc_call_vf<f>(dim3{32}, data);")
    assert find(result, "AKA2011")[0].details["autofix"] == "asc_vf_call<f>(dim3{32}, data)"


def test_dim3_assignment_and_shift_dimensions():
    result = analyze_950("dim3 block = dim3{1 << 5, 2, 1}; asc_vf_call<f>(block);",
                        preamble="__simt_vf__ __launch_bounds__(32) void f() {}")
    assert any(d.details.get("threads") == 64 for d in find(result, "AKA2012"))


def test_independent_kernels_do_not_share_launch_constants():
    src = """
        __simt_vf__ __launch_bounds__(32) void f() {}
        __simt_vf__ __launch_bounds__(512) void g() {}
        __global__ __aicore__ void a() {
            constexpr int N = 32;
            asc_vf_call<f>(dim3{N});
        }
        __global__ __aicore__ void b() {
            constexpr int N = 512;
            asc_vf_call<g>(dim3{N});
        }
    """
    result = analyze(src, chip="950pr")
    assert "AKA2012" not in codes_of(result)
    assert len(find(result, "AKA3008")) == 1


def test_global_static_array_is_charged_when_used():
    result = analyze_950("auto ptr = storage;", preamble="__ubuf__ char storage[225280];")
    assert find(result, "AKA1001")[0].details["static_bytes"] == 225280


def test_unknown_allocator_extent_is_reported_as_analysis_boundary():
    result = analyze_950("LocalMemAllocator<> alloc; auto x = alloc.Alloc<half>(runtime_size);")
    assert any("UB allocation" in d.message and d.severity is Severity.INFO for d in find(result, "AKA3002"))


def test_extern_arrays_alias_one_dynamic_base():
    result = analyze_950("auto x = a; auto y = b;", preamble="extern __ubuf__ char a[131072]; extern __ubuf__ char b[131072];")
    assert "AKA1001" not in codes_of(result)
    assert result.artifacts["ub_partition::test_kernel"]["dynamic_bytes"] == 131072


@pytest.mark.parametrize("count,partial", [(128, True), (256, False)])
def test_tscm_staging_requires_full_count(count, partial):
    result = analyze_950(f"""
        LocalTensor<half> tile;
        tile.SetTPosition(TPosition::TSCM);
        tile.SetAddr(0);
        tile.SetSize(256);
        DataCopy(tile, gm, {count});
    """)
    assert ("AKA4005" in codes_of(result)) is partial


def test_tscm_multiple_buffers_are_added_and_views_count_once():
    result = analyze_950("""
        TPipe p;
        TBuf<TPosition::TSCM> a;
        TBuf<TPosition::TSCM> b;
        p.InitBuffer(a, 262144);
        p.InitBuffer(b, 262176);
        auto a1 = a.Get<half>();
        auto a2 = a.Get<half>();
        auto b1 = b.Get<half>();
    """)
    assert find(result, "AKA4005")[0].details["allocated_bytes"] == 524320


@pytest.mark.parametrize("api", ["asc_mmad", "asc_mmad_sync", "asc_mmad_mx"])
def test_matrix_accumulator_must_match_overload(api):
    body = matrix_body(api).replace("__cc__ float", "__cc__ half")
    assert "AKA4011" in codes_of(analyze_950(body))



def test_allocator_extent_uses_callsite_constants_across_kernels():
    result = analyze("""
        __global__ __aicore__ void small() {
            constexpr int N = 1024;
            LocalMemAllocator<> a;
            auto x = a.Alloc<half>(N);
        }
        __global__ __aicore__ void large() {
            constexpr int N = 131072;
            LocalMemAllocator<> a;
            auto x = a.Alloc<half>(N);
        }
    """, chip="950pr")
    assert result.artifacts["ub_partition::small"]["dynamic_bytes"] == 2048
    assert result.artifacts["ub_partition::large"]["dynamic_bytes"] == 262144
    assert len(find(result, "AKA1001")) == 1


def test_allocator_in_inlined_helper_is_budgeted():
    result = analyze_950("allocate();", preamble="""
        __aicore__ void allocate() {
            LocalMemAllocator<> a;
            auto x = a.Alloc<half>(131072);
        }
    """)
    assert "AKA1001" in codes_of(result)


def test_partial_symbolic_tscm_capacity_reports_boundary():
    result = analyze_950("TPipe p; TBuf<TPosition::TSCM> b; p.InitBuffer(b, runtime_size);")
    assert any("TSCM capacity" in d.message for d in find(result, "AKA3002"))
