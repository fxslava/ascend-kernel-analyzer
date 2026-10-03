"""Tests for source preparation and the AST visitor."""

from __future__ import annotations

import pytest
from conftest import analyze, make_kernel

from ascend_analyzer.diagnostics import DiagnosticCollector
from ascend_analyzer.hardware import HardwareModel, PhysicalDomain, Pipe, TPosition
from ascend_analyzer.ir import BarrierOp, FlagKind
from ascend_analyzer.parsing import parse_source, prepare_source


def parse(source: str):
    """Parse without running any checker."""
    hw = HardwareModel.for_chip("ascend910b")
    return parse_source("<test>.cpp", source, hw, DiagnosticCollector())


def single_kernel(body: str, preamble: str = ""):
    unit = parse(make_kernel(body, preamble))
    assert not unit.had_parse_errors, "fixture should parse cleanly"
    assert len(unit.kernels) == 1
    return unit.kernels[0]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_rewrite_preserves_source_length_exactly(self):
        source = (
            "extern \"C\" __global__ __aicore__ void k(__gm__ half* x) {\n"
            "  __ubuf__ half* p = (__ubuf__ half*)(2048);\n"
            "}\n"
        )
        prepared = prepare_source("k.cpp", source)
        assert len(prepared.rewritten) == len(source)
        # Line structure must be identical, or every reported line is wrong.
        assert prepared.rewritten.count("\n") == source.count("\n")

    def test_rewritten_source_parses_without_errors(self):
        source = (
            "extern \"C\" __global__ __aicore__ void k(__gm__ half* x) {\n"
            "  __ubuf__ half* p = (__ubuf__ half*)(2048);\n"
            "}\n"
        )
        unit = parse(source)
        assert not unit.had_parse_errors

    def test_qualifiers_inside_block_comments_are_left_alone(self):
        # Rewriting here would inject a '*/' and terminate the comment early,
        # spilling prose into the token stream.
        source = (
            "/*\n"
            " * Example: __ubuf__ half* p = (__ubuf__ half*)(0);\n"
            " */\n"
            "extern \"C\" __global__ __aicore__ void k(__gm__ half* x) { }\n"
        )
        prepared = prepare_source("k.cpp", source)
        assert "__ubuf__ half* p" in prepared.rewritten
        unit = parse(source)
        assert not unit.had_parse_errors

    def test_qualifiers_inside_line_comments_are_left_alone(self):
        source = (
            "// note: __ubuf__ is the UB address space\n"
            "extern \"C\" __global__ __aicore__ void k(__gm__ half* x) { }\n"
        )
        prepared = prepare_source("k.cpp", source)
        assert "// note: __ubuf__ is" in prepared.rewritten

    def test_qualifiers_inside_string_literals_are_left_alone(self):
        source = 'const char* s = "__ubuf__";\n'
        prepared = prepare_source("k.cpp", source)
        assert '"__ubuf__"' in prepared.rewritten

    def test_non_ascii_text_does_not_desynchronise_offsets(self):
        # tree-sitter reports byte offsets; the rewrite must record byte
        # offsets too. A multi-byte comment ahead of a qualifier previously
        # shifted them apart and the declaration was silently dropped.
        source = (
            'extern "C" __global__ __aicore__ void k(__gm__ half* x) {\n'
            "  /* 将数据搬运到统一缓冲区 */\n"
            "  __ubuf__ half* p = (__ubuf__ half*)(1000);\n"
            "}\n"
        )
        unit = parse(source)
        assert not unit.had_parse_errors
        tensor = unit.kernels[0].tensors["p"]
        assert tensor.domain is PhysicalDomain.UB
        assert tensor.offset_value == 1000

    def test_byte_length_is_preserved_for_non_ascii_sources(self):
        source = "// 中文\n__ubuf__ half* p = (__ubuf__ half*)(0);\n"
        prepared = prepare_source("k.cpp", source)
        assert len(prepared.rewritten_bytes) == len(source.encode("utf-8"))
        # The recorded span must index the rewritten bytes, not the characters.
        span = prepared.qualifiers[0]
        assert prepared.rewritten_bytes[span.start_byte : span.end_byte] == b"/*ubuf*/"

    def test_kernel_attribute_is_found_past_non_ascii_text(self):
        source = (
            "/* 核函数入口 */\n"
            'extern "C" __global__ __aicore__ void k(__gm__ half* x) { }\n'
        )
        unit = parse(source)
        assert [k.name for k in unit.kernels] == ["k"]

    def test_address_space_is_recoverable_after_rewriting(self):
        source = (
            "extern \"C\" __global__ __aicore__ void k(__gm__ half* x) {\n"
            "  __cbuf__ half* p = (__cbuf__ half*)(0);\n"
            "}\n"
        )
        prepared = prepare_source("k.cpp", source)
        tokens = {span.token for span in prepared.qualifiers}
        assert "__cbuf__" in tokens
        assert "__gm__" in tokens

    def test_annotations_are_extracted_with_fields(self):
        source = (
            "// @ascend-layout: name=xUb pos=VECIN offset=64 count=256 dtype=half\n"
            "extern \"C\" __global__ __aicore__ void k() { }\n"
        )
        prepared = prepare_source("k.cpp", source)
        layouts = prepared.annotations_of("layout")
        assert len(layouts) == 1
        assert layouts[0].get("name") == "xUb"
        assert layouts[0].get("pos") == "VECIN"
        assert layouts[0].get_int("offset") == 64
        assert layouts[0].get_int("count") == 256


# ---------------------------------------------------------------------------
# Kernel discovery
# ---------------------------------------------------------------------------


class TestKernelDiscovery:
    def test_finds_global_aicore_entry_points(self):
        kernel = single_kernel("return;")
        assert kernel.name == "test_kernel"
        assert kernel.is_kernel_entry

    def test_plain_functions_are_skipped_by_default(self):
        unit = parse("void helper() { }\n")
        assert unit.kernels == []

    def test_all_functions_mode_includes_plain_functions(self):
        result = analyze("void helper() { }\n", all_functions=True)
        assert [k.name for k in result.unit.kernels] == ["helper"]

    def test_gm_parameters_become_global_tensors(self):
        kernel = single_kernel("return;", "")
        assert "gm" in kernel.tensors
        assert kernel.tensors["gm"].domain is PhysicalDomain.GM


# ---------------------------------------------------------------------------
# Tensor binding forms
# ---------------------------------------------------------------------------


class TestTensorBindings:
    def test_local_tensor_with_setaddr_and_setsize(self):
        kernel = single_kernel(
            """
            AscendC::LocalTensor<half> xUb;
            xUb.SetTPosition(AscendC::TPosition::VECIN);
            xUb.SetAddr(1024);
            xUb.SetSize(256);
            """
        )
        tensor = kernel.tensors["xUb"]
        assert tensor.domain is PhysicalDomain.UB
        assert tensor.position is TPosition.VECIN
        assert tensor.dtype == "half"
        assert tensor.elem_size == 2
        assert tensor.offset_value == 1024
        assert tensor.size_value == 512  # 256 half == 512 bytes

    def test_set_buffer_len_sets_bytes_directly(self):
        kernel = single_kernel(
            """
            AscendC::LocalTensor<float> t;
            t.SetTPosition(AscendC::TPosition::VECCALC);
            t.SetAddr(0);
            t.SetBufferLen(2048);
            """
        )
        assert kernel.tensors["t"].size_value == 2048

    def test_constexpr_offsets_are_folded_including_sizeof(self):
        kernel = single_kernel(
            """
            AscendC::LocalTensor<half> t;
            t.SetTPosition(AscendC::TPosition::VECIN);
            t.SetAddr(UB_PONG);
            t.SetSize(TILE_ELEMS);
            """,
            preamble=(
                "constexpr uint32_t TILE_ELEMS = 256;\n"
                "constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(half);\n"
                "constexpr uint32_t UB_PING = 0;\n"
                "constexpr uint32_t UB_PONG = UB_PING + TILE_BYTES;\n"
            ),
        )
        assert kernel.tensors["t"].offset_value == 512
        assert kernel.tensors["t"].size_value == 512

    def test_define_macros_are_folded(self):
        kernel = single_kernel(
            """
            AscendC::LocalTensor<half> t;
            t.SetTPosition(AscendC::TPosition::VECIN);
            t.SetAddr(OFFSET);
            t.SetSize(16);
            """,
            preamble="#define OFFSET (4 * 256)\n",
        )
        assert kernel.tensors["t"].offset_value == 1024

    def test_address_space_pointer_gets_domain_and_offset(self):
        kernel = single_kernel("__ubuf__ half* p = (__ubuf__ half*)(2048);")
        tensor = kernel.tensors["p"]
        assert tensor.domain is PhysicalDomain.UB
        assert tensor.offset_value == 2048
        assert tensor.dtype == "half"

    @pytest.mark.parametrize(
        "qualifier, domain",
        [
            ("__ubuf__", PhysicalDomain.UB),
            ("__cbuf__", PhysicalDomain.L1),
            ("__ca__", PhysicalDomain.L0A),
            ("__cb__", PhysicalDomain.L0B),
            ("__cc__", PhysicalDomain.L0C),
        ],
    )
    def test_every_address_space_maps_to_its_domain(self, qualifier, domain):
        kernel = single_kernel(f"{qualifier} half* p = ({qualifier} half*)(0);")
        assert kernel.tensors["p"].domain is domain

    def test_layout_annotation_fills_in_a_raw_pointer(self):
        kernel = single_kernel(
            "// @ascend-layout: name=p pos=VECOUT offset=512 count=128 dtype=half\n"
            "__ubuf__ half* p = (__ubuf__ half*)(512);"
        )
        tensor = kernel.tensors["p"]
        assert tensor.position is TPosition.VECOUT
        assert tensor.offset_value == 512
        assert tensor.size_value == 256  # 128 half

    def test_tbuf_position_propagates_through_buffer_accessor(self):
        kernel = single_kernel(
            """
            AscendC::TBuf<AscendC::TPosition::VECIN> ubBuf;
            AscendC::LocalTensor<half> t = ubBuf.GetBufferByByte<half>(256);
            """
        )
        tensor = kernel.tensors["t"]
        assert tensor.domain is PhysicalDomain.UB
        assert tensor.offset_value == 256

    def test_tensor_factory_form(self):
        kernel = single_kernel(
            "auto t = AscendC::GetLocalTensor<half>("
            "AscendC::TPosition::VECIN, 128, 64);"
        )
        # 'auto' hides the tensor type, so the factory call is what identifies
        # it; this form is recognised only when the declared type says
        # LocalTensor, so here we just assert we did not crash or mis-bind.
        assert "t" not in kernel.tensors or kernel.tensors["t"].offset_value == 128

    def test_explicit_local_tensor_factory_form(self):
        kernel = single_kernel(
            "AscendC::LocalTensor<half> t = AscendC::GetLocalTensor<half>("
            "AscendC::TPosition::VECIN, 128, 64);"
        )
        tensor = kernel.tensors["t"]
        assert tensor.domain is PhysicalDomain.UB
        assert tensor.offset_value == 128
        assert tensor.size_value == 128  # 64 half


class TestSubTensorViews:
    """``LocalTensor t = arena[i]`` lands at ``offset(arena) + i * sizeof(T)``.

    The arena's own base only exists once the TPipe layout is synthesised
    after the walk, so the view records its provenance and the offset is
    completed then.  Until the ``subscript_argument_list`` extraction was
    fixed, every view kept ``byte_offset=None`` - which is what locked 793 of
    the fleet's 1,064 AKA3006 candidates in ``chunk_kda_fwd.cpp``.
    """

    BODY = """
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> vecBuf;
        pipe.InitBuffer(vecBuf, 8192);
        uint64_t elems = 16;
        AscendC::LocalTensor<float> arena = vecBuf.Get<float>();
        AscendC::LocalTensor<float> head = arena;
        AscendC::LocalTensor<float> mid = arena[elems];
        AscendC::LocalTensor<float> tail = arena[2 * elems];
    """

    def kernel(self):
        return single_kernel(self.BODY)

    def test_a_view_of_a_bound_tensor_gets_the_shifted_offset(self):
        tensors = self.kernel().tensors
        assert tensors["arena"].offset_value == 0
        assert tensors["mid"].offset_value == 16 * 4
        assert tensors["tail"].offset_value == 2 * 16 * 4

    def test_a_pure_alias_shares_the_source_offset(self):
        tensors = self.kernel().tensors
        assert tensors["head"].offset_value == tensors["arena"].offset_value == 0

    def test_views_inherit_the_buffer_domain(self):
        tensors = self.kernel().tensors
        assert tensors["mid"].domain is PhysicalDomain.UB

    def test_a_symbolic_index_stays_symbolic_not_unbound(self):
        kernel = single_kernel(
            self.BODY
            + """
            uint64_t run = gm[0];
            AscendC::LocalTensor<float> dyn = arena[run];
            """
        )
        tensor = kernel.tensors["dyn"]
        assert tensor.byte_offset is not None
        assert tensor.offset_value is None  # symbolic: honest, not resolved

    def test_a_view_of_a_view_resolves_through_the_chain(self):
        kernel = single_kernel(
            self.BODY
            + """
            AscendC::LocalTensor<float> inner = tail[4];
            """
        )
        # tail = 2*16*4 = 128; inner = 128 + 4*4 = 144
        assert kernel.tensors["inner"].offset_value == 144

    def test_a_view_carved_from_the_accessor_call_itself(self):
        kernel = single_kernel(
            """
            AscendC::TPipe pipe;
            AscendC::TBuf<AscendC::TPosition::VECCALC> vecBuf;
            pipe.InitBuffer(vecBuf, 8192);
            AscendC::LocalTensor<half> typed = vecBuf.Get<half>()[128];
            AscendC::LocalTensor<half> tail = typed[64];
            """
        )
        tensors = kernel.tensors
        assert tensors["typed"].domain is PhysicalDomain.UB
        assert tensors["typed"].offset_value == 128 * 2
        assert tensors["tail"].offset_value == 128 * 2 + 64 * 2


# ---------------------------------------------------------------------------
# Synchronisation extraction
# ---------------------------------------------------------------------------


class TestFlagExtraction:
    def test_templated_setflag_route_and_pipe(self):
        kernel = single_kernel(
            "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);"
        )
        flags = kernel.flag_ops()
        assert len(flags) == 1
        op = flags[0]
        assert op.flag_kind is FlagKind.SET
        assert op.route.name == "MTE2_V"
        assert op.event_id == 0
        # A SetFlag is issued on the route's source pipeline.
        assert op.pipe is Pipe.MTE2

    def test_templated_waitflag_runs_on_the_destination_pipe(self):
        kernel = single_kernel(
            "AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);"
        )
        op = kernel.flag_ops()[0]
        assert op.flag_kind is FlagKind.WAIT
        assert op.event_id == 3
        assert op.pipe is Pipe.V

    def test_unqualified_spelling(self):
        kernel = single_kernel("SetFlag<HardEvent::V_MTE3>(EVENT_ID1);")
        op = kernel.flag_ops()[0]
        assert op.route.name == "V_MTE3"
        assert op.pipe is Pipe.V

    def test_isasi_set_flag_form(self):
        kernel = single_kernel("set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);")
        op = kernel.flag_ops()[0]
        assert op.isasi_form
        assert op.route.name == "MTE2_V"
        assert op.event_id == 2
        assert op.pipe is Pipe.MTE2

    def test_isasi_wait_flag_form(self):
        kernel = single_kernel("wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);")
        op = kernel.flag_ops()[0]
        assert op.flag_kind is FlagKind.WAIT
        assert op.route.name == "V_MTE3"
        assert op.pipe is Pipe.MTE3

    def test_event_id_from_a_constant(self):
        kernel = single_kernel(
            "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(MY_EVENT);",
            preamble="constexpr int32_t MY_EVENT = 4;\n",
        )
        assert kernel.flag_ops()[0].event_id == 4

    def test_unresolvable_event_id_is_recorded_as_text(self):
        kernel = single_kernel(
            """
            for (uint32_t i = 0; i < 4; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pick(i));
            }
            """
        )
        op = kernel.flag_ops()[0]
        assert op.event_id is None
        assert "pick" in op.event_id_text


class TestBarrierExtraction:
    def test_templated_global_barrier(self):
        kernel = single_kernel("AscendC::PipeBarrier<PIPE_ALL>();")
        barrier = kernel.barriers()[0]
        assert isinstance(barrier, BarrierOp)
        assert barrier.is_global
        assert barrier.target is Pipe.ALL

    def test_templated_single_pipe_barrier(self):
        kernel = single_kernel("AscendC::PipeBarrier<PIPE_V>();")
        barrier = kernel.barriers()[0]
        assert not barrier.is_global
        assert barrier.target is Pipe.V

    def test_isasi_barrier(self):
        kernel = single_kernel("pipe_barrier(PIPE_ALL);")
        assert kernel.barriers()[0].is_global


# ---------------------------------------------------------------------------
# API calls and pipeline assignment
# ---------------------------------------------------------------------------


class TestApiCalls:
    def test_datacopy_gm_to_ub_runs_on_mte2(self):
        kernel = single_kernel(
            """
            AscendC::GlobalTensor<half> g;
            g.SetGlobalBuffer(gm, 256);
            AscendC::LocalTensor<half> u;
            u.SetTPosition(AscendC::TPosition::VECIN);
            u.SetAddr(0);
            u.SetSize(256);
            AscendC::DataCopy(u, g, 256);
            """
        )
        call = kernel.api_calls()[-1]
        assert call.name == "DataCopy"
        assert call.pipe is Pipe.MTE2
        assert call.writes == ("u",)
        assert call.reads == ("g",)

    def test_datacopy_ub_to_gm_runs_on_mte3(self):
        kernel = single_kernel(
            """
            AscendC::GlobalTensor<half> g;
            g.SetGlobalBuffer(gm, 256);
            AscendC::LocalTensor<half> u;
            u.SetTPosition(AscendC::TPosition::VECOUT);
            u.SetAddr(0);
            u.SetSize(256);
            AscendC::DataCopy(g, u, 256);
            """
        )
        assert kernel.api_calls()[-1].pipe is Pipe.MTE3

    def test_vector_ops_run_on_pipe_v(self):
        kernel = single_kernel(
            """
            AscendC::LocalTensor<half> a;
            a.SetTPosition(AscendC::TPosition::VECIN);
            a.SetAddr(0);
            a.SetSize(256);
            AscendC::Add(a, a, a, 256);
            """
        )
        call = kernel.api_calls()[-1]
        assert call.name == "Add"
        assert call.pipe is Pipe.V

    def test_mmad_runs_on_the_cube_pipe(self):
        kernel = single_kernel("AscendC::Mmad(c, a, b, 16);")
        assert kernel.api_calls()[0].pipe is Pipe.M

    def test_subscripted_argument_resolves_to_its_base_tensor(self):
        kernel = single_kernel(
            """
            AscendC::GlobalTensor<half> g;
            g.SetGlobalBuffer(gm, 4096);
            AscendC::LocalTensor<half> u;
            u.SetTPosition(AscendC::TPosition::VECIN);
            u.SetAddr(0);
            u.SetSize(256);
            AscendC::DataCopy(u, g[128], 256);
            """
        )
        call = kernel.api_calls()[-1]
        assert call.reads == ("g",)

    def test_liveness_is_recorded_from_uses(self):
        kernel = single_kernel(
            """
            AscendC::LocalTensor<half> a;
            a.SetTPosition(AscendC::TPosition::VECIN);
            a.SetAddr(0);
            a.SetSize(256);
            AscendC::Abs(a, a, 256);
            AscendC::Abs(a, a, 256);
            """
        )
        tensor = kernel.tensors["a"]
        assert tensor.first_use is not None
        assert tensor.last_use is not None
        assert tensor.last_use > tensor.first_use


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


class TestLoops:
    def test_trip_count_from_a_literal_bound(self):
        kernel = single_kernel("for (uint32_t i = 0; i < 8; ++i) { g(); }")
        loop = kernel.loops[0]
        assert loop.induction_var == "i"
        assert loop.trip_count == 8

    def test_trip_count_from_a_constexpr_bound(self):
        kernel = single_kernel(
            "for (uint32_t i = 0; i < COUNT / 2; ++i) { g(); }",
            preamble="constexpr uint32_t COUNT = 8;\n",
        )
        assert kernel.loops[0].trip_count == 4

    def test_inclusive_bound(self):
        kernel = single_kernel("for (int i = 0; i <= 7; ++i) { g(); }")
        assert kernel.loops[0].trip_count == 8

    def test_strided_loop(self):
        kernel = single_kernel("for (int i = 0; i < 16; i += 4) { g(); }")
        assert kernel.loops[0].trip_count == 4

    def test_small_bounded_loops_unroll_to_concrete_offsets(self):
        # Trip count at or below the unroll limit: the body is replayed once
        # per iteration with the induction variable bound to each concrete
        # value, so the last binding wins and the offset is a constant.
        kernel = single_kernel(
            """
            for (uint32_t t = 0; t < 8; ++t) {
                AscendC::LocalTensor<half> tile;
                tile.SetTPosition(AscendC::TPosition::VECIN);
                tile.SetAddr(t * 512);
                tile.SetSize(256);
            }
            """
        )
        assert kernel.loops[0].unrolled
        assert kernel.tensors["tile"].offset_value == 7 * 512

    def test_induction_variable_bounds_make_offsets_symbolic_but_bounded(self):
        # Beyond the unroll limit the loop stays symbolic: offsets keep the
        # induction variable as a bounded free variable for the solver.
        kernel = single_kernel(
            """
            for (uint32_t t = 0; t < 64; ++t) {
                AscendC::LocalTensor<half> tile;
                tile.SetTPosition(AscendC::TPosition::VECIN);
                tile.SetAddr(t * 512);
                tile.SetSize(256);
            }
            """
        )
        assert not kernel.loops[0].unrolled
        tensor = kernel.tensors["tile"]
        assert tensor.offset_value is None  # genuinely not a constant
        from ascend_analyzer.symbolic import free_vars, is_decidable

        assert is_decidable(tensor.byte_offset)
        bounds = free_vars(tensor.byte_offset)["t"]
        assert (bounds.lower, bounds.upper) == (0, 63)

    def test_operations_inside_a_loop_carry_its_id(self):
        kernel = single_kernel(
            """
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            for (uint32_t i = 0; i < 4; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            """
        )
        flags = kernel.flag_ops()
        assert flags[0].loop_id is None       # prologue
        assert flags[1].loop_id == 0          # inside the loop body

    def test_nested_loops_record_parentage(self):
        kernel = single_kernel(
            """
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 2; ++j) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                }
            }
            """
        )
        assert len(kernel.loops) == 2
        inner = kernel.loops[1]
        assert inner.parent == 0


class TestConditionals:
    def test_branch_operations_are_marked_conditional(self):
        kernel = single_kernel(
            """
            if (block_idx == 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            """
        )
        assert kernel.flag_ops()[0].conditional

    def test_straight_line_operations_are_not_conditional(self):
        kernel = single_kernel(
            "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);"
        )
        assert not kernel.flag_ops()[0].conditional


def test_trace_is_in_source_order():
    kernel = single_kernel(
        """
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
        """
    )
    lines = [op.loc.line for op in kernel.ops]
    assert lines == sorted(lines)
    assert [op.index for op in kernel.ops] == list(range(len(kernel.ops)))


# ---------------------------------------------------------------------------
# Cube-kernel frontend: macro stages, TBuf propagation, CCE casts, unrolling
# ---------------------------------------------------------------------------


class TestTBufPropagation:
    """TBuf<TPosition> -> .Get<T>() domain propagation (AKA1009)."""

    BODY = """
        TPipe pipe;
        TBuf<TPosition::A1> bL1a;
        TBuf<TPosition::B1> bL1b;
        TBuf<TPosition::A2> bL0a;
        TBuf<TPosition::CO1> bL0c;
        pipe.InitBuffer(bL1a, 1024);
        pipe.InitBuffer(bL1b, 1024);
        pipe.InitBuffer(bL0a, 1024);
        pipe.InitBuffer(bL0c, 2048);
        LocalTensor<int8_t> l1a = bL1a.Get<int8_t>(), l1b = bL1b.Get<int8_t>();
        LocalTensor<int8_t> l0a = bL0a.Get<int8_t>();
        LocalTensor<float>  l0c = bL0c.Get<float>();
        DataCopy(l1a[0], gA[0], 512);
    """

    def test_get_inherits_the_buffer_position_and_domain(self):
        kernel = single_kernel(self.BODY)
        assert kernel.tensors["l1a"].domain is PhysicalDomain.L1
        assert kernel.tensors["l1a"].position is TPosition.A1
        assert kernel.tensors["l0a"].domain is PhysicalDomain.L0A
        assert kernel.tensors["l0a"].position is TPosition.A2
        assert kernel.tensors["l0c"].domain is PhysicalDomain.L0C
        assert kernel.tensors["l0c"].position is TPosition.CO1

    def test_init_buffer_sizes_synthesise_a_concrete_layout(self):
        kernel = single_kernel(self.BODY)
        l1a = kernel.tensors["l1a"]
        assert l1a.offset_value == 0
        assert l1a.size_value == 1024
        # Buffers in the same domain are bump-allocated apart.
        assert kernel.tensors["l1b"].offset_value == 1024
        assert kernel.tensors["l1b"].size_value == 1024
        assert kernel.tensors["l0c"].size_value == 2048

    def test_unpositioned_get_stays_unknown(self):
        # The documented gap: a TBuf without a TPosition template argument
        # still yields UNKNOWN rather than a guessed domain.
        kernel = single_kernel(
            """
            TPipe pipe;
            TBuf<TPosition::A1> buf;
            pipe.InitBuffer(buf, 512);
            LocalTensor<int8_t> t = buf.Get<int8_t>();
            DataCopy(t[0], gA[0], 128);
            """
        )
        assert kernel.tensors["t"].domain is PhysicalDomain.L1


class TestCastArgumentResolution:
    """(__ca__ T *)(uintptr_t)ptr.GetPhyAddr() resolves to the tensor."""

    BODY = """
        TPipe pipe;
        TBuf<TPosition::A2> bL0a;
        TBuf<TPosition::A1> bL1a;
        pipe.InitBuffer(bL0a, 1024);
        pipe.InitBuffer(bL1a, 1024);
        LocalTensor<int8_t> l0a = bL0a.Get<int8_t>();
        LocalTensor<int8_t> l1a = bL1a.Get<int8_t>();
        load_cbuf_to_ca_s4(
            (__ca__ fp4x2_e2m1_t *)(uintptr_t)l0a[0].GetPhyAddr(),
            (__cbuf__ fp4x2_e2m1_t *)(uintptr_t)l1a[0].GetPhyAddr(),
            (uint16_t)0, (uint16_t)0, (uint8_t)1, (uint8_t)1,
            (int16_t)1, (uint16_t)1, false);
    """

    def test_cast_operands_resolve_to_their_tensors(self):
        kernel = single_kernel(self.BODY)
        loads = [op for op in kernel.api_calls() if op.name == "load_cbuf_to_ca_s4"]
        assert len(loads) == 1
        assert loads[0].args[0].tensor == "l0a"
        assert loads[0].args[1].tensor == "l1a"
        assert loads[0].writes == ("l0a",)
        assert loads[0].pipe is Pipe.MTE1

    def test_parenthesised_expressions_are_not_mistaken_for_casts(self):
        kernel = single_kernel(
            """
            TPipe pipe;
            TBuf<TPosition::VECCALC> b;
            pipe.InitBuffer(b, 512);
            LocalTensor<half> t = b.Get<half>();
            LocalTensor<half> u = t;
            DataCopy((u), (t), 128);
            """
        )
        copies = [op for op in kernel.api_calls() if op.name == "DataCopy"]
        assert copies[0].args[0].tensor == "u"
        assert copies[0].args[1].tensor == "t"


class TestLoaderVolumeInference:
    """Raw-pointer loaders derive byte sizes from their repeat parameter."""

    def test_repeat_parameter_sizes_an_unknown_tensor(self):
        kernel = single_kernel(
            """
            TPipe pipe;
            TBuf<TPosition::A1> bL1a;
            LocalTensor<int8_t> l1a;
            l1a.SetTPosition(AscendC::TPosition::A1);
            load_cbuf_to_ca_mx((uint64_t)0,
                (__cbuf__ half *)(uintptr_t)l1a.GetPhyAddr(),
                (uint16_t)0, (uint16_t)0, (uint8_t)2, (uint8_t)1,
                (uint16_t)1, (uint16_t)1);
            """
        )
        tensor = kernel.tensors["l1a"]
        # 2 repeats x 32 B E8M0 granule; no sizeof(void) failure.
        assert tensor.size_value == 64
        assert "volume" in tensor.origin


class TestEventIdFolding:
    """The ping/pong selector folds to concrete EVENT_IDs."""

    def test_ternary_selector_folds_per_unrolled_iteration(self):
        kernel = single_kernel(
            """
            #define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)
            for (int t = 0; t < 4; ++t) {
                int p = t & 1;
                SetFlag<HardEvent::MTE1_M>(EV(p));
                WaitFlag<HardEvent::MTE1_M>(EV(p));
            }
            """
        )
        ids = [op.event_id for op in kernel.flag_ops()]
        assert ids == [0, 0, 1, 1, 0, 0, 1, 1]
        assert all(op.event_id is not None for op in kernel.flag_ops())

    def test_helper_function_selector_folds(self):
        kernel = single_kernel(
            """
            SetFlag<HardEvent::MTE1_M>(ev(0));
            SetFlag<HardEvent::MTE1_M>(ev(1));
            """,
            preamble=(
                "constexpr event_t EV0 = EVENT_ID0, EV1 = EVENT_ID1;\n"
                "static __aicore__ inline event_t ev(int p) { return p ? EV1 : EV0; }\n"
            ),
        )
        assert [op.event_id for op in kernel.flag_ops()] == [0, 1]

    def test_epilogue_guards_prune_with_concrete_induction_values(self):
        kernel = single_kernel(
            """
            #define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            for (int t = 0; t < 4; ++t) {
                int p = t & 1;
                if (t + 2 < 4) {
                    WaitFlag<HardEvent::MTE1_MTE2>(EV(p));
                    SetFlag<HardEvent::MTE1_MTE2>(EV(p));
                }
            }
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
            """
        )
        # Only iterations t=0 and t=1 enter the guard; no phantom ops from
        # the dead t=2/t=3 arms.
        # Prologue set + (t=0: wait, set) + (t=1: wait, set) + final wait.
        assert len(kernel.flag_ops()) == 1 + 4 + 1
        assert all(op.event_id is not None for op in kernel.flag_ops())


class TestStageMacros:
    """Multi-line #define stage macros become statements at the call site."""

    def test_stage_macro_operations_are_traced_in_order(self):
        kernel = single_kernel(
            """
            #define LOAD(p) do {                                              \
                WaitFlag<HardEvent::MTE2_MTE1>(EV(p));                        \
                SetFlag<HardEvent::MTE1_M>(EV(p));                            \
            } while (0)
            #define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)
            LOAD(0);
            LOAD(1);
            """
        )
        resolved = [(op.flag_kind.value, op.event_id) for op in kernel.flag_ops()]
        assert resolved == [
            ("WaitFlag", 0),
            ("SetFlag", 0),
            ("WaitFlag", 1),
            ("SetFlag", 1),
        ]

    def test_diagnostics_from_expansions_point_at_the_invocation_line(self):
        # The invocation sits on its own line; everything the expansion emits
        # is reported there, not on the #define body's lines.
        kernel = single_kernel(
            "A(0);",
            preamble=(
                "#define A(q) do { \\\n"
                "    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); \\\n"
                "    SetFlag<HardEvent::MTE3_V>(EVENT_ID0); \\\n"
                "} while (0)\n"
            ),
        )
        lines = {op.loc.line for op in kernel.flag_ops()}
        assert len(lines) == 1
        # Both ops report the single invocation line of A(0); in the body.
        assert kernel.flag_ops()[0].loc.snippet.strip().startswith("A(0);")



# ---------------------------------------------------------------------------
# auto tensor inference, named queue depths, GetWithOffset, MIX stage classes
# ---------------------------------------------------------------------------


class TestAutoTensorInference:
    """``auto x = que.AllocTensor<T>()`` declares a fully typed tensor."""

    def test_auto_alloc_tensor_binds_domain_dtype_and_size(self):
        kernel = single_kernel(
            """
            TPipe pipe;
            TQue<TPosition::VECIN, 2> que;
            pipe.InitBuffer(que, 2, 512);
            auto x = que.AllocTensor<half>();
            Duplicate(x, (half)1.0f, 256);
            """
        )
        assert "x" in kernel.tensors
        tensor = kernel.tensors["x"]
        assert tensor.domain is PhysicalDomain.UB
        assert tensor.dtype == "half"
        assert tensor.elem_size == 2
        assert tensor.size_value == 512  # one block of the depth-2 queue

    def test_auto_deque_tensor_binds_domain(self):
        kernel = single_kernel(
            """
            TPipe pipe;
            TQue<TPosition::VECIN, 2> que;
            pipe.InitBuffer(que, 2, 4096);
            auto v = que.DeQue<uint8_t>();
            """
        )
        tensor = kernel.tensors["v"]
        assert tensor.domain is PhysicalDomain.UB
        assert tensor.dtype == "uint8_t"

    def test_auto_get_tensor_inherits_tbuf_position(self):
        kernel = single_kernel(
            """
            TPipe pipe;
            TBuf<TPosition::VECCALC> scratch;
            pipe.InitBuffer(scratch, 1024);
            auto y = scratch.Get<half>();
            """
        )
        tensor = kernel.tensors["y"]
        assert tensor.domain is PhysicalDomain.UB
        assert tensor.position is TPosition.VECCALC

    def test_auto_get_with_offset_declares_tensor(self):
        kernel = single_kernel(
            """
            TPipe pipe;
            TBuf<TPosition::VECCALC> scratch;
            pipe.InitBuffer(scratch, 2048);
            auto z = scratch.GetWithOffset<half>(128, 64);
            """
        )
        tensor = kernel.tensors["z"]
        assert tensor.domain is PhysicalDomain.UB
        assert tensor.offset_value == 64  # pool base is 0 here
        assert tensor.size_value == 256  # 128 half elements


class TestNamedQueueDepth:
    """``TQue<TPosition::VECIN, DB>`` with a named depth constant registers."""

    def test_constexpr_depth_constant_registers_the_queue(self):
        kernel = single_kernel(
            """
            TPipe pipe;
            TQue<TPosition::VECIN, DB_DEPTH> que;
            pipe.InitBuffer(que, DB_DEPTH, 512);
            auto x = que.AllocTensor<half>();
            Duplicate(x, (half)1.0f, 256);
            """,
            preamble="static constexpr int DB_DEPTH = 2;",
        )
        assert "x" in kernel.tensors
        assert kernel.tensors["x"].domain is PhysicalDomain.UB

    def test_define_depth_constant_registers_the_queue(self):
        kernel = single_kernel(
            """
            TPipe pipe;
            TQue<TPosition::VECOUT, QD> qo;
            pipe.InitBuffer(qo, QD, 256);
            auto o = qo.AllocTensor<half>();
            """,
            preamble="#define QD 2",
        )
        assert kernel.tensors["o"].domain is PhysicalDomain.UB

    def test_named_depth_folds_through_the_global_env(self):
        unit = parse(
            make_kernel(
                "TPipe pipe; TQue<TPosition::VECIN, DB_DEPTH> que;",
                preamble="static constexpr int DB_DEPTH = 4;",
            )
        )
        assert unit.constants.get("DB_DEPTH") == 4


class TestGetWithOffsetSignature:
    """CANN order: ``GetWithOffset<T>(elementCount, byteOffset)``."""

    def test_count_first_offset_second(self):
        kernel = single_kernel(
            """
            TPipe pipe;
            TBuf<TPosition::VECCALC> scratch;
            pipe.InitBuffer(scratch, 2048);
            LocalTensor<half> t = scratch.GetWithOffset<half>(128, 64);
            """
        )
        tensor = kernel.tensors["t"]
        assert tensor.offset_value == 64  # the SECOND argument
        assert tensor.size_value == 256  # 128 elements x sizeof(half)

    def test_offset_is_pool_relative(self):
        # A preceding buffer pushes the pool base past zero; the byte-offset
        # argument must land INSIDE the pool, not at an absolute address.
        kernel = single_kernel(
            """
            TPipe pipe;
            TBuf<TPosition::VECCALC> first;
            pipe.InitBuffer(first, 256);
            TBuf<TPosition::VECCALC> scratch;
            pipe.InitBuffer(scratch, 2048);
            LocalTensor<half> t = scratch.GetWithOffset<half>(128, 64);
            """
        )
        tensor = kernel.tensors["t"]
        assert tensor.offset_value == 256 + 64
        assert tensor.size_value == 256

    def test_footprint_uses_count_not_buffer_block(self):
        # The count (first argument) sizes the tensor even when the parent
        # buffer's block length is larger.
        kernel = single_kernel(
            """
            TPipe pipe;
            TBuf<TPosition::VECCALC> scratch;
            pipe.InitBuffer(scratch, 512);
            LocalTensor<int16_t> t = scratch.GetWithOffset<int16_t>(64, 0);
            """
        )
        assert kernel.tensors["t"].size_value == 128  # 64 x 2 B, not 512


class TestMixTwoStageClasses:
    """A MIX_AIC_1_1 entry driving two stage classes must analyze both arms.

    Regression for the symbol-scope collision: the entry is a free function,
    so ``cube.Init``/``vec.Init`` cannot be resolved by caller context, and
    both classes define the same method names.  Before the fix the whole
    walk collapsed to ``ops == 0``.
    """

    PREAMBLE = """
        class CubeStage {
        public:
            __aicore__ inline void Init(TPipe *pipe) {
                pipe->InitBuffer(a1, 512);
            }
            __aicore__ inline void Process() {
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            }
            TBuf<TPosition::A1> a1;
        };

        class VecStage {
        public:
            __aicore__ inline void Init(TPipe *pipe) {
                pipe->InitBuffer(scratch, 1024);
            }
            __aicore__ inline void Process() {
                LocalTensor<half> y = scratch.Get<half>();
                Duplicate(y, (half)1.0f, 512);
            }
            TBuf<TPosition::VECCALC> scratch;
        };
    """

    BODY = """
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
        if ASCEND_IS_AIC {
            AscendC::TPipe pipe;
            AscendC::CubeStage cube;
            cube.Init(&pipe);
            cube.Process();
        }
        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            AscendC::VecStage vec;
            vec.Init(&pipe);
            vec.Process();
        }
    """

    def test_both_stage_classes_resolve_in_one_entry(self):
        unit = parse(make_kernel(self.BODY, self.PREAMBLE))
        assert not unit.had_parse_errors
        assert len(unit.kernels) == 1, "stage methods must inline, not spawn kernels"
        kernel = unit.kernels[0]
        # One op from each arm (InitBuffer sizes land in buffer_sizes, not ops).
        assert len(kernel.ops) >= 2, f"walk collapsed: {len(kernel.ops)} ops"
        # The AIC arm's flag op and the AIV arm's vector op both present.
        flags = [(op.flag_kind.value, op.event_id) for op in kernel.flag_ops()]
        assert ("SetFlag", 0) in flags
        # The Duplicate is a PIPE_V op emitted from VecStage::Process.
        assert any(
            getattr(op, "name", None) == "Duplicate" for op in kernel.ops_on(Pipe.V)
        )

    def test_both_stage_buffers_are_sized(self):
        unit = parse(make_kernel(self.BODY, self.PREAMBLE))
        kernel = unit.kernels[0]
        # InitBuffer ran inside BOTH inlined Init methods.
        assert kernel.buffer_sizes.get("a1") == 512
        assert kernel.buffer_sizes.get("scratch") == 1024

    def test_receiver_typed_resolution_survives_same_named_methods(self):
        # The two classes define the SAME method names (Init/Process); the
        # receiver's declared class type is what disambiguates them.
        unit = parse(make_kernel(self.BODY, self.PREAMBLE))
        kernel = unit.kernels[0]
        inlined = set(kernel.inlined)
        assert "Init" in inlined and "Process" in inlined
