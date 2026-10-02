"""Tests for AST-based tiling role inference.

A kernel that sizes its queues from a host tiling struct resolves nothing
without that struct: the ``InitBuffer`` extent will not fold, so the buffer
never takes part in the ``TPipe`` bump allocation, so every tensor the
allocator hands out loses its offset, so every offset-dependent check - the
UB bank-conflict check (AKA3006) above all - silently does not run.

Inference closes that gap by reading the *role* each field plays at its use
sites, never its name.  The guard rails are the point: only fields reached
through a verified tiling pointer are touched, an exact binding always wins,
a field compared against the core index is left alone, and a field used in
two disagreeing roles is left alone.
"""

from __future__ import annotations

from ascend_analyzer.analyzer import AnalyzerOptions, KernelAnalyzer

# A queued vector kernel of the shape the production ops use: the buffer
# manager sizes three queues from the tiling struct, and the tensors come out
# of AllocTensor.  Two VECIN queues of depth 4 x 64 B reserve 256 B each, so
# the bump allocator puts their first blocks 256 B = 8 UB blocks apart - the
# same bank - which is exactly the conflict AKA3006 exists to find.  None of
# that is visible until ``t->tileBytes`` has a value.
QUEUED_KERNEL = """
#include "kernel_operator.h"

using namespace AscendC;

struct VecTilingData {
    uint32_t tileBytes;
    uint32_t usedCoreNum;
};

class Adder {
public:
    __aicore__ inline void Init(__gm__ uint8_t *tiling)
    {
        __gm__ VecTilingData *t = reinterpret_cast<__gm__ VecTilingData *>(tiling);
        if (t->usedCoreNum <= GetBlockIdx()) {
            return;
        }
        pipe_.InitBuffer(inQueA_, 4, t->tileBytes);
        pipe_.InitBuffer(inQueB_, 4, t->tileBytes);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> a = inQueA_.AllocTensor<half>();
        LocalTensor<half> b = inQueB_.AllocTensor<half>();
        Add(a, a, b, 32);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 4> inQueA_;
    TQue<TPosition::VECIN, 4> inQueB_;
};

extern "C" __global__ __aicore__ void queued_add(__gm__ uint8_t *tiling)
{
    Adder op;
    op.Init(tiling);
    op.Process();
}
"""


def analyze(source: str, **options):
    return KernelAnalyzer(AnalyzerOptions(**options)).analyze_source("t.cpp", source)


def coverage(result) -> dict:
    out = {"candidates": 0, "evaluated": 0, "conflicts": 0}
    for key, value in result.artifacts.items():
        if key.startswith("bank_conflict_coverage::") and isinstance(value, dict):
            for field in out:
                out[field] += int(value.get(field, 0))
    return out


class TestActivation:
    """The headline claim: inference makes AKA3006 reach a verdict."""

    def test_without_inference_the_check_cannot_run(self):
        result = analyze(QUEUED_KERNEL)
        counts = coverage(result)
        assert counts["candidates"] >= 1
        assert counts["evaluated"] == 0
        assert not result.unit.inferred_tiling_bindings

    def test_with_inference_the_check_runs(self):
        result = analyze(QUEUED_KERNEL, infer_tiling_roles=True)
        counts = coverage(result)
        assert counts["candidates"] >= 1
        assert counts["evaluated"] == counts["candidates"]

    def test_with_inference_the_real_conflict_is_found(self):
        """Two depth-4 x 64 B queues land 8 UB blocks apart: one bank."""
        result = analyze(QUEUED_KERNEL, infer_tiling_roles=True)
        assert "AKA3006" in result.codes()
        assert coverage(result)["conflicts"] >= 1

    def test_inference_does_not_reject_the_kernel(self):
        """An inferred value must never be the reason a kernel is rejected."""
        result = analyze(QUEUED_KERNEL, infer_tiling_roles=True)
        assert result.fatal_count == 0


class TestScopeRestriction:
    def test_buffer_extent_is_bound_from_the_cast_pointer(self):
        result = analyze(QUEUED_KERNEL, infer_tiling_roles=True)
        assert result.unit.inferred_tiling_bindings.get("t->tileBytes") == 64

    def test_core_grid_field_is_left_symbolic(self):
        """``t->usedCoreNum <= GetBlockIdx()`` counts cores, not bytes."""
        result = analyze(QUEUED_KERNEL, infer_tiling_roles=True)
        assert "t->usedCoreNum" not in result.unit.inferred_tiling_bindings

    def test_locals_and_loop_indices_are_never_bound(self):
        source = """
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void k(__gm__ half *gm)
{
    uint32_t tileBytes = 7;
    for (int blockIdx = 0; blockIdx < 4; ++blockIdx) {
        AscendC::LocalTensor<half> x;
        x.SetTPosition(AscendC::TPosition::VECIN);
        x.SetAddr(blockIdx * tileBytes);
    }
}
"""
        result = analyze(source, infer_tiling_roles=True)
        assert result.unit.inferred_tiling_bindings == {}

    def test_member_access_on_an_unverified_pointer_is_ignored(self):
        """Only a pointer produced by a tiling cast is in scope."""
        source = """
#include "kernel_operator.h"
struct Unrelated { uint32_t tileBytes; };
extern "C" __global__ __aicore__ void k(__gm__ uint8_t *p)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> q;
    Unrelated *u = reinterpret_cast<Unrelated *>(p);
    pipe.InitBuffer(q, 1, u->tileBytes);
}
"""
        result = analyze(source, infer_tiling_roles=True)
        assert result.unit.inferred_tiling_bindings == {}

    def test_conventional_receiver_without_a_cast_is_in_scope(self):
        """``GET_TILING_DATA`` produces ``tilingData`` with no visible cast."""
        source = """
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void k(__gm__ uint8_t *p)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> q;
    pipe.InitBuffer(q, 1, tilingData->tileBytes);
}
"""
        result = analyze(source, infer_tiling_roles=True)
        assert (
            result.unit.inferred_tiling_bindings.get("tilingData->tileBytes") == 64
        )


class TestRoleValues:
    def test_cube_buffer_gets_the_fractal_width(self):
        source = """
#include "kernel_operator.h"
struct MmTilingData { uint32_t aBytes; };
extern "C" __global__ __aicore__ void k(__gm__ uint8_t *p)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> bL1;
    __gm__ MmTilingData *t = reinterpret_cast<__gm__ MmTilingData *>(p);
    pipe.InitBuffer(bL1, t->aBytes);
}
"""
        result = analyze(source, infer_tiling_roles=True)
        assert result.unit.inferred_tiling_bindings.get("t->aBytes") == 16

    def test_vector_buffer_gets_the_tile_extent(self):
        result = analyze(QUEUED_KERNEL, infer_tiling_roles=True)
        assert result.unit.inferred_tiling_bindings["t->tileBytes"] == 64

    def test_dma_params_field_gets_the_block_size(self):
        source = """
#include "kernel_operator.h"
struct CpTilingData { uint32_t srcGap; };
extern "C" __global__ __aicore__ void k(__gm__ uint8_t *p, __gm__ half *gm)
{
    __gm__ CpTilingData *t = reinterpret_cast<__gm__ CpTilingData *>(p);
    AscendC::GlobalTensor<half> g;
    g.SetGlobalBuffer(gm, 256);
    AscendC::LocalTensor<half> x;
    x.SetTPosition(AscendC::TPosition::VECIN);
    x.SetAddr(0);
    AscendC::DataCopy(x, g, 128, AscendC::DataCopyParams(1, 128, t->srcGap, 0));
}
"""
        result = analyze(source, infer_tiling_roles=True)
        assert result.unit.inferred_tiling_bindings.get("t->srcGap") == 32


class TestPrecedence:
    def test_supplied_tiling_value_wins(self):
        """A manifest value must never be overridden by an inference."""
        result = analyze(
            QUEUED_KERNEL,
            infer_tiling_roles=True,
            tiling_values={"tileBytes": 512},
        )
        assert result.unit.inferred_tiling_bindings.get("t->tileBytes") != 64

    def test_inference_is_off_by_default(self):
        assert analyze(QUEUED_KERNEL).unit.inferred_tiling_bindings == {}


class TestCliFlag:
    def test_flag_is_accepted(self):
        from ascend_analyzer.cli import build_parser

        args = build_parser().parse_args(["k.cpp", "--infer-tiling-roles"])
        assert args.infer_tiling_roles is True

    def test_flag_defaults_off(self):
        from ascend_analyzer.cli import build_parser

        assert build_parser().parse_args(["k.cpp"]).infer_tiling_roles is False

    def test_old_name_based_heuristic_is_gone(self):
        """The spec replaced name matching with role inference outright."""
        import ascend_analyzer.parsing.expr_eval as expr_eval

        assert not hasattr(expr_eval, "heuristic_tiling_value")
        assert not hasattr(expr_eval.ConstEnv, "fallback")
