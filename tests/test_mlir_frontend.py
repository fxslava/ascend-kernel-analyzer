"""Tests for the BiSheng -> MLIR frontend.

Covers the four pipeline stages end to end:

* extraction: recorded AST snapshots replay without a toolchain, and a
  tampered source invalidates its snapshot instead of serving a stale AST;
* bridge: template monomorphisation (queue depths), ``auto`` tensor
  deduction, and heterogeneous MIX lowering into per-core regions;
* verification: the lowered AnalysisUnit drives the real rule checkers -
  both spec fixtures audit completely clean (exit code 0) through
  ``--frontend=mlir``;
* determinism: bank-conflict evaluation reads exact byte offsets off the
  dialect's memref data, so equal/unequal bank placement is a unit test.

Live-toolchain tests are skipped automatically when neither a local
BiSheng/Clang nor the Docker image is reachable; every other test runs from
the committed snapshots under ``tests/data/mlir``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ascend_analyzer import AnalyzerOptions, KernelAnalyzer
from ascend_analyzer.analyzer_mlir import (
    MlirFrontendOptions,
    lower_to_unit,
    parse_source_mlir,
)
from ascend_analyzer.diagnostics import DiagnosticCollector, Severity
from ascend_analyzer.hardware import HardwareModel
from ascend_analyzer.ir.mlir_ascend import (
    AllocBufferOp,
    CoreRegionOp,
    KernelOp,
    MemorySpace,
    MteCopyOp,
    SetFlagOp,
    SsaValue,
    VectorOp,
    WaitFlagOp,
)
from ascend_analyzer.parsing.bisheng_extractor import (
    BishengError,
    discover_toolchain,
    extract_ast,
    filter_main_file,
)
from ascend_analyzer.parsing.mlir_bridge import BridgeOptions, lower_module, split_template_args
from tests.conftest import KERNEL_DIR

NAIVE = KERNEL_DIR / "naive_aiv_fp4_to_fp16.cpp"
CUBE = KERNEL_DIR / "cube_trick_fp4_to_fp16.cpp"
PINGPONG_CLEAN = KERNEL_DIR / "pingpong_clean.cpp"


def _mlir_analyzer(chip: str = "ascend910b") -> KernelAnalyzer:
    return KernelAnalyzer(AnalyzerOptions(chip=chip, frontend="mlir"))


def _module(path: Path):
    source = path.read_text(encoding="utf-8")
    result = extract_ast(str(path), source, chip="ascend910b")
    assert result.errors == [], result.errors[:3]
    return lower_module(result.ast, str(path), source), source


def _ops(kernel: KernelOp, kinds):
    return [op for op in kernel.ops if type(op).__name__ in kinds]


# ---------------------------------------------------------------------------
# Unit: structural template-argument reading (zero regex)
# ---------------------------------------------------------------------------


class TestTemplateArgSplit:
    def test_nested_angles_do_not_split(self):
        args = split_template_args("AscendC::TQue<AscendC::TPosition::VECIN, 2>")
        assert args == ["AscendC::TPosition::VECIN", "2"]

    def test_deep_nesting(self):
        args = split_template_args("A<B<C, D>, E::F<G>, 7>")
        assert args == ["B<C, D>", "E::F<G>", "7"]

    def test_no_arguments(self):
        assert split_template_args("AscendC::TPipe") == []
        assert split_template_args("") == []


# ---------------------------------------------------------------------------
# Extraction + snapshot behaviour
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_snapshots_replay_without_live_toolchain(self):
        source = NAIVE.read_text(encoding="utf-8")
        result = extract_ast(str(NAIVE), source, chip="ascend910b")
        assert result.origin == "snapshot"
        assert result.ast.get("kind") == "TranslationUnitDecl"

    def test_tampered_source_invalidates_snapshot(self, monkeypatch):
        # A snapshot whose source no longer matches must not be served.
        source = NAIVE.read_text(encoding="utf-8") + "\n// tampered\n"
        monkeypatch.delenv("ASCEND_BISHENG_BIN", raising=False)

        def _no_toolchain(*args, **kwargs):
            return None

        monkeypatch.setattr(
            "ascend_analyzer.parsing.bisheng_extractor.discover_toolchain",
            _no_toolchain,
        )
        with pytest.raises(BishengError):
            extract_ast(str(NAIVE), source, chip="ascend910b")

    def test_main_file_filter_keeps_kernel_decl(self):
        source = NAIVE.read_text(encoding="utf-8")
        result = extract_ast(str(NAIVE), source, chip="ascend910b")
        pruned = filter_main_file(result.ast, str(NAIVE))
        names = []

        def walk(node):
            if node.get("kind") == "FunctionDecl":
                names.append(node.get("name"))
            for child in node.get("inner", []) or []:
                walk(child)

        for child in pruned.get("inner", []):
            walk(child)
        assert "naive_aiv_fp4_to_fp16" in names
        # Header clutter is gone.
        assert "SetFlag" not in names and "AllocTensor" not in names

    def test_live_extraction_when_toolchain_present(self):
        if discover_toolchain(source_dir=str(KERNEL_DIR)) is None:
            pytest.skip("no BiSheng/Clang toolchain reachable")
        source = NAIVE.read_text(encoding="utf-8")
        result = extract_ast(str(NAIVE), source, chip="ascend910b",
                             use_cache=False)
        assert result.origin == "live"
        assert result.errors == []


# ---------------------------------------------------------------------------
# Bridge: monomorphisation, auto deduction, MIX regions
# ---------------------------------------------------------------------------


class TestNaiveLowering:
    def test_queue_depth_monomorphised(self):
        module, _ = _module(NAIVE)
        kernel = module.kernels[0]
        allocs = {op.buffer: op for op in kernel.ops
                  if isinstance(op, AllocBufferOp)}
        assert set(allocs) == {"inQue", "outQue", "loScratch", "hiScratch",
                               "codesScratch"}
        # TQue<VECIN, 2> and TQue<VECOUT, 2>: depths survive as integers.
        assert allocs["inQue"].depth == 2
        assert allocs["outQue"].depth == 2
        assert allocs["loScratch"].depth == 1
        # InitBuffer lengths folded: 256 B packed in, 1024 B fp16 out,
        # 1024 + 32 B bank-padded scratch.
        assert allocs["inQue"].byte_size == 256
        assert allocs["outQue"].byte_size == 1024
        assert allocs["loScratch"].byte_size == 1024 + 32

    def test_auto_tensor_deduction(self):
        module, _ = _module(NAIVE)
        kernel = module.kernels[0]
        # `auto packed = inQue.AllocTensor<uint8_t>()` and
        # `auto res = outQue.DeQue<half>()`: the desugared types arrive from
        # the C++ parser, not from any textual guess.
        assert kernel.tensors["packed"]["dtype"] == "uint8_t"
        assert kernel.tensors["packed"]["origin"] == "queue:AllocTensor"
        assert kernel.tensors["res"]["dtype"] == "half"
        assert kernel.tensors["res"]["origin"] == "queue:DeQue"
        assert kernel.tensors["in"]["dtype"] == "uint8_t"

    def test_module_dump_prints_canonical_mlir(self):
        module, _ = _module(NAIVE)
        text = module.dump()
        assert text.startswith('ascend.module @"naive_aiv_fp4_to_fp16.cpp" {')
        assert "ascend.alloc_buffer space=UB bytes=256 depth=2" in text
        assert "ascend.mte_copy" in text and "via MTE2" in text
        assert "ascend.vector" in text
        assert text.rstrip().endswith("}")

    def test_constants_folded(self):
        module, _ = _module(NAIVE)
        assert module.constants["TILE_ELEMS"] == 512
        assert module.constants["TILE_BYTES"] == 256
        assert module.constants["OUT_BYTES"] == 1024


class TestMixLowering:
    def test_heterogeneous_regions_isolated(self):
        module, _ = _module(CUBE)
        assert len(module.kernels) == 1
        kernel = module.kernels[0]
        regions = [op for op in kernel.ops if isinstance(op, CoreRegionOp)]
        cores = [r.core_type.value for r in regions]
        # Init + Process arms for both cores.
        assert cores.count("AIC") >= 2 and cores.count("AIV") >= 2

        aic_ops = [o for r in regions if r.core_type.value == "AIC"
                   for o in r.ops]
        aiv_ops = [o for r in regions if r.core_type.value == "AIV"
                   for o in r.ops]
        # Cube side: the five TBufs, Mmad, Fixpipe and the handshakes.
        assert sum(isinstance(o, SetFlagOp) for o in aic_ops) > 0
        assert sum(isinstance(o, WaitFlagOp) for o in aic_ops) > 0
        assert any(getattr(o, "api_name", "") == "Mmad" for o in aic_ops)
        assert any(getattr(o, "pipe_route", "") == "FIX" for o in aic_ops)
        # Vector side: queue-driven compute only, no cube intrinsics.
        assert any(isinstance(o, VectorOp) for o in aiv_ops)
        assert not any(isinstance(o, SetFlagOp) for o in aiv_ops)

    def test_stage_classes_never_collide(self):
        # CubeStage and VectorStage both define Init()/Process(); every
        # buffer belongs to exactly one stage's region.
        module, _ = _module(CUBE)
        kernel = module.kernels[0]
        buffers = {}
        for op in kernel.ops:
            if isinstance(op, CoreRegionOp):
                for nested in op.ops:
                    if isinstance(nested, AllocBufferOp):
                        buffers.setdefault(nested.buffer, op.core_type.value)
        cube_buffers = {b for b, c in buffers.items() if c == "AIC"}
        vec_buffers = {b for b, c in buffers.items() if c == "AIV"}
        assert any("l1A" in b for b in cube_buffers)
        assert any("inQue" in b for b in vec_buffers)
        assert not (cube_buffers & vec_buffers)

    def test_flag_routes_and_event_ids_resolve(self):
        module, _ = _module(CUBE)
        kernel = module.kernels[0]
        channels = set()
        for op in kernel.ops:
            if isinstance(op, CoreRegionOp):
                for nested in op.ops:
                    if isinstance(nested, (SetFlagOp, WaitFlagOp)):
                        assert nested.pipe_route, "unresolved HardEvent route"
                        assert nested.event_id in (0, 1)
                        channels.add((nested.pipe_route, nested.event_id))
        # The double-buffered pipeline uses both event slots on several
        # routes; routes come from evaluated enum values, joined back to
        # names through the AST's own HardEvent layout.
        routes = {route for route, _ in channels}
        assert {"MTE2_MTE1", "MTE1_M", "M_FIX", "MTE1_MTE2", "M_MTE1"} <= routes

    def test_loop_trip_counts_folded(self):
        module, _ = _module(CUBE)
        kernel = module.kernels[0]
        trips = {tuple(sorted(l.items())) for l in kernel.loops}
        assert any("t" in l["induction"] and l["trip_count"] == 4
                   for l in kernel.loops)


# ---------------------------------------------------------------------------
# Verification engine: full checker runs
# ---------------------------------------------------------------------------


class TestCleanAudits:
    def test_naive_aiv_fp4_to_fp16_clean(self):
        result = _mlir_analyzer().analyze_file(NAIVE)
        assert result.verdict == "accepted"
        assert result.diagnostics == []
        assert result.exit_code() == 0

    def test_cube_trick_fp4_to_fp16_clean(self):
        result = _mlir_analyzer().analyze_file(CUBE)
        assert result.verdict == "accepted"
        assert result.diagnostics == []
        assert result.exit_code() == 0

    def test_frontends_agree_on_both_fixtures(self):
        tree_sitter = KernelAnalyzer(AnalyzerOptions(chip="ascend910b"))
        for fixture in (NAIVE, CUBE):
            mlir = _mlir_analyzer().analyze_file(fixture)
            classic = tree_sitter.analyze_file(fixture)
            assert mlir.codes() == classic.codes() == []

    def test_layout_is_concrete(self):
        # The whole point of the MLIR path: offsets are integers, so the
        # memory audits never need the solver.
        result = _mlir_analyzer().analyze_file(CUBE)
        kernel = result.unit.kernels[0]
        static = [t for t in kernel.tensors.values() if t.is_sram]
        assert static, "no SRAM tensors lowered"
        assert all(t.is_fully_static for t in static)


class TestFallback:
    def test_unreachable_toolchain_falls_back(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise BishengError("no toolchain in this test")

        monkeypatch.setattr(
            "ascend_analyzer.analyzer_mlir.extract_ast", _raise)
        source = PINGPONG_CLEAN.read_text(encoding="utf-8")
        collector = DiagnosticCollector()
        unit = parse_source_mlir(str(PINGPONG_CLEAN), source,
                                 HardwareModel.for_chip("ascend910b"),
                                 collector)
        # The tree-sitter frontend parsed the unit: one clean kernel.
        assert [k.name for k in unit.kernels] == ["vec_add_pingpong_clean"]
        infos = [d for d in collector.items
                 if d.severity is Severity.INFO]
        assert infos and "falling back" in infos[0].message

    def test_cli_frontend_flag_end_to_end(self, capsys):
        from ascend_analyzer.cli import main

        code = main(["--frontend", "mlir", str(NAIVE)])
        assert code == 0
        captured = capsys.readouterr()
        assert "naive_aiv_fp4_to_fp16" in captured.out


# ---------------------------------------------------------------------------
# Determinism: bank conflicts from exact memref offsets
# ---------------------------------------------------------------------------


def _bank_kernel(lo_offset: int, hi_offset: int):
    """A kernel trace whose Or reads two UB tensors at given byte offsets."""
    from ascend_analyzer.diagnostics import SourceLoc
    from ascend_analyzer.hardware import PhysicalDomain, TPosition, Pipe
    from ascend_analyzer.ir import ArgRef, KernelIR, TensorDecl
    from ascend_analyzer.ir.kernel_ir import ApiCallOp
    from ascend_analyzer.symbolic import Const

    kernel = KernelIR(name="bank_probe",
                      loc=SourceLoc(file="probe", line=1))
    for name, offset in (("lo", lo_offset), ("hi", hi_offset)):
        kernel.tensors[name] = TensorDecl(
            name=name, loc=SourceLoc(file="probe", line=1),
            position=TPosition.VECCALC, domain=PhysicalDomain.UB,
            dtype="half", elem_size=2, byte_offset=Const(offset),
            byte_size=Const(128), origin="mlir:declaration",
            first_use=1, last_use=1)
    kernel.ops.append(ApiCallOp(
        index=1, loc=SourceLoc(file="probe", line=1), pipe=Pipe.V,
        scope_id=0, name="Or",
        args=(ArgRef(index=0, text="lo", tensor="lo"),
              ArgRef(index=1, text="lo", tensor="lo"),
              ArgRef(index=2, text="hi", tensor="hi"),
              ArgRef(index=3, text="64", value=64)),
        writes=("lo",), reads=("lo", "hi"), text="Or(lo, hi, 64)"))
    return kernel


class TestDeterministicBankConflicts:
    def _run(self, kernel):
        from ascend_analyzer.checkers.base import CheckerContext
        from ascend_analyzer.checkers.memory import MemoryChecker

        hardware = HardwareModel.for_chip("ascend910b")
        collector = DiagnosticCollector()
        unit = _unit_of(kernel)
        ctx = CheckerContext(unit=unit, hardware=hardware,
                             diagnostics=collector)
        MemoryChecker(ctx).run_all()
        return [d for d in collector.items if d.code.value == "AKA3006"]

    def test_same_bank_dual_operand_conflicts(self):
        # 0x500 and 0x900: (offset // 32) % 8 == 0 for both.
        findings = self._run(_bank_kernel(0x500, 0x900))
        assert len(findings) == 1
        assert "bank" in findings[0].message.lower()

    def test_adjacent_bank_stays_silent(self):
        # One 32-byte block apart in bank index: no conflict.
        findings = self._run(_bank_kernel(0x500, 0x520))
        assert findings == []


def _unit_of(kernel):
    from ascend_analyzer.ir import AnalysisUnit

    return AnalysisUnit(path="probe", source="// probe\n", kernels=[kernel])
