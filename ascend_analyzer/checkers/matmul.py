"""Ascend 950PR matmul-specific memory and signature checks."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from ..diagnostics import Code, Severity
from ..hardware import KIB, TPosition
from ..ir import KernelIR
from ..symbolic import to_int
from .base import Checker
from .source_utils import code_only, iter_calls, source_loc

__all__ = ["MatmulChecker"]

_LOW_BIT_TYPES = (
    "fp4x2_e1m2_t",
    "fp4x2_e2m1_t",
    "fp8_e4m3fn_t",
    "fp8_e5m2_t",
    "hifloat8_t",
)
_FP4_TYPES = frozenset(_LOW_BIT_TYPES[:2])
_FP8_TYPES = frozenset(_LOW_BIT_TYPES[2:4])
_MMAD_NAMES = ("asc_mmad", "asc_mmad_sync", "asc_mmad_mx", "asc_mmad_mx_sync")


class MatmulChecker(Checker):
    """Enforce 950PR sparse, low-bit, GEMV and TSCM contracts."""

    name = "matmul"

    def check(self, kernel: KernelIR) -> None:
        if self.hw.chip.name != "ascend950pr":
            return
        source = code_only(self.ctx.unit.source)
        self._check_sparse(source)
        self._check_mmad_signatures(kernel)
        self._check_tscm(kernel)

    def _check_sparse(self, source: str) -> None:
        markers = [match.start() for match in re.finditer(r"\bSparseMatmulType\b", source)]
        sparse_calls = list(iter_calls(source, "SetSparse"))
        for call in sparse_calls:
            if not call.arguments or call.arguments[0].strip().lower() not in {"true", "1"}:
                continue
            markers.append(call.start)
        for offset in markers:
            self.diags.add(
                Code.UNSUPPORTED_SPARSE_MATMUL,
                Severity.FATAL,
                "4:2 structured sparsity is unsupported on Ascend 950PR",
                source_loc(self.ctx.unit.path, source, offset),
                hardware_domain="PIPE_M",
                remediation=(
                    "Remove SparseMatmulType/SetSparse(true); use MxMatmul with "
                    "MatrixMadType::MXMODE and supported scale operands instead."
                ),
                replacement="MxMatmul / MatrixMadType::MXMODE",
            )

    def _check_mmad_signatures(self, kernel: KernelIR) -> None:
        for call in kernel.api_calls():
            if call.name not in _MMAD_NAMES:
                continue
            if len(call.args) != 10:
                self.diags.add(
                    Code.INVALID_MATMUL_TYPE, Severity.FATAL,
                    f"{call.name} requires 10 positional arguments, got {len(call.args)}",
                    call.loc, hardware_domain="PIPE_M",
                    remediation="Use (c, a, b, left_height, n_dim, right_width, "
                    "unit_flag, disable_gemv, c_matrix_source, c_matrix_init_val).",
                )
                continue
            tensors = [kernel.tensors.get(arg.tensor or "") for arg in call.args[:3]]
            types = [tensor.dtype if tensor else None for tensor in tensors]
            c, a, b = types
            mx = "_mx" in call.name
            pairs = {(x, y) for family in (_FP4_TYPES, _FP8_TYPES)
                     for x in family for y in family} if mx else (
                {(x, y) for x in _FP8_TYPES for y in _FP8_TYPES}
                | {(t, t) for t in ("half", "bfloat16_t", "float", "int8_t", "hifloat8_t")}
            )
            invalid_pair = a is not None and b is not None and (a, b) not in pairs
            expected_c = "int32_t" if a == b == "int8_t" and not mx else "float"
            invalid_c = c is not None and c != expected_c
            if invalid_pair or invalid_c:
                self.diags.add(
                    Code.INVALID_MATMUL_TYPE, Severity.FATAL,
                    f"{call.name} has unsupported (C, A, B) types {tuple(types)}",
                    call.loc, hardware_domain="PIPE_M",
                    remediation="Use float accumulation for matched FP8 pairs or "
                    "hifloat8 pairs; FP4 pairs require asc_mmad_mx. MX accepts "
                    "FP4/FP4 or FP8/FP8, never mixed families or hifloat8.",
                    api=call.name, types=tuple(types),
                )
            if any(t is None for t in types):
                self.diags.add(
                    Code.SYMBOLIC_OFFSET, Severity.INFO,
                    f"{call.name} operand types could not all be resolved",
                    call.loc, hardware_domain="PIPE_M",
                    remediation="Provide typed L0C, L0A and L0B operands to verify the overload.",
                )
            # Parameter positions are identical across the audited overloads.
            height, disabled = call.args[3].value, call.args[7].value
            if height == 1 and disabled == 1:
                self.diags.add(
                    Code.INVALID_MATMUL_TYPE, Severity.FATAL,
                    f"{call.name} disables GEMV while left_height is 1",
                    call.loc, hardware_domain="PIPE_M",
                    remediation="Set disable_gemv to false for left_height == 1 "
                    "so hardware GEMV can read the ND layout from L0A.",
                    api=call.name, left_height=height, disable_gemv=True,
                )
            elif height is None or (height == 1 and disabled is None):
                self.diags.add(
                    Code.SYMBOLIC_OFFSET, Severity.INFO,
                    f"{call.name} GEMV configuration is not statically decidable",
                    call.loc, hardware_domain="PIPE_M",
                    remediation="Ensure disable_gemv is false whenever left_height is 1.",
                )

    def _check_tscm(self, kernel: KernelIR) -> None:
        tensors = [
            tensor
            for tensor in kernel.tensors.values()
            if tensor.position is TPosition.TSCM
        ]
        counted: set[Tuple[str, Optional[int], Optional[int]]] = set()
        allocated = 0
        for tensor in tensors:
            if tensor.source_buffer is not None:
                key = (f"buffer:{tensor.source_buffer}", None, None)
            elif tensor.offset_value is not None and tensor.size_value is not None:
                key = ("range", tensor.offset_value, tensor.size_value)
            else:
                key = (f"tensor:{tensor.name}", None, None)
            if key in counted:
                continue
            counted.add(key)
            buffer_size = (
                kernel.buffer_sizes.get(tensor.source_buffer)
                if tensor.source_buffer is not None
                else None
            )
            size = buffer_size if buffer_size is not None else tensor.size_value
            if size is not None:
                allocated += size
        represented_buffers = {
            tensor.source_buffer for tensor in tensors if tensor.source_buffer is not None
        }
        for name, position in kernel.buffer_positions.items():
            if position is TPosition.TSCM and name not in represented_buffers:
                allocated += kernel.buffer_sizes.get(name, 0)
            if position is TPosition.TSCM and name not in kernel.buffer_sizes:
                self.diags.add(
                    Code.SYMBOLIC_OFFSET, Severity.INFO,
                    f"TSCM capacity cannot be proven for unsized buffer {name}",
                    kernel.loc, hardware_domain="TSCM",
                    remediation="Provide the complete single-core operand allocation size.",
                )
        if allocated > 512 * KIB:
            self.diags.add(
                Code.TSCM_RESIDENCY,
                Severity.FATAL,
                f"TSCM allocations total {allocated} B, exceeding the 512 KiB L1 capacity",
                tensors[0].loc if tensors else kernel.loc,
                hardware_domain="TSCM",
                remediation="Keep the complete resident TSCM operand set within 512 KiB.",
                allocated_bytes=allocated,
                capacity_bytes=512 * KIB,
            )
        tscm_names = {tensor.name for tensor in tensors}
        for op in kernel.api_calls():
            if op.name not in {"DataCopy", "DataCopyPad"}:
                continue
            written_tscm = [name for name in op.writes if name in tscm_names]
            if not written_tscm:
                continue
            partial = any(
                len(op.args) > 2 and op.args[2].value is not None
                and to_int(kernel.tensors[name].elem_count) is not None
                and op.args[2].value < to_int(kernel.tensors[name].elem_count)
                for name in written_tscm
            )
            if op.loop_id is None and not partial:
                continue
            self.diags.add(
                Code.TSCM_RESIDENCY,
                Severity.FATAL,
                "TSCM staging uses partial tiles or occurs inside an iteration loop",
                op.loc,
                hardware_domain="TSCM",
                remediation=(
                    "Stage the complete TSCM operand before the consuming loop and "
                    "keep it resident through the final matmul use."
                ),
                tensors=tuple(written_tscm),
                loop_id=op.loop_id,
            )
