"""Memory layout verification: capacity, alignment, aliasing and domains.

Four fatal classes of bug are detected here:

1. **SRAM capacity overflow** - a tensor's byte range leaves its physical
   domain.  With symbolic offsets the solver reports the loop iteration that
   first escapes, not just that something might.
2. **Alignment violations** - DaVinci addresses SRAM in 32-byte blocks.  A
   misaligned *base* makes the DMA engine read or write the wrong bytes; a
   misaligned *length* is just as dangerous, because the tail block is
   transferred whole and silently clobbers whatever follows it.
3. **Buffer aliasing** - two tensors that are live at the same time in the
   same physical domain must occupy disjoint byte ranges.  Ping/pong halves of
   a double buffer and in/out pairs are precisely the tensors that must not
   overlap, so this is where double-buffering bugs surface.
4. **Domain mismatch** - Ascend C hides the address space inside
   ``LocalTensor<T>``, so handing a UB-resident tensor to a Cube API that
   reads L0A type-checks in C++ and fails on hardware.  Operand domains are
   validated against the signature table in :mod:`ascend_analyzer.apis`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set

from ..apis import TRANSFER_PIPE, ArgRole, lookup_api
from ..diagnostics import Code, Severity
from ..hardware import PhysicalDomain, Pipe
from ..ir import ApiCallOp, KernelIR, TensorDecl
from ..solver import Finding, MemorySolver, Verdict, make_solver
from ..symbolic import is_decidable, render, to_int
from .base import Checker, CheckerContext

__all__ = ["MemoryChecker", "DomainUsage"]

#: Source tokens whose presence marks a translation unit as using the 351x
#: isomorphic SIMD/SIMT vector-execution path (DataCache-guarded UB budget).
_SIMT_MARKERS = (
    "__simt_vf__",
    "__simt_callee__",
    "asc_call_vf",
)


@dataclass(frozen=True)
class DomainUsage:
    """Aggregate footprint of one physical domain, for the memory map."""

    domain: PhysicalDomain
    capacity_bytes: int
    allocated_bytes: int
    high_water_bytes: int
    tensor_count: int
    symbolic_count: int

    @property
    def utilization(self) -> float:
        if self.capacity_bytes <= 0:
            return 0.0
        return self.high_water_bytes / self.capacity_bytes

    @property
    def gap_bytes(self) -> int:
        """Bytes below the high-water mark that no tensor claims."""
        return max(0, self.high_water_bytes - self.allocated_bytes)

    def to_json(self) -> Dict[str, object]:
        return {
            "domain": self.domain.value,
            "capacity_bytes": self.capacity_bytes,
            "allocated_bytes": self.allocated_bytes,
            "high_water_bytes": self.high_water_bytes,
            "utilization": round(self.utilization, 4),
            "gap_bytes": self.gap_bytes,
            "tensor_count": self.tensor_count,
            "symbolic_count": self.symbolic_count,
        }


class MemoryChecker(Checker):
    """Validates the static SRAM layout of a kernel."""

    name = "memory"

    def __init__(
        self,
        context: CheckerContext,
        solver: Optional[MemorySolver] = None,
    ) -> None:
        super().__init__(context)
        self.solver = solver or make_solver("auto")
        #: tensor name -> ids of the loops whose body references it.
        self._loops_using: Dict[str, Set[int]] = {}

    # -- entry point --------------------------------------------------------

    def check(self, kernel: KernelIR) -> None:
        self._loops_using = self._index_loop_usage(kernel)

        # On-core tensors get the full treatment. Tensors whose domain could
        # not be resolved are not skipped silently - an unverifiable buffer is
        # itself worth reporting - but only when the kernel actually binds or
        # uses them, so a declared-and-unused tensor stays quiet.
        on_core = [t for t in kernel.tensors.values() if t.is_sram]
        undetermined = [
            t
            for t in kernel.tensors.values()
            if t.domain is PhysicalDomain.UNKNOWN
            and (t.first_use is not None or not t.unbound)
        ]

        for tensor in (*on_core, *undetermined):
            self._check_extent_known(tensor)
        for tensor in on_core:
            self._check_alignment(tensor)
            self._check_bounds(tensor)

        self._check_aliasing(kernel, on_core)
        self._check_api_domains(kernel)
        self._check_vector_bank_conflicts(kernel)
        self._check_simt_datacache(kernel)
        self._check_target_features(kernel)

        tensors = on_core

        usage = self._summarize(kernel, tensors)
        self.ctx.publish(f"memory_usage::{kernel.name}", [u.to_json() for u in usage])
        self._check_fragmentation(kernel, usage)

    # -- individual tensors -------------------------------------------------

    def _check_extent_known(self, tensor: TensorDecl) -> None:
        """Flag tensors whose offset or size the analyzer could not resolve."""
        if tensor.domain is PhysicalDomain.UNKNOWN:
            self.diags.add(
                Code.UNKNOWN_DOMAIN,
                Severity.FATAL if self.ctx.strict else Severity.WARNING,
                f"tensor {tensor.name!r} has no resolvable memory domain, so its "
                "layout cannot be verified",
                tensor.loc,
                hardware_domain="UNKNOWN",
                remediation=(
                    f"Call {tensor.name}.SetTPosition(AscendC::TPosition::VECIN) "
                    "(or the correct position), allocate it from a TBuf with an "
                    "explicit TPosition, or annotate it with "
                    f"'// @ascend-layout: name={tensor.name} pos=VECIN offset=... count=...'."
                ),
                tensor=tensor.name,
                origin=tensor.origin,
            )
            return

        # A symbolic but *bounded* extent is fine: the solver reasons about it
        # exactly and will report a violation with the offending iteration.
        # Only an extent the solver cannot usefully constrain is a gap in
        # coverage worth reporting.
        missing: List[str] = []
        if not is_decidable(tensor.byte_offset):
            missing.append("base offset")
        if not is_decidable(tensor.byte_size):
            missing.append("byte size")
        if not missing:
            return

        self.diags.add(
            Code.SYMBOLIC_OFFSET,
            Severity.FATAL if self.ctx.strict else Severity.WARNING,
            f"tensor {tensor.name!r} has an unresolvable {' and '.join(missing)} "
            f"(offset={render(tensor.byte_offset)}, size={render(tensor.byte_size)}); "
            "capacity, alignment and aliasing cannot be verified for it",
            tensor.loc,
            hardware_domain=tensor.domain.value,
            remediation=(
                "Bind the layout with constexpr values, or annotate it with "
                f"'// @ascend-layout: name={tensor.name} pos=... offset=... count=...' "
                "so the full layout can be verified."
            ),
            tensor=tensor.name,
            offset_expr=render(tensor.byte_offset),
            size_expr=render(tensor.byte_size),
        )

    def _check_alignment(self, tensor: TensorDecl) -> None:
        domain = tensor.domain
        if domain is PhysicalDomain.UNKNOWN:
            return
        base_align = self.hw.base_alignment(domain)
        size_align = self.hw.size_alignment(domain)
        block = self.hw.chip.block_bytes

        if is_decidable(tensor.byte_offset):
            self._check_base_alignment(tensor, base_align)
        if is_decidable(tensor.byte_size):
            self._check_size_alignment(tensor, size_align, block)

    def _check_base_alignment(self, tensor: TensorDecl, base_align: int) -> None:
        domain = tensor.domain
        base = self.solver.check_alignment(tensor.byte_offset, base_align)
        if base.verdict is Verdict.VIOLATED:
            self.diags.add(
                Code.BASE_MISALIGNED,
                Severity.FATAL,
                f"base address of {tensor.name!r} is not {base_align}-byte aligned: "
                f"{self._value_text(tensor.byte_offset)} {base.describe_counterexample()}",
                tensor.loc,
                hardware_domain=domain.value,
                remediation=self._alignment_remediation(
                    tensor, tensor.offset_value, base_align, "base offset"
                ),
                tensor=tensor.name,
                required_alignment=base_align,
                offset=tensor.offset_value,
                offset_expr=render(tensor.byte_offset),
                counterexample=base.counterexample,
            )

    def _check_size_alignment(
        self, tensor: TensorDecl, size_align: int, block: int
    ) -> None:
        domain = tensor.domain
        size = self.solver.check_alignment(tensor.byte_size, size_align)
        if size.verdict is Verdict.VIOLATED:
            self.diags.add(
                Code.SIZE_MISALIGNED,
                Severity.FATAL,
                f"byte size of {tensor.name!r} is not {size_align}-byte aligned: "
                f"{self._value_text(tensor.byte_size)} {size.describe_counterexample()}",
                tensor.loc,
                hardware_domain=domain.value,
                remediation=(
                    f"DMA and vector instructions move whole {block}-byte blocks, so "
                    f"a length that is not a multiple of {size_align} B spills into the "
                    "following bytes. Pad the element count up to a "
                    f"{size_align // (tensor.elem_size or 1)}-element multiple "
                    f"({size_align} B), or use DataCopyPad for a ragged tail. "
                    + self._alignment_remediation(
                        tensor, tensor.size_value, size_align, "byte size"
                    )
                ),
                tensor=tensor.name,
                required_alignment=size_align,
                size=tensor.size_value,
                size_expr=render(tensor.byte_size),
                counterexample=size.counterexample,
            )

    def _check_bounds(self, tensor: TensorDecl) -> None:
        domain = tensor.domain
        capacity = self.hw.capacity(domain)
        if capacity is None:
            return
        if not (is_decidable(tensor.byte_offset) and is_decidable(tensor.byte_size)):
            return  # already reported as unresolved by _check_extent_known

        offset = tensor.offset_value
        if offset is not None and offset < 0:
            self.diags.add(
                Code.NEGATIVE_OFFSET,
                Severity.FATAL,
                f"tensor {tensor.name!r} has a negative base offset ({offset})",
                tensor.loc,
                hardware_domain=domain.value,
                remediation="Byte offsets into SRAM are unsigned; check for an "
                "underflowing subtraction in the layout arithmetic.",
                tensor=tensor.name,
                offset=offset,
            )
            return

        finding = self.solver.check_bounds(tensor.byte_offset, tensor.byte_size, capacity)
        if finding.verdict is not Verdict.VIOLATED:
            return

        end = tensor.end_value
        overflow = None if end is None else end - capacity
        self.diags.add(
            Code.SRAM_OVERFLOW,
            Severity.FATAL,
            f"tensor {tensor.name!r} does not fit in {domain.value}: "
            f"{tensor.describe_range()} against a {capacity} B "
            f"({capacity // 1024} KiB) capacity"
            + (f"; {finding.describe_counterexample()}" if finding.witness else ""),
            tensor.loc,
            hardware_domain=domain.value,
            remediation=self._overflow_remediation(
                tensor, domain, capacity, overflow, finding
            ),
            tensor=tensor.name,
            capacity_bytes=capacity,
            range_end=end,
            overflow_bytes=overflow,
            counterexample=finding.counterexample,
        )

    def _overflow_remediation(
        self,
        tensor: TensorDecl,
        domain: PhysicalDomain,
        capacity: int,
        overflow: Optional[int],
        finding: Finding,
    ) -> str:
        """Advice tailored to whether the offset is constant or loop-varying."""
        if finding.counterexample:
            # A loop-varying offset: the footprint grows per iteration, so the
            # fix is to recycle slots rather than to shrink one allocation.
            names = ", ".join(
                f"{name} = {value}" for name, value in sorted(finding.counterexample.items())
            )
            return (
                f"The offset of {tensor.name!r} grows with the loop, so the "
                f"footprint is unbounded: it first leaves {domain.value} at "
                f"{names}. Recycle a fixed set of slots instead of allocating one "
                "per iteration - two alternating (ping/pong) slots give the same "
                "pipeline overlap with a constant footprint - or tile the loop so "
                f"at most {capacity // max(1, tensor.size_value or 1)} slots are "
                f"live at once."
            )
        if overflow is not None and overflow > 0 and tensor.elem_size:
            shrink = -(-overflow // tensor.elem_size)
            return (
                f"You are {overflow} B over the {capacity} B {domain.value} "
                f"capacity. Shrink the element count by {shrink} elements, lower "
                f"the base offset, or move {tensor.name!r} to a larger domain."
            )
        return (
            f"Reduce the tile size or move {tensor.name!r} to a larger domain; "
            f"{domain.value} capacity is {capacity} B."
        )

    # -- pairwise aliasing --------------------------------------------------

    def _check_aliasing(self, kernel: KernelIR, tensors: Sequence[TensorDecl]) -> None:
        by_domain: Dict[PhysicalDomain, List[TensorDecl]] = {}
        for tensor in tensors:
            if tensor.domain is PhysicalDomain.UNKNOWN:
                continue
            by_domain.setdefault(tensor.domain, []).append(tensor)

        for domain, group in by_domain.items():
            for left, right in combinations(sorted(group, key=lambda t: t.name), 2):
                if not self._may_collide(kernel, left, right):
                    continue
                if not all(
                    is_decidable(e)
                    for e in (
                        left.byte_offset, left.byte_size,
                        right.byte_offset, right.byte_size,
                    )
                ):
                    continue  # unresolved layout; reported by _check_extent_known
                finding = self.solver.check_disjoint(
                    left.byte_offset, left.byte_size,
                    right.byte_offset, right.byte_size,
                )
                if finding.verdict is Verdict.VIOLATED:
                    self._report_collision(kernel, domain, left, right, finding)
                elif finding.verdict is Verdict.UNKNOWN and self.ctx.strict:
                    self.diags.add(
                        Code.SYMBOLIC_OFFSET,
                        Severity.WARNING,
                        f"cannot prove {left.name!r} and {right.name!r} occupy "
                        f"disjoint {domain.value} ranges",
                        left.loc,
                        hardware_domain=domain.value,
                        remediation="Give both tensors constexpr offsets and sizes, "
                        "or annotate them with @ascend-layout.",
                        related=[(f"other tensor {right.name!r}", right.loc)],
                    )

    def _may_collide(
        self, kernel: KernelIR, left: TensorDecl, right: TensorDecl
    ) -> bool:
        """``True`` when two tensors are candidates for an aliasing report."""
        # Tensors explicitly placed in the same reuse group are allowed to
        # share storage - that is deliberate buffer recycling.
        if left.reuse_group and left.reuse_group == right.reuse_group:
            return False
        # An alias/view of another tensor is intentionally the same storage.
        if "alias" in left.origin or "view" in left.origin:
            return False
        if "alias" in right.origin or "view" in right.origin:
            return False
        return self._live_ranges_overlap(kernel, left, right)

    def _live_ranges_overlap(
        self, kernel: KernelIR, left: TensorDecl, right: TensorDecl
    ) -> bool:
        """Decide whether two tensors can hold live data at the same instant.

        Lexical trace order alone is *not* sufficient here.  The point of a
        software-pipelined kernel is that MTE2 is filling the next tile while
        the vector unit is still computing the current one and MTE3 is still
        draining the previous one.  Two tensors whose uses do not overlap in
        program order are therefore routinely live simultaneously on hardware.

        So any two tensors referenced from the body of the same loop are
        treated as concurrent, which is what makes ping/pong and in/out
        collisions visible.  Deliberate buffer recycling is declared with
        ``@ascend-reuse-group`` rather than inferred from source order.
        """
        shared_loops = self._loops_using.get(left.name, set()) & self._loops_using.get(
            right.name, set()
        )
        if shared_loops:
            return True
        a_start, a_end = left.live_range(kernel.scopes)
        b_start, b_end = right.live_range(kernel.scopes)
        return a_start <= b_end and b_start <= a_end

    @staticmethod
    def _index_loop_usage(kernel: KernelIR) -> Dict[str, Set[int]]:
        """Map each tensor name to the loops whose body references it."""
        usage: Dict[str, Set[int]] = {}
        for op in kernel.api_calls():
            if op.loop_id is None:
                continue
            for name in (*op.writes, *op.reads):
                usage.setdefault(name, set()).add(op.loop_id)
        return usage

    def _concurrency_reason(self, left: TensorDecl, right: TensorDecl) -> str:
        shared = self._loops_using.get(left.name, set()) & self._loops_using.get(
            right.name, set()
        )
        if shared:
            return (
                "both are used inside the same loop, where the MTE2, Vector and MTE3 "
                "pipelines overlap across iterations, so both buffers hold live data "
                "at the same time"
            )
        return "their live ranges overlap"

    def _report_collision(
        self,
        kernel: KernelIR,
        domain: PhysicalDomain,
        left: TensorDecl,
        right: TensorDecl,
        finding: Finding,
    ) -> None:
        overlap_text = finding.witness or "ranges can overlap"
        suggestion = self._suggest_separation(left, right, domain)
        self.diags.add(
            Code.BUFFER_COLLISION,
            Severity.FATAL,
            f"tensors {left.name!r} and {right.name!r} occupy overlapping "
            f"{domain.value} ranges while both are live: "
            f"{left.name}={left.describe_range()}, "
            f"{right.name}={right.describe_range()} - {overlap_text}. They are "
            f"concurrent because {self._concurrency_reason(left, right)}",
            left.loc,
            hardware_domain=domain.value,
            remediation=suggestion,
            related=[(f"colliding tensor {right.name!r}", right.loc)],
            tensors=[left.name, right.name],
            left_range=[left.offset_value, left.end_value],
            right_range=[right.offset_value, right.end_value],
            counterexample=finding.counterexample,
        )

    def _suggest_separation(
        self, left: TensorDecl, right: TensorDecl, domain: PhysicalDomain
    ) -> str:
        align = self.hw.base_alignment(domain)
        lower, upper = sorted(
            (left, right),
            key=lambda t: (t.offset_value if t.offset_value is not None else 0),
        )
        lower_end = lower.end_value
        if lower_end is None:
            return (
                f"Give {left.name!r} and {right.name!r} disjoint {domain.value} ranges, "
                f"both {align}-byte aligned. If the overlap is deliberate buffer reuse, "
                "declare it with '// @ascend-reuse-group: group=<name> names="
                f"{left.name},{right.name}'."
            )
        suggested = -(-lower_end // align) * align
        return (
            f"Move {upper.name!r} to byte {suggested} (0x{suggested:X}) or beyond: "
            f"{lower.name!r} ends at {lower_end} (0x{lower_end:X}) and the next "
            f"{align}-byte boundary is {suggested}. If the overlap is deliberate "
            "buffer reuse, declare it with '// @ascend-reuse-group: group=<name> "
            f"names={left.name},{right.name}'."
        )

    # -- API operand domains ------------------------------------------------

    def _check_api_domains(self, kernel: KernelIR) -> None:
        for op in kernel.api_calls():
            spec = lookup_api(op.name)
            if spec is None:
                continue
            for arg in op.args:
                if not arg.tensor:
                    continue
                tensor = kernel.tensors.get(arg.tensor)
                param = spec.param_at(arg.index)
                if tensor is None or param is None or not param.is_tensor:
                    continue
                if tensor.domain is PhysicalDomain.UNKNOWN:
                    continue
                assert param.allowed is not None
                if tensor.domain in param.allowed:
                    continue
                self._report_domain_mismatch(op, arg, tensor, spec, param)
            self._check_transfer_legality(kernel, op)

    def _report_domain_mismatch(self, op, arg, tensor, spec, param) -> None:
        allowed = param.describe_allowed()
        self.diags.add(
            Code.DOMAIN_MISMATCH,
            Severity.FATAL,
            f"{op.name}() argument {arg.index} ({param.name!r}) must live in "
            f"{allowed}, but {tensor.name!r} is in {tensor.domain.value} "
            f"(TPosition::{tensor.position.value if tensor.position else '?'})",
            op.loc,
            hardware_domain=tensor.domain.value,
            remediation=(
                f"{spec.doc} Stage {tensor.name!r} into {allowed} first - for "
                f"example DataCopy it from {tensor.domain.value} to {allowed}, or "
                f"declare it with a {allowed}-mapped TPosition."
            ),
            related=[(f"declaration of {tensor.name!r}", tensor.loc)],
            api=op.name,
            argument_index=arg.index,
            parameter=param.name,
            actual_domain=tensor.domain.value,
            allowed_domains=sorted(d.value for d in param.allowed),
        )

    def _check_transfer_legality(self, kernel: KernelIR, op: ApiCallOp) -> None:
        """Reject ``DataCopy`` between domains with no direct hardware path."""
        if op.name not in {"DataCopy", "DataCopyPad"} or len(op.args) < 2:
            return
        dst = self._domain_of(kernel, op.args[0].tensor)
        src = self._domain_of(kernel, op.args[1].tensor)
        if dst is PhysicalDomain.UNKNOWN or src is PhysicalDomain.UNKNOWN:
            return
        if (dst, src) in TRANSFER_PIPE:
            return
        self.diags.add(
            Code.DOMAIN_MISMATCH,
            Severity.FATAL,
            f"{op.name}() from {src.value} to {dst.value} is not a legal direct "
            "transfer on this architecture",
            op.loc,
            hardware_domain=f"{src.value}->{dst.value}",
            remediation=(
                f"Stage the data through an intermediate buffer. {src.value} reaches "
                f"{dst.value} via "
                + (
                    "L1 (GM -> L1 -> L0A/L0B)"
                    if dst in {PhysicalDomain.L0A, PhysicalDomain.L0B}
                    else "UB or L1"
                )
                + "."
            ),
            api=op.name,
            src_domain=src.value,
            dst_domain=dst.value,
        )

    @staticmethod
    def _domain_of(kernel: KernelIR, name: Optional[str]) -> PhysicalDomain:
        if not name:
            return PhysicalDomain.UNKNOWN
        tensor = kernel.tensors.get(name)
        return tensor.domain if tensor is not None else PhysicalDomain.UNKNOWN

    # -- vector ALU bank conflicts (AKA3006) ---------------------------------

    def _check_vector_bank_conflicts(self, kernel: KernelIR) -> None:
        """Flag dual-operand vector reads that share one UB bank (AKA3006).

        The Unified Buffer is an interleaved 8-bank structure addressed in
        32-byte quantization blocks.  When one Vector ALU instruction reads
        both source operands from base offsets that land in the *same* bank
        (their 32-byte block delta is a multiple of 8), the two read ports
        contend for one bank and the hardware arbitrates them into extra
        pipeline bubbles - correct, but silently slower, so this is a
        performance warning rather than a rejection.
        """
        block = self.hw.chip.block_bytes
        banks = 8
        # Coverage is published whether or not anything is found: a check that
        # silently did not run looks identical to a clean kernel otherwise, and
        # "no conflicts" then means nothing.
        candidates = evaluated = conflicts = 0
        for op in kernel.api_calls():
            spec = lookup_api(op.name)
            if spec is None or op.pipe is not Pipe.V:
                continue
            srcs = self._dual_source_tensors(kernel, op, spec)
            if srcs is None:
                continue
            (name0, tensor0), (name1, tensor1) = srcs
            if name0 == name1:
                # The same tensor through both operand ports is served by the
                # broadcast path; there is no second bank to collide with.
                continue
            off0, off1 = tensor0.offset_value, tensor1.offset_value
            if off0 is not None and off0 == off1:
                # Two names for one region (an alias and its source) is the
                # broadcast case too: both ports read the same addresses.
                continue
            candidates += 1
            if off0 is None or off1 is None:
                continue  # unresolved layout; _check_extent_known owns that
            evaluated += 1
            delta = abs(off0 - off1) // block % banks
            if delta != 0:
                continue
            conflicts += 1
            bank = off0 // block % banks
            self.diags.add(
                Code.UB_BANK_CONFLICT,
                Severity.WARNING,
                f"Dual-operand vector instruction '{op.name}' reads "
                f"'{name0}' (offset 0x{off0:X}) and '{name1}' "
                f"(offset 0x{off1:X}) from identical UB Bank {bank}. "
                "Causes pipeline arbitration stall.",
                op.loc,
                hardware_domain="UB",
                remediation=(
                    f"Pad the allocation of '{name1}' by +32 bytes (1 DaVinci "
                    "block) or enforce bank-orthogonal base alignment."
                ),
                related=[(f"declaration of {name1!r}", tensor1.loc)],
                api=op.name,
                src0=name0,
                src1=name1,
                src0_offset=off0,
                src1_offset=off1,
                bank=bank,
                delta_blocks=abs(off0 - off1) // block,
            )

        self.ctx.publish(
            f"bank_conflict_coverage::{kernel.name}",
            {
                # Dual-operand vector reads that could collide at all.
                "candidates": candidates,
                # Of those, the ones whose two offsets both resolved, so the
                # check actually reached a verdict.
                "evaluated": evaluated,
                "conflicts": conflicts,
            },
        )

    @staticmethod
    def _dual_source_tensors(
        kernel: KernelIR, op: ApiCallOp, spec
    ) -> Optional[tuple[tuple[str, TensorDecl], tuple[str, TensorDecl]]]:
        """The ``(src0, src1)`` tensor pair of a dual-input op, when resolvable.

        Only instructions whose signature carries exactly one ``SRC0`` and one
        ``SRC1`` parameter (``Add``, ``Mul``, ``Sub``, ``Max``, ``Min``,
        ``Compare``, ...) qualify; both arguments must resolve to declared
        tensors living in UB.
        """
        picked: List[tuple[str, TensorDecl]] = []
        for role in (ArgRole.SRC0, ArgRole.SRC1):
            indices = [
                i
                for i, param in enumerate(spec.params)
                if param.role is role
            ]
            if len(indices) != 1 or indices[0] >= len(op.args):
                return None
            name = op.args[indices[0]].tensor
            if not name:
                return None
            tensor = kernel.tensors.get(name)
            if tensor is None or tensor.domain is not PhysicalDomain.UB:
                return None
            picked.append((name, tensor))
        return (picked[0], picked[1])

    # -- target instruction-set gating (AKA1011) -----------------------------

    def _check_target_features(self, kernel: KernelIR) -> None:
        """Reject intrinsics the active chip profile cannot issue (AKA1011).

        The MX Cube path (``mad_mx`` and its ``load_*_mx`` scale loaders) is
        DaVinci v3.  Analyzing such a kernel against a v2 profile like 910B
        used to model it as valid, reporting layout and pipeline findings for
        an instruction the part cannot execute at all - so the real defect,
        that the kernel is built for the wrong target, went unreported.
        """
        features = self.hw.chip.features
        for op in kernel.api_calls():
            spec = lookup_api(op.name)
            if spec is None or spec.requires_feature is None:
                continue
            if spec.requires_feature in features:
                continue
            self.diags.add(
                Code.UNSUPPORTED_INTRINSIC,
                Severity.FATAL,
                f"Intrinsic '{op.name}' is unsupported on target architecture "
                f"{self.hw.chip.display_name}: it requires the "
                f"'{spec.requires_feature}' feature, which this part does not "
                "implement",
                op.loc,
                remediation=(
                    f"Analyze this kernel against a chip whose profile carries "
                    f"'{spec.requires_feature}' (see --list-chips), or contract "
                    f"'{op.name}' to the instruction set the target does have - "
                    "on DaVinci v2 that is standard Mmad over FP16/BF16/INT8/INT4."
                ),
                api=op.name,
                requires_feature=spec.requires_feature,
                chip=self.hw.chip.name,
            )

    # -- 351x SIMD/SIMT DataCache budget (AKA1010) ---------------------------

    def _check_simt_datacache(self, kernel: KernelIR) -> None:
        """Enforce the 351x UB partition when SIMT execution is present.

        On 351x the Unified Buffer is strictly partitioned:
        ``DataCache = 256 KiB - StaticMem - DynamicMem - 8 KiB (compiler)``.
        Whenever the SIMT vector path is in use, the runtime requires
        ``DataCache >= 32 KiB``; more than 216 KiB of tensor allocation drops
        below that floor and the hardware corrupts memory at run time.
        """
        chip = self.hw.chip
        if not chip.enforces_datacache_partition:
            return
        source = self.ctx.unit.source
        if not any(
            re.search(rf"\b{re.escape(marker)}\b", source)
            for marker in _SIMT_MARKERS
        ):
            return

        allocated = self._ub_allocation_bytes(kernel)
        available = self.hw.simt_datacache_available(allocated)
        assert available is not None  # guarded by enforces_datacache_partition
        if available >= (chip.min_datacache_bytes or 0):
            return

        self.diags.add(
            Code.INSUFFICIENT_DATACACHE,
            Severity.FATAL,
            f"Insufficient UB DataCache headroom for {chip.name} SIMT "
            f"execution (available: {available} B, required: >= "
            f"{chip.min_datacache_bytes} B). Total tensor allocation exceeds "
            f"{chip.max_usable_ub_bytes // 1024} KB limit.",
            kernel.loc,
            hardware_domain="UB",
            remediation=(
                f"Free at least {chip.min_datacache_bytes - available} B of UB: "
                f"keep static + dynamic (InitBuffer) allocations at or below "
                f"{chip.max_usable_ub_bytes} B "
                f"({chip.max_usable_ub_bytes // 1024} KiB) so the SIMT DataCache "
                f"keeps its {chip.min_datacache_bytes // 1024} KiB hardware "
                "minimum, or shrink the tile until the budget closes."
            ),
            allocated_bytes=allocated,
            available_bytes=available,
            required_bytes=chip.min_datacache_bytes,
            ub_total_bytes=chip.ub_total_bytes,
            compiler_reserved_bytes=chip.compiler_reserved_bytes,
        )

    def _ub_allocation_bytes(self, kernel: KernelIR) -> int:
        """Total static + dynamic UB allocation, each storage counted once.

        ``TPipe::InitBuffer`` / ``LocalMemAllocator`` extents are counted per
        buffer; tensors that do not come from a buffer are counted by their
        own byte size, with identical ``(offset, size)`` ranges (two views of
        one raw region) deduplicated so deliberate reuse is not double-billed.
        """
        counted: Set[tuple[Optional[int], int]] = set()
        total = 0
        for tensor in kernel.tensors.values():
            if tensor.domain is not PhysicalDomain.UB or tensor.size_value is None:
                continue
            if tensor.source_buffer is not None:
                continue  # billed through its InitBuffer below
            key = (tensor.offset_value, tensor.size_value)
            if key in counted:
                continue
            counted.add(key)
            total += tensor.size_value
        return total + sum(kernel.buffer_sizes.values())

    # -- aggregate footprint ------------------------------------------------

    def _summarize(
        self, kernel: KernelIR, tensors: Sequence[TensorDecl]
    ) -> List[DomainUsage]:
        usage: List[DomainUsage] = []
        for domain in self.hw.tracked_sram_domains():
            group = [t for t in tensors if t.domain is domain]
            if not group:
                continue
            capacity = self.hw.capacity(domain) or 0
            static = [t for t in group if t.is_fully_static]
            allocated = sum(t.size_value or 0 for t in static)
            high_water = max((t.end_value or 0 for t in static), default=0)
            usage.append(
                DomainUsage(
                    domain=domain,
                    capacity_bytes=capacity,
                    allocated_bytes=allocated,
                    high_water_bytes=high_water,
                    tensor_count=len(group),
                    symbolic_count=len(group) - len(static),
                )
            )
        return usage

    def _check_fragmentation(
        self, kernel: KernelIR, usage: Sequence[DomainUsage]
    ) -> None:
        for entry in usage:
            if entry.high_water_bytes <= 0 or entry.gap_bytes <= 0:
                continue
            waste_ratio = entry.gap_bytes / entry.high_water_bytes
            if waste_ratio < 0.25 or entry.gap_bytes < self.hw.chip.block_bytes * 4:
                continue
            self.diags.add(
                Code.UB_FRAGMENTATION,
                Severity.INFO,
                f"{entry.domain.value} layout leaves {entry.gap_bytes} B "
                f"({waste_ratio:.0%}) unclaimed below the high-water mark of "
                f"{entry.high_water_bytes} B",
                kernel.loc,
                hardware_domain=entry.domain.value,
                remediation=(
                    "Compact the layout so tiles are contiguous; the reclaimed "
                    f"{entry.gap_bytes} B would allow a larger tile and fewer "
                    "loop iterations."
                ),
                gap_bytes=entry.gap_bytes,
                high_water_bytes=entry.high_water_bytes,
            )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _value_text(expr) -> str:
        folded = to_int(expr)
        if folded is not None:
            return f"{folded} (0x{folded:X})"
        return render(expr)

    def _alignment_remediation(
        self,
        tensor: TensorDecl,
        value: Optional[int],
        alignment: int,
        what: str,
    ) -> str:
        if value is None:
            return (
                f"Ensure the {what} is a multiple of {alignment} B for every value "
                "its free variables can take."
            )
        down = (value // alignment) * alignment
        up = -(-value // alignment) * alignment
        return (
            f"Round the {what} to {down} (0x{down:X}) or {up} (0x{up:X}); "
            f"{alignment} B is one DaVinci {tensor.domain.value} block."
        )
