"""Vector-ISA numerical-parity checks: split bleed (AKA1015) and repeat
encoding (AKA1020).

Two families of bug live here, both silent: the kernel compiles, runs, and
returns numbers that are wrong.

**Split-window bleed (AKA1015).**  A reduction over a flattened ``[rows,
stride]`` buffer is merged by repeatedly de-interleaving the window into its
even and odd elements and adding the halves - ``SplitColumns(merge1, merge2,
cur, len)`` on parts without a native ``DeInterleave`` is a pair of Gathers
over a byte-offset ramp.  The pairing is only row-safe while the window
covers at least two elements *per row*: at ``len == 2 * rows`` each row keeps
its own last pair; one level deeper, element ``2i`` of the flat window is row
``i`` itself, and the merge sums **rows pairwise across the batch**.  The
layout inference anchors the row count on the tree's terminal
self-accumulating ``Add(acc, acc, half, rows)`` and tracks in-place halvings,
so a window is judged against the buffer's *live* extent, not its allocation.

**Repeat encoding (AKA1020).**  The repeat-stride form of a vector
instruction - ``Cast(dst, src, mode, countPerRepeat, repeats, {dstBlk, srcBlk,
dstRep, srcRep})`` - encodes the layout twice: the repeat counter must fit
the one-byte instruction field, and for a contiguous conversion (block
strides 1, 1) the repeat strides must advance by exactly the blocks one
repeat touches: ``count * sizeof(elem) / 32`` on each side.  A stride that
respects the source but not the narrower destination (fp32 -> bf16 moves 8
src blocks per 64-element repeat but only 4 dst blocks) reads the right
elements and writes the wrong ones - parity corruption at full speed.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from ..backend.parity import (
    MergeTreeTracker,
    SplitWindow,
    replay_halving_chain,
)
from ..diagnostics import Code, Severity
from ..ir import ApiCallOp, KernelIR, LoopInfo
from .base import Checker

__all__ = ["VectorIsaChecker"]


#: ``{dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride}``.
_STRIDE_TUPLE_RE = re.compile(
    r"^\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}$"
)
#: ``while (len > bound)`` / ``while (len >= bound)`` over a halved length.
_HALVING_LOOP_RE = re.compile(
    r"while\s*\(\s*(?P<var>[A-Za-z_]\w*)\s*(?P<op>>=|>|<=|<)\s*(?P<bound>[^)]+?)\s*\)"
)
_INIT_RE = r"\b{var}\s*=\s*(?P<value>[^;+]+);"
_UPDATE_RE = r"\b{var}\s*(?:/=\s*(?P<div>\d+)|=\s*[^;]*?/\s*(?P<div2>\d+)\b)"


class VectorIsaChecker(Checker):
    """Guards the vector ISA encodings that silently corrupt numerics."""

    name = "vector_isa"

    def check(self, kernel: KernelIR) -> None:
        self._check_split_windows(kernel)
        self._check_repeat_encoding(kernel)

    # -- AKA1015: split-window bleed ----------------------------------------

    def _check_split_windows(self, kernel: KernelIR) -> None:
        windows = self._collect_split_windows(kernel)
        rows = self._infer_row_count(kernel, windows)
        tracker = MergeTreeTracker(rows)
        halvings = self._collect_halvings(kernel, windows)
        events: List[Tuple[int, object]] = sorted(
            [(w.index, w) for w in windows] + [(i, h) for i, h in halvings],
            key=lambda item: item[0],
        )
        checked = 0
        for _, event in events:
            if isinstance(event, SplitWindow):
                violation = tracker.check_window(event)
                if violation is not None:
                    checked += 1
                    self._report_split(kernel, violation, rows)
            else:
                tracker.apply_halving(event[0], event[1])

        chains = self._replay_merge_chains(kernel, rows, windows)
        self.ctx.publish(
            f"parity_coverage::{kernel.name}",
            {
                "split_windows": len(windows),
                "rows_anchor": rows,
                "halving_steps": len(halvings),
                "merge_chains_replayed": chains,
                "op_level_violations": checked,
            },
        )

    def _collect_split_windows(self, kernel: KernelIR) -> List[SplitWindow]:
        """Every de-interleave event: direct intrinsics and Gather pairs."""
        calls = kernel.api_calls()
        windows: List[SplitWindow] = []
        used: set[int] = set()
        for op in calls:
            if op.name in ("DeInterleave", "DeInterleave2") and len(op.args) >= 4:
                count = op.args[3].value
                src = op.args[2].tensor
                # The intrinsic's count is the full interleaved window; the
                # Gather idiom moves one half per call.  Normalise to halves.
                if count is None or not src or count % 2 != 0:
                    continue
                windows.append(
                    SplitWindow(
                        tensor=self._root_tensor(kernel, src),
                        half_count=count // 2,
                        index=op.index,
                        line=op.loc.line,
                        kind="deinterleave",
                    )
                )
            elif op.name == "Gather" and op.index not in used:
                twin = self._gather_twin(kernel, op)
                if twin is not None:
                    used.add(twin.index)
                    windows.append(
                        SplitWindow(
                            tensor=self._root_tensor(kernel, op.args[1].tensor),
                            half_count=twin.count,
                            index=op.index,
                            line=op.loc.line,
                            kind="gather_pair",
                        )
                    )
        return windows

    def _gather_twin(self, kernel: KernelIR, first: ApiCallOp):
        """The second Gather of a de-interleave pair, with its window count.

        The idiom (``SplitColumns`` on parts with no native de-interleave)
        gathers the even elements at ``offsets``, bumps the same offset
        tensor by one source element (``Adds(offsets, offsets, sizeof(T),
        n)``), then gathers the odd ones: two Gathers on one source with one
        element-sized offset bump between them.
        """
        if len(first.args) < 5 or not first.args[1].tensor:
            return None
        count = first.args[4].value
        offsets = first.args[2].tensor
        src_elem = self._elem_size(kernel, first.args[1].tensor)
        if count is None or not offsets or src_elem is None:
            return None
        bumped = False
        for op in kernel.ops:
            if op.index <= first.index:
                continue
            if not isinstance(op, ApiCallOp):
                continue
            if op.name == "Adds" and len(op.args) >= 4:
                if (
                    op.args[0].tensor == offsets
                    and op.args[1].tensor == offsets
                    and op.args[3].value == count
                    and op.args[2].value == src_elem
                ):
                    bumped = True
                    continue
            if op.name == "Gather" and bumped:
                if (
                    len(op.args) >= 5
                    and op.args[1].tensor == first.args[1].tensor
                    and op.args[4].value == count
                ):
                    return _Twin(op.index, count)
                return None
            if op.index - first.index > 40:
                break
        return None

    def _collect_halvings(
        self, kernel: KernelIR, windows: Sequence[SplitWindow]
    ) -> List[Tuple[int, Tuple[str, int]]]:
        """In-place merge steps: ``Add(cur, m1, m2, len/2)`` over a split src.

        The destinations of the split (``m1``/``m2``) are added back *into
        the split's own source*, halving the buffer's live extent.
        """
        split_sources = {w.tensor for w in windows}
        split_dests = {
            self._root_tensor(kernel, arg.tensor)
            for op in kernel.api_calls()
            if op.name in ("DeInterleave", "DeInterleave2", "Gather")
            for arg in op.args[:(1 if op.name == "Gather" else 2)]
            if arg.tensor
        }
        steps: List[Tuple[int, Tuple[str, int]]] = []
        for op in kernel.api_calls():
            if op.name != "Add" or len(op.args) < 4:
                continue
            dst, src0, src1 = (a.tensor for a in op.args[:3])
            count = op.args[3].value
            root = self._root_tensor(kernel, dst)
            if not dst or count is None or root not in split_sources:
                continue
            roots = {self._root_tensor(kernel, n) for n in (src0, src1) if n}
            if not roots & split_dests:
                continue
            steps.append((op.index, (root, count)))
        return steps

    def _infer_row_count(
        self, kernel: KernelIR, windows: Sequence[SplitWindow]
    ) -> Optional[int]:
        """Row count from the tree's terminal self-accumulating add.

        ``Add(acc, acc, half, n)`` folds the final two columns per row, so
        its count *is* the number of independent rows the tree reduces.  A
        candidate is accepted when it divides some split window evenly - a
        healthy tree - or, when none divides (the ragged case the check is
        about to report), when it is simply smaller than the largest window.
        """
        split_dsts = set()
        # The halves the splits produce; the anchor add consumes them.
        for op in kernel.api_calls():
            if op.name in ("DeInterleave", "DeInterleave2") and len(op.args) >= 3:
                for arg in op.args[:2]:
                    if arg.tensor:
                        split_dsts.add(arg.tensor)
        for op in kernel.api_calls():
            if op.name == "Gather" and len(op.args) >= 1 and op.args[0].tensor:
                split_dsts.add(op.args[0].tensor)
        lengths = [2 * w.half_count for w in windows]
        for op in kernel.api_calls():
            if op.name != "Add" or len(op.args) < 4:
                continue
            dst, src0, src1 = (a.tensor for a in op.args[:3])
            count = op.args[3].value
            if not dst or dst != src0 or not src1 or count is None or count < 2:
                continue
            if dst not in split_dsts and src1 not in split_dsts:
                continue
            if any(L % count == 0 for L in lengths) or (
                lengths and count < max(lengths)
            ):
                return count
        return None

    def _replay_merge_chains(
        self,
        kernel: KernelIR,
        rows: Optional[int],
        windows: Sequence[SplitWindow],
    ) -> int:
        """Replay symbolic halving loops to reach their tail split (tier 2).

        A ``while (mergeLen > 2 * chunkRows)`` tree appears in the trace as
        one body instance carrying the *entry* length; the deeper levels and
        the tail split after the loop see lengths only the loop arithmetic
        knows.  With initializer, bound and divisor all folded constants the
        chain is replayed exactly, and every level - body and tail - is held
        to the same ``2 * rows`` floor.
        """
        if rows is None or rows < 2:
            return 0
        replayed = 0
        for loop in kernel.loops.values():
            chain = self._chain_bounds(kernel, loop)
            if chain is None:
                continue
            initial, bound, divisor, strict = chain
            body_levels, tail = replay_halving_chain(
                initial, bound, divisor=divisor, strict=strict
            )
            in_loop = [w for w in windows if loop.contains(w.index)]
            if not in_loop:
                continue
            replayed += 1
            for level in body_levels:
                self._judge_level(kernel, loop, level, rows, kind="loop-body")
            self._judge_level(kernel, loop, tail, rows, kind="tail")
        return replayed

    def _chain_bounds(
        self, kernel: KernelIR, loop: LoopInfo
    ) -> Optional[Tuple[int, int, int, bool]]:
        """``(initial, bound, divisor, strict)`` of a halving-length loop."""
        header = loop.header or ""
        match = _HALVING_LOOP_RE.search(header)
        if match is None or match.group("op") not in (">", ">="):
            return None
        var = match.group("var")
        bound = _fold_arithmetic(match.group("bound"), kernel.constants)
        unit = self.ctx.unit
        body_lines = [
            unit.line_text(line)
            for line in range(loop.loc.line + 1, min(loop.loc.line + 60, len(unit.lines)) + 1)
        ]
        divisor = None
        for text in body_lines:
            update = re.search(_UPDATE_RE.format(var=re.escape(var)), text)
            if update is not None:
                divisor = int(update.group("div") or update.group("div2"))
                break
        if bound is None or divisor is None or divisor < 2:
            return None
        initial = None
        for line in range(loop.loc.line - 1, max(loop.loc.line - 30, 0), -1):
            text = unit.line_text(line)
            init = re.search(_INIT_RE.format(var=re.escape(var)), text)
            if init is not None:
                initial = _fold_arithmetic(init.group("value"), kernel.constants)
                if initial is not None:
                    break
        if initial is None:
            return None
        return initial, bound, divisor, match.group("op") == ">"

    def _judge_level(
        self, kernel: KernelIR, loop: LoopInfo, length: int, rows: int, *, kind: str
    ) -> None:
        if length % rows != 0:
            self._report_replayed(
                kernel, loop, length, rows,
                "cross_row" if length < 2 * rows else "ragged", kind,
            )
        elif length // rows < 2:
            self._report_replayed(kernel, loop, length, rows, "cross_row", kind)

    def _report_split(self, kernel, violation, rows: int) -> None:
        window = violation.window
        if violation.kind == "cross_row":
            message = (
                f"{window.kind} at line {window.line} splits '{window.tensor}' "
                f"with a window of {violation.length} elements, but the buffer "
                f"is a flattened [{rows}, stride] layout whose pairing floor "
                f"is {2 * rows}: the unit-strand de-interleave now pairs the "
                f"last element of one row with the first of the next, and the "
                f"merge tree above sums independent batch rows together"
            )
            remediation = (
                f"Stop the merge tree while the window still covers at least "
                f"two elements per row (len >= {2 * rows}); finish with "
                f"Add(even, even, odd, {rows}) so each row collapses on itself."
            )
        else:
            message = (
                f"{window.kind} at line {window.line} splits '{window.tensor}' "
                f"with a window of {violation.length} elements, not a whole "
                f"number of the {rows} rows it reduces: the trailing partial "
                f"row bleeds its columns into the next row's results"
            )
            remediation = (
                f"Keep every level's window a multiple of the row count "
                f"({rows}); derive it from the live extent, not a stale constant."
            )
        self.diags.add(
            Code.STRIDED_REDUCTION,
            Severity.FATAL,
            message,
            _line_loc(self.ctx.unit.path, window.line),
            hardware_domain="UB",
            remediation=remediation,
            tensor=window.tensor,
            window_length=violation.length,
            rows=rows,
            violation=violation.kind,
        )

    def _report_replayed(
        self, kernel, loop, length, rows, kind, level_kind
    ) -> None:
        if kind == "cross_row":
            message = (
                f"the halving merge loop at line {loop.loc.line} drives its "
                f"{level_kind} split down to a window of {length} elements, "
                f"below the {2 * rows}-element pairing floor of the "
                f"[{rows}, stride] layout: the {level_kind} pairs row 2i with "
                f"row 2i+1 and the reduction merges independent batch rows"
            )
            remediation = (
                f"Raise the loop bound to keep every window at or above "
                f"2 * rows = {2 * rows} elements, e.g. "
                f"while (len > 2 * rows)."
            )
        else:
            message = (
                f"the halving merge loop at line {loop.loc.line} runs its "
                f"{level_kind} at {length} elements, not a whole number of the "
                f"{rows} rows being reduced"
            )
            remediation = (
                f"Derive each level's window from the live extent so it stays "
                f"a multiple of the {rows}-row layout."
            )
        self.diags.add(
            Code.STRIDED_REDUCTION,
            Severity.FATAL,
            message,
            _line_loc(self.ctx.unit.path, loop.loc.line),
            hardware_domain="UB",
            remediation=remediation,
            window_length=length,
            rows=rows,
            violation=kind,
            loop_header=loop.header,
            level=level_kind,
        )

    # -- AKA1020: repeat encoding -------------------------------------------

    def _check_repeat_encoding(self, kernel: KernelIR) -> None:
        limit = self.hw.chip.max_repeat_count
        block = self.hw.chip.block_bytes
        for op in kernel.api_calls():
            if len(op.args) < 6:
                continue
            stride_text = op.args[5].text.strip()
            match = _STRIDE_TUPLE_RE.match(stride_text)
            repeats = op.args[4].value
            if repeats is not None and repeats > limit:
                self._report_repeat_overflow(op, repeats, limit)
            if match is None:
                continue
            dst_blk, src_blk, dst_rep, src_rep = (int(g) for g in match.groups())
            per_repeat = op.args[3].value
            if per_repeat is None:
                continue  # mask-driven form; no element count to reason with
            if repeats is not None and repeats < 2:
                continue  # a lone repeat never advances the strides
            dst_elem = self._elem_size(kernel, op.args[0].tensor)
            src_elem = self._elem_size(kernel, op.args[1].tensor)
            if dst_elem is None or src_elem is None:
                continue
            if dst_blk != 1 or src_blk != 1:
                continue  # deliberately strided layout; intent unknowable
            problems: List[str] = []
            dst_bytes = per_repeat * dst_elem
            src_bytes = per_repeat * src_elem
            for side, count_bytes, given in (
                ("destination", dst_bytes, dst_rep),
                ("source", src_bytes, src_rep),
            ):
                if count_bytes % block != 0:
                    problems.append(
                        f"one repeat covers {count_bytes} B of {side}, not a "
                        f"whole {block}-byte block"
                    )
                elif given != count_bytes // block:
                    problems.append(
                        f"{side} repeat stride {given} blocks advances "
                        f"{given * block} B but one repeat covers "
                        f"{count_bytes // block} blocks ({count_bytes} B)"
                    )
            if problems:
                self._report_stride_mismatch(op, per_repeat, dst_elem, src_elem,
                                             (dst_blk, src_blk, dst_rep, src_rep),
                                             problems)

    def _report_repeat_overflow(self, op, repeats: int, limit: int) -> None:
        self.diags.add(
            Code.REPEAT_ENCODING,
            Severity.FATAL,
            f"'{op.name}' requests {repeats} repeats, but the repeat counter "
            f"is a {limit}-maximum one-byte instruction field: the hardware "
            "executes repeats mod 256 and the conversion covers a fraction "
            "of the buffer with no error",
            op.loc,
            hardware_domain=op.pipe.value,
            remediation=(
                f"Split the call so every repeat count stays <= {limit}, or "
                "raise the elements-per-repeat (count) and let the strides "
                "carry the layout."
            ),
            api=op.name,
            repeats=repeats,
            max_repeats=limit,
        )

    def _report_stride_mismatch(
        self, op, per_repeat, dst_elem, src_elem, strides, problems
    ) -> None:
        self.diags.add(
            Code.REPEAT_ENCODING,
            Severity.FATAL,
            f"'{op.name}' repeat strides {{{', '.join(str(s) for s in strides)}}} "
            f"disagree with its operand geometry: {per_repeat} elements x "
            f"{src_elem} B src / {dst_elem} B dst; "
            + "; ".join(problems),
            op.loc,
            hardware_domain=op.pipe.value,
            remediation=(
                "For a contiguous conversion use repeat strides of exactly "
                "the blocks one repeat touches: "
                f"{{1, 1, {per_repeat * dst_elem // 32}, {per_repeat * src_elem // 32}}} "
                f"for {per_repeat}-element repeats of {src_elem} B -> {dst_elem} B."
            ),
            api=op.name,
            strides=list(strides),
            per_repeat=per_repeat,
            dst_elem_size=dst_elem,
            src_elem_size=src_elem,
        )

    # -- helpers -------------------------------------------------------------

    def _elem_size(self, kernel: KernelIR, name: Optional[str]) -> Optional[int]:
        if not name:
            return None
        decl = kernel.tensors.get(name)
        return decl.elem_size if decl is not None else None

    def _root_tensor(self, kernel: KernelIR, name: Optional[str]) -> str:
        """Follow ``view_source``/inline-binding chains to the base buffer.

        A split inside an inlined helper reads the ``callee::param`` view of
        the caller's buffer, while the merge steps that halve that buffer
        name it directly; both spellings must land on one identity for the
        layout tracker to follow the tree.
        """
        current = name or ""
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            decl = kernel.tensors.get(current)
            source = decl.view_source if decl is not None else None
            if not source:
                break
            current = source
        return current or name or "?"


class _Twin(NamedTuple):
    """The second Gather of a de-interleave pair."""

    index: int
    count: int


def _line_loc(path: str, line: int):
    from ..diagnostics import SourceLoc

    return SourceLoc(file=path, line=line)


def _fold_arithmetic(text: str, constants: Dict[str, int]) -> Optional[int]:
    """Fold ``2 * chunkRows``-style integer arithmetic over known constants."""
    import ast as _ast

    text = text.strip()
    text = re.sub(r"static_cast<[^>]+>\(", "(", text)
    for name, value in sorted(constants.items(), key=lambda kv: -len(kv[0])):
        if re.match(r"^[A-Za-z_]\w*$", name):
            text = re.sub(rf"\b{re.escape(name)}\b", str(value), text)
    if not re.match(r"^[\d\s()+\-*/%]+$", text):
        return None
    try:
        tree = _ast.parse(text, mode="eval")
    except SyntaxError:
        return None

    def eval_node(node) -> Optional[int]:
        if isinstance(node, _ast.Expression):
            return eval_node(node.body)
        if isinstance(node, _ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, _ast.UnaryOp) and isinstance(
            node.op, (_ast.USub, _ast.UAdd)
        ):
            value = eval_node(node.operand)
            if value is None:
                return None
            return -value if isinstance(node.op, _ast.USub) else value
        if isinstance(node, _ast.BinOp) and isinstance(
            node.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.FloorDiv, _ast.Mod)
        ):
            left, right = eval_node(node.left), eval_node(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, _ast.Add):
                return left + right
            if isinstance(node.op, _ast.Sub):
                return left - right
            if isinstance(node.op, _ast.Mult):
                return left * right
            if isinstance(node.op, _ast.FloorDiv):
                return left // right if right else None
            return left % right if right else None
        return None

    return eval_node(tree)
