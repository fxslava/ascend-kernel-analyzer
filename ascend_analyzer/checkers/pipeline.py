"""950PR SIMT dispatch and launch-bound checks."""

from __future__ import annotations

import re
from typing import Dict, Optional

from ..diagnostics import Code, Severity
from ..ir import KernelIR
from .base import Checker
from .source_utils import code_only, iter_calls, source_loc, constant_int, split_arguments

__all__ = ["PipelineChecker"]


def _threads(dim: str, names: Dict[str, int]) -> Optional[int]:
    match = re.match(r"\s*dim3\s*(?:\(|\{)(.*)(?:\)|\})\s*$", dim, re.DOTALL)
    if match is None:
        return None
    dimensions = [part.strip() for part in split_arguments(match.group(1))]
    if not 1 <= len(dimensions) <= 3:
        return 0
    values = [constant_int(part, names) for part in dimensions]
    if any(value is None for value in values):
        return None
    if any(value <= 0 for value in values):
        return 0
    values.extend([1] * (3 - len(values)))
    return values[0] * values[1] * values[2]


class PipelineChecker(Checker):
    """Validate the arch_3510 VF dispatch syntax and thread invariants."""

    name = "pipeline"

    def check(self, kernel: KernelIR) -> None:
        if self.hw.chip.name != "ascend950pr":
            return
        source = code_only(self.ctx.unit.source)
        names = {**self.ctx.unit.constants, **kernel.constants}
        self._active_lines = set(range(kernel.loc.line, (kernel.loc.end_line or kernel.loc.line) + 1))
        self._active_lines.update(op.loc.line for op in kernel.api_calls())
        self._folded_threads = {
            (op.loc.line, op.loc.column): op.launch_threads
            for op in kernel.api_calls() if op.name == "asc_vf_call"
        }
        self._check_bounds(source, self.ctx.unit.constants)
        self._check_dispatches(source, names)

    def _active(self, source: str, offset: int) -> bool:
        return source.count("\n", 0, offset) + 1 in self._active_lines

    def _check_bounds(self, source: str, names: Dict[str, int]) -> None:
        bounds_by_function: Dict[str, int] = {}
        for call in iter_calls(source, "__launch_bounds__"):
            function = re.search(r"\b([A-Za-z_]\w*)\s*\(", source[call.end :])
            expression = call.arguments[0] if len(call.arguments) == 1 else ""
            n = constant_int(expression, names)
            if n is None and expression:
                self.diags.add(Code.SYMBOLIC_OFFSET, Severity.INFO,
                    f"Cannot resolve __launch_bounds__({expression})", 
                    source_loc(self.ctx.unit.path, source, call.start),
                    remediation="Instantiate the template or provide the constant launch bound.")
                continue
            if n is None or not 1 <= n <= 2048 or n % 32:
                self.diags.add(
                    Code.INVALID_LAUNCH_BOUNDS,
                    Severity.FATAL,
                    f"__launch_bounds__ value {expression!r} must be "
                    "a constant multiple of 32 in the range 1..2048",
                    source_loc(self.ctx.unit.path, source, call.start),
                    hardware_domain="PIPE_V",
                    remediation="Choose a launch bound in 1..2048 that is a multiple of 32.",
                    value=expression,
                )
                continue
            if function is not None:
                bounds_by_function[function.group(1)] = n
        self._bounds_by_function = bounds_by_function

    def _check_dispatches(self, source: str, names: Dict[str, int]) -> None:
        for call in iter_calls(source, "asc_call_vf"):
            if not self._active(source, call.start):
                continue
            replacement = "asc_vf_call" + source[call.start + len(call.name):call.end]
            self.diags.add(
                Code.DEPRECATED_VF_DISPATCH,
                Severity.WARNING,
                "asc_call_vf is deprecated on Ascend 950PR",
                source_loc(self.ctx.unit.path, source, call.start),
                hardware_domain="PIPE_V",
                remediation="Replace with asc_vf_call<funcPtr>(dim3{...}, args...).",
                autofix=(replacement if call.template and call.arguments
                         else "asc_vf_call<funcPtr>(dim3{...}, args...)"),
            )

        dimension_variables: Dict[str, str] = {}
        for match in re.finditer(
            r"\bdim3\s+(?P<name>[A-Za-z_]\w*)\s*(?:=\s*(?:dim3\s*)?)?"
            r"(?:\{|\()(?P<dimensions>[^{}()]*)[})]",
            source,
        ):
            if not self._active(source, match.start()):
                continue
            dimension_variables[match.group("name")] = (
                "dim3{" + match.group("dimensions") + "}"
            )
        dimension_names = {
            match.group(1)
            for match in re.finditer(r"\bdim3\s+([A-Za-z_]\w*)\b", source)
            if self._active(source, match.start())
        }
        for call in iter_calls(source, "asc_vf_call"):
            if not self._active(source, call.start):
                continue
            first_argument = call.arguments[0].strip() if call.arguments else ""
            if (
                not call.template
                or len(split_arguments(call.template or "")) != 1
                or not re.match(r"&?\s*(?:::)?[A-Za-z_]", call.template or "")
                or not call.arguments
                or not (
                    re.match(r"\s*dim3\b", first_argument)
                    or first_argument in dimension_names
                )
            ):
                self.diags.add(
                    Code.INVALID_LAUNCH_BOUNDS,
                    Severity.FATAL,
                    "asc_vf_call requires a function template argument and a dim3 "
                    "thread-count as its first argument",
                    source_loc(self.ctx.unit.path, source, call.start),
                    hardware_domain="PIPE_V",
                    remediation=(
                        "Use asc_vf_call<funcPtr>(dim3{block_x, block_y, block_z}, args...)."
                    ),
                    signature=call.template,
                    first_argument=first_argument or None,
                )
                continue
            function = call.template.split(",", 1)[0].strip().lstrip("&").rsplit("::", 1)[-1]
            dimensions = dimension_variables.get(first_argument, first_argument)
            threads = _threads(dimensions, names)
            loc = source_loc(self.ctx.unit.path, source, call.start)
            if (loc.line, loc.column) in self._folded_threads:
                threads = self._folded_threads[(loc.line, loc.column)]
            if threads is None:
                self.diags.add(Code.SYMBOLIC_OFFSET, Severity.INFO,
                    f"Cannot prove the SIMT thread count for {dimensions}",
                    source_loc(self.ctx.unit.path, source, call.start),
                    remediation="Provide positive constant dim3 dimensions to verify the launch bound.")
            if threads is not None and not 1 <= threads <= 2048:
                self.diags.add(
                    Code.INVALID_LAUNCH_BOUNDS,
                    Severity.FATAL,
                    f"SIMT block has {threads} threads; dimensions must be positive and the product at most 2048",
                    source_loc(self.ctx.unit.path, source, call.start),
                    hardware_domain="PIPE_V",
                    remediation="Reduce the dim3 product to at most 2048 threads.",
                    threads=threads,
                    maximum_threads=2048,
                )
            bound = self._bounds_by_function.get(function)
            if bound is not None and threads is not None and bound < threads:
                self.diags.add(
                    Code.INVALID_LAUNCH_BOUNDS,
                    Severity.FATAL,
                    f"__launch_bounds__({bound}) is smaller than the "
                    f"{threads}-thread block dispatched to {function}",
                    source_loc(self.ctx.unit.path, source, call.start),
                    hardware_domain="PIPE_V",
                    remediation="Increase N to at least the product of the dim3 dimensions.",
                    launch_bound=bound,
                    threads=threads,
                    function=function,
                )
            if threads is None or threads <= 256:
                continue
            budget = 64 if threads <= 512 else (32 if threads <= 1024 else 16)
            self.diags.add(
                Code.REGISTER_SPILL,
                Severity.WARNING,
                f"{threads} SIMT threads reduce the per-thread register budget "
                f"to approximately {budget} registers",
                source_loc(self.ctx.unit.path, source, call.start),
                hardware_domain="REG",
                remediation=(
                    "Reduce the block size or shorten live ranges to lower "
                    "register-spill risk."
                ),
                threads=threads,
                register_budget=budget,
            )
