"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ascend_analyzer import AnalyzerOptions, KernelAnalyzer  # noqa: E402
from ascend_analyzer.analyzer import AnalysisResult  # noqa: E402

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

#: Minimal kernel scaffold so inline test sources stay readable.
KERNEL_TEMPLATE = """#include "kernel_operator.h"
{preamble}
extern "C" __global__ __aicore__ void test_kernel({params})
{{
{body}
}}
"""


def make_kernel(body: str, preamble: str = "", params: str = "__gm__ half* gm") -> str:
    """Wrap a statement list in a minimal ``__global__ __aicore__`` kernel."""
    indented = "\n".join(
        f"    {line}" if line.strip() else "" for line in body.strip("\n").splitlines()
    )
    return KERNEL_TEMPLATE.format(
        preamble=f"\n{preamble}\n" if preamble.strip() else "",
        params=params,
        body=indented,
    )


def analyze(
    source: str,
    *,
    chip: str = "ascend910b",
    solver: str = "auto",
    strict: bool = False,
    suppress: Sequence[str] = (),
    all_functions: bool = False,
    path: str = "<test>.cpp",
) -> AnalysisResult:
    """Analyze an inline source string."""
    analyzer = KernelAnalyzer(
        AnalyzerOptions(
            chip=chip,
            solver=solver,
            strict=strict,
            suppress=tuple(suppress),
            all_functions=all_functions,
        )
    )
    return analyzer.analyze_source(path, source)


def analyze_body(body: str, preamble: str = "", **kwargs) -> AnalysisResult:
    """Analyze a statement list wrapped in the kernel scaffold."""
    return analyze(make_kernel(body, preamble), **kwargs)


def codes_of(result: AnalysisResult) -> list[str]:
    return sorted(set(result.codes()))


def find(result: AnalysisResult, code: str):
    """Every diagnostic with the given code."""
    return [d for d in result.diagnostics if d.code.value == code]


def only(result: AnalysisResult, code: str):
    """The single diagnostic with the given code; fails if not exactly one."""
    matches = find(result, code)
    assert len(matches) == 1, (
        f"expected exactly one {code}, got {len(matches)}; "
        f"all codes: {codes_of(result)}"
    )
    return matches[0]


@pytest.fixture(scope="session")
def kernel_dir() -> Path:
    return KERNEL_DIR


@pytest.fixture(scope="session")
def analyzer() -> KernelAnalyzer:
    return KernelAnalyzer(AnalyzerOptions())


def pytest_configure(config: "pytest.Config") -> None:
    config.addinivalue_line(
        "markers", "slow: tests that invoke the solver repeatedly"
    )
