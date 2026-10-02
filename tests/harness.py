"""Runnable test harness for the Ascend static kernel analyzer.

    python tests/harness.py                 # run every fixture, print a summary
    python tests/harness.py --verbose       # also print the full report per fixture
    python tests/harness.py --only broken   # run fixtures whose name matches

The headline case is ``pingpong_broken.cpp``: a typical ping-pong vector
double-buffering kernel carrying an intentional pipeline deadlock (the
loop-carried ``V_MTE2`` and ``MTE3_V`` handshakes are never primed, so
iteration 0 waits for ever) and intentional unaligned memory offsets (a
250-element tile is not a whole number of 32-byte DaVinci blocks, and the
derived pong offsets land off the block boundary).  ``pingpong_clean.cpp`` is
the corrected counterpart and must come back silent.

Each fixture declares its own expectations in its header comment::

    @ascend-expect: AKA1002 AKA1003 AKA1005 AKA2003 AKA2005 AKA3001
    @ascend-expect-fatal: 13

so the fixtures stay self-documenting and this harness needs no separate
expectations table.  The same declarations drive the pytest suite in
``test_fixtures.py``.

Exits 0 when every fixture matches its declaration, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Set

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ascend_analyzer import AnalyzerOptions, KernelAnalyzer  # noqa: E402
from ascend_analyzer.analyzer import AnalysisResult  # noqa: E402
from ascend_analyzer.report.terminal import TerminalReporter  # noqa: E402

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

_EXPECT_RE = re.compile(r"@ascend-expect\s*:\s*(?P<codes>[^\n*]*)")
_EXPECT_FATAL_RE = re.compile(r"@ascend-expect-fatal\s*:\s*(?P<count>\d+)")


# ---------------------------------------------------------------------------
# Expectations
# ---------------------------------------------------------------------------


@dataclass
class Expectation:
    """What a fixture declares about itself."""

    path: Path
    codes: Set[str] = field(default_factory=set)
    fatal_count: Optional[int] = None

    @classmethod
    def from_source(cls, path: Path) -> "Expectation":
        text = path.read_text(encoding="utf-8")
        codes: Set[str] = set()
        for match in _EXPECT_RE.finditer(text):
            # Skip the longer '@ascend-expect-fatal' spelling, which the
            # generic pattern also matches.
            if text[match.start() : match.start() + 20].startswith(
                "@ascend-expect-fatal"
            ):
                continue
            codes.update(
                token.strip().upper()
                for token in match.group("codes").replace(",", " ").split()
                if token.strip()
            )
        fatal_match = _EXPECT_FATAL_RE.search(text)
        return cls(
            path=path,
            codes=codes,
            fatal_count=int(fatal_match.group("count")) if fatal_match else None,
        )


@dataclass
class Outcome:
    """The result of checking one fixture against its declaration."""

    expectation: Expectation
    result: AnalysisResult
    missing_codes: List[str] = field(default_factory=list)
    fatal_mismatch: Optional[str] = None

    @property
    def name(self) -> str:
        return self.expectation.path.name

    @property
    def passed(self) -> bool:
        return not self.missing_codes and self.fatal_mismatch is None


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def check_fixture(path: Path, analyzer: KernelAnalyzer) -> Outcome:
    """Analyze one fixture and compare the findings with its declaration."""
    expectation = Expectation.from_source(path)
    result = analyzer.analyze_file(path)
    produced = set(result.codes())

    missing = sorted(expectation.codes - produced)
    mismatch: Optional[str] = None
    if (
        expectation.fatal_count is not None
        and result.fatal_count != expectation.fatal_count
    ):
        mismatch = (
            f"expected {expectation.fatal_count} FATAL finding(s), "
            f"got {result.fatal_count}"
        )
    return Outcome(
        expectation=expectation,
        result=result,
        missing_codes=missing,
        fatal_mismatch=mismatch,
    )


def run(
    paths: Sequence[Path],
    *,
    chip: str = "ascend910b",
    solver: str = "auto",
    verbose: bool = False,
    ascii_only: bool = True,
    stream=sys.stdout,
) -> int:
    """Run every fixture and print a summary table. Returns an exit code."""
    analyzer = KernelAnalyzer(AnalyzerOptions(chip=chip, solver=solver))

    stream.write("\n")
    stream.write("Ascend Static Kernel Analyzer - test harness\n")
    stream.write("=" * 78 + "\n")
    stream.write(f"  target  {analyzer.hardware.chip.display_name}\n")
    stream.write(f"  solver  {analyzer.solver_name}\n")
    stream.write(f"  fixtures {len(paths)} in {KERNEL_DIR}\n\n")

    outcomes: List[Outcome] = []
    for path in paths:
        outcome = check_fixture(path, analyzer)
        outcomes.append(outcome)

        if verbose:
            stream.write("\n" + "=" * 78 + "\n")
            reporter = TerminalReporter(
                stream=stream, color=False, ascii_only=ascii_only
            )
            reporter.report(
                outcome.result.unit,
                outcome.result.hardware,
                outcome.result.diagnostics,
                outcome.result.artifacts,
                outcome.result.solver_name,
            )

    _write_table(outcomes, stream)
    failures = [o for o in outcomes if not o.passed]

    stream.write("\n")
    if failures:
        stream.write(f"FAILED  {len(failures)} of {len(outcomes)} fixtures\n")
        for outcome in failures:
            stream.write(f"  {outcome.name}\n")
            if outcome.missing_codes:
                stream.write(
                    f"    expected but not reported: "
                    f"{', '.join(outcome.missing_codes)}\n"
                )
            if outcome.fatal_mismatch:
                stream.write(f"    {outcome.fatal_mismatch}\n")
            stream.write(
                f"    actually reported: "
                f"{', '.join(sorted(set(outcome.result.codes()))) or '(nothing)'}\n"
            )
        return 1

    stream.write(f"PASSED  all {len(outcomes)} fixtures match their declarations\n")
    return 0


def _write_table(outcomes: Sequence[Outcome], stream) -> None:
    stream.write("\n")
    stream.write(
        f"  {'fixture':<24}{'verdict':<24}{'check':<10}"
        f"{'fatal':>6}{'warn':>6}{'info':>6}   codes\n"
    )
    stream.write("  " + "-" * 110 + "\n")
    for outcome in outcomes:
        result = outcome.result
        status = "ok" if outcome.passed else "MISMATCH"
        codes = ", ".join(sorted(set(result.codes()))) or "-"
        stream.write(
            f"  {outcome.name:<24}{result.verdict:<24}{status:<10}"
            f"{result.fatal_count:>6}{result.warning_count:>6}"
            f"{result.info_count:>6}   {codes}\n"
        )


def discover(only: Optional[str] = None) -> List[Path]:
    paths = sorted(KERNEL_DIR.glob("*.cpp"))
    if only:
        needle = only.lower()
        paths = [p for p in paths if needle in p.name.lower()]
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the analyzer against every kernel fixture and check "
        "the findings against the expectations each fixture declares.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="print the full analyzer report for each fixture",
    )
    parser.add_argument(
        "--only", metavar="SUBSTRING",
        help="run only fixtures whose filename contains SUBSTRING",
    )
    parser.add_argument("--chip", default="ascend910b", help="chip profile to target")
    parser.add_argument(
        "--solver", default="auto", choices=("auto", "z3", "interval"),
        help="memory decision procedure",
    )
    parser.add_argument(
        "--unicode", action="store_true",
        help="use Unicode box drawing in verbose reports",
    )
    args = parser.parse_args(argv)

    paths = discover(args.only)
    if not paths:
        print(f"error: no fixtures found in {KERNEL_DIR}", file=sys.stderr)
        return 1
    return run(
        paths,
        chip=args.chip,
        solver=args.solver,
        verbose=args.verbose,
        ascii_only=not args.unicode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
