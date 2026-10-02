"""Command line interface.

    ascend-analyze kernel.cpp --chip ascend910b
    ascend-analyze kernel.cpp --format json -o report.json
    ascend-analyze kernel.cpp --html report.html

Exit codes: ``0`` clean, ``1`` fatal findings, ``2`` warnings under
``--warnings-as-errors``, ``3`` a usage or I/O error.  That makes the tool
usable directly as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .analyzer import AVAILABLE_CHECKERS, TOOL_VERSION, AnalyzerOptions, KernelAnalyzer
from .baseline import Baseline, BaselineError, build_baseline_document
from .diagnostics import CODE_TITLES
from .hardware import CHIP_PROFILES, PhysicalDomain
from .report.html_report import build_html_report
from .report.json_report import build_json_report, dump_json_report
from .report.terminal import TerminalReporter

__all__ = ["main", "build_parser"]

_EXIT_USAGE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ascend-analyze",
        description=(
            "Static analyzer for Ascend C kernels written in the static tensor "
            "programming model (raw LocalTensor addressing, no TPipe/TQue). "
            "Detects SRAM overflow, alignment and aliasing violations, memory "
            "domain mismatches, and pipeline deadlocks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes:\n"
            "  0  no fatal findings\n"
            "  1  at least one FATAL finding\n"
            "  2  warnings present and --warnings-as-errors given\n"
            "  3  usage or I/O error\n"
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        metavar="SOURCE",
        help="kernel source files or directories to analyze",
    )

    target = parser.add_argument_group("target hardware")
    target.add_argument(
        "--chip",
        default="ascend910b",
        help="chip profile (default: %(default)s); see --list-chips",
    )
    target.add_argument(
        "--chip-profile",
        type=Path,
        metavar="FILE",
        help="JSON file overriding the built-in chip profile",
    )
    target.add_argument(
        "--ub-bytes", type=_positive_int, metavar="N",
        help="override the Unified Buffer capacity, in bytes",
    )
    target.add_argument(
        "--l1-bytes", type=_positive_int, metavar="N",
        help="override the L1 Buffer capacity, in bytes",
    )
    target.add_argument(
        "--list-chips", action="store_true", help="list built-in chip profiles and exit"
    )

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument(
        "--solver",
        choices=("auto", "z3", "interval"),
        default="auto",
        help="memory decision procedure (default: %(default)s)",
    )
    analysis.add_argument(
        "--solver-timeout", type=_positive_int, default=5000, metavar="MS",
        help="per-query solver timeout in milliseconds (default: %(default)s)",
    )
    analysis.add_argument(
        "--strict",
        action="store_true",
        help="promote unresolved offsets and unknown domains to FATAL",
    )
    analysis.add_argument(
        "--all-functions",
        action="store_true",
        help="analyze every function, not only __global__/__aicore__ entries",
    )
    analysis.add_argument(
        "--suppress", action="append", default=[], metavar="CODE",
        help="drop a diagnostic code entirely (repeatable), e.g. --suppress AKA3001",
    )
    analysis.add_argument(
        "--disable", action="append", default=[], metavar="CHECKER",
        choices=list(AVAILABLE_CHECKERS),
        help=f"skip a checker (repeatable): {', '.join(AVAILABLE_CHECKERS)}",
    )
    analysis.add_argument(
        "--ignore-inline-pragmas",
        action="store_true",
        help="do not honour '@ascend-ignore' annotations in the source",
    )
    analysis.add_argument(
        "--max-ops", type=_positive_int, default=20000, metavar="N",
        help="cap on tracked operations per kernel (default: %(default)s)",
    )
    analysis.add_argument(
        "--infer-tiling-roles",
        action="store_true",
        help=(
            "infer values for tiling-struct fields no manifest supplies from "
            "the role each plays at its call sites: an InitBuffer extent gets "
            "the architecture-minimal dimension (16 Cube / 64 Vector), a "
            "DataCopy stride gets the 32-byte block, and a field compared "
            "with GetBlockIdx() is left symbolic. Unblocks the TPipe layout, "
            "and with it AKA3006, on queue-managed kernels"
        ),
    )
    analysis.add_argument(
        "--tiling-data", metavar="PATH[:KEY]",
        help=(
            "bind tiling-struct fields from a JSON manifest, so layouts that "
            "depend on host tiling resolve for that configuration; append "
            "':KEY' to pick a named config (default: the first one)"
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--format",
        choices=("terminal", "json"),
        default="terminal",
        help="primary report format (default: %(default)s)",
    )
    output.add_argument(
        "-o", "--output", type=Path, metavar="FILE",
        help="write the primary report to FILE instead of stdout",
    )
    output.add_argument(
        "--html", type=Path, metavar="FILE",
        help="additionally write a standalone HTML report with the memory map",
    )
    output.add_argument(
        "--json", dest="json_path", type=Path, metavar="FILE",
        help="additionally write the JSON report to FILE",
    )
    output.add_argument(
        "--no-memory-map", action="store_true",
        help="omit the SRAM footprint chart from the terminal report",
    )
    output.add_argument(
        "--no-sync-summary", action="store_true",
        help="omit the pipeline synchronisation summary from the terminal report",
    )
    output.add_argument(
        "--no-perf-summary", action="store_true",
        help="omit the analytical performance / utilization section from the "
        "terminal report",
    )
    output.add_argument(
        "--max-findings", type=int, default=0, metavar="N",
        help="show at most N findings in the terminal report (0 = all)",
    )
    output.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    output.add_argument(
        "--ascii", action="store_true",
        help="use pure-ASCII glyphs in charts (default: auto-detect)",
    )
    output.add_argument(
        "--quiet", action="store_true", help="print only the verdict summary line"
    )
    output.add_argument(
        "--list-codes", action="store_true",
        help="list every diagnostic code and exit",
    )

    gate = parser.add_argument_group("CI gate")
    gate.add_argument(
        "--baseline", type=Path, metavar="FILE",
        help=(
            "suppress findings recorded in FILE and report only regressions; "
            "the verdict and the exit code are computed on what is left"
        ),
    )
    gate.add_argument(
        "--write-baseline", type=Path, metavar="FILE",
        help="record this run's findings as a baseline in FILE and exit 0",
    )
    gate.add_argument(
        "--baseline-root", type=Path, metavar="DIR",
        help=(
            "record and match baseline paths relative to DIR, so a baseline "
            "is portable between checkouts and between path spellings"
        ),
    )

    parser.add_argument(
        "-W", "--warnings-as-errors", action="store_true",
        help="exit non-zero when warnings are present",
    )
    parser.add_argument(
        "--version", action="version", version=f"ascend-analyze {TOOL_VERSION}"
    )
    return parser


def _load_tiling_values(spec: Optional[str]) -> dict:
    """Read ``PATH[:KEY]`` into a flat ``{field: value}`` binding.

    Two manifest shapes are accepted: a flat ``{field: value}`` mapping, and the
    harvested per-kernel form ``{"kernels": {<path>: {"configs": {KEY: {...}}}}}``.
    In the second case every kernel that carries the requested config
    contributes its fields; a field two kernels disagree on is dropped rather
    than silently resolved one way.
    """
    if not spec:
        return {}
    path_text, _, key = spec.rpartition(":")
    # A bare Windows drive letter is not a config key.
    if not path_text or len(key) == 1:
        path_text, key = spec, ""
    path = Path(path_text)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read --tiling-data {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"--tiling-data {path} is not valid JSON: {exc}") from exc

    if not isinstance(doc, dict):
        raise ValueError(f"--tiling-data {path} must hold a JSON object")

    kernels = doc.get("kernels")
    if not isinstance(kernels, dict):
        flat = {k: v for k, v in doc.items() if isinstance(v, (int, float))}
        if not flat:
            raise ValueError(f"--tiling-data {path} has no field values")
        return {k: int(v) for k, v in flat.items() if float(v).is_integer()}

    merged: dict = {}
    conflicting: set = set()
    for entry in kernels.values():
        configs = (entry or {}).get("configs")
        if not isinstance(configs, dict) or not configs:
            continue
        chosen = configs.get(key) if key else next(iter(configs.values()))
        if not isinstance(chosen, dict):
            continue
        for name, value in chosen.items():
            if not isinstance(value, (int, float)) or not float(value).is_integer():
                continue  # a float clamp is not a layout input
            value = int(value)
            if name in merged and merged[name] != value:
                conflicting.add(name)
            merged[name] = value
    for name in conflicting:
        merged.pop(name, None)
    if key and not merged:
        raise ValueError(
            f"--tiling-data {path}: no kernel defines a config named {key!r}"
        )
    return merged


def _positive_int(raw: str) -> int:
    try:
        value = int(raw, 0)
    except ValueError as exc:  # pragma: no cover - argparse formats the message
        raise argparse.ArgumentTypeError(f"{raw!r} is not an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_chips:
        _print_chips()
        return 0
    if args.list_codes:
        _print_codes()
        return 0
    if not args.paths:
        parser.print_usage(sys.stderr)
        print("error: no source files given (try --help)", file=sys.stderr)
        return _EXIT_USAGE

    overrides = {}
    if args.ub_bytes:
        overrides[PhysicalDomain.UB] = args.ub_bytes
    if args.l1_bytes:
        overrides[PhysicalDomain.L1] = args.l1_bytes

    try:
        analyzer = KernelAnalyzer(
            AnalyzerOptions(
                chip=args.chip,
                chip_profile=args.chip_profile,
                solver=args.solver,
                solver_timeout_ms=args.solver_timeout,
                strict=args.strict,
                suppress=tuple(args.suppress),
                disable=tuple(args.disable),
                all_functions=args.all_functions,
                capacity_overrides=overrides,
                max_ops=args.max_ops,
                warnings_as_errors=args.warnings_as_errors,
                honour_inline_ignores=not args.ignore_inline_pragmas,
                tiling_values=_load_tiling_values(args.tiling_data),
                infer_tiling_roles=args.infer_tiling_roles,
            )
        )
    except (KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    missing = [p for p in args.paths if not p.exists()]
    if missing:
        for path in missing:
            print(f"error: no such file or directory: {path}", file=sys.stderr)
        return _EXIT_USAGE

    try:
        results = analyzer.analyze_paths(args.paths)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    if not results:
        print("error: no analyzable sources found", file=sys.stderr)
        return _EXIT_USAGE

    if args.write_baseline is not None:
        document = build_baseline_document(
            (d for r in results for d in r.diagnostics),
            root=args.baseline_root,
            chip=args.chip,
            tool_version=TOOL_VERSION,
        )
        try:
            args.write_baseline.write_text(
                json.dumps(document, indent=2, sort_keys=False) + chr(10),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return _EXIT_USAGE
        counts = document["counts"]
        print(
            f"baseline written to {args.write_baseline}: "
            f"{counts['findings']} distinct findings, "
            f"{counts['occurrences']} occurrences"
        )
        return 0

    baselined_total = 0
    if args.baseline is not None:
        try:
            baseline = Baseline.load(args.baseline)
        except BaselineError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return _EXIT_USAGE
        if args.baseline_root is not None:
            baseline.root = args.baseline_root
        for result in results:
            kept, covered = baseline.partition(result.diagnostics)
            result.diagnostics = kept
            result.suppressed.extend(covered)
            baselined_total += len(covered)

    exit_code = _emit(args, results)
    if args.baseline is not None and not args.quiet and args.format != "json":
        print(
            f"  {baselined_total} finding(s) suppressed by baseline "
            f"{args.baseline}"
        )
    return exit_code


def _emit(args: argparse.Namespace, results: List) -> int:
    worst = 0

    # JSON is a single document per file; terminal output concatenates.
    if args.format == "json":
        payloads = [
            build_json_report(
                r.unit, r.hardware, r.diagnostics, r.artifacts,
                solver_name=r.solver_name, tool_version=TOOL_VERSION,
            )
            for r in results
        ]
        document = payloads[0] if len(payloads) == 1 else {
            "schema_version": payloads[0]["schema_version"],
            "files": payloads,
        }
        text = dump_json_report(document)
        _write_text(args.output, text, sys.stdout)
    elif not args.quiet:
        reporter = TerminalReporter(
            stream=sys.stdout,
            color=False if args.no_color else None,
            ascii_only=args.ascii,
            show_memory_map=not args.no_memory_map,
            show_sync=not args.no_sync_summary,
            max_diagnostics=max(0, args.max_findings),
        )
        if args.output is not None:
            with args.output.open("w", encoding="utf-8") as handle:
                plain = TerminalReporter(
                    stream=handle,
                    color=False,
                    ascii_only=args.ascii,
                    show_memory_map=not args.no_memory_map,
                    show_sync=not args.no_sync_summary,
                    max_diagnostics=max(0, args.max_findings),
                )
                for result in results:
                    plain.report(
                        result.unit, result.hardware, result.diagnostics,
                        result.artifacts, result.solver_name,
                    )
        for result in results:
            reporter.report(
                result.unit, result.hardware, result.diagnostics,
                result.artifacts, result.solver_name,
            )

    if args.html is not None:
        documents = [
            build_html_report(
                r.unit, r.hardware, r.diagnostics, r.artifacts,
                solver_name=r.solver_name, tool_version=TOOL_VERSION,
            )
            for r in results
        ]
        args.html.write_text(
            documents[0] if len(documents) == 1 else "\n<hr>\n".join(documents),
            encoding="utf-8",
        )
        if not args.quiet and args.format != "json":
            print(f"  HTML report written to {args.html}")

    if args.json_path is not None:
        payloads = [
            build_json_report(
                r.unit, r.hardware, r.diagnostics, r.artifacts,
                solver_name=r.solver_name, tool_version=TOOL_VERSION,
            )
            for r in results
        ]
        document = payloads[0] if len(payloads) == 1 else {
            "schema_version": payloads[0]["schema_version"],
            "files": payloads,
        }
        args.json_path.write_text(dump_json_report(document), encoding="utf-8")
        if not args.quiet and args.format != "json":
            print(f"  JSON report written to {args.json_path}")

    for result in results:
        if args.quiet:
            print(
                f"{result.unit.path}: {result.verdict} "
                f"({result.fatal_count} fatal, {result.warning_count} warning, "
                f"{result.info_count} info)"
            )
        worst = max(worst, result.exit_code(args.warnings_as_errors))
    return worst


def _write_text(path: Optional[Path], text: str, fallback) -> None:
    if path is None:
        fallback.write(text + "\n")
    else:
        path.write_text(text, encoding="utf-8")


def _print_chips() -> None:
    print("Built-in chip profiles:\n")
    for spec in CHIP_PROFILES.values():
        flag = "  (provisional)" if spec.provisional else ""
        print(f"  {spec.name:<12} {spec.display_name}{flag}")
        if spec.aliases:
            print(f"               aliases: {', '.join(spec.aliases)}")
        for domain in spec.sram_domains():
            print(
                f"               {domain.domain.value:<4} "
                f"{domain.capacity_bytes:>8} B ({domain.capacity_kib:>6.1f} KiB)  "
                f"base align {domain.base_alignment} B"
            )
        print(
            f"               reserved event ids: "
            f"{sorted(spec.reserved_event_ids)}  max: {spec.max_event_id}"
        )
        if spec.notes:
            print(f"               {spec.notes}")
        print()


def _print_codes() -> None:
    print("Diagnostic codes:\n")
    groups = {
        "1xxx  memory layout": "AKA1",
        "2xxx  synchronisation and pipeline": "AKA2",
        "3xxx  performance and analyzability": "AKA3",
        "4xxx  performance model / overlap profiling": "AKA4",
        "9xxx  analyzer infrastructure": "AKA9",
    }
    for heading, prefix in groups.items():
        print(f"  {heading}")
        for code, title in sorted(CODE_TITLES.items()):
            if code.startswith(prefix):
                print(f"    {code}  {title}")
        print()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
