"""Tests for the command line interface and its exit codes."""

from __future__ import annotations

import json

import pytest
from conftest import KERNEL_DIR

from ascend_analyzer.cli import build_parser, main

BROKEN = KERNEL_DIR / "pingpong_broken.cpp"
CLEAN = KERNEL_DIR / "pingpong_clean.cpp"
ISASI = KERNEL_DIR / "isasi_raw.cpp"


def run(args, capsys=None):
    code = main([str(a) for a in args])
    out = capsys.readouterr() if capsys else None
    return code, out


class TestExitCodes:
    def test_clean_kernel_exits_zero(self):
        assert main([str(CLEAN), "--quiet"]) == 0

    def test_fatal_findings_exit_one(self):
        assert main([str(BROKEN), "--quiet"]) == 1

    def test_warnings_alone_exit_zero(self):
        assert main([str(ISASI), "--quiet"]) == 0

    def test_warnings_as_errors_exits_two(self):
        assert main([str(ISASI), "--quiet", "--warnings-as-errors"]) == 2

    def test_clean_kernel_still_zero_under_warnings_as_errors(self):
        assert main([str(CLEAN), "--quiet", "-W"]) == 0

    def test_missing_file_exits_three(self, capsys):
        code = main([str(KERNEL_DIR / "does_not_exist.cpp"), "--quiet"])
        assert code == 3
        assert "no such file" in capsys.readouterr().err

    def test_no_arguments_exits_three(self, capsys):
        assert main([]) == 3
        assert "no source files" in capsys.readouterr().err

    def test_unknown_chip_exits_three(self, capsys):
        code = main([str(CLEAN), "--chip", "ascend999z", "--quiet"])
        assert code == 3
        assert "unknown chip" in capsys.readouterr().err

    def test_worst_exit_code_wins_across_several_files(self):
        assert main([str(CLEAN), str(BROKEN), "--quiet"]) == 1


class TestTerminalOutput:
    def test_quiet_prints_one_line_per_file(self, capsys):
        main([str(CLEAN), "--quiet"])
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        assert len(lines) == 1
        assert "accepted" in lines[0]

    def test_default_output_has_the_full_report(self, capsys):
        main([str(BROKEN), "--no-color", "--ascii"])
        out = capsys.readouterr().out
        assert "Ascend Static Kernel Analyzer" in out
        assert "Findings" in out
        assert "SRAM footprint" in out
        assert "Pipeline synchronisation" in out
        assert "REJECTED" in out

    def test_no_color_suppresses_escapes(self, capsys):
        main([str(BROKEN), "--no-color", "--ascii"])
        assert "\033[" not in capsys.readouterr().out

    def test_ascii_mode_avoids_non_ascii_glyphs(self, capsys):
        main([str(BROKEN), "--no-color", "--ascii"])
        out = capsys.readouterr().out
        out.encode("ascii")  # must not raise

    def test_output_file_receives_a_plain_report(self, tmp_path, capsys):
        target = tmp_path / "report.txt"
        main([str(BROKEN), "-o", str(target), "--no-color", "--ascii"])
        capsys.readouterr()
        text = target.read_text(encoding="utf-8")
        assert "REJECTED" in text
        assert "\033[" not in text


class TestJsonOutput:
    def test_json_to_stdout_is_valid(self, capsys):
        main([str(BROKEN), "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["verdict"] == "rejected"
        assert payload["summary"]["fatal"] > 0

    def test_json_to_a_file(self, tmp_path, capsys):
        target = tmp_path / "report.json"
        main([str(BROKEN), "--format", "json", "-o", str(target)])
        capsys.readouterr()
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["kernels"][0]["name"] == "vec_add_pingpong_broken"

    def test_side_channel_json_alongside_the_terminal_report(self, tmp_path, capsys):
        target = tmp_path / "side.json"
        main([str(BROKEN), "--json", str(target), "--no-color", "--ascii"])
        out = capsys.readouterr().out
        assert "Findings" in out
        assert json.loads(target.read_text(encoding="utf-8"))["verdict"] == "rejected"

    def test_several_files_are_wrapped_in_a_files_array(self, capsys):
        main([str(CLEAN), str(BROKEN), "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["files"]) == 2
        assert {f["verdict"] for f in payload["files"]} == {"accepted", "rejected"}


class TestHtmlOutput:
    def test_html_is_written_and_self_contained(self, tmp_path, capsys):
        target = tmp_path / "report.html"
        main([str(BROKEN), "--html", str(target), "--quiet"])
        capsys.readouterr()
        html = target.read_text(encoding="utf-8")
        assert html.startswith("<!doctype html>")
        assert "AKA1003" in html
        assert len(html) > 5000

    def test_html_path_is_announced(self, tmp_path, capsys):
        target = tmp_path / "report.html"
        main([str(BROKEN), "--html", str(target), "--no-color", "--ascii"])
        assert "HTML report written" in capsys.readouterr().out


class TestOptions:
    def test_capacity_override_changes_the_verdict(self, capsys):
        # pingpong_clean needs 2 KiB of UB; cap UB at 1 KiB and it overflows.
        assert main([str(CLEAN), "--quiet"]) == 0
        capsys.readouterr()
        assert main([str(CLEAN), "--ub-bytes", "1024", "--quiet"]) == 1

    def test_suppress_drops_a_code(self, capsys):
        main([str(BROKEN), "--format", "json", "--suppress", "AKA1005"])
        payload = json.loads(capsys.readouterr().out)
        assert "AKA1005" not in payload["summary"]["by_code"]

    def test_disable_skips_a_whole_checker(self, capsys):
        main([str(BROKEN), "--format", "json", "--disable", "memory"])
        payload = json.loads(capsys.readouterr().out)
        assert not any(c.startswith("AKA1") for c in payload["summary"]["by_code"])
        assert any(c.startswith("AKA2") for c in payload["summary"]["by_code"])

    def test_disable_deadlock_leaves_memory_findings(self, capsys):
        main([str(BROKEN), "--format", "json", "--disable", "deadlock"])
        payload = json.loads(capsys.readouterr().out)
        assert any(c.startswith("AKA1") for c in payload["summary"]["by_code"])
        assert not any(c.startswith("AKA2") for c in payload["summary"]["by_code"])

    @pytest.mark.parametrize("solver", ["z3", "interval", "auto"])
    def test_every_solver_backend_reaches_the_same_verdict(self, solver, capsys):
        main([str(BROKEN), "--format", "json", "--solver", solver])
        payload = json.loads(capsys.readouterr().out)
        assert payload["verdict"] == "rejected"
        assert payload["summary"]["by_code"]["AKA1003"] == 1

    def test_strict_mode_is_at_least_as_strict(self, capsys):
        main([str(CLEAN), "--format", "json"])
        relaxed = json.loads(capsys.readouterr().out)["summary"]["total"]
        main([str(CLEAN), "--format", "json", "--strict"])
        strict = json.loads(capsys.readouterr().out)["summary"]["total"]
        assert strict >= relaxed

    def test_directory_argument_analyzes_every_source(self, capsys):
        main([str(KERNEL_DIR), "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["files"]) >= 5

    def test_chip_alias_is_accepted(self, capsys):
        assert main([str(CLEAN), "--chip", "910b", "--quiet"]) == 0


class TestInformationalCommands:
    def test_list_chips(self, capsys):
        assert main(["--list-chips"]) == 0
        out = capsys.readouterr().out
        assert "ascend910b" in out
        assert "ascend910c" in out
        assert "provisional" in out
        assert "reserved event ids" in out

    def test_list_codes(self, capsys):
        assert main(["--list-codes"]) == 0
        out = capsys.readouterr().out
        for code in ("AKA1001", "AKA1003", "AKA2004", "AKA2005", "AKA3001"):
            assert code in out

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert "ascend-analyze" in capsys.readouterr().out

    def test_help_mentions_the_exit_codes(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert "exit codes" in out


def test_parser_rejects_a_non_positive_capacity():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["x.cpp", "--ub-bytes", "0"])
