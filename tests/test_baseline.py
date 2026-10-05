"""Tests for baseline suppression and the CI regression gate."""

from __future__ import annotations

import json

import pytest
from conftest import KERNEL_DIR

from ascend_analyzer.baseline import (
    Baseline,
    BaselineError,
    build_baseline_document,
    fingerprint,
    normalise_path,
)
from ascend_analyzer.cli import main
from ascend_analyzer.diagnostics import Code, Diagnostic, Severity, SourceLoc

BROKEN = KERNEL_DIR / "pingpong_broken.cpp"
CLEAN = KERNEL_DIR / "pingpong_clean.cpp"
MISMATCH = KERNEL_DIR / "domain_mismatch.cpp"


def diag(code=Code.DOMAIN_MISMATCH, message="boom", file="a/b.cpp", line=7):
    return Diagnostic(
        code=code,
        severity=Severity.FATAL,
        message=message,
        loc=SourceLoc(file=file, line=line),
    )


class TestPathNormalisation:
    def test_backslashes_become_forward_slashes(self):
        assert normalise_path(r"csrc\op\kernel.cpp") == "csrc/op/kernel.cpp"

    @pytest.mark.parametrize(
        "spelling",
        ["/mnt/d/Projects/x/k.cpp", "/d/Projects/x/k.cpp", r"D:\Projects\x\k.cpp"],
    )
    def test_wsl_gitbash_and_windows_spellings_agree(self, spelling):
        """The same tree reached three ways has to fingerprint the same."""
        assert normalise_path(spelling).lower() == "d:/projects/x/k.cpp"

    def test_path_under_root_becomes_relative(self, tmp_path):
        got = normalise_path("D:/Projects/x/csrc/k.cpp", root="D:/Projects/x")
        assert got == "csrc/k.cpp"

    def test_path_outside_root_is_left_absolute(self):
        got = normalise_path("D:/Other/k.cpp", root="D:/Projects/x")
        assert got == "D:/Other/k.cpp"


class TestFingerprint:
    def test_line_is_not_part_of_the_identity(self):
        """A finding that only moved down the file is not a new finding."""
        assert fingerprint(diag(line=7)) == fingerprint(diag(line=901))

    def test_message_is_part_of_the_identity(self):
        assert fingerprint(diag(message="a")) != fingerprint(diag(message="b"))

    def test_code_is_part_of_the_identity(self):
        assert fingerprint(diag(code=Code.DOMAIN_MISMATCH)) != fingerprint(
            diag(code=Code.UB_BANK_CONFLICT)
        )

    def test_whitespace_in_the_message_is_normalised(self):
        assert fingerprint(diag(message="a  b")) == fingerprint(diag(message="a b"))


class TestPartition:
    def test_recorded_finding_is_suppressed(self):
        one = diag()
        baseline = Baseline(fingerprints={fingerprint(one): 1})
        regressions, covered = baseline.partition([one])
        assert regressions == []
        assert covered == [one]

    def test_unrecorded_finding_is_a_regression(self):
        baseline = Baseline(fingerprints={fingerprint(diag(message="old")): 1})
        new = diag(message="new")
        regressions, covered = baseline.partition([new])
        assert regressions == [new]
        assert covered == []

    def test_second_occurrence_beyond_the_budget_is_a_regression(self):
        """One recorded instance must not hide a second one."""
        one = diag()
        baseline = Baseline(fingerprints={fingerprint(one): 1})
        regressions, covered = baseline.partition([one, one])
        assert len(covered) == 1
        assert len(regressions) == 1

    def test_empty_baseline_suppresses_nothing(self):
        one = diag()
        regressions, covered = Baseline().partition([one])
        assert regressions == [one]
        assert covered == []


class TestDocument:
    def test_round_trips_through_json(self, tmp_path):
        findings = [diag(message="x"), diag(message="y"), diag(message="y")]
        document = build_baseline_document(findings, chip="ascend910b")
        path = tmp_path / "base.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        baseline = Baseline.load(path)
        regressions, covered = baseline.partition(findings)
        assert regressions == []
        assert len(covered) == 3

    def test_counts_duplicate_occurrences(self):
        document = build_baseline_document([diag(message="y"), diag(message="y")])
        assert document["counts"]["occurrences"] == 2
        assert document["counts"]["findings"] == 1

    def test_findings_are_ordered_deterministically(self):
        a = build_baseline_document([diag(message="b"), diag(message="a")])
        b = build_baseline_document([diag(message="a"), diag(message="b")])
        assert a == b

    def test_rejects_non_object(self, tmp_path):
        path = tmp_path / "base.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(BaselineError):
            Baseline.load(path)

    def test_rejects_future_schema(self, tmp_path):
        path = tmp_path / "base.json"
        path.write_text(json.dumps({"schema_version": 99, "findings": []}), "utf-8")
        with pytest.raises(BaselineError):
            Baseline.load(path)

    def test_rejects_missing_findings(self, tmp_path):
        path = tmp_path / "base.json"
        path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        with pytest.raises(BaselineError):
            Baseline.load(path)

    def test_rejects_unreadable_file(self, tmp_path):
        with pytest.raises(BaselineError):
            Baseline.load(tmp_path / "absent.json")


class TestCliGate:
    def test_fully_baselined_run_exits_zero(self, tmp_path):
        """The whole point: a known backlog must not keep the gate red."""
        base = tmp_path / "base.json"
        assert main([str(MISMATCH), "--quiet", "--write-baseline", str(base)]) == 0
        assert main([str(MISMATCH), "--quiet", "--baseline", str(base)]) == 0

    def test_regression_against_a_baseline_exits_one(self, tmp_path):
        base = tmp_path / "base.json"
        # A baseline recorded from a clean kernel covers none of the broken
        # kernel's findings, so every one of them is a regression.
        assert main([str(CLEAN), "--quiet", "--write-baseline", str(base)]) == 0
        assert main([str(BROKEN), "--quiet", "--baseline", str(base)]) == 1

    def test_partially_baselined_run_reports_only_the_surplus(self, tmp_path):
        base = tmp_path / "base.json"
        main([str(MISMATCH), "--quiet", "--write-baseline", str(base)])
        document = json.loads(base.read_text(encoding="utf-8"))
        # Drop one *fatal* entry: the gate is only red for a fatal surplus
        # unless --warnings-as-errors is given, and the newest findings on
        # this fixture (AKA2010 hazard warnings) sort last.
        # ``Severity`` serialises as "FATAL"; compare case-insensitively so
        # the test does not depend on that spelling.
        fatals = [
            i for i, f in enumerate(document["findings"])
            if str(f.get("severity", "")).upper() == "FATAL"
        ]
        assert fatals, "fixture is expected to carry fatal findings"
        del document["findings"][fatals[-1]]
        base.write_text(json.dumps(document), encoding="utf-8")

        code = main([str(MISMATCH), "--quiet", "--baseline", str(base)])
        assert code == 1

    def test_write_baseline_reports_what_it_recorded(self, tmp_path, capsys):
        base = tmp_path / "base.json"
        main([str(MISMATCH), "--write-baseline", str(base)])
        out = capsys.readouterr().out
        assert "baseline written to" in out
        assert base.exists()

    def test_bad_baseline_is_a_usage_error(self, tmp_path, capsys):
        base = tmp_path / "base.json"
        base.write_text("not json", encoding="utf-8")
        assert main([str(MISMATCH), "--quiet", "--baseline", str(base)]) == 3
        assert "baseline" in capsys.readouterr().err

    def test_baseline_root_makes_the_record_relative(self, tmp_path):
        base = tmp_path / "base.json"
        main([
            str(MISMATCH), "--write-baseline", str(base),
            "--baseline-root", str(KERNEL_DIR),
        ])
        document = json.loads(base.read_text(encoding="utf-8"))
        assert document["findings"]
        for entry in document["findings"]:
            assert entry["file"] == "domain_mismatch.cpp"


def test_wsl_root_is_normalised_with_source_path():
    assert normalise_path("/mnt/c/project/csrc/kernel.cpp", root="/mnt/c/project") == "csrc/kernel.cpp"
