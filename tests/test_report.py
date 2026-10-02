"""Tests for the memory map and the three report renderers."""

from __future__ import annotations

import io
import json

import pytest
from conftest import analyze_body

from ascend_analyzer.hardware import HardwareModel
from ascend_analyzer.report.html_report import build_html_report
from ascend_analyzer.report.json_report import (
    SCHEMA_VERSION,
    build_json_report,
    dump_json_report,
)
from ascend_analyzer.report.memory_map import (
    DomainMap,
    Glyphs,
    build_memory_maps,
    render_ascii_map,
)
from ascend_analyzer.report.terminal import TerminalReporter


def ub_tensor(name: str, offset, count, pos: str = "VECIN") -> str:
    return (
        f"AscendC::LocalTensor<half> {name};\n"
        f"{name}.SetTPosition(AscendC::TPosition::{pos});\n"
        f"{name}.SetAddr({offset});\n"
        f"{name}.SetSize({count});\n"
    )


# Three contiguous tiles.  Note this layout's source pair (0, 512) is a real
# UB bank collision (16 blocks = 0 mod 8), so the Add below now warns with
# AKA3006; it stays for memory-map geometry tests, which only read the layout.
LAYOUT = (
    ub_tensor("a", 0, 256)
    + ub_tensor("b", 512, 256)
    + ub_tensor("c", 1024, 256)
    + "AscendC::Add(c, a, b, 256);\n"
)

# Bank-orthogonal counterpart for verdict tests: |0 - 544| / 32 = 17 blocks
# = 1 mod 8, so the same dual-operand Add reads distinct UB banks and the
# kernel stays clean end to end.
BANK_CLEAN_LAYOUT = (
    ub_tensor("a", 0, 256)
    + ub_tensor("b", 544, 256)
    + ub_tensor("c", 1056, 256)
    + "AscendC::Add(c, a, b, 256);\n"
)

OVERLAPPING = (
    ub_tensor("a", 0, 256)
    + ub_tensor("b", 256, 256)
    + "AscendC::Add(b, a, a, 256);\n"
)


# ---------------------------------------------------------------------------
# Memory map model
# ---------------------------------------------------------------------------


class TestMemoryMapModel:
    def test_segments_are_sorted_by_offset(self):
        result = analyze_body(LAYOUT)
        maps = build_memory_maps(result.unit.kernels[0], result.hardware)
        assert len(maps) == 1
        starts = [s.start for s in maps[0].segments]
        assert starts == sorted(starts) == [0, 512, 1024]

    def test_high_water_and_allocated_bytes(self):
        result = analyze_body(LAYOUT)
        ub = build_memory_maps(result.unit.kernels[0], result.hardware)[0]
        assert ub.high_water == 1024 + 512
        assert ub.allocated == 3 * 512
        assert ub.free_bytes == ub.capacity_bytes - ub.high_water

    def test_gaps_are_reported(self):
        result = analyze_body(LAYOUT)
        ub = build_memory_maps(result.unit.kernels[0], result.hardware)[0]
        # a ends at 512 and b starts at 512, but b ends at 1024 and c starts
        # at 1024, so this layout is contiguous: no gaps.
        assert ub.gaps() == []

    def test_gap_between_tiles_is_found(self):
        body = ub_tensor("a", 0, 256) + ub_tensor("b", 1024, 256)
        result = analyze_body(body)
        ub = build_memory_maps(result.unit.kernels[0], result.hardware)[0]
        assert ub.gaps() == [(512, 1024)]

    def test_collisions_are_annotated_both_ways(self):
        result = analyze_body(OVERLAPPING)
        ub = build_memory_maps(result.unit.kernels[0], result.hardware)[0]
        by_name = {s.name: s for s in ub.segments}
        assert by_name["a"].collides_with == ("b",)
        assert by_name["b"].collides_with == ("a",)
        assert ub.collision_count == 2

    def test_adjacent_segments_are_not_marked_as_colliding(self):
        result = analyze_body(LAYOUT)
        ub = build_memory_maps(result.unit.kernels[0], result.hardware)[0]
        assert ub.collision_count == 0

    def test_unresolved_tensors_are_listed_separately(self):
        result = analyze_body(ub_tensor("t", "hostValue", 256))
        ub = build_memory_maps(result.unit.kernels[0], result.hardware)[0]
        assert ub.segments == []
        assert any("t" in note for note in ub.unresolved)

    def test_map_is_json_serialisable(self):
        result = analyze_body(OVERLAPPING)
        ub = build_memory_maps(result.unit.kernels[0], result.hardware)[0]
        json.dumps(ub.to_json())


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------


class TestAsciiMap:
    def render(self, body: str, **kwargs):
        result = analyze_body(body)
        maps = build_memory_maps(result.unit.kernels[0], result.hardware)
        return render_ascii_map(maps[0], glyphs=Glyphs.ascii(), **kwargs)

    def test_all_lines_are_the_same_width(self):
        lines = self.render(LAYOUT, width=40)
        widths = {len(line) for line in lines}
        assert len(widths) == 1, f"ragged box: widths {sorted(widths)}"

    def test_box_is_closed(self):
        lines = self.render(LAYOUT, width=40)
        assert lines[0].startswith("+") and lines[0].endswith("+")
        assert lines[-1].startswith("+") and lines[-1].endswith("+")
        for line in lines[1:-1]:
            assert line.startswith("|") and line.endswith("|")

    def test_every_tensor_appears(self):
        text = "\n".join(self.render(LAYOUT, width=40))
        for name in ("a", "b", "c"):
            assert name in text
        assert "capacity" in text

    def test_collision_glyph_appears_only_when_ranges_truly_overlap(self):
        overlapping = "\n".join(self.render(OVERLAPPING, width=40))
        assert "X" in overlapping
        adjacent = "\n".join(self.render(LAYOUT, width=40))
        # 'X' must not appear as a bar glyph for a correct contiguous layout.
        bar_region = "\n".join(
            line[14:54] for line in adjacent.splitlines() if line.startswith("|")
        )
        assert "X" not in bar_region

    def test_small_overlap_is_still_visible(self):
        # A 12-byte overlap inside a 2 KiB window would round away under
        # nearest-integer column mapping.
        body = (
            ub_tensor("a", 512, 250)     # [512, 1012)
            + ub_tensor("b", 1000, 250)  # [1000, 1500)
            + "AscendC::Add(b, a, a, 250);\n"
        )
        text = "\n".join(self.render(body, width=40))
        assert "X" in text

    def test_unicode_and_ascii_render_the_same_shape(self):
        result = analyze_body(LAYOUT)
        maps = build_memory_maps(result.unit.kernels[0], result.hardware)
        ascii_lines = render_ascii_map(maps[0], width=40, glyphs=Glyphs.ascii())
        unicode_lines = render_ascii_map(maps[0], width=40, glyphs=Glyphs.unicode())
        assert len(ascii_lines) == len(unicode_lines)
        assert {len(line) for line in unicode_lines} == {len(line) for line in unicode_lines}

    def test_empty_domain_does_not_divide_by_zero(self):
        empty = DomainMap(
            domain=HardwareModel.for_chip("ascend910b").tracked_sram_domains()[0],
            capacity_bytes=192 * 1024,
        )
        lines = render_ascii_map(empty, width=20, glyphs=Glyphs.ascii())
        assert lines  # renders a header and a closed box, nothing more


class TestGlyphFallback:
    def test_ascii_encoding_forces_the_ascii_glyph_set(self):
        assert Glyphs.best_for("ascii").fill == "#"
        assert Glyphs.best_for("cp1252").fill == "#"

    def test_utf8_gets_the_unicode_glyph_set(self):
        assert Glyphs.best_for("utf-8").fill == "█"

    def test_unknown_encoding_falls_back_safely(self):
        assert Glyphs.best_for("not-a-real-codec").fill == "#"

    def test_force_ascii_overrides_a_capable_stream(self):
        assert Glyphs.best_for("utf-8", force_ascii=True).fill == "#"

    def test_none_encoding_falls_back(self):
        assert Glyphs.best_for(None).fill == "#"


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------


class TestTerminalReport:
    def render(self, body: str, **kwargs) -> str:
        result = analyze_body(body)
        stream = io.StringIO()
        reporter = TerminalReporter(
            stream=stream, color=False, ascii_only=True, **kwargs
        )
        reporter.report(
            result.unit, result.hardware, result.diagnostics,
            result.artifacts, result.solver_name,
        )
        return stream.getvalue()

    def test_clean_kernel_says_accepted(self):
        text = self.render(BANK_CLEAN_LAYOUT)
        assert "ACCEPTED" in text
        assert "REJECTED" not in text
        assert "No findings" in text

    def test_broken_kernel_says_rejected(self):
        text = self.render(OVERLAPPING)
        assert "REJECTED" in text
        assert "AKA1003" in text

    def test_report_includes_a_source_frame_with_a_caret(self):
        text = self.render(OVERLAPPING)
        assert "^" in text

    def test_caret_never_overruns_its_source_line(self):
        text = self.render(OVERLAPPING)
        lines = text.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("^"):
                continue
            # The preceding line is the source frame; the underline must not
            # be longer than the code it points at.
            source = lines[index - 1]
            assert len(line.rstrip()) <= len(source.rstrip()) + 1

    def test_remediation_is_present_for_every_finding(self):
        text = self.render(OVERLAPPING)
        assert "fix:" in text

    def test_no_color_means_no_escape_sequences(self):
        assert "\033[" not in self.render(OVERLAPPING)

    def test_memory_map_can_be_omitted(self):
        assert "SRAM footprint" not in self.render(LAYOUT, show_memory_map=False)
        assert "SRAM footprint" in self.render(LAYOUT)

    def test_sync_summary_can_be_omitted(self):
        body = (
            "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);\n"
            "AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);\n"
        )
        assert "Pipeline synchronisation" in self.render(body)
        assert "Pipeline synchronisation" not in self.render(
            body, show_sync=False
        )

    def test_max_diagnostics_truncates_and_says_so(self):
        text = self.render(
            ub_tensor("a", 1000, 250) + ub_tensor("b", 1100, 250)
            + "AscendC::Add(b, a, a, 250);\n",
            max_diagnostics=1,
        )
        assert "suppressed by --max-findings" in text

    def test_provisional_profile_is_disclosed(self):
        result = analyze_body(LAYOUT, chip="ascend910c")
        stream = io.StringIO()
        TerminalReporter(stream=stream, color=False, ascii_only=True).report(
            result.unit, result.hardware, result.diagnostics,
            result.artifacts, result.solver_name,
        )
        assert "provisional" in stream.getvalue()


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


class TestJsonReport:
    def build(self, body: str) -> dict:
        result = analyze_body(body)
        return build_json_report(
            result.unit, result.hardware, result.diagnostics,
            result.artifacts, solver_name=result.solver_name,
        )

    def test_is_serialisable_and_round_trips(self):
        report = self.build(OVERLAPPING)
        assert json.loads(dump_json_report(report)) == json.loads(
            dump_json_report(report)
        )

    def test_carries_the_schema_version(self):
        assert self.build(LAYOUT)["schema_version"] == SCHEMA_VERSION

    @pytest.mark.parametrize(
        "body, verdict",
        [(BANK_CLEAN_LAYOUT, "accepted"), (OVERLAPPING, "rejected")],
    )
    def test_verdict(self, body, verdict):
        assert self.build(body)["verdict"] == verdict

    def test_summary_counts_by_code(self):
        report = self.build(OVERLAPPING)
        assert report["summary"]["by_code"]["AKA1003"] == 1
        assert report["summary"]["fatal"] >= 1

    def test_diagnostics_carry_stable_fields(self):
        diag = self.build(OVERLAPPING)["diagnostics"][0]
        for key in (
            "code", "title", "severity", "hardware_domain",
            "message", "remediation", "location", "details",
        ):
            assert key in diag
        assert diag["location"]["line"] > 0

    def test_kernel_payload_includes_map_trace_and_sync(self):
        kernel = self.build(OVERLAPPING)["kernels"][0]
        for key in ("memory_map", "trace", "tensors", "constants"):
            assert key in kernel

    def test_target_describes_the_chip(self):
        target = self.build(LAYOUT)["target"]
        assert target["domains"]["UB"]["capacity_bytes"] == 192 * 1024
        assert target["reserved_event_ids"] == [6, 7]


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


class TestHtmlReport:
    def build(self, body: str) -> str:
        result = analyze_body(body)
        return build_html_report(
            result.unit, result.hardware, result.diagnostics,
            result.artifacts, solver_name=result.solver_name,
        )

    def test_is_a_complete_standalone_document(self):
        html = self.build(OVERLAPPING)
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")
        assert "<style>" in html
        # Standalone means no external assets.
        assert "src=http" not in html
        assert "href=\"http" not in html

    def test_has_a_title_and_a_verdict(self):
        html = self.build(OVERLAPPING)
        assert "<title>" in html
        assert "REJECTED" in html

    def test_renders_one_block_per_finding(self):
        result = analyze_body(OVERLAPPING)
        html = build_html_report(
            result.unit, result.hardware, result.diagnostics, result.artifacts
        )
        assert html.count("class='diag ") == len(result.diagnostics)

    def test_memory_bars_are_positioned_proportionally(self):
        html = self.build(LAYOUT)
        assert "class='lane'" in html
        assert "left:" in html and "width:" in html

    def test_collisions_get_the_collide_class(self):
        assert "collide" in self.build(OVERLAPPING)

    def test_message_text_is_escaped(self):
        # A tensor name is attacker-controlled only in the sense that it comes
        # from source text; it must still never break out of the document.
        body = ub_tensor("a", 1, 250)
        html = build_html_report(
            *_result_args(analyze_body(body))
        )
        assert "<script>" not in html

    def test_light_and_dark_theming_is_declared(self):
        html = self.build(LAYOUT)
        assert "prefers-color-scheme: light" in html
        assert ":root" in html


def _result_args(result):
    return (result.unit, result.hardware, result.diagnostics, result.artifacts)
