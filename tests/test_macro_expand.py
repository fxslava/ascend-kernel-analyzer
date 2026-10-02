"""Tests for the macro-expansion preprocessor front-end."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ascend_analyzer.parsing.macro_expand import (  # noqa: E402
    MacroExpansion,
    _rewrite_launches,
    expand_macros,
)


def expand(source: str) -> MacroExpansion:
    return expand_macros(source, base_dir=None)


class TestObjectMacros:
    def test_object_macros_are_blankable_and_recorded_in_order(self):
        source = (
            "#define NV_M 16\n"
            "#define NV_K 64\n"
            "#define NV_A_BYTES (NV_M * NV_K / 2)\n"
            "int x = NV_A_BYTES;\n"
        )
        exp = expand(source)
        assert exp.changed
        assert list(exp.object_macros) == ["NV_M", "NV_K", "NV_A_BYTES"]
        # The directive lines are blanked but keep the line count stable.
        assert exp.text.splitlines()[0].strip() == ""
        assert "int x = NV_A_BYTES;" in exp.text
        assert exp.line_origins == (1, 2, 3, 4)

    def test_trailing_comments_are_stripped_from_bodies(self):
        source = "#define TILE 512   // one packed fractal\nint x = TILE;\n"
        exp = expand(source)
        assert exp.object_macros["TILE"] == "512"

    def test_empty_guard_defines_are_recorded_but_harmless(self):
        source = (
            "#ifndef HEADER_H\n"
            "#define HEADER_H\n"
            "#endif\n"
            "int x = 1;\n"
        )
        exp = expand(source)
        assert exp.object_macros["HEADER_H"] == ""


class TestFunctionMacros:
    def test_stage_macro_is_inlined_at_invocation(self):
        source = (
            "#define STAGE(q) do { f((q)); g(); } while (0)\n"
            "void k() {\n"
            "    STAGE(7);\n"
            "}\n"
        )
        exp = expand(source)
        assert exp.stats["expansions"] == 1
        expanded = [l for l in exp.text.splitlines() if "do {" in l]
        assert expanded == ["    do { f((7)); g(); } while (0);"]

    def test_nested_event_selector_expands_inside_stage_macro(self):
        source = (
            "#define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)\n"
            "#define MAD(p) do { SetFlag<HardEvent::MTE1_M>(EV(p)); } while (0)\n"
            "void k() { MAD(0); MAD(1); }\n"
        )
        exp = expand(source)
        line = exp.text.splitlines()[-1]
        assert "((((0)) ? EVENT_ID1 : EVENT_ID0))" in line
        assert "((((1)) ? EVENT_ID1 : EVENT_ID0))" in line
        assert "EV(" not in line

    def test_expansion_lines_map_back_to_the_invocation_line(self):
        source = (
            "void k() {\n"                       # 1
            "#define A(x) do { one((x)); \\\n"   # 2
            " two((x)); } while (0)\n"           # 3
            "    A(9);\n"                        # 4
            "}\n"                                # 5
        )
        exp = expand(source)
        out_lines = exp.text.splitlines()
        # The multi-line define is blanked; the invocation line carries the
        # whole expansion and maps back to original line 4.
        invocation = next(i for i, l in enumerate(out_lines) if "one((9))" in l)
        assert "two((9))" in out_lines[invocation]
        assert exp.line_origins[invocation] == 4

    def test_invocation_before_definition_is_not_expanded(self):
        source = (
            "void k() { EARLY(1); }\n"
            "#define EARLY(x) later((x))\n"
            "void k2() { EARLY(2); }\n"
        )
        exp = expand(source)
        assert "EARLY(1);" in exp.text          # before the define: untouched
        assert "later((2));" in exp.text        # after the define: expanded

    def test_undef_retires_a_macro(self):
        source = (
            "#define G(x) g((x))\n"
            "void a() { G(1); }\n"
            "#undef G\n"
            "void b() { G(2); }\n"
        )
        exp = expand(source)
        assert "g((1));" in exp.text
        assert "G(2);" in exp.text

    def test_member_and_qualified_names_are_not_expanded(self):
        source = (
            "#define GET(x) never((x))\n"
            "void k() { obj.GET(1); ns::GET(2); ptr->GET(3); }\n"
        )
        exp = expand(source)
        line = exp.text.splitlines()[-1]
        assert "obj.GET(1);" in line
        assert "ns::GET(2);" in line
        assert "ptr->GET(3);" in line
        assert "never" not in line

    def test_self_referential_macro_terminates(self):
        source = "#define LOOP(x) LOOP((x)) f((x))\nvoid k() { LOOP(1); }\n"
        exp = expand(source)  # must not raise or hang
        assert "f((1))" in exp.text

    def test_macro_names_inside_comments_and_strings_are_left_alone(self):
        source = (
            "#define EV(p) ((p) ? EVENT_ID1 : EVENT_ID0)\n"
            "// EV(p) mentioned in a comment\n"
            'const char* s = "EV(p)";\n'
        )
        exp = expand(source)
        assert "// EV(p) mentioned in a comment" in exp.text
        assert '"EV(p)"' in exp.text


class TestLaunchSyntax:
    def test_cce_launch_becomes_a_nested_call(self):
        assert (
            _rewrite_launches("k<<<blk, s>>>(a, b);")
            == "k(blk, s)(a, b);"
        )

    def test_template_closing_brackets_are_untouched(self):
        line = "Vector<Vector<int>> v;"
        assert _rewrite_launches(line) == line

    def test_plain_lines_are_untouched(self):
        assert _rewrite_launches("a << b << c;") == "a << b << c;"


class TestIncludeInlining:
    def test_local_header_is_inlined_with_origin_mapping(self, tmp_path):
        (tmp_path / "geom.h").write_text(
            "#define TILE 512\n", encoding="utf-8"
        )
        source = (
            'line one\n'
            '#include "geom.h"\n'
            'int x = TILE;\n'
        )
        exp = expand_macros(source, base_dir=tmp_path)
        assert exp.stats["includes_inlined"] == 1
        # Header line maps to the #include line; code after it keeps its own.
        lines = exp.text.splitlines()
        assert lines[1].startswith("#define")
        assert exp.line_origins == (1, 2, 3)

    def test_unresolvable_includes_are_kept(self, tmp_path):
        source = '#include "kernel_operator.h"\nint x = 1;\n'
        exp = expand_macros(source, base_dir=tmp_path)
        assert exp.text == source

    def test_each_header_is_inlined_once(self, tmp_path):
        (tmp_path / "a.h").write_text("#define A 1\n", encoding="utf-8")
        (tmp_path / "b.h").write_text('#include "a.h"\n#define B 2\n', encoding="utf-8")
        source = '#include "a.h"\n#include "b.h"\n'
        exp = expand_macros(source, base_dir=tmp_path)
        assert exp.stats["includes_inlined"] == 2  # a.h and b.h, a.h not twice
        assert exp.text.count("#define A 1") == 1


class TestIdentityPath:
    def test_source_without_directives_is_unchanged(self):
        source = "void k() { f(1); }\n"
        exp = expand(source)
        assert not exp.changed
        assert exp.text == source
        assert exp.line_origins == (1,)
