"""Tests for the pcpp-based source frontend.

The frontend's job is narrow and load-bearing: hand tree-sitter a fully
expanded translation unit, and hand the rest of the analyzer an exact map back
to the lines the user wrote.  The tests here pin both halves, plus the four
places where Ascend C needs something a plain C preprocessor does not do.
"""

from __future__ import annotations

import pytest

from ascend_analyzer.frontend import (
    AscendCPreprocessor,
    normalize_cce_syntax,
    preprocess_source,
)
from ascend_analyzer.frontend.ascend_preprocessor import (
    CORE_GUARD_MACROS,
    ERASED_DECORATION_MACROS,
    SEEDED_MACROS,
)

nl = chr(10)
BS = chr(92)


def expand(source: str, path: str = "t.cpp") -> str:
    result = preprocess_source(path, source)
    assert not result.errors, result.errors
    return " ".join(result.text.split())


class TestLineMapping:
    def test_a_declaration_maps_to_its_own_line(self):
        source = (
            "#define TILE 64" + nl      # 1
            + "#define AREA(w,h) ((w)*(h))" + nl  # 2
            + "int a = TILE;" + nl       # 3
            + "int b = AREA(2,3);" + nl  # 4
        )
        result = preprocess_source("t.cpp", source)
        lines = result.text.split(nl)
        for needle, want in (("int a", 3), ("int b", 4)):
            index = next(i for i, l in enumerate(lines, 1) if l.strip().startswith(needle))
            assert result.origin_line(index) == want

    def test_line_markers_are_not_left_in_the_text(self):
        source = "#define X 1" + nl + "int a = X;" + nl
        assert "#line" not in preprocess_source("t.cpp", source).text

    def test_origins_cover_every_line(self):
        source = "#define X 1" + nl + "int a = X;" + nl + "int b = 2;" + nl
        result = preprocess_source("t.cpp", source)
        assert len(result.line_origins) == result.text.count(nl) + 1

    def test_origins_never_exceed_the_source(self):
        source = "#define X 1" + nl + "int a = X;" + nl
        result = preprocess_source("t.cpp", source)
        assert max(result.line_origins) <= source.count(nl) + 1

    def test_included_content_maps_to_the_include_line(self, tmp_path):
        """A diagnostic has to name a line the reader can open."""
        header = tmp_path / "geom.h"
        header.write_text("int from_header = 1;" + nl, encoding="utf-8")
        main = tmp_path / "k.cpp"
        source = (
            "int before = 0;" + nl          # 1
            + '#include "geom.h"' + nl      # 2
            + "int after = 2;" + nl         # 3
        )
        main.write_text(source, encoding="utf-8")
        result = preprocess_source(str(main), source)
        lines = result.text.split(nl)
        index = next(i for i, l in enumerate(lines, 1) if "from_header" in l)
        assert result.origin_line(index) == 2

    def test_nested_include_maps_to_the_outer_include_line(self, tmp_path):
        """A header included by a header has no #include line of its own in
        the primary file, so it keeps the outer header's attribution."""
        (tmp_path / "inner.h").write_text("int from_inner = 1;" + nl, encoding="utf-8")
        (tmp_path / "outer.h").write_text(
            '#include "inner.h"' + nl + "int from_outer = 2;" + nl, encoding="utf-8"
        )
        main = tmp_path / "k.cpp"
        source = (
            "int before = 0;" + nl        # 1
            + '#include "outer.h"' + nl   # 2
            + "int after = 3;" + nl       # 3
        )
        main.write_text(source, encoding="utf-8")
        result = preprocess_source(str(main), source)
        lines = result.text.split(nl)
        for needle in ("from_inner", "from_outer"):
            index = next(i for i, l in enumerate(lines, 1) if needle in l)
            assert result.origin_line(index) == 2, needle
        # The primary file's own lines still map to themselves.
        index = next(i for i, l in enumerate(lines, 1) if "int after" in l)
        assert result.origin_line(index) == 3

    def test_utf8_header_is_decoded(self, tmp_path):
        """These headers carry Chinese comments; cp1252 would abort the unit."""
        header = tmp_path / "zh.h"
        header.write_text(
            "/* 中文注释 */" + nl + "int decoded = 1;" + nl,
            encoding="utf-8",
        )
        main = tmp_path / "k.cpp"
        source = '#include "zh.h"' + nl + "int x = decoded;" + nl
        main.write_text(source, encoding="utf-8")
        result = preprocess_source(str(main), source)
        assert not result.errors
        assert "int decoded = 1;" in result.text


class TestCoreGuards:
    def test_guards_are_not_defined(self):
        for name in CORE_GUARD_MACROS:
            assert name not in SEEDED_MACROS

    def test_a_core_guard_conditional_survives_with_both_arms(self):
        """The visitor reads this structure to tell Cube code from Vector."""
        source = (
            "#ifdef __DAV_C220_CUBE__" + nl
            + "int cube = 1;" + nl
            + "#else" + nl
            + "int vec = 1;" + nl
            + "#endif" + nl
        )
        got = expand(source)
        assert "#ifdef __DAV_C220_CUBE__" in got
        assert "#else" in got and "#endif" in got
        assert "int cube = 1;" in got
        assert "int vec = 1;" in got

    def test_the_conditional_stays_balanced(self):
        source = (
            "#ifdef __DAV_C220_VEC__" + nl + "int v = 1;" + nl + "#endif" + nl
        )
        result = preprocess_source("t.cpp", source)
        assert not result.errors
        assert result.text.count("#ifdef") == result.text.count("#endif")

    def test_an_undecidable_conditional_keeps_both_arms(self):
        source = (
            "#ifdef SOME_FEATURE" + nl + "int on = 1;" + nl
            + "#else" + nl + "int off = 1;" + nl + "#endif" + nl
        )
        got = expand(source)
        assert "int on = 1;" in got and "int off = 1;" in got

    def test_a_decidable_conditional_is_resolved(self):
        source = (
            "#define KNOWN 1" + nl
            + "#if KNOWN" + nl + "int on = 1;" + nl
            + "#else" + nl + "int off = 1;" + nl + "#endif" + nl
        )
        got = expand(source)
        assert "int on = 1;" in got and "int off = 1;" not in got

    def test_cplusplus_is_defined(self):
        """``extern "C" {`` lives in the true arm of that guard."""
        source = (
            "#ifdef __cplusplus" + nl + 'extern "C" {' + nl + "#endif" + nl
            + "int f(int);" + nl
            + "#ifdef __cplusplus" + nl + "}" + nl + "#endif" + nl
        )
        got = expand(source)
        assert 'extern "C" {' in got
        assert "__cplusplus" not in got

    def test_core_predicates_become_the_visitor_sentinels(self):
        source = "void k() { if (ASCEND_IS_AIV) { f(); } }" + nl
        assert "__ascend_core_is_aiv" in expand(source)


class TestErasedDecoration:
    @pytest.mark.parametrize("name", ERASED_DECORATION_MACROS)
    def test_each_erased_macro_vanishes(self, name):
        source = name + " int f();" + nl
        got = expand(source)
        assert name not in got
        assert "int f();" in got

    @pytest.mark.parametrize(
        "name", ["__global__", "__aicore__", "__gm__", "__ubuf__"]
    )
    def test_address_space_and_entry_qualifiers_survive(self, name):
        """These are not noise: they carry the domain and the entry point."""
        assert name not in ERASED_DECORATION_MACROS
        source = name + " int x;" + nl
        assert name in expand(source)


class TestUnresolvedIncludes:
    def test_a_missing_include_does_not_abort(self):
        source = "#include <catlass/layout.hpp>" + nl + "int after = 1;" + nl
        result = preprocess_source("t.cpp", source)
        assert not result.errors
        assert "int after = 1;" in result.text

    def test_a_missing_system_include_is_removed(self):
        source = "#include <tla/does_not_exist.hpp>" + nl + "int a = 1;" + nl
        assert "include" not in expand(source)


class TestIncludeNormalizationCache:
    """A header edited between runs must not come back from the cache."""

    def test_an_edited_include_is_re_normalized(self, tmp_path):
        header = tmp_path / "svc.h"
        main = tmp_path / "main.cpp"
        main.write_text('#include "svc.h"' + nl, encoding="utf-8")

        header.write_text("int first_version = 1;" + nl, encoding="utf-8")
        one = preprocess_source(str(main), main.read_text(encoding="utf-8"))
        assert "first_version" in one.text

        # Longer than the first body, so even a coarse mtime cannot mask the
        # edit: the cache key is (path, mtime_ns, size).
        header.write_text(
            "int second_and_longer_version = 2;" + nl, encoding="utf-8"
        )
        two = preprocess_source(str(main), main.read_text(encoding="utf-8"))
        assert "second_and_longer_version" in two.text
        assert "first_version" not in two.text


class TestCommentsInMacroBodies:
    """A comment on a ``#define`` line is not part of the macro.

    ``on_comment`` passes comments through verbatim to keep output lines
    aligned, so ``pcpp`` would otherwise capture the trailing ``// 512B`` into
    the macro's stored value and re-emit it at every use site - where a ``//``
    swallows the rest of the line, including each closing parenthesis after
    it.  ``#define UB_BANK_DEPTH_STRIDE (...) // 512B`` alone put four
    thousand error lines into ``lightning_indexer.cpp`` that way.
    """

    def test_a_line_comment_after_a_body_stays_out_of_it(self):
        source = (
            "#define STRIDE (2 * 8)    // 512B" + nl
            + "int buf[STRIDE];" + nl
        )
        assert expand(source) == "int buf[(2 * 8)];"

    def test_a_comment_inside_a_nested_expansion_does_not_swallow_code(self):
        source = (
            "#define BLOCK 32   // 32B" + nl
            + "#define STRIDE (8 * BLOCK)    // 512B" + nl
            + "int buf[STRIDE];" + nl
        )
        assert expand(source) == "int buf[(8 * 32)];"

    def test_a_block_comment_becomes_a_space_not_a_paste(self):
        source = "#define CAT(a, b) a/*keep apart*/b" + nl + "int CAT(x, y);" + nl
        assert expand(source) == "int x y;"

    def test_the_comment_leaves_the_object_macro_body_clean(self):
        import io

        from ascend_analyzer.frontend.ascend_preprocessor import AscendCPreprocessor

        engine = AscendCPreprocessor()
        engine.parse(
            "#define WIDTH 128    // bytes per row" + nl + "int w = WIDTH;" + nl,
            source="t.cpp",
        )
        engine.write(io.StringIO())  # macros materialize as tokens are consumed
        assert engine.macros["WIDTH"].value[0].value == "128"


class TestNormalizer:
    """Byte-for-byte rewrites, so offsets still address the original file."""

    @pytest.mark.parametrize(
        "source",
        [
            "__forceinline__ [host, aicore] void f() {}" + nl,
            "inline [aicore] int g() { return 1; }" + nl,
            "void h() { k<<<cfg>>>(p); }" + nl,
            "if ASCEND_IS_AIV { f(); }" + nl,
            "ASCENDC_TPL_SEL(" + nl + "    ARGS(1, 2)," + nl + ");" + nl,
        ],
    )
    def test_every_rewrite_preserves_bytes_and_lines(self, source):
        out = normalize_cce_syntax(source)
        assert len(out.encode("utf-8")) == len(source.encode("utf-8"))
        assert out.count(nl) == source.count(nl)

    def test_subscript_is_not_mistaken_for_a_qualifier(self):
        source = "int main() { int host = 0; int b[4]; return b[host]; }" + nl
        assert normalize_cce_syntax(source) == source

    def test_qualifier_in_a_comment_is_left_alone(self):
        source = "// takes [host, aicore] in CCE" + nl + "int f();" + nl
        assert normalize_cce_syntax(source) == source

    def test_qualifier_in_a_string_is_left_alone(self):
        source = 'const char *s = "[host, aicore]";' + nl
        assert normalize_cce_syntax(source) == source

    def test_unbalanced_parenthesis_in_a_string_does_not_derail_a_span(self):
        """``TORCH_CHECK(c, "... (see docs")`` must not confuse the matcher."""
        source = (
            "int before = 1;" + nl
            + "TORCH_CHECK(" + nl
            + '    cond, "a message with ( an unbalanced paren");' + nl
            + "int after = 2;" + nl
        )
        out = normalize_cce_syntax(source)
        assert "int before = 1;" in out
        assert "int after = 2;" in out
        assert len(out.encode("utf-8")) == len(source.encode("utf-8"))

    def test_line_continuations_inside_a_define_are_preserved(self):
        """Erasing a splice truncates the macro and spills its body."""
        source = (
            "#define CHECKED(x) " + BS + nl
            + "    TORCH_CHECK(" + BS + nl
            + "        (x) != nullptr," + BS + nl
            + '        "null")' + nl
            + "void f() { CHECKED(p); }" + nl
        )
        out = normalize_cce_syntax(source)
        assert out.count(BS + nl) == source.count(BS + nl)

    def test_nothing_inside_a_directive_is_rewritten(self):
        source = "#define WRAP ASCENDC_TPL_SEL(" + BS + nl + "    a, b)" + nl
        assert normalize_cce_syntax(source) == source

    def test_shift_operator_is_not_a_launch(self):
        source = "int x = a << 3; int y = b >> 2;" + nl
        assert normalize_cce_syntax(source) == source

    def test_single_line_dsl_call_is_left_alone(self):
        source = "REGISTER_THING(a, b);" + nl
        assert normalize_cce_syntax(source) == source

    def test_a_definition_body_is_left_alone(self):
        source = (
            "TORCH_LIBRARY(" + nl + "    mod, m)" + nl + "{" + nl
            + "    int x = 1;" + nl + "}" + nl
        )
        out = normalize_cce_syntax(source)
        assert "TORCH_LIBRARY" in out and "int x = 1;" in out


class TestConfigurationErrors:
    def test_an_error_in_an_undecided_conditional_is_not_reported(self):
        """A feature flag being unset is not a fault in the kernel."""
        source = (
            "#if !defined(VLLM_ENABLE_TURBOQUANT)" + nl
            + '#error "needs the 950 build"' + nl
            + "#endif" + nl
            + "int x = 1;" + nl
        )
        result = preprocess_source("t.cpp", source)
        assert not result.errors
        assert "int x = 1;" in result.text

    def test_the_guarded_code_is_still_analyzable(self):
        source = (
            "#if !defined(SOME_FLAG)" + nl
            + '#error "unset"' + nl
            + "#else" + nl
            + "int enabled = 1;" + nl
            + "#endif" + nl
        )
        assert "int enabled = 1;" in preprocess_source("t.cpp", source).text


class TestGracefulDegradation:
    def test_a_frontend_fault_returns_the_unexpanded_source(self, monkeypatch):
        """A partial result beats none - but it has to be reported."""
        def boom(self, *args, **kwargs):
            raise RuntimeError("synthetic frontend fault")

        monkeypatch.setattr(AscendCPreprocessor, "parse", boom)
        source = "#define X 1" + nl + "int a = X;" + nl
        result = preprocess_source("t.cpp", source)
        assert result.errors
        assert "int a = X;" in result.text

    def test_the_fault_is_visible_as_a_diagnostic(self, monkeypatch):
        from ascend_analyzer.analyzer import AnalyzerOptions, KernelAnalyzer

        def boom(self, *args, **kwargs):
            raise RuntimeError("synthetic frontend fault")

        monkeypatch.setattr(AscendCPreprocessor, "parse", boom)
        result = KernelAnalyzer(AnalyzerOptions()).analyze_source(
            "t.cpp", "__global__ __aicore__ void k() {}" + nl
        )
        assert "AKA9001" in result.codes()
        assert any("preprocessing was incomplete" in d.message for d in result.diagnostics)
