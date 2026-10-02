"""Tests for CCE declaration-decoration stripping and macro fidelity.

Three production constructs used to turn a whole translation unit into one
``ERROR`` node, because ``tree-sitter-cpp`` knows none of them:

* the CCE location qualifier, ``__forceinline__ [host, aicore] void f()``;
* an object-like macro that expands to nothing but decoration, which the
  expander blanks at its ``#define`` but leaves standing at every use -
  ``HOST_DEVICE`` is the one that matters;
* a variadic macro, whose ``__VA_ARGS__`` was never substituted, and an
  operator token passed as a macro argument, which came out parenthesised as
  the ill-formed ``operator (+)``.

Each test here asserts both halves of the contract: the construct parses, and
every byte offset still addresses the original file.
"""

from __future__ import annotations

import pytest

from ascend_analyzer.frontend import normalize_cce_syntax, preprocess_source
from ascend_analyzer.parsing.ast_visitor import _make_parser, _walk
from ascend_analyzer.parsing.preprocess import (
    decoration_only_macro,
    prepare_source,
    prepare_translation_unit,
)


#: A newline and a backslash, spelled so no editing pass can mangle them.
nl = chr(10)
BS = chr(92)


def expand(source: str, path: str = "<test>.cpp") -> str:
    """The preprocessed text, with whitespace collapsed for comparison.

    Macro expansion is asserted through the real pipeline rather than against
    an internal, so what these tests pin is the contract the parser sees.
    """
    result = preprocess_source(path, source)
    assert not result.errors, result.errors
    return " ".join(result.text.split())


def parses(source: str, path: str = "<test>.cpp") -> bool:
    prepared = prepare_translation_unit(path, source)
    return not _make_parser().parse(prepared.encoded).root_node.has_error


def error_lines(source: str, path: str = "<test>.cpp") -> int:
    prepared = prepare_translation_unit(path, source)
    root = _make_parser().parse(prepared.encoded).root_node
    lines = set()
    for node in _walk(root):
        if node.type == "ERROR" or node.is_missing:
            lines.update(range(node.start_point[0] + 1, node.end_point[0] + 2))
    return len(lines)


class TestLocationQualifier:
    def test_host_aicore_declaration_parses(self):
        source = (
            "__forceinline__ [host, aicore] void\n"
            "Tiling(int m, int n, int *out)\n"
            "{\n"
            "    *out = m * n;\n"
            "}\n"
        )
        assert parses(source)

    def test_single_location_declaration_parses(self):
        assert parses("inline [aicore] int f(int x) { return x; }\n")

    def test_qualifier_is_blanked_not_deleted(self):
        """Blanking keeps every later column where it was."""
        source = "inline [aicore] int f(int x) { return x; }\n"
        prepared = prepare_source("<test>.cpp", source)
        assert len(prepared.rewritten) == len(source)
        assert "[aicore]" not in prepared.rewritten
        assert prepared.rewritten.index("int f") == source.index("int f")

    def test_declaration_after_a_qualifier_is_still_found(self):
        source = (
            "__forceinline__ [host, aicore] void Tiling(int m) {}\n"
            "__global__ __aicore__ void entry() { Tiling(4); }\n"
        )
        prepared = prepare_translation_unit("<test>.cpp", source)
        root = _make_parser().parse(prepared.encoded).root_node
        assert not root.has_error

    @pytest.mark.parametrize(
        "source",
        [
            "int main() { int host = 0; int buf[4]; return buf[host]; }\n",
            "int main() { int aicore = 1; int t[2]; t[aicore] = 3; return t[0]; }\n",
        ],
    )
    def test_array_subscript_is_not_mistaken_for_a_qualifier(self, source):
        """``buf[host]`` is a subscript; blanking it would change the program."""
        prepared = prepare_source("<test>.cpp", source)
        assert prepared.rewritten == source
        assert prepared.decorations_blanked == 0

    def test_cpp_attribute_is_left_alone(self):
        source = "[[nodiscard]] int f();\n"
        assert prepare_source("<test>.cpp", source).rewritten == source

    def test_lambda_capture_is_left_alone(self):
        source = "int main() { int host = 1, aicore = 2;\n" \
                 "  auto l = [host, aicore]() { return host + aicore; };\n" \
                 "  return l(); }\n"
        assert prepare_source("<test>.cpp", source).rewritten == source

    def test_qualifier_inside_a_comment_is_left_alone(self):
        source = "// takes [host, aicore] in CCE\nint f();\n"
        assert prepare_source("<test>.cpp", source).rewritten == source

    def test_qualifier_inside_a_string_is_left_alone(self):
        source = 'const char *s = "[host, aicore]";\n'
        assert prepare_source("<test>.cpp", source).rewritten == source


class TestDecorationOnlyMacros:
    @pytest.mark.parametrize(
        "body",
        [
            "__forceinline__ [host, aicore]",
            "inline __attribute__((always_inline))",
            "[aicore]",
            "static inline",
            "",
        ],
    )
    def test_decoration_bodies_are_recognised(self, body):
        assert decoration_only_macro(body)

    @pytest.mark.parametrize(
        "body", ["48", "(NV_M * NV_K / 2)", "((x + 31) & ~31)", "my_function()"]
    )
    def test_value_bodies_are_not_decoration(self, body):
        """A macro that carries a value must keep reaching the constant folder."""
        assert not decoration_only_macro(body)

    def test_host_device_uses_are_blanked(self):
        source = (
            "#define HOST_DEVICE __forceinline__ [host, aicore]\n"
            "struct Coord {\n"
            "    HOST_DEVICE constexpr explicit Coord(int v) : value(v) {}\n"
            "    HOST_DEVICE constexpr int get() const { return value; }\n"
            "    int value;\n"
            "};\n"
        )
        assert parses(source)

    def test_an_unexpanded_decoration_macro_does_not_cascade(self):
        """The whole point: one bad specifier must not take the file with it."""
        source = (
            "#define HOST_DEVICE __forceinline__ [host, aicore]\n"
            "struct A { HOST_DEVICE constexpr A() {} };\n"
            "int later_declaration_is_still_parsed = 7;\n"
        )
        prepared = prepare_translation_unit("<test>.cpp", source)
        root = _make_parser().parse(prepared.encoded).root_node
        assert not root.has_error
        assert "later_declaration_is_still_parsed" in prepared.rewritten

    def test_a_value_macro_is_still_folded(self):
        source = "#define TILE 64\nint x = TILE;\n"
        prepared = prepare_translation_unit("<test>.cpp", source)
        assert dict(prepared.macro_object_defs).get("TILE") == "64"

    def test_header_guard_operand_is_not_blanked(self):
        """``#ifndef HOST_DEVICE`` needs its identifier to stay an identifier."""
        source = (
            "#ifndef HOST_DEVICE\n"
            "#define HOST_DEVICE __forceinline__ [host, aicore]\n"
            "#endif\n"
            "HOST_DEVICE void f() {}\n"
        )
        assert parses(source)

    def test_force_inline_underscored_spelling_is_known(self):
        source = (
            "#ifndef __force_inline__\n"
            "#define __force_inline__ inline __attribute__((always_inline))\n"
            "#endif\n"
            "__force_inline__ unsigned Min(unsigned a, unsigned b)\n"
            "{ return a < b ? a : b; }\n"
        )
        assert parses(source)


class TestGnuAttribute:
    def test_attribute_is_blanked(self):
        source = "inline __attribute__((always_inline)) int f() { return 1; }\n"
        prepared = prepare_source("<test>.cpp", source)
        assert "__attribute__" not in prepared.rewritten
        assert len(prepared.rewritten) == len(source)

    def test_nested_parentheses_are_balanced(self):
        """The scan must consume the inner ``(32)`` too, not stop at it."""
        source = "__attribute__((aligned(32))) int x;\n"
        prepared = prepare_source("<test>.cpp", source)
        blanked = " " * len("__attribute__((aligned(32))))") 
        assert "aligned" not in prepared.rewritten
        assert prepared.rewritten.rstrip() == blanked[:-1] + " int x;"
        assert parses(source)

    def test_attribute_between_specifier_and_type(self):
        source = (
            "static __attribute__((always_inline)) inline int f()"
            " { return 1; }\n"
        )
        assert parses(source)


class TestVariadicMacros:
    """``__VA_ARGS__`` used to survive unexpanded and take a header with it."""

    def test_va_args_is_substituted(self):
        source = (
            "#define REQ(...) typename enable_if<(__VA_ARGS__)>::type" + nl
            + "REQ(A::value, B::value) x;" + nl
        )
        assert expand(source) == "typename enable_if<(A::value, B::value)>::type x;"

    def test_named_parameter_before_the_pack(self):
        source = (
            "#define LOG(fmt, ...) emit(fmt, __VA_ARGS__)" + nl
            + "void f() { LOG(m, a, b); }" + nl
        )
        assert "emit(m, a, b)" in expand(source)

    def test_empty_pack(self):
        source = (
            "#define WRAP(a, ...) call(a __VA_ARGS__)" + nl
            + "void f() { WRAP(1); }" + nl
        )
        assert "call(1" in expand(source)

    def test_sfinae_template_parameter_is_fully_expanded(self):
        source = (
            "#define REQ(...) typename enable_if<(__VA_ARGS__)>::type* = nullptr" + nl
            + "template <class T, REQ(is_int<T>::value)>" + nl
            + "T f(T t) { return t; }" + nl
        )
        prepared = prepare_translation_unit("<test>.cpp", source)
        assert "__VA_ARGS__" not in prepared.rewritten


class TestOperatorArguments:
    """An operator token passed as an argument produced ``operator (+)``."""

    def test_generated_unary_operator_parses(self):
        source = (
            "#define UNARY(OP) " + BS + nl
            + "    template <int t> " + BS + nl
            + "    constexpr C<(OP t)> operator OP (C<t>) { return {}; }" + nl
            + "template <int> struct C {};" + nl
            + "UNARY(+)" + nl
            + "UNARY(-)" + nl
        )
        assert "operator + (" in expand(source)
        assert "operator ( +" not in expand(source)
        assert error_lines(source) == 0

    def test_generated_binary_operator_parses(self):
        source = (
            "#define BINARY(OP) " + BS + nl
            + "    template <int t, int u> " + BS + nl
            + "    constexpr C<(t OP u)> operator OP (C<t>, C<u>) { return {}; }" + nl
            + "template <int> struct C {};" + nl
            + "BINARY(<<)" + nl
        )
        assert "operator << (" in expand(source)
        assert error_lines(source) == 0

    def test_type_argument_is_not_parenthesised(self):
        """``(SCFABlockCube)<ARGS>`` is not C++; the bare name is."""
        source = (
            "#define TRAITS(T) struct Traits<T<ARGS>> {};" + nl
            + "TRAITS(SCFABlockCube)" + nl
        )
        assert expand(source) == "struct Traits<SCFABlockCube<ARGS>> {};"

    def test_compound_argument_keeps_its_meaning(self):
        """Precedence must survive where it actually matters."""
        source = (
            "#define AREA(w, h) ((w) * (h))" + nl
            + "int x = AREA(1 + 2, 3);" + nl
        )
        assert expand(source) == "int x = ((1 + 2) * (3));"


class TestCplusplusGuards:
    """``#ifdef __cplusplus`` wrapping ``extern "C" {``.

    The brace opens inside one preprocessor block and closes inside another,
    which ``tree-sitter-cpp`` cannot span.  Resolving the guard is exact, not
    approximate: this analyzer always parses as C++, so the guard is true and
    its directive lines are inert.
    """

    HEADER = (
        "#ifdef __cplusplus\n"
        'extern "C" {\n'
        "#endif\n"
        "int aclnnThing(int x);\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif\n"
    )

    def test_extern_c_guard_parses(self):
        assert parses(self.HEADER)

    def test_declaration_inside_the_guard_survives(self):
        prepared = prepare_translation_unit("<test>.cpp", self.HEADER)
        assert "aclnnThing" in prepared.rewritten

    def test_guard_directives_are_blanked(self):
        prepared = prepare_translation_unit("<test>.cpp", self.HEADER)
        assert "__cplusplus" not in prepared.rewritten
        assert 'extern "C"' in prepared.rewritten

    def test_declaration_maps_back_to_its_original_line(self):
        """The token preprocessor deletes directive lines, so the contract
        is a line *mapping* rather than a line count: every output line
        reports the original line it came from."""
        prepared = prepare_translation_unit("<test>.cpp", self.HEADER)
        lines = prepared.rewritten.split(nl)
        index = next(i for i, l in enumerate(lines, 1) if "aclnnThing" in l)
        # "int aclnnThing(int x);" is line 4 of HEADER.
        assert prepared.origin_line(index) == 4

    def test_if_defined_spelling_is_resolved(self):
        source = (
            "#if defined(__cplusplus)\n"
            'extern "C" {\n'
            "#endif\n"
            "int f(int x);\n"
            "#if defined(__cplusplus)\n"
            "}\n"
            "#endif\n"
        )
        assert parses(source)

    def test_else_branch_is_dropped(self):
        """The ``#else`` arm is the C one a C++ compiler discards."""
        source = (
            "#ifdef __cplusplus\n"
            "using Flag = bool;\n"
            "#else\n"
            "this is not valid C++ at all ???\n"
            "#endif\n"
            "Flag g = true;\n"
        )
        assert parses(source)

    def test_ifndef_cplusplus_keeps_the_else_branch(self):
        """``#ifndef __cplusplus`` inverts it: the first arm is the C one."""
        source = (
            "#ifndef __cplusplus\n"
            "this is C-only and not valid C++ ???\n"
            "#else\n"
            "using Flag = bool;\n"
            "#endif\n"
            "Flag g = true;\n"
        )
        assert parses(source)

    def test_unrelated_conditional_is_left_alone(self):
        """Only ``__cplusplus`` is resolved; other arms stay for tree-sitter."""
        source = "#ifdef SOMETHING_ELSE\nint a;\n#endif\nint b;\n"
        prepared = prepare_translation_unit("<test>.cpp", source)
        assert "SOMETHING_ELSE" in prepared.rewritten

    def test_nested_guard_inside_another_conditional(self):
        source = (
            "#ifndef HEADER_H\n"
            "#define HEADER_H\n"
            "#ifdef __cplusplus\n"
            'extern "C" {\n'
            "#endif\n"
            "int f(int x);\n"
            "#ifdef __cplusplus\n"
            "}\n"
            "#endif\n"
            "#endif\n"
        )
        assert parses(source)


class TestExternalAttributeMacros:
    """Decoration macros whose ``#define`` is not in the tree at all.

    ``CATLASS_DEVICE`` (42 files) and CANN's tiling-key DSL come from external
    headers, so the body-based test in :func:`decoration_only_macro` can never
    classify them.  These rules are therefore *positional*: they key on where
    the identifier sits, not on what it is called.
    """

    def test_lone_attribute_macro_is_blanked(self):
        source = (
            "struct Layout {\n"
            "    CATLASS_DEVICE\n"
            "    static Layout create(unsigned k, unsigned n) { return Layout{}; }\n"
            "};\n"
        )
        assert parses(source)

    def test_lone_macro_before_a_destructor(self):
        source = (
            "struct BlockMmad {\n"
            "    CATLASS_DEVICE\n"
            "    ~BlockMmad() {}\n"
            "};\n"
        )
        assert parses(source)

    def test_decoration_prefix_before_struct_is_erased(self):
        """``TEMPLATES_DEF_NO_DEFAULT struct Traits`` - no reachable define.

        The token preprocessor expands it to nothing because it is listed
        in ERASED_DECORATION_MACROS; there is no text rule for it.
        """
        source = "TEMPLATES_DEF_NO_DEFAULT struct Traits { int x; };" + nl
        prepared = prepare_translation_unit("<test>.cpp", source)
        assert "TEMPLATES_DEF_NO_DEFAULT" not in prepared.rewritten
        assert "struct Traits" in prepared.rewritten
        assert parses(source)

    def test_short_or_mixed_case_identifiers_are_left_alone(self):
        """The rule needs ALL-CAPS and four characters, to stay off real code."""
        for source in (
            "int x;\nABC\nint y;\n",          # too short
            "int x;\nCamelCase\nint y;\n",    # not all caps
        ):
            prepared = prepare_source("<test>.cpp", source)
            assert prepared.rewritten == source

    def test_all_caps_constant_in_an_expression_is_left_alone(self):
        """A continued expression must not lose its operand."""
        source = "int x = 1 +\n    TILE_BYTES\n    + 2;\n"
        prepared = prepare_source("<test>.cpp", source)
        assert "TILE_BYTES" in prepared.rewritten

    def test_null_and_bool_keywords_are_left_alone(self):
        source = "int x;\nNULL\nint y;\n"
        prepared = prepare_source("<test>.cpp", source)
        assert "NULL" in prepared.rewritten

    def test_sole_enum_member_is_not_blanked(self):
        """``enum E { BARBAZ };`` - a constant the folder needs, not decoration.

        Both a decoration macro and a sole enum member can sit directly after
        a ``{``, so the two are told apart by what *follows*: a declaration
        versus a closing brace.
        """
        source = "enum E {" + nl + "    BARBAZ" + nl + "};" + nl
        prepared = prepare_source("<test>.cpp", source)
        assert "BARBAZ" in prepared.rewritten

    def test_sole_initialiser_field_is_not_blanked(self):
        source = "S s = {" + nl + "    FIELDA" + nl + "};" + nl
        prepared = prepare_source("<test>.cpp", source)
        assert "FIELDA" in prepared.rewritten

    def test_trailing_enum_member_after_a_comma_is_not_blanked(self):
        source = "enum E {" + nl + "    FOO," + nl + "    BARBAZ" + nl + "};" + nl
        prepared = prepare_source("<test>.cpp", source)
        assert "BARBAZ" in prepared.rewritten

    def test_enum_constant_still_folds(self):
        """The regression this guards: a dropped constant resolves nothing."""
        source = (
            "enum Sizes {" + nl + "    TILEBYTES = 256" + nl + "};" + nl
            + "int x = TILEBYTES;" + nl
        )
        prepared = prepare_source("<test>.cpp", source)
        assert "TILEBYTES" in prepared.rewritten

    def test_multiline_macro_statement_is_blanked(self):
        """CANN's tiling-key DSL: not valid C++ even as an expression."""
        source = (
            "int before = 1;\n"
            "ASCENDC_TPL_SEL(\n"
            "    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(X_LAYOUT, LIST, 0, 1),\n"
            "                         ASCENDC_TPL_UINT_SEL(COFF, LIST, 1, 2)),\n"
            ");\n"
            "int after = 2;\n"
        )
        assert parses(source)
        # The DSL block is erased by the pre-lexing normalizer, byte for byte.
        normalized = normalize_cce_syntax(source)
        assert "ASCENDC_TPL_SEL" not in normalized
        assert len(normalized.encode("utf-8")) == len(source.encode("utf-8"))
        assert normalized.count(nl) == source.count(nl)
        # The declarations around it survive.
        assert "int before = 1;" in normalized
        assert "int after = 2;" in normalized

    def test_single_line_macro_call_is_left_alone(self):
        """A one-line call parses as a declaration; only the multi-line form cascades."""
        source = "PYBIND_THING(a, b);\n"
        prepared = prepare_source("<test>.cpp", source)
        assert prepared.rewritten == source

    def test_macro_with_a_body_is_left_intact(self):
        """``)`` followed by ``{`` is a definition, not a statement."""
        source = (
            "TORCH_LIBRARY(\n"
            "    mymod, m)\n"
            "{\n"
            "    int x = 1;\n"
            "}\n"
        )
        prepared = prepare_source("<test>.cpp", source)
        assert "TORCH_LIBRARY" in prepared.rewritten
        assert "int x = 1;" in prepared.rewritten

    def test_blanking_preserves_bytes_and_lines(self):
        source = (
            "ASCENDC_TPL_SEL(\n"
            "    ARGS(1, 2),\n"
            ");\n"
        )
        prepared = prepare_source("<test>.cpp", source)
        assert len(prepared.rewritten.encode("utf-8")) == len(source.encode("utf-8"))
        assert prepared.rewritten.count(chr(10)) == source.count(chr(10))

    def test_non_ascii_inside_a_blanked_region_keeps_byte_and_line_counts(self):
        """These sources carry Chinese comments; blanking must stay byte-exact."""
        source = (
            "ASCENDC_TPL_SEL(\n"
            "    /* \u4e2d\u6587\u6ce8\u91ca */ ARGS(1, 2),\n"
            ");\n"
            "int after = 1;\n"
        )
        prepared = prepare_source("<test>.cpp", source)
        assert len(prepared.rewritten.encode("utf-8")) == len(source.encode("utf-8"))
        assert prepared.rewritten.count(chr(10)) == source.count(chr(10))


class TestExtraQualifiers:
    @pytest.mark.parametrize(
        "source",
        [
            "__simd_vf__ void QuantImpl(__ubuf__ float *dst) { *dst = 0; }\n",
            "void f() { auto p = (__local_mem__ float *)0; (void)p; }\n",
            "struct S { __BLOCK_LOCAL__ inline unsigned counter; };\n",
        ],
    )
    def test_v3_qualifiers_parse(self, source):
        assert parses(source)

    def test_local_mem_resolves_to_the_unified_buffer(self):
        from ascend_analyzer.hardware import ADDRESS_SPACE_TO_DOMAIN, PhysicalDomain

        assert ADDRESS_SPACE_TO_DOMAIN["__local_mem__"] is PhysicalDomain.UB


class TestStringification:
    """``#param`` renders its argument as a string literal.

    Left unexpanded the ``#`` reads as a preprocessor directive starting in
    the middle of a statement, which is what made
    ``GetOpApiFuncAddr(#aclCreateTensor)`` unparseable and alone accounted for
    thousands of error lines in ``torch_binding.cpp``.
    """

    def test_argument_becomes_a_string_literal(self):
        source = (
            "#define NAMEOF(x) #x" + nl
            + "const char *n = NAMEOF(hello);" + nl
        )
        assert expand(source) == 'const char *n = "hello";'

    def test_paste_pastes(self):
        source = (
            "#define JOIN(a, b) a##b" + nl
            + "int JOIN(foo, bar) = 1;" + nl
        )
        assert expand(source) == "int foobar = 1;"

    def test_paste_and_stringify_in_one_body(self):
        source = (
            "#define GET(name) auto _##name = f(#name)" + nl
            + "void k() { GET(acl); }" + nl
        )
        got = expand(source)
        assert "_acl" in got
        assert '"acl"' in got

    def test_the_op_api_idiom_parses(self):
        source = (
            "#define GET_ADDR(name) " + BS + nl
            + "    static const auto name = cast<_##name>(GetAddr(#name))" + nl
            + "void f() { GET_ADDR(aclCreateTensor); }" + nl
        )
        prepared = prepare_translation_unit("<test>.cpp", source)
        assert '"aclCreateTensor"' in prepared.rewritten
        assert "#aclCreateTensor" not in prepared.rewritten
        assert error_lines(source) == 0


class TestRecursionAndNesting:
    """Cases the hand-rolled expander handled badly or not at all."""

    def test_nested_expansion(self):
        source = (
            "#define INNER 4" + nl
            + "#define OUTER (INNER * 2)" + nl
            + "int x = OUTER;" + nl
        )
        assert expand(source) == "int x = (4 * 2);"

    def test_self_referential_macro_terminates(self):
        """A macro naming itself must not recurse forever."""
        source = "#define LOOP LOOP + 1" + nl + "int x = LOOP;" + nl
        assert expand(source) == "int x = LOOP + 1;"

    def test_undef_retires_a_macro(self):
        source = (
            "#define TILE 64" + nl
            + "int a = TILE;" + nl
            + "#undef TILE" + nl
            + "int b = TILE;" + nl
        )
        got = expand(source)
        assert "int a = 64;" in got
        assert "int b = TILE;" in got

    def test_invocation_before_the_definition_is_not_expanded(self):
        source = (
            "int a = TILE;" + nl
            + "#define TILE 64" + nl
            + "int b = TILE;" + nl
        )
        got = expand(source)
        assert "int a = TILE;" in got
        assert "int b = 64;" in got

    def test_conditional_on_a_known_macro_is_resolved(self):
        source = (
            "#define FEATURE 1" + nl
            + "#if FEATURE" + nl + "int on = 1;" + nl
            + "#else" + nl + "int off = 1;" + nl + "#endif" + nl
        )
        got = expand(source)
        assert "int on = 1;" in got
        assert "int off = 1;" not in got

    def test_conditional_on_an_unknown_macro_is_left_for_the_parser(self):
        """Both arms must stay analyzable; see ``on_directive_handle``."""
        source = (
            "#ifdef SOME_FEATURE" + nl + "int on = 1;" + nl
            + "#else" + nl + "int off = 1;" + nl + "#endif" + nl
        )
        got = expand(source)
        assert "int on = 1;" in got
        assert "int off = 1;" in got


class TestKernelLaunchSyntax:
    def test_launch_brackets_are_folded(self):
        source = "void host() { k<<<cfg>>>(p); }" + nl
        assert parses(source)
        assert "<<<" not in normalize_cce_syntax(source)

    def test_shift_operators_are_left_alone(self):
        source = "int x = a << 3;" + nl
        assert normalize_cce_syntax(source) == source


class TestOffsetInvariants:
    def test_blanking_preserves_byte_length(self):
        source = (
            "#define HOST_DEVICE __forceinline__ [host, aicore]\n"
            "HOST_DEVICE __attribute__((always_inline)) int f() { return 1; }\n"
        )
        prepared = prepare_source("<test>.cpp", source)
        assert len(prepared.rewritten.encode("utf-8")) == len(source.encode("utf-8"))

    def test_blanking_never_eats_a_newline(self):
        source = "__attribute__((aligned(\n    32))) int x;\n"
        prepared = prepare_source("<test>.cpp", source)
        assert prepared.rewritten.count("\n") == source.count("\n")

    def test_offsets_survive_a_non_ascii_comment(self):
        """Ascend sources carry Chinese comments; offsets are byte offsets."""
        source = "// 中文注释\ninline [aicore] int f() { return 1; }\n"
        prepared = prepare_source("<test>.cpp", source)
        encoded = prepared.rewritten.encode("utf-8")
        assert len(encoded) == len(source.encode("utf-8"))
        assert encoded.index(b"int f") == source.encode("utf-8").index(b"int f")
