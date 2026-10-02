"""Token-level preprocessing for Ascend C, on top of ``pcpp``.

This replaces the regex-and-string expander that preceded it.  ``pcpp`` is a
pure-Python ISO C99/C11 token preprocessor, so macro expansion, ``#``
stringification, ``##`` pasting, ``__VA_ARGS__`` and conditional evaluation
are all handled by a real lexer instead of by pattern matching - which is
what removes the whole class of bugs the previous expander kept producing
(corrupted parentheses, half-substituted arguments, recursion limits).

Three things this module has to add on top of ``pcpp``.

**Line fidelity.**  ``pcpp`` does not preserve line numbers: it *deletes*
directive lines rather than blanking them, so with ``line_directive = None``
the output of a real kernel drifts by tens of lines and every diagnostic
points at the wrong place.  It does, however, emit ``#line N "file"`` markers.
:func:`preprocess_source` consumes those markers to build the same
``line_origins`` map the rest of the analyzer already uses, then blanks the
marker lines.  Columns still shift on a line a macro expanded, exactly as
they did before; lines do not.

**CCE syntax the C grammar rejects.**  A location qualifier
(``[host, aicore]``) is not C, so the lexer has to be given something it can
read.  :func:`normalize_cce_syntax` erases those constructs *byte for byte*,
before tokenizing, so offsets still address the original file.

**The core-split conditionals must survive.**  ``#ifdef __DAV_C220_CUBE__`` /
``#ifdef __DAV_C220_VEC__`` is how a mix kernel divides itself between the
Cube and Vector cores, and
:meth:`~ascend_analyzer.parsing.ast_visitor._KernelWalker._handle_preproc_conditional`
walks *both* arms to tag each operation with the core it compiles into.
Defining either macro would make ``pcpp`` pick one arm and delete the other,
which would merge the two event spaces and silently stop analyzing half the
kernel.  They are therefore passed through untouched for the parser to see.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from pcpp import Action, OutputDirective, Preprocessor

__all__ = [
    "AscendCPreprocessor",
    "PreprocessResult",
    "normalize_cce_syntax",
    "preprocess_source",
    "CORE_GUARD_MACROS",
    "SEEDED_MACROS",
    "ERASED_DECORATION_MACROS",
]

#: Compile-time core selectors.  These are deliberately **not** defined: the
#: AST visitor needs the ``#ifdef`` to reach the parse tree so it can tag each
#: arm with the core it belongs to.  Any conditional mentioning one of them is
#: passed through instead of being evaluated.
CORE_GUARD_MACROS: Tuple[str, ...] = ("__DAV_C220_CUBE__", "__DAV_C220_VEC__")

#: Macros pre-seeded before parsing.
#:
#: ``ASCEND_IS_AIC``/``ASCEND_IS_AIV`` become the sentinels the visitor
#: recognises as *run-time* core predicates, which is how it narrows an
#: ``if (ASCEND_IS_AIV) { ... }`` body to one core.  ``__CCE_KT_TEST__`` picks
#: the kernel-test arm, which is the one that carries the code.
SEEDED_MACROS: Mapping[str, str] = {
    # This analyzer always parses as C++.  Without it every
    # ``#ifdef __cplusplus`` guard evaluates false and the ``extern "C" {``
    # arm - which is where the declarations are - gets deleted.
    "__cplusplus": "201703L",
    "__CCE_KT_TEST__": "1",
    "ASCEND_IS_AIC": "(__ascend_core_is_aic)",
    "ASCEND_IS_AIV": "(__ascend_core_is_aiv)",
}

#: Decoration macros expanded to nothing at the token level.
#:
#: These carry no meaning for layout or synchronisation, and their
#: definitions live in headers that are not in this tree (``CATLASS_DEVICE``)
#: or expand to a location qualifier the C grammar cannot read
#: (``HOST_DEVICE`` -> ``__forceinline__ [host, aicore]``).  Defining them
#: empty makes the lexer drop them, which is cheaper and more reliable than
#: matching them in the text afterwards.
#:
#: Deliberately absent: ``__aicore__``, ``__global__``, ``__gm__``,
#: ``__ubuf__`` and the other address-space qualifiers.  Those are *not*
#: noise - :func:`~ascend_analyzer.parsing.preprocess.prepare_source` records
#: their byte spans to recover each declaration's physical domain, and
#: ``__global__`` is how a kernel entry point is identified.  Erasing them
#: here would silently discard every domain the analyzer knows.
#: Each name here was observed decorating declarations in this fleet with no
#: reachable definition.  The list is explicit on purpose: it is auditable,
#: and a name that is wrong on it is visible rather than inferred.
ERASED_DECORATION_MACROS: Tuple[str, ...] = (
    "HOST_DEVICE",
    "CATLASS_DEVICE",
    "CATLASS_HOST_DEVICE",
    "ACLNN_API",
    "TEMPLATES_DEF_NO_DEFAULT",
    "TEMPLATES_DEF",
    "TEMPLATE_ARGS_DEF",
)

#: CCE function *location* qualifiers, erased before tokenizing.
_CCE_LOCATIONS = (
    "host", "aicore", "aicpu", "vector_core", "aicube", "mix", "mix_aic",
    "mix_aiv",
)
_CCE_BRACKET_RE = re.compile(
    r"\[[ \t]*(?:" + "|".join(_CCE_LOCATIONS) + r")"
    r"(?:[ \t]*,[ \t]*(?:" + "|".join(_CCE_LOCATIONS) + r"))*[ \t]*\]"
)
#: A bracket glued to the preceding token is a subscript, not a qualifier.
_SUBSCRIPT_LEAD = set("_)]abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
#: What may begin the return type that follows a location qualifier.
_TYPE_START = set("_*&~:abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

#: ``if ASCEND_IS_AIV {`` - CCE allows the condition without parentheses.
_BARE_COND_RE = re.compile(
    r"\b(if|while)[ \t]+(ASCEND_IS_AI[CV])[ \t]*\{"
)

#: A file-scope DSL block: a multi-line ALL-CAPS macro statement, which is how
#: CANN writes a tiling-key table.  Its argument list is not valid C even as
#: an expression, and the macro lives in headers that are not in the tree.
_DSL_CALL_RE = re.compile(r"(?:^|(?<=[;{}\n]))[ \t\n]*([A-Z][A-Z0-9_]{3,})[ \t\n]*\(")

#: CCE kernel-launch brackets, which the C grammar does not have.
_LAUNCH_OPEN_RE = re.compile(r"[A-Za-z_]\w*[ \t]*$")
_LAUNCH_TOKEN_RE = re.compile(r"<<<|>>>")


def _rewrite_launches(line: str) -> str:
    """Rewrite a CCE kernel launch ``f<<<cfg>>>(args)`` into ``f(cfg)(args)``.

    ``<<<`` and ``>>>`` do not exist in C or C++, and neither tree-sitter nor
    a C preprocessor's lexer can read them.  Folding the launch into a call
    whose callee is a call parses cleanly and keeps the same length, since
    three brackets become one parenthesis on each side plus two spaces.

    Launch syntax only appears in host-side launchers, which the analyzer does
    not inspect for layout anyway, so the approximation costs nothing.
    """
    if "<<<" not in line:
        return line
    result = line
    while True:
        match = _LAUNCH_TOKEN_RE.search(result)
        if match is None or match.group() != "<<<":
            break
        head = result[: match.start()]
        if _LAUNCH_OPEN_RE.search(head) is None:
            break
        close = result.find(">>>", match.end())
        if close < 0:
            break
        # ``<<<`` -> ``(  `` and ``>>>`` -> ``)  ``: same length either side.
        result = (
            head
            + "(  "
            + result[match.end() : close]
            + ")  "
            + result[close + 3 :]
        )
    return result


#: An ``#include`` directive, for attributing inlined content to its site.
_INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]')


def _include_sites(source: str) -> Dict[str, int]:
    """``{header basename: line of the #include}`` for the primary file.

    Content pulled in from a header is reported at the line of the directive
    that pulled it in, which is a line the reader can actually open.  Matching
    on the basename is enough here: a translation unit that includes two
    different headers of the same name would attribute one to the other's
    line, which is a far smaller error than pointing at an unrelated line.
    """
    sites: Dict[str, int] = {}
    for number, line in enumerate(source.split(chr(10)), start=1):
        match = _INCLUDE_RE.match(line)
        if match is not None:
            sites.setdefault(_path_key(match.group(1)), number)
    return sites


#: ``#line N "file"`` as ``pcpp`` emits it.
_LINE_MARKER_RE = re.compile(r'^#line[ \t]+(\d+)(?:[ \t]+"([^"]*)")?[ \t]*$')


def _directive_spans(text: str) -> List[Tuple[int, int]]:
    """Character ranges covered by preprocessor directives.

    A directive runs from its ``#`` to the end of the line, and keeps going
    across every line that ends in a backslash.  The normalizer has no
    business inside one: ``pcpp`` reads directives properly, and a rewrite
    there can only damage them.
    """
    spans: List[Tuple[int, int]] = []
    offset = 0
    lines = text.split(chr(10))
    index = 0
    while index < len(lines):
        line = lines[index]
        start = offset
        offset += len(line) + 1
        if line.lstrip().startswith("#"):
            end = start + len(line)
            while line.endswith(chr(92)) and index + 1 < len(lines):
                index += 1
                line = lines[index]
                end = offset + len(line)
                offset += len(line) + 1
            spans.append((start, end))
        index += 1
    return spans


def _non_code_spans(text: str) -> List[Tuple[int, int]]:
    """Character ranges occupied by comments and string/char literals.

    Every rewrite below has to skip these.  A ``TORCH_CHECK(cond, "... (see
    docs) ...")`` carries an unbalanced parenthesis *inside a string*, and a
    paren matcher that counts it ends the span in the wrong place - cutting a
    string literal in half and turning the rest of the file into one ERROR
    node.  The same applies to the qualifier and DSL matches: these sources
    mention ``[host, aicore]`` in their own header comments.
    """
    spans: List[Tuple[int, int]] = []
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if char == "/" and index + 1 < length:
            nxt = text[index + 1]
            if nxt == "/":
                end = text.find(chr(10), index)
                end = length if end < 0 else end
                spans.append((index, end))
                index = end
                continue
            if nxt == "*":
                end = text.find("*/", index + 2)
                end = length if end < 0 else end + 2
                spans.append((index, end))
                index = end
                continue
        if char in ('"', "'"):
            cursor = index + 1
            while cursor < length:
                if text[cursor] == chr(92):
                    cursor += 2
                    continue
                if text[cursor] == char:
                    cursor += 1
                    break
                if text[cursor] == chr(10):
                    break  # unterminated literal; stop at the line end
                cursor += 1
            spans.append((index, min(cursor, length)))
            index = max(cursor, index + 1)
            continue
        index += 1
    return spans


def _in_spans(spans: List[Tuple[int, int]], offset: int) -> bool:
    lo, hi = 0, len(spans)
    while lo < hi:
        mid = (lo + hi) // 2
        start, end = spans[mid]
        if offset < start:
            hi = mid
        elif offset >= end:
            lo = mid + 1
        else:
            return True
    return False


def _balanced_paren_end(
    text: str, open_at: int, skip: Optional[List[Tuple[int, int]]] = None
) -> int:
    """End of the parenthesis group at ``open_at``, ignoring non-code spans."""
    depth = 0
    for index in range(open_at, len(text)):
        if skip is not None and _in_spans(skip, index):
            continue
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _blank_keeping_newlines(text: str, start: int, end: int) -> str:
    """Replace ``text[start:end]`` with spaces, keeping newlines and splices.

    A backslash at the end of a line is a line *continuation*, and erasing one
    inside a multi-line ``#define`` truncates the macro: the rest of its body
    stops being a body and becomes file-scope statements, which is a far worse
    failure than the construct being blanked in the first place.
    """
    chunk = list(text[start:end])
    for index, char in enumerate(chunk):
        if char == chr(10):
            continue
        if char == chr(92) and index + 1 < len(chunk) and chunk[index + 1] == chr(10):
            continue  # a line splice: keep it
        chunk[index] = " "
    return text[:start] + "".join(chunk) + text[end:]


def normalize_cce_syntax(source: str) -> str:
    """Erase CCE constructs the C lexer cannot read, byte for byte.

    Three rewrites, each the same length as what it replaces so that every
    offset, line and column still addresses the original file:

    * a **location qualifier** - ``__forceinline__ [host, aicore] void f()`` -
      becomes spaces.  It is told apart from an array subscript
      (``buf[host]``), a C++ attribute (``[[nodiscard]]``) and a lambda by
      position: a qualifier is never glued to the preceding token, never
      doubled, and is always followed by something that starts a type.  A list
      of two or more names skips the first test, a comma inside a subscript
      not being valid C++.
    * a **parenthesis-free condition** - ``if ASCEND_IS_AIV {`` - gains its
      parentheses by giving up the two spaces around the condition, so
      ``if ASCEND_IS_AIV {`` and ``if(ASCEND_IS_AIV){`` are the same length.
    * a **file-scope DSL block** - ``ASCENDC_TPL_SEL(...)`` spanning lines -
      becomes spaces.  Only the multi-line form: a one-line ``FOO(a, b);``
      is ordinary C, and a ``)`` followed by ``{`` is a definition whose body
      is left alone.

    A fourth rewrite runs first: a **kernel launch** ``f<<<cfg>>>(args)``
    becomes ``f(  cfg)  (args)``, since ``<<<`` is not a token in any C
    grammar (see :func:`_rewrite_launches`).
    """
    # -- kernel launches -----------------------------------------------------
    text = source
    if "<<<" in text:
        text = "\n".join(_rewrite_launches(line) for line in text.split("\n"))

    # -- location qualifiers -------------------------------------------------
    skip = _non_code_spans(text)
    spans: List[Tuple[int, int]] = []
    for match in _CCE_BRACKET_RE.finditer(text):
        start, end = match.start(), match.end()
        if _in_spans(skip, start):
            continue  # a mention in a comment or a string literal
        if start > 0 and text[start - 1] == "[":
            continue
        if end < len(text) and text[end] == "]":
            continue
        if "," not in match.group(0):
            if start > 0 and text[start - 1] in _SUBSCRIPT_LEAD:
                continue
        probe = end
        while probe < len(text) and text[probe] in " \t\r\n":
            probe += 1
        if probe >= len(text) or text[probe] not in _TYPE_START:
            continue
        spans.append((start, end))
    for start, end in reversed(spans):
        text = _blank_keeping_newlines(text, start, end)

    # -- parenthesis-free conditions ----------------------------------------
    def _parenthesise(match: re.Match) -> str:
        keyword, condition = match.group(1), match.group(2)
        # Same length: two spaces traded for two parentheses.
        replacement = f"{keyword}({condition}){{"
        return replacement if len(replacement) == len(match.group(0)) else match.group(0)

    text = _BARE_COND_RE.sub(_parenthesise, text)

    # -- file-scope DSL blocks ----------------------------------------------
    # Recomputed: the rewrites above changed the text.  Directives are added
    # to the skip set, so nothing is rewritten inside a ``#define`` body.
    skip = sorted(_non_code_spans(text) + _directive_spans(text))
    spans = []
    for match in _DSL_CALL_RE.finditer(text):
        name_start = match.start(1)
        if _in_spans(skip, name_start):
            continue
        open_paren = match.end() - 1
        close = _balanced_paren_end(text, open_paren, skip)
        if close < 0 or chr(10) not in text[name_start:close]:
            continue
        probe = close
        while probe < len(text) and text[probe] in " \t\r\n":
            probe += 1
        if probe < len(text) and text[probe] == "{":
            continue  # a definition; leave its body intact
        end = probe + 1 if probe < len(text) and text[probe] == ";" else close
        spans.append((name_start, end))
    for start, end in reversed(spans):
        text = _blank_keeping_newlines(text, start, end)

    return text


class AscendCPreprocessor(Preprocessor):
    """``pcpp`` configured for Ascend C kernels.

    The hooks differ from a plain ``pcpp`` subclass in four ways, each for a
    reason the docstring on the hook gives.
    """

    def __init__(self, include_dirs: Tuple[Path, ...] = ()) -> None:
        super().__init__()
        # Keep the markers: they are the only way back to original line
        # numbers once directives have been deleted.
        self.line_directive = "#line"
        self.auto_pragma_once_enabled = True
        # These headers are UTF-8 and full of Chinese comments.  ``pcpp``
        # otherwise opens an include with the locale encoding - cp1252 on
        # Windows - and the first non-ASCII byte aborts the whole translation
        # unit.
        self.assume_encoding = "utf-8"
        #: Diagnostics ``pcpp`` would otherwise print to stderr.
        self.errors: List[str] = []
        #: One entry per open conditional, ``True`` when it was passed through.
        #: A passed-through ``#ifdef`` has to carry its ``#else`` and
        #: ``#endif`` with it, or the output holds an unbalanced conditional
        #: and ``pcpp`` reports a misplaced directive.
        self._conditional_passthru: List[bool] = []
        for directory in include_dirs:
            self.add_path(str(directory))
        for name, body in SEEDED_MACROS.items():
            self.define(f"{name} {body}")
        for name in ERASED_DECORATION_MACROS:
            self.define(name)  # empty body: expands to nothing

    # -- hooks --------------------------------------------------------------

    def on_include_not_found(
        self, is_malformed, is_system_include, curdir, includepath
    ):
        """Drop an unresolvable include instead of aborting.

        Most includes in this fleet are vendor headers that are not in the
        tree at all - ``catlass/``, ``tla/``, ``kernel_operator.h``.  The
        default raises ``OutputDirective`` *and* reports an error; this
        removes the directive silently, because a missing CANN header is the
        normal case here rather than a fault in the file being analyzed.
        """
        raise OutputDirective(Action.IgnoreAndRemove)

    def on_unknown_macro_in_defined_expr(self, tok):
        """An undefined macro in ``defined(...)`` is simply not defined."""
        return False

    def on_unknown_macro_in_expr(self, ident):
        """An undefined macro in ``#if`` arithmetic evaluates to 0, as in C."""
        return super().on_unknown_macro_in_expr(ident)

    def on_comment(self, tok):
        """Keep comments verbatim.

        Returning ``True`` passes the token through unchanged, which preserves
        every byte and newline inside it.  ``pcpp`` cannot be asked to replace
        a comment with text: returning ``False`` collapses a whole multi-line
        block comment into a *single* space, which would destroy the line
        alignment this module exists to maintain.  Comments are trivia to
        tree-sitter, so passing them through costs nothing.
        """
        return True

    def on_directive_handle(self, directive, toks, ifpassthru, precedingtoks):
        """Pass through any conditional this preprocessor cannot honestly decide.

        Two kinds are passed through rather than evaluated.

        ``#ifdef __DAV_C220_CUBE__`` is not a configuration choice to be
        resolved - it is the structure the AST visitor reads to tell Cube code
        from Vector code.  Evaluating it would delete one arm and merge the
        two event spaces.

        A conditional on a macro nothing defines - ``#ifdef SOME_FEATURE`` -
        is undecidable here, and picking the else arm would quietly stop
        analyzing the other one.  This analyzer is looking for races and
        overflows, so unexamined code is the expensive failure; the visitor is
        built to walk *both* arms and tag what it finds, which is only
        possible if the conditional survives into the parse tree.

        Everything decidable - an include guard, ``__cplusplus``, a macro the
        file or this frontend defines - is evaluated normally.
        """
        name = directive.value
        if name in ("ifdef", "ifndef", "if"):
            passthru = any(
                tok.value in CORE_GUARD_MACROS for tok in toks
            ) or self._has_undecidable_macro(toks)
            self._conditional_passthru.append(passthru)
            if passthru:
                raise OutputDirective(Action.IgnoreAndPassThrough)
        elif name in ("else", "elif"):
            if self._conditional_passthru and self._conditional_passthru[-1]:
                raise OutputDirective(Action.IgnoreAndPassThrough)
        elif name == "endif":
            if self._conditional_passthru and self._conditional_passthru.pop():
                raise OutputDirective(Action.IgnoreAndPassThrough)
        return super().on_directive_handle(directive, toks, ifpassthru, precedingtoks)

    def _has_undecidable_macro(self, toks) -> bool:
        """``True`` when a condition names an identifier nothing defines.

        ``defined`` itself, and the operators and literals around it, are not
        identifiers for this purpose.  An include guard on its first reading
        is undefined too, which is correct: passing it through means the file
        body is parsed once, and repeat inclusion is handled separately by
        ``pcpp``'s include-guard detection.
        """
        for tok in toks:
            value = getattr(tok, "value", "")
            if not value or not (value[0].isalpha() or value[0] == "_"):
                continue
            if value in ("defined", "true", "false"):
                continue
            if value not in self.macros:
                return True
        return False

    def on_file_open(self, is_system_include, includepath):
        """Open an include, normalizing the CCE syntax it contains.

        The normalizer has to run on *every* file in the translation unit, not
        just the primary one.  A tiling-key DSL block or a location qualifier
        inside an included header reaches the lexer exactly as it would in the
        main file, and the tiling-key blocks in particular hide a ``//``
        comment inside their argument list, which swallows the closing
        parenthesis and leaves the call unterminated.

        Reading through a buffer also pins the encoding: these headers are
        UTF-8, and decoding one as cp1252 aborts the whole unit.
        """
        text = io.open(includepath, "r", encoding="utf-8", errors="replace").read()
        if text.startswith("﻿"):
            text = text[1:]
        return io.StringIO(normalize_cce_syntax(text))

    def on_error(self, file, line, msg):
        """Collect diagnostics rather than printing them."""
        self.errors.append(f"{file}:{line}: {msg}")


@dataclass
class PreprocessResult:
    """What the frontend hands to the qualifier rewrite."""

    #: Preprocessed text, with ``#line`` markers consumed and blanked.
    text: str
    #: For every 1-based line of :attr:`text`, the 1-based original line.
    line_origins: Tuple[int, ...]
    #: Object-like macro bodies, for the constant folder.
    object_macros: Dict[str, str] = field(default_factory=dict)
    #: Function-like macro names, for reporting only; ``pcpp`` has already
    #: expanded every invocation.
    function_macros: Tuple[str, ...] = ()
    stats: Dict[str, int] = field(default_factory=dict)
    errors: Tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.stats)

    def origin_line(self, line: int) -> int:
        if not self.line_origins:
            return line
        if 1 <= line <= len(self.line_origins):
            return self.line_origins[line - 1]
        return self.line_origins[-1] if self.line_origins else line


def _consume_line_markers(
    text: str,
    primary: str,
    total_source_lines: int,
    include_sites: Optional[Dict[str, int]] = None,
) -> Tuple[str, Tuple[int, ...]]:
    """Strip ``#line`` markers, returning the text and its origin map.

    A marker says which file and line the *next* output line came from; lines
    after it advance one for one until the next marker.  The marker line
    itself is dropped, so the map covers exactly the lines that remain.

    Only lines from *primary* - the file being analyzed - map to their own
    line number.  Content pulled in from an ``#include`` maps instead to the
    line of the include that pulled it in, because a diagnostic has to point
    at a line that exists in the file the report names.  Reporting an included
    header's line number against the primary file would send the reader to an
    unrelated line, or past the end of the file.
    """
    out: List[str] = []
    origins: List[int] = []
    current = 1
    in_primary = True
    #: Where the content being emitted is attributed when it is not the
    #: primary file: the line of the ``#include`` that pulled it in.
    include_site = 1
    primary_key = _path_key(primary)
    sites = include_sites or {}

    for line in text.split("\n"):
        match = _LINE_MARKER_RE.match(line.strip())
        if match is not None:
            current = int(match.group(1))
            name = match.group(2)
            if name is not None:
                in_primary = _path_key(name) == primary_key
                if not in_primary:
                    # Attribute the header's content to its #include line.
                    site = sites.get(_path_key(name))
                    if site is not None:
                        include_site = site
            if in_primary:
                include_site = max(1, min(current, total_source_lines or current))
            continue
        out.append(line)
        if in_primary:
            origin = min(current, total_source_lines) if total_source_lines else current
            include_site = origin
        else:
            origin = include_site
        origins.append(max(1, origin))
        current += 1

    if not out:
        out, origins = [""], [1]
    return "\n".join(out), tuple(origins)


def _path_key(path: str) -> str:
    """Compare two spellings of one path without touching the filesystem."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1].lower()


def _macro_bodies(preprocessor: Preprocessor) -> Tuple[Dict[str, str], Tuple[str, ...]]:
    """Object-like bodies and function-like names from a finished run."""
    objects: Dict[str, str] = {}
    functions: List[str] = []
    for name, macro in getattr(preprocessor, "macros", {}).items():
        arglist = getattr(macro, "arglist", None)
        if arglist:
            functions.append(str(name))
            continue
        tokens = getattr(macro, "value", ()) or ()
        body = "".join(getattr(tok, "value", "") for tok in tokens).strip()
        if body:
            objects[str(name)] = body
    return objects, tuple(sorted(functions))


def preprocess_source(
    path: str, source: str, include_dirs: Tuple[Path, ...] = ()
) -> PreprocessResult:
    """Normalize, preprocess and re-anchor one translation unit."""
    normalized = normalize_cce_syntax(source)

    directories: List[Path] = list(include_dirs)
    if not directories:
        try:
            directories.append(Path(path).resolve().parent)
        except (OSError, ValueError):
            pass

    engine = AscendCPreprocessor(tuple(directories))
    try:
        engine.parse(normalized, source=path)
        sink = io.StringIO()
        engine.write(sink)
        expanded = sink.getvalue()
    except Exception as exc:  # a frontend fault must not lose the file
        return PreprocessResult(
            text=normalized,
            line_origins=(),
            stats={},
            errors=(f"preprocessor failed, analyzing unexpanded source: {exc}",),
        )

    total = source.count(chr(10)) + 1
    text, origins = _consume_line_markers(
        expanded, path, total, _include_sites(source)
    )
    objects, functions = _macro_bodies(engine)

    stats: Dict[str, int] = {
        "object_macros": len(objects),
        "function_macros": len(functions),
        "input_lines": total,
        "output_lines": text.count("\n") + 1,
    }
    if normalized != source:
        stats["cce_constructs_normalized"] = sum(
            1 for a, b in zip(source, normalized) if a != b
        )

    return PreprocessResult(
        text=text,
        line_origins=origins,
        object_macros=objects,
        function_macros=functions,
        stats=stats,
        errors=tuple(engine.errors),
    )
