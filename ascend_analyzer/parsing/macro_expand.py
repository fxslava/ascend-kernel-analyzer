"""A small C preprocessor front-end: local includes, ``#define`` handling and
function-like macro expansion.

Why this exists
---------------

Ascend C kernels are written as *stage macros*: the pipeline stages of a
double-buffered loop become ``#define A_MAD(t, p) do { ... } while (0)`` blocks
of ten-plus lines, invoked once per stage.  ``tree-sitter-cpp`` cannot digest a
multi-line function-like macro definition inside a function body - its
``preproc_function_def`` node terminates early and the remaining body lines
leak into the enclosing function as stray statements.  Every construct the
analyzer cares about (``SetFlag``/``WaitFlag``, ``mad_mx``, the
``(__ca__ T *)(uintptr_t)p.GetPhyAddr()`` casts) then lands inside ``ERROR``
nodes and disappears from the analysis (``AKA9001``).

Rather than tolerate the leak, this module expands the macros *before*
tree-sitter ever sees the text:

* ``#include "local.h"`` is inlined when the file exists next to the source
  (system and CANN headers such as ``kernel_operator.h`` stay untouched), so
  the tile-geometry ``#define`` chains a kernel depends on become foldable;
* every ``#define``/``#undef`` directive line is blanked, removing the
  construct tree-sitter chokes on;
* function-like macro invocations are expanded textually with argument
  substitution and rescanning, so ``A_MAD(t, p)`` becomes the statements of the
  stage at the invocation site and ``EV(p)`` becomes
  ``((p) ? EVENT_ID1 : EVENT_ID0)``, which the constant folder can evaluate
  once ``p`` is known.

Diagnostics must still point at the *original* file.  The expander therefore
tracks, for every line of expanded output, the original line it came from:
ordinary lines map to themselves, lines produced by an expansion map to the
invocation site, lines spliced from an included header map to the ``#include``
line.  :class:`MacroExpansion.line_origins` hands that map to the AST visitor,
which translates every reported location back through it.

The expander is deliberately minimal: no conditional evaluation (``#if`` arms
are left for tree-sitter), no stringification, no variadic macros.  When a
kernel uses one of those, the unexpanded construct degrades exactly as before
this module existed.

The ``##`` token-paste operator *is* supported, because production kernels do
use it: the CANN-style flag wrappers

.. code-block:: c

   #define SET_FLAG(trigger, waiter, e) \\
       AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))

build the ``HardEvent`` enumerator by pasting their arguments.  Without paste
support the invocation cannot be expanded, lands in an ``ERROR`` node, and the
flag operation never reaches the trace - so the kernel's whole synchronisation
structure silently reads as clean.  Paste runs after argument substitution and
before rescanning, per the C rules.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["MacroExpansion", "FunctionMacro", "expand_macros", "non_code_spans_str"]


#: Maximum recursive inclusion depth.
_MAX_INCLUDE_DEPTH = 8
#: Maximum macro-expansion nesting (self-referential macro guard).
_MAX_EXPANSION_DEPTH = 24

_DIRECTIVE_RE = re.compile(r"^\s*#\s*(?P<kind>[A-Za-z_]+)(?P<rest>.*)$")
_DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+(?P<name>[A-Za-z_]\w*)(?P<params>\([^)\n]*\))?(?P<body>.*)$",
    re.DOTALL,
)
_INCLUDE_QUOTED_RE = re.compile(r'^\s*#\s*include\s*"(?P<path>[^"]+)"\s*$')
_IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")
#: Characters that, when they precede an identifier, mean it is not a bare
#: macro invocation: member access, namespace qualification, directive.
_NON_CALL_PRECEDERS = frozenset({".", ":", ">", "#"})

_LAUNCH_OPEN_RE = re.compile(r"[A-Za-z_]\w*\s*$")
_LAUNCH_TOKEN_RE = re.compile(r"<<<|>>>")
#: The ``##`` token-paste operator with the horizontal whitespace it absorbs.
#: Newlines are deliberately excluded: a replacement must not change the line
#: count, or every origin mapping downstream of it would shift.
_PASTE_RE = re.compile(r"[ \t]*##[ \t]*")


def non_code_spans_str(text: str) -> List[Tuple[int, int]]:
    """Character offsets of comments and string/char literals in *text*.

    Character-level mirror of :func:`ascend_analyzer.parsing.preprocess
    .non_code_spans`; the expander works on Python strings, and byte offsets
    would desynchronise on any non-ASCII comment.
    """
    spans: List[Tuple[int, int]] = []
    index, length = 0, len(text)
    while index < length:
        ch = text[index]
        if ch == "/" and index + 1 < length:
            nxt = text[index + 1]
            if nxt == "/":
                end = text.find("\n", index)
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
        if ch in "\"'":
            quote = ch
            cursor = index + 1
            while cursor < length:
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if text[cursor] == quote:
                    cursor += 1
                    break
                if text[cursor] == "\n":
                    break  # unterminated literal; stop at the line end
                cursor += 1
            spans.append((index, min(cursor, length)))
            index = max(cursor, index + 1)
            continue
        index += 1
    return spans


def _blank_comments(text: str) -> str:
    """Replace comment interiors with spaces, keeping length and structure."""
    out = list(text)
    for start, end in non_code_spans_str(text):
        for index in range(start, end):
            if out[index] != "\n":
                out[index] = " "
    return "".join(out)


@dataclass(frozen=True)
class FunctionMacro:
    """One function-like ``#define`` with its parameter list and body."""

    name: str
    params: Tuple[str, ...]
    body: str
    #: 1-based line of the ``#define`` in the merged text.
    define_line: int
    #: ``True`` for ``#define M(...)`` / ``#define M(a, ...)``.  Trailing
    #: arguments are then bound to ``__VA_ARGS__`` rather than to a parameter.
    variadic: bool = False
    #: 1-based line of the ``#undef``, when the macro was retired.
    undef_line: Optional[int] = None

    def is_active_at(self, line: int) -> bool:
        if line <= self.define_line:
            return False
        return self.undef_line is None or line < self.undef_line


@dataclass
class MacroExpansion:
    """The output of :func:`expand_macros`."""

    #: Expanded text: directives blanked, invocations inlined.
    text: str
    #: For every 1-based line of *text*, the original 1-based line it came from.
    line_origins: Tuple[int, ...]
    #: Object-like macros in source order (name -> body text, not yet folded).
    object_macros: Dict[str, str] = field(default_factory=dict)
    #: Function-like macros by name.
    function_macros: Dict[str, FunctionMacro] = field(default_factory=dict)
    #: Counters surfaced in the report header.
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        """``True`` when anything was rewritten (expansion is not identity)."""
        return bool(
            self.stats.get("defines_blanked")
            or self.stats.get("expansions")
            or self.stats.get("includes_inlined")
            or self.stats.get("builtin_objects_substituted")
        )

    def origin_line(self, line: int) -> int:
        """Translate a 1-based line of :attr:`text` to the original file."""
        if not self.line_origins:
            return line
        if 1 <= line <= len(self.line_origins):
            return self.line_origins[line - 1]
        return self.line_origins[-1] if self.line_origins else line


# ---------------------------------------------------------------------------
# Stage 1: local include inlining
# ---------------------------------------------------------------------------


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _inline_includes(
    text: str,
    base_dir: Path,
    depth: int,
    included: Set[Path],
    stats: Dict[str, int],
) -> List[Tuple[str, int]]:
    """Replace ``#include "file"`` lines with the referenced local header.

    Returns ``(line_text, origin_line)`` pairs where *origin_line* is a 1-based
    line in *text*; spliced header lines all map to their ``#include`` line so
    every origin stays a coordinate of the root file.  Includes that cannot be
    resolved - CANN and system headers - are kept verbatim for tree-sitter.
    """
    out: List[Tuple[str, int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = _INCLUDE_QUOTED_RE.match(line)
        if match is None or depth >= _MAX_INCLUDE_DEPTH:
            out.append((line, number))
            continue
        target = (base_dir / match.group("path")).resolve()
        content = _read(target) if target not in included else None
        if content is None:
            out.append((line, number))
            continue
        included.add(target)
        stats["includes_inlined"] = stats.get("includes_inlined", 0) + 1
        # The header's own nested includes resolve relative to *its* directory.
        for inner_line, _origin in _inline_includes(
            content, target.parent, depth + 1, included, stats
        ):
            out.append((inner_line, number))
    if not out:
        out.append(("", 1))
    return out


# ---------------------------------------------------------------------------
# Stage 2: directive collection
# ---------------------------------------------------------------------------


def _line_continues(text: str, start: int, end: int) -> bool:
    """``True`` when the physical line ``[start, end)`` ends in a code ``\\``."""
    index = end - 1
    while index >= start:
        ch = text[index]
        if ch in " \t\r":
            index -= 1
            continue
        return ch == "\\"
    return False


#: Object-like macros from CANN headers that the expander substitutes itself.
#:
#: These are not inlined (vendor headers are left alone), yet they are
#: *syntactically load-bearing*: kernels write ``if ASCEND_IS_AIV {`` with no
#: parentheses of their own, because the macro supplies them.  Left alone the
#: ``if`` is not valid C++, the statement lands in an ``ERROR`` node, and the
#: whole core-guarded region - every flag operation in it - disappears from the
#: trace.  Each expands to a parenthesised sentinel identifier the AST visitor
#: recognises as a core predicate; it stays opaque to the constant folder, so
#: neither arm is pruned.
BUILTIN_OBJECT_MACROS: Dict[str, str] = {
    "ASCEND_IS_AIC": "(__ascend_core_is_aic)",
    "ASCEND_IS_AIV": "(__ascend_core_is_aiv)",
}


def _substitute_builtin_objects(
    text: str, defined: Dict[str, str]
) -> Tuple[str, int]:
    """Replace the CANN core predicates with parenthesised sentinels.

    A translation unit that defines one of these names itself wins: its own
    body was already recorded, and overriding it here would analyze code the
    compiler never sees.
    """
    pending = {
        name: body
        for name, body in BUILTIN_OBJECT_MACROS.items()
        if name not in defined and name in text
    }
    if not pending:
        return text, 0
    spans = non_code_spans_str(text)
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, pending)) + r")\b")
    pieces: List[str] = []
    cursor = 0
    count = 0
    for match in pattern.finditer(text):
        if _in_span(spans, match.start()):
            continue  # a mention in a comment or string literal
        pieces.append(text[cursor : match.start()])
        pieces.append(pending[match.group(1)])
        cursor = match.end()
        count += 1
    pieces.append(text[cursor:])
    return "".join(pieces), count


def _collect_directives(
    lines: List[str]
) -> Tuple[Dict[str, str], Dict[str, FunctionMacro], Set[int]]:
    """Record every ``#define``/``#undef`` and mark their lines for blanking.

    Line numbers in the returned structures are 1-based positions into *lines*.
    """
    text = "\n".join(lines)

    objects: Dict[str, str] = {}
    functions: Dict[str, FunctionMacro] = {}
    blank: Set[int] = set()

    # Running character offset of each physical line.
    offsets: List[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1

    index = 0
    count = len(lines)
    while index < count:
        line_no = index + 1
        directive = _DIRECTIVE_RE.match(lines[index])
        if directive is None:
            index += 1
            continue
        # Gather the full logical directive (with backslash continuations).
        logical: List[str] = [lines[index]]
        physical: List[int] = [index]
        while index < count - 1:
            start = offsets[physical[-1]]
            end = start + len(lines[physical[-1]])
            if not _line_continues(text, start, end):
                break
            index += 1
            logical.append(lines[index])
            physical.append(index)
        joined = _blank_comments("\n".join(logical)).replace("\\", " ")
        joined = " ".join(joined.split()) if len(logical) > 1 else joined
        kind = directive.group("kind")
        if kind == "define":
            match = _DEFINE_RE.match(joined)
            if match is not None:
                name = match.group("name")
                params_text = match.group("params")
                body = " ".join(match.group("body").split())
                if params_text:
                    params = tuple(
                        p.strip()
                        for p in params_text.strip("()").split(",")
                        if p.strip()
                    )
                    # ``#define TLA_REQUIRES(...)`` names no parameter; its
                    # arguments arrive through ``__VA_ARGS__``.  Treating the
                    # ``...`` as an ordinary parameter left ``__VA_ARGS__``
                    # standing in the body, and the half-expanded
                    # ``std::enable_if<(__VA_ARGS__)>`` took the whole
                    # enclosing header down with it.
                    variadic = bool(params) and params[-1] == "..."
                    if variadic:
                        params = params[:-1]
                    functions[name] = FunctionMacro(
                        name=name,
                        params=params,
                        body=body,
                        define_line=line_no,
                        variadic=variadic,
                    )
                else:
                    objects[name] = body
                blank.update(n + 1 for n in physical)
        elif kind == "undef":
            name = directive.group("rest").strip()
            macro = functions.get(name)
            if macro is not None:
                functions[name] = FunctionMacro(
                    name=macro.name,
                    params=macro.params,
                    body=macro.body,
                    define_line=macro.define_line,
                    variadic=macro.variadic,
                    undef_line=line_no,
                )
            blank.add(line_no)
        index += 1

    return objects, functions, blank


# ---------------------------------------------------------------------------
# Stage 3: invocation expansion
# ---------------------------------------------------------------------------


def _rewrite_launches(line: str) -> str:
    """Rewrite CCE kernel launches ``f<<<cfg>>>(args)`` into plain calls.

    ``<<<`` / ``>>>`` do not exist in the C++ grammar; tree-sitter derails on
    them.  ``f<<<cfg>>>(args)`` becomes ``f(cfg)(args)`` - a call whose callee
    is a call - which parses cleanly.  Only lines that mention the launch
    brackets are touched, and launch syntax never appears in kernel device
    code, so the approximation only ever affects host-side launchers the
    analyzer does not otherwise inspect.
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
        result = (
            head + "(" + result[match.end() : close] + ")"
            + result[close + 3 :]
        )
    return result


def _split_arguments(text: str) -> List[str]:
    """Split an argument list on top-level commas."""
    args: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(current))
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail or args:
        args.append(tail)
    return [a.strip() for a in args]


#: Punctuation that can make up an operator token passed as a macro argument.
_OPERATOR_CHARS = frozenset("+-*/%^&|~!<>=,[]")

#: A single token - a plain or qualified identifier, or an integer literal.
#: Parentheses around one are never needed and are sometimes fatal: a type
#: argument comes out as ``(SCFABlockCube)<TEMPLATE_ARGS>``, and a macro name
#: as ``(GEN_TRAIT_TYPE)(Q)``, neither of which is C++.
_SINGLE_TOKEN_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*|[0-9]+[uUlL]*)$")


def _needs_parentheses(arg: str) -> bool:
    """``True`` when an argument is a compound expression needing protection.

    Precedence protection matters only for an argument built from more than
    one token.  Wrapping a lone identifier changes nothing where it is an
    expression and breaks it where it is a type or a macro name.
    """
    stripped = arg.strip()
    if not stripped:
        return False
    return not _SINGLE_TOKEN_RE.match(stripped)


def _is_operator_argument(arg: str) -> bool:
    """``True`` when an argument is an operator token rather than an expression.

    ``TLA_UNARY_OP(+)`` passes the ``+`` itself, to be planted both inside
    ``C<(OP t)>`` and after the ``operator`` keyword.  Wrapping it in the
    usual protective parentheses yields ``operator (+)``, which is not C++ at
    all, and that error swallows every declaration after it.  Parentheses
    guard operator precedence *within an expression*; an argument that is not
    an expression has no precedence to guard.
    """
    stripped = arg.strip()
    if not stripped:
        return False
    return all(ch in _OPERATOR_CHARS for ch in stripped)


def _as_string_literal(arg: str) -> str:
    """Render a macro argument as the string literal ``#arg`` produces."""
    escaped = arg.strip().replace(chr(92), chr(92) * 2).replace('"', chr(92) + '"')
    return '"' + escaped + '"'


def _substitute_param(text: str, pattern: re.Pattern, arg: str) -> str:
    """Replace every occurrence of one parameter, honouring ``##`` adjacency.

    An occurrence that is an operand of a token paste is substituted *bare*.
    Pasting joins the spelling of the adjacent tokens, so the protective
    parentheses used everywhere else would make the result ill-formed:
    ``PIPE_##pipe`` has to become ``PIPE_V``, never ``PIPE_##(V)``.
    """
    bare_everywhere = _is_operator_argument(arg) or not _needs_parentheses(arg)
    pieces: List[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        before = text[: match.start()].rstrip(" \t")
        after = text[match.end() :].lstrip(" \t")
        pasted = before.endswith("##") or after.startswith("##")
        # ``#param`` stringifies.  Left unexpanded the ``#`` reads as the start
        # of a preprocessor directive in the middle of a statement, and
        # ``GetOpApiFuncAddr(#aclCreateTensor)`` took thousands of lines of
        # ``torch_binding.cpp`` down with it.
        stringified = not pasted and before.endswith("#")
        if stringified:
            pieces.append(text[cursor : len(before) - 1])
            pieces.append(_as_string_literal(arg))
        else:
            pieces.append(text[cursor : match.start()])
            pieces.append(arg if (pasted or bare_everywhere) else f"({arg})")
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces)


def _apply_pastes(text: str) -> str:
    """Execute the ``##`` operators in an already-substituted macro body.

    Paste runs after substitution and before rescanning, as C requires, so a
    concatenation that names another macro is expanded on the rescan.
    """
    if "##" not in text:
        return text
    spans = non_code_spans_str(text)
    pieces: List[str] = []
    cursor = 0
    for match in _PASTE_RE.finditer(text):
        if _in_span(spans, match.start()):
            continue  # '##' inside a comment or string literal is not an operator
        pieces.append(text[cursor : match.start()])
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces)


def _substitute(
    body: str,
    params: Sequence[str],
    args: Sequence[str],
    variadic: bool = False,
) -> str:
    """Replace parameter occurrences with arguments, then paste.

    Arguments are parenthesised to keep operator precedence, except where the
    occurrence is an operand of ``##`` (see :func:`_substitute_param`).
    """
    out = body
    for param, arg in zip(params, args):
        if not param:
            continue
        out = _substitute_param(out, re.compile(rf"\b{re.escape(param)}\b"), arg)
    if variadic:
        # The body decides whether the pack needs parentheses, as
        # ``std::enable_if<(__VA_ARGS__)>`` does, so it goes in bare.
        rest = ", ".join(a for a in args[len(params):] if a)
        out = re.sub(
            r"\b__VA_ARGS__\b", rest.replace(chr(92), chr(92) * 2), out
        )
    return _apply_pastes(out)


def _find_matching_paren(
    text: str, open_index: int, spans: Sequence[Tuple[int, int]]
) -> Optional[int]:
    depth = 0
    index = open_index
    length = len(text)
    while index < length:
        if any(s <= index < e for s, e in spans):
            index += 1
            continue
        ch = text[index]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _in_span(spans: Sequence[Tuple[int, int]], position: int) -> bool:
    return any(s <= position < e for s, e in spans)


def _prev_code_char(
    text: str, spans: Sequence[Tuple[int, int]], position: int
) -> str:
    index = position - 1
    while index >= 0:
        if any(s <= index < e for s, e in spans):
            index -= 1
            continue
        ch = text[index]
        if not ch.isspace():
            return ch
        index -= 1
    return ""


def _find_open_paren(text: str, from_index: int) -> Optional[int]:
    """The index of the ``(`` starting a call, skipping same-line whitespace."""
    index = from_index
    length = len(text)
    while index < length:
        ch = text[index]
        if ch in " \t":
            index += 1
            continue
        return index if ch == "(" else None
    return None


def _expand_inline(
    text: str,
    functions: Dict[str, FunctionMacro],
    active: Tuple[str, ...],
    activation_line: int,
    stats: Dict[str, int],
) -> str:
    """Recursively expand macro invocations inside already-substituted text.

    Everything in *text* originated at *activation_line* of the merged text
    (the line carrying the outermost invocation), so macro-activation checks
    run against that line.
    """
    if len(active) >= _MAX_EXPANSION_DEPTH:
        return text
    spans = non_code_spans_str(text)
    pieces: List[str] = []
    cursor = 0
    for match in _IDENT_RE.finditer(text):
        name = match.group()
        if name not in functions or name in active:
            continue
        if _in_span(spans, match.start()):
            continue
        if _prev_code_char(text, spans, match.start()) in _NON_CALL_PRECEDERS:
            continue
        probe = _find_open_paren(text, match.end())
        if probe is None or _in_span(spans, probe):
            continue
        macro = functions[name]
        if not macro.is_active_at(activation_line):
            continue
        close = _find_matching_paren(text, probe, spans)
        if close is None:
            continue
        args = _split_arguments(text[probe + 1 : close])
        if len(args) < len(macro.params):
            args += [""] * (len(macro.params) - len(args))
        substituted = _substitute(
            macro.body, macro.params, args, variadic=macro.variadic
        )
        pieces.append(text[cursor : match.start()])
        pieces.append(
            _expand_inline(substituted, functions, active + (name,),
                           activation_line, stats)
        )
        cursor = close + 1
        stats["expansions"] = stats.get("expansions", 0) + 1
    pieces.append(text[cursor:])
    return "".join(pieces)


def _line_starts(text: str) -> List[int]:
    starts = [0]
    for index, ch in enumerate(text):
        if ch == "\n":
            starts.append(index + 1)
    return starts


class _OutputBuilder:
    """Accumulates output lines, each tagged with its original line number."""

    def __init__(self, first_origin: int) -> None:
        self.lines: List[str] = []
        self.origins: List[int] = []
        self._buffer: List[str] = []
        self._origin: Optional[int] = None
        self._first_origin = first_origin

    def _close(self, origin: Optional[int]) -> None:
        self.lines.append("".join(self._buffer))
        self.origins.append(origin if origin is not None else self._first_origin)
        self._buffer.clear()
        self._origin = None

    def emit_source(self, text: str, start: int, end: int,
                    origin_of) -> None:
        """Copy ``text[start:end]`` verbatim, tracking line origins.

        ``origin_of(offset)`` maps a character offset of the source text to
        the original line it came from.
        """
        for index in range(start, end):
            ch = text[index]
            if ch == "\n":
                # An empty line never set its origin; the newline itself
                # still sits on that line, so ask the map directly.
                self._close(self._origin if self._origin is not None
                            else origin_of(index))
            else:
                if self._origin is None:
                    self._origin = origin_of(index)
                self._buffer.append(ch)

    def emit_replacement(self, replacement: str, origin: int) -> None:
        """Append expansion output; every line it produces maps to *origin*."""
        parts = replacement.split("\n")
        if self._origin is None:
            self._origin = origin
        self._buffer.append(parts[0])
        for part in parts[1:]:
            self._close(origin)
            self._origin = origin
            self._buffer.append(part)

    def finish(self) -> Tuple[str, Tuple[int, ...]]:
        if self._buffer:
            self._close(self._origin)
        return "\n".join(self.lines), tuple(self.origins)


def expand_macros(source: str, base_dir: Optional[Path] = None) -> MacroExpansion:
    """Prepare *source* for tree-sitter parsing.

    ``base_dir`` anchors ``#include "..."`` resolution; when ``None`` includes
    are left untouched.
    """
    stats: Dict[str, int] = {"defines_blanked": 0, "expansions": 0,
                             "includes_inlined": 0}

    # ---- stage 1: local includes ------------------------------------------
    if base_dir is not None:
        merged = _inline_includes(source, base_dir, 0, set(), stats)
    else:
        merged = [
            (line, number)
            for number, line in enumerate(source.splitlines(), start=1)
        ] or [("", 1)]
    lines = [pair[0] for pair in merged]
    origins = [pair[1] for pair in merged]

    # CCE kernel launches (``f<<<cfg>>>(args)``) are host-side only and break
    # the C++ grammar; fold them into plain nested calls.
    lines = [_rewrite_launches(line) for line in lines]

    # ---- stage 2: directives ----------------------------------------------
    objects, functions, blank = _collect_directives(lines)
    stats["defines_blanked"] = len(blank)
    for number in blank:
        lines[number - 1] = " " * len(lines[number - 1])

    text = "\n".join(lines)
    text, substituted = _substitute_builtin_objects(text, objects)
    if substituted:
        stats["builtin_objects_substituted"] = substituted
    if not functions:
        return MacroExpansion(
            text=text,
            line_origins=tuple(origins),
            object_macros=objects,
            function_macros=functions,
            stats=stats,
        )

    # ---- stage 3: invocation expansion -------------------------------------
    spans = non_code_spans_str(text)
    starts = _line_starts(text)

    def line_of(offset: int) -> int:
        return bisect.bisect_right(starts, offset)

    directive_lines = {
        index + 1
        for index, line in enumerate(lines)
        if line.lstrip().startswith("#")
    }

    sites: List[Tuple[int, int, str, int]] = []  # (start, end, replacement, origin)
    for match in _IDENT_RE.finditer(text):
        name = match.group()
        if name not in functions:
            continue
        if _in_span(spans, match.start()):
            continue
        if line_of(match.start()) in directive_lines:
            continue
        if _prev_code_char(text, spans, match.start()) in _NON_CALL_PRECEDERS:
            continue
        probe = _find_open_paren(text, match.end())
        if probe is None or _in_span(spans, probe):
            continue
        line = line_of(match.start())
        macro = functions[name]
        if not macro.is_active_at(line):
            continue
        close = _find_matching_paren(text, probe, spans)
        if close is None:
            continue
        args = _split_arguments(text[probe + 1 : close])
        if len(args) < len(macro.params):
            args += [""] * (len(macro.params) - len(args))
        substituted = _substitute(
            macro.body, macro.params, args, variadic=macro.variadic
        )
        replacement = _expand_inline(
            substituted, functions, (name,), line, stats
        )
        sites.append((match.start(), close + 1, replacement, origins[line - 1]))
        stats["expansions"] += 1

    if not sites:
        return MacroExpansion(
            text=text,
            line_origins=tuple(origins),
            object_macros=objects,
            function_macros=functions,
            stats=stats,
        )

    # ---- rebuild with per-line origins --------------------------------------
    builder = _OutputBuilder(origins[0])

    def origin_of(offset: int) -> int:
        line = bisect.bisect_right(starts, offset)
        return origins[min(max(line, 1), len(origins)) - 1]

    cursor = 0
    for start, end, replacement, origin in sites:
        builder.emit_source(text, cursor, start, origin_of)
        builder.emit_replacement(replacement, origin)
        cursor = end
    builder.emit_source(text, cursor, len(text), origin_of)
    out_text, out_origins = builder.finish()

    return MacroExpansion(
        text=out_text,
        line_origins=out_origins,
        object_macros=objects,
        function_macros=functions,
        stats=stats,
    )
