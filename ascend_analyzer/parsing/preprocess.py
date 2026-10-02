"""Source preparation for tree-sitter.

DaVinci address-space qualifiers (``__ubuf__``, ``__cbuf__``, ``__gm__``, ...)
and kernel attributes (``__global__``, ``__aicore__``) are compiler extensions
that ``tree-sitter-cpp`` does not know, and they derail its C++ grammar into
``ERROR`` nodes exactly where the interesting declarations are.

Rather than tolerate a broken tree, this module rewrites each qualifier into a
block comment of **identical byte length** - ``__ubuf__`` becomes
``/*ubuf*/``.  The grammar then parses the file cleanly, while every byte
offset, line and column in the rewritten text still matches the original
source exactly, so diagnostics point at the real file without any mapping
table.  The qualifiers themselves are not lost: their byte spans are recorded
in :class:`PreparedSource` so the AST visitor can recover the address space of
each declaration.

The same pass collects ``@ascend-*`` annotations, the documented escape hatch
for kernels whose layout the parser cannot infer on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from ..hardware import ADDRESS_SPACE_TO_DOMAIN, PhysicalDomain
from .macro_expand import expand_macros

__all__ = [
    "QualifierSpan",
    "Annotation",
    "PreparedSource",
    "prepare_source",
    "prepare_translation_unit",
    "KERNEL_ATTRIBUTES",
    "non_code_spans",
    "decoration_only_macro",
]


#: Address-space qualifiers that bind a declaration to a physical domain.
_ADDRESS_SPACES: Tuple[str, ...] = tuple(ADDRESS_SPACE_TO_DOMAIN)

#: Attributes that mark a function as an AI Core kernel entry point.
KERNEL_ATTRIBUTES: Tuple[str, ...] = ("__global__", "__aicore__", "__aicpu__")

#: Extra compiler extensions that are noise for our purposes but break parsing.
_OTHER_EXTENSIONS: Tuple[str, ...] = (
    "__restrict__",
    "__inline__",
    "__forceinline__",
    # CANN headers spell the same attribute both ways; kernels under
    # ``mla_preprocess`` use the underscored form 90 times.
    "__force_inline__",
    # Marks a v3 SIMD vector function; carries nothing for the layout, but an
    # unknown specifier ahead of a return type derails the declaration.
    "__simd_vf__",
    # Storage qualifier on a per-block static.
    "__BLOCK_LOCAL__",
)

_ALL_QUALIFIERS: Tuple[str, ...] = _ADDRESS_SPACES + KERNEL_ATTRIBUTES + _OTHER_EXTENSIONS


def _equal_length_comment(token: str) -> str:
    """Render ``__ubuf__`` as ``/*ubuf*/`` - same byte length, parses as trivia.

    A token of length *n* becomes ``/*`` + *n*-4 payload characters + ``*/``.
    Every qualifier we rewrite is at least 6 characters, so the payload is
    never empty.
    """
    inner = token.strip("_")
    budget = len(token) - 4
    if len(inner) < budget:
        inner = inner + "-" * (budget - len(inner))
    else:
        inner = inner[:budget]
    out = f"/*{inner}*/"
    assert len(out) == len(token), (token, out)
    return out


#: token -> equal-length replacement, validated at import time.
_REPLACEMENTS: Mapping[str, str] = {t: _equal_length_comment(t) for t in _ALL_QUALIFIERS}

#: The same table as bytes.  The rewrite runs over UTF-8 bytes rather than
#: characters because tree-sitter reports every node position as a *byte*
#: offset.  A single non-ASCII character anywhere earlier in the file - a
#: Chinese comment, say, which Ascend kernels are full of - would otherwise
#: desynchronise the recorded qualifier spans from the parse tree, and the
#: address space of the following declaration would be silently lost.
_REPLACEMENTS_BYTES: Mapping[bytes, bytes] = {
    token.encode("ascii"): replacement.encode("ascii")
    for token, replacement in _REPLACEMENTS.items()
}

_QUALIFIER_RE_BYTES = re.compile(
    rb"\b("
    + rb"|".join(
        re.escape(t.encode("ascii"))
        for t in sorted(_ALL_QUALIFIERS, key=len, reverse=True)
    )
    + rb")\b"
)

#: CCE function *location* qualifiers.  Ascend C spells where a function may
#: run as a bracketed list ahead of the return type::
#:
#:     __forceinline__ [host, aicore] void MoeTokenUnpermuteTiling(...)
#:
#: ``tree-sitter-cpp`` reads the ``[`` as the start of a lambda capture and
#: never recovers, so a single such declaration turns the rest of the file
#: into one ERROR node.  The names below are the documented CCE locations;
#: ``device`` is deliberately absent because a bare ``[device]`` is far more
#: likely to be an array subscript than a qualifier.
_CCE_LOCATIONS: Tuple[str, ...] = (
    "host",
    "aicore",
    "aicpu",
    "vector_core",
    "aicube",
    "mix",
    "mix_aic",
    "mix_aiv",
)

_CCE_BRACKET_RE_BYTES = re.compile(
    rb"\[[ \t]*"
    + rb"(?:" + rb"|".join(t.encode("ascii") for t in _CCE_LOCATIONS) + rb")"
    + rb"(?:[ \t]*,[ \t]*"
    + rb"(?:" + rb"|".join(t.encode("ascii") for t in _CCE_LOCATIONS) + rb")"
    + rb")*[ \t]*\]"
)

#: Bytes that, immediately before a ``[``, mean the bracket is a subscript, a
#: second C++ attribute bracket, or a lambda introducer rather than a CCE
#: location qualifier.
#: Horizontal blanks, and all whitespace, as byte sets for the scanners.
_BLANK_BYTES = frozenset((0x20, 0x09))
_SPACE_BYTES = frozenset((0x20, 0x09, 0x0A, 0x0D))

_SUBSCRIPT_LEAD = frozenset(
    b"_)]"
    b"abcdefghijklmnopqrstuvwxyz"
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    b"0123456789"
)

#: Bytes that may begin the return type following a location qualifier.
_TYPE_START = frozenset(
    b"_*&~:"
    b"abcdefghijklmnopqrstuvwxyz"
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

#: Declaration decoration that carries no meaning for this analyzer.  An
#: object-like macro whose body is *only* these tokens is safe to blank at
#: every use site: ``#define HOST_DEVICE __forceinline__ [host, aicore]``.
_DECORATION_KEYWORDS: frozenset = frozenset(
    ("inline", "static", "constexpr", "const", "explicit", "extern")
)


def _balanced_paren_end(raw: bytes, open_at: int) -> int:
    """End offset (exclusive) of the parenthesis group starting at ``open_at``."""
    depth = 0
    for index in range(open_at, len(raw)):
        byte = raw[index]
        if byte == 0x28:  # (
            depth += 1
        elif byte == 0x29:  # )
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _gnu_attribute_spans(raw: bytes, skip: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Spans of ``__attribute__((...))``, parentheses balanced."""
    spans: List[Tuple[int, int]] = []
    needle = b"__attribute__"
    cursor = 0
    while True:
        at = raw.find(needle, cursor)
        if at < 0:
            return spans
        cursor = at + len(needle)
        if _in_spans(skip, at):
            continue
        probe = cursor
        while probe < len(raw) and raw[probe] in _BLANK_BYTES:
            probe += 1
        if probe >= len(raw) or raw[probe] != 0x28:
            continue
        end = _balanced_paren_end(raw, probe)
        if end < 0:
            continue
        spans.append((at, end))
        cursor = end


def _cce_bracket_spans(
    raw: bytes,
    skip: Sequence[Tuple[int, int]],
    require_type_after: bool = True,
) -> List[Tuple[int, int]]:
    """Spans of CCE ``[host, aicore]``-style location qualifiers.

    The bracket has to be told apart from an array subscript (``buf[host]``),
    a C++ attribute (``[[nodiscard]]``) and a lambda introducer.  Three tests
    do it, and they lean on how a location qualifier actually appears -
    between the specifiers and the return type of a declaration:

    * it is never glued to the preceding token: ``buf[host]`` is a subscript,
      while ``__forceinline__ [host, aicore]`` is a qualifier;
    * it is never doubled: ``[[`` or ``]]`` is a C++ attribute;
    * what follows it starts a type, so an identifier, ``*`` or ``&`` - not
      ``;``, ``=``, ``)`` or an operator, which is what trails a subscript.

    A list of two or more names (``[host, aicore]``) skips the first test: a
    comma inside a subscript is not valid C++, so it cannot be one.
    """
    spans: List[Tuple[int, int]] = []
    for match in _CCE_BRACKET_RE_BYTES.finditer(raw):
        start, end = match.start(), match.end()
        if _in_spans(skip, start):
            continue

        # ``[[attr]]`` / ``]]`` - a C++ attribute, not a CCE qualifier.
        if start > 0 and raw[start - 1] == 0x5B:
            continue
        if end < len(raw) and raw[end] == 0x5D:
            continue

        multi = b"," in match.group(0)
        if not multi:
            # Glued to the previous token means subscript: ``buf[aicore]``.
            if start > 0 and raw[start - 1] in _SUBSCRIPT_LEAD:
                continue

        # What follows must be able to start a type.  A macro *body* is
        # classified on its own, with the return type still at the use site,
        # so the caller switches this test off for that case.
        if require_type_after:
            probe = end
            while probe < len(raw) and raw[probe] in _SPACE_BYTES:
                probe += 1
            if probe >= len(raw) or raw[probe] not in _TYPE_START:
                continue

        spans.append((start, end))
    return spans


def decoration_only_macro(body: str) -> bool:
    """``True`` when a macro body is nothing but declaration decoration.

    ``HOST_DEVICE`` (``__forceinline__ [host, aicore]``) and
    ``__force_inline__`` (``inline __attribute__((always_inline))``) both
    qualify: they decorate a declaration without contributing a type, a name
    or a value.  Such a macro can be blanked at every use site, which is what
    keeps an unexpanded object-like macro from derailing the grammar.
    """
    raw = body.encode("utf-8")
    blanks = _gnu_attribute_spans(raw, ()) + _cce_bracket_spans(
        raw, (), require_type_after=False
    )
    kept = bytearray(raw)
    for start, end in blanks:
        for index in range(start, end):
            kept[index] = 0x20
    text = bytes(kept).decode("utf-8", errors="replace")
    text = _QUALIFIER_RE_BYTES.sub(b" ", text.encode("utf-8")).decode("utf-8")
    remaining = [tok for tok in text.replace(",", " ").split() if tok]
    if not remaining:
        return True
    return all(tok in _DECORATION_KEYWORDS for tok in remaining)


_ANNOTATION_RE = re.compile(
    r"(?://|/\*)\s*@ascend-(?P<kind>[a-z][a-z0-9-]*)\s*:\s*(?P<body>[^\n*]*)"
)

#: ``key=value`` pairs inside an annotation body.  The value may itself be a
#: comma-separated list (``names=a,b,c``) as long as it has no spaces, which is
#: what keeps ``offset=0, count=256`` parsing as two fields rather than one.
_KV_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*=\s*(?P<value>[^\s,]+(?:,[^\s,]+)*)"
)


@dataclass(frozen=True)
class QualifierSpan:
    """One rewritten qualifier and where it lived in the source."""

    token: str
    start_byte: int
    end_byte: int
    line: int

    @property
    def domain(self) -> Optional[PhysicalDomain]:
        """Physical domain implied by this qualifier, if it is an address space."""
        return ADDRESS_SPACE_TO_DOMAIN.get(self.token)

    @property
    def is_kernel_attribute(self) -> bool:
        return self.token in KERNEL_ATTRIBUTES


@dataclass(frozen=True)
class Annotation:
    """A parsed ``@ascend-<kind>: ...`` comment annotation."""

    kind: str
    body: str
    line: int
    fields: Mapping[str, str] = field(default_factory=dict)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.fields.get(key, default)

    def get_int(self, key: str) -> Optional[int]:
        raw = self.fields.get(key)
        if raw is None:
            return None
        try:
            return int(raw, 0)
        except ValueError:
            return None


@dataclass
class PreparedSource:
    """The result of preparing a translation unit for parsing."""

    path: str
    original: str
    rewritten: str
    qualifiers: Tuple[QualifierSpan, ...] = ()
    annotations: Tuple[Annotation, ...] = ()
    #: The rewritten source as UTF-8 bytes.  This is what tree-sitter parses,
    #: and every offset in :attr:`qualifiers` indexes into it.
    rewritten_bytes: bytes = b""
    #: For every 1-based line of ``rewritten``, the 1-based line of the
    #: *original* file it came from.  Empty means the identity mapping (no
    #: macro expansion or include inlining happened).
    line_origins: Tuple[int, ...] = ()
    #: Object-like ``#define`` bodies in source order, for the constant
    #: folder.  Empty when no expansion pass ran (the raw ``preproc_def``
    #: nodes are then collected from the tree instead).
    macro_object_defs: Tuple[Tuple[str, str], ...] = ()
    #: Counters from the macro-expansion pass, for the report header.
    expansion_stats: Mapping[str, int] = field(default_factory=dict)
    #: How many CCE declaration decorations the preparation pass blanked.
    decorations_blanked: int = 0

    def __post_init__(self) -> None:
        if not self.rewritten_bytes:
            self.rewritten_bytes = self.rewritten.encode("utf-8")
        # The whole design rests on this invariant; assert it loudly.
        if len(self.rewritten_bytes) != len(self.rewritten.encode("utf-8")):
            raise AssertionError(
                "rewritten bytes and text disagree; tree-sitter offsets "
                "would not line up"
            )
        if not self.line_origins:
            # The invariant is over *bytes*, because that is what tree-sitter
            # reports: both node offsets and the column within a line.  It is
            # deliberately not over Python characters - blanking a multi-byte
            # comment (these sources are full of Chinese prose) replaces one
            # three-byte character with three spaces, which keeps every byte
            # offset and every line number exact while making the string
            # longer.  Lines are checked separately for the same reason.
            if len(self.rewritten_bytes) != len(self.original.encode("utf-8")):
                raise AssertionError(
                    "preparation changed the byte length of the source "
                    f"({len(self.original.encode('utf-8'))} -> "
                    f"{len(self.rewritten_bytes)}); "
                    "tree-sitter offsets would not line up"
                )
            if self.rewritten.count(chr(10)) != self.original.count(chr(10)):
                raise AssertionError(
                    "preparation changed the line count of the source; "
                    "reported line numbers would be wrong"
                )
        elif len(self.line_origins) != self.rewritten.count("\n") + 1:
            raise AssertionError(
                "line origin map does not cover the rewritten source "
                f"({len(self.line_origins)} origins for "
                f"{self.rewritten.count(chr(10)) + 1} lines)"
            )

    @property
    def encoded(self) -> bytes:
        return self.rewritten_bytes

    # -- line mapping --------------------------------------------------------

    def origin_line(self, line: int) -> int:
        """Translate a line of the parse basis to the original file."""
        if not self.line_origins:
            return line
        if 1 <= line <= len(self.line_origins):
            return self.line_origins[line - 1]
        return self.line_origins[-1] if self.line_origins else line

    # -- qualifier lookup ---------------------------------------------------

    def qualifiers_for_range(self, start_byte: int, end_byte: int) -> List[QualifierSpan]:
        """Qualifiers that apply to a node spanning ``[start_byte, end_byte)``.

        A qualifier applies when it sits *inside* the node's span (as in a cast
        expression) or immediately *before* it separated only by whitespace -
        the case for a leading ``__ubuf__`` on a declaration, which tree-sitter
        attaches as a preceding sibling comment rather than a child.
        """
        found: List[QualifierSpan] = []
        for span in self.qualifiers:
            if start_byte <= span.start_byte < end_byte:
                found.append(span)
            elif span.end_byte <= start_byte and self._only_blank_between(
                span.end_byte, start_byte
            ):
                found.append(span)
        return found

    def address_space_for_range(
        self, start_byte: int, end_byte: int
    ) -> Optional[PhysicalDomain]:
        """The single address space implied by a node's qualifiers, if any."""
        domains = [
            span.domain
            for span in self.qualifiers_for_range(start_byte, end_byte)
            if span.domain is not None
        ]
        return domains[0] if domains else None

    def _only_blank_between(self, start: int, end: int) -> bool:
        if end <= start:
            return True
        gap = self.rewritten_bytes[start:end]
        return gap.strip() == b"" and b"\n" not in gap

    def has_kernel_attribute_before(self, start_byte: int, window: int = 160) -> bool:
        """``True`` when ``__global__``/``__aicore__`` precedes ``start_byte``.

        Kernel attributes are emitted as comments by the rewrite, so they are
        siblings rather than children of the ``function_definition``; a short
        backwards window is the reliable way to find them.
        """
        lo = max(0, start_byte - window)
        return any(
            span.is_kernel_attribute and lo <= span.start_byte < start_byte
            for span in self.qualifiers
        )

    def has_launch_attribute_before(
        self, start_byte: int, window: int = 160
    ) -> bool:
        """``True`` when specifically ``__global__`` precedes ``start_byte``.

        ``__aicore__`` alone marks device code, which every helper and member
        method carries; only ``__global__`` marks a launch entry.  The
        distinction decides whether a function is analyzed in its own right or
        reached by inlining it into the entry that calls it.
        """
        lo = max(0, start_byte - window)
        return any(
            span.token == "__global__" and lo <= span.start_byte < start_byte
            for span in self.qualifiers
        )

    # -- annotation lookup --------------------------------------------------

    def annotations_of(self, kind: str) -> List[Annotation]:
        return [a for a in self.annotations if a.kind == kind]

    def annotations_near(self, line: int, kinds: Sequence[str] = ()) -> List[Annotation]:
        """Annotations on ``line`` or the line directly above it."""
        wanted = set(kinds)
        return [
            a
            for a in self.annotations
            if a.line in (line, line - 1) and (not wanted or a.kind in wanted)
        ]

    def suppressed_codes_for(self, line: int) -> List[str]:
        """Codes silenced by an ``@ascend-ignore`` annotation at ``line``."""
        out: List[str] = []
        for ann in self.annotations_near(line, ("ignore",)):
            out.extend(tok.strip().upper() for tok in ann.body.replace(",", " ").split())
        return [c for c in out if c]


def _line_of_bytes(source: bytes, offset: int) -> int:
    """1-based line number of a byte offset."""
    return source.count(b"\n", 0, offset) + 1


def _line_of(text: str, offset: int) -> int:
    """1-based line number of a character offset."""
    return text.count("\n", 0, offset) + 1


_SLASH = ord("/")
_STAR = ord("*")
_BACKSLASH = ord("\\")
_NEWLINE = ord("\n")
_QUOTES = (ord('"'), ord("'"))


def _non_code_spans(source: bytes) -> List[Tuple[int, int]]:
    """Byte ranges occupied by comments and string/char literals.

    The qualifier rewrite must not touch these.  Rewriting ``__ubuf__`` to
    ``/*ubuf*/`` *inside* a block comment would inject a ``*/`` that closes the
    comment early and spills prose into the token stream - and kernel files
    routinely mention these qualifiers in their header comments.

    Operates on bytes so the returned offsets match tree-sitter's.  Multi-byte
    UTF-8 sequences only ever appear inside comments and literals here, and
    their continuation bytes are all >= 0x80, so they never collide with the
    ASCII delimiters this scanner looks for.
    """
    spans: List[Tuple[int, int]] = []
    index, length = 0, len(source)
    while index < length:
        byte = source[index]

        if byte == _SLASH and index + 1 < length:
            nxt = source[index + 1]
            if nxt == _SLASH:
                end = source.find(b"\n", index)
                end = length if end < 0 else end
                spans.append((index, end))
                index = end
                continue
            if nxt == _STAR:
                end = source.find(b"*/", index + 2)
                end = length if end < 0 else end + 2
                spans.append((index, end))
                index = end
                continue

        if byte in _QUOTES:
            quote = byte
            cursor = index + 1
            while cursor < length:
                if source[cursor] == _BACKSLASH:
                    cursor += 2
                    continue
                if source[cursor] == quote:
                    cursor += 1
                    break
                if source[cursor] == _NEWLINE:
                    break  # unterminated literal; stop at the line end
                cursor += 1
            spans.append((index, min(cursor, length)))
            index = max(cursor, index + 1)
            continue

        index += 1
    return spans


def _in_spans(spans: Sequence[Tuple[int, int]], offset: int) -> bool:
    """``True`` when ``offset`` falls inside one of the sorted ``spans``."""
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


#: Preprocessor conditionals whose operand is a *macro name* rather than code.
#: ``#ifndef __force_inline__`` is a header guard testing whether the macro is
#: defined; rewriting that name - into a comment, or into spaces - leaves the
#: directive without an identifier and makes it an ERROR node.  The macro
#: expander blanks ``#define`` lines but keeps conditionals, so these are the
#: directives whose operands have to be left exactly as written.
_DIRECTIVE_OPERAND_RE_BYTES = re.compile(
    rb"^[ \t]*#[ \t]*(?:ifdef|ifndef|undef|if|elif)\b(.*)",
    re.MULTILINE,
)


def _directive_operand_spans(raw: bytes) -> List[Tuple[int, int]]:
    """Byte spans of preprocessor-conditional operands, which must not change."""
    return [
        (m.start(1), m.end(1))
        for m in _DIRECTIVE_OPERAND_RE_BYTES.finditer(raw)
        if m.end(1) > m.start(1)
    ]


#: Conditional directives, for resolving ``__cplusplus`` guards.
_CONDITIONAL_RE_BYTES = re.compile(
    rb"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b(.*)",
    re.MULTILINE,
)

#: ``#ifdef __cplusplus`` and the spellings that mean the same thing.
_CPLUSPLUS_TRUE_RE = re.compile(
    rb"^(?:def\s+__cplusplus|\s*defined\s*\(?\s*__cplusplus)\b"
)
_CPLUSPLUS_FALSE_RE = re.compile(
    rb"^(?:ndef\s+__cplusplus|\s*!\s*defined\s*\(?\s*__cplusplus)\b"
)


def _line_end(raw: bytes, offset: int) -> int:
    end = raw.find(bytes([10]), offset)
    return len(raw) if end < 0 else end


def _cplusplus_guard_spans(raw: bytes) -> List[Tuple[int, int]]:
    """Spans to blank so ``#ifdef __cplusplus`` guards stop breaking the parse.

    The idiom every ``aclnn_*.h`` in this fleet uses opens a brace inside one
    conditional block and closes it inside another::

        #ifdef __cplusplus
        extern "C" {
        #endif
        ...
        #ifdef __cplusplus
        }
        #endif

    ``tree-sitter-cpp`` requires each preprocessor block to be brace-balanced,
    so the unmatched ``{`` turns the remainder of the translation unit into one
    ERROR node - and because these headers are inlined, it takes the including
    ``.cpp`` with it.

    Resolving the guard is exact rather than approximate: this analyzer always
    parses as C++, so ``__cplusplus`` *is* defined.  The directive lines are
    therefore inert and are blanked, which leaves ``extern "C" { ... }`` as
    ordinary balanced code.  Where such a guard has an ``#else``, that branch
    is the C one the compiler would discard, so it is blanked too.
    """
    spans: List[Tuple[int, int]] = []
    stack: List[dict] = []
    for match in _CONDITIONAL_RE_BYTES.finditer(raw):
        name = match.group(1)
        rest = match.group(2)
        start = match.start()
        end = _line_end(raw, start)

        if name in (b"if", b"ifdef", b"ifndef"):
            kind = "other"
            if name == b"ifdef" and _CPLUSPLUS_TRUE_RE.match(b"def " + rest.strip()):
                kind = "true"
            elif name == b"ifndef" and _CPLUSPLUS_FALSE_RE.match(
                b"ndef " + rest.strip()
            ):
                kind = "true_else"
            elif name == b"if":
                if _CPLUSPLUS_TRUE_RE.match(rest):
                    kind = "true"
                elif _CPLUSPLUS_FALSE_RE.match(rest):
                    kind = "true_else"
            stack.append({"kind": kind, "open": (start, end), "else": None})
            continue

        if name in (b"else", b"elif"):
            if stack and stack[-1]["kind"] != "other" and stack[-1]["else"] is None:
                stack[-1]["else"] = (start, end)
            continue

        if name == b"endif":
            if not stack:
                continue
            frame = stack.pop()
            if frame["kind"] == "other":
                continue
            open_start, open_end = frame["open"]
            else_span = frame["else"]
            spans.append((open_start, open_end))
            spans.append((start, end))
            if frame["kind"] == "true":
                # The active branch is the first one; drop any ``#else`` body.
                if else_span is not None:
                    spans.append((else_span[0], start))
            else:
                # ``#ifndef __cplusplus``: the first branch is the C one.
                stop = else_span[1] if else_span is not None else start
                spans.append((open_end, stop))
    return spans


#: An attribute macro standing alone on its own line.  ``CATLASS_DEVICE`` is
#: the one that matters in this fleet: it decorates 42 files' member
#: declarations and is defined in the Catlass headers, which are an external
#: dependency and are not in the tree at all - so unlike ``HOST_DEVICE`` its
#: ``#define`` can never be found, and the body-based test cannot classify it.
#:
#: The test is therefore *positional*, not name-based: a bare identifier alone
#: on a line, at a declaration boundary, with no punctuation of any kind.  No
#: such line is valid C++ on its own, so one of two things is true - it is a
#: declaration-decoration macro, and blanking it fixes the parse, or it is a
#: macro expanding to a whole declaration, which was already unparseable and
#: so is lost either way.  Requiring ALL-CAPS and a length of four keeps it off
#: ordinary identifiers.
_LONE_MACRO_RE_BYTES = re.compile(
    rb"^[ \t]*([A-Z][A-Z0-9_]{3,})[ \t]*$", re.MULTILINE
)

#: Bytes that end the previous line when we are *not* at a statement boundary -
#: the lone identifier is then an operand of a continued expression, and
#: blanking it would change the program.
_CONTINUATION_TAILS = (
    b"=", b"+", b"-", b"*", b"/", b"%", b",", b"(", b"[", b"&", b"|", b"^",
    b"<", b">", b"?", b":", b"!", b"~", b"\\",
)

#: Bytes that cannot begin a declaration.  A lone identifier followed by one
#: of these is a value - an enum member, an initialiser field - not decoration.
_NOT_A_DECL_START = (b"}", b",", b"=", b")", b";", b":", b"]")

#: Keywords that an all-caps identifier could be, and must not be blanked.
_CAPS_KEYWORDS = frozenset({b"NULL", b"TRUE", b"FALSE"})


def _lone_attribute_macro_spans(
    raw: bytes, skip: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Spans of bare ALL-CAPS attribute macros occupying a whole line."""
    spans: List[Tuple[int, int]] = []
    for match in _LONE_MACRO_RE_BYTES.finditer(raw):
        start, end = match.start(1), match.end(1)
        if _in_spans(skip, start):
            continue
        if match.group(1) in _CAPS_KEYWORDS:
            continue
        # The previous non-blank line has to close a statement, so the
        # identifier cannot be an operand of a continued expression.
        probe = match.start() - 1
        while probe >= 0 and raw[probe] in _SPACE_BYTES:
            probe -= 1
        if probe >= 0 and raw[probe : probe + 1] in _CONTINUATION_TAILS:
            continue
        # Decoration precedes a *declaration*.  A lone ALL-CAPS identifier
        # followed by ``}`` is the last member of an enum or an aggregate
        # initialiser - ``enum E { BARBAZ };`` - and blanking it would delete
        # a constant the folder needs.  Checking what follows is what tells
        # the two apart, since both can sit directly after a ``{``.
        probe = end
        while probe < len(raw) and raw[probe] in _SPACE_BYTES:
            probe += 1
        if probe >= len(raw) or raw[probe : probe + 1] in _NOT_A_DECL_START:
            continue
        spans.append((start, end))
    return spans


#: Keywords a declaration can start with.  An ALL-CAPS macro sitting directly
#: in front of one is decoration - ``TEMPLATES_DEF_NO_DEFAULT struct Traits``
#: - and, being unexpanded, derails the declaration it decorates.  Matching on
#: the *following keyword* keeps this positional rather than name-based.
_DECL_KEYWORD_RE_BYTES = re.compile(
    rb"(?:^|(?<=[;{}\n]))[ \t]*([A-Z][A-Z0-9_]{3,})[ \t]+"
    rb"(?=(?:struct|class|template|union|enum|typedef|using|static|inline|"
    rb"constexpr|void|int|unsigned|signed|float|double|char|bool|auto)\b)"
)


def _decoration_prefix_spans(
    raw: bytes, skip: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Spans of ALL-CAPS macros standing directly before a declaration keyword."""
    spans: List[Tuple[int, int]] = []
    for match in _DECL_KEYWORD_RE_BYTES.finditer(raw):
        start, end = match.start(1), match.end(1)
        if _in_spans(skip, start) or match.group(1) in _CAPS_KEYWORDS:
            continue
        spans.append((start, end))
    return spans


#: An ALL-CAPS function-like macro invocation opening a statement.
_MACRO_CALL_RE_BYTES = re.compile(
    rb"(?:^|(?<=[;{}\n]))([ \t\n]*)([A-Z][A-Z0-9_]{3,})[ \t\n]*\(", re.MULTILINE
)


def _macro_statement_spans(
    raw: bytes, skip: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Spans of *multi-line* ALL-CAPS macro invocations used as a statement.

    CANN's tiling-key DSL is written as a file-scope macro call spanning
    dozens of lines::

        ASCENDC_TPL_SEL(
            ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(X_LAYOUT, ..., 0, 1),
                                 ...),
        );

    The macro is defined in the CANN headers, which are not in the tree, so it
    is never expanded - and its argument list is not valid C++ even as an
    expression (note the trailing comma), so the whole region becomes one
    ERROR node that swallows the rest of the file.

    Only invocations that **span more than one line** are blanked.  A
    single-line ``FOO(x);`` parses as an ordinary declaration and is left
    alone; the multi-line form is the one that cascades, and an unexpanded
    macro body carries nothing this analyzer could have modelled anyway.  A
    definition (``)`` followed by ``{``) is never matched, so a macro-declared
    function body is left intact.
    """
    spans: List[Tuple[int, int]] = []
    for match in _MACRO_CALL_RE_BYTES.finditer(raw):
        name_start = match.start(2)
        if _in_spans(skip, name_start):
            continue
        open_paren = match.end() - 1
        close = _balanced_paren_end(raw, open_paren)
        if close < 0:
            continue
        # Multi-line only: a single-line call is parseable and harmless.
        if raw.count(bytes([10]), name_start, close) == 0:
            continue
        end = close
        probe = close
        # Skip every kind of whitespace, newlines included: a macro such
        # as TORCH_LIBRARY opens its brace on the line after the closing
        # parenthesis, and stopping at the newline would blank the head of
        # a definition and orphan its body.
        while probe < len(raw) and raw[probe] in _SPACE_BYTES:
            probe += 1
        if probe < len(raw) and raw[probe : probe + 1] == b"{":
            continue  # a definition, not a statement; leave the body alone
        if probe < len(raw) and raw[probe : probe + 1] == b";":
            end = probe + 1
        spans.append((name_start, end))
    return spans


#: Public alias: the byte-span scanner is reused by the macro expander.
non_code_spans = _non_code_spans


def _decoration_macro_spans(
    raw: bytes, skip: Sequence[Tuple[int, int]], names: Sequence[str]
) -> List[Tuple[int, int]]:
    """Spans of every use of a decoration-only object-like macro.

    The macro expander blanks an object-like ``#define`` where it is written
    but does not substitute it at the use sites, so a name such as
    ``HOST_DEVICE`` survives into the parse basis as a bare identifier ahead
    of a declaration - which derails the grammar for the rest of the file.
    Blanking the uses costs nothing, because such a macro contributes only
    decoration.
    """
    if not names:
        return []
    pattern = re.compile(
        rb"\b(?:" + rb"|".join(re.escape(n.encode("utf-8")) for n in names) + rb")\b"
    )
    return [
        (m.start(), m.end())
        for m in pattern.finditer(raw)
        if not _in_spans(skip, m.start())
    ]


def _blank_spans(raw: bytes, spans: Sequence[Tuple[int, int]]) -> bytes:
    """Overwrite ``spans`` with spaces, keeping every other byte in place."""
    if not spans:
        return raw
    out = bytearray(raw)
    for start, end in spans:
        for index in range(start, min(end, len(out))):
            if out[index] != 0x0A:  # never swallow a newline
                out[index] = 0x20
    return bytes(out)


def prepare_source(
    path: str, source: str, attribute_macros: Sequence[str] = ()
) -> PreparedSource:
    """Rewrite DaVinci qualifiers and extract ``@ascend-*`` annotations.

    The rewrite runs over UTF-8 bytes, so the recorded qualifier offsets are
    directly comparable with the byte offsets tree-sitter reports for every
    node.  Doing it over characters would break on any file containing
    non-ASCII text ahead of a qualifier.
    """
    raw = source.encode("utf-8")
    qualifiers: List[QualifierSpan] = []
    pieces: List[bytes] = []
    cursor = 0
    skip = sorted(_non_code_spans(raw) + _directive_operand_spans(raw))

    for match in _QUALIFIER_RE_BYTES.finditer(raw):
        if _in_spans(skip, match.start()):
            continue  # inside a comment or string literal; leave it alone
        token = match.group(1)
        pieces.append(raw[cursor : match.start()])
        pieces.append(_REPLACEMENTS_BYTES[token])
        qualifiers.append(
            QualifierSpan(
                token=token.decode("ascii"),
                start_byte=match.start(),
                end_byte=match.end(),
                line=_line_of_bytes(raw, match.start()),
            )
        )
        cursor = match.end()
    pieces.append(raw[cursor:])
    rewritten_bytes = b"".join(pieces)

    # CCE declaration decoration goes next, blanked to spaces rather than
    # comments: a location qualifier sits between the specifiers and the
    # return type, where a ``/*...*/`` would be legal but pointless.  Every
    # replacement is the same byte length as what it covers, so the qualifier
    # spans recorded above - and every line and column tree-sitter reports -
    # still address the original file.
    decoration_skip = sorted(
        _non_code_spans(rewritten_bytes)
        + _directive_operand_spans(rewritten_bytes)
    )
    decoration_spans = (
        _cplusplus_guard_spans(rewritten_bytes)
        + _gnu_attribute_spans(rewritten_bytes, decoration_skip)
        + _cce_bracket_spans(rewritten_bytes, decoration_skip)
        + _decoration_macro_spans(
            rewritten_bytes, decoration_skip, attribute_macros
        )
        + _lone_attribute_macro_spans(rewritten_bytes, decoration_skip)
        + _macro_statement_spans(rewritten_bytes, decoration_skip)
        + _decoration_prefix_spans(rewritten_bytes, decoration_skip)
    )
    rewritten_bytes = _blank_spans(rewritten_bytes, decoration_spans)
    rewritten = rewritten_bytes.decode("utf-8")

    annotations: List[Annotation] = []
    for match in _ANNOTATION_RE.finditer(source):
        body = match.group("body").strip().rstrip("*/").strip()
        annotations.append(
            Annotation(
                kind=match.group("kind").lower(),
                body=body,
                line=_line_of(source, match.start()),
                fields={
                    kv.group("key").lower(): kv.group("value")
                    for kv in _KV_RE.finditer(body)
                },
            )
        )

    return PreparedSource(
        path=path,
        original=source,
        rewritten=rewritten,
        qualifiers=tuple(qualifiers),
        annotations=tuple(annotations),
        rewritten_bytes=rewritten_bytes,
        decorations_blanked=len(decoration_spans),
    )


def prepare_translation_unit(path: str, source: str) -> PreparedSource:
    """Prepare a full translation unit: expand macros, then rewrite qualifiers.

    This is the front door the AST visitor uses.  When the source contains
    preprocessor constructs worth handling - local includes or function-like
    macros - :func:`~ascend_analyzer.parsing.macro_expand.expand_macros` runs
    first and the qualifier rewrite is applied to the *expanded* text, so
    qualifier spans line up with the buffer tree-sitter parses.  Everything the
    checkers see is then translated back to original-file lines through
    :attr:`PreparedSource.line_origins`.

    Sources without ``#define``/``#include`` take the identity path and keep
    the historical byte-for-byte invariants.
    """
    base_dir: Optional[Path]
    try:
        # Pseudo paths like "<test>.cpp" resolve to the CWD; the include
        # lookup simply finds nothing there.
        base_dir = Path(path).resolve().parent
    except (OSError, ValueError):
        base_dir = None

    expansion = expand_macros(source, base_dir=base_dir)

    # Object-like macros that expand to nothing but declaration decoration are
    # blanked at their use sites.  ``HOST_DEVICE``
    # (``__forceinline__ [host, aicore]``) is the one that matters most: the
    # expander blanks its ``#define`` but leaves the 83 uses in a file such as
    # ``attn_infra/coord.hpp`` standing, and a bare identifier ahead of a
    # constructor turns the whole translation unit into one ERROR node.
    attribute_macros = tuple(
        name
        for name, body in expansion.object_macros.items()
        if decoration_only_macro(body)
    )

    if not expansion.changed:
        return prepare_source(path, source, attribute_macros)

    prepared = prepare_source(path, expansion.text, attribute_macros)
    prepared.original = source
    prepared.line_origins = expansion.line_origins
    prepared.macro_object_defs = tuple(expansion.object_macros.items())
    stats = dict(expansion.stats)
    if prepared.decorations_blanked:
        stats["cce_decorations_blanked"] = prepared.decorations_blanked
    prepared.expansion_stats = stats
    # Annotations were located on the expanded text; move them to the
    # original lines their expanded positions map back to, so suppression
    # matching against reported locations keeps working.
    prepared.annotations = tuple(
        replace(ann, line=expansion.origin_line(ann.line))
        for ann in prepared.annotations
    )
    return prepared
