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
]


#: Address-space qualifiers that bind a declaration to a physical domain.
_ADDRESS_SPACES: Tuple[str, ...] = tuple(ADDRESS_SPACE_TO_DOMAIN)

#: Attributes that mark a function as an AI Core kernel entry point.
KERNEL_ATTRIBUTES: Tuple[str, ...] = ("__global__", "__aicore__", "__aicpu__")

#: Extra compiler extensions that are noise for our purposes but break parsing.
_OTHER_EXTENSIONS: Tuple[str, ...] = ("__restrict__", "__inline__", "__forceinline__")

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
            if len(self.rewritten) != len(self.original):
                raise AssertionError(
                    "qualifier rewrite changed source length "
                    f"({len(self.original)} -> {len(self.rewritten)}); "
                    "source locations would be wrong"
                )
            if len(self.rewritten_bytes) != len(self.original.encode("utf-8")):
                raise AssertionError(
                    "qualifier rewrite changed the byte length of the source; "
                    "tree-sitter offsets would not line up"
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


#: Public alias: the byte-span scanner is reused by the macro expander.
non_code_spans = _non_code_spans


def prepare_source(path: str, source: str) -> PreparedSource:
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
    skip = _non_code_spans(raw)

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
    if not expansion.changed:
        return prepare_source(path, source)

    prepared = prepare_source(path, expansion.text)
    prepared.original = source
    prepared.line_origins = expansion.line_origins
    prepared.macro_object_defs = tuple(expansion.object_macros.items())
    prepared.expansion_stats = dict(expansion.stats)
    # Annotations were located on the expanded text; move them to the
    # original lines their expanded positions map back to, so suppression
    # matching against reported locations keeps working.
    prepared.annotations = tuple(
        replace(ann, line=expansion.origin_line(ann.line))
        for ann in prepared.annotations
    )
    return prepared
