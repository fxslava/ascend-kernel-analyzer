"""Small source scanners shared by architecture-specific checkers."""

from __future__ import annotations

import re
import ast
import operator
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from ..diagnostics import SourceLoc
from ..parsing.preprocess import non_code_spans

__all__ = ["SourceCall", "code_only", "iter_calls", "split_arguments", "source_loc"]


@dataclass(frozen=True)
class SourceCall:
    name: str
    start: int
    end: int
    template: Optional[str]
    arguments: Tuple[str, ...]


def code_only(source: str) -> str:
    """Blank comments and string literals, retaining offsets and newlines."""
    raw = source.encode("utf-8")
    masked = bytearray(raw)
    for start, end in non_code_spans(raw):
        for index in range(start, end):
            if raw[index] not in (10, 13):
                masked[index] = 32
    # Keep character offsets even when a comment contained multibyte text.
    return "".join(
        char if masked[offset] != 32 or char in " \r\n" else " "
        for offset, char in _character_offsets(source)
    )


def _character_offsets(source: str):
    offset = 0
    for char in source:
        yield offset, char
        offset += len(char.encode("utf-8"))


def _matching(source: str, opening: int, left: str, right: str) -> Optional[int]:
    depth = 0
    quote: Optional[str] = None
    escaped = False
    for index in range(opening, len(source)):
        char = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth == 0:
                return index
    return None


def split_arguments(text: str) -> Tuple[str, ...]:
    """Split C++ call arguments at top-level commas."""
    if not text.strip():
        return ()
    parts: List[str] = []
    start = 0
    parens = brackets = braces = angles = 0
    quote: Optional[str] = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            parens += 1
        elif char == ")":
            parens = max(0, parens - 1)
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets = max(0, brackets - 1)
        elif char == "{":
            braces += 1
        elif char == "}":
            braces = max(0, braces - 1)
        elif (char == "<" and not (parens or brackets or braces)
              and index > 0 and (text[index - 1].isalnum() or text[index - 1] in "_:>")
              and text[index:index + 2] not in {"<<", "<="}
              and text[index - 1] != "<"):
            angles += 1
        elif char == ">" and angles:
            angles -= 1
        elif char == "," and not (parens or brackets or braces or angles):
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return tuple(parts)


def iter_calls(source: str, name: str) -> Iterator[SourceCall]:
    """Yield balanced invocations of *name*, including optional template args."""
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    for match in pattern.finditer(source):
        cursor = match.end()
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        template: Optional[str] = None
        if cursor < len(source) and source[cursor] == "<":
            close = _matching(source, cursor, "<", ">")
            if close is None:
                continue
            template = source[cursor + 1 : close].strip()
            cursor = close + 1
            while cursor < len(source) and source[cursor].isspace():
                cursor += 1
        if cursor >= len(source) or source[cursor] != "(":
            continue
        close = _matching(source, cursor, "(", ")")
        if close is None:
            continue
        yield SourceCall(
            name=name,
            start=match.start(),
            end=close + 1,
            template=template,
            arguments=split_arguments(source[cursor + 1 : close]),
        )


def source_loc(path: str, source: str, offset: int) -> SourceLoc:
    """Build a source location for a character offset in *source*."""
    line = source.count("\n", 0, offset) + 1
    line_start = source.rfind("\n", 0, offset) + 1
    line_end = source.find("\n", offset)
    if line_end < 0:
        line_end = len(source)
    return SourceLoc(
        file=path,
        line=line,
        column=offset - line_start + 1,
        snippet=source[line_start:line_end].strip()[:240],
    )


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.floordiv,
    ast.Mod: operator.mod,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
}


def constant_int(expression: str, names: Dict[str, int]) -> Optional[int]:
    """Fold a restricted integer expression without executing source code."""
    cleaned = re.sub(r"(?<=\d)[uUlL]+\b", "", expression.strip())
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return names.get(cleaned)

    def evaluate(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return {"true": 1, "false": 0}.get(node.id, names.get(node.id))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            if value is None:
                return None
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            function = _BIN_OPS.get(type(node.op))
            left, right = evaluate(node.left), evaluate(node.right)
            if function is None or left is None or right is None:
                return None
            if isinstance(node.op, (ast.LShift, ast.RShift)) and not 0 <= right <= 63:
                return None
            try:
                return int(function(left, right))
            except (ArithmeticError, ValueError):
                return None
        return None

    return evaluate(tree)

