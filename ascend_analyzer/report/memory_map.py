"""Memory footprint maps: the SRAM layout as a picture.

A table of byte offsets is correct but hard to read; a layout bug is usually
obvious the moment you *see* the gap or the overlap.  This module turns the
resolved tensor set into per-domain :class:`DomainMap` objects and renders them
either as an ASCII chart for the terminal or as a proportional bar chart for
the HTML report.

Colliding byte ranges are marked distinctly, so a double-buffering bug shows up
as two bars visibly sharing columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..hardware import HardwareModel, PhysicalDomain
from ..ir import KernelIR
from ..symbolic import render

__all__ = ["Segment", "DomainMap", "Glyphs", "build_memory_maps", "render_ascii_map"]


@dataclass(frozen=True)
class Segment:
    """One tensor's resolved byte range within a domain."""

    name: str
    start: int
    end: int
    dtype: Optional[str] = None
    position: Optional[str] = None
    line: int = 0
    #: Names of other segments this one overlaps.
    collides_with: Tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def collides(self) -> bool:
        return bool(self.collides_with)

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "dtype": self.dtype,
            "position": self.position,
            "line": self.line,
            "collides_with": list(self.collides_with),
        }


@dataclass
class DomainMap:
    """The resolved layout of one physical memory domain."""

    domain: PhysicalDomain
    capacity_bytes: int
    description: str = ""
    base_alignment: int = 32
    segments: List[Segment] = field(default_factory=list)
    #: Tensors in this domain whose offset or size could not be resolved.
    unresolved: List[str] = field(default_factory=list)

    @property
    def high_water(self) -> int:
        return max((s.end for s in self.segments), default=0)

    @property
    def allocated(self) -> int:
        return sum(s.size for s in self.segments)

    @property
    def utilization(self) -> float:
        if self.capacity_bytes <= 0:
            return 0.0
        return self.high_water / self.capacity_bytes

    @property
    def free_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.high_water)

    @property
    def collision_count(self) -> int:
        return sum(1 for s in self.segments if s.collides)

    def gaps(self) -> List[Tuple[int, int]]:
        """Unclaimed byte ranges below the high-water mark."""
        holes: List[Tuple[int, int]] = []
        cursor = 0
        for segment in sorted(self.segments, key=lambda s: s.start):
            if segment.start > cursor:
                holes.append((cursor, segment.start))
            cursor = max(cursor, segment.end)
        return holes

    def to_json(self) -> Dict[str, object]:
        return {
            "domain": self.domain.value,
            "description": self.description,
            "capacity_bytes": self.capacity_bytes,
            "high_water_bytes": self.high_water,
            "allocated_bytes": self.allocated,
            "free_bytes": self.free_bytes,
            "utilization": round(self.utilization, 4),
            "base_alignment": self.base_alignment,
            "segments": [s.to_json() for s in self.segments],
            "gaps": [{"start": a, "end": b, "size": b - a} for a, b in self.gaps()],
            "unresolved": list(self.unresolved),
        }


# ---------------------------------------------------------------------------
# Glyph sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Glyphs:
    """Characters used to draw the chart, with an ASCII-safe fallback."""

    fill: str = "#"
    collide: str = "X"
    empty: str = "."
    partial: str = "+"
    h: str = "-"
    v: str = "|"
    tl: str = "+"
    tr: str = "+"
    bl: str = "+"
    br: str = "+"

    @classmethod
    def unicode(cls) -> "Glyphs":
        return cls(
            fill="█",      # full block
            collide="▒",   # medium shade
            empty="·",     # middle dot
            partial="▌",   # left half block
            h="─",
            v="│",
            tl="┌",
            tr="┐",
            bl="└",
            br="┘",
        )

    @classmethod
    def ascii(cls) -> "Glyphs":
        return cls()

    @classmethod
    def best_for(cls, stream_encoding: Optional[str], force_ascii: bool = False) -> "Glyphs":
        """Pick the richest glyph set the output stream can actually encode."""
        if force_ascii:
            return cls.ascii()
        candidate = cls.unicode()
        probe = "".join(
            [candidate.fill, candidate.collide, candidate.empty, candidate.partial,
             candidate.h, candidate.v, candidate.tl, candidate.tr,
             candidate.bl, candidate.br]
        )
        try:
            probe.encode(stream_encoding or "ascii")
        except (UnicodeEncodeError, LookupError):
            return cls.ascii()
        return candidate


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_memory_maps(kernel: KernelIR, hardware: HardwareModel) -> List[DomainMap]:
    """Group a kernel's resolved tensors into one :class:`DomainMap` per domain."""
    maps: Dict[PhysicalDomain, DomainMap] = {}

    for tensor in kernel.tensors.values():
        if not tensor.is_sram:
            continue
        spec = hardware.spec_for(tensor.domain)
        entry = maps.get(tensor.domain)
        if entry is None:
            entry = DomainMap(
                domain=tensor.domain,
                capacity_bytes=spec.capacity_bytes if spec else 0,
                description=spec.description if spec else "",
                base_alignment=spec.base_alignment if spec else 32,
            )
            maps[tensor.domain] = entry

        offset, size = tensor.offset_value, tensor.size_value
        if offset is None or size is None:
            entry.unresolved.append(
                f"{tensor.name} (offset={render(tensor.byte_offset)}, "
                f"size={render(tensor.byte_size)})"
            )
            continue
        entry.segments.append(
            Segment(
                name=tensor.name,
                start=offset,
                end=offset + size,
                dtype=tensor.dtype,
                position=tensor.position.value if tensor.position else None,
                line=tensor.loc.line,
            )
        )

    for entry in maps.values():
        entry.segments = _annotate_collisions(entry.segments)
        entry.segments.sort(key=lambda s: (s.start, s.name))

    return [maps[d] for d in hardware.tracked_sram_domains() if d in maps]


def _annotate_collisions(segments: Sequence[Segment]) -> List[Segment]:
    """Record, for each segment, which others it overlaps."""
    out: List[Segment] = []
    for segment in segments:
        others = tuple(
            other.name
            for other in segments
            if other is not segment
            and segment.start < other.end
            and other.start < segment.end
        )
        out.append(
            Segment(
                name=segment.name,
                start=segment.start,
                end=segment.end,
                dtype=segment.dtype,
                position=segment.position,
                line=segment.line,
                collides_with=others,
            )
        )
    return out


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------


#: Fixed column widths of the chart's text columns.
_LABEL_WIDTH = 13
_EXTENT_WIDTH = 26


def render_ascii_map(
    domain_map: DomainMap,
    *,
    width: int = 48,
    glyphs: Optional[Glyphs] = None,
    colorize=None,
) -> List[str]:
    """Render one domain as a list of terminal lines.

    The horizontal axis spans ``[0, window)`` where the window is the
    high-water mark rounded up, not the full capacity - a 2 KiB layout inside a
    192 KiB buffer would otherwise collapse into a single invisible column.
    The capacity is reported numerically instead.
    """
    g = glyphs or Glyphs.ascii()
    paint = colorize or (lambda text, _style: text)
    width = max(8, width)

    window = _window_for(domain_map)

    # (plain_text, coloured_text) pairs; padding is computed from plain_text so
    # ANSI escapes never disturb the box alignment.
    body: List[Tuple[str, str]] = []

    capacity_kib = domain_map.capacity_bytes / 1024
    body.append(
        _plain(
            f"capacity {domain_map.capacity_bytes} B ({capacity_kib:.1f} KiB)"
            f"  high-water {domain_map.high_water} B ({domain_map.utilization:.2%})"
            f"  {len(domain_map.segments)} tensors"
        )
    )
    if domain_map.segments:
        scale = max(1, -(-window // width))
        body.append(_plain(f"window 0x00000..0x{window:05X}   1 column = {scale} B"))
        body.append(_plain(""))
    elif domain_map.unresolved:
        body.append(
            _plain("no tensor in this domain has a statically resolved address")
        )
        body.append(_plain(""))

    for segment in domain_map.segments:
        bar = _render_bar(segment, domain_map.segments, window, width, g)
        style = "collision" if segment.collides else "ok"
        label = f"{segment.name[:_LABEL_WIDTH - 1]:<{_LABEL_WIDTH}}"
        extent = (
            f"[0x{segment.start:05X},0x{segment.end:05X}) {segment.size:>6} B"
        ).rjust(_EXTENT_WIDTH)
        body.append((f"{label}{bar} {extent}", f"{label}{paint(bar, style)} {extent}"))

    for gap_start, gap_end in domain_map.gaps():
        bar = _render_span(gap_start, gap_end, window, width, g.empty)
        label = f"{'(gap)':<{_LABEL_WIDTH}}"
        extent = f"{gap_end - gap_start} B free at 0x{gap_start:05X}".rjust(
            _EXTENT_WIDTH
        )
        body.append(_plain(f"{label}{bar} {extent}"))

    for note in domain_map.unresolved:
        body.append(_plain(f"{'(unresolved)':<{_LABEL_WIDTH}}{note}"))

    if domain_map.collision_count:
        body.append(
            _plain(
                f"{g.collide} = shared with another live tensor "
                f"({domain_map.collision_count} tensors affected)"
            )
        )

    inner = max(
        _LABEL_WIDTH + width + 1 + _EXTENT_WIDTH,
        max((len(plain) for plain, _ in body), default=0),
    ) + 2

    title = f" {domain_map.domain.value}"
    if domain_map.description:
        title += f" · {domain_map.description}" if g.v != "|" else f" - {domain_map.description}"
    title = title[: inner - 4] + " "

    lines = [f"{g.tl}{g.h}{title}{g.h * max(0, inner - len(title) - 1)}{g.tr}"]
    for plain, coloured in body:
        pad = " " * max(0, inner - 1 - len(plain))
        lines.append(f"{g.v} {coloured}{pad}{g.v}")
    lines.append(f"{g.bl}{g.h * inner}{g.br}")
    return lines


def _plain(text: str) -> Tuple[str, str]:
    return (text, text)


def _window_for(domain_map: DomainMap) -> int:
    """Choose a readable upper bound for the horizontal axis."""
    high = domain_map.high_water
    align = max(domain_map.base_alignment, 32)
    if high <= 0:
        return align
    rounded = -(-high // align) * align
    capacity = domain_map.capacity_bytes or rounded
    return min(max(rounded, align), capacity)


def _column_range(start: int, end: int, window: int, width: int) -> Tuple[int, int]:
    """Map a byte range to an inclusive-exclusive column range.

    The start floors and the end ceils, so a range that covers any part of a
    column claims that whole column.  That matters for collision rendering: a
    12-byte overlap inside a 2 KiB window would round away entirely under
    nearest-integer mapping, hiding exactly the bug the chart exists to show.
    """
    if window <= 0:
        return (0, 1)
    lo = min(width - 1, max(0, int(start * width // window)))
    hi = max(lo + 1, min(width, -(-end * width // window)))
    return (lo, hi)


def _collision_columns(
    segment: Segment, segments: Sequence[Segment], window: int, width: int
) -> set:
    """Columns covering a byte range this segment genuinely shares.

    Derived from exact byte intersections rather than from column overlap:
    two *adjacent* ranges share a boundary column once the end offset is
    ceiled, and marking those as a collision would cry wolf on a correct
    back-to-back layout.
    """
    columns: set = set()
    for other in segments:
        if other is segment:
            continue
        lo = max(segment.start, other.start)
        hi = min(segment.end, other.end)
        if lo < hi:
            first, last = _column_range(lo, hi, window, width)
            columns.update(range(first, last))
    return columns


def _render_bar(
    segment: Segment,
    segments: Sequence[Segment],
    window: int,
    width: int,
    glyphs: Glyphs,
) -> str:
    lo, hi = _column_range(segment.start, segment.end, window, width)
    collisions = _collision_columns(segment, segments, window, width)
    cells = []
    for column in range(width):
        if column < lo or column >= hi:
            cells.append(" ")
        elif column in collisions:
            cells.append(glyphs.collide)
        else:
            cells.append(glyphs.fill)
    return "".join(cells)


def _render_span(
    start: int, end: int, window: int, width: int, glyph: str
) -> str:
    lo, hi = _column_range(start, end, window, width)
    return "".join(glyph if lo <= c < hi else " " for c in range(width))
