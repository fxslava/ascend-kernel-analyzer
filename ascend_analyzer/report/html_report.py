"""Self-contained HTML report with a proportional memory footprint chart.

The output is a single file with no external assets, so it can be attached to
a code review or published as a CI artifact.  The chart draws each domain as a
track with one absolutely-positioned bar per tensor, which makes gaps and
overlaps immediately visible; colliding bars are hatched red.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence

from ..diagnostics import Diagnostic, Severity
from ..hardware import HardwareModel
from ..ir import AnalysisUnit
from .memory_map import DomainMap, build_memory_maps

__all__ = ["build_html_report"]


_CSS = """
:root {
  --bg: #0f1419; --panel: #171d24; --line: #263040; --text: #dbe3ec;
  --muted: #8b9bb0; --fatal: #ff5f6b; --warn: #ffb547; --info: #4fc3f7;
  --ok: #4ade80; --bar: #3d7eff; --bar2: #22d3ee;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --bg: #f7f9fc; --panel: #ffffff; --line: #dde4ee; --text: #16202c;
    --muted: #5b6b80; --fatal: #d12b39; --warn: #a66200; --info: #0b6e99;
    --ok: #15803d; --bar: #2563eb; --bar2: #0891b2;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 64px; background: var(--bg); color: var(--text);
  font: 14px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 32px 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid var(--line); }
h3 { font-size: 14px; margin: 20px 0 8px; font-weight: 600; }
.sub { color: var(--muted); margin: 0 0 20px; font-size: 13px; }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px; margin-bottom: 8px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px 14px; }
.card .k { color: var(--muted); font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.06em; }
.card .v { font-size: 18px; font-weight: 600; margin-top: 2px; }
.v.fatal { color: var(--fatal); } .v.warn { color: var(--warn); }
.v.ok { color: var(--ok); } .v.info { color: var(--info); }
.verdict { display: inline-block; padding: 4px 12px; border-radius: 999px;
  font-weight: 650; font-size: 12px; letter-spacing: 0.04em; }
.verdict.rejected { background: rgba(255,95,107,0.15); color: var(--fatal); }
.verdict.accepted_with_warnings { background: rgba(255,181,71,0.15); color: var(--warn); }
.verdict.accepted { background: rgba(74,222,128,0.15); color: var(--ok); }

.track { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px; margin-bottom: 16px; }
.track-head { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline;
  justify-content: space-between; margin-bottom: 10px; }
.track-head .name { font-weight: 650; }
.track-head .stat { color: var(--muted); font-size: 12px; }
.gauge { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden;
  margin-bottom: 14px; }
.gauge > i { display: block; height: 100%; background: var(--bar); }
.rows { display: grid; gap: 5px; }
.row { display: grid; grid-template-columns: 116px 1fr 190px; gap: 10px;
  align-items: center; font-size: 12px; }
.row .label { font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lane { position: relative; height: 18px; background: var(--line);
  border-radius: 3px; overflow: hidden; }
.lane > span { position: absolute; top: 0; bottom: 0; border-radius: 2px;
  background: var(--bar); min-width: 2px; }
.lane > span.alt { background: var(--bar2); }
.lane > span.collide { background: repeating-linear-gradient(45deg,
  var(--fatal) 0 5px, rgba(0,0,0,0.28) 5px 10px); }
.row .extent { color: var(--muted); font-family: ui-monospace, Consolas, monospace;
  text-align: right; white-space: nowrap; }
.axis { display: flex; justify-content: space-between; color: var(--muted);
  font-size: 11px; font-family: ui-monospace, Consolas, monospace; margin-top: 8px; }

.diag { background: var(--panel); border: 1px solid var(--line);
  border-left: 3px solid var(--muted); border-radius: 6px; padding: 12px 14px;
  margin-bottom: 10px; }
.diag.FATAL { border-left-color: var(--fatal); }
.diag.WARNING { border-left-color: var(--warn); }
.diag.INFO { border-left-color: var(--info); }
.diag-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  margin-bottom: 6px; }
.tag { font-size: 10.5px; font-weight: 700; letter-spacing: 0.05em;
  padding: 2px 7px; border-radius: 4px; background: var(--line); }
.tag.FATAL { background: rgba(255,95,107,0.18); color: var(--fatal); }
.tag.WARNING { background: rgba(255,181,71,0.18); color: var(--warn); }
.tag.INFO { background: rgba(79,195,247,0.18); color: var(--info); }
.diag-head .code { font-family: ui-monospace, Consolas, monospace;
  color: var(--muted); font-size: 12px; }
.diag-head .title { font-weight: 600; }
.loc { color: var(--muted); font-size: 12px;
  font-family: ui-monospace, Consolas, monospace; margin-bottom: 6px; }
.msg { white-space: pre-wrap; margin-bottom: 8px; }
.fix { background: rgba(127,127,127,0.09); border-radius: 5px; padding: 9px 11px;
  white-space: pre-wrap; font-size: 13px; }
.fix b { color: var(--ok); }
pre.frame { margin: 8px 0; padding: 9px 11px; overflow-x: auto;
  background: rgba(127,127,127,0.09); border-radius: 5px; font-size: 12px;
  font-family: ui-monospace, Consolas, monospace; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.05em; }
td.num { text-align: right; font-family: ui-monospace, Consolas, monospace; }
.pill { font-size: 11px; padding: 1px 6px; border-radius: 4px;
  background: var(--line); font-family: ui-monospace, Consolas, monospace; }
.empty { color: var(--muted); font-style: italic; }
footer { color: var(--muted); font-size: 12px; margin-top: 36px;
  padding-top: 12px; border-bottom: 0; border-top: 1px solid var(--line); }
"""


def build_html_report(
    unit: AnalysisUnit,
    hardware: HardwareModel,
    diagnostics: Sequence[Diagnostic],
    artifacts: Optional[Dict[str, object]] = None,
    *,
    solver_name: str = "interval",
    tool_version: str = "0.1.0",
) -> str:
    """Render the complete analysis as one standalone HTML document."""
    artifacts = artifacts or {}
    fatal = sum(1 for d in diagnostics if d.severity is Severity.FATAL)
    warning = sum(1 for d in diagnostics if d.severity is Severity.WARNING)
    info = sum(1 for d in diagnostics if d.severity is Severity.INFO)
    verdict = (
        "rejected" if fatal else "accepted_with_warnings" if warning else "accepted"
    )
    verdict_text = {
        "rejected": "REJECTED",
        "accepted_with_warnings": "ACCEPTED WITH WARNINGS",
        "accepted": "ACCEPTED",
    }[verdict]

    parts: List[str] = []
    e = html.escape

    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>Kernel Analysis - {e(_basename(unit.path))}</title>")
    parts.append(f"<style>{_CSS}</style></head><body><div class='wrap'>")

    parts.append("<h1>Ascend Static Kernel Analyzer</h1>")
    parts.append(
        f"<p class='sub'>{e(unit.path)} &middot; target "
        f"{e(hardware.chip.display_name)} &middot; solver {e(solver_name)}</p>"
    )

    parts.append("<div class='meta'>")
    parts.append(_card("verdict", f"<span class='verdict {verdict}'>{verdict_text}</span>"))
    parts.append(_card("fatal", str(fatal), "fatal" if fatal else "ok"))
    parts.append(_card("warnings", str(warning), "warn" if warning else "ok"))
    parts.append(_card("info", str(info), "info"))
    parts.append("</div>")

    if hardware.chip.provisional:
        parts.append(
            f"<p class='sub'>Chip profile is provisional: {e(hardware.chip.notes)}</p>"
        )

    # -- memory footprint ---------------------------------------------------
    parts.append("<h2>SRAM memory footprint</h2>")
    any_map = False
    for kernel in unit.kernels:
        maps = build_memory_maps(kernel, hardware)
        if not maps:
            continue
        any_map = True
        parts.append(f"<h3>{e(kernel.name)}</h3>")
        for domain_map in maps:
            parts.append(_render_track(domain_map))
    if not any_map:
        parts.append("<p class='empty'>No on-core SRAM tensors were resolved.</p>")

    # -- findings -----------------------------------------------------------
    parts.append(f"<h2>Findings ({len(diagnostics)})</h2>")
    if not diagnostics:
        parts.append(
            "<p class='empty'>No findings. Layout and synchronisation verified.</p>"
        )
    for diag in diagnostics:
        parts.append(_render_diagnostic(diag, unit))

    # -- synchronisation ----------------------------------------------------
    sync_rendered = False
    for kernel in unit.kernels:
        graph = artifacts.get(f"sync_graph::{kernel.name}")
        if not isinstance(graph, dict) or not graph.get("nodes"):
            continue
        if not sync_rendered:
            parts.append("<h2>Pipeline synchronisation</h2>")
            sync_rendered = True
        parts.append(f"<h3>{e(kernel.name)}</h3>")
        parts.append(_render_sync(graph))

    parts.append(
        f"<footer>Generated by ascend-kernel-analyzer {e(tool_version)}. "
        f"Diagnostic codes are stable; message text is not. "
        f"Capacity and alignment rules come from the "
        f"<code>{e(hardware.chip.name)}</code> chip profile.</footer>"
    )
    parts.append("</div></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Fragments
# ---------------------------------------------------------------------------


def _card(key: str, value: str, style: str = "") -> str:
    cls = f" {style}" if style else ""
    return (
        f"<div class='card'><div class='k'>{html.escape(key)}</div>"
        f"<div class='v{cls}'>{value}</div></div>"
    )


def _render_track(domain_map: DomainMap) -> str:
    e = html.escape
    window = max(domain_map.high_water, 1)
    rows: List[str] = []

    for index, segment in enumerate(domain_map.segments):
        left = segment.start / window * 100
        span = max(segment.size / window * 100, 0.4)
        classes = "collide" if segment.collides else ("alt" if index % 2 else "")
        title = (
            f"{segment.name}: [0x{segment.start:X}, 0x{segment.end:X}) "
            f"{segment.size} B"
            + (f" - collides with {', '.join(segment.collides_with)}"
               if segment.collides else "")
        )
        rows.append(
            "<div class='row'>"
            f"<div class='label' title='{e(title)}'>{e(segment.name)}"
            + (f" <span class='pill'>{e(segment.position)}</span>"
               if segment.position else "")
            + "</div>"
            f"<div class='lane'><span class='{classes}' "
            f"style='left:{left:.4f}%;width:{span:.4f}%' title='{e(title)}'></span></div>"
            f"<div class='extent'>0x{segment.start:05X}&ndash;0x{segment.end:05X} "
            f"&middot; {segment.size} B</div>"
            "</div>"
        )

    for gap_start, gap_end in domain_map.gaps():
        left = gap_start / window * 100
        span = max((gap_end - gap_start) / window * 100, 0.4)
        rows.append(
            "<div class='row'><div class='label empty'>(gap)</div>"
            f"<div class='lane'><span style='left:{left:.4f}%;width:{span:.4f}%;"
            "background:repeating-linear-gradient(45deg,var(--muted) 0 3px,"
            "transparent 3px 7px);opacity:0.5'></span></div>"
            f"<div class='extent'>{gap_end - gap_start} B unclaimed</div></div>"
        )

    for note in domain_map.unresolved:
        rows.append(
            "<div class='row'><div class='label empty'>(unresolved)</div>"
            f"<div class='lane'></div><div class='extent'>{e(note)}</div></div>"
        )

    collision_note = (
        f" &middot; <span style='color:var(--fatal)'>"
        f"{domain_map.collision_count} colliding</span>"
        if domain_map.collision_count
        else ""
    )
    return (
        "<div class='track'>"
        "<div class='track-head'>"
        f"<span class='name'>{e(domain_map.domain.value)}"
        + (f" <span class='stat'>{e(domain_map.description)}</span>"
           if domain_map.description else "")
        + "</span>"
        f"<span class='stat'>{domain_map.high_water} B of "
        f"{domain_map.capacity_bytes} B "
        f"({domain_map.capacity_bytes / 1024:.0f} KiB) &middot; "
        f"{domain_map.utilization:.2%} used &middot; "
        f"{len(domain_map.segments)} tensors{collision_note}</span>"
        "</div>"
        f"<div class='gauge'><i style='width:"
        f"{min(100.0, domain_map.utilization * 100):.3f}%'></i></div>"
        f"<div class='rows'>{''.join(rows)}</div>"
        f"<div class='axis'><span>0x00000</span>"
        f"<span>window = high-water mark, 0x{domain_map.high_water:05X}</span>"
        f"<span>0x{domain_map.high_water:05X}</span></div>"
        "</div>"
    )


def _render_diagnostic(diag: Diagnostic, unit: AnalysisUnit) -> str:
    e = html.escape
    severity = diag.severity.value
    frame = ""
    source_line = unit.line_text(diag.loc.line)
    if source_line.strip():
        stripped = source_line.lstrip()
        indent_removed = len(source_line) - len(stripped)
        caret_col = max(0, diag.loc.column - 1 - indent_removed)
        frame = (
            f"<pre class='frame'>{diag.loc.line:>6} | {e(stripped.rstrip())}\n"
            f"{'':>6} | {' ' * caret_col}^</pre>"
        )

    related = "".join(
        f"<div class='loc'>see also {e(loc.short)} &mdash; {e(label)}</div>"
        for label, loc in diag.related
    )
    fix = (
        f"<div class='fix'><b>fix:</b> {e(diag.remediation)}</div>"
        if diag.remediation
        else ""
    )
    return (
        f"<div class='diag {severity}'>"
        f"<div class='diag-head'><span class='tag {severity}'>{severity}</span>"
        f"<span class='code'>{e(diag.code.value)}</span>"
        f"<span class='title'>{e(diag.title)}</span>"
        f"<span class='pill'>{e(diag.hardware_domain)}</span></div>"
        f"<div class='loc'>{e(str(diag.loc))}</div>"
        f"<div class='msg'>{e(diag.message)}</div>"
        f"{frame}{related}{fix}</div>"
    )


def _render_sync(graph: Dict[str, object]) -> str:
    e = html.escape
    nodes = graph.get("nodes") or []
    pairs = graph.get("sync_pairs") or []
    acyclic = graph.get("acyclic")
    cycles = graph.get("cycles") or []

    status = (
        "<span style='color:var(--ok)'>acyclic &mdash; no circular wait</span>"
        if acyclic
        else f"<span style='color:var(--fatal)'>cyclic &mdash; "
             f"{len(cycles)} circular wait(s)</span>"
    )

    rows = "".join(
        "<tr>"
        f"<td class='num'>{e(str(pair.get('set_line')))}</td>"
        f"<td><span class='pill'>{e(str(pair.get('channel', {}).get('route')))}</span></td>"
        f"<td class='num'>{e(str(pair.get('channel', {}).get('event_id')))}</td>"
        f"<td class='num'>{e(str(pair.get('wait_line')))}</td>"
        f"<td>{'loop-carried' if pair.get('loop_carried') else 'same iteration'}</td>"
        f"<td class='num'>{e(str(pair.get('tokens')))}</td>"
        "</tr>"
        for pair in pairs
    )

    order = graph.get("topological_order")
    order_note = (
        f"<p class='sub'>A valid issue order exists over "
        f"{len(order)} synchronisation points, which is the proof of deadlock "
        f"freedom.</p>"
        if order
        else ""
    )

    return (
        "<div class='track'>"
        f"<div class='track-head'><span class='name'>dependency graph</span>"
        f"<span class='stat'>{len(nodes)} flag ops &middot; {len(pairs)} matched "
        f"pairs &middot; {status}</span></div>"
        f"{order_note}"
        "<table><thead><tr><th>set line</th><th>route</th><th>event</th>"
        "<th>wait line</th><th>distance</th><th>initial tokens</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]
