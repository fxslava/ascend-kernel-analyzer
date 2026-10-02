"""Baseline suppression, so CI gates on *new* findings only.

A repository that adopts this analyzer mid-life starts with a backlog.  Gating
on the total count means the gate is red from the first run and stays red, so
nobody looks at it.  Gating on *regressions* - findings absent from a recorded
baseline - makes the signal actionable from day one while the backlog is
burned down separately.

A finding is identified by its file, its code and its message, deliberately
*not* by its line.  Inserting a comment above a kernel shifts every line below
it; treating that as a hundred new findings would make the baseline useless
after the first unrelated edit.  The message carries the concrete offsets and
names, so two genuinely different findings of the same code in one file still
have different fingerprints.

Paths are normalised to forward slashes, and made relative to the directory
the baseline is anchored at when they sit underneath it.  That is what lets a
baseline recorded as ``D:/Projects/vllm-ascend`` match a run that spells the
same tree ``/mnt/d/Projects/vllm-ascend``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .diagnostics import Diagnostic, Severity

__all__ = [
    "Baseline",
    "BaselineError",
    "fingerprint",
    "normalise_path",
    "build_baseline_document",
]

#: Bumped only when a recorded baseline stops being readable by this code.
BASELINE_SCHEMA_VERSION = 1


class BaselineError(Exception):
    """A baseline file could not be read, or is not a baseline at all."""


def normalise_path(raw: str, root: Optional[Path] = None) -> str:
    """Canonical spelling of a source path for fingerprinting.

    Backslashes become forward slashes, and a path under *root* is reduced to
    its relative form.  A ``/mnt/d/...`` prefix is folded to ``d:/...`` so the
    WSL and Windows spellings of one tree agree.
    """
    text = str(raw).replace("\\", "/")
    lowered = text.lower()
    if lowered.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        text = f"{text[5]}:{text[6:]}"
    elif lowered.startswith("/") and len(text) > 2 and text[2] == "/":
        # Git-Bash spells the same drive ``/d/Projects/...``.
        if text[1].isalpha():
            text = f"{text[1]}:{text[2:]}"
    if root is not None:
        try:
            relative = PurePath(text).relative_to(PurePath(str(root).replace("\\", "/")))
        except ValueError:
            pass
        else:
            return relative.as_posix()
    return text


def fingerprint(diag: Diagnostic, root: Optional[Path] = None) -> str:
    """Stable identity of one finding: ``file|code|message``."""
    return "|".join(
        (normalise_path(diag.loc.file, root), diag.code.value, " ".join(diag.message.split()))
    )


@dataclass
class Baseline:
    """A recorded set of accepted findings."""

    fingerprints: Dict[str, int] = field(default_factory=dict)
    #: Where the recorded paths were relative to, if anything.
    root: Optional[Path] = None

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise BaselineError(f"cannot read baseline {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise BaselineError(f"baseline {path} is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise BaselineError(f"baseline {path} must hold a JSON object")

        version = document.get("schema_version", BASELINE_SCHEMA_VERSION)
        if not isinstance(version, int) or version > BASELINE_SCHEMA_VERSION:
            raise BaselineError(
                f"baseline {path} has schema_version {version!r}, and this "
                f"build understands up to {BASELINE_SCHEMA_VERSION}"
            )

        findings = document.get("findings")
        if not isinstance(findings, list):
            raise BaselineError(f"baseline {path} has no 'findings' list")

        root_text = document.get("root")
        root = Path(str(root_text)) if isinstance(root_text, str) and root_text else None

        counts: Dict[str, int] = {}
        for entry in findings:
            if not isinstance(entry, dict):
                continue
            key = entry.get("fingerprint")
            if not isinstance(key, str) or not key:
                continue
            counts[key] = counts.get(key, 0) + int(entry.get("count", 1) or 1)
        return cls(fingerprints=counts, root=root)

    def covers(self, diag: Diagnostic) -> bool:
        return fingerprint(diag, self.root) in self.fingerprints

    def partition(
        self, diagnostics: Sequence[Diagnostic]
    ) -> Tuple[List[Diagnostic], List[Diagnostic]]:
        """Split into ``(regressions, baselined)``.

        A code that fires *more* often than the baseline recorded reports the
        surplus as a regression, so a second instance of a known problem is
        not hidden by the first.
        """
        budget = dict(self.fingerprints)
        regressions: List[Diagnostic] = []
        baselined: List[Diagnostic] = []
        for diag in diagnostics:
            key = fingerprint(diag, self.root)
            remaining = budget.get(key, 0)
            if remaining > 0:
                budget[key] = remaining - 1
                baselined.append(diag)
            else:
                regressions.append(diag)
        return regressions, baselined


def build_baseline_document(
    diagnostics: Iterable[Diagnostic],
    *,
    root: Optional[Path] = None,
    chip: str = "",
    tool_version: str = "",
) -> Dict[str, object]:
    """Render findings as a baseline file, deterministically ordered."""
    counts: Dict[str, int] = {}
    samples: Dict[str, Diagnostic] = {}
    for diag in diagnostics:
        key = fingerprint(diag, root)
        counts[key] = counts.get(key, 0) + 1
        samples.setdefault(key, diag)

    findings = []
    for key in sorted(counts):
        diag = samples[key]
        findings.append(
            {
                "fingerprint": key,
                "count": counts[key],
                "code": diag.code.value,
                "severity": diag.severity.value,
                "file": normalise_path(diag.loc.file, root),
                # Recorded for a human reading the file; never matched on.
                "line": diag.loc.line,
                "message": " ".join(diag.message.split()),
            }
        )

    document: Dict[str, object] = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "findings": findings,
    }
    if root is not None:
        document["root"] = str(root).replace("\\", "/")
    if chip:
        document["chip"] = chip
    if tool_version:
        document["tool_version"] = tool_version
    document["counts"] = {
        "findings": len(findings),
        "occurrences": sum(counts.values()),
        "fatal": sum(
            1 for d in samples.values() if d.severity is Severity.FATAL
        ),
    }
    return document
