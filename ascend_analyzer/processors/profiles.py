"""Hardware model for Huawei Ascend / DaVinci AI Core architectures.

This module encodes the *static* hardware facts the analyzer reasons about:

* the physical on-chip SRAM domains (UB, L1, L0A/L0B/L0C, ...) and their
  capacities and alignment requirements,
* the mapping from the Ascend C ``TPosition`` logical tensor position onto
  those physical domains,
* the hardware instruction pipelines (``PIPE_MTE2``, ``PIPE_V``, ...), and
* the ``HardEvent`` synchronisation routes that connect pipeline pairs.

Capacity numbers are expressed as *chip profiles*.  Huawei does not publish a
single authoritative table for every SKU, so profiles carry a ``provisional``
flag and a ``notes`` string; the analyzer surfaces that provenance in its
report and every value can be overridden from the CLI or a JSON profile file.
Treat the shipped numbers as sensible defaults, not as gospel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

KIB = 1024

__all__ = [
    "KIB",
    "Pipe",
    "REAL_PIPES",
    "PhysicalDomain",
    "TPosition",
    "HardEventRoute",
    "DomainSpec",
    "ChipSpec",
    "HardwareModel",
    "CHIP_PROFILES",
    "TPOSITION_TO_DOMAIN",
    "ADDRESS_SPACE_TO_DOMAIN",
    "DOMAIN_TO_DEFAULT_TPOSITION",
    "resolve_chip",
    "DEFAULT_TRANSFER_PIPES",
]


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


class Pipe(Enum):
    """A DaVinci AI Core hardware instruction pipeline.

    Each pipeline consumes its own instruction queue strictly *in order*;
    different pipelines run concurrently and are only ordered with respect to
    one another by explicit synchronisation (``SetFlag``/``WaitFlag``) or by a
    ``PipeBarrier``.
    """

    S = "PIPE_S"        # scalar unit
    V = "PIPE_V"        # vector unit  (reads/writes UB)
    M = "PIPE_M"        # cube / matmul unit (reads L0A+L0B, writes L0C)
    MTE1 = "PIPE_MTE1"  # L1 -> L0A/L0B ("load data", fractal rearrange)
    MTE2 = "PIPE_MTE2"  # GM -> L1/UB   ("move in")
    MTE3 = "PIPE_MTE3"  # UB -> GM/L1   ("move out")
    FIX = "PIPE_FIX"    # fixpipe: L0C -> UB/GM (quant, relu, channel split)
    ALL = "PIPE_ALL"    # pseudo-pipe: full barrier across every pipeline

    @property
    def short(self) -> str:
        """Short mnemonic as it appears inside a ``HardEvent`` name."""
        return self.name

    @property
    def is_real(self) -> bool:
        """``False`` for the ``PIPE_ALL`` pseudo-pipe."""
        return self is not Pipe.ALL

    @classmethod
    def parse(cls, token: str) -> Optional["Pipe"]:
        """Parse ``"PIPE_MTE2"``, ``"MTE2"``, ``"pipe_t::PIPE_V"`` -> ``Pipe``."""
        t = token.strip().rsplit("::", 1)[-1].strip().upper()
        if not t:
            return None
        if not t.startswith("PIPE_"):
            t = "PIPE_" + t
        for pipe in cls:
            if pipe.value == t:
                return pipe
        return None


#: Pipelines that participate in dependency analysis, in a stable report order.
REAL_PIPES: Tuple[Pipe, ...] = (
    Pipe.S,
    Pipe.MTE2,
    Pipe.MTE1,
    Pipe.M,
    Pipe.V,
    Pipe.FIX,
    Pipe.MTE3,
)


# ---------------------------------------------------------------------------
# Memory domains
# ---------------------------------------------------------------------------


class PhysicalDomain(Enum):
    """A physically distinct memory region on (or attached to) the AI Core."""

    UB = "UB"      # Unified Buffer - vector unit scratchpad
    L1 = "L1"      # L1 Buffer     - shared staging buffer
    L0A = "L0A"    # Cube left-matrix input buffer
    L0B = "L0B"    # Cube right-matrix input buffer
    L0C = "L0C"    # Cube accumulator / output buffer
    BT = "BT"      # Bias table
    FB = "FB"      # Fixpipe parameter buffer
    GM = "GM"      # off-core global (HBM) memory
    UNKNOWN = "UNKNOWN"

    @property
    def is_on_core_sram(self) -> bool:
        """``True`` for capacity-constrained on-core SRAM domains."""
        return self in _ON_CORE_SRAM


_ON_CORE_SRAM: FrozenSet[PhysicalDomain] = frozenset(
    {
        PhysicalDomain.UB,
        PhysicalDomain.L1,
        PhysicalDomain.L0A,
        PhysicalDomain.L0B,
        PhysicalDomain.L0C,
        PhysicalDomain.BT,
        PhysicalDomain.FB,
    }
)


class TPosition(Enum):
    """Ascend C logical tensor position (``AscendC::TPosition``)."""

    GM = "GM"
    A1 = "A1"
    B1 = "B1"
    C1 = "C1"
    A2 = "A2"
    B2 = "B2"
    CO1 = "CO1"
    CO2 = "CO2"
    VECIN = "VECIN"
    VECOUT = "VECOUT"
    VECCALC = "VECCALC"
    LCM = "LCM"
    SHM = "SHM"
    TSCM = "TSCM"
    SPM = "SPM"
    C2 = "C2"
    C2PIPE2GM = "C2PIPE2GM"
    MAX = "MAX"

    @classmethod
    def parse(cls, token: str) -> Optional["TPosition"]:
        """Parse ``"AscendC::TPosition::VECIN"`` or ``"VECIN"``."""
        t = token.strip().rsplit("::", 1)[-1].strip().upper()
        for pos in cls:
            if pos.value == t:
                return pos
        return None


#: Logical ``TPosition`` -> physical SRAM domain.
TPOSITION_TO_DOMAIN: Mapping[TPosition, PhysicalDomain] = {
    TPosition.GM: PhysicalDomain.GM,
    TPosition.A1: PhysicalDomain.L1,
    TPosition.B1: PhysicalDomain.L1,
    TPosition.C1: PhysicalDomain.L1,
    TPosition.TSCM: PhysicalDomain.L1,
    TPosition.A2: PhysicalDomain.L0A,
    TPosition.B2: PhysicalDomain.L0B,
    TPosition.CO1: PhysicalDomain.L0C,
    TPosition.CO2: PhysicalDomain.UB,
    TPosition.VECIN: PhysicalDomain.UB,
    TPosition.VECOUT: PhysicalDomain.UB,
    TPosition.VECCALC: PhysicalDomain.UB,
    TPosition.LCM: PhysicalDomain.UB,
    TPosition.SHM: PhysicalDomain.L1,
    TPosition.SPM: PhysicalDomain.UNKNOWN,
    TPosition.C2: PhysicalDomain.BT,
    TPosition.C2PIPE2GM: PhysicalDomain.FB,
    TPosition.MAX: PhysicalDomain.UNKNOWN,
}

#: DaVinci address-space qualifier -> physical domain.
ADDRESS_SPACE_TO_DOMAIN: Mapping[str, PhysicalDomain] = {
    "__ubuf__": PhysicalDomain.UB,
    "__cbuf__": PhysicalDomain.L1,
    "__ca__": PhysicalDomain.L0A,
    "__cb__": PhysicalDomain.L0B,
    "__cc__": PhysicalDomain.L0C,
    "__bt__": PhysicalDomain.BT,
    "__fbuf__": PhysicalDomain.FB,
    "__gm__": PhysicalDomain.GM,
    # The v3 SIMD/SIMT vector-function model addresses the Unified Buffer as
    # plain "local memory"; 21 files in the production fleet spell it this way.
    "__local_mem__": PhysicalDomain.UB,
}

#: Default ``TPosition`` to assume for a bare address-space qualifier.
DOMAIN_TO_DEFAULT_TPOSITION: Mapping[PhysicalDomain, TPosition] = {
    PhysicalDomain.UB: TPosition.VECCALC,
    PhysicalDomain.L1: TPosition.A1,
    PhysicalDomain.L0A: TPosition.A2,
    PhysicalDomain.L0B: TPosition.B2,
    PhysicalDomain.L0C: TPosition.CO1,
    PhysicalDomain.BT: TPosition.C2,
    PhysicalDomain.FB: TPosition.C2PIPE2GM,
    PhysicalDomain.GM: TPosition.GM,
}


# ---------------------------------------------------------------------------
# HardEvent routes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardEventRoute:
    """An ``AscendC::HardEvent`` route such as ``MTE2_V``.

    The name encodes a *directed* pipeline pair ``SRC_DST``:
    ``SetFlag<HardEvent::SRC_DST>(id)`` is issued on the ``SRC`` pipeline and
    ``WaitFlag<HardEvent::SRC_DST>(id)`` stalls the ``DST`` pipeline until that
    flag is raised.  The resulting happens-before edge therefore runs
    ``SRC -> DST``.
    """

    name: str
    src: Pipe
    dst: Pipe

    @property
    def is_self_route(self) -> bool:
        """``True`` when both endpoints are the same (already program-ordered)."""
        return self.src is self.dst

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name

    @classmethod
    def parse(cls, token: str) -> Optional["HardEventRoute"]:
        """Parse ``"AscendC::HardEvent::MTE2_V"`` / ``"MTE2_V"``.

        Every ``HardEvent`` route name contains exactly one underscore
        separating the two pipeline mnemonics, so a single partition suffices.
        """
        raw = token.strip().rsplit("::", 1)[-1].strip().upper()
        if "_" not in raw:
            return None
        src_tok, _, dst_tok = raw.partition("_")
        src, dst = Pipe.parse(src_tok), Pipe.parse(dst_tok)
        if src is None or dst is None or not src.is_real or not dst.is_real:
            return None
        return cls(name=raw, src=src, dst=dst)

    @classmethod
    def from_pipes(cls, src: Pipe, dst: Pipe) -> "HardEventRoute":
        """Build the route corresponding to an explicit ISASI pipe pair."""
        return cls(name=f"{src.short}_{dst.short}", src=src, dst=dst)


# ---------------------------------------------------------------------------
# Chip profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainSpec:
    """Capacity and alignment rules for one physical memory domain."""

    domain: PhysicalDomain
    capacity_bytes: int
    #: Required alignment, in bytes, of a tensor's base byte offset.
    base_alignment: int = 32
    #: Required alignment, in bytes, of a tensor's total byte length.
    size_alignment: int = 32
    #: Required alignment, in bytes, of a DMA stride expressed in bytes.
    stride_alignment: int = 32
    description: str = ""

    @property
    def capacity_kib(self) -> float:
        return self.capacity_bytes / KIB


DEFAULT_TRANSFER_PIPES: Mapping[Tuple[PhysicalDomain, PhysicalDomain], Pipe] = {
    # move-in: global memory -> on-core SRAM
    (PhysicalDomain.UB, PhysicalDomain.GM): Pipe.MTE2,
    (PhysicalDomain.L1, PhysicalDomain.GM): Pipe.MTE2,
    (PhysicalDomain.UB, PhysicalDomain.L1): Pipe.MTE2,
    (PhysicalDomain.L1, PhysicalDomain.L1): Pipe.MTE2,
    # L1 -> cube input buffers (fractal rearrange)
    (PhysicalDomain.L0A, PhysicalDomain.L1): Pipe.MTE1,
    (PhysicalDomain.L0B, PhysicalDomain.L1): Pipe.MTE1,
    (PhysicalDomain.L0A, PhysicalDomain.UB): Pipe.MTE1,
    (PhysicalDomain.L0B, PhysicalDomain.UB): Pipe.MTE1,
    (PhysicalDomain.BT, PhysicalDomain.L1): Pipe.MTE1,
    # move-out: on-core SRAM -> global memory / L1
    (PhysicalDomain.GM, PhysicalDomain.UB): Pipe.MTE3,
    (PhysicalDomain.GM, PhysicalDomain.L1): Pipe.MTE3,
    (PhysicalDomain.L1, PhysicalDomain.UB): Pipe.MTE3,
    # intra-UB copy runs on the vector unit
    (PhysicalDomain.UB, PhysicalDomain.UB): Pipe.V,
    # cube accumulator drain goes through fixpipe
    (PhysicalDomain.UB, PhysicalDomain.L0C): Pipe.FIX,
    (PhysicalDomain.GM, PhysicalDomain.L0C): Pipe.FIX,
    (PhysicalDomain.L1, PhysicalDomain.L0C): Pipe.FIX,
}


@dataclass(frozen=True)
class ChipSpec:
    """A complete static description of one Ascend chip variant."""

    name: str
    display_name: str
    aliases: Tuple[str, ...] = ()
    domains: Mapping[PhysicalDomain, DomainSpec] = field(default_factory=dict)
    #: ``EVENT_IDn`` values reserved by the runtime and unusable by kernels.
    reserved_event_ids: FrozenSet[int] = frozenset({6, 7})
    #: Largest legal ``EVENT_IDn`` value (inclusive).
    ub_bank_count: int = 8
    l0c_base_offsets: Tuple[int, ...] = (0,)
    max_event_id: int = 7
    #: One block == the hardware's natural SRAM access granule, in bytes.
    block_bytes: int = 32
    #: Vector register width in bytes (one full vector instruction's footprint).
    vector_bytes: int = 256
    #: ``True`` when the capacity numbers are inferred rather than documented.
    provisional: bool = False
    notes: str = ""
    #: Optional instruction-set features this part implements.  An intrinsic
    #: whose ``requires_feature`` is absent here cannot be issued on this chip.
    #: Empty on DaVinci v2 (910B/910C): those parts have the baseline Cube and
    #: Vector sets, and no MX microscale path.
    features: FrozenSet[str] = frozenset()
    # -- 351x SIMD/SIMT Unified Buffer partitioning --------------------------
    #: Total Unified Buffer bytes when the architecture strictly partitions UB
    #: between tensor allocations and the SIMT DataCache.  ``None`` on parts
    #: (910B/910C) where the whole UB is available to tensors.
    ub_total_bytes: Optional[int] = None
    #: Bytes the compiler reserves inside the partitioned UB.
    compiler_reserved_bytes: int = 0
    #: Hardware/runtime minimum for the SIMT DataCache partition.
    min_datacache_bytes: Optional[int] = None
    #: Largest static+dynamic tensor allocation that still leaves the minimum
    #: DataCache: ``ub_total - compiler_reserved - min_datacache``.
    max_usable_ub_bytes: Optional[int] = None
    # -- analytical performance model ----------------------------------------
    #: Sustained GM -> L1/UB move-in bandwidth, bytes per cycle (PIPE_MTE2).
    mte2_bytes_per_cycle: int = 64
    #: Sustained L1 -> L0A/L0B fractal-load bandwidth, bytes per cycle
    #: (PIPE_MTE1).
    mte1_bytes_per_cycle: int = 64
    #: Sustained L0C -> GM/UB fixpipe drain bandwidth, bytes per cycle
    #: (PIPE_FIX; PIPE_MTE3 stores share the GM write path and use it too).
    fixpipe_bytes_per_cycle: int = 64
    #: Cube contraction throughput in multiply-accumulates per cycle: one
    #: 16x16x16 fractal multiply per cycle on a full AI core.
    cube_macs_per_cycle: int = 4096
    #: Cube operand-read throughput, bytes per cycle, used when the (M, K, N)
    #: tile shape cannot be recovered from the call site.
    cube_read_bytes_per_cycle: int = 1024
    #: Optional calibrated cross-queue hand-off penalty; zero by default. A
    #: ``SetFlag`` -> ``WaitFlag`` pair adds between two pipelines even when
    #: the producing instruction has already completed.
    sync_handoff_cycles: int = 0
    issue_cycles: int = 1
    # Flow inputs are explicit calibration, never inferred from event counts.
    flow_rates: Mapping[str, float] = field(default_factory=dict)
    flow_buffers: Mapping[str, float] = field(default_factory=dict)
    flow_roles: Mapping[str, str] = field(default_factory=dict)
    pipes: Tuple[Pipe, ...] = REAL_PIPES
    transfer_pipes: Mapping[Tuple[PhysicalDomain, PhysicalDomain], Pipe] = field(default_factory=lambda: dict(DEFAULT_TRANSFER_PIPES))
    position_domains: Mapping[TPosition, PhysicalDomain] = field(default_factory=lambda: dict(TPOSITION_TO_DOMAIN))
    queue_depths: Mapping[str, int] = field(default_factory=dict)
    supported_routes: Optional[FrozenSet[str]] = None
    core_topology: str = "split"
    ub_bank_ports: int = 1
    cube_fractal: Tuple[int, int, int] = (16, 16, 16)

    def __post_init__(self):
        from math import isfinite
        for value in (*self.flow_rates.values(), *self.flow_buffers.values()):
            if not isfinite(value) or value <= 0:
                raise ValueError("flow calibration must be finite and positive")
        if any(v not in {"compute", "dma", "control"} for v in self.flow_roles.values()):
            raise ValueError("invalid flow engine role")
        if any(not isinstance(v, int) or v < 1 for v in self.queue_depths.values()):
            raise ValueError("queue depths must be positive integers")
        if any(p not in REAL_PIPES for p in self.pipes) or len(set(self.pipes)) != len(self.pipes):
            raise ValueError("invalid processor pipelines")
        if any(len(pair) != 2 or pipe not in self.pipes for pair, pipe in self.transfer_pipes.items()):
            raise ValueError("invalid processor transfer route")
        if self.core_topology not in {"unified", "split", "unknown"}:
            raise ValueError("invalid core topology")
        if min(self.issue_cycles, self.block_bytes, self.vector_bytes, self.ub_bank_count, self.ub_bank_ports, *self.cube_fractal,
               self.mte2_bytes_per_cycle, self.mte1_bytes_per_cycle,
               self.fixpipe_bytes_per_cycle, self.cube_macs_per_cycle,
               self.cube_read_bytes_per_cycle) <= 0 or self.sync_handoff_cycles < 0:
            raise ValueError("invalid hardware granule, throughput or latency")
        for spec in self.domains.values():
            if spec.capacity_bytes < 0 or min(spec.base_alignment, spec.size_alignment, spec.stride_alignment) <= 0:
                raise ValueError("invalid memory capacity/alignment")

    def domain_spec(self, domain: PhysicalDomain) -> Optional[DomainSpec]:
        return self.domains.get(domain)

    def sram_domains(self) -> Tuple[DomainSpec, ...]:
        return tuple(
            self.domains[d]
            for d in PhysicalDomain
            if d in self.domains and d.is_on_core_sram
        )

    @property
    def enforces_datacache_partition(self) -> bool:
        """``True`` on parts where UB is split with a guarded DataCache."""
        return (
            self.ub_total_bytes is not None
            and self.min_datacache_bytes is not None
            and self.max_usable_ub_bytes is not None
        )

    def cube_contraction_cycles(self, m: int, k: int, n: int) -> int:
        """Cycles for one ``(m, k, n)`` Cube contraction.

        The cube unit retires one 16x16x16 fractal multiply per cycle, so the
        cost is the product of the ceil-rounded fractal counts along each
        dimension - exact for whole-fractal tiles and monotone otherwise.
        """
        fm, fk, fn = self.cube_fractal
        work = (-(-max(m, 1) // fm) * fm *
                -(-max(k, 1) // fk) * fk *
                -(-max(n, 1) // fn) * fn)
        return -(-work // self.cube_macs_per_cycle)


def _domains(**kw: Tuple[int, int, int, int, str]) -> Dict[PhysicalDomain, DomainSpec]:
    """Build a domain table from ``NAME=(capacity, base, size, stride, desc)``."""
    out: Dict[PhysicalDomain, DomainSpec] = {}
    for key, (cap, base, size, stride, desc) in kw.items():
        dom = PhysicalDomain[key]
        out[dom] = DomainSpec(
            domain=dom,
            capacity_bytes=cap,
            base_alignment=base,
            size_alignment=size,
            stride_alignment=stride,
            description=desc,
        )
    return out


#: Cube input/output buffers are addressed in *fractal* units.  A 16x16 block
#: of fp16 is 512 B (L0A/L0B); an fp32 accumulator fractal is 1024 B (L0C).
_FRACTAL_FP16 = 512
_FRACTAL_FP32 = 1024

CHIP_PROFILES: Dict[str, ChipSpec] = {
    "ascend310p": ChipSpec(
        name="ascend310p", display_name="Ascend 310P AI Core (provisional capacities)",
        aliases=("310p", "ascend310p1", "ascend310p3"), core_topology="unified",
        domains=_domains(
            UB=(256*KIB, 32, 32, 32, "Unified Buffer; provisional"),
            L1=(1024*KIB, 32, 32, 32, "L1; provisional"),
            L0A=(64*KIB, 512, 512, 32, "Cube A; provisional"),
            L0B=(64*KIB, 512, 512, 32, "Cube B; provisional"),
            L0C=(128*KIB, 1024, 1024, 32, "Accumulator; provisional"),
        ), provisional=True,
        notes="AI Core variant only; capacities and bank model require a SKU-specific "
              "CANN GetCoreMemSize export. Does not describe 310P Vector Core. "
              "Queue depths and route applicability remain unknown.",
    ),
    "ascend910b": ChipSpec(
        name="ascend910b",
        display_name="Ascend 910B (Atlas A2 training series)",
        aliases=("910b", "ascend910b2", "ascend910b3", "atlas-a2", "a2", "dav-c220", "ascend910b4"),
        domains=_domains(
            UB=(192 * KIB, 32, 32, 32, "Unified Buffer (vector scratchpad)"),
            L1=(512 * KIB, 32, 32, 32, "L1 Buffer (shared staging)"),
            L0A=(64 * KIB, _FRACTAL_FP16, _FRACTAL_FP16, 32, "Cube A (left matrix)"),
            L0B=(64 * KIB, _FRACTAL_FP16, _FRACTAL_FP16, 32, "Cube B (right matrix)"),
            L0C=(128 * KIB, _FRACTAL_FP32, _FRACTAL_FP32, 32, "Cube C (accumulator)"),
            BT=(1 * KIB, 64, 64, 32, "Bias table"),
            FB=(2 * KIB, 128, 128, 32, "Fixpipe parameter buffer"),
        ),
        ub_bank_count=16,
        notes="192 KiB UB and Cube capacities from NPUs/Ascend910B4.md; "
              "16 logical bank groups (64 physical banks).",
    ),
    "ascend910c": ChipSpec(
        name="ascend910c",
        display_name="Ascend 910C (Atlas A3 series)",
        aliases=("910c", "atlas-a3", "a3"),
        domains=_domains(
            UB=(256 * KIB, 32, 32, 32, "Unified Buffer (vector scratchpad)"),
            L1=(512 * KIB, 32, 32, 32, "L1 Buffer (shared staging)"),
            L0A=(64 * KIB, _FRACTAL_FP16, _FRACTAL_FP16, 32, "Cube A (left matrix)"),
            L0B=(64 * KIB, _FRACTAL_FP16, _FRACTAL_FP16, 32, "Cube B (right matrix)"),
            L0C=(256 * KIB, _FRACTAL_FP32, _FRACTAL_FP32, 32, "Cube C (accumulator)"),
            BT=(1 * KIB, 64, 64, 32, "Bias table"),
            FB=(2 * KIB, 128, 128, 32, "Fixpipe parameter buffer"),
        ),
        provisional=True,
        notes="Capacities extrapolated from the 910B profile; override with "
              "--ub-bytes / --l1-bytes or --chip-profile when you have the "
              "SKU datasheet.",
    ),
    "ascend950pr": ChipSpec(
        name="ascend950pr",
        display_name="Ascend 950PR (dav-3510)",
        aliases=("950pr", "dav-3510", "Ascend910_9589", "Ascend950"),
        domains=_domains(
            UB=(256 * KIB, 32, 32, 32, "Unified Buffer"),
            L1=(512 * KIB, 32, 32, 32, "L1 Buffer"),
            L0A=(64 * KIB, 512, 512, 32, "Cube A"),
            L0B=(64 * KIB, 512, 512, 32, "Cube B"),
            L0C=(256 * KIB, 1024, 1024, 32, "Cube accumulator"),
        ),
        reserved_event_ids=frozenset({0, 1, 2}),
        ub_bank_count=8,
        l0c_base_offsets=(0, 1024),
        features=frozenset({"mx_cube", "simt"}),
        ub_total_bytes=256 * KIB,
        compiler_reserved_bytes=8 * KIB,
        min_datacache_bytes=32 * KIB,
        max_usable_ub_bytes=216 * KIB,
        provisional=True,
        notes="dav-3510 task target: 256 KiB physical UB, TPipe events 0..2 "
              "reserved. NPUs/Ascend910_9599.md describes dav-c310, a "
              "different target; do not infer dav-3510 ISA from that SKU.",
    ),
    "ascend351x": ChipSpec(
        name="ascend351x",
        display_name="Ascend 351x (inference series)",
        aliases=("351x", "ascend3510", "ascend3519", "3510", "3519"),
        domains=_domains(
            UB=(256 * KIB, 32, 32, 32, "Unified Buffer (vector scratchpad)"),
            L1=(512 * KIB, 32, 32, 32, "L1 Buffer (shared staging)"),
            L0A=(64 * KIB, _FRACTAL_FP16, _FRACTAL_FP16, 32, "Cube A (left matrix)"),
            L0B=(64 * KIB, _FRACTAL_FP16, _FRACTAL_FP16, 32, "Cube B (right matrix)"),
            L0C=(128 * KIB, _FRACTAL_FP32, _FRACTAL_FP32, 32, "Cube C (accumulator)"),
            BT=(1 * KIB, 64, 64, 32, "Bias table"),
            FB=(2 * KIB, 128, 128, 32, "Fixpipe parameter buffer"),
        ),
        # 351x runs isomorphic SIMD/SIMT execution: the 256 KiB UB is strictly
        # partitioned between tensor memory and the SIMT DataCache, with 8 KiB
        # reserved by the compiler and a hardware-enforced DataCache floor of
        # 32 KiB.  Static+dynamic allocations beyond 216 KiB push the DataCache
        # below that floor and corrupt memory at runtime (AKA1010).
        ub_total_bytes=256 * KIB,
        compiler_reserved_bytes=8 * KIB,
        min_datacache_bytes=32 * KIB,
        max_usable_ub_bytes=216 * KIB,
        provisional=True,
        # DaVinci v3 Cube: carries the MX (microscaled FP4) operand format and
        # its scale-load path, which the v2 parts above do not.
        features=frozenset({"mx_cube"}),
        notes="Capacities extrapolated; override with --chip-profile for a "
              "specific 351x SKU. UB is partitioned: at most 216 KiB for "
              "tensors when SIMT is in use, leaving a 32 KiB DataCache.",
    ),
}


def resolve_chip(name: str) -> ChipSpec:
    """Look up a chip profile by canonical name or alias (case-insensitive)."""

    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")

    key = norm(name)
    for spec in CHIP_PROFILES.values():
        if key in {norm(c) for c in ({spec.name} | set(spec.aliases))}:
            return spec
    known = ", ".join(sorted(CHIP_PROFILES))
    raise KeyError(f"unknown chip {name!r}; known profiles: {known}")


# ---------------------------------------------------------------------------
# Hardware model facade
# ---------------------------------------------------------------------------


@dataclass
class HardwareModel:
    """Queryable facade over a :class:`ChipSpec` plus any user overrides."""

    chip: ChipSpec

    # -- construction -------------------------------------------------------

    @classmethod
    def for_chip(cls, name: str = "ascend910b") -> "HardwareModel":
        return cls(chip=resolve_chip(name))

    def with_capacity_overrides(
        self, overrides: Mapping[PhysicalDomain, Optional[int]]
    ) -> "HardwareModel":
        """Return a copy with selected domain capacities replaced."""
        applied = {d: c for d, c in overrides.items() if c is not None}
        if not applied:
            return self
        domains = dict(self.chip.domains)
        for dom, cap in applied.items():
            existing = domains.get(dom)
            domains[dom] = (
                DomainSpec(domain=dom, capacity_bytes=cap)
                if existing is None
                else replace(existing, capacity_bytes=cap)
            )
        note = f"{self.chip.notes} Capacity overridden from the command line."
        return HardwareModel(
            chip=replace(
                self.chip, domains=domains, provisional=True, notes=note.strip()
            )
        )

    @classmethod
    def from_profile_file(cls, path: Path) -> "HardwareModel":
        """Load a chip profile from JSON.

        Expected shape::

            {"name": "my910b", "base": "ascend910b",
             "reserved_event_ids": [6, 7], "max_event_id": 7,
             "domains": {"UB": {"capacity_bytes": 196608, "base_alignment": 32}}}
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        base = resolve_chip(raw["base"]) if "base" in raw else CHIP_PROFILES["ascend910b"]
        domains = dict(base.domains)
        for dom_name, cfg in (raw.get("domains") or {}).items():
            dom = PhysicalDomain[dom_name.upper()]
            cur = domains.get(dom) or DomainSpec(domain=dom, capacity_bytes=0)
            domains[dom] = replace(
                cur,
                capacity_bytes=int(cfg.get("capacity_bytes", cur.capacity_bytes)),
                base_alignment=int(cfg.get("base_alignment", cur.base_alignment)),
                size_alignment=int(cfg.get("size_alignment", cur.size_alignment)),
                stride_alignment=int(cfg.get("stride_alignment", cur.stride_alignment)),
                description=str(cfg.get("description", cur.description)),
            )
        spec = ChipSpec(
            name=str(raw.get("name", base.name)),
            display_name=str(raw.get("display_name", base.display_name)),
            aliases=tuple(raw.get("aliases", ())),
            domains=domains,
            reserved_event_ids=frozenset(
                int(x) for x in raw.get("reserved_event_ids", base.reserved_event_ids)
            ),
            features=frozenset(raw.get("features", base.features)),
            ub_bank_count=int(raw.get("ub_bank_count", base.ub_bank_count)),
            l0c_base_offsets=tuple(raw.get("l0c_base_offsets", base.l0c_base_offsets)),
            max_event_id=int(raw.get("max_event_id", base.max_event_id)),
            block_bytes=int(raw.get("block_bytes", base.block_bytes)),
            vector_bytes=int(raw.get("vector_bytes", base.vector_bytes)),
            provisional=bool(raw.get("provisional", True)),
            notes=str(raw.get("notes", f"loaded from {Path(path).name}")),
            ub_total_bytes=(
                int(raw["ub_total_bytes"]) if "ub_total_bytes" in raw else base.ub_total_bytes
            ),
            compiler_reserved_bytes=int(
                raw.get("compiler_reserved_bytes", base.compiler_reserved_bytes)
            ),
            min_datacache_bytes=(
                int(raw["min_datacache_bytes"])
                if "min_datacache_bytes" in raw
                else base.min_datacache_bytes
            ),
            max_usable_ub_bytes=(
                int(raw["max_usable_ub_bytes"])
                if "max_usable_ub_bytes" in raw
                else base.max_usable_ub_bytes
            ),
            mte2_bytes_per_cycle=int(
                raw.get("mte2_bytes_per_cycle", base.mte2_bytes_per_cycle)
            ),
            mte1_bytes_per_cycle=int(
                raw.get("mte1_bytes_per_cycle", base.mte1_bytes_per_cycle)
            ),
            fixpipe_bytes_per_cycle=int(
                raw.get("fixpipe_bytes_per_cycle", base.fixpipe_bytes_per_cycle)
            ),
            cube_macs_per_cycle=int(
                raw.get("cube_macs_per_cycle", base.cube_macs_per_cycle)
            ),
            cube_read_bytes_per_cycle=int(
                raw.get("cube_read_bytes_per_cycle", base.cube_read_bytes_per_cycle)
            ),
            sync_handoff_cycles=int(
                raw.get("sync_handoff_cycles", base.sync_handoff_cycles)
            ),
            issue_cycles=int(raw.get("issue_cycles", base.issue_cycles)),
            flow_rates={k: float(v) for k, v in raw.get("flow_rates", base.flow_rates).items()},
            flow_buffers={k: float(v) for k, v in raw.get("flow_buffers", base.flow_buffers).items()},
            flow_roles=dict(raw.get("flow_roles", base.flow_roles)),
            pipes=tuple(Pipe.parse(x) for x in raw["pipes"]) if "pipes" in raw else base.pipes,
            transfer_pipes=({tuple(PhysicalDomain[x.strip().upper()] for x in pair.split(",")): Pipe.parse(pipe)
                             for pair, pipe in raw["transfer_pipes"].items()} if "transfer_pipes" in raw else base.transfer_pipes),
            position_domains={**base.position_domains, **{TPosition[k]: PhysicalDomain[v]
                              for k, v in raw.get("position_domains", {}).items()}},
            queue_depths=dict(raw.get("queue_depths", base.queue_depths)),
            supported_routes=(frozenset(raw["supported_routes"]) if "supported_routes" in raw else base.supported_routes),
            core_topology=str(raw.get("core_topology", base.core_topology)),
            ub_bank_ports=int(raw.get("ub_bank_ports", base.ub_bank_ports)),
            cube_fractal=tuple(raw.get("cube_fractal", base.cube_fractal)),
        )
        return cls(chip=spec)

    # -- queries ------------------------------------------------------------

    def flow_rate(self, engine: str) -> float:
        if engine not in self.chip.flow_rates:
            raise ValueError(f"uncalibrated flow engine {engine!r}; provide flow_rates")
        return self.chip.flow_rates[engine]

    def flow_capacity(self, buffer: str) -> float:
        if buffer not in self.chip.flow_buffers:
            raise ValueError(f"unknown flow buffer {buffer!r}; provide flow_buffers")
        return self.chip.flow_buffers[buffer]

    def flow_role(self, engine: str) -> str:
        return self.chip.flow_roles.get(engine, "control")

    def supports_route(self, route: HardEventRoute) -> Optional[bool]:
        return None if self.chip.supported_routes is None else route.name in self.chip.supported_routes

    def active_pipes(self) -> Tuple[Pipe, ...]:
        return self.chip.pipes

    def transfer_pipe(self, dst: PhysicalDomain, src: PhysicalDomain) -> Optional[Pipe]:
        return self.chip.transfer_pipes.get((dst, src))

    def queue_depth(self, engine: str) -> Optional[int]:
        return self.chip.queue_depths.get(engine)

    def ub_bank(self, offset: int) -> int:
        return offset // self.chip.block_bytes % self.chip.ub_bank_count

    def domain_of(self, position: TPosition) -> PhysicalDomain:
        return self.chip.position_domains.get(position, PhysicalDomain.UNKNOWN)

    def capacity(self, domain: PhysicalDomain) -> Optional[int]:
        spec = self.chip.domain_spec(domain)
        return None if spec is None else spec.capacity_bytes

    def spec_for(self, domain: PhysicalDomain) -> Optional[DomainSpec]:
        return self.chip.domain_spec(domain)

    def base_alignment(self, domain: PhysicalDomain) -> int:
        spec = self.chip.domain_spec(domain)
        return self.chip.block_bytes if spec is None else spec.base_alignment

    def size_alignment(self, domain: PhysicalDomain) -> int:
        spec = self.chip.domain_spec(domain)
        return self.chip.block_bytes if spec is None else spec.size_alignment

    def stride_alignment(self, domain: PhysicalDomain) -> int:
        spec = self.chip.domain_spec(domain)
        return self.chip.block_bytes if spec is None else spec.stride_alignment

    def is_event_id_reserved(self, event_id: int) -> bool:
        return event_id in self.chip.reserved_event_ids

    def is_event_id_in_range(self, event_id: int) -> bool:
        return 0 <= event_id <= self.chip.max_event_id

    def simt_datacache_available(self, allocated_bytes: int) -> Optional[int]:
        """DataCache bytes left after *allocated_bytes* of tensor memory.

        ``DataCache = ub_total - StaticMem + DynamicMem - compiler_reserved``.
        ``None`` on architectures that do not partition the Unified Buffer.
        """
        chip = self.chip
        if chip.ub_total_bytes is None:
            return None
        return chip.ub_total_bytes - allocated_bytes - chip.compiler_reserved_bytes

    def pipe_bytes_per_cycle(self, pipe: Pipe) -> Optional[int]:
        """Modeled sustained throughput of one engine, in bytes per cycle.

        ``None`` for pipes without a bandwidth model (the scalar unit, flag
        issue, barriers): the performance profiler treats those as costing a
        nominal issue cycle each rather than moving bulk data.
        """
        chip = self.chip
        return {
            Pipe.MTE2: chip.mte2_bytes_per_cycle,
            Pipe.MTE1: chip.mte1_bytes_per_cycle,
            Pipe.MTE3: chip.fixpipe_bytes_per_cycle,
            Pipe.FIX: chip.fixpipe_bytes_per_cycle,
            # The vector unit retires vector_bytes (one full vector register
            # footprint) per cycle.
            Pipe.V: chip.vector_bytes,
        }.get(pipe)

    def tracked_sram_domains(self) -> Tuple[PhysicalDomain, ...]:
        return tuple(spec.domain for spec in self.chip.sram_domains())

    def describe(self) -> Dict[str, object]:
        """JSON-serialisable summary for the report header."""
        return {
            "name": self.chip.name,
            "display_name": self.chip.display_name,
            "provisional": self.chip.provisional,
            "notes": self.chip.notes,
            "block_bytes": self.chip.block_bytes,
            "max_event_id": self.chip.max_event_id,
            "reserved_event_ids": sorted(self.chip.reserved_event_ids),
            "flow_model": {"rates": dict(self.chip.flow_rates), "buffers": dict(self.chip.flow_buffers), "roles": dict(self.chip.flow_roles),
                           "queue_depths": dict(self.chip.queue_depths), "calibrated": bool(self.chip.flow_rates)},
            "core_topology": self.chip.core_topology,
            "pipes": [p.value for p in self.chip.pipes],
            "transfer_pipes": {f"{dst.value},{src.value}": pipe.value for (dst, src), pipe in self.chip.transfer_pipes.items()},
            "position_domains": {p.value: d.value for p, d in self.chip.position_domains.items()},
            "supported_routes": sorted(self.chip.supported_routes) if self.chip.supported_routes is not None else None,
            "ub_bank_ports": self.chip.ub_bank_ports,
            "ub_partition": {
                "ub_total_bytes": self.chip.ub_total_bytes,
                "compiler_reserved_bytes": self.chip.compiler_reserved_bytes,
                "min_datacache_bytes": self.chip.min_datacache_bytes,
                "max_usable_ub_bytes": self.chip.max_usable_ub_bytes,
            }
            if self.chip.enforces_datacache_partition
            else None,
            "perf_model": {
                "mte2_bytes_per_cycle": self.chip.mte2_bytes_per_cycle,
                "mte1_bytes_per_cycle": self.chip.mte1_bytes_per_cycle,
                "fixpipe_bytes_per_cycle": self.chip.fixpipe_bytes_per_cycle,
                "cube_macs_per_cycle": self.chip.cube_macs_per_cycle,
                "cube_read_bytes_per_cycle": self.chip.cube_read_bytes_per_cycle,
                "sync_handoff_cycles": self.chip.sync_handoff_cycles,
                "issue_cycles": self.chip.issue_cycles,
            },
            "domains": {
                spec.domain.value: {
                    "capacity_bytes": spec.capacity_bytes,
                    "capacity_kib": round(spec.capacity_kib, 2),
                    "base_alignment": spec.base_alignment,
                    "size_alignment": spec.size_alignment,
                    "stride_alignment": spec.stride_alignment,
                    "description": spec.description,
                }
                for spec in self.chip.sram_domains()
            },
        }
