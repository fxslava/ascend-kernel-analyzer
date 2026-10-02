"""Signature model for the Ascend C intrinsics the analyzer understands.

Two things are encoded per intrinsic:

1. **Which pipeline issues it.**  Most intrinsics are fixed (``Add`` is always
   ``PIPE_V``), but ``DataCopy`` is polymorphic: the issuing pipeline follows
   from the *pair* of memory domains it moves between, which is why
   :data:`TRANSFER_PIPE` is keyed by ``(dst_domain, src_domain)``.

2. **Which memory domains each operand may legally live in.**  This is what
   turns a silent "phantom type" bug - handing a UB-resident tensor to a
   Cube-targeted API that reads L0A - into a diagnosable error.  Ascend C
   hides the address space inside ``LocalTensor<T>``, so the C++ type system
   will not catch it for you.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from .hardware import PhysicalDomain as D
from .hardware import Pipe

__all__ = [
    "ArgRole",
    "ParamSpec",
    "ApiSpec",
    "TransferVolume",
    "API_TABLE",
    "TRANSFER_PIPE",
    "lookup_api",
    "data_copy_pipe",
    "is_known_api",
]


class ArgRole(Enum):
    """The semantic role of a positional argument."""

    DST = "dst"
    SRC = "src"
    SRC0 = "src0"
    SRC1 = "src1"
    SCALAR = "scalar"
    COUNT = "count"
    PARAMS = "params"
    OTHER = "other"


ANY_TENSOR: Optional[FrozenSet[D]] = None  # sentinel: operand is not domain-checked

_UB = frozenset({D.UB})
_L1 = frozenset({D.L1})
_GM = frozenset({D.GM})
_L0A = frozenset({D.L0A})
_L0B = frozenset({D.L0B})
_L0C = frozenset({D.L0C})
_UB_OR_GM = frozenset({D.UB, D.GM})
_GM_OR_UB_OR_L1 = frozenset({D.GM, D.UB, D.L1})
_L0AB = frozenset({D.L0A, D.L0B})


@dataclass(frozen=True)
class ParamSpec:
    """One positional parameter of an intrinsic."""

    role: ArgRole
    #: Domains this operand may live in; ``None`` means "not a tensor operand".
    allowed: Optional[FrozenSet[D]] = ANY_TENSOR
    name: str = ""

    @property
    def is_tensor(self) -> bool:
        return self.allowed is not None

    def describe_allowed(self) -> str:
        if self.allowed is None:
            return "any"
        return "/".join(sorted(d.value for d in self.allowed))


@dataclass(frozen=True)
class TransferVolume:
    """How to infer a loader's transaction volume from its parameters.

    Low-level CCE loaders take raw ``void*`` addresses, so ``sizeof(T)`` of the
    operand is often unavailable.  The hardware instead moves fixed-size
    *tiles* per repeat: one ``load_*_s4`` repeat moves a packed FP4 16x64
    fractal (512 B); one ``load_*_mx`` repeat moves the matching E8M0 scale
    granule for a 16x64 tile (16 rows x 2 k-groups x 1 B = 32 B).  When a
    tensor's byte size cannot be resolved any other way, the analyzer derives
    it from the repeat parameter instead of failing on ``sizeof(void)``.
    """

    #: Index of the positional parameter that counts repeats.
    repeat_index: int
    #: Bytes moved per repeat.
    unit_bytes: int
    #: Provenance note for diagnostics.
    basis: str = ""


@dataclass(frozen=True)
class ApiSpec:
    """Static description of one recognised intrinsic."""

    name: str
    params: Tuple[ParamSpec, ...]
    #: Fixed issuing pipeline, or ``None`` when it depends on operand domains.
    pipe: Optional[Pipe]
    category: str
    doc: str = ""
    #: How to derive the transferred byte volume, for raw-pointer loaders.
    volume: Optional[TransferVolume] = None

    def param_at(self, index: int) -> Optional[ParamSpec]:
        return self.params[index] if 0 <= index < len(self.params) else None


def _p(role: ArgRole, allowed: Optional[FrozenSet[D]], name: str) -> ParamSpec:
    return ParamSpec(role=role, allowed=allowed, name=name)


# ---------------------------------------------------------------------------
# DataCopy: pipeline follows the (dst, src) domain pair
# ---------------------------------------------------------------------------

#: ``(dst_domain, src_domain)`` -> issuing pipeline.  A pair absent from this
#: table is not a legal single-instruction transfer on DaVinci; the data has
#: to be staged through an intermediate buffer.
TRANSFER_PIPE: Mapping[Tuple[D, D], Pipe] = {
    # move-in: global memory -> on-core SRAM
    (D.UB, D.GM): Pipe.MTE2,
    (D.L1, D.GM): Pipe.MTE2,
    (D.UB, D.L1): Pipe.MTE2,
    (D.L1, D.L1): Pipe.MTE2,
    # L1 -> cube input buffers (fractal rearrange)
    (D.L0A, D.L1): Pipe.MTE1,
    (D.L0B, D.L1): Pipe.MTE1,
    (D.L0A, D.UB): Pipe.MTE1,
    (D.L0B, D.UB): Pipe.MTE1,
    (D.BT, D.L1): Pipe.MTE1,
    # move-out: on-core SRAM -> global memory / L1
    (D.GM, D.UB): Pipe.MTE3,
    (D.GM, D.L1): Pipe.MTE3,
    (D.L1, D.UB): Pipe.MTE3,
    # intra-UB copy runs on the vector unit
    (D.UB, D.UB): Pipe.V,
    # cube accumulator drain goes through fixpipe
    (D.UB, D.L0C): Pipe.FIX,
    (D.GM, D.L0C): Pipe.FIX,
    (D.L1, D.L0C): Pipe.FIX,
}


def data_copy_pipe(dst: D, src: D) -> Optional[Pipe]:
    """Issuing pipeline for a ``DataCopy`` between two domains.

    ``None`` means the pair is not a legal direct transfer.  Unknown domains
    fall back to a conservative guess so that a partially-resolved kernel
    still gets a plausible pipeline assignment.
    """
    if dst is D.UNKNOWN or src is D.UNKNOWN:
        if src is D.GM:
            return Pipe.MTE2
        if dst is D.GM:
            return Pipe.MTE3
        return Pipe.MTE2
    return TRANSFER_PIPE.get((dst, src))


# ---------------------------------------------------------------------------
# Intrinsic table
# ---------------------------------------------------------------------------

#: Element-wise vector intrinsics taking ``(dst, src0, src1, count)``.
_VEC_BINARY = (
    "Add", "Sub", "Mul", "Div", "Max", "Min", "And", "Or",
    "AddRelu", "SubRelu", "FusedMulAdd", "MulAddDst", "Axpy",
)
#: Element-wise vector intrinsics taking ``(dst, src, count)``.
_VEC_UNARY = (
    "Abs", "Exp", "Ln", "Sqrt", "Rsqrt", "Reciprocal", "Relu", "Not",
    "Cast", "Sign", "Floor", "Ceil", "Round", "Trunc", "Copy",
)
#: Vector intrinsics taking ``(dst, src, scalar, count)``.
_VEC_SCALAR = (
    "Adds", "Muls", "Maxs", "Mins", "Subs", "LeakyRelu", "ShiftLeft",
    "ShiftRight", "CompareScalar",
)
#: Reductions taking ``(dst, src, work, count)``.
_VEC_REDUCE = (
    "ReduceSum", "ReduceMax", "ReduceMin", "WholeReduceSum",
    "BlockReduceSum", "BlockReduceMax", "BlockReduceMin", "RepeatReduceSum",
)


def _build_table() -> Dict[str, ApiSpec]:
    table: Dict[str, ApiSpec] = {}

    def put(spec: ApiSpec) -> None:
        table[spec.name] = spec

    # -- vector unit --------------------------------------------------------
    for name in _VEC_BINARY:
        put(ApiSpec(
            name=name,
            params=(
                _p(ArgRole.DST, _UB, "dst"),
                _p(ArgRole.SRC0, _UB, "src0"),
                _p(ArgRole.SRC1, _UB, "src1"),
                _p(ArgRole.COUNT, ANY_TENSOR, "count"),
            ),
            pipe=Pipe.V,
            category="vector",
            doc="Element-wise binary vector operation; all operands live in UB.",
        ))
    for name in _VEC_UNARY:
        put(ApiSpec(
            name=name,
            params=(
                _p(ArgRole.DST, _UB, "dst"),
                _p(ArgRole.SRC, _UB, "src"),
                _p(ArgRole.COUNT, ANY_TENSOR, "count"),
            ),
            pipe=Pipe.V,
            category="vector",
            doc="Element-wise unary vector operation; all operands live in UB.",
        ))
    for name in _VEC_SCALAR:
        put(ApiSpec(
            name=name,
            params=(
                _p(ArgRole.DST, _UB, "dst"),
                _p(ArgRole.SRC, _UB, "src"),
                _p(ArgRole.SCALAR, ANY_TENSOR, "scalar"),
                _p(ArgRole.COUNT, ANY_TENSOR, "count"),
            ),
            pipe=Pipe.V,
            category="vector",
            doc="Vector-scalar operation; tensor operands live in UB.",
        ))
    for name in _VEC_REDUCE:
        put(ApiSpec(
            name=name,
            params=(
                _p(ArgRole.DST, _UB, "dst"),
                _p(ArgRole.SRC, _UB, "src"),
                _p(ArgRole.OTHER, _UB, "work"),
                _p(ArgRole.COUNT, ANY_TENSOR, "count"),
            ),
            pipe=Pipe.V,
            category="vector",
            doc="Vector reduction; all tensor operands live in UB.",
        ))

    put(ApiSpec(
        name="Duplicate",
        params=(
            _p(ArgRole.DST, _UB, "dst"),
            _p(ArgRole.SCALAR, ANY_TENSOR, "scalar"),
            _p(ArgRole.COUNT, ANY_TENSOR, "count"),
        ),
        pipe=Pipe.V,
        category="vector",
        doc="Broadcast a scalar across a UB tensor.",
    ))
    put(ApiSpec(
        name="Select",
        params=(
            _p(ArgRole.DST, _UB, "dst"),
            _p(ArgRole.SRC0, _UB, "mask"),
            _p(ArgRole.SRC0, _UB, "src0"),
            _p(ArgRole.SRC1, _UB, "src1"),
        ),
        pipe=Pipe.V,
        category="vector",
        doc="Masked select between two UB tensors.",
    ))
    put(ApiSpec(
        name="Compare",
        params=(
            _p(ArgRole.DST, _UB, "dst"),
            _p(ArgRole.SRC0, _UB, "src0"),
            _p(ArgRole.SRC1, _UB, "src1"),
        ),
        pipe=Pipe.V,
        category="vector",
        doc="Element-wise comparison producing a UB mask.",
    ))
    put(ApiSpec(
        name="Transpose",
        params=(_p(ArgRole.DST, _UB, "dst"), _p(ArgRole.SRC, _UB, "src")),
        pipe=Pipe.V,
        category="vector",
        doc="In-UB block transpose.",
    ))
    put(ApiSpec(
        name="Brcb",
        params=(
            _p(ArgRole.DST, _UB, "dst"),
            _p(ArgRole.SRC, _UB, "src"),
            _p(ArgRole.OTHER, ANY_TENSOR, "repeat"),
            _p(ArgRole.PARAMS, ANY_TENSOR, "params"),
        ),
        pipe=Pipe.V,
        category="vector",
        doc="Broadcast block within UB.",
    ))
    put(ApiSpec(
        name="GatherMask",
        params=(
            _p(ArgRole.DST, _UB, "dst"),
            _p(ArgRole.SRC0, _UB, "src0"),
            _p(ArgRole.SRC1, _UB, "src1"),
        ),
        pipe=Pipe.V,
        category="vector",
        doc="Mask-driven gather within UB.",
    ))

    # -- data movement ------------------------------------------------------
    for name in ("DataCopy", "DataCopyPad"):
        put(ApiSpec(
            name=name,
            params=(
                _p(ArgRole.DST, frozenset(D) - {D.UNKNOWN}, "dst"),
                _p(ArgRole.SRC, frozenset(D) - {D.UNKNOWN}, "src"),
                _p(ArgRole.COUNT, ANY_TENSOR, "count"),
                _p(ArgRole.PARAMS, ANY_TENSOR, "params"),
            ),
            pipe=None,  # derived from (dst, src) domains
            category="dma",
            doc="DMA transfer; the issuing pipeline follows the domain pair.",
        ))

    # -- cube / matmul ------------------------------------------------------
    put(ApiSpec(
        name="LoadData",
        params=(
            _p(ArgRole.DST, _L0AB, "dst"),
            _p(ArgRole.SRC, frozenset({D.L1, D.UB}), "src"),
            _p(ArgRole.PARAMS, ANY_TENSOR, "params"),
        ),
        pipe=Pipe.MTE1,
        category="cube",
        doc="Fractal load from L1 into a cube input buffer.",
    ))
    put(ApiSpec(
        name="LoadDataWithTranspose",
        params=(
            _p(ArgRole.DST, _L0AB, "dst"),
            _p(ArgRole.SRC, _L1, "src"),
            _p(ArgRole.PARAMS, ANY_TENSOR, "params"),
        ),
        pipe=Pipe.MTE1,
        category="cube",
        doc="Transposing fractal load from L1 into a cube input buffer.",
    ))
    put(ApiSpec(
        name="Mmad",
        params=(
            _p(ArgRole.DST, _L0C, "dstL0C"),
            _p(ArgRole.SRC0, _L0A, "a"),
            _p(ArgRole.SRC1, _L0B, "b"),
            _p(ArgRole.PARAMS, ANY_TENSOR, "params"),
        ),
        pipe=Pipe.M,
        category="cube",
        doc="Matrix multiply-accumulate: L0A x L0B -> L0C.",
    ))
    put(ApiSpec(
        name="MmadWithSparse",
        params=(
            _p(ArgRole.DST, _L0C, "dstL0C"),
            _p(ArgRole.SRC0, _L0A, "a"),
            _p(ArgRole.SRC1, _L0B, "b"),
            _p(ArgRole.PARAMS, ANY_TENSOR, "params"),
        ),
        pipe=Pipe.M,
        category="cube",
        doc="Sparse matrix multiply-accumulate: L0A x L0B -> L0C.",
    ))
    put(ApiSpec(
        name="InitConstValue",
        params=(
            _p(ArgRole.DST, _L0AB, "dst"),
            _p(ArgRole.PARAMS, ANY_TENSOR, "params"),
        ),
        pipe=Pipe.MTE1,
        category="cube",
        doc="Initialise a cube input buffer with a constant.",
    ))
    put(ApiSpec(
        name="Fixpipe",
        params=(
            _p(ArgRole.DST, _UB_OR_GM, "dst"),
            _p(ArgRole.SRC, _L0C, "srcL0C"),
            _p(ArgRole.PARAMS, ANY_TENSOR, "params"),
        ),
        pipe=Pipe.FIX,
        category="fixpipe",
        doc="Drain the cube accumulator (L0C) to UB or global memory.",
    ))
    put(ApiSpec(
        name="SetFmatrix",
        params=(_p(ArgRole.OTHER, ANY_TENSOR, "config"),),
        pipe=Pipe.S,
        category="scalar-config",
        doc="Configure the cube feature-matrix descriptor.",
    ))

    # -- low-level CCE cube loaders and contraction -------------------------
    # These are the raw-intrinsic spellings from cce_aicore_intrinsics.h that
    # Cube-only kernels (dequantization pipelines bypassing the vector unit)
    # are built from.  Operands are address-space-qualified raw pointers, so
    # the argument resolver must strip the (__ca__ T *) casts, and the volume
    # model stands in for the unavailable sizeof(void).
    _S4_VOLUME = TransferVolume(
        repeat_index=4, unit_bytes=512,
        basis="one packed-FP4 16x64 fractal (512 B) per repeat",
    )
    _MX_VOLUME = TransferVolume(
        repeat_index=4, unit_bytes=32,
        basis="one E8M0 scale granule (16 rows x 2 k-groups, 32 B) per repeat",
    )
    for name, dst, src, volume in (
        ("load_gm_to_ca_s4", _L0A, _GM, _S4_VOLUME),
        ("load_gm_to_cb_s4", _L0B, _GM, _S4_VOLUME),
        ("load_cbuf_to_ca_s4", _L0A, _L1, _S4_VOLUME),
        ("load_cbuf_to_cb_s4", _L0B, _L1, _S4_VOLUME),
        ("load_gm_to_ca_mx", _L0A, _GM, _MX_VOLUME),
        ("load_gm_to_cb_mx", _L0B, _GM, _MX_VOLUME),
        ("load_cbuf_to_ca_mx", _L0A, _L1, _MX_VOLUME),
        ("load_cbuf_to_cb_mx", _L0B, _L1, _MX_VOLUME),
    ):
        put(ApiSpec(
            name=name,
            params=(
                _p(ArgRole.DST, dst, "dst"),
                _p(ArgRole.SRC, src, "src"),
                # dstStride, srcStride, repeat, ... : scalar tuning knobs.
                _p(ArgRole.OTHER, None, "dstStride"),
                _p(ArgRole.OTHER, None, "srcStride"),
                _p(ArgRole.COUNT, None, "repeat"),
                _p(ArgRole.PARAMS, None, "params"),
            ),
            pipe=Pipe.MTE1,
            category="cube-load",
            doc=f"Raw CCE tile load; moves {volume.basis}.",
            volume=volume,
        ))
    put(ApiSpec(
        name="mad_mx",
        params=(
            _p(ArgRole.DST, _L0C, "dstL0C"),
            _p(ArgRole.OTHER, None, "dstOffset"),
            _p(ArgRole.SRC0, _L0A, "a"),
            _p(ArgRole.OTHER, None, "aOffset"),
            _p(ArgRole.SRC1, _L0B, "b"),
            _p(ArgRole.OTHER, None, "bOffset"),
            _p(ArgRole.PARAMS, ANY_TENSOR, "shape"),
            _p(ArgRole.PARAMS, ANY_TENSOR, "control"),
        ),
        pipe=Pipe.M,
        category="cube",
        doc="MX-format matrix multiply-accumulate: L0A x L0B -> L0C "
            "(fp4x2 operands with E8M0 microscales).",
    ))
    put(ApiSpec(
        name="DataCacheCleanAndInvalid",
        params=(_p(ArgRole.DST, _UB_OR_GM, "dst"),),
        pipe=Pipe.MTE3,
        category="dma",
        doc="Write back and invalidate the data cache for a tensor.",
    ))

    # -- scalar / configuration --------------------------------------------
    for name in (
        "SetVectorMask", "SetMaskNorm", "SetMaskCount", "ResetMask",
        "SetAtomicAdd", "SetAtomicNone", "SetAtomicMax", "SetAtomicMin",
        "SetMMLayoutTransform", "SetLoadDataBoundary", "SetPadValue",
    ):
        put(ApiSpec(
            name=name,
            params=(_p(ArgRole.OTHER, ANY_TENSOR, "config"),),
            pipe=Pipe.S,
            category="scalar-config",
            doc="Scalar-unit configuration write.",
        ))

    return table


API_TABLE: Dict[str, ApiSpec] = _build_table()


def lookup_api(name: str) -> Optional[ApiSpec]:
    """Look up an intrinsic by (possibly namespace-qualified) name."""
    return API_TABLE.get(name.rsplit("::", 1)[-1])


def is_known_api(name: str) -> bool:
    return lookup_api(name) is not None
