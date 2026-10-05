"""BiSheng (Clang AST) extraction engine - the MLIR frontend's first stage.

Runs the BiSheng compiler (Huawei's Clang fork; any sufficiently recent Clang
works) in ``-cc1 -ast-dump=json`` mode over an Ascend C translation unit and
returns the parsed JSON.  The extractor never regex-parses types or template
depths: monomorphisation, ``auto`` deduction, overload resolution and
constant folding are all left to the real C++ parser; this module only
orchestrates the invocation and filters the result.

Two header modes are supported:

``stub`` (default)
    Uses the bundled facade header (:mod:`.bisheng_support.stub_include`),
    which mirrors the Ascend C API surface the analyzer models.  Works on
    every machine with any clang-family compiler and spans the CANN versions
    the fixture corpus targets.

``real``
    Uses a real CANN include tree (``ASCEND_CANN_HOME``) with the shim prefix
    header for the compiler builtins the aicore front end normally provides.
    Higher fidelity, but the headers must be API-compatible with the kernel.

Toolchain discovery order:
    1. ``$ASCEND_BISHENG_BIN`` (direct binary path);
    2. the CANN 9.2 beta.2 installation on Linux/WSL;
    3. a ``bisheng`` (or ``clang``) on PATH - local or inside
       ``$ASCEND_BISHENG_CONTAINER_CMD``;
    4. WSL Docker with image ``$ASCEND_BISHENG_IMAGE`` (default
       ``cann85-cross-310p:latest``), which ships BiSheng at
       ``/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler``.

Because the compiler must run wherever the toolchain lives (often a Linux
container), the source is shipped over stdin and re-anchored with a ``#line``
directive, so every AST location still names the original file and line.

Results are memoised: extraction snapshots may be committed for tests under
``tests/data/mlir/`` and a content-addressed cache honours
``$ASCEND_AST_CACHE``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BishengToolchain",
    "ExtractionResult",
    "BishengError",
    "discover_toolchain",
    "extract_ast",
    "record_snapshot",
    "filter_main_file",
    "is_main_file",
    "node_line",
    "ARCH_DEFINES",
    "DEFAULT_DOCKER_IMAGE",
]

SUPPORT_DIR = Path(__file__).resolve().parent / "bisheng_support"
PREFIX_HEADER = SUPPORT_DIR / "extraction_prefix.h"
STUB_INCLUDE_DIR = SUPPORT_DIR / "stub_include"

#: Container image used by the Docker toolchain fallback.
DEFAULT_DOCKER_IMAGE = os.environ.get("ASCEND_BISHENG_IMAGE", "cann85-cross-310p:latest")

#: WSL distro used when Docker itself lives inside WSL (Windows hosts).
WSL_DISTRO = os.environ.get("ASCEND_WSL_DISTRO", "Ubuntu-22.04")

#: Path of the BiSheng compiler inside the default container.
CONTAINER_BISHENG = "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng"

#: Container-side paths of the BiSheng builtin and host stdlib headers.
CONTAINER_TOOLCHAIN_ROOT = "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler"
CONTAINER_STDLIB_ROOTS: Tuple[str, ...] = (
    "/usr/include/c++/11",
    "/usr/include/x86_64-linux-gnu/c++/11",
    "/usr/include/x86_64-linux-gnu",
    "/usr/include",
)

#: CANN install root inside the container (real header mode).
CONTAINER_CANN_HOME = "/usr/local/Ascend/cann-8.5.0/x86_64-linux"

#: Ascend C source roots, relative to a CANN home, for real header mode.
CANN_ASCENDC_ROOTS: Tuple[str, ...] = (
    "asc",
    "asc/impl/adv_api",
    "asc/impl/basic_api",
    "asc/impl/utils",
    "asc/include",
    "asc/include/adv_api",
    "asc/include/basic_api",
    "asc/include/aicpu_api",
    "asc/include/utils",
    "asc/include/interface",
    "ascendc/include/highlevel_api",
)

#: ``__NPU_ARCH__`` values by analyzer chip name (subset the headers branch on).
ARCH_DEFINES: Dict[str, str] = {
    "ascend910b": "2201",
    "ascend910c": "2201",
    "ascend351x": "3101",
    "ascend950pr": "3510",
}

_FALLBACK_ARCH = "2201"
CANN92_HOME = Path("/usr/local/Ascend/cann-9.2.0-beta.2")


class BishengError(RuntimeError):
    """Raised when no usable clang-family compiler could be invoked."""


@dataclass
class BishengToolchain:
    """How to reach a clang-family compiler with AST-dump support."""

    #: Shell-agnostic argv prefix that runs the compiler binary.
    argv: Tuple[str, ...]
    #: "local" - a binary on this machine; "docker" - via a container runner.
    kind: str = "local"
    #: Toolchain root holding ``lib/clang/<ver>/include`` builtin headers.
    builtin_include: Optional[str] = None
    #: System stdlib include roots usable with ``-isystem``.
    stdlib_roots: Tuple[str, ...] = ()
    #: CANN home for real header mode, when available.
    cann_home: Optional[str] = None
    #: Human-readable description for diagnostics.
    description: str = ""

    def describe(self) -> str:
        return self.description or " ".join(self.argv)


@dataclass
class ExtractionResult:
    """Outcome of one extraction run."""

    ast: dict
    errors: List[str] = field(default_factory=list)
    toolchain: Optional[BishengToolchain] = None
    header_mode: str = "stub"
    arch_define: str = _FALLBACK_ARCH
    #: Where the JSON came from: "live", "cache" or "snapshot".
    origin: str = "live"
    compiler_log: str = ""


# ---------------------------------------------------------------------------
# Toolchain discovery
# ---------------------------------------------------------------------------


def discover_toolchain(cann_home: Optional[str] = None,
                       source_dir: Optional[str] = None) -> Optional[BishengToolchain]:
    """Find a usable compiler, preferring explicit configuration.

    Order: ``$ASCEND_BISHENG_BIN``, CANN 9.2 beta.2, a local ``bisheng``/``clang`` on PATH,
    then the WSL-Docker image.  Returns ``None`` only when nothing was found
    (the caller decides whether that is fatal).
    """
    explicit = os.environ.get("ASCEND_BISHENG_BIN")
    if explicit and Path(explicit).exists():
        return _local_toolchain(Path(explicit))
    cann92_binary = CANN92_HOME / "bin" / "bisheng"
    if cann92_binary.is_file() and os.access(cann92_binary, os.X_OK):
        return _local_toolchain(cann92_binary)
    for name in ("bisheng", "bisheng.exe", "clang", "clang.exe"):
        found = shutil.which(name)
        if found:
            toolchain = _local_toolchain(Path(found))
            if toolchain is not None:
                return toolchain
    docker = _docker_runner(str(STUB_INCLUDE_DIR), source_dir or str(STUB_INCLUDE_DIR))
    if docker is not None:
        return BishengToolchain(
            argv=docker + (CONTAINER_BISHENG,),
            kind="docker",
            builtin_include=f"{CONTAINER_TOOLCHAIN_ROOT}/lib/clang/15.0.5/include",
            stdlib_roots=CONTAINER_STDLIB_ROOTS,
            cann_home=os.environ.get("ASCEND_CANN_HOME", CONTAINER_CANN_HOME),
            description=f"docker:{DEFAULT_DOCKER_IMAGE}",
        )
    return None


def _local_toolchain(binary: Path) -> Optional[BishengToolchain]:
    root = binary.resolve().parent.parent
    builtin = None
    if root.name == "bin" or (root / "lib").is_dir():
        lib = root.parent / "lib" if root.name == "bin" else root / "lib"
        for clang_dir in sorted(lib.glob("clang/*")) if lib.is_dir() else []:
            candidate = clang_dir / "include"
            if candidate.is_dir():
                builtin = str(candidate)
                break
    stdlib: Tuple[str, ...] = ()
    for guess in (
        "/usr/include/c++/11",
        "/usr/include/x86_64-linux-gnu/c++/11",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include",
    ):
        if os.path.isdir(guess):
            stdlib += (guess,)
    cann = os.environ.get("ASCEND_CANN_HOME")
    if not cann:
        for guess in (str(CANN92_HOME), "/usr/local/Ascend/cann-8.5.0/x86_64-linux",
                      "/usr/local/Ascend/ascend-toolkit/latest"):
            if os.path.isdir(guess):
                installed = Path(guess)
                cann = str(installed / "x86_64-linux") if (installed / "x86_64-linux").is_dir() else guess
                break
    return BishengToolchain(
        argv=(str(binary.resolve()),),
        kind="local",
        builtin_include=builtin,
        stdlib_roots=stdlib,
        cann_home=cann,
        description=str(binary.resolve()),
    )


#: Container-side mount points for the stub headers and the source directory.
CONTAINER_STUB_MOUNT = "/work/stub"
CONTAINER_SRC_MOUNT = "/work/src"


def _to_runner_path(path: str) -> str:
    """Convert a host path into a path the container runner can mount.

    WSL Docker on Windows mounts ``C:\\x`` as ``/mnt/c/x``; on Linux hosts the
    path is used verbatim.
    """
    if sys.platform == "win32" and len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path.replace("\\", "/")


def _docker_runner(stub_dir: str, source_dir: str) -> Optional[Tuple[str, ...]]:
    """Argv prefix that runs commands inside the BiSheng container, if any."""
    mounts = (
        "-v", f"{_to_runner_path(stub_dir)}:{CONTAINER_STUB_MOUNT}:ro",
        "-v", f"{_to_runner_path(source_dir)}:{CONTAINER_SRC_MOUNT}:ro",
    )
    explicit_cmd = os.environ.get("ASCEND_BISHENG_CONTAINER_CMD")
    if explicit_cmd:
        return tuple(explicit_cmd.split()) + mounts
    docker = shutil.which("docker")
    if docker:
        return (docker, "run", "--rm", "-i") + mounts + (DEFAULT_DOCKER_IMAGE,)
    if sys.platform == "win32":
        wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        if wsl:
            return (wsl, "-d", WSL_DISTRO, "-e", "docker", "run", "--rm", "-i") + \
                mounts + (DEFAULT_DOCKER_IMAGE,)
    return None


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def _arch_define(chip: str) -> str:
    from ..hardware import resolve_chip

    try:
        return ARCH_DEFINES.get(resolve_chip(chip).name, _FALLBACK_ARCH)
    except KeyError:
        return _FALLBACK_ARCH


def _compose_source(path: str, source: str) -> str:
    """Prefix + ``#line`` re-anchored source, piped to the compiler on stdin."""
    prefix = PREFIX_HEADER.read_text(encoding="utf-8").replace("#pragma once\n", "")
    # Forward slashes keep the path usable inside Linux containers.
    normalized = str(path).replace("\\", "/")
    return f"{prefix}\n#line 1 \"{normalized}\"\n{source}"


def _build_args(toolchain: BishengToolchain, path: str, chip: str,
                header_mode: str) -> List[str]:
    arch = _arch_define(chip)
    cce_arch = {"2201": "220", "3101": "310"}.get(arch, arch)
    args: List[str] = [
        "-cc1", "-ast-dump=json", "-std=c++17",
        "-D__CCE_KT_TEST__=1",
        f"-D__NPU_ARCH__={_arch_define(chip)}",
        f"-D__CCE_AICORE__={cce_arch}",
        f"-D__DAV_C{cce_arch}__=1",
        "-DASCEND_IS_AIC=__ascend_core_is_aic",
        "-DASCEND_IS_AIV=__ascend_core_is_aiv",
    ]
    if toolchain.builtin_include:
        args += ["-isystem", toolchain.builtin_include]
    for root in toolchain.stdlib_roots:
        args += ["-isystem", root]
    if header_mode == "real" and toolchain.cann_home:
        for rel in CANN_ASCENDC_ROOTS:
            args += ["-I", f"{toolchain.cann_home.rstrip('/')}/{rel}"]
    else:
        stub = CONTAINER_STUB_MOUNT if toolchain.kind == "docker" else \
            str(STUB_INCLUDE_DIR).replace("\\", "/")
        args += ["-I", stub]
    if toolchain.cann_home:
        for rel in ("include", "include/kernel_tiling", "include/experiment"):
            args += ["-I", f"{toolchain.cann_home.rstrip('/')}/{rel}"]
    source_dir = os.path.dirname(os.path.abspath(path)).replace("\\", "/")
    if source_dir:
        mounted = CONTAINER_SRC_MOUNT if toolchain.kind == "docker" else source_dir
        args += ["-I", mounted]
    args += ["-x", "c++", "-"]
    return args


def _run(toolchain: BishengToolchain, args: Sequence[str], source: str,
         timeout_s: int) -> Tuple[str, str]:
    argv = toolchain.argv + tuple(args)
    try:
        proc = subprocess.run(
            argv,
            input=source.encode("utf-8"),
            capture_output=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BishengError(f"failed to run {toolchain.describe()}: {exc}") from exc
    return proc.stdout.decode("utf-8", errors="replace"), proc.stderr.decode(
        "utf-8", errors="replace")


def _parse_errors(stderr: str) -> List[str]:
    """Extract diagnostics without touching type text (zero-regex policy)."""
    errors: List[str] = []
    for line in stderr.splitlines():
        marker = line.find("error: ")
        if marker >= 0:
            errors.append(line[marker + len("error: "):].strip())
    return errors


# ---------------------------------------------------------------------------
# Snapshots and cache
# ---------------------------------------------------------------------------


def _snapshot_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "data" / "mlir"


def _cache_dir() -> Optional[Path]:
    env = os.environ.get("ASCEND_AST_CACHE")
    if env:
        path = Path(env)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return None


def _content_key(path: str, source: str, chip: str, header_mode: str) -> str:
    digest = hashlib.sha256()
    digest.update(path.replace("\\", "/").encode("utf-8"))
    digest.update(b"\0")
    digest.update(source.encode("utf-8"))
    digest.update(b"\0")
    digest.update(f"{chip}|{header_mode}|{_arch_define(chip)}".encode("utf-8"))
    digest.update(b"\0")
    digest.update(PREFIX_HEADER.read_bytes())
    for stub in sorted(STUB_INCLUDE_DIR.rglob("*.h")):
        digest.update(stub.relative_to(STUB_INCLUDE_DIR).as_posix().encode("utf-8"))
        digest.update(stub.read_bytes())
    return digest.hexdigest()


#: Snapshot wrapper schema version.
SNAPSHOT_SCHEMA = 1


def _source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _snapshot_path(path: str, arch: str) -> Optional[Path]:
    stem = Path(path).stem
    candidate = _snapshot_dir() / f"{stem}.{arch}.json"
    if candidate.exists():
        return candidate
    return None


def _load_snapshot(candidate: Path, source: str) -> Optional[dict]:
    """Load a committed snapshot, validating it still matches the source."""
    try:
        wrapper = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(wrapper, dict) or wrapper.get("schema") != SNAPSHOT_SCHEMA:
        return None
    if wrapper.get("source_sha256") != _source_sha256(source):
        return None  # stale: the fixture changed after the snapshot was taken
    ast = wrapper.get("ast")
    return ast if isinstance(ast, dict) else None


def record_snapshot(path: str, source: str, chip: str = "ascend910b",
                    header_mode: str = "stub") -> Path:
    """Extract now and write a committed snapshot under ``tests/data/mlir``.

    Snapshots let the test suite exercise the full lowering without a
    toolchain; they carry the source hash so a stale snapshot is ignored
    rather than silently serving an old AST.
    """
    result = extract_ast(path, source, chip=chip, header_mode=header_mode,
                         use_cache=False)
    if result.errors:
        raise BishengError(
            f"cannot snapshot {path}: {len(result.errors)} extraction error(s), "
            f"first: {result.errors[0]}"
        )
    out_dir = _snapshot_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{Path(path).stem}.{_arch_define(chip)}.json"
    wrapper = {
        "schema": SNAPSHOT_SCHEMA,
        "source_sha256": _source_sha256(source),
        "arch": _arch_define(chip),
        "header_mode": header_mode,
        "ast": result.ast,
    }
    target.write_text(json.dumps(wrapper, separators=(",", ":")),
                      encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_ast(path: str, source: str, chip: str = "ascend910b",
                header_mode: str = "stub",
                toolchain: Optional[BishengToolchain] = None,
                timeout_s: int = 240,
                use_cache: bool = True) -> ExtractionResult:
    """Extract the Clang AST JSON of one Ascend C translation unit.

    ``path`` and ``source`` refer to the *original* file; the compiler sees a
    prefixed copy over stdin with ``#line`` re-anchoring, so AST locations
    keep the original file name and line numbers.
    """
    arch = _arch_define(chip)
    key = _content_key(path, source, chip, header_mode)
    cache = _cache_dir() if use_cache else None

    if use_cache:
        snapshot = _snapshot_path(path, arch)
        if snapshot is not None:
            data = _load_snapshot(snapshot, source)
            if data is not None:
                return ExtractionResult(ast=data, header_mode=header_mode,
                                        arch_define=arch, origin="snapshot")
        if cache is not None:
            cached = cache / f"{key}.json"
            if cached.exists():
                try:
                    data = json.loads(cached.read_text(encoding="utf-8"))
                    return ExtractionResult(ast=data, header_mode=header_mode,
                                            arch_define=arch, origin="cache")
                except (OSError, json.JSONDecodeError):
                    pass  # fall through to a live run
    if toolchain is None:
        source_dir = os.path.dirname(os.path.abspath(path)) or str(STUB_INCLUDE_DIR)
        toolchain = discover_toolchain(source_dir=source_dir)
    if toolchain is None:
        raise BishengError(
            "no BiSheng/Clang toolchain found; set ASCEND_BISHENG_BIN, install "
            "one on PATH, or provide a Docker image via ASCEND_BISHENG_IMAGE"
        )

    args = _build_args(toolchain, path, chip, header_mode)
    stdout, stderr = _run(toolchain, args, _compose_source(path, source), timeout_s)
    text = stdout.strip()
    if not text.startswith("{"):
        errors = _parse_errors(stderr) or [stderr.strip()[-400:] if stderr else "no output"]
        raise BishengError(
            f"AST extraction produced no JSON ({len(errors)} error(s)); first: "
            f"{errors[0] if errors else '?'}\n{stderr}"
        )
    data = json.loads(text)
    errors = _parse_errors(stderr)

    if cache is not None and not errors:
        try:
            (cache / f"{key}.json").write_text(
                json.dumps(data, separators=(",", ":")), encoding="utf-8")
        except OSError:
            pass
    return ExtractionResult(ast=data, errors=errors, toolchain=toolchain,
                            header_mode=header_mode, arch_define=arch, compiler_log=stderr)


# ---------------------------------------------------------------------------
# AST filtering
# ---------------------------------------------------------------------------


def _loc_file(loc: dict) -> Optional[str]:
    """File of a clang JSON location, unwrapping macro-expansion forms."""
    if not isinstance(loc, dict):
        return None
    for key in ("file", "expansionLoc"):
        value = loc.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            inner = _loc_file(value)
            if inner:
                return inner
    spelling = loc.get("spellingLoc")
    if isinstance(spelling, dict):
        return _loc_file(spelling)
    return None


def node_line(node: dict) -> int:
    """Best line number for a node (presumed lines honour ``#line``)."""
    loc = node.get("loc")
    if isinstance(loc, dict):
        for key in ("presumedLine", "line"):
            value = loc.get(key)
            if isinstance(value, int):
                return value
    rng = node.get("range")
    if isinstance(rng, dict):
        begin = rng.get("begin")
        if isinstance(begin, dict):
            for key in ("presumedLine", "line"):
                value = begin.get(key)
                if isinstance(value, int):
                    return value
    return 0


def is_main_file(node: dict, path: str) -> bool:
    """``True`` when the node's expansion is rooted in the target source file.

    With the stdin + ``#line`` extraction form, clang omits ``loc.file`` for
    presumed locations: nodes carrying an explicit ``file`` are matched
    against the target path, nodes attributed via ``includedFrom`` come from
    an included header, and the rest are main-file by construction.
    """
    loc = node.get("loc")
    if not isinstance(loc, dict):
        return False
    expansion = loc.get("expansionLoc")
    if isinstance(expansion, dict):
        inner = dict(expansion)
        inner.pop("includedFrom", None)
        if "file" in inner or "expansionLoc" in inner:
            return is_main_file({"loc": inner}, path)
    target = path.replace("\\", "/")
    target_stem = target.rsplit("/", 1)[-1]
    file = loc.get("file")
    if isinstance(file, str):
        normalized = file.replace("\\", "/")
        return normalized == target or normalized.endswith("/" + target_stem)
    if loc.get("includedFrom") is not None:
        return False
    return "presumedLine" in loc or "line" in loc


def _walk(node: dict):
    yield node
    inner = node.get("inner")
    if isinstance(inner, list):
        for child in inner:
            yield from _walk(child)


def filter_main_file(ast: dict, path: str) -> dict:
    """Return the AST pruned to nodes rooted in ``path``.

    System-header clutter (the CANN ``kernel_operator.h`` tree, libstdc++,
    compiler builtins) is discarded; decls whose expansion happened in the
    main file are kept with their full subtree.  This is the JSON equivalent
    of clang's ``isExpansionInMainFile`` filter.
    """

    def keep(node: dict) -> Optional[dict]:
        in_main = is_main_file(node, path)
        inner = node.get("inner")
        kept_children: List[dict] = []
        if isinstance(inner, list):
            for child in inner:
                kept = keep(child)
                if kept is not None:
                    kept_children.append(kept)
        if not in_main and not kept_children:
            return None
        pruned = {k: v for k, v in node.items() if k != "inner"}
        if kept_children:
            pruned["inner"] = kept_children
        return pruned

    return keep(ast) or {"kind": "TranslationUnitDecl", "inner": []}
