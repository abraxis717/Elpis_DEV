"""One local Elpis installation, RAM-only: pinned native libraries read from an immutable directory, and one volatile
ECS composite session at a time (docs/ELPIS_MISSION.md, Amendment 3).

Nothing here creates, writes, renames or removes a file. There is no home, no deployment configuration on disk and no
lineage on disk: ABI v7's ``init_deployment`` and ``Deployment.create_lineage`` (which wrote a home, its catalog, the
pinned library copies, the lineage file, a provisioned authority and a checkpoint store) are retired.

Trust. The libraries are read, read-only, from an operator build tree (each at its build path) or an installed library
directory (by file name). :func:`catalog_of` measures them into a catalog (the ``PinnedAuthority`` document: each
library's id, size and SHA-256) in memory; :class:`Installation` admits them (sealed memfd, pinned size and SHA-256,
SONAME-bound dependencies) only when that catalog's SHA-256 is the operator's pin. A changed library is refused
(``INSTALL_PIN``), never re-pinned implicitly. Pinning authenticates bytes, not authors (docs/NONCLAIMS.md).

The lineage is an ECS composite (K1 + typed memory, ELPISEC1) driven only through ``elpis.runtime.Runtime``: QUERY and
LEARN of admitted numeric stimuli and observe-only typed evidence. A :class:`Session` is one runtime session (a fresh
genesis, the configured sources founded) and the lineage's composite, created in memory from documented
deterministic initial weights (no model, no training) and anchored; closing it discards everything. A new session, in
this process or another, starts again from the same initial identity.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

from elpis.ECS.composite import CompositeConfig, CompositeLibrary, CompositeState
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.native_admission import admit_libraries, admit_library

from .authority import SessionConfig
from .composition import Runtime, RuntimeConfig
from .core import RuntimeLibrary
from .errors import CompositionError
from .observation import HacfObservationLibrary

__all__ = ("CATALOG_SCHEMA", "Installation", "LineageConfig", "Session", "catalog_of", "deployable_libraries",
           "initial_weights", "struct_digest")

CATALOG_SCHEMA = "elpis.inference-authority.v1"   # the PinnedAuthority schema
_W_DOMAIN = b"elpis.deployment.initial-w.v1"
_LIBRARY_LIMIT = 1 << 28

# (library id, file name, relative path in a build tree), in admission (dependency) order within each set.
_RUNTIME = ("elpis_runtime", "libelpis_runtime.so", "native/runtime/libelpis_runtime.so")
_ECS_SET = (("elpis_ecsg_math", "libelpis_ecsg_math.so", "native/ECS_G/libelpis_ecsg_math.so"),
            ("elpis_ecsg_k1", "libelpis_ecsg_k1.so", "native/ECS_G/libelpis_ecsg_k1.so"),
            ("elpis_ecsg_k1_fms", "libelpis_ecsg_k1_fms.so", "native/ECS_G/libelpis_ecsg_k1_fms.so"),
            ("elpis_ecs_memory", "libelpis_ecs_memory.so", "native/ECS_C/libelpis_ecs_memory.so"),
            ("elpis_ecs_composite", "libelpis_ecs_composite.so", "native/runtime/composite/libelpis_ecs_composite.so"))
_HACF = ("elpis_hacf_observation", "libelpis_hacf_observation.so", "native/runtime/libelpis_hacf_observation.so")


def deployable_libraries() -> tuple[tuple[str, str, str], ...]:
    """The native library set one installation pins and admits, in admission order: ``(library id, file name, path
    relative to an operator build tree)``. This is the one list of the files the reference runs; the catalog, the
    admission and the qualification instruments read it. (The ECS_C authority library and the testing RuntimeCore
    build of ABI v7 are retired: RuntimeCore embeds the volatile session.)"""
    return (_RUNTIME,) + _ECS_SET + (_HACF,)


def _refuse(code, detail):
    return CompositionError(code, detail)


def _read_regular(path: Path, limit: int = _LIBRARY_LIMIT) -> bytes:
    """A regular file's bytes, opened read-only without following a symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            raise _refuse("INSTALL_PATH", f"{path.name}: a regular file is required")
        chunks, left = [], st.st_size
        while left:
            chunk = os.read(fd, min(left, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _canonical(document) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"


def _relative(entry, build_tree: bool) -> str:
    return entry[2] if build_tree else entry[1]


def catalog_of(source: Path, *, build_tree: bool = False) -> tuple[bytes, str]:
    """The catalog of the libraries under ``source`` and its SHA-256: measured in memory from the files (read-only);
    nothing is written. The operator records the SHA-256 (the pin) once and passes it to every :class:`Installation`."""
    if not isinstance(source, Path) or not source.is_absolute():
        raise _refuse("INSTALL_PATH", "the library source must be an absolute path")
    entries = []
    for library_id, name, relative in deployable_libraries():
        try:
            data = _read_regular(source / (relative if build_tree else name))
        except FileNotFoundError:
            raise _refuse("INSTALL_PATH", f"{library_id} is missing under {source}") from None
        entries.append({"library_id": library_id, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    catalog = _canonical({"schema": CATALOG_SCHEMA, "provenance": "deployment",
                          "source": "elpis.runtime.deployment: an operator build's libraries, measured read-only",
                          "assets": [], "libraries": sorted(entries, key=lambda e: e["library_id"])})
    return catalog, hashlib.sha256(catalog).hexdigest()


def initial_weights(dim: int, width: int, seed: int) -> array:
    """The deterministic initial K1 weights of a lineage: ``dim * width`` binary64 values in [-0.18, 0.18), value
    i = (u64 of SHA-256(domain || seed || i)[:8] / 2^64 - 0.5) * 0.36. A documented initialization, not a model."""
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise _refuse("LINEAGE_CONFIG", "seed: a 64-bit unsigned integer")
    out = array("d")
    for i in range(dim * width):
        h = hashlib.sha256(_W_DOMAIN + seed.to_bytes(8, "little") + i.to_bytes(8, "little")).digest()
        out.append((int.from_bytes(h[:8], "little") / float(1 << 64) - 0.5) * 0.36)
    return out


@dataclass(frozen=True)
class LineageConfig:
    """The composite configuration every session starts from (K1 shape and memory shape; no memory capability is ever
    set), the seed of its initial weights and the session's entity capacity (the operator, the runtime, the
    sources)."""
    dim: int = 6
    width: int = 36
    max_rows: int = 64
    memory_dim: int = 64
    banks: int = 2
    slot_capacity: int = 64
    codebook_capacity: int = 128
    seed: int = 0
    session_capacity: int = 16

    def composite(self) -> CompositeConfig:
        return CompositeConfig(self.dim, self.width, max_rows=self.max_rows, memory_dim=self.memory_dim,
                               banks=self.banks, slot_capacity=self.slot_capacity,
                               codebook_capacity=self.codebook_capacity)


class _Libraries:
    """The admitted native libraries of one installation (one process admits them once)."""

    def __init__(self, runtime_library: RuntimeLibrary, composite: CompositeLibrary, hacf: HacfObservationLibrary,
                 identities: dict, k1, k1_fms):
        self.runtime, self.composite, self.hacf, self.identities = runtime_library, composite, hacf, identities
        self.k1, self.k1_fms = k1, k1_fms   # K1Library and the admitted K1 FMS adapter (FMS-resident K1 states)

    @classmethod
    def admit(cls, source: Path, authority: PinnedAuthority, build_tree: bool) -> "_Libraries":
        try:
            path = source / _relative(_RUNTIME, build_tree)
            runtime_library = RuntimeLibrary.admit(path.parent, path.name, authority, _RUNTIME[0])
            ecs = admit_libraries(source, authority, tuple((i, _relative((i, n, r), build_tree))
                                                           for i, n, r in _ECS_SET))
            hacf = admit_library(source, _relative(_HACF, build_tree), authority, _HACF[0])
        except CompositionError:
            raise
        except Exception as exc:   # ContractError and loader errors: an unadmissible library set
            raise _refuse("INSTALL_PIN", f"native library admission refused: {exc}") from exc
        identities = {a.library_id: {"sha256": a.identity.sha256, "size": a.identity.size} for a in ecs + (hacf,)}
        pinned = authority.libraries[_RUNTIME[0]]
        identities[_RUNTIME[0]] = {"sha256": pinned.sha256, "size": pinned.size}
        from elpis.ECS.k1 import K1Library
        return cls(runtime_library, CompositeLibrary(ecs[4].lib), HacfObservationLibrary(hacf), identities,
                   K1Library(ecs[1].lib), ecs[2].lib)


class Installation:
    """An installation, verified: the libraries under ``source`` measured into a catalog whose SHA-256 must be the
    operator's pin ``catalog_sha256``, then every library admitted from sealed memory. Read-only throughout."""

    def __init__(self, source: Path, catalog_sha256: str, *, build_tree: bool = False):
        catalog, measured = catalog_of(source, build_tree=build_tree)
        if type(catalog_sha256) is not str or measured != catalog_sha256:
            raise _refuse("INSTALL_PIN", "the measured catalog of the libraries does not match the pin")
        self.source, self.build_tree, self.catalog_sha256 = source, build_tree, measured
        try:
            self.authority = PinnedAuthority(catalog, expected_sha256=measured)
        except Exception as exc:
            raise _refuse("INSTALL_PIN", f"the native catalog is not a pinned authority: {exc}") from exc
        self.libraries = _Libraries.admit(source, self.authority, build_tree)

    def runtime_config(self, session: SessionConfig) -> RuntimeConfig:
        return RuntimeConfig(self.source / _relative(_RUNTIME, self.build_tree), native_authority=self.authority,
                             runtime_library_id=_RUNTIME[0], session=session)

    def session(self, config: LineageConfig, *, sources=(), nonce_seed: bytes | None = None) -> "Session":
        """A new session of the lineage ``config``: see :class:`Session`."""
        return Session(self, config, sources=sources, nonce_seed=nonce_seed)


class Session:
    """One live lineage, in memory: an open runtime session (the configured ``sources`` founded) and the lineage's
    composite, created from its deterministic initial weights and anchored. Everything it learns or records is
    volatile (``durability == "VOLATILE"``) and is discarded at :meth:`close`."""

    def __init__(self, installation: Installation, config: LineageConfig, *, sources=(), nonce_seed=None):
        if type(config) is not LineageConfig:
            raise _refuse("LINEAGE_CONFIG", "a LineageConfig is required")
        self.installation, self.config = installation, config
        session = SessionConfig(capacity=config.session_capacity, nonce_seed=nonce_seed, sources=tuple(sources))
        w = initial_weights(config.dim, config.width, config.seed)
        self.initial_w_sha256 = hashlib.sha256(w.tobytes()).hexdigest()
        self.runtime = Runtime(installation.runtime_config(session)).open()
        self.state = None
        try:
            self.state = CompositeState.create(installation.libraries.composite, config.composite(), memoryview(w))
            self.initial_identity = self.state.state_digest()
            self.anchored = self.runtime.anchor_cognition(self.state)
        except BaseException:
            self.close()
            raise

    def require_state(self) -> CompositeState:
        if self.state is None:
            raise _refuse("SESSION_CLOSED", "the session is closed: open a new one")
        return self.state

    def close(self) -> None:
        """End the session: the runtime session and the composite are gone (nothing was written anywhere)."""
        try:
            self.runtime.close()
        finally:
            if self.state is not None:
                self.state.close()
                self.state = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def struct_digest(values) -> str:
    """SHA-256 of binary64 values (an exact, platform-independent digest of an answer)."""
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()
