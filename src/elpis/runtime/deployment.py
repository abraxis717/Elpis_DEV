"""One local Elpis deployment: an operator-owned home directory, pinned native libraries and one ECS composite lineage.

Layout (created by this module, never edited by hand)::

    <home>/elpis.json             the deployment configuration and its trust root: the SHA-256 pin of the catalog
    <home>/native-catalog.json    the catalog of the pinned native libraries (``PinnedAuthority``)
    <home>/lib/                   the pinned libraries, copied from an operator build at ``init``
    <home>/lineage/lineage.json   the composite configuration and the provenance of its initial state
    <home>/lineage/continuity/    RuntimeCore's continuity directory (the durable identity and authority register)
    <home>/lineage/checkpoint/    the K1 Recovery R0 store, provisioned for the composite envelope

Trust. ``elpis.json`` is the operator's configuration: its ``catalog_sha256`` is the independent pin every start
verifies the catalog against, and every library is admitted (sealed memfd, pinned size and SHA-256, SONAME-bound
dependencies) from ``lib/`` at every start; nothing is ever loaded by path, from the build tree or from anywhere else.
A changed library, catalog or configuration is refused (``DEPLOYMENT_PIN``), never re-pinned implicitly. Pinning
authenticates bytes, not authors (docs/NONCLAIMS.md).

Safety. The home must be an absolute path that is not a symlink, owned by the current user and not writable by group
or others; ``init`` refuses a non-empty home and ``create`` an existing lineage: lineage data is never overwritten.
Nothing here keeps logs or history: the durable state is exactly continuity and the checkpoint slots.

The lineage is an ECS composite (K1 + typed memory, ELPISEC1) driven only through ``elpis.runtime.Runtime``: QUERY and
LEARN of admitted numeric stimuli, observe-only typed evidence, explicit recoverable publication, restart from the
last acknowledged state, operator reconciliation of an unresolved candidate. The initial K1 weights are a documented
deterministic function of a seed (no model, no training).
"""
from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

from elpis.ECS.composite import CompositeConfig, CompositeLibrary, CompositeState
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.native_admission import admit_libraries, admit_library

from .composition import Runtime, RuntimeConfig
from .core import RuntimeLibrary
from .errors import CompositionError
from .observation import HacfObservationLibrary
from .recovery import provision_k1_checkpoint

__all__ = ("DEPLOYMENT_SCHEMA", "LINEAGE_SCHEMA", "Deployment", "LineageConfig", "Session", "initial_weights",
           "init_deployment")

DEPLOYMENT_SCHEMA = "elpis.deployment.v1"
LINEAGE_SCHEMA = "elpis.lineage.v1"
CATALOG_SCHEMA = "elpis.inference-authority.v1"   # the persisted PinnedAuthority schema (kept verbatim)
_W_DOMAIN = b"elpis.deployment.initial-w.v1"

# (library id, file name, relative path in a build tree), in admission (dependency) order within each set.
_ECS_SET = (("elpis_ecsg_math", "libelpis_ecsg_math.so", "native/ECS/libelpis_ecsg_math.so"),
            ("elpis_ecsg_k1", "libelpis_ecsg_k1.so", "native/ECS/libelpis_ecsg_k1.so"),
            ("elpis_ecs_memory", "libelpis_ecs_memory.so", "native/ECS/libelpis_ecs_memory.so"),
            ("elpis_ecs_composite", "libelpis_ecs_composite.so", "native/ECS/libelpis_ecs_composite.so"))
_HACF = ("elpis_hacf_observation", "libelpis_hacf_observation.so", "native/runtime/libelpis_hacf_observation.so")
_RUNTIME = {False: ("elpis_runtime", "libelpis_runtime.so", "native/runtime/libelpis_runtime.so"),
            True: ("elpis_runtime_testing", "libelpis_runtime_testing.so",
                   "native/runtime/libelpis_runtime_testing.so")}


def _refuse(code, detail):
    return CompositionError(code, detail)


def _safe_home(home) -> Path:
    """An absolute home that is not a symlink, owned by this user and not writable by group or others (if it
    exists). Never resolved through a symlink."""
    if not isinstance(home, Path) or not home.is_absolute():
        raise _refuse("DEPLOYMENT_PATH", "the home must be an absolute path")
    if ".." in home.parts:
        raise _refuse("DEPLOYMENT_PATH", "the home path must not contain '..'")
    try:
        st = os.lstat(home)
    except FileNotFoundError:
        return home
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise _refuse("DEPLOYMENT_PATH", "the home must be a real directory, not a symlink or a file")
    if st.st_uid != os.geteuid():
        raise _refuse("DEPLOYMENT_PATH", "the home must be owned by the current user")
    if st.st_mode & 0o022:
        raise _refuse("DEPLOYMENT_PATH", "the home must not be writable by group or others")
    return home


def _write_once(path: Path, data: bytes, mode: int) -> None:
    """Create ``path`` with exactly ``data`` (never replacing an existing file), synced with its directory."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    _sync_dir(path.parent)


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_regular(path: Path, limit: int = 1 << 30) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            raise _refuse("DEPLOYMENT_PATH", f"{path.name}: a regular file is required")
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


def initial_weights(dim: int, width: int, seed: int) -> array:
    """The deterministic initial K1 weights of a new lineage: ``dim * width`` binary64 values in [-0.18, 0.18), value
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
    """The composite configuration of a lineage (K1 shape and memory shape; no memory capability is ever set)."""
    dim: int = 6
    width: int = 36
    max_rows: int = 64
    memory_dim: int = 64
    banks: int = 2
    slot_capacity: int = 64
    codebook_capacity: int = 128
    seed: int = 0

    def composite(self) -> CompositeConfig:
        return CompositeConfig(self.dim, self.width, max_rows=self.max_rows, memory_dim=self.memory_dim,
                               banks=self.banks, slot_capacity=self.slot_capacity,
                               codebook_capacity=self.codebook_capacity)


def init_deployment(home: Path, native_build: Path, *, testing: bool = False, source_commit: str | None = None):
    """Create a deployment at ``home`` from the libraries of an operator build at ``native_build`` (a CMake build
    tree). ``home`` must not exist or be empty. The libraries are copied into ``home/lib``, pinned in a catalog,
    admitted once from there (ABI checks included) before the configuration that names the catalog's pin is written;
    afterwards the build tree is never used again. ``testing`` deploys the testing RuntimeCore build (the same source
    with the testing feature's fault hooks; for crash-injection acceptance only)."""
    home = _safe_home(home)
    if not isinstance(native_build, Path) or not native_build.is_absolute():
        raise _refuse("DEPLOYMENT_PATH", "the native build must be an absolute path")
    if home.exists() and any(home.iterdir()):
        raise _refuse("DEPLOYMENT_EXISTS", f"{home} is not empty: an existing deployment is never overwritten")
    if source_commit is not None and (type(source_commit) is not str or len(source_commit) != 40 or
                                      any(c not in "0123456789abcdef" for c in source_commit)):
        raise _refuse("DEPLOYMENT_CONFIG", "source_commit: a 40-character lowercase hex commit id")
    home.mkdir(mode=0o700, parents=False, exist_ok=True)
    lib = home / "lib"
    lib.mkdir(mode=0o700)
    entries = []
    for library_id, name, relative in (_RUNTIME[testing],) + _ECS_SET + (_HACF,):
        data = _read_regular(native_build / relative)
        _write_once(lib / name, data, 0o500)
        entries.append({"library_id": library_id, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    catalog = _canonical({"schema": CATALOG_SCHEMA, "provenance": "deployment",
                          "source": "elpis.runtime.deployment: an operator build's libraries, pinned at init",
                          "assets": [], "libraries": sorted(entries, key=lambda e: e["library_id"])})
    pin = hashlib.sha256(catalog).hexdigest()
    _Libraries.admit(lib, PinnedAuthority(catalog, expected_sha256=pin), testing)   # verified before it is named
    _write_once(home / "native-catalog.json", catalog, 0o400)
    config = {"schema": DEPLOYMENT_SCHEMA, "catalog": "native-catalog.json", "catalog_sha256": pin,
              "runtime_library_id": _RUNTIME[testing][0], "runtime_library": _RUNTIME[testing][1],
              "abi": {"runtime": 5, "composite": 1}, "source_commit": source_commit}
    _write_once(home / "elpis.json", _canonical(config), 0o600)
    return Deployment(home)


class _Libraries:
    """The admitted native libraries of one deployment (one process admits them once)."""

    def __init__(self, runtime_library: RuntimeLibrary, composite: CompositeLibrary, hacf: HacfObservationLibrary,
                 identities: dict):
        self.runtime, self.composite, self.hacf, self.identities = runtime_library, composite, hacf, identities

    @classmethod
    def admit(cls, lib: Path, authority: PinnedAuthority, testing: bool) -> "_Libraries":
        runtime_id, runtime_name, _ = _RUNTIME[testing]
        try:
            runtime_library = RuntimeLibrary.admit(lib, runtime_name, authority, runtime_id)
            ecs = admit_libraries(lib, authority, tuple((i, n) for i, n, _ in _ECS_SET))
            hacf = admit_library(lib, _HACF[1], authority, _HACF[0])
        except CompositionError:
            raise
        except Exception as exc:   # ContractError and loader errors: an unadmissible library set
            raise _refuse("DEPLOYMENT_PIN", f"native library admission refused: {exc}") from exc
        identities = {a.library_id: {"sha256": a.identity.sha256, "size": a.identity.size}
                      for a in ecs + (hacf,)}
        pinned = authority.libraries[runtime_id]
        identities[runtime_id] = {"sha256": pinned.sha256, "size": pinned.size}
        return cls(runtime_library, CompositeLibrary(ecs[3].lib), HacfObservationLibrary(hacf), identities)


class Deployment:
    """An existing deployment, verified: the configuration, the catalog against its pin, every library admitted."""

    def __init__(self, home: Path):
        self.home = _safe_home(home)
        try:
            raw = _read_regular(self.home / "elpis.json", 1 << 16)
        except FileNotFoundError:
            raise _refuse("DEPLOYMENT_MISSING", f"no deployment at {self.home} (run init)") from None
        try:
            config = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _refuse("DEPLOYMENT_CONFIG", f"elpis.json: {exc}") from exc
        if type(config) is not dict or config.get("schema") != DEPLOYMENT_SCHEMA:
            raise _refuse("DEPLOYMENT_CONFIG", f"elpis.json: schema {DEPLOYMENT_SCHEMA} required")
        if config.get("abi") != {"runtime": 5, "composite": 1}:
            raise _refuse("DEPLOYMENT_ABI", f"deployment ABI {config.get('abi')} is not this runtime's (5, 1)")
        self.config = config
        self.testing = config["runtime_library_id"] == "elpis_runtime_testing"
        catalog = _read_regular(self.home / config["catalog"], 1 << 20)
        try:
            self.authority = PinnedAuthority(catalog, expected_sha256=config["catalog_sha256"])
        except Exception as exc:
            raise _refuse("DEPLOYMENT_PIN", f"the native catalog does not match its pin: {exc}") from exc
        self.libraries = _Libraries.admit(self.home / "lib", self.authority, self.testing)

    # -- the lineage -------------------------------------------------------------------------------------------------
    @property
    def lineage_dir(self) -> Path:
        return self.home / "lineage"

    def lineage(self) -> LineageConfig | None:
        """The lineage's configuration, or None when no lineage was created."""
        try:
            raw = _read_regular(self.lineage_dir / "lineage.json", 1 << 16)
        except FileNotFoundError:
            return None
        document = json.loads(raw)
        if document.get("schema") != LINEAGE_SCHEMA:
            raise _refuse("LINEAGE_CONFIG", f"lineage.json: schema {LINEAGE_SCHEMA} required")
        return LineageConfig(**document["config"])

    def runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(self.lineage_dir / "continuity", self.home / "lib" / self.config["runtime_library"],
                             native_authority=self.authority, runtime_library_id=self.config["runtime_library_id"],
                             k1_checkpoint_dir=self.lineage_dir / "checkpoint")

    def create_lineage(self, config: LineageConfig) -> dict:
        """Create the one lineage: provision its checkpoint store, create the composite from the deterministic initial
        weights, anchor it (durable first: the checkpoint slot, then continuity), record its configuration. Refused
        if a lineage (or a partial one) exists: lineage data is never overwritten."""
        if type(config) is not LineageConfig:
            raise _refuse("LINEAGE_CONFIG", "a LineageConfig is required")
        if self.lineage_dir.exists():
            raise _refuse("LINEAGE_EXISTS", f"{self.lineage_dir} exists: a lineage is never overwritten")
        composite = config.composite()
        w = initial_weights(config.dim, config.width, config.seed)
        self.lineage_dir.mkdir(mode=0o700)
        provision_k1_checkpoint(self.libraries.runtime, self.lineage_dir / "checkpoint",
                                self.libraries.composite.envelope_bytes(composite))
        with CompositeState.create(self.libraries.composite, composite, memoryview(w)) as x, \
                Runtime(self.runtime_config()) as runtime:
            anchored = runtime.anchor_cognition(x)
            identity = x.state_digest()
        document = {"schema": LINEAGE_SCHEMA, "config": asdict(config),
                    "initial_w_sha256": hashlib.sha256(w.tobytes()).hexdigest(),
                    "initial_identity": identity.hex(), "initialization": _W_DOMAIN.decode()}
        _write_once(self.lineage_dir / "lineage.json", _canonical(document), 0o400)
        return {"identity": identity.hex(), "state_format": anchored.state_format,
                "continuity_generation": anchored.generation, "initial_w_sha256": document["initial_w_sha256"]}

    def session(self) -> "Session":
        """Open the runtime over the lineage and restore exactly the last acknowledged composite (RESUMABLE), or report
        why it cannot be (no lineage, an unresolved candidate, a missing checkpoint): cognitive operations are then
        refused until the operator acts."""
        config = self.lineage()
        if config is None:
            raise _refuse("LINEAGE_MISSING", "no lineage: run create")
        return Session(self, config)


class Session:
    """One process's live lineage: the open runtime and the restored composite. Volatile work is lost at close unless
    it was published recoverably (the API says so: ``durability == "VOLATILE"``)."""

    def __init__(self, deployment: Deployment, config: LineageConfig):
        self.deployment, self.config = deployment, config
        self.runtime = Runtime(deployment.runtime_config()).open()
        self.state = None
        try:
            self.recovery = self.runtime.recover_k1()
            if self.recovery.disposition == "RESUMABLE":
                self.state = CompositeState.restore(deployment.libraries.composite, config.composite(),
                                                    self.recovery.envelope)
                if self.state.state_digest() != self.recovery.authorized:
                    raise _refuse("LINEAGE_MISMATCH", "the restored composite is not continuity's identity")
        except BaseException:
            self.close()
            raise

    def require_state(self) -> CompositeState:
        if self.state is None:
            raise _refuse(f"LINEAGE_{self.recovery.disposition}",
                          "the lineage is not resumable: see `status`; an unresolved candidate needs `reconcile`")
        return self.state

    def refresh(self) -> None:
        """After an operator reconciliation: re-read the disposition and restore the authorized composite."""
        if self.state is not None:
            raise _refuse("LINEAGE_BOUND", "a restored composite is already live")
        self.recovery = self.runtime.recover_k1()
        if self.recovery.disposition == "RESUMABLE":
            self.state = CompositeState.restore(self.deployment.libraries.composite, self.config.composite(),
                                                self.recovery.envelope)

    def close(self) -> None:
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
