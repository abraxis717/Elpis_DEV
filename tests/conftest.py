"""Repository-wide test harness.

Two policies apply to every test:

* Network access is forbidden. Any attempt to open an INET/INET6 connection or
  resolve a host name fails the test. (CI additionally runs the core suite in a
  network namespace with only loopback, which covers subprocesses too.)
* Lanes (``tests/lanes.py``): a bare ``pytest`` runs the FAST lane only; ``--lane`` selects
  one lane (or ``all``); explicit test paths run what they name.
* Native libraries are explicit. Tests that need a compiled Elpis library read
  it from the build tree named by ``ELPIS_NATIVE_BUILD`` (default: ``build`` in
  the repository root). When ``ELPIS_REQUIRE_NATIVE=1`` a missing library is a
  failure, never a skip; CI sets it.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--lane", choices=("fast", "stress", "scientific", "historical", "all"), default=None,
                     help="run one qualification lane (tests/lanes.py); a bare `pytest` runs `fast`")


def pytest_collection_modifyitems(config, items):
    from tests.lanes import lane_of

    lane = config.getoption("--lane")
    if lane is None:
        if config.option.file_or_dir:
            return  # explicit paths: run exactly what was named
        lane = "fast"
    if lane == "all":
        return
    keep, drop = [], []
    for item in items:
        (keep if lane_of(item.nodeid) == lane else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


class NetworkAccessForbidden(RuntimeError):
    pass


def _forbid(*_args, **_kwargs):
    raise NetworkAccessForbidden("network access is forbidden in Elpis tests")


_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex


def _guarded_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _forbid()
    return _REAL_CONNECT(self, address)


def _guarded_connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _forbid()
    return _REAL_CONNECT_EX(self, address)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", _forbid)
    monkeypatch.setattr(socket, "getaddrinfo", _forbid)
    monkeypatch.setattr(socket, "gethostbyname", _forbid)
    yield


# ---------------------------------------------------------------------------
# Native libraries
# ---------------------------------------------------------------------------

def native_build_dir() -> Path:
    return Path(os.environ.get("ELPIS_NATIVE_BUILD", REPO_ROOT / "build")).resolve()


def native_required() -> bool:
    return os.environ.get("ELPIS_REQUIRE_NATIVE") == "1"


def find_native_library(stem: str) -> Path | None:
    build = native_build_dir()
    if not build.is_dir():
        return None
    matches = sorted(p for p in build.rglob(f"lib{stem}.so") if p.is_file())
    return matches[0] if matches else None


def require_native_library(stem: str) -> Path:
    found = find_native_library(stem)
    if found is None:
        message = (
            f"native library lib{stem}.so not found under {native_build_dir()}; "
            "build it with `cmake -S . -B build && cmake --build build`"
        )
        if native_required():
            pytest.fail(message)
        pytest.skip(message + " (never an implicit PASS)")
    return found


def require_runtime_library(testing: bool = False) -> Path:
    """The Rust RuntimeCore library at its exact build path (native/runtime); it embeds the ECS_C authority.

    Located by path, never by search (the cargo target directories hold same-named files).
    """
    name = "libelpis_runtime_testing.so" if testing else "libelpis_runtime.so"
    path = native_build_dir() / "native" / "runtime" / name
    if not path.is_file():
        message = f"{name} not built under {native_build_dir()}; build with `cmake --build build`"
        if native_required():
            pytest.fail(message)
        pytest.skip(message + " (never an implicit PASS)")
    return path


@pytest.fixture(scope="session")
def runtime_library() -> Path:
    """Production RuntimeCore library path (no test hooks)."""
    return require_runtime_library()


@pytest.fixture
def native_workspace(tmp_path_factory):
    """A private workspace root for descriptor-capability tests.

    Descriptor capabilities resolve beneath one trusted root without following
    symlinks. The workspace is created fresh; tests place both the (copied)
    native library and their assets beneath it.
    """
    root = tmp_path_factory.mktemp("elpis-native-workspace").resolve()
    return root


@pytest.fixture
def fms_file_library(native_workspace):
    source = require_native_library("elpis_fms_file_assets")
    target = native_workspace / "native" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


# ---------------------------------------------------------------------------
# Native admission (elpis.substrate.native_admission)
# ---------------------------------------------------------------------------
#
# Production loads every high-impact native library only through admission: a deployment-pinned catalog, a trusted
# descriptor root, sealed verified bytes. In tests the test author is the trust root: the catalog below pins the
# libraries of *this* build tree. It is a TEST fixture catalog, never deployment authority.

_PINNED = {   # library id -> path relative to the build tree
    "elpis_runtime": "native/runtime/libelpis_runtime.so",
    "elpis_runtime_testing": "native/runtime/libelpis_runtime_testing.so",
    "elpis_ecsg_math": "native/ECS_G/libelpis_ecsg_math.so",
    "elpis_ecsg_k1": "native/ECS_G/libelpis_ecsg_k1.so",
    "elpis_ecsg_k1_fms": "native/ECS_G/libelpis_ecsg_k1_fms.so",
    "elpis_ecs_memory": "native/ECS_C/libelpis_ecs_memory.so",
    "elpis_ecs_composite": "native/runtime/composite/libelpis_ecs_composite.so",
    "elpis_hacf_observation": "native/runtime/libelpis_hacf_observation.so",
    "elpis_ecs_c_authority": "native/ECS_C/authority/libelpis_ecs_c_authority.so",
}
_AUTHORITY = {}


def native_authority():
    """The TEST fixture catalog pinning this build tree's libraries (PinnedAuthority; pinned by its author)."""
    import hashlib
    import json
    from elpis.substrate.authority import PinnedAuthority

    build = native_build_dir()
    key = str(build)
    if key not in _AUTHORITY:
        libraries = []
        for library_id, relative in sorted(_PINNED.items()):
            path = build / relative
            if path.is_file():
                data = path.read_bytes()
                libraries.append({"library_id": library_id, "size": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()})
        catalog = json.dumps({"schema": "elpis.inference-authority.v1",
                              "source": "TEST fixture: the build tree's libraries, pinned by the test author",
                              "provenance": "deployment", "assets": [], "libraries": libraries},
                             sort_keys=True, separators=(",", ":")).encode()
        _AUTHORITY[key] = PinnedAuthority(catalog, expected_sha256=hashlib.sha256(catalog).hexdigest())
    return _AUTHORITY[key]


def library_id_of(path: Path) -> str:
    for library_id, relative in _PINNED.items():
        if Path(relative).name == Path(path).name:
            return library_id
    raise KeyError(path)


def admit_native(path: Path):
    """Admit one built library (sealed, pinned under the test catalog)."""
    from elpis.substrate.native_admission import admit_library
    return admit_library(path.parent, path.name, native_authority(), library_id_of(path))


def admit_k1_set():
    """Admit the K1 set in dependency order (math, K1, the K1 FMS adapter): ``(math, k1, k1_fms)`` admissions."""
    from elpis.substrate.native_admission import admit_libraries
    native = native_build_dir() / "native"
    for name in ("libelpis_ecsg_math.so", "libelpis_ecsg_k1.so", "libelpis_ecsg_k1_fms.so"):
        require_native_library(name[3:-3])
    return admit_libraries(native, native_authority(), (("elpis_ecsg_math", "ECS_G/libelpis_ecsg_math.so"),
                                                        ("elpis_ecsg_k1", "ECS_G/libelpis_ecsg_k1.so"),
                                                        ("elpis_ecsg_k1_fms", "ECS_G/libelpis_ecsg_k1_fms.so")))


def admit_composite_set():
    """Admit the ECS composite set in dependency order (math, K1, the memory, the composite): its admissions. The
    composite binds K1 and the memory by SONAME, so both must be the pinned, sealed objects."""
    from elpis.substrate.native_admission import admit_libraries
    native = native_build_dir() / "native"
    names = ("ECS_G/libelpis_ecsg_math.so", "ECS_G/libelpis_ecsg_k1.so", "ECS_C/libelpis_ecs_memory.so",
             "runtime/composite/libelpis_ecs_composite.so")
    for name in names:
        require_native_library(name.rsplit("/", 1)[1][3:-3])
    return admit_libraries(native, native_authority(), tuple((name.rsplit("/", 1)[1][3:-3], name) for name in names))


# The tests' durable authority (RuntimeCore ABI v7 has only the ECS_C authority): provisioned by the test author, the
# operator here, with one ACTIVE source for every memory source authority the suites' typed observations record, so
# the authority's admission never decides what a test measures (tests of admission found their own sources).
TEST_AUTHORITY_CAPACITY = 20
# The generation of a freshly provisioned test authority, before any publication: genesis, then each test source's
# founding and activation (two transitions each).
TEST_AUTHORITY_GENERATION = 1 + 2 * 16
_AUTHORITY_LIBRARY = {}


def test_sources():
    from elpis.runtime.authority import SourceGrant
    return tuple(SourceGrant(f"test source {n}", n, 64, "test scope") for n in range(16))


def authority_library():
    """The ECS_C authority's own ABI over this build tree's library, admitted under the test catalog (once)."""
    from elpis.runtime.authority import AuthorityLibrary
    key = str(native_build_dir())
    if key not in _AUTHORITY_LIBRARY:
        path = require_native_library("elpis_ecs_c_authority")
        _AUTHORITY_LIBRARY[key] = AuthorityLibrary.admit(path.parent, path.name, native_authority())
    return _AUTHORITY_LIBRARY[key]


def provision_test_authority(root: Path, sources=None, capacity: int = TEST_AUTHORITY_CAPACITY):
    """Provision an ECS_C authority under ``root`` (the operator's act) with ``sources`` (default: the test sources)."""
    from elpis.runtime.authority import provision_authority
    return provision_authority(authority_library(), root, capacity=capacity,
                               sources=test_sources() if sources is None else sources)


AUTHORITY_SLOTS = (("state", "authority.a"), ("state", "authority.b"), ("authority", "witness.a"),
                   ("authority", "witness.b"))
AUTHORITY_FILES = sorted(["authority", "authority/deployment.key", "authority/witness.a", "authority/witness.b",
                          "state", "state/authority.a", "state/authority.b"])


def authority_slots(root: Path) -> tuple[bytes, ...]:
    """The raw bytes of the durable authority's four slots under ``root`` (its two records and two witnesses; the
    deployment key is never rewritten). Two reads are equal exactly when nothing durable was published in between."""
    return tuple((root / d / n).read_bytes() for d, n in AUTHORITY_SLOTS)


def authority_files(root: Path) -> list[str]:
    """Every path under an authority root: exactly AUTHORITY_FILES once provisioned (nothing is ever added)."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def runtime_config(authority_dir: Path, runtime_library: Path, codec_pin: str | None = None,
                   evolution_pin: str | None = None):
    """A RuntimeConfig over a built RuntimeCore library, admitted under the test catalog, publishing to the ECS_C
    authority under ``authority_dir`` (provisioned on first use: see :func:`provision_test_authority`)."""
    from elpis.runtime import RuntimeConfig
    if not authority_dir.exists():
        provision_test_authority(authority_dir)
    return RuntimeConfig(authority_dir, runtime_library, codec_pin, native_authority(),
                         library_id_of(runtime_library), evolution_policy_sha256=evolution_pin)
