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


def require_continuity_library(testing: bool = False) -> Path:
    """The Rust continuity library at its exact build path (native/continuity).

    Located by path, never by search: the cargo target directories inside the build tree
    also contain files named libelpis_continuity.so (including the testing build).
    """
    name = "libelpis_continuity_testing.so" if testing else "libelpis_continuity.so"
    path = native_build_dir() / "native" / "continuity" / name
    if not path.is_file():
        message = f"{name} not built under {native_build_dir()}; build with `cmake --build build`"
        if native_required():
            pytest.fail(message)
        pytest.skip(message + " (never an implicit PASS)")
    return path


def require_runtime_library(testing: bool = False) -> Path:
    """The Rust RuntimeCore library at its exact build path (native/runtime); it embeds continuity.

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


@pytest.fixture(scope="session")
def continuity_library() -> Path:
    """Production continuity library path (no test hooks)."""
    return require_continuity_library()


@pytest.fixture(scope="session")
def continuity_testing_library() -> Path:
    """Testing continuity library path: the same authority plus fault injection and I/O counters."""
    return require_continuity_library(testing=True)


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
