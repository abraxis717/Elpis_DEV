"""High-impact native code runs only from sealed, pinned bytes (elpis.substrate.native_admission).

RuntimeCore, native K1, the K1 FMS adapter and the continuity ABI are admitted exactly as the HACF bridges are: a
deployment-pinned catalog (its own pin from trusted configuration), a trusted descriptor root without symlink escape,
bytes copied into a sealed memfd, hashed against the pin and loaded from the seal. Every attack below is refused
before the library runs; the managed runtime additionally refuses a K1 state whose library was not admitted. The
catalogs here are TEST fixtures pinned by their author over copies of the built libraries.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from elpis.runtime import Runtime, RuntimeConfig
from elpis.runtime.cognition import QueryRequest, run_query
from elpis.runtime.composition import CompositionError
from elpis.runtime.core import RuntimeLibrary
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.contracts import ContractError
from elpis.substrate.native_admission import admission_of, admit_libraries, admit_library

from ..conftest import native_build_dir, require_native_library, require_runtime_library
from ._turn_fixtures import TEST_CODEC_PIN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import DIM, WIDTH, initial_w

REPO = Path(__file__).resolve().parents[2]


def _catalog(entries, provenance="deployment"):
    """A TEST catalog: ``{library_id: bytes}`` pinned by its author (the test)."""
    spec = {"schema": "elpis.inference-authority.v1", "source": "TEST fixture catalog", "provenance": provenance,
            "assets": [], "libraries": [{"library_id": k, "size": len(v), "sha256": hashlib.sha256(v).hexdigest()}
                                        for k, v in sorted(entries.items())]}
    document = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return PinnedAuthority(document, expected_sha256=hashlib.sha256(document).hexdigest())


@pytest.fixture
def workspace(tmp_path):
    """A trusted root holding copies of the built RuntimeCore libraries."""
    root = tmp_path / "native"
    root.mkdir()
    for testing in (False, True):
        source = require_runtime_library(testing=testing)
        shutil.copyfile(source, root / source.name)
    return root.resolve()


def _bytes(path):
    return Path(path).read_bytes()


def _refused(call, code="IDENTITY"):
    with pytest.raises((ContractError, CompositionError)) as info:
        call()
    value = info.value
    return getattr(value, "code", None) if isinstance(value, CompositionError) else value.code.name


def test_the_genuine_runtime_library_is_admitted_and_bound(workspace):
    authority = _catalog({"elpis_runtime": _bytes(workspace / "libelpis_runtime.so")})
    library = RuntimeLibrary.admit(workspace, "libelpis_runtime.so", authority)
    assert admission_of(library.admission.lib) is library.admission
    assert library.admission.lib._name.startswith("/proc/self/fd/")   # loaded from the seal, not the path


def test_a_modified_library_is_refused(workspace):
    authority = _catalog({"elpis_runtime": _bytes(workspace / "libelpis_runtime.so")})
    data = bytearray(_bytes(workspace / "libelpis_runtime.so"))
    data[len(data) // 2] ^= 0x01
    (workspace / "libelpis_runtime.so").write_bytes(bytes(data))
    with pytest.raises(CompositionError) as info:
        RuntimeLibrary.admit(workspace, "libelpis_runtime.so", authority)
    assert info.value.code == "RUNTIME_UNPINNED" and "SHA-256" in str(info.value)


def test_a_substituted_path_and_the_right_abi_with_the_wrong_bytes_are_refused(workspace):
    # The testing library exports the same RuntimeCore ABI (v3) with different bytes.
    authority = _catalog({"elpis_runtime": _bytes(workspace / "libelpis_runtime.so")})
    shutil.copyfile(workspace / "libelpis_runtime_testing.so", workspace / "substitute.so")
    with pytest.raises(CompositionError) as info:
        RuntimeLibrary.admit(workspace, "substitute.so", authority)
    assert info.value.code == "RUNTIME_UNPINNED"
    os.replace(workspace / "substitute.so", workspace / "libelpis_runtime.so")
    with pytest.raises(CompositionError) as info:
        RuntimeLibrary.admit(workspace, "libelpis_runtime.so", authority)
    assert info.value.code == "RUNTIME_UNPINNED"


def test_symlinks_and_escapes_are_refused(workspace, tmp_path):
    genuine = _bytes(workspace / "libelpis_runtime.so")
    authority = _catalog({"elpis_runtime": genuine})
    os.symlink(workspace / "libelpis_runtime.so", workspace / "link.so")
    assert _refused(lambda: admit_library(workspace, "link.so", authority, "elpis_runtime")) in ("IO", "IDENTITY",
                                                                                                  "UNSUPPORTED")
    linked_root = tmp_path / "linked-root"
    os.symlink(workspace, linked_root)
    assert _refused(lambda: admit_library(linked_root, "libelpis_runtime.so", authority, "elpis_runtime")) in (
        "IO", "UNSUPPORTED", "IDENTITY")
    assert _refused(lambda: admit_library(workspace, "../native/libelpis_runtime.so", authority,
                                          "elpis_runtime")) == "INVALID"


def test_a_wrong_digest_and_an_unauthorized_identifier_are_refused(workspace):
    genuine = _bytes(workspace / "libelpis_runtime.so")
    wrong = _catalog({"elpis_runtime": genuine + b"x"})               # the pin names other bytes
    assert _refused(lambda: admit_library(workspace, "libelpis_runtime.so", wrong, "elpis_runtime")) == "IDENTITY"
    other = _catalog({"elpis_continuity": genuine})                    # the right bytes, another identifier
    assert _refused(lambda: admit_library(workspace, "libelpis_runtime.so", other, "elpis_runtime")) == "IDENTITY"
    # Admitted under a non-RuntimeCore identifier, the right bytes still do not make a RuntimeLibrary.
    with pytest.raises(CompositionError) as info:
        RuntimeLibrary.admit(workspace, "libelpis_runtime.so", other, "elpis_continuity")
    assert info.value.code == "RUNTIME_UNPINNED"
    with pytest.raises(CompositionError) as info:
        RuntimeLibrary(admit_library(workspace, "libelpis_runtime.so", other, "elpis_continuity"))
    assert info.value.code == "RUNTIME_UNPINNED"
    synthetic = _catalog({"elpis_runtime": genuine}, provenance="synthetic-test")
    assert _refused(lambda: admit_library(workspace, "libelpis_runtime.so", synthetic, "elpis_runtime")) == "IDENTITY"


def test_the_runtime_never_loads_an_unpinned_library(workspace, tmp_path):
    for config in (RuntimeConfig(tmp_path / "c", workspace / "libelpis_runtime.so", TEST_CODEC_PIN),
                   RuntimeConfig(tmp_path / "c", workspace / "libelpis_runtime.so", TEST_CODEC_PIN,
                                 _catalog({"elpis_runtime": b"other bytes"}))):
        with pytest.raises(CompositionError) as info:
            Runtime(config)
        assert info.value.code == "RUNTIME_UNPINNED"
    with pytest.raises(CompositionError):
        RuntimeLibrary(object())   # no path constructor exists


def test_the_managed_runtime_refuses_k1_code_that_was_not_admitted(tmp_path):
    import ctypes
    from elpis.ECS.k1 import K1Library, K1State
    from .test_codec_ecs_turn import _config
    k1 = K1Library(ctypes.CDLL(str(require_native_library("elpis_ecsg_k1"))))   # research/unmanaged loading
    assert admission_of(k1._lib) is None
    with K1State.create(k1, DIM, WIDTH, initial_w().reshape(-1).tolist()) as state:
        before = state.snapshot()
        with Runtime(_config(tmp_path / "c")) as runtime:
            for call in (lambda: runtime.anchor_cognition(state),
                         lambda: runtime.run_query(state, QueryRequest("q", ByteTokens(), admitted(FixtureMap())))):
                with pytest.raises(CompositionError) as info:
                    call()
                assert info.value.code == "ECS_NATIVE_UNADMITTED"
            assert not runtime.continuity.snapshot().anchored and state.managed_lease == 0
        # Unmanaged research use of the same state stays possible.
        assert run_query(state, QueryRequest("q", ByteTokens(), admitted(FixtureMap()))).text == "ok"
        assert state.snapshot() == before


_DEPENDENCY_PROBE = r"""
import ctypes, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.native_admission import admit_libraries
root, preload, catalog, pin = Path(sys.argv[2]), sys.argv[3], sys.argv[4].encode(), sys.argv[5]
if preload != "-":
    ctypes.CDLL(preload)    # an earlier pathname load that now carries the SONAME libelpis_ecsg_k1.so
try:
    admit_libraries(root, PinnedAuthority(catalog, expected_sha256=pin),
                    (("elpis_ecsg_math", "libelpis_ecsg_math.so"), ("elpis_ecsg_k1", "libelpis_ecsg_k1.so"),
                     ("elpis_ecsg_k1_fms", "libelpis_ecsg_k1_fms.so")))
    print(json.dumps({"admitted": True}))
except Exception as exc:
    print(json.dumps({"admitted": False, "error": str(exc)}))
"""


def test_a_dependency_bound_to_unpinned_bytes_refuses_the_admission(tmp_path):
    ecs = native_build_dir() / "native" / "ECS"
    names = ("libelpis_ecsg_math.so", "libelpis_ecsg_k1.so", "libelpis_ecsg_k1_fms.so")
    for name in names:
        require_native_library(name[3:-3])
    root = tmp_path / "ecs"
    root.mkdir()
    for name in names:
        shutil.copyfile(ecs / name, root / name)
    entries = {name[3:-3]: (root / name).read_bytes() for name in names}
    document = _document(entries)
    pin = hashlib.sha256(document.encode()).hexdigest()
    # A tampered copy of K1 (one byte of its section-name table: it still loads, with the same SONAME).
    tampered = tmp_path / "tampered" / "libelpis_ecsg_k1.so"
    tampered.parent.mkdir()
    data = bytearray(entries["elpis_ecsg_k1"])
    data[-1] ^= 0x01
    tampered.write_bytes(bytes(data))

    def probe(preload):
        result = subprocess.run([sys.executable, "-c", _DEPENDENCY_PROBE, str(REPO / "src"), str(root.resolve()),
                                 str(preload), document, pin],
                                capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    assert probe("-") == {"admitted": True}                       # clean process: every dependency is sealed
    assert probe(ecs / "libelpis_ecsg_k1.so") == {"admitted": True}   # pathname-loaded, but the pinned bytes
    refused = probe(tampered)                                     # the loader would bind K1 FMS to other bytes
    assert refused["admitted"] is False and "libelpis_ecsg_k1.so" in refused["error"]


def _document(entries):
    spec = {"schema": "elpis.inference-authority.v1", "source": "TEST fixture catalog", "provenance": "deployment",
            "assets": [], "libraries": [{"library_id": k, "size": len(v), "sha256": hashlib.sha256(v).hexdigest()}
                                        for k, v in sorted(entries.items())]}
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def test_canonical_source_loads_no_native_library_by_pathname():
    """The only dlopen in canonical source is the sealed memfd load (and libc's own handle)."""
    import re
    offenders = []
    for path in sorted((REPO / "src" / "elpis").rglob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            for call in re.findall(r"CDLL\(([^)]*)\)", code):
                if call.strip() not in ("None", "None, use_errno=True") and "/proc/self/fd/" not in call:
                    offenders.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
    assert not offenders, offenders
