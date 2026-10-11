"""The Python continuity adapter is a thin, stateless binding of the Rust authority.

The continuity law itself (record format, slot selection, crash matrix, evolution reservation and
finalization, bounds, locking, refusal of old and legacy layouts) is qualified in Rust and through the
C ABI (native/continuity: ``ctest -L continuity``). These tests prove only that Python reaches that
authority faithfully: values and digests round-trip, codes surface unchanged, and the adapter owns no
format, transition law or digest of its own.
"""
from __future__ import annotations

import ast
import ctypes
import hashlib
from pathlib import Path

import pytest

from elpis.continuity import (
    ContinuityError,
    ContinuityLibrary,
    ContinuityProcessDeath,
    ContinuitySnapshot,
    ContinuityStore,
    EvolutionAuthority,
)

from ..conftest import admit_native

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "native" / "continuity" / "tests" / "fixtures"
ADAPTER = REPO / "src" / "elpis" / "continuity" / "adapter.py"
SLOTS = ("continuity.a", "continuity.b")


@pytest.fixture(scope="module")
def lib(continuity_library):
    return ContinuityLibrary(admit_native(continuity_library).lib)


@pytest.fixture(scope="module")
def testing_lib(continuity_testing_library):
    return ContinuityLibrary(admit_native(continuity_testing_library).lib)


def _d(n) -> bytes:
    return hashlib.sha256(b"k1-%d" % n).digest()


def _h(n) -> str:
    return hashlib.sha256(b"receipt-%d" % n).hexdigest()


def _code(fn):
    with pytest.raises(ContinuityError) as info:
        fn()
    return info.value.code


def test_production_library_has_no_test_hooks_and_the_testing_library_does(lib, testing_lib):
    assert not lib.testing and testing_lib.testing
    assert not hasattr(lib._lib, "elpis_continuity_testing_fault")
    assert lib.record_size == testing_lib.record_size == 176


def test_frozen_v2_vectors_round_trip_through_the_adapter(lib):
    seen = 0
    for line in (FIXTURES / "continuity_v2_vectors.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, expect, gen, anchored, k1, rev, head, pending, assertion, evd, recd, raw = line.split()
        raw = bytes.fromhex(raw)
        if expect == "corrupt":
            assert _code(lambda: lib.decode_record(raw)) == "CONTINUITY_CORRUPT", name
        elif expect == "empty":
            assert lib.decode_record(raw) is None
        else:
            snap = lib.decode_record(raw)
            expected = ContinuitySnapshot(int(gen), bytes.fromhex(k1) if anchored == "1" else None,
                                          EvolutionAuthority(int(rev), head, assertion if pending == "1" else None))
            assert snap == expected, name
            assert (snap.digest, snap.evolution.digest) == (recd, evd), name
            assert lib.encode_record(snap) == raw and lib.encode_record(expected) == raw, name
            assert lib.evolution_digest(expected.evolution) == evd, name
        seen += 1
    assert seen == 29


def test_store_operations_and_codes_surface_unchanged(lib, tmp_path):
    path = tmp_path / "c"
    with ContinuityStore(lib, path) as store:
        genesis = store.snapshot()
        assert genesis == ContinuitySnapshot(1, None, EvolutionAuthority(0, "0" * 64)) and not genesis.anchored
        assert _code(lambda: store.commit_cognition_transition(_d(0), _d(1))) == "CONTINUITY_UNANCHORED"
        assert store.anchor_cognition(_d(0)).k1_state_digest == _d(0)
        assert _code(lambda: store.anchor_cognition(_d(9))) == "CONTINUITY_ALREADY_ANCHORED"
        assert _code(lambda: store.commit_cognition_transition(_d(5), _d(6))) == "CONTINUITY_STATE_MISMATCH"
        assert _code(lambda: store.anchor_cognition(b"short")) == "CONTINUITY_INVALID"  # validity first, as before
        assert _code(lambda: store.commit_cognition_transition(_d(0), "x" * 32)) == "CONTINUITY_INVALID"
        idle = store.snapshot().evolution
        assert _code(lambda: store.commit_evolution_transition(idle, _h(1))) == "CONTINUITY_EVOLUTION_NOT_PENDING"
        for bad in (None, b"x" * 32, "", "F" * 64, "x" * 64):
            assert _code(lambda: store.reserve_evolution_assertion(idle, bad)) == "CONTINUITY_INVALID"
        assert _code(lambda: store.reserve_evolution_assertion("not an authority", _h(9))) == \
            "CONTINUITY_AUTHORITY_MISMATCH"
        pending = store.reserve_evolution_assertion(idle, _h(100)).evolution
        assert pending == EvolutionAuthority(0, "0" * 64, _h(100)) and pending.digest != idle.digest
        assert _code(lambda: store.reserve_evolution_assertion(pending, _h(101))) == "CONTINUITY_EVOLUTION_PENDING"
        assert _code(lambda: store.commit_evolution_transition(pending, "0" * 64)) == "CONTINUITY_INVALID"
        final = store.commit_evolution_transition(pending, _h(1)).evolution
        assert final == EvolutionAuthority(1, _h(1)) and final.digest == lib.evolution_digest(final)
        assert _code(lambda: store.commit_evolution_transition(pending, _h(1))) == "CONTINUITY_AUTHORITY_MISMATCH"
        assert _code(ContinuityStore(lib, path).open) == "CONTINUITY_LOCKED"
        assert _code(store.open) == "CONTINUITY_OPEN"
        current = store.snapshot()
    with ContinuityStore(lib, path) as store:
        assert store.snapshot() == current and store.snapshot().digest == current.digest
    assert sorted((p.name, p.stat().st_size) for p in path.iterdir()) == [(SLOTS[0], 176), (SLOTS[1], 176)]
    assert _code(lambda: ContinuityStore(lib, "relative")) == "CONTINUITY_PATH"
    assert _code(ContinuityStore(lib, path).snapshot) == "CONTINUITY_UNINITIALIZED"
    (tmp_path / "legacy").mkdir()
    (tmp_path / "legacy" / "events.log").write_bytes(b"x")
    assert _code(ContinuityStore(lib, tmp_path / "legacy").open) == "CONTINUITY_LEGACY_STORAGE"


def test_testing_faults_and_counters_reach_python(testing_lib, lib, tmp_path):
    store = ContinuityStore(testing_lib, tmp_path / "c").open()
    store.anchor_cognition(_d(0))
    store.testing_counters(reset=True)
    store.commit_cognition_transition(_d(0), _d(1))
    assert store.testing_counters() == {"opens": 0, "preads": 0, "pread_bytes": 0, "pwrites": 1, "pwrite_bytes": 176,
                                        "data_syncs": 1, "dir_syncs": 0, "renames": 0, "unlinks": 0}
    store.testing_fault(1, 2)  # the next publication's write fails
    assert _code(lambda: store.commit_cognition_transition(_d(1), _d(2))) == "CONTINUITY_PUBLICATION_REFUSED"
    store.testing_fault(1, 1, 1)  # death after the next write
    with pytest.raises(ContinuityProcessDeath):
        store.commit_cognition_transition(_d(1), _d(2))
    store.close()
    with ContinuityStore(testing_lib, tmp_path / "c") as again:
        assert again.snapshot().k1_state_digest == _d(2)
    with ContinuityStore(lib, tmp_path / "c") as production:
        assert _code(lambda: production.testing_fault(1, 2)) == "CONTINUITY_INVALID"


def test_the_adapter_owns_no_format_transition_law_or_digest():
    tree = ast.parse(ADAPTER.read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"__future__", "ctypes", "dataclasses", "os", "pathlib"}, imported
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "os"}
    assert calls <= {"fsencode"}, calls  # no file I/O of its own
    source = ADAPTER.read_text()
    for marker in ("ELPCONT", "elpis.continuity.register", "evolution-authority.v2", "sha256", "flock(",
                   "os.fdatasync", "os.pwrite", "generation + 1", "revision + 1"):
        assert marker not in source, marker
    import elpis.continuity as continuity
    assert sorted(continuity.__all__) == ["ContinuityError", "ContinuityLibrary", "ContinuityProcessDeath",
                                          "ContinuitySnapshot", "ContinuityStore", "EvolutionAuthority"]
    public = {n for n in vars(ContinuityStore) if not n.startswith("_")}
    assert public == {"open", "close", "snapshot", "anchor_cognition", "commit_cognition_transition",
                      "reserve_evolution_assertion", "commit_evolution_transition", "testing_fault",
                      "testing_counters"}


def test_store_handles_hold_no_python_authority_state(lib, tmp_path):
    with ContinuityStore(lib, tmp_path / "c") as store:
        for n in range(1, 200):
            pending = store.reserve_evolution_assertion(store.snapshot().evolution, _h(n)).evolution
            store.commit_evolution_transition(pending, _h(n))
        assert sorted(vars(store)) == ["_handle", "directory", "library"]
