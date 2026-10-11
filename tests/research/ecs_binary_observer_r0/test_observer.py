"""ECS Binary Observation R0 (research/ecs_binary_observer_r0). RESEARCH_ONLY; NO_RUNTIME_AUTHORITY; UNQUALIFIED.

The native observer and the pure-Python reference law agree bitwise and refuse identically; observing a real K1
QUERY never moves it; the observer cannot reach ECS, the runtime, LEARN or continuity; the frozen evidence (once
present) still binds the laboratory.
"""
from __future__ import annotations

import ast
import json
import math
import struct
from array import array
from pathlib import Path

import numpy as np
import pytest

from elpis.ECS.k1 import K1Library

from research.ecs_binary_observer_r0 import lab
from research.ecs_binary_observer_r0 import law as L
from research.ecs_binary_observer_r0 import protocol as P
from research.ecs_binary_observer_r0 import run as R
from research.ecs_binary_observer_r0.observer import Observer, ObserverError

from ...conftest import admit_k1_set, native_build_dir, native_required, require_native_library

REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "research" / "ecs_binary_observer_r0"


@pytest.fixture(scope="module")
def observer_library():
    return require_native_library("elpis_ebo_r0")


def _gray(rng) -> bytes:
    return L.gray4_blob(np.sort(rng.normal(0.0, 1.0, 15)) + np.arange(15) * 1e-3)


def _sproj(rng) -> bytes:
    index, sign = lab.sproj_tables()
    return L.sproj_blob(float(rng.normal()), float(rng.uniform(0.5, 2.0)), index, sign, rng.normal(0.0, 1.0, 256).tolist())


def test_reference_law_matches_native_bitwise(observer_library):
    rng = np.random.default_rng(11)
    for blob in (_gray(rng), _sproj(rng)):
        obs = Observer(observer_library, blob)
        law = L.parse_law(blob)
        for dim, rows in ((1, 1), (6, 5), (6, 63), (6, 64), (6, 65), (3, 300), (64, 70), (6, 4096)):
            x = rng.normal(0.0, 1.0, (rows, dim))
            values = rng.normal(0.5, 1.5, rows)
            if law.encoder == L.GRAY4:                 # exact ties and signed zeros inside the window
                values[: min(rows, 3)] = law.thresholds[: min(rows, 3)]
                values[min(rows, 3) - 1] = -0.0 if rows > 3 else values[min(rows, 3) - 1]
            digest = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
            native = obs.encode(dim, rows, np.ascontiguousarray(x), array("d", values), digest)
            reference = L.encode(law, dim, rows, x.ravel().tolist(), values.tolist(), digest)
            assert native == reference and len(native) == L.RECORD_BYTES
            assert obs.check(native) == 0
            rec = L.check_record(law, native)
            assert rec.window == min(rows, 64) and rec.rows == rows and rec.state_identity == digest[:8]
            assert bool(rec.flags & L.FLAG_TRUNCATED) == (rows > 64)


def test_refusals_are_identical_and_leave_inputs_alone(observer_library):
    rng = np.random.default_rng(5)
    blob = _gray(rng)
    obs, law = Observer(observer_library, blob), L.parse_law(blob)
    x = rng.normal(0.0, 1.0, (70, 6))
    v = rng.normal(0.0, 1.0, 70)
    digest = bytes(32)

    def both(dim, rows, xs, vs):
        codes = []
        try:
            obs.encode(dim, rows, np.ascontiguousarray(xs), array("d", vs), digest)
            codes.append(0)
        except ObserverError as exc:
            codes.append(exc.code)
        try:
            L.encode(law, dim, rows, xs.ravel().tolist(), list(vs), digest)
            codes.append(0)
        except L.Refused as exc:
            codes.append(exc.code)
        return codes

    bad_v = v.copy()
    bad_v[3] = math.nan
    bad_x = x.copy()
    bad_x[2, 1] = math.inf
    far_v = v.copy()
    far_v[66] = math.nan                               # beyond the window: never read
    snapshot = (x.copy(), v.copy())
    assert both(6, 8, x, bad_v) == [L.NONFINITE] * 2
    assert both(6, 8, bad_x, v) == [L.NONFINITE] * 2
    assert both(6, 70, x, far_v) == [0, 0]
    assert both(65, 1, np.zeros((1, 65)), v[:1]) == [L.SHAPE] * 2
    assert np.array_equal(x, snapshot[0]) and np.array_equal(v, snapshot[1])
    for mutate, code in ((lambda b: b"X" + b[1:], L.LAW), (lambda b: b[:4] + b"\x02" + b[5:], L.VERSION),
                         (lambda b: b[:-1], L.INVALID),
                         (lambda b: b[:8] + struct.pack("<d", math.nan) + b[16:], L.LAW)):
        with pytest.raises(ObserverError) as native:
            Observer(observer_library, mutate(blob))
        with pytest.raises(L.Refused) as reference:
            L.parse_law(mutate(blob))
        assert native.value.code == reference.value.code == code


def test_record_admission_matches_reference(observer_library):
    rng = np.random.default_rng(9)
    gray, sproj = _gray(rng), _sproj(rng)
    obs = Observer(observer_library, gray)
    rec = obs.encode(6, 10, np.zeros((10, 6)), array("d", [0.0] * 10), bytes(32))
    assert obs.check(rec) == 0
    assert Observer(observer_library, sproj).check(rec) == L.LAW
    with pytest.raises(L.Refused):
        L.check_record(L.parse_law(sproj), rec)
    tampered = rec[:7] + b"\x00" + rec[8:]
    assert obs.check(tampered) == L.INVALID
    with pytest.raises(L.Refused):
        L.check_record(obs.law, tampered)


def test_observing_a_real_k1_query_never_moves_it(observer_library):
    k1 = K1Library(admit_k1_set()[1].lib)
    rng = np.random.default_rng(3)
    observers = {"GRAY4": Observer(observer_library, _gray(rng)), "SPROJ": Observer(observer_library, _sproj(rng))}
    world = lab.World("test-0000")                    # outside every registry
    off = lab.run_world(k1, world)
    on = lab.run_world(k1, world, observers, parity=True)
    assert on["failures"] == [] and off["failures"] == []
    assert lab.interference_witness(off) == lab.interference_witness(on)
    assert all(len(r) == L.RECORD_BYTES for s in on["states"] for recs in s["records"].values() for r in recs.values())


def test_the_observer_cannot_reach_ecs_runtime_learn_or_continuity():
    for name in ("observer.py", "law.py"):
        tree = ast.parse((LAB / name).read_text())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not any(m.startswith("elpis") for m in imported), (name, imported)
    tree = ast.parse((LAB / "lab.py").read_text())
    elpis = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("elpis")}
    assert elpis == {"elpis.ECS.k1"}                  # the laboratory drives K1; the observer never does
    for path in (REPO / "src").rglob("*.py"):
        assert "ecs_binary_observer" not in path.read_text(encoding="utf-8"), path


def test_library_imports_and_observation_path(observer_library):
    audit = R.audit_library(native_build_dir())
    if not audit["available"]:
        if native_required():
            pytest.fail("nm/objdump unavailable")
        pytest.skip("nm/objdump unavailable (never an implicit PASS)")
    assert audit["M5"] and audit["M6"], audit


def test_frozen_evidence_still_binds_the_laboratory():
    freeze = LAB / "evidence" / "freeze" / "summary.json"
    if not freeze.is_file():
        pytest.skip("no freeze yet")
    R._verify_freeze()
    done = LAB / "evidence" / "disposition" / "summary.json"
    if done.is_file():
        d = json.loads(done.read_text())
        qual = json.loads((LAB / "evidence" / "qual" / "summary.json").read_text())
        perf = json.loads((LAB / "evidence" / "perf_qual" / "summary.json").read_text())
        assert d["disposition"] == P.disposition(qual["mechanics"]["pass"],
                                                 {n: qual["gates"][n]["USEFUL"] for n in P.ENCODERS},
                                                 {n: perf["encoders"][n]["verdict"] for n in P.ENCODERS})
