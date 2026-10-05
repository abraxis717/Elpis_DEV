"""Qualification mechanics: never construct any F world, including after qualification."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from research.ecs_k1_native import run_v2 as V
from research.ecs_k1_native import regression as G
from ...ECS_G.test_math_r0 import _library_path


@pytest.fixture(autouse=True)
def forbid_f_worlds(monkeypatch):
    build = V.X.T.build_world
    def guarded(spec, wid, gain):
        assert not wid.startswith("k1n2-"), "F is reserved for the write-once qualification"
        return build(spec, wid, gain)
    monkeypatch.setattr(V.X.T, "build_world", guarded)


def test_authority_and_empty_observations_cannot_qualify():
    p, q = V.authority()
    assert p["version"] == 2 and len(q["world_ids"]) == 32
    assert not any(V.recompute_gates({"sets": {}}).values())


@pytest.fixture(scope="module")
def development():
    library = _library_path()
    nat = V.D.Native(str(library))
    spec = copy.deepcopy(V.R3.SPEC)
    spec["regime"]["steps_per_experience"] = 20
    lab = V.X.Lab(nat.api, spec)
    # All runner paths, including the F laboratory path, exercised on a disjoint
    # synthetic development ID with reduced duration, never on an F world.
    wid = "test-k1n-v2-mechanics-0000"
    reference = V.X.qual_world(lab, wid)
    native = V.world_evidence(nat, lab, wid, reference)
    return library, nat, lab, wid, reference, native


def test_decode_and_local_operands_on_development_world(development):
    _, nat, lab, _, _, native = development
    previous = np.zeros(lab.idx.size * (lab.idx.size + 1) // 2)
    for t in V.S.STAGES:
        b = native["bounded"]["L2"][t]
        m, W, epoch = V.decode(nat, lab, bytes.fromhex(native["evidence"]["boundaries"][t]["envelope"]))
        assert np.array_equal(V.E.pack_upper(m.H), b["native_H"])
        assert np.array_equal(m.a, b["native_a"])
        assert np.array_equal(previous, b["previous_native_H"])
        assert b["h_relative"] == V.D._rel(b["native_H"], b["lab_H"])
        assert b["a_relative"] == V.D._rel(b["native_a"], b["lab_a"])
        assert b["h_relative"] <= 1e-12 and b["a_relative"] <= 1e-12
        previous = np.asarray(b["native_H"])
    for t, b in native["bounded"]["L1"].items():
        assert b["steps"] == 16 and b["native_epoch"] == b["source_epoch"] + 16
        assert not b["lab_refused"]
        assert b["w_relative"] == V.D._rel(b["native_W"], b["lab_W"])
        assert b["w_relative"] <= 1e-10
    assert all(native["checks"].values())


def test_structural_probe_on_development_world(development):
    library, _, lab, wid, _, _ = development
    result = V.structural(library, lab, lab.world(wid))
    assert all(r["calls"] == r["expected"] for r in result["operations"].values())
    assert {"learn_1", "learn_10", "learn_4000", "txn_abort", "txn_commit"} <= set(result["operations"])
    assert result["heap_allocations_before"] == result["heap_allocations_after"]
    assert V.structural_pass(result)
    del result["operations"]["learn_4000"]
    assert not V.structural_pass(result)


def test_bounds_use_raw_operands_and_never_descriptive_maxima(development):
    bounded = copy.deepcopy(development[-1]["bounded"])
    assert V.local_pass(bounded) and V.consolidation_pass(bounded)
    bounded["L1"]["A"]["native_W"][0][0] += 1.0
    bounded["L1"]["A"]["w_relative"] = 0.0  # forged summary cannot bypass raw operands
    assert not V.local_pass(bounded)
    bounded["L2"]["D"]["native_a"][0] += 1.0
    bounded["L2"]["D"]["a_relative"] = 0.0
    assert not V.consolidation_pass(bounded)


def test_disclosed_q_worker_and_clean_process():
    library = _library_path()
    V._init(str(library))
    wid = V.G.r3_qual()["world_ids"][0]
    result = V._job(("Q", wid))
    assert "error" not in result, result
    native = result["native"]
    assert V.exact_checks(native) == native["checks"]
    assert all(V.exact_checks(native).values())
    assert V.local_pass(native["bounded"]) and V.consolidation_pass(native["bounded"])
    det = V.native_determinism(library, wid, native["envelopes_sha256"])
    assert det["identical"]
    transplant = G.clean_transplant(library, {wid: native})
    assert transplant["child"] == transplant["reference"] and transplant["bitwise"]


def test_decision_uses_unchanged_laboratory_gates(development):
    _, _, lab, wid, reference, native = development
    # Decision rules are frozen. The development duration is deliberately too
    # short to qualify scientifically; equality is the machinery being tested.
    reference = {wid: reference}
    actual = V.decision(reference, {wid: native}, {"identical": True}, {"bitwise": True},
                        lab_det={"identical": True})
    expected = V.X.gates(reference, V.R3.SPEC, {"identical": True}, {"bitwise": True})["decision_record"]
    assert actual["reference"] == V.P3.plain(expected)
    assert all(actual["equal"].values())


def test_native_suite_report_is_complete():
    build = next(p for p in _library_path().parents if (p / "CMakeCache.txt").is_file())
    rec = V.native_suite(build)
    assert rec["returncode"] == 0, rec
    assert {t["name"] for t in rec["tests"]} == V.NATIVE_TESTS
    assert all(t["status"] == "run" and not t["failed"] and not t["skipped"] for t in rec["tests"])


def test_existing_record_is_refused_before_any_world(monkeypatch, tmp_path):
    record = tmp_path / "qualification.json"
    record.write_bytes(b"reserved")
    monkeypatch.setattr(V, "RECORD", record)
    with pytest.raises(V.V1.QualificationError, match="no retry"):
        V.qualify(Path("unused"))
    assert record.read_bytes() == b"reserved"


def test_dirty_tree_is_refused_before_any_world(monkeypatch):
    monkeypatch.setattr(V.V1, "_git", lambda *args: " M dirty.py")
    with pytest.raises(V.V1.QualificationError, match="entire tree clean"):
        V.implementation(Path("unused"))


def test_record_recomputes_without_constructing_worlds():
    if not V.RECORD.exists():
        pytest.skip("qualification has not run")
    record = json.loads(V.RECORD.read_bytes())
    body = record["body"]
    assert V.V1.digest("qualification", body) == record["digest"]
    assert V.recompute_gates(body) == body["gates"]
    assert body["verdict"] == ("QUALIFIED" if all(body["gates"].values()) else "NOT_QUALIFIED")
    assert body["implementation"]["dirty"] is False
