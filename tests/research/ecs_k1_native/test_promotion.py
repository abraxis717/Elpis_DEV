"""Canonical K1 admission is bound to one specifically admitted, fully verified native qualification record
(tests/research/_k1_promotion.py). A file that merely claims "QUALIFIED" admits nothing. Needs no NumPy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from .._k1_promotion import (ADMITTED_NATIVE_RECORDS, K1_FILES, R3_QUAL, admitted, native_digest,
                             native_qualified, native_record_admitted, r3_outcome_a)

REPO = Path(__file__).resolve().parents[3]
V1 = "research/ecs_k1_native/evidence/ecsg-k1-native.v1.qualification.json"


def _copy_authority(tmp: Path) -> Path:
    for rel in [str(R3_QUAL), V1, "research/ecs_k1_native/specs/ecsg-k1-native.v1.plan.json",
                *[pin["plan"] for pin in ADMITTED_NATIVE_RECORDS.values()], *ADMITTED_NATIVE_RECORDS]:
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, tmp / rel)
    return tmp


def _forge(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"body": body, "digest": native_digest("qualification", body)}), encoding="ascii")


def test_forged_qualified_records_admit_nothing(tmp_path):
    root = _copy_authority(tmp_path)
    assert r3_outcome_a(root)
    v1 = json.loads((root / V1).read_bytes())["body"]
    forged = dict(v1, verdict="QUALIFIED", experiment="ecsg-k1-native.v2")
    _forge(root / "research/ecs_k1_native/evidence/evil.qualification.json", forged)
    _forge(root / "research/ecs_k1_native/evidence/ecsg-k1-native.v3.qualification.json", forged)
    for relative in ADMITTED_NATIVE_RECORDS:   # an admitted path whose bytes were replaced, digest recomputed
        _forge(root / relative, forged)
    assert not native_qualified(root)
    assert admitted(root) == frozenset()


def test_the_v1_record_is_never_admitted():
    assert V1 not in ADMITTED_NATIVE_RECORDS
    v1 = json.loads((REPO / V1).read_bytes())
    assert v1["body"]["verdict"] == "NOT_QUALIFIED"


def test_every_admitted_record_verifies_completely_and_only_then_admits_k1():
    for relative, pin in ADMITTED_NATIVE_RECORDS.items():
        assert native_record_admitted(REPO, relative, pin), relative
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == pin["sha256"]
    assert native_qualified(REPO) == bool(ADMITTED_NATIVE_RECORDS)
    assert admitted(REPO) == (K1_FILES if ADMITTED_NATIVE_RECORDS else frozenset())


def test_a_tampered_admitted_record_is_refused(tmp_path):
    root = _copy_authority(tmp_path)
    for relative, pin in ADMITTED_NATIVE_RECORDS.items():
        data = bytearray((root / relative).read_bytes())
        data[len(data) // 2] ^= 1
        (root / relative).write_bytes(bytes(data))
        assert not native_record_admitted(root, relative, pin)


@pytest.mark.parametrize("binding", ("sha256", "raw_sha256", "digest", "experiment", "plan", "plan_sha256",
                                      "harness_commit", "r3_qual_digest"))
def test_every_admission_binding_is_required(binding):
    for relative, pin in ADMITTED_NATIVE_RECORDS.items():
        wrong = dict(pin, **{binding: "not-the-admitted-authority"})
        assert not native_record_admitted(REPO, relative, wrong), binding
