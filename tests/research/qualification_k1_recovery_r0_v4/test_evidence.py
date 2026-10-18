"""K1 Recovery R0 v4: the committed QUAL evidence validates, and the instrument in the tree is the frozen one.

The evidence (research/qualification/k1_recovery_r0_v4/evidence/) was written once by the frozen QUAL run on the
accepted source. The repository keeps no file over 2 MiB, so the sealed ``records.jsonl`` is committed gzip-compressed
(``records.jsonl.gz``); it is restored here into the sealed directory as written, and SHA256SUMS (checked by the
validator) proves it byte-identical. The independent validator (check.py evidence) then verifies that directory
(exactly its files, every SHA-256, the run bound to its freeze record and records, the observed checkout and
libraries equal to the freeze, the verdict recomputed from the records). The freeze record committed beside the
instrument is byte-identical to the evidence's, and the instrument files in the tree are still the ones the freeze
bound (a changed instrument is a new instrument and needs new evidence). Tamper resistance of the checker itself is
test_instrument.py's (DEV data).
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]
INSTRUMENT = REPO / "research" / "qualification" / "k1_recovery_r0_v4"
EVIDENCE = INSTRUMENT / "evidence"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"k1r4_{name}", INSTRUMENT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed(into: Path) -> Path:
    """The sealed directory as ``runner.py seal`` wrote it: every file copied, ``records.jsonl`` decompressed."""
    into.mkdir()
    for p in EVIDENCE.iterdir():
        if p.suffix == ".gz":
            (into / p.stem).write_bytes(gzip.decompress(p.read_bytes()))
        else:
            shutil.copyfile(p, into / p.name)
    return into


def test_the_committed_evidence_is_the_sealed_set_with_the_records_compressed():
    assert sorted(p.name for p in EVIDENCE.iterdir()) == sorted(
        ["freeze.json", "records.jsonl.gz", "run.json", "verdict.json", "SHA256SUMS"])


def test_the_sealed_evidence_validates_and_its_verdict_is_recomputed(tmp_path):
    assert _load("check").verify_evidence(_sealed(tmp_path / "sealed"), INSTRUMENT / "plan.json") == []


def test_the_committed_freeze_record_is_the_evidences_and_the_instrument_is_still_the_frozen_one():
    assert (INSTRUMENT / "FREEZE.json").read_bytes() == (EVIDENCE / "freeze.json").read_bytes()
    freeze = json.loads((EVIDENCE / "freeze.json").read_text())
    for name, digest in freeze["instrument"]["files"].items():
        assert _sha(INSTRUMENT / name) == digest, name


def test_the_run_was_the_complete_qual_plan_on_the_frozen_commit():
    run, freeze = json.loads((EVIDENCE / "run.json").read_text()), json.loads((EVIDENCE / "freeze.json").read_text())
    assert (run["inputs"], run["dev_runs"], run["only"], run["post_run_problems"]) == ("qual", None, None, [])
    assert run["observed"]["source"]["commit"] == freeze["source"]["commit"]
    assert run["observed"]["source"]["clean"] is True and run["observed"]["premise"]["problems"] == []
    verdict = json.loads((EVIDENCE / "verdict.json").read_text())
    assert verdict["records"] == verdict["planned"] and verdict["complete"] is True
