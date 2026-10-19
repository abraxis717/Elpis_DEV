"""ECS_C Recovery Q0: the committed QUAL evidence validates, and the instrument in the tree is the frozen one.

The evidence (research/qualification/ecs_c_recovery_q0/evidence/) was written once by the frozen QUAL run on the
accepted source. The repository keeps no file over 2 MiB, so the sealed ``records.jsonl`` is committed gzip-compressed
(``records.jsonl.gz``); it is restored here into the sealed directory as written, and SHA256SUMS (checked by the
validator) proves it byte-identical. The independent validator (check.py evidence) then verifies that directory:
exactly its files, every SHA-256, a complete QUAL run, and the verdict recomputed from the records. Beyond it, this
test binds the evidence to its freeze: the freeze record committed beside the instrument (published before the run)
is byte-identical to the evidence's, it is the freeze of a clean checkout whose premise held, the run's header names
the whole QUAL plan, and the instrument files in the tree are still the ones the freeze bound (a changed instrument is
a new instrument and needs new evidence). Tamper resistance of the checker itself is test_instrument.py's (DEV data).
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]
INSTRUMENT = REPO / "research" / "qualification" / "ecs_c_recovery_q0"
EVIDENCE = INSTRUMENT / "evidence"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"q0_{name}", INSTRUMENT / f"{name}.py")
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
        ["FREEZE.json", "records.jsonl.gz", "verdict.json", "SHA256SUMS"])


def test_the_sealed_evidence_validates_and_its_verdict_is_recomputed(tmp_path):
    assert _load("check").evidence(_sealed(tmp_path / "sealed")) == []


def test_the_committed_freeze_record_is_the_evidences_and_the_instrument_is_still_the_frozen_one():
    assert (INSTRUMENT / "FREEZE.json").read_bytes() == (EVIDENCE / "FREEZE.json").read_bytes()
    freeze = json.loads((EVIDENCE / "FREEZE.json").read_text())
    for name, digest in freeze["instrument"]["files"].items():
        assert _sha(INSTRUMENT / name) == digest, name
    assert sorted(freeze["instrument"]["files"]) == sorted(
        p.relative_to(INSTRUMENT).as_posix() for p in INSTRUMENT.rglob("*")
        if p.is_file() and not ({"evidence", "FREEZE.json", "__pycache__"} & set(p.relative_to(INSTRUMENT).parts)))


def test_the_run_was_the_complete_qual_plan_on_a_clean_frozen_commit(tmp_path):
    freeze = json.loads((EVIDENCE / "FREEZE.json").read_text())
    assert freeze["inputs"] == "qual" and freeze["source"]["clean"] is True and freeze["premise"]["problems"] == []
    assert freeze["python"]["isolated"] == 1 and freeze["python"]["elpis_imported"] == []
    with open(_sealed(tmp_path / "sealed") / "records.jsonl") as f:
        header = json.loads(f.readline())
    assert (header["inputs"], header["families"], header["runs"]) == ("qual", None, None)
    verdict = json.loads((EVIDENCE / "verdict.json").read_text())
    assert (verdict["inputs"], verdict["subset"], verdict["mechanical"]) == ("qual", False, [])
    assert verdict["judged"] == verdict["trials"]
