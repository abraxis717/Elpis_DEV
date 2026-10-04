"""Retention R1 evidence guards (RET1C onward): write-once bytes, binding, the recorded DEV disposition, and the
early stop it imposes.

* evidence integrity: the DEV record is byte-identical to the record written (RET1C) and its digest verifies;
* binding: it was produced by the clean RET1B laboratory, single-threaded, under the full numerical binding;
* disposition: DEV ended TASK_INVALID_ON_DEV under the registered task rule (controls only); no candidate and no
  reference was run, so there is no frozen record, no QUAL record, and freeze refuses;
* historical replay: re-running the DEV control rows reproduces them bit for bit, demanded only when the bound
  implementation and numerical profile match (skipped with the reason otherwise; never a scientific gate). The
  never-skipped current-runtime regression is test_runtime_regression.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from research.ecs_retention_r1 import protocol as P
from research.ecs_retention_r1 import run as R

from ...ECS_G.test_math_r0 import REPO, _library_path

# SHA-256 of each write-once record as written.
EVIDENCE_SHA256 = {
    "evidence/dev/ecsg-retention-r1.v1.dev.json": "0ee685f370b9d50febe2892f63a1e22a3724377755cc807659103ecd535a0d82",
}
DEV = P.load(R.DEV_PATH, "dev")
RET1B_COMMIT = "dcf73ccae26d9204afd2e1e8c2f1aba50d248560"


@pytest.mark.parametrize("name", sorted(EVIDENCE_SHA256))
def test_write_once_evidence_bytes_are_unchanged(name):
    data = (R.ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == EVIDENCE_SHA256[name], f"{name} changed after it was written"


def test_dev_record_is_bound_to_the_clean_ret1b_laboratory():
    body = DEV["body"]
    impl = body["implementation"]
    assert impl["base_commit"] == RET1B_COMMIT and not impl["dirty"]
    assert impl["lab_source_digest"] == P.source_digest_at(RET1B_COMMIT)
    assert all(body[k] == v for k, v in P.spec_digests(P.load_spec()).items())
    required = {"src/elpis/ECS_G/native.py", "src/elpis/ECS_G/cognition.py", "native/ECS_G/CMakeLists.txt",
                "CMakeLists.txt"} | {f"native/ECS_G/{d}/{n}" for d, n in (
                    ("include/elpis", "ecsg_math.h"), ("include/elpis", "ecsg_state.h"),
                    ("include/elpis", "ecsg_executor.h"), ("src", "ecsg_math.c"), ("src", "ecsg_state.c"),
                    ("src", "ecsg_executor.c"))}
    assert required <= set(impl["files"]), sorted(required - set(impl["files"]))
    prof = impl["numerical_profile"]
    assert prof["blas"]["threads"] == 1 and prof["blas"]["core"] != "unknown"
    for key in ("python", "numpy", "numpy_build", "cpu", "machine", "os", "libc", "thread_environment"):
        assert prof[key], key
    assert impl["library"]["sha256"] and impl["build"]["compiler"] and impl["build"]["cmake"]["CMAKE_BUILD_TYPE"]


def test_dev_ended_task_invalid_and_ran_no_candidate():
    body = DEV["body"]
    assert body["disposition"] == "TASK_INVALID_ON_DEV" and body["choices"] is None
    assert "candidate_table" not in body and "ranking" not in body
    spec = P.load_spec()
    assert [r["input_scale"] for r in body["task_rule"]] == spec["arms"]["S"]["input_scale_grid"]
    assert not any(r["validity"]["valid"] for r in body["task_rule"])
    assert body["world_ids"] == list(P.world_ids(spec, "DEV"))
    for rule in body["task_rule"]:
        assert list(rule["worlds"]) == body["world_ids"]
        for row in rule["worlds"].values():
            assert set(row) == {"M0", "M1", "ceilings", "witness", "W_A_refused"}   # controls only
    # The recorded verdicts follow from the recorded rows by the registered rule.
    from research.ecs_retention_r1 import experiment as X
    for rule in body["task_rule"]:
        assert X.task_validity_dev(rule["worlds"], spec["thresholds"]) == rule["validity"]


def test_no_freeze_and_no_qual_after_task_invalid_on_dev():
    assert not R.FROZEN_PATH.exists() and not R.QUAL_PATH.exists()
    with pytest.raises(P.ProtocolError, match="TASK_INVALID_ON_DEV"):
        R.freeze(RET1B_COMMIT, _library_path())


_REPLAY = """
import ctypes, json, sys
import research.ecs_retention_r1
from elpis.ECS_G.native import ECSGLibrary
from research.ecs_retention_r1 import experiment as X, protocol as P, run as R
library = sys.argv[1]
dev = P.load(R.DEV_PATH, "dev")["body"]
stale = P.binding_mismatch(dev["implementation"], P.implementation(library))
rows = None
if not stale:
    lab = X.Lab(ECSGLibrary(ctypes.CDLL(library)), P.load_spec())
    rows = [{w: P.plain(X.dev_controls_world(lab, w, r["input_scale"])) for w in ("dev-0000", "dev-0005")}
            for r in dev["task_rule"]]
sys.stdout.write(json.dumps({"stale": stale, "rows": rows}))
"""


def test_dev_controls_reproduce_exactly_under_the_recorded_binding():
    """HISTORICAL REPLAY (bitwise), two DEV worlds at every recorded scale; only under the full binding."""
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _REPLAY, str(_library_path())], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    if result["stale"]:
        pytest.skip(f"HISTORICAL_REPLAY_ENVIRONMENT_MISMATCH: bitwise DEV replay is demanded only under the "
                    f"recorded binding: {result['stale']}")
    for rule, rows in zip(DEV["body"]["task_rule"], result["rows"]):
        for wid, row in rows.items():
            assert row == rule["worlds"][wid], (rule["input_scale"], wid)


RESULTS = REPO / "docs" / "research" / "ECS_RETENTION_R1_RESULTS.md"
OVERCLAIMS = ("retention is solved", "retention_supported", "outcome_a", "elpis now remembers",
              "continual learning is achieved", "understands language", "promoted to canonical")


def test_results_report_the_disposition_and_claim_no_more():
    text = RESULTS.read_text(encoding="utf-8")
    for needle in ("TASK_INVALID_ON_DEV", DEV["digest"], DEV["body"]["implementation"]["library"]["sha256"],
                   RET1B_COMMIT[:7], "NO_CANONICAL_PROMOTION", "No candidate is", "SEMANTICS=NONE",
                   "R1 v1 established nothing about any retention mechanism", "new experiment version"):
        assert needle in text, needle
    lowered = text.lower()
    for overclaim in OVERCLAIMS:
        assert overclaim not in lowered.replace("`retention_supported` for a candidate whose state caused nothing",
                                                ""), overclaim


def test_authority_pointers_state_the_recorded_disposition():
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
    component = next(c for c in system["research"]["components"] if c["path"] == "research/ecs_retention_r1")
    ecsg = next(s for s in system["subsystems"] if s["id"] == "ECS_G")
    interface = next(i for i in ecsg["incomplete_interfaces"] if "Retention R1" in i)
    cognition = (REPO / "docs" / "COGNITION_R0.md").read_text(encoding="utf-8")
    for text in (component["classification"], interface, cognition.split("Retention R1", 1)[1].split("\n\n", 1)[0]):
        assert "TASK_INVALID_ON_DEV" in text
        assert "OUTCOME_A" not in text and "RETENTION_SUPPORTED" not in text
    assert "no candidate" in component["classification"].lower()
