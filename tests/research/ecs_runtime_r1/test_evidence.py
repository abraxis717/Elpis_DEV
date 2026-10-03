"""ECS Runtime R1 evidence guards (PERFORMANCE_ONLY): write-once, bound to what it measured, honestly reported.

Every evidence file names the commit, tree state and source digests it was
measured on. The code it measured must still be the code in this tree: a
change to the executor, the reference kernel or the Python control plane
invalidates the evidence until it is measured again. The admitted report
must state the gate verdicts the evidence actually yields.
"""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from research.ecs_runtime_r1 import gates as G
from research.ecs_runtime_r1 import workloads as WL

from ...ECS_G.test_math_r0 import REPO

EVIDENCE = REPO / "research" / "ecs_runtime_r1" / "evidence"
REPORT = REPO / "docs" / "performance" / "ECS_RUNTIME_R1.md"
# The measured code: the native executor and reference kernel, and the Python control plane.
MEASURED = ("native/ECS_G/src/ecsg_executor.c", "native/ECS_G/src/ecsg_math.c", "native/ECS_G/src/ecsg_state.c",
            "native/ECS_G/include/elpis/ecsg_executor.h", "native/ECS_G/include/elpis/ecsg_math.h",
            "native/ECS_G/include/elpis/ecsg_state.h", "src/elpis/ECS_G/native.py", "src/elpis/ECS_G/cognition.py",
            "src/elpis/runtime/cognition.py")
REFERENCE = ("native/ECS_G/src/ecsg_math.c", "native/ECS_G/src/ecsg_state.c", "native/ECS_G/include/elpis/ecsg_math.h",
             "native/ECS_G/include/elpis/ecsg_state.h")


def _sha(rel):
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def _load(name):
    return json.loads((EVIDENCE / name).read_text(encoding="ascii"))


def test_evidence_is_labelled_and_bound_to_a_clean_tree():
    for name in ("baseline.json", "final.json", "jitter.json", "overhead.json"):
        report = _load(name)
        assert report["label"].startswith("PERFORMANCE_ONLY NO_SCIENTIFIC_CLAIM"), name
        assert report["binding"]["dirty"] is False, name
        assert re.fullmatch(r"[0-9a-f]{40}", report["binding"]["head"]), name
        assert report["binding"]["library"]["sha256"] and report["binding"]["compiler"], name
        assert report["binding"]["cmake"]["CMAKE_BUILD_TYPE"] == "Release", name
    sanitizers = _load("sanitizers.json")
    assert sanitizers["dirty"] is False and sanitizers["label"].endswith("NO_SCIENTIFIC_CLAIM")


def test_final_evidence_measured_the_code_in_this_tree():
    for name in ("final.json", "jitter.json", "overhead.json"):
        sources = _load(name)["binding"]["sources"]
        stale = [rel for rel in MEASURED if sources.get(rel) != _sha(rel)]
        assert not stale, f"{name} measured different code; re-measure: {stale}"
    sanitizer_sources = _load("sanitizers.json")["sources"]
    assert all(sanitizer_sources[rel] == _sha(rel) for rel in sanitizer_sources)
    profile = _load("profile.json")
    assert profile["after"]["executor_source_sha256"] == _sha("native/ECS_G/src/ecsg_executor.c")


def test_baseline_measured_the_same_scalar_reference_before_nativeization():
    baseline = _load("baseline.json")
    assert baseline["phase"] == "baseline" and len(baseline["results"]) == 62
    assert baseline["binding"]["head"].startswith("3568a77")
    assert all(baseline["binding"]["sources"][rel] == _sha(rel) for rel in REFERENCE)
    assert "native/ECS_G/src/ecsg_executor.c" not in baseline["binding"]["sources"]


def test_every_registered_workload_and_mode_was_measured():
    final = _load("final.json")
    assert final["phase"] == "final" and final["workloads"] == WL.workloads()
    assert final["gates"] == WL.GATES and final["thresholds"] == WL.THRESHOLDS
    measured = {(r["workload"], r["mode"]) for r in final["results"]}
    for w in WL.workloads():
        if w["kind"] == "learn":
            assert {(w["id"], m) for m in ("ref_learn", "exec_learn", "exec_txn", *G.PYTHON_LEARN)} <= measured
        else:
            assert {(w["id"], m) for m in ("ref_forward", "exec_forward", "py_exec_forward_into",
                                            "py_core_query")} <= measured
    for r in final["results"]:
        assert r["samples"] >= WL.MIN_SAMPLES and r["warmup"] >= WL.MIN_WARMUP and r["first_call_ns"] > 0


def test_report_states_the_verdicts_the_evidence_yields():
    verdicts = G.evaluate()
    text = REPORT.read_text(encoding="utf-8")
    for gate in ("material_speedup", "python_overhead_k_independent", "python_overhead_split_timing", "sanitizers"):
        word = "PASS" if verdicts[gate]["pass"] else "FAIL"
        assert f"`{gate}`: {word}" in text, (gate, word)
    assert "PERFORMANCE_ONLY" in text and "NO_SCIENTIFIC_CLAIM" in text


@pytest.mark.parametrize("name", ["baseline.json", "final.json", "jitter.json", "overhead.json", "sanitizers.json",
                                  "profile.json"])
def test_evidence_files_end_cleanly(name):
    text = (EVIDENCE / name).read_text(encoding="ascii")
    assert text.endswith("}\n") and json.loads(text)
