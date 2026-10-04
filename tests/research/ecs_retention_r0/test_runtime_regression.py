"""Retention R0 recorded disposition on the current runtime. Never skipped for a host or profile difference.

``test_evidence.py`` keeps evidence integrity, chronology and historical replay (bitwise, only under the
full v1 binding and a reproducing OpenBLAS kernel) apart. This module asks whether the current runtime, on
whatever host runs it, still reaches what the Retention R0 report concludes.

Scope, fixed from a post-QUAL portability audit (docs/research/ECS_RETENTION_R0_RESULTS.md, "Reproduction
contract"): the v1 per-world numbers are not portable at a tight tolerance. Re-running all 24 QUAL worlds
with each OpenBLAS kernel class forced (AVX-512 Cooperlake/SkylakeX, which reproduce the record bitwise;
AVX2+FMA Haswell/Zen; non-FMA Prescott, the fallback on CPUs that OpenBLAS 0.3.23 does not recognize)
changes W digests and moves cumulative quantities by up to 23% (the selected candidate's late
A->B->C->D stages) and 15% (the mismatched-statistics ablation). Under all three classes the gate table,
mechanics, disposition, outcome and every per-mechanism count below were identical. Those are what this
module compares, exactly; no float and no W digest is compared with the record.

The run happens in a fresh interpreter whose BLAS/OpenMP thread environment is fixed before NumPy loads
(RET0E.3), and determinism is re-established there exactly as QUAL did (twice in process and once in a
clean process, current against current).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from research.ecs_retention_r0 import protocol as P
from research.ecs_retention_r0 import run as R

from ...ECS_G.test_math_r0 import REPO, _library_path

QUAL = P.load(R.QUAL_PATH, "qual")["body"]

_RUN = """
import ctypes, json, statistics, sys
from elpis.ECS_G.native import ECSGLibrary
from research.ecs_retention_r0 import experiment as X, protocol as P, run as R
from tests.research._blas import openblas_core, openblas_threads
library = sys.argv[1]
qual = P.load(R.QUAL_PATH, "qual")["body"]
spec, choices = P.load_spec(), qual["choices"]
lab = X.Lab(ECSGLibrary(ctypes.CDLL(library)), spec)
worlds = list(P.world_ids(spec, "QUAL"))
per_world = {w: json.loads(P.canonical_json(X.qual_world(lab, w, choices))) for w in worlds}
first = worlds[0]
runs = [X.determinism_digest(lab, first, choices), X.determinism_digest(lab, first, choices),
        R._probe(R.Path(library), first, choices)]
recorded = {"W_A": per_world[first]["mechanics"]["shared_W_A_digest"],
            "W_AB": per_world[first]["mechanisms"]["selected"]["W_digest"]}
identical = (all(r == runs[0] for r in runs) and runs[0]["W_A"] == recorded["W_A"]
             and runs[0]["W_AB"] == recorded["W_AB"])
verdict = X.gates(per_world, spec, {"world": first, "runs": runs, "recorded": recorded, "identical": identical},
                  X.consolidation_inputs(), choices)
counts = {}
for m in per_world[first]["mechanisms"]:
    rows = [per_world[w]["mechanisms"][m] for w in worlds]
    counts[m] = {"RETAINED_A": sum(r["RETAINED_A"] for r in rows), "LEARNED_A": sum(r["LEARNED_A"] for r in rows),
                 "LEARNED_B": sum(r["LEARNED_B"] for r in rows),
                 "both": sum(r["RETAINED_A"] and r["LEARNED_B"] for r in rows),
                 "CATASTROPHIC_A": sum(r["CATASTROPHIC_A"] for r in rows),
                 "refused_worlds": sum(bool(r["refused"]) for r in rows)}
sequence = {m: statistics.median(per_world[w]["sequence"][m]["held_count"] for w in worlds)
            for m in per_world[first]["sequence"]}
sys.stdout.write(json.dumps({"verdict": {k: verdict[k] for k in ("gates", "mechanics", "disposition", "outcome")},
                             "counts": counts, "sequence_median_held": sequence, "determinism_identical": identical,
                             "worlds": worlds, "core": openblas_core(), "threads": openblas_threads()}))
"""


def _recorded_counts() -> dict:
    worlds = list(QUAL["per_world"])
    counts = {}
    for m in QUAL["per_world"][worlds[0]]["mechanisms"]:
        rows = [QUAL["per_world"][w]["mechanisms"][m] for w in worlds]
        counts[m] = {"RETAINED_A": sum(r["RETAINED_A"] for r in rows), "LEARNED_A": sum(r["LEARNED_A"] for r in rows),
                     "LEARNED_B": sum(r["LEARNED_B"] for r in rows),
                     "both": sum(r["RETAINED_A"] and r["LEARNED_B"] for r in rows),
                     "CATASTROPHIC_A": sum(r["CATASTROPHIC_A"] for r in rows),
                     "refused_worlds": sum(bool(r["refused"]) for r in rows)}
    return counts


@pytest.fixture(scope="module")
def current():
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _RUN, str(_library_path())], cwd=REPO, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout)


def test_all_qual_worlds_ran_single_threaded(current):
    assert current["worlds"] == list(QUAL["per_world"])
    assert current["threads"] in (1, -1), f"effective OpenBLAS threads {current['threads']}"


def test_the_recorded_gates_mechanics_and_disposition_are_reached(current):
    v = current["verdict"]
    assert current["determinism_identical"], "G determinism failed on this host (current against current)"
    assert v["mechanics"] == QUAL["mechanics"] and all(v["mechanics"].values())
    assert v["gates"] == QUAL["gates"], (current["core"], v["gates"])
    assert (v["disposition"], v["outcome"]) == (QUAL["disposition"], QUAL["outcome"]) == (
        "PARTIAL_REDUCTION", "OUTCOME_C")
    assert v["gates"]["B_acquisition"] is False and v["gates"]["C_retention"] is False


@pytest.mark.parametrize("mechanism", sorted(_recorded_counts()))
def test_per_mechanism_counts_are_unchanged(current, mechanism):
    assert current["counts"][mechanism] == _recorded_counts()[mechanism], current["core"]


def test_longer_sequence_median_is_unchanged(current):
    worlds = list(QUAL["per_world"])
    recorded = {}
    for m in QUAL["per_world"][worlds[0]]["sequence"]:
        held = sorted(QUAL["per_world"][w]["sequence"][m]["held_count"] for w in worlds)
        recorded[m] = (held[len(held) // 2 - 1] + held[len(held) // 2]) / 2
    assert current["sequence_median_held"] == recorded
