"""Cognitive R0 scientific regression on the current runtime. Never skipped for a host or profile difference.

``test_lab.py`` keeps evidence integrity (always) and historical replay (bitwise, only under the recorded
numerical profile) apart; current executor-vs-reference parity lives in tests/ECS_G,
tests/research/ecs_runtime_r1 and the native ctest suite. This module asks the remaining question: does the
current runtime, on whatever host runs it, still reach the recorded Cognitive R0 result?

Every QUAL world is re-run through the canonical core and

* every numeric metric must agree with the QUAL record within the declared tolerance below;
* every boolean, class label and mechanics check must equal the recorded one;
* every gate (A learning, B state causality, C persistence, D continual, E controls, F determinism,
  G microstate authority), the aggregate rules and the disposition are recomputed from the current run by
  the frozen procedure and must equal the recorded ones;
* interference stays measured as found (per-world class, catastrophic flag, counts).

Exact-byte identities (learned snapshot, after-B snapshot, response digest, transition receipts) are not
compared with the record here: they are bitwise identities of one host's arithmetic, not scientific
metrics. The bitwise properties the pass rule itself demands (reset, transplant, zero state, persistence,
determinism) are re-established on this host, current against current.

Declared tolerance (fixed before use, not fitted per run): a current float ``c`` matches a recorded
float ``r`` when ``|c - r| <= ABS_TOL + REL_TOL * |r|``. Calibration on the recording host with each
OpenBLAS kernel family forced (``OPENBLAS_CORETYPE``: Cooperlake, SkylakeX, Haswell, Zen reproduce the
record exactly; Prescott, Nehalem, Sandybridge, Core2 reproduce the 61f81e6 CI values) gave at most
5e-15 relative on nmse/retention metrics, 4e-13 relative on the PTE post-step differences and 4e-16
absolute on ULP-scale quantities, with no class, flag or gate change. The tolerance leaves at least
2,600x headroom over that, and stays seven orders of magnitude below the smallest relative distance of
any recorded gated metric from its threshold (3.4%, the catastrophic flag of one world).
"""
from __future__ import annotations

import ctypes
import json
import os
import statistics
import subprocess
import sys

import pytest

from elpis.ECS_G.native import ECSGLibrary
from research.ecs_cognition_r0 import experiment as E
from research.ecs_cognition_r0 import run as R
from research.ecs_cognition_r0.protocol import load

from ...ECS_G.test_math_r0 import _library_path

REL_TOL = 1e-9
ABS_TOL = 1e-12
IDENTITY_FIELDS = frozenset({"learned_snapshot", "after_B_snapshot", "learned_responses_A", "receipt_A",
                             "receipt_B"})

QUAL = load(R.QUAL_PATH, "qual")["body"]
WORLDS = tuple(QUAL["worlds"])


@pytest.fixture(scope="module")
def current():
    """Every QUAL world through the current canonical core: {world: (metrics, checks, learned snapshot)}."""
    api = ECSGLibrary(ctypes.CDLL(str(_library_path())))
    return {w: E.evaluate_world(api, E.SPEC, w, QUAL["steps"]) for w in WORLDS}


@pytest.fixture(scope="module")
def persistence(current, tmp_path_factory):
    """C: a clean process given only the current learned snapshot bytes and the frozen spec (current vs current)."""
    tmp = tmp_path_factory.mktemp("cognition-r0-persistence")
    out = {}
    for w, (metrics, _, learned) in current.items():
        snap = tmp / f"{w}.snapshot"
        snap.write_bytes(learned)
        replay = _probe(["replay", "--world", w, "--snapshot", str(snap), "--library", str(_library_path()),
                           "--spec", str(R.FROZEN_PATH)])
        out[w] = replay["responses"] == metrics["learned_responses_A"]
    return out


@pytest.fixture(scope="module")
def determinism(current):
    """F: the first world learned again in this process and once in a clean process, against this run."""
    api = ECSGLibrary(ctypes.CDLL(str(_library_path())))
    first = WORLDS[0]
    ref = current[first][0]
    again, _, _ = E.evaluate_world(api, E.SPEC, first, QUAL["steps"])
    clean = _probe(["retrain", "--world", first, "--steps", str(QUAL["steps"]), "--library",
                      str(_library_path()), "--spec", str(R.FROZEN_PATH)])
    return {
        "in_process_repeat_equal": (again["learned_snapshot"], again["receipt_A"], again["receipt_B"])
        == (ref["learned_snapshot"], ref["receipt_A"], ref["receipt_B"]),
        "clean_process_equal": (clean["snapshot"], clean["receipt"]) == (ref["learned_snapshot"], ref["receipt_A"]),
    }


# The clean-process probes are the laboratory's own ``run replay`` / ``run retrain`` commands with the same
# minimal environment ``run._probe`` gives them, plus any BLAS selection variables of this process, so the
# child computes with the same OpenBLAS kernel and thread setting as the parent on the same host.
_BLAS_ENV = ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _probe(args: list[str]) -> dict:
    env = {"PYTHONPATH": str(R.REPO / "src") + os.pathsep + str(R.REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    env.update({k: os.environ[k] for k in _BLAS_ENV if k in os.environ})
    result = subprocess.run([sys.executable, "-m", "research.ecs_cognition_r0.run", *args], cwd=R.REPO, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, "clean-process probe failed: " + result.stderr[-2000:]
    return json.loads(result.stdout)


def _within(c: float, r: float) -> bool:
    return abs(c - r) <= ABS_TOL + REL_TOL * abs(r)


def test_the_tolerance_is_the_declared_one():
    """Widening the tolerance is a change of this contract and must be visible here and in the results doc."""
    assert (REL_TOL, ABS_TOL) == (1e-9, 1e-12)


@pytest.mark.parametrize("world", WORLDS)
def test_metrics_match_the_record_within_the_declared_tolerance(current, world):
    metrics, checks, _ = current[world]
    recorded = QUAL["per_world"][world]
    assert set(metrics) == set(recorded), "metric schema changed"
    assert IDENTITY_FIELDS <= set(recorded)
    off = []
    for key, r in sorted(recorded.items()):
        if key in IDENTITY_FIELDS:
            continue
        c = metrics[key]
        if isinstance(r, bool) or isinstance(r, str):
            if c != r:
                off.append(f"{key}: {c!r} != recorded {r!r}")
        elif isinstance(r, float):
            if not (isinstance(c, float) and _within(c, r)):
                off.append(f"{key}: {c!r} vs recorded {r!r} (|d| {abs(c - r):.3g})")
        else:
            off.append(f"{key}: unexpected recorded type {type(r).__name__}")
    assert not off, off
    assert checks == QUAL["mechanics_checks"][world]
    assert all(checks.values())


@pytest.mark.parametrize("world", WORLDS)
def test_per_world_gates_and_interference_hold(current, persistence, world):
    metrics = current[world][0]
    gates = E.gates_for_world(metrics)
    gates["C_persistence"] = persistence[world]
    assert gates == QUAL["per_world_gates"][world]
    assert all(gates.values()), gates
    measured = QUAL["retention_measured"][world]
    assert (metrics["retention_class"], metrics["catastrophic"]) == (measured["class"], measured["catastrophic"])
    assert _within(metrics["retention"], measured["retention"])


def test_the_recorded_disposition_is_reached_by_the_frozen_procedure(current, persistence, determinism):
    """The gate table, mechanics and disposition, assembled exactly as ``run.qual`` assembled them."""
    per_world = {w: v[0] for w, v in current.items()}
    checks = {w: v[1] for w, v in current.items()}
    gates = {}
    for w, m in per_world.items():
        gates[w] = E.gates_for_world(m)
        gates[w]["C_persistence"] = persistence[w]
    aggregate = E.aggregate_gates(per_world)
    leak_guard = {"core_slots_are_state_and_rate": E.slots_are_state_and_rate(),
                  "clean_process_replay_bitwise_all_worlds": all(persistence.values())}
    gate_table = {
        "A_learning": all(g["A_learning"] for g in gates.values()) and aggregate["A_median_nmse_learned_le_0.25"],
        "B_state_causality": all(g["B_state_causality"] for g in gates.values()),
        "C_persistence": all(g["C_persistence"] for g in gates.values()),
        "D_continual": all(g["D_continual"] for g in gates.values()),
        "E_controls": (all(g["E_controls"] for g in gates.values()) and aggregate["E_median_shuffled_ge_4x_learned"]
                       and all(leak_guard.values())),
        "F_determinism": all(determinism.values()),
        "G_microstate_authority": all(g["G_microstate_authority"] for g in gates.values()),
    }
    mechanics = "MECHANICS_PASS" if all(all(c.values()) for c in checks.values()) else "MECHANICS_FAIL"
    scientific = ("NOT_TESTED" if mechanics != "MECHANICS_PASS" else
                  "SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME" if all(gate_table.values()) else "DID_NOT_SUPPORT")
    assert aggregate == QUAL["aggregate"]
    assert leak_guard == QUAL["leak_guard"]
    assert determinism == QUAL["determinism"]
    assert gate_table == QUAL["gates"]
    assert (mechanics, scientific) == (QUAL["mechanics"], QUAL["scientific"]) == (
        "MECHANICS_PASS", "SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME")
    learned = [m["nmse_A_learned"] for m in per_world.values()]
    recorded_learned = [m["nmse_A_learned"] for m in QUAL["per_world"].values()]
    assert _within(statistics.median(learned), statistics.median(recorded_learned))


def test_interference_counts_are_unchanged(current):
    classes = [current[w][0]["retention_class"] for w in WORLDS]
    catastrophic = sum(current[w][0]["catastrophic"] for w in WORLDS)
    recorded = QUAL["retention_measured"].values()
    assert classes.count("INTERFERENCE") == [r["class"] for r in recorded].count("INTERFERENCE") == len(WORLDS)
    assert catastrophic == sum(r["catastrophic"] for r in recorded)
