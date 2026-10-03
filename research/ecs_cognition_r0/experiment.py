"""Cognitive R0 experiment: specification, pre-registered pass rule, DEV and QUAL procedures. RESEARCH_ONLY.

Everything that decides PASS is fixed here before DEV runs. DEV may choose
only the step budget, by the rule in ``DEV_CHOICE``, on DEV worlds only.
QUAL runs once, on disjoint QUAL worlds, against the frozen specification.

All learning and every response go through the canonical core
(``elpis.ECS_G.cognition.CognitiveCore``): responses are the native forward
map of the current W, learning is K G1 steps committed atomically. No other
model participates.
"""
from __future__ import annotations

import hashlib
import struct
import statistics

import numpy as np

from elpis.ECS_G.cognition import CognitiveCore
from elpis.ECS_G.native import WorldState

from . import task

SPEC = {
    "name": "ecsg-cognition-r0",
    "version": 1,
    "seed": 20261003,
    "dim": 6,
    "width": 36,
    "init_scale": 0.18,
    "learning_rate": 0.002,          # the frozen Branch36 reference rate of the G1 recurrence
    "teacher_width": 4,
    "teacher_scale": 0.5,
    "input_scale": 0.5,
    "region_offset": 0.5,
    "train_rows": 64,
    "test_rows": 256,
    "dev_worlds": 4,
    "qual_worlds": 8,
    "step_grid": [250, 500, 1000, 2000, 4000],
    "pte_base_width": 32,
    "pte_tail_scale": 0.125,
}

DEV_CHOICE = ("steps = the smallest step_grid value at which every DEV world reaches nmse_A <= 0.25 and "
              "nmse_A <= 0.5 * nmse_A_init (stricter than the QUAL per-world rule); otherwise the largest value.")

# Pre-registered before DEV. Gates decide the scientific disposition; measured
# items are reported with a pre-registered classification but do not gate.
PASS_RULE = {
    "A_learning": "every QUAL world: nmse_A_learned <= 0.5 and nmse_A_learned <= 0.5 * nmse_A_init; "
                  "median over QUAL worlds of nmse_A_learned <= 0.25",
    "B_state_causality": "every QUAL world: restoring the initial snapshot reproduces the initial responses "
                         "bitwise; a fresh state created from the learned W values reproduces the learned "
                         "responses bitwise; a state with W = 0 answers exactly 0 for every query",
    "C_persistence": "every QUAL world: a clean process given only the learned snapshot bytes and the frozen "
                     "specification reproduces the learned responses on the evaluation inputs bitwise",
    "D_continual": "every QUAL world, learning experience B from the A-learned state: the state identity changes, "
                   "the epoch advances by steps, the responses on B inputs change, nmse_B_after <= 0.5 and "
                   "nmse_B_after <= 0.5 * nmse_B_before",
    "D_retention_measured": "not a gate: retention = nmse_A_after_B / nmse_A_learned; class RETAINED if "
                            "nmse_A_after_B <= 0.5 and <= 0.5 * nmse_A_init, else INTERFERENCE; flag "
                            "CATASTROPHIC if nmse_A_after_B >= nmse_A_init",
    "E_controls": "every QUAL world: (i) without learning the responses equal the initial responses and the "
                  "epoch is 0; (ii) learning the same budget on permuted A targets fails the A learning "
                  "criterion on the true targets; (iii) an untrained independent fresh state fails the A "
                  "learning criterion; aggregate: median nmse_shuffled >= 4 * median nmse_A_learned; "
                  "(iv) leak guard: C holds and the core's slots are exactly the native state and the rate",
    "F_determinism": "the first QUAL world, learned twice in this process and once in a clean process from the "
                     "specification alone, yields identical learned snapshot digests and transition receipts",
    "G_microstate_authority": "every QUAL world: the PTE pair has S3 relative difference <= 1e-12 and response "
                              "relative difference <= 1e-12 before learning, and after one identical learning "
                              "step S3 relative difference >= 1e-9 and response relative difference >= 1e-9",
}

LEARNING_CRITERION = "nmse <= 0.5 and nmse <= 0.5 * nmse_A_init"


def _rows(a: np.ndarray):
    return a.tolist()


def responses_digest(values) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def snapshot_digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(a)))))


def _meets(nmse_value: float, nmse_init: float) -> bool:
    return nmse_value <= 0.5 and nmse_value <= 0.5 * nmse_init


def learn_a(api, spec: dict, world_id: str, steps: int):
    """Initialization -> experience A. Returns (core, world, initial snapshot, receipt)."""
    w = task.build_world(spec, world_id)
    core = CognitiveCore.create(api, spec["dim"], spec["width"], w.w0.reshape(-1).tolist(),
                                learning_rate=spec["learning_rate"])
    initial = core.snapshot()
    receipt = core.learn(_rows(w.xa_train), w.ya_train.tolist(), steps=steps)
    return core, w, initial, receipt


def dev_curve(api, spec: dict, world_id: str) -> dict:
    """Learning curve on experience A at every step_grid value (cumulative training)."""
    w = task.build_world(spec, world_id)
    with CognitiveCore.create(api, spec["dim"], spec["width"], w.w0.reshape(-1).tolist(),
                              learning_rate=spec["learning_rate"]) as core:
        out = {"0": task.nmse(core.query(_rows(w.xa_test)), w.ya_test)}
        done = 0
        for steps in spec["step_grid"]:
            core.learn(_rows(w.xa_train), w.ya_train.tolist(), steps=steps - done)
            done = steps
            out[str(steps)] = task.nmse(core.query(_rows(w.xa_test)), w.ya_test)
    return out


def choose_steps(spec: dict, curves: dict) -> int:
    for steps in spec["step_grid"]:
        if all(c[str(steps)] <= 0.25 and c[str(steps)] <= 0.5 * c["0"] for c in curves.values()):
            return steps
    return spec["step_grid"][-1]


def evaluate_world(api, spec: dict, world_id: str, steps: int) -> tuple[dict, dict, bytes]:
    """Every per-world measurement. Returns (metrics, checks, learned snapshot bytes)."""
    d, n, lr = spec["dim"], spec["width"], spec["learning_rate"]
    core, w, initial, receipt_a = learn_a(api, spec, world_id, steps)
    xa_t, xb_t = _rows(w.xa_test), _rows(w.xb_test)
    checks = {}
    with core:
        learned = core.snapshot()
        pa1, pb1 = core.query(xa_t), core.query(xb_t)
        with CognitiveCore.restore(api, initial, learning_rate=lr) as untouched:
            pa0, pb0 = untouched.query(xa_t), untouched.query(xb_t)
            checks["no_learning_epoch_zero"] = untouched.epoch == 0
        n_a0, n_b0, n_a1, n_b1 = (task.nmse(pa0, w.ya_test), task.nmse(pb0, w.yb_test),
                                  task.nmse(pa1, w.ya_test), task.nmse(pb1, w.yb_test))
        # B: state causality.
        with CognitiveCore.restore(api, initial, learning_rate=lr) as reset:
            reset_bitwise = reset.query(xa_t) == pa0
        with WorldState.restore(api, learned) as holder:
            learned_w = holder.w()
        with WorldState.create(api, d, n, list(learned_w)) as transplant:
            transplant_bitwise = transplant.forward(xa_t) == pa1
        with CognitiveCore.create(api, d, n, [0.0] * (d * n), learning_rate=lr) as blank:
            pz = blank.query(xa_t)
        zero_answers = pz == (0.0,) * len(xa_t)
        # E(i): the same initialization, never taught, from its W values rather than a snapshot.
        with CognitiveCore.create(api, d, n, w.w0.reshape(-1).tolist(), learning_rate=lr) as unlearned:
            no_learning_bitwise = unlearned.query(xa_t) == pa0 and unlearned.epoch == 0
        oracle_rel = _rel(task.cubic(np.asarray(learned_w).reshape(d, n), w.xa_test), pa1)
        checks["native_forward_matches_oracle"] = oracle_rel <= 1e-10
        checks["epoch_after_A"] = core.epoch == steps == receipt_a.epoch_after
        # D: continual update from the A-learned state.
        identity_before_b = core.identity
        receipt_b = core.learn(_rows(w.xb_train), w.yb_train.tolist(), steps=steps)
        pa2, pb2 = core.query(xa_t), core.query(xb_t)
        n_a2, n_b2 = task.nmse(pa2, w.ya_test), task.nmse(pb2, w.yb_test)
        after_b = core.snapshot()
        checks["epoch_after_B"] = core.epoch == 2 * steps == receipt_b.epoch_after
    # E: shuffled-target and fresh-state controls.
    with CognitiveCore.restore(api, initial, learning_rate=lr) as shuffled:
        shuffled.learn(_rows(w.xa_train), w.ya_train[w.permutation].tolist(), steps=steps)
        n_shuffled = task.nmse(shuffled.query(xa_t), w.ya_test)
    with CognitiveCore.create(api, d, n, w.w_fresh.reshape(-1).tolist(), learning_rate=lr) as fresh:
        n_fresh = task.nmse(fresh.query(xa_t), w.ya_test)
    # G: microstate authority on an exact PTE pair.
    wp, wq = task.pte_pair(spec, w.w0)
    with CognitiveCore.create(api, d, n, wp.reshape(-1).tolist(), learning_rate=lr) as p, \
            CognitiveCore.create(api, d, n, wq.reshape(-1).tolist(), learning_rate=lr) as q:
        s3_before, out_before = _rel(p.s3(), q.s3()), _rel(p.query(xa_t), q.query(xa_t))
        p.learn(_rows(w.xa_train), w.ya_train.tolist(), steps=1)
        q.learn(_rows(w.xa_train), w.ya_train.tolist(), steps=1)
        s3_after, out_after = _rel(p.s3(), q.s3()), _rel(p.query(xa_t), q.query(xa_t))
        checks["pte_microstates_differ"] = bool(np.any(wp != wq))
    metrics = {
        "nmse_A_init": n_a0, "nmse_A_learned": n_a1, "nmse_A_after_B": n_a2,
        "nmse_B_init": n_b0, "nmse_B_before": n_b1, "nmse_B_after": n_b2,
        "nmse_shuffled": n_shuffled, "nmse_fresh": n_fresh, "nmse_zero_state": task.nmse(pz, w.ya_test),
        "reset_bitwise": reset_bitwise, "transplant_bitwise": transplant_bitwise, "zero_state_answers_zero": zero_answers,
        "no_learning_bitwise": no_learning_bitwise,
        "B_identity_changed": identity_before_b != receipt_b.after and receipt_b.before == identity_before_b,
        "B_responses_changed": pb2 != pb1,
        "retention": n_a2 / n_a1,
        "retention_class": "RETAINED" if _meets(n_a2, n_a0) else "INTERFERENCE",
        "catastrophic": n_a2 >= n_a0,
        "pte_s3_rel_before": s3_before, "pte_response_rel_before": out_before,
        "pte_s3_rel_after": s3_after, "pte_response_rel_after": out_after,
        "oracle_rel": oracle_rel,
        "learned_snapshot": snapshot_digest(learned), "after_B_snapshot": snapshot_digest(after_b),
        "learned_responses_A": responses_digest(pa1),
        "receipt_A": receipt_a.digest, "receipt_B": receipt_b.digest,
    }
    return metrics, checks, learned


def gates_for_world(m: dict) -> dict:
    return {
        "A_learning": _meets(m["nmse_A_learned"], m["nmse_A_init"]),
        "B_state_causality": m["reset_bitwise"] and m["transplant_bitwise"] and m["zero_state_answers_zero"],
        "D_continual": (m["B_identity_changed"] and m["B_responses_changed"]
                        and m["nmse_B_after"] <= 0.5 and m["nmse_B_after"] <= 0.5 * m["nmse_B_before"]),
        "E_controls": (m["no_learning_bitwise"] and not _meets(m["nmse_shuffled"], m["nmse_A_init"])
                       and not _meets(m["nmse_fresh"], m["nmse_A_init"])),
        "G_microstate_authority": (m["pte_s3_rel_before"] <= 1e-12 and m["pte_response_rel_before"] <= 1e-12
                                   and m["pte_s3_rel_after"] >= 1e-9 and m["pte_response_rel_after"] >= 1e-9),
    }


def aggregate_gates(per_world: dict) -> dict:
    learned = [m["nmse_A_learned"] for m in per_world.values()]
    shuffled = [m["nmse_shuffled"] for m in per_world.values()]
    return {
        "A_median_nmse_learned_le_0.25": statistics.median(learned) <= 0.25,
        "E_median_shuffled_ge_4x_learned": statistics.median(shuffled) >= 4 * statistics.median(learned),
    }


def slots_are_state_and_rate() -> bool:
    return set(CognitiveCore.__slots__) == {"_state", "learning_rate"}
