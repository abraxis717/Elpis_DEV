"""Native K1 differential against the frozen Retention R3 record (PLAN.md). RESEARCH_ONLY.

Per R3 QUAL world: the native K1-disabled run (parity with the recorded M0 digests), the native K1 run on the
standalone and the FMS-resident paths (one transaction per experience), the boundaries against the recomputed R3
laboratory and the recorded nmse, the reset challenge, the W-only negative control, the transplant continuation and
the state semantics. ``substituted`` rebuilds the recorded per-world rows with the native ones in place, for the
unchanged R3 gates. The R3 laboratory and records are read, never written.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
from pathlib import Path
import tempfile
import time

import numpy as np

from elpis.ECS_G.k1 import K1FMSRuntime, K1Library, K1State
from elpis.ECS_G.native import ECSGLibrary, Executor
from elpis.substrate.residency import Context
from research.ecs_retention_r3 import engine as E
from research.ecs_retention_r3 import experiment as X
from research.ecs_retention_r3 import task as T

STAGES = T.TASKS
PROTECTED = X.PROTECTED


def _rel(native, reference) -> float:
    native, reference = np.asarray(native, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return float(np.max(np.abs(native - reference)) / max(1.0, float(np.max(np.abs(reference)))))


def _nmse_rel(native: float, recorded: float) -> float:
    return abs(native - recorded) / max(recorded, 1e-12)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mv(a: np.ndarray) -> memoryview:
    return memoryview(np.ascontiguousarray(a, dtype=np.float64).reshape(-1))


class Native:
    """The three loaded libraries (Runtime R1 math, K1, K1 residency adapter) beside one build."""

    def __init__(self, math_library: str):
        path = Path(math_library).resolve()
        self.api = ECSGLibrary(ctypes.CDLL(str(path)))
        self.k1 = K1Library(ctypes.CDLL(str(path.with_name("libelpis_ecsg_k1.so"))))
        self.adapter = ctypes.CDLL(str(path.with_name("libelpis_ecsg_k1_fms.so")))


def evaluate(state, world: T.World) -> dict:
    """Native nmse of every experience: the native query (W only) on its held-out inputs."""
    return {t: T.nmse(np.asarray(state.query(_mv(world.tasks[t].x_test)), dtype=np.float64), world.tasks[t].y_test)
            for t in STAGES}


def _experience(target, world, t, rate, steps, consolidate=True):
    """One experience as one transaction: K steps in one native call, then (optionally) consolidation, commit."""
    exp = world.tasks[t]
    t0 = time.perf_counter()
    with target.transaction() as txn:
        txn.learn(_mv(exp.x_train), _mv(exp.y_train), rate, steps)
        if consolidate:
            txn.consolidate(_mv(exp.x_train))
        txn.commit()
    return time.perf_counter() - t0


def standalone_run(nat: Native, world: T.World, rate: float, steps: int, *, consolidate=True):
    """A native K1 state from w0 through A..D; per boundary the envelope (and the live state for evaluation)."""
    state = K1State.create(nat.k1, *world.w0.shape, _mv(world.w0), max_rows=len(world.tasks["A"].y_train))
    heldout = _mv(np.vstack([world.tasks[t].x_test for t in STAGES]))
    out = {"init": evaluate(state, world), "stages": {}}
    try:
        for t in STAGES:
            seconds = _experience(state, world, t, rate, steps, consolidate)
            env = state.snapshot()
            out["stages"][t] = {"envelope": env, "seconds": seconds, "nmse": evaluate(state, world),
                                "W": np.asarray(state.w()).reshape(world.w0.shape), "epoch": state.epoch,
                                "provenance": state.provenance, "H": np.asarray(state.h_packed()),
                                "a": np.asarray(state.a()), "stats": state.stats(),
                                "responses": np.asarray(state.query(heldout), dtype=np.float64)}
    finally:
        state.close()
    return out


def resident_envelopes(nat: Native, world: T.World, rate: float, steps: int) -> dict:
    """The same K1 sequence on one FMS-resident state; its envelope after every boundary."""
    image = nat.k1.envelope_bytes(*world.w0.shape) - 32
    rows = len(world.tasks["A"].y_train)
    with tempfile.TemporaryDirectory() as cold, \
            Context(nat.adapter, warm_bytes=8 * image, cold_bytes=10 ** 6, max_objects=4,
                    cold_root=Path(cold) / "cold") as ctx, \
            K1FMSRuntime(nat.k1, ctx, nat.adapter, max_states=2) as runtime:
        sid = runtime.register(hashlib.sha256(world.world.encode()).digest(), *world.w0.shape, _mv(world.w0),
                               max_rows=rows)
        envelopes = {}
        for t in STAGES:
            exp = world.tasks[t]
            with runtime.transaction(sid) as txn:
                txn.learn(_mv(exp.x_train), _mv(exp.y_train), rate, steps)
                txn.consolidate(_mv(exp.x_train))
                txn.commit()
            envelopes[t] = runtime.snapshot(sid)
        runtime.close_state(sid)
    return envelopes


def continue_from(nat: Native, envelope: bytes, world: T.World, rate: float, steps: int, stages=("C", "D")) -> dict:
    """Restore an envelope in a fresh state and continue the K1 sequence; envelope per continued boundary."""
    out = {}
    with K1State.restore(nat.k1, envelope, max_rows=len(world.tasks["A"].y_train)) as state:
        out["restored"] = state.snapshot()
        for t in stages:
            _experience(state, world, t, rate, steps)
            out[t] = state.snapshot()
    return out


def _learn_c(nat, state, world, rate, steps):
    c = world.tasks["C"]
    state.learn(_mv(c.x_train), _mv(c.y_train), rate, steps)
    return np.asarray(state.w()).reshape(world.w0.shape)


def reset_branch(nat, envelope_b, world, rate, steps, th, main_c_nmse, init) -> dict:
    """R3 reset challenge, native: reset after B (W, epoch kept), learn C, read at C, stop."""
    with K1State.restore(nat.k1, envelope_b) as full, K1State.restore(nat.k1, envelope_b) as state:
        state.reset()
        mechanics = (state.w() == full.w() and state.epoch == full.epoch and not any(state.h_packed())
                     and not any(state.a()) and state.provenance == "RESET")
        x = np.vstack([world.tasks[t].x_test for t in STAGES])
        query_identical = state.query(_mv(x)) == full.query(_mv(x))
        W_c = _learn_c(nat, state, world, rate, steps)
        nmse = evaluate(state, world)
        corrected = state.stats()["corrected_steps"]
    e_reset = float(np.mean([nmse[t] for t in PROTECTED]))
    e_main = float(np.mean([main_c_nmse[t] for t in PROTECTED]))
    retained = {t: nmse[t] <= th["class_nmse_absolute"] and nmse[t] <= th["class_nmse_relative"] * init[t]
                for t in PROTECTED}
    return {"stages": {"C": {"nmse": nmse, "W_digest": E.w_digest(W_c), "refused": 0}},
            "reset_mechanics": bool(mechanics), "e_reset": e_reset, "e_uninterrupted": e_main,
            "reset_ratio": e_reset / max(e_main, 1e-300),
            "RESET_DEGRADED": bool(e_reset >= th["reset_min_factor"] * e_main
                                   and e_reset >= th["reset_min_absolute_nmse"]),
            "both_retained_at_C": bool(retained["A"] and retained["B"]),
            "_query_identical": bool(query_identical), "_corrected_steps": int(corrected)}


def w_only_branch(nat, envelope_b, W_b, world, rate, steps, reset_c_digest, main_c_digest) -> dict:
    """The Runtime R1 ELPISG01 snapshot of native W_B imported: UNCONSOLIDATED, a different state, the reset branch."""
    dim, width = W_b.shape
    with Executor.create(nat.api, dim, width, _mv(W_b)) as e:      # as R3: a Runtime R1 executor holding W_B
        snapshot = e.snapshot()
    with K1State.import_w_only(nat.k1, snapshot) as state:
        flagged = (state.provenance == "UNCONSOLIDATED_IMPORT" and not any(state.h_packed()) and not any(state.a())
                   and np.array_equal(np.asarray(state.w()).reshape(W_b.shape), W_b))
        digest_differs = state.snapshot() != envelope_b
        W_c = _learn_c(nat, state, world, rate, steps)
    reproduces = E.w_digest(W_c) == reset_c_digest
    differs = E.w_digest(W_c) != main_c_digest
    return {"snapshot_bytes": len(snapshot), "flagged_unconsolidated": bool(flagged),
            "state_digest_differs": bool(digest_differs), "reproduces_reset_at_C": bool(reproduces),
            "differs_from_uninterrupted_at_C": bool(differs),
            "pass": bool(flagged and digest_differs and reproduces and differs)}


CLASS_KEYS = ("LEARNED", "RETAINED", "CATASTROPHIC", "HELD", "SEQUENCE_HELD", "PAIR_HELD")


def _row(recorded: dict, run: dict, th: dict, *, k1: bool) -> dict:
    """The recorded mechanism row with every native quantity in place and the classes recomputed from native nmse."""
    row = copy.deepcopy(recorded)
    row["init"] = run["init"]
    for t in STAGES:
        st, n = row["stages"][t], run["stages"][t]
        st.update({"nmse": n["nmse"], "W_digest": E.w_digest(n["W"]), "state_digest": _sha(n["envelope"]),
                   "refused": 0, "learn_seconds": n["seconds"]})
        if k1:
            st["persistent_bytes"] = 8 * (n["H"].size + n["a"].size)
    row.update(X.classify({t: run["stages"][t]["nmse"] for t in STAGES}, run["init"], th))
    if k1:
        row["W_A_digest"] = E.w_digest(run["stages"]["A"]["W"])
    return row


def world_differential(nat: Native, lab: X.Lab, recorded: dict, tol: dict) -> dict:
    """Every PLAN.md comparison on one R3 QUAL world; ``rows`` are the native rows for the substituted gates."""
    world = lab.world(recorded["world"])
    P = recorded["P"]
    rec_k1, rec_m0 = P["mechanisms"]["K1"], P["mechanisms"]["M0"]
    rate, steps, th = lab.rate, lab.steps, lab.th

    zero = standalone_run(nat, world, rate, steps, consolidate=False)
    zero_digests = {t: E.w_digest(zero["stages"][t]["W"]) for t in STAGES}
    zero_parity = {"digests_equal_M0": all(zero_digests[t] == rec_m0["stages"][t]["W_digest"] for t in STAGES),
                   "corrected_steps": max(zero["stages"][t]["stats"]["corrected_steps"] for t in STAGES)}

    main = standalone_run(nat, world, rate, steps)
    ref = lab.mechanism_run(world, "K1", keep=True)["_states"]
    heldout = lab.heldout_inputs(world)
    boundaries = {}
    for i, t in enumerate(STAGES):
        n, (m, W, epoch) = main["stages"][t], ref[t]
        boundaries[t] = {
            "epoch": n["epoch"], "epoch_expected": steps * (i + 1), "provenance": n["provenance"],
            "w_relative": _rel(n["W"], W), "h_relative": _rel(n["H"], E.pack_upper(m.H)),
            "a_relative": _rel(n["a"], m.a), "query_relative": _rel(n["responses"], E.responses(lab.api, W, heldout)),
            "nmse_relative": max(_nmse_rel(n["nmse"][s], rec_k1["stages"][t]["nmse"][s]) for s in STAGES),
            "envelope_sha256": _sha(n["envelope"]),
            "reference_reproduces_recorded_state_digest": E.state_digest(m, W, epoch)
            == rec_k1["stages"][t]["state_digest"]}
    resident = resident_envelopes(nat, world, rate, steps)

    k1_row = _row(rec_k1, main, th, k1=True)
    removed_row = _row(P["removed"]["K1"], zero, th, k1=False)
    env_b = main["stages"]["B"]["envelope"]
    main_c = E.w_digest(main["stages"]["C"]["W"])
    reset = reset_branch(nat, env_b, world, rate, steps, th, main["stages"]["C"]["nmse"], main["init"])
    w_only = w_only_branch(nat, env_b, main["stages"]["B"]["W"], world, rate, steps,
                           reset["stages"]["C"]["W_digest"], main_c)
    cont = continue_from(nat, env_b, world, rate, steps)
    transplant = {"state_bytes": len(env_b), "restored_digest_equal": cont["restored"] == env_b,
                  "continuation_bitwise": cont["C"] == main["stages"]["C"]["envelope"]
                  and cont["D"] == main["stages"]["D"]["envelope"],
                  "digests": {t: _sha(cont[t]) for t in ("C", "D")}}
    semantics = {"query_identical": reset.pop("_query_identical"),
                 "learning_differs": main_c != reset["stages"]["C"]["W_digest"]}
    reset_corrected = reset.pop("_corrected_steps")
    rec_reset = P["causality"]["reset_challenge"]
    rec_w_only = P["causality"]["w_only_negative_control_at_B"]

    checks = {
        "zero_k1_parity": zero_parity["digests_equal_M0"] and zero_parity["corrected_steps"] == 0,
        "w_a": k1_row["W_A_digest"] == rec_k1["W_A_digest"],
        "boundaries": all(b["epoch"] == b["epoch_expected"] and b["provenance"] == "COMPLETE"
                          and b["w_relative"] <= tol["w_relative"] and b["h_relative"] <= tol["h_relative"]
                          and b["a_relative"] <= tol["a_relative"] and b["query_relative"] <= tol["query_relative"]
                          and b["nmse_relative"] <= tol["nmse_relative"] for b in boundaries.values()),
        "resident_equals_standalone": all(resident[t] == main["stages"][t]["envelope"] for t in STAGES),
        "classification": all(k1_row[k] == rec_k1[k] for k in CLASS_KEYS),
        "state_removal_classification": all(removed_row[k] == P["removed"]["K1"][k] for k in CLASS_KEYS),
        "reset_challenge": reset["reset_mechanics"] and reset_corrected == 0
        and _nmse_rel(reset["e_reset"], rec_reset["e_reset"]) <= tol["e_reset_relative"]
        and reset["RESET_DEGRADED"] == rec_reset["RESET_DEGRADED"]
        and reset["both_retained_at_C"] == rec_reset["both_retained_at_C"] and set(reset["stages"]) == {"C"},
        "w_only_negative_control": w_only["pass"] and w_only["pass"] == rec_w_only["pass"],
        "transplant_in_process": transplant["continuation_bitwise"] and transplant["restored_digest_equal"],
        "state_semantics": semantics["query_identical"] and semantics["learning_differs"],
    }
    comparisons = {
        "zero_k1_parity": zero_parity, "boundaries": boundaries,
        "resident_envelopes_sha256": {t: _sha(resident[t]) for t in STAGES},
        "init_equals_recorded": main["init"] == rec_k1["init"],
        "reset": {"e_reset": reset["e_reset"], "e_reset_recorded": rec_reset["e_reset"],
                  "e_reset_relative": _nmse_rel(reset["e_reset"], rec_reset["e_reset"]),
                  "reset_ratio": reset["reset_ratio"], "RESET_DEGRADED": reset["RESET_DEGRADED"],
                  "corrected_steps": reset_corrected},
        "w_only": w_only, "transplant": transplant, "state_semantics": semantics,
        "native_counters_after_D": main["stages"]["D"]["stats"],
    }
    rows = {"K1": k1_row, "removed_K1": removed_row, "reset_challenge": reset, "transplant": transplant,
            "w_only": w_only, "state_semantics": semantics}
    return {"world": world.world, "checks": checks, "pass": all(checks.values()), "comparisons": comparisons,
            "rows": rows, "_b_envelope": env_b,
            "_envelopes": {t: main["stages"][t]["envelope"] for t in STAGES}}


def substituted(recorded_per_world: dict, native: dict) -> dict:
    """The recorded per-world record with every native row in place (K1, state removal, the causal branches)."""
    out = copy.deepcopy(recorded_per_world)
    for wid, rec in out.items():
        rows, P = native[wid]["rows"], rec["P"]
        P["mechanisms"]["K1"] = rows["K1"]
        P["removed"]["K1"] = rows["removed_K1"]
        P["causality"]["state_removed"] = rows["removed_K1"]
        P["causality"]["reset_challenge"] = rows["reset_challenge"]
        P["causality"]["full_state_transplant_at_B"] = rows["transplant"]
        P["causality"]["w_only_negative_control_at_B"] = rows["w_only"]
        P["state_semantics"] = rows["state_semantics"]
    return out
