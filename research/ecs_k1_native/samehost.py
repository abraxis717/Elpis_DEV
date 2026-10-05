"""Same-host native K1 differential authority. RESEARCH_ONLY.

Every reference is computed on the host that runs the comparison, from worlds built on that host (R3 task data uses
NumPy/BLAS operations, so its bytes, and every trajectory digest derived from them, depend on the BLAS kernel; a
recorded digest from another host is historical, not a reference). Per world:

* K1-disabled native (never consolidated) is bitwise the same-host canonical Runtime R1 G1 sequence (CognitiveCore);
* native W_A is bitwise the same-host canonical W_A;
* every native query is bitwise the Runtime R1 executor's forward map of the same W;
* the FMS-resident sequence is bitwise the standalone one; every envelope restores bitwise; epochs and provenance are
  exact;
* the reset challenge, the W-only negative control, the transplant continuation and the state semantics hold exactly;
* the native rows (K1, state removal, causal branches) are returned for substitution into the R3 decision record,
  together with the same-host control rows that R3's digest-coupled mechanics compare them with (M0 and M1 through
  the canonical core, the removed C1R engine and C1R's shared W_A through the unchanged R3 laboratory), so that no
  digest from another host ever meets a digest from this one.

Nothing here forces a BLAS kernel or names a host. The v1 harness (differential.py, run.py) stays as historical
authority for K1N-v1 and is reused, unchanged, for the per-branch procedures.
"""
from __future__ import annotations

import hashlib

import numpy as np

from research.ecs_retention_r3 import engine as E
from research.ecs_retention_r3 import experiment as X
from research.ecs_retention_r3 import task as T

from . import differential as D

STAGES = T.TASKS


def canonical_stages(api, world: T.World, rate: float, steps: int) -> dict:
    """The same-host canonical Runtime R1 G1 sequence (M0's computation): W after each stage, and the refusals."""
    W, out = world.w0, {}
    for t in STAGES:
        exp = world.tasks[t]
        W, refused = E.canonical_learn(api, W, exp.x_train, exp.y_train, rate, steps)
        out[t] = {"W": W, "refused": refused}
    return out


def world_samehost(nat: D.Native, lab: X.Lab, wid: str, recorded_rows: dict) -> dict:
    """Every same-host exact invariant on one world, and the native rows for substitution.

    ``recorded_rows`` is the per-world R3 row (``{"P": ...}``) whose K1 and removed-K1 rows give the row layout
    (classes are always recomputed from native nmse; nothing recorded is compared here)."""
    world = lab.world(wid)
    P = recorded_rows["P"]
    rate, steps, th = lab.rate, lab.steps, lab.th
    canonical = canonical_stages(lab.api, world, rate, steps)
    zero = D.standalone_run(nat, world, rate, steps, consolidate=False)
    main = D.standalone_run(nat, world, rate, steps)
    resident = D.resident_envelopes(nat, world, rate, steps)
    heldout = lab.heldout_inputs(world)

    zero_parity = all(np.array_equal(zero["stages"][t]["W"], canonical[t]["W"]) and not canonical[t]["refused"]
                      for t in STAGES) and max(zero["stages"][t]["stats"]["corrected_steps"] for t in STAGES) == 0
    w_a = np.array_equal(main["stages"]["A"]["W"], canonical["A"]["W"])
    query_bitwise = all(np.array_equal(run["stages"][t]["responses"], E.responses(lab.api, run["stages"][t]["W"],
                                                                                  heldout))
                        for run in (zero, main) for t in STAGES)
    boundaries = all(main["stages"][t]["epoch"] == steps * (i + 1) and main["stages"][t]["provenance"] == "COMPLETE"
                     for i, t in enumerate(STAGES))
    restored = {t: D.continue_from(nat, main["stages"][t]["envelope"], world, rate, steps, stages=())["restored"]
                for t in STAGES}
    round_trip = all(restored[t] == main["stages"][t]["envelope"] for t in STAGES)

    k1_row = D._row(P["mechanisms"]["K1"], main, th, k1=True)
    removed_row = D._row(P["removed"]["K1"], zero, th, k1=False)
    env_b = main["stages"]["B"]["envelope"]
    main_c = E.w_digest(main["stages"]["C"]["W"])
    reset = D.reset_branch(nat, env_b, world, rate, steps, th, main["stages"]["C"]["nmse"], main["init"])
    w_only = D.w_only_branch(nat, env_b, main["stages"]["B"]["W"], world, rate, steps,
                             reset["stages"]["C"]["W_digest"], main_c)
    cont = D.continue_from(nat, env_b, world, rate, steps)
    transplant = {"state_bytes": len(env_b), "restored_digest_equal": cont["restored"] == env_b,
                  "continuation_bitwise": cont["C"] == main["stages"]["C"]["envelope"]
                  and cont["D"] == main["stages"]["D"]["envelope"],
                  "digests": {t: hashlib.sha256(cont[t]).hexdigest() for t in ("C", "D")}}
    semantics = {"query_identical": reset.pop("_query_identical"),
                 "learning_differs": main_c != reset["stages"]["C"]["W_digest"]}
    reset_corrected = reset.pop("_corrected_steps")
    checks = {
        "k1_disabled_equals_samehost_g1": bool(zero_parity),
        "w_a_equals_samehost_canonical": bool(w_a),
        "query_equals_runtime_r1_forward": bool(query_bitwise),
        "epoch_and_provenance": bool(boundaries),
        "envelope_round_trip": bool(round_trip),
        "resident_equals_standalone": all(resident[t] == main["stages"][t]["envelope"] for t in STAGES),
        "reset_mechanics": bool(reset["reset_mechanics"]) and reset_corrected == 0 and set(reset["stages"]) == {"C"},
        "w_only_semantics": bool(w_only["flagged_unconsolidated"] and w_only["state_digest_differs"]
                                 and w_only["reproduces_reset_at_C"]),
        "transplant_in_process": bool(transplant["continuation_bitwise"] and transplant["restored_digest_equal"]),
        "state_semantics": bool(semantics["query_identical"]),
        "nothing_refused": all(not canonical[t]["refused"] for t in STAGES),
    }
    controls = {"M0": X.strip_private(lab.canonical_sequence(world, "M0")),
                "M1": X.strip_private(lab.canonical_sequence(world, "M1")),
                "removed_C1R": X.strip_private(lab.run_mechanism(world, "C1R", removed=True)),
                "C1R_W_A_digest": E.w_digest(lab.w_a(world))}
    checks["k1_disabled_equals_samehost_g1"] = checks["k1_disabled_equals_samehost_g1"] and all(
        controls["M0"]["stages"][t]["W_digest"] == E.w_digest(canonical[t]["W"]) for t in STAGES)
    rows = {"K1": k1_row, "removed_K1": removed_row, "reset_challenge": reset, "transplant": transplant,
            "w_only": w_only, "state_semantics": semantics, "controls": controls}
    # Raw witnesses for v2; the comparisons and the historical regression are unchanged.
    evidence = {
        "canonical": {t: {"W_digest": E.w_digest(canonical[t]["W"]),
                           "refused": canonical[t]["refused"]} for t in STAGES},
        "zero": {t: {"W_digest": E.w_digest(zero["stages"][t]["W"]),
                      "stats": zero["stages"][t]["stats"]} for t in STAGES},
        "boundaries": {t: {"epoch": main["stages"][t]["epoch"],
                            "provenance": main["stages"][t]["provenance"],
                            "stats": main["stages"][t]["stats"],
                            "envelope": main["stages"][t]["envelope"].hex(),
                            "restored_sha256": hashlib.sha256(restored[t]).hexdigest(),
                            "resident_sha256": hashlib.sha256(resident[t]).hexdigest()}
                       for t in STAGES},
        "queries": {name: {t: {"native": run["stages"][t]["responses"].tolist(),
                                "canonical": E.responses(lab.api, run["stages"][t]["W"], heldout).tolist()}
                            for t in STAGES} for name, run in (("zero", zero), ("main", main))},
        "reset_corrected_steps": reset_corrected,
    }
    # Preserve the reset/import operands as well as their measured predicates.
    def state_values(state):
        return {"W": list(state.w()), "epoch": state.epoch, "H": list(state.h_packed()),
                "a": list(state.a()), "provenance": state.provenance,
                "envelope_sha256": hashlib.sha256(state.snapshot()).hexdigest(),
                "query": list(state.query(D._mv(heldout)))}
    with D.K1State.restore(nat.k1, env_b) as state:
        evidence["full_B"] = state_values(state)
        state.reset()
        evidence["reset_B"] = state_values(state)
    W_b = main["stages"]["B"]["W"]
    with D.Executor.create(nat.api, *W_b.shape, D._mv(W_b)) as executor:
        snapshot = executor.snapshot()
    with D.K1State.import_w_only(nat.k1, snapshot) as state:
        evidence["w_only_B"] = state_values(state)
    return {"world": wid, "checks": checks, "pass": all(checks.values()), "rows": rows,
            "evidence": evidence,
            "envelopes_sha256": {t: hashlib.sha256(main["stages"][t]["envelope"]).hexdigest() for t in STAGES},
            "_b_envelope": env_b, "_main": main}


def decision_record(per_world_rows: dict, native: dict, spec: dict, determinism: bool, transplant: bool) -> dict:
    """The unchanged R3 gates over ``per_world_rows`` with the native rows substituted; the decision record and the
    final disposition with the given gate-L value carried over."""
    substituted = D.substituted(per_world_rows, native)
    for wid, rec in substituted.items():
        same_host = native[wid]["rows"].get("controls")
        if same_host is not None:
            P = rec["P"]
            P["mechanisms"]["M0"], P["mechanisms"]["M1"] = same_host["M0"], same_host["M1"]
            P["removed"]["C1R"] = same_host["removed_C1R"]
            P["mechanisms"]["C1R"]["W_A_digest"] = same_host["C1R_W_A_digest"]
    gated = X.gates(substituted, spec, {"identical": bool(determinism)}, {"bitwise": bool(transplant)})
    return {"decision_record": gated["decision_record"], "gates": gated["gates"], "validity": gated["validity"],
            "mechanics": gated["mechanics"], "substituted": substituted}
