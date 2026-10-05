"""Retention R1 experiment: per-world procedures, classes, DEV rules, QUAL gates and disposition. RESEARCH_ONLY.

Everything that decides the disposition is in the write-once specification
(``specs/ecsg-retention-r1.v1.spec.json``); this module implements it. Stage A is learned once per world through
the canonical core and shared bitwise by every mechanism. Every response is the canonical native forward map of
the evaluated W, from one executor per state answering every experience (no task label reaches a query).
"""
from __future__ import annotations

import inspect
import statistics

import numpy as np

from elpis.ECS_G.native import Executor

from . import engine as E
from . import numerics
from . import task as T

CANDIDATES = ("K1", "K2", "K3")
ENGINE_KINDS = ("C1R", "K1", "K2", "K3")
STAGES = T.TASKS


def _median(values) -> float:
    return float(statistics.median(values))


class Lab:
    """Per-process caches over one loaded library and the specification."""

    def __init__(self, api, spec: dict):
        self.api, self.spec = api, spec
        reg = spec["regime"]
        self.dim, self.width = reg["dim"], reg["width"]
        self.rate, self.steps = reg["learning_rate"], reg["steps_per_experience"]
        self.idx = T.indices(self.dim)
        self.th = spec["thresholds"]
        self._worlds, self._wa = {}, {}

    # -- worlds ---------------------------------------------------------------------------------------------

    def world(self, arm: str, wid: str, scale: float | None = None) -> T.World:
        key = (arm, wid, scale)
        if key not in self._worlds:
            self._worlds[key] = T.build_world_s(self.spec, wid, scale) if arm == "S" else T.build_world_r(self.spec,
                                                                                                         wid)
        return self._worlds[key]

    def w_a(self, world: T.World) -> np.ndarray:
        key = (world.arm, world.world, world.scale)
        if key not in self._wa:
            a = world.tasks["A"]
            W, refused = E.canonical_learn(self.api, world.w0, a.x_train, a.y_train, self.rate, self.steps)
            self._wa[key] = (W, refused)
        return self._wa[key][0]

    def w_a_refused(self, world: T.World) -> int:
        self.w_a(world)
        return self._wa[(world.arm, world.world, world.scale)][1]

    def evaluate(self, W: np.ndarray, world: T.World) -> dict:
        """One executor created from W answers every experience's held-out inputs (no task label reaches it)."""
        with Executor.create(self.api, self.dim, self.width, E._flat(W)) as e:
            return {t: T.nmse(np.asarray(e.forward(memoryview(world.tasks[t].x_test)), dtype=np.float64),
                              world.tasks[t].y_test) for t in STAGES}

    def heldout_inputs(self, world: T.World) -> np.ndarray:
        return np.vstack([world.tasks[t].x_test for t in STAGES])

    # -- sequences ------------------------------------------------------------------------------------------

    def canonical_sequence(self, world: T.World, kind: str) -> dict:
        """M0 (plain sequential G1) or M1 (cumulative rehearsal) through the canonical core."""
        W = self.w_a(world)
        stages = {"A": self._stage_record(world, W, None, self.steps, self.w_a_refused(world), None, None)}
        stored = 0
        for i, t in enumerate(STAGES[1:], start=1):
            if kind == "M0":
                X, y = world.tasks[t].x_train, world.tasks[t].y_train
            else:
                seen = STAGES[:i + 1]
                X = np.vstack([world.tasks[s].x_train for s in seen])
                y = np.concatenate([world.tasks[s].y_train for s in seen])
                stored = 8 * (X.size + y.size)
            W, refused = E.canonical_learn(self.api, W, X, y, self.rate, self.steps)
            stages[t] = self._stage_record(world, W, None, None, refused, None, None)
            stages[t]["stored_experience_bytes"] = stored
        return self._finish(world, kind, stages, extra={"stored_experience_bytes_final": stored})

    def _stage_record(self, world, W, mech, epoch, refused, info, seconds) -> dict:
        rec = {"nmse": self.evaluate(W, world), "W_digest": E.w_digest(W), "refused": int(refused)}
        if mech is not None:
            rec.update({"state_digest": E.state_digest(mech, W, epoch), "persistent_bytes": mech.persistent_bytes(),
                        "hot_ops": mech.hot_ops(self.dim, self.width), "learn_seconds": seconds,
                        "consolidation": info or {}})
        return rec

    def candidate_run(self, world: T.World, kind: str, *, variant: str | None = None, start=None,
                      keep: bool = False) -> dict:
        """A candidate engine through the sequence from the shared W_A (or from a given state after a stage).

        variant: None, "removed", "uninformed", "mismatched", "per_experience_anchors". ``start`` is
        (stage, mechanism, W, epoch) to continue after that stage (transplant/reset ablations).
        """
        removed = variant == "removed"
        if start is None:
            if variant == "per_experience_anchors":
                mech = E.make("C1R", self.idx, lam=1.0)
            else:
                mech = E.make(kind, self.idx, removed=removed, uninformed=variant == "uninformed")
            W, epoch = self.w_a(world), self.steps
            wa_digest = E.w_digest(W)
            W, mech, info = self._consolidate(world, "A", W, mech)
            stages = {"A": self._stage_record(world, W, mech, epoch, 0, info, None)}
            first = 1
            states = {"A": (mech.clone(), W.copy(), epoch)}
        else:
            stage0, mech, W, epoch = start
            wa_digest, stages, states = None, {}, {}
            first = STAGES.index(stage0) + 1
        for t in STAGES[first:]:
            exp = world.tasks[t]
            W_new, refused, seconds = E.engine_learn(self.api, W, exp.x_train, exp.y_train, self.rate, self.steps,
                                                     mech)
            info = None
            if not refused:
                W2, m2, info = self._consolidate(world, t, W_new, mech.clone())
                if m2 is None:
                    refused = -1                   # non-finite consolidation: the whole transition is refused
                else:
                    W, mech, epoch = W2, m2, epoch + self.steps
            stages[t] = self._stage_record(world, W, mech, epoch, refused, info, seconds)
            if keep:
                states[t] = (mech.clone(), W.copy(), epoch)
        out = {"stages": stages, "W_A_digest": wa_digest, "consolidation_parameters": _consolidation_parameters(mech)}
        if keep:
            out["_states"] = states
            out["_final"] = (mech, W, epoch)
        return out

    def _consolidate(self, world, t, W, mech):
        inputs = world.mismatch[t] if (world.mismatch is not None and getattr(self, "_mismatch", False)) \
            else world.tasks[t].x_train
        rng = T.uninformed_rng(self.spec, world.world, t)
        before = None
        if mech.kind == "K3" and not mech.removed:
            before = (E.s3(W, self.idx), E.responses(self.api, W, self.heldout_inputs(world)))
        with np.errstate(over="ignore", invalid="ignore"):
            W2, info = mech.consolidate(W, inputs, rng)
        finite = np.all(np.isfinite(W2)) and all(np.all(np.isfinite(v)) for v in mech.state_vectors())
        if not finite:
            return W, None, {"refused": True}
        if before is not None:
            s_after = E.s3(W2, self.idx)
            r_after = E.responses(self.api, W2, self.heldout_inputs(world))
            info = dict(info)
            info["s3_relative"] = float(np.max(np.abs(s_after - before[0])) / max(1.0, float(np.max(np.abs(before[0])))))
            info["response_relative"] = float(np.max(np.abs(r_after - before[1]))
                                              / max(1.0, float(np.max(np.abs(before[1])))))
        return W2, mech, info

    def run_mechanism(self, world: T.World, kind: str, variant: str | None = None, keep: bool = False) -> dict:
        if kind in ("M0", "M1"):
            return self.canonical_sequence(world, kind)
        self._mismatch = variant == "mismatched"
        try:
            run = self.candidate_run(world, kind, variant=variant, keep=keep)
        finally:
            self._mismatch = False
        return self._finish(world, kind, run["stages"], run=run)

    def _finish(self, world, kind, stages, run=None, extra=None) -> dict:
        init = self.evaluate(world.w0, world)
        rec = {"kind": kind, "init": init, "stages": stages}
        rec.update(classify({s: stages[s]["nmse"] for s in STAGES}, init, self.th))
        if run is not None:
            rec["W_A_digest"] = run["W_A_digest"]
            rec["consolidation_parameters"] = run["consolidation_parameters"]
            if "_final" in run:
                rec["_run"] = run
        if extra:
            rec.update(extra)
        return rec

    # -- analyses ---------------------------------------------------------------------------------------------

    def ceilings(self, world: T.World) -> dict:
        tasks = world.tasks
        out = {"stage": {}, "previous": {}}
        for i, s in enumerate(STAGES):
            seen = [(tasks[t].x_train, tasks[t].y_train) for t in STAGES[:i + 1]]
            preds = E.ceiling(seen, [tasks[t].x_test for t in STAGES], self.idx)
            out["stage"][s] = {t: T.nmse(p, tasks[t].y_test) for t, p in zip(STAGES, preds)}
            if i:
                prev = [(tasks[t].x_train, tasks[t].y_train) for t in STAGES[:i]]
                out["previous"][s] = T.nmse(E.ceiling(prev, [tasks[s].x_test], self.idx)[0], tasks[s].y_test)
        pair = E.ceiling([(tasks["A"].x_train, tasks["A"].y_train), (tasks["B"].x_train, tasks["B"].y_train)],
                         [tasks["A"].x_test, tasks["B"].x_test], self.idx)
        init = self.evaluate(world.w0, world)
        pa, pb = T.nmse(pair[0], tasks["A"].y_test), T.nmse(pair[1], tasks["B"].y_test)
        ab, rel = self.th["class_nmse_absolute"], self.th["class_nmse_relative"]
        out["pair"] = {"A": pa, "B": pb,
                       "CEILING_FEASIBLE": pa <= ab and pa <= rel * init["A"] and pb <= ab and pb <= rel * init["B"]}
        return out

    def witness(self, world: T.World) -> dict:
        return self.evaluate(world.witness, world)

    def feature_identity(self, world: T.World) -> float:
        W_A = self.w_a(world)
        x = self.heldout_inputs(world)
        native = E.responses(self.api, W_A, x)
        coarse = 0.5 * T.features(x, self.idx) @ E.native_s3(self.api, W_A)
        return float(np.max(np.abs(native - coarse)) / max(1.0, float(np.max(np.abs(native)))))

    def causality(self, world: T.World, W: np.ndarray) -> dict:
        """(i) snapshot/restore of the executor holding W reproduces its responses; W0 reproduces untaught ones."""
        x = self.heldout_inputs(world)
        with Executor.create(self.api, self.dim, self.width, E._flat(W)) as e:
            before = e.forward(memoryview(x))
            blob = e.snapshot()
        with Executor.restore(self.api, blob) as restored:
            transplant = restored.forward(memoryview(x)) == before
        with Executor.create(self.api, self.dim, self.width, E._flat(world.w0)) as first:
            untaught = first.forward(memoryview(x))
        with Executor.create(self.api, self.dim, self.width, E._flat(world.w0)) as again:
            reset = again.forward(memoryview(x)) == untaught
        return {"snapshot_restore_bitwise": bool(transplant), "reset_bitwise": bool(reset)}

    def pte_check(self, world: T.World, kind: str) -> dict:
        """Two microstates with identical S3, each with its own consolidation from A's inputs, diverge after one
        identical candidate step on B: W stays authoritative."""
        p, q = T.pte_pair(world.w0)
        a, b = world.tasks["A"], world.tasks["B"]
        s_p, s_q = E.s3(p, self.idx), E.s3(q, self.idx)
        before = float(np.max(np.abs(s_p - s_q)) / max(1.0, float(np.max(np.abs(s_p)))))
        rng = T.uninformed_rng(self.spec, world.world, "A")
        mp, mq = E.make(kind, self.idx), E.make(kind, self.idx)
        p, _ = mp.consolidate(p, a.x_train, rng)
        q, _ = mq.consolidate(q, a.x_train, rng)
        p1 = E.engine_learn(self.api, p, b.x_train, b.y_train, self.rate, 1, mp)[0]
        q1 = E.engine_learn(self.api, q, b.x_train, b.y_train, self.rate, 1, mq)[0]
        s1p, s1q = E.s3(p1, self.idx), E.s3(q1, self.idx)
        after = float(np.max(np.abs(s1p - s1q)) / max(1.0, float(np.max(np.abs(s1p)))))
        return {"s3_rel_before": before, "s3_rel_after": after, "microstates_differ": bool(np.any(p != q)),
                "pass": before <= self.th["pte_s3_before_max"] and after >= self.th["pte_s3_after_min"]}

    def gradient_identity(self, world: T.World) -> dict:
        """Each correction is the derivative of its declared quantity (central differences, every weight)."""
        W0, WA = world.w0, self.w_a(world)
        a, b = world.tasks["A"], world.tasks["B"]
        out = {}
        h = 1e-6

        def fd(fun, W):
            g = np.zeros_like(W)
            for e in range(W.shape[0]):
                for i in range(W.shape[1]):
                    wp, wm = W.copy(), W.copy()
                    wp[e, i] += h
                    wm[e, i] -= h
                    g[e, i] = (fun(wp) - fun(wm)) / (2 * h)
            return g

        def rel(an, num):
            return float(np.max(np.abs(an - num)) / max(1e-300, float(np.max(np.abs(an)))))

        # K1 and C1R: Q consolidated from A's inputs with the anchor at W0, evaluated at W_A (non-zero there).
        k1 = E.make("K1", self.idx)
        k1.consolidate(W0, a.x_train)
        out["K1"] = rel(k1.grad(WA), fd(k1.objective, WA))
        c1 = E.make("C1R", self.idx)
        c1.consolidate(W0, a.x_train)
        anchor = E.s3(W0, self.idx)

        def c1_obj(W):
            delta = E.s3(W, self.idx) - anchor
            return 0.25 * c1.lam * float(delta @ c1.H @ delta)
        out["C1R"] = rel(c1.grad(WA), fd(c1_obj, WA))
        # K3: Omega with the protected projector of A's statistics, at W_A.
        U = E.protected_basis(E.input_statistics(a.x_train, self.idx))
        P = U @ U.T
        out["K3"] = rel(E.omega_terms(WA, P, self.idx)[1], fd(lambda W: E.omega_terms(W, P, self.idx)[0], WA))
        # K2: the projected step has no protected component (C(W) of it vanishes).
        k2 = E.make("K2", self.idx)
        k2.consolidate(WA, a.x_train)
        stepped = E.engine_learn(self.api, WA, b.x_train, b.y_train, self.rate, 1, E.make("K2", self.idx))[0]
        delta = stepped - WA
        C = k2.U.T @ E.jacobian(WA, self.idx)
        projected = delta.reshape(-1) - k2.projection(WA, delta)
        out["K2"] = float(np.linalg.norm(C @ projected) / max(1e-300, float(np.linalg.norm(C @ delta.reshape(-1)))))
        out["pass"] = (max(out["K1"], out["C1R"], out["K3"]) <= 1e-6 and out["K2"] <= 1e-10)
        return out


def _consolidation_parameters(mech) -> list:
    params = [p for p in inspect.signature(mech.consolidate).parameters if p != "self"]
    return params


def consolidation_interface() -> dict:
    """Structural: what each consolidation receives. No target, label or task parameter exists."""
    out = {}
    for cls in (E.C1R, E.K1, E.K2, E.K3):
        params = [p for p in inspect.signature(cls.consolidate).parameters if p != "self"]
        out[cls.kind] = {"parameters": params,
                         "receives_targets": any(n in ("y", "targets", "labels", "y_train", "task") for n in params)}
    return out


# --- classes ---------------------------------------------------------------------------------------------------


def classify(table: dict, init: dict, th: dict) -> dict:
    """Spec classes over one sequence. ``table[s][t]`` is nmse of experience t after stage s."""
    ab, rel = th["class_nmse_absolute"], th["class_nmse_relative"]
    learned, retained, catastrophic, held = {}, {}, {}, {}
    for i, s in enumerate(STAGES):
        prev = init[s] if i == 0 else table[STAGES[i - 1]][s]
        learned[s] = table[s][s] <= ab and table[s][s] <= rel * prev
        for t in STAGES[:i]:
            retained[f"{t}@{s}"] = table[s][t] <= ab and table[s][t] <= rel * init[t]
            catastrophic[f"{t}@{s}"] = table[s][t] >= init[t]
        held[s] = learned[s] and all(retained[f"{t}@{s}"] for t in STAGES[:i])
    final = table["D"]
    return {"LEARNED": learned, "RETAINED": retained, "CATASTROPHIC": catastrophic, "HELD": held,
            "SEQUENCE_HELD": all(held.values()),
            "PAIR_HELD": learned["B"] and retained["A@B"],
            "final_mean_nmse": float(np.mean([final[t] for t in STAGES])),
            "final_earlier_mean_nmse": float(np.mean([final[t] for t in STAGES[:3]])),
            "A_end_ratio": final["A"] / table["A"]["A"]}


def counts(rows: list) -> dict:
    """Per-mechanism counts over worlds (compared exactly across hosts and kernels; no floats)."""
    out = {"SEQUENCE_HELD": sum(r["SEQUENCE_HELD"] for r in rows), "PAIR_HELD": sum(r["PAIR_HELD"] for r in rows),
           "CATASTROPHIC_any": sum(any(r["CATASTROPHIC"].values()) for r in rows),
           "refused_worlds": sum(any(st["refused"] for st in r["stages"].values()) for r in rows)}
    for s in STAGES:
        out[f"LEARNED_{s}"] = sum(r["LEARNED"][s] for r in rows)
        out[f"HELD_{s}"] = sum(r["HELD"][s] for r in rows)
    for key in rows[0]["RETAINED"]:
        out[f"RETAINED_{key}"] = sum(r["RETAINED"][key] for r in rows)
    return out


def strip_private(rec):
    if isinstance(rec, dict):
        return {k: strip_private(v) for k, v in rec.items() if not k.startswith("_")}
    if isinstance(rec, list):
        return [strip_private(v) for v in rec]
    return rec


# --- DEV ---------------------------------------------------------------------------------------------------------


def dev_controls_world(lab: Lab, wid: str, scale: float) -> dict:
    world = lab.world("S", wid, scale)
    return {"M0": strip_private(lab.run_mechanism(world, "M0")), "M1": strip_private(lab.run_mechanism(world, "M1")),
            "ceilings": lab.ceilings(world), "witness": lab.witness(world),
            "W_A_refused": lab.w_a_refused(world)}


def task_validity_dev(rows: dict, th: dict) -> dict:
    """V1-V4 on DEV worlds for one input scale (spec dev_rules.task)."""
    worlds = list(rows)
    n = len(worlds)
    m0 = [rows[w]["M0"] for w in worlds]
    m1 = [rows[w]["M1"] for w in worlds]
    v1 = all(max(rows[w]["witness"].values()) <= th["witness_max_nmse"] for w in worlds)
    v2 = (sum(r["SEQUENCE_HELD"] for r in m0) <= th["baseline_max_sequence_held_fraction"] * n
          and _median([r["A_end_ratio"] for r in m0]) >= th["baseline_min_median_A_end_ratio"])
    v3 = all(rows[w]["ceilings"]["previous"][s] >= th["novelty_min_nmse"] for w in worlds for s in STAGES[1:])
    v4 = all(all(r["LEARNED"].values()) for r in m0) and all(r["SEQUENCE_HELD"] for r in m1)
    return {"V1_witness": v1, "V2_forgetting": v2, "V3_novelty": v3, "V4_learnability": v4,
            "valid": v1 and v2 and v3 and v4,
            "M0_sequence_held": sum(r["SEQUENCE_HELD"] for r in m0),
            "M0_median_A_end_ratio": _median([r["A_end_ratio"] for r in m0]),
            "M1_sequence_held": sum(r["SEQUENCE_HELD"] for r in m1),
            "min_novelty": min(rows[w]["ceilings"]["previous"][s] for w in worlds for s in STAGES[1:])}


def dev_candidates_world(lab: Lab, wid: str, scale: float) -> dict:
    world = lab.world("S", wid, scale)
    out = {"S": {k: strip_private(lab.run_mechanism(world, k)) for k in ("M0", "M1", "C1R") + CANDIDATES}}
    world_r = lab.world("R", wid)
    out["R"] = {k: strip_private(lab.run_mechanism(world_r, k)) for k in ("M0", "M1", "C1R") + CANDIDATES}
    out["R_ceilings"] = lab.ceilings(world_r)
    return out


def max_hot_ops(rows: list) -> int:
    return max(st.get("hot_ops", 0) for r in rows for st in r["stages"].values())


def select(table: dict) -> tuple[dict, list]:
    """The pre-registered candidate rule (spec dev_rules.candidate)."""
    ranking = []
    for position, kind in enumerate(CANDIDATES):
        rows = [table[w]["S"][kind] for w in sorted(table)]
        ranking.append({"candidate": kind, "sequence_held": sum(r["SEQUENCE_HELD"] for r in rows),
                        "median_final_mean_nmse": _median([r["final_mean_nmse"] for r in rows]),
                        "persistent_bytes": rows[0]["stages"]["A"]["persistent_bytes"],
                        "hot_ops_per_step": max_hot_ops(rows), "position": position})
    ranked = sorted(ranking, key=lambda r: (-r["sequence_held"], r["median_final_mean_nmse"], r["persistent_bytes"],
                                            r["hot_ops_per_step"], r["position"]))
    return {"candidate": ranked[0]["candidate"]}, ranked


# --- QUAL --------------------------------------------------------------------------------------------------------


def qual_world(lab: Lab, wid: str, choices: dict, arms=("S", "R")) -> dict:
    """Every registered measurement of one QUAL world (spec pass_rule, mechanics_checks, ablations, secondary)."""
    scale, sel = choices["input_scale"], choices["selected"]["candidate"]
    out = {"numerics": {"core": numerics.blas_core(), "threads": numerics.blas_threads()}}
    # The registered PTE construction perturbs coordinate 0 and steps on B: it is the R0 v1 construction and is
    # evaluated on the arm-R world (B on axis 0). In arm S, B's inputs vanish on coordinate 0, so a B step never
    # reads or moves that row; there the check is structurally degenerate and recorded as descriptive only.
    out["pte"] = lab.pte_check(lab.world("R", wid), sel)
    if "S" in arms:
        world = lab.world("S", wid, scale)
        S = {"mechanisms": {}, "removed": {}, "ablations": {}}
        for kind in ("M0", "M1", "C1R") + CANDIDATES:
            rec = lab.run_mechanism(world, kind, keep=(kind == sel))
            S["mechanisms"][kind] = rec
        for kind in ENGINE_KINDS:
            S["removed"][kind] = strip_private(lab.run_mechanism(world, kind, "removed"))
        main = S["mechanisms"][sel]["_run"]
        # Ablations of the selected candidate.
        S["ablations"]["state_removed"] = S["removed"][sel]
        S["ablations"]["mismatched_statistics"] = strip_private(lab.run_mechanism(world, sel, "mismatched"))
        S["ablations"]["uninformed_statistics"] = strip_private(lab.run_mechanism(world, sel, "uninformed"))
        if sel == "K1":
            S["ablations"]["per_experience_anchors"] = strip_private(
                lab.run_mechanism(world, "K1", "per_experience_anchors"))
        if sel == "K3":
            S["ablations"]["fibre_removed"] = strip_private(S["mechanisms"]["K1"])
        S["ablations"]["extended_state_transplant_at_B"] = transplant_at_b(lab, world, sel, main)
        S["ablations"]["consolidation_reset_at_B"] = reset_at_b(lab, world, sel, main)
        final_W = main["_final"][1]
        S["mechanisms"] = {k: strip_private(v) for k, v in S["mechanisms"].items()}
        S["ceilings"] = lab.ceilings(world)
        S["witness"] = lab.witness(world)
        S["causality"] = lab.causality(world, final_W)
        S["mechanics"] = {"feature_identity_rel": lab.feature_identity(world),
                          "pte_arm_S_descriptive": lab.pte_check(world, sel),
                          "W_A_refused": lab.w_a_refused(world)}
        if wid.endswith("-0000"):
            S["mechanics"]["gradient_identity"] = lab.gradient_identity(world)
        out["S"] = S
    if "R" in arms:
        world = lab.world("R", wid)
        R = {"mechanisms": {k: strip_private(lab.run_mechanism(world, k)) for k in ("M0", "M1", "C1R") + CANDIDATES},
             "removed": {k: strip_private(lab.run_mechanism(world, k, "removed")) for k in ENGINE_KINDS},
             "ceilings": lab.ceilings(world),
             "mechanics": {"feature_identity_rel": lab.feature_identity(world),
                           "W_A_refused": lab.w_a_refused(world)}}
        out["R"] = R
    return out


def transplant_at_b(lab: Lab, world: T.World, kind: str, main: dict) -> dict:
    """Serialize the complete declared state after B, restore it into a fresh engine, continue C and D."""
    mech, W, epoch = main["_states"]["B"]
    blob = E.serialize(mech, W, epoch)
    rng = T.uninformed_rng(lab.spec, world.world, "B")
    m2, W2, e2 = E.deserialize(blob, lab.idx, rng)
    cont = lab.candidate_run(world, kind, start=("B", m2, W2, e2))
    ref = main["stages"]
    bitwise = all(cont["stages"][s]["W_digest"] == ref[s]["W_digest"]
                  and cont["stages"][s]["state_digest"] == ref[s]["state_digest"] for s in ("C", "D"))
    return {"state_bytes": len(blob), "restored_digest_equal": E.state_digest(m2, W2, e2) == E.state_digest(mech, W,
                                                                                                          epoch),
            "continuation_bitwise": bool(bitwise),
            "digests": {s: cont["stages"][s]["state_digest"] for s in ("C", "D")}}


def reset_at_b(lab: Lab, world: T.World, kind: str, main: dict) -> dict:
    """Keep W after B, empty the consolidation state, continue C and D with the candidate engine."""
    _, W, epoch = main["_states"]["B"]
    fresh = E.make(kind, lab.idx)
    cont = lab.candidate_run(world, kind, start=("B", fresh, W.copy(), epoch))
    final = cont["stages"]["D"]["nmse"]
    init = lab.evaluate(world.w0, world)
    th = lab.th
    keep = {t: final[t] <= th["class_nmse_absolute"] and final[t] <= th["class_nmse_relative"] * init[t]
            for t in ("A", "B")}
    return {"stages": cont["stages"], "A_retained_at_D": keep["A"], "B_retained_at_D": keep["B"],
            "both_retained_at_D": keep["A"] and keep["B"]}


def determinism_digest(lab_factory, wid: str, choices: dict) -> dict:
    """The selected candidate's arm-S pipeline on one world from the specification alone: per-stage digests and
    metrics (a fresh Lab, no caches)."""
    lab = lab_factory()
    world = lab.world("S", wid, choices["input_scale"])
    rec = lab.run_mechanism(world, choices["selected"]["candidate"])
    return {"stages": {s: {k: rec["stages"][s][k] for k in ("W_digest", "state_digest", "nmse")} for s in STAGES}}


# --- gates and disposition ---------------------------------------------------------------------------------------


def gates(per_world: dict, spec: dict, choices: dict, determinism: dict, transplant_probe: dict) -> dict:
    """Validity, mechanics, gates A-L and the disposition without gate M (M is decided by the robustness runs)."""
    th = spec["thresholds"]
    sel = choices["selected"]["candidate"]
    worlds = list(per_world)
    n = len(worlds)
    S = lambda w: per_world[w]["S"]  # noqa: E731
    mech = lambda name: [S(w)["mechanisms"][name] for w in worlds]  # noqa: E731
    abl = lambda name: [S(w)["ablations"][name] for w in worlds]  # noqa: E731
    sel_rows, m0, m1 = mech(sel), mech("M0"), mech("M1")
    removed = abl("state_removed")
    stage_keys = STAGES
    candidate_stages = [st for r in sel_rows for st in r["stages"].values()]

    validity = {
        "V2_forgetting": sum(r["SEQUENCE_HELD"] for r in m0) <= th["baseline_max_sequence_held_fraction"] * n
        and _median([r["A_end_ratio"] for r in m0]) >= th["baseline_min_median_A_end_ratio"],
        "V3_novelty": sum(S(w)["ceilings"]["previous"][s] >= th["novelty_min_nmse"]
                          for w in worlds for s in STAGES[1:]) >= th["qual_validity_min_fraction"] * 3 * n,
        "V4_rehearsal_feasibility": sum(r["SEQUENCE_HELD"] for r in m1) >= th["qual_validity_min_fraction"] * n,
    }

    def engine_equivalent(arm):
        for w in worlds:
            ref = per_world[w][arm]["mechanisms"]["M0"]["stages"]
            for kind in ENGINE_KINDS:
                got = per_world[w][arm]["removed"][kind]["stages"]
                if any(got[s]["W_digest"] != ref[s]["W_digest"] for s in stage_keys):
                    return False
        return True

    arms = [a for a in ("S", "R") if a in per_world[worlds[0]]]
    fibre = []   # every K3 reconditioning recorded anywhere (mechanisms and ablations, both arms)
    for w in worlds:
        for arm in arms:
            recs = list(per_world[w][arm]["mechanisms"].values())
            if arm == "S":
                recs += [r for r in per_world[w]["S"]["ablations"].values() if isinstance(r, dict) and "stages" in r]
            for r in recs:
                for st in r["stages"].values():
                    info = st.get("consolidation") or {}
                    if "s3_relative" in info:
                        fibre.append(info["s3_relative"] <= th["fibre_s3_relative"]
                                     and info["response_relative"] <= th["fibre_response_relative"])
    gi = per_world[worlds[0]]["S"]["mechanics"].get("gradient_identity", {"pass": False})
    interface = consolidation_interface()
    mechanics = {
        "feature_identity": all(per_world[w][a]["mechanics"]["feature_identity_rel"] <= th["feature_identity_relative"]
                                for w in worlds for a in arms),
        "witness_exact": all(max(S(w)["witness"].values()) <= th["witness_max_nmse"] for w in worlds),
        "engine_equivalence": all(engine_equivalent(a) for a in arms),
        "shared_W_A": all(per_world[w][a]["mechanisms"][k]["W_A_digest"]
                          == per_world[w][a]["mechanisms"]["M0"]["stages"]["A"]["W_digest"]
                          for w in worlds for a in arms for k in ("C1R",) + CANDIDATES),
        "no_baseline_refusal": all(not any(st["refused"] for st in per_world[w][a]["mechanisms"][k]["stages"].values())
                                   and per_world[w][a]["mechanics"]["W_A_refused"] == 0
                                   for w in worlds for a in arms for k in ("M0", "M1")),
        "consolidation_interface": not any(v["receives_targets"] for v in interface.values()),
        "pte_microstate": all(per_world[w]["pte"]["pass"] for w in worlds),
        "fibre_function_preservation": all(fibre) if fibre else sel != "K3",
        "gradient_identity": bool(gi["pass"]),
        "effective_single_thread": all(per_world[w]["numerics"]["threads"] == 1 for w in worlds),
    }

    def frac(rows, pred):
        return sum(1 for r in rows if pred(r)) / n

    bytes_fixed = all(len({st["persistent_bytes"] for st in r["stages"].values()}) == 1 for r in sel_rows)
    g1 = E.g1_ops(spec["regime"]["dim"], spec["regime"]["width"], spec["regime"]["train_rows"])
    hot = max(st["hot_ops"] for st in candidate_stages)
    cold = max((st["consolidation"] or {}).get("cold_ops", 0) for st in candidate_stages)
    transplant = abl("extended_state_transplant_at_B")
    reset = abl("consolidation_reset_at_B")
    g = {
        "A_first_acquisition": all(r["LEARNED"]["A"] for r in m0)
        and _median([r["stages"]["A"]["nmse"]["A"] for r in m0]) <= th["first_acquisition_median_nmse"],
        "B_stage_acquisition": all(r["LEARNED"][s] for r in sel_rows for s in STAGES[1:]),
        "C_sequence_retention": all(all(r["RETAINED"].values()) for r in sel_rows),
        "D_no_catastrophe": not any(any(r["CATASTROPHIC"].values()) for r in sel_rows),
        "E_joint_quality": _median([r["final_mean_nmse"] for r in sel_rows])
        <= th["joint_quality_median_final_mean_nmse"],
        "F_joint_state": all(all(st.get("W_digest") for st in r["stages"].values()) for r in sel_rows),
        "G_state_causality": (
            all(S(w)["causality"]["snapshot_restore_bitwise"] and S(w)["causality"]["reset_bitwise"] for w in worlds)
            and frac(removed, lambda r: r["SEQUENCE_HELD"]) <= th["ablation_max_sequence_held_fraction"]
            and _median([r["final_earlier_mean_nmse"] for r in removed])
            >= th["ablation_min_median_factor"] * _median([r["final_earlier_mean_nmse"] for r in sel_rows])
            and all(t["continuation_bitwise"] and t["restored_digest_equal"] for t in transplant)
            and bool(transplant_probe.get("bitwise"))
            and frac(reset, lambda r: r["both_retained_at_D"]) <= th["reset_max_both_retained_fraction"]),
        "H_no_external_answer_store": (not interface[sel]["receives_targets"]) and bytes_fixed,
        "I_determinism": bool(determinism.get("identical")),
        "J_control_honesty": sel in CANDIDATES,
        "K_capacity_accounting": all("persistent_bytes" in st and "hot_ops" in st
                                     for k in ("C1R",) + CANDIDATES for r in mech(k) for st in r["stages"].values())
        and all("stored_experience_bytes_final" in r for r in m1),
        "L_native_feasibility": bytes_fixed and hot <= th["native_hot_path_max_g1_multiple"] * g1
        and cold <= th["native_cold_path_max_ops_per_consolidation"],
    }
    medians = {}
    names = ["M0", "M1", "C1R"] + list(CANDIDATES)
    for name in names:
        rows = mech(name)
        medians[name] = {"final_mean_nmse": _median([r["final_mean_nmse"] for r in rows]),
                         "final_earlier_mean_nmse": _median([r["final_earlier_mean_nmse"] for r in rows]),
                         "A_end_ratio": _median([r["A_end_ratio"] for r in rows])}
    all_counts = {name: counts(mech(name)) for name in names}
    for name in S(worlds[0])["ablations"]:
        rows = abl(name)
        if "SEQUENCE_HELD" in rows[0]:
            all_counts[f"ablation:{name}"] = counts(rows)
    all_counts["ablation:consolidation_reset_at_B"] = {"both_retained_at_D": sum(r["both_retained_at_D"]
                                                                                  for r in reset)}
    all_counts["ablation:extended_state_transplant_at_B"] = {"bitwise": sum(t["continuation_bitwise"]
                                                                            for t in transplant)}
    evidence = {"g1_ops_per_step": g1, "max_hot_ops_per_step": hot, "max_cold_ops_per_consolidation": cold,
                "persistent_bytes_fixed": bytes_fixed,
                "persistent_bytes": sel_rows[0]["stages"]["A"]["persistent_bytes"]}
    return {"validity": validity, "mechanics": mechanics, "gates": g, "medians": medians, "counts": all_counts,
            "native_feasibility_evidence": evidence, "consolidation_interface": interface,
            **disposition(validity, mechanics, g, mech, sel_rows, m0, m1, n, th)}


def disposition(validity, mechanics, g, mech, sel_rows, m0, m1, n, th) -> dict:
    """The registered disposition (spec pass_rule.disposition) over gates A-L (and M once known)."""
    if not all(mechanics.values()):
        return {"disposition": "MECHANICS_FAIL", "outcome": None}
    if not all(validity.values()):
        return {"disposition": "TASK_INVALID_UNDER_QUAL", "outcome": "OUTCOME_V"}
    if all(g.values()):
        return {"disposition": "RETENTION_SUPPORTED_UNDER_FROZEN_SYNTHETIC_REGIME", "outcome": "OUTCOME_A"}
    sel_held, m0_held = sum(r["SEQUENCE_HELD"] for r in sel_rows), sum(r["SEQUENCE_HELD"] for r in m0)
    if sel_held > m0_held and _median([r["final_earlier_mean_nmse"] for r in sel_rows]) \
            <= th["partial_reduction_max_median_factor"] * _median([r["final_earlier_mean_nmse"] for r in m0]):
        return {"disposition": "PARTIAL_REDUCTION", "outcome": "OUTCOME_C"}
    if all(r["SEQUENCE_HELD"] for r in m1):
        return {"disposition": "REPLAY_ONLY", "outcome": "OUTCOME_B"}
    return {"disposition": "NO_MATERIAL_IMPROVEMENT", "outcome": "OUTCOME_D"}


def qualifiers(per_world: dict, choices: dict, th: dict, robust_ok: bool) -> dict:
    worlds = list(per_world)
    sel = choices["selected"]["candidate"]
    rows = lambda k: [per_world[w]["S"]["mechanisms"][k] for w in worlds]  # noqa: E731
    c1r, s, k1, k3 = rows("C1R"), rows(sel), rows("K1"), rows("K3")
    held = lambda rs: sum(r["SEQUENCE_HELD"] for r in rs)  # noqa: E731
    med = lambda rs, key: _median([r[key] for r in rs])  # noqa: E731
    c1r_ok = (all(r["LEARNED"][x] for r in c1r for x in STAGES[1:]) and all(all(r["RETAINED"].values()) for r in c1r)
              and not any(any(r["CATASTROPHIC"].values()) for r in c1r)
              and med(c1r, "final_mean_nmse") <= th["joint_quality_median_final_mean_nmse"])
    exceeds = held(s) > held(c1r) or (held(s) == held(c1r) and med(s, "final_mean_nmse") < med(c1r, "final_mean_nmse"))
    e1, e3 = med(k1, "final_earlier_mean_nmse"), med(k3, "final_earlier_mean_nmse")
    micro = held(k1) != held(k3) or max(e1, e3) > 2.0 * min(e1, e3)
    return {"R0_REFERENCE_ALSO_SUFFICIENT": c1r_ok, "EXCEEDS_R0_REFERENCE": exceeds, "MICROSTATE_EFFECT": micro,
            "NUMERICALLY_FRAGILE": not robust_ok, "CAPACITY_SCOPE": True}


def arm_r_summary(per_world: dict) -> dict:
    worlds = list(per_world)
    feasible = [w for w in worlds if per_world[w]["R"]["ceilings"]["pair"]["CEILING_FEASIBLE"]]
    out = {"CEILING_FEASIBLE_worlds": len(feasible), "worlds": len(worlds), "mechanisms": {}}
    for k in ("M0", "M1", "C1R") + CANDIDATES:
        rows = [per_world[w]["R"]["mechanisms"][k] for w in worlds]
        out["mechanisms"][k] = {
            "PAIR_HELD": sum(r["PAIR_HELD"] for r in rows),
            "PAIR_HELD_in_ceiling_feasible": sum(per_world[w]["R"]["mechanisms"][k]["PAIR_HELD"] for w in feasible),
            "RETAINED_A@B": sum(r["RETAINED"]["A@B"] for r in rows), "LEARNED_B": sum(r["LEARNED"]["B"] for r in rows),
            "CATASTROPHIC_A@B": sum(r["CATASTROPHIC"]["A@B"] for r in rows),
            "SEQUENCE_HELD": sum(r["SEQUENCE_HELD"] for r in rows),
            "median_final_mean_nmse": _median([r["final_mean_nmse"] for r in rows])}
    return out
