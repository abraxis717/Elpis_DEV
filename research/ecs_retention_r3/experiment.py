"""Retention R3 experiment: per-world procedures, classes, the DEV rules, QUAL gates A-K, the decision record and
the disposition. RESEARCH_ONLY.

Everything that decides the disposition is in the write-once specification (``specs/ecsg-retention-r3.v1.spec.json``);
this module implements it. Stage A is learned once per world through the canonical core and shared bitwise by
every mechanism. Every response is the canonical native forward map of the evaluated W, from one executor per
state answering every experience (no task label reaches a query). K1 is the only eligible mechanism; C1R is a
reported reference; M0, M1, O and WSTAR are controls.
"""
from __future__ import annotations

import ast
import inspect
import statistics
import textwrap

import numpy as np

from elpis.ECS_G.native import Executor

from . import engine as E
from . import numerics
from . import task as T

STAGES = T.TASKS
ENGINE_KINDS = ("C1R", "K1")
PROTECTED = ("A", "B")


def _median(values) -> float:
    return float(statistics.median(values))


class Lab:
    """Per-process caches over one loaded library and the specification."""

    def __init__(self, api, spec: dict):
        self.api, self.spec = api, spec
        reg = spec["regime"]
        self.dim, self.width = reg["dim"], reg["width"]
        self.rate, self.steps, self.rows = reg["learning_rate"], reg["steps_per_experience"], reg["train_rows"]
        self.gain = spec["task"]["hidden_gain"]
        self.idx = T.indices(self.dim)
        self.th = spec["thresholds"]
        self._worlds, self._wa = {}, {}

    # -- worlds ---------------------------------------------------------------------------------------------

    def world(self, wid: str) -> T.World:
        if wid not in self._worlds:
            self._worlds[wid] = T.build_world(self.spec, wid, self.gain)
        return self._worlds[wid]

    def w_a(self, world: T.World) -> np.ndarray:
        if world.world not in self._wa:
            a = world.tasks["A"]
            self._wa[world.world] = E.canonical_learn(self.api, world.w0, a.x_train, a.y_train, self.rate, self.steps)
        return self._wa[world.world][0]

    def w_a_refused(self, world: T.World) -> int:
        self.w_a(world)
        return self._wa[world.world][1]

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
        stages = {"A": self._stage_record(world, W, None, None, self.w_a_refused(world))}
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
            stages[t] = self._stage_record(world, W, None, None, refused)
            stages[t]["stored_experience_bytes"] = stored
        return self._finish(world, kind, stages, extra={"stored_experience_bytes_final": stored})

    def _stage_record(self, world, W, mech, epoch, refused, seconds=None) -> dict:
        rec = {"nmse": self.evaluate(W, world), "W_digest": E.w_digest(W), "refused": int(refused)}
        if mech is not None:
            rec.update({"state_digest": E.state_digest(mech, W, epoch), "persistent_bytes": mech.persistent_bytes(),
                        "hot_ops": mech.hot_ops(self.dim, self.width),
                        "consolidation_ops": mech.consolidation_ops(self.dim, self.width, self.rows),
                        "learn_seconds": seconds})
        return rec

    def mechanism_run(self, world: T.World, kind: str, *, removed: bool = False, start=None, until: str = "D",
                      keep: bool = False) -> dict:
        """K1 or C1R through the sequence from the shared W_A (or continuing a given state after a stage).

        ``start`` is (stage, mechanism, W, epoch); ``until`` is the last stage learned. Learning one experience and
        consolidating it is one atomic transition: a refusal or a non-finite value leaves the complete state as it
        was, recorded as refused.
        """
        if start is None:
            mech = E.make(kind, self.idx, removed=removed)
            W, epoch = self.w_a(world), self.steps
            wa_digest = E.w_digest(W)
            W, mech = self._consolidate(world, "A", W, mech)
            stages = {"A": self._stage_record(world, W, mech, epoch, 0)}
            first = 1
            states = {"A": (mech.clone(), W.copy(), epoch)}
        else:
            stage0, mech, W, epoch = start
            wa_digest, stages, states = None, {}, {}
            first = STAGES.index(stage0) + 1
        for t in STAGES[first:STAGES.index(until) + 1]:
            exp = world.tasks[t]
            W_new, refused, seconds = E.engine_learn(self.api, W, exp.x_train, exp.y_train, self.rate, self.steps,
                                                     mech)
            if not refused:
                W2, m2 = self._consolidate(world, t, W_new, mech.clone())
                if m2 is None:
                    refused = -1                   # non-finite consolidation: the whole transition is refused
                else:
                    W, mech, epoch = W2, m2, epoch + self.steps
            stages[t] = self._stage_record(world, W, mech, epoch, refused, seconds)
            if keep:
                states[t] = (mech.clone(), W.copy(), epoch)
        out = {"stages": stages, "W_A_digest": wa_digest}
        if keep:
            out["_states"] = states
            out["_final"] = (mech, W, epoch)
        return out

    def _consolidate(self, world, t, W, mech):
        with np.errstate(over="ignore", invalid="ignore"):
            W2, _ = mech.consolidate(W, world.tasks[t].x_train)
        if not (np.all(np.isfinite(W2)) and all(np.all(np.isfinite(v)) for v in mech.state_vectors())):
            return W, None
        return W2, mech

    def run_mechanism(self, world: T.World, kind: str, removed: bool = False, keep: bool = False) -> dict:
        if kind in ("M0", "M1"):
            return self.canonical_sequence(world, kind)
        run = self.mechanism_run(world, kind, removed=removed, keep=keep)
        return self._finish(world, kind, run["stages"], run=run)

    def _finish(self, world, kind, stages, run=None, extra=None) -> dict:
        init = self.evaluate(world.w0, world)
        rec = {"kind": kind, "init": init, "stages": stages}
        rec.update(classify({s: stages[s]["nmse"] for s in STAGES}, init, self.th))
        if run is not None:
            rec["W_A_digest"] = run["W_A_digest"]
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
        out["capacity_max"] = max(out["stage"][s][t] for i, s in enumerate(STAGES) for t in STAGES[:i + 1])
        return out

    def witness(self, world: T.World) -> dict:
        return self.evaluate(world.witness, world)

    def feature_identity(self, world: T.World) -> float:
        W_A = self.w_a(world)
        x = self.heldout_inputs(world)
        native = E.responses(self.api, W_A, x)
        coarse = 0.5 * T.features(x, self.idx) @ E.native_s3(self.api, W_A)
        return float(np.max(np.abs(native - coarse)) / max(1.0, float(np.max(np.abs(native)))))

    def pte_check(self, world: T.World) -> dict:
        """Two microstates with identical S3, each with its own K1 consolidation from A's inputs, diverge after one
        identical K1 step on B: W stays authoritative."""
        p, q = T.pte_pair(world.w0, world.pte_axis)
        a, b = world.tasks["A"], world.tasks["B"]
        s_p, s_q = E.s3(p, self.idx), E.s3(q, self.idx)
        before = float(np.max(np.abs(s_p - s_q)) / max(1.0, float(np.max(np.abs(s_p)))))
        mp, mq = E.make("K1", self.idx), E.make("K1", self.idx)
        p, _ = mp.consolidate(p, a.x_train)
        q, _ = mq.consolidate(q, a.x_train)
        p1 = E.engine_learn(self.api, p, b.x_train, b.y_train, self.rate, 1, mp)[0]
        q1 = E.engine_learn(self.api, q, b.x_train, b.y_train, self.rate, 1, mq)[0]
        s1p, s1q = E.s3(p1, self.idx), E.s3(q1, self.idx)
        after = float(np.max(np.abs(s1p - s1q)) / max(1.0, float(np.max(np.abs(s1p)))))
        return {"s3_rel_before": before, "s3_rel_after": after, "microstates_differ": bool(np.any(p != q)),
                "pass": before <= self.th["pte_s3_before_max"] and after >= self.th["pte_s3_after_min"]}

    def gradient_identity(self, world: T.World) -> dict:
        """The K1 and C1R corrections are the derivatives of their declared quantities (central differences)."""
        W0, WA = world.w0, self.w_a(world)
        a = world.tasks["A"]
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

        k1 = E.make("K1", self.idx)
        k1.consolidate(W0, a.x_train)
        out = {"K1": rel(k1.grad(WA), fd(k1.objective, WA))}
        c1 = E.make("C1R", self.idx)
        c1.consolidate(W0, a.x_train)
        anchor = E.s3(W0, self.idx)

        def c1_obj(W):
            delta = E.s3(W, self.idx) - anchor
            return 0.25 * c1.lam * float(delta @ c1.H @ delta)
        out["C1R"] = rel(c1.grad(WA), fd(c1_obj, WA))
        out["pass"] = max(out["K1"], out["C1R"]) <= self.th["gradient_identity_relative"]
        return out


# --- structural interface checks --------------------------------------------------------------------------------


def consolidation_interface() -> dict:
    """What each consolidation receives (no target, label or task parameter) and whether it is closed form."""
    out = {}
    for cls in (E.C1R, E.K1):
        params = [p for p in inspect.signature(cls.consolidate).parameters if p != "self"]
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls.consolidate)))
        loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While, ast.AsyncFor))]
        out[cls.kind] = {"parameters": params,
                         "receives_targets": any(n in ("y", "targets", "labels", "y_train", "task", "task_id")
                                                 for n in params),
                         "closed_form": not loops}
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


# --- the causal branches (spec causality) -----------------------------------------------------------------------


def transplant_at_b(lab: Lab, world: T.World, main: dict) -> dict:
    """Serialize the complete K1 state after B, restore it into a fresh engine, continue C and D."""
    mech, W, epoch = main["_states"]["B"]
    blob = E.serialize(mech, W, epoch)
    m2, W2, e2 = E.deserialize(blob, lab.idx)
    cont = lab.mechanism_run(world, "K1", start=("B", m2, W2, e2))
    ref = main["stages"]
    bitwise = all(cont["stages"][s]["W_digest"] == ref[s]["W_digest"]
                  and cont["stages"][s]["state_digest"] == ref[s]["state_digest"] for s in ("C", "D"))
    return {"state_bytes": len(blob),
            "restored_digest_equal": E.state_digest(m2, W2, e2) == E.state_digest(mech, W, epoch),
            "continuation_bitwise": bool(bitwise),
            "digests": {s: cont["stages"][s]["state_digest"] for s in ("C", "D")}}


def reset_challenge(lab: Lab, world: T.World, main: dict) -> dict:
    """Reset H and a after B (W, epoch kept), learn C with K1 (no inherited protection), evaluate immediately at
    the C boundary and stop: no D, no further consolidation. Compared with uninterrupted K1 at the same boundary."""
    full = main["_states"]["B"]
    mech, W, epoch = E.reset(full)
    mechanics = (np.array_equal(W, full[1]) and epoch == full[2] and not mech.H.any() and not mech.a.any()
                 and mech.provenance == "RESET")
    c = world.tasks["C"]
    W_c, refused, _ = E.engine_learn(lab.api, W, c.x_train, c.y_train, lab.rate, lab.steps, mech)
    nmse = lab.evaluate(W_c, world)
    ref = main["stages"]["C"]["nmse"]
    e_reset = float(np.mean([nmse[t] for t in PROTECTED]))
    e_main = float(np.mean([ref[t] for t in PROTECTED]))
    ratio = e_reset / max(e_main, 1e-300)
    th = lab.th
    init = lab.evaluate(world.w0, world)
    retained = {t: nmse[t] <= th["class_nmse_absolute"] and nmse[t] <= th["class_nmse_relative"] * init[t]
                for t in PROTECTED}
    return {"stages": {"C": {"nmse": nmse, "W_digest": E.w_digest(W_c), "refused": int(refused)}},
            "reset_mechanics": bool(mechanics), "e_reset": e_reset, "e_uninterrupted": e_main, "reset_ratio": ratio,
            "RESET_DEGRADED": bool(e_reset >= th["reset_min_factor"] * e_main
                                   and e_reset >= th["reset_min_absolute_nmse"]),
            "both_retained_at_C": bool(retained["A"] and retained["B"])}


def w_only_control(lab: Lab, world: T.World, main: dict, reset: dict) -> dict:
    """The canonical ELPISG01 W-only snapshot of W_B, imported: UNCONSOLIDATED, a different state, the reset branch."""
    mech, W, epoch = main["_states"]["B"]
    with Executor.create(lab.api, lab.dim, lab.width, E._flat(W)) as e:
        snapshot = e.snapshot()
    m2, W2, e2 = E.import_w_only(snapshot, lab.idx, lab.dim, lab.width)
    flagged = (m2.provenance == "UNCONSOLIDATED_IMPORT" and not m2.H.any() and not m2.a.any()
               and np.array_equal(W2, W))
    digest_differs = E.state_digest(m2, W2, e2) != E.state_digest(mech, W, epoch)
    c = world.tasks["C"]
    W_c, refused, _ = E.engine_learn(lab.api, W2, c.x_train, c.y_train, lab.rate, lab.steps, m2)
    reproduces_reset = E.w_digest(W_c) == reset["stages"]["C"]["W_digest"]
    differs = E.w_digest(W_c) != main["stages"]["C"]["W_digest"]
    return {"snapshot_bytes": len(snapshot), "flagged_unconsolidated": bool(flagged),
            "state_digest_differs": bool(digest_differs), "reproduces_reset_at_C": bool(reproduces_reset),
            "differs_from_uninterrupted_at_C": bool(differs),
            "pass": bool(flagged and digest_differs and reproduces_reset and differs and not refused)}


def state_semantics(lab: Lab, world: T.World, main: dict, reset: dict) -> dict:
    """At B: identical W with and without (H, a) answers every query bitwise identically; learning C from the two
    states differs (RET3B implementation decision: the comparison is the learning of experience C, because the first
    step from the consolidation anchor carries a zero K1 correction by construction, u(W_B) = 0)."""
    full = main["_states"]["B"]
    x = lab.heldout_inputs(world)
    query_identical = np.array_equal(E.query(lab.api, full, x), E.query(lab.api, E.reset(full), x))
    learning_differs = main["stages"]["C"]["W_digest"] != reset["stages"]["C"]["W_digest"]
    return {"query_identical": bool(query_identical), "learning_differs": bool(learning_differs)}


def k1_world(lab: Lab, world: T.World) -> dict:
    """K1 with its causal branches, the removed engines and C1R (shared by DEV and QUAL)."""
    out = {"mechanisms": {}, "removed": {}, "causality": {}}
    k1 = lab.run_mechanism(world, "K1", keep=True)
    main = k1["_run"]
    out["mechanisms"]["K1"] = strip_private(k1)
    out["mechanisms"]["C1R"] = strip_private(lab.run_mechanism(world, "C1R"))
    for kind in ENGINE_KINDS:
        out["removed"][kind] = strip_private(lab.run_mechanism(world, kind, removed=True))
    reset = reset_challenge(lab, world, main)
    out["causality"]["state_removed"] = out["removed"]["K1"]
    out["causality"]["full_state_transplant_at_B"] = transplant_at_b(lab, world, main)
    out["causality"]["reset_challenge"] = reset
    out["causality"]["w_only_negative_control_at_B"] = w_only_control(lab, world, main, reset)
    out["state_semantics"] = state_semantics(lab, world, main, reset)
    return out


# --- DEV ---------------------------------------------------------------------------------------------------------


def dev_controls_world(lab: Lab, wid: str) -> dict:
    world = lab.world(wid)
    return {"M0": strip_private(lab.run_mechanism(world, "M0")), "M1": strip_private(lab.run_mechanism(world, "M1")),
            "ceilings": lab.ceilings(world), "witness": lab.witness(world), "W_A_refused": lab.w_a_refused(world)}


def task_validity_dev(rows: dict, th: dict) -> dict:
    """V1-V6 on the DEV worlds at the inherited gain (spec dev_rules.task). Controls only."""
    worlds = list(rows)
    n = len(worlds)
    m0 = [rows[w]["M0"] for w in worlds]
    m1 = [rows[w]["M1"] for w in worlds]
    v1 = all(max(rows[w]["witness"].values()) <= th["witness_max_nmse"] for w in worlds)
    v2 = (sum(r["SEQUENCE_HELD"] for r in m0) <= th["baseline_max_sequence_held_fraction"] * n
          and _median([r["final_earlier_mean_nmse"] for r in m0]) >= th["baseline_min_median_final_earlier_nmse"]
          and _median([r["A_end_ratio"] for r in m0]) >= th["baseline_min_median_A_end_ratio"])
    v3 = all(rows[w]["ceilings"]["previous"][s] >= th["novelty_min_nmse"] for w in worlds for s in STAGES[1:])
    v4 = all(all(r["LEARNED"].values()) for r in m0)
    v5 = all(r["SEQUENCE_HELD"] for r in m1)
    v6 = all(rows[w]["ceilings"]["capacity_max"] <= th["capacity_max_ceiling_nmse"] for w in worlds)
    return {"V1_realizability": v1, "V2_forgetting": v2, "V3_novelty": v3, "V4_learnability": v4,
            "V5_joint_solvability": v5, "V6_capacity": v6, "valid": v1 and v2 and v3 and v4 and v5 and v6,
            "M0_sequence_held": sum(r["SEQUENCE_HELD"] for r in m0),
            "M0_median_final_earlier_mean_nmse": _median([r["final_earlier_mean_nmse"] for r in m0]),
            "M0_median_A_end_ratio": _median([r["A_end_ratio"] for r in m0]),
            "M0_learned_all": sum(all(r["LEARNED"].values()) for r in m0),
            "M1_sequence_held": sum(r["SEQUENCE_HELD"] for r in m1),
            "min_novelty": min(rows[w]["ceilings"]["previous"][s] for w in worlds for s in STAGES[1:]),
            "max_capacity_ceiling": max(rows[w]["ceilings"]["capacity_max"] for w in worlds)}


def dev_k1_world(lab: Lab, wid: str) -> dict:
    return k1_world(lab, lab.world(wid))


def k1_criteria(rows: list, th: dict, spec: dict) -> dict:
    """The K1 criteria shared by the DEV rule and QUAL gates A-D, F, I and K (rows: one k1_world per world)."""
    n = len(rows)
    k1 = [r["mechanisms"]["K1"] for r in rows]
    removed = [r["causality"]["state_removed"] for r in rows]
    reset = [r["causality"]["reset_challenge"] for r in rows]
    transplant = [r["causality"]["full_state_transplant_at_B"] for r in rows]
    w_only = [r["causality"]["w_only_negative_control_at_B"] for r in rows]
    stages = [st for r in k1 for st in r["stages"].values()]
    reg = spec["regime"]
    g1 = E.g1_ops(reg["dim"], reg["width"], reg["train_rows"])
    hot = max(st["hot_ops"] for st in stages)
    cons = max(st["consolidation_ops"] for st in stages)
    bytes_seen = {st["persistent_bytes"] for st in stages}
    interface = consolidation_interface()["K1"]
    return {
        "acquisition": all(r["LEARNED"][s] for r in k1 for s in STAGES[1:]),
        "retention": all(all(r["RETAINED"].values()) for r in k1),
        "no_catastrophe": not any(any(r["CATASTROPHIC"].values()) for r in k1),
        "joint_quality": _median([r["final_mean_nmse"] for r in k1]) <= th["joint_quality_median_final_mean_nmse"],
        "state_removal": (sum(r["SEQUENCE_HELD"] for r in removed) <= th["ablation_max_sequence_held_fraction"] * n
                          and _median([r["final_earlier_mean_nmse"] for r in removed])
                          >= th["ablation_min_median_factor"] * _median([r["final_earlier_mean_nmse"] for r in k1])),
        "reset_challenge": (sum(r["RESET_DEGRADED"] for r in reset) >= th["reset_min_degraded_fraction"] * n
                            and _median([r["reset_ratio"] for r in reset]) >= th["reset_min_median_ratio"]),
        "no_reteaching": all(set(r["stages"]) == {"C"} for r in reset),
        "transplant": all(t["continuation_bitwise"] and t["restored_digest_equal"] for t in transplant),
        "w_only_negative_control": all(w["pass"] for w in w_only),
        "native_budget": (hot <= th["native_hot_path_max_g1_multiple"] * g1
                          and cons <= th["native_consolidation_max_ops"]
                          and bytes_seen == {th["native_persistent_extra_bytes"]}
                          and interface["closed_form"] and not interface["receives_targets"]),
        "_evidence": {"g1_ops_per_step": g1, "max_hot_ops_per_step": hot, "max_consolidation_ops": cons,
                      "persistent_extra_bytes": sorted(bytes_seen)},
    }


def k1_dev_rule(rows: dict, th: dict, spec: dict) -> dict:
    criteria = k1_criteria([rows[w] for w in rows], th, spec)
    decisive = {k: v for k, v in criteria.items() if not k.startswith("_")}
    return {"criteria": decisive, "evidence": criteria["_evidence"], "proceeds": all(decisive.values())}


# --- QUAL --------------------------------------------------------------------------------------------------------


def qual_world(lab: Lab, wid: str) -> dict:
    """Every registered measurement of one QUAL world."""
    world = lab.world(wid)
    P = k1_world(lab, world)
    P["mechanisms"]["M0"] = strip_private(lab.run_mechanism(world, "M0"))
    P["mechanisms"]["M1"] = strip_private(lab.run_mechanism(world, "M1"))
    P["ceilings"] = lab.ceilings(world)
    P["witness"] = lab.witness(world)
    P["mechanics"] = {"feature_identity_rel": lab.feature_identity(world), "pte": lab.pte_check(world),
                      "W_A_refused": lab.w_a_refused(world)}
    if wid.endswith("-0000"):
        P["mechanics"]["gradient_identity"] = lab.gradient_identity(world)
    return {"numerics": {"core": numerics.blas_core(), "threads": numerics.blas_threads()}, "P": P}


def determinism_digest(lab_factory, wid: str) -> dict:
    """K1's pipeline on one world from the specification alone (a fresh Lab, no caches)."""
    lab = lab_factory()
    rec = lab.run_mechanism(lab.world(wid), "K1")
    return {"stages": {s: {k: rec["stages"][s][k] for k in ("W_digest", "state_digest", "nmse")} for s in STAGES}}


def gates(per_world: dict, spec: dict, determinism: dict, transplant_probe: dict) -> dict:
    """Validity, mechanics, gates A-K, the decision record (without gate L) and the disposition."""
    th = spec["thresholds"]
    worlds = list(per_world)
    n = len(worlds)
    P = lambda w: per_world[w]["P"]  # noqa: E731
    mech = lambda name: [P(w)["mechanisms"][name] for w in worlds]  # noqa: E731
    k1, m0, m1 = mech("K1"), mech("M0"), mech("M1")
    crit = k1_criteria([P(w) for w in worlds], th, spec)
    interface = consolidation_interface()

    validity = {
        "V2_forgetting": sum(r["SEQUENCE_HELD"] for r in m0) <= th["baseline_max_sequence_held_fraction"] * n
        and _median([r["final_earlier_mean_nmse"] for r in m0]) >= th["baseline_min_median_final_earlier_nmse"]
        and _median([r["A_end_ratio"] for r in m0]) >= th["baseline_min_median_A_end_ratio"],
        "V3_novelty": sum(P(w)["ceilings"]["previous"][s] >= th["novelty_min_nmse"]
                          for w in worlds for s in STAGES[1:]) >= th["qual_validity_min_fraction"] * 3 * n,
        "V5_rehearsal_feasibility": sum(r["SEQUENCE_HELD"] for r in m1) >= th["qual_validity_min_fraction"] * n,
        "V6_capacity": all(P(w)["ceilings"]["capacity_max"] <= th["capacity_max_ceiling_nmse"] for w in worlds),
    }
    gi = P(worlds[0])["mechanics"].get("gradient_identity", {"pass": False})
    mechanics = {
        "feature_identity": all(P(w)["mechanics"]["feature_identity_rel"] <= th["feature_identity_relative"]
                                for w in worlds),
        "witness_exact": all(max(P(w)["witness"].values()) <= th["witness_max_nmse"] for w in worlds),
        "engine_equivalence": all(P(w)["removed"][k]["stages"][s]["W_digest"]
                                  == P(w)["mechanisms"]["M0"]["stages"][s]["W_digest"]
                                  for w in worlds for k in ENGINE_KINDS for s in STAGES),
        "shared_W_A": all(P(w)["mechanisms"][k]["W_A_digest"] == P(w)["mechanisms"]["M0"]["stages"]["A"]["W_digest"]
                          for w in worlds for k in ENGINE_KINDS),
        "no_baseline_refusal": all(not any(st["refused"] for st in P(w)["mechanisms"][k]["stages"].values())
                                   and P(w)["mechanics"]["W_A_refused"] == 0 for w in worlds for k in ("M0", "M1")),
        "consolidation_interface": all(v["closed_form"] and not v["receives_targets"] for v in interface.values()),
        "gradient_identity": bool(gi["pass"]),
        "pte_microstate": all(P(w)["mechanics"]["pte"]["pass"] for w in worlds),
        "query_ignores_consolidation_state": all(P(w)["state_semantics"]["query_identical"] for w in worlds),
        "consolidation_state_shapes_learning": all(P(w)["state_semantics"]["learning_differs"] for w in worlds),
        "reset_mechanics": all(P(w)["causality"]["reset_challenge"]["reset_mechanics"] for w in worlds),
        "effective_single_thread": all(per_world[w]["numerics"]["threads"] == 1 for w in worlds),
    }
    k1_stages = [st for r in k1 for st in r["stages"].values()]
    bytes_fixed = all(len({st["persistent_bytes"] for st in r["stages"].values()}) == 1 for r in k1)
    g = {
        "A_current_stage_acquisition": all(r["LEARNED"]["A"] for r in m0)
        and _median([r["stages"]["A"]["nmse"]["A"] for r in m0]) <= th["first_acquisition_median_nmse"]
        and crit["acquisition"],
        "B_sequential_retention": crit["retention"],
        "C_no_catastrophic_forgetting": crit["no_catastrophe"],
        "D_joint_quality": crit["joint_quality"],
        "E_single_joint_state": all(st.get("W_digest") for st in k1_stages) and not interface["K1"]["receives_targets"],
        "F_declared_state_causality": crit["state_removal"] and crit["reset_challenge"] and crit["no_reteaching"],
        "G_no_external_answer_store": (not interface["K1"]["receives_targets"]) and bytes_fixed
        and interface["K1"]["closed_form"],
        "H_deterministic_state": bool(determinism.get("identical")),
        "I_transplant_and_reset_behaviour": crit["transplant"] and bool(transplant_probe.get("bitwise"))
        and crit["w_only_negative_control"],
        "J_capacity_accounting": all("persistent_bytes" in st and "hot_ops" in st and "consolidation_ops" in st
                                     for k in ENGINE_KINDS for r in mech(k) for st in r["stages"].values())
        and all("stored_experience_bytes_final" in r for r in m1)
        and {st["persistent_bytes"] for st in k1_stages} == {th["native_persistent_extra_bytes"]},
        "K_native_feasibility": crit["native_budget"],
    }
    medians = {name: {"final_mean_nmse": _median([r["final_mean_nmse"] for r in mech(name)]),
                      "final_earlier_mean_nmse": _median([r["final_earlier_mean_nmse"] for r in mech(name)]),
                      "A_end_ratio": _median([r["A_end_ratio"] for r in mech(name)])}
               for name in ("M0", "M1", "C1R", "K1")}
    medians["state_removed"] = {"final_earlier_mean_nmse": _median([P(w)["causality"]["state_removed"][
        "final_earlier_mean_nmse"] for w in worlds])}
    reset = [P(w)["causality"]["reset_challenge"] for w in worlds]
    medians["reset_challenge"] = {"reset_ratio": _median([r["reset_ratio"] for r in reset]),
                                  "e_reset": _median([r["e_reset"] for r in reset]),
                                  "e_uninterrupted": _median([r["e_uninterrupted"] for r in reset])}
    all_counts = {name: counts(mech(name)) for name in ("M0", "M1", "C1R", "K1")}
    all_counts["state_removed"] = counts([P(w)["causality"]["state_removed"] for w in worlds])
    all_counts["reset_challenge"] = {"RESET_DEGRADED": sum(r["RESET_DEGRADED"] for r in reset),
                                     "both_retained_at_C": sum(r["both_retained_at_C"] for r in reset)}
    all_counts["full_state_transplant_at_B"] = {"bitwise": sum(
        P(w)["causality"]["full_state_transplant_at_B"]["continuation_bitwise"] for w in worlds)}
    all_counts["w_only_negative_control_at_B"] = {"pass": sum(
        P(w)["causality"]["w_only_negative_control_at_B"]["pass"] for w in worlds)}
    out = {"validity": validity, "mechanics": mechanics, "gates": g, "medians": medians, "counts": all_counts,
           "native_feasibility_evidence": crit["_evidence"], "consolidation_interface": interface,
           **disposition(validity, mechanics, g, k1, m0, m1, th)}
    out["decision_record"] = decision_record(per_world, out)
    return out


def decision_record(per_world: dict, verdict: dict) -> dict:
    """Every decision-bearing R3 quantity (spec pass_rule.decision_record); nothing descriptive."""
    worlds = list(per_world)
    P = lambda w: per_world[w]["P"]  # noqa: E731
    k1 = {w: {key: P(w)["mechanisms"]["K1"][key] for key in ("LEARNED", "RETAINED", "CATASTROPHIC", "HELD",
                                                             "SEQUENCE_HELD")} for w in worlds}
    causal = {w: {"state_removed_SEQUENCE_HELD": P(w)["causality"]["state_removed"]["SEQUENCE_HELD"],
                  "RESET_DEGRADED": P(w)["causality"]["reset_challenge"]["RESET_DEGRADED"],
                  "transplant_bitwise": P(w)["causality"]["full_state_transplant_at_B"]["continuation_bitwise"],
                  "w_only_control_pass": P(w)["causality"]["w_only_negative_control_at_B"]["pass"]} for w in worlds}
    m0 = [P(w)["mechanisms"]["M0"] for w in worlds]
    m1 = [P(w)["mechanisms"]["M1"] for w in worlds]
    return {"validity": verdict["validity"], "mechanics": verdict["mechanics"], "gates": verdict["gates"],
            "disposition": verdict["disposition"], "outcome": verdict["outcome"], "K1": k1, "causality": causal,
            "controls": {"M0_SEQUENCE_HELD": sum(r["SEQUENCE_HELD"] for r in m0),
                         "M0_LEARNED_every_stage": sum(all(r["LEARNED"].values()) for r in m0),
                         "M1_SEQUENCE_HELD": sum(r["SEQUENCE_HELD"] for r in m1)}}


def disposition(validity, mechanics, g, k1, m0, m1, th) -> dict:
    """The registered disposition (spec pass_rule.disposition) over gates A-K (and L once known)."""
    if not all(mechanics.values()):
        return {"disposition": "MECHANICS_FAIL", "outcome": None}
    if not all(validity.values()):
        return {"disposition": "TASK_INVALID_UNDER_QUAL", "outcome": "OUTCOME_V"}
    if all(g.values()):
        return {"disposition": "RETENTION_SUPPORTED_UNDER_FROZEN_SYNTHETIC_REGIME", "outcome": "OUTCOME_A"}
    k1_held, m0_held = sum(r["SEQUENCE_HELD"] for r in k1), sum(r["SEQUENCE_HELD"] for r in m0)
    if k1_held > m0_held and _median([r["final_earlier_mean_nmse"] for r in k1]) \
            <= th["partial_reduction_max_median_factor"] * _median([r["final_earlier_mean_nmse"] for r in m0]):
        return {"disposition": "PARTIAL_REDUCTION", "outcome": "OUTCOME_C"}
    if all(r["SEQUENCE_HELD"] for r in m1):
        return {"disposition": "REPLAY_ONLY", "outcome": "OUTCOME_B"}
    return {"disposition": "NO_MATERIAL_IMPROVEMENT", "outcome": "OUTCOME_D"}


def qualifiers(per_world: dict, th: dict, robust_ok: bool) -> dict:
    worlds = list(per_world)
    rows = lambda k: [per_world[w]["P"]["mechanisms"][k] for w in worlds]  # noqa: E731
    c1r, k1 = rows("C1R"), rows("K1")
    held = lambda rs: sum(r["SEQUENCE_HELD"] for r in rs)  # noqa: E731
    med = lambda rs, key: _median([r[key] for r in rs])  # noqa: E731
    c1r_ok = (all(r["LEARNED"][x] for r in c1r for x in STAGES[1:]) and all(all(r["RETAINED"].values()) for r in c1r)
              and not any(any(r["CATASTROPHIC"].values()) for r in c1r)
              and med(c1r, "final_mean_nmse") <= th["joint_quality_median_final_mean_nmse"])
    exceeds = held(k1) > held(c1r) or (held(k1) == held(c1r) and med(k1, "final_mean_nmse") < med(c1r, "final_mean_nmse"))
    return {"R0_REFERENCE_ALSO_SUFFICIENT": c1r_ok, "EXCEEDS_R0_REFERENCE": exceeds,
            "NUMERICALLY_FRAGILE": not robust_ok, "CAPACITY_SCOPE": True}
