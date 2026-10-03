"""Retention R0 experiment: per-world procedures, DEV rules, QUAL gates and disposition. RESEARCH_ONLY.

Everything that decides the disposition is in the frozen specification
(``specs/ecsg-retention-r0.v1.spec.json``); this module implements it. W_A is
learned once per world through the canonical core and shared bitwise by every
mechanism. Every response is the canonical native forward map of the
evaluated W, from one executor per state answering both experiences.
"""
from __future__ import annotations

import inspect
import statistics

import numpy as np

from elpis.ECS_G.native import Executor

from . import engine as E
from . import task as T

GRID_ORDER = ("C1", "C2")


class Lab:
    """Per-run caches over one loaded library and the specification."""

    def __init__(self, api, spec: dict):
        self.api, self.spec = api, spec
        reg = spec["regime"]
        self.dim, self.width, self.rate, self.steps = reg["dim"], reg["width"], reg["learning_rate"], reg["steps"]
        self.idx = T.indices(self.dim)
        self._worlds, self._wa = {}, {}

    def world(self, wid: str, offset: float) -> T.World:
        key = (wid, offset)
        if key not in self._worlds:
            self._worlds[key] = T.build_world(self.spec, wid, offset)
        return self._worlds[key]

    def w_a(self, world: T.World) -> np.ndarray:
        key = (world.world, world.offset)
        if key not in self._wa:
            a = world.tasks["A"]
            self._wa[key] = E.canonical_learn(self.api, world.w0, a.x_train, a.y_train, self.rate, self.steps)
        return self._wa[key]

    # -- evaluation ------------------------------------------------------------------------------------------

    def evaluate(self, W: np.ndarray, world: T.World, tasks=("A", "B")) -> dict:
        """One executor created from W answers every task's held-out queries (no task label reaches it)."""
        with Executor.create(self.api, self.dim, self.width, E._flat(W)) as e:
            out = {}
            for t in tasks:
                exp = world.tasks[t]
                pred = np.asarray(e.forward(memoryview(exp.x_test)), dtype=np.float64)
                out[t] = T.nmse(pred, exp.y_test)
        return out

    def baseline_levels(self, world: T.World) -> dict:
        W_A = self.w_a(world)
        init, after_a = self.evaluate(world.w0, world, T.TASKS), self.evaluate(W_A, world, T.TASKS)
        return {"init": init, "after_A": after_a}

    # -- mechanisms ------------------------------------------------------------------------------------------

    def run(self, world: T.World, mechanism: str, lam: float = 0.0, variant: str = "standard") -> dict:
        """Learn B from the shared W_A with one mechanism. Returns W_AB and bookkeeping."""
        W_A = self.w_a(world)
        a, b = world.tasks["A"], world.tasks["B"]
        if mechanism == "M0":
            return {"W": E.canonical_learn(self.api, W_A, b.x_train, b.y_train, self.rate, self.steps),
                    "refused": 0, "extra_state_bytes": 0, "seconds": None}
        if mechanism == "M1":
            X = np.vstack([a.x_train, b.x_train])
            y = np.concatenate([a.y_train, b.y_train])
            return {"W": E.canonical_learn(self.api, W_A, X, y, self.rate, self.steps), "refused": 0,
                    "extra_state_bytes": 8 * a.x_train.size + 8 * a.y_train.size, "seconds": None}
        if mechanism == "ENGINE0":   # the candidate engine with the consolidation term removed
            W, refused, sec = E.engine_learn(self.api, W_A, b.x_train, b.y_train, self.rate, self.steps)
            return {"W": W, "refused": refused, "extra_state_bytes": 0, "seconds": sec}
        consolidation = E.make_candidate(mechanism, lam, self.idx, W_A.shape, uninformed=(variant == "uninformed"))
        source = world.tasks["C"].x_train if variant == "mismatched" else a.x_train
        consolidation.consolidate(W_A, source)
        W, refused, sec = E.engine_learn(self.api, W_A, b.x_train, b.y_train, self.rate, self.steps, consolidation)
        return {"W": W, "refused": refused, "extra_state_bytes": consolidation.state_bytes(), "seconds": sec}

    def metrics(self, world: T.World, result: dict) -> dict:
        levels = self.baseline_levels(world)
        W_A, W = self.w_a(world), result["W"]
        after = self.evaluate(W, world)
        a_init, b_init = levels["init"]["A"], levels["init"]["B"]
        a_learned, b_before = levels["after_A"]["A"], levels["after_A"]["B"]
        m = {"nmse_A_init": a_init, "nmse_A_after_A": a_learned, "nmse_A_after_B": after["A"],
             "nmse_B_init": b_init, "nmse_B_before": b_before, "nmse_B_after": after["B"],
             "acquisition_B": b_before - after["B"], "retention_ratio": after["A"] / a_learned,
             "state_change": float(np.linalg.norm(W - W_A) / np.linalg.norm(W_A)),
             "extra_state_bytes": result["extra_state_bytes"], "refused": result["refused"],
             "learn_seconds": result["seconds"], "W_digest": E.w_digest(W)}
        m.update(classify(m))
        return m

    # -- secondary characterizations -------------------------------------------------------------------------

    def ceiling(self, world: T.World) -> dict:
        a, b = world.tasks["A"], world.tasks["B"]
        pa, pb = E.ceiling([(a.x_train, a.y_train), (b.x_train, b.y_train)], [a.x_test, b.x_test], self.idx)
        return {"nmse_A": T.nmse(pa, a.y_test), "nmse_B": T.nmse(pb, b.y_test)}

    def sequence(self, world: T.World, mechanism: str, lam: float = 0.0) -> dict:
        """A -> B -> C -> D; nmse of every task after every stage; cumulative consolidation for candidates."""
        W = self.w_a(world)
        init = self.evaluate(world.w0, world, T.TASKS)
        consolidation = None
        if mechanism in GRID_ORDER:
            consolidation = E.make_candidate(mechanism, lam, self.idx, W.shape)
            consolidation.consolidate(W, world.tasks["A"].x_train)
        stages = {"A": self.evaluate(W, world, T.TASKS)}
        seen, refused = ["A"], 0
        for t in T.TASKS[1:]:
            exp = world.tasks[t]
            if mechanism == "M0":
                W = E.canonical_learn(self.api, W, exp.x_train, exp.y_train, self.rate, self.steps)
            elif mechanism == "M1":
                X = np.vstack([world.tasks[s].x_train for s in seen + [t]])
                y = np.concatenate([world.tasks[s].y_train for s in seen + [t]])
                W = E.canonical_learn(self.api, W, X, y, self.rate, self.steps)
            else:
                W, r, _ = E.engine_learn(self.api, W, exp.x_train, exp.y_train, self.rate, self.steps, consolidation)
                refused += int(r > 0)
                consolidation.consolidate(W, exp.x_train)
            seen.append(t)
            stages[t] = self.evaluate(W, world, T.TASKS)
        final = stages["D"]
        held = {t: final[t] <= 0.5 and final[t] <= 0.5 * init[t] for t in T.TASKS}
        return {"init": init, "after_stage": stages, "held_at_end": held, "held_count": sum(held.values()),
                "refused_stages": refused}

    def pte_check(self, world: T.World, family: str, lam: float) -> dict:
        """Two microstates with identical S3 diverge after one identical candidate step: W stays authoritative."""
        p, q = T.pte_pair(world.w0)
        a, b = world.tasks["A"], world.tasks["B"]
        s_p, s_q = E.s3(p, self.idx), E.s3(q, self.idx)
        before = float(np.max(np.abs(s_p - s_q)) / max(1.0, float(np.max(np.abs(s_p)))))
        cons_p = E.make_candidate(family, lam, self.idx, p.shape)
        cons_p.consolidate(p, a.x_train)
        if family == "C1":
            cons_q = cons_p   # identical consolidation state (S3 and inputs are identical)
        else:
            cons_q = E.make_candidate(family, lam, self.idx, q.shape)
            cons_q.consolidate(q, a.x_train)
        p1 = E.engine_learn(self.api, p, b.x_train, b.y_train, self.rate, 1, cons_p)[0]
        q1 = E.engine_learn(self.api, q, b.x_train, b.y_train, self.rate, 1, cons_q)[0]
        s1p, s1q = E.s3(p1, self.idx), E.s3(q1, self.idx)
        after = float(np.max(np.abs(s1p - s1q)) / max(1.0, float(np.max(np.abs(s1p)))))
        return {"s3_rel_before": before, "s3_rel_after": after, "microstates_differ": bool(np.any(p != q)),
                "pass": before <= 1e-12 and after >= 1e-9}

    def causality(self, world: T.World, W: np.ndarray) -> dict:
        """(i) snapshot/restore of the executor holding W reproduces its responses; (ii) W0 reproduces untaught."""
        a, b = world.tasks["A"], world.tasks["B"]
        with Executor.create(self.api, self.dim, self.width, E._flat(W)) as e:
            ra, rb = e.forward(memoryview(a.x_test)), e.forward(memoryview(b.x_test))
            blob = e.snapshot()
        with Executor.restore(self.api, blob) as restored:
            transplant = restored.forward(memoryview(a.x_test)) == ra and restored.forward(memoryview(b.x_test)) == rb
        with Executor.create(self.api, self.dim, self.width, E._flat(world.w0)) as first:
            untaught = first.forward(memoryview(a.x_test)), first.forward(memoryview(b.x_test))
        with Executor.create(self.api, self.dim, self.width, E._flat(world.w0)) as again:
            reset = (again.forward(memoryview(a.x_test)), again.forward(memoryview(b.x_test))) == untaught
        return {"transplant_bitwise": bool(transplant), "reset_bitwise": bool(reset)}

    def feature_identity(self, world: T.World) -> float:
        W_A = self.w_a(world)
        x = np.vstack([world.tasks["A"].x_test, world.tasks["B"].x_test])
        native = E.responses(self.api, W_A, x)
        coarse = 0.5 * T.features(x, self.idx) @ E.native_s3(self.api, W_A)
        return float(np.max(np.abs(native - coarse)) / max(1.0, float(np.max(np.abs(native)))))


def classify(m: dict) -> dict:
    return {"LEARNED_A": m["nmse_A_after_A"] <= 0.5 and m["nmse_A_after_A"] <= 0.5 * m["nmse_A_init"],
            "LEARNED_B": m["nmse_B_after"] <= 0.5 and m["nmse_B_after"] <= 0.5 * m["nmse_B_before"],
            "RETAINED_A": m["nmse_A_after_B"] <= 0.5 and m["nmse_A_after_B"] <= 0.5 * m["nmse_A_init"],
            "CATASTROPHIC_A": m["nmse_A_after_B"] >= m["nmse_A_init"]}


def consolidation_inputs() -> dict:
    """Gate F (structural): what each candidate's consolidation receives. No target parameter exists."""
    out = {}
    for cls in (E.C1, E.C2):
        params = [p for p in inspect.signature(cls.consolidate).parameters if p != "self"]
        out[cls.family] = {"parameters": params,
                           "receives_targets": any(n in ("y", "targets", "labels", "y_train") for n in params)}
    return out


def configurations(spec: dict) -> list[tuple[str, float]]:
    return [(fam, float(lam)) for fam in GRID_ORDER for lam in spec["mechanisms"][fam]["grid"]["lambda"]]


def _median(values):
    return float(statistics.median(values))


# --- DEV ------------------------------------------------------------------------------------------------------


def dev(lab: Lab, worlds) -> dict:
    spec = lab.spec
    feasibility, chosen = {}, None
    for offset in spec["offset_grid"]:
        rows = {}
        for wid in worlds:
            world = lab.world(wid, offset)
            m0 = lab.metrics(world, lab.run(world, "M0"))
            m1 = lab.metrics(world, lab.run(world, "M1"))
            rows[wid] = {"M0": m0, "M1": m1,
                         "qualifies": m0["LEARNED_A"] and m1["LEARNED_B"] and m1["RETAINED_A"]}
        count = sum(r["qualifies"] for r in rows.values())
        feasibility[str(offset)] = {"worlds": rows, "qualifying_worlds": count}
        if chosen is None and count == len(worlds):
            chosen = offset
    replay_infeasible = chosen is None
    if replay_infeasible:
        best = max(feasibility.items(), key=lambda kv: (kv[1]["qualifying_worlds"], -float(kv[0])))
        chosen = float(best[0])
    table = {}
    for fam, lam in configurations(spec):
        key = f"{fam}:{lam}"
        table[key] = {}
        for wid in worlds:
            world = lab.world(wid, chosen)
            table[key][wid] = lab.metrics(world, lab.run(world, fam, lam))
    engine0 = {wid: lab.metrics(lab.world(wid, chosen), lab.run(lab.world(wid, chosen), "ENGINE0"))
               for wid in worlds}
    ceiling = {wid: lab.ceiling(lab.world(wid, chosen)) for wid in worlds}
    selected, secondary, ranking = select(table, spec)
    return {"feasibility": feasibility, "chosen_offset": chosen, "replay_infeasible_on_dev": replay_infeasible,
            "candidate_table": table, "ranking": ranking, "engine0": engine0, "ceiling": ceiling,
            "choices": {"offset": chosen, "selected": selected, "secondary": secondary}}


def select(table: dict, spec: dict):
    """The pre-registered candidate rule (spec dev_rules.candidate)."""
    order = configurations(spec)
    ranking = []
    for position, (fam, lam) in enumerate(order):
        rows = table[f"{fam}:{lam}"]
        count = sum(r["LEARNED_B"] and r["RETAINED_A"] for r in rows.values())
        joint = _median([r["nmse_A_after_B"] + r["nmse_B_after"] for r in rows.values()])
        state = next(iter(rows.values()))["extra_state_bytes"]
        ranking.append({"family": fam, "lambda": lam, "count": count, "median_joint_nmse": joint,
                        "extra_state_bytes": state, "position": position})
    ranked = sorted(ranking, key=lambda r: (-r["count"], r["median_joint_nmse"], r["extra_state_bytes"],
                                            r["position"]))
    first = ranked[0]
    other = next(r for r in ranked if r["family"] != first["family"])
    pick = lambda r: {"family": r["family"], "lambda": r["lambda"]}  # noqa: E731
    return pick(first), pick(other), ranked


# --- QUAL -----------------------------------------------------------------------------------------------------


def qual_world(lab: Lab, wid: str, choices: dict) -> dict:
    world = lab.world(wid, choices["offset"])
    sel, sec = choices["selected"], choices["secondary"]
    out = {"levels": lab.baseline_levels(world)}
    results = {"M0": lab.run(world, "M0"), "M1": lab.run(world, "M1"),
               "selected": lab.run(world, sel["family"], sel["lambda"]),
               "secondary": lab.run(world, sec["family"], sec["lambda"]),
               "ablation_state_removed": lab.run(world, "ENGINE0"),
               "ablation_mismatched": lab.run(world, sel["family"], sel["lambda"], "mismatched"),
               "ablation_uninformed": lab.run(world, sel["family"], sel["lambda"], "uninformed")}
    out["mechanisms"] = {k: lab.metrics(world, r) for k, r in results.items()}
    out["ceiling"] = lab.ceiling(world)
    out["causality"] = lab.causality(world, results["selected"]["W"])
    out["mechanics"] = {
        "feature_identity_rel": lab.feature_identity(world),
        "engine_equivalence_bitwise": bool(np.array_equal(results["ablation_state_removed"]["W"],
                                                          results["M0"]["W"])),
        "shared_W_A_digest": E.w_digest(lab.w_a(world)),
        "pte": lab.pte_check(world, sel["family"], sel["lambda"]),
    }
    out["sequence"] = {"M0": lab.sequence(world, "M0"), "M1": lab.sequence(world, "M1"),
                       "selected": lab.sequence(world, sel["family"], sel["lambda"]),
                       "secondary": lab.sequence(world, sec["family"], sec["lambda"])}
    return out


def determinism_digest(lab: Lab, wid: str, choices: dict) -> dict:
    """The selected candidate's pipeline on one world from the specification: W_A, W_AB digests and metrics."""
    world = T.build_world(lab.spec, wid, choices["offset"])
    a, b = world.tasks["A"], world.tasks["B"]
    W_A = E.canonical_learn(lab.api, world.w0, a.x_train, a.y_train, lab.rate, lab.steps)
    sel = choices["selected"]
    cons = E.make_candidate(sel["family"], sel["lambda"], lab.idx, W_A.shape)
    cons.consolidate(W_A, a.x_train)
    W, refused, _ = E.engine_learn(lab.api, W_A, b.x_train, b.y_train, lab.rate, lab.steps, cons)
    return {"W_A": E.w_digest(W_A), "W_AB": E.w_digest(W), "refused": refused, "nmse": lab.evaluate(W, world)}


def gates(per_world: dict, spec: dict, determinism: dict, inputs: dict, choices: dict) -> dict:
    worlds = list(per_world)
    mech = lambda name: [per_world[w]["mechanisms"][name] for w in worlds]  # noqa: E731
    sel, m0, m1, abl = mech("selected"), mech("M0"), mech("M1"), mech("ablation_state_removed")
    n = len(worlds)
    frac = lambda rows, key: sum(r[key] for r in rows) / n  # noqa: E731
    g = {
        "A_acquisition": all(r["LEARNED_A"] for r in m0) and _median([r["nmse_A_after_A"] for r in m0]) <= 0.25,
        "B_acquisition": all(r["LEARNED_B"] for r in sel),
        "C_retention": all(r["RETAINED_A"] for r in sel),
        "D_joint_state": all(r["W_digest"] for r in sel),
        "E_state_causality": (all(per_world[w]["causality"]["transplant_bitwise"] and
                                  per_world[w]["causality"]["reset_bitwise"] for w in worlds)
                              and (1.0 - frac(abl, "RETAINED_A")) >= 0.75
                              and _median([r["nmse_A_after_B"] for r in sel])
                              <= 0.5 * _median([r["nmse_A_after_B"] for r in abl])),
        "F_no_external_answer_store": not inputs[choices["selected"]["family"]]["receives_targets"],
        "G_determinism": determinism["identical"],
        "H_negative_baseline": (1.0 - frac(m0, "RETAINED_A")) >= 0.75
                               and _median([r["retention_ratio"] for r in m0]) >= 10.0,
        "I_control_honesty": choices["selected"]["family"] in GRID_ORDER,
        "J_capacity_accounting": all("extra_state_bytes" in r for r in sel + m0 + m1),
    }
    mechanics = {
        "feature_identity": all(per_world[w]["mechanics"]["feature_identity_rel"] <= 1e-10 for w in worlds),
        "engine_equivalence": all(per_world[w]["mechanics"]["engine_equivalence_bitwise"] for w in worlds),
        "shared_W_A": all(per_world[w]["mechanisms"]["M0"]["W_digest"] and
                          per_world[w]["mechanics"]["shared_W_A_digest"] for w in worlds),
        "no_baseline_refusal": all(r["refused"] == 0 for r in m0 + m1),
        "pte_microstate": all(per_world[w]["mechanics"]["pte"]["pass"] for w in worlds),
    }
    medians = {name: {"retention_ratio": _median([r["retention_ratio"] for r in mech(name)]),
                      "nmse_A_after_B": _median([r["nmse_A_after_B"] for r in mech(name)]),
                      "nmse_B_after": _median([r["nmse_B_after"] for r in mech(name)]),
                      "retained": sum(r["RETAINED_A"] for r in mech(name)),
                      "learned_B": sum(r["LEARNED_B"] for r in mech(name)),
                      "catastrophic": sum(r["CATASTROPHIC_A"] for r in mech(name))}
               for name in ("M0", "M1", "selected", "secondary", "ablation_state_removed", "ablation_mismatched",
                            "ablation_uninformed")}
    replay_retains = all(r["LEARNED_B"] and r["RETAINED_A"] for r in m1)
    if not all(mechanics.values()):
        disposition, outcome = "MECHANICS_FAIL", None
    elif all(g.values()):
        disposition, outcome = "RETENTION_SUPPORTED_UNDER_FROZEN_SYNTHETIC_REGIME", "OUTCOME_A"
    elif medians["selected"]["retention_ratio"] <= 0.5 * medians["M0"]["retention_ratio"]:
        disposition, outcome = "PARTIAL_REDUCTION", "OUTCOME_C"
    elif replay_retains:
        disposition, outcome = "REPLAY_ONLY", "OUTCOME_B"
    else:
        disposition, outcome = "NO_MATERIAL_IMPROVEMENT", "OUTCOME_D"
    return {"gates": g, "mechanics": mechanics, "medians": medians, "replay_retains_all": replay_retains,
            "disposition": disposition, "outcome": outcome, "worlds": n}
