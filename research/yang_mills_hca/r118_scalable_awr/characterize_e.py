"""R118-E bounded scientific transfer characterization. Descriptive only; does NOT clear R93.

    python -B -m research.yang_mills_hca.r118_scalable_awr.characterize_e [--out PATH]

Each OOD R77 fixture window W0 (PARTICIPANT_NONFLAT frames; source relation in
slot 0, readout relation in slot 3) is embedded among k in {0, 2, 6} distractor
windows. Distractor j binds a fixed non-empty subset of W0's intermediate
relations {1, 2} plus new zero-state relations (random members over the same ten
participants, random SO(2)xSO(2) frames). Paired samples (historical
``pair_state`` rows i and i+4) live in disjoint copies of the composed system.

References (evaluation only, never inference inputs): the R30 teacher composed
identically (same windows, same MEAN reconciliation) and the isolated teacher
(k = 0, the historical target). Nothing is fitted or selected here; evaluation
and structure seeds are fresh and checked disjoint from every historical seed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from . import engine as E
from . import provenance as P
from . import reference_adapter as RA
from .awr_state import AWRState, StateCapacity
from .fixtures import historical_frames
from .gauge import random_so2x2
from .registry import build_registry, matrix_to_blocks

EVAL_SEEDS = (118601, 118603, 118609, 118611)
STRUCTURE_SEED = 118700
CASES = ("weak", "anti")
KS = (0, 2, 6)
HORIZONS = (8, 20)
CONFIGS = tuple((c, k) for k in P.CHECKPOINTS for c in ("NATIVE", "R114"))
COPIES = 8  # pair_state(batch=4): rows i and i + 4 form sample i
FD_EPS = 1e-7


def historical_seeds() -> set:
    seeds = set()

    def walk(o, key=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
        elif isinstance(o, int) and not isinstance(o, bool) and "seed" in key.lower():
            seeds.add(o)

    for path in sorted(P.SNAPSHOT_DIR.glob("FROZEN_PROTOCOL_R*.json")):
        walk(json.loads(path.read_bytes()))
    walk(json.loads((P.FROZEN_HOST / "source/research_helpers/r92/protocol.json").read_bytes()))
    return seeds


def structure(rng, kmax=max(KS)):
    spec = []
    for _ in range(kmax):
        sub = [[1], [2], [1, 2]][int(rng.integers(0, 3))]
        new = [sorted(rng.choice(10, size=4, replace=False).tolist()) for _ in range(4 - len(sub))]
        spec.append({"fixture_relations": sub, "new_members": new, "slot_perm": rng.permutation(4).tolist(),
                     "frames": random_so2x2(rng, 4 * len(new))})
    return spec


def composed_registry(o, graph, spec):
    mats = historical_frames(o, graph, "PARTICIPANT_NONFLAT")
    base_p, base_r = 7_100_000_000_000, 9_100_000_000_000
    rel_ids, offsets, mids, frames = [], [0], [], []
    for c in range(COPIES):
        for rid, members in enumerate(graph):
            rel_ids.append(base_r + c * 1000 + rid)
            for p in sorted(members):
                mids.append(base_p + p)
                frames.append(matrix_to_blocks(mats[(rid, p)].numpy()))
            offsets.append(len(mids))
        for j, d in enumerate(spec):
            for q, members in enumerate(d["new_members"]):
                rel_ids.append(base_r + c * 1000 + 100 + j * 10 + q)
                for t, p in enumerate(members):
                    mids.append(base_p + p)
                    frames.append(d["frames"][q * 4 + t])
                offsets.append(len(mids))
    reg = build_registry([base_p + p for p in range(10)], rel_ids, offsets, mids, np.array(frames))

    def windows(k):
        out = []
        for c in range(COPIES):
            out.append([base_r + c * 1000 + r for r in range(4)])
            for j in range(k):
                d = spec[j]
                rels = [base_r + c * 1000 + a for a in d["fixture_relations"]]
                rels += [base_r + c * 1000 + 100 + j * 10 + q for q in range(len(d["new_members"]))]
                slots = [None] * 4
                for s, r in zip(d["slot_perm"], rels):
                    slots[s] = r
                out.append(slots)
        return out

    return reg, windows, (lambda c, r: base_r + c * 1000 + r)


def build_state(reg, wins, fid, st0):
    n = len(wins)
    st = AWRState(reg, StateCapacity(max_instances=n, max_materialized_relations=4 * n, max_bindings_per_relation=64))
    mask = np.array([[r is not None for r in w] for w in wins])
    rel = np.array([[0 if r is None else r for r in w] for w in wins], dtype=np.uint64)
    h, _ = st.create_instances(np.arange(n, dtype=np.uint64) + 1, rel, mask)
    for c in range(COPIES):
        for d in range(4):
            st.set_relation_state(fid(c, d), np.concatenate([st0[0][c, d].numpy(), st0[1][c, d].numpy()]))
    return st, h


def readout(st, fid):
    a3 = np.array([st.relation_state(fid(c, 3))[:4] for c in range(COPIES)])
    return a3[:4] - a3[4:]


def teacher_rollout(o, reg, st, h, fid):
    teacher = o.r30.NonlinearTeacher(o.torch)
    cur = {int(r): st.pool[r].copy() for r in np.unique(st.slot_row[h][st.slot_row[h] >= 0])}
    out = {}
    for t in range(1, max(HORIZONS) + 1):
        cands = RA.oracle_candidates(reg, st.slot_rel[h], st.slot_row[h], cur, P.CHECKPOINTS[0], "NATIVE", law=teacher)
        cur = RA.reconcile_mean(cands)
        if t in HORIZONS:
            for r, v in cur.items():
                st.pool[r] = v
            out[t] = readout(st, fid)
    return out


def hca_rollout(st, h, fid, checkpoint, controller, budgets):
    eng = E.Engine(st, checkpoint, controller=controller, budgets=budgets)
    out, norm = {}, 0.0
    for t in range(1, max(HORIZONS) + 1):
        assert eng.step(h).exact
        rows = st.slot_row[h[:1]]  # instance 0 is copy 0's W0
        norm = max(norm, float(np.linalg.norm(st.pool[rows[rows >= 0]])))
        if t in HORIZONS:
            out[t] = readout(st, fid)
    return out, norm


def metrics(v, t):
    den = (t * t).sum(-1)
    err = ((v - t) ** 2).sum(-1)
    vv = (v * v).sum(-1)
    cos = (v * t).sum(-1) / (np.sqrt(vv * den) + 1e-30)
    return {"nse": err / den, "gain": (v * t).sum(-1) / den, "cos": cos, "energy_ratio": vv / den,
            "matched_nse": 2.0 * (1.0 - cos)}


def run() -> dict:
    t0 = time.time()
    from .frozen_oracle import get_oracle

    o = get_oracle()
    hist = historical_seeds()
    used = set(EVAL_SEEDS) | {STRUCTURE_SEED + i for i in range(4)}
    if used & hist:
        raise P.ProvenanceError("R118E_SEED_LEAK:" + str(sorted(used & hist)))
    budgets = E.Budgets(max_passes=20)
    acc = {}
    consistency = 0.0
    fd = {}
    for gi, gname in enumerate(sorted(o.r77.CFG["graphs"])):
        graph = o.r77.CFG["graphs"][gname]
        spec = structure(np.random.default_rng(STRUCTURE_SEED + gi))
        reg, windows, fid = composed_registry(o, graph, spec)
        for case in CASES:
            for seed in EVAL_SEEDS:
                st0 = o.r64.pair_state(seed, case, 160, batch=4)
                teach = {}
                for k in KS:
                    st, h = build_state(reg, windows(k), fid, st0)
                    teach[k] = teacher_rollout(o, reg, st, h, fid)
                # k = 0 composed teacher must equal the historical isolated teacher rollout
                _, adj = o.graph_adjs(graph)
                with o.torch.no_grad():
                    hr = o.r64.rollout(st0, adj, o.r30.NonlinearTeacher(o.torch), 20, keep=HORIZONS)
                for H in HORIZONS:
                    consistency = max(consistency, float(np.abs(teach[0][H] - o.r63.paired(hr[H][0], 4).numpy()).max()))
                for controller, ck in CONFIGS:
                    for k in KS:
                        st, h = build_state(reg, windows(k), fid, st0)
                        hca, norm = hca_rollout(st, h, fid, ck, controller, budgets)
                        for H in HORIZONS:
                            m_c = metrics(hca[H], teach[k][H])
                            m_i = metrics(hca[H], teach[0][H])
                            shift = metrics(teach[k][H], teach[0][H])["nse"]
                            key = (gname, k, controller, ck, H)
                            rec = acc.setdefault(key, {"nse_vs_composed_teacher": [], "nse_vs_isolated_teacher": [],
                                                       "matched_nse_vs_composed_teacher": [], "cos_vs_composed_teacher": [],
                                                       "energy_ratio_vs_composed_teacher": [], "gain_vs_composed_teacher": [],
                                                       "teacher_composition_shift_nse": [], "max_w0_norm": []})
                            rec["nse_vs_composed_teacher"] += m_c["nse"].tolist()
                            rec["nse_vs_isolated_teacher"] += m_i["nse"].tolist()
                            rec["matched_nse_vs_composed_teacher"] += m_c["matched_nse"].tolist()
                            rec["cos_vs_composed_teacher"] += m_c["cos"].tolist()
                            rec["energy_ratio_vs_composed_teacher"] += m_c["energy_ratio"].tolist()
                            rec["gain_vs_composed_teacher"] += m_c["gain"].tolist()
                            rec["teacher_composition_shift_nse"] += shift.tolist()
                            rec["max_w0_norm"].append(norm)
                # finite-difference amplification of a source perturbation (first seed, R114 SIGNED8)
                if seed == EVAL_SEEDS[0]:
                    for k in KS:
                        outs = []
                        for sign in (1, -1):
                            pert = (st0[0].clone(), st0[1].clone())
                            pert[0][:, 0, :] *= (1 + sign * FD_EPS)
                            st, h = build_state(reg, windows(k), fid, pert)
                            outs.append(hca_rollout(st, h, fid, "54007/STATE8_SIGNED8", "R114", budgets)[0][20])
                        din = 2 * FD_EPS * float(np.linalg.norm(st0[0][:, 0, :].numpy()))
                        amp = float(np.linalg.norm(outs[0] - outs[1])) / din
                        fd.setdefault(f"{gname}/k{k}", []).append(amp)
    summary = {}
    for (g, k, controller, ck, H), rec in sorted(acc.items()):
        summary[f"{g}/k{k}/{controller}/{ck}/H{H}"] = {name: float(np.mean(v)) for name, v in rec.items()} | {
            "n": len(rec["nse_vs_composed_teacher"])}
    pooled = {}
    for k in KS:
        for controller in ("NATIVE", "R114"):
            for H in HORIZONS:
                rows = [v for key, v in summary.items() if f"/k{k}/{controller}/" in key and key.endswith(f"/H{H}")]
                pooled[f"k{k}/{controller}/H{H}"] = {name: float(np.mean([r[name] for r in rows]))
                                                      for name in rows[0] if name != "n"}
    return {"phase": "R118-E", "schema": "elpis.ym_hca.r118e.v1", "scope": "DESCRIPTIVE_ONLY_NO_R93_CLEARANCE",
            "eval_seeds": EVAL_SEEDS, "structure_seed": STRUCTURE_SEED, "historical_seed_count": len(hist),
            "seed_leak": False, "k_values": KS, "horizons": HORIZONS,
            "teacher_k0_vs_historical_teacher_max_abs": consistency,
            "pooled": pooled, "by_graph": summary,
            "finite_difference_amplification_h20": {k: float(np.mean(v)) for k, v in fd.items()},
            "weights_unchanged": o.parameter_pins() == P.CHECKPOINT_SHA256,
            "environment": P.environment_fingerprint(),
            "component_digest": P.component_digest(), "component_sources": P.component_sources(), "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    res = run()
    blob = json.dumps(res, indent=1, sort_keys=True, default=P.json_default)
    if args.out:
        pathlib.Path(args.out).write_text(blob + "\n")
    print(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
