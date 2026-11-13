"""Shared fixtures for the R120 tests, qualification and benchmarks (numpy only)."""
from __future__ import annotations

import numpy as np

from research.yang_mills_hca.r118_scalable_awr.awr_state import StateCapacity
from research.yang_mills_hca.r118_scalable_awr.fixtures import synthetic_registry
from research.yang_mills_hca.r118_scalable_awr.tests import _util as U118

from . import engine_r120 as E

CHECKPOINT = "54007/STATE8_SIGNED8"


def small_state(seed=5, participants=60, relations=200, windows=80, arity=(2, 6), zipf=0.5, hub=0.0, registry=None,
                state_seed=11, scale=1.0):
    """A small sparse system with overlapping 4-slot windows and random finite states."""
    reg = registry if registry is not None else synthetic_registry(participants, relations, seed=seed, arity=arity,
                                                                   zipf=zipf, hub_fraction=hub)
    rel, mask = U118.random_windows(reg, windows, seed=seed + 4)
    st = E.IncrementalState(reg, StateCapacity(max_instances=windows + 16, max_materialized_relations=4 * windows + 64,
                                               max_bindings_per_relation=256))
    handles, gens = st.create_instances(np.arange(windows, dtype=np.uint64) + 1, rel, mask)
    U118.seed_states(st, state_seed, scale=scale)
    return reg, st, handles, gens, (rel, mask)


def clone_state(reg, st, windows):
    """A fresh IncrementalState with the same windows and the same pool contents (other registry allowed)."""
    rel, mask = windows
    n = rel.shape[0]
    out = E.IncrementalState(reg, StateCapacity(max_instances=st.capacity.max_instances,
                                                max_materialized_relations=st.capacity.max_materialized_relations,
                                                max_bindings_per_relation=st.capacity.max_bindings_per_relation))
    out.create_instances(np.arange(n, dtype=np.uint64) + 1, rel, mask)
    rows = np.nonzero(st.row_ref > 0)[0]
    for row in rows.tolist():
        rid = int(st.registry.relation_ids[st.row_rel[row]])
        out.pool[out.rel_row[int(reg.relation_index([rid])[0])]] = st.pool[row]
    return out


def random_batch(st, rng, n, repeat=True, scale=1.0):
    """A valid random event batch over materialized relations (sequence increasing per relation)."""
    reg = st.registry
    rows = np.nonzero(st.row_ref > 0)[0]
    pick = rng.choice(rows, size=n, replace=repeat)
    rel_idx = st.row_rel[pick]
    ids = reg.relation_ids[rel_idx].copy()
    parts = np.array([int(rng.choice(reg.members_of(int(r)))) for r in rel_idx], dtype=np.int32)
    obs = rng.normal(size=(n, 4)) * scale
    weight = rng.uniform(-1.0, 1.0, n)
    aux = rng.uniform(0.0, 1.0, n)
    seq = np.zeros(n, dtype=np.uint64)
    seen = {}
    for i, r in enumerate(pick.tolist()):
        seen[r] = seen.get(r, 0) + 1
        seq[i] = int(st.row_seq[r]) + seen[r]
    return ids, seq, parts, obs, weight, aux


def events_of(st, ids, seq, parts, obs, weight, aux):
    """The same batch as reference event dicts (participants by stable ID)."""
    reg = st.registry
    return [{"rel_id": int(ids[i]), "seq": int(seq[i]),
             "part": None if parts[i] < 0 else int(reg.participant_ids[parts[i]]),
             "obs": [float(x) for x in obs[i]], "weight": float(weight[i]), "aux": float(aux[i])}
            for i in range(len(ids))]


def model_of(st, eng):
    """Dict model of the live system and the engine parameters, for ``reference.apply_batch``."""
    relations, instances, state = U118.simple_model(st)
    seq = {int(st.registry.relation_ids[st.row_rel[row]]): int(st.row_seq[row]) for row in np.nonzero(st.row_ref > 0)[0]}
    params = {"W": eng.W.tolist(), "bias": eng.b.tolist(), "checkpoint": eng.checkpoint, "controller": eng.controller,
              "mode": eng.reconciliation, "cgc_w": eng.cgc_w, "eta": eng.eta, "gamma": eng.gamma, "rho_b": eng.rho_b,
              "kappa": eng.kappa}
    return {"relations": relations, "instances": instances, "state": state, "seq": seq}, params


def snapshot(st, eng=None):
    """Everything an update may touch, for atomicity checks."""
    snap = {"pool": st.pool.copy(), "row_seq": st.row_seq.copy(), "row_pending": st.row_pending.copy(),
            "row_gen": st.row_gen.copy(), "row_rel": st.row_rel.copy(), "rel_row": st.rel_row.copy(),
            "slot_rel": st.slot_rel.copy(), "slot_row": st.slot_row.copy(), "inst_gen": st.inst_gen.copy(),
            "live": st.live_count}
    if eng is not None:
        snap["pending"] = eng.pending_rows
    return snap


def unchanged(st, snap, eng=None) -> bool:
    ok = all(np.array_equal(getattr(st, k).view(np.uint8) if getattr(st, k).dtype == np.float64 else getattr(st, k),
                            snap[k].view(np.uint8) if snap[k].dtype == np.float64 else snap[k])
             for k in ("pool", "row_seq", "row_pending", "row_gen", "row_rel", "rel_row", "slot_rel", "slot_row", "inst_gen"))
    ok = ok and st.live_count == snap["live"] and bool(np.all(st.mark == -1)) and not st.inst_mark.any()
    if eng is not None:
        ok = ok and eng.pending_rows == snap["pending"]
    return bool(ok)


def affected_handles(st, row_list):
    """Instances bound to the given pool rows (through the R118 binding index)."""
    codes = st.binding_codes(np.asarray(row_list, dtype=np.int64))
    return np.unique(codes // 4).astype(np.int32)
