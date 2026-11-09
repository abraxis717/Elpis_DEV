"""Shared helpers for the R118 tests (no torch)."""
from __future__ import annotations

import numpy as np

from .. import engine as E
from .. import simple_reference as SR


def simple_model(state, handles=None):
    """Dict model of the live instances (or the given handles) for simple_reference."""
    reg = state.registry
    live = np.nonzero(state.inst_live)[0] if handles is None else np.unique(np.asarray(handles))
    relations, instances, st = {}, {}, {}
    for h in live.tolist():
        slots = []
        for d in range(4):
            r = int(state.slot_rel[h, d])
            if r < 0:
                slots.append(None)
                continue
            rid = int(reg.relation_ids[r])
            slots.append(rid)
            if rid not in relations:
                lo, hi = int(reg.rel_offsets[r]), int(reg.rel_offsets[r + 1])
                relations[rid] = {int(reg.participant_ids[reg.members[i]]): reg.frames[i].tolist() for i in range(lo, hi)}
                st[rid] = state.pool[state.rel_row[r]].tolist()
        instances[int(state.inst_id[h])] = slots
    return relations, instances, st


def compare_with_simple(eng, handles, passes=1):
    """Run the engine and the simple reference from the same state; return (result, max_abs, n_equal, n)."""
    state = eng.state
    reg = state.registry
    handles = np.unique(np.asarray(handles))
    relations, instances, st = simple_model(state)
    active = [int(state.inst_id[h]) for h in handles.tolist()]
    W, b = eng.W.tolist(), eng.b.tolist()
    ref = SR.step(W, b, relations, instances, st, active, eng.checkpoint, eng.controller, passes=passes,
                  mode=eng.reconciliation)
    res = eng.step(handles, passes=passes)
    diffs, equal, n = 0.0, 0, 0
    for rid, vec in ref.items():
        r = int(reg.relation_index([rid])[0])
        got = state.pool[state.rel_row[r]]
        for k in range(8):
            diffs = max(diffs, abs(vec[k] - got[k]))
            equal += vec[k] == got[k]
            n += 1
    return res, diffs, equal, n


def random_windows(reg, n, seed, max_incidences=1 << 14):
    """``n`` windows from deterministic neighborhood extraction around random seeds."""
    from ..fixtures import extract_window

    rng = np.random.default_rng(seed)
    seeds = rng.choice(reg.num_relations, size=n, replace=reg.num_relations < n)
    rel = np.zeros((n, 4), dtype=np.uint64)
    mask = np.zeros((n, 4), dtype=bool)
    for i, s in enumerate(seeds.tolist()):
        w = extract_window(reg, s, max_incidences=max_incidences)
        perm = rng.permutation(4)[:len(w)]  # random slot roles; empty slots elsewhere
        for slot, r in zip(perm.tolist(), w):
            rel[i, slot] = reg.relation_ids[r]
            mask[i, slot] = True
    return rel, mask


def seed_states(state, seed, scale=1.0):
    """Random finite initial states for every materialized relation row (volatile input)."""
    rng = np.random.default_rng(seed)
    rows = np.nonzero(state.row_ref > 0)[0]
    state.pool[rows] = rng.normal(0.0, scale, size=(rows.size, 8))
    state.mutation_epoch += 1
    return rows


def engine(state, checkpoint="54007/STATE8_SIGNED8", controller="R114", **kw):
    budgets = kw.pop("budgets", E.Budgets(max_passes=20))
    return E.Engine(state, checkpoint, controller=controller, budgets=budgets, **kw)
