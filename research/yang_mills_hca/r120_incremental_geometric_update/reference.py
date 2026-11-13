"""R120 independent reference: the update contract in plain Python. Small graphs only.

Shares nothing with the native kernel or the engine wrapper: dictionaries, explicit loops,
``math`` scalars, IEEE-single CGC via ``cgc_bridge.forward_reference``, brute-force discovery
of affected windows (a scan over all instances, not the binding lists), and the R118
pure-Python ``simple_reference.step`` for the HCA passes. It is the oracle for

* the update law (``update_state``),
* the batch semantics: validation order, duplicates, conflicts, atomic refusal,
* selective propagation with radius and passes.

Data model (as R118 ``simple_reference``):
    relations : {relation_id: {participant_id: frame8}}
    instances : {instance_id: [relation_id or None] * 4}
    state     : {relation_id: [A0..A3, B0..B3]}
    seq       : {relation_id: last applied event sequence}
"""
from __future__ import annotations

import math

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import simple_reference as SR

from . import cgc_bridge as CGC


def _dot4(x, y):
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3]


def _apply_frame(f, x):
    return [f[0] * x[0] + f[1] * x[1], f[2] * x[0] + f[3] * x[1], f[4] * x[2] + f[5] * x[3], f[6] * x[2] + f[7] * x[3]]


def _clamp4(v):
    n = math.sqrt(_dot4(v, v))
    scale = 4.0 / (n + 1e-30)
    if scale > 1.0:
        scale = 1.0
    return [x * scale for x in v]


def update_state(s8, frame8, obs4, weight, aux, cgc_w, eta, gamma, rho_b, kappa):
    """The R120 update law on one relation state. Returns the new 8-list, or None if non-finite."""
    A = [float(x) for x in s8[:4]]
    B = [float(x) for x in s8[4:]]
    if frame8 is None:
        v = [0.0, 0.0, 0.0, 0.0]
    else:
        t = _apply_frame([float(x) for x in frame8], [float(x) for x in obs4])
        v = [float(weight) * x for x in t]
    dv = [v[k] - A[k] for k in range(4)]
    x = [math.tanh(_dot4(A, v)), math.tanh(_dot4(B, v)), math.tanh(_dot4(A, B)), math.tanh(_dot4(dv, dv)), float(aux)]
    y = CGC.forward_reference(cgc_w, [np.float32(q) for q in x])
    a = kappa * math.tanh(float(y[0]))
    b = kappa * math.tanh(float(y[1]))
    g = kappa * math.tanh(float(y[2]))
    An = [(A[k] + eta * (a * A[k] + b * B[k])) + gamma * v[k] for k in range(4)]
    Bn = [(B[k] + eta * (g * A[k] - a * B[k])) + (gamma * rho_b) * v[k] for k in range(4)]
    out = _clamp4(An) + _clamp4(Bn)
    return out if all(map(math.isfinite, out)) else None


def affected_instances(instances, seed_relations, radius):
    """Brute-force neighbourhood: instances binding a seed relation, expanded ``radius - 1`` times."""
    rels = set(seed_relations)
    active = set()
    frontier = set(seed_relations)
    for _ in range(radius):
        new = {i for i, slots in instances.items() if i not in active and any(r in frontier for r in slots if r is not None)}
        active |= new
        nxt = {r for i in new for r in instances[i] if r is not None} - rels
        rels |= nxt
        frontier = nxt
    return active


def apply_batch(model, events, params, limits=None, propagate=True, radius=1, passes=1):
    """Reference batch semantics. ``events``: dicts with rel_id, seq, part (participant id or None),
    obs, weight, aux, and optionally exp_gen/gen handled by the caller.

    Returns (disposition, new_state, new_seq, info). On refusal the inputs are returned unchanged.
    """
    relations, instances, state, seq = model["relations"], model["instances"], model["state"], model["seq"]
    W, bias, ck, ctrl, mode = params["W"], params["bias"], params["checkpoint"], params["controller"], params["mode"]
    new_state, new_seq = {}, {}
    last_payload = {}
    applied = skipped = 0
    lim = limits or {}

    def refuse(code, index):
        return code, state, seq, {"fail_index": index, "applied": 0, "skipped": 0}

    if len(events) > lim.get("max_events", 1 << 30):
        return refuse("REFUSED_EVENT_BUDGET", -1)
    for e, ev in enumerate(events):
        vals = [ev["weight"], ev["aux"]] + list(ev["obs"])
        if not all(math.isfinite(float(x)) for x in vals) or not -1.0 <= ev["weight"] <= 1.0 or not 0.0 <= ev["aux"] <= 1.0:
            return refuse("REFUSED_INVALID_VALUE", e)
        rid = ev["rel_id"]
        if rid not in relations:
            return refuse("REFUSED_UNKNOWN_RELATION", e)
        if rid not in state:
            return refuse("REFUSED_NOT_MATERIALIZED", e)
        if ev.get("stale"):
            return refuse("REFUSED_STALE_RELATION", e)
        frame = None
        if ev["part"] is not None:
            if ev["part"] not in relations[rid]:
                return refuse("REFUSED_NOT_INCIDENT", e)
            frame = relations[rid][ev["part"]]
        elif any(float(x) != 0.0 for x in ev["obs"]):
            return refuse("REFUSED_NOT_INCIDENT", e)
        last = new_seq.get(rid, seq.get(rid, 0))
        payload = (ev["part"], float(ev["weight"]), float(ev["aux"]), tuple(float(x) for x in ev["obs"]))
        if ev["seq"] <= last:
            if rid in last_payload and ev["seq"] == last and last_payload[rid] != payload:
                return refuse("REFUSED_CONFLICT", e)
            skipped += 1
            continue
        cur = new_state.get(rid, state[rid])
        nxt = update_state(cur, frame, ev["obs"], ev["weight"], ev["aux"], params["cgc_w"], params["eta"],
                           params["gamma"], params["rho_b"], params["kappa"])
        if nxt is None:
            return refuse("REFUSED_NONFINITE", e)
        new_state[rid] = nxt
        new_seq[rid] = ev["seq"]
        last_payload[rid] = payload
        applied += 1
    out_state = dict(state)
    out_state.update(new_state)
    out_seq = dict(seq)
    out_seq.update(new_seq)
    info = {"fail_index": -1, "applied": applied, "skipped": skipped, "touched": sorted(new_state), "active": []}
    if not new_state or not propagate:
        return ("OK" if not new_state else "COMMITTED_DEFERRED"), out_state, out_seq, info
    active = affected_instances(instances, list(new_state), radius)
    rows = {r for i in active for r in instances[i] if r is not None}
    if len(active) > lim.get("max_instances", 1 << 30):
        return refuse("REFUSED_INSTANCE_BUDGET", -1)
    if len(rows) > lim.get("max_rows", 1 << 30):
        return refuse("REFUSED_ROW_BUDGET", -1)
    info["active"] = sorted(active)
    stepped = SR.step(W, bias, relations, instances, out_state, active, ck, ctrl, passes=passes, mode=mode)
    return "OK", stepped, out_seq, info
