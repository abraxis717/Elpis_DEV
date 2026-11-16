"""DETERMINISTIC BOUNDED FIXTURES — fixed seeds, explicit limits.

Scenario SHAPES follow the parent G5.13A suites (delayed evidence, weak
source, rare contradiction, goal shift, misleading coactivation) so parent
findings can be re-examined, but the streams, capacities and horizons used
by this harness are its own bounded settings and are reported as such.
Nothing here is a model trace; all fixtures are synthetic.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass

from . import contract_r0 as C
from . import transport_reference as T

MEMORY_SUITES = ("delayed", "weak_source", "rare_contradiction", "goal_shift",
                 "misleading_coactivation", "interleaved_supersession",
                 "late_relevant_burst")
SEEDS = (11, 23, 37, 53)
SETTINGS = ((12, 16), (24, 32))
HORIZON = 512
WRITE_AUTHORITY = "fixture:write:r0"


def vec_for(key, kind):
    """Deterministic bounded signed activation for a synthetic fact."""
    if kind == "weak":
        return (0.025, 0.005, 0.0, 0.0)
    if kind == "contradiction":
        return (-0.03, 0.0, 0.0, 0.0)
    if kind == "strong":
        return (1.0, 0.1, 0.0, 0.0)
    x = (key * 2246822519 + 3266489917) & 0xFFFFFFFF
    out = []
    for i in range(4):
        x = (1103515245 * x + 12345) & 0xFFFFFFFF
        out.append(((1.0 if x & 1 else -1.0) *
                    (0.6 + ((x >> 11) & 511) / 1024.0)) if i in (0, 1) else 0.0)
    return tuple(out)


def relation_for(key):
    return "R" + str(key)


def build_registry(relation_ids, valid_until=10_000, role_map=None,
                   expired_ids=(), protected_ids=()):
    """Deterministic admission registry over an explicit relation namespace.

    Relation ids are the strings produced by `relation_for(key)`.
    """
    recs = []
    for rid in sorted(relation_ids):
        key = int(rid[1:])
        role = (role_map or {}).get(rid, "evidence")
        other = "counterevidence" if role == "evidence" else "evidence"
        h = random.Random(hash(("hacf-synthetic", rid)) & 0xFFFFFFFF).randrange(1 << 48)
        recs.append(C.AdmissionRecord(
            relation_id=rid,
            provenance=f"prov:{key & 0xFFFF:04x}",
            participants=((key, role, 0), (key + 1, other, 1)),
            flags=("synthetic",),
            ticket=f"tkt:{h:012x}",
            issued_tick=0,
            valid_until_tick=(0 if rid in expired_ids else valid_until),
            protected=(rid in protected_ids)))
    return C.AdmissionRegistry(recs)


def memory_stream(suite, seed, capacity, window, horizon=HORIZON):
    """One bounded episode trace plus the registry/transport it requires.

    Returns (trace, registry, transport, meta).  `meta` carries the task-relevant
    identifier sets used ONLY for retention measurement; the eviction machinery
    never sees them, and salience arrives only as explicit feedback arguments.
    """
    if suite not in MEMORY_SUITES:
        raise ValueError("UNKNOWN_SUITE")
    spec = C.SessionSpec(kv_window_tokens=window, max_active_relations=capacity,
                         max_active_entities=capacity + 1, max_control_records=8,
                         max_ops_per_tick=16, max_steps_per_call=horizon + 8)
    old = tuple(100 + i for i in range(8))
    new = tuple(200 + i for i in range(8))
    noise = tuple(10_000 + t for t in range(1, horizon + 1))
    used = set(old) | set(new) | set(noise)

    kind_of = ("weak" if suite in ("weak_source", "misleading_coactivation")
               else "strong")
    role_map, protected = {}, set()
    for k in used:
        role_map[relation_for(k)] = "evidence"
    if suite == "rare_contradiction":
        for k in old[:2]:
            role_map[relation_for(k)] = "counterevidence"
            protected.add(relation_for(k))

    registry = build_registry({relation_for(k) for k in used},
                              valid_until=horizon + 64, role_map=role_map,
                              protected_ids=protected)
    ops = {}

    def put_fact(key, kind, role=None, protect=False):
        rid = relation_for(key)
        role = role or role_map[rid]
        frame = str(key)
        ops[(rid, frame)] = _identity4()
        value = vec_for(key, kind)
        p = C.Proposal(relation_id=rid, participant=key, role=role, ordinal=0,
                       flags=("synthetic",), source_frame=frame, value4=value,
                       ticket=registry.get(rid).ticket, issued_tick=0,
                       provenance=registry.get(rid).provenance)
        trace.append(C.op("upsert_exact", key, value, WRITE_AUTHORITY))
        trace.append(C.op("propose_relation", p))
        trace.append(C.op("apply_delta", p))

    trace = [C.op("begin_session")]
    SHIFT_SUITES = ("goal_shift", "interleaved_supersession",
                    "late_relevant_burst")
    gs_start, gs_end = horizon // 3, horizon // 3 + 8
    # Post-shift the new goal keeps using the new facts until the horizon;
    # noise keeps arriving but only every 16 steps, so the bank is not
    # permanently saturated by the distractor stream.
    noise_period = 16
    for t in range(1, horizon + 1):
        trace.append(C.op("observe", (t & 0xFFFFFFFF,)))
        if t <= len(old):
            contradiction = (suite == "rare_contradiction" and t <= 2)
            put_fact(old[t - 1], "contradiction" if contradiction else kind_of,
                     protect=contradiction)
        elif t <= 192 and (suite not in SHIFT_SUITES or t < gs_start):
            if t <= 96 or t % 4:
                k = old[(t + seed) % len(old)]
                co = (0.025 if suite in ("weak_source", "misleading_coactivation")
                      else 0.0 if suite == "rare_contradiction" else 0.8)
                trace.append(C.op("reinforce", relation_for(k), co, 1.0))
            else:
                put_fact(noise[t - 1], "noise")
        elif suite == "goal_shift" and gs_start <= t < gs_end:
            put_fact(new[t - gs_start], "strong")
        elif suite == "goal_shift" and t > gs_end:
            if t % noise_period == 0:
                put_fact(noise[t - 1], "noise")
            elif t % 5:
                trace.append(C.op("reinforce", relation_for(new[(t + seed) % 8]),
                                  0.7, 1.0))
            else:
                trace.append(C.op("reinforce", relation_for(old[(t + seed) % 8]),
                                  0.0, -1.0))
        elif suite == "interleaved_supersession" and t > gs_end:
            if t % noise_period == 0:
                put_fact(noise[t - 1], "noise")
            elif t % 3:
                trace.append(C.op("reinforce", relation_for(new[(t + seed) % 8]),
                                  0.6, 1.0))
            else:
                trace.append(C.op("reinforce", relation_for(old[(t + seed) % 8]),
                                  0.1, -1.0))
        elif suite == "late_relevant_burst" and gs_start <= t < gs_start + 16:
            put_fact(new[t - gs_start] if t - gs_start < 8 else noise[t - 1], "strong")
        else:
            put_fact(noise[t - 1], "noise")
        if suite == "misleading_coactivation" and t > 192 and t % 3 == 0:
            trace.append(C.op("reinforce", relation_for(noise[t - 1]), 1.0, 0.0))
    transport = _DictTransport(ops)
    checkpoints = (horizon // 4, horizon // 2, horizon)
    meta = {"old": old, "new": new, "checkpoints": checkpoints,
            "suite": suite, "seed": seed, "capacity": capacity, "window": window,
            "spec": spec, "horizon": horizon}
    return trace, registry, transport, meta


class _DictTransport:
    """Identity transport for suites whose subject is not geometry."""
    def __init__(self, ops):
        self._ops = ops

    def bind_frame(self, relation_id, frame_id, operator):
        self._ops[(relation_id, frame_id)] = operator

    def list_frames(self):
        return tuple(sorted({f for (_r, f) in self._ops}, key=repr))

    def to_canonical(self, relation_id, frame_id, vec4):
        op = self._ops.get((relation_id, frame_id))
        if op is None:
            raise PermissionError("UNBOUND")
        return tuple(math.fsum(op[i][j] * vec4[j] for j in range(4))
                     for i in range(4))

    def to_view(self, relation_id, frame_id, vec4):
        op = self._ops.get((relation_id, frame_id))
        if op is None:
            raise PermissionError("UNBOUND")
        return tuple(math.fsum(op[j][i] * vec4[j] for j in range(4))
                     for i in range(4))


def _identity4():
    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


# ------------------------------------------------------------- gauge fixtures
def gauge_fixture(seed, participants=6, relations=3):
    """Independent local frames plus transported perspectives for suite E.

    base    : U_{relation<-participant}
    changed : U' = G_relation U G_participant^-1 with x' = G_participant x
    """
    rng = random.Random(seed)
    base, changed, inputs, gr, gp = {}, {}, {}, {}, {}
    for p in range(participants):
        frame = f"p{p}"
        gp[frame] = T._rot(0.11 + 0.29 * p + 0.05 * seed)
        inputs[frame] = tuple(rng.uniform(-1, 1) for _ in range(4))
    for r in range(relations):
        rid = f"G{r}"
        gr[rid] = T._rot(-0.31 + 0.53 * r + 0.02 * seed)
        for p in range(participants):
            frame = f"p{p}"
            u = T._rot(0.09 * (1 + p) - 0.37 * (1 + r) + 0.011 * seed)
            base[(rid, frame)] = u
            changed[(rid, frame)] = T._matmul(T._matmul(gr[rid], u),
                                              T._transpose(gp[frame]))
    return {"base": base, "changed": changed, "inputs": inputs,
            "gauge_relation": gr, "gauge_participant": gp,
            "participants": tuple(f"p{p}" for p in range(participants)),
            "relations": tuple(f"G{r}" for r in range(relations))}


# ------------------------------------------------------- authority fixtures
def authority_fixture():
    """One admitted relation plus a catalogue of deliberately invalid proposals.

    Every entry maps a label to (proposal, expected_reject_code or None).
    """
    spec = C.SessionSpec(kv_window_tokens=4, max_active_relations=4,
                         max_active_entities=4, max_control_records=8,
                         max_ops_per_tick=16, max_steps_per_call=64)
    reg = build_registry({"R10", "R11"}, valid_until=32,
                         role_map={"R10": "evidence", "R11": "counterevidence"},
                         expired_ids={"R11"}, protected_ids=set())
    ops = {}
    for rid in ("R10", "R11"):
        for frame in ("10", "11", "12"):
            ops[(rid, frame)] = _identity4()
    transport = _DictTransport(ops)
    base = dict(relation_id="R10", participant=10, role="evidence", ordinal=0,
                flags=("synthetic",), source_frame="10",
                value4=(0.5, -0.25, 0.0, 0.75),
                ticket=reg.get("R10").ticket, issued_tick=0,
                provenance=reg.get("R10").provenance)

    def mk(**over):
        return C.Proposal(**{**base, **over})

    cases = {
        "admitted_valid": (mk(), None),
        "unknown_relation": (mk(relation_id="R99"), "UNADMITTED_RELATION"),
        "provenance_mismatch": (mk(provenance="prov:dead"), "PROVENANCE_MISMATCH"),
        "unknown_participant": (mk(participant=77), "UNKNOWN_PARTICIPANT"),
        "role_change": (mk(role="counterevidence"), "ROLE_MISMATCH"),
        "ordinal_change": (mk(ordinal=1), "ORDINAL_MISMATCH"),
        "flags_change": (mk(flags=("synthetic", "extra")), "FLAGS_MISMATCH"),
        "forged_ticket": (mk(ticket="tkt:ffffffffffff"), "STALE_TICKET"),
        "expired_ticket": (mk(relation_id="R11", participant=11,
                              role="counterevidence", ordinal=0,
                              ticket=reg.get("R11").ticket,
                              provenance=reg.get("R11").provenance), "STALE_TICKET"),
        "unbound_frame": (mk(source_frame="unbound:frame"), "UNBOUND_FRAME"),
        "nonfinite_value": (mk(value4=(float("nan"), 0.0, 0.0, 0.0)), "NONFINITE_VALUE"),
        "magnitude_authority": (mk(value4=(500.0, -250.0, 0.0, 750.0)), None),
        "magnitude_with_forgery": (mk(value4=(500.0, -250.0, 0.0, 750.0),
                                      ticket="tkt:000000000000"), "STALE_TICKET"),
        "disconnected_participant": (mk(participant=11, role="counterevidence",
                                        ordinal=1), "DISCONNECTED_PARTICIPANT"),
        "unadmitted_participant_role": (mk(participant=11, role="evidence",
                                          ordinal=1), "ROLE_MISMATCH"),
    }
    return spec, reg, transport, cases


# ---------------------------------------------------------- window/rollover ops
def window_stream(rolls, window, spec=None):
    """Observe-only trace that crosses several KV rollovers."""
    s = spec or C.SessionSpec(kv_window_tokens=window, max_active_relations=4,
                              max_active_entities=4, max_control_records=8,
                              max_ops_per_tick=8, max_steps_per_call=window * rolls + 8)
    tokens = tuple(range(1, window * rolls + 1))
    trace = [C.op("begin_session")]
    marks = []
    for start in range(0, len(tokens), window):
        chunk = tokens[start:start + window]
        trace.append(C.op("observe", chunk))
        marks.append((start + len(chunk), chunk))
    return trace, marks, s
