"""ERSA FROZEN FIXTURES R0 — adversarial benchmark definitions, frozen BEFORE
the final comparisons (authority date 2026-11-04).

Every family is a deterministic op-trace plus explicit expectation predicates.
Seeds are fixed.  A backend never sees these predicates as inputs; they are
evaluation authority only.
"""
from __future__ import annotations
import random
from . import contract_ersa as C

SEEDS = {"single_relevant": 11, "many_distractors": 23,
         "similar_conflicting_keys": 37, "contradictory_evidence": 53,
         "weak_but_decisive": 71, "strong_misleading": 97,
         "goal_shift": 131, "outdated_relation": 151,
         "source_denied": 173, "disconnected_component": 191,
         "near_collision_identity": 223, "rare_evidence": 251,
         "long_no_access": 277, "repeated_overwrite": 307,
         "capacity_saturation": 331, "session_teardown": 353,
         "gauge_transform": 379, "ood_relation_family": 401}

FAMILIES = tuple(SEEDS)


def _spec(cap=12, cand=6, k=4, bits=8):
    return C.SessionSpec(max_active_relations=cap, max_candidates=cand,
                         product_key_subkeys=k, product_key_bits=bits,
                         adjacency_degree=4, key_feature_dim=4)


def _rec(rid, prov, ticket, ordinal=None, flags=(), protected=False,
         valid_until=1 << 62):
    if ordinal is None:
        ordinal = rid
    return C.AdmissionRecord(relation_id=rid, provenance=prov, role=0,
                             ordinal=ordinal, flags=tuple(sorted(flags)),
                             ticket=ticket, issued_tick=0,
                             valid_until_tick=valid_until,
                             protected=protected)


def _flag(protected=False, superseded=False, sealed=False):
    f = 0
    if protected:
        f |= C.FLAG_PROTECTED
    if superseded:
        f |= C.FLAG_SUPERSEDED
    if sealed:
        f |= C.FLAG_SEALED
    return f


def _names(f):
    return tuple(sorted(n for bit, n in C.FLAG_NAMES.items() if f & bit))


def _rv4(rng, lo=-2.0, hi=2.0):
    return (rng.uniform(lo, hi), rng.uniform(lo, hi),
            rng.uniform(lo, hi), rng.uniform(lo, hi))


def build(family):
    """Return scenario dict for one frozen family."""
    rng = random.Random(SEEDS[family])
    spec = _spec()
    cap = spec.max_active_relations
    fam = family

    def key(i, jitter=0):
        return (i + jitter, (i * 7 + 3) % 11, (i * 5 + 1) % 9, 4)

    if fam == "single_relevant":
        recs = [_rec(i, i % 3, 100 + i) for i in range(4)]
        ins = [dict(relation_id=i, key=key(i), value4=(1.0 + i, -2.0 + i,
                 0.5, -1.0 * i), provenance=i % 3, participant=i % 3,
                 role=0, ordinal=i, ticket=100 + i, utility=1.0,
                 credibility=0.8, goal=(i, 1, 2, 3),
                 angle=(0.2 * i, -0.1 * i)) for i in range(4)]
        qs = [dict(qkey=key(2), qgoal=(2, 1, 2, 3), qdir=(1, -2, 0.5, -2),
                   k=2, required=[2], forbidden=[], note="exact identity")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs)

    if fam == "many_distractors":
        recs = [_rec(i, i % 3, 100 + i) for i in range(cap)]
        ins = [dict(relation_id=i, key=key(i), value4=_rv4(rng),
                    provenance=i % 3, participant=i % 3, role=0, ordinal=i,
                    ticket=100 + i, utility=1.0, credibility=0.8,
                    goal=(i % 3, 1, 2, 3), angle=(0.1 * i, 0.05 * i))
               for i in range(cap)]
        ins[7]["key"] = key(7, 0)
        qs = [dict(qkey=key(7), qgoal=(1, 1, 2, 3), qdir=tuple(ins[7]["value4"]),
                   k=3, required=[7], forbidden=[], note="distractor pressure")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs)

    if fam == "similar_conflicting_keys":
        recs = [_rec(i, i % 3, 100 + i) for i in range(6)]
        base = key(1)
        ins = [dict(relation_id=i, key=(base[0] + (i % 2), base[1],
                    base[2], base[3] + (i % 3)),
                    value4=(1.0 + 0.1 * i, -2.0, 0.5, -1.0),
                    provenance=i % 3, participant=i % 3, role=0, ordinal=i,
                    ticket=100 + i, utility=1.0, credibility=0.8,
                    goal=(1, 1, 2, 3), angle=(0.15 * i, -0.05 * i))
               for i in range(6)]
        qs = [dict(qkey=base, qgoal=(1, 1, 2, 3), qdir=(1, -2, 0.5, -1),
                   k=2, required=[], forbidden=[],
                   note="near keys must not be conflated: distinct identities")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs)

    if fam == "contradictory_evidence":
        recs = [_rec(i, i % 3, 100 + i) for i in range(5)]
        ins = [dict(relation_id=i, key=key(i), value4=(1.0, 1.0, 0.0, 0.0),
                    provenance=i % 3, participant=i % 3, role=0, ordinal=i,
                    ticket=100 + i, utility=1.0, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0),
                    contradicts=(-1 if i == 0 else 0)) for i in range(5)]
        qs = [dict(qkey=key(2), qgoal=(1, 1, 2, 3), qdir=(1, 1, 0, 0), k=3,
                   required=[2], forbidden=[],
                   note="contradiction penalty must demote contradicted records")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs)

    if fam == "weak_but_decisive":
        recs = [_rec(i, i % 3, 100 + i) for i in range(3)]
        ins = [dict(relation_id=0, key=key(0), value4=(10.0, 0.0, 0.0, 0.0),
                    provenance=0, participant=0, role=0, ordinal=0,
                    ticket=100, utility=1.0, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0)),
               dict(relation_id=1, key=key(1), value4=(0.0, 0.0, 0.0, 0.3),
                    provenance=1, participant=1, role=0, ordinal=1,
                    ticket=101, utility=1.0, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0))]
        qs = [dict(qkey=key(1), qgoal=(1, 1, 2, 3), qdir=(0, 0, 0, 1), k=2,
                   required=[1], forbidden=[],
                   note="weak source must survive: magnitude-only aggregation fails")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs)

    if fam == "strong_misleading":
        recs = [_rec(i, i % 3, 100 + i) for i in range(4)]
        ins = [dict(relation_id=0, key=key(0), value4=(9.0, 9.0, 0.0, 0.0),
                    provenance=0, participant=0, role=0, ordinal=0,
                    ticket=100, utility=1.0, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0)),
               dict(relation_id=1, key=key(1), value4=(0.4, 0.4, 0.0, 0.0),
                    provenance=1, participant=1, role=0, ordinal=1,
                    ticket=101, utility=1.0, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0))]
        qs = [dict(qkey=key(1), qgoal=(1, 1, 2, 3), qdir=(0.4, 0.4, 0, 0),
                   k=1, required=[1], forbidden=[0],
                   note="relevance must outrank magnitude")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs)

    if fam == "goal_shift":
        recs = [_rec(i, i % 3, 100 + i) for i in range(4)]
        ins = [dict(relation_id=i, key=key(i), value4=(1.0 + i, 0.0, 1.0 - i,
                    0.0), provenance=i % 3, participant=i % 3, role=0,
                    ordinal=i, ticket=100 + i, utility=1.0, credibility=0.8,
                    goal=(1, 0, 0, 0) if i < 2 else (0, 0, 1, 0),
                    angle=(0.0, 0.0)) for i in range(4)]
        qs = [dict(qkey=key(0), qgoal=(1, 0, 0, 0), qdir=(1, 0, 1, 0), k=2,
                   required=[0], forbidden=[2, 3], note="goal A"),
              dict(qkey=key(0), qgoal=(0, 0, 1, 0), qdir=(1, 0, 1, 0), k=2,
                   required=[2], forbidden=[],
                   note="goal B: top-1 must follow the goal")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs)

    if fam == "outdated_relation":
        recs = [_rec(0, 0, 100, valid_until=0), _rec(1, 1, 101),
                _rec(2, 2, 999)]
        auth = {"expired_tickets": (999,)}
        ins = [dict(relation_id=0, key=key(0), value4=(1.0, 0.0, 0.0, 0.0),
                    provenance=0, participant=0, role=0, ordinal=0, ticket=100,
                    utility=1.0, credibility=0.9, goal=(1, 1, 2, 3),
                    angle=(0.0, 0.0)),
               dict(relation_id=1, key=key(1), value4=(0.0, 1.0, 0.0, 0.0),
                    provenance=1, participant=1, role=0, ordinal=1, ticket=101,
                    utility=1.0, credibility=0.9, goal=(1, 1, 2, 3),
                    angle=(0.0, 0.0)),
               dict(relation_id=2, key=key(2), value4=(0.0, 0.0, 1.0, 0.0),
                    provenance=2, participant=2, role=0, ordinal=2, ticket=999,
                    utility=1.0, credibility=0.9, goal=(1, 1, 2, 3),
                    angle=(0.0, 0.0))]
        qs = [dict(qkey=key(2), qgoal=(1, 1, 2, 3), qdir=(0, 0, 1, 0), k=3,
                   required=[], forbidden=[2], note="expired ticket denied")]
        return dict(spec=spec, records=recs, authority=auth, inserts=ins,
                    queries=qs, advance=2)

    if fam == "source_denied":
        recs = [_rec(i, i, 100 + i) for i in range(3)]
        auth = {"denied_provenances": (1,)}
        ins = [dict(relation_id=i, key=key(i), value4=(1.0 * (i + 1), 0.0,
                    0.0, 1.0), provenance=i, participant=i, role=0, ordinal=i,
                    ticket=100 + i, utility=1.0, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0)) for i in range(3)]
        qs = [dict(qkey=key(1), qgoal=(1, 1, 2, 3), qdir=(2, 0, 0, 1), k=3,
                   required=[], forbidden=[1],
                   note="denied provenance must never be committed")]
        return dict(spec=spec, records=recs, authority=auth, inserts=ins,
                    queries=qs)

    if fam == "disconnected_component":
        recs = [_rec(i, i % 3, 100 + i) for i in range(3)]
        tr_part = {0: (0.0, 0.0), 1: (0.5, -0.5)}
        ins = [dict(relation_id=i, key=key(i), value4=(1.0, 0.5, -0.5, 1.0),
                    provenance=i % 3, participant=i, role=0, ordinal=i,
                    ticket=100 + i, utility=1.0, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0)) for i in range(3)]
        qs = [dict(qkey=key(2), qgoal=(1, 1, 2, 3), qdir=(1, 0.5, -0.5, 1),
                   k=3, required=[], forbidden=[2],
                   note="participant 2 frame unbound -> DISCONNECTED")]
        return dict(spec=spec, records=recs,
                    authority={"participant_angles": tr_part},
                    inserts=ins, queries=qs)

    if fam == "near_collision_identity":
        recs = [_rec(0, 0, 100), _rec(1, 1, 101)]
        ins = [dict(relation_id=0, key=(5, 5, 5, 5), value4=(1.0, 2.0, 3.0, 4.0),
                    provenance=0, participant=0, role=0, ordinal=0, ticket=100,
                    utility=1.0, credibility=0.9, goal=(1, 1, 1, 1),
                    angle=(0.0, 0.0)),
               dict(relation_id=1, key=(5, 5, 5, 6), value4=(1.0, 2.0, 3.0, 4.0),
                    provenance=1, participant=1, role=0, ordinal=1, ticket=101,
                    utility=1.0, credibility=0.9, goal=(1, 1, 1, 1),
                    angle=(0.0, 0.0))]
        qs = [dict(qkey=(5, 5, 5, 5), qgoal=(1, 1, 1, 1), qdir=(1, 2, 3, 4),
                   k=2, required=[0], forbidden=[],
                   note="identical geometry, distinct identity: no merging")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs)

    if fam == "rare_evidence":
        recs = [_rec(i, i % 3, 100 + i) for i in range(8)]
        ins = [dict(relation_id=i, key=key(i), value4=(0.1 * i, 0.2, 0.3, 0.4),
                    provenance=i % 3, participant=i % 3, role=0, ordinal=i,
                    ticket=100 + i, utility=1.0, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0)) for i in range(8)]
        ins[6]["key"] = (99, 98, 97, 96)
        qs = [dict(qkey=(99, 98, 97, 96), qgoal=(1, 1, 2, 3),
                   qdir=(0.6, 0.2, 0.3, 0.4), k=2, required=[6],
                   forbidden=[], note="rare key must still be discovered")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs)

    if fam == "long_no_access":
        recs = [_rec(i, i % 3, 100 + i) for i in range(4)]
        ins = [dict(relation_id=i, key=key(i), value4=(1.0, 1.0, 1.0, 1.0),
                    provenance=i % 3, participant=i % 3, role=0, ordinal=i,
                    ticket=100 + i, utility=1.0, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0)) for i in range(4)]
        qs = [dict(qkey=key(1), qgoal=(1, 1, 2, 3), qdir=(1, 1, 1, 1), k=2,
                   required=[1], forbidden=[],
                   note="staleness must demote, not erase")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs, advance=200)

    if fam == "repeated_overwrite":
        recs = [_rec(0, 0, 100)]
        ins = [dict(relation_id=0, key=key(0), value4=(1.0, 0.0, 0.0, 0.0),
                    provenance=0, participant=0, role=0, ordinal=0, ticket=100,
                    utility=1.0, credibility=0.9, goal=(1, 1, 2, 3),
                    angle=(0.0, 0.0))]
        qs = [dict(qkey=key(0), qgoal=(1, 1, 2, 3), qdir=(2, 0, 0, 0), k=1,
                   required=[0], forbidden=[], note="after overwrite")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs,
                    extra=[("propose_commit", 0, (1.0, 0.0, 0.0, 0.0), 0.5)])

    if fam == "capacity_saturation":
        cap = 6
        spec2 = _spec(cap=cap, cand=4)
        recs = [_rec(i, i % 3, 100 + i,
                     protected=(i == 0)) for i in range(cap + 3)]
        ins = [dict(relation_id=i, key=key(i), value4=(1.0 * i, 0.0, 0.0, 1.0),
                    provenance=i % 3, participant=i % 3, role=0, ordinal=i,
                    ticket=100 + i, utility=1.0 + 0.01 * i, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0)) for i in range(cap + 3)]
        qs = [dict(qkey=key(cap + 2), qgoal=(1, 1, 2, 3),
                   qdir=(1.0 * (cap + 2), 0, 0, 1), k=3,
                   required=[cap + 2], forbidden=[],
                   note="insert at capacity must evict authorizedly, keep protected")]
        return dict(spec=spec2, records=recs, authority={}, inserts=ins,
                    queries=qs, capacity_test=True, must_retain=[0])

    if fam == "session_teardown":
        recs = [_rec(i, i % 3, 100 + i) for i in range(3)]
        ins = [dict(relation_id=i, key=key(i), value4=(1.0, 1.0, 0.0, 0.0),
                    provenance=i % 3, participant=i % 3, role=0, ordinal=i,
                    ticket=100 + i, utility=1.0, credibility=0.9,
                    goal=(1, 1, 2, 3), angle=(0.0, 0.0)) for i in range(3)]
        qs = [dict(qkey=key(0), qgoal=(1, 1, 2, 3), qdir=(1, 1, 0, 0), k=1,
                   required=[], forbidden=[], note="after teardown: empty")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs, teardown=True)

    if fam == "gauge_transform":
        recs = [_rec(i, i % 3, 100 + i) for i in range(3)]
        pa = {0: (0.3, -0.2), 1: (1.1, 0.4), 2: (-0.7, 0.9)}
        gam = (0.7, -1.3)
        gp = {}
        ins = [dict(relation_id=i, key=key(i), value4=(1.0 + i, -2.0 + i,
                    0.5, -1.0 * i), provenance=i % 3, participant=i % 3,
                    role=0, ordinal=i, ticket=100 + i, utility=1.0,
                    credibility=0.9, goal=(i, 1, 2, 3),
                    angle=(0.2 * i, -0.1 * i)) for i in range(3)]
        qs = [dict(qkey=key(1), qgoal=(1, 1, 2, 3), qdir=(1, -2, 0.5, -1),
                   k=3, required=[1], forbidden=[],
                   note="full-operator gauge covariance")]
        return dict(spec=spec, records=recs,
                    authority={"participant_angles": pa},
                    inserts=ins, queries=qs, gauge=(gam, gp))

    if fam == "ood_relation_family":
        recs = [_rec(i, i % 3, 100 + i) for i in range(4)]
        ins = [dict(relation_id=i, key=(i * 13 + 1, -i * 11, i * 3 - 5,
                    -i + 2), value4=_rv4(rng, -3.0, 3.0),
                    provenance=i % 3, participant=i % 3, role=0, ordinal=i,
                    ticket=100 + i, utility=1.0, credibility=0.9,
                    goal=(-i, 2, -1, 3), angle=(0.4 * i, 0.2 * i))
               for i in range(4)]
        qs = [dict(qkey=(7 * 13 + 1, -7 * 11, 7 * 3 - 5, -7 + 2),
                   qgoal=(-3, 2, -1, 3), qdir=(0.5, -0.25, 0.125, -0.0625),
                   k=2, required=[], forbidden=[],
                   note="out-of-distribution keys must not crash or leak")]
        return dict(spec=spec, records=recs, authority={}, inserts=ins,
                    queries=qs)

    raise ValueError("UNKNOWN_FAMILY")
