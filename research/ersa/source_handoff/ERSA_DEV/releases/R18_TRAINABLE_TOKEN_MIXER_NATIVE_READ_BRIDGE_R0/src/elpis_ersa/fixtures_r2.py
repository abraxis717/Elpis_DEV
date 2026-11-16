"""ERSA R2 PROSPECTIVE BENCHMARK — frozen BEFORE any R2 candidate tuning.

Distinct from the frozen R0 fixtures (different seeds, different families,
different task-grounded predicates).  Predicates are evaluation authority only;
backends never see them.  Targets are task-grounded (which evidence the task
requires), not agreement with one dense heuristic.
"""
from __future__ import annotations
import random
from . import contract_ersa as C

SEEDS = {
    "sparse_evidence": 5011, "dense_evidence": 5021,
    "high_cardinality_distractors": 5031, "rare_decisive": 5041,
    "weak_signed_evidence": 5051, "contradictory_attested": 5061,
    "goal_change_supersession": 5071, "index_collisions": 5081,
    "protected_long_nonaccess": 5091, "full_capacity_eviction": 5101,
    "gauge_equivalent_queries": 5111,
    "near_identical_keys_distinct_identity": 5121,
}
FAMILIES = tuple(SEEDS)


def spec(cap=16, cand=8, subkeys=4, bits=8):
    return C.SessionSpec(max_active_relations=cap, max_candidates=cand,
                         product_key_subkeys=subkeys, product_key_bits=bits,
                         adjacency_degree=4, key_feature_dim=4)


def names(f):
    return tuple(sorted(n for bit, n in C.FLAG_NAMES.items() if f & bit))


def rec(rid, prov, ticket, protected=False, valid_until=1 << 62, ordinal=None,
        flags=0):
    if ordinal is None:
        ordinal = rid
    fl = list(names(flags))
    if protected and "PROTECTED" not in fl:
        fl.append("PROTECTED")
    return C.AdmissionRecord(relation_id=rid, provenance=prov, role=0,
                             ordinal=ordinal, flags=tuple(sorted(fl)),
                             ticket=ticket,
                             issued_tick=0, valid_until_tick=valid_until,
                             protected=protected or bool(flags & C.FLAG_PROTECTED))


def flag(protected=False, superseded=False, sealed=False):
    f = 0
    if protected:
        f |= C.FLAG_PROTECTED
    if superseded:
        f |= C.FLAG_SUPERSEDED
    if sealed:
        f |= C.FLAG_SEALED
    return f


def rv(rng, lo=-2.0, hi=2.0):
    return tuple(rng.uniform(lo, hi) for _ in range(4))


def build(family, cand=8, cap=16):
    rng = random.Random(SEEDS[family])
    sp = spec(cap=cap, cand=cand)
    n = min(cap, 12)

    def key(i, j=0):
        return (i + j, (i * 7 + 3) % 11, (i * 5 + 1) % 9, 4)

    def ins(i, **over):
        d = dict(relation_id=i, key=key(i), value4=rv(rng),
                 provenance=i % 4, participant=i % 4, role=0, ordinal=i,
                 ticket=100 + i, utility=1.0, credibility=0.8,
                 goal=(i % 3, 1, 2, 3), angle=(0.1 * i, 0.05 * i))
        d.update(over)
        return d

    if family == "sparse_evidence":
        # decisive record whose key shares NO product sub-key with the query
        ins_list = [ins(i) for i in range(n)]
        ins_list[9]["key"] = (900 + 9, 1, 1, 1)
        qs = [dict(qkey=key(2), qgoal=(2, 1, 2, 3),
                   qdir=tuple(ins_list[9]["value4"]), k=4, required=[9],
                   forbidden=[], note="decisive record, no sub-key overlap")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i) for i in range(n)],
                    authority={}, inserts=ins_list, queries=qs)

    if family == "dense_evidence":
        ins_list = [ins(i) for i in range(n)]
        qs = [dict(qkey=key(i), qgoal=(i % 3, 1, 2, 3),
                   qdir=tuple(ins_list[i]["value4"]), k=4, required=[i],
                   forbidden=[], note=f"mutual relevance {i}")
                for i in (1, 4, 7)]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i) for i in range(n)],
                    authority={}, inserts=ins_list, queries=qs)

    if family == "high_cardinality_distractors":
        ins_list = [ins(i) for i in range(n)]
        qs = [dict(qkey=key(n - 1), qgoal=(1, 1, 2, 3),
                   qdir=tuple(ins_list[n - 1]["value4"]), k=2,
                   required=[n - 1], forbidden=[],
                   note="one target among cap-1 distractors")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i) for i in range(n)],
                    authority={}, inserts=ins_list, queries=qs)

    if family == "rare_decisive":
        ins_list = [ins(i) for i in range(n)]
        ins_list[6]["utility"] = 0.05          # rare, low utility
        qs = [dict(qkey=key(6), qgoal=(0, 1, 2, 3),
                   qdir=tuple(ins_list[6]["value4"]), k=2, required=[6],
                   forbidden=[], note="low-utility decisive evidence")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i) for i in range(n)],
                    authority={}, inserts=ins_list, queries=qs)

    if family == "weak_signed_evidence":
        ins_list = [ins(i) for i in range(n)]
        ins_list[5]["value4"] = (0.05, 0.0, 0.0, 0.0)     # weak magnitude
        ins_list[5]["key"] = key(5)
        qs = [dict(qkey=key(5), qgoal=(1, 1, 2, 3), qdir=(1.0, 0.0, 0.0, 0.0),
                   k=2, required=[5], forbidden=[],
                   note="weak but directionally decisive")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i) for i in range(n)],
                    authority={}, inserts=ins_list, queries=qs)

    if family == "contradictory_attested":
        ins_list = [ins(i) for i in range(6)]
        ins_list[0]["contradicts"] = 1
        ins_list[1]["contradicts"] = 0
        qs = [dict(qkey=key(0), qgoal=(1, 1, 2, 3),
                   qdir=tuple(ins_list[0]["value4"]), k=2, required=[0, 1],
                   forbidden=[], note="both sides of an attested conflict")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i) for i in range(6)],
                    authority={}, inserts=ins_list, queries=qs)

    if family == "goal_change_supersession":
        ins_list = [ins(i) for i in range(n)]
        ins_list[3]["flags"] = flag(superseded=True)
        qs = [dict(qkey=key(3), qgoal=(0, 1, 2, 3),
                   qdir=tuple(ins_list[3]["value4"]), k=2, required=[3],
                   forbidden=[], note="superseded record still addressed"),
              dict(qkey=key(8), qgoal=(2, 1, 2, 3),
                   qdir=tuple(ins_list[8]["value4"]), k=2, required=[8],
                   forbidden=[], note="goal shift follows the goal")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i,
                                          flags=flag(superseded=(i == 3)))
                                      for i in range(n)],
                    authority={}, inserts=ins_list, queries=qs)

    if family == "index_collisions":
        # every record shares three of four sub-keys with the query
        base = (7, 3, 1, 4)
        ins_list = [dict(ins(i), key=(base[0], base[1], base[2], 50 + i))
                    for i in range(n)]
        ins_list[10]["key"] = base
        qs = [dict(qkey=base, qgoal=(1, 1, 2, 3),
                   qdir=tuple(ins_list[10]["value4"]), k=2, required=[10],
                   forbidden=[], note="exact identity inside a collision storm")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i) for i in range(n)],
                    authority={}, inserts=ins_list, queries=qs)

    if family == "protected_long_nonaccess":
        ins_list = [ins(i) for i in range(n)]
        ins_list[4]["flags"] = flag(protected=True)
        qs = [dict(qkey=key(4), qgoal=(1, 1, 2, 3),
                   qdir=tuple(ins_list[4]["value4"]), k=2, required=[4],
                   forbidden=[], note="protected, never accessed")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i, protected=(i == 4))
                                      for i in range(n)],
                    authority={}, inserts=ins_list, queries=qs)

    if family == "full_capacity_eviction":
        ins_list = [ins(i) for i in range(n)]
        ins_list[2]["flags"] = flag(protected=True)
        qs = [dict(qkey=key(2), qgoal=(1, 1, 2, 3),
                   qdir=tuple(ins_list[2]["value4"]), k=2, required=[2],
                   forbidden=[], note="protected survives saturation")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i,
                                          protected=(i == 2)) for i in range(n)],
                    authority={}, inserts=ins_list, queries=qs,
                    capacity_test=True, must_retain=[2])

    if family == "gauge_equivalent_queries":
        pa = {i: (0.3 * (i % 4), -0.2 * (i % 4)) for i in range(4)}
        ins_list = [ins(i) for i in range(n)]
        qs = [dict(qkey=key(6), qgoal=(0, 1, 2, 3),
                   qdir=tuple(ins_list[6]["value4"]), k=3, required=[6],
                   forbidden=[], note="gauge-equivalent query")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i) for i in range(n)],
                    authority={"participant_angles": pa},
                    inserts=ins_list, queries=qs,
                    gauge=((0.7, -0.4), {0: (0.25, 0.55),
                                       1: (0.1, 0.2),
                                       2: (0.3, -0.1),
                                       3: (-0.2, 0.4)}))

    if family == "near_identical_keys_distinct_identity":
        base = key(3)
        ins_list = [dict(ins(i), key=(base[0] + (i % 2), base[1], base[2],
                                   base[3] + (i % 3))) for i in range(6)]
        qs = [dict(qkey=base, qgoal=(1, 1, 2, 3), qdir=(1, -2, 0.5, -1),
                   k=2, required=[0], forbidden=[1],
                   note="exact identity holder of the queried key; "
                        "nearest neighbour forbidden")]
        return dict(spec=sp, records=[rec(i, i % 4, 100 + i) for i in range(6)],
                    authority={}, inserts=ins_list, queries=qs)

    raise KeyError(family)
