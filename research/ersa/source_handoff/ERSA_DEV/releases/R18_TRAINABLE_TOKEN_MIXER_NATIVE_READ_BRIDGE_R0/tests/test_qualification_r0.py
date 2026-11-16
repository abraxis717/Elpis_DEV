"""ERSA R0 qualification instrument + adversarial mutant detection.

Every check takes the backend class explicitly so the SAME instrument can run
against the honest implementation and against an injected defect. A mutant is
DETECTED when the instrument's verdict flips to False and the evidence string
carries the concrete divergence (ids/values), not merely a name.
"""
import copy
import numpy as np
import pytest

from elpis_ersa import bank_ersa as Bk, contract_ersa as C, fixtures_ersa as F
from elpis_ersa import harness_ersa as H, geometry_ersa as G, oracle_ersa as O

FAMS = ("single_relevant", "many_distractors", "similar_conflicting_keys",
        "contradictory_evidence", "weak_but_decisive", "strong_misleading",
        "goal_shift", "outdated_relation", "source_denied",
        "disconnected_component", "near_collision_identity", "rare_evidence",
        "long_no_access", "repeated_overwrite", "capacity_saturation",
        "session_teardown", "gauge_transform", "ood_relation_family")


def _bank_cls(**modes):
    class W(Bk.ErsBank):
        def begin_session(self, spec=None, registry=None, transport=None,
                          authority=None):
            a = dict(authority or {})
            a["modes"] = {"discovery": modes.get("discovery", "indexed"),
                          "selection": modes.get("selection", "ersa")}
            return super().begin_session(spec, registry, transport, a)
    return W


HONEST = _bank_cls()


# ---------------------------------------------------------------- checks ---

def _runq(b, sc):
    """Run the fixture's first query through the backend (aggregate implies
    select, so selection and evidence are both populated)."""
    q = sc["queries"][0]
    b.aggregate(q["qkey"], query_goal=q["qgoal"], k=q["k"], qdir=q["qdir"])
    return b


def _check_authority(cls, fam):
    sc = F.build(fam)
    b = _runq(H.replay(cls, sc), sc)
    got = [r for r, _ in getattr(b, "_last_selection", [])]
    ev = [e["relation_id"] for e in getattr(b, "_last_evidence", [])]
    auth = sc.get("authority") or {}
    denied = set(auth.get("denied_provenances", ()))
    exp_t = set(auth.get("expired_tickets", ()))
    snap = b.snapshot()["relations"]
    bad = []
    for rid in sorted(set(got) | set(ev)):
        rec = snap.get(rid)
        if rec is None:
            continue
        if rec["provenance"] in denied:
            bad.append(("denied_provenance", rid, rec["provenance"]))
        if rec["ticket"] in exp_t:
            bad.append(("expired_ticket", rid, rec["ticket"]))
        if auth.get("participant_angles") and rec["participant"] \
                not in auth["participant_angles"]:
            bad.append(("unbound_frame", rid, rec["participant"]))
    return (not bad, f"committed={sorted(set(got))} violations={bad}")


def _check_signed(cls, fam):
    sc = F.build(fam)
    b = _runq(H.replay(cls, sc), sc)
    orc = H._replay_oracle(sc)
    q = sc["queries"][0]
    b.aggregate(q["qkey"], query_goal=q["qgoal"], k=q["k"], qdir=q["qdir"])
    agg_o = orc.expected_aggregate(q["qkey"], q["qgoal"], q["k"],
                                   q["qdir"])[0]
    d = G.max_abs_diff(b._last_aggregate, agg_o)
    w = [round(e["weight"], 6) for e in getattr(b, "_last_evidence", [])]
    return (d <= 1e-9, f"dir_diff={d:.3e} weights={w}")


def _check_gauge(cls, fam):
    sc = F.build(fam)
    if not sc.get("gauge"):
        return (True, "fixture has no gauge")
    gam, gp = sc["gauge"]
    b = H.replay(cls, sc)
    q = sc["queries"][0]
    agg0 = H._run_agg(b, q)
    b2 = H.replay(cls, sc, gauge=(gam, gp))
    q2 = dict(q)
    if q.get("qdir"):
        q2["qdir"] = G.rho_apply(gam, tuple(q["qdir"]))
    agg1 = H._run_agg(b2, q2)
    d = G.max_abs_diff(agg1, G.rho_apply(gam, agg0))
    return (d <= 1e-9, f"covariance_defect={d:.3e}")


def _check_determinism(cls, fam):
    a = _runq(H.replay(cls, F.build(fam)), F.build(fam))
    b = _runq(H.replay(cls, F.build(fam)), F.build(fam))
    sa = [r for r, _ in getattr(a, "_last_selection", [])]
    sb = [r for r, _ in getattr(b, "_last_selection", [])]
    da = G.max_abs_diff(a._last_aggregate, b._last_aggregate)
    return (sa == sb and da == 0.0, f"sel={sa}/{sb} agg_diff={da}")


def _check_session_isolation(cls):
    sc = F.build("single_relevant")
    b = cls()
    b.begin_session(sc["spec"], sc["records"], H._transport(sc),
                    sc["authority"])
    for ins in sc["inserts"]:
        b.insert_relation(**ins)
    live_before = b.occupancy()["live"]
    b.reset()
    occ = b.occupancy()
    snap = b.snapshot()
    b2 = cls()
    b2.begin_session(sc["spec"], [], None, None)
    fresh = b2.occupancy()["live"]
    return (occ["live"] == 0 and not snap["relations"] and fresh == 0
            and live_before > 0,
            f"before={live_before} after_reset={occ['live']} "
            f"new_session={fresh} rels={len(snap['relations'])}")


def _check_capacity_transaction(cls, fam):
    sc = F.build(fam)
    b = H.replay(cls, sc)
    occ = b.occupancy()
    cap = sc["spec"].max_active_relations
    over = occ["live"] > cap
    before = copy.deepcopy(b.snapshot())
    bad = b.commit_validated_update(dict(stage=99, target=0, delta4=(0, 0, 0, 0)))
    after = copy.deepcopy(b.snapshot())
    mutated = before["relations"] != after["relations"]
    return (not over and bad.status == C.STATUS_REJECTED and not mutated,
            f"live={occ['live']} cap={cap} bad_commit={bad.status}/"
            f"{bad.detail} mutated={mutated}")


def _check_index_raw_freshness(cls, fam):
    """Read-after-write plus eviction freshness on one bank."""
    sc = F.build(fam)
    snap0 = None
    new_rid = 4000
    extra_rec = None
    b = cls()
    # first pass to learn a live record's shape, then reopen with admission
    probe = cls()
    probe.begin_session(sc["spec"], sc["records"], H._transport(sc),
                        sc["authority"])
    for ins in sc["inserts"]:
        probe.insert_relation(**ins)
    snap0 = probe.snapshot()["relations"]
    raw_ok = fresh_ok = True
    cands = []
    if snap0:
        src = snap0[max(snap0)]
        extra_rec = C.AdmissionRecord(relation_id=new_rid,
                                      provenance=src["provenance"], role=0,
                                      ordinal=new_rid, flags=(),
                                      ticket=new_rid, issued_tick=0,
                                      valid_until_tick=1 << 62,
                                      protected=False)
        b.begin_session(sc["spec"], list(sc["records"]) + [extra_rec],
                        H._transport(sc), sc["authority"])
        for ins in sc["inserts"]:
            b.insert_relation(**ins)
        new_key = tuple(int(x) + 777 for x in src["key"])
        r = b.insert_relation(relation_id=new_rid, key=new_key,
                              value4=src["value"], provenance=src["provenance"],
                              participant=src["participant"], role=0,
                              ordinal=new_rid, ticket=new_rid, utility=1.0)
        raw_ok = r.status == C.STATUS_OK
        if raw_ok:
            b.query_candidates(new_key, query_goal=sc["queries"][0]["qgoal"])
            raw_ok = new_rid in getattr(b, "_last_candidates", [])
            b.evict(new_rid)
            b.query_candidates(new_key,
                               query_goal=sc["queries"][0]["qgoal"])
            fresh_ok = new_rid not in getattr(b, "_last_candidates", [])
        cands = getattr(b, "_last_candidates", [])
    return (raw_ok and fresh_ok,
            f"raw_visible={raw_ok} evicted_gone={fresh_ok} "
            f"rid={new_rid} cands={cands}")


def _check_oracle(cls, fam):
    """Dense backend must equal the dense oracle exactly. Indexed backend must
    rank its own candidate set exactly as the oracle scores that set (no
    phantom evidence, no corrupted state). Recall is measured, not asserted."""
    sc = F.build(fam)
    b = _runq(H.replay(cls, sc), sc)
    orc = H._replay_oracle(sc)
    q = sc["queries"][0]
    got = [r for r, _ in getattr(b, "_last_selection", [])]
    full = [r for r, _ in orc.expected_selection(
        q["qkey"], q["qgoal"], q["k"], q["qdir"])]
    cands = set(getattr(b, "_last_candidates", []))
    phantom = [r for r in got if r not in cands]
    scores = {}
    for rid in cands:
        rec = orc.state.get(rid)
        if rec is None or not orc._admitted(rid) or not orc._frame_bound(rid):
            continue
        if rec["prov"] in orc.denied:
            continue
        scores[rid] = orc.score(orc.features(rid, q["qkey"], q["qgoal"],
                                            q["qdir"] or q["qkey"]))
    exp = [r for r, _ in sorted(scores.items(),
                                key=lambda kv: (-round(kv[1], 12), kv[0]))][:len(got)]
    dense = b._modes.get("discovery") == "dense"
    ok = (got == full) if dense else (got == exp and not phantom)
    recall = (len(set(got) & set(full)) / max(len(full), 1))
    return (ok, f"got={got} exp={exp} phantom={phantom} live={len(orc.state)} "
            f"recall={recall:.3f} dense={dense}")


def _check_self_lookup(cls, fam):
    """Index self-consistency: every live relation must be recoverable by
    querying its own key (read-after-write for every record)."""
    sc = F.build(fam)
    b = H.replay(cls, sc)
    snap = b.snapshot()["relations"]
    missing = []
    for rid, rec in sorted(snap.items()):
        b.query_candidates(tuple(int(x) for x in rec["key"]),
                           query_goal=sc["queries"][0]["qgoal"])
        if rid not in getattr(b, "_last_candidates", []):
            missing.append(rid)
    return (not missing,
            f"live={len(snap)} missing_from_own_key={missing}")


CHECKS = {
    "authority": _check_authority,
    "signed": _check_signed,
    "gauge": _check_gauge,
    "determinism": _check_determinism,
    "capacity_transaction": _check_capacity_transaction,
    "index_raw_freshness": _check_index_raw_freshness,
    "oracle": _check_oracle,
    "self_lookup": _check_self_lookup,
}
FAM_FOR = {"authority": "source_denied", "signed": "weak_but_decisive",
           "gauge": "gauge_transform", "determinism": "many_distractors",
           "capacity_transaction": "capacity_saturation",
           "index_raw_freshness": "single_relevant",
           "oracle": "many_distractors"}


# ------------------------------------------------------------- mutants -----

class MutAuthority(Bk.ErsBank):
    def begin_session(self, spec=None, registry=None, transport=None,
                      authority=None):
        return super().begin_session(spec, registry, transport, {})


class MutSigned(Bk.ErsBank):
    def _score(self, feats):
        return super()._score(feats) ** 2


class MutGauge(Bk.ErsBank):
    def _transported_value(self, slot):
        a = self._arrays
        return tuple(float(x) for x in a["value"][slot])


class MutReset(Bk.ErsBank):
    def reset(self):
        return None


class MutEvictIndex(Bk.ErsBank):
    def evict(self, relation_id):
        return C.receipt(C.STATUS_OK, "", "EVICT_NO_INDEX_REMOVAL")


class MutRawIndex(Bk.ErsBank):
    def _index_insert(self, slot, kf):
        return None


class MutCapacity(Bk.ErsBank):
    def begin_session(self, spec=None, registry=None, transport=None,
                      authority=None):
        import dataclasses
        if spec is not None:
            spec = dataclasses.replace(spec,
                                       max_active_relations=int(
                                           spec.max_active_relations * 2))
        return super().begin_session(spec, registry, transport, authority)


class MutChunkState(Bk.ErsBank):
    """Chunk-boundary state loss: every 4th insert fails to register in the
    index, as if a chunk boundary dropped transient index state."""
    def _index_insert(self, slot, kf):
        self._c = getattr(self, "_c", 0) + 1
        if self._c % 4 == 0:
            return None
        return super()._index_insert(slot, kf)


class MutTransaction(Bk.ErsBank):
    def commit_validated_update(self, stage):
        stage = dict(stage)
        slot = self._slot_of(int(stage.get("target", 0)))
        if slot is not None:
            self._arrays["value"][slot] = (0.0, 0.0, 0.0, 0.0)
        return C.receipt(C.STATUS_OK, "", "COMMITTED_UNVALIDATED")


MUTANTS = {
    "authority_ignored": (MutAuthority, "authority"),
    "magnitude_only_scores": (MutSigned, "signed"),
    "gauge_dropped": (MutGauge, "gauge"),
    "reset_noop": (MutReset, "session_isolation"),
    "evict_keeps_index": (MutEvictIndex, "index_raw_freshness"),
    "insert_skips_index": (MutRawIndex, "self_lookup"),
    "capacity_slot_reuse": (MutCapacity, "capacity_transaction"),
    "chunk_boundary_state": (MutChunkState, "self_lookup"),
    "commit_bypasses_validation": (MutTransaction, "capacity_transaction"),
}


# ------------------------------------------------------------- pytest ------

@pytest.mark.parametrize("fam", FAMS)
def test_authority(fam):
    ok, ev = _check_authority(HONEST, fam)
    assert ok, ev


@pytest.mark.parametrize("fam", ("weak_but_decisive", "contradictory_evidence",
                                 "strong_misleading", "gauge_transform",
                                 "similar_conflicting_keys"))
def test_signed_aggregation(fam):
    ok, ev = _check_signed(_bank_cls(discovery="dense"), fam)
    assert ok, ev


def test_gauge_covariance():
    ok, ev = _check_gauge(_bank_cls(discovery="dense"), "gauge_transform")
    assert ok, ev


@pytest.mark.parametrize("fam", FAMS)
def test_determinism(fam):
    ok, ev = _check_determinism(HONEST, fam)
    assert ok, ev


def test_session_isolation():
    ok, ev = _check_session_isolation(HONEST)
    assert ok, ev


def test_capacity_and_transaction():
    ok, ev = _check_capacity_transaction(HONEST, "capacity_saturation")
    assert ok, ev


def test_index_raw_and_freshness():
    ok, ev = _check_index_raw_freshness(HONEST, "single_relevant")
    assert ok, ev


@pytest.mark.parametrize("fam", FAMS)
def test_self_lookup(fam):
    ok, ev = _check_self_lookup(HONEST, fam)
    assert ok, ev


@pytest.mark.parametrize("fam", FAMS)
def test_oracle_agreement(fam):
    ok, ev = _check_oracle(HONEST, fam)
    assert ok, ev


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_detected(name):
    cls, key = MUTANTS[name]
    divergences = []
    if key == "session_isolation":
        ok, ev = _check_session_isolation(cls)
        if not ok:
            divergences.append(("session_isolation", "single_relevant", ev))
    else:
        for fam in FAMS:
            ok, ev = CHECKS[key](cls, fam)
            if not ok:
                divergences.append((key, fam, ev))
    assert divergences, (f"mutant {name} NOT detected by any frozen fixture",)
    print(f"MUTANT {name}: {len(divergences)} divergences; first -> "
          f"{divergences[0][1]}: {divergences[0][2]}")
