"""ERSA R1 — independent adversarial integrity tests (R1.2).

Independent of the supplied acceptance harness: new probes for stage replay,
duplicate operations, concurrent conflicting operations, stale state,
multi-record commit failure, index consistency after eviction, protected
preservation and cross-session leakage. Every forbidden path must return a
typed receipt AND leave committed state unchanged.
"""
import copy
import unittest

from elpis_ersa import contract_ersa as C
from elpis_ersa import bank_ersa as B


def auth(rid, protected=False):
    return C.AdmissionRecord(int(rid), 7, 0, int(rid),
                             ("PROTECTED",) if protected else (),
                             int(rid), 0, 1 << 62, bool(protected))


def bank(records=(1, 2), cap=2, ops=1000, stages=4, steps=8):
    b = B.ErsBank()
    reg = [auth(x if isinstance(x, int) else x[0],
                False if isinstance(x, int) else x[1]) for x in records]
    spec = C.SessionSpec(max_active_relations=cap,
                         max_control_records=max(8, len(reg)),
                         max_ops_per_tick=ops, pending_stage_limit=stages,
                         max_steps_per_call=steps,
                         relational_namespace=(0, 10000))
    r = b.begin_session(spec, reg, None, None)
    assert r.status == C.STATUS_OK, r
    return b


def ins(b, rid, **kw):
    flags = C.FLAG_PROTECTED if kw.pop("protected", False) else 0
    if "flags" in kw:
        flags = kw.pop("flags")
    return b.insert_relation(rid, (rid, 0, 0, 0), (float(rid), 0.0, 0.0, 0.0),
                             provenance=kw.pop("provenance", 7),
                             participant=kw.pop("participant", 0),
                             role=kw.pop("role", 0),
                             ordinal=kw.pop("ordinal", rid),
                             flags=flags,
                             ticket=kw.pop("ticket", rid), **kw)


class Attack(unittest.TestCase):
    def rejected(self, r):
        self.assertIsInstance(r, C.Receipt)
        self.assertEqual(r.status, C.STATUS_REJECTED)
        self.assertIn(r.code, C.REJECT_CODES)

    def unchanged(self, b, snap):
        self.assertEqual(b.snapshot(), snap, "forbidden path mutated state")

    # ------------------------------------------------------------ stage replay
    def test_replay_single(self):
        b = bank(records=(1,), cap=1)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        st = b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.25)
        self.assertIsInstance(st, dict)
        self.assertEqual(b.commit_validated_update(st).status, C.STATUS_OK)
        after = b.snapshot()
        for _ in range(3):
            self.rejected(b.commit_validated_update(st))
            self.unchanged(b, after)
        self.assertEqual(b.occupancy()["commits"], 1)

    def test_replay_after_reclaim(self):
        b = bank(records=(1,), cap=1, stages=1)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        st = b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.25)
        self.assertIsInstance(st, dict)
        self.assertEqual(b.commit_validated_update(st).status, C.STATUS_OK)
        fresh = b.propose_update(1, (0.0, 1.0, 0.0, 0.0), 0.25)
        self.assertIsInstance(fresh, dict, "reclaimed slot must issue")
        self.assertNotEqual(fresh["token"], st["token"])
        after = b.snapshot()
        self.rejected(b.commit_validated_update(st))
        self.unchanged(b, after)

    # ------------------------------------------------- forged / mutated stages
    def test_forged_token_types(self):
        b = bank(records=(1,), cap=1)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        before = b.snapshot()
        for forged in ({"token": 1, "entries": ((1, (1.0, 0, 0, 0), (9.0, 0, 0, 0)),),
                        "issued_tick": 0},
                       {"token": "1", "entries": ((1, (1.0, 0, 0, 0), (9.0, 0, 0, 0)),)},
                       {"token": True, "entries": ((1, (1.0, 0, 0, 0), (9.0, 0, 0, 0)),)},
                       {"token": 999, "entries": ()},
                       {"token": None, "entries": ((1, (1.0, 0, 0, 0), (9.0, 0, 0, 0)),)},
                       "not-a-stage"):
            self.rejected(b.commit_validated_update(forged))
            self.unchanged(b, before)
        self.assertEqual(b.occupancy()["commits"], 0)

    def test_mutated_prev_field(self):
        b = bank(records=(1,), cap=1)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        st = b.propose_update(1, (2.0, 0.0, 0.0, 0.0), 0.25)
        self.assertIsInstance(st, dict)
        tampered = copy.deepcopy(st)
        rid, prev, cur = st["entries"][0]
        tampered["entries"] = ((rid, (0.0, 0.0, 0.0, 0.0), cur),)
        before = b.snapshot()
        self.rejected(b.commit_validated_update(tampered))
        self.unchanged(b, before)

    def test_multi_record_piggyback(self):
        b = bank(records=(1, 2), cap=2)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        self.assertEqual(ins(b, 2).status, C.STATUS_OK)
        st = b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.25)
        self.assertIsInstance(st, dict)
        rid, prev, cur = st["entries"][0]
        greedy = dict(token=st["token"], issued_tick=st["issued_tick"],
                      entries=(st["entries"][0], (2, (2.0, 0, 0, 0), (99.0, 0, 0, 0))))
        before = b.snapshot()
        self.rejected(b.commit_validated_update(greedy))
        self.unchanged(b, before)
        self.assertNotIn(99.0, b.snapshot()["relations"][2]["value"])

    # --------------------------------------------------- concurrent conflicts
    def test_conflicting_stages_same_target(self):
        b = bank(records=(1,), cap=1)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        s1 = b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.25)
        s2 = b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.25)
        self.assertIsInstance(s1, dict)
        self.assertIsInstance(s2, dict)
        self.assertEqual(b.commit_validated_update(s1).status, C.STATUS_OK)
        after = b.snapshot()
        r = b.commit_validated_update(s2)
        self.rejected(r)
        self.assertEqual(r.code, "STALE_PROPOSAL")
        self.unchanged(b, after)

    # ------------------------------------------------------------- duplicates
    def test_duplicate_insert_and_reinforce_bounds(self):
        b = bank(records=(1, 2), cap=2)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        before = b.snapshot()
        self.rejected(ins(b, 1))
        self.unchanged(b, before)
        self.assertEqual(b.occupancy()["live"], 1)
        for _ in range(40):
            b.reinforce(1)
        self.assertLessEqual(b.snapshot()["relations"][1]["utility"], 4.0)

    # ------------------------------------------------------------------- stale
    def test_stale_after_external_mutation(self):
        b = bank(records=(1,), cap=1)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        st = b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.25)
        self.assertIsInstance(st, dict)
        self.assertEqual(b.commit_validated_update(st).status, C.STATUS_OK)
        after = b.snapshot()
        self.rejected(b.commit_validated_update(st))
        self.unchanged(b, after)

    # ------------------------------------------- index / eviction consistency
    def test_index_consistency_after_eviction(self):
        b = bank(records=(1, 2, 3), cap=2)
        for rid in (1, 2, 3):
            self.assertEqual(ins(b, rid).status, C.STATUS_OK)
        self.assertNotIn(1, b.snapshot()["relations"])
        rc = b.query_candidates((1, 0, 0, 0))
        self.assertEqual(rc.status, C.STATUS_OK)
        self.assertNotIn(1, b._last_candidates)
        sel = b.select((1, 0, 0, 0), k=4)
        self.assertEqual(sel.status, C.STATUS_OK)
        self.assertNotIn(1, [rid for rid, _f in b._last_selection])
        self.assertEqual(b.occupancy()["live"], 2)
        self.assertEqual(b.occupancy()["evictions"], 1)

    def test_protected_never_evicted_under_pressure(self):
        b = bank(records=((1, True), (2, True), 3), cap=2)
        self.assertEqual(ins(b, 1, protected=True).status, C.STATUS_OK)
        self.assertEqual(ins(b, 2, protected=True).status, C.STATUS_OK)
        before = b.snapshot()
        r = ins(b, 3)
        self.rejected(r)
        self.assertEqual(r.code, "PROTECTED_CAPACITY_FULL")
        self.unchanged(b, before)
        self.assertEqual(b.occupancy()["evictions"], 0)
        self.assertEqual(set(b.snapshot()["relations"]), {1, 2})
        rc = b.query_candidates((1, 0, 0, 0))
        self.assertEqual(rc.status, C.STATUS_OK)
        self.assertIn(1, b._last_candidates)

    # ------------------------------------------------------ session leakage
    def test_no_leakage_across_sessions(self):
        b = bank(records=(1,), cap=1)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        st = b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.25)
        self.assertIsInstance(st, dict)
        b.aggregate((1, 0, 0, 0))
        self.assertEqual(b.reset().status, C.STATUS_OK)
        r = b.begin_session(C.SessionSpec(max_active_relations=1,
                                          max_control_records=8,
                                          relational_namespace=(0, 10000)),
                            [auth(1)], None, None)
        self.assertEqual(r.status, C.STATUS_OK)
        self.assertEqual(b.occupancy()["live"], 0)
        self.assertEqual(b.occupancy()["inserts"], 0)
        self.assertEqual(b.occupancy()["commits"], 0)
        before = b.snapshot()
        self.rejected(b.commit_validated_update(st))
        self.unchanged(b, before)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)

    # ---------------------------------------------------------- budget / time
    def test_budget_exhaustion_is_typed_and_stale_free(self):
        b = bank(records=(1,), cap=1, ops=2)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        r = b.aggregate((1, 0, 0, 0))
        self.rejected(r)
        self.assertIsNone(getattr(b, "_last_aggregate", None))
        self.assertEqual(b._last_selection, [])
        self.assertEqual(b.occupancy()["live"], 1)

    def test_step_never_touches_time_on_invalid(self):
        b = bank(records=(1,), cap=1, steps=4)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        before = b.snapshot()
        for bad in (-1, 0, 4.0, "2", 10 ** 12, True, float("nan")):
            self.rejected(b.step(bad))
            self.unchanged(b, before)
            self.assertEqual(b.occupancy()["tick"], 0)
        self.assertEqual(b.step(4).status, C.STATUS_OK)
        self.assertEqual(b.occupancy()["tick"], 4)

    def test_metadata_rejects_before_mutation(self):
        b = bank(records=(1, 2), cap=1)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        before = b.snapshot()
        bad_calls = (dict(goal=(1.0, 2.0)), dict(goal=(1.0, 2.0, 3.0, 4.0, 5.0)),
                     dict(utility=float("inf")), dict(credibility=-0.5),
                     dict(angle=(float("nan"), 0.0)), dict(contradicts="x"),
                     dict(flags=-1), dict(ordinal=1.5), dict(ticket=None),
                     dict(participant=True))
        for kw in bad_calls:
            r = ins(b, 2, **kw)
            self.rejected(r)
            self.unchanged(b, before)
            self.assertEqual(b.occupancy()["live"], 1)
            self.assertEqual(b.occupancy()["evictions"], 0)
        self.assertIn(1, b.snapshot()["relations"])


    # ------------------------------------------------ halt / pending accounting
    def test_halt_blocks_pending_commit(self):
        b = bank(records=(1,), cap=1)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        st = b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.25)
        self.assertIsInstance(st, dict)
        self.assertEqual(b.halt().status, C.STATUS_OK)
        before = b.snapshot()
        self.rejected(b.commit_validated_update(st))
        self.unchanged(b, before)
        self.assertEqual(b.occupancy()["commits"], 0)

    def test_pending_ring_no_silent_loss(self):
        b = bank(records=(1,), cap=1, stages=3)
        self.assertEqual(ins(b, 1).status, C.STATUS_OK)
        stages = [b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.01)
                  for _ in range(3)]
        for st in stages:
            self.assertIsInstance(st, dict)
        self.assertEqual(len({st["token"] for st in stages}), 3)
        self.assertEqual(b.occupancy()["proposals"], 3)
        r = b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.01)
        self.rejected(r)
        self.assertEqual(r.code, "PENDING_STAGE_LIMIT")
        self.assertEqual(b.occupancy()["proposals"], 3)
        self.assertEqual(b.commit_validated_update(stages[0]).status,
                         C.STATUS_OK)
        # slot reclaimed: a new stage issues with a fresh token
        fresh = b.propose_update(1, (1.0, 0.0, 0.0, 0.0), 0.01)
        self.assertIsInstance(fresh, dict)
        self.assertNotEqual(fresh["token"], stages[0]["token"])
        # conflicting pending stages are detected, not silently applied
        after = b.snapshot()
        r = b.commit_validated_update(stages[1])
        self.rejected(r)
        self.assertEqual(r.code, "STALE_PROPOSAL")
        self.unchanged(b, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
