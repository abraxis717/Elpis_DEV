"""R1 mechanics: the benchmark, the bounded relational state, invalidation exactness, causality, the host over the
real ECS_C relational view, strict admission, permutation equivariance, deterministic replay and zero writes."""
import dataclasses
import math
import sys
import unittest

import numpy as np

from relational_lao_tests import REQUIRE_NATIVE
from cgc_lao_lab import rng as rng1
from cgc_lao_v2 import env2
from cgc_lao_v2.host import HostError
from cgc_lao_v2.providers import MechanismRegistry, ProviderDeclaration, Reading, slb2_digest
from relational_lao import ecsc, harness, learners as Lr, scalar, world as W

REGISTRY = MechanismRegistry.slb2()


def independent():
    return ProviderDeclaration("p", "INDEPENDENT_REREAD", 2, slb2_digest("slb2/independent-reread"), None)


def native_or_skip(test):
    try:
        ecsc.libraries()
    except (ecsc.NativeUnavailable, OSError) as exc:
        if REQUIRE_NATIVE:
            raise
        test.skipTest(f"ECS_C relational view unavailable: {exc}")


class TestBenchmark(unittest.TestCase):
    def test_partitions_are_fresh_and_templates_enter_as_specified(self):
        import hashlib

        def key(root, *labels):
            text = root + "/".join(str(x) for x in labels)
            return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        mine = {W.derive_key(p, "world", s, "W") for p, n in W.PARTITIONS.items() for s in range(n)}
        self.assertEqual(len(mine), sum(W.PARTITIONS.values()))
        for root in ("elpis/cgc_lao/solver_lab/v2/", "elpis/cgc_lao/solver_lab/v3/", "elpis/cgc_lao/solver_lab/v4/"):
            old = {key(root, p, "router", s) for p in ("TRAIN", "DEV", "VALIDATION", "HOLDOUT", "OOD")
                   for s in range(256)}
            self.assertFalse(mine & old, root)
        self.assertFalse(mine & {rng1.derive_key(p, "router", s) for p in ("TRAIN", "DEV", "VALIDATION", "HOLDOUT")
                                 for s in range(256)})
        dev = {W.template_for("DEV", s).name for s in range(256)}
        self.assertEqual(dev, {"D_DIAMOND", "D_FAN"})
        self.assertIn("V_LADDER", {W.template_for("VALIDATION", s).name for s in range(32)})
        self.assertTrue(any(W.template_for("HOLDOUT", s).name.startswith("H_RANDOM") for s in range(64)))
        self.assertFalse(any(W.template_for(p, s).name.startswith("H_RANDOM") or W.template_for(p, s).name ==
                             "V_LADDER" for p in ("TRAIN", "DEV") for s in range(W.PARTITIONS[p])))
        with self.assertRaises(ValueError):
            W.episodes("DEV", 256, "CLEAN", 0, 1)

    def test_potential_outcomes_do_not_depend_on_batching(self):
        a = W.episodes("DEV", 5, "FLIP40", 0, 24)
        b = W.episodes("DEV", 5, "FLIP40", 16, 8)
        for name in ("x", "xq", "b0", "b1", "y", "p", "own1", "own2", "perm"):
            self.assertTrue(np.array_equal(getattr(a, name)[16:], getattr(b, name)), name)

    def test_observation_model(self):
        b = W.episodes("DEV", 7, "FLIP40", 0, 4000)
        self.assertAlmostEqual(float(b.flips.mean()), 0.4, delta=0.005)
        d = W.episodes("DEV", 7, "EVENT40", 0, 2000)
        self.assertTrue(np.array_equal(d.flips[..., 0], d.flips[..., 1]))         # shared corruption: rho = 1
        c = W.episodes("DEV", 7, "CLEAN", 0, 2000)
        self.assertFalse(c.flips.any())
        self.assertTrue(np.array_equal(c.y[..., 0], c.clean))
        p_bar = c.p.mean()
        self.assertAlmostEqual(float(c.clean.mean()), float(p_bar), delta=0.01)
        r = W.episodes("DEV", 2, "CLEAN+REV_DRIFT", 0, 500)
        drift = r.b0 != r.b1
        self.assertTrue(np.array_equal(drift, r.own1 == r.s_ev[:, None]))       # only the revised source drifts
        s = W.episodes("DEV", 2, "CLEAN+REV_SAME", 0, 500)
        self.assertTrue(np.array_equal(s.b0, s.b1))

    def test_utilities(self):
        tpl = W.D_DIAMOND
        m = np.full((1, 5, tpl.relations.__len__()), 0.5)
        u = W.utilities(m, tpl, list(range(5)))[0]
        self.assertTrue(np.allclose(u[0, :3], 0.5) and np.allclose(u[1, :3], 0.25))
        self.assertAlmostEqual(u[4, 0], 1 - 0.75 ** 3)                           # REACH: noisy-OR of 3 paths
        self.assertTrue(np.isneginf(u[2, 2]) and np.isneginf(u[4, 1]))
        self.assertEqual([q.stop for q in tpl.queries], [0.6, 0.36, 0.36, 0.36, 0.6])


class TestRelationalState(unittest.TestCase):
    def run_arms(self, regime, arms, n=64, seed=11):
        b = W.episodes("DEV", seed, regime, 0, n)
        out = harness.rollout(b, arms, next(iter(arms)), "PROBE")
        return b, out

    def test_bounded_state_and_immutable_coefficients(self):
        f = Lr.RelationalFilter()
        f.reset(4, 8)
        self.assertEqual(f.state_scalars(), 80)          # 3 x 8 x (mu, v, lambda) + 8 counts
        coeff = (f.q.copy(), f.rho.copy(), f.log_prior.copy())
        self.run_arms("FLIP20", {"R": Lr.Bayes("R")})
        s = scalar.ScalarRelational()
        self.assertEqual(s.state_scalars(), 80)
        for k in range(scalar.BUFFER):
            s.update(k % 8, 0.1, (1, 0) if k < 4 else (1,), k, ("b",))
        with self.assertRaises(OverflowError):
            s.update(0, 0.1, (1,), 9, ("b",))            # the buffer is bounded: refused, never grown
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(coeff, (f.q, f.rho, f.log_prior))))

    def test_causality_hidden_variables_never_reach_a_learner(self):
        b = W.episodes("DEV", 13, "FLIP20", 0, 96)
        arms = lambda: {"R": Lr.Bayes("R"), "I": Lr.Bayes("I", readout="independent"), "C": Lr.Counts(),  # noqa
                        "X": Lr.Exact(), "N": Lr.Noop()}
        a1 = arms()
        harness.rollout(b, a1, "R")
        poisoned = dataclasses.replace(b, b0=-b.b0, b1=-b.b1, p=1 - b.p, pq=1 - b.pq, clean=1 - b.clean,
                                       flips=1 - b.flips, q_episode=b.q_episode + 0.1)
        a2 = arms()
        harness.rollout(poisoned, a2, "R")
        zq = b.logits(b.xq)
        for k in a1:
            self.assertTrue(np.array_equal(a1[k].log.rel, a2[k].log.rel), k)
            for (w1, m1), (w2, m2) in zip(a1[k].components(zq), a2[k].components(zq)):
                self.assertTrue(np.array_equal(w1, w2) and np.array_equal(m1, m2), k)

    def test_invalidation_equals_a_learner_that_never_saw_the_retired_events(self):
        """Strategies A and B: the vectorised state equals the scalar reference fed exactly the retained events."""
        for regime in ("CLEAN+REV_DRIFT", "FLIP20+RECLAIM_DRIFT", "FLIP20+DORMANT", "CLEAN+REV_SAME"):
            for strategy in ("replay", "slot"):
                b, _ = self.run_arms(regime, {"R": (arm := Lr.Bayes("R", invalidation=strategy))}, n=40)
                keep = arm._keep(b, W.T)
                for e in range(b.count):
                    s = scalar.ScalarRelational(slots=b.R)
                    for t in range(W.T):
                        if keep[e, t]:
                            s.update(int(arm.log.rel[e, t]), float(arm.log.z[e, t]),
                                     tuple(int(v) for v in arm.log.reads[e, t, :arm.log.n[t]]), t, None)
                    for got, ref in zip((arm.filter.mu[e], arm.filter.v[e], arm.filter.lam[e]), s.state()[:3]):
                        self.assertTrue(np.allclose(got, np.array(ref), rtol=0, atol=1e-12), (regime, strategy, e))
                    self.assertEqual(list(arm.filter.counts[e]), list(s.counts), (regime, strategy, e))

    def test_stale_evidence_is_gone_and_the_stale_ablation_keeps_it(self):
        b = W.episodes("DEV", 17, "CLEAN+REV_DRIFT", 0, 200)
        arms = {"B": Lr.Bayes("B"), "A": Lr.Bayes("A", invalidation="slot"), "S": Lr.Bayes("S", invalidation="none")}
        harness.rollout(b, arms, "B")
        retired = arms["B"].log.retired(b, W.T)
        self.assertTrue(retired.any())
        for k in ("A", "B"):
            self.assertFalse((arms[k]._keep(b, W.T) & retired).any(), k)
        self.assertTrue((arms["S"]._keep(b, W.T) & retired).any())

    def test_identity_ablation_preserves_feedback_frequencies(self):
        b = W.episodes("DEV", 19, "CLEAN", 0, 128)
        arms = {"R": Lr.Bayes("R"), "P": Lr.Bayes("P", permuted=True)}
        harness.rollout(b, arms, "R")
        self.assertTrue((b.perm != np.arange(b.R)[None]).all())                      # a derangement
        # the ablation holds bitwise the same beliefs and reliability posterior, attached to the wrong relations
        R, P = arms["R"].filter, arms["P"].filter
        e = np.arange(b.count)[:, None]
        for name in ("mu", "v", "lam"):
            got = np.take_along_axis(getattr(P, name), b.perm[:, None, :], axis=2)
            self.assertTrue(np.array_equal(got, getattr(R, name)), name)
        self.assertTrue(np.array_equal(P.weights(), R.weights()))
        self.assertTrue(np.array_equal(P.counts[e, b.perm], R.counts))
        self.assertFalse(np.array_equal(P.mu, R.mu))

    def test_contradiction_eliminates_the_clean_hypothesis_without_reinforcement(self):
        s = scalar.ScalarRelational()
        s.update(0, 0.0, (1, 0), 0, ("b",))                                          # a disagreeing duplicate
        w = s.weights()
        self.assertEqual(w[0], 0.0)                                                   # q = 0 is impossible now
        before = s.predictive([0.0] * 8)[0]
        s.update(0, 0.0, (1, 1), 1, ("b",))
        s.update(0, 0.0, (0, 0), 2, ("b",))
        after = s.predictive([0.0] * 8)[0]
        self.assertLess(abs(after - 0.5), 0.05)                                       # contradictions cancel
        self.assertAlmostEqual(before, after, delta=0.05)

    def test_slot_reset_is_exact_without_replay(self):
        s = scalar.ScalarRelational()
        rng = np.random.default_rng(3)
        events = [(t, int(rng.integers(8)), float(rng.normal()), (int(rng.integers(2)), int(rng.integers(2))) if t < 4
                   else (int(rng.integers(2)),)) for t in range(8)]
        for t, r, z, reads in events:
            s.update(r, z, reads, t, ("b",))
        victim = events[2][1]
        s.reset_slot(victim)
        ref = scalar.ScalarRelational()
        for t, r, z, reads in events:
            if r != victim:
                ref.update(r, z, reads, t, ("b",))
        self.assertEqual(s.state(), ref.state())                                      # bitwise


class TestHost(unittest.TestCase):
    def setUp(self):
        native_or_skip(self)
        from relational_lao import host, hostrun
        self.host, self.hostrun = host, hostrun

    def new_host(self, invalidation="replay", decl=None):
        h = self.host.RelationalHost(REGISTRY, 0.0, dict(self.host.FROZEN_DEFAULT, invalidation=invalidation))
        h.initialize(decl or independent())
        return h

    def test_host_over_the_real_view_equals_the_vectorised_learner(self):
        cases = [(r, s_, "coupled") for r in ("FLIP20", "CLEAN+REV_DRIFT", "FLIP20+DORMANT", "CLEAN+RECLAIM_DRIFT",
                                              "EVENT40") for s_ in ("replay", "slot")]
        cases += [(r, "replay", "marginal") for r in ("FLIP20", "CLEAN+REV_DRIFT", "CLEAN+RECLAIM_DRIFT")]
        for regime, strategy, readout in cases:
            b = W.episodes("DEV", 21, regime, 0, 24)
            arm = Lr.Bayes("R", readout=readout, invalidation=strategy)
            harness.rollout(b, {"R": arm}, "R")
            _, _, st = W.query_tensors(b.template)
            dec = Lr.decide(Lr.expected_utilities(arm, b.logits(b.xq), b.template, b.query_struct),
                            st[b.query_struct])
            frozen = dict(self.host.FROZEN_DEFAULT, invalidation=strategy,
                          PRIMARY=dict(self.host.FROZEN_DEFAULT["PRIMARY"], readout=readout))
            for e in range(b.count):
                h = self.host.RelationalHost(REGISTRY, 0.0, frozen)
                h.initialize(independent())
                got, chosen = self.hostrun.run_episode(h, b, e)
                self.assertEqual(chosen, list(arm.log.rel[e]), (regime, strategy, e))
                self.assertEqual(got, list(dec[e]), (regime, strategy, e))

    def test_revoked_source_cannot_reach_later_proposals(self):
        """After a sync that retires a source, the host's learner is bitwise the learner fed only the evidence the
        view still holds LIVE; so every later proposal and decision is a function of LIVE evidence alone."""
        b = W.episodes("DEV", 23, "FLIP20+REV_DRIFT", 0, 16)
        for e in range(b.count):
            h = self.new_host()
            ew = self.hostrun.EpisodeWorld(b, e)
            h.reset_episode(ew.world, ew.relations, b.W, ew.reporters)
            for t in range(W.T):
                for table in ew.transitions(t):
                    h.sync(table)
                    fresh = scalar.ScalarRelational(slots=b.R)
                    for step, s, z, reads, binding in h.learner.buffer:
                        if h.binding_status(s, binding) == "LIVE":
                            fresh.update(s, z, reads, step, binding)
                    self.assertEqual(h.learner.state(), fresh.state(), e)
                    for step, s, z, reads, binding in h.learner.buffer:
                        self.assertNotEqual(h.binding_status(s, binding), "RETIRED")
                slot = h.propose_relation(b.x[e, t])
                rel = h.rel_of[slot]
                h.observe_selected(slot, [Reading(f"{e}/{t}", f"{e}/{t}/{k}", f"c{k}", slot, int(b.y[e, t, rel, k]))
                                          for k in range(W.READS["I"][t])])

    def test_dormancy_restores_exactly_and_reclamation_with_a_reused_authority_retires(self):
        for regime, expect in (("CLEAN+DORMANT", "SUSPENDED"), ("CLEAN+RECLAIM_DRIFT", "RETIRED")):
            b = W.episodes("DEV", 25, regime, 0, 12)
            seen = set()
            for e in range(b.count):
                h = self.new_host()
                ew = self.hostrun.EpisodeWorld(b, e)
                h.reset_episode(ew.world, ew.relations, b.W, ew.reporters)
                before_dormancy = None
                for t in range(W.T):
                    for table in ew.transitions(t):
                        state = h.learner.state()
                        info = h.sync(table)
                        seen.update(info["status"])
                        if regime.endswith("DORMANT") and t == int(b.t_ev[e]):
                            before_dormancy = state                       # suspended evidence leaves the state
                        if regime.endswith("DORMANT") and t == int(b.t_ev[e]) + 2:
                            # reactivation at the same version: every buffered event is LIVE again, and the
                            # state is exactly the learner fed all of them (the suspended ones restored)
                            self.assertTrue(all(h.binding_status(s_, bd) == "LIVE"
                                                for _, s_, _, _, bd in h.learner.buffer))
                            fresh = scalar.ScalarRelational(slots=b.R)
                            for step, s_, z, reads, bd in h.learner.buffer:
                                fresh.update(s_, z, reads, step, bd)
                            self.assertEqual(h.learner.state(), fresh.state())
                            del state
                    slot = h.propose_relation(b.x[e, t])
                    rel = h.rel_of[slot]
                    h.observe_selected(slot, [Reading(f"{e}/{t}", f"{e}/{t}/{k}", f"c{k}", slot,
                                                      int(b.y[e, t, rel, k])) for k in range(W.READS["I"][t])])
                del before_dormancy
            self.assertIn(expect, seen, regime)

    def test_integrity_and_contract_failures_fail_closed(self):
        b = W.episodes("DEV", 27, "CLEAN", 0, 4)
        cases = {
            "QUALIFIED_READING_COUNT_MISMATCH": lambda t, rs: rs[:1] if t == 0 else rs,
            "READING_REPLAYED": lambda t, rs: [rs[0], dataclasses.replace(rs[1], reading_id=rs[0].reading_id)]
            if len(rs) == 2 else rs,
            "REREAD_ON_SAME_CHANNEL": lambda t, rs: [rs[0], dataclasses.replace(rs[1], channel_id=rs[0].channel_id)]
            if len(rs) == 2 else rs,
            "READING_VALUE": lambda t, rs: [dataclasses.replace(rs[0], value=2)] + rs[1:],
            "STALE_OR_FOREIGN_EVENT": lambda t, rs: [dataclasses.replace(r, event_id="0/0") for r in rs] if t == 1
            else rs,
        }
        for reason, tamper in cases.items():
            h = self.new_host()
            with self.assertRaises(HostError) as cm:
                self.hostrun.run_episode(h, b, 0, tamper=tamper)
            self.assertEqual(str(cm.exception), reason)
            self.assertTrue(all(x == 0.0 for row in h.learner.mu for x in row))          # zeroised
            self.assertTrue(all(x == 0.0 for row in h.learner.v for x in row))
            with self.assertRaises(HostError):
                h.propose_relation(b.x[0, 0])

    def test_strict_admission_and_contract_two(self):
        single = ProviderDeclaration("p", "INDEPENDENT_REREAD", 1, slb2_digest("slb2/independent-reread"), None)
        shared = ProviderDeclaration("p", "SHARED_SOURCE", 2, slb2_digest("slb2/shared-source"), None)
        for decl in (single, shared):
            h = self.new_host(decl=decl)
            self.assertEqual((h.admission.contract, h.admission.mode), ("II", "FIRST_ONLY"))
            b = W.episodes("DEV", 29, "EVENT40", 0, 2)
            self.hostrun.run_episode(h, b, 0, tamper=None)
            self.assertEqual(h.learner.M, 4)                                              # the fallback's hypotheses
            self.assertTrue(all(len(e[3]) == 1 for e in h.learner.buffer))                # first readings only
        h = self.new_host()
        self.assertEqual((h.admission.contract, h.admission.mode), ("I", "DUPLICATE"))
        with self.assertRaises(ValueError):
            self.host.RelationalHost(REGISTRY, 0.0, dict(self.host.FROZEN_DEFAULT, reads=(1,) * 8))

    def test_permutation_equivariance_and_deterministic_replay(self):
        b = W.episodes("DEV", 31, "FLIP20", 0, 16)
        for e in range(b.count):
            first = self.hostrun.run_episode(self.new_host(), b, e)
            again = self.hostrun.run_episode(self.new_host(), b, e)
            self.assertEqual(first, again)                                                # deterministic
            relabelled = dataclasses.replace(b, partition=b.partition, seed=b.seed)
            original = self.hostrun.EpisodeWorld.__init__

            def salted(self_, batch, e_):
                original(self_, batch, e_)
                tag = ("salt", e_)
                ents = [ecsc.digest("rlb1/renamed", *tag, n) for n in batch.template.entities]
                self_.relations = [(ents[a], ents[c]) for a, c in batch.template.relations]
            self.hostrun.EpisodeWorld.__init__ = salted
            try:
                renamed = self.hostrun.run_episode(self.new_host(), relabelled, e)
            finally:
                self.hostrun.EpisodeWorld.__init__ = original
            self.assertEqual(first, renamed, e)                       # new names, new canonical slot order

    def test_capacity_scope_and_no_writes(self):
        h = self.new_host()
        b = W.episodes("DEV", 33, "CLEAN", 0, 2)
        ew = self.hostrun.EpisodeWorld(b, 0)
        too_many = ew.relations + [(ecsc.digest("x"), ecsc.digest("y"))]
        with self.assertRaises(HostError):
            h.reset_episode(ew.world, too_many, b.W, ew.reporters)
        # a scope change comes with a version bump: the old evidence is retired by the view
        h = self.new_host()
        h.reset_episode(ew.world, ew.relations, b.W, ew.reporters)
        slot = h.propose_relation(b.x[0, 0])
        h.observe_selected(slot, [Reading("s/0", f"s/0/{k}", f"c{k}", slot, 1) for k in range(2)])
        row = ew.reporters(0, h.rel_of[slot])
        ew.world.set_version(row, 2, ecsc.digest("another scope"))
        info = h.sync(ew.world)
        self.assertEqual(info["status"], ["RETIRED"])
        self.assertEqual(h.learner.counts[slot], 0)
        # zero persistent writes during inference (Python audit hook on every write-capable open)
        writes, armed = [], [True]

        def hook(event, args):
            if not armed[0]:
                return
            if event == "open" and len(args) > 1 and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
                writes.append(args)
            elif event == "open" and len(args) > 2 and isinstance(args[2], int) and args[2] & 0o3:
                writes.append(args)
            elif event in ("os.remove", "os.rename", "os.mkdir", "os.truncate", "shutil.copyfile"):
                writes.append((event,) + tuple(args))
        sys.addaudithook(hook)                       # audit hooks cannot be removed: disarmed after the window
        try:
            for e in range(2):
                self.hostrun.run_episode(self.new_host(), b, e)
        finally:
            armed[0] = False
        self.assertEqual(writes, [])


if __name__ == "__main__":
    unittest.main()
