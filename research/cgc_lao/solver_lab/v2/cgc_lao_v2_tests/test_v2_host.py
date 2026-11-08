"""Provider contract and research host, end to end (Milestones B and E).

- The audit: every registered SLB2 measurement mechanism is checked statistically with privileged simulator
  access (the "testable measurement provenance" the host registry relies on).
- Admission: Contract I only for audited independent re-reads; everything else falls back (Contract II).
- Lifecycle: complete episodes through the host with the native learner, decision-for-decision equal to the
  vectorised frozen learners; every integrity or lifecycle violation fails closed.
"""
import dataclasses
import os
import unittest

import numpy as np

from cgc_lao_v2_tests import V2  # noqa: F401
from cgc_lao_v2 import agents2, candidates2, env2, harness2, native
from cgc_lao_v2.host import HostError, ResearchHost, admit, run_episode
from cgc_lao_v2.providers import (MechanismRegistry, ProviderDeclaration, Reading, SimulatedProvider,
                                  slb2_digest)

REQUIRE = os.environ.get("ELPIS_REQUIRE_NATIVE") == "1"
FROZEN = {"PRIMARY": candidates2.BAYES_CONFIGS["RMF_NOEVENT"], "FALLBACK": candidates2.BAYES_CONFIGS["SAFE4_P8"],
          "horizon": 8, "reads": list(env2.CONTRACTS["DUP2_4"].reads)}
REGISTRY = MechanismRegistry.slb2()


def declare(relation, mechanism, bound=None, name="p"):
    return ProviderDeclaration(name, relation, 2, slb2_digest(mechanism) if mechanism else "0" * 64, bound)


def honest_independent():
    return declare("INDEPENDENT_REREAD", "slb2/independent-reread")


class TestProviderAudit(unittest.TestCase):
    """Privileged statistical audit of the registered mechanisms (never available to a deployed host)."""

    def corr(self, regime, partition="DEV"):
        b = env2.episodes(partition, 250, regime, 0, 6000)
        f = b.flips.astype(float)
        return float(np.corrcoef(f[..., 0].ravel(), f[..., 1].ravel())[0, 1]), b

    def test_independent_reread(self):
        c, b = self.corr("FLIP40")
        self.assertLess(abs(c), 0.01)
        self.assertAlmostEqual(float((b.y[..., 0] != b.y[..., 1]).mean()), 0.48, delta=0.006)

    def test_correlated(self):
        for regime, rho in (("CORR10", 0.1), ("CORR50", 0.5)):
            c, _ = self.corr(regime)
            self.assertLessEqual(c, rho + 0.01, regime)
            self.assertGreater(c, rho - 0.01, regime)

    def test_shared_source(self):
        c, _ = self.corr("EVENT40")
        self.assertEqual(c, 1.0)

    def test_second_event(self):
        b = env2.episodes("OOD", 3, "SECOND_EVENT40", 0, 6000)
        pr = 0.4 + 0.2 * b.p
        self.assertAlmostEqual(float((b.y[..., 0] != b.y[..., 1]).mean()), float((2 * pr * (1 - pr)).mean()), delta=0.006)

    def test_registry_is_host_owned_and_digest_keyed(self):
        a = REGISTRY.lookup(slb2_digest("slb2/independent-reread"))
        self.assertEqual((a.relation, a.correlation_bound), ("INDEPENDENT_REREAD", 0.0))
        self.assertIsNone(REGISTRY.lookup("0" * 64))
        self.assertEqual({f.name for f in dataclasses.fields(Reading)},
                         {"event_id", "reading_id", "channel_id", "expert", "value"})   # no corruption label


class TestAdmission(unittest.TestCase):
    def test_decision_table(self):
        cases = [
            (honest_independent(), 0.0, ("I", "PRIMARY", "DUPLICATE")),
            (declare("INDEPENDENT_REREAD", None), 0.0, ("II", "FALLBACK", "FIRST_ONLY")),
            (declare("INDEPENDENT_REREAD", "slb2/shared-source"), 0.0, ("II", "FALLBACK", "FIRST_ONLY")),
            (declare("CORRELATED", "slb2/correlated-0.1", 0.1), 0.0, ("II", "FALLBACK", "FIRST_ONLY")),
            (declare("CORRELATED", "slb2/correlated-0.1", 0.1), 0.1, ("I", "PRIMARY", "DUPLICATE")),
            (declare("CORRELATED", "slb2/correlated-0.1", 0.05), 0.05, ("II", "FALLBACK", "FIRST_ONLY")),
            (declare("CORRELATED", "slb2/correlated-0.5", 0.5), 0.1, ("II", "FALLBACK", "FIRST_ONLY")),
            (declare("CORRELATED", "slb2/correlated-0.1", None), 0.1, ("II", "FALLBACK", "FIRST_ONLY")),
            (declare("SHARED_SOURCE", "slb2/shared-source"), 1.0, ("II", "FALLBACK", "FIRST_ONLY")),
            (declare("COPY", "slb2/independent-reread"), 1.0, ("II", "FALLBACK", "FIRST_ONLY")),
            (declare("SECOND_EVENT", "slb2/second-event"), 1.0, ("II", "FALLBACK", "FIRST_ONLY")),
            (declare("SINGLE", "slb2/independent-reread"), 1.0, ("II", "FALLBACK", "FIRST_ONLY")),
            (declare("TELEPATHY", "slb2/independent-reread"), 1.0, ("II", "FALLBACK", "FIRST_ONLY")),
            (ProviderDeclaration("p", "INDEPENDENT_REREAD", 3, slb2_digest("slb2/independent-reread")), 0.0,
             ("II", "FALLBACK", "FIRST_ONLY")),
            ("not a declaration", 0.0, ("II", "FALLBACK", "FIRST_ONLY")),
        ]
        for decl, rho_admit, want in cases:
            got = admit(decl, REGISTRY, rho_admit)
            self.assertEqual((got.contract, got.configuration, got.mode), want, (decl, rho_admit, got.reasons))


class TestHostEndToEnd(unittest.TestCase):
    def setUp(self):
        try:
            native.library()
        except native.NativeUnavailable as exc:
            if REQUIRE:
                raise
            self.skipTest(str(exc))

    def host(self, decl, rho_admit=0.0, backend="native"):
        h = ResearchHost(FROZEN, REGISTRY, rho_admit, backend)
        h.initialize(decl)
        return h

    def parity(self, regime, decl, behaviour, config, mode, n=48):
        batch = env2.episodes("DEV", 252, regime, 0, n)
        cfg = FROZEN[config]
        agent = agents2.BayesAgent2(cfg["hypotheses"], cfg["prior"])
        ref = harness2.rollout2(agent, batch, env2.CONTRACTS["DUP2_4"], mode)
        h = self.host(decl)
        self.assertEqual((h.admission.configuration, h.admission.mode), (config, mode))
        mismatches = 0
        for e in range(n):
            provider = SimulatedProvider(batch, e, decl, behaviour, FROZEN["reads"])
            got = run_episode(h, batch, e, provider)
            mismatches += sum(int(a != b) for a, b in zip(got, ref.decisions[e]))
        self.assertLessEqual(mismatches, 1, (regime, config))       # near-ties only (see test_v2_native)

    def test_contract_one_lifecycle_matches_frozen_primary(self):
        for regime in ("CLEAN", "FLIP40", "FLIP45"):
            self.parity(regime, honest_independent(), "reread", "PRIMARY", "DUPLICATE")

    def test_contract_two_lifecycle_matches_frozen_fallback(self):
        for regime, decl, behaviour in (("EVENT40", declare("SHARED_SOURCE", "slb2/shared-source"), "reread"),
                                        ("FLIP40", declare("INDEPENDENT_REREAD", None), "reread"),
                                        ("CLEAN", declare("COPY", "slb2/independent-reread"), "copy")):
            self.parity(regime, decl, behaviour, "FALLBACK", "FIRST_ONLY")

    def test_scalar_backend_agrees_with_native_backend(self):
        batch = env2.episodes("DEV", 251, "FLIP40", 0, 12)
        for e in range(batch.count):
            prov = lambda: SimulatedProvider(batch, e, honest_independent(), "reread", FROZEN["reads"])  # noqa: E731
            self.assertEqual(run_episode(self.host(honest_independent()), batch, e, prov()),
                             run_episode(self.host(honest_independent(), backend="scalar"), batch, e, prov()))

    def test_integrity_violations_fail_closed(self):
        batch = env2.episodes("DEV", 253, "FLIP40", 0, 4)
        expected = {"replay_reading_id": "READING_REPLAYED", "same_channel": "REREAD_ON_SAME_CHANNEL",
                    "foreign_event": "REREAD_OF_ANOTHER_EVENT", "wrong_expert": "READING_FOR_UNSELECTED_EXPERT",
                    "bad_value": "READING_VALUE", "stale_event": "STALE_EVENT"}
        for tamper, reason in expected.items():
            h = self.host(honest_independent())
            provider = SimulatedProvider(batch, 0, honest_independent(), "reread", FROZEN["reads"], tamper=tamper)
            with self.assertRaises(HostError) as cm:
                run_episode(h, batch, 0, provider)
            self.assertEqual(str(cm.exception), reason)
            self.assertEqual(h.learner.state_bytes(), bytes(len(h.learner.state_bytes())))   # zeroised
            with self.assertRaises(HostError):
                h.propose_expert(batch.x[0, 0])                                              # refused until reset
            run_episode(h, batch, 1, SimulatedProvider(batch, 1, honest_independent(), "reread", FROZEN["reads"]))

    def test_lifecycle_and_input_contracts_fail_closed(self):
        batch = env2.episodes("DEV", 253, "CLEAN", 0, 2)
        h = ResearchHost(FROZEN, REGISTRY)
        with self.assertRaises(HostError):
            h.reset_episode(batch.router_w)                                   # NOT_INITIALIZED
        h.initialize(honest_independent())
        with self.assertRaises(HostError):
            h.initialize(honest_independent())
        with self.assertRaises(HostError):
            h.reset_episode(np.zeros((3, 3)))
        checks = [
            ("SELECT_REQUIRED", lambda: h.observe_selected(0, [Reading("e", "r", "c", 0, 1)])),
            ("CONTEXT_CONTRACT", lambda: h.propose_expert(np.array([0.0, np.nan, 0.0, 0.0]))),
            ("CONTEXT_CONTRACT", lambda: h.propose_expert(np.zeros(3))),
            ("TERMINAL_PREMATURE", lambda: h.propose_terminal_decision(batch.xq[0])),
        ]
        for reason, call in checks:
            h.reset_episode(batch.router_w)
            with self.assertRaises(HostError) as cm:
                call()
            self.assertEqual(str(cm.exception), reason)
        h.reset_episode(batch.router_w)
        a = h.propose_expert(batch.x[0, 0])
        with self.assertRaises(HostError) as cm:
            h.propose_expert(batch.x[0, 0])
        self.assertEqual(str(cm.exception), "OBSERVATION_MISSING")
        h.reset_episode(batch.router_w)
        a = h.propose_expert(batch.x[0, 0])
        with self.assertRaises(HostError) as cm:
            h.observe_selected((a + 1) % 3, [Reading("e", "r", "c", (a + 1) % 3, 1)])
        self.assertEqual(str(cm.exception), "SELECTION_NOT_ACCEPTED")
        h.reset_episode(batch.router_w)
        a = h.propose_expert(batch.x[0, 0])
        rs = [Reading("e", f"r{i}", f"c{i}", a, 1) for i in range(3)]
        with self.assertRaises(HostError) as cm:
            h.observe_selected(a, rs)
        self.assertEqual(str(cm.exception), "READING_COUNT")
        h.reset_episode(batch.router_w)
        for t in range(8):
            a = h.propose_expert(batch.x[0, t])
            h.observe_selected(a, [Reading(f"e{t}", f"r{t}", "c", a, 1)])
        with self.assertRaises(HostError) as cm:
            h.propose_expert(batch.x[0, 0])
        self.assertEqual(str(cm.exception), "HORIZON_EXHAUSTED")
        h.reset_episode(batch.router_w)
        for t in range(8):
            a = h.propose_expert(batch.x[0, t])
            h.observe_selected(a, [Reading(f"e{t}", f"r{t}", "c", a, t % 2)])
        with self.assertRaises(HostError) as cm:
            h.propose_terminal_decision(np.zeros((65, 4)))
        self.assertEqual(str(cm.exception), "QUERY_CONTRACT")
        summary = h.close_episode()
        self.assertEqual(summary["status"], "FAILED")

    def test_contract_two_never_passes_a_second_reading(self):
        class Counting:
            def __init__(self, inner):
                self.inner, self.sizes = inner, []

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def update(self, a, z, values, advance=True):
                self.sizes.append(len(values))
                return self.inner.update(a, z, values, advance)
        batch = env2.episodes("DEV", 252, "EVENT40", 0, 3)
        decl = declare("SHARED_SOURCE", "slb2/shared-source")
        h = self.host(decl)
        h.learner = Counting(h.learner)
        for e in range(3):
            run_episode(h, batch, e, SimulatedProvider(batch, e, decl, "reread", FROZEN["reads"]))
        self.assertEqual(set(h.learner.sizes), {1})
        h1 = self.host(honest_independent())
        h1.learner = Counting(h1.learner)
        run_episode(h1, batch, 0, SimulatedProvider(batch, 0, honest_independent(), "reread", FROZEN["reads"]))
        self.assertEqual(h1.learner.sizes, [2, 2, 2, 2, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
