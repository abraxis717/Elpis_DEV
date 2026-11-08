"""Deployment-contract tests for every arm: causality, hidden-variable poisoning, privilege refusal, reset,
bounded state, frozen coefficients, deterministic replay."""
import dataclasses
import unittest

import numpy as np

from cgc_lao_lab_tests import LAB  # noqa: F401  (puts the lab on sys.path)
from cgc_lao_lab import agents as ag
from cgc_lao_lab import arms, cgc, env, harness, rng
from cgc_lao_lab.candidates import CANDIDATES

ORIGINAL = env.CONTRACTS["ORIGINAL"]
BUDGET = 64


def frozen_unit_weights():
    return cgc.frozen_weights(0.2 * rng.normal(rng.derive_key("unit", "boundary-w"), 120))


def admissible_factories():
    w = frozen_unit_weights()
    facs = {
        "NOOP": arms.factory("NOOP"),
        "SGD": arms.factory("SGD", sgd_eta=0.5),
        "BETA": arms.factory("BETA", beta_kappa=8.0),
        "CGC_REFERENCE": arms.factory("CGC_REFERENCE", cgc_w=w),
        "STATIC_ATTENUATION": arms.factory("STATIC_ATTENUATION", cgc_w=w, attenuation=0.5),
        "CLEAN_BAYES": arms.factory("CLEAN_BAYES"),
    }
    for name in CANDIDATES:
        facs[name] = arms.factory(name, cgc_w=w)
    return facs


def contracts_for(agent):
    if isinstance(agent, (ag.CGCAgent, ag.GatedCGCAgent)):
        return [ORIGINAL]
    return [ORIGINAL, env.CONTRACTS["DUP2"], env.CONTRACTS["T12"]]


def poisoned(batch: env.EpisodeBatch, actions: np.ndarray, contract) -> env.EpisodeBatch:
    """Same selected readings; every unselected potential outcome inverted; hidden truth destroyed."""
    y = 1 - batch.y
    taken = np.zeros_like(y, dtype=bool)
    for t in range(contract.horizon):
        taken[np.arange(batch.count), t, actions[:, t], :] = True
    y = np.where(taken, batch.y, y).astype(np.uint8)
    nan = lambda a: np.full_like(a, np.nan, dtype=np.float64)  # noqa: E731
    return dataclasses.replace(batch, y=y, b=nan(batch.b), clean=1 - batch.clean, flips=1 - batch.flips,
                               q_episode=nan(batch.q_episode), regime=env.REGIMES["CLEAN"], access_log=[])


class TestDeploymentBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facs = admissible_factories()
        cls.batch = env.episodes("DEV", 5, "FLIP40", 0, 48)

    def run_arm(self, name, batch=None, contract=ORIGINAL, regime=None):
        batch = batch if batch is not None else self.batch
        agent, priv = self.facs[name](regime or batch.regime, contract)
        self.assertIsNone(priv)
        return agent, harness.rollout(agent, batch, contract)

    def test_state_budget_and_fixed_size(self):
        for name, fac in self.facs.items():
            agent, _ = fac(None, ORIGINAL)
            for contract in contracts_for(agent):
                agent, _ = fac(None, contract)
                agent.begin(self.batch.router_w, self.batch.count, contract)
                sizes0 = {k: v.shape for k, v in agent.volatile_state().items()}
                harness.rollout(agent, self.batch, contract)
                state = agent.volatile_state()
                self.assertEqual({k: v.shape for k, v in state.items()}, sizes0, name)
                per_episode = sum(v.size for v in state.values()) // self.batch.count
                self.assertTrue(all(v.shape[0] == self.batch.count for v in state.values()), name)
                self.assertEqual(per_episode, agent.state_scalars(), name)
                self.assertLessEqual(agent.state_scalars(), BUDGET, name)

    def test_deterministic_replay(self):
        for name in self.facs:
            _, a = self.run_arm(name)
            _, b = self.run_arm(name)
            self.assertTrue(np.array_equal(a.actions, b.actions) and np.array_equal(a.decisions, b.decisions), name)
            self.assertTrue(np.array_equal(a.terminal, b.terminal), name)

    def test_reset_leaves_no_carry_over(self):
        other = env.episodes("DEV", 6, "CLEAN", 0, 48)
        for name, fac in self.facs.items():
            agent, _ = fac(None, ORIGINAL)
            harness.rollout(agent, self.batch, ORIGINAL)
            reused = harness.rollout(agent, other, ORIGINAL)
            fresh = harness.rollout(fac(None, ORIGINAL)[0], other, ORIGINAL)
            self.assertTrue(np.array_equal(reused.decisions, fresh.decisions), name)
            self.assertTrue(np.array_equal(reused.actions, fresh.actions), name)

    def test_hidden_variable_poisoning_changes_nothing(self):
        for name, fac in self.facs.items():
            for contract in contracts_for(fac(None, ORIGINAL)[0]):
                _, ref = self.run_arm(name, contract=contract)
                bad = poisoned(self.batch, ref.actions, contract)
                agent, _ = fac(env.REGIMES["CLEAN"], contract)   # a wrong regime label is ignored as well
                pub_before = (bad.x.copy(), bad.xq.copy())
                ag_actions, ag_decisions = [], []
                agent.begin(bad.router_w, bad.count, contract)
                for t in range(contract.horizon):
                    a = agent.act(t, bad.x[:, t])
                    agent.observe(t, bad.x[:, t], a, bad.readings(t, a, contract.reads[t]))
                    ag_actions.append(a)
                ag_decisions = agent.decide(bad.xq)
                self.assertTrue(np.array_equal(np.stack(ag_actions, 1), ref.actions), (name, contract.name))
                self.assertTrue(np.array_equal(ag_decisions, ref.decisions), (name, contract.name))
                self.assertTrue(np.array_equal(pub_before[0], bad.x) and np.array_equal(pub_before[1], bad.xq))

    def test_causality_future_readings_do_not_change_past_actions(self):
        for name in self.facs:
            _, ref = self.run_arm(name)
            for t_cut in (2, 5):
                y = self.batch.y.copy()
                y[:, t_cut + 1:] = 1 - y[:, t_cut + 1:]
                alt = dataclasses.replace(self.batch, y=y, access_log=[])
                _, out = self.run_arm(name, batch=alt)
                self.assertTrue(np.array_equal(out.actions[:, :t_cut + 2], ref.actions[:, :t_cut + 2]), (name, t_cut))

    def test_only_selected_readings_are_accessed(self):
        _, out = self.run_arm("RMF_U3")
        log = self.batch.access_log[-8:]
        for t, (step, acts, reads) in enumerate(log):
            self.assertEqual((step, reads), (t, 1))
            self.assertTrue(np.array_equal(acts, out.actions[:, t]))

    def test_privileged_inputs_are_refused_by_admissible_agents(self):
        for name, fac in self.facs.items():
            agent, _ = fac(None, ORIGINAL)
            self.assertFalse(agent.privileged, name)
            with self.assertRaises(PermissionError):
                agent.begin(self.batch.router_w, 4, ORIGINAL, {"q_override": np.zeros((4, 1))})
        oracle, _ = arms.factory("KNOWN_Q_ORACLE")(env.REGIMES["FLIP40"], ORIGINAL)
        self.assertTrue(oracle.privileged)
        self.assertNotIn("KNOWN_Q_ORACLE", arms.BASELINES)

    def test_frozen_coefficients_are_not_mutated(self):
        w = frozen_unit_weights()
        before = cgc.weights_digest(w)
        for attenuation in (1.0, 0.5):
            agent = ag.CGCAgent(w, attenuation)
            harness.rollout(agent, self.batch, ORIGINAL)
            self.assertFalse(agent.w.flags.writeable)
            self.assertEqual(cgc.weights_digest(agent.w), before)
        agent = arms.factory("RMF_U3")(None, ORIGINAL)[0]
        qs, prior = agent.filter.qs.copy(), agent.filter.prior.copy()
        harness.rollout(agent, self.batch, ORIGINAL)
        self.assertTrue(np.array_equal(agent.filter.qs, qs) and np.array_equal(agent.filter.prior, prior))

    def test_fixed_log_mode_replays_supplied_actions(self):
        _, noop = self.run_arm("NOOP")
        agent, _ = self.facs["RMF_U3"](None, ORIGINAL)
        out = harness.rollout(agent, self.batch, ORIGINAL, actions=noop.actions)
        self.assertTrue(np.array_equal(out.actions, noop.actions))
        self.assertTrue(np.allclose(out.online, noop.online))

    def test_noop_terminal_decision_ignores_feedback(self):
        _, a = self.run_arm("NOOP")
        inverted = dataclasses.replace(self.batch, y=(1 - self.batch.y).astype(np.uint8), access_log=[])
        _, b = self.run_arm("NOOP", batch=inverted)
        self.assertTrue(np.array_equal(a.decisions, b.decisions))
        _, c = self.run_arm("RMF_U3", batch=inverted)
        _, d = self.run_arm("RMF_U3")
        self.assertFalse(np.array_equal(c.decisions, d.decisions))   # a learner does use its feedback


if __name__ == "__main__":
    unittest.main()
