"""Q4E5 parity and fail-closed boundary tests; torch CPU only."""
import math
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from cgc.lao_q4e5 import (
    FrozenCGC3, ProbeEarlyEpisode, choose_probe_early, HORIZON, EXPERTS,
    STEP_CAP, STATE_CAP,
)


def reference_choose_action(t, state, count, base):
    """Verbatim Q4E4/Q4E5 selector semantics for PROBE_EARLY (lines 462-498)."""
    pred = (base + state).sigmoid()
    best = pred.argmax(-1)
    unseen = count == 0
    need = unseen.sum(-1)
    unseen_best = pred.masked_fill(~unseen, -1e9).argmax(-1)
    deadline_force = need >= (8 - t)
    offer = t in (1, 2)
    probe = (torch.ones_like(deadline_force) if offer
             else torch.zeros_like(deadline_force)) & (need > 0)
    return torch.where(deadline_force | probe, unseen_best, best)


class TestLAOQ4E5(unittest.TestCase):
    def test_selector_parity_all_steps(self):
        g = torch.Generator().manual_seed(62023)
        for t in range(HORIZON):
            for size in (1, 3, 32):
                base = torch.randn(size, 3, generator=g)
                state = torch.randn(size, 3, generator=g).clamp(-STATE_CAP, STATE_CAP)
                # Every row represents exactly t acquired observations.
                observations = torch.randint(0, 3, (size, t), generator=g)
                count = torch.zeros(size, 3)
                for idx in range(t):
                    count.scatter_add_(1, observations[:, idx:idx+1],
                                       torch.ones(size, 1))
                got = choose_probe_early(t, base, state, count)
                expected = reference_choose_action(t, state, count, base)
                self.assertTrue(torch.equal(got, expected), (t, size))

    def test_tie_and_saturated_logit_parity(self):
        for t in range(HORIZON):
            base = torch.tensor([[90., 91., 92.], [0., 0., 0.]])
            state = torch.zeros_like(base)
            count = torch.zeros_like(base)
            if t > 0:
                count[:, 0] = float(t)
            self.assertTrue(torch.equal(choose_probe_early(t, base, state, count),
                                        reference_choose_action(t, state, count, base)))

    def test_full_episode_reference_parity_and_coverage(self):
        g = torch.Generator().manual_seed(88711)
        n = 27
        w = 0.65 * torch.randn(3, 4, generator=g)
        trained = 0.05 * torch.randn(3, 5, 8, generator=g)
        model = FrozenCGC3(trained).eval()
        self.assertEqual(trained.numel(), 120)
        self.assertEqual(list(model.parameters()), [])
        episode = ProbeEarlyEpisode(w, model, n)
        state = torch.zeros(n, 3)
        count = torch.zeros_like(state)
        context = torch.randn(8, n, 4, generator=g)
        query = torch.randn(16, n, 4, generator=g)
        potential = (torch.rand(8, n, 3, generator=g) > .48).float()
        # The full potential-outcome tensor is used only by this test harness.
        for t in range(HORIZON):
            base = torch.einsum('nd,ed->ne', context[t], w)
            act_ref = reference_choose_action(t, state, count, base)
            act = episode.select(context[t])
            self.assertTrue(torch.equal(act, act_ref))
            selected = potential[t].gather(1, act_ref[:, None]).squeeze(1)
            pred = (base + state).sigmoid()
            onehot = F.one_hot(act_ref, 3).to(state.dtype)
            residual = pred.gather(1, act_ref[:, None]).squeeze(1) - selected
            grad = 3 * onehot * residual[:, None]
            surprise = residual.square()[:, None]
            clock = torch.full_like(surprise, (t + 1) / HORIZON)
            feat = torch.cat((torch.tanh(grad), torch.tanh(surprise), clock), -1)
            z = torch.einsum('bd,odn->bon', feat, trained)
            delta = STEP_CAP * torch.tanh(0.5 * (z + z.square() + z.pow(3)).sum(-1))
            state = (state - delta.clamp(-STEP_CAP, STEP_CAP)).clamp(
                -STATE_CAP, STATE_CAP)
            count = count + onehot
            episode.observe(selected)
            self.assertTrue(torch.equal(state, episode.state), f'state step {t}')
            self.assertTrue(torch.equal(count, episode.counts), f'count step {t}')
        self.assertEqual(episode.t, HORIZON)
        self.assertTrue(bool((count > 0).all()))
        ref_prob = (torch.einsum('qnd,ed->qne', query, w)
                    + state.unsqueeze(0)).sigmoid()
        self.assertTrue(torch.equal(ref_prob, episode.terminal_expert_probabilities(query)))
        self.assertEqual(episode.state.shape, (n, 3))

    def test_invalid_inputs_fail_closed(self):
        w = torch.zeros(3, 4)
        model = FrozenCGC3(torch.zeros(3, 5, 8)).eval()
        episode = ProbeEarlyEpisode(w, model, 2)
        with self.assertRaisesRegex(ValueError, 'SELECT_REQUIRED'):
            episode.observe(torch.zeros(2))
        with self.assertRaisesRegex(ValueError, 'TERMINAL_PREMATURE'):
            episode.terminal_expert_probabilities(torch.zeros(1, 2, 4))
        episode.select(torch.zeros(2, 4))
        with self.assertRaisesRegex(ValueError, 'OBSERVATION_MISSING'):
            episode.select(torch.zeros(2, 4))
        with self.assertRaisesRegex(ValueError, 'BERNOULLI'):
            episode.observe(torch.tensor([0., .2]))
        episode.observe(torch.zeros(2))
        with self.assertRaisesRegex(ValueError, 'COUNTS_DO_NOT_MATCH_STEP'):
            choose_probe_early(3, torch.zeros(2, 3), torch.zeros(2, 3),
                               torch.zeros(2, 3))
        with self.assertRaisesRegex(ValueError, 'DEVICE_MISMATCH|FLOAT_DTYPE'):
            choose_probe_early(0, torch.zeros(2, 3, dtype=torch.float64),
                               torch.zeros(2, 3), torch.zeros(2, 3))
        with self.assertRaisesRegex(ValueError, 'MUST_BE_FROZEN_EVAL'):
            ProbeEarlyEpisode(w, nn.Linear(5, 3).eval(), 2)

    def test_no_model_weight_updates_or_storage_writes(self):
        g = torch.Generator().manual_seed(19)
        w = torch.randn(3, 4, generator=g)
        model = FrozenCGC3(torch.randn(3, 5, 8, generator=g)).eval()
        before = model.w.clone()
        e = ProbeEarlyEpisode(w, model, 4)
        for t in range(8):
            e.select(torch.randn(4, 4, generator=g))
            e.observe(torch.randint(0, 2, (4,), generator=g).float())
        self.assertTrue(torch.equal(model.w, before))
        self.assertEqual(len(list(model.parameters())), 0)
        self.assertEqual(model.state_dict(), {})


if __name__ == '__main__':
    unittest.main()
