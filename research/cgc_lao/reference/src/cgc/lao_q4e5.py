"""Opt-in CGC-LAO Q4E5 research reference: PROBE_EARLY (3 experts, T=8).

Qualified only for the Q4E5 synthetic selected-expert feedback experiment.
This module adds no online weight training, adaptive HALT, terminal STOP policy,
or hidden-state/counterfactual access. The learned CGC controller weights are
supplied externally as an already-frozen inference module.

Historical scientific evidence belongs to the separate Q4E5 qualification ZIP.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

EXPERTS = 3
HORIZON = 8
STEP_CAP = 0.25
STATE_CAP = 1.5
MAX_BATCH = 4096
MAX_QUERIES = 64


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def choose_probe_early(
    t: int, base_logits: torch.Tensor, state: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Q4E5 acquisition selector, returning one expert index per episode.

    Three-expert, fixed-eight-observation policy. Arguments are PUBLIC router
    logits W*x, bounded transient correction state, and observed-expert counts.
    The policy never receives latent biases, reward counterfactuals, corruption
    rates, future terminal queries, or a trained model's parameters.

    State update and STOP/HALT are deliberately outside this pure function.
    """
    _require(isinstance(t, int) and not isinstance(t, bool) and 0 <= t < HORIZON,
             'Q4E5_STEP_OUT_OF_RANGE')
    _require(all(isinstance(z, torch.Tensor) for z in (base_logits, state, counts)),
             'Q4E5_TENSORS_REQUIRED')
    _require(base_logits.ndim == 2 and base_logits.shape[1] == EXPERTS
             and 1 <= base_logits.shape[0] <= MAX_BATCH,
             'Q4E5_LOGIT_SHAPE_B3')
    _require(state.shape == base_logits.shape and counts.shape == base_logits.shape,
             'Q4E5_STATE_COUNT_SHAPE_B3')
    _require(base_logits.dtype in (torch.float32, torch.float64)
             and state.dtype == base_logits.dtype,
             'Q4E5_FLOAT_DTYPE')
    _require(counts.dtype in (torch.float32, torch.float64, torch.int32, torch.int64),
             'Q4E5_COUNT_DTYPE')
    _require(base_logits.device == state.device == counts.device,
             'Q4E5_DEVICE_MISMATCH')
    _require(all(_finite(z) for z in (base_logits, state, counts)), 'Q4E5_NONFINITE')
    _require(bool(((counts >= 0) & (counts == counts.floor())).all())
             and bool((counts.sum(-1) == t).all()),
             'Q4E5_COUNTS_DO_NOT_MATCH_STEP')
    _require(bool((state.abs() <= STATE_CAP + 1e-6).all()), 'Q4E5_STATE_CAP')

    # Keep the exact Q4E4/Q4E5 sigmoid/rank/mask/tie semantics.
    pred = (base_logits + state).sigmoid()
    best = pred.argmax(-1)
    unseen = counts == 0
    need = unseen.sum(-1)
    unseen_best = pred.masked_fill(~unseen, -1e9).argmax(-1)
    remaining = HORIZON - t
    deadline_force = need >= remaining
    offer = t in (1, 2)
    probe = (torch.ones_like(deadline_force) if offer
             else torch.zeros_like(deadline_force)) & (need > 0)
    return torch.where(deadline_force | probe, unseen_best, best)


class FrozenCGC3(nn.Module):
    """Float32, 3-output/120-scalar Q4E5 CGC forward with frozen RAM weights.

    NOT the 4-output CGC120 wrapper; no training/checkpoint synthesis here.
    The supplied [3,5,8] weights must come from separately verified training.
    The buffer is nonpersistent so this wrapper is not a checkpoint authority.
    """

    def __init__(self, trained_weights: torch.Tensor):
        super().__init__()
        _require(isinstance(trained_weights, torch.Tensor)
                 and trained_weights.shape == (3, 5, 8)
                 and trained_weights.dtype == torch.float32
                 and _finite(trained_weights), 'Q4E5_CGC_WEIGHT_CONTRACT')
        self.register_buffer('w', trained_weights.detach().clone(), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _require(isinstance(features, torch.Tensor) and features.ndim == 2
                 and features.shape[1] == 5
                 and features.dtype == self.w.dtype
                 and features.device == self.w.device, 'Q4E5_CGC_FEATURE_CONTRACT')
        z = torch.einsum('bd,odn->bon', features, self.w)
        return 0.5 * (z + z.square() + z.pow(3)).sum(-1)


class ProbeEarlyEpisode:
    """Bounded volatile on-policy feedback episode for the frozen CGC adapter.

    Each call to select(context[B,4]) must be paired with observe(reward[B]).
    The caller owns the external reward environment. Only selected rewards may
    be supplied to observe. A fresh instance begins with empty episode state.
    This object does not perform a terminal STOP action or choose a HALT time.
    """

    def __init__(self, router_w: torch.Tensor, frozen_controller: nn.Module,
                 batch_size: int):
        _require(isinstance(router_w, torch.Tensor)
                 and router_w.shape == (EXPERTS, 4)
                 and router_w.dtype == torch.float32 and _finite(router_w),
                 'Q4E5_ROUTER_CONTRACT')
        _require(isinstance(batch_size, int) and not isinstance(batch_size, bool)
                 and 1 <= batch_size <= MAX_BATCH, 'Q4E5_BATCH_CAP')
        _require(isinstance(frozen_controller, nn.Module), 'Q4E5_MODULE_REQUIRED')
        _require(not frozen_controller.training and
                 not any(p.requires_grad for p in frozen_controller.parameters()),
                 'Q4E5_CONTROLLER_MUST_BE_FROZEN_EVAL')
        for tensor in list(frozen_controller.parameters()) + list(frozen_controller.buffers()):
            _require(tensor.device == router_w.device and _finite(tensor),
                     'Q4E5_CONTROLLER_DEVICE_OR_NONFINITE')
        self.router_w = router_w.detach().clone()  # bounded immutable source copy
        self.controller = frozen_controller
        self.state = torch.zeros(batch_size, EXPERTS, dtype=router_w.dtype,
                                 device=router_w.device)
        self.counts = torch.zeros_like(self.state)
        self.t = 0
        self._pending = None

    def select(self, context: torch.Tensor) -> torch.Tensor:
        _require(self._pending is None, 'Q4E5_OBSERVATION_MISSING')
        _require(self.t < HORIZON, 'Q4E5_HORIZON_EXHAUSTED')
        _require(isinstance(context, torch.Tensor)
                 and context.shape == (self.state.shape[0], 4)
                 and context.dtype == self.state.dtype
                 and context.device == self.state.device
                 and _finite(context), 'Q4E5_CONTEXT_CONTRACT')
        with torch.no_grad():
            base = torch.einsum('nd,ed->ne', context, self.router_w)
            action = choose_probe_early(self.t, base, self.state, self.counts)
            self._pending = (base, action.clone())
            return action.clone()

    def observe(self, selected_reward: torch.Tensor) -> None:
        _require(self._pending is not None, 'Q4E5_SELECT_REQUIRED')
        _require(isinstance(selected_reward, torch.Tensor)
                 and selected_reward.shape == (self.state.shape[0],)
                 and selected_reward.dtype == self.state.dtype
                 and selected_reward.device == self.state.device
                 and _finite(selected_reward)
                 and bool(((selected_reward == 0) | (selected_reward == 1)).all()),
                 'Q4E5_SELECTED_BERNOULLI_REWARD_REQUIRED')
        base, action = self._pending
        with torch.no_grad():
            pred = (base + self.state).sigmoid()
            onehot = F.one_hot(action, EXPERTS).to(self.state.dtype)
            residual = pred.gather(1, action[:, None]).squeeze(1) - selected_reward
            grad = 3 * onehot * residual[:, None]
            surprise = residual.square()[:, None]
            clock = torch.full_like(surprise, (self.t + 1) / HORIZON)
            features = torch.cat((torch.tanh(grad), torch.tanh(surprise), clock), -1)
            delta = STEP_CAP * torch.tanh(self.controller(features))
            _require(delta.shape == self.state.shape and delta.dtype == self.state.dtype
                     and delta.device == self.state.device and _finite(delta),
                     'Q4E5_CONTROLLER_OUTPUT_CONTRACT')
            next_state = (self.state - delta.clamp(-STEP_CAP, STEP_CAP)).clamp(
                -STATE_CAP, STATE_CAP)
            self.state = next_state
            self.counts = self.counts + onehot
            self.t += 1
            self._pending = None

    def terminal_expert_probabilities(self, queries: torch.Tensor) -> torch.Tensor:
        _require(self.t == HORIZON and self._pending is None, 'Q4E5_TERMINAL_PREMATURE')
        _require(bool((self.counts > 0).all()), 'Q4E5_COVERAGE_FAILURE')
        _require(isinstance(queries, torch.Tensor) and queries.ndim == 3
                 and 1 <= queries.shape[0] <= MAX_QUERIES
                 and queries.shape[1:] == (self.state.shape[0], 4)
                 and queries.dtype == self.state.dtype
                 and queries.device == self.state.device and _finite(queries),
                 'Q4E5_QUERY_CONTRACT')
        with torch.no_grad():
            base = torch.einsum('qnd,ed->qne', queries, self.router_w)
            return (base + self.state.unsqueeze(0)).sigmoid()
