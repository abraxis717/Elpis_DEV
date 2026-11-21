"""R12: the Stage C seam harness. RESEARCH_ONLY. R11's Stage C regression stack
(``specialist_stack.RegressionStack``, unchanged), with the episode-local fast update of its LAST block's specialist
core (``fast_state``).

The episode (R11 Stage C, unchanged)
    The context holds ``k`` support tokens ``[x, y, 0]``. Then come 48 query tokens ``[x, 0, 1]``, and the model
    predicts ``y`` at the queries.

The teaching signal (the leakage contract)
    The fast update learns from the SAME ``k`` observed support pairs the context holds, but through PROBES: the
    support inputs in query format ``[x, 0, 1]``, run WITHOUT any context.
    - A probe token carries no target value (asserted).
    - A probe pass contains no support context, so no probe can read its own target from the sequence. The support
      loss therefore trains predictions that never saw their targets.
    - Query targets are read only to score, after the update.
    (``test_r12.py`` shows why probes must be context-free: after a context that holds the pairs, a probe's
    prediction moves with its own target.)

Two paths, one mathematics
    - **Evaluation** runs one transactional ``SpecialistFastState`` per episode (``session.bounded_update``).
    - **Training** (the episodic objective) runs the same 8 clipped SGD steps for a batch of episodes at once
      (``torch.func.vmap``), each episode with its own clip.
    ``test_r12.py`` checks the two against each other.
"""
from __future__ import annotations

import math
import statistics
import time

import torch
from torch.func import functional_call

from . import specialist_stack as ss
from . import specialist_tasks as st
from .cgc_specialist import specialist_core_names
from .fast_state import (FastStateConfig, ObservedSupport, SpecialistFastState, frozen_pass, head, last_channel)

TARGET_SLOT = slice(st.INPUTS, st.INPUTS + st.OUTPUTS)      # the y columns of a token [x (5), y (3), is_query (1)]
STEP_GRID = (0, 1, 2, 4, 8)


# -- inputs -------------------------------------------------------------------------------------------------------

def probe_inputs(namespace: str, family: str, seed: int, split: str, index: int, k: int):
    """The episode's first ``k`` support inputs as query-format tokens ``[x, 0, 1]`` (no context) and their observed
    targets: ``([1, k, 9], [1, k, 3])``. No probe token carries a target value."""
    e = st.teacher(namespace, family, seed, split, index)
    sx, sy = e.support[0][:k].float(), e.support[1][:k].float()
    tokens = torch.cat([sx, torch.zeros(k, st.OUTPUTS), torch.ones(k, 1)], dim=-1)
    if bool(tokens[:, TARGET_SLOT].any()):
        raise AssertionError("a probe token carries a target value")
    return tokens[None], sy[None]


def batch_probes(namespace, families, seed, split, start, size, k):
    """Probes of ``size`` episodes, laid out like ``specialist_stack.batch_tokens`` (family ``b % len(families)``)."""
    rows, targets = [], []
    for b in range(size):
        tokens, y = probe_inputs(namespace, families[b % len(families)], seed, split, start + b, k)
        rows.append(tokens[0])
        targets.append(y[0])
    return torch.stack(rows), torch.stack(targets)


def ood_inputs(namespace: str, family: str, seed: int, split: str, index: int, k: int):
    """``k`` support tokens then the episode's 32 OOD inputs (x 1.4) as queries: ``([1, k + 32, 9], [1, 32, 3])``."""
    e = st.teacher(namespace, family, seed, split, index)
    sx, sy = e.support[0][:k].float(), e.support[1][:k].float()
    ox, oy = e.ood[0].float(), e.ood[1].float()
    support = torch.cat([sx, sy, torch.zeros(k, 1)], dim=-1)
    queries = torch.cat([ox, torch.zeros(ox.shape[0], st.OUTPUTS), torch.ones(ox.shape[0], 1)], dim=-1)
    return torch.cat([support, queries])[None], oy[None]


def r_squared(features: torch.Tensor, inputs: torch.Tensor) -> float:
    """Share of the variance of ``features [n, f]`` explained linearly by ``inputs [n, d]`` (least squares)."""
    f, x = features.double(), inputs.double()
    coef = torch.linalg.lstsq(x, f).solution
    residual = f - x @ coef
    centred = f - f.mean(0, keepdim=True)
    total = float(centred.pow(2).sum())
    return 1.0 - float(residual.pow(2).sum()) / total if total > 0 else math.nan


def _mse(a, b) -> float:
    return float((a - b).pow(2).mean())


def _cos(a: dict, b: dict) -> float:
    dot = sum(float((a[n] * b[n]).sum()) for n in a)
    na = math.sqrt(sum(float(a[n].pow(2).sum()) for n in a))
    nb = math.sqrt(sum(float(b[n].pow(2).sum()) for n in b))
    return dot / (na * nb) if na > 0 and nb > 0 else math.nan


# -- evaluation (transactional fast state, one session per episode) ------------------------------------------------

@torch.no_grad()
def _state_equal(model, tokens, overrides) -> bool:
    """Do the recurrent states a full recomputation leaves behind equal the frozen model's, bit for bit?"""
    frozen = model.new_state(1)
    model(tokens, frozen)
    adapted = model.new_state(1)
    functional_call(model, overrides, (tokens, adapted))
    return all(torch.equal(a, b) for a, b in zip(frozen[0], adapted[0])) and all(
        (a is None and b is None) or torch.equal(a, b) for a, b in zip(frozen[1], adapted[1]))


def evaluate_fast(model, namespace: str, family: str, seed: int, split: str, episodes: int, *,
                  config: FastStateConfig = FastStateConfig(), k: int = 32) -> list:
    """Per test episode: the frozen and fast-adapted query MSE and every control (see ``PREREGISTRATION_R12.md``).

    ``model`` must be frozen. For an FFN-only stack only the frozen measurements exist."""
    has_core = bool(specialist_core_names(last_channel(model)[2]))
    rows, deltas, pres = [], [], []
    for index in range(episodes):
        tokens, y = ss.episode_tokens(namespace, family, seed, split, index, k)
        tokens, y = tokens[None], y[None]
        free_tokens, free_y = ss.episode_tokens(namespace, family, seed, split, index, 0)
        free_tokens, free_y = free_tokens[None], free_y[None]
        ood_tokens, ood_y = ood_inputs(namespace, family, seed, split, index, k)
        e = st.teacher(namespace, family, seed, split, index)
        disjoint = not bool((e.support[0][:k, None, :] == e.query[0][None, :, :]).all(-1).any())
        q = y.shape[1]
        out, pre = frozen_pass(model, tokens)
        free_out, free_pre = frozen_pass(model, free_tokens)
        ood_out, ood_pre = frozen_pass(model, ood_tokens)
        row = {"episode": index, "frozen": {"query": _mse(out[:, -q:], y), "ood": _mse(ood_out[:, -ood_y.shape[1]:], ood_y),
                                             "context_free": _mse(free_out, free_y)},
               "null": float(y.pow(2).mean()), "leakage": {"query_support_disjoint": disjoint}}
        if not has_core:
            rows.append(row)
            continue
        probes, probe_y = probe_inputs(namespace, family, seed, split, index, k)
        row["leakage"]["probe_tokens_carry_no_target"] = not bool(probes[..., TARGET_SLOT].any())
        support = ObservedSupport.observed(probes, probe_y, source=f"teacher {namespace}/{family}/{seed}/{split}/{index}")
        _, probe_pre = frozen_pass(model, probes)
        session = f"{namespace}/{family}/{seed}/{split}/{index}"
        fast = SpecialistFastState(model, session=session, config=config)
        loss_of = fast.support_loss(support, probe_pre)
        curve, trace, increments, previous = {"0": row["frozen"]["query"]}, [], [], fast.delta()
        for step in range(1, config.max_steps + 1):
            (result,) = fast.adapt(loss_of, steps=1)
            trace.append({"step": step, "status": result.status.name, "loss_before": result.loss_before,
                          "loss_after": result.loss_after, "grad_norm": result.grad_norm, "clipped": result.clipped,
                          "projected": result.projected, "delta_norm": result.delta_norm})
            now = fast.delta()
            increments.append({n: now[n] - previous[n] for n in now})
            previous = now
            with torch.no_grad():
                if str(step) in map(str, STEP_GRID):
                    curve[str(step)] = _mse(fast.output(pre)[:, -q:], y)
        with torch.no_grad():
            row["fast"] = curve
            row["fast_ood"] = _mse(fast.output(ood_pre)[:, -ood_y.shape[1]:], ood_y)
            row["fast_context_free"] = _mse(fast.output(free_pre), free_y)
            full = functional_call(model, fast.overrides(), (tokens, model.new_state(1)))
            row["parity"] = {"full_vs_cached_max_abs": float((full - fast.output(pre)).abs().max()),
                             "states_unchanged_by_the_core": _state_equal(model, tokens, fast.overrides())}
            block, _, expert = last_channel(model)
            code = (expert.cgc.proj_in if expert.cgc is not None else expert.mlp.proj_in)(block.channel_norm(probe_pre))
            row["alignment_r2"] = r_squared(code[0], probes[0, :, :st.INPUTS])
        row["trace"] = trace
        row["cosine_successive_updates"] = [_cos(increments[i], increments[i - 1]) for i in range(1, len(increments))]
        deltas.append(fast.delta())
        pres.append((pre, y))
        fast.reset()
        with torch.no_grad():
            reset_out = fast.output(pre)[:, -q:]
        row["reset"] = {"query": _mse(reset_out, y), "exact": bool(torch.equal(reset_out, out[:, -q:]))}
        fast.close()
        off = SpecialistFastState(model, session=session + "/disabled", config=config.replace(learning_enabled=False))
        statuses = [r.status.name for r in off.adapt(off.support_loss(support, probe_pre), steps=config.max_steps)]
        with torch.no_grad():
            off_out = off.output(pre)[:, -q:]
        row["disabled"] = {"query": _mse(off_out, y), "exact": bool(torch.equal(off_out, out[:, -q:])), "status": statuses}
        off.close()
        perm = st.randperm(k, namespace, "shuffle", family, seed, split, index)
        shuffled = ObservedSupport.observed(probes, probe_y[:, perm], source="CONTROL: permuted support targets")
        sh = SpecialistFastState(model, session=session + "/shuffled", config=config)
        sh.adapt(sh.support_loss(shuffled, probe_pre), steps=config.max_steps)
        with torch.no_grad():
            row["shuffled"] = _mse(sh.output(pre)[:, -q:], y)
        sh.close()
        rows.append(row)
    if has_core:              # interference: this episode's queries under the PREVIOUS episode's adapted core
        stale = SpecialistFastState(model, session=f"{namespace}/{family}/{seed}/{split}/interference", config=config)
        with torch.no_grad():
            for index, row in enumerate(rows):
                pre, y = pres[index]
                row["previous_episode_core"] = _mse(stale.output(pre, deltas[index - 1])[:, -y.shape[1]:], y)
        stale.close()
    return rows


# -- training (the episodic objective: batched fast update, first-order outer gradient) ----------------------------

def _detached(module, prefix: str = "") -> dict:
    return {prefix + n: p.detach() for n, p in module.named_parameters()}


def batched_fast_update(model, pre: torch.Tensor, targets: torch.Tensor, config: FastStateConfig) -> dict:
    """R11's inner loop for ``B`` episodes at once: ``config.max_steps`` SGD steps on the last block's specialist core,
    each episode with its own global-norm clip. Returns the adapted cores ``{name: [B, ...]}``, detached.
    The bound of ``FastStateConfig`` cannot bind within its own step cap (documented, tested)."""
    block, channel, expert = last_channel(model)
    names = specialist_core_names(expert)
    params = _detached(expert)
    norm_p, read_p = _detached(model.final_norm), _detached(model.readout)
    batch = pre.shape[0]
    with torch.no_grad():
        h = functional_call(block.channel_norm, _detached(block.channel_norm), (pre,))
    cores = {n: params[n].unsqueeze(0).expand(batch, *params[n].shape).clone() for n in names}

    def loss_one(core, h_b, pre_b, y_b):
        out = functional_call(expert, {**params, **core}, (h_b,))
        pred = functional_call(model.readout, read_p, (functional_call(model.final_norm, norm_p, (pre_b + out,)),))
        return (pred - y_b).pow(2).mean()

    gradient = torch.func.vmap(torch.func.grad(loss_one))
    for _ in range(config.max_steps):
        g = gradient(cores, h, pre, targets)
        norm = torch.sqrt(sum(g[n].pow(2).flatten(1).sum(1) for n in names))
        scale = torch.where(norm > 0, torch.clamp(config.grad_clip / norm, max=1.0), torch.ones_like(norm))
        cores = {n: cores[n] - config.lr * scale.view(batch, *([1] * (cores[n].ndim - 1))) * g[n] for n in names}
    return {n: c.detach() for n, c in cores.items()}


class override_last_core:
    """Within the context, the last block's routed channel computes, for batch element ``b``, its single expert under
    ``cores[name][b]`` (``functional_call`` through ``ChannelExpert``, vectorized over episodes). Equal to the routed
    channel itself when every core is the base (one expert, weight exactly 1; tested)."""

    def __init__(self, model, cores: dict):
        self.model, self.cores = model, cores

    def __enter__(self):
        _, channel, expert = last_channel(self.model)
        live = dict(expert.named_parameters())

        def one(core, h):
            return functional_call(expert, {**live, **core}, (h,))
        batched = torch.func.vmap(one)
        self.handle = channel.register_forward_hook(lambda module, args, out: batched(self.cores, args[0]))
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def adapted_query_mse(model, tokens, targets, probes, probe_targets, config: FastStateConfig, *, first_order=True):
    """``[B]`` query MSE with the last block's core adapted on each episode's probes (``probes`` None: frozen core).
    ``first_order``: the outer gradient reaches the base core at the adapted point (R11 Stage B's FOMAML)."""
    q = targets.shape[1]
    names = specialist_core_names(last_channel(model)[2])
    if probes is None or not names:
        return ss.query_mse(model, tokens, targets)
    _, pre = frozen_pass(model, probes)
    fast = batched_fast_update(model, pre, probe_targets, config)
    live = dict(last_channel(model)[2].named_parameters())
    cores = {n: (live[n] + (fast[n] - live[n].detach())) if first_order else fast[n] for n in names}
    with override_last_core(model, cores):
        pred = model(tokens, model.new_state(tokens.shape[0]))[:, -q:]
    return (pred - targets).pow(2).mean(dim=(1, 2))


def train_fast(model, namespace: str, families, seed: int, plan: dict, *, config: FastStateConfig = FastStateConfig()):
    """R11 Stage C's ``train_stack`` with ONE change. For a specialist arm, each step's query pass runs under the last
    block's core adapted on that step's observed support (its probes), and the outer gradient is first order.
    - Kept from ``train_stack``: the optimizer, the schedule, the k schedule, the batches and the selection rule
      (DEV, k = 32, every ``select_every`` steps; here scored after adaptation).
    - k = 0 steps (no support) and FFN-only arms run ``train_stack``'s step exactly."""
    names = specialist_core_names(last_channel(model)[2])
    opt = torch.optim.Adam(model.parameters(), lr=plan["adam_lr"])
    steps = plan["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
    rng = torch.Generator().manual_seed(st.seed_of(namespace, "k-schedule", seed))
    best = {"score": math.inf, "step": None, "state": None}
    trace, started = [], time.time()
    for step in range(steps + 1):
        if step % plan["select_every"] == 0 or step == steps:
            scores = []
            for family in families:
                tokens, targets = ss.batch_tokens(namespace, [family], seed, "DEV", 0, plan["dev_episodes"], 32)
                probes = batch_probes(namespace, [family], seed, "DEV", 0, plan["dev_episodes"], 32) if names else (None, None)
                with torch.no_grad():
                    scores.append(float(adapted_query_mse(model, tokens, targets, *probes, config).mean()))
            score = statistics.fmean(scores)
            trace.append({"step": step, "dev_k32": score, "seconds": round(time.time() - started, 1)})
            if math.isfinite(score) and score < best["score"]:
                best.update(score=score, step=step, state={k: v.detach().clone() for k, v in model.state_dict().items()})
        if step == steps:
            break
        k = ss.K_GRID[int(torch.randint(len(ss.K_GRID), (1,), generator=rng))]
        tokens, targets = ss.batch_tokens(namespace, families, seed, "TRAIN", step * plan["batch"], plan["batch"], k)
        probes = batch_probes(namespace, families, seed, "TRAIN", step * plan["batch"], plan["batch"], k) \
            if names and k > 0 else (None, None)
        loss = adapted_query_mse(model, tokens, targets, *probes, config).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.load_state_dict(best["state"])
    return {"selected_step": best["step"], "selected_dev_score": best["score"], "trace": trace}


def freeze(model):
    """The trained stack as an immutable checkpoint for evaluation: eval mode, no parameter requires grad."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
