"""R11: episodic specialist tasks and the microscopic learners of Stage A. RESEARCH_ONLY, float64, offline.

The task family is R97W's synthetic teacher, re-implemented here from the R97W PORTABLE specification
(abraxis717/CGC ``research/benchmarks/r97w/portable``, commit 0895c2b; ``SPEC_PORTABLE.json`` and
``experiment.py`` digests are recorded in R11's preregistration). Inputs ``z`` (4 features) and the constant 1 form
``x = [z, 1]`` (5 inputs); targets are 3-dimensional:

    smooth       y = .35 sin(1.5 z A1) + .15 tanh(z A2) + .09 z_0 z_1          + .28 nonce + .12 z EP
    conditional  y = .40 (g sin(z A1) + (1 - g) tanh(z A2)),  g = sigmoid(3 z_0) + .28 nonce + .12 z EP

``A1, A2`` are shared within a (family, seed) group; ``EP`` (slope) and ``nonce`` (sign per output) are drawn per
episode. Every draw comes from SHA-256 of a namespaced key: R11 uses its OWN namespaces, so no R11 episode is an R97W
episode and no confirmation episode was seen in development.

Learners (Stage A). ``E0``: three independent E0 cells, ``x [5] -> 3``, core ``[5, 3, 8]`` = 120 parameters
(``cgc.CGCCore`` layout and mathematics, tested against the pinned upstream E0). ``GELU``: R97W's 120-parameter
network 5 -> 13 -> 3. ``projected``: the transformer adapter geometry around the same core, ``y = R P_out core(P_in
E x)`` with a fixed embedding ``E`` (5 -> H) and readout ``R`` (H -> 3), ``P_in`` (H -> 5) and ``P_out`` (3 -> H).
Only the 120 core parameters ever change in an inner (episode) loop.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math

import torch
from torch.nn import functional as F

from .cgc import phi, soft_bound

DTYPE = torch.float64
FEATURES, OUTPUTS, UNITS, GELU_WIDTH = 4, 3, 8, 13
INPUTS = FEATURES + 1
FAMILIES = ("smooth", "conditional")
PROTOCOL = {"support": 32, "query": 48, "ood": 32, "inner_steps": 8, "inner_lr": 0.08, "outer_interpolation": 0.28,
            "gradient_clip_l2": 2.0, "eval_steps": (0, 1, 2, 4, 8), "train_episodes": 64, "test_episodes": 12,
            "dev_episodes": 5, "dev_checkpoints": (0, 16, 32, 64), "forget_probe_episodes": 8}


def seed_of(*parts) -> int:
    key = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2 ** 31 - 1)


def rand(shape, *parts) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed_of(*parts)), dtype=DTYPE)


def randperm(n: int, *parts) -> torch.Tensor:
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed_of(*parts)))


def orthonormal(rows: int, cols: int, *parts) -> torch.Tensor:
    """A ``[rows, cols]`` matrix with orthonormal columns (rows >= cols), from a namespaced seed."""
    q, r = torch.linalg.qr(rand((rows, cols), *parts))
    return q * torch.sign(torch.diagonal(r))


@dataclasses.dataclass(frozen=True)
class Episode:
    support: tuple
    query: tuple
    ood: tuple


def teacher(namespace: str, family: str, seed: int, split: str, episode: int, protocol=PROTOCOL) -> Episode:
    d, out = FEATURES, OUTPUTS
    a1 = rand((d, out), namespace, "shared", family, seed, "A") / math.sqrt(d)
    a2 = rand((d, out), namespace, "shared", family, seed, "B") / math.sqrt(d)
    ep = rand((d, out), namespace, "episode", family, seed, split, episode, "slope") / math.sqrt(d)
    nonce = torch.where(rand((1, out), namespace, "episode", family, seed, split, episode, "nonce") > 0, 1.0, -1.0)

    def target(z):
        if family == "smooth":
            y = .35 * torch.sin(1.5 * z @ a1) + .15 * torch.tanh(z @ a2) + .09 * (z[:, 0:1] * z[:, 1:2])
        elif family == "conditional":
            g = torch.sigmoid(3 * z[:, 0:1])
            y = .40 * (g * torch.sin(z @ a1) + (1 - g) * torch.tanh(z @ a2))
        else:
            raise ValueError(f"unknown family {family!r}")
        return y + .28 * nonce + .12 * (z @ ep)

    def data(kind, n):
        z = rand((n, d), namespace, "inputs", family, seed, split, episode, kind)
        if kind == "ood":
            z = z * 1.4
        return torch.cat([z, torch.ones(n, 1, dtype=DTYPE)], dim=-1), target(z)

    return Episode(data("support", protocol["support"]), data("query", protocol["query"]),
                   data("ood", protocol["ood"]))


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).pow(2).mean()


# -- Stage A learners ---------------------------------------------------------------------------------------------

def e0_cells(w: torch.Tensor, x: torch.Tensor, bound: float | None) -> torch.Tensor:
    """``CGCCore``'s mathematics, functionally: ``w [d, C, N]``, ``x [n, d]`` -> ``[n, C]``."""
    d, c, n = w.shape
    u = soft_bound(x @ w.reshape(d, c * n), bound)
    return phi(u).unflatten(-1, (c, n)).sum(-1)


def gelu_cells(core: tuple, x: torch.Tensor) -> torch.Tensor:
    w1, b1, w2, b2 = core
    return F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)


@dataclasses.dataclass(frozen=True)
class Learner:
    """One Stage A arm. ``core`` is ``"e0"`` or ``"gelu"``; ``hidden`` None = microscopic (no projections)."""
    core: str
    hidden: int | None = None
    bound: float | None = None
    learn_projections: bool = False

    def init_core(self, seed: int, namespace: str) -> tuple:
        if self.core == "e0":                    # R97W's ECS initialization scale, in the [d, C, N] layout
            return (rand((INPUTS, OUTPUTS, UNITS), namespace, "init", "e0", seed) * (.17 / math.sqrt(INPUTS)),)
        if self.core == "gelu":                  # R97W's FFN initialization
            return (rand((GELU_WIDTH, INPUTS), namespace, "init", "gelu", seed, "in") * (.22 / math.sqrt(INPUTS)),
                    torch.zeros(GELU_WIDTH, dtype=DTYPE),
                    rand((OUTPUTS, GELU_WIDTH), namespace, "init", "gelu", seed, "out") * (.22 / math.sqrt(GELU_WIDTH)),
                    torch.zeros(OUTPUTS, dtype=DTYPE))
        raise ValueError(self.core)

    def fixed(self, seed: int, namespace: str) -> dict:
        """The context of the adapter (never trained): embedding ``E`` (5 -> H, unit-RMS hidden code) and
        readout ``R`` (H -> 3, orthonormal rows)."""
        if self.hidden is None:
            return {}
        h = self.hidden
        return {"E": orthonormal(h, INPUTS, namespace, "embed", seed) * math.sqrt(h / INPUTS),
                "R": orthonormal(h, OUTPUTS, namespace, "readout", seed).T.contiguous()}

    def init_projections(self, seed: int, namespace: str) -> tuple:
        """``P_in [5, H]`` (std H^-1/2: a unit-variance code from a unit-RMS hidden state, as ``ModelConfig``'s
        rule) and ``P_out [H, 3]`` (std 3^-1/2: unit gain through the fixed readout). Same for every core."""
        if self.hidden is None:
            return ()
        h = self.hidden
        return (rand((INPUTS, h), namespace, "init", "proj_in", seed) * h ** -0.5,
                rand((h, OUTPUTS), namespace, "init", "proj_out", seed) * OUTPUTS ** -0.5)

    def predict(self, core: tuple, projections: tuple, fixed: dict, x: torch.Tensor) -> torch.Tensor:
        if self.hidden is None:
            code = x
        else:
            code = (x @ fixed["E"].T) @ projections[0].T
        cells = e0_cells(core[0], code, self.bound) if self.core == "e0" else gelu_cells(core, code)
        if self.hidden is None:
            return cells
        return (cells @ projections[1].T) @ fixed["R"].T

    @property
    def inner_parameters(self) -> int:
        return INPUTS * OUTPUTS * UNITS if self.core == "e0" else (GELU_WIDTH * INPUTS + GELU_WIDTH
                                                                   + OUTPUTS * GELU_WIDTH + OUTPUTS)

    def parameter_accounting(self) -> dict:
        inner = self.inner_parameters
        proj = 0 if self.hidden is None else (INPUTS + OUTPUTS) * self.hidden
        return {"inner_loop_parameters": inner, "projection_parameters": proj,
                "outer_trainable_parameters": inner + (proj if self.learn_projections else 0),
                "resident_parameters": inner + proj,
                "fixed_context_tensors": 0 if self.hidden is None else (INPUTS + OUTPUTS) * self.hidden}


def adapt(learner: Learner, core: tuple, projections: tuple, fixed: dict, support: tuple, steps: int, lr: float,
          clip: float, *, record=None, query=None, ood=None):
    """``steps`` explicit SGD steps on the CORE only (global L2 clip), as R97W's inner loop. Returns the adapted
    core (detached) and, when ``record`` is given, query and OOD losses at those step counts."""
    cur = tuple(p.detach().requires_grad_(True) for p in core)
    proj = tuple(p.detach() for p in projections)
    trajectory = {}

    def note(t):
        with torch.no_grad():
            trajectory[t] = [float(mse(learner.predict(cur, proj, fixed, data[0]), data[1])) for data in (query, ood)]

    if record is not None and 0 in record:
        note(0)
    for t in range(1, steps + 1):
        loss = mse(learner.predict(cur, proj, fixed, support[0]), support[1])
        grads = torch.autograd.grad(loss, cur)
        norm = math.sqrt(sum(float(g.detach().pow(2).sum()) for g in grads))
        scale = min(1.0, clip / (norm + 1e-12))
        cur = tuple((p - lr * scale * g).detach().requires_grad_(True) for p, g in zip(cur, grads))
        if record is not None and t in record:
            note(t)
    return tuple(p.detach() for p in cur), trajectory


def projection_step(learner: Learner, core: tuple, projections: tuple, fixed: dict, query: tuple, lr: float,
                    clip: float) -> tuple:
    """First-order outer step for LEARNED projections: one clipped SGD step on the query loss of the ADAPTED core
    (the core is held fixed). Only for ``learn_projections`` arms; the extra capacity is accounted separately."""
    proj = tuple(p.detach().requires_grad_(True) for p in projections)
    loss = mse(learner.predict(tuple(c.detach() for c in core), proj, fixed, query[0]), query[1])
    grads = torch.autograd.grad(loss, proj)
    norm = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))
    scale = min(1.0, clip / (norm + 1e-12))
    return tuple((p - lr * scale * g).detach() for p, g in zip(proj, grads))


# -- Stage B: the specialist inside the real ChannelExpert ------------------------------------------------------------

STAGE_B = {"hidden": 64, "ffn_hidden": 86, "nuisance": 1.0, "inner_steps": 8, "inner_lr": 0.08,
           "gradient_clip_l2": 2.0, "eval_steps": (0, 1, 2, 4, 8), "test_episodes": 12, "dev_episodes": 5}

# objective "specialist": first-order MAML (the query loss AFTER the inner loop trains everything, the core through
# its adapted value); "pooled": ordinary supervised training on support and query with no inner loop.
ARMS_B = {
    "G0": {"kind": "ffn", "gate_mode": "learned", "objective": "pooled"},
    "G1": {"kind": "ffn+cgc", "gate_mode": "learned", "objective": "specialist"},
    "G2": {"kind": "ffn+cgc", "gate_mode": "fixed", "objective": "specialist"},
    "G3": {"kind": "ffn+mlp", "gate_mode": "fixed", "objective": "specialist"},
    "G2_POOLED": {"kind": "ffn+cgc", "gate_mode": "fixed", "objective": "pooled"},
}


def hidden_context(namespace: str, seed: int, hidden: int = 64) -> dict:
    """The fixed embedding of ``x`` into a hidden state (5-dim task subspace + orthogonal nuisance) and the fixed
    readout ``R`` (orthonormal rows). Never trained: they stand for the surrounding model."""
    basis = orthonormal(hidden, hidden, namespace, "basis", seed)
    return {"task": basis[:, :INPUTS], "nuisance": basis[:, INPUTS:],
            "R": orthonormal(hidden, OUTPUTS, namespace, "readout", seed).T.contiguous()}


def hidden_states(context: dict, x: torch.Tensor, namespace: str, seed: int, *parts, sigma: float = 1.0):
    """``u = Q_task x + sigma Q_nuisance xi`` (unit RMS on average; the nuisance is drawn per example)."""
    xi = rand((x.shape[0], context["nuisance"].shape[1]), namespace, "nuisance", seed, *parts)
    return x @ context["task"].T + sigma * (xi @ context["nuisance"].T)


def build_arm(arm: str, profile: str, seed: int, *, dtype=torch.float64):
    """One ``ChannelExpert`` of the arm: the repository's class and its own initialization rules."""
    import types
    from .config import ModelConfig
    from .model import ElpisTorchLM, _seeded_normal
    from .sparse_channel import ChannelExpert, SparseConfig
    from .cgc_specialist import micro_model, micro_sparse
    spec = ARMS_B[arm]
    m = micro_model(ModelConfig(vocab_size=2, d_model=STAGE_B["hidden"], n_layers=1, n_heads=1, max_seq_len=8,
                                channel="ffn+cgc", ffn_hidden=STAGE_B["ffn_hidden"], dtype="float64", seed=seed),
                    profile=profile)
    base = SparseConfig(experts=1, top_k=1, ffn_hidden=STAGE_B["ffn_hidden"], gate_init=0.1)
    s = base.replace(kind="ffn") if spec["kind"] == "ffn" else micro_sparse(
        base, profile=profile, gate_mode=spec["gate_mode"], kind=spec["kind"])
    expert = ChannelExpert(m, s, dtype)
    holder = types.SimpleNamespace(config=m)
    with torch.no_grad():
        for name, p in expert.named_parameters():
            std = ElpisTorchLM._init_std(holder, name)
            if std is not None:
                p.copy_(_seeded_normal(seed, "r11b:" + name, p.shape, std, p.dtype))
            elif name == "cgc_gate":
                p.fill_(s.gate_init)
            else:
                p.zero_()                                 # the GELU control's biases
    return expert, m, s


def specialist_cells(expert, u: torch.Tensor) -> torch.Tensor | None:
    if expert.cgc is not None:
        return expert.cgc(u)
    return None if expert.mlp is None else expert.mlp(u)


@torch.no_grad()
def init_branch_like_stage_a(expert, seed: int, namespace: str) -> dict:
    """The A1 residual-scale calibration, fixed BEFORE any Stage B data: the specialist branch gets Stage A's
    projected-adapter initialization, the one under which Stage A measures adaptation. ``P_in`` std H^-1/2 (unit-
    variance code), the core at R97W's scale (cubic 0.17/sqrt(5); GELU R97W's FFN rule), ``P_out`` std 3^-1/2 (unit
    gain through the orthonormal readout). The repository's residual rule (std 0.014 here) leaves the inner loop a
    gain near 0.1 and the cell offsets dominate: an uncontrolled magnitude, not a property of either specialist.
    Same rule for every specialist arm; the historical gate (G1, 0.1) is NOT changed."""
    if expert.cgc is None and expert.mlp is None:
        return {}
    branch = expert.cgc if expert.cgc is not None else expert.mlp
    h = branch.proj_in.weight.shape[1]
    branch.proj_in.weight.copy_(rand(tuple(branch.proj_in.weight.shape), namespace, "branch", seed, "proj_in") * h ** -0.5)
    branch.proj_out.weight.copy_(rand(tuple(branch.proj_out.weight.shape), namespace, "branch", seed, "proj_out")
                                 * OUTPUTS ** -0.5)
    if expert.cgc is not None:
        expert.cgc.core.weight.copy_(rand(tuple(expert.cgc.core.weight.shape), namespace, "branch", seed, "core")
                                     * (.17 / math.sqrt(INPUTS)))
    else:
        mlp = expert.mlp
        mlp.hidden.weight.copy_(rand(tuple(mlp.hidden.weight.shape), namespace, "branch", seed, "w1")
                                * (.22 / math.sqrt(INPUTS)))
        mlp.out.weight.copy_(rand(tuple(mlp.out.weight.shape), namespace, "branch", seed, "w2")
                             * (.22 / math.sqrt(GELU_WIDTH)))
        mlp.hidden.bias.zero_()
        mlp.out.bias.zero_()
    return {"proj_in_std": h ** -0.5, "proj_out_std": OUTPUTS ** -0.5,
            "core_rule": "R97W ECS 0.17/sqrt(5)" if expert.cgc is not None else "R97W FFN 0.22/sqrt(fan_in), zero bias"}


def _functional(expert, overrides: dict, u: torch.Tensor, *, live: bool, ablate: bool = False):
    from torch.func import functional_call
    params = {n: (p if live else p.detach()) for n, p in expert.named_parameters()}
    params.update(overrides)
    if ablate:                                            # the expert with its specialist branch removed
        return functional_call(expert.ffn, {k[4:]: v for k, v in params.items() if k.startswith("ffn.")}, (u,))
    return functional_call(expert, params, (u,))


def inner_adapt(expert, core_names, u, y, readout, steps, lr, clip, *, start=None, record=None, probes=None):
    """``steps`` explicit SGD steps on the specialist CORE only, everything else frozen (the R0 session's
    ``W_base + dW`` convention through ``functional_call``). ``record``: step counts at which the probes'
    (query, OOD) losses are noted. Returns ``(core, trajectory)``."""
    base = dict(expert.named_parameters())
    cur = {n: (base[n] if start is None else start[n]).detach().clone().requires_grad_(True) for n in core_names}
    trajectory = {}

    def note(t):
        with torch.no_grad():
            trajectory[t] = [float(mse(_functional(expert, cur, pu, live=False) @ readout.T, py)) for pu, py in probes]

    if record is not None and 0 in record:
        note(0)
    for t in range(1, steps + 1):
        loss = mse(_functional(expert, cur, u, live=False) @ readout.T, y)
        grads = torch.autograd.grad(loss, list(cur.values()))
        norm = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))
        scale = min(1.0, clip / (norm + 1e-12))
        cur = {n: (p - lr * scale * g).detach().requires_grad_(True) for (n, p), g in zip(cur.items(), grads)}
        if record is not None and t in record:
            note(t)
    return {n: p.detach() for n, p in cur.items()}, trajectory


def episode_states(context: dict, episode: Episode, namespace: str, seed: int, split: str, index: int,
                   sigma: float) -> dict:
    """The hidden states of an episode's support, query and OOD inputs."""
    return {kind: (hidden_states(context, getattr(episode, kind)[0], namespace, seed, split, index, kind,
                                 sigma=sigma), getattr(episode, kind)[1])
            for kind in ("support", "query", "ood")}


def train_arm(arm: str, profile: str, family: str, seed: int, namespace: str, plan: dict, *, trace_every=None):
    """Offline (Mode A) training of one arm's expert for one (family, seed) group. Returns the expert, the fixed
    context and a training trace (gate, branch/FFN RMS, P_out norm, gradient norms, saturation)."""
    from .cgc_specialist import branch_diagnostics, gradient_norms, specialist_core_names
    spec = ARMS_B[arm]
    expert, _, _ = build_arm(arm, profile, seed)
    context = hidden_context(namespace, seed, STAGE_B["hidden"])
    readout = context["R"]
    sigma = STAGE_B["nuisance"]
    dev = teacher(namespace, family, seed, "DIAGNOSE", 0)
    dev_u = hidden_states(context, torch.cat([dev.support[0], dev.query[0]]), namespace, seed, "DIAGNOSE", 0, "all",
                          sigma=sigma)
    factor = init_branch_like_stage_a(expert, seed, namespace)
    core_names = specialist_core_names(expert)
    optimizer = torch.optim.Adam(expert.parameters(), lr=plan["adam_lr"])
    steps, batch = plan["outer_steps"], plan["meta_batch"]
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
    trace = []
    initial = {n: p.detach().clone() for n, p in expert.named_parameters()}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for b in range(batch):
            index = step * batch + b
            ep = episode_states(context, teacher(namespace, family, seed, "TRAIN", index), namespace, seed, "TRAIN",
                                index, sigma)
            su, sy = ep["support"]
            qu, qy = ep["query"]
            if spec["objective"] == "specialist" and core_names:
                fast, _ = inner_adapt(expert, core_names, su, sy, readout, STAGE_B["inner_steps"], STAGE_B["inner_lr"],
                                      STAGE_B["gradient_clip_l2"])
                live = dict(expert.named_parameters())
                overrides = {n: live[n] + (fast[n] - live[n].detach()) for n in core_names}   # first order
                loss = mse(_functional(expert, overrides, qu, live=True) @ readout.T, qy)
            else:
                u = torch.cat([su, qu])
                loss = mse(expert(u) @ readout.T, torch.cat([sy, qy]))
            (loss / batch).backward()
            total += float(loss.detach()) / batch
        grads = gradient_norms(expert)
        optimizer.step()
        schedule.step()
        if trace_every and (step % trace_every == 0 or step == steps - 1):
            diag = branch_diagnostics(expert, dev_u)
            trace.append({"step": step, "loss": total, "gradient_norms": grads,
                          **{k: diag.get(k) for k in ("gate", "branch_rms", "branch_to_ffn_rms", "ffn_rms",
                                                      "preact_rms", "preact_abs_max", "saturated_fraction",
                                                      "nonfinite")},
                          "proj_out_norm": _proj_out_norm(expert)})
    change = {n: float((p.detach() - initial[n]).norm() / (initial[n].norm() + 1e-30))
              for n, p in expert.named_parameters()}
    return expert, context, {"branch_initialization": factor, "trace": trace, "relative_change": change,
                             "final_diagnostics": branch_diagnostics(expert, dev_u)}


def _proj_out_norm(expert):
    out = expert.cgc.proj_out if expert.cgc is not None else (expert.mlp.proj_out if expert.mlp is not None else None)
    return None if out is None else float(out.weight.detach().norm())


def evaluate_arm(expert, context: dict, family: str, seed: int, namespace: str, split: str, episodes: int, *,
                    start_from=None):
    """Per test episode: query/OOD losses after k explicit core steps (Mode A), the frozen k = 0 value (Mode B), and
    the causal ablation (specialist branch removed). ``start_from`` swaps in another expert's specialist (its core,
    projections and gate) on this expert's FFN: the specialist-swap control."""
    from .cgc_specialist import specialist_core_names
    readout, sigma = context["R"], STAGE_B["nuisance"]
    core_names = specialist_core_names(expert)
    rows = []
    for index in range(episodes):
        ep = episode_states(context, teacher(namespace, family, seed, split, index), namespace, seed, split, index,
                            sigma)
        su, sy = ep["support"]
        probes = [ep["query"], ep["ood"]]
        row = {"episode": index, "null": [float(py.pow(2).mean()) for _, py in probes]}   # the zero predictor
        with torch.no_grad():
            row["ablated"] = [float(mse(_functional(expert, {}, pu, live=False, ablate=True) @ readout.T, py))
                              for pu, py in probes]
        if core_names:
            overrides = {} if start_from is None else _specialist_state(start_from)
            if overrides:
                live = expert
                saved = {n: p.detach().clone() for n, p in live.named_parameters() if n in overrides}
                with torch.no_grad():
                    for n, p in live.named_parameters():
                        if n in overrides:
                            p.copy_(overrides[n])
            _, trajectory = inner_adapt(expert, core_names, su, sy, readout, STAGE_B["inner_steps"],
                                        STAGE_B["inner_lr"], STAGE_B["gradient_clip_l2"],
                                        record=set(STAGE_B["eval_steps"]), probes=probes)
            if overrides:
                with torch.no_grad():
                    for n, p in expert.named_parameters():
                        if n in saved:
                            p.copy_(saved[n])
        else:
            with torch.no_grad():
                value = [float(mse(expert(pu) @ readout.T, py)) for pu, py in probes]
            trajectory = {k: value for k in STAGE_B["eval_steps"]}
        row["curve"] = {str(k): v for k, v in trajectory.items()}
        rows.append(row)
    return rows


def _specialist_state(expert) -> dict:
    """The specialist's own parameters (projections, core, gate), for the swap control."""
    return {n: p.detach().clone() for n, p in expert.named_parameters() if not n.startswith("ffn.")}
